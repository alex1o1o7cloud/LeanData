import Mathlib

namespace frequency_of_group_samples_l118_118462

-- Conditions
def sample_capacity : ℕ := 32
def group_frequency : ℝ := 0.125

-- Theorem statement
theorem frequency_of_group_samples : group_frequency * sample_capacity = 4 :=
by sorry

end frequency_of_group_samples_l118_118462


namespace third_car_year_l118_118563

theorem third_car_year (y1 y2 y3 : ℕ) (h1 : y1 = 1970) (h2 : y2 = y1 + 10) (h3 : y3 = y2 + 20) : y3 = 2000 :=
by
  sorry

end third_car_year_l118_118563


namespace function_intersects_all_lines_l118_118632

theorem function_intersects_all_lines :
  (∃ f : ℝ → ℝ, (∀ a : ℝ, ∃ y : ℝ, y = f a) ∧ (∀ k b : ℝ, ∃ x : ℝ, f x = k * x + b)) :=
sorry

end function_intersects_all_lines_l118_118632


namespace correlation_comparison_l118_118851

/-- The data for variables x and y are (1, 3), (2, 5.3), (3, 6.9), (4, 9.1), and (5, 10.8) -/
def xy_data : List (Int × Float) := [(1, 3), (2, 5.3), (3, 6.9), (4, 9.1), (5, 10.8)]

/-- The data for variables U and V are (1, 12.7), (2, 10.2), (3, 7), (4, 3.6), and (5, 1) -/
def UV_data : List (Int × Float) := [(1, 12.7), (2, 10.2), (3, 7), (4, 3.6), (5, 1)]

/-- r1 is the linear correlation coefficient between y and x -/
noncomputable def r1 : Float := sorry

/-- r2 is the linear correlation coefficient between V and U -/
noncomputable def r2 : Float := sorry

/-- The problem is to prove that r2 < 0 < r1 given the data conditions -/
theorem correlation_comparison : r2 < 0 ∧ 0 < r1 := 
by 
  sorry

end correlation_comparison_l118_118851


namespace range_eq_domain_l118_118330

def f (x : ℝ) : ℝ := |x - 2| - 2

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem range_eq_domain : (Set.range f) = M :=
by
  sorry

end range_eq_domain_l118_118330


namespace find_m_l118_118894

noncomputable def curve (x : ℝ) : ℝ := (1 / 4) * x^2
noncomputable def line (x : ℝ) : ℝ := 1 - 2 * x

theorem find_m (m n : ℝ) (h_curve : curve m = n) (h_perpendicular : (1 / 2) * m * (-2) = -1) : m = 1 := 
  sorry

end find_m_l118_118894


namespace monotonic_exponential_decreasing_l118_118201

variable (a : ℝ) (f : ℝ → ℝ)

theorem monotonic_exponential_decreasing {m n : ℝ}
  (h0 : a = (Real.sqrt 5 - 1) / 2)
  (h1 : ∀ x, f x = a^x)
  (h2 : 0 < a ∧ a < 1)
  (h3 : f m > f n) :
  m < n :=
sorry

end monotonic_exponential_decreasing_l118_118201


namespace length_of_intersection_segment_l118_118824

-- Define the polar coordinates conditions
def curve_1 (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def curve_2 (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

-- Convert polar equations to Cartesian coordinates
def curve_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4 * y
def curve_2_cartesian (x y : ℝ) : Prop := x = 1

-- Define the intersection points and the segment length function
def segment_length (y1 y2 : ℝ) : ℝ := abs (y1 - y2)

-- The statement to prove
theorem length_of_intersection_segment :
  (curve_1_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_1_cartesian 1 (2 - Real.sqrt 3)) →
  (curve_2_cartesian 1 (2 + Real.sqrt 3)) ∧ (curve_2_cartesian 1 (2 - Real.sqrt 3)) →
  segment_length (2 + Real.sqrt 3) (2 - Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end length_of_intersection_segment_l118_118824


namespace angle_CDB_45_degrees_l118_118917

theorem angle_CDB_45_degrees
  (α β γ δ : ℝ)
  (triangle_isosceles_right : α = β)
  (triangle_angle_BCD : γ = 90)
  (square_angle_DCE : δ = 90)
  (triangle_angle_ABC : α = β)
  (isosceles_triangle_angle : α + β + γ = 180)
  (isosceles_triangle_right : α = 45)
  (isosceles_triangle_sum : α + α + 90 = 180)
  (square_geometry : δ = 90) :
  γ + δ = 180 →  180 - (γ + α) = 45 :=
by
  sorry

end angle_CDB_45_degrees_l118_118917


namespace sin_18_eq_l118_118109

theorem sin_18_eq : ∃ x : Real, x = (Real.sin (Real.pi / 10)) ∧ x = (Real.sqrt 5 - 1) / 4 := by
  sorry

end sin_18_eq_l118_118109


namespace g_g_is_odd_l118_118116

def f (x : ℝ) : ℝ := x^3

def g (x : ℝ) : ℝ := f (f x)

theorem g_g_is_odd : ∀ x : ℝ, g (g (-x)) = -g (g x) :=
by 
-- proof will go here
sorry

end g_g_is_odd_l118_118116


namespace AB_vector_eq_l118_118443

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C D : V)
variables (a b : V)
variable (ABCD_parallelogram : is_parallelogram A B C D)

-- Definition of the diagonals
def AC_vector : V := C - A
def BD_vector : V := D - B

-- The given condition that diagonals AC and BD are equal to a and b respectively
axiom AC_eq_a : AC_vector A C = a
axiom BD_eq_b : BD_vector B D = b

-- Proof problem
theorem AB_vector_eq : (B - A) = (1/2) • (a - b) :=
sorry

end AB_vector_eq_l118_118443


namespace percentage_problem_l118_118260

theorem percentage_problem (P : ℝ) :
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 :=
by
  intro h
  sorry

end percentage_problem_l118_118260


namespace find_k_l118_118430

theorem find_k (a b : ℕ) (k : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a^2 + b^2) = k * (a * b - 1)) :
  k = 5 :=
sorry

end find_k_l118_118430


namespace percent_difference_calculation_l118_118592

theorem percent_difference_calculation :
  (0.80 * 45) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_calculation_l118_118592


namespace combined_share_is_50000_l118_118190

def profit : ℝ := 80000

def majority_owner_share : ℝ := 0.25 * profit

def remaining_profit : ℝ := profit - majority_owner_share

def partner_share : ℝ := 0.25 * remaining_profit

def combined_share_majority_two_owners : ℝ := majority_owner_share + 2 * partner_share

theorem combined_share_is_50000 :
  combined_share_majority_two_owners = 50000 := 
by 
  sorry

end combined_share_is_50000_l118_118190


namespace pole_intersection_height_l118_118319

theorem pole_intersection_height 
  (h1 h2 d : ℝ) 
  (h1pos : h1 = 30) 
  (h2pos : h2 = 90) 
  (dpos : d = 150) : 
  ∃ y, y = 22.5 :=
by
  sorry

end pole_intersection_height_l118_118319


namespace range_of_a_l118_118387

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - 2 * x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x0 : ℝ, 0 < x0 ∧ x0 < 1 →
  (0 < (deriv (fun x => f a x - x)) x0)) →
  a > (4 / Real.exp (3 / 4)) :=
by
  intro h
  sorry

end range_of_a_l118_118387


namespace shortest_tree_height_is_correct_l118_118098

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end shortest_tree_height_is_correct_l118_118098


namespace α_plus_β_eq_two_l118_118140

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

theorem α_plus_β_eq_two
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 := 
sorry

end α_plus_β_eq_two_l118_118140


namespace largest_fraction_l118_118957

theorem largest_fraction (p q r s : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) :
  (∃ (x : ℝ), x = (r + s) / (p + q) ∧ 
  (x > (p + s) / (q + r)) ∧ 
  (x > (p + q) / (r + s)) ∧ 
  (x > (q + r) / (p + s)) ∧ 
  (x > (q + s) / (p + r))) :=
sorry

end largest_fraction_l118_118957


namespace dot_product_a_b_l118_118986

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end dot_product_a_b_l118_118986


namespace union_M_N_l118_118861

-- Define the set M
def M : Set ℤ := {x | x^2 - x = 0}

-- Define the set N
def N : Set ℤ := {y | y^2 + y = 0}

-- Prove that the union of M and N is {-1, 0, 1}
theorem union_M_N :
  M ∪ N = {-1, 0, 1} :=
by
  sorry

end union_M_N_l118_118861


namespace find_x_l118_118495

-- Definitions based on the given conditions
variables {B C D : Type} (A : Type)

-- Angles in degrees
variables (angle_ACD : ℝ := 100)
variables (angle_ADB : ℝ)
variables (angle_ABD : ℝ := 2 * angle_ADB)
variables (angle_DAC : ℝ)
variables (angle_BAC : ℝ := angle_DAC)
variables (angle_ACB : ℝ := 180 - angle_ACD)
variables (y : ℝ := angle_DAC)
variables (x : ℝ := angle_ADB)

-- The proof statement
theorem find_x (h1 : B = C) (h2 : C = D) 
    (h3: angle_ACD = 100) 
    (h4: angle_ADB = x) 
    (h5: angle_ABD = 2 * x) 
    (h6: angle_DAC = angle_BAC) 
    (h7: angle_DAC = y)
    : x = 20 :=
sorry

end find_x_l118_118495


namespace global_phone_company_customers_l118_118044

theorem global_phone_company_customers :
  (total_customers = 25000) →
  (us_percentage = 0.20) →
  (canada_percentage = 0.12) →
  (australia_percentage = 0.15) →
  (uk_percentage = 0.08) →
  (india_percentage = 0.05) →
  (us_customers = total_customers * us_percentage) →
  (canada_customers = total_customers * canada_percentage) →
  (australia_customers = total_customers * australia_percentage) →
  (uk_customers = total_customers * uk_percentage) →
  (india_customers = total_customers * india_percentage) →
  (mentioned_countries_customers = us_customers + canada_customers + australia_customers + uk_customers + india_customers) →
  (other_countries_customers = total_customers - mentioned_countries_customers) →
  (other_countries_customers = 10000) ∧ (us_customers / other_countries_customers = 1 / 2) :=
by
  -- The further proof steps would go here if needed
  sorry

end global_phone_company_customers_l118_118044


namespace rhombus_area_from_roots_l118_118589

-- Definition of the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 10 * x + 24 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b

-- Final mathematical statement to prove
theorem rhombus_area_from_roots (a b : ℝ) (h : roots a b) :
  a * b = 24 → (1 / 2) * a * b = 12 := 
by
  sorry

end rhombus_area_from_roots_l118_118589


namespace students_playing_long_tennis_l118_118169

theorem students_playing_long_tennis (n F B N L : ℕ)
  (h1 : n = 35)
  (h2 : F = 26)
  (h3 : B = 17)
  (h4 : N = 6)
  (h5 : L = (n - N) - (F - B)) :
  L = 20 :=
by
  sorry

end students_playing_long_tennis_l118_118169


namespace correct_statements_l118_118923

namespace ProofProblem

def P1 : Prop := (-4) + (-5) = -9
def P2 : Prop := -5 - (-6) = 11
def P3 : Prop := -2 * (-10) = -20
def P4 : Prop := 4 / (-2) = -2

theorem correct_statements : P1 ∧ P4 ∧ ¬P2 ∧ ¬P3 := by
  -- proof to be filled in later
  sorry

end ProofProblem

end correct_statements_l118_118923


namespace hula_hoop_ratio_l118_118788

variable (Nancy Casey Morgan : ℕ)
variable (hula_hoop_time_Nancy : Nancy = 10)
variable (hula_hoop_time_Casey : Casey = Nancy - 3)
variable (hula_hoop_time_Morgan : Morgan = 21)

theorem hula_hoop_ratio (hula_hoop_time_Nancy : Nancy = 10) (hula_hoop_time_Casey : Casey = Nancy - 3) (hula_hoop_time_Morgan : Morgan = 21) :
  Morgan / Casey = 3 := by
  sorry

end hula_hoop_ratio_l118_118788


namespace problem_1_l118_118042

theorem problem_1
  (α : ℝ)
  (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := 
sorry

end problem_1_l118_118042


namespace equal_lengths_imply_equal_segments_l118_118562

theorem equal_lengths_imply_equal_segments 
  (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h₁ : a₁ = a₂) 
  (h₂ : b₁ = b₂) : 
  x = y := 
sorry

end equal_lengths_imply_equal_segments_l118_118562


namespace num_persons_initially_l118_118618

theorem num_persons_initially (N : ℕ) (avg_weight : ℝ) 
  (h_increase_avg : avg_weight + 5 = avg_weight + 40 / N) :
  N = 8 := by
    sorry

end num_persons_initially_l118_118618


namespace problem_l118_118245

theorem problem (y : ℝ) (hy : 5 = y^2 + 4 / y^2) : y + 2 / y = 3 ∨ y + 2 / y = -3 :=
by
  sorry

end problem_l118_118245


namespace triangle_inequality_l118_118968

theorem triangle_inequality (a : ℝ) (h₁ : a > 5) (h₂ : a < 19) : 5 < a ∧ a < 19 :=
by
  exact ⟨h₁, h₂⟩

end triangle_inequality_l118_118968


namespace walt_age_l118_118749

variable (W M P : ℕ)

-- Conditions
def condition1 := M = 3 * W
def condition2 := M + 12 = 2 * (W + 12)
def condition3 := P = 4 * W
def condition4 := P + 15 = 3 * (W + 15)

theorem walt_age (W M P : ℕ) (h1 : condition1 W M) (h2 : condition2 W M) (h3 : condition3 W P) (h4 : condition4 W P) : 
  W = 30 :=
sorry

end walt_age_l118_118749


namespace solution_to_quadratic_inequality_l118_118873

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 - 5 * x > 9

theorem solution_to_quadratic_inequality (x : ℝ) : quadratic_inequality x ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_to_quadratic_inequality_l118_118873


namespace min_value_c_plus_d_l118_118390

theorem min_value_c_plus_d (c d : ℤ) (h : c * d = 144) : c + d = -145 :=
sorry

end min_value_c_plus_d_l118_118390


namespace taxi_fare_l118_118468

theorem taxi_fare (x : ℝ) (h : 3.00 + 0.25 * ((x - 0.75) / 0.1) = 12) : x = 4.35 :=
  sorry

end taxi_fare_l118_118468


namespace evaluate_expression_l118_118355

theorem evaluate_expression (b : ℚ) (h : b = 4 / 3) :
  (6 * b ^ 2 - 17 * b + 8) * (3 * b - 4) = 0 :=
by 
  -- Proof goes here
  sorry

end evaluate_expression_l118_118355


namespace pythagorean_triangle_divisible_by_5_l118_118030

theorem pythagorean_triangle_divisible_by_5 {a b c : ℕ} (h : a^2 + b^2 = c^2) : 
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := 
by
  sorry

end pythagorean_triangle_divisible_by_5_l118_118030


namespace divisible_by_8_l118_118697

theorem divisible_by_8 (k : ℤ) : 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  8 ∣ (7 * m^2 - 5 * n^2 - 2) :=
by 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  sorry

end divisible_by_8_l118_118697


namespace apple_tree_total_production_l118_118212

noncomputable def first_season_production : ℕ := 200
noncomputable def second_season_production : ℕ := 
  first_season_production - (first_season_production * 20 / 100)
noncomputable def third_season_production : ℕ := 
  second_season_production * 2
noncomputable def total_production : ℕ := 
  first_season_production + second_season_production + third_season_production

theorem apple_tree_total_production :
  total_production = 680 := by
  sorry

end apple_tree_total_production_l118_118212


namespace inscribed_circle_radius_inequality_l118_118573

open Real

variables (ABC ABD BDC : Type) -- Representing the triangles

noncomputable def r (ABC : Type) : ℝ := sorry -- radius of the inscribed circle in ABC
noncomputable def r1 (ABD : Type) : ℝ := sorry -- radius of the inscribed circle in ABD
noncomputable def r2 (BDC : Type) : ℝ := sorry -- radius of the inscribed circle in BDC

noncomputable def p (ABC : Type) : ℝ := sorry -- semiperimeter of ABC
noncomputable def p1 (ABD : Type) : ℝ := sorry -- semiperimeter of ABD
noncomputable def p2 (BDC : Type) : ℝ := sorry -- semiperimeter of BDC

noncomputable def S (ABC : Type) : ℝ := sorry -- area of ABC
noncomputable def S1 (ABD : Type) : ℝ := sorry -- area of ABD
noncomputable def S2 (BDC : Type) : ℝ := sorry -- area of BDC

lemma triangle_area_sum (ABC ABD BDC : Type) :
  S ABC = S1 ABD + S2 BDC := sorry

lemma semiperimeter_area_relation (ABC ABD BDC : Type) :
  S ABC = p ABC * r ABC ∧
  S1 ABD = p1 ABD * r1 ABD ∧
  S2 BDC = p2 BDC * r2 BDC := sorry

theorem inscribed_circle_radius_inequality (ABC ABD BDC : Type) :
  r1 ABD + r2 BDC > r ABC := sorry

end inscribed_circle_radius_inequality_l118_118573


namespace jenna_average_speed_l118_118541

theorem jenna_average_speed (total_distance : ℕ) (total_time : ℕ) 
(first_segment_speed : ℕ) (second_segment_speed : ℕ) (third_segment_speed : ℕ) : 
  total_distance = 150 ∧ total_time = 2 ∧ first_segment_speed = 50 ∧ 
  second_segment_speed = 70 → third_segment_speed = 105 := 
by 
  intros h
  sorry

end jenna_average_speed_l118_118541


namespace line_shift_upwards_l118_118017

theorem line_shift_upwards (x y : ℝ) (h : y = -2 * x) : y + 3 = -2 * x + 3 :=
by sorry

end line_shift_upwards_l118_118017


namespace total_deposit_amount_l118_118601

def markDeposit : ℕ := 88
def bryanDeposit (markAmount : ℕ) : ℕ := 5 * markAmount - 40
def totalDeposit (markAmount bryanAmount : ℕ) : ℕ := markAmount + bryanAmount

theorem total_deposit_amount : totalDeposit markDeposit (bryanDeposit markDeposit) = 488 := 
by sorry

end total_deposit_amount_l118_118601


namespace contradiction_proof_l118_118321

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end contradiction_proof_l118_118321


namespace num_int_values_N_l118_118185

theorem num_int_values_N (N : ℕ) : 
  (∃ M, M ∣ 72 ∧ M > 3 ∧ N = M - 3) ↔ N ∈ ({1, 3, 5, 6, 9, 15, 21, 33, 69} : Finset ℕ) :=
by
  sorry

end num_int_values_N_l118_118185


namespace ratio_of_inquisitive_tourist_l118_118545

theorem ratio_of_inquisitive_tourist (questions_per_tourist : ℕ)
                                     (num_group1 : ℕ) (num_group2 : ℕ) (num_group3 : ℕ) (num_group4 : ℕ)
                                     (total_questions : ℕ) 
                                     (inquisitive_tourist_questions : ℕ) :
  questions_per_tourist = 2 ∧ 
  num_group1 = 6 ∧ 
  num_group2 = 11 ∧ 
  num_group3 = 8 ∧ 
  num_group4 = 7 ∧ 
  total_questions = 68 ∧ 
  inquisitive_tourist_questions = (total_questions - (num_group1 * questions_per_tourist + num_group2 * questions_per_tourist +
                                                        (num_group3 - 1) * questions_per_tourist + num_group4 * questions_per_tourist)) →
  (inquisitive_tourist_questions : ℕ) / questions_per_tourist = 3 :=
by sorry

end ratio_of_inquisitive_tourist_l118_118545


namespace alcohol_water_ratio_l118_118138

theorem alcohol_water_ratio (a b : ℚ) (h₁ : a = 3/5) (h₂ : b = 2/5) : a / b = 3 / 2 :=
by
  sorry

end alcohol_water_ratio_l118_118138


namespace find_other_number_l118_118095

def integers_three_and_four_sum (a b : ℤ) : Prop :=
  3 * a + 4 * b = 131

def one_of_the_numbers_is (x : ℤ) : Prop :=
  x = 17

theorem find_other_number (a b : ℤ) (h1 : integers_three_and_four_sum a b) (h2 : one_of_the_numbers_is a ∨ one_of_the_numbers_is b) :
  (a = 21 ∨ b = 21) :=
sorry

end find_other_number_l118_118095


namespace decorations_count_l118_118674

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l118_118674


namespace solve_system_of_equations_l118_118620

theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x - 2 * y = 1) (h2 : x + y = 2) : x^2 - 2 * y^2 = -1 :=
by
  sorry

end solve_system_of_equations_l118_118620


namespace same_solution_k_value_l118_118404

theorem same_solution_k_value 
  (x : ℝ)
  (k : ℝ)
  (m : ℝ)
  (h₁ : 2 * x + 4 = 4 * (x - 2))
  (h₂ : k * x + m = 2 * x - 1) 
  (h₃ : k = 17) : 
  k = 17 ∧ m = -91 :=
by
  sorry

end same_solution_k_value_l118_118404


namespace paint_needed_for_new_statues_l118_118035

-- Conditions
def pint_for_original : ℕ := 1
def original_height : ℕ := 8
def num_statues : ℕ := 320
def new_height : ℕ := 2
def scale_ratio : ℚ := (new_height : ℚ) / (original_height : ℚ)
def area_ratio : ℚ := scale_ratio ^ 2

-- Correct Answer
def total_paint_needed : ℕ := 20

-- Theorem to be proved
theorem paint_needed_for_new_statues :
  pint_for_original * num_statues * area_ratio = total_paint_needed := 
by
  sorry

end paint_needed_for_new_statues_l118_118035


namespace number_of_triangles_l118_118776

theorem number_of_triangles (points : List ℝ) (h₀ : points.length = 12)
  (h₁ : ∀ p ∈ points, p ≠ A ∧ p ≠ B ∧ p ≠ C ∧ p ≠ D): 
  (∃ triangles : ℕ, triangles = 216) :=
  sorry

end number_of_triangles_l118_118776


namespace smallest_positive_integer_l118_118263

theorem smallest_positive_integer (n : ℕ) (hn : 0 < n) (h : 19 * n ≡ 1456 [MOD 11]) : n = 6 :=
by
  sorry

end smallest_positive_integer_l118_118263


namespace tangent_line_MP_l118_118418

theorem tangent_line_MP
  (O : Type)
  (circle : O → O → Prop)
  (K M N P L : O)
  (is_tangent : O → O → Prop)
  (is_diameter : O → O → O)
  (K_tangent : is_tangent K M)
  (eq_segments : ∀ {P Q R}, circle P Q → circle Q R → circle P R → (P, Q) = (Q, R))
  (diam_opposite : L = is_diameter K L)
  (line_intrsc : ∀ {X Y}, is_tangent X Y → circle X Y → (Y = Y) → P = Y)
  (circ : ∀ {X Y}, circle X Y) :
  is_tangent M P :=
by
  sorry

end tangent_line_MP_l118_118418


namespace number_solution_exists_l118_118267

theorem number_solution_exists (x : ℝ) (h : 0.80 * x = (4 / 5 * 15) + 20) : x = 40 :=
sorry

end number_solution_exists_l118_118267


namespace find_x_when_y_is_6_l118_118719

-- Condition for inverse variation
def inverse_var (k y : ℝ) (x : ℝ) : Prop := x = k / y^2

-- Given values
def given_value_x : ℝ := 1
def given_value_y : ℝ := 2
def new_value_y : ℝ := 6

-- The theorem to prove
theorem find_x_when_y_is_6 :
  ∃ k, inverse_var k given_value_y given_value_x → inverse_var k new_value_y (1/9) :=
by
  sorry

end find_x_when_y_is_6_l118_118719


namespace parabola_line_intersect_at_one_point_l118_118283

theorem parabola_line_intersect_at_one_point (a : ℚ) :
  (∃ x : ℚ, ax^2 + 5 * x + 4 = 0) → a = 25 / 16 :=
by
  -- Conditions and computation here
  sorry

end parabola_line_intersect_at_one_point_l118_118283


namespace complex_power_sum_eq_five_l118_118694

noncomputable def w : ℂ := sorry

theorem complex_power_sum_eq_five (h : w^3 + w^2 + 1 = 0) : 
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 :=
sorry

end complex_power_sum_eq_five_l118_118694


namespace exists_constant_C_inequality_for_difference_l118_118340

theorem exists_constant_C (a : ℕ → ℝ) (C : ℝ) (hC : 0 < C) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a n ≤ C * n^2) := sorry

theorem inequality_for_difference (a : ℕ → ℝ) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a (n + 1) - a n ≤ 4 * n + 3) := sorry

end exists_constant_C_inequality_for_difference_l118_118340


namespace total_race_distance_l118_118218

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l118_118218


namespace necessary_not_sufficient_condition_l118_118401

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

end necessary_not_sufficient_condition_l118_118401


namespace no_solutions_l118_118635

/-- Prove that there are no pairs of positive integers (x, y) such that x² + y² + x = 2x³. -/
theorem no_solutions : ∀ x y : ℕ, 0 < x → 0 < y → (x^2 + y^2 + x = 2 * x^3) → false :=
by
  sorry

end no_solutions_l118_118635


namespace parking_lot_wheels_l118_118494

-- Define the total number of wheels for each type of vehicle
def car_wheels (n : ℕ) : ℕ := n * 4
def motorcycle_wheels (n : ℕ) : ℕ := n * 2
def truck_wheels (n : ℕ) : ℕ := n * 6
def van_wheels (n : ℕ) : ℕ := n * 4

-- Number of each type of guests' vehicles
def num_cars : ℕ := 5
def num_motorcycles : ℕ := 4
def num_trucks : ℕ := 3
def num_vans : ℕ := 2

-- Number of parents' vehicles and their wheels
def parents_car_wheels : ℕ := 4
def parents_jeep_wheels : ℕ := 4

-- Summing up all the wheels
def total_wheels : ℕ :=
  car_wheels num_cars +
  motorcycle_wheels num_motorcycles +
  truck_wheels num_trucks +
  van_wheels num_vans +
  parents_car_wheels +
  parents_jeep_wheels

theorem parking_lot_wheels : total_wheels = 62 := by
  sorry

end parking_lot_wheels_l118_118494


namespace solution_set_of_inequality_l118_118699

theorem solution_set_of_inequality :
  { x : ℝ | x ≠ 5 ∧ (x * (x + 1)) / ((x - 5) ^ 3) ≥ 25 } = 
  { x : ℝ | x ≤ 5 / 3 } ∪ { x : ℝ | x > 5 } := by
  sorry

end solution_set_of_inequality_l118_118699


namespace glued_cubes_surface_area_l118_118962

theorem glued_cubes_surface_area (L l : ℝ) (h1 : L = 2) (h2 : l = L / 2) : 
  6 * L^2 + 4 * l^2 = 28 :=
by
  sorry

end glued_cubes_surface_area_l118_118962


namespace new_average_after_doubling_l118_118581

theorem new_average_after_doubling
  (avg : ℝ) (num_students : ℕ) (h_avg : avg = 40) (h_num_students : num_students = 10) :
  let total_marks := avg * num_students
  let new_total_marks := total_marks * 2
  let new_avg := new_total_marks / num_students
  new_avg = 80 :=
by
  sorry

end new_average_after_doubling_l118_118581


namespace mirella_orange_books_read_l118_118082

-- Definitions based on the conditions in a)
def purpleBookPages : ℕ := 230
def orangeBookPages : ℕ := 510
def purpleBooksRead : ℕ := 5
def extraOrangePages : ℕ := 890

-- The total number of purple pages read
def purplePagesRead := purpleBooksRead * purpleBookPages

-- The number of orange books read
def orangeBooksRead (O : ℕ) := O * orangeBookPages

-- Statement to be proved
theorem mirella_orange_books_read (O : ℕ) :
  orangeBooksRead O = purplePagesRead + extraOrangePages → O = 4 :=
by
  sorry

end mirella_orange_books_read_l118_118082


namespace combination_20_6_l118_118480

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l118_118480


namespace range_of_a_l118_118725

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1))
  ↔ (-1 < a ∧ a < 3) :=
by sorry

end range_of_a_l118_118725


namespace g_18_value_l118_118554

-- Define the function g as taking positive integers to positive integers
variable (g : ℕ+ → ℕ+)

-- Define the conditions for the function g
axiom increasing (n : ℕ+) : g (n + 1) > g n
axiom multiplicative (m n : ℕ+) : g (m * n) = g m * g n
axiom power_property (m n : ℕ+) (h : m ≠ n ∧ m ^ (n : ℕ) = n ^ (m : ℕ)) :
  g m = n ∨ g n = m

-- Prove that g(18) is 72
theorem g_18_value : g 18 = 72 :=
sorry

end g_18_value_l118_118554


namespace circle_radius_l118_118593

theorem circle_radius (A B C O : Type) (AB AC : ℝ) (OA : ℝ) (r : ℝ) 
  (h1 : AB * AC = 60)
  (h2 : OA = 8) 
  (h3 : (8 + r) * (8 - r) = 60) : r = 2 :=
sorry

end circle_radius_l118_118593


namespace perpendicular_lines_a_value_l118_118158

theorem perpendicular_lines_a_value :
  ∀ a : ℝ, 
    (∀ x y : ℝ, 2*x + a*y - 7 = 0) → 
    (∀ x y : ℝ, (a-3)*x + y + 4 = 0) → a = 2 :=
by
  sorry

end perpendicular_lines_a_value_l118_118158


namespace range_of_expression_l118_118306

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
sorry

end range_of_expression_l118_118306


namespace paper_area_difference_l118_118168

def sheet1_length : ℕ := 14
def sheet1_width : ℕ := 12
def sheet2_length : ℕ := 9
def sheet2_width : ℕ := 14

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def combined_area (length : ℕ) (width : ℕ) : ℕ := 2 * area length width

theorem paper_area_difference :
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 84 := 
by 
  sorry

end paper_area_difference_l118_118168


namespace dobarulho_problem_l118_118755

def is_divisible_by (x d : ℕ) : Prop := d ∣ x

def valid_quadruple (A B C D : ℕ) : Prop :=
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (A ≤ 8) ∧ (D > 1) ∧
  is_divisible_by (100 * A + 10 * B + C) D ∧
  is_divisible_by (100 * B + 10 * C + A) D ∧
  is_divisible_by (100 * C + 10 * A + B) D ∧
  is_divisible_by (100 * (A + 1) + 10 * C + B) D ∧
  is_divisible_by (100 * C + 10 * B + (A + 1)) D ∧
  is_divisible_by (100 * B + 10 * (A + 1) + C) D 

theorem dobarulho_problem :
  ∀ (A B C D : ℕ), valid_quadruple A B C D → 
  (A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨ 
  (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
  (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37) :=
by sorry

end dobarulho_problem_l118_118755


namespace fifth_powers_sum_eq_l118_118476

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l118_118476


namespace calc_perimeter_l118_118243

noncomputable def width (w: ℝ) (h: ℝ) : Prop :=
  h = w + 10

noncomputable def cost (P: ℝ) (rate: ℝ) (total_cost: ℝ) : Prop :=
  total_cost = P * rate

noncomputable def perimeter (w: ℝ) (P: ℝ) : Prop :=
  P = 2 * (w + (w + 10))

theorem calc_perimeter {w P : ℝ} (h_rate : ℝ) (h_total_cost : ℝ)
  (h1 : width w (w + 10))
  (h2 : cost (2 * (w + (w + 10))) h_rate h_total_cost) :
  P = 2 * (w + (w + 10)) →
  h_total_cost = 910 →
  h_rate = 6.5 →
  w = 30 →
  P = 140 :=
sorry

end calc_perimeter_l118_118243


namespace paula_travel_fraction_l118_118551

theorem paula_travel_fraction :
  ∀ (f : ℚ), 
    (∀ (L_time P_time travel_total : ℚ), 
      L_time = 70 →
      P_time = 70 * f →
      travel_total = 504 →
      (L_time + 5 * L_time + P_time + P_time = travel_total) →
      f = 3/5) :=
by
  sorry

end paula_travel_fraction_l118_118551


namespace petya_recover_x_y_l118_118396

theorem petya_recover_x_y (x y a b c d : ℝ)
    (hx_pos : x > 0) (hy_pos : y > 0)
    (ha : a = x + y) (hb : b = x - y) (hc : c = x / y) (hd : d = x * y) :
    ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a = x' + y' ∧ b = x' - y' ∧ c = x' / y' ∧ d = x' * y' :=
sorry

end petya_recover_x_y_l118_118396


namespace find_ks_l118_118427

def is_valid_function (f : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

theorem find_ks (f : ℕ → ℤ) :
  (f 2006 = 2007) →
  is_valid_function f k →
  k = 0 ∨ k = -1 :=
sorry

end find_ks_l118_118427


namespace percentage_of_girls_l118_118202

def total_students : ℕ := 100
def boys : ℕ := 50
def girls : ℕ := total_students - boys

theorem percentage_of_girls :
  (girls / total_students) * 100 = 50 := sorry

end percentage_of_girls_l118_118202


namespace sum_of_squares_ge_two_ab_l118_118560

theorem sum_of_squares_ge_two_ab (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := 
  sorry

end sum_of_squares_ge_two_ab_l118_118560


namespace determine_a_l118_118877

theorem determine_a (a : ℝ) (x1 x2 : ℝ) :
  (x1 * x1 + (2 * a - 1) * x1 + a * a = 0) ∧
  (x2 * x2 + (2 * a - 1) * x2 + a * a = 0) ∧
  ((x1 + 2) * (x2 + 2) = 11) →
  a = -1 :=
by
  sorry

end determine_a_l118_118877


namespace fewest_reciprocal_keypresses_l118_118257

theorem fewest_reciprocal_keypresses (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0) 
  (h1 : f 50 = 1 / 50) (h2 : f (1 / 50) = 50) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, (m < n) → (f^[m] 50 ≠ 50)) :=
by
  sorry

end fewest_reciprocal_keypresses_l118_118257


namespace calf_probability_l118_118026

theorem calf_probability 
  (P_B1 : ℝ := 0.6)  -- Proportion of calves from the first farm
  (P_B2 : ℝ := 0.3)  -- Proportion of calves from the second farm
  (P_B3 : ℝ := 0.1)  -- Proportion of calves from the third farm
  (P_B1_A : ℝ := 0.15)  -- Conditional probability of a calf weighing more than 300 kg given it is from the first farm
  (P_B2_A : ℝ := 0.25)  -- Conditional probability of a calf weighing more than 300 kg given it is from the second farm
  (P_B3_A : ℝ := 0.35)  -- Conditional probability of a calf weighing more than 300 kg given it is from the third farm)
  (P_A : ℝ := P_B1 * P_B1_A + P_B2 * P_B2_A + P_B3 * P_B3_A) : 
  P_B3 * P_B3_A / P_A = 0.175 := 
by
  sorry

end calf_probability_l118_118026


namespace decompose_zero_l118_118891

theorem decompose_zero (a : ℤ) : 0 = 0 * a := by
  sorry

end decompose_zero_l118_118891


namespace larger_angle_measure_l118_118510

-- Defining all conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90

def angle_ratio (a b : ℝ) : Prop := a / b = 5 / 4

-- Main proof statement
theorem larger_angle_measure (a b : ℝ) (h1 : is_complementary a b) (h2 : angle_ratio a b) : a = 50 :=
by
  sorry

end larger_angle_measure_l118_118510


namespace horner_method_v1_l118_118070

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_v1 (x : ℝ) (h : x = 5) : 
  ((4 * x + 2) * x + 3.5) = 22 := by
  rw [h]
  norm_num
  sorry

end horner_method_v1_l118_118070


namespace product_of_real_roots_of_equation_l118_118830

theorem product_of_real_roots_of_equation : 
  ∀ x : ℝ, (x^4 + (x - 4)^4 = 32) → x = 2 :=
sorry

end product_of_real_roots_of_equation_l118_118830


namespace clothes_washer_final_price_l118_118636

theorem clothes_washer_final_price
  (P : ℝ) (d1 d2 d3 : ℝ)
  (hP : P = 500)
  (hd1 : d1 = 0.10)
  (hd2 : d2 = 0.20)
  (hd3 : d3 = 0.05) :
  (P * (1 - d1) * (1 - d2) * (1 - d3)) / P = 0.684 :=
by
  sorry

end clothes_washer_final_price_l118_118636


namespace geometric_sequence_a2_value_l118_118948

theorem geometric_sequence_a2_value
  (a : ℕ → ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, a (n + 1) = a 1 * (a 2) ^ n)
  (h_sum : a1 + a2 + a3 = 18)
  (h_inverse_sum : 1/a1 + 1/a2 + 1/a3 = 2)
  : a2 = 3 :=
sorry

end geometric_sequence_a2_value_l118_118948


namespace find_two_digit_integers_l118_118172

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem find_two_digit_integers
    (a b : ℕ) :
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    (a = b + 12 ∨ b = a + 12) ∧
    (a / 10 = b / 10 ∨ a % 10 = b % 10) ∧
    (sum_of_digits a = sum_of_digits b + 3 ∨ sum_of_digits b = sum_of_digits a + 3) :=
sorry

end find_two_digit_integers_l118_118172


namespace largest_sum_of_distinct_factors_l118_118365

theorem largest_sum_of_distinct_factors (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) (h_product : A * B * C = 3003) :
  A + B + C ≤ 105 :=
sorry  -- Proof is not required, just the statement.

end largest_sum_of_distinct_factors_l118_118365


namespace calculate_expression_l118_118442

variable (x y : ℝ)

theorem calculate_expression : (-2 * x^2 * y) ^ 2 = 4 * x^4 * y^2 := by
  sorry

end calculate_expression_l118_118442


namespace radishes_in_first_basket_l118_118411

theorem radishes_in_first_basket :
  ∃ x : ℕ, ∃ y : ℕ, x + y = 88 ∧ y = x + 14 ∧ x = 37 :=
by
  -- Proof goes here
  sorry

end radishes_in_first_basket_l118_118411


namespace paths_A_to_D_through_B_and_C_l118_118555

-- Define points and paths in a grid
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 3⟩
def C : Point := ⟨6, 4⟩
def D : Point := ⟨9, 6⟩

-- Calculate binomial coefficient
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Number of paths from one point to another in a grid
def numPaths (p1 p2 : Point) : ℕ :=
  let stepsRight := p2.x - p1.x
  let stepsDown := p2.y - p1.y
  choose (stepsRight + stepsDown) stepsRight

theorem paths_A_to_D_through_B_and_C : numPaths A B * numPaths B C * numPaths C D = 500 := by
  -- Using the conditions provided:
  -- numPaths A B = choose 5 2 = 10
  -- numPaths B C = choose 5 1 = 5
  -- numPaths C D = choose 5 2 = 10
  -- Therefore, numPaths A B * numPaths B C * numPaths C D = 10 * 5 * 10 = 500
  sorry

end paths_A_to_D_through_B_and_C_l118_118555


namespace number_of_rabbits_l118_118180

theorem number_of_rabbits (C D : ℕ) (hC : C = 49) (hD : D = 37) (h : D + R = C + 9) :
  R = 21 :=
by
    sorry

end number_of_rabbits_l118_118180


namespace man_l118_118457

theorem man's_age_twice_son's_age_in_2_years
  (S : ℕ) (M : ℕ) (Y : ℕ)
  (h1 : M = S + 24)
  (h2 : S = 22)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 := by
  sorry

end man_l118_118457


namespace proportion_of_boys_correct_l118_118419

noncomputable def proportion_of_boys : ℚ :=
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let expected_children := 3 -- (2 boys and 1 girl)
  let expected_boys := 2 -- Expected number of boys in each family
  
  expected_boys / expected_children

theorem proportion_of_boys_correct : proportion_of_boys = 2 / 3 := by
  sorry

end proportion_of_boys_correct_l118_118419


namespace areas_of_geometric_figures_with_equal_perimeter_l118_118519

theorem areas_of_geometric_figures_with_equal_perimeter (l : ℝ) (h : (l > 0)) :
  let s1 := l^2 / (4 * Real.pi)
  let s2 := l^2 / 16
  let s3 := (Real.sqrt 3) * l^2 / 36
  s1 > s2 ∧ s2 > s3 := by
  sorry

end areas_of_geometric_figures_with_equal_perimeter_l118_118519


namespace partition_exists_iff_l118_118075

theorem partition_exists_iff (k : ℕ) :
  (∃ (A B : Finset ℕ), A ∪ B = Finset.range (1990 + k + 1) ∧ A ∩ B = ∅ ∧ 
  (A.sum id + 1990 * A.card = B.sum id + 1990 * B.card)) ↔ 
  (k % 4 = 3 ∨ (k % 4 = 0 ∧ k ≥ 92)) :=
by
  sorry

end partition_exists_iff_l118_118075


namespace zion_dad_age_difference_in_10_years_l118_118301

/-
Given:
1. Zion's age is 8 years.
2. Zion's dad's age is 3 more than 4 times Zion's age.
Prove:
In 10 years, the difference in age between Zion's dad and Zion will be 27 years.
-/

theorem zion_dad_age_difference_in_10_years :
  let zion_age := 8
  let dad_age := 4 * zion_age + 3
  (dad_age + 10) - (zion_age + 10) = 27 := by
  sorry

end zion_dad_age_difference_in_10_years_l118_118301


namespace minimum_b_value_l118_118842

theorem minimum_b_value (k : ℕ) (x y z b : ℕ) (h1 : x = 3 * k) (h2 : y = 4 * k)
  (h3 : z = 7 * k) (h4 : y = 15 * b - 5) (h5 : ∀ n : ℕ, n = 4 * k + 5 → n % 15 = 0) : 
  b = 3 :=
by
  sorry

end minimum_b_value_l118_118842


namespace triangle_ABC_is_right_triangle_l118_118208

-- Define the triangle and the given conditions
variable (a b c : ℝ)
variable (h1 : a + c = 2*b)
variable (h2 : c - a = 1/2*b)

-- State the problem
theorem triangle_ABC_is_right_triangle : c^2 = a^2 + b^2 :=
by
  sorry

end triangle_ABC_is_right_triangle_l118_118208


namespace mark_exceeded_sugar_intake_by_100_percent_l118_118823

-- Definitions of the conditions
def softDrinkCalories : ℕ := 2500
def sugarPercentage : ℝ := 0.05
def caloriesPerCandy : ℕ := 25
def numCandyBars : ℕ := 7
def recommendedSugarIntake : ℕ := 150

-- Calculating the amount of added sugar in the soft drink
def addedSugarSoftDrink : ℝ := sugarPercentage * softDrinkCalories

-- Calculating the total added sugar from the candy bars
def addedSugarCandyBars : ℕ := numCandyBars * caloriesPerCandy

-- Summing the added sugar from the soft drink and the candy bars
def totalAddedSugar : ℝ := addedSugarSoftDrink + (addedSugarCandyBars : ℝ)

-- Calculate the excess intake of added sugar over the recommended amount
def excessSugarIntake : ℝ := totalAddedSugar - (recommendedSugarIntake : ℝ)

-- Prove that the percentage by which Mark exceeded the recommended intake of added sugar is 100%
theorem mark_exceeded_sugar_intake_by_100_percent :
  (excessSugarIntake / (recommendedSugarIntake : ℝ)) * 100 = 100 :=
by
  sorry

end mark_exceeded_sugar_intake_by_100_percent_l118_118823


namespace juanitas_dessert_cost_is_correct_l118_118084

noncomputable def brownie_cost := 2.50
noncomputable def regular_scoop_cost := 1.00
noncomputable def premium_scoop_cost := 1.25
noncomputable def deluxe_scoop_cost := 1.50
noncomputable def syrup_cost := 0.50
noncomputable def nuts_cost := 1.50
noncomputable def whipped_cream_cost := 0.75
noncomputable def cherry_cost := 0.25
noncomputable def discount_tuesday := 0.10

noncomputable def total_cost_of_juanitas_dessert :=
    let discounted_brownie := brownie_cost * (1 - discount_tuesday)
    let ice_cream_cost := 2 * regular_scoop_cost + premium_scoop_cost
    let syrup_total := 2 * syrup_cost
    let additional_toppings := nuts_cost + whipped_cream_cost + cherry_cost
    discounted_brownie + ice_cream_cost + syrup_total + additional_toppings
   
theorem juanitas_dessert_cost_is_correct:
  total_cost_of_juanitas_dessert = 9.00 := by
  sorry

end juanitas_dessert_cost_is_correct_l118_118084


namespace find_angle_A_find_sum_b_c_l118_118499

-- Given the necessary conditions
variables (a b c : ℝ)
variables (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

-- Assuming necessary trigonometric identities
axiom sin_squared_add_cos_squared : ∀ (x : ℝ), sin x * sin x + cos x * cos x = 1
axiom cos_sum : ∀ (x y : ℝ), cos (x + y) = cos x * cos y - sin x * sin y

-- Condition: 2 sin^2(A) + 3 cos(B+C) = 0
axiom condition1 : 2 * sin A * sin A + 3 * cos (B + C) = 0

-- Condition: The area of the triangle is S = 5 √3
axiom condition2 : 1 / 2 * b * c * sin A = 5 * Real.sqrt 3

-- Condition: The length of side a = √21
axiom condition3 : a = Real.sqrt 21

-- Part (1): Prove the measure of angle A
theorem find_angle_A : A = π / 3 :=
sorry

-- Part (2): Given S = 5√3 and a = √21, find b + c.
theorem find_sum_b_c : b + c = 9 :=
sorry

end find_angle_A_find_sum_b_c_l118_118499


namespace parabola_translation_l118_118303

theorem parabola_translation :
  (∀ x : ℝ, y = x^2 → y' = (x - 1)^2 + 3) :=
sorry

end parabola_translation_l118_118303


namespace min_distance_to_line_l118_118307

-- Given that a point P(x, y) lies on the line x - y - 1 = 0
-- We need to prove that the minimum value of (x - 2)^2 + (y - 2)^2 is 1/2
theorem min_distance_to_line (x y: ℝ) (h: x - y - 1 = 0) :
  ∃ P : ℝ, P = (x - 2)^2 + (y - 2)^2 ∧ P = 1 / 2 :=
by
  sorry

end min_distance_to_line_l118_118307


namespace calculate_a5_l118_118250

variable {a1 : ℝ} -- geometric sequence first term
variable {a : ℕ → ℝ} -- geometric sequence
variable {n : ℕ} -- sequence index
variable {r : ℝ} -- common ratio

-- Definitions based on the given conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a1 * r ^ n

-- Given conditions
axiom common_ratio_is_two : r = 2
axiom product_condition : a 2 * a 10 = 16 -- indices offset by 1, so a3 = a 2 and a11 = a 10
axiom positive_terms : ∀ n, a n > 0

-- Goal: calculate a 4
theorem calculate_a5 : a 4 = 1 :=
sorry

end calculate_a5_l118_118250


namespace hayley_friends_l118_118883

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends_l118_118883


namespace triangle_area_example_l118_118505

def point := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example :
  let A : point := (3, -2)
  let B : point := (12, 5)
  let C : point := (3, 8)
  triangle_area A B C = 45 :=
by
  sorry

end triangle_area_example_l118_118505


namespace tan_identity_l118_118896

theorem tan_identity :
  let t5 := Real.tan (Real.pi / 36) -- 5 degrees in radians
  let t40 := Real.tan (Real.pi / 9)  -- 40 degrees in radians
  t5 + t40 + t5 * t40 = 1 :=
by
  sorry

end tan_identity_l118_118896


namespace eq_of_operation_l118_118900

theorem eq_of_operation {x : ℝ} (h : 60 + 5 * 12 / (x / 3) = 61) : x = 180 :=
by
  sorry

end eq_of_operation_l118_118900


namespace letters_with_dot_not_line_l118_118643

-- Definitions from conditions
def D_inter_S : ℕ := 23
def S : ℕ := 42
def Total_letters : ℕ := 70

-- Problem statement
theorem letters_with_dot_not_line : (Total_letters - S - D_inter_S) = 5 :=
by sorry

end letters_with_dot_not_line_l118_118643


namespace trig_evaluation_l118_118429

noncomputable def sin30 := 1 / 2
noncomputable def cos45 := Real.sqrt 2 / 2
noncomputable def tan30 := Real.sqrt 3 / 3
noncomputable def sin60 := Real.sqrt 3 / 2

theorem trig_evaluation : 4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end trig_evaluation_l118_118429


namespace find_b_l118_118311

noncomputable def a : ℂ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℂ := sorry

-- Given conditions
axiom sum_eq : a + b + c = 4
axiom prod_pairs_eq : a * b + b * c + c * a = 5
axiom prod_triple_eq : a * b * c = 6

-- Prove that b = 1
theorem find_b : b = 1 :=
by
  -- Proof omitted
  sorry

end find_b_l118_118311


namespace problems_per_page_l118_118641

theorem problems_per_page (total_problems : ℕ) (percent_solved : ℝ) (pages_left : ℕ)
  (h_total : total_problems = 550)
  (h_percent : percent_solved = 0.65)
  (h_pages : pages_left = 3) :
  (total_problems - Nat.ceil (percent_solved * total_problems)) / pages_left = 64 := by
  sorry

end problems_per_page_l118_118641


namespace hose_Z_fill_time_l118_118633

theorem hose_Z_fill_time (P X Y Z : ℝ) (h1 : X + Y = P / 3) (h2 : Y = P / 9) (h3 : X + Z = P / 4) (h4 : X + Y + Z = P / 2.5) : Z = P / 15 :=
sorry

end hose_Z_fill_time_l118_118633


namespace percentage_error_l118_118717

theorem percentage_error (x : ℝ) : ((x * 3 - x / 5) / (x * 3) * 100) = 93.33 := 
  sorry

end percentage_error_l118_118717


namespace movie_hours_sum_l118_118234

noncomputable def total_movie_hours 
  (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : ℕ :=
  Joyce + Michael + Nikki + Ryn + Sam

theorem movie_hours_sum (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : 
  total_movie_hours Michael Joyce Nikki Ryn Sam h1 h2 h3 h4 h5 = 94 :=
by 
  -- The actual proof will go here, to demonstrate the calculations resulting in 94 hours
  sorry

end movie_hours_sum_l118_118234


namespace original_faculty_members_l118_118992

theorem original_faculty_members
  (x : ℝ) (h : 0.87 * x = 195) : x = 224 := sorry

end original_faculty_members_l118_118992


namespace focus_of_parabola_l118_118181

-- Problem statement
theorem focus_of_parabola (x y : ℝ) : (2 * x^2 = -y) → (focus_coordinates = (0, -1 / 8)) :=
by
  sorry

end focus_of_parabola_l118_118181


namespace volume_of_intersection_of_two_perpendicular_cylinders_l118_118863

theorem volume_of_intersection_of_two_perpendicular_cylinders (R : ℝ) : 
  ∃ V : ℝ, V = (16 / 3) * R^3 := 
sorry

end volume_of_intersection_of_two_perpendicular_cylinders_l118_118863


namespace range_of_a_l118_118812

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * (a + 2)

-- We are given that for f to have both maximum and minimum values, f' must have two distinct roots
-- Thus we translate the mathematical condition to the discriminant of f' being greater than 0
def discriminant_greater_than_zero (a : ℝ) : Prop :=
  (6 * a)^2 - 4 * 3 * 3 * (a + 2) > 0

-- Finally, we want to prove that this simplifies to a condition on a
theorem range_of_a (a : ℝ) : discriminant_greater_than_zero a ↔ (a > 2 ∨ a < -1) :=
by
  -- Write the proof here
  sorry

end range_of_a_l118_118812


namespace max_marks_l118_118531

theorem max_marks (M : ℝ) (h1 : 0.42 * M = 80) : M = 190 :=
by
  sorry

end max_marks_l118_118531


namespace suff_but_not_necess_condition_l118_118368

theorem suff_but_not_necess_condition (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : a + a * b < 0 :=
  sorry

end suff_but_not_necess_condition_l118_118368


namespace find_t_l118_118751

open Real

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (t : ℝ) :
  let m := (t + 1, 1)
  let n := (t + 2, 2)
  dot_product (vector_add m n) (vector_sub m n) = 0 → 
  t = -3 :=
by
  intro h
  sorry

end find_t_l118_118751


namespace purely_imaginary_a_eq_1_fourth_quadrant_a_range_l118_118996

-- Definitions based on given conditions
def z (a : ℝ) := (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I

-- Purely imaginary proof statement
theorem purely_imaginary_a_eq_1 (a : ℝ) 
  (hz : (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I = (0 : ℂ) + (a^2 - 5 * a - 6) * Complex.I) :
  a = 1 := by 
  sorry

-- Fourth quadrant proof statement
theorem fourth_quadrant_a_range (a : ℝ) 
  (hz1 : a^2 - 7 * a + 6 > 0) 
  (hz2 : a^2 - 5 * a - 6 < 0) : 
  -1 < a ∧ a < 1 := by 
  sorry

end purely_imaginary_a_eq_1_fourth_quadrant_a_range_l118_118996


namespace child_ticket_cost_l118_118238

variable (A C : ℕ) -- A stands for the number of adults, C stands for the cost of one child's ticket

theorem child_ticket_cost 
  (number_of_adults : ℕ) 
  (number_of_children : ℕ) 
  (cost_concessions : ℕ) 
  (total_cost_trip : ℕ)
  (cost_adult_ticket : ℕ) 
  (ticket_costs : ℕ) 
  (total_adult_cost : ℕ) 
  (remaining_ticket_cost : ℕ) 
  (child_ticket : ℕ) :
  number_of_adults = 5 →
  number_of_children = 2 →
  cost_concessions = 12 →
  total_cost_trip = 76 →
  cost_adult_ticket = 10 →
  ticket_costs = total_cost_trip - cost_concessions →
  total_adult_cost = number_of_adults * cost_adult_ticket →
  remaining_ticket_cost = ticket_costs - total_adult_cost →
  child_ticket = remaining_ticket_cost / number_of_children →
  C = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Adding sorry since the proof is not required
  sorry

end child_ticket_cost_l118_118238


namespace asymptote_sum_l118_118994

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

def holes := 0 -- a
def vertical_asymptotes := 2 -- b
def horizontal_asymptotes := 1 -- c
def oblique_asymptotes := 0 -- d

theorem asymptote_sum : holes + 2 * vertical_asymptotes + 3 * horizontal_asymptotes + 4 * oblique_asymptotes = 7 :=
by
  unfold holes vertical_asymptotes horizontal_asymptotes oblique_asymptotes
  norm_num

end asymptote_sum_l118_118994


namespace police_officer_placement_l118_118714

-- The given problem's conditions
def intersections : Finset String := {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}

def streets : List (Finset String) := [
    {"A", "B", "C", "D"},        -- Horizontal streets
    {"E", "F", "G"},
    {"H", "I", "J", "K"},
    {"A", "E", "H"},             -- Vertical streets
    {"B", "F", "I"},
    {"D", "G", "J"},
    {"H", "F", "C"},             -- Diagonal streets
    {"C", "G", "K"}
]

def chosen_intersections : Finset String := {"B", "G", "H"}

-- Proof problem
theorem police_officer_placement :
  ∀ street ∈ streets, ∃ p ∈ chosen_intersections, p ∈ street := by
  sorry

end police_officer_placement_l118_118714


namespace least_number_to_subtract_l118_118237

theorem least_number_to_subtract (n : ℕ) (h1 : n = 157632)
  (h2 : ∃ k : ℕ, k = 12 * 18 * 24 / (gcd 12 (gcd 18 24)) ∧ k ∣ n - 24) :
  n - 24 = 24 := 
sorry

end least_number_to_subtract_l118_118237


namespace cube_sum_equals_36_l118_118527

variable {a b c k : ℝ}

theorem cube_sum_equals_36 (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (heq : (a^3 - 12) / a = (b^3 - 12) / b)
    (heq_another : (b^3 - 12) / b = (c^3 - 12) / c) :
    a^3 + b^3 + c^3 = 36 := by
  sorry

end cube_sum_equals_36_l118_118527


namespace shelves_needed_is_five_l118_118685

-- Definitions for the conditions
def initial_bears : Nat := 15
def additional_bears : Nat := 45
def bears_per_shelf : Nat := 12

-- Adding the number of bears received to the initial stock
def total_bears : Nat := initial_bears + additional_bears

-- Calculating the number of shelves used
def shelves_used : Nat := total_bears / bears_per_shelf

-- Statement to prove
theorem shelves_needed_is_five : shelves_used = 5 :=
by
  -- Insert specific step only if necessary, otherwise use sorry
  sorry

end shelves_needed_is_five_l118_118685


namespace equation_of_line_passing_through_A_equation_of_circle_l118_118050

variable {α β γ : ℝ}
variable {a b c u v w : ℝ}
variable (A : ℝ × ℝ × ℝ) -- Barycentric coordinates of point A

-- Statement for the equation of a line passing through point A in barycentric coordinates
theorem equation_of_line_passing_through_A (A : ℝ × ℝ × ℝ) : 
  ∃ (u v w : ℝ), u * α + v * β + w * γ = 0 := by
  sorry

-- Statement for the equation of a circle in barycentric coordinates
theorem equation_of_circle {u v w : ℝ} :
  -a^2 * β * γ - b^2 * γ * α - c^2 * α * β +
  (u * α + v * β + w * γ) * (α + β + γ) = 0 := by
  sorry

end equation_of_line_passing_through_A_equation_of_circle_l118_118050


namespace least_integer_of_sum_in_ratio_l118_118021

theorem least_integer_of_sum_in_ratio (a b c : ℕ) (h1 : a + b + c = 90) (h2 : a * 3 = b * 2) (h3 : a * 5 = c * 2) : a = 18 :=
by
  sorry

end least_integer_of_sum_in_ratio_l118_118021


namespace combined_weight_l118_118882

-- Definition of conditions
def regular_dinosaur_weight := 800
def five_regular_dinosaurs_weight := 5 * regular_dinosaur_weight
def barney_weight := five_regular_dinosaurs_weight + 1500

-- Statement to prove
theorem combined_weight (h1: five_regular_dinosaurs_weight = 5 * regular_dinosaur_weight)
                        (h2: barney_weight = five_regular_dinosaurs_weight + 1500) : 
        (barney_weight + five_regular_dinosaurs_weight = 9500) :=
by
    sorry

end combined_weight_l118_118882


namespace friend_balloon_count_l118_118022

theorem friend_balloon_count (you_balloons friend_balloons : ℕ) (h1 : you_balloons = 7) (h2 : you_balloons = friend_balloons + 2) : friend_balloons = 5 :=
by
  sorry

end friend_balloon_count_l118_118022


namespace eq1_solution_eq2_solution_l118_118493

theorem eq1_solution (x : ℝ) : (x - 1)^2 - 1 = 15 ↔ x = 5 ∨ x = -3 := by sorry

theorem eq2_solution (x : ℝ) : (1 / 3) * (x + 3)^3 - 9 = 0 ↔ x = 0 := by sorry

end eq1_solution_eq2_solution_l118_118493


namespace number_of_trees_is_correct_l118_118680

-- Define the conditions
def length_of_plot := 120
def width_of_plot := 70
def distance_between_trees := 5

-- Define the calculated number of intervals along each side
def intervals_along_length := length_of_plot / distance_between_trees
def intervals_along_width := width_of_plot / distance_between_trees

-- Define the number of trees along each side including the boundaries
def trees_along_length := intervals_along_length + 1
def trees_along_width := intervals_along_width + 1

-- Define the total number of trees
def total_number_of_trees := trees_along_length * trees_along_width

-- The theorem we want to prove
theorem number_of_trees_is_correct : total_number_of_trees = 375 :=
by sorry

end number_of_trees_is_correct_l118_118680


namespace right_triangle_area_l118_118672

variable {AB BC AC : ℕ}

theorem right_triangle_area : ∀ (AB BC AC : ℕ), (AC = 50) → (AB + BC = 70) → (AB^2 + BC^2 = AC^2) → (1 / 2) * AB * BC = 300 :=
by
  intros AB BC AC h1 h2 h3
  -- Proof steps will be added here
  sorry

end right_triangle_area_l118_118672


namespace combined_percent_of_6th_graders_l118_118520

theorem combined_percent_of_6th_graders (num_students_pineview : ℕ) 
                                        (percent_6th_pineview : ℝ) 
                                        (num_students_oakridge : ℕ)
                                        (percent_6th_oakridge : ℝ)
                                        (num_students_maplewood : ℕ)
                                        (percent_6th_maplewood : ℝ) 
                                        (total_students : ℝ) :
    num_students_pineview = 150 →
    percent_6th_pineview = 0.15 →
    num_students_oakridge = 180 →
    percent_6th_oakridge = 0.17 →
    num_students_maplewood = 170 →
    percent_6th_maplewood = 0.15 →
    total_students = 500 →
    ((percent_6th_pineview * num_students_pineview) + 
     (percent_6th_oakridge * num_students_oakridge) + 
     (percent_6th_maplewood * num_students_maplewood)) / 
    total_students * 100 = 15.72 :=
by
  sorry

end combined_percent_of_6th_graders_l118_118520


namespace total_weight_of_8_moles_of_BaCl2_l118_118264

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular weight of BaCl2
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

-- Define the number of moles
def moles : ℝ := 8

-- Define the total weight calculation
def total_weight : ℝ := molecular_weight_BaCl2 * moles

-- The theorem to prove
theorem total_weight_of_8_moles_of_BaCl2 : total_weight = 1665.84 :=
by sorry

end total_weight_of_8_moles_of_BaCl2_l118_118264


namespace sum_mod_9_l118_118774

theorem sum_mod_9 (x y z : ℕ) (h1 : x < 9) (h2 : y < 9) (h3 : z < 9) 
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : (x * y * z) % 9 = 1) (h8 : (7 * z) % 9 = 4) (h9 : (8 * y) % 9 = (5 + y) % 9) :
  (x + y + z) % 9 = 7 := 
by {
  sorry
}

end sum_mod_9_l118_118774


namespace sweets_ratio_l118_118101

theorem sweets_ratio (total_sweets : ℕ) (mother_ratio : ℚ) (eldest_sweets second_sweets : ℕ)
  (h1 : total_sweets = 27) (h2 : mother_ratio = 1 / 3) (h3 : eldest_sweets = 8) (h4 : second_sweets = 6) :
  let mother_sweets := mother_ratio * total_sweets
  let remaining_sweets := total_sweets - mother_sweets
  let other_sweets := eldest_sweets + second_sweets
  let youngest_sweets := remaining_sweets - other_sweets
  youngest_sweets / eldest_sweets = 1 / 2 :=
by
  sorry

end sweets_ratio_l118_118101


namespace mario_savings_percentage_l118_118489

-- Define the price of one ticket
def ticket_price : ℝ := sorry

-- Define the conditions
-- Condition 1: 5 tickets can be purchased for the usual price of 3 tickets
def price_for_5_tickets := 3 * ticket_price

-- Condition 2: Mario bought 5 tickets
def mario_tickets := 5 * ticket_price

-- Condition 3: Usual price for 5 tickets
def usual_price_5_tickets := 5 * ticket_price

-- Calculate the amount saved
def amount_saved := usual_price_5_tickets - price_for_5_tickets

theorem mario_savings_percentage
  (ticket_price: ℝ)
  (h1 : price_for_5_tickets = 3 * ticket_price)
  (h2 : mario_tickets = 5 * ticket_price)
  (h3 : usual_price_5_tickets = 5 * ticket_price)
  (h4 : amount_saved = usual_price_5_tickets - price_for_5_tickets):
  (amount_saved / usual_price_5_tickets) * 100 = 40 := 
by {
    -- Placeholder
    sorry
}

end mario_savings_percentage_l118_118489


namespace factor_polynomial_l118_118559

-- Statement of the proof problem
theorem factor_polynomial (x y z : ℝ) :
    x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
    (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) :=
by
  sorry

end factor_polynomial_l118_118559


namespace man_l118_118277

noncomputable def speed_of_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_100_meters_downstream : ℝ := 19.99840012798976 -- in seconds
noncomputable def distance_covered : ℝ := 0.1 -- in kilometers (100 meters)

noncomputable def speed_in_still_water : ℝ :=
  (distance_covered / (time_to_cover_100_meters_downstream / 3600)) - speed_of_current

theorem man's_speed_in_still_water :
  speed_in_still_water = 14.9997120913593 :=
  by
    sorry

end man_l118_118277


namespace largest_angle_in_triangle_PQR_l118_118392

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l118_118392


namespace eval_f_a_plus_1_l118_118653

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the condition
axiom a : ℝ

-- State the theorem to be proven
theorem eval_f_a_plus_1 : f (a + 1) = a^2 + 2*a + 1 :=
by
  sorry

end eval_f_a_plus_1_l118_118653


namespace sq_sum_ge_one_third_l118_118678

theorem sq_sum_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sq_sum_ge_one_third_l118_118678


namespace centipede_and_earthworm_meeting_time_l118_118638

noncomputable def speed_centipede : ℚ := 5 / 3
noncomputable def speed_earthworm : ℚ := 5 / 2
noncomputable def initial_gap : ℚ := 20

theorem centipede_and_earthworm_meeting_time : 
  ∃ t : ℚ, (5 / 2) * t = initial_gap + (5 / 3) * t ∧ t = 24 := 
by
  sorry

end centipede_and_earthworm_meeting_time_l118_118638


namespace maximum_shapes_in_grid_l118_118546

-- Define the grid size and shape properties
def grid_width : Nat := 8
def grid_height : Nat := 14
def shape_area : Nat := 3
def shape_grid_points : Nat := 8

-- Define the total grid points in the rectangular grid
def total_grid_points : Nat := (grid_width + 1) * (grid_height + 1)

-- Define the question and the condition that needs to be proved
theorem maximum_shapes_in_grid : (total_grid_points / shape_grid_points) = 16 := by
  sorry

end maximum_shapes_in_grid_l118_118546


namespace grade_representation_l118_118062

theorem grade_representation :
  (8, 1) = (8, 1) :=
by
  sorry

end grade_representation_l118_118062


namespace inequality_real_equation_positive_integers_solution_l118_118890

-- Prove the inequality for real numbers a and b
theorem inequality_real (a b : ℝ) :
  (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * ((2 * a + 1) * (3 * b + 1)) :=
  sorry

-- Find all positive integers n and p such that the equation holds
theorem equation_positive_integers_solution :
  ∃ (n p : ℕ), 0 < n ∧ 0 < p ∧ (n^2 + 1) * (p^2 + 1) + 45 = 2 * ((2 * n + 1) * (3 * p + 1)) ∧ n = 2 ∧ p = 2 :=
  sorry

end inequality_real_equation_positive_integers_solution_l118_118890


namespace martha_total_butterflies_l118_118837

variable (Yellow Blue Black : ℕ)

def butterfly_equations (Yellow Blue Black : ℕ) : Prop :=
  (Blue = 2 * Yellow) ∧ (Blue = 6) ∧ (Black = 10)

theorem martha_total_butterflies 
  (h : butterfly_equations Yellow Blue Black) : 
  (Yellow + Blue + Black = 19) :=
by
  sorry

end martha_total_butterflies_l118_118837


namespace emerie_dimes_count_l118_118434

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count_l118_118434


namespace remainder_31_l118_118081

theorem remainder_31 (x : ℤ) (h : x % 62 = 7) : (x + 11) % 31 = 18 := by
  sorry

end remainder_31_l118_118081


namespace rhombus_area_600_l118_118885

noncomputable def area_of_rhombus (x y : ℝ) : ℝ := (x * y) * 2

theorem rhombus_area_600 (x y : ℝ) (qx qy : ℝ)
  (hx : x = 15) (hy : y = 20)
  (hr1 : qx = 15) (hr2 : qy = 20)
  (h_ratio : qy / qx = 4 / 3) :
  area_of_rhombus (2 * (x + y - 2)) (x + y) = 600 :=
by
  rw [hx, hy]
  sorry

end rhombus_area_600_l118_118885


namespace minimum_value_h_at_a_eq_2_range_of_a_l118_118897

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x

theorem minimum_value_h_at_a_eq_2 : ∃ x, h 2 x = 3 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 1, h a x ≥ 1) ↔ a ≥ 1 :=
sorry

end minimum_value_h_at_a_eq_2_range_of_a_l118_118897


namespace side_of_square_is_25_l118_118598

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l118_118598


namespace sports_club_membership_l118_118273

theorem sports_club_membership :
  ∀ (total T B_and_T neither : ℕ),
    total = 30 → 
    T = 19 →
    B_and_T = 9 →
    neither = 2 →
  ∃ (B : ℕ), B = 18 :=
by
  intros total T B_and_T neither ht hT hBandT hNeither
  let B := total - neither - T + B_and_T
  use B
  sorry

end sports_club_membership_l118_118273


namespace negation_proposition_l118_118159

theorem negation_proposition (p : Prop) : 
  (∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ ¬ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by
  sorry

end negation_proposition_l118_118159


namespace expression_for_C_value_of_C_l118_118639

variables (x y : ℝ)

-- Definitions based on the given conditions
def A := x^2 - 2 * x * y + y^2
def B := x^2 + 2 * x * y + y^2

-- The algebraic expression for C
def C := - x^2 + 10 * x * y - y^2

-- Prove that the expression for C is correct
theorem expression_for_C (h : 3 * A x y - 2 * B x y + C x y = 0) : 
  C x y = - x^2 + 10 * x * y - y^2 := 
by {
  sorry
}

-- Prove the value of C when x = 1/2 and y = -2
theorem value_of_C : C (1/2) (-2) = -57/4 :=
by {
  sorry
}

end expression_for_C_value_of_C_l118_118639


namespace pure_imaginary_x_l118_118470

theorem pure_imaginary_x (x : ℝ) (h: (x - 2008) = 0) : x = 2008 :=
by
  sorry

end pure_imaginary_x_l118_118470


namespace distance_to_origin_eq_three_l118_118914

theorem distance_to_origin_eq_three :
  let P := (1, 2, 2)
  let origin := (0, 0, 0)
  dist P origin = 3 := by
  sorry

end distance_to_origin_eq_three_l118_118914


namespace problem1_problem2_problem3_l118_118023

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l118_118023


namespace solve_problem_l118_118866

theorem solve_problem :
  ∃ (x y : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y = 5 :=
by
  sorry

end solve_problem_l118_118866


namespace area_dodecagon_equals_rectangle_l118_118937

noncomputable def area_regular_dodecagon (r : ℝ) : ℝ := 3 * r^2

theorem area_dodecagon_equals_rectangle (r : ℝ) :
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  area_dodecagon = area_rectangle :=
by
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  show area_dodecagon = area_rectangle
  sorry

end area_dodecagon_equals_rectangle_l118_118937


namespace cubic_roots_expression_l118_118965

theorem cubic_roots_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -1) (h3 : a * b * c = 2) :
  2 * a * (b - c) ^ 2 + 2 * b * (c - a) ^ 2 + 2 * c * (a - b) ^ 2 = -36 :=
by
  sorry

end cubic_roots_expression_l118_118965


namespace find_natural_number_n_l118_118295

theorem find_natural_number_n : 
  ∃ (n : ℕ), (∃ k : ℕ, n + 15 = k^2) ∧ (∃ m : ℕ, n - 14 = m^2) ∧ n = 210 :=
by
  sorry

end find_natural_number_n_l118_118295


namespace shop_width_correct_l118_118451

-- Definition of the shop's monthly rent
def monthly_rent : ℝ := 2400

-- Definition of the shop's length in feet
def shop_length : ℝ := 10

-- Definition of the annual rent per square foot
def annual_rent_per_sq_ft : ℝ := 360

-- The mathematical assertion that the width of the shop is 8 feet
theorem shop_width_correct (width : ℝ) :
  (monthly_rent * 12) / annual_rent_per_sq_ft / shop_length = width :=
by
  sorry

end shop_width_correct_l118_118451


namespace sum_of_x_intercepts_l118_118422

theorem sum_of_x_intercepts (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (5 : ℤ) * (3 : ℤ) = (a : ℤ) * (b : ℤ)) : 
  ((-5 : ℤ) / (a : ℤ)) + ((-5 : ℤ) / (3 : ℤ)) + ((-1 : ℤ) / (1 : ℤ)) + ((-1 : ℤ) / (15 : ℤ)) = -8 := 
by 
  sorry

end sum_of_x_intercepts_l118_118422


namespace determinant_trig_matrix_eq_one_l118_118676

theorem determinant_trig_matrix_eq_one (α θ : ℝ) :
  Matrix.det ![
  ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
  ![Real.sin θ, -Real.cos θ, 0],
  ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 :=
by
  sorry

end determinant_trig_matrix_eq_one_l118_118676


namespace least_number_added_to_divide_l118_118756

-- Definitions of conditions
def lcm_three_five_seven_eight : ℕ := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 8
def remainder_28523_lcm := 28523 % lcm_three_five_seven_eight

-- Lean statement to prove the correct answer
theorem least_number_added_to_divide (n : ℕ) :
  n = lcm_three_five_seven_eight - remainder_28523_lcm :=
sorry

end least_number_added_to_divide_l118_118756


namespace mary_received_more_l118_118150

theorem mary_received_more (investment_Mary investment_Harry profit : ℤ)
  (one_third_profit divided_equally remaining_profit : ℤ)
  (total_Mary total_Harry difference : ℤ)
  (investment_ratio_Mary investment_ratio_Harry : ℚ) :
  investment_Mary = 700 →
  investment_Harry = 300 →
  profit = 3000 →
  one_third_profit = profit / 3 →
  divided_equally = one_third_profit / 2 →
  remaining_profit = profit - one_third_profit →
  investment_ratio_Mary = 7/10 →
  investment_ratio_Harry = 3/10 →
  total_Mary = divided_equally + investment_ratio_Mary * remaining_profit →
  total_Harry = divided_equally + investment_ratio_Harry * remaining_profit →
  difference = total_Mary - total_Harry →
  difference = 800 := by
  sorry

end mary_received_more_l118_118150


namespace siblings_gmat_scores_l118_118028

-- Define the problem conditions
variables (x y z : ℝ)

theorem siblings_gmat_scores (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) : 
  y = x - 1/3 ∧ z = x - 1/6 :=
by
  sorry

end siblings_gmat_scores_l118_118028


namespace fourth_equation_general_expression_l118_118735

theorem fourth_equation :
  (10 : ℕ)^2 - 4 * (4 : ℕ)^2 = 36 := 
sorry

theorem general_expression (n : ℕ) (hn : n > 0) :
  (2 * n + 2)^2 - 4 * n^2 = 8 * n + 4 :=
sorry

end fourth_equation_general_expression_l118_118735


namespace possible_red_ball_draws_l118_118951

/-- 
Given two balls in a bag where one is white and the other is red, 
if a ball is drawn and returned, and then another ball is drawn, 
prove that the possible number of times a red ball is drawn is 0, 1, or 2.
-/
theorem possible_red_ball_draws : 
  (∀ balls : Finset (ℕ × ℕ), 
    balls = {(0, 1), (1, 0)} →
    ∀ draw1 draw2 : ℕ × ℕ, 
    draw1 ∈ balls →
    draw2 ∈ balls →
    ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ 
    n = (if draw1 = (1, 0) then 1 else 0) + 
        (if draw2 = (1, 0) then 1 else 0)) → 
    True := sorry

end possible_red_ball_draws_l118_118951


namespace fraction_of_tea_in_final_cup2_is_5_over_8_l118_118790

-- Defining the initial conditions and the transfers
structure CupContents where
  tea : ℚ
  milk : ℚ

def initialCup1 : CupContents := { tea := 6, milk := 0 }
def initialCup2 : CupContents := { tea := 0, milk := 3 }

def transferOneThird (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let teaTransferred := (1 / 3) * cup1.tea
  ( { cup1 with tea := cup1.tea - teaTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk } )

def transferOneFourth (cup2 : CupContents) (cup1 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup2.tea + cup2.milk
  let amountTransferred := (1 / 4) * mixedTotal
  let teaTransferred := amountTransferred * (cup2.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup2.milk / mixedTotal)
  ( { tea := cup1.tea + teaTransferred, milk := cup1.milk + milkTransferred },
    { tea := cup2.tea - teaTransferred, milk := cup2.milk - milkTransferred } )

def transferOneHalf (cup1 : CupContents) (cup2 : CupContents) : CupContents × CupContents :=
  let mixedTotal := cup1.tea + cup1.milk
  let amountTransferred := (1 / 2) * mixedTotal
  let teaTransferred := amountTransferred * (cup1.tea / mixedTotal)
  let milkTransferred := amountTransferred * (cup1.milk / mixedTotal)
  ( { tea := cup1.tea - teaTransferred, milk := cup1.milk - milkTransferred },
    { tea := cup2.tea + teaTransferred, milk := cup2.milk + milkTransferred } )

def finalContents (cup1 cup2 : CupContents) : CupContents × CupContents :=
  let (cup1Transferred, cup2Transferred) := transferOneThird cup1 cup2
  let (cup1Mixed, cup2Mixed) := transferOneFourth cup2Transferred cup1Transferred
  transferOneHalf cup1Mixed cup2Mixed

-- Statement to be proved
theorem fraction_of_tea_in_final_cup2_is_5_over_8 :
  ((finalContents initialCup1 initialCup2).snd.tea / ((finalContents initialCup1 initialCup2).snd.tea + (finalContents initialCup1 initialCup2).snd.milk) = 5 / 8) :=
sorry

end fraction_of_tea_in_final_cup2_is_5_over_8_l118_118790


namespace no_such_real_x_exists_l118_118828

theorem no_such_real_x_exists :
  ¬ ∃ (x : ℝ), ⌊ x ⌋ + ⌊ 2 * x ⌋ + ⌊ 4 * x ⌋ + ⌊ 8 * x ⌋ + ⌊ 16 * x ⌋ + ⌊ 32 * x ⌋ = 12345 := 
sorry

end no_such_real_x_exists_l118_118828


namespace balloons_given_by_mom_l118_118816

-- Definitions of the initial and total number of balloons
def initial_balloons := 26
def total_balloons := 60

-- Theorem: Proving the number of balloons Tommy's mom gave him
theorem balloons_given_by_mom : total_balloons - initial_balloons = 34 :=
by
  -- This proof is obvious from the setup, so we write sorry to skip the proof.
  sorry

end balloons_given_by_mom_l118_118816


namespace problem1_correct_problem2_correct_l118_118325

noncomputable def problem1 := 5 + (-6) + 3 - 8 - (-4)
noncomputable def problem2 := -2^2 - 3 * (-1)^3 - (-1) / (-1 / 2)^2

theorem problem1_correct : problem1 = -2 := by
  rw [problem1]
  sorry

theorem problem2_correct : problem2 = 3 := by
  rw [problem2]
  sorry

end problem1_correct_problem2_correct_l118_118325


namespace new_profit_is_220_percent_l118_118778

noncomputable def cost_price (CP : ℝ) : ℝ := 100

def initial_profit_percentage : ℝ := 60

noncomputable def initial_selling_price (CP : ℝ) : ℝ :=
  CP + (initial_profit_percentage / 100) * CP

noncomputable def new_selling_price (SP : ℝ) : ℝ :=
  2 * SP

noncomputable def new_profit_percentage (CP SP2 : ℝ) : ℝ :=
  ((SP2 - CP) / CP) * 100

theorem new_profit_is_220_percent : 
  new_profit_percentage (cost_price 100) (new_selling_price (initial_selling_price (cost_price 100))) = 220 :=
by
  sorry

end new_profit_is_220_percent_l118_118778


namespace ratio_of_areas_l118_118258

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l118_118258


namespace remainder_numGreenRedModal_l118_118682

def numGreenMarbles := 7
def numRedMarbles (n : ℕ) := 7 + n
def validArrangement (g r : ℕ) := (g + r = numGreenMarbles + numRedMarbles r) ∧ 
  (g = r)

theorem remainder_numGreenRedModal (N' : ℕ) :
  N' % 1000 = 432 :=
sorry

end remainder_numGreenRedModal_l118_118682


namespace residue_7_pow_1234_l118_118275

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l118_118275


namespace first_shaded_complete_cycle_seat_190_l118_118016

theorem first_shaded_complete_cycle_seat_190 : 
  ∀ (n : ℕ), (n ≥ 1) → 
  ∃ m : ℕ, 
    ((m ≥ n) ∧ 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ 12) → 
    ∃ k : ℕ, (k ≤ m ∧ (k * (k + 1) / 2) % 12 = (i - 1) % 12))) ↔ 
  ∃ m : ℕ, (m = 19 ∧ 190 = (m * (m + 1)) / 2) :=
by
  sorry

end first_shaded_complete_cycle_seat_190_l118_118016


namespace smallest_integer_larger_than_expr_is_248_l118_118211

noncomputable def small_int_larger_than_expr : ℕ :=
  let expr := (Real.sqrt 5 + Real.sqrt 3)^4
  248

theorem smallest_integer_larger_than_expr_is_248 :
    ∃ (n : ℕ), n > (Real.sqrt 5 + Real.sqrt 3)^4 ∧ n = small_int_larger_than_expr := 
by
  -- We introduce the target integer 248
  use (248 : ℕ)
  -- The given conditions should lead us to 248 being greater than the expression.
  sorry

end smallest_integer_larger_than_expr_is_248_l118_118211


namespace geometric_sequence_common_ratio_l118_118798

theorem geometric_sequence_common_ratio
  (q a_1 : ℝ)
  (h1: a_1 * q = 1)
  (h2: a_1 + a_1 * q^2 = -2) :
  q = -1 :=
by
  sorry

end geometric_sequence_common_ratio_l118_118798


namespace simplify_expression_l118_118525

theorem simplify_expression (a : ℤ) :
  ((36 * a^9)^4 * (63 * a^9)^4) = a^4 :=
sorry

end simplify_expression_l118_118525


namespace value_of_k_l118_118949

theorem value_of_k (k : ℕ) (h : 24 / k = 4) : k = 6 := by
  sorry

end value_of_k_l118_118949


namespace points_earned_l118_118879

-- Definition of the conditions explicitly stated in the problem
def points_per_bag := 8
def total_bags := 4
def bags_not_recycled := 2

-- Calculation of bags recycled
def bags_recycled := total_bags - bags_not_recycled

-- The main theorem stating the proof equivalent
theorem points_earned : points_per_bag * bags_recycled = 16 := 
by
  sorry

end points_earned_l118_118879


namespace sqrt_two_irrational_l118_118006

def irrational (x : ℝ) := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem sqrt_two_irrational : irrational (Real.sqrt 2) := 
by 
  sorry

end sqrt_two_irrational_l118_118006


namespace gcd_problem_l118_118490

theorem gcd_problem (b : ℕ) (h : ∃ k : ℕ, b = 3150 * k) :
  gcd (b^2 + 9 * b + 54) (b + 4) = 2 := by
  sorry

end gcd_problem_l118_118490


namespace packing_objects_in_boxes_l118_118157

theorem packing_objects_in_boxes 
  (n k : ℕ) (n_pos : 0 < n) (k_pos : 0 < k) 
  (objects : Fin (n * k) → Fin k) 
  (boxes : Fin k → Fin n → Fin k) :
  ∃ (pack : Fin (n * k) → Fin k), 
    (∀ i, ∃ c1 c2, 
      ∀ j, pack i = pack j → 
      (objects i = c1 ∨ objects i = c2 ∧
      objects j = c1 ∨ objects j = c2)) := 
sorry

end packing_objects_in_boxes_l118_118157


namespace ratio_and_lcm_l118_118375

noncomputable def common_factor (a b : ℕ) := ∃ x : ℕ, a = 3 * x ∧ b = 4 * x

theorem ratio_and_lcm (a b : ℕ) (h1 : common_factor a b) (h2 : Nat.lcm a b = 180) (h3 : a = 60) : b = 45 :=
by sorry

end ratio_and_lcm_l118_118375


namespace sum_of_sequence_l118_118114

noncomputable def f (n x : ℝ) : ℝ := (1 / (8 * n)) * x^2 + 2 * n * x

theorem sum_of_sequence (n : ℕ) (hn : n > 0) :
  let a : ℝ := 1 / (8 * n)
  let b : ℝ := 2 * n
  let f' := 2 * a * ((-n : ℝ )) + b 
  ∃ S : ℝ, S = (n - 1) * 2^(n + 1) + 2 := 
sorry

end sum_of_sequence_l118_118114


namespace maximal_segment_number_l118_118315

theorem maximal_segment_number (n : ℕ) (h : n > 4) : 
  ∃ k, k = if n % 2 = 0 then 2 * n - 4 else 2 * n - 3 :=
sorry

end maximal_segment_number_l118_118315


namespace original_cost_price_40_l118_118266

theorem original_cost_price_40
  (selling_price : ℝ)
  (decrease_rate : ℝ)
  (profit_increase_rate : ℝ)
  (new_selling_price := selling_price)
  (original_cost_price : ℝ)
  (new_cost_price := (1 - decrease_rate) * original_cost_price)
  (original_profit_margin := (selling_price - original_cost_price) / original_cost_price)
  (new_profit_margin := (new_selling_price - new_cost_price) / new_cost_price)
  (profit_margin_increase := profit_increase_rate)
  (h1 : selling_price = 48)
  (h2 : decrease_rate = 0.04)
  (h3 : profit_increase_rate = 0.05)
  (h4 : new_profit_margin = original_profit_margin + profit_margin_increase) :
  original_cost_price = 40 := 
by 
  sorry

end original_cost_price_40_l118_118266


namespace weeks_to_work_l118_118091

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end weeks_to_work_l118_118091


namespace total_bending_angle_l118_118726

theorem total_bending_angle (n : ℕ) (h : n > 4) (θ : ℝ) (hθ : θ = 360 / (2 * n)) : 
  ∃ α : ℝ, α = 180 :=
by
  sorry

end total_bending_angle_l118_118726


namespace infinite_grid_rectangles_l118_118246

theorem infinite_grid_rectangles (m : ℕ) (hm : m > 12) : 
  ∃ (x y : ℕ), x * y > m ∧ x * (y - 1) < m := 
  sorry

end infinite_grid_rectangles_l118_118246


namespace total_slices_l118_118068

def pizzas : ℕ := 2
def slices_per_pizza : ℕ := 8

theorem total_slices : pizzas * slices_per_pizza = 16 :=
by
  sorry

end total_slices_l118_118068


namespace base_number_of_equation_l118_118665

theorem base_number_of_equation (n : ℕ) (h_n: n = 17)
  (h_eq: 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^18) : some_number = 2 := by
  sorry

end base_number_of_equation_l118_118665


namespace tim_youth_comparison_l118_118867

theorem tim_youth_comparison :
  let Tim_age : ℕ := 5
  let Rommel_age : ℕ := 3 * Tim_age
  let Jenny_age : ℕ := Rommel_age + 2
  Jenny_age - Tim_age = 12 := 
by 
  sorry

end tim_youth_comparison_l118_118867


namespace num_sides_regular_polygon_l118_118318

-- Define the perimeter and side length of the polygon
def perimeter : ℝ := 160
def side_length : ℝ := 10

-- Theorem to prove the number of sides
theorem num_sides_regular_polygon : 
  (perimeter / side_length) = 16 := by
    sorry  -- Proof is omitted

end num_sides_regular_polygon_l118_118318


namespace chipmunks_initial_count_l118_118382

variable (C : ℕ) (total : ℕ) (morning_beavers : ℕ) (afternoon_beavers : ℕ) (decrease_chipmunks : ℕ)

axiom chipmunks_count : morning_beavers = 20 
axiom beavers_double : afternoon_beavers = 2 * morning_beavers
axiom decrease_chipmunks_initial : decrease_chipmunks = 10
axiom total_animals : total = 130

theorem chipmunks_initial_count : 
  20 + C + (2 * 20) + (C - 10) = 130 → C = 40 :=
by
  intros h
  sorry

end chipmunks_initial_count_l118_118382


namespace remainder_of_7_pow_308_mod_11_l118_118215

theorem remainder_of_7_pow_308_mod_11 :
  (7 ^ 308) % 11 = 9 :=
by
  sorry

end remainder_of_7_pow_308_mod_11_l118_118215


namespace find_incorrect_statement_l118_118471

def statement_A := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
def statement_B := ∀ (P : Prop), ((¬P) → false) → P
def statement_C := ∀ (shape : Type), (∃ s : shape, true) → false
def statement_D := ∀ (P : ℕ → Prop), P 0 → (∀ n, P n → P (n + 1)) → ∀ n, P n
def statement_E := ∀ {α : Type} (p : Prop), (¬p ∨ p)

theorem find_incorrect_statement : statement_C :=
sorry

end find_incorrect_statement_l118_118471


namespace Taylor_needs_14_jars_l118_118547

noncomputable def standard_jar_volume : ℕ := 60
noncomputable def big_container_volume : ℕ := 840

theorem Taylor_needs_14_jars : big_container_volume / standard_jar_volume = 14 :=
by sorry

end Taylor_needs_14_jars_l118_118547


namespace bryan_initial_pushups_l118_118808

def bryan_pushups (x : ℕ) : Prop :=
  let totalPushups := x + x + (x - 5)
  totalPushups = 40

theorem bryan_initial_pushups (x : ℕ) (hx : bryan_pushups x) : x = 15 :=
by {
  sorry
}

end bryan_initial_pushups_l118_118808


namespace percentage_of_students_on_trip_l118_118424

-- Define the problem context
variable (total_students : ℕ)
variable (students_more_100 : ℕ)
variable (students_on_trip : ℕ)

-- Define the conditions as per the problem
def condition_1 : Prop := students_more_100 = total_students * 15 / 100
def condition_2 : Prop := students_more_100 = students_on_trip * 25 / 100

-- Define the problem statement
theorem percentage_of_students_on_trip
  (h1 : condition_1 total_students students_more_100)
  (h2 : condition_2 students_more_100 students_on_trip) :
  students_on_trip = total_students * 60 / 100 :=
by
  sorry

end percentage_of_students_on_trip_l118_118424


namespace geometric_sequence_m_solution_l118_118385

theorem geometric_sequence_m_solution (m : ℝ) (h : ∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 4 ∧ a * c = b^2) :
  m = 2 ∨ m = -2 :=
by
  sorry

end geometric_sequence_m_solution_l118_118385


namespace first_divisor_l118_118160

theorem first_divisor (k : ℤ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 7 = 3) (h4 : k < 42) (hk : k = 17) : 5 ≤ 6 ∧ 5 ≤ 7 ∧ 5 = 5 :=
by {
  sorry
}

end first_divisor_l118_118160


namespace total_weight_mason_hotdogs_l118_118669

-- Definitions from conditions
def weight_hotdog := 2
def weight_burger := 5
def weight_pie := 10
def noah_burgers := 8
def jacob_pies := noah_burgers - 3
def mason_hotdogs := 3 * jacob_pies

-- Statement to prove
theorem total_weight_mason_hotdogs : mason_hotdogs * weight_hotdog = 30 := 
by 
  sorry

end total_weight_mason_hotdogs_l118_118669


namespace cubic_function_increasing_l118_118391

noncomputable def f (a x : ℝ) := x ^ 3 + a * x ^ 2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) (h : 0 ≤ a ∧ a ≤ 21) :
    ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
sorry

end cubic_function_increasing_l118_118391


namespace ab_value_l118_118220

theorem ab_value (a b : ℝ) :
  (A = { x : ℝ | x^2 - 8 * x + 15 = 0 }) ∧
  (B = { x : ℝ | x^2 - a * x + b = 0 }) ∧
  (A ∪ B = {2, 3, 5}) ∧
  (A ∩ B = {3}) →
  (a * b = 30) :=
by
  sorry

end ab_value_l118_118220


namespace v2004_eq_1_l118_118974

def g (x: ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- assuming default value for undefined cases

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n + 1)

theorem v2004_eq_1 : v 2004 = 1 :=
  sorry

end v2004_eq_1_l118_118974


namespace cubic_inches_in_one_cubic_foot_l118_118290

-- Definition for the given conversion between foot and inches
def foot_to_inches : ℕ := 12

-- The theorem to prove the cubic conversion
theorem cubic_inches_in_one_cubic_foot : (foot_to_inches ^ 3) = 1728 := by
  -- Skipping the actual proof
  sorry

end cubic_inches_in_one_cubic_foot_l118_118290


namespace tank_fill_fraction_l118_118503

theorem tank_fill_fraction (a b c : ℝ) (h1 : a=9) (h2 : b=54) (h3 : c=3/4) : (c * b + a) / b = 23 / 25 := 
by 
  sorry

end tank_fill_fraction_l118_118503


namespace part1_part2_l118_118225

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0 ∧ x < 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 + m * y + 1 = 0 ∧ y < 0)
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0

-- Lean statement for part 1
theorem part1 (m : ℝ) :
  ¬ ¬ p m → m > 2 :=
sorry

-- Lean statement for part 2
theorem part2 (m : ℝ) :
  (p m ∨ q m) ∧ (¬(p m ∧ q m)) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end part1_part2_l118_118225


namespace red_grapes_count_l118_118945

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l118_118945


namespace customer_initial_amount_l118_118145

theorem customer_initial_amount (d c : ℕ) (h1 : c = 100 * d) (h2 : c = 2 * d) : d = 0 ∧ c = 0 := by
  sorry

end customer_initial_amount_l118_118145


namespace diagonals_in_nine_sided_polygon_l118_118587

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l118_118587


namespace common_ratio_of_geometric_series_l118_118248

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l118_118248


namespace james_savings_l118_118040

-- Define the conditions
def cost_vest : ℝ := 250
def weight_plates_pounds : ℕ := 200
def cost_per_pound : ℝ := 1.2
def original_weight_vest_cost : ℝ := 700
def discount : ℝ := 100

-- Define the derived quantities based on conditions
def cost_weight_plates : ℝ := weight_plates_pounds * cost_per_pound
def total_cost_setup : ℝ := cost_vest + cost_weight_plates
def discounted_weight_vest_cost : ℝ := original_weight_vest_cost - discount
def savings : ℝ := discounted_weight_vest_cost - total_cost_setup

-- The statement to prove the savings
theorem james_savings : savings = 110 := by
  sorry

end james_savings_l118_118040


namespace find_number_exceeds_sixteen_percent_l118_118673

theorem find_number_exceeds_sixteen_percent (x : ℝ) (h : x - 0.16 * x = 63) : x = 75 :=
sorry

end find_number_exceeds_sixteen_percent_l118_118673


namespace no_seating_in_four_consecutive_seats_l118_118147

theorem no_seating_in_four_consecutive_seats :
  let total_arrangements := Nat.factorial 10
  let grouped_arrangements := Nat.factorial 7 * Nat.factorial 4
  let acceptable_arrangements := total_arrangements - grouped_arrangements
  acceptable_arrangements = 3507840 :=
by
  sorry

end no_seating_in_four_consecutive_seats_l118_118147


namespace anika_age_l118_118744

/-- Given:
 1. Anika is 10 years younger than Clara.
 2. Clara is 5 years older than Ben.
 3. Ben is 20 years old.
 Prove:
 Anika's age is 15 years.
 -/
theorem anika_age (Clara Anika Ben : ℕ) 
  (h1 : Anika = Clara - 10) 
  (h2 : Clara = Ben + 5) 
  (h3 : Ben = 20) : Anika = 15 := 
by
  sorry

end anika_age_l118_118744


namespace quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l118_118681

theorem quadratic_has_two_real_roots_for_any_m (m : ℝ) : 
  ∃ (α β : ℝ), (α^2 - 3*α + 2 - m^2 - m = 0) ∧ (β^2 - 3*β + 2 - m^2 - m = 0) :=
sorry

theorem find_m_given_roots_conditions (α β : ℝ) (m : ℝ) 
  (h1 : α^2 - 3*α + 2 - m^2 - m = 0) 
  (h2 : β^2 - 3*β + 2 - m^2 - m = 0) 
  (h3 : α^2 + β^2 = 9) : 
  m = -2 ∨ m = 1 :=
sorry

end quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l118_118681


namespace speed_of_water_l118_118226

variable (v : ℝ) -- the speed of the water in km/h
variable (t : ℝ) -- time taken to swim back in hours
variable (d : ℝ) -- distance swum against the current in km
variable (s : ℝ) -- speed in still water

theorem speed_of_water :
  ∀ (v t d s : ℝ),
  s = 20 -> t = 5 -> d = 40 -> d = (s - v) * t -> v = 12 :=
by
  intros v t d s ht hs hd heq
  sorry

end speed_of_water_l118_118226


namespace find_mnp_l118_118752

noncomputable def equation_rewrite (a b x y : ℝ) (m n p : ℕ): Prop :=
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) ∧
  (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5

theorem find_mnp (a b x y : ℝ): 
  equation_rewrite a b x y 2 1 4 ∧ (2 * 1 * 4 = 8) :=
by 
  sorry

end find_mnp_l118_118752


namespace polynomial_factorization_l118_118626

theorem polynomial_factorization : (∀ x : ℤ, x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1)) := by
  intro x
  sorry

end polynomial_factorization_l118_118626


namespace no_second_quadrant_l118_118048

theorem no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (x < 0 → 3 * x + k - 2 ≤ 0)) → k ≤ 2 :=
by
  intro h
  sorry

end no_second_quadrant_l118_118048


namespace initial_men_l118_118739

/-- Initial number of men M being catered for. 
Proof that the initial number of men M is equal to 760 given the conditions. -/
theorem initial_men (M : ℕ)
  (H1 : 22 * M = 20 * M)
  (H2 : 2 * (M + 3040) = M) : M = 760 := 
sorry

end initial_men_l118_118739


namespace solve_for_y_l118_118950

theorem solve_for_y (y : ℝ) (h : 5^(3 * y) = Real.sqrt 125) : y = 1 / 2 :=
by sorry

end solve_for_y_l118_118950


namespace distinct_roots_quadratic_l118_118565

theorem distinct_roots_quadratic (a x₁ x₂ : ℝ) (h₁ : x^2 + a*x + 8 = 0) 
  (h₂ : x₁ ≠ x₂) (h₃ : x₁ - 64 / (17 * x₂^3) = x₂ - 64 / (17 * x₁^3)) : 
  a = 12 ∨ a = -12 := 
sorry

end distinct_roots_quadratic_l118_118565


namespace power_function_value_at_two_l118_118760

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value_at_two (a : ℝ) (h : f (1/2) a = 8) : f 2 a = 1 / 8 := by
  sorry

end power_function_value_at_two_l118_118760


namespace revenue_equation_l118_118252

theorem revenue_equation (x : ℝ) (r_j r_t : ℝ) (h1 : r_j = 90) (h2 : r_t = 144) :
  r_j + r_j * (1 + x) + r_j * (1 + x)^2 = r_t :=
by
  rw [h1, h2]
  sorry

end revenue_equation_l118_118252


namespace all_values_equal_l118_118137

noncomputable def f : ℤ × ℤ → ℕ :=
sorry

theorem all_values_equal (f : ℤ × ℤ → ℕ)
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ x y, f (x, y) = 1/4 * (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1))) :
  ∀ (x1 y1 x2 y2 : ℤ), f (x1, y1) = f (x2, y2) := 
sorry

end all_values_equal_l118_118137


namespace cube_of_odd_number_minus_itself_divisible_by_24_l118_118373

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end cube_of_odd_number_minus_itself_divisible_by_24_l118_118373


namespace chord_property_l118_118703

noncomputable def chord_length (R r k : ℝ) : Prop :=
  k = 2 * Real.sqrt (R^2 - r^2)

theorem chord_property (P O : Point) (R k : ℝ) (hR : 0 < R) (hk : 0 < k) :
  ∃ r, r = Real.sqrt (R^2 - k^2 / 4) ∧ chord_length R r k :=
sorry

end chord_property_l118_118703


namespace fiona_observe_pairs_l118_118549

def classroom_pairs (n : ℕ) : ℕ :=
  if n > 1 then n - 1 else 0

theorem fiona_observe_pairs :
  classroom_pairs 12 = 11 :=
by
  sorry

end fiona_observe_pairs_l118_118549


namespace range_of_a_l118_118603

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l118_118603


namespace jennifer_fruits_left_l118_118880

theorem jennifer_fruits_left:
  (apples = 2 * pears) →
  (cherries = oranges / 2) →
  (grapes = 3 * apples) →
  pears = 15 →
  oranges = 30 →
  pears_given = 3 →
  oranges_given = 5 →
  apples_given = 5 →
  cherries_given = 7 →
  grapes_given = 3 →
  (remaining_fruits =
    (pears - pears_given) +
    (oranges - oranges_given) +
    (apples - apples_given) +
    (cherries - cherries_given) +
    (grapes - grapes_given)) →
  remaining_fruits = 157 :=
by
  intros
  sorry

end jennifer_fruits_left_l118_118880


namespace only_setA_forms_triangle_l118_118461

-- Define the sets of line segments
def setA := [3, 5, 7]
def setB := [3, 6, 10]
def setC := [5, 5, 11]
def setD := [5, 6, 11]

-- Define a function to check the triangle inequality
def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Formalize the question
theorem only_setA_forms_triangle :
  satisfies_triangle_inequality 3 5 7 ∧
  ¬(satisfies_triangle_inequality 3 6 10) ∧
  ¬(satisfies_triangle_inequality 5 5 11) ∧
  ¬(satisfies_triangle_inequality 5 6 11) :=
by
  sorry

end only_setA_forms_triangle_l118_118461


namespace gain_percentage_l118_118522

theorem gain_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 + C2 = 540) (h2 : C1 = 315)
    (h3 : SP1 = C1 - (0.15 * C1)) (h4 : SP1 = SP2) :
    ((SP2 - C2) / C2) * 100 = 19 :=
by
  sorry

end gain_percentage_l118_118522


namespace train_crossing_pole_time_l118_118919

theorem train_crossing_pole_time :
  ∀ (speed_kmph length_m: ℝ), speed_kmph = 160 → length_m = 400.032 → 
  length_m / (speed_kmph * 1000 / 3600) = 9.00072 :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- The proof is omitted as per instructions
  sorry

end train_crossing_pole_time_l118_118919


namespace repeat_45_fraction_repeat_245_fraction_l118_118689

-- Define the repeating decimal 0.454545... == n / d
def repeating_45_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.45454545 = (n : ℚ) / (d : ℚ))

-- First problem statement: 0.4545... == 5 / 11
theorem repeat_45_fraction : 0.45454545 = (5 : ℚ) / (11 : ℚ) :=
by
  sorry

-- Define the repeating decimal 0.2454545... == n / d
def repeating_245_equiv : Prop := ∃ n d : ℕ, (d ≠ 0) ∧ (0.2454545 = (n : ℚ) / (d : ℚ))

-- Second problem statement: 0.2454545... == 27 / 110
theorem repeat_245_fraction : 0.2454545 = (27 : ℚ) / (110 : ℚ) :=
by
  sorry

end repeat_45_fraction_repeat_245_fraction_l118_118689


namespace equivalent_single_percentage_increase_l118_118498

noncomputable def calculate_final_price (p : ℝ) : ℝ :=
  let p1 := p * (1 + 0.15)
  let p2 := p1 * (1 + 0.20)
  let p_final := p2 * (1 - 0.10)
  p_final

theorem equivalent_single_percentage_increase (p : ℝ) : 
  calculate_final_price p = p * 1.242 :=
by
  sorry

end equivalent_single_percentage_increase_l118_118498


namespace tree_original_height_l118_118853

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l118_118853


namespace quadratic_sum_eq_504_l118_118947

theorem quadratic_sum_eq_504 :
  ∃ (a b c : ℝ), (∀ x : ℝ, 20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧ a + b + c = 504 :=
by sorry

end quadratic_sum_eq_504_l118_118947


namespace determinant_is_zero_l118_118072

-- Define the matrix
def my_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, x + z, y - z],
    ![1, x + y + z, y - z],
    ![1, x + z, x + y]]

-- Define the property to prove
theorem determinant_is_zero (x y z : ℝ) :
  Matrix.det (my_matrix x y z) = 0 :=
by sorry

end determinant_is_zero_l118_118072


namespace quadratic_expression_value_l118_118576

theorem quadratic_expression_value (a : ℝ)
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 + 2 * (a - 1) * x₁ + a^2 - 7 * a - 4 = 0 ∧ x₂^2 + 2 * (a - 1) * x₂ + a^2 - 7 * a - 4 = 0)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ * x₂ - 3 * x₁ - 3 * x₂ - 2 = 0) :
  (1 + 4 / (a^2 - 4)) * (a + 2) / a = 2 := 
sorry

end quadratic_expression_value_l118_118576


namespace tan_ratio_of_triangle_sides_l118_118265

theorem tan_ratio_of_triangle_sides (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : α + β + γ = π)
  (h3 : c ≠ 0):
  ( (Real.tan γ) / (Real.tan α + Real.tan β) ) = (a * b) / (1011 * c^2) := 
sorry

end tan_ratio_of_triangle_sides_l118_118265


namespace tan_theta_eq_l118_118105

variables (k θ : ℝ)

-- Condition: k > 0
axiom k_pos : k > 0

-- Condition: k * cos θ = 12
axiom k_cos_theta : k * Real.cos θ = 12

-- Condition: k * sin θ = 5
axiom k_sin_theta : k * Real.sin θ = 5

-- To prove: tan θ = 5 / 12
theorem tan_theta_eq : Real.tan θ = 5 / 12 := by
  sorry

end tan_theta_eq_l118_118105


namespace max_int_solution_of_inequality_system_l118_118876

theorem max_int_solution_of_inequality_system :
  ∃ (x : ℤ), (∀ (y : ℤ), (3 * y - 1 < y + 1) ∧ (2 * (2 * y - 1) ≤ 5 * y + 1) → y ≤ x) ∧
             (3 * x - 1 < x + 1) ∧ (2 * (2 * x - 1) ≤ 5 * x + 1) ∧
             x = 0 :=
by
  sorry

end max_int_solution_of_inequality_system_l118_118876


namespace sin_cos_identity_l118_118677

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) 
  - Real.cos (200 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 := 
by
  -- This would be where the proof goes
  sorry

end sin_cos_identity_l118_118677


namespace tyler_total_puppies_l118_118282

/-- 
  Tyler has 15 dogs, and each dog has 5 puppies.
  We want to prove that the total number of puppies is 75.
-/
def tyler_dogs : Nat := 15
def puppies_per_dog : Nat := 5
def total_puppies_tyler_has : Nat := tyler_dogs * puppies_per_dog

theorem tyler_total_puppies : total_puppies_tyler_has = 75 := by
  sorry

end tyler_total_puppies_l118_118282


namespace sprockets_produced_by_machines_l118_118450

noncomputable def machine_sprockets (t : ℝ) : Prop :=
  let machineA_hours := t + 10
  let machineA_rate := 4
  let machineA_sprockets := machineA_hours * machineA_rate
  let machineB_hours := t
  let machineB_rate := 4.4
  let machineB_sprockets := machineB_hours * machineB_rate
  machineA_sprockets = 440 ∧ machineB_sprockets = 440

theorem sprockets_produced_by_machines (t : ℝ) (h : machine_sprockets t) : t = 100 :=
  sorry

end sprockets_produced_by_machines_l118_118450


namespace S_10_is_65_l118_118773

variable (a_1 d : ℤ)
variable (S : ℤ → ℤ)

-- Define the arithmetic sequence conditions
def a_3 : ℤ := a_1 + 2 * d
def S_n (n : ℤ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a_3_is_4 : a_3 = 4
axiom S_9_minus_S_6_is_27 : S 9 - S 6 = 27

-- The target statement to be proven
theorem S_10_is_65 : S 10 = 65 :=
by
  sorry

end S_10_is_65_l118_118773


namespace total_amount_l118_118590

def shares (a b c : ℕ) : Prop :=
  b = 1800 ∧ 2 * b = 3 * a ∧ 3 * c = 4 * b

theorem total_amount (a b c : ℕ) (h : shares a b c) : a + b + c = 5400 :=
by
  have h₁ : 2 * b = 3 * a := h.2.1
  have h₂ : 3 * c = 4 * b := h.2.2
  have hb : b = 1800 := h.1
  sorry

end total_amount_l118_118590


namespace problem1_problem2_l118_118772

theorem problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x + y ≥ 6 :=
sorry

theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x * y ≥ 9 :=
sorry

end problem1_problem2_l118_118772


namespace x_squared_plus_y_squared_l118_118280

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 17) (h2 : x * y = 6) : x^2 + y^2 = 301 :=
by sorry

end x_squared_plus_y_squared_l118_118280


namespace inheritance_amount_l118_118976

theorem inheritance_amount (x : ℝ) 
    (federal_tax : ℝ := 0.25 * x) 
    (remaining_after_federal_tax : ℝ := x - federal_tax) 
    (state_tax : ℝ := 0.15 * remaining_after_federal_tax) 
    (total_taxes : ℝ := federal_tax + state_tax) 
    (taxes_paid : total_taxes = 15000) : 
    x = 41379 :=
sorry

end inheritance_amount_l118_118976


namespace atomic_weight_of_chlorine_l118_118910

theorem atomic_weight_of_chlorine (molecular_weight_AlCl3 : ℝ) (atomic_weight_Al : ℝ) (atomic_weight_Cl : ℝ) :
  molecular_weight_AlCl3 = 132 ∧ atomic_weight_Al = 26.98 →
  132 = 26.98 + 3 * atomic_weight_Cl →
  atomic_weight_Cl = 35.007 :=
by
  intros h1 h2
  sorry

end atomic_weight_of_chlorine_l118_118910


namespace books_bound_l118_118723

theorem books_bound (x : ℕ) (w c : ℕ) (h₀ : w = 92) (h₁ : c = 135) 
(h₂ : 92 - x = 2 * (135 - x)) :
x = 178 :=
by
  sorry

end books_bound_l118_118723


namespace sandbag_weight_proof_l118_118492

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end sandbag_weight_proof_l118_118492


namespace trapezoid_area_correct_l118_118289

noncomputable def calculate_trapezoid_area : ℕ :=
  let parallel_side_1 := 6
  let parallel_side_2 := 12
  let leg := 5
  let radius := 5
  let height := radius
  let area := (1 / 2) * (parallel_side_1 + parallel_side_2) * height
  area

theorem trapezoid_area_correct :
  calculate_trapezoid_area = 45 :=
by {
  sorry
}

end trapezoid_area_correct_l118_118289


namespace total_notebooks_l118_118400

-- Definitions from the conditions
def Yoongi_notebooks : Nat := 3
def Jungkook_notebooks : Nat := 3
def Hoseok_notebooks : Nat := 3

-- The proof problem
theorem total_notebooks : Yoongi_notebooks + Jungkook_notebooks + Hoseok_notebooks = 9 := 
by 
  sorry

end total_notebooks_l118_118400


namespace square_side_length_s2_l118_118705

theorem square_side_length_s2 (s1 s2 s3 : ℕ)
  (h1 : s1 + s2 + s3 = 3322)
  (h2 : s1 - s2 + s3 = 2020) :
  s2 = 651 :=
by sorry

end square_side_length_s2_l118_118705


namespace find_number_l118_118710

theorem find_number (a b N : ℕ) (h1 : b = 7) (h2 : b - a = 2) (h3 : a * b = 2 * (a + b) + N) : N = 11 :=
  sorry

end find_number_l118_118710


namespace translate_right_l118_118108

-- Definition of the initial point and translation distance
def point_A : ℝ × ℝ := (2, -1)
def translation_distance : ℝ := 3

-- The proof statement
theorem translate_right (x_A y_A : ℝ) (d : ℝ) 
  (h1 : point_A = (x_A, y_A))
  (h2 : translation_distance = d) : 
  (x_A + d, y_A) = (5, -1) := 
sorry

end translate_right_l118_118108


namespace ratio_b_to_c_l118_118817

variables (a b c d e f : ℝ)

theorem ratio_b_to_c 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := 
sorry

end ratio_b_to_c_l118_118817


namespace kate_change_l118_118700

def first_candy_cost : ℝ := 0.54
def second_candy_cost : ℝ := 0.35
def third_candy_cost : ℝ := 0.68
def amount_given : ℝ := 5.00

theorem kate_change : amount_given - (first_candy_cost + second_candy_cost + third_candy_cost) = 3.43 := by
  sorry

end kate_change_l118_118700


namespace incorrect_transformation_is_not_valid_l118_118960

-- Define the system of linear equations
def eq1 (x y : ℝ) := 2 * x + y = 5
def eq2 (x y : ℝ) := 3 * x + 4 * y = 7

-- The definition of the correct transformation for x from equation eq2
def correct_transformation (x y : ℝ) := x = (7 - 4 * y) / 3

-- The definition of the incorrect transformation for x from equation eq2
def incorrect_transformation (x y : ℝ) := x = (7 + 4 * y) / 3

theorem incorrect_transformation_is_not_valid (x y : ℝ) 
  (h1 : eq1 x y) 
  (h2 : eq2 x y) :
  ¬ incorrect_transformation x y := 
by
  sorry

end incorrect_transformation_is_not_valid_l118_118960


namespace peach_trees_count_l118_118337

theorem peach_trees_count : ∀ (almond_trees: ℕ), almond_trees = 300 → 2 * almond_trees - 30 = 570 :=
by
  intros
  sorry

end peach_trees_count_l118_118337


namespace find_c_l118_118899

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ x ∈ Set.Ioo (-(7 / 3) : ℝ) (2 : ℝ)) → c = 14 :=
by
  intro h
  sorry

end find_c_l118_118899


namespace isosceles_trapezoid_side_length_l118_118024

theorem isosceles_trapezoid_side_length (A b1 b2 h half_diff s : ℝ) (h0 : A = 44) (h1 : b1 = 8) (h2 : b2 = 14) 
    (h3 : A = 0.5 * (b1 + b2) * h)
    (h4 : h = 4) 
    (h5 : half_diff = (b2 - b1) / 2) 
    (h6 : half_diff = 3)
    (h7 : s^2 = h^2 + half_diff^2)
    (h8 : s = 5) : 
    s = 5 :=
by 
    apply h8

end isosceles_trapezoid_side_length_l118_118024


namespace compound_interest_rate_l118_118112

theorem compound_interest_rate
(SI : ℝ) (CI : ℝ) (P1 : ℝ) (r : ℝ) (t1 t2 : ℕ) (P2 R : ℝ)
(h1 : SI = (P1 * r * t1) / 100)
(h2 : SI = CI / 2)
(h3 : CI = P2 * (1 + R / 100) ^ t2 - P2)
(h4 : P1 = 3500)
(h5 : r = 6)
(h6 : t1 = 2)
(h7 : P2 = 4000)
(h8 : t2 = 2) : R = 10 := by
  sorry

end compound_interest_rate_l118_118112


namespace cassidy_grades_below_B_l118_118278

theorem cassidy_grades_below_B (x : ℕ) (h1 : 26 = 14 + 3 * x) : x = 4 := 
by 
  sorry

end cassidy_grades_below_B_l118_118278


namespace necklace_sum_l118_118557

theorem necklace_sum (H J x S : ℕ) (hH : H = 25) (h1 : H = J + 5) (h2 : x = J / 2) (h3 : S = 2 * H) : H + J + x + S = 105 :=
by 
  sorry

end necklace_sum_l118_118557


namespace students_that_do_not_like_either_sport_l118_118294

def total_students : ℕ := 30
def students_like_basketball : ℕ := 15
def students_like_table_tennis : ℕ := 10
def students_like_both : ℕ := 3

theorem students_that_do_not_like_either_sport : (total_students - (students_like_basketball + students_like_table_tennis - students_like_both)) = 8 := 
by
  sorry

end students_that_do_not_like_either_sport_l118_118294


namespace average_DE_l118_118901

theorem average_DE 
  (a b c d e : ℝ) 
  (avg_all : (a + b + c + d + e) / 5 = 80) 
  (avg_abc : (a + b + c) / 3 = 78) : 
  (d + e) / 2 = 83 := 
sorry

end average_DE_l118_118901


namespace betty_needs_five_boxes_l118_118743

def betty_oranges (total_oranges first_box second_box max_per_box : ℕ) : ℕ :=
  let remaining_oranges := total_oranges - (first_box + second_box)
  let full_boxes := remaining_oranges / max_per_box
  let extra_box := if remaining_oranges % max_per_box == 0 then 0 else 1
  full_boxes + 2 + extra_box

theorem betty_needs_five_boxes :
  betty_oranges 120 30 25 30 = 5 := 
by
  sorry

end betty_needs_five_boxes_l118_118743


namespace arithmetic_sequence_max_sum_l118_118556

theorem arithmetic_sequence_max_sum (a d t : ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a > 0) 
  (h2 : (9 * t) = a + 5 * d) 
  (h3 : (11 * t) = a + 4 * d) 
  (h4 : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2) :
  n = 10 :=
sorry

end arithmetic_sequence_max_sum_l118_118556


namespace point_in_second_quadrant_condition_l118_118361

theorem point_in_second_quadrant_condition (a : ℤ)
  (h1 : 3 * a - 9 < 0)
  (h2 : 10 - 2 * a > 0)
  (h3 : |3 * a - 9| = |10 - 2 * a|):
  (a + 2) ^ 2023 - 1 = 0 := 
sorry

end point_in_second_quadrant_condition_l118_118361


namespace problem_l118_118758

variable (m n : ℝ)
variable (h1 : m + n = -1994)
variable (h2 : m * n = 7)

theorem problem (m n : ℝ) (h1 : m + n = -1994) (h2 : m * n = 7) : 
  (m^2 + 1993 * m + 6) * (n^2 + 1995 * n + 8) = 1986 := 
by
  sorry

end problem_l118_118758


namespace reaction2_follows_markovnikov_l118_118961

-- Define Markovnikov's rule - applying to case with protic acid (HX) to an alkene.
def follows_markovnikov_rule (HX : String) (initial_molecule final_product : String) : Prop :=
  initial_molecule = "CH3-CH=CH2 + HBr" ∧ final_product = "CH3-CHBr-CH3"

-- Example reaction data
def reaction1_initial : String := "CH2=CH2 + Br2"
def reaction1_final : String := "CH2Br-CH2Br"

def reaction2_initial : String := "CH3-CH=CH2 + HBr"
def reaction2_final : String := "CH3-CHBr-CH3"

def reaction3_initial : String := "CH4 + Cl2"
def reaction3_final : String := "CH3Cl + HCl"

def reaction4_initial : String := "CH ≡ CH + HOH"
def reaction4_final : String := "CH3''-C-H"

-- Proof statement
theorem reaction2_follows_markovnikov : follows_markovnikov_rule "HBr" reaction2_initial reaction2_final := by
  sorry

end reaction2_follows_markovnikov_l118_118961


namespace sum_gcd_lcm_is_244_l118_118029

-- Definitions of the constants
def a : ℕ := 12
def b : ℕ := 80

-- Main theorem statement
theorem sum_gcd_lcm_is_244 : Nat.gcd a b + Nat.lcm a b = 244 := by
  sorry

end sum_gcd_lcm_is_244_l118_118029


namespace triangle_construction_possible_l118_118569

-- Define the entities involved
variables {α β : ℝ} {a c : ℝ}

-- State the theorem
theorem triangle_construction_possible (a c : ℝ) (h : α = 2 * β) : a > (2 / 3) * c :=
sorry

end triangle_construction_possible_l118_118569


namespace max_clouds_through_planes_l118_118702

-- Define the problem parameters and conditions
def max_clouds (n : ℕ) : ℕ :=
  n + 1

-- Mathematically equivalent proof problem statement in Lean 4
theorem max_clouds_through_planes : max_clouds 10 = 11 :=
  by
    sorry  -- Proof skipped as required

end max_clouds_through_planes_l118_118702


namespace solve_cubic_equation_l118_118343

theorem solve_cubic_equation (x : ℝ) (h : 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3)) : x = 343 := by
  sorry

end solve_cubic_equation_l118_118343


namespace exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l118_118417

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

end exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l118_118417


namespace isosceles_triangle_leg_length_l118_118629

theorem isosceles_triangle_leg_length
  (P : ℝ) (base : ℝ) (L : ℝ)
  (h_isosceles : true)
  (h_perimeter : P = 24)
  (h_base : base = 10)
  (h_perimeter_formula : P = base + 2 * L) :
  L = 7 := 
by
  sorry

end isosceles_triangle_leg_length_l118_118629


namespace sum_bn_2999_l118_118630

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_bn_2999 : (Finset.range 3000).sum b_n = 572 := by
  sorry

end sum_bn_2999_l118_118630


namespace positive_integer_solutions_l118_118496

theorem positive_integer_solutions :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x + y + x * y = 2008 ∧
  ((x = 6 ∧ y = 286) ∨ (x = 286 ∧ y = 6) ∨ (x = 40 ∧ y = 48) ∨ (x = 48 ∧ y = 40)) :=
by
  sorry

end positive_integer_solutions_l118_118496


namespace employed_females_percentage_l118_118610

theorem employed_females_percentage (E M : ℝ) (hE : E = 60) (hM : M = 42) : ((E - M) / E) * 100 = 30 := by
  sorry

end employed_females_percentage_l118_118610


namespace correct_number_is_650_l118_118767

theorem correct_number_is_650 
  (n : ℕ) 
  (h : n - 152 = 346): 
  n + 152 = 650 :=
by
  sorry

end correct_number_is_650_l118_118767


namespace total_ladybugs_correct_l118_118309

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end total_ladybugs_correct_l118_118309


namespace new_average_age_l118_118104

theorem new_average_age:
  ∀ (initial_avg_age new_persons_avg_age : ℝ) (initial_count new_persons_count : ℕ),
    initial_avg_age = 16 →
    new_persons_avg_age = 15 →
    initial_count = 20 →
    new_persons_count = 20 →
    (initial_avg_age * initial_count + new_persons_avg_age * new_persons_count) / 
    (initial_count + new_persons_count) = 15.5 :=
by
  intros initial_avg_age new_persons_avg_age initial_count new_persons_count
  intros h1 h2 h3 h4
  
  sorry

end new_average_age_l118_118104


namespace find_constants_l118_118771

theorem find_constants :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ 3 → x ≠ 4 → 
  (6 * x / ((x - 4) * (x - 3) ^ 2)) = (A / (x - 4) + B / (x - 3) + C / (x - 3) ^ 2)) ∧
  A = 24 ∧
  B = - 162 / 7 ∧
  C = - 18 :=
by
  use 24, -162 / 7, -18
  sorry

end find_constants_l118_118771


namespace probability_x_lt_2y_is_2_over_5_l118_118971

noncomputable def rectangle_area : ℝ :=
  5 * 2

noncomputable def triangle_area : ℝ :=
  1 / 2 * 4 * 2

noncomputable def probability_x_lt_2y : ℝ :=
  triangle_area / rectangle_area

theorem probability_x_lt_2y_is_2_over_5 :
  probability_x_lt_2y = 2 / 5 := by
  sorry

end probability_x_lt_2y_is_2_over_5_l118_118971


namespace family_reunion_people_l118_118713

theorem family_reunion_people (pasta_per_person : ℚ) (total_pasta : ℚ) (recipe_people : ℚ) : 
  pasta_per_person = 2 / 7 ∧ total_pasta = 10 -> recipe_people = 35 :=
by
  sorry

end family_reunion_people_l118_118713


namespace sum_of_natural_numbers_eq_4005_l118_118578

theorem sum_of_natural_numbers_eq_4005 :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 4005 ∧ n = 89 :=
by
  sorry

end sum_of_natural_numbers_eq_4005_l118_118578


namespace squareable_numbers_l118_118446

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end squareable_numbers_l118_118446


namespace complex_real_part_of_product_l118_118658

theorem complex_real_part_of_product (z1 z2 : ℂ) (i : ℂ) 
  (hz1 : z1 = 4 + 29 * Complex.I)
  (hz2 : z2 = 6 + 9 * Complex.I)
  (hi : i = Complex.I) : 
  ((z1 - z2) * i).re = 20 := 
by
  sorry

end complex_real_part_of_product_l118_118658


namespace logic_problem_l118_118233

theorem logic_problem (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬ (p ∨ q) :=
sorry

end logic_problem_l118_118233


namespace sum_of_valid_m_values_l118_118397

-- Variables and assumptions
variable (m x : ℝ)

-- Conditions from the given problem
def inequality_system (m x : ℝ) : Prop :=
  (x - 4) / 3 < x - 4 ∧ (m - x) / 5 < 0

def solution_set_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality_system m x → x > 4

def fractional_equation (m x : ℝ) : Prop :=
  6 / (x - 3) + 1 = (m * x - 3) / (x - 3)

-- Lean statement to prove the sum of integers satisfying the conditions
theorem sum_of_valid_m_values : 
  (∀ m : ℝ, solution_set_condition m ∧ 
            (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ fractional_equation m x) →
            (∃ (k : ℕ), k = 2 ∨ k = 4) → 
            2 + 4 = 6) :=
sorry

end sum_of_valid_m_values_l118_118397


namespace geometric_progression_nonzero_k_l118_118110

theorem geometric_progression_nonzero_k (k : ℝ) : k ≠ 0 ↔ (40*k)^2 = (10*k) * (160*k) := by sorry

end geometric_progression_nonzero_k_l118_118110


namespace total_candies_l118_118235

-- Define variables and conditions
variables (x y z : ℕ)
axiom h1 : x = y / 2
axiom h2 : x + z = 24
axiom h3 : y + z = 34

-- The statement to be proved
theorem total_candies : x + y + z = 44 :=
by
  sorry

end total_candies_l118_118235


namespace find_monthly_fee_l118_118370

-- Definitions from conditions
def monthly_fee (total_bill : ℝ) (cost_per_minute : ℝ) (minutes_used : ℝ) : ℝ :=
  total_bill - cost_per_minute * minutes_used

-- Theorem stating the question
theorem find_monthly_fee :
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  total_bill - cost_per_minute * minutes_used = 5.00 :=
by
  -- Definition of variables used in the theorem
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  
  -- The statement of the theorem and leaving the proof as an exercise
  show total_bill - cost_per_minute * minutes_used = 5.00
  sorry

end find_monthly_fee_l118_118370


namespace problem_inequality_l118_118916

theorem problem_inequality (a b c m n p : ℝ) (h1 : a + b + c = 1) (h2 : m + n + p = 1) :
  -1 ≤ a * m + b * n + c * p ∧ a * m + b * n + c * p ≤ 1 := by
  sorry

end problem_inequality_l118_118916


namespace range_of_a_l118_118750

noncomputable section

def f (a x : ℝ) := a * x^2 + 2 * a * x - Real.log (x + 1)
def g (x : ℝ) := (Real.exp x - x - 1) / (Real.exp x * (x + 1))

theorem range_of_a
  (a : ℝ)
  (h : ∀ x > 0, f a x + Real.exp (-a) > 1 / (x + 1)) : a ∈ Set.Ici (1 / 2) := 
sorry

end range_of_a_l118_118750


namespace boat_distance_downstream_l118_118247

theorem boat_distance_downstream (v_s : ℝ) (h : 8 - v_s = 5) :
  8 + v_s = 11 :=
by
  sorry

end boat_distance_downstream_l118_118247


namespace birds_problem_l118_118079

theorem birds_problem 
  (x y z : ℕ) 
  (h1 : x + y + z = 30) 
  (h2 : (1 / 3 : ℚ) * x + (1 / 2 : ℚ) * y + 2 * z = 30) 
  : x = 9 ∧ y = 10 ∧ z = 11 := 
  by {
  -- Proof steps would go here
  sorry
}

end birds_problem_l118_118079


namespace desired_interest_percentage_l118_118097

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end desired_interest_percentage_l118_118097


namespace find_integer_n_l118_118425

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l118_118425


namespace length_of_AB_l118_118980

noncomputable def isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem length_of_AB 
  (a b c d e : ℕ)
  (h_iso_ABC : isosceles_triangle a b c)
  (h_iso_CDE : isosceles_triangle c d e)
  (h_perimeter_CDE : c + d + e = 25)
  (h_perimeter_ABC : a + b + c = 24)
  (h_CE : c = 9)
  (h_AB_DE : a = e) : a = 7 :=
by
  sorry

end length_of_AB_l118_118980


namespace asymptote_equations_l118_118039

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (e = sqrt 3) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

theorem asymptote_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : sqrt (a^2 + b^2) / a = sqrt 3) :
  ∀ (x : ℝ), ∃ (y : ℝ), y = sqrt 2 * x ∨ y = -sqrt 2 * x :=
sorry

end asymptote_equations_l118_118039


namespace min_value_of_expression_l118_118870

theorem min_value_of_expression (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
    (h1 : ∀ n, S_n n = (4/3) * (a_n n - 1)) :
  ∃ (n : ℕ), (4^(n - 2) + 1) * (16 / a_n n + 1) = 4 :=
by
  sorry

end min_value_of_expression_l118_118870


namespace stewart_farm_food_l118_118067

variable (S H : ℕ) (HorseFoodPerHorsePerDay : Nat) (TotalSheep : Nat)

theorem stewart_farm_food (ratio_sheep_horses : 6 * H = 7 * S) 
  (total_sheep_count : S = 48) 
  (horse_food : HorseFoodPerHorsePerDay = 230) : 
  HorseFoodPerHorsePerDay * (7 * 48 / 6) = 12880 :=
by
  sorry

end stewart_farm_food_l118_118067


namespace find_a3_l118_118561

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

end find_a3_l118_118561


namespace total_candies_in_store_l118_118111

-- Define the quantities of chocolates in each box
def box_chocolates_1 := 200
def box_chocolates_2 := 320
def box_chocolates_3 := 500
def box_chocolates_4 := 500
def box_chocolates_5 := 768
def box_chocolates_6 := 768

-- Define the quantities of candies in each tub
def tub_candies_1 := 1380
def tub_candies_2 := 1150
def tub_candies_3 := 1150
def tub_candies_4 := 1720

-- Sum of all chocolates and candies
def total_chocolates := box_chocolates_1 + box_chocolates_2 + box_chocolates_3 + box_chocolates_4 + box_chocolates_5 + box_chocolates_6
def total_candies := tub_candies_1 + tub_candies_2 + tub_candies_3 + tub_candies_4
def total_store_candies := total_chocolates + total_candies

theorem total_candies_in_store : total_store_candies = 8456 := by
  sorry

end total_candies_in_store_l118_118111


namespace garden_dimensions_l118_118924

theorem garden_dimensions (w l : ℕ) (h₁ : l = w + 3) (h₂ : 2 * (l + w) = 26) : w = 5 ∧ l = 8 :=
by
  sorry

end garden_dimensions_l118_118924


namespace exponent_fraction_equals_five_fourths_l118_118139

theorem exponent_fraction_equals_five_fourths :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5 / 4 :=
by
  sorry

end exponent_fraction_equals_five_fourths_l118_118139


namespace find_number_l118_118151

theorem find_number (a b : ℕ) (h₁ : a = 555) (h₂ : b = 445) :
  let S := a + b
  let D := a - b
  let Q := 2 * D
  let R := 30
  let N := (S * Q) + R
  N = 220030 := by
  sorry

end find_number_l118_118151


namespace storm_deposit_l118_118796

theorem storm_deposit (C : ℝ) (original_amount after_storm_rate before_storm_rate : ℝ) (after_storm full_capacity : ℝ) :
  before_storm_rate = 0.40 →
  after_storm_rate = 0.60 →
  original_amount = 220 * 10^9 →
  before_storm_rate * C = original_amount →
  C = full_capacity →
  after_storm = after_storm_rate * full_capacity →
  after_storm - original_amount = 110 * 10^9 :=
by
  sorry

end storm_deposit_l118_118796


namespace complement_of_M_in_U_l118_118580

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M_in_U : compl M ∩ U = {1, 2, 3} :=
by
  sorry

end complement_of_M_in_U_l118_118580


namespace intersection_A_B_l118_118607

-- Definition of set A based on the given inequality
def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Definition of set B
def B : Set ℝ := {-3, -1, 1, 3}

-- Prove the intersection A ∩ B equals the expected set {-1, 1, 3}
theorem intersection_A_B : A ∩ B = {-1, 1, 3} := 
by
  sorry

end intersection_A_B_l118_118607


namespace thabo_books_l118_118076

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books_l118_118076


namespace trail_mix_total_weight_l118_118763

def peanuts : ℝ := 0.16666666666666666
def chocolate_chips : ℝ := 0.16666666666666666
def raisins : ℝ := 0.08333333333333333
def trail_mix_weight : ℝ := 0.41666666666666663

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = trail_mix_weight :=
sorry

end trail_mix_total_weight_l118_118763


namespace trigonometric_identity_l118_118930

theorem trigonometric_identity :
  8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 :=
by
  sorry

end trigonometric_identity_l118_118930


namespace part1_part2_l118_118439

def proposition_p (m : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 2 * x - 4 ≥ m^2 - 5 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - 2 * x + m - 1 ≤ 0

theorem part1 (m : ℝ) : proposition_p m → 1 ≤ m ∧ m ≤ 4 := 
sorry

theorem part2 (m : ℝ) : (proposition_p m ∨ proposition_q m) → m ≤ 4 := 
sorry

end part1_part2_l118_118439


namespace B_listing_method_l118_118944

-- Definitions for given conditions
def A : Set ℤ := {-2, -1, 1, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t*t}

-- The mathematically equivalent proof problem
theorem B_listing_method :
  B = {4, 1, 9, 16} := 
by {
  sorry
}

end B_listing_method_l118_118944


namespace sum_of_three_numbers_l118_118793

theorem sum_of_three_numbers (a b c : ℝ) (h1 : (a + b + c) / 3 = a - 15) (h2 : (a + b + c) / 3 = c + 10) (h3 : b = 10) :
  a + b + c = 45 :=
  sorry

end sum_of_three_numbers_l118_118793


namespace num_solutions_gcd_lcm_l118_118274

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem num_solutions_gcd_lcm (x y : ℕ) :
  (Nat.gcd x y = factorial 20) ∧ (Nat.lcm x y = factorial 30) →
  2^10 = 1024 :=
  by
  intro h
  sorry

end num_solutions_gcd_lcm_l118_118274


namespace sqrt_123400_l118_118222

theorem sqrt_123400 (h1: Real.sqrt 12.34 = 3.512) : Real.sqrt 123400 = 351.2 :=
by 
  sorry

end sqrt_123400_l118_118222


namespace square_units_digit_eq_9_l118_118609

/-- The square of which whole number has a units digit of 9? -/
theorem square_units_digit_eq_9 (n : ℕ) (h : ∃ m : ℕ, n = m^2 ∧ m % 10 = 9) : n = 3 ∨ n = 7 := by
  sorry

end square_units_digit_eq_9_l118_118609


namespace probability_of_different_cousins_name_l118_118579

theorem probability_of_different_cousins_name :
  let total_letters := 19
  let amelia_letters := 6
  let bethany_letters := 7
  let claire_letters := 6
  let probability := 
    2 * ((amelia_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)) +
         (amelia_letters / (total_letters : ℚ)) * (claire_letters / (total_letters - 1 : ℚ)) +
         (claire_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)))
  probability = 40 / 57 := sorry

end probability_of_different_cousins_name_l118_118579


namespace car_meeting_points_l118_118191

-- Define the conditions for the problem
variables {A B : ℝ}
variables {speed_ratio : ℝ} (ratio_pos : speed_ratio = 5 / 4)
variables {T1 T2 : ℝ} (T1_pos : T1 = 145) (T2_pos : T2 = 201)

-- The proof problem statement
theorem car_meeting_points (A B : ℝ) (ratio_pos : speed_ratio = 5 / 4) 
  (T1 T2 : ℝ) (T1_pos : T1 = 145) (T2_pos : T2 = 201) :
  A = 103 ∧ B = 229 :=
sorry

end car_meeting_points_l118_118191


namespace rect_garden_width_l118_118367

theorem rect_garden_width (w l : ℝ) (h1 : l = 3 * w) (h2 : l * w = 768) : w = 16 := by
  sorry

end rect_garden_width_l118_118367


namespace range_of_a_l118_118001

variable (a x : ℝ)

def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4

def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

theorem range_of_a (h : ∀ (x : ℝ), p a x → q x) : a <= -2 ∨ a >= 7 := 
by sorry

end range_of_a_l118_118001


namespace Debby_daily_bottles_is_six_l118_118143

def daily_bottles (total_bottles : ℕ) (total_days : ℕ) : ℕ :=
  total_bottles / total_days

theorem Debby_daily_bottles_is_six : daily_bottles 12 2 = 6 := by
  sorry

end Debby_daily_bottles_is_six_l118_118143


namespace borrowed_amount_l118_118627

theorem borrowed_amount (P : ℝ) 
    (borrow_rate : ℝ := 4) 
    (lend_rate : ℝ := 6) 
    (borrow_time : ℝ := 2) 
    (lend_time : ℝ := 2) 
    (gain_per_year : ℝ := 140) 
    (h₁ : ∀ (P : ℝ), P / 8.333 - P / 12.5 = 280) 
    : P = 7000 := 
sorry

end borrowed_amount_l118_118627


namespace reflect_point_x_axis_correct_l118_118037

-- Definition of the transformation reflecting a point across the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the original point coordinates
def P : ℝ × ℝ := (-2, 3)

-- The Lean proof statement
theorem reflect_point_x_axis_correct :
  reflect_x_axis P = (-2, -3) :=
sorry

end reflect_point_x_axis_correct_l118_118037


namespace sin_theta_of_triangle_area_side_median_l118_118502

-- Defining the problem statement and required conditions
theorem sin_theta_of_triangle_area_side_median (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (hA : A = 30)
  (ha : a = 12)
  (hm : m = 8)
  (hTriangleArea : A = 1/2 * a * m * Real.sin θ) :
  Real.sin θ = 5 / 8 :=
by
  -- Proof omitted
  sorry

end sin_theta_of_triangle_area_side_median_l118_118502


namespace order_of_expressions_l118_118734

theorem order_of_expressions (k : ℕ) (hk : k > 4) : (k + 2) < (2 * k) ∧ (2 * k) < (k^2) ∧ (k^2) < (2^k) := by
  sorry

end order_of_expressions_l118_118734


namespace f_2007_2007_l118_118008

def f (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d * d) |>.sum

def f_k : ℕ → ℕ → ℕ
| 0, n => n
| (k+1), n => f (f_k k n)

theorem f_2007_2007 : f_k 2007 2007 = 145 :=
by
  sorry -- Proof omitted

end f_2007_2007_l118_118008


namespace value_of_business_calculation_l118_118507

noncomputable def value_of_business (total_shares_sold_value : ℝ) (shares_fraction_sold : ℝ) (ownership_fraction : ℝ) : ℝ :=
  (total_shares_sold_value / shares_fraction_sold) * ownership_fraction⁻¹

theorem value_of_business_calculation :
  value_of_business 45000 (3/4) (2/3) = 90000 :=
by
  sorry

end value_of_business_calculation_l118_118507


namespace x_100_equals_2_power_397_l118_118007

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 5*n - 3

-- Define the merged sequence x_n
noncomputable def x_n (k : ℕ) : ℕ := 2^(4*k - 3)

-- Prove x_100 is 2^397
theorem x_100_equals_2_power_397 : x_n 100 = 2^397 := by
  unfold x_n
  show 2^(4*100 - 3) = 2^397
  rfl

end x_100_equals_2_power_397_l118_118007


namespace prime_count_of_first_10_sums_is_2_l118_118484

open Nat

def consecutivePrimes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def consecutivePrimeSums (n : Nat) : List Nat :=
  (List.range n).scanl (λ sum i => sum + consecutivePrimes.getD i 0) 0

theorem prime_count_of_first_10_sums_is_2 :
  let sums := consecutivePrimeSums 10;
  (sums.count isPrime) = 2 :=
by
  sorry

end prime_count_of_first_10_sums_is_2_l118_118484


namespace number_of_ordered_pairs_l118_118608

theorem number_of_ordered_pairs (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / 24)) : 
  ∃ n : ℕ, n = 41 :=
by
  sorry

end number_of_ordered_pairs_l118_118608


namespace find_m_l118_118259

variable (m : ℝ)

-- Definitions of the vectors
def AB : ℝ × ℝ := (m + 3, 2 * m + 1)
def CD : ℝ × ℝ := (m + 3, -5)

-- Definition of perpendicular vectors, dot product is zero
def perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_m (h : perp (AB m) (CD m)) : m = 2 := by
  sorry

end find_m_l118_118259


namespace trigonometric_equation_solution_l118_118852

theorem trigonometric_equation_solution (n : ℕ) (h_pos : 0 < n) (x : ℝ) (hx1 : ∀ k : ℤ, x ≠ k * π / 2) :
  (1 / (Real.sin x)^(2 * n) + 1 / (Real.cos x)^(2 * n) = 2^(n + 1)) ↔ ∃ k : ℤ, x = (2 * k + 1) * π / 4 :=
by sorry

end trigonometric_equation_solution_l118_118852


namespace least_number_to_subtract_l118_118695

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : ∃ k : ℕ, (n - k) % 11 = 0 ∧ k = 4 :=
by
  sorry

end least_number_to_subtract_l118_118695


namespace volume_calculation_l118_118144

-- Define the dimensions of the rectangular parallelepiped
def a : ℕ := 2
def b : ℕ := 3
def c : ℕ := 4

-- Define the radius for spheres and cylinders
def r : ℝ := 2

theorem volume_calculation : 
  let l := 384
  let o := 140
  let q := 3
  (l + o + q = 527) :=
by
  sorry

end volume_calculation_l118_118144


namespace benjamin_collects_6_dozen_eggs_l118_118096

theorem benjamin_collects_6_dozen_eggs (B : ℕ) (h : B + 3 * B + (B - 4) = 26) : B = 6 :=
by sorry

end benjamin_collects_6_dozen_eggs_l118_118096


namespace cos_2alpha_2beta_l118_118409

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l118_118409


namespace total_pages_l118_118829

-- Conditions
variables (B1 B2 : ℕ)
variable (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90)
variable (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120)

-- Theorem statement
theorem total_pages (B1 B2 : ℕ) (h1 : (2 / 3 : ℚ) * B1 - (1 / 3 : ℚ) * B1 = 90) (h2 : (3 / 4 : ℚ) * B2 - (1 / 4 : ℚ) * B2 = 120) :
  B1 + B2 = 510 :=
sorry

end total_pages_l118_118829


namespace parallelogram_area_l118_118395

theorem parallelogram_area {a b : ℝ} (h₁ : a = 9) (h₂ : b = 12) (angle : ℝ) (h₃ : angle = 150) : 
  ∃ (area : ℝ), area = 54 * Real.sqrt 3 :=
by
  sorry

end parallelogram_area_l118_118395


namespace a_range_l118_118517

noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem a_range (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ a ∈ Set.Ico (1/7 : ℝ) (1/3 : ℝ) :=
by
  sorry

end a_range_l118_118517


namespace beads_needed_for_jewelry_l118_118596

/-
  We define the parameters based on the problem statement.
-/

def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def total_beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

def repeats_per_bracelet : ℕ := 3
def repeats_per_necklace : ℕ := 5

/-
  We calculate the total number of beads for 1 bracelet and 10 necklaces.
-/

def beads_per_bracelet : ℕ := total_beads_per_pattern * repeats_per_bracelet
def beads_per_necklace : ℕ := total_beads_per_pattern * repeats_per_necklace
def total_beads_needed : ℕ := beads_per_bracelet + beads_per_necklace * 10

theorem beads_needed_for_jewelry:
  total_beads_needed = 742 :=
by 
  sorry

end beads_needed_for_jewelry_l118_118596


namespace upper_limit_of_raise_l118_118887

theorem upper_limit_of_raise (lower upper : ℝ) (h_lower : lower = 0.05)
  (h_upper : upper > 0.08) (h_inequality : ∀ r, lower < r → r < upper)
  : upper < 0.09 :=
sorry

end upper_limit_of_raise_l118_118887


namespace problem_statement_l118_118528

theorem problem_statement :
  ∃ (n : ℕ), n = 101 ∧
  (∀ (x : ℕ), x < 4032 → ((x^2 - 20) % 16 = 0) ∧ ((x^2 - 16) % 20 = 0) ↔ (∃ k1 k2 : ℕ, (x = 80 * k1 + 6 ∨ x = 80 * k2 + 74) ∧ k1 + k2 + 1 = n)) :=
by sorry

end problem_statement_l118_118528


namespace minimum_omega_l118_118171

theorem minimum_omega (ω : ℕ) (h_pos : ω ∈ {n : ℕ | n > 0}) (h_cos_center : ∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + π / 2) :
  ω = 2 :=
by { sorry }

end minimum_omega_l118_118171


namespace range_of_a_l118_118288

noncomputable def f (x : ℤ) (a : ℝ) := (3 * x^2 + a * x + 26) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℕ+, f x a ≤ 2) → a ≤ -15 :=
by
  sorry

end range_of_a_l118_118288


namespace calculate_expression_l118_118552

theorem calculate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 :=
by
  sorry

end calculate_expression_l118_118552


namespace parabola_focus_l118_118166

theorem parabola_focus (x : ℝ) : ∃ f : ℝ × ℝ, f = (0, 1 / 4) ∧ ∀ y : ℝ, y = x^2 → f = (0, 1 / 4) :=
by
  sorry

end parabola_focus_l118_118166


namespace smallest_prime_divisor_and_cube_root_l118_118530

theorem smallest_prime_divisor_and_cube_root (N : ℕ) (p : ℕ) (q : ℕ)
  (hN_composite : N > 1 ∧ ¬ (∃ p : ℕ, p > 1 ∧ p < N ∧ N = p))
  (h_divisor : N = p * q)
  (h_p_prime : Nat.Prime p)
  (h_min_prime : ∀ (d : ℕ), Nat.Prime d → d ∣ N → p ≤ d)
  (h_cube_root : p > Nat.sqrt (Nat.sqrt N)) :
  Nat.Prime q := 
sorry

end smallest_prime_divisor_and_cube_root_l118_118530


namespace total_cost_of_purchase_l118_118187

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l118_118187


namespace intersection_A_B_l118_118652

def A : Set ℝ := { x | Real.sqrt x ≤ 3 }
def B : Set ℝ := { x | x^2 ≤ 9 }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end intersection_A_B_l118_118652


namespace initial_men_count_l118_118227

theorem initial_men_count
  (M : ℕ)
  (h1 : ∀ T : ℕ, (M * 8 * 10 = T) → (5 * 16 * 12 = T)) :
  M = 12 :=
by
  sorry

end initial_men_count_l118_118227


namespace common_ratio_is_63_98_l118_118329

/-- Define the terms of the geometric series -/
def term (n : Nat) : ℚ := 
  match n with
  | 0 => 4 / 7
  | 1 => 18 / 49
  | 2 => 162 / 343
  | _ => sorry  -- For simplicity, we can define more terms if needed, but it's irrelevant for our proof

/-- Define the common ratio of the geometric series -/
def common_ratio (a b : ℚ) : ℚ := b / a

/-- The problem states that the common ratio of first two terms of the given series is equal to 63/98 -/
theorem common_ratio_is_63_98 : common_ratio (term 0) (term 1) = 63 / 98 :=
by
  -- leave the proof as sorry for now
  sorry

end common_ratio_is_63_98_l118_118329


namespace smallest_x_with_18_factors_and_factors_18_21_exists_l118_118473

def has_18_factors (x : ℕ) : Prop :=
(x.factors.length == 18)

def is_factor (a b : ℕ) : Prop :=
(b % a == 0)

theorem smallest_x_with_18_factors_and_factors_18_21_exists :
  ∃ x : ℕ, has_18_factors x ∧ is_factor 18 x ∧ is_factor 21 x ∧ ∀ y : ℕ, has_18_factors y ∧ is_factor 18 y ∧ is_factor 21 y → y ≥ x :=
sorry

end smallest_x_with_18_factors_and_factors_18_21_exists_l118_118473


namespace largest_possible_s_l118_118314

noncomputable def max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) : ℝ :=
  2 + 3 * Real.sqrt 2

theorem largest_possible_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) :
  s ≤ max_value_of_s p q r s h1 h2 := 
sorry

end largest_possible_s_l118_118314


namespace interest_rate_proof_l118_118644

noncomputable def remaining_interest_rate (total_investment yearly_interest part_investment interest_rate_part amount_remaining_interest : ℝ) : Prop :=
  (part_investment * interest_rate_part) + amount_remaining_interest = yearly_interest ∧
  (total_investment - part_investment) * (amount_remaining_interest / (total_investment - part_investment)) = amount_remaining_interest

theorem interest_rate_proof :
  remaining_interest_rate 3000 256 800 0.1 176 :=
by
  sorry

end interest_rate_proof_l118_118644


namespace how_many_correct_l118_118141

def calc1 := (2 * Real.sqrt 3) * (3 * Real.sqrt 3) = 6 * Real.sqrt 3
def calc2 := Real.sqrt 2 + Real.sqrt 3 = Real.sqrt 5
def calc3 := (5 * Real.sqrt 5) - (2 * Real.sqrt 2) = 3 * Real.sqrt 3
def calc4 := (Real.sqrt 2) / (Real.sqrt 3) = (Real.sqrt 6) / 3

theorem how_many_correct : (¬ calc1) ∧ (¬ calc2) ∧ (¬ calc3) ∧ calc4 → 1 = 1 :=
by { sorry }

end how_many_correct_l118_118141


namespace main_theorem_l118_118371

def f (m: ℕ) : ℕ := m * (m + 1) / 2

lemma f_1 : f 1 = 1 := by 
  -- placeholder for proof
  sorry

lemma f_functional_eq (m n : ℕ) : f m + f n = f (m + n) - m * n := by
  -- placeholder for proof
  sorry

theorem main_theorem (m : ℕ) : f m = m * (m + 1) / 2 := by
  -- Combining the conditions to conclude the result
  sorry

end main_theorem_l118_118371


namespace sqrt_529000_pow_2_5_l118_118577

theorem sqrt_529000_pow_2_5 : (529000 ^ (1 / 2) ^ (5 / 2)) = 14873193 := by
  sorry

end sqrt_529000_pow_2_5_l118_118577


namespace total_amount_paid_l118_118544

-- Definitions based on conditions
def original_price : ℝ := 100
def discount_rate : ℝ := 0.20
def additional_discount : ℝ := 5
def sales_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_amount_paid :
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price - additional_discount
  let total_price_with_tax := final_price * (1 + sales_tax_rate)
  total_price_with_tax = 81 := sorry

end total_amount_paid_l118_118544


namespace number_of_circumcenter_quadrilaterals_l118_118611

-- Definitions for each type of quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def is_square (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_kite (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def has_circumcenter (q : Quadrilateral) : Prop := sorry

-- List of quadrilaterals
def square : Quadrilateral := sorry
def rectangle : Quadrilateral := sorry
def rhombus : Quadrilateral := sorry
def kite : Quadrilateral := sorry
def trapezoid : Quadrilateral := sorry

-- Proof that the number of quadrilaterals with a point equidistant from all vertices is 2
theorem number_of_circumcenter_quadrilaterals :
  (has_circumcenter square) ∧
  (has_circumcenter rectangle) ∧
  ¬ (has_circumcenter rhombus) ∧
  ¬ (has_circumcenter kite) ∧
  ¬ (has_circumcenter trapezoid) →
  2 = 2 :=
by
  sorry

end number_of_circumcenter_quadrilaterals_l118_118611


namespace avg_speed_3x_km_l118_118232

-- Definitions based on the conditions
def distance1 (x : ℕ) : ℕ := x
def speed1 : ℕ := 90
def distance2 (x : ℕ) : ℕ := 2 * x
def speed2 : ℕ := 20

-- The total distance covered
def total_distance (x : ℕ) : ℕ := distance1 x + distance2 x

-- The time taken for each part of the journey
def time1 (x : ℕ) : ℚ := distance1 x / speed1
def time2 (x : ℕ) : ℚ := distance2 x / speed2

-- The total time taken
def total_time (x : ℕ) : ℚ := time1 x + time2 x

-- The average speed
def average_speed (x : ℕ) : ℚ := total_distance x / total_time x

-- The theorem we want to prove
theorem avg_speed_3x_km (x : ℕ) : average_speed x = 27 := by
  sorry

end avg_speed_3x_km_l118_118232


namespace total_lunch_bill_l118_118014

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l118_118014


namespace triangle_side_ratio_l118_118184

variables (a b c S : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
    and given a=1, B=π/4, and the area S=2, we prove that b / sin(B) = 5√2. -/
theorem triangle_side_ratio (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : b / Real.sin B = 5 * Real.sqrt 2 :=
sorry

end triangle_side_ratio_l118_118184


namespace no_integer_solutions_l118_118242

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end no_integer_solutions_l118_118242


namespace find_number_l118_118721

theorem find_number (x : ℝ) (h : 3034 - (1002 / x) = 2984) : x = 20.04 :=
by
  sorry

end find_number_l118_118721


namespace sum_of_selected_sections_l118_118712

-- Given volumes of a bamboo, we denote them as a1, a2, ..., a9 forming an arithmetic sequence.
-- Where the sum of the volumes of the top four sections is 3 liters, and the
-- sum of the volumes of the bottom three sections is 4 liters.

-- Definitions based on the conditions
def arith_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℝ} {d : ℝ}
variable (sum_top_four : a 1 + a 2 + a 3 + a 4 = 3)
variable (sum_bottom_three : a 7 + a 8 + a 9 = 4)
variable (seq_condition : arith_seq a d)

theorem sum_of_selected_sections 
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 7 + a 8 + a 9 = 4)
  (h_seq : arith_seq a d) : 
  a 2 + a 3 + a 8 = 17 / 6 := 
sorry -- proof goes here

end sum_of_selected_sections_l118_118712


namespace find_b_l118_118034

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

-- Assertion we need to prove
theorem find_b (b : ℝ) (h : p (q 3 b) = 3) : b = 4 := 
by
  sorry

end find_b_l118_118034


namespace arithmetic_sequence_closed_form_l118_118606

noncomputable def B_n (n : ℕ) : ℝ :=
  2 * (1 - (-2)^n) / 3

theorem arithmetic_sequence_closed_form (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : a_n 1 = 1) (h2 : S_n 3 = 0) :
  B_n n = 2 * (1 - (-2)^n) / 3 := sorry

end arithmetic_sequence_closed_form_l118_118606


namespace find_number_l118_118613

theorem find_number (x : ℤ) (h : 3 * x - 6 = 2 * x) : x = 6 :=
by
  sorry

end find_number_l118_118613


namespace total_games_played_l118_118814

theorem total_games_played (won_games : ℕ) (won_ratio : ℕ) (lost_ratio : ℕ) (tied_ratio : ℕ) (total_games : ℕ) :
  won_games = 42 →
  won_ratio = 7 →
  lost_ratio = 4 →
  tied_ratio = 5 →
  total_games = won_games + lost_ratio * (won_games / won_ratio) + tied_ratio * (won_games / won_ratio) →
  total_games = 96 :=
by
  intros h_won h_won_ratio h_lost_ratio h_tied_ratio h_total
  sorry

end total_games_played_l118_118814


namespace eccentricity_of_ellipse_l118_118821

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse:
  ∀ (a b : ℝ) (c : ℝ), 
    0 < b ∧ b < a ∧ a = 3 * c → 
    ellipse_eccentricity a b c = 1/3 := by
  intros a b c h
  let e := ellipse_eccentricity a b c
  have h1 : 0 < b := h.1
  have h2 : b < a := h.2.left
  have h3 : a = 3 * c := h.2.right
  simp [ellipse_eccentricity, h3]
  sorry

end eccentricity_of_ellipse_l118_118821


namespace man_l118_118990

theorem man's_age (x : ℕ) : 6 * (x + 6) - 6 * (x - 6) = x → x = 72 :=
by
  sorry

end man_l118_118990


namespace mean_of_remaining_four_numbers_l118_118285

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h1 : (a + b + c + d + 106) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.5 := 
sorry

end mean_of_remaining_four_numbers_l118_118285


namespace cubic_polynomial_sum_l118_118449

-- Define the roots and their properties according to Vieta's formulas
variables {p q r : ℝ}
axiom root_poly : p * q * r = -1
axiom pq_sum : p * q + p * r + q * r = -3
axiom roots_sum : p + q + r = 0

-- Define the target equality to prove
theorem cubic_polynomial_sum :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 3 :=
by
  sorry

end cubic_polynomial_sum_l118_118449


namespace find_j_l118_118504

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_j
  (a b c : ℤ)
  (h1 : f a b c 2 = 0)
  (h2 : 200 < f a b c 10 ∧ f a b c 10 < 300)
  (h3 : 400 < f a b c 9 ∧ f a b c 9 < 500)
  (j : ℤ)
  (h4 : 1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1)) :
  j = 36 := sorry

end find_j_l118_118504


namespace find_constants_l118_118393

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 3 then a * x^2 + b else 10 - 2 * x

theorem find_constants (a b : ℝ)
  (H : ∀ x, f a b (f a b x) = x) :
  a + b = 13 / 3 := by 
  sorry

end find_constants_l118_118393


namespace Amanda_hiking_trip_l118_118783

-- Define the conditions
variable (x : ℝ) -- the total distance of Amanda's hiking trip
variable (forest_path : ℝ) (plain_path : ℝ)
variable (stream_path : ℝ) (mountain_path : ℝ)

-- Given conditions
axiom h1 : stream_path = (1/4) * x
axiom h2 : forest_path = 25
axiom h3 : mountain_path = (1/6) * x
axiom h4 : plain_path = 2 * forest_path
axiom h5 : stream_path + forest_path + mountain_path + plain_path = x

-- Proposition to prove
theorem Amanda_hiking_trip : x = 900 / 7 :=
by
  sorry

end Amanda_hiking_trip_l118_118783


namespace intersection_of_A_and_B_l118_118534

namespace SetsIntersectionProof

def setA : Set ℝ := { x | |x| ≤ 2 }
def setB : Set ℝ := { x | x < 1 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | -2 ≤ x ∧ x < 1 } :=
sorry

end SetsIntersectionProof

end intersection_of_A_and_B_l118_118534


namespace cost_per_yellow_ink_l118_118123

def initial_amount : ℕ := 50
def cost_per_black_ink : ℕ := 11
def num_black_inks : ℕ := 2
def cost_per_red_ink : ℕ := 15
def num_red_inks : ℕ := 3
def additional_amount_needed : ℕ := 43
def num_yellow_inks : ℕ := 2

theorem cost_per_yellow_ink :
  let total_cost_needed := initial_amount + additional_amount_needed
  let total_black_ink_cost := cost_per_black_ink * num_black_inks
  let total_red_ink_cost := cost_per_red_ink * num_red_inks
  let total_non_yellow_cost := total_black_ink_cost + total_red_ink_cost
  let total_yellow_ink_cost := total_cost_needed - total_non_yellow_cost
  let cost_per_yellow_ink := total_yellow_ink_cost / num_yellow_inks
  cost_per_yellow_ink = 13 :=
by
  sorry

end cost_per_yellow_ink_l118_118123


namespace haley_money_l118_118136

variable (x : ℕ)

def initial_amount : ℕ := 2
def difference : ℕ := 11
def total_amount (x : ℕ) : ℕ := x

theorem haley_money : total_amount x - initial_amount = difference → total_amount x = 13 := by
  sorry

end haley_money_l118_118136


namespace age_of_youngest_person_l118_118871

theorem age_of_youngest_person :
  ∃ (a1 a2 a3 a4 : ℕ), 
  (a1 < a2) ∧ (a2 < a3) ∧ (a3 < a4) ∧ 
  (a4 = 50) ∧ 
  (a1 + a2 + a3 + a4 = 158) ∧ 
  (a2 - a1 = a3 - a2) ∧ (a3 - a2 = a4 - a3) ∧ 
  a1 = 29 :=
by
  sorry

end age_of_youngest_person_l118_118871


namespace right_angled_triangle_setB_l118_118659

def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

theorem right_angled_triangle_setB :
  isRightAngledTriangle 1 1 (Real.sqrt 2) ∧
  ¬isRightAngledTriangle 1 2 3 ∧
  ¬isRightAngledTriangle 6 8 11 ∧
  ¬isRightAngledTriangle 2 3 4 :=
by
  sorry

end right_angled_triangle_setB_l118_118659


namespace product_102_108_l118_118420

theorem product_102_108 : (102 = 105 - 3) → (108 = 105 + 3) → (102 * 108 = 11016) := by
  sorry

end product_102_108_l118_118420


namespace product_of_four_consecutive_integers_divisible_by_24_l118_118514

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_24_l118_118514


namespace relationship_between_m_and_n_l118_118497

variable (x : ℝ)

def m := x^2 + 2*x + 3
def n := 2

theorem relationship_between_m_and_n :
  m x ≥ n := by
  sorry

end relationship_between_m_and_n_l118_118497


namespace even_sum_count_l118_118679

theorem even_sum_count (x y : ℕ) 
  (hx : x = (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)) 
  (hy : y = ((60 - 40) / 2 + 1)) : 
  x + y = 561 := 
by 
  sorry

end even_sum_count_l118_118679


namespace range_of_m_l118_118100

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * x - m < 0 ∧ -1 ≤ x ∧ x ≤ 2) →
  (∃ x : ℝ, x^2 - x - 2 > 0) →
  (∀ x : ℝ, 4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) →
  m > 8 :=
sorry

end range_of_m_l118_118100


namespace negate_exactly_one_even_l118_118342

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even :
  ¬(is_even a ∧ is_odd b ∧ is_odd c ∨ is_odd a ∧ is_even b ∧ is_odd c ∨ is_odd a ∧ is_odd b ∧ is_even c) ↔
  (is_even a ∧ is_even b ∨ is_even a ∧ is_even c ∨ is_even b ∧ is_even c ∨ is_odd a ∧ is_odd b ∧ is_odd c) := sorry

end negate_exactly_one_even_l118_118342


namespace second_number_is_sixty_l118_118071

theorem second_number_is_sixty (x : ℕ) (h_sum : 2 * x + x + (2 / 3) * x = 220) : x = 60 :=
by
  sorry

end second_number_is_sixty_l118_118071


namespace find_b_age_l118_118506

variable (a b c : ℕ)
-- Condition 1: a is two years older than b
variable (h1 : a = b + 2)
-- Condition 2: b is twice as old as c
variable (h2 : b = 2 * c)
-- Condition 3: The total of the ages of a, b, and c is 17
variable (h3 : a + b + c = 17)

theorem find_b_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 17) : b = 6 :=
by
  sorry

end find_b_age_l118_118506


namespace classes_Mr_Gates_has_l118_118224

theorem classes_Mr_Gates_has (buns_per_package packages_bought students_per_class buns_per_student : ℕ) :
  buns_per_package = 8 → 
  packages_bought = 30 → 
  students_per_class = 30 → 
  buns_per_student = 2 → 
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 := 
by
  sorry

end classes_Mr_Gates_has_l118_118224


namespace distinct_banners_l118_118394

inductive Color
| red
| white
| blue
| green
| yellow

def adjacent_different (a b : Color) : Prop := a ≠ b

theorem distinct_banners : 
  ∃ n : ℕ, n = 320 ∧ ∀ strips : Fin 4 → Color, 
    adjacent_different (strips 0) (strips 1) ∧ 
    adjacent_different (strips 1) (strips 2) ∧ 
    adjacent_different (strips 2) (strips 3) :=
sorry

end distinct_banners_l118_118394


namespace chenny_friends_count_l118_118183

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l118_118183


namespace sum_first_seven_arithmetic_l118_118706

theorem sum_first_seven_arithmetic (a : ℕ) (d : ℕ) (h : a + 3 * d = 3) :
    let a1 := a
    let a2 := a + d
    let a3 := a + 2 * d
    let a4 := a + 3 * d
    let a5 := a + 4 * d
    let a6 := a + 5 * d
    let a7 := a + 6 * d
    a1 + a2 + a3 + a4 + a5 + a6 + a7 = 21 :=
by
  sorry

end sum_first_seven_arithmetic_l118_118706


namespace part1_part2_part3_l118_118436

universe u

def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12 * x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}
def CR_A : Set ℝ := {x | x < -3 ∨ x ≥ 7}

theorem part1 : A ∪ B = {x | -3 ≤ x ∧ x < 10} := by
  sorry

theorem part2 : CR_A ∩ B = {x | 7 ≤ x ∧ x < 10} := by
  sorry

theorem part3 (a : ℝ) (h : (A ∩ C a).Nonempty) : a > -3 := by
  sorry

end part1_part2_part3_l118_118436


namespace remainder_problem_l118_118802

theorem remainder_problem :
  (1234567 % 135 = 92) ∧ ((92 * 5) % 27 = 1) := by
  sorry

end remainder_problem_l118_118802


namespace percent_value_in_quarters_l118_118213

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l118_118213


namespace sufficient_not_necessary_condition_l118_118801

-- Definition of the conditions
def Q (x : ℝ) : Prop := x^2 - x - 2 > 0
def P (x a : ℝ) : Prop := |x| > a

-- Main statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, P x a → Q x) → a ≥ 2 :=
by
  sorry

end sufficient_not_necessary_condition_l118_118801


namespace trig_simplification_l118_118727

theorem trig_simplification :
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
sorry

end trig_simplification_l118_118727


namespace cost_of_pen_is_five_l118_118748

-- Define the given conditions
def pencils_per_box := 80
def num_boxes := 15
def total_pencils := num_boxes * pencils_per_box
def cost_per_pencil := 4
def total_cost_of_stationery := 18300
def additional_pens := 300
def num_pens := 2 * total_pencils + additional_pens

-- Calculate total cost of pencils
def total_cost_of_pencils := total_pencils * cost_per_pencil

-- Calculate total cost of pens
def total_cost_of_pens := total_cost_of_stationery - total_cost_of_pencils

-- The conjecture to prove
theorem cost_of_pen_is_five :
  (total_cost_of_pens / num_pens) = 5 :=
sorry

end cost_of_pen_is_five_l118_118748


namespace logarithm_base_l118_118444

theorem logarithm_base (x : ℝ) (b : ℝ) : (9 : ℝ)^(x + 5) = (16 : ℝ)^x → b = 16 / 9 → x = Real.log 9^5 / Real.log b := by sorry

end logarithm_base_l118_118444


namespace prob_red_blue_calc_l118_118757

noncomputable def prob_red_blue : ℚ :=
  let p_yellow := (6 : ℚ) / 13
  let p_red_blue_given_yellow := (7 : ℚ) / 12
  let p_red_blue_given_not_yellow := (7 : ℚ) / 13
  p_red_blue_given_yellow * p_yellow + p_red_blue_given_not_yellow * (1 - p_yellow)

/-- The probability of drawing a red or blue marble from the updated bag contents is 91/169. -/
theorem prob_red_blue_calc : prob_red_blue = 91 / 169 :=
by
  -- This proof is omitted as per instructions.
  sorry

end prob_red_blue_calc_l118_118757


namespace problem_proof_l118_118650

-- Define the mixed numbers and their conversions to improper fractions
def mixed_number_1 := 84 * 19 + 4  -- 1600
def mixed_number_2 := 105 * 19 + 5 -- 2000 

-- Define the improper fractions
def improper_fraction_1 := mixed_number_1 / 19
def improper_fraction_2 := mixed_number_2 / 19

-- Define the decimals and their conversions to fractions
def decimal_1 := 11 / 8  -- 1.375
def decimal_2 := 9 / 10  -- 0.9

-- Perform the multiplications
def multiplication_1 := (improper_fraction_1 * decimal_1 : ℚ)
def multiplication_2 := (improper_fraction_2 * decimal_2 : ℚ)

-- Perform the addition
def addition_result := multiplication_1 + multiplication_2

-- The final result is converted to a fraction for comparison
def final_result := 4000 / 19

-- Define and state the theorem
theorem problem_proof : addition_result = final_result := by
  sorry

end problem_proof_l118_118650


namespace a_10_eq_505_l118_118214

-- The sequence definition
def a (n : ℕ) : ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.sum (List.range' start n)

-- Theorem that the 10th term of the sequence is 505
theorem a_10_eq_505 : a 10 = 505 := 
by
  sorry

end a_10_eq_505_l118_118214


namespace parabola_line_intersect_solutions_count_l118_118844

theorem parabola_line_intersect_solutions_count :
  ∃ b1 b2 : ℝ, (b1 ≠ b2 ∧ (b1^2 - b1 - 3 = 0) ∧ (b2^2 - b2 - 3 = 0)) :=
by
  sorry

end parabola_line_intersect_solutions_count_l118_118844


namespace book_cost_l118_118268

theorem book_cost (p : ℝ) (h1 : 14 * p < 25) (h2 : 16 * p > 28) : 1.75 < p ∧ p < 1.7857 :=
by
  -- This is where the proof would go
  sorry

end book_cost_l118_118268


namespace cos_theta_is_correct_l118_118833

def vector_1 : ℝ × ℝ := (4, 5)
def vector_2 : ℝ × ℝ := (2, 7)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1 * v1.1 + v1.2 * v1.2) * Real.sqrt (v2.1 * v2.1 + v2.2 * v2.2))

theorem cos_theta_is_correct :
  cos_theta vector_1 vector_2 = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
by
  -- proof goes here
  sorry

end cos_theta_is_correct_l118_118833


namespace max_sum_of_lengths_l118_118907

def length_of_integer (k : ℤ) (hk : k > 1) : ℤ := sorry

theorem max_sum_of_lengths (x y : ℤ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
  length_of_integer x hx + length_of_integer y hy = 15 :=
sorry

end max_sum_of_lengths_l118_118907


namespace five_less_than_sixty_percent_of_cats_l118_118482

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l118_118482


namespace integer_solutions_eq_l118_118312

theorem integer_solutions_eq (x y : ℤ) (h : y^2 = x^3 + (x + 1)^2) : (x, y) = (0, 1) ∨ (x, y) = (0, -1) :=
by
  sorry

end integer_solutions_eq_l118_118312


namespace geometric_sequence_n_terms_l118_118118

/-- In a geometric sequence with the first term a₁ and common ratio q,
the number of terms n for which the nth term aₙ has a given value -/
theorem geometric_sequence_n_terms (a₁ aₙ q : ℚ) (n : ℕ)
  (h1 : a₁ = 9/8)
  (h2 : aₙ = 1/3)
  (h3 : q = 2/3)
  (h_seq : aₙ = a₁ * q^(n-1)) :
  n = 4 := sorry

end geometric_sequence_n_terms_l118_118118


namespace find_nm_l118_118572

theorem find_nm :
  ∃ n m : Int, (-120 : Int) ≤ n ∧ n ≤ 120 ∧ (-120 : Int) ≤ m ∧ m ≤ 120 ∧ 
  (Real.sin (n * Real.pi / 180) = Real.sin (580 * Real.pi / 180)) ∧ 
  (Real.cos (m * Real.pi / 180) = Real.cos (300 * Real.pi / 180)) ∧ 
  n = -40 ∧ m = -60 := by
  sorry

end find_nm_l118_118572


namespace trapezoid_prob_l118_118188

noncomputable def trapezoid_probability_not_below_x_axis : ℝ :=
  let P := (4, 4)
  let Q := (-4, -4)
  let R := (-10, -4)
  let S := (-2, 4)
  -- Coordinates of intersection points
  let T := (0, 0)
  let U := (-6, 0)
  -- Compute the probability
  (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40)

theorem trapezoid_prob :
  trapezoid_probability_not_below_x_axis = (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40) :=
sorry

end trapezoid_prob_l118_118188


namespace problem_1_problem_2_l118_118308

noncomputable def A := Real.pi / 3
noncomputable def b := 5
noncomputable def c := 4 -- derived from the solution
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem problem_1 (A : ℝ) 
  (h : Real.cos (2 * A) - 3 * Real.cos (Real.pi - A) = 1) 
  : A = Real.pi / 3 :=
sorry

theorem problem_2 (a : ℝ) 
  (b : ℝ) 
  (S : ℝ) 
  (h_b : b = 5) 
  (h_S : S = 5 * Real.sqrt 3) 
  : a = Real.sqrt 21 :=
sorry

end problem_1_problem_2_l118_118308


namespace mandy_more_than_three_friends_l118_118978

noncomputable def stickers_given_to_three_friends : ℕ := 4 * 3
noncomputable def total_initial_stickers : ℕ := 72
noncomputable def stickers_left : ℕ := 42
noncomputable def total_given_away : ℕ := total_initial_stickers - stickers_left
noncomputable def mandy_justin_total : ℕ := total_given_away - stickers_given_to_three_friends
noncomputable def mandy_stickers : ℕ := 14
noncomputable def three_friends_stickers : ℕ := stickers_given_to_three_friends

theorem mandy_more_than_three_friends : 
  mandy_stickers - three_friends_stickers = 2 :=
by
  sorry

end mandy_more_than_three_friends_l118_118978


namespace log5_x_equals_neg_two_log5_2_l118_118130

theorem log5_x_equals_neg_two_log5_2 (x : ℝ) (h : x = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3)) :
  Real.log x / Real.log 5 = -2 * (Real.log 2 / Real.log 5) :=
by
  sorry

end log5_x_equals_neg_two_log5_2_l118_118130


namespace duration_of_period_l118_118483

variable (t : ℝ)

theorem duration_of_period:
  (2800 * 0.185 * t - 2800 * 0.15 * t = 294) ↔ (t = 3) :=
by
  sorry

end duration_of_period_l118_118483


namespace polynomial_properties_l118_118134

noncomputable def p (x : ℕ) : ℕ := 2 * x^3 + x + 4

theorem polynomial_properties :
  p 1 = 7 ∧ p 10 = 2014 := 
by
  -- Placeholder for proof
  sorry

end polynomial_properties_l118_118134


namespace bisection_method_third_interval_l118_118229

theorem bisection_method_third_interval 
  (f : ℝ → ℝ) (a b : ℝ) (H1 : a = -2) (H2 : b = 4) 
  (H3 : f a * f b ≤ 0) : 
  ∃ c d : ℝ, c = -1/2 ∧ d = 1 ∧ f c * f d ≤ 0 :=
by 
  sorry

end bisection_method_third_interval_l118_118229


namespace correct_answers_is_36_l118_118724

noncomputable def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 50) ∧ (4 * c - w = 130)

theorem correct_answers_is_36 (c w : ℕ) (h : num_correct_answers c w) : c = 36 :=
by
  sorry

end correct_answers_is_36_l118_118724


namespace bus_car_ratio_l118_118762

variable (R C Y : ℝ)

noncomputable def ratio_of_bus_to_car (R C Y : ℝ) : ℝ :=
  R / C

theorem bus_car_ratio 
  (h1 : R = 48) 
  (h2 : Y = 3.5 * C) 
  (h3 : Y = R - 6) : 
  ratio_of_bus_to_car R C Y = 4 :=
by sorry

end bus_car_ratio_l118_118762


namespace non_congruent_parallelograms_l118_118352

def side_lengths_sum (a b : ℕ) : Prop :=
  a + b = 25

def is_congruent (a b : ℕ) (a' b' : ℕ) : Prop :=
  (a = a' ∧ b = b') ∨ (a = b' ∧ b = a')

def non_congruent_count (n : ℕ) : Prop :=
  ∀ (a b : ℕ), side_lengths_sum a b → 
  ∃! (m : ℕ), is_congruent a b m b

theorem non_congruent_parallelograms :
  ∃ (n : ℕ), non_congruent_count n ∧ n = 13 :=
sorry

end non_congruent_parallelograms_l118_118352


namespace xiaohongs_mother_deposit_l118_118162

theorem xiaohongs_mother_deposit (x : ℝ) :
  x + x * 3.69 / 100 * 3 * (1 - 20 / 100) = 5442.8 :=
by
  sorry

end xiaohongs_mother_deposit_l118_118162


namespace simplify_expression_l118_118935

theorem simplify_expression (x : ℝ) : 3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + x^3) = -x^3 + 9 * x^2 + 6 * x - 3 :=
by
  sorry -- Proof is omitted.

end simplify_expression_l118_118935


namespace geom_seq_sum_eqn_l118_118481

theorem geom_seq_sum_eqn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 2 + 2 * a 3 = a 1)
  (h2 : a 1 * a 4 = a 6)
  (h3 : ∀ n, a (n + 1) = a 1 * (1 / 2) ^ n)
  (h4 : ∀ n, S n = 2 * ((1 - (1 / 2) ^ n) / (1 - (1 / 2)))) :
  a n + S n = 4 :=
sorry

end geom_seq_sum_eqn_l118_118481


namespace highway_length_l118_118881

theorem highway_length 
  (speed_car1 speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 54)
  (h_speed_car2 : speed_car2 = 57)
  (h_time : time = 3) : 
  speed_car1 * time + speed_car2 * time = 333 := by
  sorry

end highway_length_l118_118881


namespace area_of_inscribed_square_l118_118860

theorem area_of_inscribed_square (a : ℝ) : 
    ∃ S : ℝ, S = 3 * a^2 / (7 - 4 * Real.sqrt 3) :=
by
  sorry

end area_of_inscribed_square_l118_118860


namespace no_solution_system_l118_118299

theorem no_solution_system (a : ℝ) :
  (∀ (x : ℝ), (a ≠ 0 → (a^2 * x + 2 * a) / (a * x - 2 + a^2) < 0 ∨ ax + a ≤ 5/4)) ∧ 
  (a = 0 → ¬ ∃ (x : ℝ), (a^2 * x + 2 * a) / (a * x - 2 + a^2) ≥ 0 ∧ ax + a > 5/4) ↔ 
  a ∈ Set.Iic (-1/2) ∪ {0} :=
by sorry

end no_solution_system_l118_118299


namespace sum_of_first_and_third_is_68_l118_118660

theorem sum_of_first_and_third_is_68
  (A B C : ℕ)
  (h1 : A + B + C = 98)
  (h2 : A * 3 = B * 2)  -- implying A / B = 2 / 3
  (h3 : B * 8 = C * 5)  -- implying B / C = 5 / 8
  (h4 : B = 30) :
  A + C = 68 :=
sorry

end sum_of_first_and_third_is_68_l118_118660


namespace sum_of_fractions_l118_118472

theorem sum_of_fractions : (1 / 1) + (2 / 2) + (3 / 3) = 3 := 
by 
  norm_num

end sum_of_fractions_l118_118472


namespace largest_angle_is_90_degrees_l118_118236

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem largest_angle_is_90_degrees (u : ℝ) (a b c : ℝ) (v : ℝ) (h_v : v = 1)
  (h_a : a = Real.sqrt (2 * u - 1))
  (h_b : b = Real.sqrt (2 * u + 3))
  (h_c : c = 2 * Real.sqrt (u + v)) :
  is_right_triangle a b c :=
by
  sorry

end largest_angle_is_90_degrees_l118_118236


namespace probability_red_white_green_probability_any_order_l118_118840

-- Definitions based on the conditions
def total_balls := 28
def red_balls := 15
def white_balls := 9
def green_balls := 4

-- Part (a): Probability of first red, second white, third green
theorem probability_red_white_green : 
  (red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2)) = 5 / 182 :=
by 
  sorry

-- Part (b): Probability of red, white, and green in any order
theorem probability_any_order :
  6 * ((red_balls / total_balls) * (white_balls / (total_balls - 1)) * (green_balls / (total_balls - 2))) = 15 / 91 :=
by
  sorry

end probability_red_white_green_probability_any_order_l118_118840


namespace teena_speed_l118_118103

theorem teena_speed (T : ℝ) : 
  (∀ (d₀ d_poe d_ahead : ℝ), 
    d₀ = 7.5 ∧ d_poe = 40 * 1.5 ∧ d_ahead = 15 →
    T = (d₀ + d_poe + d_ahead) / 1.5) → 
  T = 55 :=
by
  intros
  sorry

end teena_speed_l118_118103


namespace side_length_square_eq_6_l118_118722

theorem side_length_square_eq_6
  (width length : ℝ)
  (h_width : width = 2)
  (h_length : length = 18) :
  (∃ s : ℝ, s^2 = width * length) ∧ (∀ s : ℝ, s^2 = width * length → s = 6) :=
by
  sorry

end side_length_square_eq_6_l118_118722


namespace pablo_days_to_complete_all_puzzles_l118_118410

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l118_118410


namespace binomial_identity_l118_118832

-- Given:
variables {k n : ℕ}

-- Conditions:
axiom h₁ : 1 < k
axiom h₂ : 1 < n

-- Statement:
theorem binomial_identity (h₁ : 1 < k) (h₂ : 1 < n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := 
sorry

end binomial_identity_l118_118832


namespace question_proof_l118_118176

theorem question_proof (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) : xy + y^2 = y^2 + y + 12 :=
by
  sorry

end question_proof_l118_118176


namespace arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l118_118447

-- Question 1
theorem arithmetic_sequence_n (a1 a4 a10 : ℤ) (d : ℤ) (n : ℤ) (Sn : ℤ) 
  (h1 : a1 + 3 * d = a4) 
  (h2 : a1 + 9 * d = a10)
  (h3 : Sn = n * (2 * a1 + (n - 1) * d) / 2)
  (h4 : a4 = 10)
  (h5 : a10 = -2)
  (h6 : Sn = 60)
  : n = 5 ∨ n = 6 := 
sorry

-- Question 2
theorem sum_arithmetic_sequence_S17 (a1 : ℤ) (d : ℤ) (a_n1 : ℤ → ℤ) (S17 : ℤ)
  (h1 : a1 = -7)
  (h2 : ∀ n, a_n1 (n + 1) = a_n1 n + d)
  (h3 : S17 = 17 * (2 * a1 + 16 * d) / 2)
  : S17 = 153 := 
sorry

-- Question 3
theorem arithmetic_sequence_S13 (a_2 a_7 a_12 : ℤ) (S13 : ℤ)
  (h1 : a_2 + a_7 + a_12 = 24)
  (h2 : S13 = a_7 * 13)
  : S13 = 104 := 
sorry

end arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l118_118447


namespace spherical_coordinates_neg_z_l118_118078

theorem spherical_coordinates_neg_z (x y z : ℝ) (h₀ : ρ = 5) (h₁ : θ = 3 * Real.pi / 4) (h₂ : φ = Real.pi / 3)
  (hx : x = ρ * Real.sin φ * Real.cos θ) 
  (hy : y = ρ * Real.sin φ * Real.sin θ) 
  (hz : z = ρ * Real.cos φ) : 
  (ρ, θ, π - φ) = (5, 3 * Real.pi / 4, 2 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_neg_z_l118_118078


namespace sphere_center_x_axis_eq_l118_118055

theorem sphere_center_x_axis_eq (a : ℝ) (R : ℝ) (x y z : ℝ) :
  (x - a) ^ 2 + y ^ 2 + z ^ 2 = R ^ 2 → (0 - a) ^ 2 + (0 - 0) ^ 2 + (0 - 0) ^ 2 = R ^ 2 →
  a = R →
  (x ^ 2 - 2 * a * x + y ^ 2 + z ^ 2 = 0) :=
by
  sorry

end sphere_center_x_axis_eq_l118_118055


namespace chips_probability_l118_118953

def total_chips : ℕ := 12
def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5

def total_ways : ℕ := Nat.factorial total_chips

def blue_group_ways : ℕ := Nat.factorial blue_chips
def green_group_ways : ℕ := Nat.factorial green_chips
def red_group_ways : ℕ := Nat.factorial red_chips
def group_permutations : ℕ := Nat.factorial 3

def satisfying_arrangements : ℕ :=
  group_permutations * blue_group_ways * green_group_ways * red_group_ways

noncomputable def probability_of_event_B : ℚ :=
  (satisfying_arrangements : ℚ) / (total_ways : ℚ)

theorem chips_probability :
  probability_of_event_B = 1 / 4620 :=
by
  sorry

end chips_probability_l118_118953


namespace problem_1_problem_2_l118_118256

noncomputable def f (x p : ℝ) := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) := 2 * Real.exp 1 / x

theorem problem_1 (p : ℝ) : 
  (∀ x : ℝ, 0 < x → p * x - p / x - 2 * Real.log x ≥ 0) ↔ p ≥ 1 := 
by sorry

theorem problem_2 (p : ℝ) : 
  (∃ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ Real.exp 1 ∧ f x_0 p > g x_0) ↔ 
  p > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
by sorry

end problem_1_problem_2_l118_118256


namespace parabola_line_non_intersect_l118_118051

def P (x : ℝ) : ℝ := x^2 + 3 * x + 1
def Q : ℝ × ℝ := (10, 50)

def line_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - Q.1) + Q.2

theorem parabola_line_non_intersect (r s : ℝ) (h : ∀ m, (r < m ∧ m < s) ↔ (∀ x, 
  x^2 + (3 - m) * x + (10 * m - 49) ≠ 0)) : r + s = 46 := 
sorry

end parabola_line_non_intersect_l118_118051


namespace min_value_p_plus_q_l118_118353

-- Definitions related to the conditions.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_equations (a b p q : ℕ) : Prop :=
  20 * a + 17 * b = p ∧ 17 * a + 20 * b = q ∧ is_prime p ∧ is_prime q

def distinct_positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- The main proof problem.
theorem min_value_p_plus_q (a b p q : ℕ) :
  distinct_positive_integers a b →
  satisfies_equations a b p q →
  p + q = 296 :=
by
  sorry

end min_value_p_plus_q_l118_118353


namespace selena_left_with_l118_118662

/-- Selena got a tip of $99 and spent money on various foods whose individual costs are provided. 
Prove that she will be left with $38. -/
theorem selena_left_with : 
  let tip := 99
  let steak_cost := 24
  let num_steaks := 2
  let burger_cost := 3.5
  let num_burgers := 2
  let ice_cream_cost := 2
  let num_ice_cream := 3
  let total_spent := (steak_cost * num_steaks) + (burger_cost * num_burgers) + (ice_cream_cost * num_ice_cream)
  tip - total_spent = 38 := 
by 
  sorry

end selena_left_with_l118_118662


namespace sum_of_fractions_l118_118316

theorem sum_of_fractions:
  (2 / 5) + (3 / 8) + (1 / 4) = 1 + (1 / 40) :=
by
  sorry

end sum_of_fractions_l118_118316


namespace original_number_is_7_l118_118857

theorem original_number_is_7 (x : ℤ) (h : (((3 * (x + 3) + 3) - 3) / 3) = 10) : x = 7 :=
sorry

end original_number_is_7_l118_118857


namespace arithmetic_seq_a7_l118_118041

structure arith_seq (a : ℕ → ℤ) : Prop :=
  (step : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_seq_a7
  {a : ℕ → ℤ}
  (h_seq : arith_seq a)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  : a 7 = 8 :=
by
  sorry

end arithmetic_seq_a7_l118_118041


namespace isosceles_triangle_perimeter_l118_118959

theorem isosceles_triangle_perimeter (a b : ℕ) (h_isosceles : a = 3 ∨ a = 7 ∨ b = 3 ∨ b = 7) (h_ineq1 : 3 + 3 ≤ b ∨ b + b ≤ 3) (h_ineq2 : 7 + 7 ≥ a ∨ a + a ≥ 7) :
  (a = 3 ∧ b = 7) → 3 + 7 + 7 = 17 :=
by
  -- To be completed
  sorry

end isosceles_triangle_perimeter_l118_118959


namespace john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l118_118251

theorem john_needs_to_sell_1200_pencils_to_make_120_dollars_profit :
  ∀ (buy_rate_pencils : ℕ) (buy_rate_dollars : ℕ) (sell_rate_pencils : ℕ) (sell_rate_dollars : ℕ),
    buy_rate_pencils = 5 →
    buy_rate_dollars = 7 →
    sell_rate_pencils = 4 →
    sell_rate_dollars = 6 →
    ∃ (n_pencils : ℕ), n_pencils = 1200 ∧ 
                        (sell_rate_dollars / sell_rate_pencils - buy_rate_dollars / buy_rate_pencils) * n_pencils = 120 :=
by
  sorry

end john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l118_118251


namespace parabola_point_b_l118_118619

variable {a b : ℝ}

theorem parabola_point_b (h1 : 6 = 2^2 + 2*a + b) (h2 : -14 = (-2)^2 - 2*a + b) : b = -8 :=
by
  -- sorry as a placeholder for the actual proof.
  sorry

end parabola_point_b_l118_118619


namespace find_remainder_when_q_divided_by_x_plus_2_l118_118358

noncomputable def q (x : ℝ) (D E F : ℝ) := D * x^4 + E * x^2 + F * x + 5

theorem find_remainder_when_q_divided_by_x_plus_2 (D E F : ℝ) :
  q 2 D E F = 15 → q (-2) D E F = 15 :=
by
  intro h
  sorry

end find_remainder_when_q_divided_by_x_plus_2_l118_118358


namespace quadratic_has_negative_root_l118_118524

def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks for a range of m such that the quadratic function intersects the negative x-axis
theorem quadratic_has_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ quadratic_function m x = 0) ↔ (1 ≤ m ∧ m < 2 ∨ 2 < m ∧ m < 3) :=
sorry

end quadratic_has_negative_root_l118_118524


namespace digit_sum_2001_not_perfect_square_l118_118854

theorem digit_sum_2001_not_perfect_square (n : ℕ) (h : (n.digits 10).sum = 2001) : ¬ ∃ k : ℕ, n = k * k := 
sorry

end digit_sum_2001_not_perfect_square_l118_118854


namespace ryan_more_hours_english_than_spanish_l118_118217

-- Define the time spent on various languages as constants
def hoursEnglish : ℕ := 7
def hoursSpanish : ℕ := 4

-- State the problem as a theorem
theorem ryan_more_hours_english_than_spanish : hoursEnglish - hoursSpanish = 3 :=
by sorry

end ryan_more_hours_english_than_spanish_l118_118217


namespace problem1_problem2_problem3_problem4_problem5_problem6_l118_118032

-- First problem: \(\frac{1}{3} + \left(-\frac{1}{2}\right) = -\frac{1}{6}\)
theorem problem1 : (1 / 3 : ℚ) + (-1 / 2) = -1 / 6 := by sorry

-- Second problem: \(-2 - \left(-9\right) = 7\)
theorem problem2 : (-2 : ℚ) - (-9) = 7 := by sorry

-- Third problem: \(\frac{15}{16} - \left(-7\frac{1}{16}\right) = 8\)
theorem problem3 : (15 / 16 : ℚ) - (-(7 + 1 / 16)) = 8 := by sorry

-- Fourth problem: \(-\left|-4\frac{2}{7}\right| - \left|+1\frac{5}{7}\right| = -6\)
theorem problem4 : -|(-4 - 2 / 7 : ℚ)| - |(1 + 5 / 7)| = -6 := by sorry

-- Fifth problem: \(6 + \left(-12\right) + 8.3 + \left(-7.5\right) = -5.2\)
theorem problem5 : (6 : ℚ) + (-12) + (83 / 10) + (-75 / 10) = -52 / 10 := by sorry

-- Sixth problem: \(\left(-\frac{1}{8}\right) + 3.25 + 2\frac{3}{5} + \left(-5.875\right) + 1.15 = 1\)
theorem problem6 : (-1 / 8 : ℚ) + 3 + 1 / 4 + 2 + 3 / 5 + (-5 - 875 / 1000) + 1 + 15 / 100 = 1 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l118_118032


namespace base_six_equals_base_b_l118_118364

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  6 * 6 + 2

noncomputable def base_b_to_decimal (b : ℕ) : ℕ :=
  b^2 + 2 * b + 4

theorem base_six_equals_base_b (b : ℕ) : b^2 + 2 * b - 34 = 0 → b = 4 := 
by sorry

end base_six_equals_base_b_l118_118364


namespace tan_double_angle_l118_118271

theorem tan_double_angle (α : ℝ) (h : 3 * Real.cos α + Real.sin α = 0) : 
    Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l118_118271


namespace sarahs_monthly_fee_l118_118156

noncomputable def fixed_monthly_fee (x y : ℝ) : Prop :=
  x + 4 * y = 30.72 ∧ 1.1 * x + 8 * y = 54.72

theorem sarahs_monthly_fee : ∃ x y : ℝ, fixed_monthly_fee x y ∧ x = 7.47 :=
by
  sorry

end sarahs_monthly_fee_l118_118156


namespace sequence_no_limit_l118_118785

noncomputable def sequence_limit (x : ℕ → ℝ) (a : ℝ) : Prop :=
    ∀ ε > 0, ∃ N, ∀ n > N, abs (x n - a) < ε

theorem sequence_no_limit (x : ℕ → ℝ) (a : ℝ) (ε : ℝ) (k : ℕ) :
    (ε > 0) ∧ (∀ n, n > k → abs (x n - a) ≥ ε) → ¬ sequence_limit x a :=
by
  sorry

end sequence_no_limit_l118_118785


namespace arithmetic_sequence_a10_l118_118938

theorem arithmetic_sequence_a10 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h_seq : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h_S4 : S 4 = 10)
  (h_S9 : S 9 = 45) :
  a 10 = 10 :=
sorry

end arithmetic_sequence_a10_l118_118938


namespace choose_positions_from_8_people_l118_118474

theorem choose_positions_from_8_people : 
  ∃ (ways : ℕ), ways = 8 * 7 * 6 := 
sorry

end choose_positions_from_8_people_l118_118474


namespace problem_solution_l118_118516

variable {A B : Set ℝ}

def definition_A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def definition_B : Set ℝ := {x | x ≤ 3}

theorem problem_solution : definition_A ∩ definition_B = definition_A :=
by
  sorry

end problem_solution_l118_118516


namespace option_A_option_C_option_D_l118_118958

noncomputable def ratio_12_11 := (12 : ℝ) / 11
noncomputable def ratio_11_10 := (11 : ℝ) / 10

theorem option_A : ratio_12_11^11 > ratio_11_10^10 := sorry

theorem option_C : ratio_12_11^10 > ratio_11_10^9 := sorry

theorem option_D : ratio_11_10^12 > ratio_12_11^13 := sorry

end option_A_option_C_option_D_l118_118958


namespace min_side_length_of_square_l118_118045

theorem min_side_length_of_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ s : ℝ, s = 
    if a < (Real.sqrt 2 + 1) * b then 
      a 
    else 
      (Real.sqrt 2 / 2) * (a + b) := 
sorry

end min_side_length_of_square_l118_118045


namespace subset_relation_l118_118475

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x + 2}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (x - 4) / Real.log 2}

-- State the proof problem
theorem subset_relation : N ⊆ M := 
sorry

end subset_relation_l118_118475


namespace harvesting_days_l118_118809

theorem harvesting_days :
  (∀ (harvesters : ℕ) (days : ℕ) (mu : ℕ), 2 * 3 * (75 : ℕ) = 450) →
  (7 * 4 * (75 : ℕ) = 2100) :=
by
  sorry

end harvesting_days_l118_118809


namespace cost_of_apples_l118_118825

theorem cost_of_apples (price_per_six_pounds : ℕ) (pounds_to_buy : ℕ) (expected_cost : ℕ) :
  price_per_six_pounds = 5 → pounds_to_buy = 18 → (expected_cost = 15) → 
  (price_per_six_pounds / 6) * pounds_to_buy = expected_cost :=
by
  intro price_per_six_pounds_eq pounds_to_buy_eq expected_cost_eq
  rw [price_per_six_pounds_eq, pounds_to_buy_eq, expected_cost_eq]
  -- the actual proof would follow, using math steps similar to the solution but skipped here
  sorry

end cost_of_apples_l118_118825


namespace part_a_l118_118403

theorem part_a (a b c : ℝ) : 
  (∀ n : ℝ, (n + 2)^2 = a * (n + 1)^2 + b * n^2 + c * (n - 1)^2) ↔ (a = 3 ∧ b = -3 ∧ c = 1) :=
by 
  sorry

end part_a_l118_118403


namespace Joe_speed_first_part_l118_118346

theorem Joe_speed_first_part
  (dist1 dist2 : ℕ)
  (speed2 avg_speed total_distance total_time : ℕ)
  (h1 : dist1 = 180)
  (h2 : dist2 = 120)
  (h3 : speed2 = 40)
  (h4 : avg_speed = 50)
  (h5 : total_distance = dist1 + dist2)
  (h6 : total_distance = 300)
  (h7 : total_time = total_distance / avg_speed)
  (h8 : total_time = 6) :
  ∃ v : ℕ, (dist1 / v + dist2 / speed2 = total_time) ∧ v = 60 :=
by
  sorry

end Joe_speed_first_part_l118_118346


namespace pairwise_coprime_circle_l118_118707

theorem pairwise_coprime_circle :
  ∃ (circle : Fin 100 → ℕ),
    (∀ i, Nat.gcd (circle i) (Nat.gcd (circle ((i + 1) % 100)) (circle ((i - 1) % 100))) = 1) → 
    ∀ i j, i ≠ j → Nat.gcd (circle i) (circle j) = 1 :=
by
  sorry

end pairwise_coprime_circle_l118_118707


namespace max_value_of_sum_on_ellipse_l118_118323

theorem max_value_of_sum_on_ellipse (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : x + y ≤ 2 :=
sorry

end max_value_of_sum_on_ellipse_l118_118323


namespace side_length_of_smaller_square_l118_118036

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square_l118_118036


namespace find_number_eq_l118_118858

theorem find_number_eq (x : ℝ) (h : (35 / 100) * x = (20 / 100) * 40) : x = 160 / 7 :=
by
  sorry

end find_number_eq_l118_118858


namespace find_value_of_S_l118_118286

theorem find_value_of_S (S : ℝ)
  (h1 : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 180) :
  S = 180 :=
sorry

end find_value_of_S_l118_118286


namespace total_coins_l118_118964

theorem total_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end total_coins_l118_118964


namespace cube_vertices_faces_edges_l118_118591

theorem cube_vertices_faces_edges (V F E : ℕ) (hv : V = 8) (hf : F = 6) (euler : V - E + F = 2) : E = 12 :=
by
  sorry

end cube_vertices_faces_edges_l118_118591


namespace height_of_fourth_person_l118_118272

/-- There are 4 people of different heights standing in order of increasing height.
    The difference is 2 inches between the first person and the second person,
    and also between the second person and the third person.
    The difference between the third person and the fourth person is 6 inches.
    The average height of the four people is 76 inches.
    Prove that the height of the fourth person is 82 inches. -/
theorem height_of_fourth_person 
  (h1 h2 h3 h4 : ℕ) 
  (h2_def : h2 = h1 + 2)
  (h3_def : h3 = h2 + 2)
  (h4_def : h4 = h3 + 6)
  (average_height : (h1 + h2 + h3 + h4) / 4 = 76) 
  : h4 = 82 :=
by sorry

end height_of_fourth_person_l118_118272


namespace students_not_like_any_l118_118927

variables (F B P T F_cap_B F_cap_P F_cap_T B_cap_P B_cap_T P_cap_T F_cap_B_cap_P_cap_T : ℕ)

def total_students := 30

def students_like_F := 18
def students_like_B := 12
def students_like_P := 14
def students_like_T := 10

def students_like_F_and_B := 8
def students_like_F_and_P := 6
def students_like_F_and_T := 4
def students_like_B_and_P := 5
def students_like_B_and_T := 3
def students_like_P_and_T := 7

def students_like_all_four := 2

theorem students_not_like_any :
  total_students - ((students_like_F + students_like_B + students_like_P + students_like_T)
                    - (students_like_F_and_B + students_like_F_and_P + students_like_F_and_T
                      + students_like_B_and_P + students_like_B_and_T + students_like_P_and_T)
                    + students_like_all_four) = 11 :=
by sorry

end students_not_like_any_l118_118927


namespace sum_digits_2_2005_times_5_2007_times_3_l118_118888

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem sum_digits_2_2005_times_5_2007_times_3 : 
  sum_of_digits (2^2005 * 5^2007 * 3) = 12 := 
by 
  sorry

end sum_digits_2_2005_times_5_2007_times_3_l118_118888


namespace evaluate_expression_l118_118351

theorem evaluate_expression (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9 / 4 :=
sorry

end evaluate_expression_l118_118351


namespace sufficient_not_necessary_condition_l118_118207

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < 1 → x < 2) ∧ ¬ (x < 2 → x < 1) :=
by
  sorry

end sufficient_not_necessary_condition_l118_118207


namespace total_equipment_cost_l118_118716

-- Definitions of costs in USD
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80
def number_of_players : ℝ := 16

-- Statement to prove
theorem total_equipment_cost :
  number_of_players * (jersey_cost + shorts_cost + socks_cost) = 752 :=
by
  sorry

end total_equipment_cost_l118_118716


namespace find_number_l118_118988

theorem find_number
  (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 328 - (100 * a + 10 * b + c) = a + b + c) :
  100 * a + 10 * b + c = 317 :=
sorry

end find_number_l118_118988


namespace solve_quadratic_equation_l118_118529

theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1)^2 = 36 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l118_118529


namespace investment_at_6_percent_l118_118797

theorem investment_at_6_percent
  (x y : ℝ) 
  (total_investment : x + y = 15000)
  (total_interest : 0.06 * x + 0.075 * y = 1023) :
  x = 6800 :=
sorry

end investment_at_6_percent_l118_118797


namespace exist_A_B_l118_118284

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exist_A_B : ∃ (A B : ℕ), A = 2016 * B ∧ sum_of_digits A + 2016 * sum_of_digits B < 0 := sorry

end exist_A_B_l118_118284


namespace simplify_polynomial_l118_118553

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial_l118_118553


namespace distance_home_to_school_l118_118179

theorem distance_home_to_school :
  ∃ T D : ℝ, 6 * (T + 7/60) = D ∧ 12 * (T - 8/60) = D ∧ 9 * T = D ∧ D = 2.1 :=
by
  sorry

end distance_home_to_school_l118_118179


namespace bill_experience_l118_118293

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l118_118293


namespace shopkeeper_weight_l118_118149

/-- A shopkeeper sells his goods at cost price but uses a certain weight instead of kilogram weight.
    His profit percentage is 25%. Prove that the weight he uses is 0.8 kilograms. -/
theorem shopkeeper_weight (c s p : ℝ) (x : ℝ) (h1 : s = c * (1 + p / 100))
  (h2 : p = 25) (h3 : c = 1) (h4 : s = 1.25) : x = 0.8 :=
by
  sorry

end shopkeeper_weight_l118_118149


namespace find_constants_l118_118452

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_constants (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_min_value : ∃ x : ℝ, a * csc (b * x + c) = 3)
  (h_period : ∀ x, a * csc (b * (x + 4 * Real.pi) + c) = a * csc (b * x + c)) :
  a = 3 ∧ b = (1 / 2) :=
by
  sorry

end find_constants_l118_118452


namespace negation_example_l118_118921

theorem negation_example :
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 1 < 0 :=
sorry

end negation_example_l118_118921


namespace y_coordinate_of_A_l118_118305

theorem y_coordinate_of_A (a : ℝ) (y : ℝ) (h1 : y = a * 1) (h2 : y = (4 - a) / 1) : y = 2 :=
by
  sorry

end y_coordinate_of_A_l118_118305


namespace find_root_and_m_l118_118648

theorem find_root_and_m (x₁ m : ℝ) (h₁ : -2 * x₁ = 2) (h₂ : x^2 + m * x + 2 = 0) : x₁ = -1 ∧ m = 3 := 
by 
  -- Proof omitted
  sorry

end find_root_and_m_l118_118648


namespace find_smallest_n_l118_118985

theorem find_smallest_n (k : ℕ) (hk: 0 < k) :
        ∃ n : ℕ, (∀ (s : Finset ℤ), s.card = n → 
        ∃ (x y : ℤ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % (2 * k) = 0 ∨ (x - y) % (2 * k) = 0) 
        ∧ n = k + 2 :=
sorry

end find_smallest_n_l118_118985


namespace total_amount_l118_118668

noncomputable def A : ℝ := 396.00000000000006
noncomputable def B : ℝ := A * (3 / 2)
noncomputable def C : ℝ := B * 4

theorem total_amount (A_eq : A = 396.00000000000006) (A_B_relation : A = (2 / 3) * B) (B_C_relation : B = (1 / 4) * C) :
  396.00000000000006 + B + C = 3366.000000000001 := by
  sorry

end total_amount_l118_118668


namespace second_train_length_l118_118453

theorem second_train_length
  (train1_length : ℝ)
  (train1_speed_kmph : ℝ)
  (train2_speed_kmph : ℝ)
  (time_to_clear : ℝ)
  (h1 : train1_length = 135)
  (h2 : train1_speed_kmph = 80)
  (h3 : train2_speed_kmph = 65)
  (h4 : time_to_clear = 7.447680047665153) :
  ∃ l2 : ℝ, l2 = 165 :=
by
  let train1_speed_mps := train1_speed_kmph * 1000 / 3600
  let train2_speed_mps := train2_speed_kmph * 1000 / 3600
  let total_distance := (train1_speed_mps + train2_speed_mps) * time_to_clear
  have : total_distance = 300 := by sorry
  have l2 := total_distance - train1_length
  use l2
  have : l2 = 165 := by sorry
  assumption

end second_train_length_l118_118453


namespace ratio_of_logs_eq_golden_ratio_l118_118438

theorem ratio_of_logs_eq_golden_ratio
  (r s : ℝ) (hr : 0 < r) (hs : 0 < s)
  (h : Real.log r / Real.log 4 = Real.log s / Real.log 18 ∧ Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) :
  s / r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_eq_golden_ratio_l118_118438


namespace min_moves_l118_118568

theorem min_moves (n : ℕ) : (n * (n + 1)) / 2 > 100 → n = 15 :=
by
  sorry

end min_moves_l118_118568


namespace initial_numbers_count_l118_118571

theorem initial_numbers_count (n : ℕ) (S : ℝ)
  (h1 : S / n = 56)
  (h2 : (S - 100) / (n - 2) = 56.25) :
  n = 50 :=
sorry

end initial_numbers_count_l118_118571


namespace basic_computer_price_l118_118004

theorem basic_computer_price (C P : ℝ)
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3)
  : C = 1500 :=
sorry

end basic_computer_price_l118_118004


namespace min_value_y_l118_118583

theorem min_value_y : ∃ x : ℝ, (y = 2 * x^2 + 8 * x + 18) ∧ (∀ x : ℝ, y ≥ 10) :=
by
  sorry

end min_value_y_l118_118583


namespace total_market_cost_l118_118086

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end total_market_cost_l118_118086


namespace solve_inequality_l118_118146

theorem solve_inequality (x : ℝ) : 
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2 / 3 ∨ x > 1) := 
by
  sorry

end solve_inequality_l118_118146


namespace set_D_is_empty_l118_118381

-- Definitions based on the conditions from the original problem
def set_A : Set ℝ := {x | x + 3 = 3}
def set_B : Set (ℝ × ℝ) := {(x, y) | y^2 = -x^2}
def set_C : Set ℝ := {x | x^2 ≤ 0}
def set_D : Set ℝ := {x | x^2 - x + 1 = 0}

-- The theorem statement
theorem set_D_is_empty : set_D = ∅ :=
sorry

end set_D_is_empty_l118_118381


namespace projectile_time_l118_118206

theorem projectile_time : ∃ t : ℝ, (60 - 8 * t - 5 * t^2 = 30) ∧ t = 1.773 := by
  sorry

end projectile_time_l118_118206


namespace problem_1_problem_2_l118_118943

def setA (x : ℝ) : Prop := 2 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 3 < x ∧ x ≤ 10
def setC (a : ℝ) (x : ℝ) : Prop := a - 5 < x ∧ x < a

theorem problem_1 (x : ℝ) :
  (setA x ∧ setB x ↔ 3 < x ∧ x < 7) ∧
  (setA x ∨ setB x ↔ 2 ≤ x ∧ x ≤ 10) := 
by sorry

theorem problem_2 (a : ℝ) :
  (∀ x, setC a x → (2 ≤ x ∧ x ≤ 10)) ↔ (7 ≤ a ∧ a ≤ 10) :=
by sorry

end problem_1_problem_2_l118_118943


namespace tom_paths_avoiding_construction_l118_118920

def tom_home : (ℕ × ℕ) := (0, 0)
def friend_home : (ℕ × ℕ) := (4, 3)
def construction_site : (ℕ × ℕ) := (2, 2)

def total_paths_without_restriction : ℕ := Nat.choose 7 4
def paths_via_construction_site : ℕ := (Nat.choose 4 2) * (Nat.choose 3 1)
def valid_paths : ℕ := total_paths_without_restriction - paths_via_construction_site

theorem tom_paths_avoiding_construction : valid_paths = 17 := by
  sorry

end tom_paths_avoiding_construction_l118_118920


namespace cubic_root_form_addition_l118_118487

theorem cubic_root_form_addition (p q r : ℕ) 
(h_root_form : ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧ x = (p^(1/3) + q^(1/3) + 2) / r) : 
  p + q + r = 10 :=
sorry

end cubic_root_form_addition_l118_118487


namespace line_slope_product_l118_118730

theorem line_slope_product (x y : ℝ) (h1 : (x, 6) = (x, 6)) (h2 : (10, y) = (10, y)) (h3 : ∀ x, y = (1 / 2) * x) : x * y = 60 :=
sorry

end line_slope_product_l118_118730


namespace tangent_line_through_P_line_through_P_chord_length_8_l118_118338

open Set

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_P : ℝ × ℝ := (3, 4)

def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 25 = 0

def line_m_case1 (x : ℝ) : Prop := x = 3

def line_m_case2 (x y : ℝ) : Prop := 7 * x - 24 * y + 75 = 0

theorem tangent_line_through_P :
  tangent_line point_P.1 point_P.2 :=
sorry

theorem line_through_P_chord_length_8 :
  (∀ x y, circle x y → line_m_case1 x ∨ line_m_case2 x y) :=
sorry

end tangent_line_through_P_line_through_P_chord_length_8_l118_118338


namespace handshake_problem_l118_118054

theorem handshake_problem (x y : ℕ) 
  (H : (x * (x - 1)) / 2 + y = 159) : 
  x = 18 ∧ y = 6 := 
sorry

end handshake_problem_l118_118054


namespace votes_cast_l118_118818

theorem votes_cast (V : ℝ) (h1 : 0.35 * V + 2250 = 0.65 * V) : V = 7500 := 
by
  sorry

end votes_cast_l118_118818


namespace part_I_part_II_l118_118155

def f (x a : ℝ) : ℝ := |x - 4 * a| + |x|

theorem part_I (a : ℝ) (h : -4 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, f x a ≥ a^2 := 
sorry

theorem part_II (x y z : ℝ) (h : 4 * x + 2 * y + z = 4) :
  (x + y)^2 + y^2 + z^2 ≥ 16 / 21 :=
sorry

end part_I_part_II_l118_118155


namespace perfect_square_condition_l118_118433

def is_perfect_square (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

noncomputable def a_n (n : ℕ) : ℤ := (10^n - 1) / 9

theorem perfect_square_condition (n b : ℕ) (h1 : 0 < b) (h2 : b < 10) :
  is_perfect_square ((a_n (2 * n)) - b * (a_n n)) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) := by
  sorry

end perfect_square_condition_l118_118433


namespace min_photos_for_condition_l118_118684

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l118_118684


namespace Kelly_initial_games_l118_118903

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end Kelly_initial_games_l118_118903


namespace max_a_no_lattice_point_l118_118379

theorem max_a_no_lattice_point (a : ℚ) : a = 35 / 51 ↔ 
  (∀ (m : ℚ), (2 / 3 < m ∧ m < a) → 
    (∀ (x : ℤ), (0 < x ∧ x ≤ 50) → 
      ¬ ∃ (y : ℤ), y = m * x + 5)) :=
sorry

end max_a_no_lattice_point_l118_118379


namespace cleaning_cost_l118_118359

theorem cleaning_cost (num_cleanings : ℕ) (chemical_cost : ℕ) (monthly_cost : ℕ) (tip_percentage : ℚ) 
  (cleaning_sessions_per_month : num_cleanings = 30 / 3)
  (monthly_chemical_cost : chemical_cost = 2 * 200)
  (total_monthly_cost : monthly_cost = 2050)
  (cleaning_cost_with_tip : monthly_cost - chemical_cost =  num_cleanings * (1 + tip_percentage) * x) : 
  x = 150 := 
by
  sorry

end cleaning_cost_l118_118359


namespace binary_op_property_l118_118173

variable (X : Type)
variable (star : X → X → X)
variable (h : ∀ x y : X, star (star x y) x = y)

theorem binary_op_property (x y : X) : star x (star y x) = y := 
by 
  sorry

end binary_op_property_l118_118173


namespace monthly_food_expense_l118_118406

-- Definitions based on the given conditions
def E : ℕ := 6000
def R : ℕ := 640
def EW : ℕ := E / 4
def I : ℕ := E / 5
def L : ℕ := 2280

-- Define the monthly food expense F
def F : ℕ := E - (R + EW + I) - L

-- The theorem stating that the monthly food expense is 380
theorem monthly_food_expense : F = 380 := 
by
  -- proof goes here
  sorry

end monthly_food_expense_l118_118406


namespace xyz_inequality_l118_118210

theorem xyz_inequality : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by
  intros
  sorry

end xyz_inequality_l118_118210


namespace nikka_us_stamp_percentage_l118_118009

/-- 
Prove that 20% of Nikka's stamp collection are US stamps given the following conditions:
1. Nikka has a total of 100 stamps.
2. 35 of those stamps are Chinese.
3. 45 of those stamps are Japanese.
-/
theorem nikka_us_stamp_percentage
  (total_stamps : ℕ)
  (chinese_stamps : ℕ)
  (japanese_stamps : ℕ)
  (h1 : total_stamps = 100)
  (h2 : chinese_stamps = 35)
  (h3 : japanese_stamps = 45) :
  ((total_stamps - (chinese_stamps + japanese_stamps)) / total_stamps) * 100 = 20 := 
by
  sorry

end nikka_us_stamp_percentage_l118_118009


namespace simplify_polynomial_problem_l118_118178

theorem simplify_polynomial_problem (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) = 2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := 
by
  sorry

end simplify_polynomial_problem_l118_118178


namespace exists_multiple_digits_0_1_l118_118241

theorem exists_multiple_digits_0_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k ≤ n) ∧ (∃ m : ℕ, m * n = k) ∧ (∀ d : ℕ, ∃ i : ℕ, i ≤ n ∧ d = 0 ∨ d = 1) :=
sorry

end exists_multiple_digits_0_1_l118_118241


namespace factorization_of_a_square_minus_one_l118_118605

theorem factorization_of_a_square_minus_one (a : ℤ) : a^2 - 1 = (a + 1) * (a - 1) := 
  by sorry

end factorization_of_a_square_minus_one_l118_118605


namespace train_cross_time_l118_118339

noncomputable def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_time :
  let length := 100
  let speed := 126
  abs (time_to_cross_pole length speed - 2.8571) < 0.0001 :=
by
  let length := 100
  let speed := 126
  have h1 : abs (time_to_cross_pole length speed - 2.8571) < 0.0001
  sorry
  exact h1

end train_cross_time_l118_118339


namespace find_number_l118_118209

def divisor : ℕ := 22
def quotient : ℕ := 12
def remainder : ℕ := 1
def number : ℕ := (divisor * quotient) + remainder

theorem find_number : number = 265 := by
  sorry

end find_number_l118_118209


namespace sequence_general_term_l118_118733

theorem sequence_general_term (n : ℕ) : 
  (∀ (a : ℕ → ℚ), (a 1 = 1) ∧ (a 2 = 2 / 3) ∧ (a 3 = 3 / 7) ∧ (a 4 = 4 / 15) ∧ (a 5 = 5 / 31) → a n = n / (2^n - 1)) :=
by
  sorry

end sequence_general_term_l118_118733


namespace sum_of_first_15_terms_l118_118898

theorem sum_of_first_15_terms (a : ℕ → ℝ) (r : ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * r)
    (h1 : a 1 + a 2 + a 3 = 1)
    (h2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 +
   a 10 + a 11 + a 12 + a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_first_15_terms_l118_118898


namespace inverse_of_f_inverse_of_f_inv_l118_118642

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + 1

noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

theorem inverse_of_f (x : ℝ) (hx : x > 1) : f_inv (f x) = x :=
by
  sorry

theorem inverse_of_f_inv (x : ℝ) (hx : x > 1) : f (f_inv x) = x :=
by
  sorry

end inverse_of_f_inverse_of_f_inv_l118_118642


namespace arc_length_of_sector_l118_118416

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h : r = Real.pi ∧ θ = 120) : 
  r * θ / 180 * Real.pi = 2 * Real.pi * Real.pi / 3 :=
by
  sorry

end arc_length_of_sector_l118_118416


namespace length_of_second_platform_l118_118328

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform_l118_118328


namespace gcd_1729_1309_eq_7_l118_118320

theorem gcd_1729_1309_eq_7 : Nat.gcd 1729 1309 = 7 := by
  sorry

end gcd_1729_1309_eq_7_l118_118320


namespace range_of_t_l118_118761

theorem range_of_t (t : ℝ) (h : ∃ x : ℝ, x ∈ Set.Iic t ∧ (x^2 - 4*x + t ≤ 0)) : 0 ≤ t ∧ t ≤ 4 :=
sorry

end range_of_t_l118_118761


namespace find_x_values_l118_118912

theorem find_x_values (
  x : ℝ
) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ 2) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ 
  (x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)) :=
by
  sorry

end find_x_values_l118_118912


namespace unique_k_n_m_solution_l118_118448

-- Problem statement
theorem unique_k_n_m_solution :
  ∃ (k : ℕ) (n : ℕ) (m : ℕ), k = 1 ∧ n = 2 ∧ m = 3 ∧ 3^k + 5^k = n^m ∧
  ∀ (k₀ : ℕ) (n₀ : ℕ) (m₀ : ℕ), (3^k₀ + 5^k₀ = n₀^m₀ ∧ m₀ ≥ 2) → (k₀ = 1 ∧ n₀ = 2 ∧ m₀ = 3) :=
by
  sorry

end unique_k_n_m_solution_l118_118448


namespace sin_product_ge_one_l118_118107

theorem sin_product_ge_one (x : ℝ) (n : ℤ) :
  (∀ α, |Real.sin α| ≤ 1) →
  ∀ x,
  (Real.sin x) * (Real.sin (1755 * x)) * (Real.sin (2011 * x)) ≥ 1 ↔
  ∃ n : ℤ, x = π / 2 + 2 * π * n := by {
    sorry
}

end sin_product_ge_one_l118_118107


namespace number_of_boys_in_second_class_l118_118661

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end number_of_boys_in_second_class_l118_118661


namespace triangle_third_side_l118_118087

theorem triangle_third_side (AB AC AD : ℝ) (hAB : AB = 25) (hAC : AC = 30) (hAD : AD = 24) :
  ∃ BC : ℝ, (BC = 25 ∨ BC = 11) :=
by
  sorry

end triangle_third_side_l118_118087


namespace min_value_expression_l118_118815

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l118_118815


namespace simplify_polynomial_l118_118477

theorem simplify_polynomial (x : ℝ) :
  3 + 5 * x - 7 * x^2 - 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 = 9 - x - x^2 := 
  by {
  -- placeholder for the proof
  sorry
}

end simplify_polynomial_l118_118477


namespace nests_count_l118_118781

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l118_118781


namespace proof_system_l118_118977

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  6 * x - 2 * y = 1 ∧ 2 * x + y = 2

-- Define the solution to the system of equations
def solution_equations (x y : ℝ) : Prop :=
  x = 0.5 ∧ y = 1

-- Define the system of inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1

-- Define the solution set for the system of inequalities
def solution_inequalities (x : ℝ) : Prop :=
  2 < x ∧ x < 5

-- The final theorem to be proved
theorem proof_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution_equations x y ∧ system_of_inequalities x ∧ solution_inequalities x :=
by
  sorry

end proof_system_l118_118977


namespace simplify_fraction_l118_118538

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l118_118538


namespace rate_of_interest_is_4_l118_118093

theorem rate_of_interest_is_4 (R : ℝ) : 
  ∀ P : ℝ, ∀ T : ℝ, P = 3000 → T = 5 → (P * R * T / 100 = P - 2400) → R = 4 :=
by
  sorry

end rate_of_interest_is_4_l118_118093


namespace MutualExclusivity_Of_A_C_l118_118194

-- Definitions of events using conditions from a)
def EventA (products : List Bool) : Prop :=
  products.all (λ p => p = true)

def EventB (products : List Bool) : Prop :=
  products.all (λ p => p = false)

def EventC (products : List Bool) : Prop :=
  products.any (λ p => p = false)

-- The main theorem using correct answer from b)
theorem MutualExclusivity_Of_A_C (products : List Bool) :
  EventA products → ¬ EventC products :=
by
  sorry

end MutualExclusivity_Of_A_C_l118_118194


namespace weight_of_each_piece_l118_118647

theorem weight_of_each_piece 
  (x : ℝ)
  (h : 2 * x + 0.08 = 0.75) : 
  x = 0.335 :=
by
  sorry

end weight_of_each_piece_l118_118647


namespace opposite_of_neg_six_is_six_l118_118501

theorem opposite_of_neg_six_is_six : 
  ∃ (x : ℝ), (-6 + x = 0) ∧ x = 6 := by
  sorry

end opposite_of_neg_six_is_six_l118_118501


namespace initial_ratio_of_liquids_l118_118688

theorem initial_ratio_of_liquids (p q : ℕ) (h1 : p + q = 40) (h2 : p / (q + 15) = 5 / 6) : p / q = 5 / 3 :=
by
  sorry

end initial_ratio_of_liquids_l118_118688


namespace domain_of_f_l118_118908

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f :
  { x : ℝ | f x ≠ 0 } = { x : ℝ | x ≠ -5 }
:= sorry

end domain_of_f_l118_118908


namespace MrFletcher_paid_l118_118892

noncomputable def total_payment (hours_day1 hours_day2 hours_day3 rate_per_hour men : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := total_hours * men
  total_man_hours * rate_per_hour

theorem MrFletcher_paid
  (hours_day1 hours_day2 hours_day3 : ℕ)
  (rate_per_hour men : ℕ)
  (h1 : hours_day1 = 10)
  (h2 : hours_day2 = 8)
  (h3 : hours_day3 = 15)
  (h4 : rate_per_hour = 10)
  (h5 : men = 2) :
  total_payment hours_day1 hours_day2 hours_day3 rate_per_hour men = 660 := 
by {
  -- skipped proof details
  sorry
}

end MrFletcher_paid_l118_118892


namespace number_of_valid_numbers_l118_118053

-- Define a function that checks if a number is composed of digits from the set {1, 2, 3}
def composed_of_123 (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 1 ∨ d = 2 ∨ d = 3

-- Define a predicate for a number being less than 200,000
def less_than_200000 (n : ℕ) : Prop := n < 200000

-- Define a predicate for a number being divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main theorem statement
theorem number_of_valid_numbers : ∃ (count : ℕ), count = 202 ∧ 
  (∀ (n : ℕ), less_than_200000 n → composed_of_123 n → divisible_by_3 n → n < count) :=
sorry

end number_of_valid_numbers_l118_118053


namespace quarters_dimes_equivalence_l118_118126

theorem quarters_dimes_equivalence (m : ℕ) (h : 25 * 30 + 10 * 20 = 25 * 15 + 10 * m) : m = 58 :=
by
  sorry

end quarters_dimes_equivalence_l118_118126


namespace ratio_of_adults_to_children_l118_118604

-- Definitions based on conditions
def adult_ticket_price : ℝ := 5.50
def child_ticket_price : ℝ := 2.50
def total_receipts : ℝ := 1026
def number_of_adults : ℝ := 152

-- Main theorem to prove ratio of adults to children is 2:1
theorem ratio_of_adults_to_children : 
  ∃ (number_of_children : ℝ), adult_ticket_price * number_of_adults + child_ticket_price * number_of_children = total_receipts ∧ 
  number_of_adults / number_of_children = 2 :=
by
  sorry

end ratio_of_adults_to_children_l118_118604


namespace task_assignments_count_l118_118737

theorem task_assignments_count (S : Finset (Fin 5)) :
  ∃ (assignments : Fin 5 → Fin 3),  
    (∀ t, assignments t ≠ t) ∧ 
    (∀ v, ∃ t, assignments t = v) ∧ 
    (∀ t, (t = 4 → assignments t = 1)) ∧ 
    S.card = 60 :=
by sorry

end task_assignments_count_l118_118737


namespace num_letters_with_line_no_dot_l118_118415

theorem num_letters_with_line_no_dot :
  ∀ (total_letters with_dot_and_line : ℕ) (with_dot_only with_line_only : ℕ),
    (total_letters = 60) →
    (with_dot_and_line = 20) →
    (with_dot_only = 4) →
    (total_letters = with_dot_and_line + with_dot_only + with_line_only) →
    with_line_only = 36 :=
by
  intros total_letters with_dot_and_line with_dot_only with_line_only
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end num_letters_with_line_no_dot_l118_118415


namespace min_expression_l118_118645

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l118_118645


namespace abs_x_plus_2_l118_118777

theorem abs_x_plus_2 (x : ℤ) (h : x = -3) : |x + 2| = 1 :=
by sorry

end abs_x_plus_2_l118_118777


namespace lilly_can_buy_flowers_l118_118981

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l118_118981


namespace necessary_not_sufficient_l118_118740

theorem necessary_not_sufficient (a b c : ℝ) : (a < b) → (ac^2 < b * c^2) ∧ ∀a b c : ℝ, (ac^2 < b * c^2) → (a < b) :=
sorry

end necessary_not_sufficient_l118_118740


namespace angle_in_third_quadrant_l118_118064

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) :
  (2 * ↑k * Real.pi + Real.pi < α ∧ α < 2 * ↑k * Real.pi + 3 * Real.pi / 2) →
  (∃ (m : ℤ), (0 < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < Real.pi ∨
                π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 3 * Real.pi / 2 ∨ 
                -π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 0)) :=
by
  sorry

end angle_in_third_quadrant_l118_118064


namespace original_number_l118_118341

theorem original_number (N m a b c : ℕ) (hN : N = 3306) 
  (h_eq : 3306 + m = 222 * (a + b + c)) 
  (hm_digits : m = 100 * a + 10 * b + c) 
  (h1 : a + b + c = 15) 
  (h2 : ∃ (a b c : ℕ), a + b + c = 15 ∧ 100 * a + 10 * b + c = 78): 
  100 * a + 10 * b + c = 753 := 
by sorry

end original_number_l118_118341


namespace minimum_value_expression_l118_118628

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end minimum_value_expression_l118_118628


namespace sum_of_distinct_product_GH_l118_118670

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_single_digit (d : ℕ) : Prop :=
  d < 10

theorem sum_of_distinct_product_GH : 
  ∀ (G H : ℕ), 
    is_single_digit G ∧ is_single_digit H ∧ 
    divisible_by_45 (8620000307 + 10000000 * G + H) → 
    (if H = 5 then GH = 6 else if H = 0 then GH = 0 else GH = 0) := 
  sorry

-- Note: This is a simplified representation; tailored more complex conditions and steps may be encapsulated in separate definitions and theorems as needed.

end sum_of_distinct_product_GH_l118_118670


namespace cucumbers_count_l118_118765

theorem cucumbers_count:
  ∀ (C T : ℕ), C + T = 420 ∧ T = 4 * C → C = 84 :=
by
  intros C T h
  sorry

end cucumbers_count_l118_118765


namespace irreducible_fraction_l118_118795

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l118_118795


namespace books_sold_l118_118153

theorem books_sold (original_books : ℕ) (remaining_books : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 51) 
  (h2 : remaining_books = 6) 
  (h3 : sold_books = original_books - remaining_books) : 
  sold_books = 45 :=
by 
  sorry

end books_sold_l118_118153


namespace perfect_square_a_i_l118_118845

theorem perfect_square_a_i (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n + 2) = 18 * a (n + 1) - a n) :
  ∀ i, ∃ k, 5 * (a i) ^ 2 - 1 = k ^ 2 :=
by
  -- The proof is missing the skipped definitions from the problem and solution context
  sorry

end perfect_square_a_i_l118_118845


namespace find_k_hyperbola_l118_118407

-- Define the given conditions
variables (k : ℝ)
def condition1 : Prop := k < 0
def condition2 : Prop := 2 * k^2 + k - 2 = -1

-- State the proof goal
theorem find_k_hyperbola (h1 : condition1 k) (h2 : condition2 k) : k = -1 :=
by
  sorry

end find_k_hyperbola_l118_118407


namespace calc_j_inverse_l118_118106

noncomputable def i : ℂ := Complex.I  -- Equivalent to i^2 = -1 definition of complex imaginary unit
noncomputable def j : ℂ := i + 1      -- Definition of j

theorem calc_j_inverse :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by 
  -- The statement here only needs to declare the equivalence, 
  -- without needing the proof
  sorry

end calc_j_inverse_l118_118106


namespace cars_with_neither_features_l118_118244

-- Define the given conditions
def total_cars : ℕ := 65
def cars_with_power_steering : ℕ := 45
def cars_with_power_windows : ℕ := 25
def cars_with_both_features : ℕ := 17

-- Define the statement to be proved
theorem cars_with_neither_features : total_cars - (cars_with_power_steering + cars_with_power_windows - cars_with_both_features) = 12 :=
by
  sorry

end cars_with_neither_features_l118_118244


namespace correct_operation_l118_118956

theorem correct_operation (x : ℝ) : (2 * x ^ 3) ^ 2 = 4 * x ^ 6 := 
  sorry

end correct_operation_l118_118956


namespace total_books_l118_118822

-- Conditions
def TimsBooks : Nat := 44
def SamsBooks : Nat := 52
def AlexsBooks : Nat := 65
def KatiesBooks : Nat := 37

-- Theorem Statement
theorem total_books :
  TimsBooks + SamsBooks + AlexsBooks + KatiesBooks = 198 :=
by
  sorry

end total_books_l118_118822


namespace water_balloon_packs_l118_118270

theorem water_balloon_packs (P : ℕ) : 
  (6 * P + 12 = 30) → P = 3 := by
  sorry

end water_balloon_packs_l118_118270


namespace decorative_object_height_l118_118199

def diameter_fountain := 20 -- meters
def radius_fountain := diameter_fountain / 2 -- meters

def max_height := 8 -- meters
def distance_to_max_height := 2 -- meters

-- The initial height of the water jets at the decorative object
def initial_height := 7.5 -- meters

theorem decorative_object_height :
  initial_height = 7.5 :=
  sorry

end decorative_object_height_l118_118199


namespace eight_hash_four_eq_ten_l118_118983

def operation (a b : ℚ) : ℚ := a + a / b

theorem eight_hash_four_eq_ten : operation 8 4 = 10 :=
by
  sorry

end eight_hash_four_eq_ten_l118_118983


namespace second_quadrant_condition_l118_118800

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ -270 < α ∧ α < -180

theorem second_quadrant_condition (α : ℝ) : 
  (is_obtuse α → is_in_second_quadrant α) ∧ ¬(is_in_second_quadrant α → is_obtuse α) := 
by
  sorry

end second_quadrant_condition_l118_118800


namespace minimum_g_value_l118_118122

noncomputable def g (x : ℝ) := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem minimum_g_value :
  ∀ x ≥ (1 : ℝ), g x = (47 / 16) := sorry

end minimum_g_value_l118_118122


namespace singer_arrangements_l118_118031

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l118_118031


namespace work_completion_time_equal_l118_118331

/-- Define the individual work rates of a, b, c, and d --/
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 6
def work_rate_c : ℚ := 1 / 12
def work_rate_d : ℚ := 1 / 10

/-- Define the combined work rate when they work together --/
def combined_work_rate : ℚ := work_rate_a + work_rate_b + work_rate_c + work_rate_d

/-- Define total work as one unit divided by the combined work rate --/
def total_days_to_complete : ℚ := 1 / combined_work_rate

/-- Main theorem to prove: When a, b, c, and d work together, they complete the work in 120/47 days --/
theorem work_completion_time_equal : total_days_to_complete = 120 / 47 :=
by
  sorry

end work_completion_time_equal_l118_118331


namespace intersection_A_B_l118_118513

def A (x : ℝ) : Prop := ∃ y, y = Real.log (-x^2 - 2*x + 8) ∧ -x^2 - 2*x + 8 > 0
def B (x : ℝ) : Prop := Real.log x / Real.log 2 < 1 ∧ x > 0

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l118_118513


namespace problem_1_problem_2_l118_118834

-- Define the sets M and N as conditions and include a > 0 condition.
def M (a : ℝ) : Set ℝ := {x : ℝ | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x : ℝ | 4 * x ^ 2 - 4 * x - 3 < 0}

-- Problem 1: Prove that a = 2 given the set conditions.
theorem problem_1 (a : ℝ) (h_pos : a > 0) :
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3 / 2} → a = 2 :=
sorry

-- Problem 2: Prove the range of a is 0 < a ≤ 1 / 2 given the set conditions.
theorem problem_2 (a : ℝ) (h_pos : a > 0) :
  N ∪ (compl (M a)) = Set.univ → 0 < a ∧ a ≤ 1 / 2 :=
sorry

end problem_1_problem_2_l118_118834


namespace gold_coins_count_l118_118708

theorem gold_coins_count (G : ℕ) 
  (h1 : 50 * G + 125 + 30 = 305) :
  G = 3 := 
by
  sorry

end gold_coins_count_l118_118708


namespace bus_total_distance_l118_118625

theorem bus_total_distance
  (distance40 : ℝ)
  (distance60 : ℝ)
  (speed40 : ℝ)
  (speed60 : ℝ)
  (total_time : ℝ)
  (distance40_eq : distance40 = 100)
  (speed40_eq : speed40 = 40)
  (speed60_eq : speed60 = 60)
  (total_time_eq : total_time = 5)
  (time40 : ℝ)
  (time40_eq : time40 = distance40 / speed40)
  (time_equation : time40 + distance60 / speed60 = total_time) :
  distance40 + distance60 = 250 := sorry

end bus_total_distance_l118_118625


namespace jade_statue_ratio_l118_118715

/-!
Nancy carves statues out of jade. A giraffe statue takes 120 grams of jade and sells for $150.
An elephant statue sells for $350. Nancy has 1920 grams of jade, and the revenue from selling all
elephant statues is $400 more than selling all giraffe statues.
Prove that the ratio of the amount of jade used for an elephant statue to the amount used for a
giraffe statue is 2.
-/

theorem jade_statue_ratio
  (g_grams : ℕ := 120) -- grams of jade for a giraffe statue
  (g_price : ℕ := 150) -- price of a giraffe statue
  (e_price : ℕ := 350) -- price of an elephant statue
  (total_jade : ℕ := 1920) -- total grams of jade Nancy has
  (additional_revenue : ℕ := 400) -- additional revenue from elephant statues
  (r : ℕ) -- ratio of jade usage of elephant to giraffe statue
  (h : total_jade / g_grams * g_price + additional_revenue = (total_jade / (g_grams * r)) * e_price) :
  r = 2 :=
sorry

end jade_statue_ratio_l118_118715


namespace mark_hourly_wage_before_raise_40_l118_118488

-- Mark's hourly wage before the raise
def hourly_wage_before_raise (x : ℝ) : Prop :=
  let weekly_hours := 40
  let raise_percentage := 0.05
  let new_hourly_wage := x * (1 + raise_percentage)
  let new_weekly_earnings := weekly_hours * new_hourly_wage
  let old_bills := 600
  let personal_trainer := 100
  let new_expenses := old_bills + personal_trainer
  let leftover_income := 980
  new_weekly_earnings = new_expenses + leftover_income

-- Proving that Mark's hourly wage before the raise was 40 dollars
theorem mark_hourly_wage_before_raise_40 : hourly_wage_before_raise 40 :=
by
  -- Proof goes here
  sorry

end mark_hourly_wage_before_raise_40_l118_118488


namespace contrapositive_statement_l118_118333

theorem contrapositive_statement (x : ℝ) : (x ≤ -3 → x < 0) → (x ≥ 0 → x > -3) := 
by
  sorry

end contrapositive_statement_l118_118333


namespace rhombus_region_area_l118_118060

noncomputable def region_area (s : ℝ) (angleB : ℝ) : ℝ :=
  let h := (s / 2) * (Real.sin (angleB / 2))
  let area_triangle := (1 / 2) * (s / 2) * h
  3 * area_triangle

theorem rhombus_region_area : region_area 3 150 = 0.87345 := by
    sorry

end rhombus_region_area_l118_118060


namespace initial_points_l118_118261

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l118_118261


namespace number_of_children_l118_118345

-- Definitions of the conditions
def crayons_per_child : ℕ := 8
def total_crayons : ℕ := 56

-- Statement of the problem
theorem number_of_children : total_crayons / crayons_per_child = 7 := by
  sorry

end number_of_children_l118_118345


namespace expression_equality_l118_118543

theorem expression_equality : 
  (∀ (x : ℝ) (a k n : ℝ), (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n → a = 6 ∧ k = -5 ∧ n = -6) → 
  ∀ (a k n : ℝ), a = 6 → k = -5 → n = -6 → a - n + k = 7 :=
by
  intro h
  intros a k n ha hk hn
  rw [ha, hk, hn]
  norm_num

end expression_equality_l118_118543


namespace abs_fraction_eq_sqrt_three_over_two_l118_118152

variable (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b)

theorem abs_fraction_eq_sqrt_three_over_two (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b) : 
  |(a + b) / (a - b)| = Real.sqrt (3 / 2) := by
  sorry

end abs_fraction_eq_sqrt_three_over_two_l118_118152


namespace original_cost_price_of_car_l118_118614

theorem original_cost_price_of_car
    (S_m S_f C : ℝ)
    (h1 : S_m = 0.86 * C)
    (h2 : S_f = 54000)
    (h3 : S_f = 1.20 * S_m) :
    C = 52325.58 :=
by
    sorry

end original_cost_price_of_car_l118_118614


namespace total_fencing_cost_l118_118384

-- Definitions based on the conditions
def cost_per_side : ℕ := 69
def number_of_sides : ℕ := 4

-- The proof problem statement
theorem total_fencing_cost : number_of_sides * cost_per_side = 276 := by
  sorry

end total_fencing_cost_l118_118384


namespace bond_selling_price_l118_118616

def bond_face_value : ℝ := 5000
def bond_interest_rate : ℝ := 0.06
def interest_approx : ℝ := bond_face_value * bond_interest_rate
def selling_price_interest_rate : ℝ := 0.065
def approximate_selling_price : ℝ := 4615.38

theorem bond_selling_price :
  interest_approx = selling_price_interest_rate * approximate_selling_price :=
sorry

end bond_selling_price_l118_118616


namespace adam_first_year_students_l118_118720

theorem adam_first_year_students (X : ℕ) 
  (remaining_years_students : ℕ := 9 * 50)
  (total_students : ℕ := 490) 
  (total_years_students : X + remaining_years_students = total_students) : X = 40 :=
by { sorry }

end adam_first_year_students_l118_118720


namespace carrot_broccoli_ratio_l118_118539

variables (total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings : ℕ)

-- Define the conditions
def is_condition_satisfied :=
  total_earnings = 380 ∧
  broccoli_earnings = 57 ∧
  cauliflower_earnings = 136 ∧
  spinach_earnings = (carrot_earnings / 2 + 16)

-- Define the proof problem that checks the ratio
theorem carrot_broccoli_ratio (h : is_condition_satisfied total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings) :
  ((carrot_earnings + ((carrot_earnings / 2) + 16)) + broccoli_earnings + cauliflower_earnings = total_earnings) →
  (carrot_earnings / broccoli_earnings = 2) :=
sorry

end carrot_broccoli_ratio_l118_118539


namespace distance_per_trip_l118_118915

--  Define the conditions as assumptions
variables (total_distance : ℝ) (num_trips : ℝ)
axiom h_total_distance : total_distance = 120
axiom h_num_trips : num_trips = 4

-- Define the question converted into a statement to be proven
theorem distance_per_trip : total_distance / num_trips = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end distance_per_trip_l118_118915


namespace arithmetic_sequence_expression_l118_118435

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

theorem arithmetic_sequence_expression
  (h_arith_seq : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -3) :
  ∀ n : ℕ, a n = -2 * n + 3 :=
  sorry

end arithmetic_sequence_expression_l118_118435


namespace min_value_x_3y_min_value_x_3y_iff_l118_118124

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y ≥ 25 :=
sorry

theorem min_value_x_3y_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y = 25 ↔ x = 10 ∧ y = 5 :=
sorry

end min_value_x_3y_min_value_x_3y_iff_l118_118124


namespace p_sufficient_not_necessary_for_q_l118_118905

-- Define the conditions p and q
def p (x : ℝ) := x^2 < 5 * x - 6
def q (x : ℝ) := |x + 1| ≤ 4

-- The goal to prove
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬ (∀ x, q x → p x) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l118_118905


namespace cost_of_ice_cream_l118_118718

/-- Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups. 
    The cost of each chapati is Rs. 6, that of each plate of rice is Rs. 45, and that of mixed 
    vegetable is Rs. 70. Alok paid the cashier Rs. 931. Prove the cost of each ice-cream cup is Rs. 20. -/
theorem cost_of_ice_cream (n_chapatis n_rice n_vegetable n_ice_cream : ℕ) 
    (cost_chapati cost_rice cost_vegetable total_paid : ℕ)
    (h_chapatis : n_chapatis = 16) 
    (h_rice : n_rice = 5)
    (h_vegetable : n_vegetable = 7)
    (h_ice_cream : n_ice_cream = 6)
    (h_cost_chapati : cost_chapati = 6)
    (h_cost_rice : cost_rice = 45)
    (h_cost_vegetable : cost_vegetable = 70)
    (h_total_paid : total_paid = 931) :
    (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetable * cost_vegetable)) / n_ice_cream = 20 := 
by
  sorry

end cost_of_ice_cream_l118_118718


namespace percentage_paid_X_vs_Y_l118_118281

theorem percentage_paid_X_vs_Y (X Y : ℝ) (h1 : X + Y = 528) (h2 : Y = 240) :
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_X_vs_Y_l118_118281


namespace bill_profit_difference_l118_118872

theorem bill_profit_difference 
  (SP : ℝ) 
  (hSP : SP = 1.10 * (SP / 1.10)) 
  (hSP_val : SP = 989.9999999999992) 
  (NP : ℝ) 
  (hNP : NP = 0.90 * (SP / 1.10)) 
  (NSP : ℝ) 
  (hNSP : NSP = 1.30 * NP) 
  : NSP - SP = 63.0000000000008 := 
by 
  sorry

end bill_profit_difference_l118_118872


namespace greatest_integer_a_for_domain_of_expression_l118_118973

theorem greatest_integer_a_for_domain_of_expression :
  ∃ a : ℤ, (a^2 < 60 ∧ (∀ b : ℤ, b^2 < 60 → b ≤ a)) :=
  sorry

end greatest_integer_a_for_domain_of_expression_l118_118973


namespace kim_pairs_of_shoes_l118_118164

theorem kim_pairs_of_shoes : ∃ n : ℕ, 2 * n + 1 = 14 ∧ (1 : ℚ) / (2 * n - 1) = (0.07692307692307693 : ℚ) :=
by
  sorry

end kim_pairs_of_shoes_l118_118164


namespace last_four_digits_pow_product_is_5856_l118_118287

noncomputable def product : ℕ := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349

theorem last_four_digits_pow_product_is_5856 :
  (product % 10000) ^ 4 % 10000 = 5856 := by
  sorry

end last_four_digits_pow_product_is_5856_l118_118287


namespace combined_profit_percentage_correct_l118_118575

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l118_118575


namespace greatest_consecutive_integers_sum_36_l118_118363

theorem greatest_consecutive_integers_sum_36 : ∀ (x : ℤ), (x + (x + 1) + (x + 2) = 36) → (x + 2 = 13) :=
by
  sorry

end greatest_consecutive_integers_sum_36_l118_118363


namespace distance_to_post_office_l118_118963

variable (D : ℚ)
variable (rate_to_post : ℚ := 25)
variable (rate_back : ℚ := 4)
variable (total_time : ℚ := 5 + 48 / 60)

theorem distance_to_post_office : (D / rate_to_post + D / rate_back = total_time) → D = 20 := by
  sorry

end distance_to_post_office_l118_118963


namespace angle_A_sides_b_c_l118_118324

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

end angle_A_sides_b_c_l118_118324


namespace largest_divisor_of_expression_l118_118805

theorem largest_divisor_of_expression (x : ℤ) (h_even : x % 2 = 0) :
  ∃ k, (∀ x, x % 2 = 0 → k ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) ∧ 
       (∀ m, (∀ x, x % 2 = 0 → m ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) → m ≤ k) ∧ 
       k = 32 :=
sorry

end largest_divisor_of_expression_l118_118805


namespace general_term_arithmetic_sequence_l118_118753

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n, a n = 2 * n - 1 := 
by
  intros h1 h2 n
  sorry

end general_term_arithmetic_sequence_l118_118753


namespace cover_rectangle_with_polyomino_l118_118142

-- Defining the conditions under which the m x n rectangle can be covered by the given polyomino
theorem cover_rectangle_with_polyomino (m n : ℕ) :
  (6 ∣ (m * n)) →
  (m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) →
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) →
  ((3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ (m * n))) :=
sorry

end cover_rectangle_with_polyomino_l118_118142


namespace probability_zhang_watches_entire_news_l118_118440

noncomputable def broadcast_time_start := 12 * 60 -- 12:00 in minutes
noncomputable def broadcast_time_end := 12 * 60 + 30 -- 12:30 in minutes
noncomputable def news_report_duration := 5 -- 5 minutes
noncomputable def zhang_on_tv_time := 12 * 60 + 20 -- 12:20 in minutes
noncomputable def favorable_time_start := zhang_on_tv_time
noncomputable def favorable_time_end := zhang_on_tv_time + news_report_duration -- 12:20 to 12:25

theorem probability_zhang_watches_entire_news : 
  let total_broadcast_time := broadcast_time_end - broadcast_time_start
  let favorable_time_span := favorable_time_end - favorable_time_start
  favorable_time_span / total_broadcast_time = 1 / 6 :=
by
  sorry

end probability_zhang_watches_entire_news_l118_118440


namespace smallest_magnitude_z_theorem_l118_118129

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem smallest_magnitude_z_theorem : 
  ∃ z : ℂ, (Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) ∧
  smallest_magnitude_z z = 36 / Real.sqrt 97 := 
sorry

end smallest_magnitude_z_theorem_l118_118129


namespace inclination_angle_of_line_l118_118804

-- Definitions drawn from the condition.
def line_equation (x y : ℝ) := x - y + 1 = 0

-- The statement of the theorem (equivalent proof problem).
theorem inclination_angle_of_line : ∀ x y : ℝ, line_equation x y → θ = π / 4 :=
sorry

end inclination_angle_of_line_l118_118804


namespace sqrt_inequality_l118_118428

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l118_118428


namespace maria_total_earnings_l118_118803

-- Definitions of the conditions
def day1_tulips := 30
def day1_roses := 20
def day2_tulips := 2 * day1_tulips
def day2_roses := 2 * day1_roses
def day3_tulips := day2_tulips / 10
def day3_roses := 16
def tulip_price := 2
def rose_price := 3

-- Definition of the total earnings calculation
noncomputable def total_earnings : ℤ :=
  let total_tulips := day1_tulips + day2_tulips + day3_tulips
  let total_roses := day1_roses + day2_roses + day3_roses
  (total_tulips * tulip_price) + (total_roses * rose_price)

-- The proof statement
theorem maria_total_earnings : total_earnings = 420 := by
  sorry

end maria_total_earnings_l118_118803


namespace spherical_to_rectangular_correct_l118_118163

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l118_118163


namespace frequency_of_middle_rectangle_l118_118249

theorem frequency_of_middle_rectangle
    (n : ℕ)
    (A : ℕ)
    (h1 : A + (n - 1) * A = 160) :
    A = 32 :=
by
  sorry

end frequency_of_middle_rectangle_l118_118249


namespace acme_vs_beta_l118_118465

theorem acme_vs_beta (x : ℕ) :
  (80 + 10 * x < 20 + 15 * x) → (13 ≤ x) :=
by
  intro h
  sorry

end acme_vs_beta_l118_118465


namespace socks_cost_5_l118_118421

theorem socks_cost_5
  (jeans t_shirt socks : ℕ)
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) :
  socks = 5 :=
by
  sorry

end socks_cost_5_l118_118421


namespace max_value_of_expression_l118_118033

def real_numbers (m n : ℝ) := m > 0 ∧ n < 0 ∧ (1 / m + 1 / n = 1)

theorem max_value_of_expression (m n : ℝ) (h : real_numbers m n) : 4 * m + n ≤ 1 :=
  sorry

end max_value_of_expression_l118_118033


namespace exists_triangle_with_edges_l118_118991

variable {A B C D: Type}
variables (AB AC AD BC BD CD : ℝ)
variables (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

def x := AB * CD
def y := AC * BD
def z := AD * BC

theorem exists_triangle_with_edges :
  ∃ (x y z : ℝ), 
  ∃ (A B C D: Type),
  ∃ (AB AC AD BC BD CD : ℝ) (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D),
  x = AB * CD ∧ y = AC * BD ∧ z = AD * BC → 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end exists_triangle_with_edges_l118_118991


namespace eleven_percent_greater_than_seventy_l118_118911

theorem eleven_percent_greater_than_seventy : ∀ x : ℝ, (x = 70 * (1 + 11 / 100)) → (x = 77.7) :=
by
  intro x
  intro h
  sorry

end eleven_percent_greater_than_seventy_l118_118911


namespace take_home_pay_correct_l118_118615

noncomputable def faith_take_home_pay : Float :=
  let regular_hourly_rate := 13.50
  let regular_hours_per_day := 8
  let days_per_week := 5
  let regular_hours_per_week := regular_hours_per_day * days_per_week
  let regular_earnings_per_week := regular_hours_per_week * regular_hourly_rate

  let overtime_rate_multiplier := 1.5
  let overtime_hourly_rate := regular_hourly_rate * overtime_rate_multiplier
  let overtime_hours_per_day := 2
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week
  let overtime_earnings_per_week := overtime_hours_per_week * overtime_hourly_rate

  let total_sales := 3200.0
  let commission_rate := 0.10
  let commission := total_sales * commission_rate

  let total_earnings_before_deductions := regular_earnings_per_week + overtime_earnings_per_week + commission

  let deduction_rate := 0.25
  let amount_withheld := total_earnings_before_deductions * deduction_rate
  let amount_withheld_rounded := (amount_withheld * 100).round / 100

  let take_home_pay := total_earnings_before_deductions - amount_withheld_rounded
  take_home_pay

theorem take_home_pay_correct : faith_take_home_pay = 796.87 :=
by
  /- Proof omitted -/
  sorry

end take_home_pay_correct_l118_118615


namespace seats_on_each_bus_l118_118354

-- Define the given conditions
def totalStudents : ℕ := 45
def totalBuses : ℕ := 5

-- Define what we need to prove - 
-- that the number of seats on each bus is 9
def seatsPerBus (students : ℕ) (buses : ℕ) : ℕ := students / buses

theorem seats_on_each_bus : seatsPerBus totalStudents totalBuses = 9 := by
  -- Proof to be filled in later
  sorry

end seats_on_each_bus_l118_118354


namespace range_of_k_l118_118128

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := 
sorry

end range_of_k_l118_118128


namespace dolls_count_l118_118102

theorem dolls_count (lisa_dolls : ℕ) (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ)
  (h1 : vera_dolls = 2 * lisa_dolls)
  (h2 : sophie_dolls = 2 * vera_dolls)
  (h3 : aida_dolls = 2 * sophie_dolls)
  (hl : lisa_dolls = 20) :
  aida_dolls + sophie_dolls + vera_dolls + lisa_dolls = 300 :=
by
  sorry

end dolls_count_l118_118102


namespace gcf_3150_7350_l118_118987

theorem gcf_3150_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end gcf_3150_7350_l118_118987


namespace evaluate_expression_l118_118869

noncomputable def expression_equal : Prop :=
  let a := (11: ℝ)
  let b := (11 : ℝ)^((1 : ℝ) / 6)
  let c := (11 : ℝ)^((1 : ℝ) / 5)
  (b / c = a^(-((1 : ℝ) / 30)))

theorem evaluate_expression :
  expression_equal :=
sorry

end evaluate_expression_l118_118869


namespace roots_geometric_progression_two_complex_conjugates_l118_118698

theorem roots_geometric_progression_two_complex_conjugates (a : ℝ) :
  (∃ b k : ℝ, b ≠ 0 ∧ k ≠ 0 ∧ (k + 1/ k = 2) ∧ 
    (b * (1 + k + 1/k) = 9) ∧ (b^2 * (k + 1 + 1/k) = 27) ∧ (b^3 = -a)) →
  a = -27 :=
by sorry

end roots_geometric_progression_two_complex_conjugates_l118_118698


namespace boat_speed_in_still_water_l118_118895

theorem boat_speed_in_still_water (V_b : ℝ) : 
    (∀ (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ), 
        stream_speed = 5 ∧ 
        travel_time = 5 ∧ 
        distance = 105 →
        distance = (V_b + stream_speed) * travel_time) → 
    V_b = 16 := 
by 
    intro h
    specialize h 5 5 105 
    have h1 : 105 = (V_b + 5) * 5 := h ⟨rfl, ⟨rfl, rfl⟩⟩
    sorry

end boat_speed_in_still_water_l118_118895


namespace shorter_piece_length_l118_118372

noncomputable def total_length : ℝ := 140
noncomputable def ratio : ℝ := 2 / 5

theorem shorter_piece_length (x : ℝ) (y : ℝ) (h1 : x + y = total_length) (h2 : x = ratio * y) : x = 40 :=
by
  sorry

end shorter_piece_length_l118_118372


namespace factorize_difference_of_squares_l118_118933

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l118_118933


namespace inequality_always_holds_l118_118574

theorem inequality_always_holds (a b : ℝ) (h : a * b > 0) : (b / a + a / b) ≥ 2 :=
sorry

end inequality_always_holds_l118_118574


namespace pile_splitting_l118_118038

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l118_118038


namespace lcm_is_2310_l118_118617

def a : ℕ := 210
def b : ℕ := 605
def hcf : ℕ := 55

theorem lcm_is_2310 (lcm : ℕ) : Nat.lcm a b = 2310 :=
by 
  have h : a * b = lcm * hcf := by sorry
  sorry

end lcm_is_2310_l118_118617


namespace new_acute_angle_ACB_l118_118567

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end new_acute_angle_ACB_l118_118567


namespace repeating_decimal_to_fraction_l118_118954

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 56 / 9900) : x = 3969 / 11100 := 
sorry

end repeating_decimal_to_fraction_l118_118954


namespace dave_deleted_apps_l118_118357

theorem dave_deleted_apps : 
  ∀ (a_initial a_left a_deleted : ℕ), a_initial = 16 → a_left = 5 → a_deleted = a_initial - a_left → a_deleted = 11 :=
by
  intros a_initial a_left a_deleted h_initial h_left h_deleted
  rw [h_initial, h_left] at h_deleted
  exact h_deleted

end dave_deleted_apps_l118_118357


namespace coins_from_brother_l118_118018

-- Defining the conditions as variables
variables (piggy_bank_coins : ℕ) (father_coins : ℕ) (given_to_Laura : ℕ) (left_coins : ℕ)

-- Setting the conditions
def conditions : Prop :=
  piggy_bank_coins = 15 ∧
  father_coins = 8 ∧
  given_to_Laura = 21 ∧
  left_coins = 15

-- The main theorem statement
theorem coins_from_brother (B : ℕ) :
  conditions piggy_bank_coins father_coins given_to_Laura left_coins →
  piggy_bank_coins + B + father_coins - given_to_Laura = left_coins →
  B = 13 :=
by
  sorry

end coins_from_brother_l118_118018


namespace machines_in_first_scenario_l118_118709

theorem machines_in_first_scenario :
  ∃ M : ℕ, (∀ (units1 units2 : ℕ) (hours1 hours2 : ℕ),
    units1 = 20 ∧ hours1 = 10 ∧ units2 = 200 ∧ hours2 = 25 ∧
    (M * units1 / hours1 = 20 * units2 / hours2)) → M = 5 :=
by
  sorry

end machines_in_first_scenario_l118_118709


namespace real_solutions_count_l118_118558

theorem real_solutions_count : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), (2 : ℝ) ^ (3 * x ^ 2 - 8 * x + 4) = 1 → x = 2 ∨ x = 2 / 3 :=
by
  sorry

end real_solutions_count_l118_118558


namespace mike_picked_l118_118296

-- Define the number of pears picked by Jason, Keith, and the total number of pears picked.
def jason_picked : ℕ := 46
def keith_picked : ℕ := 47
def total_picked : ℕ := 105

-- Define the goal that we need to prove: the number of pears Mike picked.
theorem mike_picked (jason_picked keith_picked total_picked : ℕ) 
  (h1 : jason_picked = 46) 
  (h2 : keith_picked = 47) 
  (h3 : total_picked = 105) 
  : (total_picked - (jason_picked + keith_picked)) = 12 :=
by sorry

end mike_picked_l118_118296


namespace two_digit_perfect_squares_divisible_by_3_l118_118736

theorem two_digit_perfect_squares_divisible_by_3 :
  ∃! n1 n2 : ℕ, (10 ≤ n1^2 ∧ n1^2 < 100 ∧ n1^2 % 3 = 0) ∧
               (10 ≤ n2^2 ∧ n2^2 < 100 ∧ n2^2 % 3 = 0) ∧
                (n1 ≠ n2) :=
by sorry

end two_digit_perfect_squares_divisible_by_3_l118_118736


namespace coordinates_of_A_l118_118512

-- Definitions based on conditions
def origin : ℝ × ℝ := (0, 0)
def similarity_ratio : ℝ := 2
def point_A : ℝ × ℝ := (2, 3)
def point_A' (P : ℝ × ℝ) : Prop :=
  P = (similarity_ratio * point_A.1, similarity_ratio * point_A.2) ∨
  P = (-similarity_ratio * point_A.1, -similarity_ratio * point_A.2)

-- Statement of the theorem
theorem coordinates_of_A' :
  ∃ P : ℝ × ℝ, point_A' P :=
by
  use (4, 6)
  left
  sorry

end coordinates_of_A_l118_118512


namespace sandwich_cost_90_cents_l118_118154

def sandwich_cost (bread_cost ham_cost cheese_cost : ℕ) : ℕ :=
  2 * bread_cost + ham_cost + cheese_cost

theorem sandwich_cost_90_cents :
  sandwich_cost 15 25 35 = 90 :=
by
  -- Proof goes here
  sorry

end sandwich_cost_90_cents_l118_118154


namespace probability_of_5_odd_in_6_rolls_l118_118485

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l118_118485


namespace arith_seq_largest_portion_l118_118432

theorem arith_seq_largest_portion (a1 d : ℝ) (h_d_pos : d > 0) 
  (h_sum : 5 * a1 + 10 * d = 100)
  (h_ratio : (3 * a1 + 9 * d) / 7 = 2 * a1 + d) : 
  a1 + 4 * d = 115 / 3 := by
  sorry

end arith_seq_largest_portion_l118_118432


namespace final_result_is_106_l118_118223

def chosen_number : ℕ := 122
def multiplied_by_2 (x : ℕ) : ℕ := 2 * x
def subtract_138 (y : ℕ) : ℕ := y - 138

theorem final_result_is_106 : subtract_138 (multiplied_by_2 chosen_number) = 106 :=
by
  -- proof is omitted
  sorry

end final_result_is_106_l118_118223


namespace smallest_perfect_square_divisible_by_5_and_6_l118_118413

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l118_118413


namespace Johann_oranges_l118_118362

-- Define the given conditions
def initial_oranges := 60
def eaten_oranges := 10
def half_remaining_oranges := (initial_oranges - eaten_oranges) / 2
def returned_oranges := 5

-- Define the statement to prove
theorem Johann_oranges :
  initial_oranges - eaten_oranges - half_remaining_oranges + returned_oranges = 30 := by
  sorry

end Johann_oranges_l118_118362


namespace solution_of_inequality_l118_118148

theorem solution_of_inequality (a b : ℝ) (h : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (x^2 < a * x + b)) :
  b^a = 81 := 
sorry

end solution_of_inequality_l118_118148


namespace select_monkey_l118_118918

theorem select_monkey (consumption : ℕ → ℕ) (n bananas minutes : ℕ)
  (h1 : consumption 1 = 1) (h2 : consumption 2 = 2) (h3 : consumption 3 = 3)
  (h4 : consumption 4 = 4) (h5 : consumption 5 = 5) (h6 : consumption 6 = 6)
  (h_total_minutes : minutes = 18) (h_total_bananas : bananas = 18) :
  consumption 1 * minutes = bananas :=
by
  sorry

end select_monkey_l118_118918


namespace relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l118_118745

-- Let a, b, and c be the sides of a right triangle with c as the hypotenuse.
variables (a b c : ℝ) (n : ℕ)

-- Assume the triangle is a right triangle
-- and assume n is a positive integer.
axiom right_triangle : a^2 + b^2 = c^2
axiom positive_integer : n > 0 

-- The relationships we need to prove:
theorem relationship_a_plus_b_greater_c : n = 1 → a + b > c := sorry
theorem relationship_a_squared_plus_b_squared_equals_c_squared : n = 2 → a^2 + b^2 = c^2 := sorry
theorem relationship_a_n_plus_b_n_less_than_c_n : n ≥ 3 → a^n + b^n < c^n := sorry

end relationship_a_plus_b_greater_c_relationship_a_squared_plus_b_squared_equals_c_squared_relationship_a_n_plus_b_n_less_than_c_n_l118_118745


namespace star_inequalities_not_all_true_simultaneously_l118_118646

theorem star_inequalities_not_all_true_simultaneously
  (AB BC CD DE EF FG GH HK KL LA : ℝ)
  (h1 : BC > AB)
  (h2 : DE > CD)
  (h3 : FG > EF)
  (h4 : HK > GH)
  (h5 : LA > KL) :
  False :=
  sorry

end star_inequalities_not_all_true_simultaneously_l118_118646


namespace binomial_square_l118_118437

theorem binomial_square (p : ℝ) : (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 24 * x + p) → p = 16 := by
  sorry

end binomial_square_l118_118437


namespace max_perimeter_convex_quadrilateral_l118_118847

theorem max_perimeter_convex_quadrilateral :
  ∃ (AB BC AD CD AC BD : ℝ), 
    AB = 1 ∧ BC = 1 ∧
    AD ≤ 1 ∧ CD ≤ 1 ∧ AC ≤ 1 ∧ BD ≤ 1 ∧
    2 + 4 * Real.sin (Real.pi / 12) = 
      AB + BC + AD + CD :=
sorry

end max_perimeter_convex_quadrilateral_l118_118847


namespace students_total_l118_118770

theorem students_total (position_eunjung : ℕ) (following_students : ℕ) (h1 : position_eunjung = 6) (h2 : following_students = 7) : 
  position_eunjung + following_students = 13 :=
by
  sorry

end students_total_l118_118770


namespace total_points_after_3_perfect_games_l118_118125

def perfect_score := 21
def number_of_games := 3

theorem total_points_after_3_perfect_games : perfect_score * number_of_games = 63 := 
by
  sorry

end total_points_after_3_perfect_games_l118_118125


namespace product_of_numbers_l118_118997

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 85) : x * y = 18 := by
  sorry

end product_of_numbers_l118_118997


namespace fraction_of_l118_118521

theorem fraction_of (a b : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) : (a / b) = 3/5 :=
by sorry

end fraction_of_l118_118521


namespace ranking_most_economical_l118_118300

theorem ranking_most_economical (c_T c_R c_J q_T q_R q_J : ℝ)
  (hR_cost : c_R = 1.25 * c_T)
  (hR_quantity : q_R = 0.75 * q_J)
  (hJ_quantity : q_J = 2.5 * q_T)
  (hJ_cost : c_J = 1.2 * c_R) :
  ((c_J / q_J) ≤ (c_R / q_R)) ∧ ((c_R / q_R) ≤ (c_T / q_T)) :=
by {
  sorry
}

end ranking_most_economical_l118_118300


namespace statement_C_correct_l118_118588

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l118_118588


namespace find_number_l118_118955

theorem find_number:
  ∃ x: ℕ, (∃ k: ℕ, ∃ r: ℕ, 5 * (x + 3) = 8 * k + r ∧ k = 156 ∧ r = 2) ∧ x = 247 :=
by 
  sorry

end find_number_l118_118955


namespace maximize_x4y3_l118_118792

theorem maximize_x4y3 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = 40) : 
    (x, y) = (160 / 7, 120 / 7) ↔ x ^ 4 * y ^ 3 ≤ (160 / 7) ^ 4 * (120 / 7) ^ 3 := 
sorry

end maximize_x4y3_l118_118792


namespace average_runs_in_30_matches_l118_118941

theorem average_runs_in_30_matches 
  (avg1 : ℕ) (matches1 : ℕ) (avg2 : ℕ) (matches2 : ℕ) (total_matches : ℕ)
  (h1 : avg1 = 40) (h2 : matches1 = 20) (h3 : avg2 = 13) (h4 : matches2 = 10) (h5 : total_matches = 30) :
  ((avg1 * matches1 + avg2 * matches2) / total_matches) = 31 := by
  sorry

end average_runs_in_30_matches_l118_118941


namespace transform_eq_l118_118564

theorem transform_eq (m n x y : ℕ) (h1 : m + x = n + y) (h2 : x = y) : m = n :=
sorry

end transform_eq_l118_118564


namespace max_initial_number_l118_118335

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l118_118335


namespace josh_found_marbles_l118_118348

theorem josh_found_marbles :
  ∃ (F : ℕ), (F + 14 = 23) → (F = 9) :=
by
  existsi 9
  intro h
  linarith

end josh_found_marbles_l118_118348


namespace calculate_triple_transform_l118_118690

def transformation (N : ℝ) : ℝ :=
  0.4 * N + 2

theorem calculate_triple_transform :
  transformation (transformation (transformation 20)) = 4.4 :=
by
  sorry

end calculate_triple_transform_l118_118690


namespace rectangular_prism_pairs_l118_118230

def total_pairs_of_edges_in_rect_prism_different_dimensions (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : ℕ :=
66

theorem rectangular_prism_pairs (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : total_pairs_of_edges_in_rect_prism_different_dimensions length width height h1 h2 h3 = 66 := 
sorry

end rectangular_prism_pairs_l118_118230


namespace solve_x_l118_118775

theorem solve_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.84) : x = 72 := 
by
  sorry

end solve_x_l118_118775


namespace coordinates_of_F_double_prime_l118_118532

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem coordinates_of_F_double_prime :
  let F : ℝ × ℝ := (3, 3)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_x_axis F'
  F'' = (-3, -3) :=
by
  sorry

end coordinates_of_F_double_prime_l118_118532


namespace thirty_six_forty_five_nine_eighteen_l118_118836

theorem thirty_six_forty_five_nine_eighteen :
  18 * 36 + 45 * 18 - 9 * 18 = 1296 :=
by
  sorry

end thirty_six_forty_five_nine_eighteen_l118_118836


namespace coefficient_x_squared_l118_118469

theorem coefficient_x_squared (a : ℝ) (x : ℝ) (h : x = 0.5) (eqn : a * x^2 + 9 * x - 5 = 0) : a = 2 :=
by
  sorry

end coefficient_x_squared_l118_118469


namespace range_of_m_l118_118380

-- Define sets A and B
def A := {x : ℝ | x ≤ 1}
def B (m : ℝ) := {x : ℝ | x ≤ m}

-- Statement: Prove the range of m such that B ⊆ A
theorem range_of_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ (m ≤ 1) :=
by sorry

end range_of_m_l118_118380


namespace value_of_star_l118_118878

theorem value_of_star (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : (a + b) % 4 = 0) : a^2 + 2*a*b + b^2 = 64 :=
by
  sorry

end value_of_star_l118_118878


namespace problem_statement_l118_118115

theorem problem_statement (x : ℝ) (h : 2 * x^2 + 1 = 17) : 4 * x^2 + 1 = 33 :=
by sorry

end problem_statement_l118_118115


namespace difference_of_squares_is_40_l118_118766

theorem difference_of_squares_is_40 {x y : ℕ} (h1 : x + y = 20) (h2 : x * y = 99) (hx : x > y) : x^2 - y^2 = 40 :=
sorry

end difference_of_squares_is_40_l118_118766


namespace satisfy_conditions_l118_118197

variable (x : ℝ)

theorem satisfy_conditions :
  (3 * x^2 + 4 * x - 9 < 0) ∧ (x ≥ -2) ↔ (-2 ≤ x ∧ x < 1) := by
  sorry

end satisfy_conditions_l118_118197


namespace triangle_side_length_l118_118463

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h₁ : a * Real.cos B = b * Real.sin A)
  (h₂ : C = Real.pi / 6) (h₃ : c = 2) : b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l118_118463


namespace integer_value_l118_118228

theorem integer_value (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (h3 : x > 0) (h4 : y > 0) (h5 : z > 0) :
  ∃ a : ℕ, a + y + z = 26 ∧ a = 15 := by
  sorry

end integer_value_l118_118228


namespace third_offense_fraction_l118_118922

-- Define the conditions
def sentence_assault : ℕ := 3
def sentence_poisoning : ℕ := 24
def total_sentence : ℕ := 36

-- The main theorem to prove
theorem third_offense_fraction :
  (total_sentence - (sentence_assault + sentence_poisoning)) / (sentence_assault + sentence_poisoning) = 1 / 3 := by
  sorry

end third_offense_fraction_l118_118922


namespace range_of_a_l118_118019

noncomputable def odd_function_periodic_real (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ -- odd function condition
  (∀ x, f (x + 5) = f x) ∧ -- periodic function condition
  (f 1 < -1) ∧ -- given condition
  (f 4 = Real.log a / Real.log 2) -- condition using log base 2

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : odd_function_periodic_real f a) : a > 2 :=
by sorry 

end range_of_a_l118_118019


namespace earnings_difference_l118_118174

theorem earnings_difference (x y : ℕ) 
  (h1 : 3 * 6 + 4 * 5 + 5 * 4 = 58)
  (h2 : x * y = 12500) 
  (total_earnings : (3 * 6 * x * y / 100 + 4 * 5 * x * y / 100 + 5 * 4 * x * y / 100) = 7250) :
  4 * 5 * x * y / 100 - 3 * 6 * x * y / 100 = 250 := 
by 
  sorry

end earnings_difference_l118_118174


namespace triangle_B_eq_2A_range_of_a_l118_118768

theorem triangle_B_eq_2A (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = c) : B = 2 * A := 
sorry

theorem range_of_a (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : a + 2 * a * Real.cos B = 2) (h6 : 0 < (π - A - B)) (h7 : (π - A - B) < π/2) : 1 < a ∧ a < 2 := 
sorry

end triangle_B_eq_2A_range_of_a_l118_118768


namespace min_value_expr_l118_118813

theorem min_value_expr (x y : ℝ) : ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + x * y + y^2 ≥ m) ∧ m = 0 :=
by
  sorry

end min_value_expr_l118_118813


namespace prize_distribution_l118_118843

theorem prize_distribution (x y z : ℕ) (h₁ : 15000 * x + 10000 * y + 5000 * z = 1000000) (h₂ : 93 ≤ z - x) (h₃ : z - x < 96) :
  x + y + z = 147 :=
sorry

end prize_distribution_l118_118843


namespace range_m_l118_118667

def A (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ -3 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_m_l118_118667


namespace velocity_at_t1_l118_118623

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2 * t

-- Define the velocity function as the derivative of s
def velocity (t : ℝ) : ℝ := -2 * t + 2

-- Prove that the velocity at t = 1 is 0
theorem velocity_at_t1 : velocity 1 = 0 :=
by
  -- Apply the definition of velocity
    sorry

end velocity_at_t1_l118_118623


namespace sum_geometric_sequence_l118_118113

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_geometric_sequence_l118_118113


namespace eval_expression_l118_118952

theorem eval_expression : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := 
by
  sorry

end eval_expression_l118_118952


namespace time_between_four_and_five_straight_line_l118_118687

theorem time_between_four_and_five_straight_line :
  ∃ t : ℚ, t = 21 + 9/11 ∨ t = 54 + 6/11 :=
by
  sorry

end time_between_four_and_five_straight_line_l118_118687


namespace intersection_of_complements_l118_118360

theorem intersection_of_complements 
  (U : Set ℕ) (A B : Set ℕ)
  (hU : U = { x | x ≤ 5 }) 
  (hA : A = {1, 2, 3}) 
  (hB : B = {1, 4}) :
  ((U \ A) ∩ (U \ B)) = {0, 5} :=
by sorry

end intersection_of_complements_l118_118360


namespace find_y_l118_118011

theorem find_y 
  (x y : ℕ) 
  (hx : x % y = 9) 
  (hxy : (x : ℝ) / y = 96.12) : y = 75 :=
sorry

end find_y_l118_118011


namespace hydrogen_atoms_in_compound_l118_118046

theorem hydrogen_atoms_in_compound : 
  ∀ (C O H : ℕ) (molecular_weight : ℕ), 
  C = 1 → 
  O = 3 → 
  molecular_weight = 62 → 
  (12 * C + 16 * O + H = molecular_weight) → 
  H = 2 := 
by
  intros C O H molecular_weight hc ho hmw hcalc
  sorry

end hydrogen_atoms_in_compound_l118_118046


namespace inequality_generalization_l118_118219

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : n > 0) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2) (h2 : x + 4 / (x ^ 2) = (x / 2) + (x / 2) + 4 / (x ^ 2) ∧ (x / 2) + (x / 2) + 4 / (x ^ 2) ≥ 3) : 
  x + n^n / x^n ≥ n + 1 := 
sorry

end inequality_generalization_l118_118219


namespace rectangle_area_l118_118025

theorem rectangle_area (b l : ℕ) (P : ℕ) (h1 : l = 3 * b) (h2 : P = 64) (h3 : P = 2 * (l + b)) :
  l * b = 192 :=
by
  sorry

end rectangle_area_l118_118025


namespace fraction_of_track_Scottsdale_to_Forest_Grove_l118_118970

def distance_between_Scottsdale_and_Sherbourne : ℝ := 200
def round_trip_duration : ℝ := 5
def time_Harsha_to_Sherbourne : ℝ := 2

theorem fraction_of_track_Scottsdale_to_Forest_Grove :
  ∃ f : ℝ, f = 1/5 ∧
    ∀ (d : ℝ) (t : ℝ) (h : ℝ),
    d = distance_between_Scottsdale_and_Sherbourne →
    t = round_trip_duration →
    h = time_Harsha_to_Sherbourne →
    (2.5 - h) / t = f :=
sorry

end fraction_of_track_Scottsdale_to_Forest_Grove_l118_118970


namespace horner_value_at_2_l118_118649

noncomputable def f (x : ℝ) := 2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

theorem horner_value_at_2 : f 2 = 12 := sorry

end horner_value_at_2_l118_118649


namespace triangle_fraction_correct_l118_118839

def point : Type := ℤ × ℤ

def area_triangle (A B C : point) : ℚ :=
  (1 / 2 : ℚ) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℚ))

def area_grid (length width : ℚ) : ℚ :=
  length * width

noncomputable def fraction_covered (A B C : point) (grid_length grid_width : ℚ) : ℚ :=
  area_triangle A B C / area_grid grid_length grid_width

theorem triangle_fraction_correct :
  fraction_covered (-2, 3) (2, -2) (3, 5) 8 6 = 11 / 32 :=
by
  sorry

end triangle_fraction_correct_l118_118839


namespace alice_profit_l118_118692

-- Define the variables and conditions
def total_bracelets : ℕ := 52
def material_cost : ℝ := 3.00
def bracelets_given_away : ℕ := 8
def sale_price : ℝ := 0.25

-- Calculate the number of bracelets sold
def bracelets_sold : ℕ := total_bracelets - bracelets_given_away

-- Calculate the revenue from selling the bracelets
def revenue : ℝ := bracelets_sold * sale_price

-- Define the profit as revenue minus material cost
def profit : ℝ := revenue - material_cost

-- The statement to prove
theorem alice_profit : profit = 8.00 := 
by
  sorry

end alice_profit_l118_118692


namespace angle_A_measure_l118_118165

theorem angle_A_measure (A B C D E : ℝ) 
(h1 : A = 3 * B)
(h2 : A = 4 * C)
(h3 : A = 5 * D)
(h4 : A = 6 * E)
(h5 : A + B + C + D + E = 540) : 
A = 277 :=
by
  sorry

end angle_A_measure_l118_118165


namespace four_digit_solution_l118_118317

-- Definitions for the conditions.
def condition1 (u z x : ℕ) : Prop := u + z - 4 * x = 1
def condition2 (u z y : ℕ) : Prop := u + 10 * z - 2 * y = 14

-- The theorem to prove that the four-digit number xyz is either 1014, 2218, or 1932
theorem four_digit_solution (x y z u : ℕ) (h1 : condition1 u z x) (h2 : condition2 u z y) :
  (x = 1 ∧ y = 0 ∧ z = 1 ∧ u = 4) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ u = 8) ∨
  (x = 1 ∧ y = 9 ∧ z = 3 ∧ u = 2) := 
sorry

end four_digit_solution_l118_118317


namespace range_of_m_l118_118741

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : 
  (x + y ≥ m) → m ≤ 18 :=
sorry

end range_of_m_l118_118741


namespace min_value_expression_l118_118216

open Real

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_abc : a * b * c = 1 / 2) : 
    a^2 + 8 * a * b + 32 * b^2 + 16 * b * c + 8 * c^2 ≥ 18 :=
sorry

end min_value_expression_l118_118216


namespace simplify_expression_l118_118841

theorem simplify_expression :
  (18 / 17) * (13 / 24) * (68 / 39) = 1 := 
by
  sorry

end simplify_expression_l118_118841


namespace profit_percentage_l118_118926

theorem profit_percentage (CP SP : ℝ) (h : 18 * CP = 16 * SP) : 
  (SP - CP) / CP * 100 = 12.5 := by
sorry

end profit_percentage_l118_118926


namespace inscribed_regular_polygon_sides_l118_118849

theorem inscribed_regular_polygon_sides (n : ℕ) (h_central_angle : 360 / n = 72) : n = 5 :=
by
  sorry

end inscribed_regular_polygon_sides_l118_118849


namespace min_handshakes_l118_118344

theorem min_handshakes 
  (people : ℕ) 
  (handshakes_per_person : ℕ) 
  (total_people : people = 30) 
  (handshakes_rule : handshakes_per_person = 3) 
  (unique_handshakes : people * handshakes_per_person % 2 = 0) 
  (multiple_people : people > 0):
  (people * handshakes_per_person / 2) = 45 :=
by
  sorry

end min_handshakes_l118_118344


namespace option_A_option_B_option_C_option_D_l118_118595

-- Define the equation of the curve
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Prove that when k=2, the curve is a circle
theorem option_A (x y : ℝ) : curve 2 x y ↔ x^2 + y^2 = 3 :=
by
  sorry

-- Prove the necessary and sufficient condition for the curve to be an ellipse
theorem option_B (k : ℝ) : (-1 < k ∧ k < 5) ↔ ∃ x y, curve k x y ∧ (k ≠ 2) :=
by
  sorry

-- Prove the condition for the curve to be a hyperbola with foci on the y-axis
theorem option_C (k : ℝ) : k < -1 ↔ ∃ x y, curve k x y ∧ (k < -1 ∧ k < 5) :=
by
  sorry

-- Prove that there does not exist a real number k such that the curve is a parabola
theorem option_D : ¬ (∃ k x y, curve k x y ∧ ∃ a b, x = a ∧ y = b) :=
by
  sorry

end option_A_option_B_option_C_option_D_l118_118595


namespace john_average_speed_l118_118377

theorem john_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time_is_45_minutes : uphill_time = 45)
  (downhill_time_is_15_minutes : downhill_time = 15)
  (uphill_distance_is_3_km : uphill_distance = 3)
  (downhill_distance_is_3_km : downhill_distance = 3)
  : (uphill_distance + downhill_distance) / ((uphill_time + downhill_time) / 60) = 6 := 
by
  sorry

end john_average_speed_l118_118377


namespace Karen_tote_weight_l118_118117

variable (B T F : ℝ)
variable (Papers Laptop : ℝ)

theorem Karen_tote_weight (h1: T = 2 * B)
                         (h2: F = 2 * T)
                         (h3: Papers = (1 / 6) * F)
                         (h4: Laptop = T + 2)
                         (h5: F = B + Laptop + Papers):
  T = 12 := 
sorry

end Karen_tote_weight_l118_118117


namespace sum_of_squares_positive_l118_118784

theorem sum_of_squares_positive (x_1 x_2 k : ℝ) (h : x_1 ≠ x_2) 
  (hx1 : x_1^2 + 2*x_1 - k = 0) (hx2 : x_2^2 + 2*x_2 - k = 0) :
  x_1^2 + x_2^2 > 0 :=
by
  sorry

end sum_of_squares_positive_l118_118784


namespace age_ratio_l118_118182

theorem age_ratio (S : ℕ) (M : ℕ) (h1 : S = 28) (h2 : M = S + 30) : 
  ((M + 2) / (S + 2) = 2) := 
by
  sorry

end age_ratio_l118_118182


namespace find_a_l118_118868

noncomputable def lines_perpendicular (a : ℝ) (l1: ℝ × ℝ × ℝ) (l2: ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1
  let (A2, B2, C2) := l2
  (B1 ≠ 0) ∧ (B2 ≠ 0) ∧ (-A1 / B1) * (-A2 / B2) = -1

theorem find_a (a : ℝ) :
  lines_perpendicular a (a, 1, 1) (2*a, a - 3, 1) → a = 1 ∨ a = -3/2 :=
by
  sorry

end find_a_l118_118868


namespace stans_average_speed_l118_118594

/-- Given that Stan drove 420 miles in 6 hours, 480 miles in 7 hours, and 300 miles in 5 hours,
prove that his average speed for the entire trip is 1200/18 miles per hour. -/
theorem stans_average_speed :
  let total_distance := 420 + 480 + 300
  let total_time := 6 + 7 + 5
  total_distance / total_time = 1200 / 18 :=
by
  sorry

end stans_average_speed_l118_118594


namespace triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l118_118742

theorem triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero
  (A B C : ℝ) (h : A + B + C = 180): 
    (A = 60 ∨ B = 60 ∨ C = 60) ↔ (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 0) := 
by
  sorry

end triangle_angle_60_iff_sin3A_sin3B_sin3C_eq_zero_l118_118742


namespace math_problem_l118_118204

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l118_118204


namespace not_54_after_one_hour_l118_118779

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

end not_54_after_one_hour_l118_118779


namespace one_gallon_fills_one_cubic_foot_l118_118389

theorem one_gallon_fills_one_cubic_foot
  (total_water : ℕ)
  (drinking_cooking : ℕ)
  (shower_water : ℕ)
  (num_showers : ℕ)
  (pool_length : ℕ)
  (pool_width : ℕ)
  (pool_height : ℕ)
  (h_total_water : total_water = 1000)
  (h_drinking_cooking : drinking_cooking = 100)
  (h_shower_water : shower_water = 20)
  (h_num_showers : num_showers = 15)
  (h_pool_length : pool_length = 10)
  (h_pool_width : pool_width = 10)
  (h_pool_height : pool_height = 6) :
  (pool_length * pool_width * pool_height) / 
  (total_water - drinking_cooking - num_showers * shower_water) = 1 := by
  sorry

end one_gallon_fills_one_cubic_foot_l118_118389


namespace intersection_A_B_eq_l118_118664

def A : Set ℝ := { x | (x / (x - 1)) ≥ 0 }

def B : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B_eq :
  (A ∩ B) = { y : ℝ | 1 < y } :=
sorry

end intersection_A_B_eq_l118_118664


namespace shaded_area_is_20_l118_118967

-- Represents the square PQRS with the necessary labeled side lengths
noncomputable def square_side_length : ℝ := 8

-- Represents the four labeled smaller squares' positions and their side lengths
noncomputable def smaller_square_side_lengths : List ℝ := [2, 2, 2, 6]

-- The coordinates or relations to describe their overlaying positions are not needed for the proof.

-- Define the calculated areas from the solution steps
noncomputable def vertical_rectangle_area : ℝ := 6 * 2
noncomputable def horizontal_rectangle_area : ℝ := 6 * 2
noncomputable def overlap_area : ℝ := 2 * 2

-- The total shaded T-shaped region area calculation
noncomputable def total_shaded_area : ℝ := vertical_rectangle_area + horizontal_rectangle_area - overlap_area

-- Theorem statement to prove the area of the T-shaped region is 20
theorem shaded_area_is_20 : total_shaded_area = 20 :=
by
  -- Proof steps are not required as per the instruction.
  sorry

end shaded_area_is_20_l118_118967


namespace max_cars_with_ac_but_not_rs_l118_118865

namespace CarProblem

variables (total_cars : ℕ) 
          (cars_without_ac : ℕ)
          (cars_with_rs : ℕ)
          (cars_with_ac : ℕ := total_cars - cars_without_ac)
          (cars_with_ac_and_rs : ℕ)
          (cars_with_ac_but_not_rs : ℕ := cars_with_ac - cars_with_ac_and_rs)

theorem max_cars_with_ac_but_not_rs 
        (h1 : total_cars = 100)
        (h2 : cars_without_ac = 37)
        (h3 : cars_with_rs ≥ 51)
        (h4 : cars_with_ac_and_rs = min cars_with_rs cars_with_ac) :
        cars_with_ac_but_not_rs = 12 := by
    sorry

end CarProblem

end max_cars_with_ac_but_not_rs_l118_118865


namespace age_difference_l118_118464

variable (A B C : ℕ)

def age_relationship (B C : ℕ) : Prop :=
  B = 2 * C

def total_ages (A B C : ℕ) : Prop :=
  A + B + C = 72

theorem age_difference (B : ℕ) (hB : B = 28) (h1 : age_relationship B C) (h2 : total_ages A B C) :
  A - B = 2 :=
sorry

end age_difference_l118_118464


namespace negation_of_proposition_l118_118787

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → (x + 1/x) ≥ 2

-- Define the negation of the original proposition
def negation_prop : Prop := ∃ x > 0, x + 1/x < 2

-- State that the negation of the original proposition is the stated negation
theorem negation_of_proposition : (¬ ∀ x, original_prop x) ↔ negation_prop := 
by sorry

end negation_of_proposition_l118_118787


namespace original_wattage_l118_118831

theorem original_wattage (W : ℝ) (new_W : ℝ) (h1 : new_W = 1.25 * W) (h2 : new_W = 100) : W = 80 :=
by
  sorry

end original_wattage_l118_118831


namespace total_art_cost_l118_118255

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l118_118255


namespace largest_divisor_of_n_l118_118388

theorem largest_divisor_of_n (n : ℕ) (hn : 0 < n) (h : 50 ∣ n^2) : 5 ∣ n :=
sorry

end largest_divisor_of_n_l118_118388


namespace christen_potatoes_peeled_l118_118794

-- Define the initial conditions and setup
def initial_potatoes := 50
def homer_rate := 4
def christen_rate := 6
def time_homer_alone := 5
def combined_rate := homer_rate + christen_rate

-- Calculate the number of potatoes peeled by Homer alone in the first 5 minutes
def potatoes_peeled_by_homer_alone := time_homer_alone * homer_rate

-- Calculate the remaining potatoes after Homer peeled alone
def remaining_potatoes := initial_potatoes - potatoes_peeled_by_homer_alone

-- Calculate the time taken for Homer and Christen to peel the remaining potatoes together
def time_to_finish_together := remaining_potatoes / combined_rate

-- Calculate the number of potatoes peeled by Christen during the shared work period
def potatoes_peeled_by_christen := christen_rate * time_to_finish_together

-- The final theorem we need to prove
theorem christen_potatoes_peeled : potatoes_peeled_by_christen = 18 := by
  sorry

end christen_potatoes_peeled_l118_118794


namespace min_41x_2y_eq_nine_l118_118121

noncomputable def min_value_41x_2y (x y : ℝ) : ℝ :=
  41*x + 2*y

theorem min_41x_2y_eq_nine (x y : ℝ) (h : ∀ n : ℕ, 0 < n →  n*x + (1/n)*y ≥ 1) :
  min_value_41x_2y x y = 9 :=
sorry

end min_41x_2y_eq_nine_l118_118121


namespace correct_mark_l118_118893

theorem correct_mark 
  (avg_wrong : ℝ := 60)
  (wrong_mark : ℝ := 90)
  (num_students : ℕ := 30)
  (avg_correct : ℝ := 57.5) :
  (wrong_mark - (avg_wrong * num_students - avg_correct * num_students)) = 15 :=
by
  sorry

end correct_mark_l118_118893


namespace solve_x4_minus_inv_x4_l118_118634

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end solve_x4_minus_inv_x4_l118_118634


namespace intersection_sets_l118_118838

open Set

def A := {x : ℤ | abs x < 3}
def B := {x : ℤ | abs x > 1}

theorem intersection_sets :
  A ∩ B = {-2, 2} := by
  sorry

end intersection_sets_l118_118838


namespace cost_of_one_stamp_l118_118906

-- Defining the conditions
def cost_of_four_stamps := 136
def number_of_stamps := 4

-- Prove that if 4 stamps cost 136 cents, then one stamp costs 34 cents
theorem cost_of_one_stamp : cost_of_four_stamps / number_of_stamps = 34 :=
by
  sorry

end cost_of_one_stamp_l118_118906


namespace tn_range_l118_118313

noncomputable def a (n : ℕ) : ℚ :=
  (2 * n - 1) / 10

noncomputable def b (n : ℕ) : ℚ :=
  2^(n - 1)

noncomputable def c (n : ℕ) : ℚ :=
  (1 + a n) / (4 * b n)

noncomputable def T (n : ℕ) : ℚ :=
  (1 / 10) * (2 - (n + 2) / (2^n)) + (9 / 20) * (2 - 1 / (2^(n-1)))

theorem tn_range (n : ℕ) : (101 / 400 : ℚ) ≤ T n ∧ T n < (103 / 200 : ℚ) :=
sorry

end tn_range_l118_118313


namespace number_of_truthful_monkeys_l118_118460

-- Define the conditions of the problem
def num_tigers : ℕ := 100
def num_foxes : ℕ := 100
def num_monkeys : ℕ := 100
def total_groups : ℕ := 100
def animals_per_group : ℕ := 3
def yes_tiger : ℕ := 138
def yes_fox : ℕ := 188

-- Problem statement to be proved
theorem number_of_truthful_monkeys :
  ∃ m : ℕ, m = 76 ∧
  ∃ (x y z m n : ℕ),
    -- The number of monkeys mixed with tigers
    x + 2 * (74 - y) = num_monkeys ∧

    -- Given constraints
    m ∈ {n : ℕ | n ≤ x} ∧
    n ∈ {n : ℕ | n ≤ (num_foxes - x)} ∧

    -- Equation setup and derived equations
    (x - m) + (num_foxes - y) + n = yes_tiger ∧
    m + (num_tigers - x - n) + (num_tigers - z) = yes_fox ∧
    y + z = 74 ∧
    
    -- ensuring the groups are valid
    2 * (74 - y) = z :=

sorry

end number_of_truthful_monkeys_l118_118460


namespace ab_leq_one_fraction_inequality_l118_118789

-- Part 1: Prove that ab ≤ 1
theorem ab_leq_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) : a * b ≤ 1 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

-- Part 2: Prove that (1/a^3 - 1/b^3) > 3 * (1/a - 1/b) given b > a
theorem fraction_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 1/(a * b) + 3) (h4 : b > a) :
  1/(a^3) - 1/(b^3) > 3 * (1/a - 1/b) :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end ab_leq_one_fraction_inequality_l118_118789


namespace extreme_value_range_of_a_l118_118791

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (1 - a * x)

theorem extreme_value_range_of_a (a : ℝ) :
  a ∈ Set.Ioo (2 / 3 : ℝ) 2 ↔
    ∃ c ∈ Set.Ioo 0 1, ∀ x : ℝ, f a c = f a x :=
by
  sorry

end extreme_value_range_of_a_l118_118791


namespace minimum_red_chips_l118_118786

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ w / 4) (h2 : b ≤ r / 6) (h3 : w + b ≥ 75) : r ≥ 90 :=
sorry

end minimum_red_chips_l118_118786


namespace job_completion_time_l118_118005

theorem job_completion_time (x : ℤ) (hx : (4 : ℝ) / x + (2 : ℝ) / 3 = 1) : x = 12 := by
  sorry

end job_completion_time_l118_118005


namespace max_height_of_rock_l118_118402

theorem max_height_of_rock : 
    ∃ t_max : ℝ, (∀ t : ℝ, -5 * t^2 + 25 * t + 10 ≤ -5 * t_max^2 + 25 * t_max + 10) ∧ (-5 * t_max^2 + 25 * t_max + 10 = 165 / 4) := 
sorry

end max_height_of_rock_l118_118402


namespace total_pounds_of_food_l118_118602

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end total_pounds_of_food_l118_118602


namespace remainder_of_3y_l118_118302

theorem remainder_of_3y (y : ℕ) (hy : y % 9 = 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_of_3y_l118_118302


namespace remaining_minutes_proof_l118_118336

def total_series_minutes : ℕ := 360

def first_session_end : ℕ := 17 * 60 + 44  -- in minutes
def first_session_start : ℕ := 15 * 60 + 20  -- in minutes
def second_session_end : ℕ := 20 * 60 + 40  -- in minutes
def second_session_start : ℕ := 19 * 60 + 15  -- in minutes
def third_session_end : ℕ := 22 * 60 + 30  -- in minutes
def third_session_start : ℕ := 21 * 60 + 35  -- in minutes

def first_session_duration : ℕ := first_session_end - first_session_start
def second_session_duration : ℕ := second_session_end - second_session_start
def third_session_duration : ℕ := third_session_end - third_session_start

def total_watched : ℕ := first_session_duration + second_session_duration + third_session_duration

def remaining_time : ℕ := total_series_minutes - total_watched

theorem remaining_minutes_proof : remaining_time = 76 := 
by 
  sorry  -- Proof goes here

end remaining_minutes_proof_l118_118336


namespace range_of_fx_a_eq_2_range_of_a_increasing_fx_l118_118383

-- Part (1)
theorem range_of_fx_a_eq_2 (x : ℝ) (h : x ∈ Set.Icc (-2 : ℝ) (3 : ℝ)) :
  ∃ y ∈ Set.Icc (-21 / 4 : ℝ) (15 : ℝ), y = x^2 + 3 * x - 3 :=
sorry

-- Part (2)
theorem range_of_a_increasing_fx (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) → 2 * x + 2 * a - 1 ≥ 0) ↔ a ∈ Set.Ici (3 / 2 : ℝ) :=
sorry

end range_of_fx_a_eq_2_range_of_a_increasing_fx_l118_118383


namespace solve_for_y_l118_118969

theorem solve_for_y (y : ℝ) (h : (5 - 1 / y)^(1/3) = -3) : y = 1 / 32 :=
by
  sorry

end solve_for_y_l118_118969


namespace divisible_by_27_l118_118077

theorem divisible_by_27 (n : ℕ) : 27 ∣ (2^(5*n+1) + 5^(n+2)) :=
by
  sorry

end divisible_by_27_l118_118077


namespace factorize_expr_l118_118356

-- Define the variables a and b as elements of an arbitrary ring
variables {R : Type*} [CommRing R] (a b : R)

-- Prove the factorization identity
theorem factorize_expr : a^2 * b - b = b * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expr_l118_118356


namespace correct_inequality_l118_118292

variable (a b : ℝ)

theorem correct_inequality (h : a > b) : a - 3 > b - 3 :=
by
  sorry

end correct_inequality_l118_118292


namespace calculate_earths_atmosphere_mass_l118_118975

noncomputable def mass_of_earths_atmosphere (R p0 g : ℝ) : ℝ :=
  (4 * Real.pi * R^2 * p0) / g

theorem calculate_earths_atmosphere_mass (R p0 g : ℝ) (h : 0 < g) : 
  mass_of_earths_atmosphere R p0 g = 5 * 10^18 := 
sorry

end calculate_earths_atmosphere_mass_l118_118975


namespace sum_of_solutions_l118_118065

theorem sum_of_solutions (x : ℝ) :
  (∀ x, x^2 - 17 * x + 54 = 0) → 
  (∃ r s : ℝ, r ≠ s ∧ r + s = 17) :=
by
  sorry

end sum_of_solutions_l118_118065


namespace sequence_5th_term_l118_118827

theorem sequence_5th_term (a b c : ℚ) (h1 : a = 1 / 4 * (4 + b)) (h2 : b = 1 / 4 * (a + 40)) (h3 : 40 = 1 / 4 * (b + c)) : 
  c = 2236 / 15 := 
by 
  sorry

end sequence_5th_term_l118_118827


namespace no_real_roots_of_quadratic_l118_118764

theorem no_real_roots_of_quadratic (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b ≠ 0) ↔ ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry

end no_real_roots_of_quadratic_l118_118764


namespace min_k_value_l118_118548

noncomputable def f (k x : ℝ) : ℝ := k * (x^2 - x + 1) - x^4 * (1 - x)^4

theorem min_k_value : ∃ k : ℝ, (k = 1 / 192) ∧ ∀ x : ℝ, (0 ≤ x) → (x ≤ 1) → (f k x ≥ 0) :=
by
  existsi (1 / 192)
  sorry

end min_k_value_l118_118548


namespace min_possible_A_div_C_l118_118167

theorem min_possible_A_div_C (x : ℝ) (A C : ℝ) (h1 : x^2 + (1/x)^2 = A) (h2 : x + 1/x = C) (h3 : 0 < A) (h4 : 0 < C) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ (∀ B, (x^2 + (1/x)^2 = B) → (x + 1/x = C) → (B / C = 0 → B = 0)) :=
by
  sorry

end min_possible_A_div_C_l118_118167


namespace combined_gold_cost_l118_118835

def gary_gold_weight : ℕ := 30
def gary_gold_cost_per_gram : ℕ := 15
def anna_gold_weight : ℕ := 50
def anna_gold_cost_per_gram : ℕ := 20

theorem combined_gold_cost : (gary_gold_weight * gary_gold_cost_per_gram) + (anna_gold_weight * anna_gold_cost_per_gram) = 1450 :=
by {
  sorry -- Proof goes here
}

end combined_gold_cost_l118_118835


namespace sum_of_integers_c_with_four_solutions_l118_118586

noncomputable def g (x : ℝ) : ℝ :=
  ((x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 120) - 2

theorem sum_of_integers_c_with_four_solutions :
  (∃ (c : ℤ), ∀ x : ℝ, -4.5 ≤ x ∧ x ≤ 4.5 → g x = c ↔ c = -2) → c = -2 :=
by
  sorry

end sum_of_integers_c_with_four_solutions_l118_118586


namespace rationalize_denominator_l118_118074

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l118_118074


namespace parabola_equation_l118_118131

theorem parabola_equation (x y : ℝ) :
  (∃p : ℝ, x = 4 ∧ y = -2 ∧ (x^2 = -2 * p * y ∨ y^2 = 2 * p * x) → (x^2 = -8 * y ∨ y^2 = x)) :=
by
  sorry

end parabola_equation_l118_118131


namespace sum_of_distances_l118_118177

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_2 = d_1 + 5) (h2 : d_1 + d_2 = 13) :
  d_1 + d_2 = 13 :=
by sorry

end sum_of_distances_l118_118177


namespace M_equals_N_l118_118193

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {y | 0 ≤ y}

-- State the main proof goal
theorem M_equals_N : M = N :=
by
  sorry

end M_equals_N_l118_118193


namespace find_m_value_l118_118015

theorem find_m_value : ∃ m : ℤ, 81 - 6 = 25 + m ∧ m = 50 :=
by
  sorry

end find_m_value_l118_118015


namespace avg_speed_of_car_l118_118386

noncomputable def average_speed (distance1 distance2 : ℕ) (time1 time2 : ℕ) : ℕ :=
  (distance1 + distance2) / (time1 + time2)

theorem avg_speed_of_car :
  average_speed 65 45 1 1 = 55 := by
  sorry

end avg_speed_of_car_l118_118386


namespace maximize_Sn_l118_118454

theorem maximize_Sn (a1 : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : a1 > 0)
  (h2 : a1 + 9 * (a1 + 5 * d) = 0)
  (h_sn : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)) :
  ∃ n_max, ∀ n, S n ≤ S n_max ∧ n_max = 5 :=
by
  sorry

end maximize_Sn_l118_118454


namespace surface_area_rectangular_solid_l118_118675

def length := 5
def width := 4
def depth := 1

def surface_area (l w d : ℕ) := 2 * (l * w) + 2 * (l * d) + 2 * (w * d)

theorem surface_area_rectangular_solid : surface_area length width depth = 58 := 
by 
sorry

end surface_area_rectangular_solid_l118_118675


namespace car_average_speed_l118_118500

noncomputable def average_speed (speeds : List ℝ) (distances : List ℝ) (times : List ℝ) : ℝ :=
  (distances.sum + times.sum) / times.sum

theorem car_average_speed :
  let distances := [30, 35, 35, 52 / 3, 15]
  let times := [30 / 45, 35 / 55, 30 / 60, 20 / 60, 15 / 65]
  average_speed [45, 55, 70, 52, 65] distances times = 64.82 := by
  sorry

end car_average_speed_l118_118500


namespace trigonometric_identity_solution_l118_118347

theorem trigonometric_identity_solution 
  (alpha beta : ℝ)
  (h1 : π / 4 < alpha)
  (h2 : alpha < 3 * π / 4)
  (h3 : 0 < beta)
  (h4 : beta < π / 4)
  (h5 : Real.cos (π / 4 + alpha) = -4 / 5)
  (h6 : Real.sin (3 * π / 4 + beta) = 12 / 13) :
  (Real.sin (alpha + beta) = 63 / 65) ∧
  (Real.cos (alpha - beta) = -33 / 65) :=
by
  sorry

end trigonometric_identity_solution_l118_118347


namespace calc_expr_l118_118696

theorem calc_expr : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 :=
by
  sorry

end calc_expr_l118_118696


namespace least_n_probability_lt_1_over_10_l118_118508

theorem least_n_probability_lt_1_over_10 : 
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < 1 / 10 ∧ ∀ m < n, ¬ ((1 / 2 : ℝ) ^ m < 1 / 10) :=
by
  sorry

end least_n_probability_lt_1_over_10_l118_118508


namespace greatest_value_of_x_l118_118934

theorem greatest_value_of_x (x : ℕ) : (Nat.lcm (Nat.lcm x 12) 18 = 180) → x ≤ 180 :=
by
  sorry

end greatest_value_of_x_l118_118934


namespace inequality_solution_l118_118537

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end inequality_solution_l118_118537


namespace difference_between_c_and_a_l118_118942

variables (a b c : ℝ)

theorem difference_between_c_and_a
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end difference_between_c_and_a_l118_118942


namespace village_population_l118_118479

theorem village_population (x : ℝ) (h : 0.96 * x = 23040) : x = 24000 := sorry

end village_population_l118_118479


namespace part1_part2_l118_118754

def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2 * a|

-- Define the first part of the problem
theorem part1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
sorry

-- Define the second part of the problem
theorem part2 (a x : ℝ) (h1 : a ≥ 1) : f x a ≥ 2 :=
sorry

end part1_part2_l118_118754


namespace find_beta_l118_118458

open Real

theorem find_beta (α β : ℝ) (h1 : cos α = 1 / 7) (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : β = π / 3 :=
by
  sorry

end find_beta_l118_118458


namespace profit_percent_eq_20_l118_118631

-- Define cost price 'C' and original selling price 'S'
variable (C S : ℝ)

-- Hypothesis: selling at 2/3 of the original price results in a 20% loss 
def condition (C S : ℝ) : Prop :=
  (2 / 3) * S = 0.8 * C

-- Main theorem: profit percent when selling at the original price is 20%
theorem profit_percent_eq_20 (C S : ℝ) (h : condition C S) : (S - C) / C * 100 = 20 :=
by
  -- Proof steps would go here but we use sorry to indicate the proof is omitted
  sorry

end profit_percent_eq_20_l118_118631


namespace numberOfWaysToChooseLeadership_is_correct_l118_118466

noncomputable def numberOfWaysToChooseLeadership (totalMembers : ℕ) : ℕ :=
  let choicesForGovernor := totalMembers
  let remainingAfterGovernor := totalMembers - 1

  let choicesForDeputies := Nat.choose remainingAfterGovernor 3
  let remainingAfterDeputies := remainingAfterGovernor - 3

  let choicesForLieutenants1 := Nat.choose remainingAfterDeputies 3
  let remainingAfterLieutenants1 := remainingAfterDeputies - 3

  let choicesForLieutenants2 := Nat.choose remainingAfterLieutenants1 3
  let remainingAfterLieutenants2 := remainingAfterLieutenants1 - 3

  let choicesForLieutenants3 := Nat.choose remainingAfterLieutenants2 3
  let remainingAfterLieutenants3 := remainingAfterLieutenants2 - 3

  let choicesForSubordinates : List ℕ := 
    (List.range 8).map (λ i => Nat.choose (remainingAfterLieutenants3 - 2*i) 2)

  choicesForGovernor 
  * choicesForDeputies 
  * choicesForLieutenants1 
  * choicesForLieutenants2 
  * choicesForLieutenants3 
  * List.prod choicesForSubordinates

theorem numberOfWaysToChooseLeadership_is_correct : 
  numberOfWaysToChooseLeadership 35 = 
    35 * Nat.choose 34 3 * Nat.choose 31 3 * Nat.choose 28 3 * Nat.choose 25 3 *
    Nat.choose 16 2 * Nat.choose 14 2 * Nat.choose 12 2 * Nat.choose 10 2 *
    Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 :=
by
  sorry

end numberOfWaysToChooseLeadership_is_correct_l118_118466


namespace sum_inverses_of_roots_l118_118441

open Polynomial

theorem sum_inverses_of_roots (a b c : ℝ) (h1 : a^3 - 2020 * a + 1010 = 0)
    (h2 : b^3 - 2020 * b + 1010 = 0) (h3 : c^3 - 2020 * c + 1010 = 0) :
    (1/a) + (1/b) + (1/c) = 2 := 
  sorry

end sum_inverses_of_roots_l118_118441


namespace regular_polygon_perimeter_l118_118276

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l118_118276


namespace expand_polynomial_l118_118083

variable (x : ℝ)

theorem expand_polynomial :
  (7 * x - 3) * (2 * x ^ 3 + 5 * x ^ 2 - 4) = 14 * x ^ 4 + 29 * x ^ 3 - 15 * x ^ 2 - 28 * x + 12 := by
  sorry

end expand_polynomial_l118_118083


namespace right_triangle_properties_l118_118399

theorem right_triangle_properties (a b c h : ℝ)
  (ha: a = 5) (hb: b = 12) (h_right_angle: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c * h) :
  c = 13 ∧ h = 60 / 13 :=
by
  sorry

end right_triangle_properties_l118_118399


namespace initial_amount_l118_118203

theorem initial_amount (x : ℕ) (h1 : x - 3 + 14 = 22) : x = 11 :=
sorry

end initial_amount_l118_118203


namespace white_pairs_coincide_l118_118020

theorem white_pairs_coincide 
  (red_half : ℕ) (blue_half : ℕ) (white_half : ℕ)
  (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) :
  red_half = 2 → blue_half = 4 → white_half = 6 →
  red_pairs = 1 → blue_pairs = 2 → red_white_pairs = 2 →
  2 * (red_half - red_pairs + blue_half - 2 * blue_pairs + 
       white_half - 2 * red_white_pairs) = 4 :=
by
  intros 
    h_red_half h_blue_half h_white_half 
    h_red_pairs h_blue_pairs h_red_white_pairs
  rw [h_red_half, h_blue_half, h_white_half, 
      h_red_pairs, h_blue_pairs, h_red_white_pairs]
  sorry

end white_pairs_coincide_l118_118020


namespace number_of_questions_in_test_l118_118637

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end number_of_questions_in_test_l118_118637


namespace polynomial_transformation_l118_118671

noncomputable def p : ℝ → ℝ := sorry

variable (k : ℕ)

axiom ax1 (x : ℝ) : p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))

theorem polynomial_transformation (k : ℕ) (p : ℝ → ℝ)
  (h_p : ∀ x : ℝ, p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))) :
  ∀ x : ℝ, p (3 * x) = 3^(k - 1) * (p x + p (x + 1/3) + p (x + 2/3)) := sorry

end polynomial_transformation_l118_118671


namespace shark_fin_falcata_area_is_correct_l118_118622

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

end shark_fin_falcata_area_is_correct_l118_118622


namespace probability_closer_to_6_l118_118013

theorem probability_closer_to_6 :
  let interval : Set ℝ := Set.Icc 0 6
  let subinterval : Set ℝ := Set.Icc 3 6
  let length_interval := 6
  let length_subinterval := 3
  (length_subinterval / length_interval) = 0.5 := by
    sorry

end probability_closer_to_6_l118_118013


namespace number_4_div_p_equals_l118_118599

-- Assume the necessary conditions
variables (p q : ℝ)
variables (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778)

-- Define the proof problem
theorem number_4_div_p_equals (N : ℝ) (hN : 4 / p = N) : N = 8 :=
by 
  sorry

end number_4_div_p_equals_l118_118599


namespace number_of_math_fun_books_l118_118624

def intelligence_challenge_cost := 18
def math_fun_cost := 8
def total_spent := 92

theorem number_of_math_fun_books (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : intelligence_challenge_cost * x + math_fun_cost * y = total_spent) : y = 7 := 
by
  sorry

end number_of_math_fun_books_l118_118624


namespace PedoeInequalityHolds_l118_118729

noncomputable def PedoeInequality 
  (a b c a1 b1 c1 : ℝ) (Δ Δ1 : ℝ) :
  Prop :=
  a^2 * (b1^2 + c1^2 - a1^2) + 
  b^2 * (c1^2 + a1^2 - b1^2) + 
  c^2 * (a1^2 + b1^2 - c1^2) >= 16 * Δ * Δ1 

axiom areas_triangle 
  (a b c : ℝ) : ℝ 

axiom areas_triangle1 
  (a1 b1 c1 : ℝ) : ℝ 

theorem PedoeInequalityHolds 
  (a b c a1 b1 c1 : ℝ) 
  (Δ := areas_triangle a b c) 
  (Δ1 := areas_triangle1 a1 b1 c1) :
  PedoeInequality a b c a1 b1 c1 Δ Δ1 :=
sorry

end PedoeInequalityHolds_l118_118729


namespace price_decrease_required_to_initial_l118_118456

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

end price_decrease_required_to_initial_l118_118456


namespace green_ball_probability_l118_118240

def containerA := (8, 2) -- 8 green, 2 red
def containerB := (6, 4) -- 6 green, 4 red
def containerC := (5, 5) -- 5 green, 5 red
def containerD := (8, 2) -- 8 green, 2 red

def probability_of_green : ℚ :=
  (1 / 4) * (8 / 10) + (1 / 4) * (6 / 10) + (1 / 4) * (5 / 10) + (1 / 4) * (8 / 10)
  
theorem green_ball_probability :
  probability_of_green = 43 / 160 :=
sorry

end green_ball_probability_l118_118240


namespace each_child_apples_l118_118221

-- Define the given conditions
def total_apples : ℕ := 450
def num_adults : ℕ := 40
def num_adults_apples : ℕ := 3
def num_children : ℕ := 33

-- Define the theorem to prove
theorem each_child_apples : 
  let total_apples_eaten_by_adults := num_adults * num_adults_apples
  let total_apples_for_children := total_apples - total_apples_eaten_by_adults
  let apples_per_child := total_apples_for_children / num_children
  apples_per_child = 10 :=
by
  sorry

end each_child_apples_l118_118221


namespace abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l118_118254

theorem abs_neg_two_eq_two : |(-2)| = 2 :=
sorry

theorem neg_two_pow_zero_eq_one : (-2)^0 = 1 :=
sorry

end abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l118_118254


namespace total_population_of_towns_l118_118884

theorem total_population_of_towns :
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  num_towns * estimated_avg_pop = 95000 :=
by
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  show num_towns * estimated_avg_pop = 95000
  sorry

end total_population_of_towns_l118_118884


namespace evaluate_expression_l118_118445

theorem evaluate_expression :
  ((Int.ceil ((21 : ℚ) / 5 - Int.ceil ((35 : ℚ) / 23))) : ℚ) /
  (Int.ceil ((35 : ℚ) / 5 + Int.ceil ((5 * 23 : ℚ) / 35))) = 3 / 11 := by
  sorry

end evaluate_expression_l118_118445


namespace f_zero_f_odd_solve_inequality_l118_118874

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom increasing_on_nonneg : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem f_zero : f 0 = 0 :=
by sorry

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

theorem solve_inequality {x : ℝ} (h : 0 < x) : f (Real.log x / Real.log 10 - 1) < 0 ↔ 0 < x ∧ x < 10 :=
by sorry

end f_zero_f_odd_solve_inequality_l118_118874


namespace wholesale_price_l118_118731

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end wholesale_price_l118_118731


namespace a_takes_30_minutes_more_l118_118414

noncomputable def speed_ratio := 3 / 4
noncomputable def time_A := 2 -- 2 hours
noncomputable def time_diff (b_time : ℝ) := time_A - b_time

theorem a_takes_30_minutes_more (b_time : ℝ) 
  (h_ratio : speed_ratio = 3 / 4)
  (h_a : time_A = 2) :
  time_diff b_time = 0.5 →  -- because 0.5 hours = 30 minutes
  time_diff b_time * 60 = 30 :=
by sorry

end a_takes_30_minutes_more_l118_118414


namespace visitors_saturday_l118_118584

def friday_visitors : ℕ := 3575
def saturday_visitors : ℕ := 5 * friday_visitors

theorem visitors_saturday : saturday_visitors = 17875 := by
  -- proof details would go here
  sorry

end visitors_saturday_l118_118584


namespace fraction_proof_l118_118349

theorem fraction_proof (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
by
  sorry

end fraction_proof_l118_118349


namespace find_x2_l118_118175

theorem find_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 + x3 = 17) (h3 : x2 + x3 = 33) : x2 = 15 :=
by
  sorry

end find_x2_l118_118175


namespace product_of_conversions_l118_118132

-- Define the binary number 1101
def binary_number := 1101

-- Convert binary 1101 to decimal
def binary_to_decimal : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 212
def ternary_number := 212

-- Convert ternary 212 to decimal
def ternary_to_decimal : ℕ := 2 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Statement to prove
theorem product_of_conversions : (binary_to_decimal) * (ternary_to_decimal) = 299 := by
  sorry

end product_of_conversions_l118_118132


namespace evaluate_expression_l118_118799

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (1/3 : ℚ)
  let z := (-12 : ℚ)
  let w := (5 : ℚ)
  x^2 * y^3 * z + w = (179/36 : ℚ) :=
by
  sorry

end evaluate_expression_l118_118799


namespace arun_speed_ratio_l118_118350

namespace SpeedRatio

variables (V_a V_n V_a' : ℝ)
variable (distance : ℝ := 30)
variable (original_speed_Arun : ℝ := 5)
variable (time_Arun time_Anil time_Arun_new_speed : ℝ)

-- Conditions
theorem arun_speed_ratio :
  V_a = original_speed_Arun →
  time_Arun = distance / V_a →
  time_Anil = distance / V_n →
  time_Arun = time_Anil + 2 →
  time_Arun_new_speed = distance / V_a' →
  time_Arun_new_speed = time_Anil - 1 →
  V_a' / V_a = 2 := 
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1] at *
  sorry

end SpeedRatio

end arun_speed_ratio_l118_118350


namespace simplify_fraction_expression_l118_118701

theorem simplify_fraction_expression (d : ℤ) :
  (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 :=
by
  -- skip the proof by adding sorry
  sorry

end simplify_fraction_expression_l118_118701


namespace function_graph_second_quadrant_l118_118585

theorem function_graph_second_quadrant (b : ℝ) (h : ∀ x, 2 ^ x + b - 1 ≥ 0): b ≤ 0 :=
sorry

end function_graph_second_quadrant_l118_118585


namespace math_problem_l118_118663

noncomputable def problem_statement (f : ℚ → ℝ) : Prop :=
  (∀ r s : ℚ, ∃ n : ℤ, f (r + s) = f r + f s + n) →
  ∃ (q : ℕ) (p : ℤ), abs (f (1 / q) - p) ≤ 1 / 2012

-- To state this problem as a theorem in Lean 4
theorem math_problem (f : ℚ → ℝ) :
  problem_statement f :=
sorry

end math_problem_l118_118663


namespace old_hen_weight_unit_l118_118099

theorem old_hen_weight_unit (w : ℕ) (units : String) (opt1 opt2 opt3 opt4 : String)
  (h_opt1 : opt1 = "grams") (h_opt2 : opt2 = "kilograms") (h_opt3 : opt3 = "tons") (h_opt4 : opt4 = "meters") (h_w : w = 2) : 
  (units = opt2) :=
sorry

end old_hen_weight_unit_l118_118099


namespace triangular_pyramid_surface_area_l118_118656

theorem triangular_pyramid_surface_area
  (base_area : ℝ)
  (side_area : ℝ) :
  base_area = 3 ∧ side_area = 6 → base_area + 3 * side_area = 21 :=
by
  sorry

end triangular_pyramid_surface_area_l118_118656


namespace yellow_papers_count_l118_118550

theorem yellow_papers_count (n : ℕ) (total_papers : ℕ) (periphery_papers : ℕ) (inner_papers : ℕ) 
  (h1 : n = 10) 
  (h2 : total_papers = n * n) 
  (h3 : periphery_papers = 4 * n - 4)
  (h4 : inner_papers = total_papers - periphery_papers) :
  inner_papers = 64 :=
by
  sorry

end yellow_papers_count_l118_118550


namespace problem_statement_l118_118423

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l118_118423


namespace fifth_term_arithmetic_sequence_is_19_l118_118683

def arithmetic_sequence_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem fifth_term_arithmetic_sequence_is_19 :
  arithmetic_sequence_nth_term 3 4 5 = 19 := 
  by
  sorry

end fifth_term_arithmetic_sequence_is_19_l118_118683


namespace time_to_cook_one_potato_l118_118279

-- Definitions for the conditions
def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def remaining_minutes : ℕ := 45

-- Lean theorem that asserts the equivalence of the problem statement to the correct answer
theorem time_to_cook_one_potato (total_potatoes cooked_potatoes remaining_minutes : ℕ) 
  (h_total : total_potatoes = 16) 
  (h_cooked : cooked_potatoes = 7) 
  (h_remaining : remaining_minutes = 45) :
  (remaining_minutes / (total_potatoes - cooked_potatoes) = 5) :=
by
  -- Using sorry to skip proof
  sorry

end time_to_cook_one_potato_l118_118279


namespace trapezoid_area_l118_118047

theorem trapezoid_area (x : ℝ) :
  let base1 := 4 * x
  let base2 := 6 * x
  let height := x
  (base1 + base2) / 2 * height = 5 * x^2 :=
by
  sorry

end trapezoid_area_l118_118047


namespace program_output_is_201_l118_118732

theorem program_output_is_201 :
  ∃ x S n, x = 3 + 2 * n ∧ S = n^2 + 4 * n ∧ S ≥ 10000 ∧ x = 201 :=
by
  sorry

end program_output_is_201_l118_118732


namespace coin_combinations_l118_118491

theorem coin_combinations (pennies nickels dimes quarters : ℕ) :
  (1 * pennies + 5 * nickels + 10 * dimes + 25 * quarters = 50) →
  ∃ (count : ℕ), count = 35 := by
  sorry

end coin_combinations_l118_118491


namespace ratio_eq_one_l118_118376

variable {a b : ℝ}

theorem ratio_eq_one (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) : (a / 8) / (b / 7) = 1 := 
by
  sorry

end ratio_eq_one_l118_118376


namespace negation_of_exists_proposition_l118_118819

theorem negation_of_exists_proposition :
  ¬ (∃ x₀ : ℝ, x₀^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 ≥ 0 :=
by
  sorry

end negation_of_exists_proposition_l118_118819


namespace problem1_problem2_l118_118057

variable (α : ℝ)

axiom tan_alpha_condition : Real.tan (Real.pi + α) = -1/2

-- Problem 1 Statement
theorem problem1 
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) : 
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) / 
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3 * Real.pi / 2 - α)) = -7/9 := 
sorry

-- Problem 2 Statement
theorem problem2
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) :
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := 
sorry

end problem1_problem2_l118_118057


namespace rug_area_is_180_l118_118536

variables (w l : ℕ)

def length_eq_width_plus_eight (l w : ℕ) : Prop :=
  l = w + 8

def uniform_width_between_rug_and_room (d : ℕ) : Prop :=
  d = 8

def area_uncovered_by_rug (area : ℕ) : Prop :=
  area = 704

def area_of_rug (w l : ℕ) : ℕ :=
  l * w

theorem rug_area_is_180 (w l : ℕ) (hwld : length_eq_width_plus_eight l w)
  (huw : uniform_width_between_rug_and_room 8)
  (huar : area_uncovered_by_rug 704) :
  area_of_rug w l = 180 :=
sorry

end rug_area_is_180_l118_118536


namespace parallel_lines_slope_eq_l118_118889

theorem parallel_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 ↔ 6 * x + m * y + 11 = 0) → m = 8 :=
by
  sorry

end parallel_lines_slope_eq_l118_118889


namespace contrapositive_example_l118_118509

theorem contrapositive_example (x : ℝ) : 
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := 
by
  sorry

end contrapositive_example_l118_118509


namespace range_of_a_l118_118063

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((1 - a) * x > 1 - a) → (x < 1)) → (1 < a) :=
by sorry

end range_of_a_l118_118063


namespace expense_of_three_yuan_l118_118566

def isIncome (x : Int) : Prop := x > 0
def isExpense (x : Int) : Prop := x < 0
def incomeOfTwoYuan : Int := 2

theorem expense_of_three_yuan : isExpense (-3) :=
by
  -- Assuming the conditions:
  -- Income is positive: isIncome incomeOfTwoYuan (which is 2)
  -- Expenses are negative
  -- Expenses of 3 yuan should be denoted as -3 yuan
  sorry

end expense_of_three_yuan_l118_118566


namespace external_tangent_inequality_l118_118982

variable (x y z : ℝ)
variable (a b c T : ℝ)

-- Definitions based on conditions
def a_def : a = x + y := sorry
def b_def : b = y + z := sorry
def c_def : c = z + x := sorry
def T_def : T = π * x^2 + π * y^2 + π * z^2 := sorry

-- The theorem to prove
theorem external_tangent_inequality
    (a_def : a = x + y) 
    (b_def : b = y + z) 
    (c_def : c = z + x) 
    (T_def : T = π * x^2 + π * y^2 + π * z^2) : 
    π * (a + b + c) ^ 2 ≤ 12 * T := 
sorry

end external_tangent_inequality_l118_118982


namespace f_zero_f_positive_all_f_increasing_f_range_l118_118298

universe u

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_positive : ∀ x : ℝ, 0 < x → f x > 1
axiom f_add_prop : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(0) = 1
theorem f_zero : f 0 = 1 := sorry

-- Problem 2: Prove that for any x in ℝ, f(x) > 0
theorem f_positive_all (x : ℝ) : f x > 0 := sorry

-- Problem 3: Prove that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 4: Given f(x) * f(2x - x²) > 1, find the range of x
theorem f_range (x : ℝ) (h : f x * f (2*x - x^2) > 1) : 0 < x ∧ x < 3 := sorry

end f_zero_f_positive_all_f_increasing_f_range_l118_118298


namespace probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l118_118094

noncomputable def p_a : ℝ := 0.18
noncomputable def p_b : ℝ := 0.5
noncomputable def p_b_given_a : ℝ := 0.2
noncomputable def p_c : ℝ := 0.3
noncomputable def p_c_given_a : ℝ := 0.4
noncomputable def p_c_given_b : ℝ := 0.6

noncomputable def p_a_and_b : ℝ := p_a * p_b_given_a
noncomputable def p_a_and_b_and_c : ℝ := p_c_given_a * p_a_and_b
noncomputable def p_a_and_b_given_c : ℝ := p_a_and_b_and_c / p_c
noncomputable def p_a_and_c_given_b : ℝ := p_a_and_b_and_c / p_b

theorem probability_a_and_b_and_c : p_a_and_b_and_c = 0.0144 := by
  sorry

theorem probability_a_and_b_given_c : p_a_and_b_given_c = 0.048 := by
  sorry

theorem probability_a_and_c_given_b : p_a_and_c_given_b = 0.0288 := by
  sorry

end probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l118_118094


namespace value_of_abs_sum_l118_118925

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem value_of_abs_sum (θ x : ℝ) (h : Real.log x / Real.log 2 = 3 - 2 * cos_squared θ) :
  |x - 2| + |x - 8| = 6 := by
    sorry

end value_of_abs_sum_l118_118925


namespace original_wage_before_increase_l118_118486

theorem original_wage_before_increase (new_wage : ℝ) (increase_rate : ℝ) (original_wage : ℝ) (h : new_wage = original_wage + increase_rate * original_wage) : 
  new_wage = 42 → increase_rate = 0.50 → original_wage = 28 :=
by
  intros h_new_wage h_increase_rate
  have h1 : new_wage = 42 := h_new_wage
  have h2 : increase_rate = 0.50 := h_increase_rate
  have h3 : new_wage = original_wage + increase_rate * original_wage := h
  sorry

end original_wage_before_increase_l118_118486


namespace sum_of_prime_factors_is_prime_l118_118170

/-- Define the specific number in question -/
def num := 30030

/-- List the prime factors of the number -/
def prime_factors := [2, 3, 5, 7, 11, 13]

/-- Sum of the prime factors -/
def sum_prime_factors := prime_factors.sum

theorem sum_of_prime_factors_is_prime :
  sum_prime_factors = 41 ∧ Prime 41 := 
by
  -- The conditions are encapsulated in the definitions above
  -- Now, establish the required proof goal using these conditions
  sorry

end sum_of_prime_factors_is_prime_l118_118170


namespace greatest_power_of_2_divides_10_1004_minus_4_502_l118_118089

theorem greatest_power_of_2_divides_10_1004_minus_4_502 :
  ∃ k, 10^1004 - 4^502 = 2^1007 * k :=
sorry

end greatest_power_of_2_divides_10_1004_minus_4_502_l118_118089


namespace math_proof_problem_l118_118979

noncomputable def f (a b : ℚ) : ℝ := sorry

axiom f_cond1 (a b c : ℚ) : f (a * b) c = f a c * f b c ∧ f c (a * b) = f c a * f c b
axiom f_cond2 (a : ℚ) : f a (1 - a) = 1

theorem math_proof_problem (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (f a a = 1) ∧ 
  (f a (-a) = 1) ∧
  (f a b * f b a = 1) := 
by 
  sorry

end math_proof_problem_l118_118979


namespace puppy_weight_l118_118859

variable (p s l r : ℝ)

theorem puppy_weight :
  p + s + l + r = 40 ∧ 
  p^2 + l^2 = 4 * s ∧ 
  p^2 + s^2 = l^2 → 
  p = Real.sqrt 2 :=
sorry

end puppy_weight_l118_118859


namespace nathaniel_initial_tickets_l118_118989

theorem nathaniel_initial_tickets (a b c : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) :
  a * b + c = 11 :=
by
  sorry

end nathaniel_initial_tickets_l118_118989


namespace value_of_f_at_3_l118_118058

def f (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_3 : f 3 = 9 / 7 := by
  sorry

end value_of_f_at_3_l118_118058


namespace angle_of_elevation_proof_l118_118192

noncomputable def height_of_lighthouse : ℝ := 100

noncomputable def distance_between_ships : ℝ := 273.2050807568877

noncomputable def angle_of_elevation_second_ship : ℝ := 45

noncomputable def distance_from_second_ship := height_of_lighthouse

noncomputable def distance_from_first_ship := distance_between_ships - distance_from_second_ship

noncomputable def tanθ := height_of_lighthouse / distance_from_first_ship

noncomputable def angle_of_elevation_first_ship := Real.arctan tanθ

theorem angle_of_elevation_proof :
  angle_of_elevation_first_ship = 30 := by
    sorry

end angle_of_elevation_proof_l118_118192


namespace sequence_periodic_mod_l118_118189

-- Define the sequence (u_n) recursively
def sequence_u (a : ℕ) : ℕ → ℕ
  | 0     => a  -- Note: u_1 is defined as the initial term a, treating the starting index as 0 for compatibility with Lean's indexing.
  | (n+1) => a ^ (sequence_u a n)

-- The theorem stating there exist integers k and N such that for all n ≥ N, u_{n+k} ≡ u_n (mod m)
theorem sequence_periodic_mod (a m : ℕ) (hm : 0 < m) (ha : 0 < a) :
  ∃ k N : ℕ, ∀ n : ℕ, N ≤ n → (sequence_u a (n + k) ≡ sequence_u a n [MOD m]) :=
by
  sorry

end sequence_periodic_mod_l118_118189


namespace lily_typing_break_time_l118_118195

theorem lily_typing_break_time :
  ∃ t : ℝ, (15 * t + 15 * t = 255) ∧ (19 = 2 * t + 2) ∧ (t = 8) := 
sorry

end lily_typing_break_time_l118_118195


namespace area_of_original_triangle_l118_118061

theorem area_of_original_triangle (a : Real) (S_intuitive : Real) : 
  a = 2 -> S_intuitive = (Real.sqrt 3) -> (S_intuitive / (Real.sqrt 2 / 4)) = 2 * Real.sqrt 6 := 
by
  sorry

end area_of_original_triangle_l118_118061


namespace johns_contribution_l118_118856

theorem johns_contribution (A : ℝ) (J : ℝ) : 
  (1.7 * A = 85) ∧ ((5 * A + J) / 6 = 85) → J = 260 := 
by
  sorry

end johns_contribution_l118_118856


namespace christina_total_payment_l118_118570

def item1_ticket_price : ℝ := 200
def item1_discount1 : ℝ := 0.25
def item1_discount2 : ℝ := 0.15
def item1_tax_rate : ℝ := 0.07

def item2_ticket_price : ℝ := 150
def item2_discount : ℝ := 0.30
def item2_tax_rate : ℝ := 0.10

def item3_ticket_price : ℝ := 100
def item3_discount : ℝ := 0.20
def item3_tax_rate : ℝ := 0.05

def expected_total : ℝ := 335.93

theorem christina_total_payment :
  let item1_final_price :=
    (item1_ticket_price * (1 - item1_discount1) * (1 - item1_discount2)) * (1 + item1_tax_rate)
  let item2_final_price :=
    (item2_ticket_price * (1 - item2_discount)) * (1 + item2_tax_rate)
  let item3_final_price :=
    (item3_ticket_price * (1 - item3_discount)) * (1 + item3_tax_rate)
  item1_final_price + item2_final_price + item3_final_price = expected_total :=
by
  sorry

end christina_total_payment_l118_118570


namespace find_n_in_sequence_l118_118640

theorem find_n_in_sequence (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = 2 * a n) 
    (h3 : S n = 126) 
    (h4 : S n = 2^(n+1) - 2) : 
  n = 6 :=
sorry

end find_n_in_sequence_l118_118640


namespace perimeter_one_face_of_cube_is_24_l118_118806

noncomputable def cube_volume : ℝ := 216
def perimeter_of_face_of_cube (V : ℝ) : ℝ := 4 * (V^(1/3) : ℝ)

theorem perimeter_one_face_of_cube_is_24 :
  perimeter_of_face_of_cube cube_volume = 24 := 
by
  -- This proof will invoke the calculation shown in the problem.
  sorry

end perimeter_one_face_of_cube_is_24_l118_118806


namespace total_amount_paid_l118_118691

variable (n : ℕ) (each_paid : ℕ)

/-- This is a statement that verifies the total amount paid given the number of friends and the amount each friend pays. -/
theorem total_amount_paid (h1 : n = 7) (h2 : each_paid = 70) : n * each_paid = 490 := by
  -- This proof will validate that the total amount paid is 490
  sorry

end total_amount_paid_l118_118691


namespace min_time_to_same_side_l118_118654

def side_length : ℕ := 50
def speed_A : ℕ := 5
def speed_B : ℕ := 3

def time_to_same_side (side_length speed_A speed_B : ℕ) : ℕ :=
  30

theorem min_time_to_same_side :
  time_to_same_side side_length speed_A speed_B = 30 :=
by
  -- The proof goes here
  sorry

end min_time_to_same_side_l118_118654


namespace solution_set_inequality_l118_118875

theorem solution_set_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (x - a) * (x - (1 / a)) < 0} = {x : ℝ | a < x ∧ x < 1 / a} := sorry

end solution_set_inequality_l118_118875


namespace trigonometric_identity_l118_118012

theorem trigonometric_identity :
  (1 / Real.cos (40 * Real.pi / 180) - 2 * Real.sqrt 3 / Real.sin (40 * Real.pi / 180)) = -4 * Real.tan (20 * Real.pi / 180) := 
sorry

end trigonometric_identity_l118_118012


namespace solve_system_of_equations_l118_118455

theorem solve_system_of_equations :
  {p : ℝ × ℝ | 
    (p.1^2 + p.2 + 1) * (p.2^2 + p.1 + 1) = 4 ∧
    (p.1^2 + p.2)^2 + (p.2^2 + p.1)^2 = 2} =
  {(0, 1), (1, 0), 
   ( (-1 + Real.sqrt 5) / 2, (-1 + Real.sqrt 5) / 2),
   ( (-1 - Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2) } :=
by
  sorry

end solve_system_of_equations_l118_118455


namespace infinite_series_sum_l118_118326

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) / 10^(n + 1) = 10 / 81 :=
sorry

end infinite_series_sum_l118_118326


namespace min_distance_ants_l118_118239

open Real

theorem min_distance_ants (points : Fin 1390 → ℝ × ℝ) :
  (∀ i j : Fin 1390, i ≠ j → dist (points i) (points j) > 0.02) → 
  (∀ i : Fin 1390, |(points i).snd| < 0.01) → 
  ∃ i j : Fin 1390, i ≠ j ∧ dist (points i) (points j) > 10 :=
by
  sorry

end min_distance_ants_l118_118239


namespace difference_between_possible_values_of_x_l118_118161

noncomputable def difference_of_roots (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ℝ :=
  let sol1 := 11  -- First root
  let sol2 := -11 -- Second root
  sol1 - sol2

theorem difference_between_possible_values_of_x (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) :
  difference_of_roots x h = 22 :=
sorry

end difference_between_possible_values_of_x_l118_118161


namespace no_three_digit_numbers_meet_conditions_l118_118135

theorem no_three_digit_numbers_meet_conditions :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (n % 10 = 5) ∧ (n % 10 = 0) → false := 
by {
  sorry
}

end no_three_digit_numbers_meet_conditions_l118_118135


namespace distance_between_cities_l118_118811

noncomputable def speed_a : ℝ := 1 / 10
noncomputable def speed_b : ℝ := 1 / 15
noncomputable def time_to_meet : ℝ := 6
noncomputable def distance_diff : ℝ := 12

theorem distance_between_cities : 
  (time_to_meet * (speed_a + speed_b) = 60) →
  time_to_meet * speed_a - time_to_meet * speed_b = distance_diff →
  time_to_meet * (speed_a + speed_b) = 60 :=
by
  intros h1 h2
  sorry

end distance_between_cities_l118_118811


namespace total_cost_is_734_l118_118000

-- Define the cost of each ice cream flavor
def cost_vanilla : ℕ := 99
def cost_chocolate : ℕ := 129
def cost_strawberry : ℕ := 149

-- Define the amount of each flavor Mrs. Hilt buys
def num_vanilla : ℕ := 2
def num_chocolate : ℕ := 3
def num_strawberry : ℕ := 1

-- Calculate the total cost in cents
def total_cost : ℕ :=
  (num_vanilla * cost_vanilla) +
  (num_chocolate * cost_chocolate) +
  (num_strawberry * cost_strawberry)

-- Statement of the proof problem
theorem total_cost_is_734 : total_cost = 734 :=
by
  sorry

end total_cost_is_734_l118_118000


namespace sum_of_inserted_numbers_eq_12_l118_118909

theorem sum_of_inserted_numbers_eq_12 (a b : ℝ) (r d : ℝ) 
  (h1 : a = 2 * r) 
  (h2 : b = 2 * r^2) 
  (h3 : b = a + d) 
  (h4 : 12 = b + d) : 
  a + b = 12 :=
by
  sorry

end sum_of_inserted_numbers_eq_12_l118_118909


namespace find_perimeter_square3_l118_118738

-- Define the conditions: perimeter of first and second square
def perimeter_square1 := 60
def perimeter_square2 := 48

-- Calculate side lengths based on the perimeter
def side_length_square1 := perimeter_square1 / 4
def side_length_square2 := perimeter_square2 / 4

-- Calculate areas of the two squares
def area_square1 := side_length_square1 * side_length_square1
def area_square2 := side_length_square2 * side_length_square2

-- Calculate the area of the third square
def area_square3 := area_square1 - area_square2

-- Calculate the side length of the third square
def side_length_square3 := Nat.sqrt area_square3

-- Define the perimeter of the third square
def perimeter_square3 := 4 * side_length_square3

/-- Theorem: The perimeter of the third square is 36 cm -/
theorem find_perimeter_square3 : perimeter_square3 = 36 := by
  sorry

end find_perimeter_square3_l118_118738


namespace price_of_one_table_l118_118929

variable (C T : ℝ)

def cond1 := 2 * C + T = 0.6 * (C + 2 * T)
def cond2 := C + T = 60
def solution := T = 52.5

theorem price_of_one_table (h1 : cond1 C T) (h2 : cond2 C T) : solution T :=
by
  sorry

end price_of_one_table_l118_118929


namespace eggs_in_each_basket_l118_118231

theorem eggs_in_each_basket
  (total_red_eggs : ℕ)
  (total_orange_eggs : ℕ)
  (h_red : total_red_eggs = 30)
  (h_orange : total_orange_eggs = 45)
  (eggs_in_each_basket : ℕ)
  (h_at_least : eggs_in_each_basket ≥ 5) :
  (total_red_eggs % eggs_in_each_basket = 0) ∧ 
  (total_orange_eggs % eggs_in_each_basket = 0) ∧
  eggs_in_each_basket = 15 := sorry

end eggs_in_each_basket_l118_118231


namespace cups_per_serving_l118_118049

theorem cups_per_serving (total_cups servings : ℝ) (h1 : total_cups = 36) (h2 : servings = 18.0) :
  total_cups / servings = 2 :=
by 
  sorry

end cups_per_serving_l118_118049


namespace european_postcards_cost_l118_118327

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end european_postcards_cost_l118_118327


namespace sum_zero_of_cubic_identity_l118_118913

theorem sum_zero_of_cubic_identity (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a^3 + b^3 + c^3 = 3 * a * b * c) : 
  a + b + c = 0 :=
by
  sorry

end sum_zero_of_cubic_identity_l118_118913


namespace problem1_problem2_l118_118966

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 - 3 * x

theorem problem1 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → a ≤ 0 :=
sorry

theorem problem2 (a : ℝ) (h : a = 6) :
  x = 3 ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 6 → f x 6 ≤ -6 ∧ f x 6 ≥ -18) :=
sorry

end problem1_problem2_l118_118966


namespace twenty_four_game_l118_118542

theorem twenty_four_game : 8 / (3 - 8 / 3) = 24 := 
by
  sorry

end twenty_four_game_l118_118542


namespace probability_three_red_balls_l118_118459

open scoped BigOperators

noncomputable def hypergeometric_prob (r : ℕ) (b : ℕ) (k : ℕ) (d : ℕ) : ℝ :=
  (Nat.choose r d * Nat.choose b (k - d) : ℝ) / Nat.choose (r + b) k

theorem probability_three_red_balls :
  hypergeometric_prob 10 5 5 3 = 1200 / 3003 :=
by sorry

end probability_three_red_balls_l118_118459


namespace sum_of_first_53_odd_numbers_l118_118932

theorem sum_of_first_53_odd_numbers :
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  let sum := 53 / 2 * (first_term + last_term)
  sum = 2809 :=
by
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  have last_term_val : last_term = 105 := by
    sorry
  let sum := 53 / 2 * (first_term + last_term)
  have sum_val : sum = 2809 := by
    sorry
  exact sum_val

end sum_of_first_53_odd_numbers_l118_118932


namespace range_AD_dot_BC_l118_118846

noncomputable def vector_dot_product_range (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : ℝ :=
  let ab := 2
  let ac := 1
  let bc := ac - ab
  let ad := x * ac + (1 - x) * ab
  ad * bc

theorem range_AD_dot_BC : 
  ∃ (a b : ℝ), vector_dot_product_range x h1 h2 = a ∧ ∀ (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1), a ≤ vector_dot_product_range x h1 h2 ∧ vector_dot_product_range x h1 h2 ≤ b :=
sorry

end range_AD_dot_BC_l118_118846


namespace opposite_of_negative_fraction_l118_118085

theorem opposite_of_negative_fraction :
  -(-1 / 2023) = (1 / 2023) :=
by
  sorry

end opposite_of_negative_fraction_l118_118085


namespace no_integer_roots_l118_118119

theorem no_integer_roots (a b x : ℤ) : 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 ≠ 0 :=
sorry

end no_integer_roots_l118_118119


namespace greatest_x_lcm_l118_118120

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l118_118120


namespace find_constants_l118_118657

variable (x : ℝ)

def A := 3
def B := -3
def C := 11

theorem find_constants (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  (5 * x + 2) / ((x - 2) * (x - 4)^2) = A / (x - 2) + B / (x - 4) + C / (x - 4)^2 :=
by
  unfold A B C
  sorry

end find_constants_l118_118657


namespace total_is_correct_l118_118511

-- Define the given conditions.
def dividend : ℕ := 55
def divisor : ℕ := 11
def quotient := dividend / divisor
def total := dividend + quotient + divisor

-- State the theorem to be proven.
theorem total_is_correct : total = 71 := by sorry

end total_is_correct_l118_118511


namespace calculate_expression_solve_quadratic_l118_118334

-- Problem 1
theorem calculate_expression (x : ℝ) (hx : x > 0) :
  (2 / 3) * Real.sqrt (9 * x) + 6 * Real.sqrt (x / 4) - x * Real.sqrt (1 / x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem solve_quadratic (x : ℝ) (h : x^2 - 4 * x + 1 = 0) :
  x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
sorry

end calculate_expression_solve_quadratic_l118_118334


namespace all_sets_B_l118_118998

open Set

theorem all_sets_B (B : Set ℕ) :
  { B | {1, 2} ∪ B = {1, 2, 3} } =
  ({ {3}, {1, 3}, {2, 3}, {1, 2, 3} } : Set (Set ℕ)) :=
sorry

end all_sets_B_l118_118998


namespace determine_a7_l118_118198

noncomputable def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => a1
| (n+1) => a1 + n * d

noncomputable def sum_arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a1 * (n + 1) + (n * (n + 1) * d) / 2

theorem determine_a7 (a1 d : ℤ) (a2 : a1 + d = 7) (S7 : sum_arithmetic_seq a1 d 7 = -7) : arithmetic_seq a1 d 7 = -13 :=
by
  sorry

end determine_a7_l118_118198


namespace oliver_more_money_l118_118332

noncomputable def totalOliver : ℕ := 10 * 20 + 3 * 5
noncomputable def totalWilliam : ℕ := 15 * 10 + 4 * 5

theorem oliver_more_money : totalOliver - totalWilliam = 45 := by
  sorry

end oliver_more_money_l118_118332


namespace simplify_and_find_ratio_l118_118518

theorem simplify_and_find_ratio (m : ℤ) : 
  let expr := (6 * m + 18) / 6 
  let c := 1
  let d := 3
  (c / d : ℚ) = 1 / 3 := 
by
  -- Conditions and transformations are stated here
  -- (6 * m + 18) / 6 can be simplified step-by-step
  sorry

end simplify_and_find_ratio_l118_118518


namespace inequality_x2_8_over_xy_y2_l118_118003

open Real

theorem inequality_x2_8_over_xy_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^2 + 8 / (x * y) + y^2 ≥ 8 := 
sorry

end inequality_x2_8_over_xy_y2_l118_118003


namespace total_cost_of_fencing_l118_118133

theorem total_cost_of_fencing (length breadth : ℕ) (cost_per_metre : ℕ) 
  (h1 : length = breadth + 20) 
  (h2 : length = 200) 
  (h3 : cost_per_metre = 26): 
  2 * (length + breadth) * cost_per_metre = 20140 := 
by sorry

end total_cost_of_fencing_l118_118133


namespace cannot_determine_both_correct_l118_118366

-- Definitions
def total_students : ℕ := 40
def answered_q1_correctly : ℕ := 30
def did_not_take_test : ℕ := 10

-- Assertion that the number of students answering both questions correctly cannot be determined
theorem cannot_determine_both_correct (answered_q2_correctly : ℕ) :
  (∃ (both_correct : ℕ), both_correct ≤ answered_q1_correctly ∧ both_correct ≤ answered_q2_correctly)  ↔ answered_q2_correctly > 0 :=
by 
 sorry

end cannot_determine_both_correct_l118_118366


namespace find_parallel_line_l118_118995

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end find_parallel_line_l118_118995


namespace length_of_bridge_is_230_l118_118855

noncomputable def train_length : ℚ := 145
noncomputable def train_speed_kmh : ℚ := 45
noncomputable def time_to_cross_bridge : ℚ := 30
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 1000) / 3600
noncomputable def bridge_length : ℚ := (train_speed_ms * time_to_cross_bridge) - train_length

theorem length_of_bridge_is_230 :
  bridge_length = 230 :=
sorry

end length_of_bridge_is_230_l118_118855


namespace slower_train_speed_l118_118533

theorem slower_train_speed
  (v : ℝ) -- the speed of the slower train (kmph)
  (faster_train_speed : ℝ := 72)        -- the speed of the faster train
  (time_to_cross_man : ℝ := 18)         -- time to cross a man in the slower train (seconds)
  (faster_train_length : ℝ := 180)      -- length of the faster train (meters))
  (conversion_factor : ℝ := 5 / 18)     -- conversion factor from kmph to m/s
  (relative_speed_m_s : ℝ := ((faster_train_speed - v) * conversion_factor)) :
  ((faster_train_length : ℝ) = (relative_speed_m_s * time_to_cross_man)) →
  v = 36 :=
by
  -- the actual proof needs to be filled here
  sorry

end slower_train_speed_l118_118533


namespace union_A_B_l118_118467

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

theorem union_A_B (x : ℝ) : (x ∈ A ∨ x ∈ B) ↔ x < 3 := by
  sorry

end union_A_B_l118_118467


namespace spinner_probability_l118_118052

theorem spinner_probability (P_D P_E : ℝ) (hD : P_D = 2/5) (hE : P_E = 1/5) 
  (hTotal : P_D + P_E + P_F = 1) : P_F = 2/5 :=
by
  sorry

end spinner_probability_l118_118052


namespace line_through_two_points_l118_118904

theorem line_through_two_points (x y : ℝ) (hA : (x, y) = (3, 0)) (hB : (x, y) = (0, 2)) :
  2 * x + 3 * y - 6 = 0 :=
sorry 

end line_through_two_points_l118_118904


namespace kernel_count_in_final_bag_l118_118205

namespace PopcornKernelProblem

def percentage_popped (popped total : ℕ) : ℤ := ((popped : ℤ) * 100) / (total : ℤ)

def first_bag_percentage := percentage_popped 60 75
def second_bag_percentage := percentage_popped 42 50
def final_bag_percentage (x : ℕ) : ℤ := percentage_popped 82 x

theorem kernel_count_in_final_bag :
  (first_bag_percentage + second_bag_percentage + final_bag_percentage 100) / 3 = 82 := 
sorry

end PopcornKernelProblem

end kernel_count_in_final_bag_l118_118205


namespace simplify_and_evaluate_expression_l118_118939

/-
Problem: Prove ( (a + 1) / (a - 1) + 1 ) / ( 2a / (a^2 - 1) ) = 2024 given a = 2023.
-/

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  ( (a + 1) / (a - 1) + 1 ) / ( 2 * a / (a^2 - 1) ) = 2024 :=
by
  sorry

end simplify_and_evaluate_expression_l118_118939


namespace problem_proof_l118_118782

noncomputable def original_number_of_buses_and_total_passengers : Nat × Nat :=
  let k := 24
  let total_passengers := 529
  (k, total_passengers)

theorem problem_proof (k n : Nat) (h₁ : n = 22 + 23 / (k - 1)) (h₂ : 22 * k + 1 = n * (k - 1)) (h₃ : k ≥ 2) (h₄ : n ≤ 32) :
  (k, 22 * k + 1) = original_number_of_buses_and_total_passengers :=
by
  sorry

end problem_proof_l118_118782


namespace energy_calculation_l118_118010

noncomputable def stormy_day_energy_production 
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (proportional_increase : ℝ) : ℝ :=
  proportional_increase * (energy_per_day * days * number_of_windmills)

theorem energy_calculation
  (energy_per_day : ℝ) (days : ℝ) (number_of_windmills : ℝ) (wind_speed_proportion : ℝ)
  (stormy_day_energy_per_windmill : ℝ) (s : ℝ)
  (H1 : energy_per_day = 400) 
  (H2 : days = 2) 
  (H3 : number_of_windmills = 3) 
  (H4 : stormy_day_energy_per_windmill = s * energy_per_day)
  : stormy_day_energy_production energy_per_day days number_of_windmills s = s * (400 * 3 * 2) :=
by
  sorry

end energy_calculation_l118_118010


namespace find_a_and_a100_l118_118850

def seq (a : ℝ) (n : ℕ) : ℝ := (-1)^n * n + a

theorem find_a_and_a100 :
  ∃ a : ℝ, (seq a 1 + seq a 4 = 3 * seq a 2) ∧ (seq a 100 = 97) :=
by
  sorry

end find_a_and_a100_l118_118850


namespace center_of_circle_l118_118478

theorem center_of_circle (
  center : ℝ × ℝ
) :
  (∀ (p : ℝ × ℝ), (p.1 * 3 + p.2 * 4 = 24) ∨ (p.1 * 3 + p.2 * 4 = -6) → (dist center p = dist center p)) ∧
  (center.1 * 3 - center.2 = 0)
  → center = (3 / 5, 9 / 5) :=
by
  sorry

end center_of_circle_l118_118478


namespace calculate_value_l118_118848

-- Definition of the given values
def val1 : ℕ := 444
def val2 : ℕ := 44
def val3 : ℕ := 4

-- Theorem statement proving the value of the expression
theorem calculate_value : (val1 - val2 - val3) = 396 := 
by 
  sorry

end calculate_value_l118_118848


namespace triangle_perimeter_l118_118431

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (hc : c ^ 2 - 3 * c = c - 3) 
  (h3 : 3 + 3 > 5) 
  (h4 : 3 + 5 > 3) 
  (h5 : 5 + 3 > 3) : 
  a + b + c = 11 :=
by
  sorry

end triangle_perimeter_l118_118431


namespace ball_returns_to_bob_after_13_throws_l118_118931

theorem ball_returns_to_bob_after_13_throws:
  ∃ n : ℕ, n = 13 ∧ (∀ k, k < 13 → (1 + 3 * k) % 13 = 0) :=
sorry

end ball_returns_to_bob_after_13_throws_l118_118931


namespace horse_bags_problem_l118_118090

theorem horse_bags_problem (x y : ℤ) 
  (h1 : x - 1 = y + 1) : 
  x + 1 = 2 * (y - 1) :=
sorry

end horse_bags_problem_l118_118090


namespace proof_problem_l118_118820

-- Define the propositions as Lean terms
def prop1 : Prop := ∀ (l1 l2 : ℝ) (h1 : l1 ≠ 0 ∧ l2 ≠ 0), (l1 * l2 = -1) → (l1 ≠ l2)  -- Two perpendicular lines must intersect (incorrect definition)
def prop2 : Prop := ∀ (l : ℝ), ∃! (m : ℝ), (l * m = -1)  -- There is only one perpendicular line (incorrect definition)
def prop3 : Prop := (∀ (α β γ : ℝ), α = β → γ = 90 → α + γ = β + γ)  -- Equal corresponding angles when intersecting a third (incorrect definition)
def prop4 : Prop := ∀ (A B C : ℝ), (A = B ∧ B = C) → (A = C)  -- Transitive property of parallel lines

-- The statement that only one of these propositions is true, and it is the fourth one
theorem proof_problem (h1 : ¬ prop1) (h2 : ¬ prop2) (h3 : ¬ prop3) (h4 : prop4) : 
  ∃! (i : ℕ), i = 4 := 
by
  sorry

end proof_problem_l118_118820


namespace perpendicular_lines_l118_118374

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y - 1 = 0 → x + 2 * y = 0) →
  (a = 1) :=
by
  sorry

end perpendicular_lines_l118_118374


namespace dinner_cakes_today_6_l118_118747

-- Definitions based on conditions
def lunch_cakes_today : ℕ := 5
def dinner_cakes_today (x : ℕ) : ℕ := x
def yesterday_cakes : ℕ := 3
def total_cakes_served : ℕ := 14

-- Lean statement to prove the mathematical equivalence
theorem dinner_cakes_today_6 (x : ℕ) (h : lunch_cakes_today + dinner_cakes_today x + yesterday_cakes = total_cakes_served) : x = 6 :=
by {
  sorry -- Proof to be completed.
}

end dinner_cakes_today_6_l118_118747


namespace classroom_width_perimeter_ratio_l118_118412

theorem classroom_width_perimeter_ratio
  (L : Real) (W : Real) (P : Real)
  (hL : L = 15) (hW : W = 10)
  (hP : P = 2 * (L + W)) :
  W / P = 1 / 5 :=
sorry

end classroom_width_perimeter_ratio_l118_118412


namespace unique_root_range_l118_118993

theorem unique_root_range (a : ℝ) :
  (x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → (∃! x : ℝ, x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → - (Real.sqrt 3) / 2 < a ∧ a < (Real.sqrt 3) / 2 :=
by
  sorry

end unique_root_range_l118_118993


namespace surface_area_of_brick_l118_118127

namespace SurfaceAreaProof

def brick_length : ℝ := 8
def brick_width : ℝ := 6
def brick_height : ℝ := 2

theorem surface_area_of_brick :
  2 * (brick_length * brick_width + brick_length * brick_height + brick_width * brick_height) = 152 :=
by
  sorry

end SurfaceAreaProof

end surface_area_of_brick_l118_118127


namespace cost_of_marker_l118_118196

theorem cost_of_marker (n m : ℝ) (h1 : 3 * n + 2 * m = 7.45) (h2 : 4 * n + 3 * m = 10.40) : m = 1.40 :=
  sorry

end cost_of_marker_l118_118196


namespace parabola_directrix_l118_118886

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (x1 x2 t : ℝ) 
  (h_intersect : ∃ y1 y2, y1 = x1 + t ∧ y2 = x2 + t ∧ x1^2 = 2 * p * y1 ∧ x2^2 = 2 * p * y2)
  (h_midpoint : (x1 + x2) / 2 = 2) :
  p = 2 → ∃ d : ℝ, d = -1 := 
by
  sorry

end parabola_directrix_l118_118886


namespace pump_out_time_l118_118310

theorem pump_out_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (rate : ℝ)
  (H_length : length = 50)
  (H_width : width = 30)
  (H_depth : depth = 1.8)
  (H_rate : rate = 2.5) : 
  (length * width * depth) / rate / 60 = 18 :=
by
  sorry

end pump_out_time_l118_118310


namespace power_increased_by_four_l118_118526

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l118_118526


namespace sequence_geometric_and_sum_l118_118186

variables {S : ℕ → ℝ} (a1 : S 1 = 1)
variable (n : ℕ)
def a := (S (n+1) - 2 * S n, S n)
def b := (2, n)

/-- Prove that the sequence {S n / n} is a geometric sequence 
with first term 1 and common ratio 2, and find the sum of the first 
n terms of the sequence {S n} -/
theorem sequence_geometric_and_sum {S : ℕ → ℝ} (a1 : S 1 = 1)
  (n : ℕ)
  (parallel : ∀ n, n * (S (n + 1) - 2 * S n) = 2 * S n) :
  ∃ r : ℝ, r = 2 ∧ ∃ T : ℕ → ℝ, T n = (n-1)*2^n + 1 :=
by
  sorry

end sequence_geometric_and_sum_l118_118186


namespace find_second_bank_account_balance_l118_118253

theorem find_second_bank_account_balance : 
  (exists (X : ℝ),  
    let raw_material_cost := 100
    let machinery_cost := 125
    let raw_material_tax := 0.05 * raw_material_cost
    let discounted_machinery_cost := machinery_cost - (0.1 * machinery_cost)
    let machinery_tax := 0.08 * discounted_machinery_cost
    let total_raw_material_cost := raw_material_cost + raw_material_tax
    let total_machinery_cost := discounted_machinery_cost + machinery_tax
    let total_spent := total_raw_material_cost + total_machinery_cost
    let total_cash := 900 + X
    let spent_proportion := 0.2 * total_cash
    total_spent = spent_proportion → X = 232.50) :=
by {
  sorry
}

end find_second_bank_account_balance_l118_118253


namespace table_tennis_basketball_teams_l118_118092

theorem table_tennis_basketball_teams (X Y : ℕ)
  (h1 : X + Y = 50) 
  (h2 : 7 * Y = 3 * X)
  (h3 : 2 * (X - 8) = 3 * (Y + 8)) :
  X = 35 ∧ Y = 15 :=
by
  sorry

end table_tennis_basketball_teams_l118_118092


namespace triangle_third_side_lengths_l118_118769

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end triangle_third_side_lengths_l118_118769


namespace max_ratio_three_digit_sum_l118_118621

theorem max_ratio_three_digit_sum (N a b c : ℕ) (hN : N = 100 * a + 10 * b + c) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) :
  (∀ (N' a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' → 1 ≤ a' → b' ≤ 9 → c' ≤ 9 → (N' : ℚ) / (a' + b' + c') ≤ 100) :=
sorry

end max_ratio_three_digit_sum_l118_118621


namespace trigonometric_sign_l118_118936

open Real

theorem trigonometric_sign :
  (0 < 1 ∧ 1 < π / 2) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → sin x ≤ sin y)) ∧ 
  (∀ x y, (0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → cos x ≥ cos y)) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 :=
by
  sorry

end trigonometric_sign_l118_118936


namespace first_pipe_fill_time_l118_118600

theorem first_pipe_fill_time 
  (T : ℝ)
  (h1 : 48 * (1 / T - 1 / 24) + 18 * (1 / T) = 1) :
  T = 22 :=
by
  sorry

end first_pipe_fill_time_l118_118600


namespace inequality_cannot_hold_l118_118408

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_cannot_hold (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) := 
by {
  sorry
}

end inequality_cannot_hold_l118_118408


namespace rose_days_to_complete_work_l118_118999

theorem rose_days_to_complete_work (R : ℝ) (h1 : 1 / 10 + 1 / R = 1 / 8) : R = 40 := 
sorry

end rose_days_to_complete_work_l118_118999


namespace min_value_of_x_plus_y_l118_118043

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy: 0 < y) (h: 9 * x + y = x * y) : x + y ≥ 16 := 
sorry

end min_value_of_x_plus_y_l118_118043


namespace age_proof_l118_118780

theorem age_proof (M S Y : ℕ) (h1 : M = 36) (h2 : S = 12) (h3 : M = 3 * S) : 
  (M + Y = 2 * (S + Y)) ↔ (Y = 12) :=
by 
  sorry

end age_proof_l118_118780


namespace solve_for_y_l118_118002

variable {b c y : Real}

theorem solve_for_y (h : b > c) (h_eq : y^2 + c^2 = (b - y)^2) : y = (b^2 - c^2) / (2 * b) := 
sorry

end solve_for_y_l118_118002


namespace gcd_polynomial_l118_118655

theorem gcd_polynomial (b : ℤ) (h : 1729 ∣ b) : Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := 
by
  sorry

end gcd_polynomial_l118_118655


namespace quadratic_completes_square_l118_118426

theorem quadratic_completes_square (b c : ℤ) :
  (∃ b c : ℤ, (∀ x : ℤ, x^2 - 12 * x + 49 = (x + b)^2 + c) ∧ b + c = 7) :=
sorry

end quadratic_completes_square_l118_118426


namespace ceil_square_count_ceil_x_eq_15_l118_118405

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l118_118405


namespace find_weekday_rate_l118_118972

-- Definitions of given conditions
def num_people : ℕ := 6
def days_weekdays : ℕ := 2
def days_weekend : ℕ := 2
def weekend_rate : ℕ := 540
def payment_per_person : ℕ := 320

-- Theorem to prove the weekday rental rate
theorem find_weekday_rate (W : ℕ) :
  (num_people * payment_per_person) = (days_weekdays * W) + (days_weekend * weekend_rate) →
  W = 420 :=
by 
  intros h
  sorry

end find_weekday_rate_l118_118972


namespace distance_MN_is_2R_l118_118080

-- Definitions for the problem conditions
variable (R : ℝ) (A B C M N : ℝ) (alpha : ℝ)
variable (AC AB : ℝ)

-- Assumptions based on the problem statement
axiom circle_radius (r : ℝ) : r = R
axiom chord_length_AC (ch_AC : ℝ) : ch_AC = AC
axiom chord_length_AB (ch_AB : ℝ) : ch_AB = AB
axiom distance_M_to_AC (d_M_AC : ℝ) : d_M_AC = AC
axiom distance_N_to_AB (d_N_AB : ℝ) : d_N_AB = AB
axiom angle_BAC (ang_BAC : ℝ) : ang_BAC = alpha

-- To prove: the distance between M and N is 2R
theorem distance_MN_is_2R : |MN| = 2 * R := sorry

end distance_MN_is_2R_l118_118080


namespace circumscribed_radius_of_triangle_ABC_l118_118059

variable (A B C R : ℝ) (a b c : ℝ)

noncomputable def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ B = 2 * A ∧ C = 3 * A

noncomputable def side_length (A a : ℝ) : Prop :=
  a = 6

noncomputable def circumscribed_radius (A B C a R : ℝ) : Prop :=
  2 * R = a / (Real.sin (Real.pi * A / 180))

theorem circumscribed_radius_of_triangle_ABC:
  triangle_ABC A B C →
  side_length A a →
  circumscribed_radius A B C a R →
  R = 6 :=
by
  intros
  sorry

end circumscribed_radius_of_triangle_ABC_l118_118059


namespace cartons_being_considered_l118_118369

-- Definitions based on conditions
def packs_per_box : ℕ := 10
def boxes_per_carton : ℕ := 12
def price_per_pack : ℕ := 1
def total_cost : ℕ := 1440

-- Calculate total cost per carton
def cost_per_carton : ℕ := boxes_per_carton * packs_per_box * price_per_pack

-- Formulate the main theorem
theorem cartons_being_considered : (total_cost / cost_per_carton) = 12 :=
by
  -- The relevant steps would go here, but we're only providing the statement
  sorry

end cartons_being_considered_l118_118369


namespace find_constant_A_l118_118297

theorem find_constant_A :
  ∀ (x : ℝ)
  (A B C D : ℝ),
      (
        (1 : ℝ) / (x^4 - 20 * x^3 + 147 * x^2 - 490 * x + 588) = 
        (A / (x + 3)) + (B / (x - 4)) + (C / ((x - 4)^2)) + (D / (x - 7))
      ) →
      A = - (1 / 490) := 
by 
  intro x A B C D h
  sorry

end find_constant_A_l118_118297


namespace number_of_child_workers_l118_118262

-- Define the conditions
def number_of_male_workers : ℕ := 20
def number_of_female_workers : ℕ := 15
def wage_per_male : ℕ := 35
def wage_per_female : ℕ := 20
def wage_per_child : ℕ := 8
def average_wage : ℕ := 26

-- Define the proof goal
theorem number_of_child_workers (C : ℕ) : 
  ((number_of_male_workers * wage_per_male +
    number_of_female_workers * wage_per_female +
    C * wage_per_child) /
   (number_of_male_workers + number_of_female_workers + C) = average_wage) → 
  C = 5 :=
by 
  sorry

end number_of_child_workers_l118_118262


namespace parallel_lines_necessary_not_sufficient_l118_118759

theorem parallel_lines_necessary_not_sufficient {a : ℝ} 
  (h1 : ∀ x y : ℝ, a * x + (a + 2) * y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + a * y + 2 = 0) 
  (h3 : ∀ x y : ℝ, a * (1 * y + 2) = 1 * (a * y + 2)) : 
  (a = -1) -> (a = 2 ∨ a = -1 ∧ ¬(∀ b, a = b → a = -1)) :=
by
  -- proof goes here
  sorry

end parallel_lines_necessary_not_sufficient_l118_118759


namespace y_equals_x_l118_118535

theorem y_equals_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 1 + 1 / y) (h2 : y = 1 + 1 / x) : y = x :=
sorry

end y_equals_x_l118_118535


namespace tax_on_other_items_l118_118902

theorem tax_on_other_items (total_amount clothing_amount food_amount other_items_amount tax_on_clothing tax_on_food total_tax : ℝ) (tax_percent_other : ℝ) 
(h1 : clothing_amount = 0.5 * total_amount)
(h2 : food_amount = 0.2 * total_amount)
(h3 : other_items_amount = 0.3 * total_amount)
(h4 : tax_on_clothing = 0.04 * clothing_amount)
(h5 : tax_on_food = 0) 
(h6 : total_tax = 0.044 * total_amount)
: 
(tax_percent_other = 8) := 
by
  -- Definitions from the problem
  -- Define the total tax paid as the sum of taxes on clothing, food, and other items
  let tax_other_items : ℝ := tax_percent_other / 100 * other_items_amount
  
  -- Total tax equation
  have h7 : tax_on_clothing + tax_on_food + tax_other_items = total_tax
  sorry

  -- Substitution values into the given conditions and solving
  have h8 : tax_on_clothing + tax_percent_other / 100 * other_items_amount = total_tax
  sorry
  
  have h9 : 0.04 * 0.5 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h10 : 0.02 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h11 : tax_percent_other / 100 * 0.3 * total_amount = 0.024 * total_amount
  sorry

  have h12 : tax_percent_other / 100 * 0.3 = 0.024
  sorry

  have h13 : tax_percent_other / 100 = 0.08
  sorry

  have h14 : tax_percent_other = 8
  sorry

  exact h14

end tax_on_other_items_l118_118902


namespace true_or_false_is_true_l118_118612

theorem true_or_false_is_true (p q : Prop) (hp : p = true) (hq : q = false) : p ∨ q = true :=
by
  sorry

end true_or_false_is_true_l118_118612


namespace rainfall_second_week_value_l118_118826

-- Define the conditions
variables (rainfall_first_week rainfall_second_week : ℝ)
axiom condition1 : rainfall_first_week + rainfall_second_week = 30
axiom condition2 : rainfall_second_week = 1.5 * rainfall_first_week

-- Define the theorem we want to prove
theorem rainfall_second_week_value : rainfall_second_week = 18 := by
  sorry

end rainfall_second_week_value_l118_118826


namespace smartphone_cost_decrease_l118_118523

theorem smartphone_cost_decrease :
  ∀ (cost2010 cost2020 : ℝ),
  cost2010 = 600 →
  cost2020 = 450 →
  ((cost2010 - cost2020) / cost2010) * 100 = 25 :=
by
  intros cost2010 cost2020 h1 h2
  sorry

end smartphone_cost_decrease_l118_118523


namespace find_N_l118_118693

-- Define the problem parameters
def certain_value : ℝ := 0
def x : ℝ := 10

-- Define the main statement to be proved
theorem find_N (N : ℝ) : 3 * x = (N - x) + certain_value → N = 40 :=
  by sorry

end find_N_l118_118693


namespace inequality_correct_l118_118540

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_correct_l118_118540


namespace rational_solutions_product_l118_118582

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem rational_solutions_product :
  ∀ c : ℕ, (c > 0) → (is_perfect_square (49 - 12 * c)) → (∃ a b : ℕ, a = 4 ∧ b = 2 ∧ a * b = 8) :=
by sorry

end rational_solutions_product_l118_118582


namespace ab_inequality_l118_118069

theorem ab_inequality
  {a b : ℝ}
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_b_sum : a + b = 2) :
  ∀ n : ℕ, 2 ≤ n → (a^n + 1) * (b^n + 1) ≥ 4 :=
by
  sorry

end ab_inequality_l118_118069


namespace four_letters_three_mailboxes_l118_118066

theorem four_letters_three_mailboxes : (3 ^ 4) = 81 :=
  by sorry

end four_letters_three_mailboxes_l118_118066


namespace perpendicular_x_intercept_l118_118291

theorem perpendicular_x_intercept (x : ℝ) :
  (∃ y : ℝ, 2 * x + 3 * y = 9) ∧ (∃ y : ℝ, y = 5) → x = -10 / 3 :=
by sorry -- Proof omitted

end perpendicular_x_intercept_l118_118291


namespace even_k_l118_118651

theorem even_k :
  ∀ (a b n k : ℕ),
  1 ≤ a → 1 ≤ b → 0 < n →
  2^n - 1 = a * b →
  (a * b + a - b - 1) % 2^k = 0 →
  (a * b + a - b - 1) % 2^(k+1) ≠ 0 →
  Even k :=
by
  intros a b n k ha hb hn h1 h2 h3
  sorry

end even_k_l118_118651


namespace fraction_of_kiwis_l118_118073

theorem fraction_of_kiwis (total_fruits : ℕ) (num_strawberries : ℕ) (h₁ : total_fruits = 78) (h₂ : num_strawberries = 52) :
  (total_fruits - num_strawberries) / total_fruits = 1 / 3 :=
by
  -- proof to be provided, this is just the statement
  sorry

end fraction_of_kiwis_l118_118073


namespace sandwich_cost_90_cents_l118_118704

theorem sandwich_cost_90_cents :
  let cost_bread := 0.15
  let cost_ham := 0.25
  let cost_cheese := 0.35
  (2 * cost_bread + cost_ham + cost_cheese) * 100 = 90 := 
by
  sorry

end sandwich_cost_90_cents_l118_118704


namespace billy_can_play_l118_118864

-- Define the conditions
def total_songs : ℕ := 52
def songs_to_learn : ℕ := 28

-- Define the statement to be proved
theorem billy_can_play : total_songs - songs_to_learn = 24 := by
  -- Proof goes here
  sorry

end billy_can_play_l118_118864


namespace find_angle_between_planes_l118_118322

noncomputable def angle_between_planes (α β : ℝ) : ℝ := Real.arcsin ((Real.sqrt 6 + 1) / 5)

theorem find_angle_between_planes (α β : ℝ) (h : α = β) : 
  (∃ (cube : Type) (A B C D A₁ B₁ C₁ D₁ : cube),
    α = Real.arcsin ((Real.sqrt 6 - 1) / 5) ∨ α = Real.arcsin ((Real.sqrt 6 + 1) / 5)) 
    :=
sorry

end find_angle_between_planes_l118_118322


namespace simplify_expression_l118_118304

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 :=
sorry

end simplify_expression_l118_118304


namespace semicircle_parametric_equation_correct_l118_118940

-- Define the conditions of the problem in terms of Lean definitions and propositions.

def semicircle_parametric_equation : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (Real.pi / 2) →
    ∃ α : ℝ, α = 2 * θ ∧ 0 ≤ α ∧ α ≤ Real.pi ∧
    (∃ (x y : ℝ), x = 1 + Real.cos α ∧ y = Real.sin α)

-- Statement that we will prove
theorem semicircle_parametric_equation_correct : semicircle_parametric_equation :=
  sorry

end semicircle_parametric_equation_correct_l118_118940


namespace smallest_possible_bob_number_l118_118984

theorem smallest_possible_bob_number : 
  let alices_number := 60
  let bobs_smallest_number := 30
  ∃ (bob_number : ℕ), (∀ p : ℕ, Prime p → p ∣ alices_number → p ∣ bob_number) ∧ bob_number = bobs_smallest_number :=
by
  sorry

end smallest_possible_bob_number_l118_118984


namespace steps_to_get_down_empire_state_building_l118_118810

theorem steps_to_get_down_empire_state_building (total_steps : ℕ) (steps_building_to_garden : ℕ) (steps_to_madison_square : ℕ) :
  total_steps = 991 -> steps_building_to_garden = 315 -> steps_to_madison_square = total_steps - steps_building_to_garden -> steps_to_madison_square = 676 :=
by
  intros
  subst_vars
  sorry

end steps_to_get_down_empire_state_building_l118_118810


namespace midpoint_x_sum_l118_118946

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end midpoint_x_sum_l118_118946


namespace sufficient_condition_l118_118515

theorem sufficient_condition (m : ℝ) (x : ℝ) : -3 < m ∧ m < 1 → ((m - 1) * x^2 + (m - 1) * x - 1 < 0) :=
by
  sorry

end sufficient_condition_l118_118515


namespace find_f_seven_l118_118088

theorem find_f_seven 
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 3) = x^2 - 2 * x + 3) :
  f 7 = 3 := 
sorry

end find_f_seven_l118_118088


namespace parallel_lines_condition_l118_118398

theorem parallel_lines_condition (k_1 k_2 : ℝ) :
  (k_1 = k_2) ↔ (∀ x y : ℝ, k_1 * x + y + 1 = 0 → k_2 * x + y - 1 = 0) :=
sorry

end parallel_lines_condition_l118_118398


namespace graph_single_point_c_eq_7_l118_118027

theorem graph_single_point_c_eq_7 (x y : ℝ) (c : ℝ) :
  (∃ p : ℝ × ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + c = 0 ↔ (x, y) = p) →
  c = 7 :=
by
  sorry

end graph_single_point_c_eq_7_l118_118027


namespace move_up_4_units_l118_118728

-- Define the given points M and N
def M : ℝ × ℝ := (-1, -1)
def N : ℝ × ℝ := (-1, 3)

-- State the theorem to be proved
theorem move_up_4_units (M N : ℝ × ℝ) :
  (M = (-1, -1)) → (N = (-1, 3)) → (N = (M.1, M.2 + 4)) :=
by
  intros hM hN
  rw [hM, hN]
  sorry

end move_up_4_units_l118_118728


namespace intersection_points_circle_l118_118686

-- Defining the two lines based on the parameter u
def line1 (u : ℝ) (x y : ℝ) : Prop := 2 * u * x - 3 * y - 2 * u = 0
def line2 (u : ℝ) (x y : ℝ) : Prop := x - 3 * u * y + 2 = 0

-- Proof statement that shows the intersection points lie on a circle
theorem intersection_points_circle (u x y : ℝ) :
  line1 u x y → line2 u x y → (x - 1)^2 + y^2 = 1 :=
by {
  -- This completes the proof statement, but leaves implementation as exercise
  sorry
}

end intersection_points_circle_l118_118686


namespace additional_rows_added_l118_118666

theorem additional_rows_added
  (initial_tiles : ℕ) (initial_rows : ℕ) (initial_columns : ℕ) (new_columns : ℕ) (new_rows : ℕ)
  (h1 : initial_tiles = 48)
  (h2 : initial_rows = 6)
  (h3 : initial_columns = initial_tiles / initial_rows)
  (h4 : new_columns = initial_columns - 2)
  (h5 : new_rows = initial_tiles / new_columns) :
  new_rows - initial_rows = 2 := by sorry

end additional_rows_added_l118_118666


namespace solve_for_m_l118_118928

open Real

theorem solve_for_m (a b m : ℝ)
  (h1 : (1/2)^a = m)
  (h2 : 3^b = m)
  (h3 : 1/a - 1/b = 2) :
  m = sqrt 6 / 6 := 
  sorry

end solve_for_m_l118_118928


namespace arithmetic_seq_sum_ratio_l118_118056

theorem arithmetic_seq_sum_ratio (a1 d : ℝ) (S : ℕ → ℝ) 
  (hSn : ∀ n, S n = n * a1 + d * (n * (n - 1) / 2))
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 9 / S 6 = 2 :=
by
  sorry

end arithmetic_seq_sum_ratio_l118_118056


namespace chessboard_no_single_black_square_l118_118862

theorem chessboard_no_single_black_square :
  (∀ (repaint : (Fin 8) × Bool → (Fin 8) × Bool), False) :=
by 
  sorry

end chessboard_no_single_black_square_l118_118862


namespace similar_triangles_same_heights_ratio_l118_118200

theorem similar_triangles_same_heights_ratio (h1 h2 : ℝ) 
  (sim_ratio : h1 / h2 = 1 / 4) : h1 / h2 = 1 / 4 :=
by
  sorry

end similar_triangles_same_heights_ratio_l118_118200


namespace binom_divisible_by_prime_l118_118711

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h1 : 1 ≤ k) (h2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
sorry

end binom_divisible_by_prime_l118_118711


namespace multiplication_factor_l118_118269

theorem multiplication_factor
  (n : ℕ) (avg_orig avg_new : ℝ) (F : ℝ)
  (H1 : n = 7)
  (H2 : avg_orig = 24)
  (H3 : avg_new = 120)
  (H4 : (n * avg_new) = F * (n * avg_orig)) :
  F = 5 :=
by {
  sorry
}

end multiplication_factor_l118_118269


namespace ratio_of_shares_l118_118597

-- Definitions
variable (A B C : ℝ)   -- Representing the shares of a, b, and c
variable (x : ℝ)       -- Fraction

-- Conditions
axiom h1 : A = 80
axiom h2 : A + B + C = 200
axiom h3 : A = x * (B + C)
axiom h4 : B = (6 / 9) * (A + C)

-- Statement to prove
theorem ratio_of_shares : A / (B + C) = 2 / 3 :=
by sorry

end ratio_of_shares_l118_118597


namespace gcd_123456_789012_l118_118807

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := sorry

end gcd_123456_789012_l118_118807


namespace sum_of_circle_center_coordinates_l118_118746

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

end sum_of_circle_center_coordinates_l118_118746


namespace harrys_mothers_age_l118_118378

theorem harrys_mothers_age 
  (h : ℕ)  -- Harry's age
  (f : ℕ)  -- Father's age
  (m : ℕ)  -- Mother's age
  (h_age : h = 50)
  (f_age : f = h + 24)
  (m_age : m = f - h / 25) 
  : (m - h = 22) := 
by
  sorry

end harrys_mothers_age_l118_118378
