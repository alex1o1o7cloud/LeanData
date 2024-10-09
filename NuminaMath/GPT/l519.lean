import Mathlib

namespace algebraic_expression_value_l519_51939

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 6 - Real.sqrt 2) : 2 * x^2 + 4 * Real.sqrt 2 * x = 8 :=
sorry

end algebraic_expression_value_l519_51939


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l519_51969

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l519_51969


namespace q_value_l519_51908

noncomputable def prove_q (a b m p q : Real) :=
  (a * b = 5) → 
  (b + 1/a) * (a + 1/b) = q →
  q = 36/5

theorem q_value (a b : ℝ) (h_roots : a * b = 5) : (b + 1/a) * (a + 1/b) = 36 / 5 :=
by 
  sorry

end q_value_l519_51908


namespace maximum_enclosed_area_l519_51951

theorem maximum_enclosed_area (P : ℝ) (A : ℝ) : 
  P = 100 → (∃ l w : ℝ, P = 2 * l + 2 * w ∧ A = l * w) → A ≤ 625 :=
by
  sorry

end maximum_enclosed_area_l519_51951


namespace good_carrots_l519_51929

theorem good_carrots (haley_picked : ℕ) (mom_picked : ℕ) (bad_carrots : ℕ) :
  haley_picked = 39 → mom_picked = 38 → bad_carrots = 13 →
  (haley_picked + mom_picked - bad_carrots) = 64 :=
by
  sorry  -- Proof is omitted.

end good_carrots_l519_51929


namespace equilateral_triangle_perimeter_twice_side_area_l519_51946

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l519_51946


namespace equivalent_single_reduction_l519_51991

theorem equivalent_single_reduction :
  ∀ (P : ℝ), P * (1 - 0.25) * (1 - 0.20) = P * (1 - 0.40) :=
by
  intros P
  -- Proof will be skipped
  sorry

end equivalent_single_reduction_l519_51991


namespace find_number_l519_51948

theorem find_number : 
  (15^2 * 9^2) / x = 51.193820224719104 → x = 356 :=
by
  sorry

end find_number_l519_51948


namespace students_not_enrolled_l519_51956

-- Declare the conditions
def total_students : Nat := 79
def students_french : Nat := 41
def students_german : Nat := 22
def students_both : Nat := 9

-- Define the problem statement
theorem students_not_enrolled : total_students - (students_french + students_german - students_both) = 25 := by
  sorry

end students_not_enrolled_l519_51956


namespace complex_conjugate_x_l519_51923

theorem complex_conjugate_x (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3 * x + 2 : ℂ) * Complex.I = 4 + 20 * Complex.I) : x = -3 := sorry

end complex_conjugate_x_l519_51923


namespace tan_beta_is_neg3_l519_51907

theorem tan_beta_is_neg3 (α β : ℝ) (h1 : Real.tan α = -2) (h2 : Real.tan (α + β) = 1) : Real.tan β = -3 := 
sorry

end tan_beta_is_neg3_l519_51907


namespace tan_two_x_is_odd_l519_51919

noncomputable def tan_two_x (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_two_x_is_odd :
  ∀ x : ℝ,
  (∀ k : ℤ, x ≠ (k * Real.pi / 2) + (Real.pi / 4)) →
  tan_two_x (-x) = -tan_two_x x :=
by
  sorry

end tan_two_x_is_odd_l519_51919


namespace total_distance_travelled_l519_51975

theorem total_distance_travelled (D : ℝ) (h1 : (D / 2) / 30 + (D / 2) / 25 = 11) : D = 150 :=
sorry

end total_distance_travelled_l519_51975


namespace find_positive_integer_l519_51986

theorem find_positive_integer (n : ℕ) (h1 : n % 14 = 0) (h2 : 676 ≤ n ∧ n ≤ 702) : n = 700 :=
sorry

end find_positive_integer_l519_51986


namespace successive_product_l519_51912

theorem successive_product (n : ℤ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_l519_51912


namespace truncated_cone_sphere_radius_l519_51911

noncomputable def radius_of_sphere (r1 r2 h : ℝ) : ℝ := 
  (Real.sqrt (h^2 + (r1 - r2)^2)) / 2

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 h : ℝ), r1 = 20 → r2 = 6 → h = 15 → radius_of_sphere r1 r2 h = Real.sqrt 421 / 2 :=
by
  intros r1 r2 h h1 h2 h3
  simp [radius_of_sphere]
  rw [h1, h2, h3]
  sorry

end truncated_cone_sphere_radius_l519_51911


namespace sum_of_squares_greater_than_cubics_l519_51987

theorem sum_of_squares_greater_than_cubics (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a)
  : 
  (2 * (a + b + c) * (a^2 + b^2 + c^2)) / 3 > a^3 + b^3 + c^3 + a * b * c := 
by 
  sorry

end sum_of_squares_greater_than_cubics_l519_51987


namespace evaluate_expression_l519_51963

theorem evaluate_expression : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end evaluate_expression_l519_51963


namespace product_of_integers_l519_51999

theorem product_of_integers (a b : ℤ) (h_lcm : Int.lcm a b = 45) (h_gcd : Int.gcd a b = 9) : a * b = 405 :=
by
  sorry

end product_of_integers_l519_51999


namespace evaluate_expression_l519_51949

theorem evaluate_expression :
  3000 * (3000 ^ 1500 + 3000 ^ 1500) = 2 * 3000 ^ 1501 :=
by sorry

end evaluate_expression_l519_51949


namespace area_of_annulus_l519_51909

variable {b c h : ℝ}
variable (hb : b > c)
variable (h2 : h^2 = b^2 - 2 * c^2)

theorem area_of_annulus (hb : b > c) (h2 : h^2 = b^2 - 2 * c^2) :
    π * (b^2 - c^2) = π * h^2 := by
  sorry

end area_of_annulus_l519_51909


namespace ellie_oil_needs_l519_51985

def oil_per_wheel : ℕ := 10
def number_of_wheels : ℕ := 2
def oil_for_rest : ℕ := 5
def total_oil_needed : ℕ := oil_per_wheel * number_of_wheels + oil_for_rest

theorem ellie_oil_needs : total_oil_needed = 25 := by
  sorry

end ellie_oil_needs_l519_51985


namespace simplify_log_expression_l519_51990

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l519_51990


namespace fraction_used_first_day_l519_51927

theorem fraction_used_first_day (x : ℝ) :
  let initial_supplies := 400
  let supplies_remaining_after_first_day := initial_supplies * (1 - x)
  let supplies_remaining_after_three_days := (2/5 : ℝ) * supplies_remaining_after_first_day
  supplies_remaining_after_three_days = 96 → 
  x = (2/5 : ℝ) :=
by
  intros
  sorry

end fraction_used_first_day_l519_51927


namespace initial_friends_count_l519_51989

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count_l519_51989


namespace maximize_sector_area_l519_51995

noncomputable def sector_radius_angle (r l α : ℝ) : Prop :=
  2 * r + l = 40 ∧ α = l / r

theorem maximize_sector_area :
  ∃ r α : ℝ, sector_radius_angle r 20 α ∧ r = 10 ∧ α = 2 :=
by
  sorry

end maximize_sector_area_l519_51995


namespace max_angle_B_l519_51955

-- We define the necessary terms to state our problem
variables {A B C : Real} -- The angles of triangle ABC
variables {cot_A cot_B cot_C : Real} -- The cotangents of angles A, B, and C

-- The main theorem stating that given the conditions the maximum value of angle B is pi/3
theorem max_angle_B (h1 : cot_B = (cot_A + cot_C) / 2) (h2 : A + B + C = Real.pi) :
  B ≤ Real.pi / 3 := by
  sorry

end max_angle_B_l519_51955


namespace relationship_among_a_b_c_l519_51906

noncomputable def a : ℝ := 0.99 ^ (1.01 : ℝ)
noncomputable def b : ℝ := 1.01 ^ (0.99 : ℝ)
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l519_51906


namespace overlapping_region_area_l519_51900

noncomputable def radius : ℝ := 15
noncomputable def central_angle_radians : ℝ := Real.pi / 2
noncomputable def area_of_sector : ℝ := (1 / 4) * Real.pi * (radius^2)
noncomputable def side_length_equilateral_triangle : ℝ := radius
noncomputable def area_of_equilateral_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_equilateral_triangle^2)
noncomputable def overlapping_area : ℝ := 2 * area_of_sector - area_of_equilateral_triangle

theorem overlapping_region_area :
  overlapping_area = 112.5 * Real.pi - 56.25 * Real.sqrt 3 :=
by
  sorry
 
end overlapping_region_area_l519_51900


namespace expand_expression_l519_51954

theorem expand_expression (y : ℝ) : (7 * y + 12) * 3 * y = 21 * y ^ 2 + 36 * y := by
  sorry

end expand_expression_l519_51954


namespace area_of_shaded_region_l519_51928

theorem area_of_shaded_region :
  let v1 := (0, 0)
  let v2 := (15, 0)
  let v3 := (45, 30)
  let v4 := (45, 45)
  let v5 := (30, 45)
  let v6 := (0, 15)
  let area_large_rectangle := 45 * 45
  let area_triangle1 := 1 / 2 * 15 * 15
  let area_triangle2 := 1 / 2 * 15 * 15
  let shaded_area := area_large_rectangle - (area_triangle1 + area_triangle2)
  shaded_area = 1800 :=
by
  sorry

end area_of_shaded_region_l519_51928


namespace square_diagonal_l519_51959

theorem square_diagonal (p : ℤ) (h : p = 28) : ∃ d : ℝ, d = 7 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l519_51959


namespace subtracting_five_equals_thirtyfive_l519_51973

variable (x : ℕ)

theorem subtracting_five_equals_thirtyfive (h : x - 5 = 35) : x / 5 = 8 :=
sorry

end subtracting_five_equals_thirtyfive_l519_51973


namespace election_winner_margin_l519_51937

theorem election_winner_margin (V : ℝ) 
    (hV: V = 3744 / 0.52) 
    (w_votes: ℝ := 3744) 
    (l_votes: ℝ := 0.48 * V) :
    w_votes - l_votes = 288 := by
  sorry

end election_winner_margin_l519_51937


namespace value_set_for_a_non_empty_proper_subsets_l519_51978

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

theorem value_set_for_a (M : Set ℝ) : 
  (∀ (a : ℝ), B a ⊆ A → a ∈ M) :=
sorry

theorem non_empty_proper_subsets (M : Set ℝ) :
  M = {0, 3, -3} →
  (∃ S : Set (Set ℝ), S = {{0}, {3}, {-3}, {0, 3}, {0, -3}, {3, -3}}) :=
sorry

end value_set_for_a_non_empty_proper_subsets_l519_51978


namespace polynomial_is_first_degree_l519_51970

theorem polynomial_is_first_degree (k m : ℝ) (h : (k - 1) = 0) : k = 1 :=
by
  sorry

end polynomial_is_first_degree_l519_51970


namespace infections_first_wave_l519_51996

theorem infections_first_wave (x : ℕ)
  (h1 : 4 * x * 14 = 21000) : x = 375 :=
  sorry

end infections_first_wave_l519_51996


namespace find_k_from_direction_vector_l519_51974

/-- Given points p1 and p2, the direction vector's k component
    is -3 when the x component is 3. -/
theorem find_k_from_direction_vector
  (p1 : ℤ × ℤ) (p2 : ℤ × ℤ)
  (h1 : p1 = (2, -1))
  (h2 : p2 = (-4, 5))
  (dv_x : ℤ) (dv_k : ℤ)
  (h3 : (dv_x, dv_k) = (3, -3)) :
  True :=
by
  sorry

end find_k_from_direction_vector_l519_51974


namespace find_a5_l519_51960

variables {a : ℕ → ℝ}  -- represent the arithmetic sequence

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom a3_a8_sum : a 3 + a 8 = 22
axiom a6_value : a 6 = 8
axiom arithmetic : is_arithmetic_sequence a

-- Target proof statement
theorem find_a5 (a : ℕ → ℝ) (arithmetic : is_arithmetic_sequence a) (a3_a8_sum : a 3 + a 8 = 22) (a6_value : a 6 = 8) : a 5 = 14 :=
by {
  sorry
}

end find_a5_l519_51960


namespace reduce_consumption_percentage_l519_51983

theorem reduce_consumption_percentage :
  ∀ (current_rate old_rate : ℝ), 
  current_rate = 20 → 
  old_rate = 16 → 
  ((current_rate - old_rate) / old_rate * 100) = 25 :=
by
  intros current_rate old_rate h_current h_old
  sorry

end reduce_consumption_percentage_l519_51983


namespace range_of_m_empty_solution_set_inequality_l519_51976

theorem range_of_m_empty_solution_set_inequality (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 ≥ 0 → false) ↔ -4 < m ∧ m < 0 := 
sorry

end range_of_m_empty_solution_set_inequality_l519_51976


namespace complex_expression_eq_l519_51905

open Real

theorem complex_expression_eq (p q : ℝ) (hpq : p ≠ q) :
  (sqrt ((p^4 + q^4)/(p^4 - p^2 * q^2) + (2 * q^2)/(p^2 - q^2)) * (p^3 - p * q^2) - 2 * q * sqrt p) /
  (sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)) = 
  sqrt (p^2 - q^2) / sqrt p := 
sorry

end complex_expression_eq_l519_51905


namespace second_divisor_l519_51931

theorem second_divisor (x : ℕ) (k q : ℤ) : 
  (197 % 13 = 2) → 
  (x > 13) → 
  (197 % x = 5) → 
  x = 16 :=
by sorry

end second_divisor_l519_51931


namespace inequality_solution_l519_51997

theorem inequality_solution (x : ℚ) : (3 * x - 5 ≥ 9 - 2 * x) → (x ≥ 14 / 5) :=
by
  sorry

end inequality_solution_l519_51997


namespace first_machine_defect_probability_l519_51933

/-- Probability that a randomly selected defective item was made by the first machine is 0.5 
given certain conditions. -/
theorem first_machine_defect_probability :
  let PFirstMachine := 0.4
  let PSecondMachine := 0.6
  let DefectRateFirstMachine := 0.03
  let DefectRateSecondMachine := 0.02
  let TotalDefectProbability := PFirstMachine * DefectRateFirstMachine + PSecondMachine * DefectRateSecondMachine
  let PDefectGivenFirstMachine := PFirstMachine * DefectRateFirstMachine / TotalDefectProbability
  PDefectGivenFirstMachine = 0.5 :=
by
  sorry

end first_machine_defect_probability_l519_51933


namespace problem_l519_51953

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem problem (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 :=
by
  sorry

end problem_l519_51953


namespace flower_shop_ratio_l519_51965

theorem flower_shop_ratio (V C T R : ℕ) 
(total_flowers : V + C + T + R > 0)
(tulips_ratio : T = V / 4)
(roses_tulips_equal : R = T)
(carnations_fraction : C = 2 / 3 * (V + T + R + C)) 
: V / C = 1 / 3 := 
by
  -- Proof omitted
  sorry

end flower_shop_ratio_l519_51965


namespace solve_quadratic_1_solve_quadratic_2_l519_51915

theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 - 7 * x - 1 = 0 ↔ 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) := 
by 
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 3)^2 = 10 * x - 15 ↔ 
  (x = 3 / 2 ∨ x = 4) := 
by 
  sorry

end solve_quadratic_1_solve_quadratic_2_l519_51915


namespace area_of_square_ABCD_l519_51901

theorem area_of_square_ABCD :
  (∃ (x y : ℝ), 2 * x + 2 * y = 40) →
  ∃ (s : ℝ), s = 20 ∧ s * s = 400 :=
by
  sorry

end area_of_square_ABCD_l519_51901


namespace product_of_digits_l519_51944

theorem product_of_digits (n A B : ℕ) (h1 : n % 6 = 0) (h2 : A + B = 12) (h3 : n = 10 * A + B) : 
  (A * B = 32 ∨ A * B = 36) :=
by 
  sorry

end product_of_digits_l519_51944


namespace compute_trig_expression_l519_51934

theorem compute_trig_expression : 
  (1 - 1 / (Real.cos (37 * Real.pi / 180))) *
  (1 + 1 / (Real.sin (53 * Real.pi / 180))) *
  (1 - 1 / (Real.sin (37 * Real.pi / 180))) *
  (1 + 1 / (Real.cos (53 * Real.pi / 180))) = 1 :=
sorry

end compute_trig_expression_l519_51934


namespace current_price_of_soda_l519_51981

theorem current_price_of_soda (C S : ℝ) (h1 : 1.25 * C = 15) (h2 : C + S = 16) : 1.5 * S = 6 :=
by
  sorry

end current_price_of_soda_l519_51981


namespace math_problem_solution_l519_51935

theorem math_problem_solution (pA : ℚ) (pB : ℚ)
  (hA : pA = 1/2) (hB : pB = 1/3) :
  let pNoSolve := (1 - pA) * (1 - pB)
  let pSolve := 1 - pNoSolve
  pNoSolve = 1/3 ∧ pSolve = 2/3 :=
by
  sorry

end math_problem_solution_l519_51935


namespace quadrilateral_area_l519_51971

def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (4, 3)
def vertex3 : ℝ × ℝ := (7, 1)
def vertex4 : ℝ × ℝ := (4, 6)

noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
       (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 7.5 :=
by
  sorry

end quadrilateral_area_l519_51971


namespace fraction_simplification_l519_51941

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (2 * x - 5) / (x ^ 2 - 1) + 3 / (1 - x) = - (x + 8) / (x ^ 2 - 1) :=
  sorry

end fraction_simplification_l519_51941


namespace find_a11_l519_51932

variable (a : ℕ → ℝ)

axiom geometric_seq (a : ℕ → ℝ) (r : ℝ) : ∀ n, a (n + 1) = a n * r

variable (r : ℝ)
variable (h3 : a 3 = 4)
variable (h7 : a 7 = 12)

theorem find_a11 : a 11 = 36 := by
  sorry

end find_a11_l519_51932


namespace find_perpendicular_line_l519_51977

theorem find_perpendicular_line (x y : ℝ) (h₁ : y = (1/2) * x + 1)
    (h₂ : (x, y) = (2, 0)) : y = -2 * x + 4 :=
sorry

end find_perpendicular_line_l519_51977


namespace trees_in_garden_l519_51992

theorem trees_in_garden (yard_length distance_between_trees : ℕ) (h1 : yard_length = 800) (h2 : distance_between_trees = 32) :
  ∃ n : ℕ, n = (yard_length / distance_between_trees) + 1 ∧ n = 26 :=
by
  sorry

end trees_in_garden_l519_51992


namespace trig_identity_l519_51967

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := 
by
  sorry

end trig_identity_l519_51967


namespace unique_providers_count_l519_51994

theorem unique_providers_count :
  let num_children := 4
  let num_providers := 25
  (∀ s : Fin num_children, s.val < num_providers)
  → num_providers * (num_providers - 1) * (num_providers - 2) * (num_providers - 3) = 303600
:= sorry

end unique_providers_count_l519_51994


namespace remainder_3203_4507_9929_mod_75_l519_51984

theorem remainder_3203_4507_9929_mod_75 :
  (3203 * 4507 * 9929) % 75 = 34 :=
by
  have h1 : 3203 % 75 = 53 := sorry
  have h2 : 4507 % 75 = 32 := sorry
  have h3 : 9929 % 75 = 29 := sorry
  -- complete the proof using modular arithmetic rules.
  sorry

end remainder_3203_4507_9929_mod_75_l519_51984


namespace value_of_a_l519_51920

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_a (a : ℝ) :
  (1 / log_base 2 a) + (1 / log_base 3 a) + (1 / log_base 4 a) + (1 / log_base 5 a) = 7 / 4 ↔
  a = 120 ^ (4 / 7) :=
by
  sorry

end value_of_a_l519_51920


namespace residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l519_51958

noncomputable def phi (n : ℕ) : ℕ := Nat.totient n

theorem residues_exponent (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d ∧ ∀ x ∈ S, x^d % p = 1 :=
by sorry

theorem residues_divides_p_minus_one (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d :=
by sorry
  
theorem primitive_roots_phi (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ∃ (S : Finset ℕ), S.card = phi (p-1) ∧ ∀ g ∈ S, IsPrimitiveRoot g p :=
by sorry

end residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l519_51958


namespace solve_speeds_ratio_l519_51916

noncomputable def speeds_ratio (v_A v_B : ℝ) : Prop :=
  v_A / v_B = 1 / 3

theorem solve_speeds_ratio (v_A v_B : ℝ) (h1 : ∃ t : ℝ, t = 1 ∧ v_A = 300 - v_B ∧ v_A = v_B ∧ v_B = 300) 
  (h2 : ∃ t : ℝ, t = 7 ∧ 7 * v_A = 300 - 7 * v_B ∧ 7 * v_A = 300 - v_B ∧ 7 * v_B = v_A): 
    speeds_ratio v_A v_B :=
sorry

end solve_speeds_ratio_l519_51916


namespace sum_of_roots_l519_51972

theorem sum_of_roots (z1 z2 : ℂ) (h : z1^2 + 5*z1 - 14 = 0 ∧ z2^2 + 5*z2 - 14 = 0) :
  z1 + z2 = -5 :=
sorry

end sum_of_roots_l519_51972


namespace drinking_problem_solution_l519_51922

def drinking_rate (name : String) (hours : ℕ) (total_liters : ℕ) : ℚ :=
  total_liters / hours

def total_wine_consumed_in_x_hours (x : ℚ) :=
  x * (
  drinking_rate "assistant1" 12 40 +
  drinking_rate "assistant2" 10 40 +
  drinking_rate "assistant3" 8 40
  )

theorem drinking_problem_solution : 
  (∃ x : ℚ, total_wine_consumed_in_x_hours x = 40) →
  ∃ x : ℚ, x = 120 / 37 :=
by 
  sorry

end drinking_problem_solution_l519_51922


namespace floor_sqrt_18_squared_eq_16_l519_51910

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end floor_sqrt_18_squared_eq_16_l519_51910


namespace prime_solution_exists_l519_51921

theorem prime_solution_exists :
  ∃ (p q r : ℕ), p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 = r^4) ∧ (p = 7) ∧ (q = 3) ∧ (r = 2) := 
by
  sorry

end prime_solution_exists_l519_51921


namespace range_of_x_l519_51982

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_x (x : ℝ) (h₀ : -1 < x ∧ x < 1) (h₁ : f 0 = 0) (h₂ : f (1 - x) + f (1 - x^2) < 0) :
  1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end range_of_x_l519_51982


namespace candy_distribution_problem_l519_51914

theorem candy_distribution_problem (n : ℕ) :
  (n - 1) * (n - 2) / 2 - 3 * (n/2 - 1) / 6 = n + 1 → n = 18 :=
sorry

end candy_distribution_problem_l519_51914


namespace nth_term_150_l519_51957

-- Conditions
def a : ℕ := 2
def d : ℕ := 5
def arithmetic_sequence (n : ℕ) : ℕ := a + (n - 1) * d

-- Question and corresponding answer proof
theorem nth_term_150 : arithmetic_sequence 150 = 747 := by
  sorry

end nth_term_150_l519_51957


namespace reciprocal_roots_l519_51942

theorem reciprocal_roots (a b : ℝ) (h : a ≠ 0) :
  ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + a = 0) ∧ (a * x2^2 + b * x2 + a = 0) → x1 = 1 / x2 ∧ x2 = 1 / x1 :=
by
  intros x1 x2 hroots
  have hsum : x1 + x2 = -b / a := by sorry
  have hprod : x1 * x2 = 1 := by sorry
  sorry

end reciprocal_roots_l519_51942


namespace James_vegetable_intake_in_third_week_l519_51966

noncomputable def third_week_vegetable_intake : ℝ :=
  let asparagus_per_day_first_week : ℝ := 0.25
  let broccoli_per_day_first_week : ℝ := 0.25
  let cauliflower_per_day_first_week : ℝ := 0.5

  let asparagus_per_day_second_week := 2 * asparagus_per_day_first_week
  let broccoli_per_day_second_week := 3 * broccoli_per_day_first_week
  let cauliflower_per_day_second_week := cauliflower_per_day_first_week * 1.75
  let spinach_per_day_second_week : ℝ := 0.5
  
  let daily_intake_second_week := asparagus_per_day_second_week +
                                  broccoli_per_day_second_week +
                                  cauliflower_per_day_second_week +
                                  spinach_per_day_second_week
  
  let kale_per_day_third_week : ℝ := 0.5
  let zucchini_per_day_third_week : ℝ := 0.15
  
  let daily_intake_third_week := asparagus_per_day_second_week +
                                 broccoli_per_day_second_week +
                                 cauliflower_per_day_second_week +
                                 spinach_per_day_second_week +
                                 kale_per_day_third_week +
                                 zucchini_per_day_third_week
  
  daily_intake_third_week * 7

theorem James_vegetable_intake_in_third_week : 
  third_week_vegetable_intake = 22.925 :=
  by
    sorry

end James_vegetable_intake_in_third_week_l519_51966


namespace dodecahedron_equilateral_triangles_l519_51924

-- Definitions reflecting the conditions
def vertices_of_dodecahedron := 20
def faces_of_dodecahedron := 12
def vertices_per_face := 5
def equilateral_triangles_per_face := 5

theorem dodecahedron_equilateral_triangles :
  (faces_of_dodecahedron * equilateral_triangles_per_face) = 60 := by
  sorry

end dodecahedron_equilateral_triangles_l519_51924


namespace mans_rate_in_still_water_l519_51940

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water_l519_51940


namespace triangle_property_l519_51926

theorem triangle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_perimeter : a + b + c = 12) (h_inradius : 2 * (a + b + c) = 24) :
    ¬((a^2 + b^2 = c^2) ∨ (a^2 + b^2 > c^2) ∨ (c^2 > a^2 + b^2)) := 
sorry

end triangle_property_l519_51926


namespace amoeba_count_after_week_l519_51947

-- Definition of the initial conditions
def amoeba_splits_daily (n : ℕ) : ℕ := 2^n

-- Theorem statement translating the problem to Lean
theorem amoeba_count_after_week : amoeba_splits_daily 7 = 128 :=
by
  sorry

end amoeba_count_after_week_l519_51947


namespace rank_from_left_l519_51962

theorem rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h_total : total_students = 31) (h_right : rank_from_right = 21) : 
  rank_from_left = 11 := by
  sorry

end rank_from_left_l519_51962


namespace factorization_correct_l519_51904

theorem factorization_correct : 
  ¬(∃ x : ℝ, -x^2 + 4 * x = -x * (x + 4)) ∧
  ¬(∃ x y: ℝ, x^2 + x * y + x = x * (x + y)) ∧
  (∀ x y: ℝ, x * (x - y) + y * (y - x) = (x - y)^2) ∧
  ¬(∃ x : ℝ, x^2 - 4 * x + 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correct_l519_51904


namespace inequality_amgm_l519_51952

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l519_51952


namespace smallest_positive_multiple_l519_51993

theorem smallest_positive_multiple (a : ℕ) :
  (37 * a) % 97 = 7 → 37 * a = 481 :=
sorry

end smallest_positive_multiple_l519_51993


namespace slope_of_perpendicular_line_l519_51950

-- Define what it means to be the slope of a line in a certain form
def slope_of_line (a b c : ℝ) (m : ℝ) : Prop :=
  b ≠ 0 ∧ m = -a / b

-- Define what it means for two slopes to be perpendicular
def are_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Given conditions
def given_line : Prop := slope_of_line 4 5 20 (-4 / 5)

-- The theorem to be proved
theorem slope_of_perpendicular_line : ∃ m : ℝ, given_line ∧ are_perpendicular_slopes (-4 / 5) m ∧ m = 5 / 4 :=
  sorry

end slope_of_perpendicular_line_l519_51950


namespace man_speed_in_still_water_l519_51930

theorem man_speed_in_still_water (c_speed : ℝ) (distance_m : ℝ) (time_sec : ℝ) (downstream_distance_km : ℝ) (downstream_time_hr : ℝ) :
    c_speed = 3 →
    distance_m = 15 →
    time_sec = 2.9997600191984644 →
    downstream_distance_km = distance_m / 1000 →
    downstream_time_hr = time_sec / 3600 →
    (downstream_distance_km / downstream_time_hr) - c_speed = 15 :=
by
  intros hc hd ht hdownstream_distance hdownstream_time 
  sorry

end man_speed_in_still_water_l519_51930


namespace maximize_S_n_decreasing_arithmetic_sequence_l519_51925

theorem maximize_S_n_decreasing_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d < 0)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (h4 : S 5 = S 10) :
  S 7 = S 8 :=
sorry

end maximize_S_n_decreasing_arithmetic_sequence_l519_51925


namespace function_symmetric_about_point_l519_51902

theorem function_symmetric_about_point :
  ∃ x₀ y₀, (x₀, y₀) = (Real.pi / 3, 0) ∧ ∀ x y, y = Real.sin (2 * x + Real.pi / 3) →
    (Real.sin (2 * (2 * x₀ - x) + Real.pi / 3) = y) :=
sorry

end function_symmetric_about_point_l519_51902


namespace find_integer_x_l519_51998

theorem find_integer_x (x y : ℕ) (h_gt : x > y) (h_gt_zero : y > 0) (h_eq : x + y + x * y = 99) : x = 49 :=
sorry

end find_integer_x_l519_51998


namespace sufficient_not_necessary_condition_l519_51943

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 + t * x - t

-- The proof statement about the condition for roots
theorem sufficient_not_necessary_condition (t : ℝ) :
  (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) :=
sorry

end sufficient_not_necessary_condition_l519_51943


namespace factory_produces_6500_toys_per_week_l519_51961

theorem factory_produces_6500_toys_per_week
    (days_per_week : ℕ)
    (toys_per_day : ℕ)
    (h1 : days_per_week = 5)
    (h2 : toys_per_day = 1300) :
    days_per_week * toys_per_day = 6500 := 
by 
  sorry

end factory_produces_6500_toys_per_week_l519_51961


namespace problem_statement_l519_51988

-- Definitions and conditions
def f (x : ℝ) : ℝ := x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

-- Given the specific condition
def f_symmetric_about_1 : Prop := is_symmetric_about f 1

-- We need to prove that this implies g(x) = 3x - 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f_symmetric_about_1 → ∀ x, g x = 3 * x - 2 := 
by
  intro h
  sorry -- Detailed proof is omitted

end problem_statement_l519_51988


namespace percentage_second_division_l519_51936

theorem percentage_second_division (total_students : ℕ) 
                                  (first_division_percentage : ℝ) 
                                  (just_passed : ℕ) 
                                  (all_students_passed : total_students = 300) 
                                  (percentage_first_division : first_division_percentage = 26) 
                                  (students_just_passed : just_passed = 60) : 
  (26 / 100 * 300 + (total_students - (26 / 100 * 300 + 60)) + 60) = 300 → 
  ((total_students - (26 / 100 * 300 + 60)) / total_students * 100) = 54 := 
by 
  sorry

end percentage_second_division_l519_51936


namespace mark_owes_linda_l519_51964

-- Define the payment per room and the number of rooms painted
def payment_per_room := (13 : ℚ) / 3
def rooms_painted := (8 : ℚ) / 5

-- State the theorem and the proof
theorem mark_owes_linda : (payment_per_room * rooms_painted) = (104 : ℚ) / 15 := by
  sorry

end mark_owes_linda_l519_51964


namespace find_y_l519_51918

theorem find_y 
  (x y : ℝ) 
  (h1 : (6 : ℝ) = (1/2 : ℝ) * x) 
  (h2 : y = (1/2 : ℝ) * 10) 
  (h3 : x * y = 60) 
: y = 5 := 
by 
  sorry

end find_y_l519_51918


namespace minimize_expression_l519_51945

theorem minimize_expression : 
  let a := -1
  let b := -0.5
  (a + b) ≤ (a - b) ∧ (a + b) ≤ (a * b) ∧ (a + b) ≤ (a / b) := by
  let a := -1
  let b := -0.5
  sorry

end minimize_expression_l519_51945


namespace china_math_olympiad_34_2023_l519_51903

-- Defining the problem conditions and verifying the minimum and maximum values of S.
theorem china_math_olympiad_34_2023 {a b c d e : ℝ}
  (h1 : a ≥ -1)
  (h2 : b ≥ -1)
  (h3 : c ≥ -1)
  (h4 : d ≥ -1)
  (h5 : e ≥ -1)
  (h6 : a + b + c + d + e = 5) :
  (-512 ≤ (a + b) * (b + c) * (c + d) * (d + e) * (e + a)) ∧
  ((a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288) :=
sorry

end china_math_olympiad_34_2023_l519_51903


namespace misha_card_numbers_l519_51979

-- Define the context for digits
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Define conditions
def proper_fraction (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a < b

-- Original problem statement rewritten for Lean
theorem misha_card_numbers (L O M N S B : ℕ) :
  is_digit L → is_digit O → is_digit M → is_digit N → is_digit S → is_digit B →
  proper_fraction O M → proper_fraction O S →
  L + O / M + O + N + O / S = 10 + B :=
sorry

end misha_card_numbers_l519_51979


namespace student_count_estimate_l519_51913

theorem student_count_estimate 
  (n : Nat) 
  (h1 : 80 ≤ n) 
  (h2 : 100 ≤ n) 
  (h3 : 20 * n = 8000) : 
  n = 400 := 
by 
  sorry

end student_count_estimate_l519_51913


namespace alloy_chromium_l519_51938

variable (x : ℝ)

theorem alloy_chromium (h : 0.15 * 15 + 0.08 * x = 0.101 * (15 + x)) : x = 35 := by
  sorry

end alloy_chromium_l519_51938


namespace find_number_l519_51968

theorem find_number (x : ℕ) (h : 24 * x = 2376) : x = 99 :=
by
  sorry

end find_number_l519_51968


namespace range_of_a_l519_51980

open Complex Real

theorem range_of_a (a : ℝ) (h : abs (1 + a * Complex.I) ≤ 2) : a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_a_l519_51980


namespace a_plus_b_is_18_over_5_l519_51917

noncomputable def a_b_sum (a b : ℚ) : Prop :=
  (∃ (x y : ℚ), x = 2 ∧ y = 3 ∧ x = (1 / 3) * y + a ∧ y = (1 / 5) * x + b) → a + b = (18 / 5)

-- No proof provided, just the statement.
theorem a_plus_b_is_18_over_5 (a b : ℚ) : a_b_sum a b :=
sorry

end a_plus_b_is_18_over_5_l519_51917
