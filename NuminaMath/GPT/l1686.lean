import Mathlib

namespace odd_and_monotonically_decreasing_l1686_168613

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem odd_and_monotonically_decreasing :
  is_odd (fun x : ℝ => -x^3) ∧ is_monotonically_decreasing (fun x : ℝ => -x^3) :=
by
  sorry

end odd_and_monotonically_decreasing_l1686_168613


namespace rate_of_current_l1686_168691

theorem rate_of_current
  (D U R : ℝ)
  (hD : D = 45)
  (hU : U = 23)
  (hR : R = 34)
  : (D - R = 11) ∧ (R - U = 11) :=
by
  sorry

end rate_of_current_l1686_168691


namespace solve_linear_equation_l1686_168670

theorem solve_linear_equation (a b x : ℝ) (h : a - b = 0) (ha : a ≠ 0) : ax + b = 0 ↔ x = -1 :=
by sorry

end solve_linear_equation_l1686_168670


namespace sqrt_170569_sqrt_175561_l1686_168604

theorem sqrt_170569 : Nat.sqrt 170569 = 413 := 
by 
  sorry 

theorem sqrt_175561 : Nat.sqrt 175561 = 419 := 
by 
  sorry

end sqrt_170569_sqrt_175561_l1686_168604


namespace part1_part2_part3_l1686_168698

noncomputable def p1_cost (t : ℕ) : ℕ := 
  if t <= 150 then 58 else 58 + 25 * (t - 150) / 100

noncomputable def p2_cost (t : ℕ) (a : ℕ) : ℕ := 
  if t <= 350 then 88 else 88 + a * (t - 350)

-- Part 1: Prove the costs for 260 minutes
theorem part1 : p1_cost 260 = 855 / 10 ∧ p2_cost 260 30 = 88 :=
by 
  sorry

-- Part 2: Prove the existence of t for given a
theorem part2 (t : ℕ) : (a = 30) → (∃ t, p1_cost t = p2_cost t a) :=
by 
  sorry

-- Part 3: Prove a=45 and the range for which Plan 1 is cheaper
theorem part3 : 
  (a = 45) ↔ (p1_cost 450 = p2_cost 450 a) ∧ (∀ t, (0 ≤ t ∧ t < 270) ∨ (t > 450) → p1_cost t < p2_cost t 45 ) :=
by 
  sorry

end part1_part2_part3_l1686_168698


namespace max_abc_value_l1686_168693

theorem max_abc_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_equation : a * b + c = (a + c) * (b + c))
  (h_sum : a + b + c = 2) : abc ≤ 1/27 :=
by sorry

end max_abc_value_l1686_168693


namespace train_complete_time_l1686_168657

noncomputable def train_time_proof : Prop :=
  ∃ (t_x : ℕ) (v_x : ℝ) (v_y : ℝ),
    v_y = 140 / 3 ∧
    t_x = 140 / v_x ∧
    (∃ t : ℝ, 
      t * v_x = 60.00000000000001 ∧
      t * v_y = 140 - 60.00000000000001) ∧
    t_x = 4

theorem train_complete_time : train_time_proof := by
  sorry

end train_complete_time_l1686_168657


namespace area_of_sector_l1686_168649

theorem area_of_sector
  (θ : ℝ) (l : ℝ) (r : ℝ := l / θ)
  (h1 : θ = 2)
  (h2 : l = 4) :
  1 / 2 * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l1686_168649


namespace tan_theta_value_l1686_168627

theorem tan_theta_value (θ k : ℝ) 
  (h1 : Real.sin θ = (k + 1) / (k - 3)) 
  (h2 : Real.cos θ = (k - 1) / (k - 3)) 
  (h3 : (Real.sin θ ≠ 0) ∧ (Real.cos θ ≠ 0)) : 
  Real.tan θ = 3 / 4 := 
sorry

end tan_theta_value_l1686_168627


namespace part_a_part_b_part_c_l1686_168632

variable (p : ℕ) (k : ℕ)

theorem part_a (hp : Prime p) (h : p = 4 * k + 1) :
  ∃ x : ℤ, (x^2 + 1) % p = 0 :=
by
  sorry

theorem part_b (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)) :
  ∃ (r1 r2 s1 s2 : ℕ), (r1 * x + s1) % p = (r2 * x + s2) % p :=
by
  sorry

theorem part_c (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)):
  p = (Int.ofNat (r1 - r2))^2 + (Int.ofNat (s1 - s2))^2 :=
by
  sorry

end part_a_part_b_part_c_l1686_168632


namespace find_prime_powers_l1686_168692

open Nat

theorem find_prime_powers (p x y : ℕ) (hp : p.Prime) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 ↔
  (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
sorry

end find_prime_powers_l1686_168692


namespace cyclic_quadrilateral_eq_l1686_168643

theorem cyclic_quadrilateral_eq (A B C D : ℝ) (AB AD BC DC : ℝ)
  (h1 : AB = AD) (h2 : based_on_laws_of_cosines) : AC ^ 2 = BC * DC + AB ^ 2 :=
sorry

end cyclic_quadrilateral_eq_l1686_168643


namespace method_is_systematic_sampling_l1686_168603

-- Define the conditions
def rows : ℕ := 25
def seats_per_row : ℕ := 20
def filled_auditorium : Prop := True
def seat_numbered_15_sampled : Prop := True
def interval : ℕ := 20

-- Define the concept of systematic sampling
def systematic_sampling (rows seats_per_row interval : ℕ) : Prop :=
  (rows > 0 ∧ seats_per_row > 0 ∧ interval > 0 ∧ (interval = seats_per_row))

-- State the problem in terms of proving that the sampling method is systematic
theorem method_is_systematic_sampling :
  filled_auditorium → seat_numbered_15_sampled → systematic_sampling rows seats_per_row interval :=
by
  intros h1 h2
  -- Assume that the proof goes here
  sorry

end method_is_systematic_sampling_l1686_168603


namespace ploughing_solution_l1686_168685

/-- Definition representing the problem of A and B ploughing the field together and alone --/
noncomputable def ploughing_problem : Prop :=
  ∃ (A : ℝ), (A > 0) ∧ (1 / A + 1 / 30 = 1 / 10) ∧ A = 15

theorem ploughing_solution : ploughing_problem :=
  by sorry

end ploughing_solution_l1686_168685


namespace Q_2_plus_Q_neg2_l1686_168636

variable {k : ℝ}

noncomputable def Q (x : ℝ) : ℝ := 0 -- Placeholder definition, real polynomial will be defined in proof.

theorem Q_2_plus_Q_neg2 (hQ0 : Q 0 = 2 * k)
  (hQ1 : Q 1 = 3 * k)
  (hQ_minus1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 16 * k :=
sorry

end Q_2_plus_Q_neg2_l1686_168636


namespace factorize_polynomial_l1686_168600

def p (a b : ℝ) : ℝ := a^2 - b^2 + 2 * a + 1

theorem factorize_polynomial (a b : ℝ) : 
  p a b = (a + 1 + b) * (a + 1 - b) :=
by
  sorry

end factorize_polynomial_l1686_168600


namespace max_f_value_l1686_168659

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_f_value : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), f x₀ = 1 :=
by
  sorry

end max_f_value_l1686_168659


namespace height_of_circular_segment_l1686_168644

theorem height_of_circular_segment (d a : ℝ) (h : ℝ) :
  (h = (d - Real.sqrt (d^2 - a^2)) / 2) ↔ 
  ((a / 2)^2 + (d / 2 - h)^2 = (d / 2)^2) :=
sorry

end height_of_circular_segment_l1686_168644


namespace parabola_unique_intersection_x_axis_l1686_168683

theorem parabola_unique_intersection_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ ∀ y, y^2 - 6*y + m = 0 → y = x) → m = 9 :=
by
  sorry

end parabola_unique_intersection_x_axis_l1686_168683


namespace a_alone_days_l1686_168662

theorem a_alone_days 
  (B_days : ℕ)
  (B_days_eq : B_days = 8)
  (C_payment : ℝ)
  (C_payment_eq : C_payment = 450)
  (total_payment : ℝ)
  (total_payment_eq : total_payment = 3600)
  (combined_days : ℕ)
  (combined_days_eq : combined_days = 3)
  (combined_rate_eq : (1 / A + 1 / B_days + C = 1 / combined_days)) 
  (rate_proportion : (1 / A) / (1 / B_days) = 7 / 1) 
  : A = 56 :=
sorry

end a_alone_days_l1686_168662


namespace female_officers_count_l1686_168656

theorem female_officers_count
  (total_on_duty : ℕ)
  (on_duty_females : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 240)
  (h2 : on_duty_females = total_on_duty / 2)
  (h3 : on_duty_females = (40 * total_female_officers) / 100) : 
  total_female_officers = 300 := 
by
  sorry

end female_officers_count_l1686_168656


namespace certain_number_is_8000_l1686_168641

theorem certain_number_is_8000 (x : ℕ) (h : x / 10 - x / 2000 = 796) : x = 8000 :=
sorry

end certain_number_is_8000_l1686_168641


namespace proof_question_1_l1686_168615

noncomputable def question_1 (x : ℝ) : ℝ :=
  (Real.sin (2 * x) + 2 * (Real.sin x)^2) / (1 - Real.tan x)

theorem proof_question_1 :
  ∀ x : ℝ, (Real.cos (π / 4 + x) = 3 / 5) →
  (17 * π / 12 < x ∧ x < 7 * π / 4) →
  question_1 x = -9 / 20 :=
by
  intros x h1 h2
  sorry

end proof_question_1_l1686_168615


namespace ratio_A_to_B_l1686_168611

noncomputable def A_annual_income : ℝ := 436800.0000000001
noncomputable def B_increase_rate : ℝ := 0.12
noncomputable def C_monthly_income : ℝ := 13000

noncomputable def A_monthly_income : ℝ := A_annual_income / 12
noncomputable def B_monthly_income : ℝ := C_monthly_income + (B_increase_rate * C_monthly_income)

theorem ratio_A_to_B :
  ((A_monthly_income / 80) : ℝ) = 455 ∧
  ((B_monthly_income / 80) : ℝ) = 182 :=
by
  sorry

end ratio_A_to_B_l1686_168611


namespace log_10_7_eqn_l1686_168666

variables (p q : ℝ)
noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_10_7_eqn (h1 : log_base 4 5 = p) (h2 : log_base 5 7 = q) : 
  log_base 10 7 = (2 * p * q) / (2 * p + 1) :=
by 
  sorry

end log_10_7_eqn_l1686_168666


namespace meet_time_opposite_directions_catch_up_time_same_direction_l1686_168667

def length_of_track := 440
def speed_A := 5
def speed_B := 6

theorem meet_time_opposite_directions :
  (length_of_track / (speed_A + speed_B)) = 40 :=
by
  sorry

theorem catch_up_time_same_direction :
  (length_of_track / (speed_B - speed_A)) = 440 :=
by
  sorry

end meet_time_opposite_directions_catch_up_time_same_direction_l1686_168667


namespace cities_below_50000_l1686_168682

theorem cities_below_50000 (p1 p2 : ℝ) (h1 : p1 = 20) (h2: p2 = 65) :
  p1 + p2 = 85 := 
  by sorry

end cities_below_50000_l1686_168682


namespace parabola_focus_coincides_ellipse_focus_l1686_168655

theorem parabola_focus_coincides_ellipse_focus (p : ℝ) :
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ ∀ x y : ℝ, y^2 = 2 * p * x <-> x = p / 2)
  → p = 4 := 
by
  sorry 

end parabola_focus_coincides_ellipse_focus_l1686_168655


namespace expression_value_l1686_168631

theorem expression_value : 2013 * (2015 / 2014) + 2014 * (2016 / 2015) + (4029 / (2014 * 2015)) = 4029 :=
by
  sorry

end expression_value_l1686_168631


namespace rectangle_area_l1686_168619

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end rectangle_area_l1686_168619


namespace pyramid_prism_sum_l1686_168642

-- Definitions based on conditions
structure Prism :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

-- The initial cylindrical-prism object
noncomputable def initial_prism : Prism :=
  { vertices := 8,
    edges := 10,
    faces := 5 }

-- Structure for Pyramid Addition
structure PyramidAddition :=
  (new_vertices : ℕ)
  (new_edges : ℕ)
  (new_faces : ℕ)

noncomputable def pyramid_addition : PyramidAddition := 
  { new_vertices := 1,
    new_edges := 4,
    new_faces := 4 }

-- Function to add pyramid to the prism
noncomputable def add_pyramid (prism : Prism) (pyramid : PyramidAddition) : Prism :=
  { vertices := prism.vertices + pyramid.new_vertices,
    edges := prism.edges + pyramid.new_edges,
    faces := prism.faces - 1 + pyramid.new_faces }

-- The resulting prism after adding the pyramid
noncomputable def resulting_prism := add_pyramid initial_prism pyramid_addition

-- Proof problem statement
theorem pyramid_prism_sum : 
  resulting_prism.vertices + resulting_prism.edges + resulting_prism.faces = 31 :=
by sorry

end pyramid_prism_sum_l1686_168642


namespace total_wages_l1686_168622

theorem total_wages (A_days B_days : ℝ) (A_wages : ℝ) (W : ℝ) 
  (h1 : A_days = 10)
  (h2 : B_days = 15)
  (h3 : A_wages = 2100) :
  W = 3500 :=
by sorry

end total_wages_l1686_168622


namespace isosceles_base_length_l1686_168674

theorem isosceles_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter side_length base_length : ℕ), 
  equilateral_perimeter = 60 →  -- Condition: Perimeter of the equilateral triangle is 60
  isosceles_perimeter = 45 →    -- Condition: Perimeter of the isosceles triangle is 45
  side_length = equilateral_perimeter / 3 →   -- Condition: Each side of the equilateral triangle
  isosceles_perimeter = side_length + side_length + base_length  -- Condition: Perimeter relation in isosceles triangle
  → base_length = 5  -- Result: The base length of the isosceles triangle is 5
:= 
sorry

end isosceles_base_length_l1686_168674


namespace negation_proposition_l1686_168661

theorem negation_proposition (a b c : ℝ) : 
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) := 
by
  -- proof goes here
  sorry

end negation_proposition_l1686_168661


namespace count_similar_divisors_l1686_168625

def is_integrally_similar_divisible (a b c : ℕ) : Prop :=
  ∃ x y z : ℕ, a * c = b * z ∧
  x ≤ y ∧ y ≤ z ∧
  b = 2023 ∧ a * c = 2023^2

theorem count_similar_divisors (b : ℕ) (hb : b = 2023) :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (a c : ℕ), a ≤ b ∧ b ≤ c → is_integrally_similar_divisible a b c) :=
by
  sorry

end count_similar_divisors_l1686_168625


namespace lowest_price_for_butter_l1686_168606

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end lowest_price_for_butter_l1686_168606


namespace solve_for_x_l1686_168689

def f (x : ℝ) : ℝ := x^2 + x - 1

theorem solve_for_x (x : ℝ) (h : f x = 5) : x = 2 ∨ x = -3 := 
by {
  sorry
}

end solve_for_x_l1686_168689


namespace sum_of_roots_l1686_168675

theorem sum_of_roots {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h1 : c + d = -a) (h2 : c * d = b) (h3 : a + b = -c) (h4 : a * b = d) : 
    a + b + c + d = -2 := 
by
  sorry

end sum_of_roots_l1686_168675


namespace polynomial_is_perfect_cube_l1686_168676

theorem polynomial_is_perfect_cube (p q n : ℚ) :
  (∃ a : ℚ, x^3 + p * x^2 + q * x + n = (x + a)^3) ↔ (q = p^2 / 3 ∧ n = p^3 / 27) :=
by sorry

end polynomial_is_perfect_cube_l1686_168676


namespace largest_digit_M_l1686_168605

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l1686_168605


namespace fifth_power_last_digit_l1686_168646

theorem fifth_power_last_digit (n : ℕ) : 
  (n % 10)^5 % 10 = n % 10 :=
by sorry

end fifth_power_last_digit_l1686_168646


namespace remainder_when_divided_by_9_l1686_168637

variable (k : ℕ)

theorem remainder_when_divided_by_9 :
  (∃ k, k % 5 = 2 ∧ k % 6 = 3 ∧ k % 8 = 7 ∧ k < 100) →
  k % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l1686_168637


namespace total_pieces_of_gum_l1686_168660

def packages : ℕ := 12
def pieces_per_package : ℕ := 20

theorem total_pieces_of_gum : packages * pieces_per_package = 240 :=
by
  -- proof is skipped
  sorry

end total_pieces_of_gum_l1686_168660


namespace at_least_one_not_less_than_one_l1686_168668

theorem at_least_one_not_less_than_one (x : ℝ) (a b c : ℝ) 
  (ha : a = x^2 + 1/2) 
  (hb : b = 2 - x) 
  (hc : c = x^2 - x + 1) : 
  (1 ≤ a) ∨ (1 ≤ b) ∨ (1 ≤ c) := 
sorry

end at_least_one_not_less_than_one_l1686_168668


namespace proposition_false_at_4_l1686_168635

open Nat

def prop (n : ℕ) : Prop := sorry -- the actual proposition is not specified, so we use sorry

theorem proposition_false_at_4 :
  (∀ k : ℕ, k > 0 → (prop k → prop (k + 1))) →
  ¬ prop 5 →
  ¬ prop 4 :=
by
  intros h_induction h_proposition_false_at_5
  sorry

end proposition_false_at_4_l1686_168635


namespace negation_proposition_l1686_168684

theorem negation_proposition :
  ∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n :=
sorry

end negation_proposition_l1686_168684


namespace volume_relation_l1686_168653

theorem volume_relation 
  (r h : ℝ) 
  (heightC_eq_three_times_radiusD : h = 3 * r)
  (radiusC_eq_heightD : r = h)
  (volumeD_eq_three_times_volumeC : ∀ (π : ℝ), 3 * (π * h^2 * r) = π * r^2 * h) :
  3 = (3 : ℝ) := 
by
  sorry

end volume_relation_l1686_168653


namespace shaded_square_area_l1686_168628

noncomputable def Pythagorean_area (a b c : ℕ) (area_a area_b area_c : ℕ) : Prop :=
  area_a = a^2 ∧ area_b = b^2 ∧ area_c = c^2 ∧ a^2 + b^2 = c^2

theorem shaded_square_area 
  (area1 area2 area3 : ℕ)
  (area_unmarked : ℕ)
  (h1 : area1 = 5)
  (h2 : area2 = 8)
  (h3 : area3 = 32)
  (h_unmarked: area_unmarked = area2 + area3)
  (h_shaded : area1 + area_unmarked = 45) :
  area1 + area_unmarked = 45 :=
by
  exact h_shaded

end shaded_square_area_l1686_168628


namespace algebraic_expression_value_l1686_168607

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 7 = -6 :=
by
  sorry

end algebraic_expression_value_l1686_168607


namespace document_completion_time_l1686_168690

-- Define the typing rates for different typists
def fast_typist_rate := 1 / 4
def slow_typist_rate := 1 / 9
def additional_typist_rate := 1 / 4

-- Define the number of typists
def num_fast_typists := 2
def num_slow_typists := 3
def num_additional_typists := 2

-- Define the distraction time loss per typist every 30 minutes
def distraction_loss := 1 / 6

-- Define the combined rate without distractions
def combined_rate : ℚ :=
  (num_fast_typists * fast_typist_rate) +
  (num_slow_typists * slow_typist_rate) +
  (num_additional_typists * additional_typist_rate)

-- Define the distraction rate loss per hour (two distractions per hour)
def distraction_rate_loss_per_hour := 2 * distraction_loss

-- Define the effective combined rate considering distractions
def effective_combined_rate : ℚ := combined_rate - distraction_rate_loss_per_hour

-- Prove that the document is completed in 1 hour with the effective rate
theorem document_completion_time :
  effective_combined_rate = 1 :=
sorry

end document_completion_time_l1686_168690


namespace simplify_fraction_product_l1686_168620

theorem simplify_fraction_product :
  4 * (18 / 5) * (35 / -63) * (8 / 14) = - (32 / 7) :=
by sorry

end simplify_fraction_product_l1686_168620


namespace ordinary_eq_from_param_eq_l1686_168617

theorem ordinary_eq_from_param_eq (α : ℝ) :
  (∃ (x y : ℝ), x = 3 * Real.cos α + 1 ∧ y = - Real.cos α → x + 3 * y - 1 = 0 ∧ (-2 ≤ x ∧ x ≤ 4)) := 
sorry

end ordinary_eq_from_param_eq_l1686_168617


namespace investment_difference_l1686_168601

theorem investment_difference (x y z : ℕ) 
  (h1 : x + (x + y) + (x + 2 * y) = 9000)
  (h2 : (z / 9000) = (800 / 1800)) 
  (h3 : z = x + 2 * y) :
  y = 1000 := 
by
  -- omitted proof steps
  sorry

end investment_difference_l1686_168601


namespace cylinder_radius_exists_l1686_168648

theorem cylinder_radius_exists (r h : ℕ) (pr : r ≥ 1) :
  (π * ↑r ^ 2 * ↑h = 2 * π * ↑r * (↑h + ↑r)) ↔
  (r = 3 ∨ r = 4 ∨ r = 6) :=
by
  sorry

end cylinder_radius_exists_l1686_168648


namespace circle_through_focus_l1686_168652

open Real

-- Define the parabola as a set of points
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 - 3) ^ 2 = 8 * (P.1 - 2)

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 3)

-- Define the circle with center P and radius the distance from P to the y-axis
def is_tangent_circle (P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 + (P.2 - 3) ^ 2 = (C.1) ^ 2 + (C.2) ^ 2 ∧ C = (4, 3))

-- The main theorem
theorem circle_through_focus (P : ℝ × ℝ) 
  (hP_on_parabola : is_on_parabola P) 
  (hP_tangent_circle : is_tangent_circle P (4, 3)) :
  is_tangent_circle P (4, 3) :=
by sorry

end circle_through_focus_l1686_168652


namespace statistical_measure_mode_l1686_168658

theorem statistical_measure_mode (fav_dishes : List ℕ) :
  (∀ measure, (measure = "most frequently occurring value" → measure = "Mode")) :=
by
  intro measure
  intro h
  sorry

end statistical_measure_mode_l1686_168658


namespace arithmetic_sequence_sum_l1686_168665

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  a 1 + a 2 = -1 →
  a 3 = 4 →
  (a 1 + 2 * d = 4) →
  ∀ n, a n = a 1 + (n - 1) * d →
  a 4 + a 5 = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_sequence_sum_l1686_168665


namespace complement_union_M_N_eq_set_l1686_168664

open Set

-- Define the universe U
def U : Set (ℝ × ℝ) := { p | True }

-- Define the set M
def M : Set (ℝ × ℝ) := { p | (p.snd - 3) / (p.fst - 2) ≠ 1 }

-- Define the set N
def N : Set (ℝ × ℝ) := { p | p.snd ≠ p.fst + 1 }

-- Define the complement of M ∪ N in U
def complement_MN : Set (ℝ × ℝ) := compl (M ∪ N)

theorem complement_union_M_N_eq_set : complement_MN = { (2, 3) } :=
  sorry

end complement_union_M_N_eq_set_l1686_168664


namespace drum_wife_leopard_cost_l1686_168650

-- Definitions
variables (x y z : ℤ)

def system1 := 2 * x + 3 * y + z = 111
def system2 := 3 * x + 4 * y - 2 * z = -8
def even_condition := z % 2 = 0

theorem drum_wife_leopard_cost:
  system1 x y z ∧ system2 x y z ∧ even_condition z →
  x = 20 ∧ y = 9 ∧ z = 44 :=
by
  intro h
  -- Full proof can be provided here
  sorry

end drum_wife_leopard_cost_l1686_168650


namespace part1_solution_set_part2_range_of_a_l1686_168633

open Real

-- For part (1)
theorem part1_solution_set (x a : ℝ) (h : a = 3) : |2 * x - a| + a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := 
by {
  sorry
}

-- For part (2)
theorem part2_range_of_a (f g : ℝ → ℝ) (hf : ∀ x, f x = |2 * x - a| + a) (hg : ∀ x, g x = |2 * x - 3|) :
  (∀ x, f x + g x ≥ 5) ↔ a ≥ 11 / 3 :=
by {
  sorry
}

end part1_solution_set_part2_range_of_a_l1686_168633


namespace largest_sum_fraction_l1686_168673

theorem largest_sum_fraction :
  max 
    ((1/3) + (1/2))
    (max 
      ((1/3) + (1/5))
      (max 
        ((1/3) + (1/6))
        (max 
          ((1/3) + (1/9))
          ((1/3) + (1/10))
        )
      )
    ) = 5/6 :=
by sorry

end largest_sum_fraction_l1686_168673


namespace least_number_of_tablets_l1686_168677

theorem least_number_of_tablets (tablets_A : ℕ) (tablets_B : ℕ) (hA : tablets_A = 10) (hB : tablets_B = 13) :
  ∃ n, ((tablets_A ≤ 10 → n ≥ tablets_A + 2) ∧ (tablets_B ≤ 13 → n ≥ tablets_B + 2)) ∧ n = 12 :=
by
  sorry

end least_number_of_tablets_l1686_168677


namespace total_short_trees_after_planting_l1686_168612

def initial_short_trees : ℕ := 31
def planted_short_trees : ℕ := 64

theorem total_short_trees_after_planting : initial_short_trees + planted_short_trees = 95 := by
  sorry

end total_short_trees_after_planting_l1686_168612


namespace tom_age_difference_l1686_168672

/-- 
Tom Johnson's age is some years less than twice as old as his sister.
The sum of their ages is 14 years.
Tom's age is 9 years.
Prove that the number of years less Tom's age is than twice his sister's age is 1 year. 
-/ 
theorem tom_age_difference (T S : ℕ) 
  (h₁ : T = 9) 
  (h₂ : T + S = 14) : 
  2 * S - T = 1 := 
by 
  sorry

end tom_age_difference_l1686_168672


namespace part1_part2_part3_l1686_168640

-- Definition of the given expression
def expr (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

-- Condition 1: Given final result 2x^2 - 4x + 2
def target_expr1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 2

-- Condition 2: Given values for a and b by Student B
def student_b_expr (x : ℝ) : ℝ := (5 * x^2 - 3 * x + 2) - (5 * x^2 + 3 * x)

-- Condition 3: Result independent of x
def target_expr3 : ℝ := 2

-- Prove conditions and answers
theorem part1 (a b : ℝ) : (∀ x : ℝ, expr a b x = target_expr1 x) → a = 7 ∧ b = -1 :=
sorry

theorem part2 : (∀ x : ℝ, student_b_expr x = -6 * x + 2) :=
sorry

theorem part3 (a b : ℝ) : (∀ x : ℝ, expr a b x = 2) → a = 5 ∧ b = 3 :=
sorry

end part1_part2_part3_l1686_168640


namespace arithmetic_seq_sum_l1686_168651

theorem arithmetic_seq_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 :=
by 
  sorry

end arithmetic_seq_sum_l1686_168651


namespace not_sum_six_odd_squares_l1686_168629

-- Definition stating that a number is odd.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Given that the square of any odd number is 1 modulo 8.
lemma odd_square_mod_eight (n : ℕ) (h : is_odd n) : (n^2) % 8 = 1 :=
sorry

-- Main theorem stating that 1986 cannot be the sum of six squares of odd numbers.
theorem not_sum_six_odd_squares : ¬ ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    is_odd n1 ∧ is_odd n2 ∧ is_odd n3 ∧ is_odd n4 ∧ is_odd n5 ∧ is_odd n6 ∧
    n1^2 + n2^2 + n3^2 + n4^2 + n5^2 + n6^2 = 1986 :=
sorry

end not_sum_six_odd_squares_l1686_168629


namespace fishing_ratio_l1686_168686

variables (B C : ℝ)
variable (brian_per_trip : ℝ)
variable (chris_per_trip : ℝ)

-- Given conditions
def conditions : Prop :=
  C = 10 ∧
  brian_per_trip = 400 ∧
  chris_per_trip = 400 * (5 / 3) ∧
  B * brian_per_trip + 10 * chris_per_trip = 13600

-- The ratio of the number of times Brian goes fishing to the number of times Chris goes fishing
def ratio_correct : Prop :=
  B / C = 26 / 15

theorem fishing_ratio (h : conditions B C brian_per_trip chris_per_trip) : ratio_correct B C :=
by
  sorry

end fishing_ratio_l1686_168686


namespace student_community_arrangements_l1686_168630

theorem student_community_arrangements :
  (3 ^ 4) = 81 :=
by
  sorry

end student_community_arrangements_l1686_168630


namespace solve_quadratic_eq_l1686_168626

theorem solve_quadratic_eq (x : ℝ) :
  (3 * (2 * x + 1) = (2 * x + 1)^2) →
  (x = -1/2 ∨ x = 1) :=
by
  sorry

end solve_quadratic_eq_l1686_168626


namespace history_books_count_l1686_168616

theorem history_books_count :
  ∃ (total_books reading_books math_books science_books history_books : ℕ),
    total_books = 10 ∧
    reading_books = (2 * total_books) / 5 ∧
    math_books = (3 * total_books) / 10 ∧
    science_books = math_books - 1 ∧
    history_books = total_books - (reading_books + math_books + science_books) ∧
    history_books = 1 :=
by
  sorry

end history_books_count_l1686_168616


namespace weight_of_10m_l1686_168680

-- Defining the proportional weight conditions
variable (weight_of_rod : ℝ → ℝ)

-- Conditional facts about the weight function
axiom weight_proportional : ∀ (length1 length2 : ℝ), length1 ≠ 0 → length2 ≠ 0 → 
  weight_of_rod length1 / length1 = weight_of_rod length2 / length2
axiom weight_of_6m : weight_of_rod 6 = 14.04

-- Theorem stating the weight of a 10m rod
theorem weight_of_10m : weight_of_rod 10 = 23.4 := 
sorry

end weight_of_10m_l1686_168680


namespace total_amount_spent_l1686_168610
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l1686_168610


namespace car_speed_second_hour_l1686_168671

/-- The speed of the car in the first hour is 85 km/h, the average speed is 65 km/h over 2 hours,
proving that the speed of the car in the second hour is 45 km/h. -/
theorem car_speed_second_hour (v1 : ℕ) (v_avg : ℕ) (t : ℕ) (d1 : ℕ) (d2 : ℕ) 
  (h1 : v1 = 85) (h2 : v_avg = 65) (h3 : t = 2) (h4 : d1 = v1 * 1) (h5 : d2 = (v_avg * t) - d1) :
  d2 = 45 :=
sorry

end car_speed_second_hour_l1686_168671


namespace general_term_formula_l1686_168602

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ := 3^n + a
noncomputable def an (n : ℕ) : ℝ := 2 * 3^(n-1)

theorem general_term_formula {a : ℝ} (n : ℕ) (h : Sn n a = 3^n + a) :
  Sn n a - Sn (n-1) a = an n :=
sorry

end general_term_formula_l1686_168602


namespace correct_calculation_l1686_168681

-- Definition of the expressions in the problem
def exprA (a : ℝ) : Prop := 2 * a^2 + a^3 = 3 * a^5
def exprB (x y : ℝ) : Prop := ((-3 * x^2 * y)^2 / (x * y) = 9 * x^5 * y^3)
def exprC (b : ℝ) : Prop := (2 * b^2)^3 = 8 * b^6
def exprD (x : ℝ) : Prop := (2 * x * 3 * x^5 = 6 * x^5)

-- The proof problem
theorem correct_calculation (a x y b : ℝ) : exprC b ∧ ¬ exprA a ∧ ¬ exprB x y ∧ ¬ exprD x :=
by {
  sorry
}

end correct_calculation_l1686_168681


namespace units_digit_sum_base8_l1686_168621

theorem units_digit_sum_base8 : 
  let n1 := 53 
  let n2 := 64 
  let sum_base8 := n1 + n2 
  (sum_base8 % 8) = 7 := 
by 
  sorry

end units_digit_sum_base8_l1686_168621


namespace largest_possible_e_l1686_168647

noncomputable def diameter := (2 : ℝ)
noncomputable def PX := (4 / 5 : ℝ)
noncomputable def PY := (3 / 4 : ℝ)
noncomputable def e := (41 - 16 * Real.sqrt 25 : ℝ)
noncomputable def u := 41
noncomputable def v := 16
noncomputable def w := 25

theorem largest_possible_e (P Q X Y Z R S : Real) (d : diameter = 2)
  (PX_len : P - X = 4/5) (PY_len : P - Y = 3/4)
  (e_def : e = 41 - 16 * Real.sqrt 25)
  : u + v + w = 82 :=
by
  sorry

end largest_possible_e_l1686_168647


namespace number_of_zeros_of_f_is_3_l1686_168678

def f (x : ℝ) : ℝ := x^3 - 64 * x

theorem number_of_zeros_of_f_is_3 : ∃ x1 x2 x3, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (x1 ≠ x2) ∧ (x2 ≠ x3) ∧ (x1 ≠ x3) :=
by
  sorry

end number_of_zeros_of_f_is_3_l1686_168678


namespace convert_spherical_to_rectangular_l1686_168697

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem convert_spherical_to_rectangular : spherical_to_rectangular 5 (Real.pi / 2) (Real.pi / 3) = 
  (0, 5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  sorry

end convert_spherical_to_rectangular_l1686_168697


namespace probability_one_white_ball_conditional_probability_P_B_given_A_l1686_168663

-- Definitions for Problem 1
def red_balls : Nat := 4
def white_balls : Nat := 2
def total_balls : Nat := red_balls + white_balls

def C (n k : ℕ) : ℕ := n.choose k

theorem probability_one_white_ball :
  (C 2 1 * C 4 2 : ℚ) / C 6 3 = 3 / 5 :=
by sorry

-- Definitions for Problem 2
def total_after_first_draw : Nat := total_balls - 1
def remaining_red_balls : Nat := red_balls - 1

theorem conditional_probability_P_B_given_A :
  (remaining_red_balls : ℚ) / total_after_first_draw = 3 / 5 :=
by sorry

end probability_one_white_ball_conditional_probability_P_B_given_A_l1686_168663


namespace irrational_implies_irrational_l1686_168699

-- Define irrational number proposition
def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

-- Define the main proposition to prove
theorem irrational_implies_irrational (a : ℝ) : is_irrational (a - 2) → is_irrational a :=
by
  sorry

end irrational_implies_irrational_l1686_168699


namespace paisa_per_rupee_z_gets_l1686_168687

theorem paisa_per_rupee_z_gets
  (y_share : ℝ)
  (y_per_x_paisa : ℝ)
  (total_amount : ℝ)
  (x_share : ℝ)
  (z_share : ℝ)
  (paisa_per_rupee : ℝ)
  (h1 : y_share = 36)
  (h2 : y_per_x_paisa = 0.45)
  (h3 : total_amount = 140)
  (h4 : x_share = y_share / y_per_x_paisa)
  (h5 : z_share = total_amount - (x_share + y_share))
  (h6 : paisa_per_rupee = (z_share / x_share) * 100) :
  paisa_per_rupee = 30 :=
by
  sorry

end paisa_per_rupee_z_gets_l1686_168687


namespace female_participation_fraction_l1686_168609

noncomputable def fraction_of_females (males_last_year : ℕ) (females_last_year : ℕ) : ℚ :=
  let males_this_year := (1.10 * males_last_year : ℚ)
  let females_this_year := (1.25 * females_last_year : ℚ)
  females_this_year / (males_this_year + females_this_year)

theorem female_participation_fraction
  (males_last_year : ℕ) (participation_increase : ℚ)
  (males_increase : ℚ) (females_increase : ℚ)
  (h_males_last_year : males_last_year = 30)
  (h_participation_increase : participation_increase = 1.15)
  (h_males_increase : males_increase = 1.10)
  (h_females_increase : females_increase = 1.25)
  (h_females_last_year : females_last_year = 15) :
  fraction_of_females males_last_year females_last_year = 19 / 52 := by
  sorry

end female_participation_fraction_l1686_168609


namespace translation_4_units_upwards_l1686_168623

theorem translation_4_units_upwards (M N : ℝ × ℝ) (hx : M.1 = N.1) (hy_diff : N.2 - M.2 = 4) :
  N = (M.1, M.2 + 4) :=
by
  sorry

end translation_4_units_upwards_l1686_168623


namespace min_value_of_expr_l1686_168608

theorem min_value_of_expr (a : ℝ) (ha : a > 1) : a + a^2 / (a - 1) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l1686_168608


namespace axis_center_symmetry_sine_shifted_l1686_168639
  noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 3 * Real.pi / 4 + k * Real.pi

  noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (Real.pi / 4 + k * Real.pi, 0)

  theorem axis_center_symmetry_sine_shifted :
    ∀ (k : ℤ),
    ∃ x y : ℝ,
      (x = axis_of_symmetry k) ∧ (y = 0) ∧ (y, 0) = center_of_symmetry k := 
  sorry
  
end axis_center_symmetry_sine_shifted_l1686_168639


namespace area_of_bounded_curve_is_64_pi_l1686_168645

noncomputable def bounded_curve_area : Real :=
  let curve_eq (x y : ℝ) : Prop := (2 * x + 3 * y + 5) ^ 2 + (x + 2 * y - 3) ^ 2 = 64
  let S : Real := 64 * Real.pi
  S

theorem area_of_bounded_curve_is_64_pi : bounded_curve_area = 64 * Real.pi := 
by
  sorry

end area_of_bounded_curve_is_64_pi_l1686_168645


namespace minimum_value_of_expression_l1686_168654

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l1686_168654


namespace calculate_triple_hash_l1686_168695

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem calculate_triple_hash : hash (hash (hash 100)) = 9 := by
  sorry

end calculate_triple_hash_l1686_168695


namespace min_distance_sum_l1686_168618

open Real EuclideanGeometry

-- Define the parabola y^2 = 4x
noncomputable def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1 

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the line l: x = -1
def line_l (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Define the distance from point P to point M
def distance_to_M (P : ℝ × ℝ) : ℝ := dist P M

-- Define the distance from point P to line l
def distance_to_line (P : ℝ × ℝ) := line_l P 

-- Define the sum of distances
def sum_of_distances (P : ℝ × ℝ) : ℝ := distance_to_M P + distance_to_line P

-- Prove the minimum value of the sum of distances
theorem min_distance_sum : ∃ P, parabola P ∧ sum_of_distances P = sqrt 10 := sorry

end min_distance_sum_l1686_168618


namespace eventually_composite_appending_threes_l1686_168679

theorem eventually_composite_appending_threes (n : ℕ) :
  ∃ n' : ℕ, n' = 10 * n + 3 ∧ ∃ k : ℕ, k > 0 ∧ (3 * k + 3) % 7 ≠ 1 ∧ (3 * k + 3) % 7 ≠ 2 ∧ (3 * k + 3) % 7 ≠ 3 ∧
  (3 * k + 3) % 7 ≠ 5 ∧ (3 * k + 3) % 7 ≠ 6 :=
sorry

end eventually_composite_appending_threes_l1686_168679


namespace part1_part2_l1686_168669

def A (x : ℝ) : Prop := -2 < x ∧ x < 10
def B (x a : ℝ) : Prop := (x ≥ 1 + a ∨ x ≤ 1 - a) ∧ a > 0
def p (x : ℝ) : Prop := A x
def q (x a : ℝ) : Prop := B x a

theorem part1 (a : ℝ) (hA : ∀ x, A x → ¬ B x a) : a ≥ 9 :=
sorry

theorem part2 (a : ℝ) (hSuff : ∀ x, (x ≥ 10 ∨ x ≤ -2) → B x a) (hNotNec : ∃ x, ¬ (x ≥ 10 ∨ x ≤ -2) ∧ B x a) : 0 < a ∧ a ≤ 3 :=
sorry

end part1_part2_l1686_168669


namespace triangle_ratio_l1686_168624

variables (A B C : ℝ) (a b c : ℝ)

theorem triangle_ratio (h_cosB : Real.cos B = 4/5)
    (h_a : a = 5)
    (h_area : 1/2 * a * c * Real.sin B = 12) :
    (a + c) / (Real.sin A + Real.sin C) = 25 / 3 :=
sorry

end triangle_ratio_l1686_168624


namespace value_of_expression_l1686_168638

theorem value_of_expression : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 :=
by
  sorry

end value_of_expression_l1686_168638


namespace solve_for_x_l1686_168694

theorem solve_for_x (x : ℚ) (h : (x - 75) / 4 = (5 - 3 * x) / 7) : x = 545 / 19 :=
sorry

end solve_for_x_l1686_168694


namespace perfect_square_trinomial_l1686_168696

-- Define the conditions
theorem perfect_square_trinomial (k : ℤ) : 
  ∃ (a b : ℤ), (a^2 = 1 ∧ b^2 = 16 ∧ (x^2 + k * x * y + 16 * y^2 = (a * x + b * y)^2)) ↔ (k = 8 ∨ k = -8) :=
by
  sorry

end perfect_square_trinomial_l1686_168696


namespace simplify_eval_l1686_168614

variable (x : ℝ)
def expr := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)

theorem simplify_eval (h : x = -2) : expr x = 6 := by
  sorry

end simplify_eval_l1686_168614


namespace factor_expression_l1686_168634

theorem factor_expression (b : ℤ) : 52 * b ^ 2 + 208 * b = 52 * b * (b + 4) := 
by {
  sorry
}

end factor_expression_l1686_168634


namespace difference_between_max_and_min_change_l1686_168688

-- Define percentages as fractions for Lean
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

-- Define the minimum and maximum possible values of changes (in percentage as a fraction)
def min_change : ℚ := 10 / 100
def max_change : ℚ := 50 / 100

-- The theorem we need to prove
theorem difference_between_max_and_min_change : (max_change - min_change) = 40 / 100 :=
by
  sorry

end difference_between_max_and_min_change_l1686_168688
