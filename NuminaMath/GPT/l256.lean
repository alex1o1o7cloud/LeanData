import Mathlib

namespace arccos_half_eq_pi_div_three_l256_256863

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256863


namespace arccos_half_is_pi_div_three_l256_256856

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256856


namespace slope_of_line_l256_256464

theorem slope_of_line (x y : ℝ) : (4 * y = 5 * x - 20) → (y = (5/4) * x - 5) :=
by
  intro h
  sorry

end slope_of_line_l256_256464


namespace algebra_expression_value_l256_256950

theorem algebra_expression_value (a b c : ℝ) (h1 : a - b = 3) (h2 : b + c = -5) : 
  ac - bc + a^2 - ab = -6 := by
  sorry

end algebra_expression_value_l256_256950


namespace percentage_above_wholesale_cost_l256_256305

def wholesale_cost : ℝ := 200
def paid_price : ℝ := 228
def discount_rate : ℝ := 0.05

theorem percentage_above_wholesale_cost :
  ∃ P : ℝ, P = 20 ∧ 
    paid_price = (1 - discount_rate) * (wholesale_cost + P/100 * wholesale_cost) :=
by
  sorry

end percentage_above_wholesale_cost_l256_256305


namespace balls_distribution_l256_256220

theorem balls_distribution : 
  ∃ (ways : ℕ), ways = 15 ∧ (∀ (distribution : Fin 3 → ℕ), (∀ i, 2 ≤ distribution i) → sum distribution = 10) :=
sorry

end balls_distribution_l256_256220


namespace sqrt_mul_l256_256689

theorem sqrt_mul (h₁ : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3) : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 :=
by
  sorry

end sqrt_mul_l256_256689


namespace find_z_l256_256331

theorem find_z (a b p q : ℝ) (z : ℝ) 
  (cond : (z + a + b = q * (p * z - a - b))) : 
  z = (a + b) * (q + 1) / (p * q - 1) :=
sorry

end find_z_l256_256331


namespace mobius_trip_proof_l256_256589

noncomputable def mobius_trip_time : ℝ :=
  let speed_no_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let distance_total := 257
  let distance_typical := 120
  let distance_light := distance_total - distance_typical
  let time_typical := distance_typical / speed_typical_load
  let time_light := distance_light / speed_light_load
  let time_return := distance_total / speed_no_load
  let rest_first := (20 + 25 + 35) / 60.0
  let rest_second := (45 + 30) / 60.0
  time_typical + time_light + time_return + rest_first + rest_second

theorem mobius_trip_proof : mobius_trip_time = 44.6783 :=
  by sorry

end mobius_trip_proof_l256_256589


namespace expression_meaningful_l256_256402

theorem expression_meaningful (x : ℝ) : (∃ y, y = 4 / (x - 5)) ↔ x ≠ 5 :=
by
  sorry

end expression_meaningful_l256_256402


namespace latus_rectum_parabola_l256_256703

theorem latus_rectum_parabola : 
  ∀ (x y : ℝ), (x = 4 * y^2) → (x = -1/16) :=
by 
  sorry

end latus_rectum_parabola_l256_256703


namespace total_cost_l256_256584

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l256_256584


namespace geometric_sequence_proof_l256_256345

theorem geometric_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (h1 : a 2 = 8) 
    (h2 : S 3 = 28) 
    (h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
    (h4 : ∀ n, a n = a 1 * q^(n-1)) 
    (h5 : q > 1) :
    (∀ n, a n = 2^(n + 1)) ∧ (∀ n, (a n)^2 > S n + 7) := sorry

end geometric_sequence_proof_l256_256345


namespace balls_picked_at_random_eq_two_l256_256291

-- Define the initial conditions: number of balls of each color
def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 4
def num_green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls

-- Define the given probability
def given_probability : ℚ := 0.15151515151515152

-- Define the probability calculation for picking two red balls
def probability_two_reds : ℚ :=
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1))

-- The theorem to prove
theorem balls_picked_at_random_eq_two :
  probability_two_reds = given_probability → n = 2 :=
by
  sorry

end balls_picked_at_random_eq_two_l256_256291


namespace train_speed_180_kmph_l256_256817

def train_speed_in_kmph (length_meters : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_m_per_s := length_meters / time_seconds
  let speed_km_per_h := speed_m_per_s * 36 / 10
  speed_km_per_h

theorem train_speed_180_kmph:
  train_speed_in_kmph 400 8 = 180 := by
  sorry

end train_speed_180_kmph_l256_256817


namespace expand_expression_l256_256938

theorem expand_expression (x y : ℝ) : 
  (2 * x + 3) * (5 * y + 7) = 10 * x * y + 14 * x + 15 * y + 21 := 
by sorry

end expand_expression_l256_256938


namespace min_value_of_function_product_inequality_l256_256672

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality_l256_256672


namespace range_of_2alpha_minus_beta_over_3_l256_256349

theorem range_of_2alpha_minus_beta_over_3 (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
sorry

end range_of_2alpha_minus_beta_over_3_l256_256349


namespace monotonicity_tangent_points_l256_256364

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l256_256364


namespace arccos_half_is_pi_div_three_l256_256855

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256855


namespace arccos_half_is_pi_div_three_l256_256853

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256853


namespace replace_asterisk_with_2x_l256_256993

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l256_256993


namespace fisherman_total_fish_l256_256162

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l256_256162


namespace fruit_placement_l256_256024

def Box : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 4}

noncomputable def fruit_positions (B1 B2 B3 B4 : Box) : Prop :=
  (B1 ≠ 1 → B3 ≠ 2 ∨ B3 ≠ 4) ∧
  (B2 ≠ 2) ∧
  (B3 ≠ 3 → B1 ≠ 1) ∧
  (B4 ≠ 4) ∧
  B1 = 1 ∧ B2 = 2 ∧ B3 = 3 ∧ B4 = 4

theorem fruit_placement :
  ∃ (B1 B2 B3 B4 : Box), B1 = 2 ∧ B2 = 4 ∧ B3 = 3 ∧ B4 = 1 := sorry

end fruit_placement_l256_256024


namespace Ed_lost_marble_count_l256_256698

variable (D : ℕ) -- Number of marbles Doug has

noncomputable def Ed_initial := D + 19 -- Ed initially had D + 19 marbles
noncomputable def Ed_now := D + 8 -- Ed now has D + 8 marbles
noncomputable def Ed_lost := Ed_initial D - Ed_now D -- Ed lost Ed_initial - Ed_now marbles

theorem Ed_lost_marble_count : Ed_lost D = 11 := by 
  sorry

end Ed_lost_marble_count_l256_256698


namespace solve_math_problem_l256_256348

-- Math problem definition
def math_problem (A : ℝ) : Prop :=
  (0 < A ∧ A < (Real.pi / 2)) ∧ (Real.cos A = 3 / 5) →
  Real.sin (2 * A) = 24 / 25

-- Example theorem statement in Lean
theorem solve_math_problem (A : ℝ) : math_problem A :=
sorry

end solve_math_problem_l256_256348


namespace park_area_is_120000_l256_256155

noncomputable def area_of_park : ℕ :=
  let speed_km_hr := 12
  let speed_m_min := speed_km_hr * 1000 / 60
  let time_min := 8
  let perimeter := speed_m_min * time_min
  let ratio_l_b := (1, 3)
  let length := perimeter / (2 * (ratio_l_b.1 + ratio_l_b.2))
  let breadth := ratio_l_b.2 * length
  length * breadth

theorem park_area_is_120000 :
  area_of_park = 120000 :=
by
  sorry

end park_area_is_120000_l256_256155


namespace collinear_GFE_l256_256598

/-- Pentagon ABCDE is inscribed in a circle. Its diagonals AC and BD intersect at F. -/
variables (A B C D E F G H I J : Point)
variables (circle : Circle A B C D E) (intersect_AC_BD : Intersects (Line A C) (Line B D) F)
variables (bisector_BAC_CDB : AngleBisector (A B C) (C D B) G)
variables (intersect_AG_BD : Intersects (Line A G) (Line B D) H)
variables (intersect_DG_AC : Intersects (Line D G) (Line A C) I)
variables (intersect_EG_AD : Intersects (Line E G) (Line A D) J)

/-- The quadrilateral FHGI is cyclic. -/
variable (cyclic_FHGI : CyclicQuadrilateral F H G I)

/-- The product equality condition. -/
variable (product_equality : (Distance J A) * (Distance F C) * (Distance G H) 
                             = (Distance J D) * (Distance F B) * (Distance G I))

theorem collinear_GFE : Collinear G F E := 
by
  sorry

end collinear_GFE_l256_256598


namespace average_star_rating_is_four_l256_256261

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l256_256261


namespace three_pow_12_mul_three_pow_8_equals_243_pow_4_l256_256553

theorem three_pow_12_mul_three_pow_8_equals_243_pow_4 : 3^12 * 3^8 = 243^4 := 
by sorry

end three_pow_12_mul_three_pow_8_equals_243_pow_4_l256_256553


namespace monotonicity_tangent_intersection_points_l256_256354

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l256_256354


namespace intervals_of_positivity_l256_256044

theorem intervals_of_positivity :
  {x : ℝ | (x + 1) * (x - 1) * (x - 2) > 0} = {x : ℝ | (-1 < x ∧ x < 1) ∨ (2 < x)} :=
by
  sorry

end intervals_of_positivity_l256_256044


namespace min_value_abs_x1_x2_l256_256959

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_value_abs_x1_x2 
  (a : ℝ) (x1 x2 : ℝ)
  (h_symm : ∃ k : ℤ, -π / 6 - (Real.arctan (Real.sqrt 3 / a)) = (k * π + π / 2))
  (h_diff : f a x1 - f a x2 = -4) :
  |x1 + x2| = (2 * π) / 3 := 
sorry

end min_value_abs_x1_x2_l256_256959


namespace value_of_a_l256_256652

theorem value_of_a (a : ℝ) (h : 1 ∈ ({a, a ^ 2} : Set ℝ)) : a = -1 :=
sorry

end value_of_a_l256_256652


namespace arccos_one_half_l256_256926

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256926


namespace P_and_Q_equivalent_l256_256968

def P (x : ℝ) : Prop := 3 * x - x^2 ≤ 0
def Q (x : ℝ) : Prop := |x| ≤ 2
def P_intersection_Q (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

theorem P_and_Q_equivalent : ∀ x, (P x ∧ Q x) ↔ P_intersection_Q x :=
by {
  sorry
}

end P_and_Q_equivalent_l256_256968


namespace vector_addition_in_triangle_l256_256410

theorem vector_addition_in_triangle
  (A B C D : Type)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] 
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (AB AC AD BD DC : A)
  (h1 : BD = 2 • DC) :
  AD = (1/3 : ℝ) • AB + (2/3 : ℝ) • AC :=
sorry

end vector_addition_in_triangle_l256_256410


namespace example_is_fraction_l256_256000

def is_fraction (a b : ℚ) : Prop := ∃ x y : ℚ, a = x ∧ b = y ∧ y ≠ 0

-- Example condition relevant to the problem
theorem example_is_fraction (x : ℚ) : is_fraction x (x + 2) :=
by
  sorry

end example_is_fraction_l256_256000


namespace jacob_age_proof_l256_256697

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l256_256697


namespace solve_quad_linear_system_l256_256965

theorem solve_quad_linear_system :
  (∃ x y : ℝ, x^2 - 6 * x + 8 = 0 ∧ y + 2 * x = 12 ∧ ((x, y) = (4, 4) ∨ (x, y) = (2, 8))) :=
sorry

end solve_quad_linear_system_l256_256965


namespace prob_either_A_or_B_hired_l256_256296

-- Definitions
def graduates := ["A", "B", "C", "D", "E"]
def num_graduates := 5
def num_to_hire := 2
def total_combinations : ℕ := Nat.choose num_graduates num_to_hire
def combinations_without_A_B : ℕ := Nat.choose (num_graduates - 2) num_to_hire

-- Probability Calculation
def prob_none_A_B : ℚ := combinations_without_A_B / total_combinations
def prob_either_A_B : ℚ := 1 - prob_none_A_B

-- Proof statement
theorem prob_either_A_or_B_hired : prob_either_A_B = 7 / 10 := by
  sorry

end prob_either_A_or_B_hired_l256_256296


namespace Max_students_count_l256_256253

variables (M J : ℕ)

theorem Max_students_count :
  (M = 2 * J + 100) → 
  (M + J = 5400) → 
  M = 3632 := 
by 
  intros h1 h2
  sorry

end Max_students_count_l256_256253


namespace how_many_women_left_l256_256573

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l256_256573


namespace op_5_2_l256_256107

def op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem op_5_2 : op 5 2 = 30 := 
by sorry

end op_5_2_l256_256107


namespace distance_to_origin_l256_256440

theorem distance_to_origin (a : ℝ) (h: |a| = 5) : 3 - a = -2 ∨ 3 - a = 8 :=
sorry

end distance_to_origin_l256_256440


namespace arccos_one_half_l256_256923

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256923


namespace speed_of_jakes_dad_second_half_l256_256978

theorem speed_of_jakes_dad_second_half :
  let distance_to_park := 22
  let total_time := 0.5
  let time_half_journey := total_time / 2
  let speed_first_half := 28
  let distance_first_half := speed_first_half * time_half_journey
  let remaining_distance := distance_to_park - distance_first_half
  let time_second_half := time_half_journey
  let speed_second_half := remaining_distance / time_second_half
  speed_second_half = 60 :=
by
  sorry

end speed_of_jakes_dad_second_half_l256_256978


namespace length_BD_l256_256061

noncomputable def length_segments (CB : ℝ) : ℝ := 4 * CB

noncomputable def circle_radius_AC (CB : ℝ) : ℝ := (4 * CB) / 2

noncomputable def circle_radius_CB (CB : ℝ) : ℝ := CB / 2

noncomputable def tangent_touch_point (CB BD : ℝ) : Prop :=
  ∃ x, CB = x ∧ BD = x

theorem length_BD (CB BD : ℝ) (h : tangent_touch_point CB BD) : BD = CB :=
by
  sorry

end length_BD_l256_256061


namespace cartesian_line_eq_range_m_common_points_l256_256088

-- Definitions of given conditions
def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def polar_line (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Problem part 1: Cartesian equation of the line l
theorem cartesian_line_eq (ρ θ m : ℝ) :
  (ρ * sin (θ + π / 3) + m = 0) →
  (sqrt 3 * ρ * cos θ + ρ * sin θ + 2 * m = 0) :=
by sorry

-- Problem part 2: Range of m for common points with curve C
theorem range_m_common_points (m : ℝ) :
  (∃ t : ℝ, parametric_curve t ∈ set_of (λ p : ℝ × ℝ, ∃ ρ θ, p = (ρ * cos θ, ρ * sin θ) ∧ polar_line ρ θ m)) ↔
  (-19 / 12 ≤ m ∧ m ≤ 5 / 2) :=
by sorry

end cartesian_line_eq_range_m_common_points_l256_256088


namespace product_of_all_possible_N_l256_256314

theorem product_of_all_possible_N (A B N : ℝ) 
  (h1 : A = B + N)
  (h2 : A - 4 = B + N - 4)
  (h3 : B + 5 = B + 5)
  (h4 : |((B + N - 4) - (B + 5))| = 1) :
  ∃ N₁ N₂ : ℝ, (|N₁ - 9| = 1 ∧ |N₂ - 9| = 1) ∧ N₁ * N₂ = 80 :=
by {
  -- We know the absolute value equation leads to two solutions
  -- hence we will consider N₁ and N₂ such that |N - 9| = 1
  -- which eventually yields N = 10 and N = 8, making their product 80.
  sorry
}

end product_of_all_possible_N_l256_256314


namespace cans_of_chili_beans_ordered_l256_256038

theorem cans_of_chili_beans_ordered (T C : ℕ) (h1 : 2 * T = C) (h2 : T + C = 12) : C = 8 := by
  sorry

end cans_of_chili_beans_ordered_l256_256038


namespace highest_prob_red_ball_l256_256232

-- Definitions
def total_red_balls : ℕ := 5
def total_white_balls : ℕ := 12
def total_balls : ℕ := total_red_balls + total_white_balls

-- Condition that neither bag is empty
def neither_bag_empty (r1 w1 r2 w2 : ℕ) : Prop :=
  (r1 + w1 > 0) ∧ (r2 + w2 > 0)

-- Define the probability of drawing a red ball from a bag
def prob_red (r w : ℕ) : ℚ :=
  if (r + w) = 0 then 0 else r / (r + w)

-- Define the overall probability if choosing either bag with equal probability
def overall_prob_red (r1 w1 r2 w2 : ℕ) : ℚ :=
  (prob_red r1 w1 + prob_red r2 w2) / 2

-- Problem statement to be proved
theorem highest_prob_red_ball :
  ∃ (r1 w1 r2 w2 : ℕ),
    neither_bag_empty r1 w1 r2 w2 ∧
    r1 + r2 = total_red_balls ∧
    w1 + w2 = total_white_balls ∧
    (overall_prob_red r1 w1 r2 w2 = 0.625) :=
sorry

end highest_prob_red_ball_l256_256232


namespace probability_sum_is_perfect_square_l256_256781

-- Define the conditions: Two 6-sided dice
def num_faces := 6
def total_outcomes := num_faces * num_faces

-- Define the problem: The probability that the sum is a perfect square
theorem probability_sum_is_perfect_square : ∃ p : ℚ, 
  (p = 7 / 36) ∧ 
  (let perfect_squares := {4, 9},
       valid_outcomes := 
         { (d1, d2) | (d1 + d2 ∈ perfect_squares) } in
    set.size valid_outcomes / total_outcomes = p) :=
begin
  sorry
end

end probability_sum_is_perfect_square_l256_256781


namespace find_C_l256_256819

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 300) 
  (h2 : A + C = 200) 
  (h3 : B + C = 350) : 
  C = 250 := 
  by sorry

end find_C_l256_256819


namespace calculate_change_l256_256831

def price_cappuccino := 2
def price_iced_tea := 3
def price_cafe_latte := 1.5
def price_espresso := 1

def quantity_cappuccinos := 3
def quantity_iced_teas := 2
def quantity_cafe_lattes := 2
def quantity_espressos := 2

def amount_paid := 20

theorem calculate_change :
  let total_cost := (price_cappuccino * quantity_cappuccinos) + 
                    (price_iced_tea * quantity_iced_teas) + 
                    (price_cafe_latte * quantity_cafe_lattes) + 
                    (price_espresso * quantity_espressos) in
  amount_paid - total_cost = 3 :=
by
  sorry

end calculate_change_l256_256831


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256875

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256875


namespace prove_rectangular_selection_l256_256194

def number_of_ways_to_choose_rectangular_region (horizontals verticals : ℕ) : ℕ :=
  (Finset.choose horizontals 2) * (Finset.choose verticals 2)

theorem prove_rectangular_selection :
  number_of_ways_to_choose_rectangular_region 5 5 = 100 :=
by
  sorry

end prove_rectangular_selection_l256_256194


namespace area_of_square_with_perimeter_l256_256136

def perimeter_of_square (s : ℝ) : ℝ := 4 * s

def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_square_with_perimeter (p : ℝ) (h : perimeter_of_square (3 * p) = 12 * p) : area_of_square (3 * p) = 9 * p^2 := by
  sorry

end area_of_square_with_perimeter_l256_256136


namespace find_B_squared_l256_256051

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 23 + 105 / x

theorem find_B_squared :
  ∃ B : ℝ, (B = (Real.sqrt 443)) ∧ (B^2 = 443) :=
by
  sorry

end find_B_squared_l256_256051


namespace repeating_decimal_base_l256_256057

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end repeating_decimal_base_l256_256057


namespace equal_share_payments_l256_256240

theorem equal_share_payments (j n : ℝ) 
  (jack_payment : ℝ := 80) 
  (emma_payment : ℝ := 150) 
  (noah_payment : ℝ := 120)
  (liam_payment : ℝ := 200) 
  (total_cost := jack_payment + emma_payment + noah_payment + liam_payment) 
  (individual_share := total_cost / 4) 
  (jack_due := individual_share - jack_payment) 
  (emma_due := emma_payment - individual_share) 
  (noah_due := individual_share - noah_payment) 
  (liam_due := liam_payment - individual_share) 
  (j := jack_due) 
  (n := noah_due) : 
  j - n = 40 := 
by 
  sorry

end equal_share_payments_l256_256240


namespace area_ratio_of_isosceles_triangle_l256_256245

variable (x : ℝ)
variable (hx : 0 < x)

def isosceles_triangle (AB AC : ℝ) (BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 * x ∧ BC = x

def extend_side (B_length AB_length : ℝ) : Prop :=
  B_length = 2 * AB_length

def ratio_of_areas (area_AB'B'C' area_ABC : ℝ) : Prop :=
  area_AB'B'C' / area_ABC = 9

theorem area_ratio_of_isosceles_triangle
  (AB AC BC : ℝ) (BB' B'C' area_ABC area_AB'B'C' : ℝ)
  (h_isosceles : isosceles_triangle x AB AC BC)
  (h_extend_A : extend_side BB' AB)
  (h_extend_C : extend_side B'C' AC) :
  ratio_of_areas area_AB'B'C' area_ABC := by
  sorry

end area_ratio_of_isosceles_triangle_l256_256245


namespace floor_length_l256_256171

theorem floor_length (width length : ℕ) 
  (cost_per_square total_cost : ℕ)
  (square_side : ℕ)
  (h1 : width = 64) 
  (h2 : square_side = 8)
  (h3 : cost_per_square = 24)
  (h4 : total_cost = 576) 
  : length = 24 :=
by
  -- Placeholder for the proof, using sorry
  sorry

end floor_length_l256_256171


namespace remaining_card_number_l256_256526

theorem remaining_card_number (A B C D E F G H : ℕ) (cards : Finset ℕ) 
  (hA : A + B = 10) 
  (hB : C - D = 1) 
  (hC : E * F = 24) 
  (hD : G / H = 3) 
  (hCards : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : A ∉ cards ∧ B ∉ cards ∧ C ∉ cards ∧ D ∉ cards ∧ E ∉ cards ∧ F ∉ cards ∧ G ∉ cards ∧ H ∉ cards) :
  7 ∈ cards := 
by
  sorry

end remaining_card_number_l256_256526


namespace circles_intersect_at_2_points_l256_256074

theorem circles_intersect_at_2_points :
  let circle1 := { p : ℝ × ℝ | (p.1 - 5 / 2) ^ 2 + p.2 ^ 2 = 25 / 4 }
  let circle2 := { p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 7 / 2) ^ 2 = 49 / 4 }
  ∃ (P1 P2 : ℝ × ℝ), P1 ∈ circle1 ∧ P1 ∈ circle2 ∧
                     P2 ∈ circle1 ∧ P2 ∈ circle2 ∧
                     P1 ≠ P2 ∧ ∀ (P : ℝ × ℝ), P ∈ circle1 ∧ P ∈ circle2 → P = P1 ∨ P = P2 := 
by 
  sorry

end circles_intersect_at_2_points_l256_256074


namespace number_of_packs_of_cake_l256_256250

-- Define the total number of packs of groceries
def total_packs : ℕ := 14

-- Define the number of packs of cookies
def packs_of_cookies : ℕ := 2

-- Define the number of packs of cake as total packs minus packs of cookies
def packs_of_cake : ℕ := total_packs - packs_of_cookies

theorem number_of_packs_of_cake :
  packs_of_cake = 12 := by
  -- Placeholder for the proof
  sorry

end number_of_packs_of_cake_l256_256250


namespace arithmetic_progression_numbers_l256_256776

theorem arithmetic_progression_numbers :
  ∃ (a d : ℚ), (3 * (2 * a - d) = 2 * (a + d)) ∧ ((a - d) * (a + d) = (a - 2)^2) ∧
  ((a = 5 ∧ d = 4 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 ∧ c = 9) 
   ∨ (a = 5 / 4 ∧ d = 1 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 / 4 ∧ c = 9 / 4)) :=
by
  sorry

end arithmetic_progression_numbers_l256_256776


namespace multiply_vars_l256_256512

variables {a b : ℝ}

theorem multiply_vars : -3 * a * b * 2 * a = -6 * a^2 * b := by
  sorry

end multiply_vars_l256_256512


namespace total_songs_purchased_is_162_l256_256789

variable (c_country : ℕ) (c_pop : ℕ) (c_jazz : ℕ) (c_rock : ℕ)
variable (s_country : ℕ) (s_pop : ℕ) (s_jazz : ℕ) (s_rock : ℕ)

-- Setting up the conditions
def num_country_albums := 6
def num_pop_albums := 2
def num_jazz_albums := 4
def num_rock_albums := 3

-- Number of songs per album
def country_album_songs := 9
def pop_album_songs := 9
def jazz_album_songs := 12
def rock_album_songs := 14

theorem total_songs_purchased_is_162 :
  num_country_albums * country_album_songs +
  num_pop_albums * pop_album_songs +
  num_jazz_albums * jazz_album_songs +
  num_rock_albums * rock_album_songs = 162 := by
  sorry

end total_songs_purchased_is_162_l256_256789


namespace train_speed_and_length_l256_256803

theorem train_speed_and_length 
  (x y : ℝ)
  (h1 : 60 * x = 1000 + y)
  (h2 : 40 * x = 1000 - y) :
  x = 20 ∧ y = 200 :=
by
  sorry

end train_speed_and_length_l256_256803


namespace imaginary_part_of_fraction_l256_256108

open Complex

theorem imaginary_part_of_fraction :
  ∃ z : ℂ, z = ⟨0, 1⟩ / ⟨1, 1⟩ ∧ z.im = 1 / 2 :=
by
  sorry

end imaginary_part_of_fraction_l256_256108


namespace find_other_number_l256_256287

theorem find_other_number 
  {A B : ℕ} 
  (h_A : A = 24)
  (h_hcf : Nat.gcd A B = 14)
  (h_lcm : Nat.lcm A B = 312) :
  B = 182 :=
by
  -- Proof skipped
  sorry

end find_other_number_l256_256287


namespace translation_preserves_coordinates_l256_256656

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end translation_preserves_coordinates_l256_256656


namespace thirty_thousand_times_thirty_thousand_l256_256827

-- Define the number thirty thousand
def thirty_thousand : ℕ := 30000

-- Define the product of thirty thousand times thirty thousand
def product_thirty_thousand : ℕ := thirty_thousand * thirty_thousand

-- State the theorem that this product equals nine hundred million
theorem thirty_thousand_times_thirty_thousand :
  product_thirty_thousand = 900000000 :=
sorry -- Proof goes here

end thirty_thousand_times_thirty_thousand_l256_256827


namespace base6_addition_correct_l256_256135

-- We define the numbers in base 6
def a_base6 : ℕ := 2 * 6^3 + 4 * 6^2 + 5 * 6^1 + 3 * 6^0
def b_base6 : ℕ := 1 * 6^4 + 6 * 6^3 + 4 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Define the expected result in base 6 and its base 10 equivalent
def result_base6 : ℕ := 2 * 6^4 + 5 * 6^3 + 5 * 6^2 + 4 * 6^1 + 5 * 6^0
def result_base10 : ℕ := 3881

-- The proof statement
theorem base6_addition_correct : (a_base6 + b_base6 = result_base6) ∧ (result_base6 = result_base10) := by
  sorry

end base6_addition_correct_l256_256135


namespace infinite_positive_sequence_geometric_l256_256742

theorem infinite_positive_sequence_geometric {a : ℕ → ℝ} (h : ∀ n ≥ 1, a (n + 2) = a n - a (n + 1)) 
  (h_pos : ∀ n, a n > 0) :
  ∃ (a1 : ℝ) (q : ℝ), q = (Real.sqrt 5 - 1) / 2 ∧ (∀ n, a n = a1 * q^(n - 1)) := by
  sorry

end infinite_positive_sequence_geometric_l256_256742


namespace solve_for_x_l256_256761

theorem solve_for_x : ∀ (x : ℕ), (1000 = 10^3) → (40 = 2^3 * 5) → 1000^5 = 40^x → x = 15 :=
by
  intros x h1 h2 h3
  sorry

end solve_for_x_l256_256761


namespace max_value_in_interval_l256_256536

variable {R : Type*} [OrderedCommRing R]

variables (f : R → R)
variables (odd_f : ∀ x, f (-x) = -f (x))
variables (f_increasing : ∀ x y, 0 < x → x < y → f x < f y)
variables (additive_f : ∀ x y, f (x + y) = f x + f y)
variables (f1_eq_2 : f 1 = 2)

theorem max_value_in_interval : ∀ x ∈ Set.Icc (-3 : R) (-2 : R), f x ≤ f (-2) ∧ f (-2) = -4 :=
by
  sorry

end max_value_in_interval_l256_256536


namespace part1_monotonicity_part2_tangent_intersection_l256_256359

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l256_256359


namespace age_difference_constant_l256_256618

theorem age_difference_constant (a b x : ℕ) : (a + x) - (b + x) = a - b :=
by
  sorry

end age_difference_constant_l256_256618


namespace sequences_count_n3_sequences_count_n6_sequences_count_n9_l256_256816

inductive Shape
  | triangle
  | square
  | rectangle (k : ℕ)

open Shape

def transition (s : Shape) : List Shape :=
  match s with
  | triangle => [triangle, square]
  | square => [rectangle 1]
  | rectangle k =>
    if k = 0 then [rectangle 1] else [rectangle (k - 1), rectangle (k + 1)]

def count_sequences (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (shapes : List Shape) : ℕ :=
    if m = 0 then shapes.length
    else
      let next_shapes := shapes.bind transition
      aux (m - 1) next_shapes
  aux n [square]

theorem sequences_count_n3 : count_sequences 3 = 5 :=
  by sorry

theorem sequences_count_n6 : count_sequences 6 = 24 :=
  by sorry

theorem sequences_count_n9 : count_sequences 9 = 149 :=
  by sorry

end sequences_count_n3_sequences_count_n6_sequences_count_n9_l256_256816


namespace total_cost_l256_256585

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end total_cost_l256_256585


namespace x_is_integer_l256_256794

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ k1 : ℤ, x^2 - x = k1)
  (h2 : ∃ (n : ℕ) (_ : n > 2) (k2 : ℤ), x^n - x = k2) : 
  ∃ (m : ℤ), x = m := 
sorry

end x_is_integer_l256_256794


namespace tangent_product_l256_256419

noncomputable section

open Real

theorem tangent_product 
  (x y k1 k2 : ℝ) :
  (x / 2) ^ 2 + y ^ 2 = 1 ∧ 
  (x, y) = (-3, -3) ∧ 
  k1 + k2 = 18 / 5 ∧
  k1 * k2 = 8 / 5 → 
  (3 * k1 - 3) * (3 * k2 - 3) = 9 := 
by
  intros 
  sorry

end tangent_product_l256_256419


namespace function_monotonically_increasing_on_interval_l256_256199

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_monotonically_increasing_on_interval (e : ℝ) (h_e_pos : 0 < e) (h_ln_e_pos : 0 < Real.log e) :
  ∀ x : ℝ, e < x → 0 < Real.log x - 1 := 
sorry

end function_monotonically_increasing_on_interval_l256_256199


namespace floor_e_eq_2_l256_256326

noncomputable def e_approx : ℝ := 2.71828

theorem floor_e_eq_2 : ⌊e_approx⌋ = 2 :=
sorry

end floor_e_eq_2_l256_256326


namespace find_acute_angle_l256_256550

theorem find_acute_angle (α : ℝ) (a b : ℝ × ℝ)
  (h_a : a = (3/2, Real.sin α))
  (h_b : b = (Real.cos α, 1/3))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  α = Real.pi / 4 := sorry

end find_acute_angle_l256_256550


namespace desired_average_l256_256425

variable (avg_4_tests : ℕ)
variable (score_5th_test : ℕ)

theorem desired_average (h1 : avg_4_tests = 78) (h2 : score_5th_test = 88) : (4 * avg_4_tests + score_5th_test) / 5 = 80 :=
by
  sorry

end desired_average_l256_256425


namespace reciprocal_roots_k_value_l256_256548

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end reciprocal_roots_k_value_l256_256548


namespace monotonicity_tangent_points_l256_256363

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l256_256363


namespace cartesian_equation_of_l_range_of_m_l256_256090

-- Definitions from conditions
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def polar_line_equation (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

theorem cartesian_equation_of_l (x y m : ℝ) (θ : ℝ) :
  (∃ (ρ : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ) →
  polar_line_equation (sqrt (x^2 + y^2)) (atan2 y x) m →
  (sqrt 3 * x + y + 2 * m = 0) := by
  sorry

theorem range_of_m (m t : ℝ) :
  (parametric_curve_C t = (sqrt 3 * cos (2 * t), 2 * sin t)) →
  (2 * sin t ∈ Icc (-2 : ℝ) 2) →
  ((-19/12) ≤ m ∧ m ≤ 5/2) := by
  sorry

end cartesian_equation_of_l_range_of_m_l256_256090


namespace arccos_half_eq_pi_div_three_l256_256870

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256870


namespace last_years_rate_per_mile_l256_256244

-- Definitions from the conditions
variables (m : ℕ) (x : ℕ)

-- Condition 1: This year, walkers earn $2.75 per mile
def amount_per_mile_this_year : ℝ := 2.75

-- Condition 2: Last year's winner collected $44
def last_years_total_amount : ℕ := 44

-- Condition 3: Elroy will walk 5 more miles than last year's winner
def elroy_walks_more_miles (m : ℕ) : ℕ := m + 5

-- The main goal is to prove that last year's rate per mile was $4 given the conditions
theorem last_years_rate_per_mile (h1 : last_years_total_amount = m * x)
  (h2 : last_years_total_amount = (elroy_walks_more_miles m) * amount_per_mile_this_year) :
  x = 4 :=
by {
  sorry
}

end last_years_rate_per_mile_l256_256244


namespace fisherman_total_fish_l256_256164

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l256_256164


namespace arccos_one_half_is_pi_div_three_l256_256892

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256892


namespace arccos_half_eq_pi_div_3_l256_256911

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256911


namespace paint_used_l256_256581

theorem paint_used (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) 
  (first_week_paint : ℚ) (remaining_paint : ℚ) (second_week_paint : ℚ) (total_used_paint : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/6 →
  second_week_fraction = 1/5 →
  first_week_paint = first_week_fraction * total_paint →
  remaining_paint = total_paint - first_week_paint →
  second_week_paint = second_week_fraction * remaining_paint →
  total_used_paint = first_week_paint + second_week_paint →
  total_used_paint = 120 := sorry

end paint_used_l256_256581


namespace smallest_x_value_min_smallest_x_value_l256_256282

noncomputable def smallest_x_not_defined : ℝ := ( 47 - (Real.sqrt 2041) ) / 12

theorem smallest_x_value :
  ∀ x : ℝ, (6 * x^2 - 47 * x + 7 = 0) → x = smallest_x_not_defined ∨ (x = (47 + (Real.sqrt 2041)) / 12) :=
sorry

theorem min_smallest_x_value :
  smallest_x_not_defined < (47 + (Real.sqrt 2041)) / 12 :=
sorry

end smallest_x_value_min_smallest_x_value_l256_256282


namespace polygon_sides_l256_256719

theorem polygon_sides (n : ℕ) (sum_of_angles : ℕ) (missing_angle : ℕ) 
  (h1 : sum_of_angles = 3240) 
  (h2 : missing_angle * n / (n - 1) = 2 * sum_of_angles) : 
  n = 20 := 
sorry

end polygon_sides_l256_256719


namespace determine_even_condition_l256_256692

theorem determine_even_condition (x : ℤ) (m : ℤ) (h : m = x % 2) : m = 0 ↔ x % 2 = 0 :=
by sorry

end determine_even_condition_l256_256692


namespace simplify_sqrt3_7_pow6_l256_256610

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l256_256610


namespace seniors_playing_all_three_l256_256459

variables {α : Type} [Fintype α]
variables (F B L : Finset α)

def total_seniors := F ∪ B ∪ L
def football := F
def baseball := B
def lacrosse := L
def football_lacrosse := F ∩ L
def baseball_football := B ∩ F
def baseball_lacrosse := B ∩ L
def all_three := F ∩ B ∩ L

variable (n : ℕ)

theorem seniors_playing_all_three (h1 : total_seniors = 85)
    (h2 : football.card = 74)
    (h3 : baseball.card = 26)
    (h4 : football_lacrosse.card = 17)
    (h5 : baseball_football.card = 18)
    (h6 : baseball_lacrosse.card = 13)
    (h7 : lacrosse.card = 2 * n) :
    all_three.card = 11 :=
by sorry

end seniors_playing_all_three_l256_256459


namespace range_of_m_l256_256388

noncomputable def f (m : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1/2 then 10 * x - m else x * Real.exp x - 2 * m * x + m

theorem range_of_m (m e : ℝ) : 
  (∀ (x : ℝ), f m x = 0 → ∃ ! (x : ℝ), f m x = 0) → e < m ∧ m ≤ 5 :=
sorry

end range_of_m_l256_256388


namespace simplify_root_power_l256_256605

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l256_256605


namespace drawings_in_five_pages_l256_256241

theorem drawings_in_five_pages :
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  a₁ + a₂ + a₃ + a₄ + a₅ = 155 :=
by
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  sorry

end drawings_in_five_pages_l256_256241


namespace cos_sum_seventh_roots_of_unity_l256_256530

noncomputable def cos_sum (α : ℝ) : ℝ := 
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)

theorem cos_sum_seventh_roots_of_unity (z : ℂ) (α : ℝ)
  (hz : z^7 = 1) (hz_ne_one : z ≠ 1) (hα : z = Complex.exp (Complex.I * α)) :
  cos_sum α = -1/2 :=
by
  sorry

end cos_sum_seventh_roots_of_unity_l256_256530


namespace g_value_l256_256387

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) - 1

theorem g_value (ω φ : ℝ) (h : ∀ x : ℝ, f ω φ (π / 4 - x) = f ω φ (π / 4 + x)) :
  g ω φ (π / 4) = -1 :=
sorry

end g_value_l256_256387


namespace expenditure_ratio_l256_256168

/-- A man saves 35% of his income in the first year. -/
def saving_rate_first_year : ℝ := 0.35

/-- His income increases by 35% in the second year. -/
def income_increase_rate : ℝ := 0.35

/-- His savings increase by 100% in the second year. -/
def savings_increase_rate : ℝ := 1.0

theorem expenditure_ratio
  (I : ℝ)  -- first year income
  (S1 : ℝ := saving_rate_first_year * I)  -- first year saving
  (E1 : ℝ := I - S1)  -- first year expenditure
  (I2 : ℝ := I + income_increase_rate * I)  -- second year income
  (S2 : ℝ := 2 * S1)  -- second year saving (increases by 100%)
  (E2 : ℝ := I2 - S2)  -- second year expenditure
  :
  (E1 + E2) / E1 = 2
  :=
  sorry

end expenditure_ratio_l256_256168


namespace monotonicity_of_f_tangent_intersection_points_l256_256375

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l256_256375


namespace fraction_decomposition_l256_256050

theorem fraction_decomposition :
  (1 : ℚ) / 4 = (1 : ℚ) / 8 + (1 : ℚ) / 8 := 
by
  -- proof goes here
  sorry

end fraction_decomposition_l256_256050


namespace part1_monotonicity_part2_tangent_intersection_l256_256357

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l256_256357


namespace average_star_rating_l256_256267

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l256_256267


namespace mixed_number_eval_l256_256317

theorem mixed_number_eval :
  -|-(18/5 : ℚ)| - (- (12 /5 : ℚ)) + (4/5 : ℚ) = - (2 / 5 : ℚ) :=
by
  sorry

end mixed_number_eval_l256_256317


namespace final_score_is_83_l256_256122

def running_score : ℕ := 90
def running_weight : ℚ := 0.5

def fancy_jump_rope_score : ℕ := 80
def fancy_jump_rope_weight : ℚ := 0.3

def jump_rope_score : ℕ := 70
def jump_rope_weight : ℚ := 0.2

noncomputable def final_score : ℚ := 
  running_score * running_weight + 
  fancy_jump_rope_score * fancy_jump_rope_weight + 
  jump_rope_score * jump_rope_weight

theorem final_score_is_83 : final_score = 83 := 
  by
    sorry

end final_score_is_83_l256_256122


namespace sum_arithmetic_sequence_l256_256954

theorem sum_arithmetic_sequence {a : ℕ → ℤ} (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 13 = S 2000 →
  S 2013 = 0 :=
by
  sorry

end sum_arithmetic_sequence_l256_256954


namespace one_fourth_difference_l256_256688

theorem one_fourth_difference :
  (1 / 4) * ((9 * 5) - (7 + 3)) = 35 / 4 :=
by sorry

end one_fourth_difference_l256_256688


namespace june_ride_time_l256_256744

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end june_ride_time_l256_256744


namespace part1_monotonicity_part2_tangent_intersection_l256_256358

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

theorem part1_monotonicity (a : ℝ) :
  (a ≥ 1/3 → ∀ x y, x ≤ y → f a x ≤ f a y) ∧
  (a < 1/3 → (∀ x ≤ (1 - real.sqrt (1 - 3 * a)) / 3, f a x ≤ f a ( (1 - real.sqrt (1 - 3 * a)) / 3)) ∧
                  ∀ x ≥ (1 + real.sqrt (1 - 3 * a)) / 3, f a ((1 + real.sqrt (1 - 3 * a)) / 3) ≤ f a x) :=
by sorry

theorem part2_tangent_intersection (a: ℝ) :
  (f a 1 = a + 1 ∧ f a (-1) = -a - 1) :=
by sorry

end part1_monotonicity_part2_tangent_intersection_l256_256358


namespace age_proof_l256_256594

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l256_256594


namespace incorrect_average_initially_l256_256616

theorem incorrect_average_initially (S : ℕ) :
  (S + 25) / 10 = 46 ↔ (S + 65) / 10 = 50 := by
  sorry

end incorrect_average_initially_l256_256616


namespace calculate_series_l256_256511

theorem calculate_series : 20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 200 := 
by
  sorry

end calculate_series_l256_256511


namespace gallons_bought_l256_256332

variable (total_needed : ℕ) (existing_paint : ℕ) (needed_more : ℕ)

theorem gallons_bought (H : total_needed = 70) (H1 : existing_paint = 36) (H2 : needed_more = 11) : 
  total_needed - existing_paint - needed_more = 23 := 
sorry

end gallons_bought_l256_256332


namespace find_m_value_l256_256077

-- Define the conditions
def is_direct_proportion_function (m : ℝ) : Prop :=
  let y := (m - 1) * x^(|m|)
  ∃ k : ℝ, ∀ x : ℝ, y = k * x

-- State the theorem
theorem find_m_value (m : ℝ) (h1 : is_direct_proportion_function m)
  (h2 : m - 1 ≠ 0) (h3 : |m| = 1) : m = -1 := by
  sorry

end find_m_value_l256_256077


namespace my_age_is_five_times_son_age_l256_256592

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l256_256592


namespace geo_seq_a6_eight_l256_256408

-- Definitions based on given conditions
variable (a : ℕ → ℝ) -- the sequence
variable (q : ℝ) -- common ratio
-- Conditions for a_1 * a_3 = 4 and a_4 = 4
def geometric_sequence := ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q
def condition1 := a 1 * a 3 = 4
def condition2 := a 4 = 4

-- Proof problem: Prove a_6 = 8 given the conditions above
theorem geo_seq_a6_eight (h1 : condition1 a) (h2 : condition2 a) (hs : geometric_sequence a) : 
  a 6 = 8 :=
sorry

end geo_seq_a6_eight_l256_256408


namespace arccos_half_is_pi_div_three_l256_256851

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256851


namespace find_number_l256_256668

theorem find_number (x : ℤ) : 17 * (x + 99) = 3111 → x = 84 :=
by
  sorry

end find_number_l256_256668


namespace tournament_draws_l256_256657

open Finset

theorem tournament_draws (n : ℕ) (hn : n = 12) :
  let matches := n.choose 2
  let wins := n
  matches - wins = 54 :=
by
  sorry

end tournament_draws_l256_256657


namespace number_of_sides_l256_256228

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l256_256228


namespace amount_after_two_years_l256_256007

def present_value : ℝ := 70400
def rate : ℝ := 0.125
def years : ℕ := 2
def final_amount := present_value * (1 + rate) ^ years

theorem amount_after_two_years : final_amount = 89070 :=
by sorry

end amount_after_two_years_l256_256007


namespace arccos_one_half_is_pi_div_three_l256_256890

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256890


namespace cone_sphere_ratio_l256_256172

theorem cone_sphere_ratio (r h : ℝ) (π_pos : 0 < π) (r_pos : 0 < r) :
  (1/3) * π * r^2 * h = (1/3) * (4/3) * π * r^3 → h / r = 4/3 :=
by
  sorry

end cone_sphere_ratio_l256_256172


namespace ellipse_equation_is_correct_line_equation_is_correct_l256_256205

-- Given conditions
variable (a b e x y : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (ab_order : b < a)
variable (minor_axis_half_major_axis : 2 * a * (1 / 2) = 2 * b)
variable (right_focus_shortest_distance : a - e = 2 - Real.sqrt 3)
variable (ellipse_equation : a^2 = b^2 + e^2)
variable (m : ℝ)
variable (area_triangle_AOB_is_1 : 1 = 1)

-- Part (I) Prove the equation of ellipse C
theorem ellipse_equation_is_correct :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part (II) Prove the equation of line l
theorem line_equation_is_correct :
  (∀ x y : ℝ, (y = x + m) ↔ ((y = x + (Real.sqrt 10 / 2)) ∨ (y = x - (Real.sqrt 10 / 2)))) :=
sorry

end ellipse_equation_is_correct_line_equation_is_correct_l256_256205


namespace change_is_five_l256_256304

noncomputable def haircut_cost := 15
noncomputable def payment := 20
noncomputable def counterfeit := 20
noncomputable def exchanged_amount := (10 : ℤ) + 10
noncomputable def flower_shop_amount := 20

def change_given (payment haircut_cost: ℕ) : ℤ :=
payment - haircut_cost

theorem change_is_five : 
  change_given payment haircut_cost = 5 :=
by 
  sorry

end change_is_five_l256_256304


namespace identify_correct_statement_l256_256471

-- Definitions based on conditions
def population (athletes : ℕ) : Prop := athletes = 1000
def is_individual (athlete : ℕ) : Prop := athlete ≤ 1000
def is_sample (sampled_athletes : ℕ) (sample_size : ℕ) : Prop := sampled_athletes = 100 ∧ sample_size = 100

-- Theorem statement based on the conclusion
theorem identify_correct_statement (athletes : ℕ) (sampled_athletes : ℕ) (sample_size : ℕ)
    (h1 : population athletes) (h2 : ∀ a, is_individual a) (h3 : is_sample sampled_athletes sample_size) : 
    (sampled_athletes = 100) ∧ (sample_size = 100) :=
by
  sorry

end identify_correct_statement_l256_256471


namespace geometric_sequence_expression_l256_256551

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (h_q : q = 4)
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_sum : a 0 + a 1 + a 2 = 21) :
  ∀ n, a n = 4 ^ n :=
by sorry

end geometric_sequence_expression_l256_256551


namespace hypotenuse_is_2_sqrt_25_point_2_l256_256306

open Real

noncomputable def hypotenuse_length_of_right_triangle (ma mb : ℝ) (a b c : ℝ) : ℝ :=
  if h1 : ma = 6 ∧ mb = sqrt 27 then
    c
  else
    0

theorem hypotenuse_is_2_sqrt_25_point_2 :
  hypotenuse_length_of_right_triangle 6 (sqrt 27) a b (2 * sqrt 25.2) = 2 * sqrt 25.2 :=
by
  sorry -- proof to be filled

end hypotenuse_is_2_sqrt_25_point_2_l256_256306


namespace tangent_parallel_line_l256_256555

open Function

def f (x : ℝ) : ℝ := x^4 - x

def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel_line {P : ℝ × ℝ} (hP : ∃ x y, P = (x, y) ∧ f' x = 3) :
  P = (1, 0) := by
  sorry

end tangent_parallel_line_l256_256555


namespace num_pairs_in_arithmetic_progression_l256_256933

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end num_pairs_in_arithmetic_progression_l256_256933


namespace total_rainfall_l256_256578

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l256_256578


namespace women_left_room_is_3_l256_256564

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l256_256564


namespace complex_polygon_area_l256_256333

noncomputable def area_of_resulting_polygon (side_length : ℝ) (angles : List ℝ) : ℝ :=
  let r := (side_length * Real.sqrt 2) / 2
  let triangle_area (a b : ℝ) := (1 / 2) * r^2 * Real.sin (b - a)
  let total_triang_area := List.sum $ List.map (λ pair, triangle_area pair.1 pair.2) (angles.zip angles.tail)
  4 * total_triang_area + 4 * side_length^2

theorem complex_polygon_area (side_length : ℝ) (angles : List ℝ) (h1 : side_length = 8)
    (h2 : angles = [0, 20 * Real.pi / 180, 50 * Real.pi / 180, 80 * Real.pi / 180]) :
    area_of_resulting_polygon side_length angles = 685.856 :=
  sorry

end complex_polygon_area_l256_256333


namespace fisherman_total_fish_l256_256163

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l256_256163


namespace arccos_half_eq_pi_over_three_l256_256915

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256915


namespace range_of_a_l256_256248

theorem range_of_a (a : ℝ) (x : ℝ) :
  (x^2 - 4 * a * x + 3 * a^2 < 0 → (x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0)) → a < 0 → 
  (a ≤ -4 ∨ -2 / 3 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l256_256248


namespace arccos_one_half_l256_256924

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256924


namespace number_of_sides_l256_256230

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l256_256230


namespace inequality_solution_l256_256942

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := 
by 
  sorry

end inequality_solution_l256_256942


namespace quadratic_real_roots_m_range_l256_256731

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l256_256731


namespace yoongi_rank_l256_256110

def namjoon_rank : ℕ := 2
def yoongi_offset : ℕ := 10

theorem yoongi_rank : namjoon_rank + yoongi_offset = 12 := 
by
  sorry

end yoongi_rank_l256_256110


namespace arccos_half_eq_pi_div_three_l256_256865

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256865


namespace find_point_N_l256_256956

theorem find_point_N 
  (M N : ℝ × ℝ) 
  (MN_length : Real.sqrt (((N.1 - M.1) ^ 2) + ((N.2 - M.2) ^ 2)) = 4)
  (MN_parallel_y_axis : N.1 = M.1)
  (M_coord : M = (-1, 2)) 
  : (N = (-1, 6)) ∨ (N = (-1, -2)) :=
sorry

end find_point_N_l256_256956


namespace paul_mowing_money_l256_256424

theorem paul_mowing_money (M : ℝ) 
  (h1 : 2 * M = 6) : 
  M = 3 :=
by 
  sorry

end paul_mowing_money_l256_256424


namespace sqrt_seven_lt_three_l256_256036

theorem sqrt_seven_lt_three : real.sqrt 7 < 3 := 
by 
  sorry

end sqrt_seven_lt_three_l256_256036


namespace min_le_mult_l256_256748

theorem min_le_mult {x y z m : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
    (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : m ≤ x * y^2 * z^3 :=
by
  sorry

end min_le_mult_l256_256748


namespace replace_asterisk_with_2x_l256_256994

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l256_256994


namespace initial_bushes_l256_256277

theorem initial_bushes (b : ℕ) (h1 : b + 4 = 6) : b = 2 :=
by {
  sorry
}

end initial_bushes_l256_256277


namespace arccos_one_half_l256_256883

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256883


namespace arccos_one_half_is_pi_div_three_l256_256894

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256894


namespace alfred_gain_percent_l256_256178

theorem alfred_gain_percent (P : ℝ) (R : ℝ) (S : ℝ) (H1 : P = 4700) (H2 : R = 800) (H3 : S = 6000) : 
  (S - (P + R)) / (P + R) * 100 = 9.09 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end alfred_gain_percent_l256_256178


namespace speed_at_perigee_l256_256800

-- Define the conditions
def semi_major_axis (a : ℝ) := a > 0
def perigee_distance (a : ℝ) := 0.5 * a
def point_P_distance (a : ℝ) := 0.75 * a
def speed_at_P (v1 : ℝ) := v1 > 0

-- Define what we need to prove
theorem speed_at_perigee (a v1 v2 : ℝ) (h1 : semi_major_axis a) (h2 : speed_at_P v1) :
  v2 = (3 / Real.sqrt 5) * v1 :=
sorry

end speed_at_perigee_l256_256800


namespace scoops_per_carton_l256_256293

-- Definitions for scoops required by everyone
def ethan_vanilla := 1
def ethan_chocolate := 1
def lucas_danny_connor_chocolate_each := 2
def lucas_danny_connor := 3
def olivia_vanilla := 1
def olivia_strawberry := 1
def shannon_vanilla := 2 * olivia_vanilla
def shannon_strawberry := 2 * olivia_strawberry

-- Definitions for total scoops taken
def total_vanilla_taken := ethan_vanilla + olivia_vanilla + shannon_vanilla
def total_chocolate_taken := ethan_chocolate + (lucas_danny_connor_chocolate_each * lucas_danny_connor)
def total_strawberry_taken := olivia_strawberry + shannon_strawberry
def total_scoops_taken := total_vanilla_taken + total_chocolate_taken + total_strawberry_taken

-- Definitions for remaining scoops and original total scoops
def remaining_scoops := 16
def original_scoops := total_scoops_taken + remaining_scoops

-- Definition for number of cartons
def total_cartons := 3

-- Proof goal: scoops per carton
theorem scoops_per_carton : original_scoops / total_cartons = 10 := 
by
  -- Add your proof steps here
  sorry

end scoops_per_carton_l256_256293


namespace num_ways_to_form_rectangle_l256_256193

theorem num_ways_to_form_rectangle (n : ℕ) (h : n = 5) :
  (nat.choose n 2) * (nat.choose n 2) = 100 :=
by {
  rw h,
  exact nat.choose_five_two_mul_five_two 100
}

lemma nat.choose_five_two_mul_five_two :
  ((5.choose 2) * (5.choose 2) = 100) :=
by norm_num

end num_ways_to_form_rectangle_l256_256193


namespace crayon_ratio_l256_256417

theorem crayon_ratio :
  ∀ (Karen Beatrice Gilbert Judah : ℕ),
    Karen = 128 →
    Beatrice = Karen / 2 →
    Beatrice = Gilbert →
    Gilbert = 4 * Judah →
    Judah = 8 →
    Beatrice / Gilbert = 1 :=
by
  intros Karen Beatrice Gilbert Judah hKaren hBeatrice hEqual hGilbert hJudah
  sorry

end crayon_ratio_l256_256417


namespace hypercube_paths_24_l256_256103

-- Define the 4-dimensional hypercube
structure Hypercube4 :=
(vertices : Fin 16) -- Using Fin 16 to represent the 16 vertices
(edges : Fin 32)    -- Using Fin 32 to represent the 32 edges

def valid_paths (start : Fin 16) : Nat :=
  -- This function should calculate the number of valid paths given the start vertex
  24 -- placeholder, as we are giving the pre-computed total number here

theorem hypercube_paths_24 (start : Fin 16) :
  valid_paths start = 24 :=
by sorry

end hypercube_paths_24_l256_256103


namespace bob_shuck_2_hours_l256_256824

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l256_256824


namespace diaries_ratio_l256_256254

variable (initial_diaries : ℕ)
variable (final_diaries : ℕ)
variable (lost_fraction : ℚ)
variable (bought_diaries : ℕ)

theorem diaries_ratio 
  (h1 : initial_diaries = 8)
  (h2 : final_diaries = 18)
  (h3 : lost_fraction = 1 / 4)
  (h4 : ∃ x : ℕ, (initial_diaries + x - lost_fraction * (initial_diaries + x) = final_diaries) ∧ x = 16) :
  (16 / initial_diaries : ℚ) = 2 := 
by
  sorry

end diaries_ratio_l256_256254


namespace gcd_pow_minus_one_l256_256663

theorem gcd_pow_minus_one (n m : ℕ) (hn : n = 1030) (hm : m = 1040) :
  Nat.gcd (2^n - 1) (2^m - 1) = 1023 := 
by
  sorry

end gcd_pow_minus_one_l256_256663


namespace problem_statement_l256_256148

theorem problem_statement (x : ℕ) (h : 423 - x = 421) : (x * 423) + 421 = 1267 := by
  sorry

end problem_statement_l256_256148


namespace square_diff_l256_256549

-- Definitions and conditions from the problem
def three_times_sum_eq (a b : ℝ) : Prop := 3 * (a + b) = 18
def diff_eq (a b : ℝ) : Prop := a - b = 4

-- Goal to prove that a^2 - b^2 = 24 under the given conditions
theorem square_diff (a b : ℝ) (h₁ : three_times_sum_eq a b) (h₂ : diff_eq a b) : a^2 - b^2 = 24 :=
sorry

end square_diff_l256_256549


namespace women_left_l256_256569

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l256_256569


namespace total_votes_l256_256563

theorem total_votes (votes_brenda : ℕ) (total_votes : ℕ) 
  (h1 : votes_brenda = 50) 
  (h2 : votes_brenda = (1/4 : ℚ) * total_votes) : 
  total_votes = 200 :=
by 
  sorry

end total_votes_l256_256563


namespace book_pages_l256_256180

theorem book_pages (n days_n : ℕ) (first_day_pages break_days : ℕ) (common_difference total_pages_read : ℕ) (portion_of_book : ℚ) :
    n = 14 → days_n = 12 → first_day_pages = 10 → break_days = 2 → common_difference = 2 →
    total_pages_read = 252 → portion_of_book = 3/4 →
    (total_pages_read : ℚ) * (4/3) = 336 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end book_pages_l256_256180


namespace box_weights_l256_256590

theorem box_weights (a b c : ℕ) (h1 : a + b = 132) (h2 : b + c = 135) (h3 : c + a = 137) (h4 : a > 40) (h5 : b > 40) (h6 : c > 40) : a + b + c = 202 :=
by 
  sorry

end box_weights_l256_256590


namespace num_sequences_l256_256204

theorem num_sequences (a : ℕ → ℤ) (h1 : a 1 = 0) (h100 : a 100 = 475)
  (h_diff : ∀ k, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) :
  (Nat.choose 99 2) = 4851 := 
by
  sorry

end num_sequences_l256_256204


namespace arccos_half_eq_pi_over_three_l256_256919

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256919


namespace second_polygon_sides_l256_256780

theorem second_polygon_sides (a b n m : ℕ) (s : ℝ) 
  (h1 : a = 45) 
  (h2 : b = 3 * s)
  (h3 : n * b = m * s)
  (h4 : n = 45) : m = 135 := 
by
  sorry

end second_polygon_sides_l256_256780


namespace projection_of_a_onto_b_l256_256960

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

theorem projection_of_a_onto_b : projection vec_a vec_b = Real.sqrt 5 :=
by
  sorry

end projection_of_a_onto_b_l256_256960


namespace lucy_l256_256421

-- Define rounding function to nearest ten
def round_to_nearest_ten (x : Int) : Int :=
  if x % 10 < 5 then x - x % 10 else x + (10 - x % 10)

-- Define the problem with given conditions
def lucy_problem : Prop :=
  let sum := 68 + 57
  round_to_nearest_ten sum = 130

-- Statement of proof problem
theorem lucy's_correct_rounded_sum : lucy_problem := by
  sorry

end lucy_l256_256421


namespace arccos_half_eq_pi_div_three_l256_256873

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256873


namespace second_bucket_capacity_l256_256804

-- Define the initial conditions as given in the problem.
def tank_capacity : ℕ := 48
def bucket1_capacity : ℕ := 4

-- Define the number of times the 4-liter bucket is used.
def bucket1_uses : ℕ := tank_capacity / bucket1_capacity

-- Define a condition related to bucket uses.
def buckets_use_relation (x : ℕ) : Prop :=
  bucket1_uses = (tank_capacity / x) - 4

-- Formulate the theorem that states the capacity of the second bucket.
theorem second_bucket_capacity (x : ℕ) (h : buckets_use_relation x) : x = 3 :=
by {
  sorry
}

end second_bucket_capacity_l256_256804


namespace arccos_half_eq_pi_over_three_l256_256920

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256920


namespace rate_of_current_l256_256099

def downstream_eq (b c : ℝ) : Prop := (b + c) * 4 = 24
def upstream_eq (b c : ℝ) : Prop := (b - c) * 6 = 24

theorem rate_of_current (b c : ℝ) (h1 : downstream_eq b c) (h2 : upstream_eq b c) : c = 1 :=
by sorry

end rate_of_current_l256_256099


namespace wine_division_l256_256660

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end wine_division_l256_256660


namespace abs_add_conditions_l256_256966

theorem abs_add_conditions (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  a + b = 1 ∨ a + b = 7 :=
by
  sorry

end abs_add_conditions_l256_256966


namespace arccos_of_half_eq_pi_over_three_l256_256838

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256838


namespace multiply_digits_correctness_l256_256991

theorem multiply_digits_correctness (a b c : ℕ) :
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c :=
by sorry

end multiply_digits_correctness_l256_256991


namespace right_triangle_area_l256_256770

theorem right_triangle_area (a b c p : ℝ) (h1 : a = b) (h2 : 3 * p = a + b + c)
  (h3 : c = Real.sqrt (2 * a ^ 2)) :
  (1/2) * a ^ 2 = (9 * p ^ 2 * (3 - 2 * Real.sqrt 2)) / 4 :=
by
  sorry

end right_triangle_area_l256_256770


namespace geometric_seq_sum_l256_256541

noncomputable def a_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

theorem geometric_seq_sum :
  let a1 := a_n 1
  let a2 := a_n 2
  let a3 := a_n 3
  let a4 := a_n 4
  let a5 := a_n 5
  a1 + |a2| + a3 + |a4| + a5 = 121 :=
by
  sorry

end geometric_seq_sum_l256_256541


namespace max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l256_256682

noncomputable def point_in_circle : Prop :=
  let P := (-Real.sqrt 3, 2)
  ∃ (x y : ℝ), x^2 + y^2 = 12 ∧ x = -Real.sqrt 3 ∧ y = 2

theorem max_min_AB_length (α : ℝ) (h1 : -Real.sqrt 3 ≤ α ∧ α ≤ Real.pi / 2) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let R := Real.sqrt 12
  ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12 ∧ (P.1, P.2) = (-Real.sqrt 3, 2)) →
    ((max (dist A B) (dist P P)) = 4 * Real.sqrt 3 ∧ (min (dist A B) (dist P P)) = 2 * Real.sqrt 5) :=
sorry

theorem chord_length_at_angle (α : ℝ) (h2 : α = 120 / 180 * Real.pi) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let A := (Real.sqrt 12, 0)
  let B := (-Real.sqrt 12, 0)
  let AB := (dist A B)
  AB = Real.sqrt 47 :=
sorry

theorem trajectory_midpoint_chord :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  ∀ (M : ℝ × ℝ), (∀ k : ℝ, P.2 - 2 = k * (P.1 + Real.sqrt 3) ∧ M.2 = - 1 / k * M.1) → 
  (M.1^2 + M.2^2 + Real.sqrt 3 * M.1 + 2 * M.2 = 0) :=
sorry

end max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l256_256682


namespace animals_remaining_correct_l256_256127

-- Definitions from the conditions
def initial_cows : ℕ := 184
def initial_dogs : ℕ := initial_cows / 2

def cows_sold : ℕ := initial_cows / 4
def remaining_cows : ℕ := initial_cows - cows_sold

def dogs_sold : ℕ := (3 * initial_dogs) / 4
def remaining_dogs : ℕ := initial_dogs - dogs_sold

def total_remaining_animals : ℕ := remaining_cows + remaining_dogs

-- Theorem to be proved
theorem animals_remaining_correct : total_remaining_animals = 161 := 
by
  sorry

end animals_remaining_correct_l256_256127


namespace barrel_contents_lost_l256_256484

theorem barrel_contents_lost (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 220) 
  (h2 : remaining_amount = 198) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 10 :=
by
  rw [h1, h2]
  sorry

end barrel_contents_lost_l256_256484


namespace arccos_one_half_l256_256927

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256927


namespace jacob_current_age_l256_256694

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l256_256694


namespace triangle_area_proof_l256_256487

noncomputable def segment_squared (a b : ℝ) : ℝ := a ^ 2 - b ^ 2

noncomputable def triangle_conditions (a b c : ℝ): Prop :=
  segment_squared b a = a ^ 2 - c ^ 2

noncomputable def area_triangle_OLK (r a b c : ℝ) (cond : triangle_conditions a b c): ℝ :=
  (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3))

theorem triangle_area_proof (r a b c : ℝ) (cond : triangle_conditions a b c) :
  area_triangle_OLK r a b c cond = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3)) :=
sorry

end triangle_area_proof_l256_256487


namespace inequality_trig_l256_256207

theorem inequality_trig 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (hz : 0 < z ∧ z < (π / 2)) :
  (π / 2) + 2 * (Real.sin x) * (Real.cos y) + 2 * (Real.sin y) * (Real.cos z) > 
  (Real.sin (2 * x)) + (Real.sin (2 * y)) + (Real.sin (2 * z)) :=
by
  sorry  -- The proof is omitted

end inequality_trig_l256_256207


namespace carson_gardening_l256_256691

theorem carson_gardening : 
  ∀ (lines_to_mow : ℕ) (time_per_line : ℕ) (total_gardening_time : ℕ)
    (flowers_per_row : ℕ) (time_per_flower : ℚ),
    lines_to_mow = 40 →
    time_per_line = 2 →
    total_gardening_time = 108 →
    flowers_per_row = 7 →
    time_per_flower = (1/2 : ℚ) →
    let time_mowing := lines_to_mow * time_per_line in
    let remaining_time := total_gardening_time - time_mowing in
    let total_flowers := remaining_time / time_per_flower in
    let rows_of_flowers := total_flowers / flowers_per_row in
    rows_of_flowers = 8 :=
begin
  intros lines_to_mow time_per_line total_gardening_time flowers_per_row time_per_flower,
  intros h1 h2 h3 h4 h5,
  let time_mowing := lines_to_mow * time_per_line,
  let remaining_time := total_gardening_time - time_mowing,
  let total_flowers := remaining_time / time_per_flower,
  let rows_of_flowers := total_flowers / flowers_per_row,
  sorry,
end

end carson_gardening_l256_256691


namespace no_real_solution_for_ap_l256_256934

theorem no_real_solution_for_ap : 
  (¬∃ (a b : ℝ), 15, a, b, a * b form_an_arithmetic_progression) :=
sorry

end no_real_solution_for_ap_l256_256934


namespace amount_given_to_second_set_of_families_l256_256997

theorem amount_given_to_second_set_of_families
  (total_spent : ℝ) (amount_first_set : ℝ) (amount_last_set : ℝ)
  (h_total_spent : total_spent = 900)
  (h_amount_first_set : amount_first_set = 325)
  (h_amount_last_set : amount_last_set = 315) :
  total_spent - amount_first_set - amount_last_set = 260 :=
by
  -- sorry is placed to skip the proof
  sorry

end amount_given_to_second_set_of_families_l256_256997


namespace marie_finishes_third_task_at_3_30_PM_l256_256251

open Time

def doesThreeEqualTasksInARow (start first second third : TimeClock) : Prop :=
  ∃ d : TimeClock, -- duration of one task
  first = start + d ∧
  second = first + d ∧
  third = second + d

theorem marie_finishes_third_task_at_3_30_PM :
  ∀ start first second third : TimeClock,
  start.hours = 13 ∧ start.minutes = 0 ∧ -- 1:00 PM
  second.hours = 14 ∧ second.minutes = 40 ∧ -- 2:40 PM
  doesThreeEqualTasksInARow start first second third →
  third.hours = 15 ∧ third.minutes = 30 := -- 3:30 PM
by
  intros start first second third start_cond second_cond equal_tasks_cond
  sorry

end marie_finishes_third_task_at_3_30_PM_l256_256251


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256881

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256881


namespace arccos_of_half_eq_pi_over_three_l256_256834

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256834


namespace difference_of_numbers_l256_256764

theorem difference_of_numbers : 
  ∃ (L S : ℕ), L = 1631 ∧ L = 6 * S + 35 ∧ L - S = 1365 := 
by
  sorry

end difference_of_numbers_l256_256764


namespace maximize_sum_of_arithmetic_seq_l256_256504

theorem maximize_sum_of_arithmetic_seq (a d : ℤ) (n : ℤ) : d < 0 → a^2 = (a + 10 * d)^2 → n = 5 ∨ n = 6 :=
by
  intro h_d_neg h_a1_eq_a11
  have h_a1_5d_neg : a + 5 * d = 0 := sorry
  have h_sum_max : n = 5 ∨ n = 6 := sorry
  exact h_sum_max

end maximize_sum_of_arithmetic_seq_l256_256504


namespace age_difference_is_40_l256_256152

-- Define the ages of the daughter and the mother
variables (D M : ℕ)

-- Conditions
-- 1. The mother's age is the digits of the daughter's age reversed
def mother_age_is_reversed_daughter_age : Prop :=
  M = 10 * D + D

-- 2. In thirteen years, the mother will be twice as old as the daughter
def mother_twice_as_old_in_thirteen_years : Prop :=
  M + 13 = 2 * (D + 13)

-- The theorem: The difference in their current ages is 40
theorem age_difference_is_40
  (h1 : mother_age_is_reversed_daughter_age D M)
  (h2 : mother_twice_as_old_in_thirteen_years D M) :
  M - D = 40 :=
sorry

end age_difference_is_40_l256_256152


namespace property_tax_increase_is_800_l256_256792

-- Define conditions as constants
def tax_rate : ℝ := 0.10
def initial_value : ℝ := 20000
def new_value : ℝ := 28000

-- Define the increase in property tax
def tax_increase : ℝ := (new_value * tax_rate) - (initial_value * tax_rate)

-- Statement to be proved
theorem property_tax_increase_is_800 : tax_increase = 800 :=
by
  sorry

end property_tax_increase_is_800_l256_256792


namespace boat_travel_distance_downstream_l256_256292

-- Define the conditions given in the problem
def speed_boat_still_water := 22 -- in km/hr
def speed_stream := 5 -- in km/hr
def time_downstream := 2 -- in hours

-- Define a function to compute the effective speed downstream
def effective_speed_downstream (speed_boat: ℝ) (speed_stream: ℝ) : ℝ :=
  speed_boat + speed_stream

-- Define a function to compute the distance travelled downstream
def distance_downstream (speed: ℝ) (time: ℝ) : ℝ :=
  speed * time

-- The main theorem to prove
theorem boat_travel_distance_downstream :
  distance_downstream (effective_speed_downstream speed_boat_still_water speed_stream) time_downstream = 54 :=
by
  -- Proof is to be filled in later
  sorry

end boat_travel_distance_downstream_l256_256292


namespace solve_system_l256_256060

theorem solve_system (x y : ℚ) 
  (h1 : x + 2 * y = -1) 
  (h2 : 2 * x + y = 3) : 
  x + y = 2 / 3 := 
sorry

end solve_system_l256_256060


namespace presidency_meeting_arrangements_l256_256677

theorem presidency_meeting_arrangements : 
  let total_ways := 3 * (nat.choose 6 3) * (nat.choose 6 1) * (nat.choose 6 1) in 
  total_ways = 2160 := 
by
  sorry

end presidency_meeting_arrangements_l256_256677


namespace solve_for_x_l256_256434

theorem solve_for_x (x : ℚ) : 
  (5 * x + 8 * x = 350 - 9 * (x + 8)) → 
  (x = 139 / 11) :=
by
  intro h
  sorry

end solve_for_x_l256_256434


namespace women_left_l256_256568

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l256_256568


namespace probability_student_less_than_25_l256_256478

def total_students : ℕ := 100

-- Percentage conditions translated to proportions
def proportion_male : ℚ := 0.48
def proportion_female : ℚ := 0.52

def proportion_male_25_or_older : ℚ := 0.50
def proportion_female_25_or_older : ℚ := 0.20

-- Definition of probability that a randomly selected student is less than 25 years old.
def probability_less_than_25 : ℚ :=
  (proportion_male * (1 - proportion_male_25_or_older)) +
  (proportion_female * (1 - proportion_female_25_or_older))

theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.656 := by
  sorry

end probability_student_less_than_25_l256_256478


namespace arccos_one_half_l256_256882

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256882


namespace smallest_n_19n_congruent_1453_mod_8_l256_256139

theorem smallest_n_19n_congruent_1453_mod_8 : 
  ∃ (n : ℕ), 19 * n % 8 = 1453 % 8 ∧ ∀ (m : ℕ), (19 * m % 8 = 1453 % 8 → n ≤ m) := 
sorry

end smallest_n_19n_congruent_1453_mod_8_l256_256139


namespace exists_coprime_positive_sum_le_m_l256_256669

theorem exists_coprime_positive_sum_le_m (m : ℕ) (a b : ℤ) 
  (ha : 0 < a) (hb : 0 < b) (hcoprime : Int.gcd a b = 1)
  (h1 : a ∣ (m + b^2)) (h2 : b ∣ (m + a^2)) 
  : ∃ a' b', 0 < a' ∧ 0 < b' ∧ Int.gcd a' b' = 1 ∧ a' ∣ (m + b'^2) ∧ b' ∣ (m + a'^2) ∧ a' + b' ≤ m + 1 :=
by
  sorry

end exists_coprime_positive_sum_le_m_l256_256669


namespace jane_current_age_l256_256413

noncomputable def JaneAge : ℕ := 34

theorem jane_current_age : 
  ∃ J : ℕ, 
    (∀ t : ℕ, t ≥ 18 ∧ t - 18 ≤ JaneAge - 18 → t ≤ JaneAge / 2) ∧
    (JaneAge - 12 = 23 - 12 * 2) ∧
    (23 = 23) →
    J = 34 := by
  sorry

end jane_current_age_l256_256413


namespace problem_statement_l256_256534

theorem problem_statement (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n : ℕ, a (n + 2) = a n)
  (h_b : ∀ n : ℕ, b (n + 1) - b n = a n)
  (h_repeat : ∀ k : ℕ, ∃ m : ℕ, (b (2 * m) / a m) = k)
  : b 1 = 2 :=
sorry

end problem_statement_l256_256534


namespace cube_sum_gt_l256_256747

variable (a b c d : ℝ)
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
variable (h1 : a + b = c + d)
variable (h2 : a^2 + b^2 > c^2 + d^2)

theorem cube_sum_gt : a^3 + b^3 > c^3 + d^3 := by
  sorry

end cube_sum_gt_l256_256747


namespace monotonicity_of_f_tangent_intersection_l256_256360

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l256_256360


namespace line_equation_is_correct_l256_256272

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

theorem line_equation_is_correct (x y t : ℝ)
  (h1: x = 3 * t + 6)
  (h2: y = 5 * t - 7) :
  y = (5 / 3) * x - 17 :=
sorry

end line_equation_is_correct_l256_256272


namespace probability_of_selected_cubes_l256_256298

-- Total number of unit cubes
def total_unit_cubes : ℕ := 125

-- Number of cubes with exactly two blue faces (from edges not corners)
def two_blue_faces : ℕ := 9

-- Number of unpainted unit cubes
def unpainted_cubes : ℕ := 51

-- Calculate total combinations of choosing 2 cubes out of 125
def total_combinations : ℕ := Nat.choose total_unit_cubes 2

-- Calculate favorable outcomes: one cube with 2 blue faces and one unpainted cube
def favorable_outcomes : ℕ := two_blue_faces * unpainted_cubes

-- Calculate probability
def probability : ℚ := favorable_outcomes / total_combinations

-- The theorem we want to prove
theorem probability_of_selected_cubes :
  probability = 3 / 50 :=
by
  -- Show that the probability indeed equals 3/50
  sorry

end probability_of_selected_cubes_l256_256298


namespace remainder_when_divided_l256_256198

theorem remainder_when_divided (P D Q R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D * D') = R + R' * D :=
by
  sorry

end remainder_when_divided_l256_256198


namespace exists_solution_in_range_l256_256693

open Function

theorem exists_solution_in_range : ∃ z ∈ set.Icc (-10 : ℝ) 10, exp (2 * z) = (z - 2) / (z + 2) := 
sorry

end exists_solution_in_range_l256_256693


namespace find_x_value_l256_256455

theorem find_x_value {C S x : ℝ}
  (h1 : C = 100 * (1 + x / 100))
  (h2 : S - C = 10 / 9)
  (h3 : S = 100 * (1 + x / 100)):
  x = 10 :=
by
  sorry

end find_x_value_l256_256455


namespace students_with_uncool_family_l256_256084

-- Define the conditions as given in the problem.
variables (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ)
          (cool_siblings : ℕ) (cool_siblings_and_dads : ℕ)

-- Provide the known values as conditions.
def problem_conditions := 
  total_students = 50 ∧
  cool_dads = 20 ∧
  cool_moms = 25 ∧
  both_cool_parents = 12 ∧
  cool_siblings = 5 ∧
  cool_siblings_and_dads = 3

-- State the problem: prove the number of students with all uncool family members.
theorem students_with_uncool_family : problem_conditions total_students cool_dads cool_moms 
                                            both_cool_parents cool_siblings cool_siblings_and_dads →
                                    (50 - ((20 - 12) + (25 - 12) + 12 + (5 - 3)) = 15) :=
by intros h; cases h; sorry

end students_with_uncool_family_l256_256084


namespace monotonicity_and_tangent_intersection_l256_256366

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l256_256366


namespace savannah_wrapped_gifts_with_second_roll_l256_256431

theorem savannah_wrapped_gifts_with_second_roll (total_gifts rolls_used roll_1_gifts roll_3_gifts roll_2_gifts : ℕ) 
  (h1 : total_gifts = 12) 
  (h2 : rolls_used = 3) 
  (h3 : roll_1_gifts = 3) 
  (h4 : roll_3_gifts = 4)
  (h5 : total_gifts - roll_1_gifts - roll_3_gifts = roll_2_gifts) :
  roll_2_gifts = 5 := 
by
  sorry

end savannah_wrapped_gifts_with_second_roll_l256_256431


namespace solve_proportion_l256_256476

noncomputable def x : ℝ := 0.6

theorem solve_proportion (x : ℝ) (h : 0.75 / x = 10 / 8) : x = 0.6 :=
by
  sorry

end solve_proportion_l256_256476


namespace arc_length_problem_l256_256769

noncomputable def arc_length (r : ℝ) (theta : ℝ) : ℝ :=
  r * theta

theorem arc_length_problem :
  ∀ (r : ℝ) (theta_deg : ℝ), r = 1 ∧ theta_deg = 150 → 
  arc_length r (theta_deg * (Real.pi / 180)) = (5 * Real.pi / 6) :=
by
  intro r theta_deg h
  sorry

end arc_length_problem_l256_256769


namespace fish_caught_in_second_catch_l256_256560

theorem fish_caught_in_second_catch
  (tagged_fish_released : Int)
  (tagged_fish_in_second_catch : Int)
  (total_fish_in_pond : Int)
  (C : Int)
  (h_tagged_fish_count : tagged_fish_released = 60)
  (h_tagged_in_second_catch : tagged_fish_in_second_catch = 2)
  (h_total_fish : total_fish_in_pond = 1800) :
  C = 60 :=
by
  sorry

end fish_caught_in_second_catch_l256_256560


namespace unique_zero_function_l256_256520

theorem unique_zero_function {f : ℕ → ℕ} (h : ∀ m n, f (m + f n) = f m + f n + f (n + 1)) : ∀ n, f n = 0 :=
by {
  sorry
}

end unique_zero_function_l256_256520


namespace find_bananas_l256_256125

theorem find_bananas 
  (bananas apples persimmons : ℕ) 
  (h1 : apples = 4 * bananas) 
  (h2 : persimmons = 3 * bananas) 
  (h3 : apples + persimmons = 210) : 
  bananas = 30 := 
  sorry

end find_bananas_l256_256125


namespace EdProblem_l256_256937

/- Define the conditions -/
def EdConditions := 
  ∃ (m : ℕ) (N : ℕ), 
    m = 16 ∧ 
    N = Nat.choose 15 5 ∧
    N % 1000 = 3

/- The statement to be proven -/
theorem EdProblem : EdConditions :=
  sorry

end EdProblem_l256_256937


namespace range_of_m_l256_256216

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) → m < 2 + 2 * Real.sqrt 2 :=
sorry

end range_of_m_l256_256216


namespace or_false_iff_not_p_l256_256797

theorem or_false_iff_not_p (p q : Prop) : (p ∨ q → false) ↔ ¬p :=
by sorry

end or_false_iff_not_p_l256_256797


namespace prism_volume_l256_256935

theorem prism_volume (a b c : ℝ) (h1 : a * b = 12) (h2 : b * c = 8) (h3 : a * c = 4) : a * b * c = 8 * Real.sqrt 6 :=
by 
  sorry

end prism_volume_l256_256935


namespace problem_statement_l256_256211

theorem problem_statement (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = - f x)
  (hf_deriv : ∀ x < 0, 2 * f x + x * deriv f x < 0) :
  f 1 < 2016 * f (Real.sqrt 2016) ∧ 2016 * f (Real.sqrt 2016) < 2017 * f (Real.sqrt 2017) := 
  sorry

end problem_statement_l256_256211


namespace value_divided_by_l256_256143

theorem value_divided_by {x : ℝ} : (5 / x) * 12 = 10 → x = 6 :=
by
  sorry

end value_divided_by_l256_256143


namespace find_expression_and_area_l256_256716

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := - x^2 - 4 * x + 1

theorem find_expression_and_area :
  (∃ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c = 0 ∧ (∃ x : ℝ, f'(x) = 2 * x + 2) ∧ ∀ x, (f(x) = a * x^2 + b * x + c)) ∧
  let f (x : ℝ) := x^2 + 2 * x + 1 in
  let g (x : ℝ) := - x^2 - 4 * x + 1 in
  (∃ (area : ℝ), area = ∫ x in (-3)..0, (g x - f x) dx ∧ area = 9) :=
begin
  have h_deriv : ∀ x : ℝ, has_deriv_at f (2 * x + 2) x := sorry,
  use [1, 2, 1],
  split,
  { split,
    { norm_num },
    { split,
      { sorry },
      { intro x,
        simp [f] } } },
  { unfold f g,
    use 9,
    split,
    { sorry },
    { norm_num } }
end

end find_expression_and_area_l256_256716


namespace problem_solution_l256_256214

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f x + 2 * Real.cos x ^ 2

theorem problem_solution :
  (∀ x, (∃ ω > 0, ∃ φ, |φ| < Real.pi / 2 ∧ Real.sin (ω * x - φ) = 0 ∧ 2 * ω = Real.pi)) →
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (g x ≤ 2 ∧ g x ≥ 1 / 2)) :=
by
  sorry

end problem_solution_l256_256214


namespace largest_five_digit_palindromic_number_l256_256537

def is_five_digit_palindrome (n : ℕ) : Prop := n / 10000 = n % 10 ∧ (n / 1000) % 10 = (n / 10) % 10

def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

theorem largest_five_digit_palindromic_number :
  ∃ (abcba deed : ℕ), is_five_digit_palindrome abcba ∧ 10000 ≤ abcba ∧ abcba < 100000 ∧ is_four_digit_palindrome deed ∧ 1000 ≤ deed ∧ deed < 10000 ∧ abcba = 45 * deed ∧ abcba = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l256_256537


namespace arccos_half_eq_pi_div_3_l256_256912

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256912


namespace correct_systematic_sampling_l256_256129

-- Definitions for conditions in a)
def num_bags := 50
def num_selected := 5
def interval := num_bags / num_selected

-- We encode the systematic sampling selection process
def systematic_sampling (n : Nat) (start : Nat) (interval: Nat) (count : Nat) : List Nat :=
  List.range count |>.map (λ i => start + i * interval)

-- Theorem to prove that the selection of bags should have an interval of 10
theorem correct_systematic_sampling :
  ∃ (start : Nat), systematic_sampling num_selected start interval num_selected = [7, 17, 27, 37, 47] := sorry

end correct_systematic_sampling_l256_256129


namespace monotonicity_of_f_tangent_line_intersection_points_l256_256352

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l256_256352


namespace train_passing_time_correct_l256_256411

noncomputable def train_passing_time (L1 L2 : ℕ) (S1 S2 : ℕ) : ℝ :=
  let S1_mps := S1 * (1000 / 3600)
  let S2_mps := S2 * (1000 / 3600)
  let relative_speed := S1_mps + S2_mps
  let total_length := L1 + L2
  total_length / relative_speed

theorem train_passing_time_correct :
  train_passing_time 105 140 45 36 = 10.89 := by
  sorry

end train_passing_time_correct_l256_256411


namespace mapping_sum_l256_256957

theorem mapping_sum (f : ℝ × ℝ → ℝ × ℝ) (a b : ℝ)
(h1 : ∀ x y, f (x, y) = (x, x + y))
(h2 : (a, b) = f (1, 3)) :
  a + b = 5 :=
sorry

end mapping_sum_l256_256957


namespace joao_chocolates_l256_256242

theorem joao_chocolates (n : ℕ) (hn1 : 30 < n) (hn2 : n < 100) (h1 : n % 7 = 1) (h2 : n % 10 = 2) : n = 92 :=
sorry

end joao_chocolates_l256_256242


namespace sand_total_weight_l256_256048

variable (Eden_buckets : ℕ)
variable (Mary_buckets : ℕ)
variable (Iris_buckets : ℕ)
variable (bucket_weight : ℕ)
variable (total_weight : ℕ)

axiom Eden_buckets_eq : Eden_buckets = 4
axiom Mary_buckets_eq : Mary_buckets = Eden_buckets + 3
axiom Iris_buckets_eq : Iris_buckets = Mary_buckets - 1
axiom bucket_weight_eq : bucket_weight = 2
axiom total_weight_eq : total_weight = (Eden_buckets + Mary_buckets + Iris_buckets) * bucket_weight

theorem sand_total_weight : total_weight = 34 := by
  rw [total_weight_eq, Eden_buckets_eq, Mary_buckets_eq, Iris_buckets_eq, bucket_weight_eq]
  sorry

end sand_total_weight_l256_256048


namespace harry_morning_ratio_l256_256700

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio_l256_256700


namespace greatest_integer_of_set_is_152_l256_256623

-- Define the conditions
def median (s : Set ℤ) : ℤ := 150
def smallest_integer (s : Set ℤ) : ℤ := 140
def consecutive_even_integers (s : Set ℤ) : Prop := 
  ∀ x ∈ s, ∃ y ∈ s, x = y ∨ x = y + 2

-- The main theorem
theorem greatest_integer_of_set_is_152 (s : Set ℤ) 
  (h_median : median s = 150)
  (h_smallest : smallest_integer s = 140)
  (h_consecutive : consecutive_even_integers s) : 
  ∃ greatest : ℤ, greatest = 152 := 
sorry

end greatest_integer_of_set_is_152_l256_256623


namespace right_triangle_side_lengths_l256_256206

theorem right_triangle_side_lengths (a S : ℝ) (b c : ℝ)
  (h1 : S = b + c)
  (h2 : c^2 = a^2 + b^2) :
  b = (S^2 - a^2) / (2 * S) ∧ c = (S^2 + a^2) / (2 * S) :=
by
  sorry

end right_triangle_side_lengths_l256_256206


namespace arccos_half_eq_pi_div_three_l256_256842

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256842


namespace find_first_number_l256_256190

theorem find_first_number (x : ℤ) (k : ℤ) :
  (29 > 0) ∧ (x % 29 = 8) ∧ (1490 % 29 = 11) → x = 29 * k + 8 :=
by
  intros h
  sorry

end find_first_number_l256_256190


namespace negative_number_from_operations_l256_256787

theorem negative_number_from_operations :
  (∀ (a b : Int), a + b < 0 → a = -1 ∧ b = -3) ∧
  (∀ (a b : Int), a - b < 0 → a = 1 ∧ b = 4) ∧
  (∀ (a b : Int), a * b > 0 → a = 3 ∧ b = -2) ∧
  (∀ (a b : Int), a / b = 0 → a = 0 ∧ b = -7) :=
by
  sorry

end negative_number_from_operations_l256_256787


namespace next_sales_amount_l256_256492

theorem next_sales_amount
  (royalties1: ℝ)
  (sales1: ℝ)
  (royalties2: ℝ)
  (percentage_decrease: ℝ)
  (X: ℝ)
  (h1: royalties1 = 4)
  (h2: sales1 = 20)
  (h3: royalties2 = 9)
  (h4: percentage_decrease = 58.333333333333336 / 100)
  (h5: royalties2 / X = royalties1 / sales1 - ((royalties1 / sales1) * percentage_decrease)): 
  X = 108 := 
  by 
    -- Proof omitted
    sorry

end next_sales_amount_l256_256492


namespace partitions_count_l256_256525

open Finset

noncomputable def numberOfPartitions (n : ℕ) : ℕ :=
  (choose (n+2) 2) - (numberOfInvalidPartitions n)

def validPartition (A1 A2 A3 : Finset ℕ) : Prop :=
  (∀ (a ∈ A1) (b ∈ A1), a < b → ((a % 2) ≠ (b % 2))) ∧
  (∀ (a ∈ A2) (b ∈ A2), a < b → ((a % 2) ≠ (b % 2))) ∧
  (∀ (a ∈ A3) (b ∈ A3), a < b → ((a % 2) ≠ (b % 2))) ∧
  (A1.nonempty → A2.nonempty → A3.nonempty → (∃! (m ∈ {A1, A2, A3}), ∃ x ∈ m, x % 2 = 0))

def numberOfInvalidPartitions (n : ℕ) : ℕ :=
  sorry -- Define specific counting of invalid partitions as explained

theorem partitions_count (n : ℕ) :
  (∃ A1 A2 A3 : Finset ℕ, (A1 ∪ A2 ∪ A3 = range n) ∧ disjoint A1 A2 ∧ disjoint A2 A3 ∧ disjoint A1 A3 ∧ validPartition A1 A2 A3) →
  numberOfPartitions n = choose (n+2) 2 - numberOfInvalidPartitions n :=
sorry -- Proof construction

end partitions_count_l256_256525


namespace arccos_half_eq_pi_div_3_l256_256913

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256913


namespace sum_of_ages_five_years_from_now_l256_256662

noncomputable def viggo_age_when_brother_was_2 (brother_age: ℕ) : ℕ :=
  10 + 2 * brother_age

noncomputable def current_viggo_age (viggo_age_at_2: ℕ) (current_brother_age: ℕ) : ℕ :=
  viggo_age_at_2 + (current_brother_age - 2)

def sister_age (viggo_age: ℕ) : ℕ :=
  viggo_age + 5

noncomputable def cousin_age (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) : ℕ :=
  ((viggo_age + brother_age + sister_age) / 3)

noncomputable def future_ages_sum (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) (cousin_age: ℕ) : ℕ :=
  viggo_age + 5 + brother_age + 5 + sister_age + 5 + cousin_age + 5

theorem sum_of_ages_five_years_from_now :
  let current_brother_age := 10
  let viggo_age_at_2 := viggo_age_when_brother_was_2 2
  let current_viggo_age := current_viggo_age viggo_age_at_2 current_brother_age
  let current_sister_age := sister_age current_viggo_age
  let current_cousin_age := cousin_age current_viggo_age current_brother_age current_sister_age
  future_ages_sum current_viggo_age current_brother_age current_sister_age current_cousin_age = 99 := sorry

end sum_of_ages_five_years_from_now_l256_256662


namespace sum_reciprocal_of_shifted_roots_l256_256985

noncomputable def roots_of_cubic (a b c : ℝ) : Prop := 
    ∀ x : ℝ, x^3 - x - 2 = (x - a) * (x - b) * (x - c)

theorem sum_reciprocal_of_shifted_roots (a b c : ℝ) 
    (h : roots_of_cubic a b c) : 
    (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = 1 :=
by
  sorry

end sum_reciprocal_of_shifted_roots_l256_256985


namespace Jack_sent_correct_number_of_BestBuy_cards_l256_256412

def price_BestBuy_gift_card : ℕ := 500
def price_Walmart_gift_card : ℕ := 200
def initial_BestBuy_gift_cards : ℕ := 6
def initial_Walmart_gift_cards : ℕ := 9

def total_price_of_initial_gift_cards : ℕ :=
  (initial_BestBuy_gift_cards * price_BestBuy_gift_card) +
  (initial_Walmart_gift_cards * price_Walmart_gift_card)

def price_of_Walmart_sent : ℕ := 2 * price_Walmart_gift_card
def value_of_gift_cards_remaining : ℕ := 3900

def prove_sent_BestBuy_worth : Prop :=
  total_price_of_initial_gift_cards - value_of_gift_cards_remaining - price_of_Walmart_sent = 1 * price_BestBuy_gift_card

theorem Jack_sent_correct_number_of_BestBuy_cards :
  prove_sent_BestBuy_worth :=
by
  sorry

end Jack_sent_correct_number_of_BestBuy_cards_l256_256412


namespace exists_monochromatic_triangle_in_K6_l256_256438

/-- In a complete graph with 6 vertices where each edge is colored either red or blue,
    there exists a set of 3 vertices such that the edges joining them are all the same color. -/
theorem exists_monochromatic_triangle_in_K6 (color : Fin 6 → Fin 6 → Prop)
  (h : ∀ {i j : Fin 6}, i ≠ j → (color i j ∨ ¬ color i j)) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  ((color i j ∧ color j k ∧ color k i) ∨ (¬ color i j ∧ ¬ color j k ∧ ¬ color k i)) :=
by
  sorry

end exists_monochromatic_triangle_in_K6_l256_256438


namespace arccos_half_eq_pi_div_3_l256_256906

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256906


namespace cartesian_equation_of_l_range_of_m_l256_256089

-- Define the parametric equations for curve C
def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

-- Define the polar equation for line l
def polar_line_l (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

-- Define the Cartesian equation of line l
def cartesian_line_l (x y m : ℝ) : Prop :=
  sqrt 3 * x + y + 2 * m = 0

-- Proof that the Cartesian equation of l is sqrt(3)x + y + 2m = 0
theorem cartesian_equation_of_l (ρ θ m : ℝ) (h : polar_line_l ρ θ m) :
  ∃ x y, cartesian_line_l x y m :=
by
  let x := ρ * cos θ
  let y := ρ * sin θ
  use [x, y]
  sorry

-- Proof for the range of values of m for l and C to have common points
theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, let x := sqrt 3 * cos (2 * t) in
            let y := 2 * sin t in
            cartesian_line_l x y m) ↔ -19/12 ≤ m ∧ m ≤ 5/2 :=
by
  sorry

end cartesian_equation_of_l_range_of_m_l256_256089


namespace triangle_perimeter_correct_l256_256309

def side_a : ℕ := 15
def side_b : ℕ := 8
def side_c : ℕ := 10
def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter_correct :
  perimeter side_a side_b side_c = 33 := by
sorry

end triangle_perimeter_correct_l256_256309


namespace probability_neither_square_nor_cube_l256_256639

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l256_256639


namespace mow_lawn_payment_l256_256790

theorem mow_lawn_payment (bike_cost weekly_allowance babysitting_rate babysitting_hours money_saved target_savings mowing_payment : ℕ) 
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : babysitting_rate = 7)
  (h4 : babysitting_hours = 2)
  (h5 : money_saved = 65)
  (h6 : target_savings = 6) :
  mowing_payment = 10 :=
sorry

end mow_lawn_payment_l256_256790


namespace simplify_root_power_l256_256604

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l256_256604


namespace percentage_of_rotten_bananas_l256_256175

-- Define the initial conditions and the question as a Lean theorem statement
theorem percentage_of_rotten_bananas (oranges bananas : ℕ) (perc_rot_oranges perc_good_fruits : ℝ) 
  (total_fruits good_fruits good_oranges good_bananas rotten_bananas perc_rot_bananas : ℝ) :
  oranges = 600 →
  bananas = 400 →
  perc_rot_oranges = 0.15 →
  perc_good_fruits = 0.886 →
  total_fruits = (oranges + bananas) →
  good_fruits = (perc_good_fruits * total_fruits) →
  good_oranges = ((1 - perc_rot_oranges) * oranges) →
  good_bananas = (good_fruits - good_oranges) →
  rotten_bananas = (bananas - good_bananas) →
  perc_rot_bananas = ((rotten_bananas / bananas) * 100) →
  perc_rot_bananas = 6 :=
by
  intros; sorry

end percentage_of_rotten_bananas_l256_256175


namespace sequence_sixth_term_l256_256974

theorem sequence_sixth_term (a b c d : ℚ) : 
  (a = 1/4 * (5 + b)) →
  (b = 1/4 * (a + 45)) →
  (45 = 1/4 * (b + c)) →
  (c = 1/4 * (45 + d)) →
  d = 1877 / 3 :=
by
  sorry

end sequence_sixth_term_l256_256974


namespace sqrt_9_eq_pm3_l256_256771

theorem sqrt_9_eq_pm3 : ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_9_eq_pm3_l256_256771


namespace bonus_percentage_correct_l256_256233

/-
Tom serves 10 customers per hour and works for 8 hours, earning 16 bonus points.
We need to find the percentage of bonus points per customer served.
-/

def customers_per_hour : ℕ := 10
def hours_worked : ℕ := 8
def total_bonus_points : ℕ := 16

def total_customers_served : ℕ := customers_per_hour * hours_worked
def bonus_percentage : ℕ := (total_bonus_points * 100) / total_customers_served

theorem bonus_percentage_correct : bonus_percentage = 20 := by
  sorry

end bonus_percentage_correct_l256_256233


namespace find_C_and_D_l256_256702

theorem find_C_and_D :
  (∀ x, x^2 - 3 * x - 10 ≠ 0 → (4 * x - 3) / (x^2 - 3 * x - 10) = (17 / 7) / (x - 5) + (11 / 7) / (x + 2)) :=
by
  sorry

end find_C_and_D_l256_256702


namespace women_left_room_is_3_l256_256566

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l256_256566


namespace monotonicity_and_tangent_intersections_l256_256381

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l256_256381


namespace radius_of_base_of_cone_is_3_l256_256013

noncomputable def radius_of_base_of_cone (θ R : ℝ) : ℝ :=
  ((θ / 360) * 2 * Real.pi * R) / (2 * Real.pi)

theorem radius_of_base_of_cone_is_3 :
  radius_of_base_of_cone 120 9 = 3 := 
by 
  simp [radius_of_base_of_cone]
  sorry

end radius_of_base_of_cone_is_3_l256_256013


namespace SmallestPositiveAngle_l256_256712

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l256_256712


namespace final_bicycle_price_is_225_l256_256174

noncomputable def final_selling_price (cp_A : ℝ) (profit_A : ℝ) (profit_B : ℝ) : ℝ :=
  let sp_B := cp_A * (1 + profit_A / 100)
  let sp_C := sp_B * (1 + profit_B / 100)
  sp_C

theorem final_bicycle_price_is_225 :
  final_selling_price 114.94 35 45 = 224.99505 :=
by
  sorry

end final_bicycle_price_is_225_l256_256174


namespace sum_of_repeating_decimals_l256_256033

-- Declare the repeating decimals as constants
def x : ℚ := 2/3
def y : ℚ := 7/9

-- The problem statement
theorem sum_of_repeating_decimals : x + y = 13 / 9 := by
  sorry

end sum_of_repeating_decimals_l256_256033


namespace student_solved_correctly_l256_256020

theorem student_solved_correctly (c e : ℕ) (h1 : c + e = 80) (h2 : 5 * c - 3 * e = 8) : c = 31 :=
sorry

end student_solved_correctly_l256_256020


namespace range_of_m_l256_256426

theorem range_of_m (a : ℝ) (h : a ≠ 0) (x1 x2 y1 y2 : ℝ) (m : ℝ)
  (hx1 : -2 < x1 ∧ x1 < 0) (hx2 : m < x2 ∧ x2 < m + 1)
  (h_on_parabola_A : y1 = a * x1^2 - 2 * a * x1 - 3)
  (h_on_parabola_B : y2 = a * x2^2 - 2 * a * x2 - 3)
  (h_diff_y : y1 ≠ y2) :
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end range_of_m_l256_256426


namespace max_area_14_5_l256_256989

noncomputable def rectangle_max_area (P D : ℕ) (x y : ℝ) : ℝ :=
  if (2 * x + 2 * y = P) ∧ (x^2 + y^2 = D^2) then x * y else 0

theorem max_area_14_5 :
  ∃ (x y : ℝ), (2 * x + 2 * y = 14) ∧ (x^2 + y^2 = 5^2) ∧ rectangle_max_area 14 5 x y = 12.25 :=
by
  sorry

end max_area_14_5_l256_256989


namespace probability_neither_square_nor_cube_l256_256624

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l256_256624


namespace sam_needs_change_l256_256406

noncomputable def toy_prices : List ℝ := [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50]

def sam_initial_quarters : ℝ := 12.0 * 0.25

def favorite_toy_price : ℝ := 4.00

def total_permutations (n : ℕ) := (Finset.range n).card.fact

def favorable_permutations : ℕ := (Finset.range 7).card.fact

def probability_no_change_needed : ℝ := favorable_permutations / total_permutations 10

def probability_change_needed : ℝ := 1 - probability_no_change_needed

theorem sam_needs_change : probability_change_needed = 719 / 720 :=
by
  sorry

end sam_needs_change_l256_256406


namespace ticket_cost_correct_l256_256030

theorem ticket_cost_correct : 
  ∀ (a : ℝ), 
  (3 * a + 5 * (a / 2) = 30) → 
  10 * a + 8 * (a / 2) ≥ 10 * a + 8 * (a / 2) * 0.9 →
  10 * a + 8 * (a / 2) * 0.9 = 68.733 :=
by
  intro a
  intro h1 h2
  sorry

end ticket_cost_correct_l256_256030


namespace arithmetic_progression_product_difference_le_one_l256_256506

theorem arithmetic_progression_product_difference_le_one 
  (a b : ℝ) :
  ∃ (m n k l : ℤ), |(a + b * m) * (a + b * n) - (a + b * k) * (a + b * l)| ≤ 1 :=
sorry

end arithmetic_progression_product_difference_le_one_l256_256506


namespace A_8_coords_l256_256977

-- Define point as a structure
structure Point where
  x : Int
  y : Int

-- Initial point A
def A : Point := {x := 3, y := 2}

-- Symmetric point about the y-axis
def sym_y (p : Point) : Point := {x := -p.x, y := p.y}

-- Symmetric point about the origin
def sym_origin (p : Point) : Point := {x := -p.x, y := -p.y}

-- Symmetric point about the x-axis
def sym_x (p : Point) : Point := {x := p.x, y := -p.y}

-- Function to get the n-th symmetric point in the sequence
def sym_point (n : Nat) : Point :=
  match n % 3 with
  | 0 => A
  | 1 => sym_y A
  | 2 => sym_origin (sym_y A)
  | _ => A  -- Fallback case (should not be reachable for n >= 0)

theorem A_8_coords : sym_point 8 = {x := 3, y := -2} := sorry

end A_8_coords_l256_256977


namespace rectangle_enclosure_l256_256195
open BigOperators

theorem rectangle_enclosure (n m : ℕ) (hn : n = 5) (hm : m = 5) : 
  (∑ i in finset.range n, ∑ j in finset.range i, 1) * 
  (∑ k in finset.range m, ∑ l in finset.range k, 1) = 100 := by
  sorry

end rectangle_enclosure_l256_256195


namespace arccos_half_eq_pi_div_three_l256_256859

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256859


namespace find_first_offset_l256_256053

theorem find_first_offset 
  (diagonal : ℝ) (second_offset : ℝ) (area : ℝ) (first_offset : ℝ)
  (h_diagonal : diagonal = 20)
  (h_second_offset : second_offset = 4)
  (h_area : area = 90)
  (h_area_formula : area = (diagonal * (first_offset + second_offset)) / 2) :
  first_offset = 5 :=
by 
  rw [h_diagonal, h_second_offset, h_area] at h_area_formula 
  -- This would be the place where you handle solving the formula using the given conditions
  sorry

end find_first_offset_l256_256053


namespace determine_GH_l256_256452

-- Define a structure for a Tetrahedron with edge lengths as given conditions
structure Tetrahedron :=
  (EF FG EH FH EG GH : ℕ)

-- Instantiate the Tetrahedron with the given edge lengths
def tetrahedron_EFGH := Tetrahedron.mk 42 14 37 19 28 14

-- State the theorem
theorem determine_GH (t : Tetrahedron) (hEF : t.EF = 42) :
  t.GH = 14 :=
sorry

end determine_GH_l256_256452


namespace part1_part2_l256_256218

def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 4}

theorem part1 : A ∩ (B 3)ᶜ = Set.Icc 3 5 := by
  sorry

theorem part2 : A ∩ B m = C → m = 8 := by
  sorry

end part1_part2_l256_256218


namespace problem_solution_l256_256531

noncomputable def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_xplus1_even (x : ℝ) : f (x + 1) = f (-x + 1) := sorry

theorem problem_solution : f 2015 = -1 := 
by 
  sorry

end problem_solution_l256_256531


namespace simplify_sqrt3_7_pow6_l256_256612

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l256_256612


namespace spadesuit_value_l256_256519

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l256_256519


namespace arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l256_256137

theorem arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125 :
  (16 + 23 + 38 + 11.5) / 4 = 22.125 :=
by
  sorry

end arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l256_256137


namespace tangent_parallel_x_axis_monotonically_increasing_intervals_l256_256215

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

theorem tangent_parallel_x_axis (m n : ℝ) (h : m ≠ 0) (h_tangent : 3 * m * (2:ℝ)^2 + 2 * n * (2:ℝ) = 0) :
  n = -3 * m :=
by
  sorry

theorem monotonically_increasing_intervals (m : ℝ) (h : m ≠ 0) : 
  (∀ x : ℝ, 3 * m * x * (x - (2 : ℝ)) > 0 ↔ 
    if m > 0 then x < 0 ∨ 2 < x else 0 < x ∧ x < 2) :=
by
  sorry

end tangent_parallel_x_axis_monotonically_increasing_intervals_l256_256215


namespace simplify_sqrt3_7_pow6_l256_256611

theorem simplify_sqrt3_7_pow6 : (∛7)^6 = 49 :=
by
  -- we can use the properties of exponents directly in Lean
  have h : (∛7)^6 = (7^(1/3))^6 := by rfl
  rw h
  rw [←real.rpow_mul 7 (1/3) 6]
  norm_num
  -- additional steps to deal with the specific operations might be required to provide the final proof
  sorry

end simplify_sqrt3_7_pow6_l256_256611


namespace mashed_potatoes_suggestion_count_l256_256433

def number_of_students_suggesting_bacon := 394
def extra_students_suggesting_mashed_potatoes := 63
def number_of_students_suggesting_mashed_potatoes := number_of_students_suggesting_bacon + extra_students_suggesting_mashed_potatoes

theorem mashed_potatoes_suggestion_count :
  number_of_students_suggesting_mashed_potatoes = 457 := by
  sorry

end mashed_potatoes_suggestion_count_l256_256433


namespace fraction_of_cookies_l256_256334

-- Given conditions
variables 
  (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3)

-- Proof statement
theorem fraction_of_cookies (Millie_cookies Mike_cookies Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3) : 
  (Frank_cookies / Mike_cookies : ℚ) = 1 / 4 :=
by
  sorry

end fraction_of_cookies_l256_256334


namespace cars_through_toll_booth_l256_256407

noncomputable def total_cars_in_week (n_mon n_tue n_wed n_thu n_fri n_sat n_sun : ℕ) : ℕ :=
  n_mon + n_tue + n_wed + n_thu + n_fri + n_sat + n_sun 

theorem cars_through_toll_booth : 
  let n_mon : ℕ := 50
  let n_tue : ℕ := 50
  let n_wed : ℕ := 2 * n_mon
  let n_thu : ℕ := 2 * n_mon
  let n_fri : ℕ := 50
  let n_sat : ℕ := 50
  let n_sun : ℕ := 50
  total_cars_in_week n_mon n_tue n_wed n_thu n_fri n_sat n_sun = 450 := 
by 
  sorry

end cars_through_toll_booth_l256_256407


namespace tangent_line_inclination_range_l256_256621

theorem tangent_line_inclination_range:
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 → (0 ≤ 2*x ∧ 2*x ≤ 1 ∨ -1 ≤ 2*x ∧ 2*x < 0) →
    ∃ (α : ℝ), (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π) :=
sorry

end tangent_line_inclination_range_l256_256621


namespace cost_of_article_l256_256150

theorem cost_of_article 
    (C G : ℝ) 
    (h1 : 340 = C + G) 
    (h2 : 350 = C + G + 0.05 * G) 
    : C = 140 :=
by
    -- We do not need to provide the proof; 'sorry' is sufficient.
    sorry

end cost_of_article_l256_256150


namespace arccos_half_eq_pi_over_three_l256_256918

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256918


namespace c_share_l256_256012

theorem c_share (x y z a b c : ℝ) 
  (H1 : b = (65/100) * a)
  (H2 : c = (40/100) * a)
  (H3 : a + b + c = 328) : 
  c = 64 := 
sorry

end c_share_l256_256012


namespace assignment_increase_l256_256783

-- Define what an assignment statement is
def assignment_statement (lhs rhs : ℕ) : ℕ := rhs

-- Define the conditions and the problem
theorem assignment_increase (n : ℕ) : assignment_statement n (n + 1) = n + 1 :=
by
  -- Here we would prove that the assignment statement increases n by 1
  sorry

end assignment_increase_l256_256783


namespace part1_part2_l256_256493

-- Part 1: Proving the solutions for (x-1)^2 = 49
theorem part1 (x : ℝ) (h : (x - 1)^2 = 49) : x = 8 ∨ x = -6 :=
sorry

-- Part 2: Proving the time for the object to reach the ground
theorem part2 (t : ℝ) (h : 4.9 * t^2 = 10) : t = 10 / 7 :=
sorry

end part1_part2_l256_256493


namespace tangent_line_equation_at_point_l256_256765

def curve (x : ℝ) : ℝ := -x^2 + 1
def point : ℝ × ℝ := (1, 0)

theorem tangent_line_equation_at_point :
  ∀ x y : ℝ, point = (1, 0) → y = curve x → 
  ∃ m b : ℝ, m = -2 ∧ b = 2 ∧ m * x + b = y :=
by
  sorry

end tangent_line_equation_at_point_l256_256765


namespace probability_x_lt_y_in_rectangle_l256_256809

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l256_256809


namespace sqrt_k_kn_eq_k_sqrt_kn_l256_256458

theorem sqrt_k_kn_eq_k_sqrt_kn (k n : ℕ) (h : k = Nat.sqrt (n + 1)) : 
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := 
sorry

end sqrt_k_kn_eq_k_sqrt_kn_l256_256458


namespace wrench_weight_relation_l256_256670

variables (h w : ℕ)

theorem wrench_weight_relation (h w : ℕ) 
  (cond : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) : w = 2 * h := 
by sorry

end wrench_weight_relation_l256_256670


namespace carson_gold_stars_l256_256182

theorem carson_gold_stars (gold_stars_yesterday gold_stars_today : ℕ) (h1 : gold_stars_yesterday = 6) (h2 : gold_stars_today = 9) : 
  gold_stars_yesterday + gold_stars_today = 15 := 
by
  sorry

end carson_gold_stars_l256_256182


namespace combined_weight_difference_l256_256746

def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := -5.25
def biology_weight : ℝ := 3.755

theorem combined_weight_difference :
  (chemistry_weight - calculus_weight) - (geometry_weight + biology_weight) = 7.995 :=
by
  sorry

end combined_weight_difference_l256_256746


namespace max_profit_l256_256500

theorem max_profit : ∃ v p : ℝ, 
  v + p ≤ 5 ∧
  v + 3 * p ≤ 12 ∧
  100000 * v + 200000 * p = 850000 :=
by
  sorry

end max_profit_l256_256500


namespace area_increase_l256_256814

theorem area_increase (original_length original_width new_length : ℝ)
  (h1 : original_length = 20)
  (h2 : original_width = 5)
  (h3 : new_length = original_length + 10) :
  (new_length * original_width - original_length * original_width) = 50 := by
  sorry

end area_increase_l256_256814


namespace median_interval_60_64_l256_256653

theorem median_interval_60_64 
  (students : ℕ) 
  (f_45_49 f_50_54 f_55_59 f_60_64 : ℕ) :
  students = 105 ∧ 
  f_45_49 = 8 ∧ 
  f_50_54 = 15 ∧ 
  f_55_59 = 20 ∧ 
  f_60_64 = 18 ∧ 
  (8 + 15 + 20 + 18) ≥ (105 + 1) / 2
  → 60 ≤ (105 + 1) / 2  ∧ (105 + 1) / 2 ≤ 64 :=
sorry

end median_interval_60_64_l256_256653


namespace pants_cost_correct_l256_256687

-- Define the conditions as variables
def initial_money : ℕ := 71
def shirt_cost : ℕ := 5
def num_shirts : ℕ := 5
def remaining_money : ℕ := 20

-- Define intermediates necessary to show the connection between conditions and the question
def money_spent_on_shirts : ℕ := num_shirts * shirt_cost
def money_left_after_shirts : ℕ := initial_money - money_spent_on_shirts
def pants_cost : ℕ := money_left_after_shirts - remaining_money

-- The main theorem to prove the question is equal to the correct answer
theorem pants_cost_correct : pants_cost = 26 :=
by
  sorry

end pants_cost_correct_l256_256687


namespace exists_multiple_of_10_of_three_distinct_integers_l256_256114

theorem exists_multiple_of_10_of_three_distinct_integers
    (a b c : ℤ) 
    (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    ∃ x y : ℤ, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ (10 ∣ (x^5 * y^3 - x^3 * y^5)) :=
by
  sorry

end exists_multiple_of_10_of_three_distinct_integers_l256_256114


namespace seashells_calculation_l256_256755

theorem seashells_calculation :
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  leigh_seashells = 16 :=
by
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  show leigh_seashells = 16
  sorry

end seashells_calculation_l256_256755


namespace arithmetic_sequence_general_formula_and_sum_max_l256_256739

theorem arithmetic_sequence_general_formula_and_sum_max :
  ∀ (a : ℕ → ℤ), 
  (a 7 = -8) → (a 17 = -28) → 
  (∀ n, a n = -2 * n + 6) ∧ 
  (∀ S : ℕ → ℤ, (∀ n, S n = -n^2 + 5 * n) → ∀ n, S n ≤ 6) :=
by
  sorry

end arithmetic_sequence_general_formula_and_sum_max_l256_256739


namespace total_boxes_l256_256774
namespace AppleBoxes

theorem total_boxes (initial_boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ)
  (apples_per_bag : ℕ) (bags_per_box : ℕ) (good_apples : ℕ) (final_boxes : ℕ) :
  initial_boxes = 14 →
  apples_per_box = 105 →
  rotten_apples = 84 →
  apples_per_bag = 6 →
  bags_per_box = 7 →
  final_boxes = (initial_boxes * apples_per_box - rotten_apples) / (apples_per_bag * bags_per_box) →
  final_boxes = 33 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  exact h6

end AppleBoxes

end total_boxes_l256_256774


namespace num_ways_award_medals_l256_256775

-- There are 8 sprinters in total
def num_sprinters : ℕ := 8

-- Three of the sprinters are Americans
def num_americans : ℕ := 3

-- The number of non-American sprinters
def num_non_americans : ℕ := num_sprinters - num_americans

-- The question to prove: the number of ways the medals can be awarded if at most one American gets a medal
theorem num_ways_award_medals 
  (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = num_sprinters) (h2 : m = num_americans) 
  (h3 : k = num_non_americans) 
  (no_american : ℕ := k * (k - 1) * (k - 2)) 
  (one_american : ℕ := m * 3 * k * (k - 1)) 
  : no_american + one_american = 240 :=
sorry

end num_ways_award_medals_l256_256775


namespace ratio_mn_eq_x_plus_one_over_two_x_plus_one_l256_256999

theorem ratio_mn_eq_x_plus_one_over_two_x_plus_one (x : ℝ) (m n : ℝ) 
  (hx : x > 0) 
  (hmn : m * n ≠ 0) 
  (hineq : m * x > n * x + n) : 
  m / (m + n) = (x + 1) / (2 * x + 1) := 
by 
  sorry

end ratio_mn_eq_x_plus_one_over_two_x_plus_one_l256_256999


namespace larger_of_two_numbers_with_hcf_25_l256_256154

theorem larger_of_two_numbers_with_hcf_25 (a b : ℕ) (h_hcf: Nat.gcd a b = 25)
  (h_lcm_factors: 13 * 14 = (25 * 13 * 14) / (Nat.gcd a b)) :
  max a b = 350 :=
sorry

end larger_of_two_numbers_with_hcf_25_l256_256154


namespace pizza_boxes_sold_l256_256521

variables (P : ℕ) -- Representing the number of pizza boxes sold

def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2

def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def goal_amount : ℝ := 500
def more_needed : ℝ := 258
def current_amount : ℝ := goal_amount - more_needed

-- Total earnings calculation
def total_earnings : ℝ := (P : ℝ) * pizza_price + fries_sold * fries_price + soda_sold * soda_price

theorem pizza_boxes_sold (h : total_earnings P = current_amount) : P = 15 := 
by
  sorry

end pizza_boxes_sold_l256_256521


namespace chemistry_marks_more_than_physics_l256_256276

theorem chemistry_marks_more_than_physics (M P C x : ℕ) 
  (h1 : M + P = 32) 
  (h2 : (M + C) / 2 = 26) 
  (h3 : C = P + x) : 
  x = 20 := 
by
  sorry

end chemistry_marks_more_than_physics_l256_256276


namespace complete_the_square_l256_256614

-- Define the initial condition
def initial_eqn (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0

-- Theorem statement for completing the square
theorem complete_the_square (x : ℝ) : initial_eqn x → (x - 3)^2 = 4 :=
by sorry

end complete_the_square_l256_256614


namespace linear_inequality_solution_l256_256343

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l256_256343


namespace chinese_characters_digits_l256_256722

theorem chinese_characters_digits:
  ∃ (a b g s t : ℕ), -- Chinese characters represented by digits
    -- Different characters represent different digits
    a ≠ b ∧ a ≠ g ∧ a ≠ s ∧ a ≠ t ∧
    b ≠ g ∧ b ≠ s ∧ b ≠ t ∧
    g ≠ s ∧ g ≠ t ∧
    s ≠ t ∧
    -- Equation: 业步高 * 业步高 = 高升抬步高
    (a * 100 + b * 10 + g) * (a * 100 + b * 10 + g) = (g * 10000 + s * 1000 + t * 100 + b * 10 + g) :=
by {
  -- We need to prove that the number represented by "高升抬步高" is 50625.
  sorry
}

end chinese_characters_digits_l256_256722


namespace expand_polynomials_l256_256940

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l256_256940


namespace probability_x_lt_y_l256_256811

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l256_256811


namespace Jane_mom_jars_needed_l256_256414

theorem Jane_mom_jars_needed : 
  ∀ (total_tomatoes jar_capacity : ℕ), 
  total_tomatoes = 550 → 
  jar_capacity = 14 → 
  ⌈(total_tomatoes: ℚ) / jar_capacity⌉ = 40 := 
by 
  intros total_tomatoes jar_capacity htotal hcapacity
  sorry

end Jane_mom_jars_needed_l256_256414


namespace max_area_of_garden_l256_256683

theorem max_area_of_garden (L : ℝ) (hL : 0 ≤ L) :
  ∃ x y : ℝ, x + 2 * y = L ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y = L^2 / 8 :=
by
  sorry

end max_area_of_garden_l256_256683


namespace geometric_sequence_eighth_term_l256_256620

theorem geometric_sequence_eighth_term (a r : ℝ) (h1 : a * r ^ 3 = 12) (h2 : a * r ^ 11 = 3) : 
  a * r ^ 7 = 6 * Real.sqrt 2 :=
sorry

end geometric_sequence_eighth_term_l256_256620


namespace fish_estimation_l256_256015

noncomputable def number_caught := 50
noncomputable def number_marked_caught := 2
noncomputable def number_released := 30

theorem fish_estimation (N : ℕ) (h1 : number_caught = 50) 
  (h2 : number_marked_caught = 2) 
  (h3 : number_released = 30) :
  (number_marked_caught : ℚ) / number_caught = number_released / N → 
  N = 750 :=
by
  sorry

end fish_estimation_l256_256015


namespace arccos_half_eq_pi_div_three_l256_256868

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256868


namespace arccos_half_eq_pi_div_three_l256_256844

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256844


namespace sum_S10_equals_10_div_21_l256_256724

def a (n : ℕ) : ℚ := 1 / (4 * n^2 - 1)
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_S10_equals_10_div_21 : S 10 = 10 / 21 :=
by
  sorry

end sum_S10_equals_10_div_21_l256_256724


namespace max_area_of_triangle_l256_256720

noncomputable def max_triangle_area (v1 v2 v3 : ℝ) (S : ℝ) : Prop :=
  2 * S + Real.sqrt 3 * (v1 * v2 + v3) = 0 ∧ v3 = Real.sqrt 3 → S ≤ Real.sqrt 3 / 4

theorem max_area_of_triangle (v1 v2 v3 S : ℝ) :
  max_triangle_area v1 v2 v3 S :=
by
  sorry

end max_area_of_triangle_l256_256720


namespace quadratic_decreasing_l256_256533

-- Define the quadratic function and the condition a < 0
def quadratic_function (a x : ℝ) := a * x^2 - 2 * a * x + 1

-- Define the main theorem to be proven
theorem quadratic_decreasing (a m : ℝ) (ha : a < 0) : 
  (∀ x, x > m → quadratic_function a x < quadratic_function a (x+1)) ↔ m ≥ 1 :=
by
  sorry

end quadratic_decreasing_l256_256533


namespace monotonicity_of_f_intersection_points_of_tangent_l256_256386

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l256_256386


namespace change_in_expression_l256_256557

theorem change_in_expression {x b : ℝ} (hx : 0 ≤ b) : 
  let initial_expr := 2*x^2 + 5
  let increased_expr := 2*(x + b)^2 + 5
  let decreased_expr := 2*(x - b)^2 + 5
  in increased_expr - initial_expr = 4*x*b + 2*b^2 ∨ 
     decreased_expr - initial_expr = -4*x*b + 2*b^2 :=
by
  sorry

end change_in_expression_l256_256557


namespace andy_diana_weight_l256_256179

theorem andy_diana_weight :
  ∀ (a b c d : ℝ),
  a + b = 300 →
  b + c = 280 →
  c + d = 310 →
  a + d = 330 := by
  intros a b c d h₁ h₂ h₃
  -- Proof goes here
  sorry

end andy_diana_weight_l256_256179


namespace stratified_sampling_female_athletes_l256_256308

-- Conditions
def total_male_athletes : ℕ := 56
def total_female_athletes : ℕ := 42
def total_sample_size : ℕ := 28

-- Proportion of female athletes in stratified sampling
def proportion_female_athletes : ℚ := total_female_athletes / (total_male_athletes + total_female_athletes)
def expected_female_athletes_in_sample : ℕ := (proportion_female_athletes * total_sample_size).natAbs

-- The proof statement
theorem stratified_sampling_female_athletes :
  expected_female_athletes_in_sample = 12 :=
sorry

end stratified_sampling_female_athletes_l256_256308


namespace emily_lemon_juice_fraction_l256_256325

/-- 
Emily places 6 ounces of tea into a twelve-ounce cup and 6 ounces of honey into a second cup
of the same size. Then she adds 3 ounces of lemon juice to the second cup. Next, she pours half
the tea from the first cup into the second, mixes thoroughly, and then pours one-third of the
mixture in the second cup back into the first. 
Prove that the fraction of the mixture in the first cup that is lemon juice is 1/7.
--/
theorem emily_lemon_juice_fraction :
  let cup1_tea := 6
  let cup2_honey := 6
  let cup2_lemon_juice := 3
  let cup1_tea_transferred := cup1_tea / 2
  let cup1 := cup1_tea - cup1_tea_transferred
  let cup2 := cup2_honey + cup2_lemon_juice + cup1_tea_transferred
  let mix_ratio (x y : ℕ) := (x : ℚ) / (x + y)
  let cup1_after_transfer := cup1 + (cup2 / 3)
  let cup2_tea := cup1_tea_transferred
  let cup2_honey := cup2_honey
  let cup2_lemon_juice := cup2_lemon_juice
  let cup1_lemon_transferred := 1
  cup1_tea + (cup2 / 3) = 3 + (cup2_tea * (1 / 3)) + 1 + (cup2_honey * (1 / 3)) + cup2_lemon_juice / 3 →
  cup1 / (cup1 + cup2_honey) = 1/7 :=
sorry

end emily_lemon_juice_fraction_l256_256325


namespace parabola_line_intersection_l256_256727

/-- 
Given the parabola y^2 = -x and the line l: y = k(x + 1) intersect at points A and B,
(Ⅰ) Find the range of values for k;
(Ⅱ) Let O be the vertex of the parabola, prove that OA ⟂ OB.
-/
theorem parabola_line_intersection (k : ℝ) (A B : ℝ × ℝ)
  (hA : A.2 ^ 2 = -A.1) (hB : B.2 ^ 2 = -B.1)
  (hlineA : A.2 = k * (A.1 + 1)) (hlineB : B.2 = k * (B.1 + 1)) :
  (k ≠ 0) ∧ ((A.2 * B.2 = -1) → A.1 * B.1 * (A.2 * B.2) = -1) :=
by
  sorry

end parabola_line_intersection_l256_256727


namespace ant_meet_at_QS_is_five_l256_256457

-- Condition definitions
def triangle_sides (PQ QR RP : ℕ) : Prop := 
  PQ = 7 ∧ QR = 8 ∧ RP = 9

def ant_meeting_point (P Q R S : ℕ) : Prop :=
  ∃ (PQ QR RP : ℕ), triangle_sides PQ QR RP ∧ 
  (ant_meeting_distance PQ QR RP = 12)

-- Ant meeting distance function definition
def ant_meeting_distance (PQ QR RP : ℕ) : ℕ :=
  (PQ + QR + RP) / 2

-- Theorem statement expressing the problem
theorem ant_meet_at_QS_is_five (P Q R S : ℕ) 
  (h: triangle_sides 7 8 9) (a: ant_meeting_point P Q R S) : 
  ( ∃ QS : ℕ, QS = 5 ) :=
sorry

end ant_meet_at_QS_is_five_l256_256457


namespace replace_star_with_2x_l256_256995

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l256_256995


namespace tom_total_money_l256_256983

theorem tom_total_money :
  let initial_amount := 74
  let additional_amount := 86
  initial_amount + additional_amount = 160 :=
by
  let initial_amount := 74
  let additional_amount := 86
  show initial_amount + additional_amount = 160
  sorry

end tom_total_money_l256_256983


namespace arccos_half_eq_pi_div_3_l256_256910

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256910


namespace monotonicity_of_f_tangent_line_intersection_points_l256_256353

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l256_256353


namespace number_of_sides_l256_256229

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end number_of_sides_l256_256229


namespace quotient_of_fifths_l256_256547

theorem quotient_of_fifths : (2 / 5) / (1 / 5) = 2 := 
  by 
    sorry

end quotient_of_fifths_l256_256547


namespace arccos_half_eq_pi_div_three_l256_256901

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256901


namespace total_rainfall_over_3_days_l256_256576

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l256_256576


namespace probability_neither_square_nor_cube_l256_256636

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l256_256636


namespace comparison_of_products_l256_256786

def A : ℕ := 8888888888888888888 -- 19 digits, all 8's
def B : ℕ := 3333333333333333333333333333333333333333333333333333333333333333 -- 68 digits, all 3's
def C : ℕ := 4444444444444444444 -- 19 digits, all 4's
def D : ℕ := 6666666666666666666666666666666666666666666666666666666666666667 -- 68 digits, first 67 are 6's, last is 7

theorem comparison_of_products : C * D > A * B ∧ C * D - A * B = 4444444444444444444 := sorry

end comparison_of_products_l256_256786


namespace find_n_l256_256052

theorem find_n 
  (n : ℕ) (h₁ : n > 0) 
  (h₂ : ∃ (k : ℤ), (1/3 : ℚ) + (1/4 : ℚ) + (1/8 : ℚ) + 1/↑n = k) : 
  n = 24 :=
by
  sorry

end find_n_l256_256052


namespace triangle_area_l256_256462

theorem triangle_area : 
  let p1 := (3, 2)
  let p2 := (3, -4)
  let p3 := (12, 2)
  let height := |2 - (-4)|
  let base := |12 - 3|
  let area := (1 / 2) * base * height
  area = 27 := sorry

end triangle_area_l256_256462


namespace category_B_count_solution_hiring_probability_l256_256491

-- Definitions and conditions
def category_A_count : Nat := 12

def total_selected_housekeepers : Nat := 20
def category_B_selected_housekeepers : Nat := 16
def category_A_selected_housekeepers := total_selected_housekeepers - category_B_selected_housekeepers

-- The value of x
def category_B_count (x : Nat) : Prop :=
  (category_A_selected_housekeepers * x) / category_A_count = category_B_selected_housekeepers

-- Assertion for the value of x
theorem category_B_count_solution : category_B_count 48 :=
by sorry

-- Conditions for the second part of the problem
def remaining_category_A : Nat := 3
def remaining_category_B : Nat := 2
def total_remaining := remaining_category_A + remaining_category_B

def possible_choices := remaining_category_A * (remaining_category_A - 1) / 2 + remaining_category_A * remaining_category_B + remaining_category_B * (remaining_category_B - 1) / 2
def successful_choices := remaining_category_A * remaining_category_B

def probability (a b : Nat) := (successful_choices % total_remaining) / (possible_choices % total_remaining)

-- Assertion for the probability
theorem hiring_probability : probability remaining_category_A remaining_category_B = 3 / 5 :=
by sorry

end category_B_count_solution_hiring_probability_l256_256491


namespace fixed_errors_correct_l256_256338

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l256_256338


namespace gcd_of_polynomial_l256_256208

theorem gcd_of_polynomial (b : ℕ) (hb : b % 780 = 0) : Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 65) b = 65 := by
  sorry

end gcd_of_polynomial_l256_256208


namespace number_exceeds_twenty_percent_by_forty_l256_256791

theorem number_exceeds_twenty_percent_by_forty (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 :=
by
  sorry

end number_exceeds_twenty_percent_by_forty_l256_256791


namespace probability_face_not_red_is_five_sixths_l256_256134

-- Definitions based on the conditions
def total_faces : ℕ := 6
def green_faces : ℕ := 3
def blue_faces : ℕ := 2
def red_faces : ℕ := 1

-- Definition for the probability calculation
def probability_not_red (total : ℕ) (not_red : ℕ) : ℚ := not_red / total

-- The main statement to prove
theorem probability_face_not_red_is_five_sixths :
  probability_not_red total_faces (green_faces + blue_faces) = 5 / 6 :=
by sorry

end probability_face_not_red_is_five_sixths_l256_256134


namespace arrangement_plans_l256_256975

-- Definition of the problem conditions
def numChineseTeachers : ℕ := 2
def numMathTeachers : ℕ := 4
def numTeachersPerSchool : ℕ := 3

-- Definition of the problem statement
theorem arrangement_plans
  (c : ℕ) (m : ℕ) (s : ℕ)
  (h1 : numChineseTeachers = c)
  (h2 : numMathTeachers = m)
  (h3 : numTeachersPerSchool = s)
  (h4 : ∀ a b : ℕ, a + b = numChineseTeachers → a = 1 ∧ b = 1)
  (h5 : ∀ a b : ℕ, a + b = numMathTeachers → a = 2 ∧ b = 2) :
  (c * (1 / 2 * m * (m - 1) / 2)) = 12 :=
sorry

end arrangement_plans_l256_256975


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l256_256379

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l256_256379


namespace arthur_hot_dogs_first_day_l256_256028

theorem arthur_hot_dogs_first_day (H D n : ℕ) (h₀ : D = 1)
(h₁ : 3 * H + n = 10)
(h₂ : 2 * H + 3 * D = 7) : n = 4 :=
by sorry

end arthur_hot_dogs_first_day_l256_256028


namespace right_triangle_area_l256_256443

theorem right_triangle_area (a b c : ℕ) (habc : a = 3 ∧ b = 4 ∧ c = 5) : 
  (a * a + b * b = c * c) → 
  1 / 2 * (a * b) = 6 :=
by
  sorry

end right_triangle_area_l256_256443


namespace remaining_animals_l256_256128
open Nat

theorem remaining_animals (dogs : ℕ) (cows : ℕ)
  (h1 : cows = 2 * dogs)
  (h2 : cows = 184) :
  let cows_sold := cows / 4 in
  let remaining_cows := cows - cows_sold in
  let dogs_sold := 3 * dogs / 4 in
  let remaining_dogs := dogs - dogs_sold in
  remaining_cows + remaining_dogs = 161 :=
by
  sorry

end remaining_animals_l256_256128


namespace range_of_a_l256_256404

theorem range_of_a (a : ℝ) :
  (∀ x, (x - 2)/5 + 2 ≤ x - 4/5 ∨ x ≤ a) → a ≥ 3 :=
by
  sorry

end range_of_a_l256_256404


namespace rectangle_count_l256_256196

theorem rectangle_count (h_lines v_lines : Finset ℕ) (h_card : h_lines.card = 5) (v_card : v_lines.card = 5) :
  ∃ (n : ℕ), n = (h_lines.choose 2).card * (v_lines.choose 2).card ∧ n = 100 :=
by
  sorry 

end rectangle_count_l256_256196


namespace range_of_expression_l256_256064

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ∧ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ≤ 22 + 4 * Real.sqrt 5 :=
sorry

end range_of_expression_l256_256064


namespace is_composite_1010_pattern_l256_256256

theorem is_composite_1010_pattern (k : ℕ) (h : k ≥ 2) : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (1010^k + 101 = a * b)) :=
  sorry

end is_composite_1010_pattern_l256_256256


namespace maddie_watched_8_episodes_l256_256588

def minutes_per_episode : ℕ := 44
def minutes_monday : ℕ := 138
def minutes_tuesday_wednesday : ℕ := 0
def minutes_thursday : ℕ := 21
def episodes_friday : ℕ := 2
def minutes_per_episode_friday := episodes_friday * minutes_per_episode
def minutes_weekend : ℕ := 105
def total_minutes := minutes_monday + minutes_tuesday_wednesday + minutes_thursday + minutes_per_episode_friday + minutes_weekend
def answer := total_minutes / minutes_per_episode

theorem maddie_watched_8_episodes : answer = 8 := by
  sorry

end maddie_watched_8_episodes_l256_256588


namespace total_pencils_l256_256183

theorem total_pencils (pencils_per_box : ℕ) (friends : ℕ) (total_pencils : ℕ) : 
  pencils_per_box = 7 ∧ friends = 5 → total_pencils = pencils_per_box + friends * pencils_per_box → total_pencils = 42 :=
by
  intros h1 h2
  sorry

end total_pencils_l256_256183


namespace fisherman_total_fish_l256_256161

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l256_256161


namespace triangle_base_length_l256_256615

/-
Theorem: Given a triangle with height 5.8 meters and area 24.36 square meters,
the length of the base is 8.4 meters.
-/

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) :
  h = 5.8 ∧ A = 24.36 ∧ A = (b * h) / 2 → b = 8.4 :=
by
  sorry

end triangle_base_length_l256_256615


namespace embankment_height_bounds_l256_256494

theorem embankment_height_bounds
  (a : ℝ) (b : ℝ) (h : ℝ)
  (a_eq : a = 5)
  (b_lower_bound : 2 ≤ b)
  (vol_lower_bound : 400 ≤ (25 * (a^2 - b^2)))
  (vol_upper_bound : (25 * (a^2 - b^2)) ≤ 500) :
  1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by
  sorry

end embankment_height_bounds_l256_256494


namespace circle_B_area_l256_256513

theorem circle_B_area
  (r R : ℝ)
  (h1 : ∀ (x : ℝ), x = 5)  -- derived from r = 5
  (h2 : R = 2 * r)
  (h3 : 25 * Real.pi = Real.pi * r^2)
  (h4 : R = 10)  -- derived from diameter relation
  : ∃ A_B : ℝ, A_B = 100 * Real.pi :=
by
  sorry

end circle_B_area_l256_256513


namespace age_problem_l256_256405

variables (K M A B : ℕ)

theorem age_problem
  (h1 : K + 7 = 3 * M)
  (h2 : M = 5)
  (h3 : A + B = 2 * M + 4)
  (h4 : A = B - 3)
  (h5 : K + B = M + 9) :
  K = 8 ∧ M = 5 ∧ B = 6 ∧ A = 3 :=
sorry

end age_problem_l256_256405


namespace oz_lost_words_count_l256_256286
-- We import the necessary library.

-- Define the context.
def total_letters := 69
def forbidden_letter := 7

-- Define function to calculate lost words when a specific letter is forbidden.
def lost_words (total_letters : ℕ) (forbidden_letter : ℕ) : ℕ :=
  let one_letter_lost := 1
  let two_letter_lost := 2 * (total_letters - 1)
  one_letter_lost + two_letter_lost

-- State the theorem.
theorem oz_lost_words_count :
  lost_words total_letters forbidden_letter = 139 :=
by
  sorry

end oz_lost_words_count_l256_256286


namespace min_distance_l256_256173

theorem min_distance (W : ℝ) (b : ℝ) (n : ℕ) (H_W : W = 42) (H_b : b = 3) (H_n : n = 8) : 
  ∃ d : ℝ, d = 2 ∧ (W - n * b = 9 * d) := 
by 
  -- Here should go the proof
  sorry

end min_distance_l256_256173


namespace oil_level_drop_l256_256808

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25
noncomputable def truck_tank_radius : ℝ := 7
noncomputable def truck_tank_height : ℝ := 10

noncomputable def π : ℝ := Real.pi
noncomputable def truck_tank_volume := π * truck_tank_radius^2 * truck_tank_height
noncomputable def stationary_tank_area := π * stationary_tank_radius^2

theorem oil_level_drop (volume_truck: ℝ) (area_stationary: ℝ) : volume_truck = 490 * π → area_stationary = π * 10000 → (volume_truck / area_stationary) = 0.049 :=
by
  intros h1 h2
  sorry

end oil_level_drop_l256_256808


namespace total_money_raised_l256_256301

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l256_256301


namespace red_balls_probability_l256_256485

-- Definitions of the conditions
def red_balls : ℕ := 4
def blue_balls : ℕ := 5
def green_balls : ℕ := 3
def total_balls : ℕ := red_balls + blue_balls + green_balls
def balls_picked : ℕ := 3

-- Probability calculation functions
def first_draw_prob : ℚ := red_balls / total_balls
def second_draw_prob : ℚ := (red_balls - 1) / (total_balls - 1)
def third_draw_prob : ℚ := (red_balls - 2) / (total_balls - 2)

-- Combined probability
def combined_prob : ℚ := first_draw_prob * second_draw_prob * third_draw_prob

-- Lean statement to prove the probability
theorem red_balls_probability : combined_prob = 1 / 55 :=
by
  sorry

end red_balls_probability_l256_256485


namespace total_length_of_XYZ_l256_256116

noncomputable def length_XYZ : ℝ :=
  let length_X := 2 + 2 + 2 * Real.sqrt 2
  let length_Y := 3 + 2 * Real.sqrt 2
  let length_Z := 3 + 3 + Real.sqrt 10
  length_X + length_Y + length_Z

theorem total_length_of_XYZ :
  length_XYZ = 13 + 4 * Real.sqrt 2 + Real.sqrt 10 :=
by
  sorry

end total_length_of_XYZ_l256_256116


namespace problem1_problem2_l256_256289

open Nat

-- Definitions used based on conditions
def A (n m : ℕ) : ℕ := factorial n / factorial (n - m)
def C (n r : ℕ) : ℕ := n.choose r

-- Problem 1: Prove that 2 * A 8 5 + 7 * A 8 4 / A 8 8 - A 9 5 = 1 / 15
theorem problem1 : (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := sorry

-- Problem 2: Prove that C 200 198 + C 200 196 + 2 * C 200 197 = 67331650
theorem problem2 : C 200 198 + C 200 196 + 2 * C 200 197 = 67331650 := sorry

end problem1_problem2_l256_256289


namespace elephants_ratio_l256_256444

theorem elephants_ratio (x : ℝ) (w : ℝ) (g : ℝ) (total : ℝ) :
  w = 70 →
  total = 280 →
  g = x * w →
  w + g = total →
  x = 3 :=
by 
  intros h1 h2 h3 h4
  sorry

end elephants_ratio_l256_256444


namespace arithmetic_sequence_general_term_l256_256451

theorem arithmetic_sequence_general_term
    (a : ℕ → ℤ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_mean_26 : (a 2 + a 6) / 2 = 5)
    (h_mean_37 : (a 3 + a 7) / 2 = 7) :
    ∀ n, a n = 2 * n - 3 := 
by
  sorry

end arithmetic_sequence_general_term_l256_256451


namespace valentines_count_l256_256596

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 52) : x * y = 108 :=
by sorry

end valentines_count_l256_256596


namespace num_ways_placing_2015_bishops_l256_256063

-- Define the concept of placing bishops on a 2 x n chessboard without mutual attacks
def max_bishops (n : ℕ) : ℕ := n

-- Define the calculation of the number of ways to place these bishops
def num_ways_to_place_bishops (n : ℕ) : ℕ := 2 ^ n

-- The proof statement for our specific problem
theorem num_ways_placing_2015_bishops :
  num_ways_to_place_bishops 2015 = 2 ^ 2015 :=
by
  sorry

end num_ways_placing_2015_bishops_l256_256063


namespace total_sand_weight_is_34_l256_256047

-- Define the conditions
def eden_buckets : ℕ := 4
def mary_buckets : ℕ := eden_buckets + 3
def iris_buckets : ℕ := mary_buckets - 1
def weight_per_bucket : ℕ := 2

-- Define the total weight calculation
def total_buckets : ℕ := eden_buckets + mary_buckets + iris_buckets
def total_weight : ℕ := total_buckets * weight_per_bucket

-- The proof statement
theorem total_sand_weight_is_34 : total_weight = 34 := by
  sorry

end total_sand_weight_is_34_l256_256047


namespace angle_2016_in_third_quadrant_l256_256284

def quadrant (θ : ℤ) : ℤ :=
  let angle := θ % 360
  if 0 ≤ angle ∧ angle < 90 then 1
  else if 90 ≤ angle ∧ angle < 180 then 2
  else if 180 ≤ angle ∧ angle < 270 then 3
  else 4

theorem angle_2016_in_third_quadrant : 
  quadrant 2016 = 3 := 
by
  sorry

end angle_2016_in_third_quadrant_l256_256284


namespace no_such_function_exists_l256_256741

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := 
by
  sorry

end no_such_function_exists_l256_256741


namespace total_apples_l256_256078

-- Define the number of apples given to each person
def apples_per_person : ℝ := 15.0

-- Define the number of people
def number_of_people : ℝ := 3.0

-- Goal: Prove that the total number of apples is 45.0
theorem total_apples : apples_per_person * number_of_people = 45.0 := by
  sorry

end total_apples_l256_256078


namespace round_robin_pairing_possible_l256_256820

def players : Set String := {"A", "B", "C", "D", "E", "F"}

def is_pairing (pairs : List (String × String)) : Prop :=
  ∀ (p : String × String), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ players ∧ p.2 ∈ players

def unique_pairs (rounds : List (List (String × String))) : Prop :=
  ∀ r, r ∈ rounds → is_pairing r ∧ (∀ p1 p2, p1 ∈ r → p2 ∈ r → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

def all_players_paired (rounds : List (List (String × String))) : Prop :=
  ∀ p, p ∈ players →
  (∀ q, q ∈ players → p ≠ q → 
    (∃ r, r ∈ rounds ∧ (p,q) ∈ r ∨ (q,p) ∈ r))

theorem round_robin_pairing_possible : 
  ∃ rounds, List.length rounds = 5 ∧ unique_pairs rounds ∧ all_players_paired rounds :=
  sorry

end round_robin_pairing_possible_l256_256820


namespace ratio_of_voters_l256_256737

open Real

theorem ratio_of_voters (X Y : ℝ) (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) : X / Y = 2 :=
by
  sorry

end ratio_of_voters_l256_256737


namespace rationalize_denominator_l256_256601

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = (Real.sqrt 3) / 5 :=
by
  sorry

end rationalize_denominator_l256_256601


namespace fourth_power_nested_sqrt_l256_256316

noncomputable def nested_sqrt := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end fourth_power_nested_sqrt_l256_256316


namespace junk_items_count_l256_256252

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end junk_items_count_l256_256252


namespace probability_three_heads_l256_256667

noncomputable def fair_coin_flip: ℝ := 1 / 2

theorem probability_three_heads :
  (fair_coin_flip * fair_coin_flip * fair_coin_flip) = 1 / 8 :=
by
  -- proof would go here
  sorry

end probability_three_heads_l256_256667


namespace solve_for_x_l256_256613

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 8 * x - 16 = 0) : x = 4 / 3 :=
by
  sorry

end solve_for_x_l256_256613


namespace line_interparabola_length_l256_256679

theorem line_interparabola_length :
  (∀ (x y : ℝ), y = x - 2 → y^2 = 4 * x) →
  ∃ (A B : ℝ × ℝ), (∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2)) →
  (dist A B = 4 * Real.sqrt 6) :=
by
  intros
  sorry

end line_interparabola_length_l256_256679


namespace second_number_value_l256_256955

theorem second_number_value
  (a b : ℝ)
  (h1 : a * (a - 6) = 7)
  (h2 : b * (b - 6) = 7)
  (h3 : a ≠ b)
  (h4 : a + b = 6) :
  b = 7 := by
sorry

end second_number_value_l256_256955


namespace fruit_box_assignment_proof_l256_256025

-- Definitions of the boxes with different fruits
inductive Fruit | Apple | Pear | Orange | Banana
open Fruit

-- Define a function representing the placement of fruits in the boxes
def box_assignment := ℕ → Fruit

-- Conditions based on the problem statement
def conditions (assign : box_assignment) : Prop :=
  assign 1 ≠ Orange ∧
  assign 2 ≠ Pear ∧
  (assign 1 = Banana → assign 3 ≠ Apple ∧ assign 3 ≠ Pear) ∧
  assign 4 ≠ Apple

-- The correct assignment of fruits to boxes
def correct_assignment (assign : box_assignment) : Prop :=
  assign 1 = Banana ∧
  assign 2 = Apple ∧
  assign 3 = Orange ∧
  assign 4 = Pear

-- Theorem statement
theorem fruit_box_assignment_proof : ∃ assign : box_assignment, conditions assign ∧ correct_assignment assign :=
sorry

end fruit_box_assignment_proof_l256_256025


namespace f_not_monotonic_l256_256717

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x:ℝ)) = -f x

def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ¬ ( (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x) )

variable (f : ℝ → ℝ)

axiom periodicity : ∀ x, f (x + 3/2) = -f x 
axiom odd_shifted : is_odd_function (λ x => f (x - 3/4))

theorem f_not_monotonic : is_not_monotonic f := by
  sorry

end f_not_monotonic_l256_256717


namespace harrys_morning_routine_time_l256_256049

theorem harrys_morning_routine_time :
  (15 + 20 + 25 + 2 * 15 = 90) :=
by
  sorry

end harrys_morning_routine_time_l256_256049


namespace cube_root_power_simplify_l256_256607

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l256_256607


namespace find_y_perpendicular_l256_256346

theorem find_y_perpendicular (y : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (2, y))
  (ha : a = (2, 1))
  (h_perp : (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0) :
  y = -4 :=
sorry

end find_y_perpendicular_l256_256346


namespace cost_price_radio_l256_256439

theorem cost_price_radio (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1305) 
  (h2 : loss_percentage = 0.13) 
  (h3 : SP = C * (1 - loss_percentage)) :
  C = 1500 := 
by 
  sorry

end cost_price_radio_l256_256439


namespace circles_tangent_radii_product_eq_l256_256071

/-- Given two circles that pass through a fixed point \(M(x_1, y_1)\)
    and are tangent to both the x-axis and y-axis, with radii \(r_1\) and \(r_2\),
    prove that \(r_1 r_2 = x_1^2 + y_1^2\). -/
theorem circles_tangent_radii_product_eq (x1 y1 r1 r2 : ℝ)
  (h1 : (∃ (a : ℝ), ∃ (circle1 : ℝ → ℝ → ℝ), ∀ x y, circle1 x y = (x - a)^2 + (y - a)^2 - r1^2)
    ∧ (∃ (b : ℝ), ∃ (circle2 : ℝ → ℝ → ℝ), ∀ x y, circle2 x y = (x - b)^2 + (y - b)^2 - r2^2))
  (hm1 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r1)^2 + (p.snd - r1)^2 = r1^2 })
  (hm2 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r2)^2 + (p.snd - r2)^2 = r2^2 }) :
  r1 * r2 = x1^2 + y1^2 := sorry

end circles_tangent_radii_product_eq_l256_256071


namespace find_x_for_parallel_vectors_l256_256392

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4, x)
def b : ℝ × ℝ := (-4, 4)

-- Define parallelism condition for two 2D vectors
def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the main theorem statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : are_parallel (a x) b) : x = -4 :=
by sorry

end find_x_for_parallel_vectors_l256_256392


namespace average_star_rating_is_four_l256_256262

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l256_256262


namespace how_many_women_left_l256_256574

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l256_256574


namespace old_man_gold_coins_l256_256312

theorem old_man_gold_coins (x y : ℕ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := 
sorry

end old_man_gold_coins_l256_256312


namespace volume_of_tetrahedron_is_zero_l256_256947

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

-- Definition of the points as given in the problem
def P1 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci n, fibonacci (n + 1), fibonacci (n + 2))
def P2 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 3), fibonacci (n + 4), fibonacci (n + 5))
def P3 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 6), fibonacci (n + 7), fibonacci (n + 8))
def P4 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 9), fibonacci (n + 10), fibonacci (n + 11))

-- Function to calculate the volume of the tetrahedron
noncomputable def tetrahedron_volume (v1 v2 v3 v4 : ℝ × ℝ × ℝ) : ℝ :=
  let matrix := λ i j, match (i, j) with
  | (0, 0) => v2.1 - v1.1 | (0, 1) => v2.2 - v1.2 | (0, 2) => v2.3 - v1.3
  | (1, 0) => v3.1 - v1.1 | (1, 1) => v3.2 - v1.2 | (1, 2) => v3.3 - v1.3
  | (2, 0) => v4.1 - v1.1 | (2, 1) => v4.2 - v1.2 | (2, 2) => v4.3 - v1.3
  | _ => 0 end
  (1/6) * (matrix.det).abs

-- Statement of the theorem
theorem volume_of_tetrahedron_is_zero (n : ℕ) : 
  tetrahedron_volume (P1 n) (P2 n) (P3 n) (P4 n) = 0 :=
sorry

end volume_of_tetrahedron_is_zero_l256_256947


namespace avoid_loss_maximize_profit_max_profit_per_unit_l256_256490

-- Definitions of the functions as per problem conditions
noncomputable def C (x : ℝ) : ℝ := 2 + x
noncomputable def R (x : ℝ) : ℝ := if x ≤ 4 then 4 * x - (1 / 2) * x^2 - (1 / 2) else 7.5
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Proof statements

-- 1. Range to avoid loss
theorem avoid_loss (x : ℝ) : 1 ≤ x ∧ x ≤ 5.5 ↔ L x ≥ 0 :=
by
  sorry

-- 2. Production to maximize profit
theorem maximize_profit (x : ℝ) : x = 3 ↔ ∀ y, L y ≤ L 3 :=
by
  sorry

-- 3. Maximum profit per unit selling price
theorem max_profit_per_unit (x : ℝ) : x = 3 ↔ (R 3 / 3 = 2.33) :=
by
  sorry

end avoid_loss_maximize_profit_max_profit_per_unit_l256_256490


namespace pants_price_100_l256_256990

-- Define the variables and conditions
variables (x y : ℕ)

-- Define the prices according to the conditions
def coat_price_pants := x + 340
def coat_price_shoes_pants := y + x + 180
def total_price := (coat_price_pants x) + x + y

-- The theorem to prove
theorem pants_price_100 (h1: coat_price_pants x = coat_price_shoes_pants x y) (h2: total_price x y = 700) : x = 100 :=
sorry

end pants_price_100_l256_256990


namespace rate_of_current_in_river_l256_256101

theorem rate_of_current_in_river (b c : ℝ) (h1 : 4 * (b + c) = 24) (h2 : 6 * (b - c) = 24) : c = 1 := by
  sorry

end rate_of_current_in_river_l256_256101


namespace arccos_half_eq_pi_div_three_l256_256903

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256903


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256876

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256876


namespace sum_of_roots_l256_256713

theorem sum_of_roots : 
  let a := 1
  let b := 2001
  let c := -2002
  ∀ x y: ℝ, (x^2 + b*x + c = 0) ∧ (y^2 + b*y + c = 0) -> (x + y = -b) :=
by
  sorry

end sum_of_roots_l256_256713


namespace smallest_k_l256_256056

def u (n : ℕ) : ℕ := n^4 + 3 * n^2 + 2

def delta (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => u
  | k+1 => fun n => delta k u (n+1) - delta k u n

theorem smallest_k (n : ℕ) : ∃ k, (forall m, delta k u m = 0) ∧ 
                            (forall j, (∀ m, delta j u m = 0) → j ≥ k) := sorry

end smallest_k_l256_256056


namespace min_max_expression_l256_256796

theorem min_max_expression (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 19) (h2 : b^2 + b * c + c^2 = 19) :
  ∃ (min_val max_val : ℝ), 
    min_val = 0 ∧ max_val = 57 ∧ 
    (∀ x, x = c^2 + c * a + a^2 → min_val ≤ x ∧ x ≤ max_val) :=
by sorry

end min_max_expression_l256_256796


namespace arccos_half_eq_pi_div_3_l256_256907

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256907


namespace fare_midpoint_to_b_l256_256176

-- Define the conditions
def initial_fare : ℕ := 5
def initial_distance : ℕ := 2
def additional_fare_per_km : ℕ := 2
def total_fare : ℕ := 35
def walked_distance_meters : ℕ := 800

-- Define the correct answer
def fare_from_midpoint_to_b : ℕ := 19

-- Prove that the fare from the midpoint between A and B to B is 19 yuan
theorem fare_midpoint_to_b (y : ℝ) (h1 : 16.8 < y ∧ y ≤ 17) : 
  let half_distance := y / 2
  let total_taxi_distance := half_distance - 2
  let total_additional_fare := ⌈total_taxi_distance⌉ * additional_fare_per_km
  initial_fare + total_additional_fare = fare_from_midpoint_to_b := 
by
  sorry

end fare_midpoint_to_b_l256_256176


namespace eighth_term_sum_of_first_15_terms_l256_256784

-- Given definitions from the conditions
def a1 : ℚ := 5
def a30 : ℚ := 100
def n8 : ℕ := 8
def n15 : ℕ := 15
def n30 : ℕ := 30

-- Formulate the arithmetic sequence properties
def common_difference : ℚ := (a30 - a1) / (n30 - 1)

def nth_term (n : ℕ) : ℚ :=
  a1 + (n - 1) * common_difference

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * common_difference)

-- Statements to be proven
theorem eighth_term :
  nth_term n8 = 25 + 1/29 := by sorry

theorem sum_of_first_15_terms :
  sum_of_first_n_terms n15 = 393 + 2/29 := by sorry

end eighth_term_sum_of_first_15_terms_l256_256784


namespace arithmetic_example_l256_256665

theorem arithmetic_example : 4 * (9 - 6) - 8 = 4 := by
  sorry

end arithmetic_example_l256_256665


namespace inequality_sol_set_a_eq_2_inequality_sol_set_general_l256_256542

theorem inequality_sol_set_a_eq_2 :
  ∀ x : ℝ, (x^2 - x + 2 - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem inequality_sol_set_general (a : ℝ) :
  (∀ x : ℝ, (x^2 - x + a - a^2 ≤ 0) ↔
    (if a < 1/2 then a ≤ x ∧ x ≤ 1 - a
    else if a > 1/2 then 1 - a ≤ x ∧ x ≤ a
    else x = 1/2)) :=
by sorry

end inequality_sol_set_a_eq_2_inequality_sol_set_general_l256_256542


namespace day_of_week_2_2312_wednesday_l256_256818

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem day_of_week_2_2312_wednesday (birth_year : ℕ) (birth_day : String) 
  (h1 : birth_year = 2312 - 300)
  (h2 : birth_day = "Wednesday") :
  "Monday" = "Monday" :=
sorry

end day_of_week_2_2312_wednesday_l256_256818


namespace total_questions_attempted_l256_256738

/-- 
In an examination, a student scores 3 marks for every correct answer and loses 1 mark for
every wrong answer. He attempts some questions and secures 180 marks. The number of questions
he attempts correctly is 75. Prove that the total number of questions he attempts is 120. 
-/
theorem total_questions_attempted
  (marks_per_correct : ℕ := 3)
  (marks_lost_per_wrong : ℕ := 1)
  (total_marks : ℕ := 180)
  (correct_answers : ℕ := 75) :
  ∃ (wrong_answers total_questions : ℕ), 
    total_marks = (marks_per_correct * correct_answers) - (marks_lost_per_wrong * wrong_answers) ∧
    total_questions = correct_answers + wrong_answers ∧
    total_questions = 120 := 
by {
  sorry -- proof omitted
}

end total_questions_attempted_l256_256738


namespace periodic_odd_function_example_l256_256186

open Real

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem periodic_odd_function_example (f : ℝ → ℝ) 
  (h_odd : odd f) 
  (h_periodic : periodic f 2) : 
  f 1 + f 4 + f 7 = 0 := 
sorry

end periodic_odd_function_example_l256_256186


namespace rate_of_current_in_river_l256_256102

theorem rate_of_current_in_river (b c : ℝ) (h1 : 4 * (b + c) = 24) (h2 : 6 * (b - c) = 24) : c = 1 := by
  sorry

end rate_of_current_in_river_l256_256102


namespace florist_total_roses_l256_256167

-- Define the known quantities
def originalRoses : ℝ := 37.0
def firstPick : ℝ := 16.0
def secondPick : ℝ := 19.0

-- The theorem stating the total number of roses
theorem florist_total_roses : originalRoses + firstPick + secondPick = 72.0 :=
  sorry

end florist_total_roses_l256_256167


namespace arccos_of_half_eq_pi_over_three_l256_256840

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256840


namespace distinct_possible_lunches_l256_256508

namespace SchoolCafeteria

def main_courses : List String := ["Hamburger", "Veggie Burger", "Chicken Sandwich", "Pasta"]
def beverages_when_meat_free : List String := ["Water", "Soda"]
def beverages_when_meat : List String := ["Water"]
def snacks : List String := ["Apple Pie", "Fruit Cup"]

-- Count the total number of distinct possible lunches
def count_distinct_lunches : Nat :=
  let count_options (main_course : String) : Nat :=
    if main_course = "Hamburger" ∨ main_course = "Chicken Sandwich" then
      beverages_when_meat.length * snacks.length
    else
      beverages_when_meat_free.length * snacks.length
  (main_courses.map count_options).sum

theorem distinct_possible_lunches : count_distinct_lunches = 12 := by
  sorry

end SchoolCafeteria

end distinct_possible_lunches_l256_256508


namespace minimum_questions_needed_to_determine_birthday_l256_256133

def min_questions_to_determine_birthday : Nat := 9

theorem minimum_questions_needed_to_determine_birthday : min_questions_to_determine_birthday = 9 :=
sorry

end minimum_questions_needed_to_determine_birthday_l256_256133


namespace least_number_to_add_l256_256785

theorem least_number_to_add (a : ℕ) (p q r : ℕ) (h : a = 1076) (hp : p = 41) (hq : q = 59) (hr : r = 67) :
  ∃ k : ℕ, k = 171011 ∧ (a + k) % (lcm p (lcm q r)) = 0 :=
sorry

end least_number_to_add_l256_256785


namespace distance_city_A_C_l256_256699

-- Define the conditions
def starts_simultaneously (A : Prop) (Eddy Freddy : Prop) := Eddy ∧ Freddy
def travels (A B C : Prop) (Eddy Freddy : Prop) := Eddy → 3 = 3 ∧ Freddy → 4 = 4
def distance_AB (A B : Prop) := 600
def speed_ratio (Eddy_speed Freddy_speed : ℝ) := Eddy_speed / Freddy_speed = 1.7391304347826086

noncomputable def distance_AC (Eddy_time Freddy_time : ℝ) (Eddy_speed Freddy_speed : ℝ) 
  := (Eddy_speed / 1.7391304347826086) * Freddy_time

theorem distance_city_A_C 
  (A B C Eddy Freddy : Prop)
  (Eddy_time Freddy_time : ℝ) 
  (Eddy_speed effective_Freddy_speed : ℝ)
  (h1 : starts_simultaneously A Eddy Freddy)
  (h2 : travels A B C Eddy Freddy)
  (h3 : distance_AB A B = 600)
  (h4 : speed_ratio Eddy_speed effective_Freddy_speed)
  (h5 : Eddy_speed = 200)
  (h6 : effective_Freddy_speed = 115)
  : distance_AC Eddy_time Freddy_time Eddy_speed effective_Freddy_speed = 460 := 
  by sorry

end distance_city_A_C_l256_256699


namespace triangle_structure_twelve_rows_l256_256177

theorem triangle_structure_twelve_rows :
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  rods 12 + connectors 13 = 325 :=
by
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  sorry

end triangle_structure_twelve_rows_l256_256177


namespace second_neighbor_brought_less_l256_256123

theorem second_neighbor_brought_less (n1 n2 : ℕ) (htotal : ℕ) (h1 : n1 = 75) (h_total : n1 + n2 = 125) :
  n1 - n2 = 25 :=
by
  sorry

end second_neighbor_brought_less_l256_256123


namespace monotonicity_f_tangent_points_l256_256373

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l256_256373


namespace arccos_half_eq_pi_over_three_l256_256916

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256916


namespace solve_system_eq_l256_256762

theorem solve_system_eq (x y z : ℝ) 
  (h1 : x * y = 6 * (x + y))
  (h2 : x * z = 4 * (x + z))
  (h3 : y * z = 2 * (y + z)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = -24 ∧ y = 24 / 5 ∧ z = 24 / 7) :=
  sorry

end solve_system_eq_l256_256762


namespace assignment_plans_count_correct_l256_256238

open Finset
open Fintype

variables (roles : Finset (role)) [Fintype role] 
          (volunteers : Finset (volunteer)) [Fintype volunteer]

def role := {translation, tour_guiding, etiquette, driving}
def volunteer := {Zhang, Liu, Li, Song, Wang}

def valid_volunteer_role (v : volunteer) : Finset role :=
  match v with
  | Zhang => {translation, tour_guiding}
  | _ => role

noncomputable def count_assignment_plans : ℕ :=
  ((volunteers.powerset.filter (λ s, s.card = 4)).toFinset.card) *
  ((powerset (roles ∪ valid_volunteer_role(translation) ∪ valid_volunteer_role(tour_guiding))).toFinset.card)

theorem assignment_plans_count_correct : count_assignment_plans role volunteer = 36 :=
by sorry

end assignment_plans_count_correct_l256_256238


namespace min_value_2a_minus_ab_l256_256757

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_lt_11 : a < 11) (hb_lt_11 : b < 11) : 
  ∃ (min_val : ℤ), min_val = -80 ∧ ∀ x y : ℕ, 0 < x → 0 < y → x < 11 → y < 11 → 2 * x - x * y ≥ min_val :=
by
  use -80
  sorry

end min_value_2a_minus_ab_l256_256757


namespace inequality_proof_l256_256988

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)^2 + bc / (b + c)^2 + ca / (c + a)^2) + (3 * (a^2 + b^2 + c^2)) / (a + b + c)^2 ≥ 7 / 4 := 
by
  sorry

end inequality_proof_l256_256988


namespace arccos_half_is_pi_div_three_l256_256850

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256850


namespace jacob_current_age_l256_256695

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l256_256695


namespace balls_problem_l256_256973

noncomputable def red_balls_initial := 420
noncomputable def total_balls_initial := 600
noncomputable def percent_red_required := 60 / 100

theorem balls_problem :
  ∃ (x : ℕ), 420 - x = (3 / 5) * (600 - x) :=
by
  sorry

end balls_problem_l256_256973


namespace power_function_properties_l256_256441

theorem power_function_properties (α : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ α) 
    (h_point : f (1/2) = 2 ) :
    (∀ x : ℝ, f x = 1 / x) ∧ (∀ x : ℝ, 0 < x → (f x) < (f (x / 2))) ∧ (∀ x : ℝ, f (-x) = - (f x)) :=
by
  sorry

end power_function_properties_l256_256441


namespace projection_of_a_on_b_l256_256538

theorem projection_of_a_on_b (a b : ℝ) (θ : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 1)
  (hθ : θ = 60) : 
  (|a| * Real.cos (θ * Real.pi / 180)) = 1 := 
sorry

end projection_of_a_on_b_l256_256538


namespace find_k_l256_256394

theorem find_k {k : ℚ} (h : (3 : ℚ)^3 + 7 * (3 : ℚ)^2 + k * (3 : ℚ) + 23 = 0) : k = -113 / 3 :=
by
  sorry

end find_k_l256_256394


namespace ellipse_intersects_x_axis_at_four_l256_256505

theorem ellipse_intersects_x_axis_at_four
    (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 0))
    (h2 : f2 = (4, 0))
    (h3 : ∃ P : ℝ × ℝ, P = (1, 0) ∧ (dist P f1 + dist P f2 = 4)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ (dist Q f1 + dist Q f2 = 4) :=
sorry

end ellipse_intersects_x_axis_at_four_l256_256505


namespace problem_statement_l256_256730

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 14) : 6 * (x + 3) = 4.8 :=
sorry

end problem_statement_l256_256730


namespace gcd_14m_21n_126_l256_256222

theorem gcd_14m_21n_126 {m n : ℕ} (hm_pos : 0 < m) (hn_pos : 0 < n) (h_gcd : Nat.gcd m n = 18) : 
  Nat.gcd (14 * m) (21 * n) = 126 :=
by
  sorry

end gcd_14m_21n_126_l256_256222


namespace net_progress_l256_256016

def lost_yards : Int := 5
def gained_yards : Int := 7

theorem net_progress : gained_yards - lost_yards = 2 := 
by
  sorry

end net_progress_l256_256016


namespace cartesian_equation_of_l_range_of_m_l256_256096

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

def polar_line (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) + m = 0

theorem cartesian_equation_of_l (x y m : ℝ) (ρ θ : ℝ)
  (h₁ : polar_line ρ θ m)
  (h₂ : ρ = sqrt (x ^ 2 + y ^ 2))
  (h₃ : θ = Real.atan2 y x) :
  sqrt 3 * x + y + 2 * m = 0 := sorry

theorem range_of_m (x y t m : ℝ)
  (h₁ : (x, y) = parametric_curve t)
  (h₂ : sqrt 3 * x + y + 2 * m = 0) :
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 := sorry

end cartesian_equation_of_l_range_of_m_l256_256096


namespace genevieve_errors_fixed_l256_256335

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l256_256335


namespace proof_a_squared_plus_b_squared_l256_256963

theorem proof_a_squared_plus_b_squared (a b : ℝ) (h1 : (a + b) ^ 2 = 4) (h2 : a * b = 1) : a ^ 2 + b ^ 2 = 2 := 
by 
  sorry

end proof_a_squared_plus_b_squared_l256_256963


namespace eagles_points_l256_256559

theorem eagles_points (x y : ℕ) (h₁ : x + y = 82) (h₂ : x - y = 18) : y = 32 :=
sorry

end eagles_points_l256_256559


namespace triceratops_count_l256_256004

theorem triceratops_count (r t : ℕ) 
  (h_legs : 4 * r + 4 * t = 48) 
  (h_horns : 2 * r + 3 * t = 31) : 
  t = 7 := 
by 
  hint

/- The given conditions are:
1. Each rhinoceros has 2 horns.
2. Each triceratops has 3 horns.
3. Each animal has 4 legs.
4. There is a total of 31 horns.
5. There is a total of 48 legs.

Using these conditions and the equations derived from them, we need to prove that the number of triceratopses (t) is 7.
-/

end triceratops_count_l256_256004


namespace cost_of_producing_one_component_l256_256488

-- Define the conditions as constants
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_cost : ℕ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℕ := 195

-- Define the cost of producing one component as a variable
variable (C : ℕ)

/-- Prove that C must be less than or equal to 80 given the conditions -/
theorem cost_of_producing_one_component : 
  150 * C + 150 * shipping_cost_per_unit + fixed_monthly_cost ≤ 150 * selling_price_per_component → C ≤ 80 :=
by
  sorry

end cost_of_producing_one_component_l256_256488


namespace smallest_positive_angle_l256_256708

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l256_256708


namespace total_people_in_boats_l256_256124

theorem total_people_in_boats (bo_num : ℝ) (avg_people : ℝ) (bo_num_eq : bo_num = 3.0) (avg_people_eq : avg_people = 1.66666666699999) : ∃ total_people : ℕ, total_people = 6 := 
by
  sorry

end total_people_in_boats_l256_256124


namespace line_inters_curve_l256_256095

noncomputable def curve_C_x (t : ℝ) : ℝ := sqrt 3 * cos (2 * t)
noncomputable def curve_C_y (t : ℝ) : ℝ := 2 * sin t

theorem line_inters_curve (m : ℝ) :
  ∀ t : ℝ, (sqrt 3 * cos (2 * t)) + 2 * sin (t) + 2 * m = 0 →
  m ∈ Icc (-(19 / 12)) (5 / 2) :=
by {
  sorry
}

end line_inters_curve_l256_256095


namespace volume_ratio_john_emma_l256_256981

theorem volume_ratio_john_emma (r_J h_J r_E h_E : ℝ) (diam_J diam_E : ℝ)
  (h_diam_J : diam_J = 8) (h_r_J : r_J = diam_J / 2) (h_h_J : h_J = 15)
  (h_diam_E : diam_E = 10) (h_r_E : r_E = diam_E / 2) (h_h_E : h_E = 12) :
  (π * r_J^2 * h_J) / (π * r_E^2 * h_E) = 4 / 5 := by
  sorry

end volume_ratio_john_emma_l256_256981


namespace bob_shucks_240_oysters_in_2_hours_l256_256823

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l256_256823


namespace smallest_exponentiated_number_l256_256003

theorem smallest_exponentiated_number :
  127^8 < 63^10 ∧ 63^10 < 33^12 := 
by 
  -- Proof omitted
  sorry

end smallest_exponentiated_number_l256_256003


namespace mom_t_shirts_total_l256_256156

-- Definitions based on the conditions provided in the problem
def packages : ℕ := 71
def t_shirts_per_package : ℕ := 6

-- The statement to prove that the total number of white t-shirts is 426
theorem mom_t_shirts_total : packages * t_shirts_per_package = 426 := by sorry

end mom_t_shirts_total_l256_256156


namespace preimage_of_3_1_is_2_half_l256_256532

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * p.2, p.1 - 2 * p.2)

theorem preimage_of_3_1_is_2_half :
  (∃ x y : ℝ, f (x, y) = (3, 1) ∧ (x = 2 ∧ y = 1/2)) :=
by
  sorry

end preimage_of_3_1_is_2_half_l256_256532


namespace simplify_and_rationalize_l256_256602

theorem simplify_and_rationalize :
  ( (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) *
    (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 ) :=
by
  sorry

end simplify_and_rationalize_l256_256602


namespace boat_stream_speed_l256_256486

/-- A boat can travel with a speed of 22 km/hr in still water. 
If the speed of the stream is unknown, the boat takes 7 hours 
to go 189 km downstream. What is the speed of the stream?
-/
theorem boat_stream_speed (v : ℝ) : (22 + v) * 7 = 189 → v = 5 :=
by
  intro h
  sorry

end boat_stream_speed_l256_256486


namespace square_prism_surface_area_eq_volume_l256_256202

theorem square_prism_surface_area_eq_volume :
  ∃ (a b : ℕ), (a > 0) ∧ (2 * a^2 + 4 * a * b = a^2 * b)
  ↔ (a = 12 ∧ b = 3) ∨ (a = 8 ∧ b = 4) ∨ (a = 6 ∧ b = 6) ∨ (a = 5 ∧ b = 10) :=
by
  sorry

end square_prism_surface_area_eq_volume_l256_256202


namespace arccos_one_half_l256_256925

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256925


namespace reciprocals_of_roots_l256_256246

variable (a b c k : ℝ)

theorem reciprocals_of_roots (kr ks : ℝ) (h_eq : a * kr^2 + k * c * kr + b = 0) (h_eq2 : a * ks^2 + k * c * ks + b = 0) :
  (1 / (kr^2)) + (1 / (ks^2)) = (k^2 * c^2 - 2 * a * b) / (b^2) :=
by
  sorry

end reciprocals_of_roots_l256_256246


namespace det1_value_system_solution_l256_256758

-- Definitions from conditions
def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

-- Given values from the problem
def det1 := determinant (-2) 3 2 1

def a1 := 2
def b1 := -1
def c1 := 1
def a2 := 3
def b2 := 2
def c2 := 11

def D := determinant a1 b1 a2 b2
def Dx := determinant c1 b1 c2 b2
def Dy := determinant a1 c1 a2 c2

def x := Dx / D
def y := Dy / D

-- Theorem statements to be proved
theorem det1_value : det1 = -8 := 
by 
  unfold det1 determinant 
  sorry

theorem system_solution : x = 13 / 7 ∧ y = 19 / 7 := 
by 
  unfold x y Dx Dy D determinant 
  sorry

end det1_value_system_solution_l256_256758


namespace women_left_l256_256570

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l256_256570


namespace correct_option_d_l256_256723

variable (m t x1 x2 y1 y2 : ℝ)

theorem correct_option_d (h_m : m > 0)
  (h_y1 : y1 = m * x1^2 - 2 * m * x1 + 1)
  (h_y2 : y2 = m * x2^2 - 2 * m * x2 + 1)
  (h_x1 : t < x1 ∧ x1 < t + 1)
  (h_x2 : t + 2 < x2 ∧ x2 < t + 3)
  (h_t_geq1 : t ≥ 1) :
  y1 < y2 := sorry

end correct_option_d_l256_256723


namespace indigo_restaurant_average_rating_l256_256265

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l256_256265


namespace function_nonnegative_l256_256932

noncomputable def f (x : ℝ) := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem function_nonnegative (x : ℝ) : 
  (f x ≥ 0) ↔ (0 ≤ x ∧ x ≤ (1 / 7)) ∨ (3 ≤ x) :=
sorry

end function_nonnegative_l256_256932


namespace combined_distance_all_birds_two_seasons_l256_256482

-- Definition of the given conditions
def number_of_birds : Nat := 20
def distance_jim_to_disney : Nat := 50
def distance_disney_to_london : Nat := 60

-- The conclusion we need to prove
theorem combined_distance_all_birds_two_seasons :
  (distance_jim_to_disney + distance_disney_to_london) * number_of_birds = 2200 :=
by
  sorry

end combined_distance_all_birds_two_seasons_l256_256482


namespace smallest_positive_angle_l256_256707

theorem smallest_positive_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = 9 :=
sorry

end smallest_positive_angle_l256_256707


namespace women_left_l256_256571

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end women_left_l256_256571


namespace probability_non_square_non_cube_l256_256634

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l256_256634


namespace total_cost_is_correct_l256_256582

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l256_256582


namespace concentration_of_spirit_in_vessel_a_l256_256271

theorem concentration_of_spirit_in_vessel_a :
  ∀ (x : ℝ), 
    (∀ (v1 v2 v3 : ℝ), v1 * (x / 100) + v2 * (30 / 100) + v3 * (10 / 100) = 15 * (26 / 100) →
      v1 + v2 + v3 = 15 →
      v1 = 4 → v2 = 5 → v3 = 6 →
      x = 45) :=
by
  intros x v1 v2 v3 h h_volume h_v1 h_v2 h_v3
  sorry

end concentration_of_spirit_in_vessel_a_l256_256271


namespace number_of_valid_pairs_l256_256728

open Finset

-- Define set A
def A := {x | 1 ≤ x ∧ x ≤ 9}

-- Define set B as the Cartesian product of A with itself
def B := {ab : ℤ × ℤ | ab.1 ∈ A ∧ ab.2 ∈ A}

-- Define the function f
def f (ab : ℤ × ℤ) : ℤ := ab.1 * ab.2 - ab.1 - ab.2

-- The main theorem to be proved
theorem number_of_valid_pairs : (filter (λ ab, f ab = 11) B).card = 4 :=
by sorry

end number_of_valid_pairs_l256_256728


namespace arccos_one_half_is_pi_div_three_l256_256891

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256891


namespace identify_set_A_l256_256147

open Set

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem identify_set_A : A = {0, 1, 2} := 
by
  sorry

end identify_set_A_l256_256147


namespace books_distribution_l256_256801

noncomputable def distribution_ways : ℕ :=
  let books := 5
  let people := 4
  let combination := Nat.choose books 2
  let arrangement := Nat.factorial people
  combination * arrangement ^ people

theorem books_distribution : distribution_ways = 240 := by
  sorry

end books_distribution_l256_256801


namespace ratio_of_perimeters_of_squares_l256_256659

theorem ratio_of_perimeters_of_squares (A B : ℝ) (h: A / B = 16 / 25) : ∃ (P1 P2 : ℝ), P1 / P2 = 4 / 5 :=
by
  sorry

end ratio_of_perimeters_of_squares_l256_256659


namespace digit_A_value_l256_256197

theorem digit_A_value :
  ∃ (A : ℕ), A < 10 ∧ (45 % A = 0) ∧ (172 * 10 + A * 10 + 6) % 8 = 0 ∧
    ∀ (B : ℕ), B < 10 ∧ (45 % B = 0) ∧ (172 * 10 + B * 10 + 6) % 8 = 0 → B = A := sorry

end digit_A_value_l256_256197


namespace monotonicity_of_f_tangent_intersection_coordinates_l256_256369

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l256_256369


namespace range_of_m_l256_256086

open Real

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def polar_line (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

def cartesian_line (x y m : ℝ) : Prop :=
  sqrt 3 * x + y + 2 * m = 0

theorem range_of_m (t m : ℝ) :
  let C := parametric_curve t,
      x := C.1,
      y := C.2 in
  polar_line x y m →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by sorry

end range_of_m_l256_256086


namespace ratio_of_numbers_l256_256273

theorem ratio_of_numbers
  (greater less : ℕ)
  (h1 : greater = 64)
  (h2 : less = 32)
  (h3 : greater + less = 96)
  (h4 : ∃ k : ℕ, greater = k * less) :
  greater / less = 2 := by
  sorry

end ratio_of_numbers_l256_256273


namespace number_of_bricks_l256_256297

theorem number_of_bricks (b1_hours b2_hours combined_hours: ℝ) (reduction_rate: ℝ) (x: ℝ):
  b1_hours = 12 ∧ 
  b2_hours = 15 ∧ 
  combined_hours = 6 ∧ 
  reduction_rate = 15 ∧ 
  (combined_hours * ((x / b1_hours) + (x / b2_hours) - reduction_rate) = x) → 
  x = 1800 :=
by
  sorry

end number_of_bricks_l256_256297


namespace dave_winfield_home_runs_correct_l256_256502

def dave_winfield_home_runs (W : ℕ) : Prop :=
  755 = 2 * W - 175

theorem dave_winfield_home_runs_correct : dave_winfield_home_runs 465 :=
by
  -- The proof is omitted as requested
  sorry

end dave_winfield_home_runs_correct_l256_256502


namespace abs_x_lt_2_sufficient_but_not_necessary_l256_256671

theorem abs_x_lt_2_sufficient_but_not_necessary (x : ℝ) :
  (|x| < 2) → (x ^ 2 - x - 6 < 0) ∧ ¬ ((x ^ 2 - x - 6 < 0) → (|x| < 2)) := by
  sorry

end abs_x_lt_2_sufficient_but_not_necessary_l256_256671


namespace arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l256_256037

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  sorry

theorem arccos_sqrt_three_over_two_eq_pi_six : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l256_256037


namespace union_complement_B_A_equals_a_values_l256_256070

namespace ProofProblem

-- Define the universal set R as real numbers
def R := Set ℝ

-- Define set A and set B as per the conditions
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Complement of B in R
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}

-- Union of complement of B with A
def union_complement_B_A : Set ℝ := complement_B ∪ A

-- The first statement to be proven
theorem union_complement_B_A_equals : 
  union_complement_B_A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by
  sorry

-- Define set C as per the conditions
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- The second statement to be proven
theorem a_values (a : ℝ) (h : C a ⊆ B) : 
  2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end ProofProblem

end union_complement_B_A_equals_a_values_l256_256070


namespace num_rectangular_arrays_with_48_chairs_l256_256806

theorem num_rectangular_arrays_with_48_chairs : 
  ∃ n, (∀ (r c : ℕ), 2 ≤ r ∧ 2 ≤ c ∧ r * c = 48 → (n = 8 ∨ n = 0)) ∧ (n = 8) :=
by 
  sorry

end num_rectangular_arrays_with_48_chairs_l256_256806


namespace at_op_subtraction_l256_256076

-- Define the operation @
def at_op (x y : ℝ) : ℝ := 3 * x * y - 2 * x + y

-- Prove the problem statement
theorem at_op_subtraction :
  at_op 6 4 - at_op 4 6 = -6 :=
by
  sorry

end at_op_subtraction_l256_256076


namespace arccos_half_eq_pi_over_three_l256_256921

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256921


namespace holidays_per_month_l256_256311

theorem holidays_per_month (total_holidays : ℕ) (months_in_year : ℕ) (holidays_in_month : ℕ) 
    (h1 : total_holidays = 48) (h2 : months_in_year = 12) : holidays_in_month = 4 := 
by
  sorry

end holidays_per_month_l256_256311


namespace deductive_reasoning_option_l256_256821

inductive ReasoningType
| deductive
| inductive
| analogical

-- Definitions based on conditions
def option_A : ReasoningType := ReasoningType.inductive
def option_B : ReasoningType := ReasoningType.deductive
def option_C : ReasoningType := ReasoningType.inductive
def option_D : ReasoningType := ReasoningType.analogical

-- The main theorem to prove
theorem deductive_reasoning_option : option_B = ReasoningType.deductive :=
by sorry

end deductive_reasoning_option_l256_256821


namespace solution_set_leq_2_l256_256340

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l256_256340


namespace initial_students_count_eq_16_l256_256234

variable (n T : ℕ)
variable (h1 : (T:ℝ) / n = 62.5)
variable (h2 : ((T - 70):ℝ) / (n - 1) = 62.0)

theorem initial_students_count_eq_16 :
  n = 16 :=
by
  sorry

end initial_students_count_eq_16_l256_256234


namespace cartesian_from_polar_range_of_m_for_intersection_l256_256085

def polar_to_cartesian_l (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

-- Statement for part 1: Prove the Cartesian equation from the polar equation
theorem cartesian_from_polar (ρ θ m : ℝ) (h : polar_to_cartesian_l ρ θ m) : Prop :=
  ∃ x y, x = ρ * cos θ ∧ y = ρ * sin θ ∧ sqrt 3 * x + y + 2 * m = 0

-- Statement for part 2: Prove the range of m for intersection
theorem range_of_m_for_intersection (m : ℝ) :
  (∃ t : ℝ, let (x, y) := parametric_curve_C t in sqrt 3 * x + y + 2 * m = 0) ↔ -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by
  intros
  sorry

end cartesian_from_polar_range_of_m_for_intersection_l256_256085


namespace find_largest_even_integer_l256_256773

-- Define the sum of the first 30 positive even integers
def sum_first_30_even : ℕ := 2 * (30 * 31 / 2)

-- Assume five consecutive even integers and their sum
def consecutive_even_sum (m : ℕ) : ℕ := (m - 8) + (m - 6) + (m - 4) + (m - 2) + m

-- Statement of the theorem to be proven
theorem find_largest_even_integer : ∃ (m : ℕ), consecutive_even_sum m = sum_first_30_even ∧ m = 190 :=
by
  sorry

end find_largest_even_integer_l256_256773


namespace good_set_exists_l256_256460

def is_good_set (A : List ℕ) : Prop :=
  ∀ i ∈ A, i > 0 ∧ ∀ j ∈ A, i ≠ j → i ^ 2015 % (List.prod (A.erase i)) = 0

theorem good_set_exists (n : ℕ) (h : 3 ≤ n ∧ n ≤ 2015) : 
  ∃ A : List ℕ, A.length = n ∧ ∀ (a : ℕ), a ∈ A → a > 0 ∧ is_good_set A :=
sorry

end good_set_exists_l256_256460


namespace x_eq_1_sufficient_but_not_necessary_l256_256721

theorem x_eq_1_sufficient_but_not_necessary (x : ℝ) : x^2 - 3 * x + 2 = 0 → (x = 1 ↔ true) ∧ (x ≠ 1 → ∃ y : ℝ, y ≠ x ∧ y^2 - 3 * y + 2 = 0) :=
by
  sorry

end x_eq_1_sufficient_but_not_necessary_l256_256721


namespace fish_count_and_total_l256_256595

-- Definitions of each friend's number of fish
def max_fish : ℕ := 6
def sam_fish : ℕ := 3 * max_fish
def joe_fish : ℕ := 9 * sam_fish
def harry_fish : ℕ := 5 * joe_fish

-- Total number of fish for all friends combined
def total_fish : ℕ := max_fish + sam_fish + joe_fish + harry_fish

-- The theorem stating the problem and corresponding solution
theorem fish_count_and_total :
  max_fish = 6 ∧
  sam_fish = 3 * max_fish ∧
  joe_fish = 9 * sam_fish ∧
  harry_fish = 5 * joe_fish ∧
  total_fish = (max_fish + sam_fish + joe_fish + harry_fish) :=
by
  repeat { sorry }

end fish_count_and_total_l256_256595


namespace total_allocation_is_1800_l256_256239

-- Definitions from conditions.
def part_value (amount_food : ℕ) (ratio_food : ℕ) : ℕ :=
  amount_food / ratio_food

def total_parts (ratio_household : ℕ) (ratio_food : ℕ) (ratio_misc : ℕ) : ℕ :=
  ratio_household + ratio_food + ratio_misc

def total_amount (part_value : ℕ) (total_parts : ℕ) : ℕ :=
  part_value * total_parts

-- Given conditions
def ratio_household := 5
def ratio_food := 4
def ratio_misc := 1
def amount_food := 720

-- Prove the total allocation
theorem total_allocation_is_1800 
  (amount_food : ℕ := 720) 
  (ratio_household : ℕ := 5) 
  (ratio_food : ℕ := 4) 
  (ratio_misc : ℕ := 1) : 
  total_amount (part_value amount_food ratio_food) (total_parts ratio_household ratio_food ratio_misc) = 1800 :=
by
  sorry

end total_allocation_is_1800_l256_256239


namespace jonas_needs_35_pairs_of_socks_l256_256982

def JonasWardrobeItems (socks_pairs shoes_pairs pants_items tshirts : ℕ) : ℕ :=
  2 * socks_pairs + 2 * shoes_pairs + pants_items + tshirts

def itemsNeededToDouble (initial_items : ℕ) : ℕ :=
  2 * initial_items - initial_items

theorem jonas_needs_35_pairs_of_socks (socks_pairs : ℕ) 
                                      (shoes_pairs : ℕ) 
                                      (pants_items : ℕ) 
                                      (tshirts : ℕ) 
                                      (final_socks_pairs : ℕ) 
                                      (initial_items : ℕ := JonasWardrobeItems socks_pairs shoes_pairs pants_items tshirts) 
                                      (needed_items : ℕ := itemsNeededToDouble initial_items) 
                                      (needed_pairs_of_socks := needed_items / 2) : 
                                      final_socks_pairs = 35 :=
by
  sorry

end jonas_needs_35_pairs_of_socks_l256_256982


namespace community_service_arrangements_l256_256200

noncomputable def total_arrangements : ℕ :=
  let case1 := Nat.choose 6 3
  let case2 := 2 * Nat.choose 6 2
  let case3 := case2
  case1 + case2 + case3

theorem community_service_arrangements :
  total_arrangements = 80 :=
by
  sorry

end community_service_arrangements_l256_256200


namespace sequence_has_both_max_and_min_l256_256217

noncomputable def a_n (n : ℕ) : ℝ :=
  (n + 1) * ((-10 / 11) ^ n)

theorem sequence_has_both_max_and_min :
  ∃ (max min : ℝ) (N M : ℕ), 
    (∀ n : ℕ, a_n n ≤ max) ∧ (∀ n : ℕ, min ≤ a_n n) ∧ 
    (a_n N = max) ∧ (a_n M = min) := 
sorry

end sequence_has_both_max_and_min_l256_256217


namespace manolo_rate_change_after_one_hour_l256_256753

variable (masks_in_first_hour : ℕ)
variable (masks_in_remaining_time : ℕ)
variable (total_masks : ℕ)

-- Define conditions as Lean definitions
def first_hour_rate := 1 / 4  -- masks per minute
def remaining_time_rate := 1 / 6  -- masks per minute
def total_time := 4  -- hours
def masks_produced_in_first_hour (t : ℕ) := t * 15  -- t hours, 60 minutes/hour, at 15 masks/hour
def masks_produced_in_remaining_time (t : ℕ) := t * 10 -- (total_time - 1) hours, 60 minutes/hour, at 10 masks/hour

-- Main proof problem statement
theorem manolo_rate_change_after_one_hour :
  masks_in_first_hour = masks_produced_in_first_hour 1 →
  masks_in_remaining_time = masks_produced_in_remaining_time (total_time - 1) →
  total_masks = masks_in_first_hour + masks_in_remaining_time →
  (∃ t : ℕ, t = 1) :=
by
  -- Placeholder, proof not required
  sorry

end manolo_rate_change_after_one_hour_l256_256753


namespace genevieve_errors_fixed_l256_256336

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l256_256336


namespace odd_square_diff_div_by_eight_l256_256600

theorem odd_square_diff_div_by_eight (n p : ℤ) : 
  (2 * n + 1)^2 - (2 * p + 1)^2 % 8 = 0 := 
by 
-- Here we declare the start of the proof.
  sorry

end odd_square_diff_div_by_eight_l256_256600


namespace fisherman_catch_total_l256_256166

theorem fisherman_catch_total :
  let bass := 32
  let trout := bass / 4
  let blue_gill := bass * 2
in bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_total_l256_256166


namespace simplify_expression_l256_256603

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ( (x^6 - 1) / (3 * x^3) )^2) = Real.sqrt (x^12 + 7 * x^6 + 1) / (3 * x^3) :=
by sorry

end simplify_expression_l256_256603


namespace solve_for_x_minus_y_l256_256400

theorem solve_for_x_minus_y (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 24) : x - y = 4 := 
by
  sorry

end solve_for_x_minus_y_l256_256400


namespace x_percent_more_than_y_l256_256736

theorem x_percent_more_than_y (z : ℝ) (hz : z ≠ 0) (y : ℝ) (x : ℝ)
  (h1 : y = 0.70 * z) (h2 : x = 0.84 * z) :
  x = y + 0.20 * y :=
by
  -- proof goes here
  sorry

end x_percent_more_than_y_l256_256736


namespace common_points_range_for_m_l256_256087

open Real

noncomputable def polar_to_cartesian_equation (m : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, √3 * x + y + 2 * m = 0

theorem common_points_range_for_m :
  (∀ t : ℝ, ∃ m : ℝ, let x := √3 * cos (2 * t),
                         y := 2 * sin t in
                         polar_to_cartesian_equation m (x, y)) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 := 
sorry

end common_points_range_for_m_l256_256087


namespace quadratic_complete_square_l256_256231

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 7 * x + 6) = (x - 7 / 2) ^ 2 - 25 / 4 :=
by
  sorry

end quadratic_complete_square_l256_256231


namespace calc_1_calc_2_calc_3_calc_4_l256_256828

section
variables {m n x y z : ℕ} -- assuming all variables are natural numbers for simplicity.
-- Problem 1
theorem calc_1 : (2 * m * n) / (3 * m ^ 2) * (6 * m * n) / (5 * n) = (4 * n) / 5 :=
sorry

-- Problem 2
theorem calc_2 : (5 * x - 5 * y) / (3 * x ^ 2 * y) * (9 * x * y ^ 2) / (x ^ 2 - y ^ 2) = 
  15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem calc_3 : ((x ^ 3 * y ^ 2) / z) ^ 2 * ((y * z) / x ^ 2) ^ 3 = y ^ 7 * z :=
sorry

-- Problem 4
theorem calc_4 : (4 * x ^ 2 * y ^ 2) / (2 * x + y) * (4 * x ^ 2 + 4 * x * y + y ^ 2) / (2 * x + y) / 
  ((2 * x * y) * (2 * x - y) / (4 * x ^ 2 - y ^ 2)) = 4 * x ^ 2 * y + 2 * x * y ^ 2 :=
sorry
end

end calc_1_calc_2_calc_3_calc_4_l256_256828


namespace meet_time_first_l256_256802

-- Define the conditions
def track_length := 1800 -- track length in meters
def speed_A_kmph := 36   -- speed of A in kmph
def speed_B_kmph := 54   -- speed of B in kmph

-- Convert speeds from kmph to mps
def speed_A_mps := speed_A_kmph * 1000 / 3600 -- speed of A in meters per second
def speed_B_mps := speed_B_kmph * 1000 / 3600 -- speed of B in meters per second

-- Calculate lap times in seconds
def lap_time_A := track_length / speed_A_mps
def lap_time_B := track_length / speed_B_mps

-- Problem statement
theorem meet_time_first :
  Nat.lcm (Int.natAbs lap_time_A) (Int.natAbs lap_time_B) = 360 := by
  sorry

end meet_time_first_l256_256802


namespace number_of_groups_l256_256480

-- Define constants
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6

-- Define the theorem to be proven
theorem number_of_groups :
  (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end number_of_groups_l256_256480


namespace find_smallest_angle_l256_256710

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l256_256710


namespace solution_set_of_inequality_l256_256275

theorem solution_set_of_inequality :
  {x : ℝ | |x^2 - 2| < 2} = {x : ℝ | (x > -2 ∧ x < 0) ∨ (x > 0 ∧ x < 2)} :=
by
  sorry

end solution_set_of_inequality_l256_256275


namespace interest_first_year_l256_256132
-- Import the necessary math library

-- Define the conditions and proof the interest accrued in the first year
theorem interest_first_year :
  ∀ (P B₁ : ℝ) (r₂ increase_ratio: ℝ),
    P = 1000 →
    B₁ = 1100 →
    r₂ = 0.20 →
    increase_ratio = 0.32 →
    (B₁ - P) = 100 :=
by
  intros P B₁ r₂ increase_ratio P_def B₁_def r₂_def increase_ratio_def
  sorry

end interest_first_year_l256_256132


namespace initial_cost_renting_car_l256_256005

theorem initial_cost_renting_car
  (initial_cost : ℝ)
  (miles_monday : ℝ := 620)
  (miles_thursday : ℝ := 744)
  (cost_per_mile : ℝ := 0.50)
  (total_spent : ℝ := 832)
  (total_miles : ℝ := miles_monday + miles_thursday)
  (expected_initial_cost : ℝ := 150) :
  total_spent = initial_cost + cost_per_mile * total_miles → initial_cost = expected_initial_cost :=
by
  sorry

end initial_cost_renting_car_l256_256005


namespace order_of_logs_l256_256528

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem order_of_logs : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l256_256528


namespace sum_of_areas_l256_256307

theorem sum_of_areas (radii : ℕ → ℝ) (areas : ℕ → ℝ) (h₁ : radii 0 = 2) 
  (h₂ : ∀ n, radii (n + 1) = radii n / 3) 
  (h₃ : ∀ n, areas n = π * (radii n) ^ 2) : 
  ∑' n, areas n = (9 * π) / 2 := 
by 
  sorry

end sum_of_areas_l256_256307


namespace sin_cos_from_tan_l256_256210

variable {α : Real} (hα : α > 0) -- Assume α is an acute angle

theorem sin_cos_from_tan (h : Real.tan α = 2) : 
  Real.sin α = 2 / Real.sqrt 5 ∧ Real.cos α = 1 / Real.sqrt 5 := 
by sorry

end sin_cos_from_tan_l256_256210


namespace christopher_age_l256_256201

variable (C G F : ℕ)

theorem christopher_age (h1 : G = C + 8) (h2 : F = C - 2) (h3 : C + G + F = 60) : C = 18 := by
  sorry

end christopher_age_l256_256201


namespace arccos_one_half_is_pi_div_three_l256_256893

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256893


namespace total_cost_is_correct_l256_256583

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l256_256583


namespace overall_average_score_l256_256972

theorem overall_average_score (students_total : ℕ) (scores_day1 : ℕ) (avg1 : ℝ)
  (scores_day2 : ℕ) (avg2 : ℝ) (scores_day3 : ℕ) (avg3 : ℝ)
  (h1 : students_total = 45)
  (h2 : scores_day1 = 35)
  (h3 : avg1 = 0.65)
  (h4 : scores_day2 = 8)
  (h5 : avg2 = 0.75)
  (h6 : scores_day3 = 2)
  (h7 : avg3 = 0.85) :
  (scores_day1 * avg1 + scores_day2 * avg2 + scores_day3 * avg3) / students_total = 0.68 :=
by
  -- Lean proof goes here
  sorry

end overall_average_score_l256_256972


namespace arccos_half_eq_pi_div_three_l256_256869

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256869


namespace probability_neither_square_nor_cube_l256_256625

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l256_256625


namespace total_spent_after_three_years_l256_256752

def iPhone_cost : ℝ := 1000
def contract_cost_per_month : ℝ := 200
def case_cost_before_discount : ℝ := 0.20 * iPhone_cost
def headphones_cost_before_discount : ℝ := 0.5 * case_cost_before_discount
def charger_cost : ℝ := 60
def warranty_cost_for_two_years : ℝ := 150
def discount_rate : ℝ := 0.10
def time_in_years : ℝ := 3

def contract_cost_for_three_years := contract_cost_per_month * 12 * time_in_years
def case_cost_after_discount := case_cost_before_discount * (1 - discount_rate)
def headphones_cost_after_discount := headphones_cost_before_discount * (1 - discount_rate)

def total_cost : ℝ :=
  iPhone_cost +
  contract_cost_for_three_years +
  case_cost_after_discount +
  headphones_cost_after_discount +
  charger_cost +
  warranty_cost_for_two_years

theorem total_spent_after_three_years : total_cost = 8680 :=
  by
    sorry

end total_spent_after_three_years_l256_256752


namespace perfect_square_trinomial_solution_l256_256398

theorem perfect_square_trinomial_solution (m : ℝ) :
  (∃ a : ℝ, (∀ x : ℝ, x^2 - 2*(m+3)*x + 9 = (x - a)^2))
  → m = 0 ∨ m = -6 :=
by
  sorry

end perfect_square_trinomial_solution_l256_256398


namespace total_investment_amount_l256_256022

-- Define the initial conditions
def amountAt8Percent : ℝ := 3000
def interestAt8Percent (amount : ℝ) : ℝ := amount * 0.08
def interestAt10Percent (amount : ℝ) : ℝ := amount * 0.10
def totalAmount (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem total_investment_amount : 
    let x := 2400
    totalAmount amountAt8Percent x = 5400 :=
by
  sorry

end total_investment_amount_l256_256022


namespace common_diff_necessary_sufficient_l256_256067

section ArithmeticSequence

variable {α : Type*} [OrderedAddCommGroup α] {a : ℕ → α} {d : α}

-- Define an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Prove that d > 0 is the necessary and sufficient condition for a_2 > a_1
theorem common_diff_necessary_sufficient (a : ℕ → α) (d : α) :
    (is_arithmetic_sequence a d) → (d > 0 ↔ a 2 > a 1) :=
by
  sorry

end ArithmeticSequence

end common_diff_necessary_sufficient_l256_256067


namespace probability_not_square_or_cube_l256_256646

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l256_256646


namespace arccos_half_eq_pi_div_three_l256_256904

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256904


namespace problem_statement_l256_256953

noncomputable def tan_plus_alpha_half_pi (α : ℝ) : ℝ := -1 / (Real.tan α)

theorem problem_statement (α : ℝ) (h : tan_plus_alpha_half_pi α = -1 / 2) :
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -5 := by
  sorry

end problem_statement_l256_256953


namespace distinct_real_roots_range_l256_256969

theorem distinct_real_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ax^2 + 2 * x + 1 = 0 ∧ ay^2 + 2 * y + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
by
  sorry

end distinct_real_roots_range_l256_256969


namespace min_possible_frac_l256_256401

theorem min_possible_frac (x A C : ℝ) (hx : x ≠ 0) (hC_pos : 0 < C) (hA_pos : 0 < A)
  (h1 : x^2 + (1/x)^2 = A)
  (h2 : x - 1/x = C)
  (hC : C = Real.sqrt 3):
  A / C = (5 * Real.sqrt 3) / 3 := by
  sorry

end min_possible_frac_l256_256401


namespace sam_bought_17_mystery_books_l256_256315

def adventure_books := 13
def used_books := 15
def new_books := 15
def total_books := used_books + new_books
def mystery_books := total_books - adventure_books

theorem sam_bought_17_mystery_books : mystery_books = 17 := by
  sorry

end sam_bought_17_mystery_books_l256_256315


namespace other_root_of_equation_l256_256967

theorem other_root_of_equation (c : ℝ) (h : 3^2 - 5 * 3 + c = 0) : 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5 * x + c = 0 ∧ x = 2 := 
by 
  sorry

end other_root_of_equation_l256_256967


namespace table_tennis_probability_l256_256236

-- Define the given conditions
def prob_A_wins_set : ℚ := 2 / 3
def prob_B_wins_set : ℚ := 1 / 3
def best_of_five_sets := 5
def needed_wins_for_A := 3
def needed_losses_for_A := 2

-- Define the problem to prove
theorem table_tennis_probability :
  ((prob_A_wins_set ^ 2) * prob_B_wins_set * prob_A_wins_set) = 8 / 27 :=
by
  sorry

end table_tennis_probability_l256_256236


namespace sandy_change_from_twenty_dollar_bill_l256_256830

theorem sandy_change_from_twenty_dollar_bill :
  let cappuccino_cost := 2
  let iced_tea_cost := 3
  let cafe_latte_cost := 1.5
  let espresso_cost := 1
  let num_cappuccinos := 3
  let num_iced_teas := 2
  let num_cafe_lattes := 2
  let num_espressos := 2
  let total_cost := num_cappuccinos * cappuccino_cost
                  + num_iced_teas * iced_tea_cost
                  + num_cafe_lattes * cafe_latte_cost
                  + num_espressos * espresso_cost
  20 - total_cost = 3 := 
by
  sorry

end sandy_change_from_twenty_dollar_bill_l256_256830


namespace find_initial_divisor_l256_256597

theorem find_initial_divisor (N D : ℤ) (h1 : N = 2 * D) (h2 : N % 4 = 2) : D = 3 :=
by
  sorry

end find_initial_divisor_l256_256597


namespace quadratic_has_real_roots_iff_l256_256734

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l256_256734


namespace algebraic_expression_value_l256_256397

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (∃ x, x = -2 ∧ a * x - b = 1) → 4 * a + 2 * b + 7 = 5 :=
by
  intros a b h
  cases' h with x hx
  cases' hx with hx1 hx2
  rw [hx1] at hx2
  sorry

end algebraic_expression_value_l256_256397


namespace projection_ratio_zero_l256_256106

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end projection_ratio_zero_l256_256106


namespace john_total_cost_l256_256743

-- Definitions based on given conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_next_10_years : ℕ := 20000
def university_tuition : ℕ := 250000
def years_first_phase : ℕ := 8
def years_second_phase : ℕ := 10

-- We need to prove the total cost John pays
theorem john_total_cost : 
  (years_first_phase * yearly_cost_first_8_years + years_second_phase * yearly_cost_next_10_years + university_tuition) / 2 = 265000 :=
by sorry

end john_total_cost_l256_256743


namespace expression_equivalence_l256_256523

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by
  sorry

end expression_equivalence_l256_256523


namespace women_left_room_is_3_l256_256565

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l256_256565


namespace arccos_half_eq_pi_div_three_l256_256846

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256846


namespace constant_term_proof_l256_256043

noncomputable def constant_term_in_binomial_expansion (c : ℚ) (x : ℚ) : ℚ :=
  if h : (c = (2 : ℚ) - (1 / (8 * x^3))∧ x ≠ 0) then 
    28
  else 
    0

theorem constant_term_proof : 
  constant_term_in_binomial_expansion ((2 : ℚ) - (1 / (8 * (1 : ℚ)^3))) 1 = 28 := 
by
  sorry

end constant_term_proof_l256_256043


namespace arccos_half_eq_pi_div_three_l256_256902

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256902


namespace mean_score_l256_256149

variable (mean stddev : ℝ)

-- Conditions
axiom condition1 : 42 = mean - 5 * stddev
axiom condition2 : 67 = mean + 2.5 * stddev

theorem mean_score : mean = 58.67 := 
by 
  -- You would need to provide proof here
  sorry

end mean_score_l256_256149


namespace find_a1_l256_256097

-- Defining the conditions
variables (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_monotone : ∀ n, a n ≥ a (n + 1)) -- Monotonically decreasing

-- Specific values from the problem
axiom h_a3 : a 3 = 1
axiom h_a2_a4 : a 2 + a 4 = 5 / 2
axiom h_geom_seq : ∀ n, a (n + 1) = a n * q  -- Geometric sequence property

-- The goal is to prove that a 1 = 4
theorem find_a1 : a 1 = 4 :=
by
  -- Insert proof here
  sorry

end find_a1_l256_256097


namespace cost_of_renting_per_month_l256_256949

namespace RentCarProblem

def cost_new_car_per_month : ℕ := 30
def months_per_year : ℕ := 12
def yearly_difference : ℕ := 120

theorem cost_of_renting_per_month (R : ℕ) :
  (cost_new_car_per_month * months_per_year + yearly_difference) / months_per_year = R → 
  R = 40 :=
by
  sorry

end RentCarProblem

end cost_of_renting_per_month_l256_256949


namespace find_P_eq_30_l256_256798

theorem find_P_eq_30 (P Q R S : ℕ) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  P * Q = 120 ∧ R * S = 120 ∧ P - Q = R + S → P = 30 :=
by
  sorry

end find_P_eq_30_l256_256798


namespace quadratic_has_real_roots_iff_l256_256733

theorem quadratic_has_real_roots_iff (m : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + m + 5 = 0) ↔ m ≤ -1 :=
by
  -- Proof omitted
  sorry

end quadratic_has_real_roots_iff_l256_256733


namespace find_original_price_l256_256279

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end find_original_price_l256_256279


namespace cryptarithm_solved_l256_256941

-- Definitions for the digits A, B, C
def valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

-- Given conditions, where A, B, C are distinct non-zero digits
def conditions (A B C : ℕ) : Prop :=
  valid_digit A ∧ valid_digit B ∧ valid_digit C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C

-- Definitions of the two-digit and three-digit numbers
def two_digit (A B : ℕ) : ℕ := 10 * A + B
def three_digit_rep (C : ℕ) : ℕ := 111 * C

-- Main statement of the proof problem
theorem cryptarithm_solved (A B C : ℕ) (h : conditions A B C) :
  two_digit A B + A * three_digit_rep C = 247 → A * 100 + B * 10 + C = 251 :=
sorry -- Proof goes here

end cryptarithm_solved_l256_256941


namespace probability_of_even_product_l256_256188

def spinnerA := {2, 4, 5, 7, 9}
def spinnerB := {1, 2, 3, 4, 5, 6}
def prob_even (s₁ s₂ : finset ℕ) : ℚ :=
  1 - ((finset.filter (λ x : ℕ, x % 2 = 1) s₁).card * (finset.filter (λ y : ℕ, y % 2 = 1) s₂).card : ℚ) / 
        (s₁.card * s₂.card)

theorem probability_of_even_product :
  prob_even spinnerA spinnerB = 7 / 10 :=
by
  sorry

end probability_of_even_product_l256_256188


namespace fraction_BC_AD_l256_256237

-- Defining points and segments
variables (A B C D : Point)
variable (len : Point → Point → ℝ) -- length function

-- Conditions
axiom AB_eq_3BD : len A B = 3 * len B D
axiom AC_eq_7CD : len A C = 7 * len C D
axiom B_mid_AD : 2 * len A B = len A D

-- Theorem: Proving the fraction of BC relative to AD is 2/3
theorem fraction_BC_AD : (len B C) / (len A D) = 2 / 3 :=
sorry

end fraction_BC_AD_l256_256237


namespace correct_fraction_subtraction_l256_256073

theorem correct_fraction_subtraction (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  ((1 / x) - (1 / (x - 1))) = - (1 / (x^2 - x)) :=
by
  sorry

end correct_fraction_subtraction_l256_256073


namespace learning_machine_price_reduction_l256_256450

theorem learning_machine_price_reduction (x : ℝ) (h1 : 2000 * (1 - x) * (1 - x) = 1280) : 2000 * (1 - x)^2 = 1280 :=
by
  sorry

end learning_machine_price_reduction_l256_256450


namespace cylinder_original_radius_l256_256522

theorem cylinder_original_radius
  (r : ℝ)
  (h_original : ℝ := 4)
  (h_increased : ℝ := 3 * h_original)
  (volume_eq : π * (r + 8)^2 * h_original = π * r^2 * h_increased) :
  r = 4 + 4 * Real.sqrt 5 :=
sorry

end cylinder_original_radius_l256_256522


namespace monotonicity_and_tangent_intersection_l256_256368

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l256_256368


namespace prob_rain_both_days_l256_256649

-- Declare the probabilities involved
def P_Monday : ℝ := 0.40
def P_Tuesday : ℝ := 0.30
def P_Tuesday_given_Monday : ℝ := 0.30

-- Prove the probability of it raining on both days
theorem prob_rain_both_days : P_Monday * P_Tuesday_given_Monday = 0.12 :=
by
  sorry

end prob_rain_both_days_l256_256649


namespace markup_percentage_is_ten_l256_256274

theorem markup_percentage_is_ten (S C : ℝ)
  (h1 : S - C = 0.0909090909090909 * S) :
  (S - C) / C * 100 = 10 :=
by
  sorry

end markup_percentage_is_ten_l256_256274


namespace correct_choice_D_l256_256146

theorem correct_choice_D (a : ℝ) :
  (2 * a ^ 2) ^ 3 = 8 * a ^ 6 ∧ 
  (a ^ 10 * a ^ 2 ≠ a ^ 20) ∧ 
  (a ^ 10 / a ^ 2 ≠ a ^ 5) ∧ 
  ((Real.pi - 3) ^ 0 ≠ 0) :=
by {
  sorry
}

end correct_choice_D_l256_256146


namespace extreme_points_inequality_l256_256540

noncomputable def f (a x : ℝ) : ℝ := a * x - (a / x) - 2 * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := (a * x^2 - 2 * x + a) / x^2

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : a > 0) (h2 : 1 < x1) (h3 : x1 < Real.exp 1)
  (h4 : f a x1 = 0) (h5 : f a x2 = 0) (h6 : x1 ≠ x2) : |f a x1 - f a x2| < 1 :=
by
  sorry

end extreme_points_inequality_l256_256540


namespace manager_salary_l256_256153

theorem manager_salary (avg_salary_employees : ℝ) (num_employees : ℕ) (salary_increase : ℝ) (manager_salary : ℝ) :
  avg_salary_employees = 1500 →
  num_employees = 24 →
  salary_increase = 400 →
  (num_employees + 1) * (avg_salary_employees + salary_increase) - num_employees * avg_salary_employees = manager_salary →
  manager_salary = 11500 := 
by
  intros h_avg_salary_employees h_num_employees h_salary_increase h_computation
  sorry

end manager_salary_l256_256153


namespace free_throws_count_l256_256446

-- Definitions based on the conditions
variables (a b x : ℕ) -- Number of 2-point shots, 3-point shots, and free throws respectively.

-- Condition: Points from two-point shots equal the points from three-point shots
def points_eq : Prop := 2 * a = 3 * b

-- Condition: Number of free throws is twice the number of two-point shots
def free_throws_eq : Prop := x = 2 * a

-- Condition: Total score is adjusted to 78 points
def total_score : Prop := 2 * a + 3 * b + x = 78

-- Proof problem statement
theorem free_throws_count (h1 : points_eq a b) (h2 : free_throws_eq a x) (h3 : total_score a b x) : x = 26 :=
sorry

end free_throws_count_l256_256446


namespace family_ages_sum_today_l256_256454

theorem family_ages_sum_today (A B C D E : ℕ) (h1 : A + B + C + D = 114) (h2 : E = D - 14) :
    (A + 5) + (B + 5) + (C + 5) + (E + 5) = 120 :=
by
  sorry

end family_ages_sum_today_l256_256454


namespace total_students_l256_256661

def Varsity_students : ℕ := 1300
def Northwest_students : ℕ := 1400
def Central_students : ℕ := 1800
def Greenbriar_students : ℕ := 1650

theorem total_students : Varsity_students + Northwest_students + Central_students + Greenbriar_students = 6150 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end total_students_l256_256661


namespace initial_apples_l256_256429

theorem initial_apples (X : ℕ) (h : X - 2 + 3 = 5) : X = 4 :=
sorry

end initial_apples_l256_256429


namespace max_a_value_l256_256390

theorem max_a_value (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 3| ≥ 2 * a) : a ≤ 1 :=
sorry

end max_a_value_l256_256390


namespace one_of_sum_of_others_l256_256257

theorem one_of_sum_of_others (a b c : ℝ) 
  (cond1 : |a - b| ≥ |c|)
  (cond2 : |b - c| ≥ |a|)
  (cond3 : |c - a| ≥ |b|) :
  (a = b + c) ∨ (b = c + a) ∨ (c = a + b) :=
by
  sorry

end one_of_sum_of_others_l256_256257


namespace star_test_one_star_test_two_l256_256040

def star (x y : ℤ) : ℤ :=
  if x = 0 then Int.natAbs y
  else if y = 0 then Int.natAbs x
  else if (x < 0) = (y < 0) then Int.natAbs x + Int.natAbs y
  else -(Int.natAbs x + Int.natAbs y)

theorem star_test_one :
  star 11 (star 0 (-12)) = 23 :=
by
  sorry

theorem star_test_two (a : ℤ) :
  2 * (2 * star 1 a) - 1 = 3 * a ↔ a = 3 ∨ a = -5 :=
by
  sorry

end star_test_one_star_test_two_l256_256040


namespace solve_equation_l256_256453

theorem solve_equation :
  {x : ℝ | (x + 1) * (x + 3) = x + 1} = {-1, -2} :=
sorry

end solve_equation_l256_256453


namespace tim_kittens_l256_256778

theorem tim_kittens (initial_kittens : ℕ) (given_to_jessica_fraction : ℕ) (saras_kittens : ℕ) (adopted_fraction : ℕ) 
  (h_initial : initial_kittens = 12)
  (h_fraction_to_jessica : given_to_jessica_fraction = 3)
  (h_saras_kittens : saras_kittens = 14)
  (h_adopted_fraction : adopted_fraction = 2) :
  let kittens_after_jessica := initial_kittens - initial_kittens / given_to_jessica_fraction
  let total_kittens_after_sara := kittens_after_jessica + saras_kittens
  let adopted_kittens := saras_kittens / adopted_fraction
  let final_kittens := total_kittens_after_sara - adopted_kittens
  final_kittens = 15 :=
by {
  sorry
}

end tim_kittens_l256_256778


namespace comic_books_exclusive_count_l256_256507

theorem comic_books_exclusive_count 
  (shared_comics : ℕ) 
  (total_andrew_comics : ℕ) 
  (john_exclusive_comics : ℕ) 
  (h_shared_comics : shared_comics = 15) 
  (h_total_andrew_comics : total_andrew_comics = 22) 
  (h_john_exclusive_comics : john_exclusive_comics = 10) : 
  (total_andrew_comics - shared_comics + john_exclusive_comics = 17) := by 
  sorry

end comic_books_exclusive_count_l256_256507


namespace function_satisfies_conditions_l256_256328

def f (m n : ℕ) : ℕ := m * n

theorem function_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := 
by {
  sorry
}

end function_satisfies_conditions_l256_256328


namespace arccos_half_eq_pi_div_three_l256_256861

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256861


namespace f_positive_for_all_x_f_increasing_solution_set_inequality_l256_256930

namespace ProofProblem

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one_eq_two : f 1 = 2
axiom f_pos_when_pos : ∀ x : ℝ, x > 0 → f x > 1
axiom f_add_mul : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(x) > 0 for all x ∈ ℝ
theorem f_positive_for_all_x : ∀ x : ℝ, f x > 0 := sorry

-- Problem 2: Prove that f(x) is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 3: Find the solution set of the inequality f(3-2x) > 4
theorem solution_set_inequality : { x : ℝ | f (3 - 2 * x) > 4 } = { x | x < 1 / 2 } := sorry

end ProofProblem

end f_positive_for_all_x_f_increasing_solution_set_inequality_l256_256930


namespace work_completion_days_l256_256673

structure WorkProblem :=
  (total_work : ℝ := 1) -- Assume total work to be 1 unit
  (days_A : ℝ := 30)
  (days_B : ℝ := 15)
  (days_together : ℝ := 5)

noncomputable def total_days_taken (wp : WorkProblem) : ℝ :=
  let work_per_day_A := 1 / wp.days_A
  let work_per_day_B := 1 / wp.days_B
  let work_per_day_together := work_per_day_A + work_per_day_B
  let work_done_together := wp.days_together * work_per_day_together
  let remaining_work := wp.total_work - work_done_together
  let days_for_A := remaining_work / work_per_day_A
  wp.days_together + days_for_A

theorem work_completion_days (wp : WorkProblem) : total_days_taken wp = 20 :=
by
  sorry

end work_completion_days_l256_256673


namespace build_wall_30_persons_l256_256483

-- Defining the conditions
def work_rate (persons : ℕ) (days : ℕ) : ℚ := 1 / (persons * days)

-- Total work required to build the wall by 8 persons in 42 days
def total_work : ℚ := work_rate 8 42 * 8 * 42

-- Work rate for 30 persons
def combined_work_rate (persons : ℕ) : ℚ := persons * work_rate 8 42

-- Days required for 30 persons to complete the same work
def days_required (persons : ℕ) (work : ℚ) : ℚ := work / combined_work_rate persons

-- Expected result is 11.2 days for 30 persons
theorem build_wall_30_persons : days_required 30 total_work = 11.2 := 
by
  sorry

end build_wall_30_persons_l256_256483


namespace monotonicity_and_tangent_intersections_l256_256383

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l256_256383


namespace delores_initial_money_l256_256931

-- Definitions and conditions based on the given problem
def original_computer_price : ℝ := 400
def original_printer_price : ℝ := 40
def original_headphones_price : ℝ := 60

def computer_discount : ℝ := 0.10
def computer_tax : ℝ := 0.08
def printer_tax : ℝ := 0.05
def headphones_tax : ℝ := 0.06

def leftover_money : ℝ := 10

-- Final proof problem statement
theorem delores_initial_money :
  original_computer_price * (1 - computer_discount) * (1 + computer_tax) +
  original_printer_price * (1 + printer_tax) +
  original_headphones_price * (1 + headphones_tax) + leftover_money = 504.40 := by
  sorry -- Proof is not required

end delores_initial_money_l256_256931


namespace rectangular_solid_volume_l256_256191

theorem rectangular_solid_volume
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : b = 2 * a) :
  a * b * c = 12 := 
by
  sorry

end rectangular_solid_volume_l256_256191


namespace area_of_QCA_l256_256321

noncomputable def area_of_triangle (x p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) : ℝ :=
  1 / 2 * x * (15 - p)

theorem area_of_QCA (x : ℝ) (p : ℝ) (hx_pos : 0 < x) (hp_bounds : 0 < p ∧ p < 15) :
  area_of_triangle x p hx_pos hp_bounds = 1 / 2 * x * (15 - p) :=
sorry

end area_of_QCA_l256_256321


namespace arccos_half_eq_pi_div_three_l256_256860

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256860


namespace competition_end_time_l256_256807

-- Definitions for the problem conditions
def start_time : ℕ := 15 * 60  -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 1300       -- competition duration in minutes
def end_time : ℕ := start_time + duration

-- The expected end time in minutes from midnight, where 12:40 p.m. is (12*60 + 40) = 760 + 40 = 800 minutes from midnight.
def expected_end_time : ℕ := 12 * 60 + 40 

-- The theorem to prove
theorem competition_end_time : end_time = expected_end_time := by
  sorry

end competition_end_time_l256_256807


namespace arccos_one_half_l256_256886

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256886


namespace probability_neither_square_nor_cube_l256_256630

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l256_256630


namespace inequality_neg_mul_l256_256552

theorem inequality_neg_mul (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
    sorry

end inequality_neg_mul_l256_256552


namespace strawberries_to_grapes_ratio_l256_256586

-- Define initial conditions
def initial_grapes : ℕ := 100
def fruits_left : ℕ := 96

-- Define the number of strawberries initially
def strawberries_init (S : ℕ) : Prop :=
  (S - (2 * (1/5) * S) = fruits_left - initial_grapes + ((2 * (1/5)) * initial_grapes))

-- Define the ratio problem in Lean
theorem strawberries_to_grapes_ratio (S : ℕ) (h : strawberries_init S) : (S / initial_grapes = 3 / 5) :=
sorry

end strawberries_to_grapes_ratio_l256_256586


namespace log_product_zero_l256_256510

theorem log_product_zero :
  (Real.log 3 / Real.log 2 + Real.log 27 / Real.log 2) *
  (Real.log 4 / Real.log 4 + Real.log (1 / 4) / Real.log 4) = 0 := by
  -- Place proof here
  sorry

end log_product_zero_l256_256510


namespace volume_ratio_of_spheres_l256_256079

theorem volume_ratio_of_spheres (r R : ℝ) (h : (4 * real.pi * r^2) / (4 * real.pi * R^2) = 4 / 9) :
  (4 / 3 * real.pi * r^3) / (4 / 3 * real.pi * R^3) = 8 / 27 :=
by
  sorry

end volume_ratio_of_spheres_l256_256079


namespace arccos_half_eq_pi_div_three_l256_256899

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256899


namespace probability_bulls_win_series_l256_256435

theorem probability_bulls_win_series :
  let P_Bulls := 1 / 4 in
  let P_Heat := 3 / 4 in
  let scenarios := 6 in  -- number of ways to have 2 wins for each team out of first 4 games
  let prob_2wins_bulls := scenarios * (P_Bulls ^ 2) * (P_Heat ^ 2) in
  let prob_bulls_win_game_5 := P_Bulls in
  prob_2wins_bulls * prob_bulls_win_game_5 = 27 / 512 := by
  -- proof goes here
  sorry

end probability_bulls_win_series_l256_256435


namespace arccos_one_half_l256_256887

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256887


namespace probability_neither_square_nor_cube_l256_256640

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l256_256640


namespace find_possible_K_l256_256735

theorem find_possible_K (K : ℕ) (N : ℕ) (h1 : K * (K + 1) / 2 = N^2) (h2 : N < 150)
  (h3 : ∃ m : ℕ, N^2 = m * (m + 1) / 2) : K = 1 ∨ K = 8 ∨ K = 39 ∨ K = 92 ∨ K = 168 := by
  sorry

end find_possible_K_l256_256735


namespace cos_alpha_values_l256_256223

theorem cos_alpha_values (α : ℝ) (h : Real.sin (π + α) = -3 / 5) :
  Real.cos α = 4 / 5 ∨ Real.cos α = -4 / 5 := 
sorry

end cos_alpha_values_l256_256223


namespace find_x_l256_256666

def cube_volume (s : ℝ) := s^3
def cube_surface_area (s : ℝ) := 6 * s^2

theorem find_x (x : ℝ) (s : ℝ) 
  (hv : cube_volume s = 7 * x)
  (hs : cube_surface_area s = x) : 
  x = 42 := 
by
  sorry

end find_x_l256_256666


namespace expected_attempts_for_10_keys_10_suitcases_l256_256299

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (n * (n + 3)) / 4 - (Real.log n + 0.577)

theorem expected_attempts_for_10_keys_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 0.01 := 
by
  sorry

end expected_attempts_for_10_keys_10_suitcases_l256_256299


namespace max_product_not_less_than_993_squared_l256_256445

theorem max_product_not_less_than_993_squared :
  ∀ (a : Fin 1985 → ℕ), 
    (∀ i, ∃ j, a j = i + 1) →  -- representation of permutation
    (∃ i : Fin 1985, i * (a i) ≥ 993 * 993) :=
by
  intros a h
  sorry

end max_product_not_less_than_993_squared_l256_256445


namespace probability_x_lt_y_l256_256812

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l256_256812


namespace problem_statement_l256_256058

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : (a - c) ^ 3 > (b - c) ^ 3 :=
by
  sorry

end problem_statement_l256_256058


namespace D_72_l256_256984

/-- D(n) denotes the number of ways of writing the positive integer n
    as a product n = f1 * f2 * ... * fk, where k ≥ 1, the fi are integers
    strictly greater than 1, and the order in which the factors are
    listed matters. -/
def D (n : ℕ) : ℕ := sorry

theorem D_72 : D 72 = 43 := sorry

end D_72_l256_256984


namespace no_integer_solutions_l256_256322

theorem no_integer_solutions :
  ∀ x y : ℤ, x^3 + 4 * x^2 - 11 * x + 30 ≠ 8 * y^3 + 24 * y^2 + 18 * y + 7 :=
by sorry

end no_integer_solutions_l256_256322


namespace probability_neither_square_nor_cube_l256_256641

theorem probability_neither_square_nor_cube (A : Finset ℕ) (hA : A = Finset.range 201) :
  (A.filter (λ n, ¬ (∃ k, k^2 = n) ∧ ¬ (∃ k, k^3 = n))).card / A.card = 183 / 200 := 
by sorry

end probability_neither_square_nor_cube_l256_256641


namespace problem1_problem2_l256_256829

-- Problem 1
theorem problem1 (b : ℝ) :
  4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1) :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  a - a * abs (-a^2 - 1) < 1 - a^2 * (a - 1) :=
by
  sorry

end problem1_problem2_l256_256829


namespace farm_section_areas_l256_256685

theorem farm_section_areas (n : ℕ) (total_area : ℕ) (sections : ℕ) 
  (hn : sections = 5) (ht : total_area = 300) : total_area / sections = 60 :=
by
  sorry

end farm_section_areas_l256_256685


namespace solution_is_correct_l256_256260

noncomputable def satisfies_inequality (x y : ℝ) : Prop := 
  x + 3 * y + 14 ≤ 0

noncomputable def satisfies_equation (x y : ℝ) : Prop := 
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solution_is_correct : satisfies_inequality (-2) (-4) ∧ satisfies_equation (-2) (-4) :=
  by sorry

end solution_is_correct_l256_256260


namespace how_many_women_left_l256_256572

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l256_256572


namespace monotonicity_of_f_intersection_points_of_tangent_l256_256385

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l256_256385


namespace hyperbola_foci_property_l256_256389

noncomputable def hyperbola (x y b : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / b^2) = 1

theorem hyperbola_foci_property (x y b : ℝ) (h : hyperbola x y b) (b_pos : b > 0) (PF1 : ℝ) (PF2 : ℝ) (hPF1 : PF1 = 5) :
  PF2 = 11 :=
by
  sorry

end hyperbola_foci_property_l256_256389


namespace find_b_in_triangle_l256_256066

theorem find_b_in_triangle (a B C A b : ℝ)
  (ha : a = Real.sqrt 3)
  (hB : Real.sin B = 1 / 2)
  (hC : C = Real.pi / 6)
  (hA : A = 2 * Real.pi / 3) :
  b = 1 :=
by
  -- proof omitted
  sorry

end find_b_in_triangle_l256_256066


namespace total_pay_l256_256131

-- Definitions based on the conditions
def y_pay : ℕ := 290
def x_pay : ℕ := (120 * y_pay) / 100

-- The statement to prove that the total pay is Rs. 638
theorem total_pay : x_pay + y_pay = 638 := 
by
  -- skipping the proof for now
  sorry

end total_pay_l256_256131


namespace TommysFirstHousePrice_l256_256280

theorem TommysFirstHousePrice :
  ∃ (P : ℝ), 1.25 * P = 0.25 * 500000 ∧ P = 100000 :=
by
  use 100000
  split
  · norm_num
  · norm_num
  sorry

end TommysFirstHousePrice_l256_256280


namespace replace_star_with_2x_l256_256996

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l256_256996


namespace max_blocks_in_box_l256_256463

def volume (l w h : ℕ) : ℕ := l * w * h

-- Define the dimensions of the box and the block
def box_length := 4
def box_width := 3
def box_height := 2
def block_length := 3
def block_width := 1
def block_height := 1

-- Define the volumes of the box and the block using the dimensions
def V_box : ℕ := volume box_length box_width box_height
def V_block : ℕ := volume block_length block_width block_height

theorem max_blocks_in_box : V_box / V_block = 8 :=
  sorry

end max_blocks_in_box_l256_256463


namespace june_bernard_travel_time_l256_256745

-- Defining the given conditions
def distance_june_julia : ℝ := 2
def time_june_julia : ℝ := 6
def distance_june_bernard : ℝ := 5

-- Rate calculation premise
def travel_rate : ℝ := distance_june_julia / time_june_julia

-- Proof problem statement
theorem june_bernard_travel_time :
  travel_rate * distance_june_bernard = 15 := sorry

end june_bernard_travel_time_l256_256745


namespace min_sum_of_dimensions_l256_256946

theorem min_sum_of_dimensions (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 3003) : 
  a + b + c ≥ 57 := sorry

end min_sum_of_dimensions_l256_256946


namespace potion_kit_cost_is_18_l256_256072

def price_spellbook : ℕ := 5
def count_spellbooks : ℕ := 5
def price_owl : ℕ := 28
def count_potion_kits : ℕ := 3
def payment_total_silver : ℕ := 537
def silver_per_gold : ℕ := 9

def cost_each_potion_kit_in_silver (payment_total_silver : ℕ)
                                   (price_spellbook : ℕ)
                                   (count_spellbooks : ℕ)
                                   (price_owl : ℕ)
                                   (count_potion_kits : ℕ)
                                   (silver_per_gold : ℕ) : ℕ :=
  let total_gold := payment_total_silver / silver_per_gold
  let cost_spellbooks := count_spellbooks * price_spellbook
  let cost_remaining_gold := total_gold - cost_spellbooks - price_owl
  let cost_each_potion_kit_gold := cost_remaining_gold / count_potion_kits
  cost_each_potion_kit_gold * silver_per_gold

theorem potion_kit_cost_is_18 :
  cost_each_potion_kit_in_silver payment_total_silver
                                 price_spellbook
                                 count_spellbooks
                                 price_owl
                                 count_potion_kits
                                 silver_per_gold = 18 :=
by sorry

end potion_kit_cost_is_18_l256_256072


namespace third_box_nuts_l256_256098

theorem third_box_nuts
  (A B C : ℕ)
  (h1 : A = B + C - 6)
  (h2 : B = A + C - 10) :
  C = 8 :=
by
  sorry

end third_box_nuts_l256_256098


namespace arccos_half_eq_pi_div_three_l256_256898

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256898


namespace correct_options_l256_256017

-- Given conditions
def f : ℝ → ℝ := sorry -- We will assume there is some function f that satisfies the conditions

axiom xy_identity (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = x * f y + y * f x
axiom f_positive (x : ℝ) (hx : 1 < x) : 0 < f x

-- Proof of the required conclusion
theorem correct_options (h1 : f 1 = 0) (h2 : ∀ x y, f (x * y) ≠ f x * f y)
  (h3 : ∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y)
  (h4 : ∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) : 
  f 1 = 0 ∧ (∀ x y, f (x * y) ≠ f x * f y) ∧ (∀ x, 1 < x → ∀ y, 1 < y → x < y → f x < f y) ∧ (∀ x, 2 ≤ x → x * f (x - 3 / 2) ≥ (3 / 2 - x) * f x) :=
sorry

end correct_options_l256_256017


namespace algebraic_expression_value_l256_256952

variable {a b c : ℝ}

theorem algebraic_expression_value
  (h1 : (a + b) * (b + c) * (c + a) = 0)
  (h2 : a * b * c < 0) :
  (a / |a|) + (b / |b|) + (c / |c|) = 1 := by
  sorry

end algebraic_expression_value_l256_256952


namespace monotonicity_tangent_intersection_points_l256_256356

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l256_256356


namespace arccos_half_is_pi_div_three_l256_256857

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256857


namespace arccos_half_eq_pi_div_three_l256_256843

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256843


namespace commonPointsLineCurve_l256_256093

noncomputable def CartesianEquationOfL (ρ θ m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = ρ * cos θ) ∧ (y = ρ * sin θ) →
  (ρ * sin (θ + π / 3) + m = 0) →
  (√3 * x + y + 2 * m = 0)

noncomputable def parametricCurveC (t : ℝ) : Prop :=
  ∃ x y : ℝ, (x = √3 * cos (2 * t)) ∧ (y = 2 * sin t)

noncomputable def rangeOfMForCommonPoints (m : ℝ) : Prop :=
  -19 / 12 ≤ m ∧ m ≤ 5 / 2

theorem commonPointsLineCurve (ρ θ m t : ℝ) :
  CartesianEquationOfL ρ θ m →
  parametricCurveC t →
  rangeOfMForCommonPoints m :=
begin
  sorry
end

end commonPointsLineCurve_l256_256093


namespace probability_neither_perfect_square_nor_cube_l256_256627

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l256_256627


namespace arccos_one_half_l256_256888

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256888


namespace sample_size_l256_256971

theorem sample_size (w_under30 : ℕ) (w_30to40 : ℕ) (w_40plus : ℕ) (sample_40plus : ℕ) (total_sample : ℕ) :
  w_under30 = 2400 →
  w_30to40 = 3600 →
  w_40plus = 6000 →
  sample_40plus = 60 →
  total_sample = 120 :=
by
  intros
  sorry

end sample_size_l256_256971


namespace elsa_final_marbles_l256_256324

def start_marbles : ℕ := 40
def lost_breakfast : ℕ := 3
def given_susie : ℕ := 5
def new_marbles : ℕ := 12
def returned_marbles : ℕ := 2 * given_susie

def final_marbles : ℕ :=
  start_marbles - lost_breakfast - given_susie + new_marbles + returned_marbles

theorem elsa_final_marbles : final_marbles = 54 := by
  sorry

end elsa_final_marbles_l256_256324


namespace parallel_lines_condition_l256_256546

theorem parallel_lines_condition (k1 k2 b : ℝ) (l1 l2 : ℝ → ℝ) (H1 : ∀ x, l1 x = k1 * x + 1)
  (H2 : ∀ x, l2 x = k2 * x + b) : (∀ x, l1 x = l2 x ↔ k1 = k2 ∧ b = 1) → (k1 = k2) ↔ (∀ x, l1 x ≠ l2 x ∧ l1 x - l2 x = 1 - b) := 
by
  sorry

end parallel_lines_condition_l256_256546


namespace right_triangle_area_l256_256285

variable {AB BC AC : ℕ}

theorem right_triangle_area : ∀ (AB BC AC : ℕ), (AC = 50) → (AB + BC = 70) → (AB^2 + BC^2 = AC^2) → (1 / 2) * AB * BC = 300 :=
by
  intros AB BC AC h1 h2 h3
  -- Proof steps will be added here
  sorry

end right_triangle_area_l256_256285


namespace my_age_is_five_times_son_age_l256_256591

theorem my_age_is_five_times_son_age (son_age_next : ℕ) (my_age : ℕ) (h1 : son_age_next = 8) (h2 : my_age = 5 * (son_age_next - 1)) : my_age = 35 :=
by
  -- skip the proof
  sorry

end my_age_is_five_times_son_age_l256_256591


namespace ruby_height_l256_256399

/-- Height calculations based on given conditions -/
theorem ruby_height (Janet_height : ℕ) (Charlene_height : ℕ) (Pablo_height : ℕ) (Ruby_height : ℕ) 
  (h₁ : Janet_height = 62) 
  (h₂ : Charlene_height = 2 * Janet_height)
  (h₃ : Pablo_height = Charlene_height + 70)
  (h₄ : Ruby_height = Pablo_height - 2) : Ruby_height = 192 := 
by
  sorry

end ruby_height_l256_256399


namespace arccos_one_half_l256_256885

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256885


namespace standard_deviation_of_applicants_l256_256270

theorem standard_deviation_of_applicants (σ : ℕ) 
  (h1 : ∃ avg : ℕ, avg = 30)
  (h2 : ∃ n : ℕ, n = 17)
  (h3 : ∃ range_count : ℕ, range_count = (30 + σ) - (30 - σ) + 1) :
  σ = 8 :=
by
  sorry

end standard_deviation_of_applicants_l256_256270


namespace arccos_one_half_is_pi_div_three_l256_256897

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256897


namespace max_value_200_max_value_attained_l256_256751

noncomputable def max_value (X Y Z : ℕ) : ℕ := 
  X * Y * Z + X * Y + Y * Z + Z * X

theorem max_value_200 (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  max_value X Y Z ≤ 200 :=
sorry

theorem max_value_attained (X Y Z : ℕ) (h : X = 5) (h1 : Y = 5) (h2 : Z = 5) : 
  max_value X Y Z = 200 :=
sorry

end max_value_200_max_value_attained_l256_256751


namespace complement_union_example_l256_256481

open Set

theorem complement_union_example :
  ∀ (U A B : Set ℕ), 
  U = {1, 2, 3, 4, 5, 6, 7, 8} → 
  A = {1, 3, 5, 7} → 
  B = {2, 4, 5} → 
  (U \ (A ∪ B)) = {6, 8} := by 
  intros U A B hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_union_example_l256_256481


namespace general_term_of_an_l256_256545

theorem general_term_of_an (a : ℕ → ℕ) (h1 : a 1 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) :
    ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end general_term_of_an_l256_256545


namespace bob_shucks_240_oysters_in_2_hours_l256_256822

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l256_256822


namespace probability_neither_perfect_square_nor_cube_l256_256629

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l256_256629


namespace smallest_product_is_neg280_l256_256715

theorem smallest_product_is_neg280 :
  let S := {-8, -6, -4, 0, 3, 5, 7}
  in ∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = -280 := by
sorry

end smallest_product_is_neg280_l256_256715


namespace wheel_distance_covered_l256_256499

noncomputable def diameter : ℝ := 15
noncomputable def revolutions : ℝ := 11.210191082802547
noncomputable def pi : ℝ := Real.pi -- or you can use the approximate value if required: 3.14159
noncomputable def circumference : ℝ := pi * diameter
noncomputable def distance_covered : ℝ := circumference * revolutions

theorem wheel_distance_covered :
  distance_covered = 528.316820577 := 
by
  unfold distance_covered
  unfold circumference
  unfold diameter
  unfold revolutions
  norm_num
  sorry

end wheel_distance_covered_l256_256499


namespace tangent_line_through_point_l256_256054

-- Definitions based purely on the conditions given in the problem.
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
def point_on_line (x y : ℝ) : Prop := 3 * x - 4 * y + 25 = 0
def point_given : ℝ × ℝ := (-3, 4)

-- The theorem statement to be proven
theorem tangent_line_through_point : point_on_line point_given.1 point_given.2 := 
sorry

end tangent_line_through_point_l256_256054


namespace factorize_cubic_l256_256327

theorem factorize_cubic (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_cubic_l256_256327


namespace triangle_area_fraction_l256_256111

-- Define the grid size
def grid_size : ℕ := 6

-- Define the vertices of the triangle
def vertex_A : (ℕ × ℕ) := (3, 3)
def vertex_B : (ℕ × ℕ) := (3, 5)
def vertex_C : (ℕ × ℕ) := (5, 5)

-- Define the area of the larger grid
def area_square := grid_size ^ 2

-- Compute the base and height of the triangle
def base_triangle := vertex_C.1 - vertex_B.1
def height_triangle := vertex_B.2 - vertex_A.2

-- Compute the area of the triangle
def area_triangle := (base_triangle * height_triangle) / 2

-- Define the fraction of the area of the larger square inside the triangle
def area_fraction := area_triangle / area_square

-- State the theorem
theorem triangle_area_fraction :
  area_fraction = 1 / 18 :=
by
  sorry

end triangle_area_fraction_l256_256111


namespace monotonicity_of_f_tangent_intersection_coordinates_l256_256370

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l256_256370


namespace initial_mixture_amount_l256_256496

/-- A solution initially contains an unknown amount of a mixture consisting of 15% sodium chloride
(NaCl), 30% potassium chloride (KCl), 35% sugar, and 20% water. To this mixture, 50 grams of sodium chloride
and 80 grams of potassium chloride are added. If the new salt content of the solution (NaCl and KCl combined)
is 47.5%, how many grams of the mixture were present initially?

Given:
  * The initial mixture consists of 15% NaCl and 30% KCl.
  * 50 grams of NaCl and 80 grams of KCl are added.
  * The new mixture has 47.5% NaCl and KCl combined.
  
Prove that the initial amount of the mixture was 2730 grams. -/
theorem initial_mixture_amount
    (x : ℝ)
    (h_initial_mixture : 0.15 * x + 50 + 0.30 * x + 80 = 0.475 * (x + 130)) :
    x = 2730 := by
  sorry

end initial_mixture_amount_l256_256496


namespace smallest_three_digit_solution_l256_256465

theorem smallest_three_digit_solution (n : ℕ) : 
  75 * n ≡ 225 [MOD 345] → 100 ≤ n ∧ n ≤ 999 → n = 118 :=
by
  intros h1 h2
  sorry

end smallest_three_digit_solution_l256_256465


namespace average_age_of_team_l256_256437

theorem average_age_of_team (A : ℝ) : 
    (11 * A =
         9 * (A - 1) + 53) → 
    A = 31 := 
by 
  sorry

end average_age_of_team_l256_256437


namespace total_money_raised_l256_256302

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l256_256302


namespace leonine_cats_l256_256144

theorem leonine_cats (n : ℕ) (h : n = (4 / 5 * n) + (4 / 5)) : n = 4 :=
by
  sorry

end leonine_cats_l256_256144


namespace arccos_half_eq_pi_div_three_l256_256866

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256866


namespace area_difference_l256_256109

-- Define the areas of individual components
def area_of_square : ℕ := 1
def area_of_small_triangle : ℚ := (1 / 2) * area_of_square
def area_of_large_triangle : ℚ := (1 / 2) * (1 * 2 * area_of_square)

-- Define the total area of the first figure
def first_figure_area : ℚ := 
    8 * area_of_square +
    6 * area_of_small_triangle +
    2 * area_of_large_triangle

-- Define the total area of the second figure
def second_figure_area : ℚ := 
    4 * area_of_square +
    6 * area_of_small_triangle +
    8 * area_of_large_triangle

-- Define the statement to prove the difference in areas
theorem area_difference : second_figure_area - first_figure_area = 2 := by
    -- sorry is used to indicate that the proof is omitted
    sorry

end area_difference_l256_256109


namespace geo_seq_arith_seq_l256_256767

theorem geo_seq_arith_seq (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_gp : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, a_n n > 0) (h_arith : a_n 4 - a_n 3 = a_n 5 - a_n 4) 
  (hq_pos : q > 0) (hq_neq1 : q ≠ 1) :
  S 6 / S 3 = 2 := by
  sorry

end geo_seq_arith_seq_l256_256767


namespace find_a_decreasing_l256_256766

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem find_a_decreasing : 
  (∀ x : ℝ, x < 6 → f x a ≤ f (x - 1) a) → a ≥ 6 := 
sorry

end find_a_decreasing_l256_256766


namespace probability_neither_square_nor_cube_l256_256632

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l256_256632


namespace probability_x_lt_y_in_rectangle_l256_256810

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l256_256810


namespace quadratic_m_ge_neg2_l256_256224

-- Define the quadratic equation and condition for real roots
def quadratic_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, (x + 2) ^ 2 = m + 2

-- The theorem to prove
theorem quadratic_m_ge_neg2 (m : ℝ) (h : quadratic_has_real_roots m) : m ≥ -2 :=
by {
  sorry
}

end quadratic_m_ge_neg2_l256_256224


namespace cartesian_line_equiv_ranges_l256_256091

variable (t m : ℝ)

def parametric_curve (t : ℝ) : ℝ × ℝ := (sqrt 3 * cos (2 * t), 2 * sin t)

def polar_line (ρ θ m : ℝ) : Prop := ρ * sin (θ + π / 3) + m = 0

noncomputable def cartesian_line (x y m : ℝ) : Prop := sqrt 3 * x + y + 2 * m = 0

def curve_equation (x y : ℝ) : Prop := x = sqrt 3 * cos (2 * t) ∧ y = 2 * sin t

def range_of_m (m : ℝ) : Prop := -19 / 12 ≤ m ∧ m ≤ 5 / 2

theorem cartesian_line_equiv (t m : ℝ) : 
  (∀ ρ θ, polar_line ρ θ m ↔ cartesian_line 
    (ρ * cos θ) (ρ * sin θ) m) :=
by
  sorry

theorem ranges(m : ℝ) : 
  (∀ t, curve_equation (sqrt 3 * cos (2 * t)) (2 * sin t) → ∃ m, range_of_m m) :=
by
  sorry

end cartesian_line_equiv_ranges_l256_256091


namespace minimum_value_2_only_in_option_b_l256_256503

noncomputable def option_a (x : ℝ) : ℝ := x + 1 / x
noncomputable def option_b (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def option_c (x : ℝ) : ℝ := (Real.log x) + 1 / (Real.log x)
noncomputable def option_d (x : ℝ) : ℝ := (Real.sin x) + 1 / (Real.sin x)

theorem minimum_value_2_only_in_option_b :
  (∀ x > 0, option_a x ≠ 2) ∧
  (∃ x, option_b x = 2) ∧
  (∀ x (h: 0 < x) (h' : x < 1), option_c x ≠ 2) ∧
  (∀ x (h: 0 < x) (h' : x < π / 2), option_d x ≠ 2) :=
by
  sorry

end minimum_value_2_only_in_option_b_l256_256503


namespace zero_points_sum_gt_one_l256_256958

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1 / (2 * x))

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (hx₁ : g x₁ m = 0) (hx₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := 
  by
    sorry

end zero_points_sum_gt_one_l256_256958


namespace points_on_same_line_l256_256045

theorem points_on_same_line (p : ℝ) :
  (∃ m : ℝ, m = ( -3.5 - 0.5 ) / ( 3 - (-1)) ∧ ∀ x y : ℝ, 
    (x = -1 ∧ y = 0.5) ∨ (x = 3 ∧ y = -3.5) ∨ (x = 7 ∧ y = p) → y = m * x + (0.5 - m * (-1))) →
    p = -7.5 :=
by
  sorry

end points_on_same_line_l256_256045


namespace cost_of_goods_l256_256777

theorem cost_of_goods
  (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 315)
  (h2 : 4 * x + 10 * y + z = 420) :
  x + y + z = 105 :=
by
  sorry

end cost_of_goods_l256_256777


namespace tangent_line_find_a_l256_256403

theorem tangent_line_find_a (a : ℝ) (f : ℝ → ℝ) (tangent : ℝ → ℝ) (x₀ : ℝ)
  (hf : ∀ x, f x = x + 1/x - a * Real.log x)
  (h_tangent : ∀ x, tangent x = x + 1)
  (h_deriv : deriv f x₀ = deriv tangent x₀)
  (h_eq : f x₀ = tangent x₀) :
  a = -1 :=
sorry

end tangent_line_find_a_l256_256403


namespace average_star_rating_l256_256268

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l256_256268


namespace indigo_restaurant_average_rating_l256_256264

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l256_256264


namespace arccos_half_eq_pi_div_3_l256_256909

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256909


namespace even_binomial_coefficients_l256_256318

theorem even_binomial_coefficients (n : ℕ) (h_pos: 0 < n) : 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → 2 ∣ Nat.choose n k) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end even_binomial_coefficients_l256_256318


namespace graph_location_l256_256768

theorem graph_location (k : ℝ) (H : k > 0) :
    (∀ x : ℝ, (0 < x → 0 < y) → (y = 2/x) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
    sorry

end graph_location_l256_256768


namespace simplify_root_power_l256_256606

theorem simplify_root_power :
  (7^(1/3))^6 = 49 := by
  sorry

end simplify_root_power_l256_256606


namespace complement_of_A_in_U_l256_256729

def U : Set ℤ := {x | -2 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {x | ∃ n : ℕ, (x = 2 * n ∧ n ≤ 3)}

theorem complement_of_A_in_U : (U \ A) = {-2, -1, 1, 3, 5} :=
by
  sorry

end complement_of_A_in_U_l256_256729


namespace bread_products_wasted_l256_256580

theorem bread_products_wasted :
  (50 * 8 - (20 * 5 + 15 * 4 + 10 * 10 * 1.5)) / 1.5 = 60 := by
  -- The proof steps are omitted here
  sorry

end bread_products_wasted_l256_256580


namespace sum_of_tangents_l256_256184

noncomputable def function_f (x : ℝ) : ℝ :=
  max (max (4 * x + 20) (-x + 2)) (5 * x - 3)

theorem sum_of_tangents (q : ℝ → ℝ) (a b c : ℝ) (h1 : ∀ x, q x - (4 * x + 20) = q x - function_f x)
  (h2 : ∀ x, q x - (-x + 2) = q x - function_f x)
  (h3 : ∀ x, q x - (5 * x - 3) = q x - function_f x) :
  a + b + c = -83 / 10 :=
sorry

end sum_of_tangents_l256_256184


namespace fisherman_catch_total_l256_256165

theorem fisherman_catch_total :
  let bass := 32
  let trout := bass / 4
  let blue_gill := bass * 2
in bass + trout + blue_gill = 104 := by
  sorry

end fisherman_catch_total_l256_256165


namespace find_min_length_seg_O1O2_l256_256115

noncomputable def minimum_length_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ) (dist_YZ : ℝ) (dist_YW : ℝ)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : ℝ :=
  dist O1 O2

theorem find_min_length_seg_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ := 1)
  (dist_YZ : ℝ := 3)
  (dist_YW : ℝ := 5)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : minimum_length_O1O2 X Y Z W dist_XY dist_YZ dist_YW O1 O2 circumcenter1 circumcenter2 h1 h2 h3 hO1 hO2 = 2 :=
sorry

end find_min_length_seg_O1O2_l256_256115


namespace smallest_dividend_l256_256145

   theorem smallest_dividend (b a : ℤ) (q : ℤ := 12) (r : ℤ := 3) (h : a = b * q + r) (h' : r < b) : a = 51 :=
   by
     sorry
   
end smallest_dividend_l256_256145


namespace four_is_square_root_of_sixteen_l256_256472

theorem four_is_square_root_of_sixteen : (4 : ℝ) * (4 : ℝ) = 16 :=
by
  sorry

end four_is_square_root_of_sixteen_l256_256472


namespace arccos_of_half_eq_pi_over_three_l256_256835

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256835


namespace distance_to_left_focus_l256_256065

theorem distance_to_left_focus (P : ℝ × ℝ) 
  (h1 : P.1^2 / 100 + P.2^2 / 36 = 1) 
  (h2 : dist P (50 - 100 / 9, P.2) = 17 / 2) :
  dist P (-50 - 100 / 9, P.2) = 66 / 5 :=
sorry

end distance_to_left_focus_l256_256065


namespace cube_root_power_simplify_l256_256609

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l256_256609


namespace derivative_of_exp_sin_l256_256221

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (fun x => Real.exp x * Real.sin x)) x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
sorry

end derivative_of_exp_sin_l256_256221


namespace polygon_sides_l256_256225

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l256_256225


namespace binary_to_octal_101110_l256_256319

theorem binary_to_octal_101110 : 
  ∀ (binary_to_octal : ℕ → ℕ), 
  binary_to_octal 0b101110 = 0o56 :=
by
  sorry

end binary_to_octal_101110_l256_256319


namespace find_r_l256_256750

theorem find_r (r : ℝ) (h_curve : r = -2 * r^2 + 5 * r - 2) : r = 1 :=
sorry

end find_r_l256_256750


namespace probability_not_E_four_spins_l256_256062

theorem probability_not_E_four_spins :
  (5/6)^4 = 625/1296 :=
by
  sorry

end probability_not_E_four_spins_l256_256062


namespace arccos_half_eq_pi_div_three_l256_256905

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256905


namespace least_number_to_subtract_l256_256140

theorem least_number_to_subtract (x : ℕ) :
  (2590 - x) % 9 = 6 ∧ 
  (2590 - x) % 11 = 6 ∧ 
  (2590 - x) % 13 = 6 ↔ 
  x = 16 := 
sorry

end least_number_to_subtract_l256_256140


namespace cost_per_pancake_correct_l256_256415

-- Define the daily rent expense
def daily_rent := 30

-- Define the daily supplies expense
def daily_supplies := 12

-- Define the number of pancakes needed to cover expenses
def number_of_pancakes := 21

-- Define the total daily expenses
def total_daily_expenses := daily_rent + daily_supplies

-- Define the cost per pancake calculation
def cost_per_pancake := total_daily_expenses / number_of_pancakes

-- The theorem to prove the cost per pancake
theorem cost_per_pancake_correct :
  cost_per_pancake = 2 := 
by
  sorry

end cost_per_pancake_correct_l256_256415


namespace square_area_proof_l256_256497

theorem square_area_proof
  (x s : ℝ)
  (h1 : x^2 = 3 * s)
  (h2 : 4 * x = s^2) :
  x^2 = 6 :=
  sorry

end square_area_proof_l256_256497


namespace how_many_women_left_l256_256575

theorem how_many_women_left
  (M W : ℕ) -- Initial number of men and women
  (h_ratio : 5 * M = 4 * W) -- Initial ratio 4:5
  (h_men_final : M + 2 = 14) -- 2 men entered the room to make it 14 men
  (h_women_final : 2 * (W - x) = 24) -- Some women left, number of women doubled to 24
  :
  x = 3 := 
sorry

end how_many_women_left_l256_256575


namespace arithmetic_sequence_sum_l256_256032

-- Define the variables and conditions
def a : ℕ := 71
def d : ℕ := 2
def l : ℕ := 99

-- Calculate the number of terms in the sequence
def n : ℕ := ((l - a) / d) + 1

-- Define the sum of the arithmetic sequence
def S : ℕ := (n * (a + l)) / 2

-- Statement to be proven
theorem arithmetic_sequence_sum :
  3 * S = 3825 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_sum_l256_256032


namespace vertical_asymptote_x_value_l256_256714

theorem vertical_asymptote_x_value (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 :=
by
  sorry

end vertical_asymptote_x_value_l256_256714


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l256_256380

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l256_256380


namespace estimate_of_high_scores_l256_256676

open ProbabilityTheory

noncomputable def number_of_students_with_high_scores : ℝ :=
  let n := 100 in -- Number of students
  let mu := 100 in -- Mean of the normal distribution
  let sigma := 10 in -- Standard deviation of the normal distribution
  let ξ := NormalDistr μ σ in -- Normal distribution with mean 100 and standard deviation 10
  let p90_100 := 0.3 in -- Given probability P(90 ≤ ξ ≤ 100)
  let p110_plus := (0.5 - p90_100) in -- Calculated probability P(ξ ≥ 110)
  p110_plus * n -- Number of students with math scores ≥ 110

theorem estimate_of_high_scores : number_of_students_with_high_scores = 20 := sorry

end estimate_of_high_scores_l256_256676


namespace solution_set_intersection_l256_256212

theorem solution_set_intersection (a b : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2) →
  (∀ x : ℝ, x^2 + a * x + b < 0 ↔ (-1 < x ∧ x < 2)) →
  a + b = -3 :=
by 
  sorry

end solution_set_intersection_l256_256212


namespace arccos_half_eq_pi_div_three_l256_256867

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256867


namespace part1_part2_part3_l256_256294

noncomputable def functional_relationship (x : ℝ) : ℝ := -x + 26

theorem part1 (x y : ℝ) (hx6 : x = 6 ∧ y = 20) (hx8 : x = 8 ∧ y = 18) (hx10 : x = 10 ∧ y = 16) :
  ∀ (x : ℝ), functional_relationship x = -x + 26 := 
by
  sorry

theorem part2 (x : ℝ) (h_price_range : 6 ≤ x ∧ x ≤ 12) : 
  14 ≤ functional_relationship x ∧ functional_relationship x ≤ 20 :=
by
  sorry

noncomputable def gross_profit (x : ℝ) : ℝ := x * (functional_relationship x - 4)

theorem part3 (hx : 1 ≤ x) (hy : functional_relationship x ≤ 10):
  gross_profit (16 : ℝ) = 120 :=
by
  sorry

end part1_part2_part3_l256_256294


namespace clock_hand_swap_times_l256_256466

noncomputable def time_between_2_and_3 : ℚ := (2 * 143 + 370) / 143
noncomputable def time_between_6_and_7 : ℚ := (6 * 143 + 84) / 143

theorem clock_hand_swap_times :
  time_between_2_and_3 = 2 + 31 * 7 / 143 ∧
  time_between_6_and_7 = 6 + 12 * 84 / 143 :=
by
  -- Math proof will go here
  sorry

end clock_hand_swap_times_l256_256466


namespace arccos_half_eq_pi_div_three_l256_256848

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256848


namespace olivia_total_earnings_l256_256992

variable (rate : ℕ) (hours_monday : ℕ) (hours_wednesday : ℕ) (hours_friday : ℕ)

def olivia_earnings : ℕ := hours_monday * rate + hours_wednesday * rate + hours_friday * rate

theorem olivia_total_earnings :
  rate = 9 → hours_monday = 4 → hours_wednesday = 3 → hours_friday = 6 → olivia_earnings rate hours_monday hours_wednesday hours_friday = 117 :=
by
  sorry

end olivia_total_earnings_l256_256992


namespace arccos_half_eq_pi_div_three_l256_256872

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256872


namespace range_of_a_for_two_unequal_roots_l256_256539

theorem range_of_a_for_two_unequal_roots (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * Real.log x₁ = x₁ ∧ a * Real.log x₂ = x₂) ↔ a > Real.exp 1 :=
sorry

end range_of_a_for_two_unequal_roots_l256_256539


namespace part1_part2_l256_256068

def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part1 (x : ℝ) : (f x 2 > 2) ↔ (x > 3 / 2) :=
sorry

theorem part2 (a : ℝ) (ha : a > 0) : (∀ x, f x a < 2 * a) ↔ (1 < a) :=
sorry

end part1_part2_l256_256068


namespace function_increasing_range_l256_256119

theorem function_increasing_range (a : ℝ) : 
    (∀ x : ℝ, x ≥ 4 → (2*x + 2*(a-1)) > 0) ↔ a ≥ -3 := 
by
  sorry

end function_increasing_range_l256_256119


namespace other_train_length_l256_256281

noncomputable def relative_speed (speed1 speed2 : ℝ) : ℝ :=
  speed1 + speed2

noncomputable def speed_in_km_per_sec (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr / 3600

noncomputable def total_distance_crossed (relative_speed : ℝ) (time_sec : ℕ) : ℝ :=
  relative_speed * (time_sec : ℝ)

noncomputable def length_of_other_train (total_distance length_of_first_train : ℝ) : ℝ :=
  total_distance - length_of_first_train

theorem other_train_length :
  let speed1 := 210
  let speed2 := 90
  let length_of_first_train := 0.9
  let time_taken := 24
  let relative_speed_km_per_hr := relative_speed speed1 speed2
  let relative_speed_km_per_sec := speed_in_km_per_sec relative_speed_km_per_hr
  let total_distance := total_distance_crossed relative_speed_km_per_sec time_taken
  length_of_other_train total_distance length_of_first_train = 1.1 := 
by
  sorry

end other_train_length_l256_256281


namespace circle_passing_through_points_l256_256587

noncomputable def parabola (x: ℝ) (a b: ℝ) : ℝ :=
  x^2 + a * x + b

theorem circle_passing_through_points (a b α β k: ℝ) :
  parabola 0 a b = b ∧ parabola α a b = 0 ∧ parabola β a b = 0 ∧
  ((0 - (α + β) / 2)^2 + (1 - k)^2 = ((α + β) / 2)^2 + (k - b)^2) →
  b = 1 :=
by
  sorry

end circle_passing_through_points_l256_256587


namespace max_f_eq_find_a_l256_256961

open Real

noncomputable def f (α : ℝ) : ℝ :=
  let a := (sin α, cos α)
  let b := (6 * sin α + cos α, 7 * sin α - 2 * cos α)
  a.1 * b.1 + a.2 * b.2

theorem max_f_eq : 
  ∃ α : ℝ, f α = 4 * sqrt 2 + 2 :=
sorry

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_A : ℝ)

noncomputable def f_triangle (A : ℝ) : ℝ :=
  let a := (sin A, cos A)
  let b := (6 * sin A + cos A, 7 * sin A - 2 * cos A)
  a.1 * b.1 + a.2 * b.2

axiom f_A_eq (A : ℝ) : f_triangle A = 6

theorem find_a (A B C a b c : ℝ) (h₁ : f_triangle A = 6) (h₂ : 1 / 2 * b * c * sin A = 3) (h₃ : b + c = 2 + 3 * sqrt 2) :
  a = sqrt 10 :=
sorry

end max_f_eq_find_a_l256_256961


namespace calculate_perimeter_l256_256117

def four_squares_area : ℝ := 144 -- total area of the figure in cm²
noncomputable def area_of_one_square : ℝ := four_squares_area / 4 -- area of one square in cm²
noncomputable def side_length_of_square : ℝ := Real.sqrt area_of_one_square -- side length of one square in cm

def number_of_vertical_segments : ℕ := 4 -- based on the arrangement
def number_of_horizontal_segments : ℕ := 6 -- based on the arrangement

noncomputable def total_perimeter : ℝ := (number_of_vertical_segments + number_of_horizontal_segments) * side_length_of_square

theorem calculate_perimeter : total_perimeter = 60 := by
  sorry

end calculate_perimeter_l256_256117


namespace find_a_l256_256209

theorem find_a (a b c : ℕ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c) (h_eq : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end find_a_l256_256209


namespace sqrt_seven_lt_three_l256_256035

theorem sqrt_seven_lt_three : Real.sqrt 7 < 3 :=
by
  sorry

end sqrt_seven_lt_three_l256_256035


namespace unique_fish_total_l256_256754

-- Define the conditions as stated in the problem
def Micah_fish : ℕ := 7
def Kenneth_fish : ℕ := 3 * Micah_fish
def Matthias_fish : ℕ := Kenneth_fish - 15
def combined_fish : ℕ := Micah_fish + Kenneth_fish + Matthias_fish
def Gabrielle_fish : ℕ := 2 * combined_fish

def shared_fish_Micah_Matthias : ℕ := 4
def shared_fish_Kenneth_Gabrielle : ℕ := 6

-- Define the total unique fish computation
def total_unique_fish : ℕ := (Micah_fish + Kenneth_fish + Matthias_fish + Gabrielle_fish) - (shared_fish_Micah_Matthias + shared_fish_Kenneth_Gabrielle)

-- State the theorem
theorem unique_fish_total : total_unique_fish = 92 := by
  -- Proof omitted
  sorry

end unique_fish_total_l256_256754


namespace only_zero_sol_l256_256329

theorem only_zero_sol (x y z t : ℤ) : x^2 + y^2 + z^2 + t^2 = 2 * x * y * z * t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end only_zero_sol_l256_256329


namespace monotonicity_of_f_tangent_intersection_l256_256362

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l256_256362


namespace soccer_team_players_l256_256495

theorem soccer_team_players
  (first_half_starters : ℕ)
  (first_half_subs : ℕ)
  (second_half_mult : ℕ)
  (did_not_play : ℕ)
  (players_prepared : ℕ) :
  first_half_starters = 11 →
  first_half_subs = 2 →
  second_half_mult = 2 →
  did_not_play = 7 →
  players_prepared = 20 :=
by
  -- Proof steps go here
  sorry

end soccer_team_players_l256_256495


namespace travel_time_is_correct_l256_256006

-- Define the conditions
def speed : ℕ := 60 -- Speed in km/h
def distance : ℕ := 120 -- Distance between points A and B in km

-- Time calculation from A to B 
def time_AB : ℕ := distance / speed

-- Time calculation from B to A (since speed and distance are the same)
def time_BA : ℕ := distance / speed

-- Total time calculation
def total_time : ℕ := time_AB + time_BA

-- The proper statement to prove
theorem travel_time_is_correct : total_time = 4 := by
  -- Additional steps and arguments would go here
  -- skipping proof
  sorry

end travel_time_is_correct_l256_256006


namespace convert_to_polar_coordinates_l256_256517

open Real

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := sqrt (x^2 + y^2)
  let θ := if y < 0 then 2 * π - arctan (abs y / abs x) else arctan (abs y / abs x)
  (r, θ)

theorem convert_to_polar_coordinates : 
  polar_coordinates 3 (-3) = (3 * sqrt 2, 7 * π / 4) :=
by
  sorry

end convert_to_polar_coordinates_l256_256517


namespace polar_to_rectangular_l256_256516

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 3) :
    (r * Real.cos θ, r * Real.sin θ) = (2, 2 * Real.sqrt 3) :=
by
  rw [h_r, h_θ]
  norm_num
  rw [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  exact ⟨rfl, rfl⟩

end polar_to_rectangular_l256_256516


namespace initial_extra_planks_l256_256832

-- Definitions corresponding to the conditions
def charlie_planks : Nat := 10
def father_planks : Nat := 10
def total_planks : Nat := 35

-- The proof problem statement
theorem initial_extra_planks : total_planks - (charlie_planks + father_planks) = 15 := by
  sorry

end initial_extra_planks_l256_256832


namespace julia_tuesday_kids_l256_256416

theorem julia_tuesday_kids :
  ∃ x : ℕ, (∃ y : ℕ, y = 6 ∧ y = x + 1) → x = 5 := 
by
  sorry

end julia_tuesday_kids_l256_256416


namespace number_of_meters_sold_l256_256021

-- Define the given conditions
def price_per_meter : ℕ := 436 -- in kopecks
def total_revenue_end : ℕ := 728 -- in kopecks
def max_total_revenue : ℕ := 50000 -- in kopecks

-- State the problem formally in Lean 4
theorem number_of_meters_sold (x : ℕ) :
  price_per_meter * x ≡ total_revenue_end [MOD 1000] ∧
  price_per_meter * x ≤ max_total_revenue →
  x = 98 :=
sorry

end number_of_meters_sold_l256_256021


namespace ratio_of_managers_to_non_managers_l256_256082

theorem ratio_of_managers_to_non_managers 
  (M N : ℕ) 
  (hM : M = 9) 
  (hN : N = 47) : 
  M.gcd N = 1 ∧ M / N = 9 / 47 := 
by {
  -- Proof is omitted
  sorry
}

end ratio_of_managers_to_non_managers_l256_256082


namespace smaller_circle_x_coordinate_l256_256779

theorem smaller_circle_x_coordinate (h : ℝ) 
  (P : ℝ × ℝ) (S : ℝ × ℝ)
  (H1 : P = (9, 12))
  (H2 : S = (h, 0))
  (r_large : ℝ)
  (r_small : ℝ)
  (H3 : r_large = 15)
  (H4 : r_small = 10) :
  S.1 = 10 ∨ S.1 = -10 := 
sorry

end smaller_circle_x_coordinate_l256_256779


namespace megan_average_speed_l256_256187

theorem megan_average_speed :
  ∃ s : ℕ, s = 100 / 3 ∧ ∃ (o₁ o₂ : ℕ), o₁ = 27472 ∧ o₂ = 27572 ∧ o₂ - o₁ = 100 :=
by
  sorry

end megan_average_speed_l256_256187


namespace total_rainfall_over_3_days_l256_256577

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l256_256577


namespace expand_product_correct_l256_256701

noncomputable def expand_product (x : ℝ) : ℝ :=
  3 * (x + 4) * (x + 5)

theorem expand_product_correct (x : ℝ) :
  expand_product x = 3 * x^2 + 27 * x + 60 :=
by
  unfold expand_product
  sorry

end expand_product_correct_l256_256701


namespace ratio_of_volumes_l256_256080

theorem ratio_of_volumes (r1 r2 : ℝ) (h : (4 * π * r1^2) / (4 * π * r2^2) = 4 / 9) :
  (4/3 * π * r1^3) / (4/3 * π * r2^3) = 8 / 27 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_volumes_l256_256080


namespace number_div_addition_l256_256142

-- Define the given conditions
def original_number (q d r : ℕ) : ℕ := (q * d) + r

theorem number_div_addition (q d r a b : ℕ) (h1 : d = 6) (h2 : q = 124) (h3 : r = 4) (h4 : a = 24) (h5 : b = 8) :
  ((original_number q d r + a) / b : ℚ) = 96.5 :=
by 
  sorry

end number_div_addition_l256_256142


namespace probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l256_256018

noncomputable def total_members := 42
noncomputable def boys := 28
noncomputable def girls := 14
noncomputable def selected := 6

theorem probability_athlete_A_selected :
  (selected : ℚ) / total_members = 1 / 7 :=
by sorry

theorem number_of_males_selected :
  (selected * (boys : ℚ)) / total_members = 4 :=
by sorry

theorem number_of_females_selected :
  (selected * (girls : ℚ)) / total_members = 2 :=
by sorry

end probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l256_256018


namespace shortest_fence_length_l256_256160

-- We define the conditions given in the problem.
def triangle_side_length : ℕ := 50
def number_of_dotted_lines : ℕ := 13

-- We need to prove that the shortest total length of the fences required to protect all the cabbage from goats equals 650 meters.
theorem shortest_fence_length : number_of_dotted_lines * triangle_side_length = 650 :=
by
  -- The proof steps are omitted as per instructions.
  sorry

end shortest_fence_length_l256_256160


namespace fraction_to_decimal_l256_256514

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
sorry

end fraction_to_decimal_l256_256514


namespace production_line_probabilities_l256_256031

noncomputable def production_lines : Type :=
  { p : ℕ × ℕ × ℕ // p.1 + p.2 + p.3 = 100 }

def p_H1 : ℝ := 0.30
def p_H2 : ℝ := 0.25
def p_H3 : ℝ := 0.45

def p_A_given_H1 : ℝ := 0.03
def p_A_given_H2 : ℝ := 0.02
def p_A_given_H3 : ℝ := 0.04

noncomputable def p_A :=
  (p_A_given_H1 * p_H1) +
  (p_A_given_H2 * p_H2) +
  (p_A_given_H3 * p_H3)

def p_H1_given_A :=
  (p_H1 * p_A_given_H1) / p_A

def p_H2_given_A :=
  (p_H2 * p_A_given_H2) / p_A

def p_H3_given_A :=
  (p_H3 * p_A_given_H3) / p_A

theorem production_line_probabilities:
  p_H1_given_A = 0.281 ∧
  p_H2_given_A = 0.156 ∧
  p_H3_given_A = 0.563 :=
sorry

end production_line_probabilities_l256_256031


namespace probability_neither_perfect_square_nor_cube_l256_256644

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l256_256644


namespace carrots_chloe_l256_256034

theorem carrots_chloe (c_i c_t c_p : ℕ) (H1 : c_i = 48) (H2 : c_t = 45) (H3 : c_p = 42) : 
  c_i - c_t + c_p = 45 := by
  sorry

end carrots_chloe_l256_256034


namespace number_put_in_machine_l256_256681

theorem number_put_in_machine (x : ℕ) (y : ℕ) (h1 : y = x + 15 - 6) (h2 : y = 77) : x = 68 :=
by
  sorry

end number_put_in_machine_l256_256681


namespace total_storage_l256_256979

variable (barrels largeCasks smallCasks : ℕ)
variable (cap_barrel cap_largeCask cap_smallCask : ℕ)

-- Given conditions
axiom h1 : barrels = 4
axiom h2 : largeCasks = 3
axiom h3 : smallCasks = 5
axiom h4 : cap_largeCask = 20
axiom h5 : cap_smallCask = cap_largeCask / 2
axiom h6 : cap_barrel = 2 * cap_largeCask + 3

-- Target statement
theorem total_storage : 4 * cap_barrel + 3 * cap_largeCask + 5 * cap_smallCask = 282 := 
by
  -- Proof is not required
  sorry

end total_storage_l256_256979


namespace original_number_is_45_l256_256554

theorem original_number_is_45 (x : ℕ) (h : x - 30 = x / 3) : x = 45 :=
by {
  sorry
}

end original_number_is_45_l256_256554


namespace probability_at_least_one_special_l256_256011

-- Define the number of each type of cards
def numDiamonds := 13
def numAces := 4
def numKings := 4
def totalDeckSize := 52 + numKings
def numSpecialCards := numDiamonds + numAces + numKings - 1 -- Count ace of diamonds only once

-- Probability calculations
def probNotSpecial := (totalDeckSize - numSpecialCards : ℚ) / totalDeckSize
def probBothNotSpecial := probNotSpecial * probNotSpecial
def probAtLeastOneSpecial := 1 - probBothNotSpecial

-- Theorems showing intermediate steps and the final answer
theorem probability_at_least_one_special :
  probAtLeastOneSpecial = 115 / 196 :=
by sorry

end probability_at_least_one_special_l256_256011


namespace total_notebooks_distributed_l256_256475

theorem total_notebooks_distributed :
  ∀ (N C : ℕ), 
    (N / C = C / 8) →
    (N = 16 * (C / 2)) →
    N = 512 := 
by
  sorry

end total_notebooks_distributed_l256_256475


namespace monotonicity_and_tangent_intersection_l256_256367

def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersection :
  ∀ a : ℝ,
  (if a ≥ 1/3 then ∀ x : ℝ, diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) ≥ 0 else 
    (∀ x : ℝ, x < (1 - real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0) ∧
    (∀ x : ℝ, (1 - real.sqrt(1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) < 0) ∧
    (∀ x : ℝ, x > (1 + real.sqrt(1 - 3 * a)) / 3 ∧ diffable ℝ ℝ (λ x : ℝ, f x a) x ∧ (derivative (λ x : ℝ, f x a) x) > 0)) ∧
  (let px1 := 1, px2 := -1 in (∃ y1 y2 : ℝ, f 1 a = y1 ∧ f (-1) a = y2 ∧ y1 = a + 1 ∧ y2 = -a - 1)) :=
sorry

end monotonicity_and_tangent_intersection_l256_256367


namespace sin_double_angle_sin_multiple_angle_l256_256799

-- Prove that |sin(2x)| <= 2|sin(x)| for any value of x
theorem sin_double_angle (x : ℝ) : |Real.sin (2 * x)| ≤ 2 * |Real.sin x| := 
by sorry

-- Prove that |sin(nx)| <= n|sin(x)| for any positive integer n and any value of x
theorem sin_multiple_angle (n : ℕ) (x : ℝ) (h : 0 < n) : |Real.sin (n * x)| ≤ n * |Real.sin x| :=
by sorry

end sin_double_angle_sin_multiple_angle_l256_256799


namespace polygon_sides_l256_256227

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l256_256227


namespace probability_neither_perfect_square_nor_cube_l256_256642

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l256_256642


namespace toy_cost_price_l256_256169

theorem toy_cost_price (C : ℕ) (h : 18 * C + 3 * C = 25200) : C = 1200 := by
  -- The proof is not required
  sorry

end toy_cost_price_l256_256169


namespace temperature_difference_l256_256121

theorem temperature_difference (T_south T_north : ℝ) (h_south : T_south = 6) (h_north : T_north = -3) :
  T_south - T_north = 9 :=
by 
  -- Proof goes here
  sorry

end temperature_difference_l256_256121


namespace fence_cost_l256_256141

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (side_length perimeter cost : ℝ) 
  (h1 : area = 289) 
  (h2 : price_per_foot = 55)
  (h3 : side_length = Real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : cost = perimeter * price_per_foot) :
  cost = 3740 := 
sorry

end fence_cost_l256_256141


namespace arccos_of_half_eq_pi_over_three_l256_256839

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256839


namespace choosing_one_student_is_50_l256_256083

-- Define the number of male students and female students
def num_male_students : Nat := 26
def num_female_students : Nat := 24

-- Define the total number of ways to choose one student
def total_ways_to_choose_one_student : Nat := num_male_students + num_female_students

-- Theorem statement proving the total number of ways to choose one student is 50
theorem choosing_one_student_is_50 : total_ways_to_choose_one_student = 50 := by
  sorry

end choosing_one_student_is_50_l256_256083


namespace hydrogen_atomic_weight_is_correct_l256_256706

-- Definitions and assumptions based on conditions
def molecular_weight : ℝ := 68
def number_of_hydrogen_atoms : ℕ := 1
def number_of_chlorine_atoms : ℕ := 1
def number_of_oxygen_atoms : ℕ := 2
def atomic_weight_chlorine : ℝ := 35.45
def atomic_weight_oxygen : ℝ := 16.00

-- Definition for the atomic weight of hydrogen to be proved
def atomic_weight_hydrogen (w : ℝ) : Prop :=
  w * number_of_hydrogen_atoms
  + atomic_weight_chlorine * number_of_chlorine_atoms
  + atomic_weight_oxygen * number_of_oxygen_atoms = molecular_weight

-- The theorem to prove the atomic weight of hydrogen
theorem hydrogen_atomic_weight_is_correct : atomic_weight_hydrogen 1.008 :=
by
  unfold atomic_weight_hydrogen
  simp
  sorry

end hydrogen_atomic_weight_is_correct_l256_256706


namespace collinear_points_in_triangle_l256_256247

noncomputable section

open EuclideanGeometry

variables (A B C D J M : Point)

-- Assume the triangle and the relevant points in the plane
variable (hABC : Triangle A B C)

-- D is the point of tangency of the incircle with side BC
variable (hD : inCircleTangentPoint D (side B C) (inCircle A B C))

-- J is the center of the excircle opposite vertex A
variable (hJ : J = exCircleCenterOpposite A B C)

-- M is the midpoint of the altitude from vertex A
variable (hM : isMidpoint M (altitudeFromVertex A B C))

-- Prove D, M, and J are collinear
theorem collinear_points_in_triangle : collinear {D, M, J} :=
  sorry

end collinear_points_in_triangle_l256_256247


namespace fraction_books_sold_l256_256295

theorem fraction_books_sold (B : ℕ) (F : ℚ)
  (hc1 : F * B * 4 = 288)
  (hc2 : F * B + 36 = B) :
  F = 2 / 3 :=
by
  sorry

end fraction_books_sold_l256_256295


namespace length_of_bridge_l256_256442

theorem length_of_bridge (length_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (total_distance : ℝ) (bridge_length : ℝ) :
  length_train = 160 →
  speed_kmh = 45 →
  time_sec = 30 →
  speed_ms = 45 * (1000 / 3600) →
  total_distance = speed_ms * time_sec →
  bridge_length = total_distance - length_train →
  bridge_length = 215 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end length_of_bridge_l256_256442


namespace cricket_bat_weight_l256_256323

-- Define the conditions as Lean definitions
def weight_of_basketball : ℕ := 36
def weight_of_basketballs (n : ℕ) := n * weight_of_basketball
def weight_of_cricket_bats (m : ℕ) := m * (weight_of_basketballs 4 / 8)

-- State the theorem and skip the proof
theorem cricket_bat_weight :
  weight_of_cricket_bats 1 = 18 :=
by
  sorry

end cricket_bat_weight_l256_256323


namespace calculate_max_income_l256_256235

variables 
  (total_lunch_pasta : ℕ) (total_lunch_chicken : ℕ) (total_lunch_fish : ℕ)
  (sold_lunch_pasta : ℕ) (sold_lunch_chicken : ℕ) (sold_lunch_fish : ℕ)
  (dinner_pasta : ℕ) (dinner_chicken : ℕ) (dinner_fish : ℕ)
  (price_pasta : ℝ) (price_chicken : ℝ) (price_fish : ℝ)
  (discount : ℝ)
  (max_income : ℝ)

def unsold_lunch_pasta := total_lunch_pasta - sold_lunch_pasta
def unsold_lunch_chicken := total_lunch_chicken - sold_lunch_chicken
def unsold_lunch_fish := total_lunch_fish - sold_lunch_fish

def discounted_price (price : ℝ) := price * (1 - discount)

def income_lunch (sold : ℕ) (price : ℝ) := sold * price
def income_dinner (fresh : ℕ) (price : ℝ) := fresh * price
def income_unsold (unsold : ℕ) (price : ℝ) := unsold * discounted_price price

theorem calculate_max_income 
  (h_pasta_total : total_lunch_pasta = 8) (h_chicken_total : total_lunch_chicken = 5) (h_fish_total : total_lunch_fish = 4)
  (h_pasta_sold : sold_lunch_pasta = 6) (h_chicken_sold : sold_lunch_chicken = 3) (h_fish_sold : sold_lunch_fish = 3)
  (h_dinner_pasta : dinner_pasta = 2) (h_dinner_chicken : dinner_chicken = 2) (h_dinner_fish : dinner_fish = 1)
  (h_price_pasta: price_pasta = 12) (h_price_chicken: price_chicken = 15) (h_price_fish: price_fish = 18)
  (h_discount: discount = 0.10) 
  : max_income = 136.80 :=
  sorry

end calculate_max_income_l256_256235


namespace line_to_cartesian_eq_line_and_curve_common_points_l256_256094

theorem line_to_cartesian_eq (m : ℝ) :
  ∀ (ρ θ : ℝ), (ρ * Real.sin (θ + Real.pi / 3) + m = 0) ↔ (sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ + 2 * m = 0) :=
by sorry

theorem line_and_curve_common_points (m : ℝ) :
  (∃ (t : ℝ), sqrt 3 * Real.cos (2 * t) * sqrt 3 + 2 * Real.sin t + 2 * m = 0) ↔ (m ∈ Set.Icc (-19 / 12 : ℝ) (5 / 2 : ℝ)) :=
by sorry

end line_to_cartesian_eq_line_and_curve_common_points_l256_256094


namespace fabulous_integers_l256_256686

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (a^n - a) % n = 0

theorem fabulous_integers (n : ℕ) : is_fabulous n ↔ ¬(∃ k : ℕ, n = 2^k ∧ k ≥ 1) := 
sorry

end fabulous_integers_l256_256686


namespace min_value_inequality_l256_256987

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 * x / (2 * y + z) + 3 * y / (x + 2 * z) + 9 * z / (x + y) ≥ 83 :=
sorry

end min_value_inequality_l256_256987


namespace cost_of_5kg_l256_256158

def cost_of_seeds (x : ℕ) : ℕ :=
  if x ≤ 2 then 5 * x else 4 * x + 2

theorem cost_of_5kg : cost_of_seeds 5 = 22 := by
  sorry

end cost_of_5kg_l256_256158


namespace quotient_of_a_by_b_l256_256008

-- Definitions based on given conditions
def a : ℝ := 0.0204
def b : ℝ := 17

-- Statement to be proven
theorem quotient_of_a_by_b : a / b = 0.0012 := 
by
  sorry

end quotient_of_a_by_b_l256_256008


namespace necessary_condition_not_sufficient_condition_l256_256288

variable (x : ℝ)

def quadratic_condition : Prop := x^2 - 3 * x + 2 > 0
def interval_condition : Prop := x < 1 ∨ x > 4

theorem necessary_condition : interval_condition x → quadratic_condition x := by sorry

theorem not_sufficient_condition : ¬ (quadratic_condition x → interval_condition x) := by sorry

end necessary_condition_not_sufficient_condition_l256_256288


namespace average_star_rating_l256_256269

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l256_256269


namespace race_distance_l256_256684

variable (distance : ℝ)

theorem race_distance :
  (0.25 * distance = 50) → (distance = 200) :=
by
  intro h
  sorry

end race_distance_l256_256684


namespace polar_to_rectangular_coordinates_l256_256515

-- Define the given conditions in Lean 4
def r : ℝ := 4
def theta : ℝ := Real.pi / 3

-- Define the conversion formulas
def x : ℝ := r * Real.cos theta
def y : ℝ := r * Real.sin theta

-- State the proof problem
theorem polar_to_rectangular_coordinates : (x = 2) ∧ (y = 2 * Real.sqrt 3) := by
  -- Sorry is used to indicate the proof is omitted
  sorry

end polar_to_rectangular_coordinates_l256_256515


namespace rate_of_current_l256_256100

def downstream_eq (b c : ℝ) : Prop := (b + c) * 4 = 24
def upstream_eq (b c : ℝ) : Prop := (b - c) * 6 = 24

theorem rate_of_current (b c : ℝ) (h1 : downstream_eq b c) (h2 : upstream_eq b c) : c = 1 :=
by sorry

end rate_of_current_l256_256100


namespace arccos_one_half_is_pi_div_three_l256_256895

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256895


namespace monotonicity_of_f_tangent_intersection_points_l256_256376

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l256_256376


namespace arccos_of_half_eq_pi_over_three_l256_256841

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256841


namespace reggie_loses_by_21_points_l256_256112

-- Define the points for each type of shot.
def layup_points := 1
def free_throw_points := 2
def three_pointer_points := 3
def half_court_points := 5

-- Define Reggie's shot counts.
def reggie_layups := 4
def reggie_free_throws := 3
def reggie_three_pointers := 2
def reggie_half_court_shots := 1

-- Define Reggie's brother's shot counts.
def brother_layups := 3
def brother_free_throws := 2
def brother_three_pointers := 5
def brother_half_court_shots := 4

-- Calculate Reggie's total points.
def reggie_total_points :=
  reggie_layups * layup_points +
  reggie_free_throws * free_throw_points +
  reggie_three_pointers * three_pointer_points +
  reggie_half_court_shots * half_court_points

-- Calculate Reggie's brother's total points.
def brother_total_points :=
  brother_layups * layup_points +
  brother_free_throws * free_throw_points +
  brother_three_pointers * three_pointer_points +
  brother_half_court_shots * half_court_points

-- Calculate the difference in points.
def point_difference := brother_total_points - reggie_total_points

-- Prove that the difference in points Reggie lost by is 21.
theorem reggie_loses_by_21_points : point_difference = 21 := by
  sorry

end reggie_loses_by_21_points_l256_256112


namespace probability_neither_perfect_square_nor_cube_l256_256643

theorem probability_neither_perfect_square_nor_cube :
  let numbers := finset.range 201
  let perfect_squares := finset.filter (λ n, ∃ k, k * k = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ k, k * k * k = n) numbers
  let perfect_sixth_powers := finset.filter (λ n, ∃ k, k * k * k * k * k * k = n) numbers
  let n := finset.card numbers
  let p := finset.card perfect_squares
  let q := finset.card perfect_cubes
  let r := finset.card perfect_sixth_powers
  let neither := n - (p + q - r)
  (neither: ℚ) / n = 183 / 200 := by
  -- proof is omitted
  sorry

end probability_neither_perfect_square_nor_cube_l256_256643


namespace arccos_one_half_l256_256929

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256929


namespace outfits_count_l256_256962

def num_outfits (n : Nat) (total_colors : Nat) : Nat :=
  let total_combinations := n * n * n
  let undesirable_combinations := total_colors
  total_combinations - undesirable_combinations

theorem outfits_count : num_outfits 5 5 = 120 :=
  by
  sorry

end outfits_count_l256_256962


namespace distance_with_father_l256_256243

variable (total_distance driven_with_mother driven_with_father: ℝ)

theorem distance_with_father :
  total_distance = 0.67 ∧ driven_with_mother = 0.17 → driven_with_father = 0.50 := 
by
  sorry

end distance_with_father_l256_256243


namespace parallel_lines_slope_l256_256970

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0) ∧ (∀ x y : ℝ, 2 * x + (m + 5) * y - 8 = 0) →
  m = -7 :=
by
  intro H
  sorry

end parallel_lines_slope_l256_256970


namespace rectangle_enclosed_by_four_lines_l256_256192

theorem rectangle_enclosed_by_four_lines : 
  let h_lines := 5
  let v_lines := 5
  (choose h_lines 2) * (choose v_lines 2) = 100 :=
by {
  sorry
}

end rectangle_enclosed_by_four_lines_l256_256192


namespace fixed_errors_correct_l256_256337

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l256_256337


namespace blueprint_conversion_proof_l256_256157

-- Let inch_to_feet be the conversion factor from blueprint inches to actual feet.
def inch_to_feet : ℝ := 500

-- Let line_segment_inch be the length of the line segment on the blueprint in inches.
def line_segment_inch : ℝ := 6.5

-- Then, line_segment_feet is the actual length of the line segment in feet.
def line_segment_feet : ℝ := line_segment_inch * inch_to_feet

-- Theorem statement to prove
theorem blueprint_conversion_proof : line_segment_feet = 3250 := by
  -- Proof goes here
  sorry

end blueprint_conversion_proof_l256_256157


namespace find_y_l256_256477

theorem find_y (x y : Int) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := 
by 
  sorry

end find_y_l256_256477


namespace average_star_rating_is_four_l256_256263

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end average_star_rating_is_four_l256_256263


namespace women_left_room_is_3_l256_256567

-- Definitions and conditions
variables (M W x : ℕ)
variables (ratio : M * 5 = W * 4) 
variables (men_entered : M + 2 = 14) 
variables (women_left : 2 * (W - x) = 24)

-- Theorem statement
theorem women_left_room_is_3 
  (ratio : M * 5 = W * 4) 
  (men_entered : M + 2 = 14) 
  (women_left : 2 * (W - x) = 24) : 
  x = 3 :=
sorry

end women_left_room_is_3_l256_256567


namespace find_third_card_value_l256_256120

noncomputable def point_values (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b = 25 ∧
  b + c = 13

theorem find_third_card_value :
  ∃ a b c : ℕ, point_values a b c ∧ c = 1 :=
by {
  sorry
}

end find_third_card_value_l256_256120


namespace find_smallest_angle_l256_256709

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l256_256709


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256880

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256880


namespace lcm_multiplied_by_2_is_72x_l256_256159

-- Define the denominators
def denom1 (x : ℕ) := 4 * x
def denom2 (x : ℕ) := 6 * x
def denom3 (x : ℕ) := 9 * x

-- Define the least common multiple of three natural numbers
def lcm_three (a b c : ℕ) := Nat.lcm a (Nat.lcm b c)

-- Define the multiplication by 2
def multiply_by_2 (n : ℕ) := 2 * n

-- Define the final result
def final_result (x : ℕ) := 72 * x

-- The proof statement
theorem lcm_multiplied_by_2_is_72x (x : ℕ): 
  multiply_by_2 (lcm_three (denom1 x) (denom2 x) (denom3 x)) = final_result x := 
by
  sorry

end lcm_multiplied_by_2_is_72x_l256_256159


namespace interval_length_difference_l256_256041

noncomputable def log2_abs (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem interval_length_difference :
  ∀ (a b : ℝ), (∀ x, a ≤ x ∧ x ≤ b → 0 ≤ log2_abs x ∧ log2_abs x ≤ 2) → 
               (b - a = 15 / 4 - 3 / 4) :=
by
  intros a b h
  sorry

end interval_length_difference_l256_256041


namespace parabola_focus_l256_256118

theorem parabola_focus (y x : ℝ) (h : y^2 = 4 * x) : x = 1 → y = 0 → (1, 0) = (1, 0) :=
by 
  sorry

end parabola_focus_l256_256118


namespace polygon_area_l256_256740

theorem polygon_area (sides : ℕ) (perpendicular_adjacent : Bool) (congruent_sides : Bool) (perimeter : ℝ) (area : ℝ) :
  sides = 32 → 
  perpendicular_adjacent = true → 
  congruent_sides = true →
  perimeter = 64 →
  area = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end polygon_area_l256_256740


namespace arccos_half_eq_pi_over_three_l256_256917

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256917


namespace probability_not_square_or_cube_l256_256647

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l256_256647


namespace prime_square_minus_one_divisible_by_24_l256_256427

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (h_prime : Prime p) (h_gt_3 : p > 3) : 
  ∃ k : ℕ, p^2 - 1 = 24 * k := by
sorry

end prime_square_minus_one_divisible_by_24_l256_256427


namespace business_value_l256_256474

-- Define the conditions
variable (V : ℝ) -- Total value of the business
variable (man_shares : ℝ := (2/3) * V) -- Man's share in the business
variable (sold_shares_value : ℝ := (3/4) * man_shares) -- Value of sold shares
variable (sale_price : ℝ := 45000) -- Price the shares were sold for

-- State the theorem to be proven
theorem business_value (h : (3/4) * (2/3) * V = 45000) : V = 90000 := by
  sorry

end business_value_l256_256474


namespace k_satisfies_triangle_condition_l256_256189

theorem k_satisfies_triangle_condition (k : ℤ) 
  (hk_pos : 0 < k) (a b c : ℝ) (ha_pos : 0 < a) 
  (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (h_ineq : (k : ℝ) * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : k = 6 → 
  (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end k_satisfies_triangle_condition_l256_256189


namespace angle_between_lines_at_most_l256_256059
-- Import the entire Mathlib library for general mathematical definitions

-- Define the problem statement in Lean 4
theorem angle_between_lines_at_most (n : ℕ) (h : n > 0) :
  ∃ (l1 l2 : ℝ), l1 ≠ l2 ∧ (n : ℝ) > 0 → ∃ θ, 0 ≤ θ ∧ θ ≤ 180 / n := by
  sorry

end angle_between_lines_at_most_l256_256059


namespace shipping_cost_per_unit_l256_256489

-- Define the conditions
def cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def num_components : ℝ := 150
def lowest_selling_price : ℝ := 196.67

-- Define the revenue and total cost
def total_cost (S : ℝ) : ℝ := (cost_per_component * num_components) + fixed_monthly_cost + (num_components * S)
def total_revenue : ℝ := lowest_selling_price * num_components

-- Define the proposition to be proved
theorem shipping_cost_per_unit (S : ℝ) :
  total_cost S ≤ total_revenue → S ≤ 6.67 :=
by sorry

end shipping_cost_per_unit_l256_256489


namespace monotonicity_of_f_tangent_intersection_points_l256_256377

-- Definitions based on the condition in a)
noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a*x + 1
noncomputable def f' (x a : ℝ) : ℝ := 3*x^2 - 2*x + a

-- Monotonicity problem statement
theorem monotonicity_of_f (a : ℝ) :
  (a >= 1/3 ∧ ∀ x y : ℝ, (x ≤ y) → f x a ≤ f y a) ∨
  (a < 1/3 ∧ ∀ x y : ℝ, (x < 1 - real.sqrt(1-3*a)/3 ∨ x > 1 + real.sqrt(1-3*a)/3) →
    f x a ≤ f y a) ∧
  (a < 1/3 ∧ ∀ x y : ℝ, (1 - real.sqrt(1-3*a)/3 ≤ x ∧ x < y ∧ y ≤ 1 + real.sqrt(1-3*a)/3) →
    f x a < f y a) :=
sorry

-- Tangent intersection problem statement
theorem tangent_intersection_points (x₀ : ℝ) (y₀ : ℝ) (a : ℝ) :
  (y₀ = a + 1 ∧ x₀ = 1) ∨ (y₀ = -a - 1 ∧ x₀ = -1) → 
  (∃ x₀ : ℝ, 2*x₀^3 - x₀^2 - 1 = 0) :=
sorry

end monotonicity_of_f_tangent_intersection_points_l256_256377


namespace trigonometric_identity_l256_256760

open Real

theorem trigonometric_identity :
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  4 * cos_18 ^ 2 - 1 = 1 / (4 * sin_18 ^ 2) :=
by
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  sorry

end trigonometric_identity_l256_256760


namespace problems_left_to_grade_l256_256310

theorem problems_left_to_grade 
  (problems_per_worksheet : ℕ)
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (h1 : problems_per_worksheet = 2)
  (h2 : total_worksheets = 14)
  (h3 : graded_worksheets = 7) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by 
  sorry

end problems_left_to_grade_l256_256310


namespace probability_non_square_non_cube_l256_256633

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l256_256633


namespace points_for_level_completion_l256_256002

-- Condition definitions
def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def total_points : ℕ := 62

-- Derived definitions (based on the problem steps):
def points_from_enemies : ℕ := enemies_defeated * points_per_enemy
def points_for_completing_level : ℕ := total_points - points_from_enemies

-- Theorem statement
theorem points_for_level_completion : points_for_completing_level = 8 := by
  sorry

end points_for_level_completion_l256_256002


namespace minimum_value_expression_l256_256705

theorem minimum_value_expression :
  ∃ x y : ℝ, (∀ a b : ℝ, (a^2 + 4*a*b + 5*b^2 - 8*a - 6*b) ≥ -41) ∧ (x^2 + 4*x*y + 5*y^2 - 8*x - 6*y) = -41 := 
sorry

end minimum_value_expression_l256_256705


namespace monotonicity_f_tangent_points_l256_256372

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l256_256372


namespace difference_in_distances_l256_256448

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (revolutions : ℕ) : ℝ :=
  circumference r * revolutions

theorem difference_in_distances :
  let r1 := 22.4
  let r2 := 34.2
  let revolutions := 400
  let D1 := distance_covered r1 revolutions
  let D2 := distance_covered r2 revolutions
  D2 - D1 = 29628 :=
by
  sorry

end difference_in_distances_l256_256448


namespace spadesuit_value_l256_256518

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l256_256518


namespace vector_combination_l256_256219

-- Definitions for vectors a, b, and c with the conditions provided
def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

-- The statement we want to prove
theorem vector_combination (t m n : ℝ)
  (h : c t = m • a + n • b) :
  t = 11 ∧ m + n = 11 / 2 :=
by
  sorry

end vector_combination_l256_256219


namespace original_number_is_correct_l256_256170

theorem original_number_is_correct (x : ℝ) (h : 10 * x = x + 34.65) : x = 3.85 :=
sorry

end original_number_is_correct_l256_256170


namespace monotonicity_of_f_tangent_line_intersection_coordinates_l256_256378

noncomputable def f (a x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) : 
  (if a ≥ (1 : ℝ) / 3 then ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2 else 
  ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 > f a x2 ∧ ∀ x x' : ℝ, x < x' → 
  ((x < x1 ∨ x > x2) → f a x < f a x' ∧ (x1 < x < x2) → f a x > f a x'))) :=
sorry

theorem tangent_line_intersection_coordinates (a : ℝ) :
   (f a 1 = a + 1) ∧ (f a (-1) = -a - 1) :=
sorry

end monotonicity_of_f_tangent_line_intersection_coordinates_l256_256378


namespace cube_root_power_simplify_l256_256608

theorem cube_root_power_simplify :
  (∛7) ^ 6 = 49 := 
by
  sorry

end cube_root_power_simplify_l256_256608


namespace consecutive_even_numbers_average_35_greatest_39_l256_256617

-- Defining the conditions of the problem
def average_of_even_numbers (n : ℕ) (S : ℕ) : ℕ := (n * S + (2 * n * (n - 1)) / 2) / n

-- Main statement to be proven
theorem consecutive_even_numbers_average_35_greatest_39 : 
  ∃ (n : ℕ), average_of_even_numbers n (38 - (n - 1) * 2) = 35 ∧ (38 - (n - 1) * 2) + (n - 1) * 2 = 38 :=
by
  sorry

end consecutive_even_numbers_average_35_greatest_39_l256_256617


namespace probability_neither_square_nor_cube_l256_256637

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l256_256637


namespace find_n_l256_256055

theorem find_n (n : ℕ) : (10^n = (10^5)^3) → n = 15 :=
by sorry

end find_n_l256_256055


namespace seeds_per_can_l256_256104

theorem seeds_per_can (total_seeds : ℝ) (number_of_cans : ℝ) (h1 : total_seeds = 54.0) (h2 : number_of_cans = 9.0) : (total_seeds / number_of_cans = 6.0) :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end seeds_per_can_l256_256104


namespace optimal_selection_exists_l256_256725

-- Define the given 5x5 matrix
def matrix : Matrix (Fin 5) (Fin 5) ℕ := ![
  ![11, 17, 25, 19, 16],
  ![24, 10, 13, 15, 3],
  ![12, 5, 14, 2, 18],
  ![23, 4, 1, 8, 22],
  ![6, 20, 7, 21, 9]
]

-- Define the statement of the proof problem
theorem optimal_selection_exists:
  ∃ (s : Finset (Fin 5 × Fin 5)), 
    (s.card = 5) ∧
    (∀ (i j : Fin 5 × Fin 5), i ∈ s → j ∈ s → i.1 ≠ j.1 ∧ i.2 ≠ j.2) ∧
    (∀ (t : Finset ℕ), (t = s.image (λ ij, matrix ij.1 ij.2)) → t.min' (by decide) = 15) :=
sorry

end optimal_selection_exists_l256_256725


namespace indigo_restaurant_average_rating_l256_256266

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l256_256266


namespace area_estimation_correct_l256_256473

-- definition for the region Ω under the curve y = x^2 and bounded by y = 4
def region_under_curve (x : ℝ) : ℝ :=
  x ^ 2

-- define the condition for the random point generation.
def point_in_region (a b : ℝ) : Prop :=
  let a1 := 4 * a - 2 in
  let b1 := 4 * b in
  b1 < a1 ^ 2

noncomputable def estimate_area_of_region (m n : ℕ) (output_n : ℕ) : ℝ :=
  let total_area := 16 in -- area of the bounding rectangle
  let probability_in_region := (output_n.to_real) / (m.to_real) in
  total_area * probability_in_region

-- given condition M = 100
def M : ℕ := 100

-- given output n = 34
def output_n : ℕ := 34

theorem area_estimation_correct :
  estimate_area_of_region M output_n = 10.56 :=
by {
  sorry
}

end area_estimation_correct_l256_256473


namespace probability_neither_square_nor_cube_l256_256631

theorem probability_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  (∑ i in Finset.range n, if (∃ k : ℕ, k ^ 2 = i + 1) ∨ (∃ k : ℕ, k ^ 3 = i + 1) then 0 else 1) / n = 183 / 200 := 
by
  sorry

end probability_neither_square_nor_cube_l256_256631


namespace arccos_of_half_eq_pi_over_three_l256_256837

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256837


namespace calculate_expression_l256_256986

theorem calculate_expression (p q : ℝ) (hp : p + q = 7) (hq : p * q = 12) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 3691 := 
by sorry

end calculate_expression_l256_256986


namespace page_numbers_sum_l256_256447

theorem page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 136080) : n + (n + 1) + (n + 2) = 144 :=
by
  sorry

end page_numbers_sum_l256_256447


namespace arccos_half_eq_pi_div_three_l256_256871

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l256_256871


namespace floor_sum_min_value_l256_256396

theorem floor_sum_min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋) = 7 :=
sorry

end floor_sum_min_value_l256_256396


namespace linear_polynomial_divisible_49_l256_256648

theorem linear_polynomial_divisible_49 {P : ℕ → Polynomial ℚ} :
    let Q := Polynomial.C 1 * (Polynomial.X ^ 8) + Polynomial.C 1 * (Polynomial.X ^ 7)
    ∃ a b x, (P x) = Polynomial.C a * Polynomial.X + Polynomial.C b ∧ a ≠ 0 ∧ 
              (∀ i, P (i + 1) = (Polynomial.C 1 * Polynomial.X + Polynomial.C 1) * P i ∨ 
                            P (i + 1) = Polynomial.derivative (P i)) →
              (a - b) % 49 = 0 :=
by
  sorry

end linear_polynomial_divisible_49_l256_256648


namespace complement_and_intersection_l256_256249

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {-2, -1, 0}
def B : Set ℤ := {0, 1, 2}

theorem complement_and_intersection :
  ((U \ A) ∩ B) = {1, 2} := 
by
  sorry

end complement_and_intersection_l256_256249


namespace monotonicity_tangent_intersection_points_l256_256355

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l256_256355


namespace arccos_one_half_l256_256922

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256922


namespace part1_monotonicity_part2_minimum_range_l256_256726

noncomputable def f (k x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

theorem part1_monotonicity (x : ℝ) (h : x ≠ 1) :
    k = 0 → f k x = (x / (x - 1)) * Real.log x ∧ 
    (0 < x ∧ x < 1 ∨ 1 < x) → Monotone (f k) :=
sorry

theorem part2_minimum_range (k : ℝ) :
    (∃ x ∈ Set.Ioi 1, IsLocalMin (f k) x) ↔ k ∈ Set.Ioi 1 :=
sorry

end part1_monotonicity_part2_minimum_range_l256_256726


namespace probability_neither_square_nor_cube_l256_256626

theorem probability_neither_square_nor_cube :
  let count_squares := 14
  let count_cubes := 5
  let overlap := 2
  let total_range := 200
  let neither_count := total_range - (count_squares + count_cubes - overlap)
  let probability := (neither_count : ℚ) / total_range
  probability = 183 / 200 :=
by {
  sorry
}

end probability_neither_square_nor_cube_l256_256626


namespace shortest_chord_eqn_of_circle_l256_256081

theorem shortest_chord_eqn_of_circle 
    (k x y : ℝ)
    (C_eq : x^2 + y^2 - 2*x - 24 = 0)
    (line_l : y = k * (x - 2) - 1) :
  y = x - 3 :=
by
  sorry

end shortest_chord_eqn_of_circle_l256_256081


namespace arccos_one_half_l256_256889

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256889


namespace trajectory_of_M_l256_256535

variables {x y : ℝ}

theorem trajectory_of_M (h : y / (x + 2) + y / (x - 2) = 2) (hx : x ≠ 2) (hx' : x ≠ -2) :
  x * y - x^2 + 4 = 0 :=
by sorry

end trajectory_of_M_l256_256535


namespace monotonicity_of_f_tangent_intersection_l256_256361

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 / 3) → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 / 3) → 
    ∀ x : ℝ, 
      (x < (1 - real.sqrt (1 - 3 * a)) / 3 ∨ x > (1 + real.sqrt (1 - 3 * a)) / 3) → 
      f x a ≥ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ 
      f x a ≥ f ((1 + real.sqrt (1 - 3 * a)) / 3) a ∧
      ((1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f ((1 - real.sqrt (1 - 3 * a)) / 3) a ∧ f x a ≤ f ((1 + real.sqrt (1 - 3 * a)) / 3) a)) :=
sorry

theorem tangent_intersection (a : ℝ) :
  (∃ x0 : ℝ, 2 * x0^3 - x0^2 - 1 = 0 ∧ (f x0 a = (f x0 a) * x0) ∧ 
  (x0 = 1 ∨ x0 = -1) ∧
  ((x0 = 1 → (1, a + 1) ∈ set.range (λ x : ℝ, (x, f x a))) ∧
  (x0 = -1 → (-1, -a - 1) ∈ set.range (λ x : ℝ, (x, f x a))))) :=
sorry

end monotonicity_of_f_tangent_intersection_l256_256361


namespace smallest_number_l256_256479

theorem smallest_number (a b c d e : ℕ) (h₁ : a = 12) (h₂ : b = 16) (h₃ : c = 18) (h₄ : d = 21) (h₅ : e = 28) : 
    ∃ n : ℕ, (n - 4) % Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 0 ∧ n = 1012 :=
by
    sorry

end smallest_number_l256_256479


namespace arccos_half_is_pi_div_three_l256_256854

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256854


namespace integer_coefficient_equation_calculate_expression_l256_256509

noncomputable def a : ℝ := (Real.sqrt 5 - 1) / 2

theorem integer_coefficient_equation :
  a ^ 2 + a - 1 = 0 :=
sorry

theorem calculate_expression :
  a ^ 3 - 2 * a + 2015 = 2014 :=
sorry

end integer_coefficient_equation_calculate_expression_l256_256509


namespace train_speeds_proof_l256_256456

-- Defining the initial conditions
variables (v_g v_p v_e : ℝ)
variables (t_g t_p t_e : ℝ) -- t_g, t_p, t_e are the times for goods, passenger, and express trains respectively

-- Conditions given in the problem
def goods_train_speed := v_g 
def passenger_train_speed := 90 
def express_train_speed := 1.5 * 90

-- Passenger train catches up with the goods train after 4 hours
def passenger_goods_catchup := 90 * 4 = v_g * (t_g + 4) - v_g * t_g

-- Express train catches up with the passenger train after 3 hours
def express_passenger_catchup := 1.5 * 90 * 3 = 90 * (3 + 4)

-- Theorem to prove the speeds of each train
theorem train_speeds_proof (h1 : 90 * 4 = v_g * (t_g + 4) - v_g * t_g)
                           (h2 : 1.5 * 90 * 3 = 90 * (3 + 4)) :
    v_g = 90 ∧ v_p = 90 ∧ v_e = 135 :=
by {
  sorry
}

end train_speeds_proof_l256_256456


namespace quadratic_function_proof_l256_256543

theorem quadratic_function_proof (a c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^2 - 4 * x + c)
  (h_sol_set : ∀ x, f x < 0 → (-1 < x ∧ x < 5)) :
  (a = 1 ∧ c = -5) ∧ (∀ x, 0 ≤ x ∧ x ≤ 3 → -9 ≤ f x ∧ f x ≤ -5) :=
by
  sorry

end quadratic_function_proof_l256_256543


namespace Person3IsTriussian_l256_256423

def IsTriussian (person : ℕ) : Prop := if person = 3 then True else False

def Person1Statement : Prop := ∀ i j k : ℕ, i = 1 → j = 2 → k = 3 → (IsTriussian i = (IsTriussian j ∧ IsTriussian k) ∨ (¬IsTriussian j ∧ ¬IsTriussian k))

def Person2Statement : Prop := ∀ i j : ℕ, i = 2 → j = 3 → (IsTriussian j = False)

def Person3Statement : Prop := ∀ i j : ℕ, i = 3 → j = 1 → (IsTriussian j = False)

theorem Person3IsTriussian : (Person1Statement ∧ Person2Statement ∧ Person3Statement) → IsTriussian 3 :=
by 
  sorry

end Person3IsTriussian_l256_256423


namespace max_photo_area_correct_l256_256075

def frame_area : ℝ := 59.6
def num_photos : ℕ := 4
def max_photo_area : ℝ := 14.9

theorem max_photo_area_correct : frame_area / num_photos = max_photo_area :=
by sorry

end max_photo_area_correct_l256_256075


namespace sum_of_solutions_eq_neg2_l256_256529

noncomputable def sum_of_real_solutions (a : ℝ) (h : a > 2) : ℝ :=
  -2

theorem sum_of_solutions_eq_neg2 (a : ℝ) (h : a > 2) :
  sum_of_real_solutions a h = -2 := sorry

end sum_of_solutions_eq_neg2_l256_256529


namespace arccos_one_half_is_pi_div_three_l256_256896

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l256_256896


namespace payment_of_employee_B_l256_256793

-- Define the variables and conditions
variables (A B : ℝ) (total_payment : ℝ) (payment_ratio : ℝ)

-- Assume the given conditions
def conditions : Prop := 
  (A + B = total_payment) ∧ 
  (A = payment_ratio * B) ∧ 
  (total_payment = 550) ∧ 
  (payment_ratio = 1.5)

-- Prove the payment of employee B is 220 given the conditions
theorem payment_of_employee_B : conditions A B total_payment payment_ratio → B = 220 := 
by
  sorry

end payment_of_employee_B_l256_256793


namespace arccos_half_eq_pi_div_three_l256_256847

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256847


namespace union_M_N_eq_N_l256_256418

def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

theorem union_M_N_eq_N : M ∪ N = N := by
  sorry

end union_M_N_eq_N_l256_256418


namespace van_helsing_earnings_l256_256782

theorem van_helsing_earnings (V W : ℕ) 
  (h1 : W = 4 * V) 
  (h2 : W = 8) :
  let E_v := 5 * (V / 2)
  let E_w := 10 * 8
  let E_total := E_v + E_w
  E_total = 85 :=
by
  sorry

end van_helsing_earnings_l256_256782


namespace cost_of_large_fries_l256_256130

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end cost_of_large_fries_l256_256130


namespace arccos_half_eq_pi_div_three_l256_256849

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256849


namespace number_of_sides_of_polygon_l256_256014

theorem number_of_sides_of_polygon (n : ℕ) (h1 : (n * (n - 3)) = 340) : n = 20 :=
by
  sorry

end number_of_sides_of_polygon_l256_256014


namespace largest_even_integer_product_l256_256527

theorem largest_even_integer_product (n : ℕ) (h : 2 * n * (2 * n + 2) * (2 * n + 4) * (2 * n + 6) = 5040) :
  2 * n + 6 = 20 :=
by
  sorry

end largest_even_integer_product_l256_256527


namespace f_when_x_lt_4_l256_256213

noncomputable def f : ℝ → ℝ := sorry

theorem f_when_x_lt_4 (x : ℝ) (h1 : ∀ y : ℝ, y > 4 → f y = 2^(y-1)) (h2 : ∀ y : ℝ, f (4-y) = f (4+y)) (hx : x < 4) : f x = 2^(7-x) :=
by
  sorry

end f_when_x_lt_4_l256_256213


namespace arccos_of_half_eq_pi_over_three_l256_256836

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l256_256836


namespace initial_dogs_count_is_36_l256_256278

-- Conditions
def initial_cats := 29
def adopted_dogs := 20
def additional_cats := 12
def total_pets := 57

-- Calculate total cats
def total_cats := initial_cats + additional_cats

-- Calculate initial dogs
def initial_dogs (initial_dogs : ℕ) : Prop :=
(initial_dogs - adopted_dogs) + total_cats = total_pets

-- Prove that initial dogs (D) is 36
theorem initial_dogs_count_is_36 : initial_dogs 36 :=
by
-- Here should contain the proof which is omitted
sorry

end initial_dogs_count_is_36_l256_256278


namespace girls_in_class_l256_256562

theorem girls_in_class (g b : ℕ) (h1 : g + b = 28) (h2 : g * 4 = b * 3) : g = 12 := by
  sorry

end girls_in_class_l256_256562


namespace monotonicity_tangent_points_l256_256365

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity (a : ℝ) : 
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧ 
  (a < (1 : ℝ) / 3 → (∀ x y : ℝ, x ≤ y → (x < (1 - real.sqrt (1 - 3 * a)) / 3 → f x a ≤ f y a) ∧ 
                               ((1 + real.sqrt (1 - 3 * a)) / 3 < x → f x a ≤ f y a) ∧ 
                               ((1 - real.sqrt (1 - 3 * a)) / 3 < x → x < (1 + real.sqrt (1 - 3 * a)) / 3 → f x a ≥ f y a))) := 
sorry

theorem tangent_points (a : ℝ) : 
  ({ x // 2 * x^3 - x^2 - 1 = 0 } = {1} ∧ (f 1 a = a + 1) ∨ { x // 2 * x^3 - x^2 - 1 = 0 } = { -1 } ∧ (f (-1) a = -a - 1)) := 
sorry

end monotonicity_tangent_points_l256_256365


namespace age_proof_l256_256593

-- Let's define the conditions first
variable (s f : ℕ) -- s: age of the son, f: age of the father

-- Conditions derived from the problem statement
def son_age_condition : Prop := s = 8 - 1
def father_age_condition : Prop := f = 5 * s

-- The goal is to prove that the father's age is 35
theorem age_proof (s f : ℕ) (h₁ : son_age_condition s) (h₂ : father_age_condition s f) : f = 35 :=
by sorry

end age_proof_l256_256593


namespace arccos_half_eq_pi_div_three_l256_256900

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l256_256900


namespace monotonicity_of_f_tangent_line_intersection_points_l256_256351

section
  variable {a : ℝ}
  def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

  theorem monotonicity_of_f (a : ℝ) :
    (a ≥ 1/3 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≤ f(x2)) ∧
    (a < 1/3 → ∀ x1 x2 : ℝ, 
    (x1 < x2 ∧ x2 ≤ 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) > f(x2)) ∧
    (x2 > 1 / 3 + (1 / 3) * sqrt(1 - 3 * a) → f(x1) < f(x2))) :=
  sorry

  theorem tangent_line_intersection_points (a : ℝ) :
    ∃ x : ℝ, (y = (a+1)*x) ∧ ((x = 1 ∧ y = a+1) ∨  (x = -1 ∧ y = -a-1)) :=
  sorry
end

end monotonicity_of_f_tangent_line_intersection_points_l256_256351


namespace molar_weight_of_BaF2_l256_256664

theorem molar_weight_of_BaF2 (Ba_weight : Real) (F_weight : Real) (num_moles : ℕ) 
    (Ba_weight_val : Ba_weight = 137.33) (F_weight_val : F_weight = 18.998) 
    (num_moles_val : num_moles = 6) 
    : (137.33 + 2 * 18.998) * 6 = 1051.956 := 
by
  sorry

end molar_weight_of_BaF2_l256_256664


namespace number_of_people_l256_256313

theorem number_of_people (clinks : ℕ) (h : clinks = 45) : ∃ x : ℕ, x * (x - 1) / 2 = clinks ∧ x = 10 :=
by
  sorry

end number_of_people_l256_256313


namespace problem_solution_l256_256690

theorem problem_solution :
  50000 - ((37500 / 62.35) ^ 2 + Real.sqrt 324) = -311752.222 :=
by
  sorry

end problem_solution_l256_256690


namespace adam_paper_tearing_l256_256501

theorem adam_paper_tearing (n : ℕ) :
  let starts_with_one_piece : ℕ := 1
  let increment_to_four : ℕ := 3
  let increment_to_ten : ℕ := 9
  let target_pieces : ℕ := 20000
  let start_modulo : ℤ := 1

  -- Modulo 3 analysis
  starts_with_one_piece % 3 = start_modulo ∧
  increment_to_four % 3 = 0 ∧ 
  increment_to_ten % 3 = 0 ∧ 
  target_pieces % 3 = 2 → 
  n % 3 = start_modulo ∧ ∀ m, m % 3 = 0 → n + m ≠ target_pieces :=
sorry

end adam_paper_tearing_l256_256501


namespace smallest_even_sum_is_102_l256_256772

theorem smallest_even_sum_is_102 (s : Int) (h₁ : ∃ a b c d e f g : Int, s = a + b + c + d + e + f + g)
    (h₂ : ∀ a b c d e f g : Int, b = a + 2 ∧ c = a + 4 ∧ d = a + 6 ∧ e = a + 8 ∧ f = a + 10 ∧ g = a + 12)
    (h₃ : s = 756) : ∃ a : Int, a = 102 :=
  by
    sorry

end smallest_even_sum_is_102_l256_256772


namespace sandy_fingernails_length_l256_256113

/-- 
Sandy, who just turned 12 this month, has a goal for tying the world record for longest fingernails, 
which is 26 inches. Her fingernails grow at a rate of one-tenth of an inch per month. 
She will be 32 when she achieves the world record. 
Prove that her fingernails are currently 2 inches long.
-/
theorem sandy_fingernails_length 
  (current_age : ℕ) (world_record_length : ℝ) (growth_rate : ℝ) (years_to_achieve : ℕ) : 
  current_age = 12 → 
  world_record_length = 26 → 
  growth_rate = 0.1 → 
  years_to_achieve = 20 →
  (world_record_length - growth_rate * 12 * years_to_achieve) = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_fingernails_length_l256_256113


namespace max_value_dn_l256_256185

def a (n : ℕ) : ℕ := 100 + 2 * n * n
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_value_dn : ∀ n : ℕ, d n ≤ 2 :=
by
  -- Definitions used in proof
  intro n
  let an := a n
  let an1 := a (n + 1)
  have h : d n = Nat.gcd an an1, from rfl
  -- Simplifications and Euclidean steps go here (not shown in Lean statement)
  -- Maximum value assertion
  sorry

end max_value_dn_l256_256185


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256877

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256877


namespace example_table_tennis_probability_l256_256651

def table_tennis_game (A_serves_first : Bool) (score_probability_A : ℝ) 
  (independent_serves : ∀ n m : ℕ, A n ≠ A m → Independent (events! n) (events! m))
  (serve_points_A : Fin 4 → Fin 5)
  (serve_points_B : Fin 4 → Fin 5) : Prop :=
P(A₀) = 0.16 ∧
P(A₁) = 0.48 ∧
P(B₀) = 0.36 ∧
P(B₁) = 0.48 ∧
P(B₂) = 0.16 ∧
P(A₂) = 0.36 ∧
P(B) = 0.352 ∧
P(C) = 0.3072

theorem example_table_tennis_probability :
  ∀ (prob_A : ℝ) (prob_B : ℝ) (score_A : ℕ) (score_B : ℕ),
  table_tennis_game true 0.6
  (λ n m h, by sorry)
  (λ x, by sorry)
  (λ x, by sorry) :=
by sorry

end example_table_tennis_probability_l256_256651


namespace purple_cars_count_l256_256980

theorem purple_cars_count
    (P R G : ℕ)
    (h1 : R = P + 6)
    (h2 : G = 4 * R)
    (h3 : P + R + G = 312) :
    P = 47 :=
by 
  sorry

end purple_cars_count_l256_256980


namespace arccos_one_half_l256_256928

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l256_256928


namespace curve_is_parabola_l256_256944

-- Define the condition: the curve is defined by the given polar equation
def polar_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- The main theorem statement: Prove that the curve defined by the equation is a parabola
theorem curve_is_parabola (r θ : ℝ) (h : polar_eq r θ) : ∃ x y : ℝ, x = 1 + 2 * y :=
sorry

end curve_is_parabola_l256_256944


namespace division_result_l256_256181

-- Define the arithmetic expression
def arithmetic_expression : ℕ := (20 + 15 * 3) - 10

-- Define the main problem
def problem : Prop := 250 / arithmetic_expression = 250 / 55

-- The theorem statement that needs to be proved
theorem division_result : problem := by
    sorry

end division_result_l256_256181


namespace eq_970299_l256_256467

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l256_256467


namespace min_colors_shapes_l256_256678

def representable_centers (C S : Nat) : Nat :=
  C + (C * (C - 1)) / 2 + S + S * (S - 1)

theorem min_colors_shapes (C S : Nat) :
  ∀ (C S : Nat), (C + (C * (C - 1)) / 2 + S + S * (S - 1)) ≥ 12 → (C, S) = (3, 3) :=
sorry

end min_colors_shapes_l256_256678


namespace distance_between_A_and_B_l256_256290

theorem distance_between_A_and_B 
  (d : ℕ) -- The distance we want to prove
  (ha : ∀ (t : ℕ), d = 700 * t)
  (hb : ∀ (t : ℕ), d + 400 = 2100 * t) :
  d = 1700 := 
by
  sorry

end distance_between_A_and_B_l256_256290


namespace monotonicity_of_f_intersection_points_of_tangent_l256_256384

section
variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f' x ≤ f' y) ↔ (a ≥ 1 / 3) :=
sorry

theorem intersection_points_of_tangent :
  ∃ (x₁ x₂ : ℝ), (f x₁ = (a + 1) * x₁) ∧ (f x₂ = - (a + 1) * x₂) ∧
  x₁ = 1 ∧ x₂ = -1 :=
sorry

end

end monotonicity_of_f_intersection_points_of_tangent_l256_256384


namespace total_money_raised_l256_256300

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l256_256300


namespace max_sum_arithmetic_sequence_l256_256203

theorem max_sum_arithmetic_sequence (n : ℕ) (M : ℝ) (hM : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ k, a (k + 1) - a k = a 1 - a 0) 
  (h_constraint : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S, S = (n + 1) * (Real.sqrt (10 * M)) / 2 :=
sorry

end max_sum_arithmetic_sequence_l256_256203


namespace min_value_of_fraction_l256_256391

theorem min_value_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (4 / (a + 2) + 1 / (b + 1)) = 9 / 4 :=
sorry

end min_value_of_fraction_l256_256391


namespace octagon_area_difference_l256_256029

theorem octagon_area_difference (side_length : ℝ) (h : side_length = 1) : 
  let A := 2 * (1 + Real.sqrt 2)
  let triangle_area := (1 / 2) * (1 / 2) * (1 / 2)
  let gray_area := 4 * triangle_area
  let part_with_lines := A - gray_area
  (gray_area - part_with_lines) = 1 / 4 :=
by
  sorry

end octagon_area_difference_l256_256029


namespace f_is_decreasing_max_k_value_l256_256350

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_is_decreasing : ∀ x > 0, (∃ y > x, f y < f x) :=
by
  sorry

theorem max_k_value : ∃ k : ℕ, (∀ x > 0, f x > k / (x + 1)) ∧ k = 3 :=
by
  sorry

end f_is_decreasing_max_k_value_l256_256350


namespace arithmetic_sequence_value_y_l256_256138

theorem arithmetic_sequence_value_y :
  ∀ (a₁ a₃ y : ℤ), 
  a₁ = 3 ^ 3 →
  a₃ = 5 ^ 3 →
  y = (a₁ + a₃) / 2 →
  y = 76 :=
by 
  intros a₁ a₃ y h₁ h₃ hy 
  sorry

end arithmetic_sequence_value_y_l256_256138


namespace probability_neither_perfect_square_nor_cube_l256_256628

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l256_256628


namespace find_common_ratio_l256_256718

-- We need to state that q is the common ratio of the geometric sequence

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first three terms for the geometric sequence
def S_3 (a : ℕ → ℝ) := a 0 + a 1 + a 2

-- State the Lean 4 declaration of the proof problem
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : (S_3 a) / (a 2) = 3) :
  q = 1 := 
sorry

end find_common_ratio_l256_256718


namespace monotonicity_and_tangent_intersections_l256_256382

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

theorem monotonicity_and_tangent_intersections (a : ℝ) :
  (a ≥ 1/3 → ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∧
  (a < 1/3 → 
  (∀ x : ℝ, x < (1 - sqrt(1 - 3 * a))/3 → f x a < f ((1 - sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, x > (1 + sqrt(1 - 3 * a))/3 → f x a > f ((1 + sqrt(1 - 3 * a))/3) a) ∧
  (∀ x : ℝ, (1 - sqrt(1 - 3 * a))/3 < x ∧ x < (1 + sqrt(1 - 3 * a))/3 → 
  f ((1 - sqrt(1 - 3 * a))/3) a > f x a ∧ f x a > f ((1 + sqrt(1 - 3 * a))/3) a)) ∧
  (f 1 a = a + 1 ∧ f (-1) a = -a - 1) := 
by sorry

end monotonicity_and_tangent_intersections_l256_256382


namespace mass_percentage_of_H_in_ascorbic_acid_l256_256704

-- Definitions based on the problem conditions
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

def ascorbic_acid_molecular_formula_C : ℝ := 6
def ascorbic_acid_molecular_formula_H : ℝ := 8
def ascorbic_acid_molecular_formula_O : ℝ := 6

noncomputable def ascorbic_acid_molar_mass : ℝ :=
  ascorbic_acid_molecular_formula_C * molar_mass_C + 
  ascorbic_acid_molecular_formula_H * molar_mass_H + 
  ascorbic_acid_molecular_formula_O * molar_mass_O

noncomputable def hydrogen_mass_in_ascorbic_acid : ℝ :=
  ascorbic_acid_molecular_formula_H * molar_mass_H

noncomputable def hydrogen_mass_percentage_in_ascorbic_acid : ℝ :=
  (hydrogen_mass_in_ascorbic_acid / ascorbic_acid_molar_mass) * 100

theorem mass_percentage_of_H_in_ascorbic_acid :
  hydrogen_mass_percentage_in_ascorbic_acid = 4.588 :=
by
  sorry

end mass_percentage_of_H_in_ascorbic_acid_l256_256704


namespace packing_big_boxes_l256_256393

def total_items := 8640
def items_per_small_box := 12
def small_boxes_per_big_box := 6

def num_big_boxes (total_items items_per_small_box small_boxes_per_big_box : ℕ) : ℕ :=
  (total_items / items_per_small_box) / small_boxes_per_big_box

theorem packing_big_boxes : num_big_boxes total_items items_per_small_box small_boxes_per_big_box = 120 :=
by
  sorry

end packing_big_boxes_l256_256393


namespace john_weekly_earnings_before_raise_l256_256105

theorem john_weekly_earnings_before_raise :
  ∀(x : ℝ), (70 = 1.0769 * x) → x = 64.99 :=
by
  intros x h
  sorry

end john_weekly_earnings_before_raise_l256_256105


namespace zongzi_packing_l256_256756

theorem zongzi_packing (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (8 * x + 10 * y = 200) ↔ (x, y) = (5, 16) ∨ (x, y) = (10, 12) ∨ (x, y) = (15, 8) ∨ (x, y) = (20, 4) := 
sorry

end zongzi_packing_l256_256756


namespace eq_970299_l256_256468

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l256_256468


namespace max_books_l256_256936

theorem max_books (cost_per_book : ℝ) (total_money : ℝ) (h_cost : cost_per_book = 8.75) (h_money : total_money = 250.0) :
  ∃ n : ℕ, n = 28 ∧ cost_per_book * n ≤ total_money ∧ ∀ m : ℕ, cost_per_book * m ≤ total_money → m ≤ 28 :=
by
  sorry

end max_books_l256_256936


namespace arccos_half_eq_pi_div_three_l256_256862

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256862


namespace max_y_difference_l256_256945

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference : 
  ∃ x1 x2 : ℝ, 
    f x1 = g x1 ∧ f x2 = g x2 ∧ 
    (∀ x : ℝ, f x = g x → x = x1 ∨ x = x2) ∧ 
    abs ((f x1) - (f x2)) = 2 := 
by
  sorry

end max_y_difference_l256_256945


namespace total_house_rent_l256_256759

theorem total_house_rent (P S R : ℕ)
  (h1 : S = 5 * P)
  (h2 : R = 3 * P)
  (h3 : R = 1800) : 
  S + P + R = 5400 :=
by
  sorry

end total_house_rent_l256_256759


namespace monotonicity_f_tangent_points_l256_256374

def f (a x : ℝ) := x^3 - x^2 + a * x + 1
def f_prime (a x : ℝ) := 3 * x^2 - 2 * x + a

theorem monotonicity_f (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x, 0 ≤ f_prime a x) ∧
  (a < 1 / 3 → ∃ x1 x2, x1 = (1 - real.sqrt (1 - 3 * a)) / 3 ∧
                        x2 = (1 + real.sqrt (1 - 3 * a)) / 3 ∧
                        ∀ x, (x < x1 ∨ x > x2 → 0 ≤ f_prime a x) ∧
                             (x1 < x ∧ x < x2 → 0 > f_prime a x)) :=
sorry

theorem tangent_points (a : ℝ) :
  ∀ (x0 : ℝ), f_prime a x0 * x0 = 2 * x0^3 - x0^2 - 1 →
              (f a x0 = f_prime a x0 * x0 + (1 - x0^2)) →
              (x0 = 1 ∧ f a 1 = a + 1) ∨
              (x0 = -1 ∧ f a (-1) = -a - 1) :=
sorry

end monotonicity_f_tangent_points_l256_256374


namespace probability_non_square_non_cube_l256_256635

theorem probability_non_square_non_cube :
  let numbers := finset.Icc 1 200
  let perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers
  let perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers
  let perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers
  let total := finset.card numbers
  let square_count := finset.card perfect_squares
  let cube_count := finset.card perfect_cubes
  let sixth_count := finset.card perfect_sixths
  let non_square_non_cube_count := total - (square_count + cube_count - sixth_count)
  (non_square_non_cube_count : ℚ) / total = 183 / 200 := by
{
  numbers := finset.Icc 1 200,
  perfect_squares := finset.filter (λ n, ∃ m, m * m = n) numbers,
  perfect_cubes := finset.filter (λ n, ∃ m, m * m * m = n) numbers,
  perfect_sixths := finset.filter (λ n, ∃ m, m * m * m * m * m * m = n) numbers,
  total := finset.card numbers,
  square_count := finset.card perfect_squares,
  cube_count := finset.card perfect_cubes,
  sixth_count := finset.card perfect_sixths,
  non_square_non_cube_count := total - (square_count + cube_count - sixth_count),
  (non_square_non_cube_count : ℚ) / total := 183 / 200
}

end probability_non_square_non_cube_l256_256635


namespace positive_root_condition_negative_root_condition_zero_root_condition_l256_256943

-- Positive root condition
theorem positive_root_condition {a b : ℝ} (h : a * b < 0) : ∃ x : ℝ, a * x + b = 0 ∧ x > 0 :=
by
  sorry

-- Negative root condition
theorem negative_root_condition {a b : ℝ} (h : a * b > 0) : ∃ x : ℝ, a * x + b = 0 ∧ x < 0 :=
by
  sorry

-- Root equal to zero condition
theorem zero_root_condition {a b : ℝ} (h₁ : b = 0) (h₂ : a ≠ 0) : ∃ x : ℝ, a * x + b = 0 ∧ x = 0 :=
by
  sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l256_256943


namespace length_sum_l256_256001

theorem length_sum : 
  let m := 1 -- Meter as base unit
  let cm := 0.01 -- 1 cm in meters
  let mm := 0.001 -- 1 mm in meters
  2 * m + 3 * cm + 5 * mm = 2.035 * m :=
by sorry

end length_sum_l256_256001


namespace gigi_has_15_jellybeans_l256_256430

variable (G : ℕ) -- G is the number of jellybeans Gigi has
variable (R : ℕ) -- R is the number of jellybeans Rory has
variable (L : ℕ) -- L is the number of jellybeans Lorelai has eaten

-- Conditions
def condition1 := R = G + 30
def condition2 := L = 3 * (G + R)
def condition3 := L = 180

-- Proof statement
theorem gigi_has_15_jellybeans (G R L : ℕ) (h1 : condition1 G R) (h2 : condition2 G R L) (h3 : condition3 L) : G = 15 := by
  sorry

end gigi_has_15_jellybeans_l256_256430


namespace sequence_non_existence_l256_256347

variable (α β : ℝ)
variable (r : ℝ)

theorem sequence_non_existence 
  (hαβ : α * β > 0) :  
  (∃ (x : ℕ → ℝ), x 0 = r ∧ ∀ n, x (n + 1) = (x n + α) / (β * (x n) + 1) → false) ↔ 
  r = - (1 / β) :=
sorry

end sequence_non_existence_l256_256347


namespace min_ab_minus_cd_l256_256544

theorem min_ab_minus_cd (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9) (h5 : a^2 + b^2 + c^2 + d^2 = 21) : ab - cd ≥ 2 := sorry

end min_ab_minus_cd_l256_256544


namespace prob_sum_divisible_by_4_is_1_4_l256_256046

/-- 
  Given two wheels each with numbers from 1 to 8, 
  the probability that the sum of two selected numbers from the wheels is divisible by 4.
-/
noncomputable def prob_sum_divisible_by_4 : ℚ :=
  let outcomes : ℕ := 8 * 8
  let favorable_outcomes : ℕ := 16
  favorable_outcomes / outcomes

theorem prob_sum_divisible_by_4_is_1_4 : prob_sum_divisible_by_4 = 1 / 4 := 
  by
    -- Statement is left as sorry as the proof steps are not required.
    sorry

end prob_sum_divisible_by_4_is_1_4_l256_256046


namespace twin_primes_divisible_by_12_l256_256428

def isTwinPrime (p q : ℕ) : Prop :=
  p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p = 2

theorem twin_primes_divisible_by_12 {p q r s : ℕ} 
  (h1 : isTwinPrime p q) 
  (h2 : p > 3) 
  (h3 : isTwinPrime r s) 
  (h4 : r > 3) :
  12 ∣ (p * r - q * s) := by
  sorry

end twin_primes_divisible_by_12_l256_256428


namespace largest_unshaded_area_l256_256815

theorem largest_unshaded_area (s : ℝ) (π_approx : ℝ) :
    (let r := s / 2
     let area_square := s^2
     let area_circle := π_approx * r^2
     let area_triangle := (1 / 2) * (s / 2) * (s / 2)
     let unshaded_square := area_square - area_circle
     let unshaded_circle := area_circle - area_triangle
     unshaded_circle) > (unshaded_square) := by
        sorry

end largest_unshaded_area_l256_256815


namespace fraction_neither_cable_nor_vcr_l256_256151

variable (T : ℕ)
variable (units_with_cable : ℕ := T / 5)
variable (units_with_vcrs : ℕ := T / 10)
variable (units_with_cable_and_vcrs : ℕ := (T / 5) / 3)

theorem fraction_neither_cable_nor_vcr (T : ℕ)
  (h1 : units_with_cable = T / 5)
  (h2 : units_with_vcrs = T / 10)
  (h3 : units_with_cable_and_vcrs = (units_with_cable / 3)) :
  (T - (units_with_cable + (units_with_vcrs - units_with_cable_and_vcrs))) / T = 7 / 10 := 
by
  sorry

end fraction_neither_cable_nor_vcr_l256_256151


namespace find_divisor_l256_256009

open Nat

theorem find_divisor 
  (d n : ℕ)
  (h1 : n % d = 3)
  (h2 : 2 * n % d = 2) : 
  d = 4 := 
sorry

end find_divisor_l256_256009


namespace simplify_expression_l256_256432

variable (a b : ℤ)

theorem simplify_expression : 
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := 
by sorry

end simplify_expression_l256_256432


namespace minimize_quadratic_l256_256948

theorem minimize_quadratic (c : ℝ) : ∃ b : ℝ, (∀ x : ℝ, 3 * x^2 + 2 * x + c ≥ 3 * b^2 + 2 * b + c) ∧ b = -1/3 :=
by
  sorry

end minimize_quadratic_l256_256948


namespace part1_part2_l256_256069

def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = (x - 1)^2 + 1}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

theorem part1 : A ∩ B = {x : ℝ | x > 2} := 
by sorry

theorem part2 (a : ℝ) (h : (C a \ A) ⊆ C a) : 2 ≤ a :=
by sorry

end part1_part2_l256_256069


namespace minimum_k_exists_l256_256795

theorem minimum_k_exists :
  ∀ (s : Finset ℝ), s.card = 3 → (∀ (a b : ℝ), a ∈ s → b ∈ s → (|a - b| ≤ (1.5 : ℝ) ∨ |(1 / a) - (1 / b)| ≤ 1.5)) :=
by
  sorry

end minimum_k_exists_l256_256795


namespace arccos_half_eq_pi_div_three_l256_256864

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256864


namespace fruit_box_assignment_l256_256026

variable (B1 B2 B3 B4 : Nat)

theorem fruit_box_assignment :
  (¬(B1 = 1) ∧ ¬(B2 = 2) ∧ ¬(B3 = 4 ∧ B2 ∨ B3 = 3 ∧ B2) ∧ ¬(B4 = 4)) →
  B1 = 2 ∧ B2 = 4 ∧ B3 = 3 ∧ B4 = 1 :=
by
  sorry

end fruit_box_assignment_l256_256026


namespace midpoint_sum_l256_256283

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -4) (hy2 : y2 = -7) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 1 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num

end midpoint_sum_l256_256283


namespace boat_travel_time_l256_256674

theorem boat_travel_time (x : ℝ) (T : ℝ) (h0 : 0 ≤ x) (h1 : x ≠ 15.6) 
    (h2 : 96 = (15.6 - x) * T) 
    (h3 : 96 = (15.6 + x) * 5) : 
    T = 8 :=
by 
  sorry

end boat_travel_time_l256_256674


namespace ratio_of_height_to_radius_max_volume_l256_256449

theorem ratio_of_height_to_radius_max_volume (r h : ℝ) (h_surface_area : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi) :
  (exists (max_r : ℝ) (max_h : ℝ), 2 * r * max_r + 2 * r * max_h = 6 * Real.pi ∧ 
                                  max_r = 1 ∧ 
                                  max_h = 2 ∧ 
                                  (max_h / max_r) = 2) :=
by
  sorry

end ratio_of_height_to_radius_max_volume_l256_256449


namespace locus_of_midpoint_of_chord_l256_256951

theorem locus_of_midpoint_of_chord
  (x y : ℝ)
  (hx : (x - 1)^2 + y^2 ≠ 0)
  : (x - 1) * (x - 1) + y * y = 1 :=
by
  sorry

end locus_of_midpoint_of_chord_l256_256951


namespace area_of_EFGH_l256_256655

-- Define the dimensions of the smaller rectangles
def smaller_rectangle_short_side : ℕ := 7
def smaller_rectangle_long_side : ℕ := 2 * smaller_rectangle_short_side

-- Define the configuration of rectangles
def width_EFGH : ℕ := 2 * smaller_rectangle_short_side
def length_EFGH : ℕ := smaller_rectangle_long_side

-- Prove that the area of rectangle EFGH is 196 square feet
theorem area_of_EFGH : width_EFGH * length_EFGH = 196 := by
  sorry

end area_of_EFGH_l256_256655


namespace fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l256_256461

def positive_integers_up_to (n : ℕ) : List ℕ :=
  List.range' 1 n

def divisible_by_lcm (lcm : ℕ) (lst : List ℕ) : List ℕ :=
  lst.filter (λ x => x % lcm = 0)

noncomputable def fraction_divisible_by_both (n a b : ℕ) : ℚ :=
  let lcm_ab := Nat.lcm a b
  let elems := positive_integers_up_to n
  let divisible_elems := divisible_by_lcm lcm_ab elems
  divisible_elems.length / n

theorem fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25 :
  fraction_divisible_by_both 100 3 4 = (2 : ℚ) / 25 :=
by
  sorry

end fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l256_256461


namespace value_of_x2_plus_4y2_l256_256964

theorem value_of_x2_plus_4y2 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : x * y = -12) : x^2 + 4*y^2 = 84 := 
  sorry

end value_of_x2_plus_4y2_l256_256964


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256879

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256879


namespace winner_percentage_l256_256976

theorem winner_percentage (W L V : ℕ) 
    (hW : W = 868) 
    (hDiff : W - L = 336)
    (hV : V = W + L) : 
    (W * 100 / V) = 62 := 
by 
    sorry

end winner_percentage_l256_256976


namespace eq1_solution_eq2_solution_l256_256259

theorem eq1_solution (x : ℝ) : (x - 1)^2 - 1 = 15 ↔ x = 5 ∨ x = -3 := by sorry

theorem eq2_solution (x : ℝ) : (1 / 3) * (x + 3)^3 - 9 = 0 ↔ x = 0 := by sorry

end eq1_solution_eq2_solution_l256_256259


namespace find_w_l256_256320

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![4, 1]]

def w : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![1 / 243], ![-1 / 324]]

def I2 : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem find_w : 
  ((B ^ 6 + B ^ 4 + B ^ 2 + I2) ⬝ w = ![![1], ![16]]) :=
begin
  sorry
end

end find_w_l256_256320


namespace total_rainfall_l256_256579

-- Given conditions
def sunday_rainfall : ℕ := 4
def monday_rainfall : ℕ := sunday_rainfall + 3
def tuesday_rainfall : ℕ := 2 * monday_rainfall

-- Question: Total rainfall over the 3 days
theorem total_rainfall : sunday_rainfall + monday_rainfall + tuesday_rainfall = 25 := by
  sorry

end total_rainfall_l256_256579


namespace sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l256_256650

variable (p q : ℝ) (x1 x2 : ℝ)

-- Define the condition: Roots of the quadratic equation
def quadratic_equation_condition : Prop :=
  x1^2 + p * x1 + q = 0 ∧ x2^2 + p * x2 + q = 0

-- Define the identities for calculations based on properties of roots
def properties_of_roots : Prop :=
  x1 + x2 = -p ∧ x1 * x2 = q

-- First proof problem
theorem sum_of_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                           (h2 : properties_of_roots p q x1 x2) :
  1 / x1 + 1 / x2 = -p / q := 
by sorry

-- Second proof problem
theorem sum_of_square_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                  (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^2) + 1 / (x2^2) = (p^2 - 2*q) / (q^2) := 
by sorry

-- Third proof problem
theorem sum_of_cubic_reciprocals (h1 : quadratic_equation_condition p q x1 x2) 
                                 (h2 : properties_of_roots p q x1 x2) :
  1 / (x1^3) + 1 / (x2^3) = p * (3*q - p^2) / (q^3) := 
by sorry

end sum_of_reciprocals_sum_of_square_reciprocals_sum_of_cubic_reciprocals_l256_256650


namespace sin_A_eq_one_half_l256_256558

theorem sin_A_eq_one_half (a b : ℝ) (sin_B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_B = 2/3) : 
  ∃ (sin_A : ℝ), sin_A = 1/2 := 
by
  let sin_A := a * sin_B / b
  existsi sin_A
  sorry

end sin_A_eq_one_half_l256_256558


namespace polygon_sides_l256_256226

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l256_256226


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256874

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256874


namespace value_of_polynomial_l256_256470

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l256_256470


namespace crayons_selection_l256_256654

theorem crayons_selection : 
  ∃ (n : ℕ), n = Nat.choose 14 4 ∧ n = 1001 := by
  sorry

end crayons_selection_l256_256654


namespace sum_of_possible_values_l256_256344

variable {S : ℝ} (h : S ≠ 0)

theorem sum_of_possible_values (h : S ≠ 0) : ∃ N : ℝ, N ≠ 0 ∧ 6 * N + 2 / N = S → ∀ N1 N2 : ℝ, (6 * N1 + 2 / N1 = S ∧ 6 * N2 + 2 / N2 = S) → (N1 + N2) = S / 6 :=
by
  sorry

end sum_of_possible_values_l256_256344


namespace quadratic_real_roots_m_range_l256_256732

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l256_256732


namespace find_original_number_l256_256680

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_original_number_l256_256680


namespace expand_polynomials_l256_256939

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l256_256939


namespace washing_machine_capacity_l256_256010

-- Define the problem conditions
def families : Nat := 3
def people_per_family : Nat := 4
def days : Nat := 7
def towels_per_person_per_day : Nat := 1
def loads : Nat := 6

-- Define the statement to prove
theorem washing_machine_capacity :
  (families * people_per_family * days * towels_per_person_per_day) / loads = 14 := by
  sorry

end washing_machine_capacity_l256_256010


namespace arccos_half_eq_pi_div_three_l256_256858

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l256_256858


namespace pool_surface_area_l256_256813

/-
  Given conditions:
  1. The width of the pool is 3 meters.
  2. The length of the pool is 10 meters.

  To prove:
  The surface area of the pool is 30 square meters.
-/
def width : ℕ := 3
def length : ℕ := 10
def surface_area (length width : ℕ) : ℕ := length * width

theorem pool_surface_area : surface_area length width = 30 := by
  unfold surface_area
  rfl

end pool_surface_area_l256_256813


namespace value_of_polynomial_l256_256469

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l256_256469


namespace cartesian_equation_of_line_range_of_m_l256_256092

variable (m t : ℝ)

def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos (2 * t), 2 * sin t)

def polar_line (ρ θ m : ℝ) : Prop :=
  ρ * sin (θ + π / 3) + m = 0

theorem cartesian_equation_of_line (m : ℝ) : 
  ∃ (x y : ℝ), √3 * x + y + 2 * m = 0 :=
by
  use (m, m)
  sorry

theorem range_of_m (m : ℝ) :
  ∃ t : ℝ, parametric_curve t = (sqrt 3 * cos (2 * t), 2 * sin t) →
  -19 / 12 ≤ m ∧ m ≤ 5 / 2 :=
by
  use m
  sorry

end cartesian_equation_of_line_range_of_m_l256_256092


namespace circle_area_from_diameter_points_l256_256599

theorem circle_area_from_diameter_points (C D : ℝ × ℝ)
    (hC : C = (-2, 3)) (hD : D = (4, -1)) :
    ∃ (A : ℝ), A = 13 * Real.pi :=
by
  let distance := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  have diameter : distance = Real.sqrt (6^2 + (-4)^2) := sorry -- this follows from the coordinates
  have radius : distance / 2 = Real.sqrt 13 := sorry -- half of the diameter
  exact ⟨13 * Real.pi, sorry⟩ -- area of the circle

end circle_area_from_diameter_points_l256_256599


namespace linear_inequality_solution_l256_256342

theorem linear_inequality_solution {x y m n : ℤ} 
  (h_table: (∀ x, if x = -2 then y = 3 
                else if x = -1 then y = 2 
                else if x = 0 then y = 1 
                else if x = 1 then y = 0 
                else if x = 2 then y = -1 
                else if x = 3 then y = -2 
                else true)) 
  (h_eq: m * x - n = y) : 
  x ≥ -1 :=
sorry

end linear_inequality_solution_l256_256342


namespace slope_of_line_l256_256330

-- Definition of the line equation
def lineEquation (x y : ℝ) : Prop := 4 * x - 7 * y = 14

-- The statement that we need to prove
theorem slope_of_line : ∀ x y, lineEquation x y → ∃ m, m = 4 / 7 :=
by {
  sorry
}

end slope_of_line_l256_256330


namespace bob_shuck_2_hours_l256_256825

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l256_256825


namespace vertex_farthest_from_origin_l256_256998

theorem vertex_farthest_from_origin (center : ℝ × ℝ) (area : ℝ) (top_side_horizontal : Prop) (dilation_center : ℝ × ℝ) (scale_factor : ℝ) :
  center = (10, -5) ∧ area = 16 ∧ top_side_horizontal ∧ dilation_center = (0, 0) ∧ scale_factor = 3 →
  ∃ (vertex_farthest : ℝ × ℝ), vertex_farthest = (36, -21) :=
by
  sorry

end vertex_farthest_from_origin_l256_256998


namespace number_of_extra_spacy_subsets_l256_256039

def is_extra_spacy (s : Finset ℕ) : Prop :=
  ∀ (a b c d : ℕ), a ∈ s → b ∈ s → c ∈ s → d ∈ s → ¬ (a + 1 = b ∧ b + 1 = c ∧ c + 1 = d)

def d : ℕ → ℕ
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 5
| n := if n ≥ 5 then d (n - 1) + d (n - 4) else 0

theorem number_of_extra_spacy_subsets :
  d 12 = 69 :=
by
  sorry

end number_of_extra_spacy_subsets_l256_256039


namespace jacob_age_proof_l256_256696

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l256_256696


namespace stella_dolls_count_l256_256763

variables (D : ℕ) (clocks glasses P_doll P_clock P_glass cost profit : ℕ)

theorem stella_dolls_count (h_clocks : clocks = 2)
                     (h_glasses : glasses = 5)
                     (h_P_doll : P_doll = 5)
                     (h_P_clock : P_clock = 15)
                     (h_P_glass : P_glass = 4)
                     (h_cost : cost = 40)
                     (h_profit : profit = 25) :
  D = 3 :=
by sorry

end stella_dolls_count_l256_256763


namespace value_of_x4_plus_1_div_x4_l256_256556

theorem value_of_x4_plus_1_div_x4 (x : ℝ) (hx : x^2 + 1 / x^2 = 2) : x^4 + 1 / x^4 = 2 := 
sorry

end value_of_x4_plus_1_div_x4_l256_256556


namespace range_of_f_x_lt_1_l256_256395

theorem range_of_f_x_lt_1 (x : ℝ) (f : ℝ → ℝ) (h : f x = x^3) : f x < 1 ↔ x < 1 := by
  sorry

end range_of_f_x_lt_1_l256_256395


namespace arccos_half_eq_pi_over_three_l256_256914

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l256_256914


namespace fruits_in_boxes_l256_256023

theorem fruits_in_boxes :
  ∃ (B1 B2 B3 B4 : string), 
    ¬((B1 = "Orange") ∧ (B2 = "Pear") ∧ (B3 = "Banana" → (B4 = "Apple" ∨ B4 = "Pear")) ∧ (B4 = "Apple")) ∧
    B1 = "Banana" ∧ B2 = "Apple" ∧ B3 = "Orange" ∧ B4 = "Pear" :=
by {
  sorry
}

end fruits_in_boxes_l256_256023


namespace find_x_l256_256339

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end find_x_l256_256339


namespace speed_ratio_l256_256658

variable (v1 v2 : ℝ) -- Speeds of A and B respectively
variable (dA dB : ℝ) -- Distances to destinations A and B respectively

-- Conditions:
-- 1. Both reach their destinations in 1 hour
def condition_1 : Prop := dA = v1 ∧ dB = v2

-- 2. When they swap destinations, A takes 35 minutes more to reach B's destination
def condition_2 : Prop := dB / v1 = dA / v2 + 35 / 60

-- Given these conditions, prove that the ratio of v1 to v2 is 3
theorem speed_ratio (h1 : condition_1 v1 v2 dA dB) (h2 : condition_2 v1 v2 dA dB) : v1 = 3 * v2 :=
sorry

end speed_ratio_l256_256658


namespace simplify_expression_l256_256826

theorem simplify_expression : 
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2)) - 
  (Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2 / 3))) = -Real.sqrt 2 :=
by
  sorry

end simplify_expression_l256_256826


namespace max_3cosx_4sinx_l256_256524

theorem max_3cosx_4sinx (x : ℝ) : (3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧ (∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5) :=
  sorry

end max_3cosx_4sinx_l256_256524


namespace solution_set_leq_2_l256_256341

theorem solution_set_leq_2 (x y m n : ℤ)
  (h1 : m * 0 - n = 1)
  (h2 : m * 1 - n = 0)
  (h3 : y = m * x - n) :
  x ≥ -1 ↔ m * x - n ≤ 2 :=
by {
  sorry
}

end solution_set_leq_2_l256_256341


namespace total_money_raised_l256_256303

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l256_256303


namespace complement_B_in_U_l256_256420

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x = 1}
def U : Set ℕ := A ∪ B

theorem complement_B_in_U : (U \ B) = {2, 3} := by
  sorry

end complement_B_in_U_l256_256420


namespace find_number_l256_256019

theorem find_number :
  ∃ n : ℤ,
    (n % 12 = 11) ∧ 
    (n % 11 = 10) ∧ 
    (n % 10 = 9) ∧ 
    (n % 9 = 8) ∧ 
    (n % 8 = 7) ∧ 
    (n % 7 = 6) ∧ 
    (n % 6 = 5) ∧ 
    (n % 5 = 4) ∧ 
    (n % 4 = 3) ∧ 
    (n % 3 = 2) ∧ 
    (n % 2 = 1) ∧
    n = 27719 :=
sorry

end find_number_l256_256019


namespace arccos_half_eq_pi_div_three_l256_256845

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l256_256845


namespace combined_weight_of_two_new_students_l256_256561

theorem combined_weight_of_two_new_students (W : ℕ) (X : ℕ) 
  (cond1 : (W - 150 + X) / 8 = (W / 8) - 2) :
  X = 134 := 
sorry

end combined_weight_of_two_new_students_l256_256561


namespace find_sum_l256_256498

noncomputable def principal_sum (P R : ℝ) := 
  let I := (P * R * 10) / 100
  let new_I := (P * (R + 5) * 10) / 100
  I + 600 = new_I

theorem find_sum (P R : ℝ) (h : principal_sum P R) : P = 1200 := 
  sorry

end find_sum_l256_256498


namespace ratio_son_grandson_l256_256422

-- Define the conditions
variables (Markus_age Son_age Grandson_age : ℕ)
axiom Markus_twice_son : Markus_age = 2 * Son_age
axiom sum_ages : Markus_age + Son_age + Grandson_age = 140
axiom Grandson_age_20 : Grandson_age = 20

-- Define the goal to prove
theorem ratio_son_grandson : (Son_age : ℚ) / Grandson_age = 2 :=
by
  sorry

end ratio_son_grandson_l256_256422


namespace fraction_compare_l256_256042

theorem fraction_compare : 
  let a := (1 : ℝ) / 4
  let b := 250000025 / (10^9)
  let diff := a - b
  diff = (1 : ℝ) / (4 * 10^7) :=
by
  sorry

end fraction_compare_l256_256042


namespace find_number_of_violas_l256_256675

theorem find_number_of_violas (cellos : ℕ) (pairs : ℕ) (probability : ℚ) 
    (h1 : cellos = 800) 
    (h2 : pairs = 100) 
    (h3 : probability = 0.00020833333333333335) : 
    ∃ V : ℕ, V = 600 := 
by 
    sorry

end find_number_of_violas_l256_256675


namespace average_bmi_is_correct_l256_256258

-- Define Rachel's parameters
def rachel_weight : ℕ := 75
def rachel_height : ℕ := 60  -- in inches

-- Define Jimmy's parameters based on the conditions
def jimmy_weight : ℕ := rachel_weight + 6
def jimmy_height : ℕ := rachel_height + 3

-- Define Adam's parameters based on the conditions
def adam_weight : ℕ := rachel_weight - 15
def adam_height : ℕ := rachel_height - 2

-- Define the BMI formula
def bmi (weight : ℕ) (height : ℕ) : ℚ := (weight * 703 : ℚ) / (height * height)

-- Rachel's, Jimmy's, and Adam's BMIs
def rachel_bmi : ℚ := bmi rachel_weight rachel_height
def jimmy_bmi : ℚ := bmi jimmy_weight jimmy_height
def adam_bmi : ℚ := bmi adam_weight adam_height

-- Proving the average BMI
theorem average_bmi_is_correct : 
  (rachel_bmi + jimmy_bmi + adam_bmi) / 3 = 13.85 := 
by
  sorry

end average_bmi_is_correct_l256_256258


namespace arccos_one_half_l256_256884

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l256_256884


namespace SmallestPositiveAngle_l256_256711

theorem SmallestPositiveAngle (x : ℝ) (h1 : 0 < x) :
  (sin (4 * real.to_radians x) * sin (6 * real.to_radians x) = cos (4 * real.to_radians x) * cos (6 * real.to_radians x)) →
  x = 9 :=
by
  sorry

end SmallestPositiveAngle_l256_256711


namespace inequality_solution_l256_256788

theorem inequality_solution (x : ℝ) :
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3 / 2) :=
by
  sorry

end inequality_solution_l256_256788


namespace expected_up_right_paths_l256_256027

def lattice_points := {p : ℕ × ℕ // p.1 ≤ 5 ∧ p.2 ≤ 5}

def total_paths : ℕ := Nat.choose 10 5

def calculate_paths (x y : ℕ) : ℕ :=
  if h : x ≤ 5 ∧ y ≤ 5 then
    let F := total_paths * 25
    F / 36
  else
    0

theorem expected_up_right_paths : ∃ S, S = 175 :=
  sorry

end expected_up_right_paths_l256_256027


namespace findWorkRateB_l256_256805

-- Define the work rates of A and C given in the problem
def workRateA : ℚ := 1 / 8
def workRateC : ℚ := 1 / 16

-- Combined work rate when A, B, and C work together to complete the work in 4 days
def combinedWorkRate : ℚ := 1 / 4

-- Define the work rate of B that we need to prove
def workRateB : ℚ := 1 / 16

-- Theorem to prove that workRateB is equal to B's work rate given the conditions
theorem findWorkRateB : workRateA + workRateB + workRateC = combinedWorkRate :=
  by
  sorry

end findWorkRateB_l256_256805


namespace arccos_half_eq_pi_div_3_l256_256908

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l256_256908


namespace hexagon_side_equalities_l256_256749

variables {A B C D E F : Type}

-- Define the properties and conditions of the problem
noncomputable def convex_hexagon (A B C D E F : Type) : Prop :=
  True -- Since we neglect geometric properties in this abstract.

def parallel (a b : Type) : Prop := True -- placeholder for parallel condition
def equal_length (a b : Type) : Prop := True -- placeholder for length

-- Given conditions
variables (h1 : convex_hexagon A B C D E F)
variables (h2 : parallel AB DE)
variables (h3 : parallel BC FA)
variables (h4 : parallel CD FA)
variables (h5 : equal_length AB DE)

-- Statement to prove
theorem hexagon_side_equalities : equal_length BC DE ∧ equal_length CD FA := sorry

end hexagon_side_equalities_l256_256749


namespace squirrels_in_tree_l256_256126

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) (h1 : nuts = 2) (h2 : squirrels = nuts + 2) : squirrels = 4 :=
by
    rw [h1] at h2
    exact h2

end squirrels_in_tree_l256_256126


namespace probability_not_square_or_cube_l256_256645

theorem probability_not_square_or_cube : 
  let total_numbers := 200
  let perfect_squares := {n | n^2 ≤ 200}.card
  let perfect_cubes := {n | n^3 ≤ 200}.card
  let perfect_sixth_powers := {n | n^6 ≤ 200}.card
  let total_perfect_squares_cubes := perfect_squares + perfect_cubes - perfect_sixth_powers
  let neither_square_nor_cube := total_numbers - total_perfect_squares_cubes
  neither_square_nor_cube / total_numbers = 183 / 200 := 
by
  sorry

end probability_not_square_or_cube_l256_256645


namespace probability_neither_square_nor_cube_l256_256638

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube :
  ∃ p : ℚ, p = 183 / 200 ∧
           p = 
           (((finset.range 200).filter (λ n, ¬ is_perfect_square (n + 1) ∧ ¬ is_perfect_cube (n + 1))).card).to_nat / 200 :=
by
  sorry

end probability_neither_square_nor_cube_l256_256638


namespace fraction_of_air_conditioned_rooms_rented_l256_256255

variable (R : ℚ)
variable (h1 : R > 0)
variable (rented_rooms : ℚ := (3/4) * R)
variable (air_conditioned_rooms : ℚ := (3/5) * R)
variable (not_rented_rooms : ℚ := (1/4) * R)
variable (air_conditioned_not_rented_rooms : ℚ := (4/5) * not_rented_rooms)
variable (air_conditioned_rented_rooms : ℚ := air_conditioned_rooms - air_conditioned_not_rented_rooms)
variable (fraction_air_conditioned_rented : ℚ := air_conditioned_rented_rooms / air_conditioned_rooms)

theorem fraction_of_air_conditioned_rooms_rented :
  fraction_air_conditioned_rented = (2/3) := by
  sorry

end fraction_of_air_conditioned_rooms_rented_l256_256255


namespace cos_pi_over_3_arccos_property_arccos_one_half_l256_256878

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l256_256878


namespace monotonicity_of_f_tangent_intersection_coordinates_l256_256371

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x + 1

-- Part 1: Monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (a ≥ (1 : ℝ) / 3 → ∀ x y : ℝ, x ≤ y → f x a ≤ f y a) ∧
  (a < (1 : ℝ) / 3 → 
    (∀ x : ℝ, x < (1 - real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, x > (1 + real.sqrt (1 - 3 * a)) / 3 → ∀ y : ℝ, x ≤ y → f x a ≤ f y a) ∧
    (∀ x : ℝ, (1 - real.sqrt (1 - 3 * a)) / 3 < x ∧ x < (1 + real.sqrt (1 - 3 * a)) / 3 → 
              ∀ y : ℝ, x ≤ y → f x a ≥ f y a)) :=
sorry

-- Part 2: Tangent intersection coordinates
theorem tangent_intersection_coordinates (a : ℝ) :
  (2 * (1 : ℝ)^3 - (1 : ℝ)^2 - 1 = 0) →
  (f 1 a, f (-1) a) = (a + 1, -a - 1) :=
sorry

end monotonicity_of_f_tangent_intersection_coordinates_l256_256371


namespace length_of_plot_57_meters_l256_256622

section RectangleProblem

variable (b : ℝ) -- breadth of the plot
variable (l : ℝ) -- length of the plot
variable (cost_per_meter : ℝ) -- cost per meter
variable (total_cost : ℝ) -- total cost

-- Given conditions
def length_eq_breadth_plus_14 (b l : ℝ) : Prop := l = b + 14
def cost_eq_perimeter_cost_per_meter (cost_per_meter total_cost perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

-- Definition of perimeter
def perimeter (b l : ℝ) : ℝ := 2 * l + 2 * b

-- Problem statement
theorem length_of_plot_57_meters
  (h1 : length_eq_breadth_plus_14 b l)
  (h2 : cost_eq_perimeter_cost_per_meter cost_per_meter total_cost (perimeter b l))
  (h3 : cost_per_meter = 26.50)
  (h4 : total_cost = 5300) :
  l = 57 :=
by
  sorry

end RectangleProblem

end length_of_plot_57_meters_l256_256622


namespace power_difference_mod_7_l256_256833

theorem power_difference_mod_7 :
  (45^2011 - 23^2011) % 7 = 5 := by
  have h45 : 45 % 7 = 3 := by norm_num
  have h23 : 23 % 7 = 2 := by norm_num
  sorry

end power_difference_mod_7_l256_256833


namespace probability_of_passing_test_l256_256409

theorem probability_of_passing_test (p : ℝ) (h : p + p * (1 - p) + p * (1 - p)^2 = 0.784) : p = 0.4 :=
sorry

end probability_of_passing_test_l256_256409


namespace baguettes_sold_third_batch_l256_256436

-- Definitions of the conditions
def daily_batches : ℕ := 3
def baguettes_per_batch : ℕ := 48
def baguettes_sold_first_batch : ℕ := 37
def baguettes_sold_second_batch : ℕ := 52
def baguettes_left : ℕ := 6

theorem baguettes_sold_third_batch : 
  daily_batches * baguettes_per_batch - (baguettes_sold_first_batch + baguettes_sold_second_batch + baguettes_left) = 49 :=
by sorry

end baguettes_sold_third_batch_l256_256436


namespace roots_sum_reciprocal_squares_l256_256619

theorem roots_sum_reciprocal_squares (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + bc + ca = 20) (h3 : abc = 3) :
  (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2) = 328 / 9 := 
by
  sorry

end roots_sum_reciprocal_squares_l256_256619


namespace arccos_half_is_pi_div_three_l256_256852

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l256_256852
