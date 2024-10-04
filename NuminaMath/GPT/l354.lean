import Mathlib

namespace number_of_solutions_l354_354079

theorem number_of_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, 20 ≤ x ∧ x ≤ 150 ∧ Odd x ∧ (x + 17) % 29 = 65 % 29) ∧ S.card = 3 :=
by
  sorry

end number_of_solutions_l354_354079


namespace pyramid_volume_l354_354031

-- Definitions of vertices
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (30, 0)
def C : (ℝ × ℝ) := (10, 20)

-- Midpoints calculation (given as conditions)
def D : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- midpoint AB
def E : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- midpoint BC
def F : (ℝ × ℝ) := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)  -- midpoint CA

-- Centroid of triangle ABC
def centroid : (ℝ × ℝ) := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Area calculation of base triangle
def area_ABC : ℝ := abs (1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Height of pyramid
def height_pyramid : ℝ := (20 - 0) / 2

-- Volume calculation of the pyramid
def volume_pyramid : ℝ := (1 / 3) * area_ABC * height_pyramid

theorem pyramid_volume : volume_pyramid = 1000 := by
  -- Proof would go here
  sorry

end pyramid_volume_l354_354031


namespace positions_after_347_moves_l354_354127

-- Define the possible positions for the cat
inductive CatPosition
| top_vertex
| right_upper_vertex
| right_lower_vertex
| left_lower_vertex
| left_upper_vertex

-- Define the possible positions for the mouse
inductive MousePosition
| top_left_edge
| left_upper_vertex
| left_middle_edge
| left_lower_vertex
| bottom_edge
| right_lower_vertex
| right_middle_edge
| right_upper_vertex
| top_right_edge
| top_vertex

-- Define the movement function for the cat
def cat_position_after_moves (moves : Nat) : CatPosition :=
  match moves % 5 with
  | 0 => CatPosition.top_vertex
  | 1 => CatPosition.right_upper_vertex
  | 2 => CatPosition.right_lower_vertex
  | 3 => CatPosition.left_lower_vertex
  | 4 => CatPosition.left_upper_vertex
  | _ => CatPosition.top_vertex  -- This case is unreachable due to % 5

-- Define the movement function for the mouse
def mouse_position_after_moves (moves : Nat) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.top_left_edge
  | 1 => MousePosition.left_upper_vertex
  | 2 => MousePosition.left_middle_edge
  | 3 => MousePosition.left_lower_vertex
  | 4 => MousePosition.bottom_edge
  | 5 => MousePosition.right_lower_vertex
  | 6 => MousePosition.right_middle_edge
  | 7 => MousePosition.right_upper_vertex
  | 8 => MousePosition.top_right_edge
  | 9 => MousePosition.top_vertex
  | _ => MousePosition.top_left_edge  -- This case is unreachable due to % 10

-- Prove the positions after 347 moves
theorem positions_after_347_moves :
  cat_position_after_moves 347 = CatPosition.right_upper_vertex ∧
  mouse_position_after_moves 347 = MousePosition.right_middle_edge :=
by
  sorry

end positions_after_347_moves_l354_354127


namespace correct_number_of_propositions_l354_354078

variable (a b : Line) (α β : Plane)

-- Conditions for each proposition
variable (h1a : Perpendicular a α) (h1b : Perpendicular b α) 
variable (h2a : Parallel a α) (h2b : Parallel b α) 
variable (h3a : Perpendicular a α) (h3b : Perpendicular a β) 
variable (h4a : Parallel α b) (h4b : Parallel β b)

theorem correct_number_of_propositions :
  (if (a ∥ b) then 1 else 0) + 
  (if (a ∥ b) then 0 else 1) + 
  (if (α ∥ β) then 1 else 0) + 
  (if (α ∥ β) then 0 else 1) = 2 := 
sorry

end correct_number_of_propositions_l354_354078


namespace sin_cos_sum_l354_354116

variable (θ a : Real)
variable h1 : θ ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 4)
variable h2 : Real.sin (2 * θ) = a

theorem sin_cos_sum (h1 : θ ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 4)) 
(h2 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) :=
sorry

end sin_cos_sum_l354_354116


namespace mod_equiv_l354_354688

theorem mod_equiv (a b c d e : ℤ) (n : ℤ) (h1 : a = 101)
                                    (h2 : b = 15)
                                    (h3 : c = 7)
                                    (h4 : d = 9)
                                    (h5 : e = 5)
                                    (h6 : n = 17) :
  (a * b - c * d + e) % n = 7 := by
  sorry

end mod_equiv_l354_354688


namespace irreducible_polynomial_l354_354864

noncomputable def f (a : Fin n → ℕ) (x : ℤ) : ℤ :=
  (finset.univ : finset (Fin n)).sum (λ i, a i * x ^ (i : ℤ))

theorem irreducible_polynomial (b k : ℤ) (n : ℕ) 
  (a : Fin n → ℕ) (p : ℤ) 
  (h1 : 1 < k) (h2 : k < b) 
  (h3 : (finset.univ : finset (Fin n)).sum (λ i, ↑(a i) * b ^ (i : ℤ)) = k * p) 
  (h4 : ∀ (r : ℤ), f a r = 0 → |r - b| > int.sqrt k) :
  irreducible (univ.sum (λ i, a i * polynomial.X ^ (i : ℕ))) :=
sorry

end irreducible_polynomial_l354_354864


namespace ellipse_and_point_exists_l354_354086

noncomputable def hyperbola_eq := ∀ x y : ℝ, x^2 - y^2 / 2 = 1

noncomputable def ellipse_params (a b c : ℝ) := 
  a = sqrt 3 ∧ b = sqrt 2 ∧ c = 1

noncomputable def ellipse_eq := ∀ x y : ℝ, (x^2 / 3) + (y^2 / 2) = 1

noncomputable def point_exists (ellipse_eq : ∀ x y : ℝ, (x^2 / 3) + (y^2 / 2) = 1) := 
  ∃ P : ℝ × ℝ, P = (3,0) ∧
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧
  (∃ l : ℝ → ℝ, l A.1 = A.2 ∧ l B.1 = B.2)
  → 
  let slope_P_A := (A.2 / (A.1 - P.1))
  let slope_P_B := (B.2 / (B.1 - P.1))
  in (slope_P_A + slope_P_B = 0)

theorem ellipse_and_point_exists :
  (hyperbola_eq ∧ ellipse_params (sqrt 3) (sqrt 2) 1 → 
  ∃ (ellipse : ∀ x y : ℝ, (x^2 / 3) + (y^2 / 2) = 1), ellipse ∧ point_exists ellipse) :=
sorry

end ellipse_and_point_exists_l354_354086


namespace sqrt_pi_expr_equals_pi_sub_3_l354_354983

theorem sqrt_pi_expr_equals_pi_sub_3 : sqrt (π^2 - 6 * π + 9) = π - 3 := by
  -- The proof would go here
  sorry

end sqrt_pi_expr_equals_pi_sub_3_l354_354983


namespace hyperbola_focal_length_l354_354913

theorem hyperbola_focal_length : 
  ∀ {x y : ℝ}, 
    (x^2 / 10 - y^2 / 2 = 1) → 2 * sqrt (10 + 2) = 4 * sqrt 3 :=
by
  intro x y h
  have a_sq : ℝ := 10
  have b_sq : ℝ := 2
  have c_sq : ℝ := a_sq + b_sq
  have c : ℝ := sqrt c_sq
  have focal_length := 2 * c
  rw [←sqrt_add, @eq.complex 12 12, mul_comm] at focal_length
  exact focal_length

noncomputable example : 2 * sqrt (10 + 2) = 4 * sqrt 3 := 
by
  rw [show 10 + 2 = 12, from rfl, sqrt_mul, sqrt_mul, show (4 * 3 : ℝ) = 12 by norm_num]
  congr
  norm_num

end hyperbola_focal_length_l354_354913


namespace distribute_fruits_evenly_l354_354506

theorem distribute_fruits_evenly :
  let k_strawberries_baskets := 8 * 3 in -- Kimberly's 24 strawberry baskets
  let k_strawberries := k_strawberries_baskets * 15 in -- Each basket has 15 strawberries
  let k_blueberries_baskets := 5 in
  let k_blueberries := k_blueberries_baskets * 40 in -- Each basket has 40 blueberries

  let b_strawberries_baskets := 3 in
  let b_strawberries := b_strawberries_baskets * 15 in -- Each basket has 15 strawberries
  let b_blackberries_baskets := 4 in
  let b_blackberries := b_blackberries_baskets * 30 in -- Each basket has 30 blackberries

  let p_blackberries := b_blackberries - 75 in -- Parents pick 75 less blackberries than brother
  let p_blueberries_baskets := k_blueberries_baskets + 4 in
  let p_blueberries := p_blueberries_baskets * 55 in -- Each of their baskets contains 55 blueberries

  let total_fruits := k_strawberries + k_blueberries + b_strawberries + b_blackberries + p_blackberries + p_blueberries in
  total_fruits = 1265 →
  (total_fruits / 4).natAbs = 316 :=
by
  intros
  sorry

end distribute_fruits_evenly_l354_354506


namespace BP_PA_ratio_l354_354835

section

variable (A B C P : Type)
variable {AC BC PA PB BP : ℕ}

-- Conditions:
-- 1. In triangle ABC, the ratio AC:CB = 2:5.
axiom AC_CB_ratio : 2 * BC = 5 * AC

-- 2. The bisector of the exterior angle at C intersects the extension of BA at P,
--    such that B is between P and A.
axiom Angle_Bisector_Theorem : PA * BC = PB * AC

theorem BP_PA_ratio (h1 : 2 * BC = 5 * AC) (h2 : PA * BC = PB * AC) :
  BP * PA = 5 * PA := sorry

end

end BP_PA_ratio_l354_354835


namespace cubic_identity_l354_354466

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 40) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1575 := 
by
  sorry

end cubic_identity_l354_354466


namespace rotating_isosceles_trapezoid_forms_frustum_l354_354547

theorem rotating_isosceles_trapezoid_forms_frustum (T : Type) [trapezoid T] [isosceles T] [rotate_around_axis_of_symmetry T] : shape T = frustum :=
by sorry

end rotating_isosceles_trapezoid_forms_frustum_l354_354547


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354424

theorem option_A_correct (x : ℝ) (hx : x ≥ 1) : 
  let y := (x - 2) / (2 * x + 1)
  in y ∈ set.Ico (-1/3 : ℝ) (1/2 : ℝ) :=
sorry

theorem option_B_correct (f : ℝ → ℝ) (domain_of_f : set.Icc (-1 : ℝ) 1) :
  let y := λ x, f(x - 1) / real.sqrt(x - 1)
  in ∀ x, x ∈ set.Ioo (1 : ℝ) (2 : ℝ) → x - 1 ∈ domain_of_f :=
sorry

theorem option_C_correct (f : ℝ → ℝ) :
  (∀ x, f x = x^2) ∧ ∃ A : set ℝ, (A = {-2}) ∨ (A = {2}) ∨ (A = {-2, 2}) :=
sorry

theorem option_D_incorrect (f : ℝ → ℝ) (m : ℝ) (hyp1 : ∀ x, f(x + 1/x) = x^2 + 1/x^2) (hyp2 : f m = 4) :
  m ≠ real.sqrt 6 :=
sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354424


namespace chess_tournament_points_difference_l354_354816

theorem chess_tournament_points_difference :
  let total_participants := 12
  let games_played := 66
  let total_points := 66
  let vasya_points := 5
  let petya_points := 6
  let points_difference := petya_points - vasya_points
  in points_difference = 1 :=
by
  sorry

end chess_tournament_points_difference_l354_354816


namespace taxi_fare_l354_354293

theorem taxi_fare :
  ∀ (initial_fee rate_per_increment increment_distance total_distance : ℝ),
    initial_fee = 2.35 →
    rate_per_increment = 0.35 →
    increment_distance = (2 / 5) →
    total_distance = 3.6 →
    (initial_fee + rate_per_increment * (total_distance / increment_distance)) = 5.50 :=
by
  intros initial_fee rate_per_increment increment_distance total_distance
  intro h1 h2 h3 h4
  sorry -- Proof is not required.

end taxi_fare_l354_354293


namespace abby_correct_percentage_l354_354823

-- Defining the scores and number of problems for each test
def score_test1 := 85 / 100
def score_test2 := 75 / 100
def score_test3 := 60 / 100
def score_test4 := 90 / 100

def problems_test1 := 30
def problems_test2 := 50
def problems_test3 := 20
def problems_test4 := 40

-- Define the total number of problems
def total_problems := problems_test1 + problems_test2 + problems_test3 + problems_test4

-- Calculate the number of problems Abby answered correctly on each test
def correct_problems_test1 := score_test1 * problems_test1
def correct_problems_test2 := score_test2 * problems_test2
def correct_problems_test3 := score_test3 * problems_test3
def correct_problems_test4 := score_test4 * problems_test4

-- Calculate the total number of correctly answered problems
def total_correct_problems := correct_problems_test1 + correct_problems_test2 + correct_problems_test3 + correct_problems_test4

-- Calculate the overall percentage of problems answered correctly
def overall_percentage_correct := (total_correct_problems / total_problems) * 100

-- The theorem to be proved
theorem abby_correct_percentage : overall_percentage_correct = 80 := by
  -- Skipping the actual proof
  sorry

end abby_correct_percentage_l354_354823


namespace number_of_valid_rectangles_l354_354238

-- Define the conditions
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_rectangle_sides (a b : ℕ) : Prop :=
  is_even a ∧ is_even b ∧ a * b = 96

-- State the problem as a Lean theorem
theorem number_of_valid_rectangles :
  {p : ℕ × ℕ | valid_rectangle_sides p.1 p.2}.to_finset.card = 4 :=
sorry

end number_of_valid_rectangles_l354_354238


namespace solve_equation_l354_354060

theorem solve_equation {x y z : ℝ} (h₁ : x + 95 / 12 * y + 4 * z = 0)
  (h₂ : 4 * x + 95 / 12 * y - 3 * z = 0)
  (h₃ : 3 * x + 5 * y - 4 * z = 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 * z / y^3 = -60 :=
sorry

end solve_equation_l354_354060


namespace log_base_2_b_range_of_x_l354_354391

-- Definitions based on conditions
def a (x : ℝ) : ℝ := 2^x
def b : ℝ := 4^(2/3)

-- Prove that log base 2 of b equals 4/3
theorem log_base_2_b : Real.logBase 2 b = 4 / 3 :=
by
  sorry

-- Prove the range of x satisfying log base a of b <= 1
theorem range_of_x (x : ℝ) : Real.logBase (a x) b ≤ 1 ↔ x < 0 ∨ x ≥ 4 / 3 :=
by
  sorry


end log_base_2_b_range_of_x_l354_354391


namespace geometric_sequence_min_l354_354125

theorem geometric_sequence_min (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_condition : 2 * (a 4) + (a 3) - 2 * (a 2) - (a 1) = 8)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  ∃ min_val, min_val = 12 * Real.sqrt 3 ∧ min_val = 2 * (a 5) + (a 4) :=
sorry

end geometric_sequence_min_l354_354125


namespace find_s6_l354_354113

noncomputable theory
open Classical

variables {a b x y : ℝ}
variables {s: ℕ → ℝ}

-- Define the initial conditions
def s1 : ℝ := a * x + b * y
def s2 : ℝ := a * x^2 + b * y^2
def s3 : ℝ := a * x^3 + b * y^3
def s4 : ℝ := a * x^4 + b * y^4
def s5 : ℝ := a * x^5 + b * y^5

-- Assert the given values
axiom s1_eq : s1 = 5
axiom s2_eq : s2 = 11
axiom s3_eq : s3 = 23
axiom s4_eq : s4 = 50
axiom s5_eq : s5 = 106

-- State and prove the problem
theorem find_s6 (h : ∀ n, s (n + 1) = (x + y) * s n - x * y * s (n - 1)) : a * x^6 + b * y^6 = 238 :=
sorry

end find_s6_l354_354113


namespace hyperbola_eccentricity_range_hyperbola_equation_given_conditions_l354_354061

open Real

theorem hyperbola_eccentricity_range (b : ℝ) (e : ℝ) (m : ℝ) 
  (h₀ : b > 0)
  (h_line_intersect : ∀ m : ℝ, ∃ x y : ℝ, y = x + m ∧ (x^2 / 2 - y^2 / b^2 = 1)) : 
  e > sqrt 2 := sorry

theorem hyperbola_equation_given_conditions (c b : ℝ) (m : ℝ) 
  (h₀ : b > 0)
  (h_focus : ∀ x y : ℝ, x = y - c ∧ (x^2 / 2 - (y - c)^2 / b^2 = 1))
  (h_focus_points : ∃ P Q : ℝ × ℝ, 
    ∀ y1 y2 : ℝ, (overrightarrow P = (1/5 : ℝ) * overrightarrow Q)) 
  : b^2 = 7 ∧ (∀ x y : ℝ, x^2 / 2 - y^2 / 7 = 1) := sorry

end hyperbola_eccentricity_range_hyperbola_equation_given_conditions_l354_354061


namespace percentage_of_water_in_raisins_l354_354770

theorem percentage_of_water_in_raisins
  (original_weight : ℝ)
  (raisins_weight : ℝ)
  (water_content_percentage : ℝ)
  (h_original_weight : original_weight = 95.99999999999999)
  (h_raisins_weight : raisins_weight = 8)
  (h_water_content_percentage : water_content_percentage = 93) :
  let water_weight := original_weight * (water_content_percentage / 100)
      water_lost := original_weight - raisins_weight
      water_in_raisins := water_weight - water_lost
      water_percentage := (water_in_raisins / raisins_weight) * 100
  in water_percentage = 16 :=
sorry

end percentage_of_water_in_raisins_l354_354770


namespace sum_b_sequence_l354_354414

noncomputable def sequence_a (n : ℕ) : ℕ := n

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def sequence_b (n : ℕ) (p : ℝ) [fact (0 < p)] : ℝ :=
  sequence_a n * p ^ sequence_a n

noncomputable def T (n : ℕ) (p : ℝ) [fact (0 < p)] : ℝ :=
  if p = 1 then
    (n * (n + 1) / 2)
  else
    (p * (1 - p ^ n) / (1 - p) ^ 2 - n * p ^ (n + 1) / (1 - p))

theorem sum_b_sequence (n : ℕ) (p : ℝ) [fact (0 < p)] :
  T n p = if p = 1 then
            (n * (n + 1) / 2)
          else
            (p * (1 - p^n) / (1 - p)^2 - n * p^(n+1) / (1 - p))
:= sorry

end sum_b_sequence_l354_354414


namespace solution_of_inequalities_l354_354699

theorem solution_of_inequalities (x : ℝ) :
  (2 * x / 5 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) ↔ (-5 ≤ x ∧ x < -3 / 2) := by
  sorry

end solution_of_inequalities_l354_354699


namespace negation_of_proposition_l354_354570

variable (x : ℝ)

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by
  simp
  sorry

end negation_of_proposition_l354_354570


namespace julia_garden_area_l354_354138

theorem julia_garden_area
  (length perimeter walk_distance : ℝ)
  (h_length : length * 30 = walk_distance)
  (h_perimeter : perimeter * 12 = walk_distance)
  (h_perimeter_def : perimeter = 2 * (length + width))
  (h_walk_distance : walk_distance = 1500) :
  (length * width = 625) :=
by
  sorry

end julia_garden_area_l354_354138


namespace rice_amount_l354_354923

variable (P M : Real)
variable (price_drop_perc : Real := 0.2)
variable (current_rice : Real := 25)
variable (new_price := 0.8 * P)
variable (M := 25 * new_price)

theorem rice_amount :
  let previous_rice := M / P
  in previous_rice = 20 := by
sorry

end rice_amount_l354_354923


namespace gondor_repaired_3_phones_on_monday_l354_354105

theorem gondor_repaired_3_phones_on_monday :
  ∃ P : ℕ, 
    (10 * P + 10 * 5 + 20 * 2 + 20 * 4 = 200) ∧
    P = 3 :=
by
  sorry

end gondor_repaired_3_phones_on_monday_l354_354105


namespace find_fraction_l354_354716

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l354_354716


namespace roots_sum_product_l354_354169

variable {a b : ℝ}

theorem roots_sum_product (ha : a + b = 6) (hp : a * b = 8) : 
  a^4 + b^4 + a^3 * b + a * b^3 = 432 :=
by
  sorry

end roots_sum_product_l354_354169


namespace circletangent_inequality_l354_354675

theorem circletangent_inequality {O1 O2 A B M N : Type}
  (R r a b : ℝ)
  (h1 : circle O1)
  (h2 : circle O2)
  (h3 : R = radius O1)
  (h4 : r = radius O2)
  (h_inter : intersect O1 O2 A B)
  (h_tangent1 : tangent O1 M)
  (h_tangent2 : tangent O2 N)
  (h_circum1 : circumradius (triangle A M N) = a)
  (h_circum2 : circumradius (triangle B M N) = b) :
  R + r ≥ a + b :=
sorry

end circletangent_inequality_l354_354675


namespace ratio_of_side_lengths_sum_l354_354247

theorem ratio_of_side_lengths_sum (a b c : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 25) :
  a + b + c = 44 := 
by
  sorry

end ratio_of_side_lengths_sum_l354_354247


namespace radius_vector_l354_354150

theorem radius_vector 
  (O M : Euclidean_space ℝ (fin 3)) 
  (x y z : ℝ)
  (orig : O = (λ _, 0))
  (pointM : M = (λ i, if i = 0 then -2 else if i = 1 then 5 else if i = 2 then 0 else 0)) : 
  (λ i, if i = 0 then -2 else if i = 1 then 5 else if i = 2 then 0 else 0) = M :=
begin
  sorry
end

end radius_vector_l354_354150


namespace bernardo_larger_than_silvia_l354_354016

open BigOperators

-- Define the sets Bernardo and Silvia pick from
def bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def silvia_set := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable theory

-- Define the problem statement
theorem bernardo_larger_than_silvia : 
  (∑ b in bernardo_set.powerset.filter (λ s, s.card = 3), 1 : ℚ) /
  (∑ b in finset.powerset_len 3 bernardo_set, 1) /
  ((∑ s in silvia_set.powerset.filter (λ s, s.card = 3), 1 : ℚ) /
  (∑ s in finset.powerset_len 3 silvia_set, 1)) = 49 / 96 := 
sorry

end bernardo_larger_than_silvia_l354_354016


namespace woman_alone_days_l354_354307

theorem woman_alone_days (M W : ℝ) (h1 : (10 * M + 15 * W) * 5 = 1) (h2 : M * 100 = 1) : W * 150 = 1 :=
by
  sorry

end woman_alone_days_l354_354307


namespace harvey_sold_17_steaks_l354_354771

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l354_354771


namespace bookshelf_orderings_count_l354_354817

theorem bookshelf_orderings_count :
  let books := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  (stack : Finset ℕ) (h₀ : stack ⊆ Finset.range 9) -- books 1 through 8 might form the stack
  (h₁ : books \ stack = {9}) -- book 9 has been shelved
  (h₂ : ∃ k, stack.card = k ∧ ∑ k in Finset.range (k + 1), choose 8 k * (k + 2) = 1280)
  : ∑ k in Finset.range 9, choose 8 k * (k + 2) = 1280 :=
begin
  sorry -- proof to be filled
end

end bookshelf_orderings_count_l354_354817


namespace power_function_monotonically_decreasing_interval_l354_354915

open Real

noncomputable def power_function_passes_through_point (a : ℝ) : Prop :=
  2^a = 1 / 4

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x

theorem power_function_monotonically_decreasing_interval (a : ℝ) 
  (h : power_function_passes_through_point a) : 
  is_monotonically_decreasing_on (λ x, x^a) (Set.Ioi 0) :=
sorry

end power_function_monotonically_decreasing_interval_l354_354915


namespace system_solution_l354_354215

theorem system_solution (x y : ℚ) (h1 : 2 * x - 3 * y = 1) (h2 : (y + 1) / 4 + 1 = (x + 2) / 3) : x = 3 ∧ y = 5 / 3 :=
by
  sorry

end system_solution_l354_354215


namespace platform_length_calc_l354_354314

theorem platform_length_calc (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : 
  train_speed_kmph = 72 → crossing_time_s = 26 → train_length_m = 290.04 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600) in
  let distance_covered := train_speed_mps * crossing_time_s in
  let platform_length := distance_covered - train_length_m in
  platform_length = 229.96 := by
  intros
  sorry

end platform_length_calc_l354_354314


namespace double_recipe_total_l354_354246

theorem double_recipe_total 
  (butter_ratio : ℕ) (flour_ratio : ℕ) (sugar_ratio : ℕ) 
  (flour_cups : ℕ) 
  (h_ratio : butter_ratio = 2) 
  (h_flour : flour_ratio = 5) 
  (h_sugar : sugar_ratio = 3) 
  (h_flour_cups : flour_cups = 15) : 
  2 * ((butter_ratio * (flour_cups / flour_ratio)) + flour_cups + (sugar_ratio * (flour_cups / flour_ratio))) = 60 := 
by 
  sorry

end double_recipe_total_l354_354246


namespace square_area_l354_354578

theorem square_area (A B : ℝ × ℝ) (hA : A = (0, 5)) (hB : B = (5, 0)) :
  let s := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  s^2 = 50 :=
by
  have hs : s = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2), from rfl,
  rw [hA, hB] at hs,
  sorry

end square_area_l354_354578


namespace closest_number_to_fraction_l354_354370

noncomputable def closest_to_fraction : ℝ :=
  let fraction := 403 / 0.21
  let options := [0.2, 2, 20, 200, 2000]
  2000

theorem closest_number_to_fraction : 
  ∃ (closest : ℝ), closest ∈ [0.2, 2, 20, 200, 2000] ∧ closest = 2000 :=
by
  use 2000
  split
  · simp [List.mem]; tauto
  · rfl

end closest_number_to_fraction_l354_354370


namespace shaded_area_is_correct_l354_354664

theorem shaded_area_is_correct :
  (∃ (A B C D E : Point),
    is_square ABCD ∧
    side_length ABCD = 6 ∧
    distance D E = 10) →
  shaded_area ABCD DE = 10.8 :=
begin
  sorry
end

end shaded_area_is_correct_l354_354664


namespace median_line_eqn_triangle_area_l354_354748

open Real
open Set

noncomputable def midpoint (p1 p2 : ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def line_eqn (p1 p2 : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p, (p2.2 - p1.2) * (p.1 - p1.1) = (p.2 - p1.2) * (p2.1 - p1.1)

theorem median_line_eqn :
  ∀ {(A B C : ℝ × ℝ)},
    A = (0, -2) →
    B = (4, -3) →
    C = (3, 1) →
    ∃ l : ℝ × ℝ → Prop, (line_eqn B (midpoint A C)) = l ∧ l = λ p, p.1 + p.2 - 1 = 0 :=
by sorry

def distance_line_to_point (a b c : ℝ) (x y : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem triangle_area :
  ∀ {A B C : ℝ × ℝ},
    A = (0, -2) →
    B = (4, -3) →
    C = (3, 1) →
    ∃ S : ℝ, S = 15 / 2 ∧ S = 1 / 2 * dist A C * distance_line_to_point 1 (-1) (-2) B.1 B.2 :=
by sorry

end median_line_eqn_triangle_area_l354_354748


namespace sqrt_pi_squared_minus_six_pi_plus_nine_l354_354981

noncomputable def pi : ℝ := Real.pi

theorem sqrt_pi_squared_minus_six_pi_plus_nine : sqrt (pi^2 - 6 * pi + 9) = pi - 3 :=
by
  sorry

end sqrt_pi_squared_minus_six_pi_plus_nine_l354_354981


namespace simplify_and_evaluate_expression_l354_354549

/-
Problem: Prove ( (a + 1) / (a - 1) + 1 ) / ( 2a / (a^2 - 1) ) = 2024 given a = 2023.
-/

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  ( (a + 1) / (a - 1) + 1 ) / ( 2 * a / (a^2 - 1) ) = 2024 :=
by
  sorry

end simplify_and_evaluate_expression_l354_354549


namespace integral_sin_cos_pow_eq_l354_354297

theorem integral_sin_cos_pow_eq :
  ∫ x in -Real.pi / 2 .. 0, 2 ^ 8 * (Real.sin x) ^ 4 * (Real.cos x) ^ 4 = 3 * Real.pi := 
sorry

end integral_sin_cos_pow_eq_l354_354297


namespace count_3_digit_numbers_divisible_by_5_l354_354779

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354779


namespace number_of_ways_to_play_cards_l354_354325

theorem number_of_ways_to_play_cards :
  let num_twos := 2
      num_aces := 3
      total_cards := 5
  in
  (∃ (ways : ℕ), ways = 5! + 2! + 4! + (nat.choose 3 2) * 3! + 3! + (nat.choose 3 2) * 4! ) → ways = 242 :=
by
  let num_twos := 2
  let num_aces := 3
  let total_cards := 5
  let cases := 5! + 2! + 4! + (nat.choose 3 2) * 3! + 3! + (nat.choose 3 2) * 4!
  show cases = 242 from
  sorry

end number_of_ways_to_play_cards_l354_354325


namespace boat_arrival_interval_l354_354527

-- Define the speeds and the launching interval
def losyash_speed := 4 -- in km/h
def boat_speed := 10 -- in km/h
def launch_interval := 0.5 -- in hours

-- Define the expected time interval of boats arriving at Sovunya's house
def expected_interval := 18 -- in minutes

-- Conversion from minutes to hours
def min_to_hour (m : ℕ) : ℚ := m / 60

theorem boat_arrival_interval :
  let interval := 18 in
  min_to_hour interval = ((boat_speed - losyash_speed) * launch_interval) / boat_speed :=
by {
  sorry
}

end boat_arrival_interval_l354_354527


namespace find_number_of_arithmetic_sequences_l354_354037

def isArithmeticSequence (s : Finset ℕ) : Prop :=
  s.card = 4 ∧
  ∃ d a b c : ℕ, d ≠ 0 ∧
    s = {a, a + d, a + 2 * d, a + 3 * d} ∧
    {a, a + d, a + 2 * d, a + 3 * d} ⊆ (Finset.range 13)

theorem find_number_of_arithmetic_sequences :
  (Finset.filter isArithmeticSequence (Finset.powersetLen 4 (Finset.range 13))).card = 21 :=
by
  sorry

end find_number_of_arithmetic_sequences_l354_354037


namespace pyramid_volume_l354_354707

noncomputable def volume_of_pyramid (a h : ℝ) : ℝ :=
  (a^2 * h) / (4 * Real.sqrt 3)

theorem pyramid_volume (d x y : ℝ) (a h : ℝ) (edge_distance lateral_face_distance : ℝ)
  (H1 : edge_distance = 2) (H2 : lateral_face_distance = Real.sqrt 12)
  (H3 : x = 2) (H4 : y = 2 * Real.sqrt 3) (H5 : d = (a * Real.sqrt 3) / 6)
  (H6 : h = Real.sqrt (48 / 5)) :
  volume_of_pyramid a h = 216 * Real.sqrt 3 := by
  sorry

end pyramid_volume_l354_354707


namespace f_is_increasing_on_interval_l354_354754

noncomputable def f (ω x ϕ : ℝ) : ℝ := 
  sin (ω * x + ϕ) + cos (ω * x + ϕ)

theorem f_is_increasing_on_interval
  (ω ϕ : ℝ) (hω : ω > 0) (hϕ : 0 < ϕ) (hϕπ : ϕ < π)
  (f_odd : ∀ x : ℝ, f ω x ϕ = - f ω (-x) ϕ)
  (intersection_dist : ∀ x1 x2 : ℝ, f ω x1 ϕ = √2 ∧ f ω x2 ϕ = √2 → abs (x2 - x1) = π / 2) :
  ∀ x : ℝ, x ∈ Ioo (π / 8) (3 * π / 8) → strict_mono (λ t, f ω t ϕ) x :=
sorry

end f_is_increasing_on_interval_l354_354754


namespace roses_in_vase_l354_354595

theorem roses_in_vase (initial_roses added_roses total_roses : ℕ) 
  (h₁ : initial_roses = 6)
  (h₂ : added_roses = 16) 
  (h₃ : total_roses = initial_roses + added_roses) : 
  total_roses = 22 := 
by 
  rw [h₁, h₂] at h₃
  rw h₃
  exact rfl

end roses_in_vase_l354_354595


namespace sine_identity_condition_l354_354839

theorem sine_identity_condition 
  (A B : ℝ) 
  (h : sin (A - B) * cos B + cos (A - B) * sin B ≥ 1) 
  (hB : ∃ C : ℝ, 0 < C ∧ C + A + B = π ∧ 0 < A ∧ 0 < B): 
  (A = π / 2 → true) ∧ ¬(B = π / 2 → (sin A ≥ 1)) :=
by sorry

end sine_identity_condition_l354_354839


namespace cosine_sum_l354_354029

theorem cosine_sum :
  (∑ k in finset.range 91, 2 * (real.cos (10 + k : ℝ))^2) = 142 :=
by
  sorry

end cosine_sum_l354_354029


namespace square_of_binomial_conditions_l354_354613

variable (x a b m : ℝ)

theorem square_of_binomial_conditions :
  ∃ u v : ℝ, (x + a) * (x - a) = u^2 - v^2 ∧
             ∃ e f : ℝ, (-x - b) * (x - b) = - (e^2 - f^2) ∧
             ∃ g h : ℝ, (b + m) * (m - b) = g^2 - h^2 ∧
             ¬ ∃ p q : ℝ, (a + b) * (-a - b) = p^2 - q^2 :=
by
  sorry

end square_of_binomial_conditions_l354_354613


namespace students_not_enrolled_in_biology_class_l354_354290

theorem students_not_enrolled_in_biology_class (total_students : ℕ) (percent_biology : ℕ) 
  (h1 : total_students = 880) (h2 : percent_biology = 35) : 
  total_students - (percent_biology * total_students / 100) = 572 := by
  sorry

end students_not_enrolled_in_biology_class_l354_354290


namespace point_motion_eq_l354_354254

theorem point_motion_eq (t : ℝ) (C : ℝ) :
    (∃ s : ℝ → ℝ, (∀ t, deriv s t = t^2 - 8 * t + 3) ∧ (s = λ t, (t^3) / 3 - 4 * t^2 + 3 * t + C)) :=
begin
  sorry
end

end point_motion_eq_l354_354254


namespace perimeter_triangle_ABC_eq_18_l354_354300

theorem perimeter_triangle_ABC_eq_18 (h1 : ∀ (Δ : ℕ), Δ = 9) 
(h2 : ∀ (p : ℕ), p = 6) : 
∀ (perimeter_ABC : ℕ), perimeter_ABC = 18 := by
sorry

end perimeter_triangle_ABC_eq_18_l354_354300


namespace cashback_percentage_l354_354551

theorem cashback_percentage
  (total_cost : ℝ) (rebate : ℝ) (final_cost : ℝ)
  (H1 : total_cost = 150) (H2 : rebate = 25) (H3 : final_cost = 110) :
  (total_cost - rebate - final_cost) / (total_cost - rebate) * 100 = 12 := by
  sorry

end cashback_percentage_l354_354551


namespace part1_max_f_part2_range_a_l354_354091

-- Define the functions as per the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

-- Part (1): Prove that the maximum value of f(x) for a = -1 is -1
theorem part1_max_f (x : ℝ) : 
  let f := f x (-1) 
  (∀ x > 0, f x ≤ -1 ∧ 
     (∃ x₀ > 0, f x₀ = -1)) := sorry

-- Define the given g(x) and h(x) functions in terms of f
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * f x a
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := 2 * a * x ^ 2 - (2 * a - 1) * x + a - 1

-- Part (2): Prove the range of a is [1, +∞) if g(x) ≤ h(x) for all x ≥ 1
theorem part2_range_a (a : ℝ) : 
  (∀ x ≥ 1, g x a ≤ h x a) → a ≥ 1 := sorry

end part1_max_f_part2_range_a_l354_354091


namespace locus_of_midpoint_l354_354851

open Locale Classical

variables (A B C H B' C' : Point)
variables (BB' CC' : Line)
variables (l : Line) [LineThrough H BB'] [LineThrough H CC'] 
variables (M N P Q : Point)

-- Define H as the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) (BB' CC' : Line) : Prop :=
⊥ BB' ∧ ⊥ CC' ∧ H = BB'.intersection CC'

-- Define BB' and CC' as altitudes of triangle ABC
def is_altitude (BB' : Line) (A B C : Point) : Prop :=
B ∈ BB' ∧ line_through BB' (A,C') ∧ C' ∈ BB'

def is_altitude (CC' : Line) (A B C : Point) : Prop :=
C ∈ CC' ∧ line_through CC' (A,B') ∧ B' ∈ CC'

-- Define the line l passing through H
def is_line_through (l : Line) (H : Point) : Prop :=
H ∈ l 

-- Define M and N as points where l intersects [BC'] and [CB']
def intersects (l : Line) (p1 p2 : Point) : Point :=
{ p : Point // p ∈ l ∧ p ∈ Line.segment p1 p2 }

def M := intersects l B C'
def N := intersects l C B'

-- Define P and Q as the feet of the perpendiculars from M and N to l
def is_perpendicular_foot (P : Point) (M : Point) (l : Line) (BB' : Line) : Prop :=
line_perpendicular P l ∧ P ∈ BB' ∧ P = BB'.intersection l

def is_perpendicular_foot (Q : Point) (N : Point) (l : Line) (CC' : Line) : Prop :=
line_perpendicular Q l ∧ Q ∈ CC' ∧ Q = CC'.intersection l

-- Define the midpoint of PQ
def midpoint (P Q : Point) : Point :=
(P + Q) / 2

-- Define the locus of midpoints of PQ
def loc_midpoints (P Q : Point) : Subset Point :=
λ (mid : Point), mid = midpoint P Q

-- Prove: the locus of midpoint of PQ is the line passing through H and parallel to BC
theorem locus_of_midpoint (H : Point) (A B C B' C' : Point) (BB' CC' : Line) (l : Line)
  [is_orthocenter H A B C BB' CC'] [is_altitude BB' A B C] [is_altitude CC' A B C] 
  [is_line_through l H] [is_perpendicular_foot P M l BB'] [is_perpendicular_foot Q N l CC'] :
  loc_midpoints P Q = LineThrough H (Parallel BC) :=
sorry

end locus_of_midpoint_l354_354851


namespace find_constant_l354_354652

-- Define the relationship between Fahrenheit and Celsius
def temp_rel (c f k : ℝ) : Prop :=
  f = (9 / 5) * c + k

-- Temperature increases
def temp_increase (c1 c2 f1 f2 : ℝ) : Prop :=
  (f2 - f1 = 30) ∧ (c2 - c1 = 16.666666666666668)

-- Freezing point condition
def freezing_point (f : ℝ) : Prop :=
  f = 32

-- Main theorem to prove
theorem find_constant (k : ℝ) :
  ∃ (c1 c2 f1 f2: ℝ), temp_rel c1 f1 k ∧ temp_rel c2 f2 k ∧ 
  temp_increase c1 c2 f1 f2 ∧ freezing_point f1 → k = 32 :=
by sorry

end find_constant_l354_354652


namespace symmetric_y_axis_sine_eq_l354_354810

theorem symmetric_y_axis_sine_eq (α β : ℝ) (h : ∃ p : ℝ × ℝ, p.1 = -p.1 ∧ (cos α, sin α) = p ∧ (cos β, sin β) = (-p.1, p.2)) : 
  sin α = sin β :=
sorry

end symmetric_y_axis_sine_eq_l354_354810


namespace oriented_segment_reciprocal_sum_zero_l354_354937

-- Define the necessary elements: Points, Lines, and the triangle
variables {Point Line : Type}
variables (A B C M A₁ B₁ C₁ : Point)
variables (BC CA AB : Line)

-- Conditions: M is the centroid of triangle ABC, and M is the intersection of medians
-- We'll use the Lean definition of centroid and relevant properties directly
axiom M_is_centroid : centroid A B C M

-- Conditional intersection points on the sides of the triangle
axiom intersection_BC : BC.contains A₁
axiom intersection_CA : CA.contains B₁
axiom intersection_AB : AB.contains C₁

-- Main statement to prove
theorem oriented_segment_reciprocal_sum_zero
    (hM : centroid A B C M)
    (hA₁ : BC.contains A₁)
    (hB₁ : CA.contains B₁)
    (hC₁ : AB.contains C₁) :
    (1 / (vector_length M A₁)) + (1 / (vector_length M B₁)) + (1 / (vector_length M C₁)) = 0 :=
sorry

end oriented_segment_reciprocal_sum_zero_l354_354937


namespace find_angle_ADB_l354_354536

-- Given conditions
variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables (D : A)
variables (E : B)
variables (F : C)
variables (D_perp_BC : E -> B -> C -> Prop)
variables (triangle_equilateral : A -> B -> C -> Prop)

-- Definitions relating to the conditions
noncomputable def midpoint (x y : A) := (x + y) / 2
noncomputable def angle {A B C: Type*} [metric_space A] [metric_space B] [metric_space C] (A : A) (B : B) (C : C) : ℝ := sorry

-- Main proof statement
theorem find_angle_ADB 
  (h1 : 2 * (dist D A) = dist D C) -- 2AD = DC
  (h2 : D_perp_BC D B C) -- E is foot of perpendicular from D to BC
  (h3 : ∃ F : A, F == BD ∩ AE) -- F is intersection of BD and AE
  (h4 : triangle_equilateral B E F) -- Triangle BEF is equilateral
  :
  angle A D B = 90 := 
sorry

end find_angle_ADB_l354_354536


namespace scheduling_ways_l354_354452

-- Define the conditions
constant Courses : Set String := {"algebra", "geometry", "number_theory", "calculus"}
constant Periods : Fin 8
constant algebra_in_first_period : Periods := 0
constant no_two_consecutive : Set (Fin 8 × Fin 8)

-- Define the problem statement
theorem scheduling_ways : 
  ∑ (arrangements : List (Fin 8)), 
    (arrangements.head = algebra_in_first_period ∧
     ∀ i j, (i, j) ∈ no_two_consecutive → abs(arrangements.nth i - arrangements.nth j) > 1) →
    List.card arrangements = 12
:=
sorry

end scheduling_ways_l354_354452


namespace find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5_l354_354698

theorem find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), (n > 0) ∧ (n % 15 = 0) ∧ (30 ≤ Real.sqrt n) ∧ (Real.sqrt n ≤ 30.5) ∧ (n = 900) :=
by {
  use 900,
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  sorry
}

end find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5_l354_354698


namespace capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l354_354768

-- Step 1: Define the capacities of type A and B cars
def typeACarCapacity := 3
def typeBCarCapacity := 4

-- Step 2: Prove transportation capacities x and y
theorem capacities_correct (x y: ℕ) (h1 : 3 * x + 2 * y = 17) (h2 : 2 * x + 3 * y = 18) :
    x = typeACarCapacity ∧ y = typeBCarCapacity :=
by
  sorry

-- Step 3: Define a rental plan to transport 35 tons
theorem rental_plan_exists (a b : ℕ) : 3 * a + 4 * b = 35 :=
by
  sorry

-- Step 4: Prove the minimal cost solution
def typeACarCost := 300
def typeBCarCost := 320

def rentalCost (a b : ℕ) : ℕ := a * typeACarCost + b * typeBCarCost

theorem minimal_rental_cost_exists :
    ∃ a b, 3 * a + 4 * b = 35 ∧ rentalCost a b = 2860 :=
by
  sorry

end capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l354_354768


namespace sum_of_first_5_terms_arith_seq_l354_354072

noncomputable def arith_seq_sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

variables {a : ℕ → ℤ} (h1 : a 0 = 1) 
          (h2 : 4 * a 0 = 4 * 1 + a 2)
          (h3 : ∀ n, a (n + 1) - a n = a 1 - a 0)

theorem sum_of_first_5_terms_arith_seq : arith_seq_sum_first_n_terms a 5 = 31 :=
by
  sorry

end sum_of_first_5_terms_arith_seq_l354_354072


namespace cos_B_unique_value_l354_354364

theorem cos_B_unique_value (B : ℝ) (h : tan B + sec B = 3) : cos B = 3/5 :=
by
  sorry

end cos_B_unique_value_l354_354364


namespace dopey_games_l354_354212

/-- Each dwarf played the following number of games on Monday:
- Grumpy: 1 game,
- Sneezy: 2 games,
- Sleepy: 3 games,
- Bashful: 4 games,
- Happy: 5 games,
- Doc: 6 games.
Prove that Dopey played 3 games on Monday. --/
theorem dopey_games :
  ∀ (dwarves : Type) (played_games : dwarves → ℕ),
  (played_games Grumpy = 1) ∧
  (played_games Sneezy = 2) ∧
  (played_games Sleepy = 3) ∧
  (played_games Bashful = 4) ∧
  (played_games Happy = 5) ∧
  (played_games Doc = 6) → 
  ∃ (Dopey : dwarves), played_games Dopey = 3 :=
by
  sorry

end dopey_games_l354_354212


namespace bernardo_prob_l354_354015

-- Define the sets from which Bernardo and Silvia pick their numbers
def set_b := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def set_s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Bernardo picks 3 distinct numbers from set_b:
def bernardo_comb := { p | p ∈ set_b ∧ fintype.card p = 3 }

-- Silvia picks 3 distinct numbers from set_s:
def silvia_comb := { q | q ∈ set_s ∧ fintype.card q = 3 }

-- Function to calculate the number formed by digits in descending order
def form_number (s : finset ℕ) : ℕ :=
  s.val.sort (≥)

-- Probability that Bernardo's number is larger than Silvia's
def prob_bernardo_larger := 
  let favorable := finset.card { (b, s) |
    b ∈ bernardo_comb ∧
    s ∈ silvia_comb ∧
    form_number b > form_number s } in
  let total := finset.card bernardo_comb * finset.card silvia_comb in
  (favorable : ℚ) / total

-- Main theorem statement
theorem bernardo_prob : 
  prob_bernardo_larger = 7 / 20 := sorry

end bernardo_prob_l354_354015


namespace after_lunch_typing_orders_l354_354827

theorem after_lunch_typing_orders :
  let S := {1, 2, 3, 4, 5, 6, 7}
  let k := S.size
  (∑ k in range 8, choose 7 k * (k + 2)) = 704 :=
by
  sorry

end after_lunch_typing_orders_l354_354827


namespace resistance_at_least_2000_l354_354415

variable (U : ℝ) (I : ℝ) (R : ℝ)

-- Given conditions:
def voltage := U = 220
def max_current := I ≤ 0.11

-- Ohm's law in this context
def ohms_law := I = U / R

-- Proof problem statement:
theorem resistance_at_least_2000 (voltage : U = 220) (max_current : I ≤ 0.11) (ohms_law : I = U / R) : R ≥ 2000 :=
sorry

end resistance_at_least_2000_l354_354415


namespace impossible_to_form_11x12x13_parallelepiped_l354_354606

def is_possible_to_form_parallelepiped
  (brick_shapes_form_unit_cubes : Prop)
  (dimensions : ℕ × ℕ × ℕ) : Prop :=
  ∃ bricks : ℕ, 
    (bricks * 4 = dimensions.fst * dimensions.snd * dimensions.snd.fst)

theorem impossible_to_form_11x12x13_parallelepiped 
  (dimensions := (11, 12, 13)) 
  (brick_shapes_form_unit_cubes : Prop) : 
  ¬ is_possible_to_form_parallelepiped brick_shapes_form_unit_cubes dimensions := 
sorry

end impossible_to_form_11x12x13_parallelepiped_l354_354606


namespace solve_for_x_l354_354461

noncomputable def x_solution (x : ℝ) : Prop := sqrt (3 / x + 3) = 4 / 3

theorem solve_for_x (x : ℝ) (h : x_solution x) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354461


namespace measuring_cup_size_l354_354875

-- Defining the conditions
def total_flour := 8
def flour_needed := 6
def scoops_removed := 8 

-- Defining the size of the cup
def cup_size (x : ℚ) := 8 - scoops_removed * x = flour_needed

-- Stating the theorem
theorem measuring_cup_size : ∃ x : ℚ, cup_size x ∧ x = 1 / 4 :=
by {
    sorry
}

end measuring_cup_size_l354_354875


namespace correct_answer_is_D_l354_354316

-- Define the students and their numbering in the problem
def num_students := 300
def students_first_grade := 120
def students_second_grade := 90
def students_third_grade := 90

-- Define the sample sequences
def sample1 := [7, 37, 67, 97, 127, 157, 187, 217, 247, 277]
def sample2 := [5, 9, 100, 107, 121, 180, 195, 221, 265, 299]
def sample3 := [11, 41, 71, 101, 131, 161, 191, 221, 251, 281]
def sample4 := [31, 61, 91, 121, 151, 181, 211, 241, 271, 300]

-- Define a function to check if a sample can be systematic sampling
def is_systematic (s : List Nat) : Bool := 
  (List.inits s).tail.all (λ ps, ps.pairwise (λ a b => b - a = 30))

-- Define a function to check if a sample can be stratified sampling (mocked for simplicity)
-- In reality, it will need more conditions but assuming all samples can be stratified sampling as per problem statement
def is_stratified (s : List Nat) : Bool := true

-- Now to state the problem
theorem correct_answer_is_D :
  (is_systematic sample2 = false ∧ is_systematic sample3 = false) ∧
  (is_stratified sample2 = true) ∧
  (is_systematic sample1 = true ∧ is_systematic sample4 = true) ∧
  (is_stratified sample1 = true ∧ is_stratified sample3 = true) :=
  sorry

end correct_answer_is_D_l354_354316


namespace tile_position_l354_354812

theorem tile_position (grid : fin 7 → fin 7 → Prop) (tiles_1x3 : fin 16 → list (fin 7 × fin 7))
  (tile_1x1 : fin 7 × fin 7) :
  (∃ (i j : fin 7), tile_1x1 = (i, j) ∧ (i = 3 ∧ j = 3 ∨ 
  i = 0 ∨ i = 6 ∨ j = 0 ∨ j = 6)) :=
sorry

end tile_position_l354_354812


namespace f_prime_even_l354_354093

def exp (x : ℝ) : ℝ := Real.exp x

def f (x : ℝ) : ℝ := exp x - exp (-x)

def f_prime (x : ℝ) : ℝ := exp x + exp (-x)

theorem f_prime_even (x : ℝ) : f_prime (-x) = f_prime x :=
by
  sorry

end f_prime_even_l354_354093


namespace zero_point_interval_l354_354917

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.logb 2 x

theorem zero_point_interval :
  (∀ x > 0, 6 / x - Real.logb 2 x = f x) →
  (∀ x > 0, StrictMonoDecrOn f {y | y > 0}) →
  (f 3 > 0) →
  (f 4 < 0) →
  ∃ c, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  intros h_domain h_mono h_f3 h_f4
  sorry  -- Proof is omitted.

end zero_point_interval_l354_354917


namespace min_possible_range_l354_354985

def scores : List (ℕ × ℕ) := [(17, 31), (28, 47), (35, 60), (45, 75), (52, 89)]

def min_score : ℕ := scores.map Prod.fst |>.minimum' (by decide)
def max_score : ℕ := scores.map Prod.snd |>.maximum' (by decide)

theorem min_possible_range : max_score - min_score = 72 :=
by
  sorry

end min_possible_range_l354_354985


namespace parabola_PQ_length_l354_354097

-- Given conditions
def parabola_eq (x y : ℝ) := y^2 = 2 * x
def focus := (1/2 : ℝ, 0 : ℝ)
def directrix (x : ℝ) := x = -1/2

-- Points P and Q on the parabola with their x-coordinates
variables (x1 y1 x2 y2 : ℝ)
axiom on_parabola_P : parabola_eq x1 y1
axiom on_parabola_Q : parabola_eq x2 y2
axiom x1_x2_sum : x1 + x2 = 3

theorem parabola_PQ_length : |x1 + x2 + 1 - x1 - x2| = 4 :=
by
  sorry

end parabola_PQ_length_l354_354097


namespace angle_bisector_ACB_perpendicular_to_CD_l354_354833

variables {A B C D : Type}
variables [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D]

-- Given conditions as variables and assumptions
variables (ABCD : convex_quadrilateral A B C D)
variables (angle_B_eq_C : ∠B = ∠C)
variables (angle_D_eq_90 : ∠D = 90)
variables (AB_2CD : |AB| = 2 * |CD|)

-- Theorem stating the conclusion
theorem angle_bisector_ACB_perpendicular_to_CD :
  angle_bisector (∠ ACB) ⟂ CD :=
sorry

end angle_bisector_ACB_perpendicular_to_CD_l354_354833


namespace exists_set_with_properties_l354_354365

theorem exists_set_with_properties : 
  ∃ (M : set (ℝ × ℝ × ℝ)) (hM_finite : set.finite M),
  (¬ ∃ (P : set (ℝ × ℝ × ℝ)), ∀ (p ∈ M), p ∈ P ∧ affine_span ℝ (set.to_finset P).val = ⊤) ∧
  (∀ (A B : ℝ × ℝ × ℝ), A ≠ B ∧ A ∈ M ∧ B ∈ M →
    ∃ (C D : ℝ × ℝ × ℝ), C ≠ D ∧ C ∈ M ∧ D ∈ M ∧
    ∃ (par_line : vector ℝ), (B.1 - A.1, B.2 - A.2, B.3 - A.3) ∥ par_line ∧ 
                              (D.1 - C.1, D.2 - C.2, D.3 - C.3) ∥ par_line) :=
sorry

end exists_set_with_properties_l354_354365


namespace max_value_arithmetic_sequence_l354_354400

theorem max_value_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S n = n * (a (n/2 + 1)) -- Assuming 1-based indexing and only even indices (simplied for illustration)
  (h_s17_pos : S 17 > 0)
  (h_s18_neg : S 18 < 0) :
  ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 15 ∧ ∀ k, 1 ≤ k ∧ k ≤ 15 → S n / a n ≥ S k / a k :=
begin
  use 9,
  split,
  { -- 1 ≤ 9 ≤ 15
    split, simp, linarith, },
  intros k hk,
  -- provide hints here if necessary
  sorry
end

end max_value_arithmetic_sequence_l354_354400


namespace unit_digit_of_expression_l354_354348
-- Lean 4 statement

theorem unit_digit_of_expression : 
    (∃ (n : ℕ), n = (∏ i in (Finset.range 6), (2^(2^i) + 1)) ∧ n % 10 = 5) := 
by 
  sorry

end unit_digit_of_expression_l354_354348


namespace intersection_A_B_l354_354763

def A : Set ℕ := { x | x ≤ 1 }
def B : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : A = ({0, 1} : Set ℕ) ∧ ↑0 ∈ A ∧ ↑1 ∈ A ∧ B = { x : ℤ | -1 ≤ x ∧ x ≤ 2 } ∧ (∀ x : ℤ, x ∈ A → x ∈ B → x = 0 ∨ x = 1) ↔ A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l354_354763


namespace pattern_is_composite_l354_354889

def is_digit_pattern (n : ℕ) (k : ℕ) : Prop :=
  n = (List.range (k + 1)).foldr (λ i acc, acc + 10^(2 * i)) 0

theorem pattern_is_composite (n k : ℕ) (hk : k ≥ 2) (h : is_digit_pattern n k) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q :=
by
  sorry

end pattern_is_composite_l354_354889


namespace seventh_term_ratio_l354_354857

noncomputable def seventh_term_ratio_proof (r f s g : ℚ) : Prop :=
  let R_n := λ n : ℕ, (n / 2) * (2 * r + (n - 1) * f)
  let U_n := λ n : ℕ, (n / 2) * (2 * s + (n - 1) * g)
  ∀ n : ℕ, R_n n / U_n n = (3 * n + 5 : ℚ) / (2 * n + 13) → (r + 6 * f) / (s + 6 * g) = (4 : ℚ) / 3

theorem seventh_term_ratio (r f s g : ℚ) (h : ∀ n, R_n n / U_n n = (3 * n + 5 : ℚ) / (2 * n + 13)) :
  (r + 6 * f) / (s + 6 * g) = (4 : ℚ) / 3 :=
begin
  sorry
end

end seventh_term_ratio_l354_354857


namespace total_students_correct_l354_354249

-- Given conditions
def number_of_buses : ℕ := 95
def number_of_seats_per_bus : ℕ := 118

-- Definition for the total number of students
def total_number_of_students : ℕ := number_of_buses * number_of_seats_per_bus

-- Problem statement
theorem total_students_correct :
  total_number_of_students = 11210 :=
by
  -- Proof is omitted, hence we use sorry.
  sorry

end total_students_correct_l354_354249


namespace octahedron_nonconsecutive_probability_l354_354576

theorem octahedron_nonconsecutive_probability :
  let faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let pairs := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1)]
  let total_configurations := 7!
  let valid_configurations := 60
  let m := 1
  let n := 84
  m + n = 85 :=
by {
  -- Total number of permutations of the numbers on the faces of the octahedron
  let total_permutations := factorial 7
  
  -- Valid configurations that meet the condition (calculated manually)
  let valid_count := 60
  
  -- The probability
  let probability := valid_count / total_permutations
  
  -- Simplify the fraction and check the values of m and n
  have h_prob : probability = 1 / 84 :=
    by sorry,
  
  -- Therefore, m = 1 and n = 84, thus m + n = 1 + 84 = 85
  exact (1 + 84)
}

end octahedron_nonconsecutive_probability_l354_354576


namespace tank_fill_time_l354_354002

theorem tank_fill_time (T : ℕ) (a_time b_time c_time d_time : ℕ) :
  a_time = 60 → b_time = 40 → c_time = 30 → d_time = 24 →
  let ra := 1 / a_time in
  let rb := 1 / b_time in
  let rc := 1 / c_time in
  let rd := 1 / d_time in
  let combined_rate1 := ra + rb in
  let combined_rate2 := rb + rc + rd in
  let combined_rate3 := ra + rc + rd in
  (T / 3) * combined_rate1 = 1/3 → 
  (T / 3) * combined_rate2 = 1/3 →
  (T / 3) * combined_rate3 = 1/3 →
  ((5 * T / 360) + (12 * T / 360) + (11 * T / 360)) = 1 →
  T = 13 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tank_fill_time_l354_354002


namespace negation_example_l354_354919

theorem negation_example (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 3 * x - 1 ≤ 0 :=
sorry

end negation_example_l354_354919


namespace find_percentage_l354_354996

theorem find_percentage (p : ℝ) (h : (p / 100) * 8 = 0.06) : p = 0.75 := 
by 
  sorry

end find_percentage_l354_354996


namespace find_x_l354_354530

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0)
  (coins_megan : ℤ := 42)
  (coins_shana : ℤ := 35)
  (shana_win : ℕ := 2)
  (total_megan : shana_win * x + (total_races - shana_win) * y = coins_shana)
  (total_shana : (total_races - shana_win) * x + shana_win * y = coins_megan) :
  x = 4 := by
  sorry

end find_x_l354_354530


namespace inclination_angle_of_tangent_at_point_l354_354224

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x + 4

theorem inclination_angle_of_tangent_at_point :
  let x := 1,
      y := 3,
      α := 45
  in
  curve 1 = 3 →
  (deriv curve 1) = 1 →
  Real.arctan 1 = Real.pi / 4 :=
by
  intro x y α h1 h2
  sorry

end inclination_angle_of_tangent_at_point_l354_354224


namespace convert_157_to_base_8_l354_354036

theorem convert_157_to_base_8 : base10_to_baseN 157 8 = [2, 3, 5] :=
sorry

end convert_157_to_base_8_l354_354036


namespace hexagon_diagonals_concurrent_l354_354517

theorem hexagon_diagonals_concurrent
  (A B C D E F : Type)
  [convex_hexagon ABCDEF : convex ABCDEF]
  (h1 : dist A B = dist D E)
  (h2 : dist B C = dist E F)
  (h3 : dist C D = dist F A)
  (h4 : (\angle A - \angle D = \angle C - \angle F) ∧ (\angle C - \angle F = \angle E - \angle B)) :
  concurrent AD BE CF :=
sorry

end hexagon_diagonals_concurrent_l354_354517


namespace college_application_distributions_l354_354210

theorem college_application_distributions : 
  let total_students := 6
  let colleges := 3
  ∃ n : ℕ, n = 540 ∧ 
    (n = (colleges^total_students - colleges * (2^total_students) + 
      (colleges.choose 2) * 1)) := sorry

end college_application_distributions_l354_354210


namespace number_of_zero_points_l354_354575

def f (x : ℝ) : ℝ :=
if x >= 1 then 1 + log x / log 5 else 2 * x - 1

theorem number_of_zero_points :
  {x : ℝ | f x = 0}.to_finset.card = 1 :=
sorry

end number_of_zero_points_l354_354575


namespace parallel_vectors_eq_zero_l354_354442

theorem parallel_vectors_eq_zero (m : ℝ) : 
  let a := (-2 : ℝ, 3) in
  let b := (1 : ℝ, m - 3 / 2) in
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → m = 0 :=
by
  intros
  sorry

end parallel_vectors_eq_zero_l354_354442


namespace intersection_points_curve_circle_l354_354492

theorem intersection_points_curve_circle :
  let z : ℂ := complex.exp (complex.I * θ)
  in let curve := abs (z - 1 / z) = 1
  in let circle := abs z = 1
  in  ∃ θ_vals : set ℝ, (circle -> curve) ∧ θ_vals = {π/6, 5*π/6, 7*π/6, 11*π/6} ∧ θ_vals.card = 4.
sorry

end intersection_points_curve_circle_l354_354492


namespace average_increase_by_19_point_8_l354_354532

theorem average_increase_by_19_point_8 (num_list : List ℕ) (N : ℕ) 
  (original_number reversed_number : ℕ) (diff : ℕ) :
  num_list.length = 10 → 
  original_number < 1000 →
  reversed_number < 1000 →
  diff = reversed_number - original_number →
  diff = 198 →
  N = (reversed_number - original_number) / 10 →
  N = 19.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_increase_by_19_point_8_l354_354532


namespace manufacturing_employees_percentage_l354_354556

theorem manufacturing_employees_percentage 
  (total_circle_deg : ℝ := 360) 
  (manufacturing_deg : ℝ := 18) 
  (sector_proportion : ∀ x y, x / y = (x/y : ℝ)) 
  (percentage : ∀ x, x * 100 = (x * 100 : ℝ)) :
  (manufacturing_deg / total_circle_deg) * 100 = 5 := 
by sorry

end manufacturing_employees_percentage_l354_354556


namespace fortieth_sequence_number_l354_354667

theorem fortieth_sequence_number :
  (∃ r n : ℕ, ((r * (r + 1)) - 40 = n) ∧ (40 ≤ r * (r + 1)) ∧ (40 > (r - 1) * r) ∧ n = 2 * r) :=
sorry

end fortieth_sequence_number_l354_354667


namespace part1_solution_set_part2_range_of_a_l354_354033

-- Define the function f
def f (a x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Problem 1: When a = 1, solve the inequality f(x) ≤ 5
theorem part1_solution_set : 
  { x : ℝ | f 1 x ≤ 5 } = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 4} := 
  by 
  sorry

-- Problem 2: Determine the range of a for which f(x) has a minimum
theorem part2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x < 1/3 → f a x ≤ f a 1/3) → 
           (∀ x : ℝ, x ≥ 1/3 → f a x ≥ f a 1/3) ↔ 
           (-3 ≤ a ∧ a ≤ 3) := 
  by
  sorry

end part1_solution_set_part2_range_of_a_l354_354033


namespace tetrahedron_minimum_distance_height_inequality_l354_354170

theorem tetrahedron_minimum_distance_height_inequality 
  (d h : ℝ) 
  (h_min_dist : ∀ (A B C D : ℝ), d = min_dist_between_opposite_edges A B C D)
  (h_shortest_height : ∀ (A B C D : ℝ), h = shortest_height A B C D) :
  2 * d > h :=
sorry

end tetrahedron_minimum_distance_height_inequality_l354_354170


namespace remainder_of_3_pow_17_mod_7_l354_354278

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7_l354_354278


namespace sarah_remaining_problems_time_l354_354897

theorem sarah_remaining_problems_time:
  let initial_math_problems := 35
  let initial_science_problems := 25
  let saturday_math_done := 15
  let saturday_science_done := 10
  let saturday_math_time := 2 -- hours
  let saturday_science_time := 1.5  -- hours
  let remaining_math_problems := initial_math_problems - saturday_math_done
  let remaining_science_problems := initial_science_problems - saturday_science_done
  let total_remaining_problems := remaining_math_problems + remaining_science_problems
  let pages_left := 5
  let problems_per_page := total_remaining_problems / pages_left
  let average_time_per_problem := ((saturday_math_time / saturday_math_done) + (saturday_science_time / saturday_science_done)) / 2
  let total_remaining_time := total_remaining_problems * average_time_per_problem
  total_remaining_time ≈ 4.96 ∧ problems_per_page > 0 :=
by
  sorry

end sarah_remaining_problems_time_l354_354897


namespace harmonic_difference_l354_354080

open BigOperators

-- Define the summation for harmonic series
def harmonic_sum (n : ℕ) : ℚ := ∑ y in Finset.range (n + 1) \ {0}, (1 / y : ℚ)

-- Given problem conditions translated to Lean 4
-- Express the sum from 3 to 10 for both sequences and the difference
def sum1 := ∑ y in Finset.range' 3 11, (1 / (y - 2) : ℚ)
def sum2 := ∑ y in Finset.range' 3 11, (1 / (y - 1) : ℚ)

-- Main theorem to be proven
theorem harmonic_difference :
  sum1 - sum2 = 8 / 9 := sorry

end harmonic_difference_l354_354080


namespace percent_volume_removed_l354_354311

-- Define the dimensions of the original box
def length := 20
def width := 12
def height := 10

-- Define the side length of the cubes removed from each corner
def cube_side := 4

-- Define the volume of the original box
def volume_box := length * width * height

-- Define the volume of one cube removed
def volume_cube := cube_side * cube_side * cube_side

-- Define the number of cubes removed
def num_cubes := 8

-- Define the total volume of the cubes removed
def volume_removed := num_cubes * volume_cube

-- Define the percentage of the volume removed
def percentage_removed := (volume_removed * 100) / volume_box

-- Theorem: The percentage of the original volume removed is 21.333%
theorem percent_volume_removed :
  percentage_removed = 21.333 := by
  sorry

end percent_volume_removed_l354_354311


namespace bf_distance_l354_354640

open Set

-- Definitions based on conditions in a)
def parabola : Set (ℝ × ℝ) := {p | (p.snd)^2 = 4 * p.fst}
def focus : ℝ × ℝ := (1, 0)
def directrix : Set (ℝ × ℝ) := {p | p.fst = -1}
def pointA : ℝ × ℝ
def pointB : ℝ × ℝ
def pointC : ℝ × ℝ

def lineL (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | ∃ a b c : ℝ, a * q.fst + b * q.snd + c = 0 ∧ a * p1.fst + b * p1.snd + c = 0 ∧ a * p2.fst + b * p2.snd + c = 0}

axiom is_on_parabola_A : pointA ∈ parabola
axiom is_on_parabola_B : pointB ∈ parabola
axiom is_on_directrix_C : pointC ∈ directrix
axiom pointA_and_C_same_side_x_axis : pointA.snd * pointC.snd > 0
axiom distance_AC_twice_AF : dist pointA pointC = 2 * dist pointA focus

theorem bf_distance : dist pointB focus = 4 :=
by sorry

end bf_distance_l354_354640


namespace completion_count_l354_354630

def isLatinSquare (n : Nat) (M : Matrix (Fin n) (Fin n) Nat) : Prop :=
  ∀ i : Fin n, ∀ k : Fin n, ∃ j1 j2 : Fin n, 
    (M i j1 = k ∧ ∀ (j : Fin n), j ≠ j1 → M i j ≠ k) ∧
    (M j2 i = k ∧ ∀ (j : Fin n), j ≠ j2 → M j i ≠ k)

def partiallyFilled4x4 : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![![some 1, none, none, none],
    ![none, none, none, none],
    ![none, some 3, none, none],
    ![none, none, none, none]]

noncomputable def countCompletions (partial : Matrix (Fin 4) (Fin 4) (Option Nat)) : Nat := sorry

theorem completion_count : countCompletions partiallyFilled4x4 = 48 :=
sorry

end completion_count_l354_354630


namespace at_least_two_acute_angles_l354_354949

theorem at_least_two_acute_angles (T : Type) [triangle T] :
  (∀ t : T, ∃ (a b c : angle t), a + b + c = 180 ∧ (a < 90 ∧ b < 90) ∨ (a < 90 ∧ c < 90) ∨ (b < 90 ∧ c < 90)) := 
sorry

end at_least_two_acute_angles_l354_354949


namespace solution_set_for_f_lt_0_l354_354737

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x - 1 else -x - 1

theorem solution_set_for_f_lt_0 :
  {x : ℝ | f x < 0} = set.Ioo (-1) 1 :=
begin
  unfold f,
  ext,
  split; intros hx,
  sorry,
  sorry
end

end solution_set_for_f_lt_0_l354_354737


namespace floor_sq_sum_eq_four_l354_354211

noncomputable def regions (x y : ℝ) : Prop :=
(x >= 2 ∧ x < 3 ∧ y >= 0 ∧ y < 1) ∨ 
(x >= -2 ∧ x < -1 ∧ y >= 0 ∧ y < 1) ∨ 
(x >= 0 ∧ x < 1 ∧ y >= 2 ∧ y < 3) ∨ 
(x >= 0 ∧ x < 1 ∧ y >= -2 ∧ y < -1)

theorem floor_sq_sum_eq_four (x y : ℝ) (hx : x.floor^2 + y.floor^2 = 4) : 
  regions x y :=
sorry

end floor_sq_sum_eq_four_l354_354211


namespace delta_five_three_l354_354040

def Δ (a b : ℕ) : ℕ := 4 * a - 6 * b

theorem delta_five_three :
  Δ 5 3 = 2 := by
  sorry

end delta_five_three_l354_354040


namespace eval_expression_l354_354048

theorem eval_expression : 3 ^ 4 - 4 * 3 ^ 3 + 6 * 3 ^ 2 - 4 * 3 + 1 = 16 := 
by 
  sorry

end eval_expression_l354_354048


namespace isabella_score_sixth_test_l354_354847

noncomputable def isabella_scores_valid (scores : List ℕ) : Prop :=
  scores.length = 7 ∧
  (∀ s, s ∈ scores → 92 ≤ s ∧ s ≤ 101) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 7 → (List.take n scores).sum / n = (List.take n scores).sum / n) ∧
  scores.nth 6 = some 96

theorem isabella_score_sixth_test (scores : List ℕ) (h : isabella_scores_valid scores):
  scores.nth 5 = some 101 :=
by
  sorry

end isabella_score_sixth_test_l354_354847


namespace cart_total_distance_l354_354994

-- Definitions for the conditions
def first_section_distance := (15/2) * (8 + (8 + 14 * 10))
def second_section_distance := (15/2) * (148 + (148 + 14 * 6))

-- Combining both distances
def total_distance := first_section_distance + second_section_distance

-- Statement to be proved
theorem cart_total_distance:
  total_distance = 4020 :=
by
  sorry

end cart_total_distance_l354_354994


namespace min_a_sq_plus_b_sq_l354_354121
open Real

noncomputable def f (x a b : ℝ) := exp x + a * (x - 1) + b
noncomputable def g (t : ℝ) := exp (2 * t) / ((t - 1) ^ 2 + 1)

theorem min_a_sq_plus_b_sq : 
  (∃ t ∈ Icc (1/2) 1, f t a b = 0) → (∃ (a b : ℝ), a^2 + b^2 = 4 * exp(1) / 5) :=
sorry

end min_a_sq_plus_b_sq_l354_354121


namespace trigonometric_function_analysis_l354_354545

/-- Given the function f(x) = cos(2 * x - π / 3) + cos(2 * x + π / 6), prove the following statements:
1. The maximum value of f(x) is √2.
2. The smallest positive period of f(x) is π.
3. f(x) is a decreasing function in the interval (π / 24, 13 * π / 24).
4. After shifting the graph of y = √2 * cos(2 * x) to the right by π / 24 units, it will coincide with the graph of the function.
--/
theorem trigonometric_function_analysis (x : ℝ) : 
  (∃ f : ℝ → ℝ, f(x) = cos(2 * x - π / 3) + cos(2 * x + π / 6) ∧ 
  (∀ x, f(x) ≤ sqrt 2 ∧ 
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ 
  ∀ x, (π / 24 < x ∧ x < 13 * π / 24) → f'(x) < 0 ∧ 
  ∀ x, (let g(x) = sqrt 2 * cos(2 * (x - π / 24)) in g(x) = f(x))
sorry

end trigonometric_function_analysis_l354_354545


namespace sequence_properties_l354_354762

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n, a (n + 1) = a n + 2

theorem sequence_properties {a : ℕ → ℤ} (h : arithmetic_sequence a) :
  a 2 + a 4 = 6 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end sequence_properties_l354_354762


namespace value_of_f_at_neg1_l354_354947

def f (x : ℤ) : ℤ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

theorem value_of_f_at_neg1 : f (-1) = 6 :=
by
  sorry

end value_of_f_at_neg1_l354_354947


namespace num_3_digit_div_by_5_l354_354783

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l354_354783


namespace fibonacci_expression_l354_354757

variables {F : ℕ → ℕ}

def fibonacci_property (n : ℕ) : Prop :=
  (λ n, F (n + 1) * F (n - 1) - F n ^ 2 = (-1) ^ n)

def fibonacci_identity (M : ℕ → Matrix (Fin 2) (Fin 2) ℕ) : Prop :=
  ∀ n, M n =
    (λ n, Matrix.mul_vec
      (Matrix.vec_mul (Matrix.vec_cons (Matrix.vec_cons 1 1) (Matrix.vec_cons 1 0)) (Matrix.vec_cons (Matrix.vec_cons (F (n + 1)) (F n)) (Matrix.vec_cons (F n) (F (n - 1)))))
    )

noncomputable def compute_expression (F : ℕ → ℕ) (k : ℤ) : ℤ :=
  (F 785 + k) * (F 787 + k) - (F 786 + k) * (F 786 + k)

theorem fibonacci_expression (F : ℕ → ℕ) (k : ℤ) (F_prop : ∀ n, fibonacci_property F n) (M_prop : fibonacci_identity F) :
compute_expression F k = -1 :=
by
  sorry

end fibonacci_expression_l354_354757


namespace zuminglish_mod_1000_l354_354811

def zuminglish_sequences (n : ℕ) : ℕ × ℕ × ℕ :=
  if n = 2 then (4, 2, 2)
  else
    let (a, b, c) := zuminglish_sequences (n - 1)
    (2 * (a + c), a, 2 * b)

def valid_zuminglish_words (n : ℕ) : ℕ :=
  let (a_n, b_n, c_n) := zuminglish_sequences n
  a_n + b_n + c_n

theorem zuminglish_mod_1000 : valid_zuminglish_words 7 % 1000 = 912 :=
  sorry

end zuminglish_mod_1000_l354_354811


namespace correct_statements_true_l354_354417

theorem correct_statements_true :
  (∀ x : ℝ, x ≥ 1 → (let y := (x - 2)/(2*x + 1) in y ∈ [-1/3, 1/2)) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, 2*x - 1 ∈ [-1, 1] → f (2*x -1) ∈ ℝ) →
    let domain_y := {x | 1 < x ∧ x ≤ 2} in ∀ x ∈ domain_y, f (x - 1) / sqrt (x - 1) ∈ ℝ ) ∧
  (∃! f, ∀ x, (x ∈ {-2, 2}) → f x = x^2) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, x ≥ 1 → f (x + 1/x) = x^2 + 1/x^2) →
    let m := sqrt 6 in f m = 4) :=
begin
  sorry
end

end correct_statements_true_l354_354417


namespace conjugate_of_z_is_1_plus_I_find_a_and_b_l354_354069

/-- Define complex number z -/
def z : ℂ := (3 - I) / (2 + I)

/-- Define the conjugate of z -/
def z_conjugate : ℂ := conj z

/-- 1. Prove that the conjugate of z is 1 + I -/
theorem conjugate_of_z_is_1_plus_I : z_conjugate = 1 + I := sorry

/-- Define real variables a and b -/
variable (a b : ℝ)

/-- Define function f(z) -/
def f (z : ℂ) : ℂ := z^2 + a * z + b

/-- Condition: f(z) = conj(z) -/
theorem find_a_and_b (z : ℂ) (hz : z = 1 - I) (hc : f z = conj z) : a = -3 ∧ b = 4 :=
sorry

end conjugate_of_z_is_1_plus_I_find_a_and_b_l354_354069


namespace exists_good_word_l354_354475

open Classical

variable (α : Type) [Fintype α] [DecidableEq α]

def is_allowed_word (nonallowed_words : List (List α)) (word : List α) : Prop :=
  ¬∃ (bad_word : List α), bad_word ∈ nonallowed_words ∧ bad_word <:+ word

theorem exists_good_word (alphabet : Finset α) (nonallowed_words : List (List α))
  (distinct_lengths : Nonempty α) :
  ∀ n : ℕ, ∃ word : List α, word.length = n ∧ is_allowed_word nonallowed_words word :=
by sorry

end exists_good_word_l354_354475


namespace area_of_triangle_BEC_l354_354153

-- Given conditions definitions
namespace Geometry

variables {Point : Type} [ordered_comm_group Point]

structure Trapezoid (A B C D E : Point) extends Quadrilateral A B C D where
  AD_perp_DC : perpendicular AD DC
  AD_length : distance A D = 4
  AB_length : distance A B = 5
  DC_length : distance D C = 10
  DE_on_DC : on_line_segment D E C
  DE_length : distance D E = 6
  BE_parallel_AD : parallel B E A D

-- Define points and the area of a triangle
def area_BEC {A B C D E : Point} [trapezoid : Trapezoid A B C D E] : ℝ :=
  let BE_length := trapezoid.AD_length in
  let EC_length := trapezoid.DC_length - trapezoid.DE_length in
  (1/2) * BE_length * EC_length

-- Theorem statement
theorem area_of_triangle_BEC (A B C D E : Point) [trapezoid : Trapezoid A B C D E] : 
  area_BEC = 8 :=
by
  sorry

end Geometry

end area_of_triangle_BEC_l354_354153


namespace milk_remaining_l354_354591

def initial_whole_milk := 15
def initial_low_fat_milk := 12
def initial_almond_milk := 8

def jason_buys := 5
def jason_promotion := 2 -- every 2 bottles he gets 1 free

def harry_buys_low_fat := 4
def harry_gets_free_low_fat := 1
def harry_buys_almond := 2

theorem milk_remaining : 
  (initial_whole_milk - jason_buys = 10) ∧ 
  (initial_low_fat_milk - (harry_buys_low_fat + harry_gets_free_low_fat) = 7) ∧ 
  (initial_almond_milk - harry_buys_almond = 6) :=
by
  sorry

end milk_remaining_l354_354591


namespace total_lucky_tickets_even_sum_lucky_tickets_divisible_by_999_l354_354019

def isLucky (n : ℕ) : Prop :=
  let digits := (List.range 6).map (fun i => (n / 10^i) % 10)
  (digits.take 3).sum = (digits.drop 3).sum

theorem total_lucky_tickets_even : (Oper.length filter isLucky (List.range 1000000)) % 2 = 0 :=
sorry

theorem sum_lucky_tickets_divisible_by_999 : (List.range 1000000 |>.filter isLucky |>.sum) % 999 = 0 :=
sorry

end total_lucky_tickets_even_sum_lucky_tickets_divisible_by_999_l354_354019


namespace simplify_power_of_power_l354_354901

variable (x : ℝ)

theorem simplify_power_of_power : (3 * x^4)^4 = 81 * x^16 := 
by 
sorry

end simplify_power_of_power_l354_354901


namespace problem_solution_l354_354252

noncomputable def sequence_x : ℕ → ℤ
| 0     := 11
| (n+1) := 3 * sequence_x n + 2 * sequence_y n

noncomputable def sequence_y : ℕ → ℤ
| 0     := 7
| (n+1) := 4 * sequence_x n + 3 * sequence_y n

theorem problem_solution : 
  (sequence_y 1854)^(2018) - 2 * (sequence_x 1854)^(2018) % 2018 = 1825 := 
sorry

end problem_solution_l354_354252


namespace unique_red_number_under_2014_l354_354046

def is_red (N : ℕ) : Prop :=
  let divisors := (List.range (N + 1)).filter (λ d => N % d = 0)
  divisors.length = 8 ∧ divisors[4] = 3 * divisors[2] - 4

theorem unique_red_number_under_2014 :
  ∃! N < 2014, is_red N ∧ N = 621 :=
by
  sorry

end unique_red_number_under_2014_l354_354046


namespace line_AB_not_through_specific_point_l354_354733

-- Definitions of the given circles and the point M
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define M is a point on Circle C
def on_circle_C (a b : ℝ) : Prop := circle_C a b

-- Define the equation of Circle M with center at point M and radius |OM|
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the equation of line AB
def line_AB (a b : ℝ) : (ℝ × ℝ) → Prop := λ p, (2 * a - 2) * p.1 + 2 * b * p.2 - 3 = 0

-- Given points to check
def points_to_check : list (ℝ × ℝ) := [
  (3/4, 1/2),
  (3/2, 1),
  (1/2, 1/2),
  (1/2, real.sqrt 2 / 2)
]

-- Theorem stating the line AB does not pass through a particular point
theorem line_AB_not_through_specific_point (a b : ℝ)
  (hM_on_C : on_circle_C a b) :
  ¬ line_AB a b (1/2, 1/2) :=
sorry

end line_AB_not_through_specific_point_l354_354733


namespace parts_per_hour_equality_l354_354200

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l354_354200


namespace find_f_2011_l354_354405

-- Definitions of given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_of_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Main theorem to be proven
theorem find_f_2011 (f : ℝ → ℝ) 
  (hf_even: is_even_function f) 
  (hf_periodic: is_periodic_of_period f 4) 
  (hf_at_1: f 1 = 1) : 
  f 2011 = 1 := 
by 
  sorry

end find_f_2011_l354_354405


namespace temperature_decrease_l354_354136

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end temperature_decrease_l354_354136


namespace basketball_team_wins_l354_354631

theorem basketball_team_wins (wins_first_60 : ℕ) (remaining_games : ℕ) (total_games : ℕ) (target_win_percentage : ℚ) (winning_games : ℕ) : 
  wins_first_60 = 45 → remaining_games = 40 → total_games = 100 → target_win_percentage = 0.75 → 
  winning_games = 30 := by
  intros h1 h2 h3 h4
  sorry

end basketball_team_wins_l354_354631


namespace sum_of_solutions_l354_354957

-- Define the conditions in Lean
def is_solution (x : ℕ) : Prop := 7 * (5 * x - 3) % 10 = 35 % 10
def within_range (x : ℕ) : Prop := x > 0 ∧ x <= 30
def valid_solution (x : ℕ) : Prop := is_solution x ∧ within_range x

-- Define the main theorem
theorem sum_of_solutions : ∑ x in (Finset.filter valid_solution (Finset.range 31)), x = 60 :=
by sorry

end sum_of_solutions_l354_354957


namespace only_f1_is_odd_l354_354089

-- Define the four functions
def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := abs x
def f3 (x : ℝ) : ℝ := log x
def f4 (x : ℝ) : ℝ := x^3 + 1

-- Define a predicate for odd functions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Prove that f1 is the only odd function among the four
theorem only_f1_is_odd :
  is_odd_function f1 ∧
  ¬ (is_odd_function f2) ∧
  ¬ (is_odd_function f3) ∧
  ¬ (is_odd_function f4) :=
by
  -- Proof omitted
  sorry

end only_f1_is_odd_l354_354089


namespace manufacturer_cost_price_l354_354322

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end manufacturer_cost_price_l354_354322


namespace geese_problem_l354_354005

theorem geese_problem 
  (G : ℕ)  -- Total number of geese in the original V formation
  (T : ℕ)  -- Number of geese that flew up from the trees to join the new V formation
  (h1 : G / 2 + T = 12)  -- Final number of geese flying in the V formation was 12 
  (h2 : T = G / 2)  -- Number of geese that flew out from the trees is the same as the number of geese that landed initially
: T = 6 := 
sorry

end geese_problem_l354_354005


namespace divisible_by_two_of_square_l354_354798

theorem divisible_by_two_of_square {a : ℤ} (h : 2 ∣ a^2) : 2 ∣ a :=
sorry

end divisible_by_two_of_square_l354_354798


namespace cos_alpha_l354_354726

noncomputable def prism := sorry -- Define the prism with necessary attributes omitted for brevity

variable (prism : Type)
variables (a d BK : ℝ) (B C K : prism) -- Define relevant points and values in the prism

-- Assumptions
axiom BK_eq_3_CK : BK = 3 * (distance C K)
axiom distance_CK : d = (1/3) * (distance B K)

-- Proof goal
theorem cos_alpha : cos α = sqrt (2 / 3) := 
sorry

end cos_alpha_l354_354726


namespace inconsistent_proportion_l354_354403

theorem inconsistent_proportion (a b : ℝ) (h1 : 3 * a = 5 * b) (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (a / b = 3 / 5) :=
sorry

end inconsistent_proportion_l354_354403


namespace sequence_periodicity_l354_354250

-- Define the initial conditions: p is a prime number with exactly 300 non-zero digits.
def is_prime_with_300_digits (p : ℕ) : Prop :=
  Nat.Prime p ∧ (p / 10^299 > 0) ∧ (p / 10^300 = 0)

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ → ℕ
| 0     := p
| n + 1 := 2 * (a (n) / 2)

-- Theorem statement to prove that a_{2003} = 2p given the conditions
theorem sequence_periodicity (p : ℕ) (h1 : is_prime_with_300_digits p) : a p 2003 = 2 * p :=
by
  sorry

end sequence_periodicity_l354_354250


namespace cis_sum_angle_eq_100_l354_354672

theorem cis_sum_angle_eq_100 :
  ∃ r > 0, ∃ θ, (θ = 100 ∧ 0 ≤ θ ∧ θ < 360) ∧ 
  (Complex.exp (Complex.I * Real.pi / 180 * 55) + 
   Complex.exp (Complex.I * Real.pi / 180 * 65) + 
   Complex.exp (Complex.I * Real.pi / 180 * 75) + 
   Complex.exp (Complex.I * Real.pi / 180 * 85) + 
   Complex.exp (Complex.I * Real.pi / 180 * 95) + 
   Complex.exp (Complex.I * Real.pi / 180 * 105) +
   Complex.exp (Complex.I * Real.pi / 180 * 115) + 
   Complex.exp (Complex.I * Real.pi / 180 * 125) + 
   Complex.exp (Complex.I * Real.pi / 180 * 135) + 
   Complex.exp (Complex.I * Real.pi / 180 * 145)
  = r * Complex.exp (Complex.I * Real.pi / 180 * θ)) := 
sorry

end cis_sum_angle_eq_100_l354_354672


namespace max_area_of_rectangular_garden_l354_354329

noncomputable def max_rectangle_area (x y : ℝ) (h1 : 2 * (x + y) = 36) (h2 : x > 0) (h3 : y > 0) : ℝ :=
  x * y

theorem max_area_of_rectangular_garden
  (x y : ℝ)
  (h1 : 2 * (x + y) = 36)
  (h2 : x > 0)
  (h3 : y > 0) :
  max_rectangle_area x y h1 h2 h3 = 81 :=
sorry

end max_area_of_rectangular_garden_l354_354329


namespace card_probability_l354_354262

theorem card_probability :
  let S := {-2, 0, 1, 2, 3}
  let cond1 (a : ℤ) := a < 2
  let cond2 (a : ℤ) := a ≥ -1 / 3
  let valid_a := {a ∈ S | cond1 a ∧ cond2 a ∧ a ≠ 0}
  valid_a.card / S.card = 1 / 5 :=
by
  sorry

end card_probability_l354_354262


namespace min_sum_of_fractions_is_43_over_72_l354_354165

noncomputable def digits := {2, 3, 4, 5, 6, 7, 8, 9}
def distinct_digits (A B C D : ℕ) : Prop := 
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem min_sum_of_fractions_is_43_over_72 (A B C D : ℕ) 
  (h_distinct : distinct_digits A B C D) :
  (A : ℚ) / B + (C : ℚ) / D = 43 / 72 :=
sorry

end min_sum_of_fractions_is_43_over_72_l354_354165


namespace cube_root_of_3x_add_4y_l354_354732

def cube_root (z : ℝ) : ℝ :=
  z ^ (1 / 3 : ℝ)

theorem cube_root_of_3x_add_4y (x y : ℝ) 
  (h1 : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 3) 
  (h2 : x = 5) 
  (h3 : y = 3) : 
  cube_root (3 * x + 4 * y) = 3 := 
by 
  sorry

end cube_root_of_3x_add_4y_l354_354732


namespace grasshopper_trap_condition_l354_354315

theorem grasshopper_trap_condition (N : ℕ) : 
  (∃ k : ℕ, N = 2^k - 1) ↔ 
  (∀ (g_pos : ℤ), (g_pos = 0 ∨ g_pos = -(N + 1)) → 
  (∃ num : ℕ, ∃ direction : g_pos → g_pos, 
    g_pos + num * direction g_pos = 0 ∨ g_pos + num * direction g_pos = -(N + 1))) :=
sorry

end grasshopper_trap_condition_l354_354315


namespace right_triangle_sum_of_squares_l354_354139

theorem right_triangle_sum_of_squares (A B C : Type) [InnerProductSpace ℝ A] [MetricSpace (B)] [MetricSpace (C)] 
    (h1: is_right_triangle A B C) (h2: dist B C = 5) :
    dist A B ^ 2 + dist A C ^ 2 + dist B C ^ 2 = 50 := by
  sorry

end right_triangle_sum_of_squares_l354_354139


namespace even_sums_of_two_cubes_l354_354775

theorem even_sums_of_two_cubes :
  let valid_cube_sums := {n | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000 ∧ n % 2 = 0}
  in valid_cube_sums.card = 25 :=
by sorry

end even_sums_of_two_cubes_l354_354775


namespace identity_1_identity_2_l354_354802

noncomputable def r : ℝ := sorry

noncomputable def a_6  : ℝ := r
noncomputable def a_3  : ℝ := r * Real.sqrt 3
noncomputable def a_4  : ℝ := r * Real.sqrt 2
noncomputable def a_10 : ℝ := r * (Real.sqrt 5 - 1) / 2
noncomputable def a_5  : ℝ := r / 2 * Real.sqrt (10 - 2 * Real.sqrt 5)

-- Now, we assert the identities provided in the problem.

theorem identity_1 : a_6^2 + a_4^2 = a_3^2 := by
  sorry

theorem identity_2 : a_6^2 + a_{10}^2 = a_{5}^2 := by
  sorry

end identity_1_identity_2_l354_354802


namespace complex_number_fourth_quadrant_l354_354067

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_fourth_quadrant (z : ℂ) (h : z + z * complex.I = 3 + 2 * complex.I) :
  is_in_fourth_quadrant z :=
sorry

end complex_number_fourth_quadrant_l354_354067


namespace initial_amount_invested_l354_354385

noncomputable def initial_investment (r : ℝ) (t : ℕ) (A : ℝ) : ℝ :=
  A / ((1 + r) ^ t)

theorem initial_amount_invested :
  initial_investment 0.08 5 500 ≈ 340.28 :=
by
  sorry

end initial_amount_invested_l354_354385


namespace medians_similar_l354_354727

theorem medians_similar {A B C : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (a b : ℝ) (a_gt_b : a > b) : 
  let leg1 := 2 * a,
      leg2 := 2 * b,
      hypotenuse := real.sqrt ((2 * a) ^ 2 + (2 * b) ^ 2),
      s_c := hypotenuse / 2,
      s_a := real.sqrt (4 * b ^ 2 + a ^ 2),
      s_b := real.sqrt (a ^ 2 + b ^ 2),
      right_triangle_med := s_c ^ 2 = s_a ^ 2 + s_b ^ 2 in
  triangle_similar (2 * a) (2 * b) hypotenuse s_a s_b s_c :=
begin
  sorry
end

end medians_similar_l354_354727


namespace total_cash_correct_l354_354908

def stock1_price : ℝ := 120.50
def stock1_brokerage_percent : ℝ := (1/4)/100

def stock2_price : ℝ := 210.75
def stock2_brokerage_percent : ℝ := 0.5/100

def stock3_price : ℝ := 310.25
def stock3_brokerage_percent : ℝ := 0.75/100

def brokerage (price brokerage_percent : ℝ) : ℝ := price * brokerage_percent
def cash_realized (price brokerage_percent : ℝ) : ℝ := price - brokerage price brokerage_percent

def total_cash_realized : ℝ := 
  cash_realized stock1_price stock1_brokerage_percent + 
  cash_realized stock2_price stock2_brokerage_percent + 
  cash_realized stock3_price stock3_brokerage_percent

theorem total_cash_correct : total_cash_realized = 637.818125 := by
  sorry

end total_cash_correct_l354_354908


namespace painters_needed_days_l354_354383

-- Let P be the total work required in painter-work-days
def total_painter_work_days : ℕ := 5

-- Let E be the effective number of workers with advanced tools
def effective_workers : ℕ := 4

-- Define the number of days, we need to prove this equals 1.25
def days_to_complete_work (P E : ℕ) : ℚ := P / E

-- The main theorem to prove: for total_painter_work_days and effective_workers, the days to complete the work is 1.25
theorem painters_needed_days :
  days_to_complete_work total_painter_work_days effective_workers = 5 / 4 :=
by
  sorry

end painters_needed_days_l354_354383


namespace tangent_line_at_2_l354_354393

noncomputable def f (x : ℝ) : ℝ := x * Real.log (x - 1)

theorem tangent_line_at_2 : 
  let f' (x : ℝ) : ℝ := Real.log (x - 1) + x / (x - 1)
  f' 2 = 2 ∧ f 2 = 0 → ∃ m b : ℝ, (∀ x : ℝ, (m = 2 ∧ b = -4) ∧ (f 2 = 0 ∧ f' 2 = 2 → ∀ x, y = m * x + b)) :=
begin
  intro h,
  use 2,
  use -4,
  sorry
end

end tangent_line_at_2_l354_354393


namespace equidistant_x_coord_on_x_axis_l354_354951

def distance (P Q : ℝ × ℝ) : ℝ := (Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2))

def P_on_x_axis (x : ℝ) : ℝ × ℝ := (x, 0)

def A := (-3, 0)
def B := (2, 5)

theorem equidistant_x_coord_on_x_axis (x : ℝ) : distance (P_on_x_axis x) A = distance (P_on_x_axis x) B → x = 2 :=
by
  sorry

end equidistant_x_coord_on_x_axis_l354_354951


namespace cartesian_equation_of_curve_length_AB_l354_354760

-- Define the parametric equations of line l
def line_param (t: ℝ): ℝ × ℝ := (1 + (1/2) * t, t + 1)

-- Define the polar equation of curve C
def polar_eq (ρ θ: ℝ): ℝ := ρ * sin(θ) ^ 2 - 4 * cos(θ)

-- Define the Cartesian equation of curve C derived from the polar equation
def cartesian_eq (x y: ℝ): Prop := y^2 = 4 * x

-- Define the intersection points and the distance between them
def intersection_points (t: ℝ): ℝ × ℝ := (1 + (1/2) * t, t + 1)

def distance (p1 p2: ℝ × ℝ): ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  sqrt((x2 - x1)^2 + (y2 - y1)^2)

-- Prove that the Cartesian equation of curve C is y^2 = 4x
theorem cartesian_equation_of_curve : ∀ (ρ θ: ℝ), polar_eq ρ θ = 0 → ∃ x y: ℝ, cartesian_eq x y :=
sorry

-- Prove that the length of |AB| is √15
theorem length_AB : ∀ t1 t2: ℝ, cartesian_eq (fst (line_param t1)) (snd (line_param t1)) ∧ cartesian_eq (fst (line_param t2)) (snd (line_param t2)) →
  distance (line_param t1) (line_param t2) = sqrt 15 :=
sorry

end cartesian_equation_of_curve_length_AB_l354_354760


namespace smallest_m_l354_354177

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def existsPerfectSquareSum {X : Finset ℕ} (W : Finset ℕ) : Prop :=
  ∃ u v ∈ W, isPerfectSquare (u + v)

theorem smallest_m (X : Finset ℕ) (hX : X = Finset.range 2002 \ {0}) :
  ∃ (m : ℕ), (∀ W ⊆ X, W.card = m → existsPerfectSquareSum W) ∧
  (∀ k, (∀ W ⊆ X, W.card = k → existsPerfectSquareSum W) → k ≥ m) :=
begin
  use 1000,
  split,
  {
    intros W hW h_card,
    sorry
  },
  {
    intros k h_k,
    by_contradiction,
    sorry
  }
end

end smallest_m_l354_354177


namespace probability_sum_even_for_three_cubes_l354_354936

-- Define the probability function
def probability_even_sum (n: ℕ) : ℚ :=
  if n > 0 then 1 / 2 else 0

theorem probability_sum_even_for_three_cubes : probability_even_sum 3 = 1 / 2 :=
by
  sorry

end probability_sum_even_for_three_cubes_l354_354936


namespace symmetric_complex_product_l354_354910

-- Definition and condition
def is_symmetric_about_y_eq_x (z1 z2 : ℂ) : Prop :=
  ∃ a b : ℝ, z1 = a + b * complex.I ∧ z2 = b + a * complex.I

theorem symmetric_complex_product (z1 z2 : ℂ) (h_symm : is_symmetric_about_y_eq_x z1 z2) (h_neq : z1 ≠ 3 + 2 * complex.I) :
  z1 * z2 = 13 * complex.I :=
by
  sorry

end symmetric_complex_product_l354_354910


namespace find_initial_cars_l354_354992

-- Definitions based on given conditions
def initial_cars (x : ℕ) := x
def initial_silver_cars (x : ℕ) := 0.10 * (x : ℝ)
def new_shipment := 80
def silver_in_new_shipment := 0.75 * 80
def total_silver_cars (x : ℕ) := initial_silver_cars x + silver_in_new_shipment
def total_cars (x : ℕ) := x + new_shipment
def percentage_of_silver_cars (x : ℕ) := 0.20 * (total_cars x : ℝ)

-- The proof statement
theorem find_initial_cars (x : ℕ) (h : total_silver_cars x = percentage_of_silver_cars x) : x = 440 :=
by
  sorry

end find_initial_cars_l354_354992


namespace pasha_wins_9_games_l354_354197

theorem pasha_wins_9_games :
  ∃ w l : ℕ, (w + l = 12) ∧ (2^w * (2^l - 1) - (2^l - 1) * 2^(w - 1) = 2023) ∧ (w = 9) :=
by
  sorry

end pasha_wins_9_games_l354_354197


namespace number_of_consecutive_pages_is_188_l354_354988

def is_consecutive_page (n : Nat) : Prop :=
  ∃ (a b : Nat), a < b ∧ n = (b - a + 1) * (a + b) / 2

theorem number_of_consecutive_pages_is_188 :
  (Finset.range 2020).filter is_consecutive_page).card = 188 := 
  sorry

end number_of_consecutive_pages_is_188_l354_354988


namespace symmetric_circle_eq_l354_354226

def center_of_circle (h k : ℝ) (r : ℝ) : Type :=
  { center_x : ℝ // center_x = h } × { center_y : ℝ // center_y = k }

def equation_of_circle (h k : ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r

theorem symmetric_circle_eq (
    h k r : ℝ) :
  center_of_circle (-h) (k) (sqrt r) →
  center_of_circle (h) (-k) (sqrt r) :=
sorry

end symmetric_circle_eq_l354_354226


namespace sine_graph_shift_l354_354939

theorem sine_graph_shift :
  ∀ (x : ℝ), sin (2 * x + π / 3) = sin (2 * (x + π / 6)) :=
by
  intro x
  sorry

end sine_graph_shift_l354_354939


namespace area_union_DEF_D_l354_354496

-- Definitions of the problem's conditions
variable {D E F H D' E' F' : Type}

-- Given conditions
axiom DE_eq_8 : distance D E = 8
axiom EF_eq_17 : distance E F = 17
axiom DF_eq_15 : distance D F = 15
axiom H_is_centroid : is_centroid H D E F
axiom rotate_180 : rotate_180_about H DEF = D'E'F'

-- Objective: Prove the area of the union of triangles DEF and D'E'F' is 60
theorem area_union_DEF_D'E'F' : 
  area (union (triangle D E F) (triangle D' E' F')) = 60 :=
sorry

end area_union_DEF_D_l354_354496


namespace total_height_increase_l354_354232

def height_increase_per_decade : ℕ := 90
def decades_in_two_centuries : ℕ := (2 * 100) / 10

theorem total_height_increase :
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end total_height_increase_l354_354232


namespace sum_of_divisors_prime_count_l354_354171

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum id

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_prime_sum_divisors : ℕ :=
  (Finset.range 26).filter (λ n, is_prime (sum_of_divisors n)).card

theorem sum_of_divisors_prime_count :
  count_prime_sum_divisors = 5 :=
by
  continuity_except_eq_okay_exfinset_uname_inhed_sorry_have
  end sorry

end sum_of_divisors_prime_count_l354_354171


namespace value_of_n_l354_354801

theorem value_of_n (n : ℕ) : (1 / 5 : ℝ) ^ n * (1 / 4 : ℝ) ^ 18 = 1 / (2 * (10 : ℝ) ^ 35) → n = 35 :=
by
  intro h
  sorry

end value_of_n_l354_354801


namespace find_pointC_l354_354730

-- Define the conditions from the problem
def pointA : ℝ := 2
def pointB : ℝ := pointA - 5 + 2
def pointC : ℝ := pointA + 4

-- Given the number represented by point B is -1
def given_condition : pointB = -1 := sorry

-- The theorem to prove
theorem find_pointC : pointC = 6 :=
by
  -- Reference the given condition
  have h := given_condition
  -- Unrolling the definitions and calculations
  unfold pointA pointB pointC at h ⊢
  linarith

end find_pointC_l354_354730


namespace bus_speed_is_48_l354_354711

theorem bus_speed_is_48 (v1 v2 : ℝ) (h1 : ∀ (t : ℝ), v1 * t + v2 * t = 8)
  (h2 : ∀ (d AB : ℝ), d(C, t(AB v1)) ∧ d(C, t(AB v2)))
  (h3 : ∀ (t : ℝ), d(2/3) = lim t(AB v1) - 0)
  (h4 : dAB = 4) :
  v2 = 48 :=
  sorry

end bus_speed_is_48_l354_354711


namespace shaded_region_area_is_correct_l354_354144

-- Define the basic elements of the problem
def radius : ℝ := 2
def rectangle_area : ℝ := 8
def semicircle_area (r : ℝ) := (1 / 2) * Mathlib.pi * r^2
def shaded_area (rect_area : ℝ) (semicircle_area : ℝ) := rect_area - semicircle_area

-- The main statement: proving the shaded area is 8 - 2 * π
theorem shaded_region_area_is_correct :
  shaded_area rectangle_area (semicircle_area radius) = 8 - 2 * Mathlib.pi := by
  sorry

end shaded_region_area_is_correct_l354_354144


namespace area_of_quadrilateral_EFGH_l354_354834

noncomputable def trapezium_ABCD_midpoints_area : ℝ :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (4, 3)
  let D := (0, 3)
  let E := ((B.1 + C.1)/2, (B.2 + C.2)/2) -- midpoint of BC
  let F := ((C.1 + D.1)/2, (C.2 + D.2)/2) -- midpoint of CD
  let G := ((A.1 + D.1)/2, (A.2 + D.2)/2) -- midpoint of AD
  let H := ((G.1 + E.1)/2, (G.2 + E.2)/2) -- midpoint of GE
  let area := (E.1 * F.2 + F.1 * G.2 + G.1 * H.2 + H.1 * E.2 - F.1 * E.2 - G.1 * F.2 - H.1 * G.2 - E.1 * H.2) / 2
  abs area

theorem area_of_quadrilateral_EFGH : trapezium_ABCD_midpoints_area = 0.75 := by
  sorry

end area_of_quadrilateral_EFGH_l354_354834


namespace students_present_in_class_l354_354826

def num_students (D B : ℕ) : Prop := 
  D / (D + B) = 0.6 ∧ (D - 1) / ((D - 1) + (B - 2)) = 0.625

theorem students_present_in_class : ∃ (D B : ℕ), num_students D B ∧ D = 21 ∧ B = 14 := by
  sorry

end students_present_in_class_l354_354826


namespace parallelogram_ratio_l354_354236

variables (A B C D E F G : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G]

-- Definition of a Parallelogram
def is_parallelogram (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] : Prop :=
  dist A B = dist C D ∧ dist A D = dist B C ∧ segment A C ∩ segment B D ≠ ∅

-- Function to measure lengths AE, AF, and AG
noncomputable def measure_length (p1 p2 : Type) [metric_space p1] [metric_space p2] : ℝ := dist p1 p2

-- Given conditions of line l intersecting the sides and diagonal
def line_intersections (A B C D E F G : Type) [metric_space A] [metric_space B] [metric_space C] 
                      [metric_space D] [metric_space E] [metric_space F] [metric_space G] : Prop :=
  E ∈ segment A B ∧ F ∈ segment A D ∧ G ∈ segment A C

-- Statement of the proof we need to show in Lean
theorem parallelogram_ratio (A B C D E F G : Type) [metric_space A] [metric_space B] [metric_space C] 
                            [metric_space D] [metric_space E] [metric_space F] [metric_space G]
                            (h_parallelogram: is_parallelogram A B C D)
                            (h_intersections: line_intersections A B C D E F G) :
  measure_length A B / measure_length A E + measure_length A D / measure_length A F = 
  measure_length A C / measure_length A G :=
sorry

end parallelogram_ratio_l354_354236


namespace max_value_of_t_l354_354860

theorem max_value_of_t (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t, t = min x (y / (x^2 + y^2)) ∧ t ≤ sqrt 2 / 2 :=
sorry

end max_value_of_t_l354_354860


namespace paint_area_is_correct_l354_354896

-- Define the dimensions of the wall, window, and door
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def door_height : ℕ := 1
def door_length : ℕ := 7

-- Calculate area
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def door_area : ℕ := door_height * door_length

-- Calculate area to be painted
def area_to_be_painted : ℕ := wall_area - window_area - door_area

-- The theorem statement
theorem paint_area_is_correct : area_to_be_painted = 128 := 
by
  -- The proof would go here (omitted)
  sorry

end paint_area_is_correct_l354_354896


namespace min_value_of_a_l354_354216

noncomputable def smallest_possible_a (a b c: ℚ) (h1: a > 0) (h2: a + b + c ∈ ℚ) (vertex: (ℚ × ℚ)) : ℚ :=
  if vertex = (1 / 3, -2 / 3) then
    if h2 then
      ∃ (n: ℚ), (4 * a) = 9 * n + 6 ∧ (-2 / 3 < n) ∧ (a = (9 * n + 6) / 4) ∧ a = 3 / 8
    else
      0
  else
    0

theorem min_value_of_a : smallest_possible_a a b c h1 h2 (1 / 3, -2 / 3) = 3 / 8 :=
  sorry

end min_value_of_a_l354_354216


namespace probability_sales_greater_than_10000_l354_354486

/-- Define the probability that the sales of new energy vehicles in a randomly selected city are greater than 10000 -/
theorem probability_sales_greater_than_10000 :
  (1 / 2) * (2 / 10) + (1 / 2) * (6 / 10) = 2 / 5 :=
by sorry

end probability_sales_greater_than_10000_l354_354486


namespace correct_statements_true_l354_354418

theorem correct_statements_true :
  (∀ x : ℝ, x ≥ 1 → (let y := (x - 2)/(2*x + 1) in y ∈ [-1/3, 1/2)) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, 2*x - 1 ∈ [-1, 1] → f (2*x -1) ∈ ℝ) →
    let domain_y := {x | 1 < x ∧ x ≤ 2} in ∀ x ∈ domain_y, f (x - 1) / sqrt (x - 1) ∈ ℝ ) ∧
  (∃! f, ∀ x, (x ∈ {-2, 2}) → f x = x^2) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, x ≥ 1 → f (x + 1/x) = x^2 + 1/x^2) →
    let m := sqrt 6 in f m = 4) :=
begin
  sorry
end

end correct_statements_true_l354_354418


namespace exterior_angle_measure_l354_354825

-- Condition 1: In a triangle, two of the interior angles measure 70° and 40°
def interior_angle1 : ℝ := 70
def interior_angle2 : ℝ := 40

-- Condition 2: One side of the triangle is extended to form an exterior angle adjacent to the 70° angle
def adjacent_to_angle1 : Prop := True -- this is just to indicate that there is an exterior angle adjacent to the 70° angle.

-- Question: Prove that the exterior angle adjacent to the 70° angle is 110°
theorem exterior_angle_measure :
  let exterior_angle := 180 - interior_angle1 in
  exterior_angle = 110 :=
by
  sorry

end exterior_angle_measure_l354_354825


namespace units_digit_147_25_50_l354_354960

theorem units_digit_147_25_50 :
  let units_digit (n : ℕ) : ℕ := n % 10 in
  units_digit ((147 ^ 25) ^ 50) = 9 :=
by
  let units_digit : ℕ → ℕ := λ n, n % 10
  have h1 : units_digit 147 = 7 := by rfl
  have pattern : List ℕ := [7, 9, 3, 1] 
  have pattern_length : pattern.length = 4 := by rfl
  have h2 : ∀ n, units_digit (7^n) = pattern.get ((n - 1) % 4) := sorry
  have rem_25 : 25 % 4 = 1 := by rfl
  have rem_50 : 50 % 4 = 2 := by rfl
  have h3 : units_digit ((147^25)^50) = units_digit (7^50) := by
    rw [←nat.pow_mod (147 ^ 25) 10, nat.pow_mod 7 10, nat_mod_mod_to_pos _ 4]
    simp only [units_digit]
  rw [h2 50, List.get?]
  simp only [pattern, rem_50]
  rfl

end units_digit_147_25_50_l354_354960


namespace find_c_198_l354_354693

theorem find_c_198 (a b c : ℕ) (h1 : c = ((a + b * complex.I)^3).re) (h2 : (a + b * complex.I)^3.im = 107) (h3 : (a - b * complex.I = b * complex.I → h1 : c = a a^3 - 3 * a * b^2 +c = a^3 -3*a*b^2)
: c = 198 :=
sorry

end find_c_198_l354_354693


namespace probability_of_parabolas_intersect_l354_354270

theorem probability_of_parabolas_intersect (a b h k : ℕ) (ha : 1 ≤ a ∧ a ≤ 6)
                                            (hb : 1 ≤ b ∧ b ≤ 6) (hh : 1 ≤ h ∧ h ≤ 8)
                                            (hk : 1 ≤ k ∧ k ≤ 8) : 
  (h^2 - b + k = 0 → a = -2 * h) ∨ ((a + 2 * h ≠ 0) → true) →
  (a, b) = (1,1) ∨ (a, b) = (1,2) ∨ (a, b) = (1,3) ∨ (a, b) = (1,4) ∨ (a, b) = (1,5) ∨ (a, b) = (1,6) ∨
  (a, b) = (2,1) ∨ (a, b) = (2,2) ∨ (a, b) = (2,3) ∨ (a, b) = (2,4) ∨ (a, b) = (2,5) ∨ (a, b) = (2,6) ∨
  (a, b) = (3,1) ∨ (a, b) = (3,2) ∨ (a, b) = (3,3) ∨ (a, b) = (3,4) ∨ (a, b) = (3,5) ∨ (a, b) = (3,6) ∨
  (a, b) = (4,1) ∨ (a, b) = (4,2) ∨ (a, b) = (4,3) ∨ (a, b) = (4,4) ∨ (a, b) = (4,5) ∨ (a, b) = (4,6) ∨
  (a, b) = (5,1) ∨ (a, b) = (5,2) ∨ (a, b) = (5,3) ∨ (a, b) = (5,4) ∨ (a, b) = (5,5) ∨ (a, b) = (5,6) ∨
  (a, b) = (6,1) ∨ (a, b) = (6,2) ∨ (a, b) = (6,3) ∨ (a, b) = (6,4)∨ (a, b) = (6,5) ∨ (a, b) = (6,6) →
  observe 15 / 16 :=
by
sorry

end probability_of_parabolas_intersect_l354_354270


namespace cost_comparison_l354_354998

def cost_function_A (x : ℕ) : ℕ := 450 * x + 1000
def cost_function_B (x : ℕ) : ℕ := 500 * x

theorem cost_comparison (x : ℕ) : 
  if x = 20 then cost_function_A x = cost_function_B x 
  else if x < 20 then cost_function_A x > cost_function_B x 
  else cost_function_A x < cost_function_B x :=
sorry

end cost_comparison_l354_354998


namespace number_of_pieces_in_each_small_load_l354_354299

theorem number_of_pieces_in_each_small_load 
  (total_pieces : ℕ)
  (first_load : ℕ)
  (small_loads : ℕ)
  (remaining_pieces : ℕ)
  (pieces_per_load : ℕ) :
  total_pieces = 59 →
  first_load = 32 →
  small_loads = 9 →
  remaining_pieces = total_pieces - first_load →
  pieces_per_load = remaining_pieces / small_loads →
  pieces_per_load = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  simp at h4
  rw h4 at h5
  simp at h5
  exact h5

end number_of_pieces_in_each_small_load_l354_354299


namespace solve_for_x_l354_354458

theorem solve_for_x (x : ℚ) (h : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354458


namespace quadratic_transformation_l354_354924

theorem quadratic_transformation
    (a b c : ℝ)
    (h : ℝ)
    (cond : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    (∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = 20 * (x - h)^2 + 80) → h = 5 :=
by
  sorry

end quadratic_transformation_l354_354924


namespace find_bracket_403_n_l354_354011

-- Define the sequence of our problem
def seq (n : ℕ) : ℕ := 2 * n + 1

-- Define a helper function to sum the sequence of bracket sizes
def bracket_size_sum (m : ℕ) : ℕ :=
  let group_size := 10 * (m / 4)
  let remainder := m % 4
  group_size + match remainder with
               | 0 => 0
               | 1 => 1
               | 2 => 3
               | 3 => 6
               | _ => 0

theorem find_bracket_403_n (m : ℕ) : seq 2013 ∈ (bracket_size_sum 403) := by
  sorry

end find_bracket_403_n_l354_354011


namespace area_of_portion_of_circle_l354_354274

open Real

def circle_eq (x y : ℝ) : Prop := x^2 - 16 * x + y^2 - 8 * y = 32
def line_eq (x y : ℝ) : Prop := y = 2 * x - 20

theorem area_of_portion_of_circle :
  let r := 4 * Real.sqrt 5,
      total_area := pi * r^2,
      area_of_interest := total_area / 4
  in circle_eq x y → line_eq x y → x < (y + 20) / 2 → y < 0 → area_of_interest = 20 * pi :=
sorry

end area_of_portion_of_circle_l354_354274


namespace solve_for_m_l354_354524

def f (x : ℝ) : ℝ := if x >= 2 then x^2 - 1 else log x / log 2

theorem solve_for_m (m : ℝ) (h : f(m) = 3) : m = 2 :=
sorry

end solve_for_m_l354_354524


namespace points_per_game_without_bonus_l354_354188

-- Definition of the conditions
def b : ℕ := 82
def n : ℕ := 79
def P : ℕ := 15089

-- Theorem statement
theorem points_per_game_without_bonus :
  (P - b * n) / n = 109 :=
by
  -- Proof will be filled in here
  sorry

end points_per_game_without_bonus_l354_354188


namespace coefficient_x_squared_l354_354073

variable {a w c d : ℝ}

/-- The coefficient of x^2 in the expanded form of the equation (ax + w)(cx + d) = 6x^2 + x - 12 -/
theorem coefficient_x_squared (h1 : (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
                             (h2 : abs a + abs w + abs c + abs d = 12) :
  a * c = 6 :=
  sorry

end coefficient_x_squared_l354_354073


namespace parallelogram_area_and_existence_l354_354678

-- Define the points in space
def A : ℝ × ℝ × ℝ := (2, -5, 3)
def B : ℝ × ℝ × ℝ := (4, -9, 6)
def C : ℝ × ℝ × ℝ := (1, -4, 1)
def D : ℝ × ℝ × ℝ := (3, -8, 4)

-- Define vectors for the sides of the parallelogram
def AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def CD := (D.1 - C.1, D.2 - C.2, D.3 - C.3)
def CA := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Cross product of vectors AB and CA
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  (v.1 * v.1 + v.2 * v.2 + v.3 * v.3).sqrt

-- Theorem statement
theorem parallelogram_area_and_existence :
  AB = CD ∧ magnitude (cross_product AB CA) = Real.sqrt 110 :=
by
  sorry

end parallelogram_area_and_existence_l354_354678


namespace find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5_l354_354697

theorem find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), (n > 0) ∧ (n % 15 = 0) ∧ (30 ≤ Real.sqrt n) ∧ (Real.sqrt n ≤ 30.5) ∧ (n = 900) :=
by {
  use 900,
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  sorry
}

end find_positive_integer_divisible_by_15_and_sqrt_between_30_and_30_5_l354_354697


namespace find_chosen_number_l354_354651

-- Define the conditions
def condition (x : ℝ) : Prop := (3 / 2) * x + 53.4 = -78.9

-- State the theorem
theorem find_chosen_number : ∃ x : ℝ, condition x ∧ x = -88.2 :=
sorry

end find_chosen_number_l354_354651


namespace num_3_digit_div_by_5_l354_354785

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l354_354785


namespace mole_can_sustain_l354_354134

noncomputable def mole_winter_sustainability : Prop :=
  ∃ (grain millet : ℕ), 
    grain = 8 ∧ 
    millet = 0 ∧ 
    ∀ (month : ℕ), 1 ≤ month ∧ month ≤ 3 → 
      ((grain ≥ 3 ∧ (grain - 3) + millet <= 12) ∨ 
      (grain ≥ 1 ∧ millet ≥ 3 ∧ (grain - 1) + (millet - 3) <= 12)) ∧
      ((∃ grain_exchanged millet_gained : ℕ, 
         grain_exchanged ≤ grain ∧
         millet_gained = 2 * grain_exchanged ∧
         grain - grain_exchanged + millet_gained <= 12 ∧
         grain = grain - grain_exchanged) → 
      (grain = 0 ∧ millet = 0))

theorem mole_can_sustain : mole_winter_sustainability := 
sorry 

end mole_can_sustain_l354_354134


namespace probability_not_both_red_l354_354387

theorem probability_not_both_red (r w : ℕ) (draws : ℕ) (balls : ℕ) (total_choices red_choices : ℕ) :
    r = 3 → w = 2 → draws = 2 → balls = r + w →
    total_choices = Nat.choose balls draws →
    red_choices = Nat.choose r draws →
    (1 - (red_choices / total_choices : ℚ) = (7 / 10 : ℚ)) :=
by
  intros hr hw hdraws hballs htotal hred
  rw [hr, hw, hdraws, hballs, htotal, hred]
  -- Further steps would involve evaluating Nat.choose 5 2 and Nat.choose 3 2,
  -- and then doing the necessary calculations
  sorry

end probability_not_both_red_l354_354387


namespace partition_integers_to_perfect_square_triples_l354_354684

theorem partition_integers_to_perfect_square_triples :
  ∃ (P : ℤ → ℤ → ℤ → Prop), (∀ a b c, P a b c → a + b + c = 0) ∧
  (∀ a b c, P a b c → is_perfect_square (|a^3 * b + b^3 * c + c^3 * a|)) :=
sorry

-- Define a predicate that checks whether a number is a perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

end partition_integers_to_perfect_square_triples_l354_354684


namespace maximumPhoenixNumber_l354_354120

def isPhoenixNumber (M : ℕ) : Prop :=
  M / 1000 ∈ (2 : ℕ) .. 9 ∧
  M / 100 % 10 ∈ (2 : ℕ) .. 9 ∧
  M / 10 % 10 ∈ (2 : ℕ) .. 9 ∧
  M % 10 ∈ (2 : ℕ) .. 9 ∧ 
  let a := M / 1000 in
  let b := M / 100 % 10 in
  let c := M / 10 % 10 in
  let d := M % 10 in
  b - a = 2 * (d - c) ∧
  2 ≤ a ∧ a ≤ b ∧ b < c ∧ c ≤ d ∧ d ≤ 9

noncomputable def G (M : ℕ) : ℕ :=
  let a := M / 1000 in
  let b := M / 100 % 10 in
  let c := M / 10 % 10 in
  let d := M % 10 in
  (49 * a * c - 2 * a + 2 * d + 23 * b - 6) / 24

theorem maximumPhoenixNumber : ∃ M, isPhoenixNumber M ∧ G M ∈ ℤ ∧ M = 6699 := 
by
  sorry

end maximumPhoenixNumber_l354_354120


namespace inscribed_circle_radius_l354_354277

noncomputable def s (DE DF EF : ℝ) := (DE + DF + EF) / 2

noncomputable def area (s DE DF EF : ℝ) := real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

theorem inscribed_circle_radius {DE DF EF : ℝ} (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) :
  let s := s DE DF EF in
  let K := area s DE DF EF in
  let r := K / s in
  r = 2 * real.sqrt 14 :=
by
  sorry

end inscribed_circle_radius_l354_354277


namespace evaluate_f_at_2_plus_0_evaluate_f_at_2_minus_0_l354_354518

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 1 else 2*x + 1

theorem evaluate_f_at_2_plus_0 :
  f 2 = 5 :=
by
  sorry

theorem evaluate_f_at_2_minus_0 :
  f 2 = -1 :=
by
  sorry

end evaluate_f_at_2_plus_0_evaluate_f_at_2_minus_0_l354_354518


namespace log_problem_l354_354353

open Real

noncomputable def lg (x : ℝ) := log x / log 10

theorem log_problem :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 5 = 1 :=
by
  sorry

end log_problem_l354_354353


namespace sum_x_coords_Q3_eq_Q1_l354_354629

theorem sum_x_coords_Q3_eq_Q1 (x : Fin 23 → ℝ) : (∑ i, x i) = 46 → 
  (∑ i, (x i + x ((i + 1) % 23)) / 2) = 46 → 
  (∑ i, ((x i + x ((i + 1) % 23)) / 2 + (x ((i + 1) % 23) + x ((i + 2) % 23)) / 2) / 2) = 46 := 
by
  intros hQ1 hQ2
  -- skip the proof
  exact hQ2 -- This step uses the symmetry of the problem across iterations

end sum_x_coords_Q3_eq_Q1_l354_354629


namespace sum_of_complex_numbers_l354_354523

open Complex

theorem sum_of_complex_numbers :
  (∀ z : ℂ, |z| = 1 → ∃ x : ℝ, z * x^2 + 2 * conj(z) * x + 2 = 0) →
  ∑ z in { z : ℂ | |z| = 1 ∧ (∃ x : ℝ, z * x^2 + 2 * conj(z) * x + 2 = 0) }, z = -3/2 := 
by
  sorry

end sum_of_complex_numbers_l354_354523


namespace find_f_12_16_l354_354117

noncomputable def f : ℕ+ × ℕ+ → ℕ+
| (x, y) := sorry

-- Define the conditions
axiom cond1 : ∀ (x : ℕ+), f (x, x) = x
axiom cond2 : ∀ (x y : ℕ+), f (x, y) = f (y, x)
axiom cond3 : ∀ (x y : ℕ+), (x + y) * f (x, y) = y * f (x, x + y)

-- The theorem to prove
theorem find_f_12_16 (hf : ∀ x y : ℕ+, f(x, y) = f(y, x)) :
  f (⟨12, nat.pos_of_ne_zero (by norm_num)⟩, ⟨16, nat.pos_of_ne_zero (by norm_num)⟩) = ⟨48, nat.pos_of_ne_zero (by norm_num)⟩ :=
sorry

end find_f_12_16_l354_354117


namespace lcm_5_711_is_3555_l354_354279

theorem lcm_5_711_is_3555 : Nat.lcm 5 711 = 3555 := by
  sorry

end lcm_5_711_is_3555_l354_354279


namespace floor_sum_l354_354687

theorem floor_sum :
  (Int.floor 19.7) + (Int.floor (-19.7)) = -1 :=
by 
  sorry

end floor_sum_l354_354687


namespace ramesh_transport_cost_l354_354543

noncomputable def calculate_transport_cost (LP : ℝ) (discounted_price : ℝ) (install_cost : ℝ) (selling_price_needed : ℝ) : ℝ :=
  let SP := 1.10 * LP in
  selling_price_needed - SP - install_cost

theorem ramesh_transport_cost :
  let LP := 16500 / 0.80 in
  calculate_transport_cost LP 16500 250 23100 = 162.5 :=
by
  sorry

end ramesh_transport_cost_l354_354543


namespace stationery_sales_l354_354555

theorem stationery_sales :
  let pen_percentage : ℕ := 42
  let pencil_percentage : ℕ := 27
  let total_sales_percentage : ℕ := 100
  total_sales_percentage - (pen_percentage + pencil_percentage) = 31 :=
by
  sorry

end stationery_sales_l354_354555


namespace regular_tetrahedron_ineq_l354_354163

variables (A B C D M N : ℝ)

def is_regular_tetrahedron (A B C D : ℝ) : Prop := sorry

theorem regular_tetrahedron_ineq (h : is_regular_tetrahedron A B C D) :
  (AM * AN + BM * BN + CM * CN) ≥ (DM * DN) :=
sorry

end regular_tetrahedron_ineq_l354_354163


namespace June_sweets_count_l354_354009

variable (A M J : ℕ)

-- condition: May has three-quarters of the number of sweets that June has
def May_sweets := M = (3/4) * J

-- condition: April has two-thirds of the number of sweets that May has
def April_sweets := A = (2/3) * M

-- condition: April, May, and June have 90 sweets between them
def Total_sweets := A + M + J = 90

-- proof problem: How many sweets does June have?
theorem June_sweets_count : 
  May_sweets M J ∧ April_sweets A M ∧ Total_sweets A M J → J = 40 :=
by
  sorry

end June_sweets_count_l354_354009


namespace part1_part2_l354_354500

-- Definitions
def parametric_circle (α : ℝ) (θ : ℝ) : ℝ × ℝ := (α + α * cos θ, α * sin θ)
def polar_line (ρ : ℝ) (θ : ℝ) : ℝ := ρ * sin (θ + π / 4)
def distance_between_points (A B : ℝ × ℝ) : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Conditions
def circle_center_and_radius (a : ℝ) : Prop := (x - a)^2 + y^2 = a^2
def line_intersects_circle (a : ℝ) (θ ρ : ℝ) : Prop := polar_line ρ θ = 2 * real.sqrt 2
def circle_arc_distance (α : ℝ) : Prop := ∃ A B : ℝ × ℝ, distance_between_points A B = 2 * real.sqrt 2
def max_value_of_sum (α θ1 θ2 : ℝ) : Prop := θ1 + θ2 = π / 3 ∧ ∃ M N : ℝ × ℝ, 
  parametric_circle α θ1 = M ∧ parametric_circle α θ2 = N ∧ |α * cos θ1 + α * cos θ2| + |α * sin θ1 + α * sin θ2| = 4 * real.sqrt 3

-- Proof focus
theorem part1 (α : ℝ) (θ ρ : ℝ) (h1 : 0 < α ∧ α < 5) (h2 : circle_center_and_radius α) 
  (h3 : line_intersects_circle α θ ρ) (h4 : circle_arc_distance α) : 
  α = 2 := sorry

theorem part2 (α : ℝ) (θ1 θ2 : ℝ) (h1 : 0 < α ∧ α < 5) (h2 : θ1 + θ2 = π / 3) 
  (h3 : ∃ M N : ℝ × ℝ, parametric_circle α θ1 = M ∧ parametric_circle α θ2 = N) :
  max_value_of_sum α θ1 θ2 := sorry

end part1_part2_l354_354500


namespace simplified_expression_evaluation_l354_354904

def expression (x y : ℝ) : ℝ :=
  3 * (x^2 - 2 * x^2 * y) - 3 * x^2 + 2 * y - 2 * (x^2 * y + y)

def x := 1/2
def y := -3

theorem simplified_expression_evaluation : expression x y = 6 :=
  sorry

end simplified_expression_evaluation_l354_354904


namespace max_value_of_expression_l354_354511

noncomputable def max_expression_value (a b : ℝ) := a * b * (100 - 5 * a - 2 * b)

theorem max_value_of_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 2 * b < 100 →
  max_expression_value a b ≤ 78125 / 36 := by
  intros a b ha hb h
  sorry

end max_value_of_expression_l354_354511


namespace find_fruit_juice_amount_l354_354026

def total_punch : ℕ := 14 * 10
def mountain_dew : ℕ := 6 * 12
def ice : ℕ := 28
def fruit_juice : ℕ := total_punch - mountain_dew - ice

theorem find_fruit_juice_amount : fruit_juice = 40 := by
  sorry

end find_fruit_juice_amount_l354_354026


namespace no_equal_probability_totals_for_two_loaded_dice_l354_354945

-- Variables representing the probabilities of rolling numbers on the two dice
variables {a b c d : ℝ}

-- Assume each possible total (2 through 12) has an equal probability
def equal_probability (a b c d : ℝ) : Prop :=
  let p := 1 / 11 in
  (a * c = p ∧ -- Total of 2
   b * d = p ∧ -- Total of 12
   (a / b + b / a) = 1 ∧ -- Simplified expression for total of 7 with conditions a*d + b*c = p
   ... ) -- Add further conditions for other totals here

-- Theorem stating the impossibility
theorem no_equal_probability_totals_for_two_loaded_dice (a b c d : ℝ) : ¬equal_probability a b c d :=
sorry

end no_equal_probability_totals_for_two_loaded_dice_l354_354945


namespace ratio_m_n_l354_354190

theorem ratio_m_n (m n : ℕ) (h1 : m > n) (h2 : ¬ (m % n = 0)) (h3 : (m % n) = ((m + n) % (m - n))) : (m : ℚ) / n = 5 / 2 := by
  sorry

end ratio_m_n_l354_354190


namespace inv_function_correct_l354_354161

def f (x : ℝ) : ℝ :=
  if x < 15 then x^2 + 3 else 3 * x - 2

noncomputable def f_inv : ℝ → ℝ 
| 10 := Real.sqrt 7
| 49 := 17
| _ := 0  -- This is a total function definition with placeholder cases

theorem inv_function_correct :
  f_inv 10 + f_inv 49 = Real.sqrt 7 + 17 :=
  by 
    -- the proof would go here
    sorry

end inv_function_correct_l354_354161


namespace sum_of_sequence_l354_354010

noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℝ := n * Real.pi
def b (n : ℕ) : ℝ := 3^n * a n

-- Define the sum T_n of the first n terms of sequence b
def T (n : ℕ) : ℝ := ∑ k in Finset.range n, b (k + 1)

-- The theorem to prove
theorem sum_of_sequence (n : ℕ) : 
  T n = ( (2 * n - 1) * 3^(n + 1) + 3 ) / 4 * Real.pi :=
by
  sorry

end sum_of_sequence_l354_354010


namespace ratio_of_female_to_male_officers_on_duty_l354_354880

theorem ratio_of_female_to_male_officers_on_duty 
    (p : ℝ) (T : ℕ) (F : ℕ) 
    (hp : p = 0.19) (hT : T = 152) (hF : F = 400) : 
    (76 / 76) = 1 :=
by
  sorry

end ratio_of_female_to_male_officers_on_duty_l354_354880


namespace edge_length_of_cube_l354_354408

noncomputable def sphere_volume : ℝ := (9 / 16) * Real.pi

theorem edge_length_of_cube
  (V : ℝ)
  (hV : V = sphere_volume)
  (a : ℝ)
  (h1 : V = (4 / 3) * Real.pi * (sqrt 3 * a / 2) ^ 3)
  : a = sqrt 3 / 2 := 
  sorry

end edge_length_of_cube_l354_354408


namespace manufacturer_cost_price_l354_354321

theorem manufacturer_cost_price (final_price : ℝ) (m_profit r1 r2 r3 : ℝ) : 
  final_price = 30.09 → 
  m_profit = 0.18 → 
  r1 = 1.20 → 
  r2 = 1.25 → 
  let C := final_price / ((1 + m_profit) * r1 * r2) in 
  C ≈ 17 :=
by sorry

end manufacturer_cost_price_l354_354321


namespace exponent_zero_rule_l354_354607

theorem exponent_zero_rule :
  (-(763215432: ℤ) / (19080360805: ℤ)) ^ (0: ℤ) = 1 :=
by
  sorry

end exponent_zero_rule_l354_354607


namespace line_through_circumcenter_l354_354225

structure Point where
  x : ℝ
  y : ℝ

variables {A B C M M1 : Point}

def dist (P Q : Point) : ℝ := real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Given conditions
axiom dist_M_A : dist M A = 1
axiom dist_M_B : dist M B = 2
axiom dist_M_C : dist M C = 3

axiom dist_M1_A : dist M1 A = 3
axiom dist_M1_B : dist M1 B = real.sqrt 15
axiom dist_M1_C : dist M1 C = 5

axiom weighted_sum_zero (P : Point) : 
  5 * (dist P A)^2 - 8 * (dist P B)^2 + 3 * (dist P C)^2 = 0

-- Theorem stating that the line MM1 passes through the circumcenter of triangle ABC
theorem line_through_circumcenter (circumcenter : Point)
  (hM : weighted_sum_zero M)
  (hM1 : weighted_sum_zero M1)
  (hO : weighted_sum_zero circumcenter) : 
  ∃ k : ℝ, ∀ t : ℝ, circumcenter.x = M.x + t * (M1.x - M.x) ∧ circumcenter.y = M.y + t * (M1.y - M.y) :=
sorry

end line_through_circumcenter_l354_354225


namespace triangle_max_area_l354_354473

theorem triangle_max_area (A B C : Type) [EuclideanSpace E] 
(a b c : ℝ) 
(h₁ : angle A = real.pi / 3) 
(h₂ : dist B C = 2 * real.sqrt 3) : 
area A B C ≤ 3 * real.sqrt 3 :=
sorry

end triangle_max_area_l354_354473


namespace find_term_of_sequence_l354_354360

theorem find_term_of_sequence :
  ∀ (a d n : ℤ), a = -5 → d = -4 → (-4)*n + 1 = -401 → n = 100 :=
by
  intros a d n h₁ h₂ h₃
  sorry

end find_term_of_sequence_l354_354360


namespace count_of_green_hats_l354_354604

-- Defining the total number of hats
def total_hats : ℕ := 85

-- Defining the costs of each hat type
def blue_cost : ℕ := 6
def green_cost : ℕ := 7
def red_cost : ℕ := 8

-- Defining the total cost
def total_cost : ℕ := 600

-- Defining the ratio as 3:2:1
def ratio_blue : ℕ := 3
def ratio_green : ℕ := 2
def ratio_red : ℕ := 1

-- Defining the multiplication factor
def x : ℕ := 14

-- Number of green hats based on the ratio
def G : ℕ := ratio_green * x

-- Proving that we bought 28 green hats
theorem count_of_green_hats : G = 28 := by
  -- proof steps intention: sorry to skip the proof
  sorry

end count_of_green_hats_l354_354604


namespace mutually_exclusive_iff_complementary_l354_354731

variables {Ω : Type} (A₁ A₂ : Set Ω) (S : Set Ω)

/-- Proposition A: Events A₁ and A₂ are mutually exclusive. -/
def mutually_exclusive : Prop := A₁ ∩ A₂ = ∅

/-- Proposition B: Events A₁ and A₂ are complementary. -/
def complementary : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = S

/-- Proposition A is a necessary but not sufficient condition for Proposition B. -/
theorem mutually_exclusive_iff_complementary :
  mutually_exclusive A₁ A₂ → (complementary A₁ A₂ S → mutually_exclusive A₁ A₂) ∧
  (¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂ S)) :=
by
  sorry

end mutually_exclusive_iff_complementary_l354_354731


namespace least_integer_sum_of_primes_l354_354953

-- Define what it means to be prime and greater than a number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greater_than_ten (n : ℕ) : Prop := n > 10

-- Main theorem statement
theorem least_integer_sum_of_primes :
  ∃ n, (∀ p1 p2 p3 p4 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
                        greater_than_ten p1 ∧ greater_than_ten p2 ∧ greater_than_ten p3 ∧ greater_than_ten p4 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
                        n = p1 + p2 + p3 + p4 → n ≥ 60) ∧
        n = 60 :=
  sorry

end least_integer_sum_of_primes_l354_354953


namespace average_after_17th_inning_l354_354310

theorem average_after_17th_inning (A : ℝ) (total_runs_16th_inning : ℝ) 
  (average_before_17th : A * 16 = total_runs_16th_inning) 
  (increased_average_by_3 : (total_runs_16th_inning + 83) / 17 = A + 3) :
  (A + 3) = 35 := 
sorry

end average_after_17th_inning_l354_354310


namespace checkerboard_sum_l354_354999

-- Define the row-wise numbering function
def f (i j : ℕ) : ℕ := 13 * (i - 1) + j

-- Define the column-wise numbering function
def g (i j : ℕ) : ℕ := 17 * (j - 1) + i

-- Define the condition where the numbering systems agree
def agrees (i j : ℕ) : Prop := 3 * i - 4 * j = -1

theorem checkerboard_sum :
  (∑ i in (Finset.range 18), ∑ j in (Finset.range 14), if agrees i j then f i j else 0) = 171 :=
sorry

end checkerboard_sum_l354_354999


namespace find_constant_l354_354288

-- Given function f satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the given conditions
variable (h1 : ∀ x : ℝ, f x + 3 * f (c - x) = x)
variable (h2 : f 2 = 2)

-- Statement to prove the constant c
theorem find_constant (c : ℝ) : (f x + 3 * f (c - x) = x) → (f 2 = 2) → c = 8 :=
by
  intro h1 h2
  sorry

end find_constant_l354_354288


namespace multiplicative_inverse_123_mod_456_l354_354359

theorem multiplicative_inverse_123_mod_456 :
  ∃ a : ℤ, 0 ≤ a ∧ a ≤ 455 ∧ (123 * a ≡ 1 [MOD 456]) :=
sorry

end multiplicative_inverse_123_mod_456_l354_354359


namespace no_real_solution_for_equation_l354_354681

theorem no_real_solution_for_equation (p : ℝ) (x : ℝ) (h₁ : 0 ≤ x) : ¬ (sqrt (x^2 - p) + 2 * sqrt (x^2 - 1) = x) :=
by
  sorry

end no_real_solution_for_equation_l354_354681


namespace count_valid_x_l354_354447

theorem count_valid_x :
  let x_min := 25
  let x_max := 33
  let valid_count := list.range' x_min (x_max - x_min + 1)
  valid_count.length = 9 := by
  sorry

end count_valid_x_l354_354447


namespace flour_for_third_of_cake_l354_354989

theorem flour_for_third_of_cake :
  let cups_of_flour := 6 + 1/3 in
  let one_third := (1/3 : ℚ) in
  let needed_flour := cups_of_flour * one_third in
  needed_flour = 2 + 1/9 := by 
    let cups_of_flour : ℚ := 19/3
    let needed_flour : ℚ := (1/3) * cups_of_flour
    have : needed_flour = 19 / 9 := by norm_num
    sorry

end flour_for_third_of_cake_l354_354989


namespace max_distance_OC_l354_354140

theorem max_distance_OC : 
  ∀ (A B C : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 1) ∧ (B = (3, 0)) ∧ 
    (equilateral_triangle A B C) → 
    |dist (0, 0) C| ≤ 4 := 
by
sorry

end max_distance_OC_l354_354140


namespace probability_of_range_l354_354568

-- Assuming the normal distribution and the probability properties
noncomputable def math_scores_distribution : ℝ → ℝ :=
  sorry

theorem probability_of_range :
  ∀ (ξ : ℝ),
  (∀ x, math_scores_distribution x ~ Normal 100 (5 ^ 2)) →
  (∀ x, (P (ξ < 110) = 0.98)) →
  (P (90 < ξ < 100) = 0.48) :=
by
  intros ξ distribution_property probability_property
  sorry

end probability_of_range_l354_354568


namespace first_bin_cans_l354_354343

variables (n1 n2 n3 n4 n5 : ℕ)

-- Conditions
def cond1 : n2 = 4 := by sorry
def cond2 : n3 = 7 := by sorry
def cond3 : n4 = 11 := by sorry
def cond4 : n5 = 16 := by sorry
def cond5 : (n3 - n2) = 3 ∧ (n4 - n3) = 4 ∧ (n5 - n4) = 5 := by sorry

-- Conclusion
theorem first_bin_cans : n1 = 2 :=
by 
  have h1 : n2 = 4 := cond1
  have h2 : n3 = 7 := cond2
  have h3 : n4 = 11 := cond3
  have h4 : n5 = 16 := cond4
  have h_diff : (n3 - n2) = 3 ∧ (n4 - n3) = 4 ∧ (n5 - n4) = 5 := cond5
  -- use the conditions to deduce n1
  sorry

end first_bin_cans_l354_354343


namespace bus_speed_is_48_l354_354710

theorem bus_speed_is_48 (v1 v2 : ℝ) (h1 : ∀ (t : ℝ), v1 * t + v2 * t = 8)
  (h2 : ∀ (d AB : ℝ), d(C, t(AB v1)) ∧ d(C, t(AB v2)))
  (h3 : ∀ (t : ℝ), d(2/3) = lim t(AB v1) - 0)
  (h4 : dAB = 4) :
  v2 = 48 :=
  sorry

end bus_speed_is_48_l354_354710


namespace sum_of_faces_edges_vertices_eq_26_diff_of_vertices_edges_eq_neg4_l354_354280

/-- Definitions and conditions for a square prism -/
def square_prism_faces : Nat := 2 + 4
def square_prism_edges : Nat := 8 + 4
def square_prism_vertices : Nat := 4 + 4

/-- Main theorems to prove -/
theorem sum_of_faces_edges_vertices_eq_26 :
  square_prism_faces + square_prism_edges + square_prism_vertices = 26 := by
  sorry

theorem diff_of_vertices_edges_eq_neg4 :
  square_prism_vertices - square_prism_edges = -4 := by
  sorry

end sum_of_faces_edges_vertices_eq_26_diff_of_vertices_edges_eq_neg4_l354_354280


namespace max_score_with_fifteen_cards_l354_354286

-- Definitions for the conditions given
variable (r b y : Nat)

-- Total number of cards condition
def total_cards : Prop := r + b + y = 15

-- Point values conditions
def point_value_red : Nat := r
def point_value_blue : Nat := 2 * r * b
def point_value_yellow : Nat := 3 * b * y

-- Total score
def total_score : Nat := point_value_red + point_value_blue + point_value_yellow

-- The main theorem: The maximum score with fifteen cards
theorem max_score_with_fifteen_cards (h : total_cards r b y) : total_score r b y = 168 :=
sorry -- Proof omitted

end max_score_with_fifteen_cards_l354_354286


namespace perpendicular_vectors_l354_354443

variables (a b : ℝ × ℝ) (k : ℝ)
def a := (1, 2)
def b := (-3, 2)

theorem perpendicular_vectors :
  (k * a + b) ∙ (a - 3 * b) = 0 → k = 19 :=
by sorry

end perpendicular_vectors_l354_354443


namespace cylinder_height_l354_354515

noncomputable def height_of_cylinder (R α β : ℝ) : ℝ :=
  2 * R * tan β * (sqrt ((sin (α + β)) * (sin (α - β)))) / (sin α * cos β)

theorem cylinder_height (R α β : ℝ) (h : ℝ) 
  (condition_1 : 0 < R) -- Radius of the base is positive.
  (condition_2 : 0 < α ∧ α < π/2) -- α is an acute angle.
  (condition_3 : 0 < β ∧ β < π/2) -- β is an acute angle.
  : 
  h = height_of_cylinder R α β :=
sorry

end cylinder_height_l354_354515


namespace max_subsets_with_arithmetic_intersections_l354_354174

theorem max_subsets_with_arithmetic_intersections : 
  ∃ A : set (set (fin 2016)), (∀ (i j : set (fin 2016)), i ≠ j → ∃ (d : ℕ), d > 0 ∧ ∀ m n ∈ (i ∩ j : set (fin 2016)), 
  (n = m + d) ∨ (n = m + 2*d)) ∧ fintype.card A = 1362432297 :=
begin
  sorry
end

end max_subsets_with_arithmetic_intersections_l354_354174


namespace equations_have_solutions_l354_354205

theorem equations_have_solutions (a b c : ℝ) : 
  ∃ x : ℝ, a * Real.sin x + b * Real.cos x + c = 0 ∨ 
           2 * a * Real.tan x + b * Real.cot x + 2 * c = 0 :=
by 
  sorry

end equations_have_solutions_l354_354205


namespace company_hired_22_additional_males_l354_354589

theorem company_hired_22_additional_males
  (E M : ℕ) 
  (initial_percentage_female : ℝ)
  (final_total_employees : ℕ)
  (final_percentage_female : ℝ)
  (initial_female_count : initial_percentage_female * E = 0.6 * E)
  (final_employee_count : E + M = 264) 
  (final_female_count : initial_percentage_female * E = final_percentage_female * (E + M)) :
  M = 22 := 
by
  sorry

end company_hired_22_additional_males_l354_354589


namespace max_xy_sum_l354_354952

theorem max_xy_sum (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x * y = 18) : x + y ≤ Real.sqrt 166 :=
by
  have h := (x + y)^2
  have : (x + y)^2 = x^2 + y^2 + 2 * (x * y) by sorry
  rw [h1, h2] at this
  have : (x + y)^2 = 166 by sorry
  have : x + y = Real.sqrt 166 ∨ x + y = -Real.sqrt 166 by sorry
  sorry

end max_xy_sum_l354_354952


namespace total_distance_travelled_l354_354644

-- Condition definitions
def distancesNumber := 5
def speeds := [6, 12, 18, 24, 30] -- in km/hr
def totalTime := 17 / 60 -- in hours

-- Proof statement
theorem total_distance_travelled : 
  ∃ d, d > 0 ∧ 5 * d ≈ 3.728 ∧ 
  totalTime = d / speeds[0] + d / speeds[1] + d / speeds[2] + d / speeds[3] + d / speeds[4] := 
by
  sorry

end total_distance_travelled_l354_354644


namespace probability_of_5_blue_marbles_l354_354843

/--
Jane has a bag containing 9 blue marbles and 6 red marbles. 
She draws a marble, records its color, returns it to the bag, and repeats this process 8 times. 
We aim to prove that the probability that she draws exactly 5 blue marbles is \(0.279\).
-/
theorem probability_of_5_blue_marbles :
  let blue_probability := 9 / 15 
  let red_probability := 6 / 15
  let single_combination_prob := (blue_probability^5) * (red_probability^3)
  let combinations := (Nat.choose 8 5)
  let total_probability := combinations * single_combination_prob
  (Float.round (total_probability.toFloat * 1000) / 1000) = 0.279 :=
by
  sorry

end probability_of_5_blue_marbles_l354_354843


namespace ellipse_major_minor_axis_l354_354626

theorem ellipse_major_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) ∧
  (∃ a b : ℝ, a = 2 * b ∧ b^2 = 1 ∧ a^2 = 1/m) →
  m = 1/4 :=
by {
  sorry
}

end ellipse_major_minor_axis_l354_354626


namespace angle_AB_KM_60_l354_354516

variables {A B C K M : Type}

-- Conditions: ABC is an equilateral triangle, BCKM is a parallelogram
variable [equilateral_triangle ABC]
variable [parallelogram BCKM]

-- Statement: The angle between line AB and line KM is 60 degrees
theorem angle_AB_KM_60 : angle_between_lines AB KM = 60 := 
sorry

end angle_AB_KM_60_l354_354516


namespace carpet_area_cost_l354_354327

theorem carpet_area_cost (len_ft : ℝ) (width_ft : ℝ) (ft_to_m : ℝ) (cost_per_m2 : ℝ) 
  (room_area_ft2 := len_ft * width_ft)
  (conversion_factor := ft_to_m * ft_to_m)
  (room_area_m2 := room_area_ft2 / conversion_factor)
  (total_cost := room_area_m2 * cost_per_m2) : 
  len_ft = 18 ∧ width_ft = 15 ∧ ft_to_m = 3.28 ∧ cost_per_m2 = 12 →
  room_area_m2 = 25.08 ∧ total_cost = 300.96 :=
by
  intros h
  cases h with h1 h22
  cases h22 with h2 h3
  cases h3 with h3 h4
  sorry

end carpet_area_cost_l354_354327


namespace phase_shift_f_l354_354378

noncomputable def f (x : ℝ) : ℝ := 5 * sin (2 * (x - π / 4))

theorem phase_shift_f : phase_shift f = π / 4 :=
sorry

end phase_shift_f_l354_354378


namespace length_of_PQ_l354_354054

-- Let P, Q, and R be points such that QPR is a right triangle with a right angle at P
-- and angle PQR is 45 degrees, and PR is 10 units. Prove that PQ = 5 * sqrt 2.
theorem length_of_PQ
  (P Q R : Point)
  (h_right : angle QPR = π / 2)
  (h_45 : angle PQR = π / 4)
  (h_PR : dist P R = 10) :
  dist P Q = 5 * Real.sqrt 2 := by
  sorry

end length_of_PQ_l354_354054


namespace fibby_numbers_l354_354645

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧ ∃ (n : ℕ) (d : ℕ → ℕ),
  (∀ j, 1 ≤ j ∧ j ≤ k - 2 → d (j + 2) = d (j + 1) + d j) ∧
  (∀ (j : ℕ), 1 ≤ j ∧ j ≤ k → d j ∣ n) ∧
  (∀ (m : ℕ), m ∣ n → m < d 1 ∨ m > d k)

theorem fibby_numbers : ∀ (k : ℕ), is_fibby k → k = 3 ∨ k = 4 :=
sorry

end fibby_numbers_l354_354645


namespace temperature_decrease_l354_354137

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end temperature_decrease_l354_354137


namespace points_three_planes_points_two_planes_one_point_points_one_plane_two_points_points_three_points_l354_354035

-- (a) Prove that the number of points at a given distance from three intersecting planes is eight.
theorem points_three_planes (p1 p2 p3 : Plane) (d : ℝ) (h1 : intersects p1 p2) (h2 : intersects p2 p3) (h3 : intersects p1 p3) :
    number_of_points_at_distance_from_planes p1 p2 p3 d = 8 := 
sorry

-- (b) Prove that the number of points at a given distance from two intersecting planes and one point is eight.
theorem points_two_planes_one_point (p1 p2 : Plane) (pt : Point) (d : ℝ) (h : intersects p1 p2) :
    number_of_points_at_distance_from_planes_and_point p1 p2 pt d = 8 := 
sorry

-- (c) Prove that the number of points at a given distance from one plane and two points is four.
theorem points_one_plane_two_points (p : Plane) (pt1 pt2 : Point) (d : ℝ) :
    number_of_points_at_distance_from_plane_and_points p pt1 pt2 d = 4 :=
sorry

-- (d) Prove that the number of points at a given distance from three points can be zero, one, or two.
theorem points_three_points (pt1 pt2 pt3 : Point) (d : ℝ) :
    number_of_points_at_distance_from_points pt1 pt2 pt3 d ∈ {0, 1, 2} := 
sorry

end points_three_planes_points_two_planes_one_point_points_one_plane_two_points_points_three_points_l354_354035


namespace geo_seq_ineq_problem_statement_l354_354479

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n+1) = r * a n

theorem geo_seq_ineq (a : ℕ → ℝ) (r : ℝ) (h_seq : geometric_sequence a) : 
  ∀ n : ℕ, r < 1 → a(n+1) < a n := sorry

theorem problem_statement (a : ℕ → ℝ) (r : ℝ) 
  (h_seq : geometric_sequence a)
  (h_ineq: ∀ n : ℕ, r < 1 → a(n+1) < a n)
  (h1 : a 2 * a 8 = 6) 
  (h2 : a 4 + a 6 = 5) 
  : a 5 / a 7 = 3 / 2 := 
sorry

end geo_seq_ineq_problem_statement_l354_354479


namespace election_votes_total_l354_354824

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total_l354_354824


namespace solution_set_of_system_of_inequalities_l354_354585

theorem solution_set_of_system_of_inequalities :
  {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3 * x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end solution_set_of_system_of_inequalities_l354_354585


namespace additional_june_sales_l354_354233

-- Definitions
def normal_sales : ℕ := 21122
def total_sales_june_july : ℕ := 46166
def july_sales := normal_sales

-- We need to prove that the hobby store sold 3922 more trading cards in June than normal.
theorem additional_june_sales:
  let june_sales := total_sales_june_july - july_sales in
  june_sales - normal_sales = 3922 :=
by
  let june_sales := total_sales_june_july - july_sales
  have h : june_sales = 25044 := by sorry
  have h2 : june_sales - normal_sales = 3922 := by sorry
  show june_sales - normal_sales = 3922 from h2
  sorry

end additional_june_sales_l354_354233


namespace find_y_in_terms_of_x_and_n_l354_354512

variable (x n y : ℝ)

theorem find_y_in_terms_of_x_and_n
  (h : n = 3 * x * y / (x - y)) :
  y = n * x / (3 * x + n) :=
  sorry

end find_y_in_terms_of_x_and_n_l354_354512


namespace average_marks_two_classes_correct_l354_354289

axiom average_marks_first_class : ℕ → ℕ → ℕ
axiom average_marks_second_class : ℕ → ℕ → ℕ
axiom combined_average_marks_correct : ℕ → ℕ → Prop

theorem average_marks_two_classes_correct :
  average_marks_first_class 39 45 = 39 * 45 →
  average_marks_second_class 35 70 = 35 * 70 →
  combined_average_marks_correct (average_marks_first_class 39 45) (average_marks_second_class 35 70) :=
by
  intros h1 h2
  sorry

end average_marks_two_classes_correct_l354_354289


namespace jacob_pencils_count_l354_354841

variables (x y : ℕ)

-- Condition statements
def total_pencils := 21
def zain_monday := x
def zain_tuesday := y
def jacob_monday := (2 * x) / 3
def jacob_tuesday := y / 2
def all_pencils_taken := x + jacob_monday + y + jacob_tuesday = total_pencils

theorem jacob_pencils_count :
  all_pencils_taken →
  jacob_monday + jacob_tuesday = 8 :=
sorry

end jacob_pencils_count_l354_354841


namespace quadratic_form_l354_354925

-- Define the constants b and c based on the problem conditions
def b : ℤ := 900
def c : ℤ := -807300

-- Create a statement that represents the proof goal
theorem quadratic_form (c_eq : c = -807300) (b_eq : b = 900) : c / b = -897 :=
by
  sorry

end quadratic_form_l354_354925


namespace sum_b_formula_l354_354929

noncomputable def a (n : ℕ) : ℚ := n^2 + 3*n + 2
noncomputable def b (n : ℕ) : ℚ := 1 / a n

def sum_b (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, b (k + 1))

theorem sum_b_formula (n : ℕ) :
  sum_b n = 1 / 2 - 1 / (n + 2) :=
by
  sorry

end sum_b_formula_l354_354929


namespace number_of_girls_l354_354132

theorem number_of_girls (total_people : ℕ) (num_boys : ℕ) (num_teachers : ℕ) (h1 : total_people = 1396) (h2 : num_boys = 309) (h3 : num_teachers = 772) : 
  total_people - num_boys - num_teachers = 315 := 
by
  rw [h1, h2, h3]
  norm_num

end number_of_girls_l354_354132


namespace vector_BE_eq_b_sub_half_a_l354_354883

theorem vector_BE_eq_b_sub_half_a
  (a b : ℝ)  -- Vectors a and b as real numbers for simplicity
  (E_midpoint_DC : E = midpoint D C)
  (square_ABCD : is_square A B C D)
  (AB_eq_a : vector_eq AB a)
  (AD_eq_b : vector_eq AD b) :
  vector_eq BE (b - (1/2) * a) := 
sorry

end vector_BE_eq_b_sub_half_a_l354_354883


namespace can_identify_counterfeits_l354_354149

-- Define the problem setup and conditions
noncomputable def counterfeit_problem (N : ℕ) (h : N ≥ 5) : Prop :=
  ∃ (m : Fin N → ℝ), -- We have N coins represented by their weights
    (∀ i, i < N → m i > 0) ∧ -- All weights are positive
    (∃ i j, i ≠ j ∧ m i = m j) ∧ -- There are exactly 2 coins with the same weight
    (∃ c1 c2, c1 ≠ c2 ∧ m c1 = m c2 ∧ m c1 < m (Fin.ofNat 0) ∧ m c2 < m (Fin.ofNat 1)) -- and these 2 coins are lighter

-- The goal is to show that it is possible to identify and demonstrate the even weights given only two weighings
theorem can_identify_counterfeits (N : ℕ) (h : N ≥ 5) : 
    ∃ (m : Fin N → ℝ), -- We have N coins represented by their weights
      (∀ i, i < N → m i > 0) ∧ -- All weights are positive
      (∃ c1 c2, c1 ≠ c2 ∧ m c1 = m c2 ∧ m c1 < m (Fin.ofNat 0) ∧ m c2 < m (Fin.ofNat 1)) → 
      ∃ w1 w2, -- exist two weighings
      (w1.cases_on (λ a b, m a + m b) = m (Fin.ofNat 0) ↔ m a < m (Fin.ofNat 0)) ∧
      (w2.cases_on (λ a b, m a + m b) = m (Fin.ofNat 1) ↔ m a < m (Fin.ofNat 1)) :=
sorry

end can_identify_counterfeits_l354_354149


namespace median_intersection_l354_354861

noncomputable def is_common_point (a b c : ℝ) (z : ℂ) : Prop :=
  ∃ t ∈ ℝ, z = (a * Complex.I) * (Real.cos t ^ 4 : ℂ) + 
            2 * (1/2 + b * Complex.I) * (Real.cos t ^ 2 : ℂ) * (Real.sin t ^ 2 : ℂ) + 
            (1 + c * Complex.I) * (Real.sin t ^ 4 : ℂ)

theorem median_intersection (a b c : ℝ) :
  ∃ z : ℂ, is_common_point a b c z ∧ z.re = 1/2 ∧ z.im = (a + c + 2 * b) / 4 :=
sorry

end median_intersection_l354_354861


namespace equivalence_of_fractions_l354_354552

variable (a b c : ℝ)
variable (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def x : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)
def y : ℝ := (c^2 + a^2 - b^2) / (2 * c * a)
def z : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

theorem equivalence_of_fractions (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (1 - (x a b c)^2) / a^2 = (1 - (y a b c)^2) / b^2 ∧ 
  (1 - (y a b c)^2) / b^2 = (1 - (z a b c)^2) / c^2 :=
by {
  sorry -- Proof goes here
}

end equivalence_of_fractions_l354_354552


namespace Tamika_probability_greater_than_Carlos_l354_354217

noncomputable def T_products : Finset ℕ :=
  {7, 8, 9, 10}.powerset.filter (λ s, s.card = 2).image (λ s, s.prod id)

noncomputable def C_products : Finset ℕ :=
  {2, 3, 5, 6}.powerset.filter (λ s, s.card = 3).image (λ s, s.prod id)

def count_favorable_outcomes : ℕ :=
  (T_products ×ˢ C_products).filter (λ p, p.1 > p.2).card

def total_possible_outcomes : ℕ :=
  T_products.card * C_products.card

theorem Tamika_probability_greater_than_Carlos :
  count_favorable_outcomes.to_rat / total_possible_outcomes.to_rat = 17 / 24 :=
by
  -- This is a statement placeholder
  sorry

end Tamika_probability_greater_than_Carlos_l354_354217


namespace problem_statement_l354_354050

def digit_sum (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

theorem problem_statement :
  ∀ n : ℕ, (∃ a b : ℕ, n = digit_sum a ∧ n = digit_sum b ∧ n = digit_sum (a + b)) ↔ (∃ k : ℕ, n = 9 * k) :=
by
  sorry

end problem_statement_l354_354050


namespace equation_of_circle_min_area_of_quadrilateral_l354_354635

/- Definitions related to the problem -/
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line_through_center (x y : ℝ) : Prop := x + y = 2
def circle_eq (a b r x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r

/- Circle center must lie on the line x + y - 2 = 0 -/
def center_eq_line (a b : ℝ) : Prop := a + b = 2

/- Condition on P being on the line 3x + 4y + 8 = 0 -/
def P_on_line (x y : ℝ) : Prop := 3*x + 4*y + 8 = 0

/- Definition for the distance between two points -/
def dist (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/- Assertions that need to be proved -/

-- (1) Prove the equation of circle M 
theorem equation_of_circle :
  ∃ (a b r : ℝ), (circle_eq a b r 1 (-1)) ∧ (circle_eq a b r (-1) 1) ∧ (center_eq_line a b) ∧ (a = 1) ∧ (b = 1) ∧ (r = 2) :=
sorry

-- (2) Prove the minimum area of quadrilateral PAMB is 2√5
theorem min_area_of_quadrilateral :
  ∃ (P : ℝ × ℝ), (P_on_line P.1 P.2) ∧ (dist P (1, 1) = 3) ∧ (2 * real.sqrt (dist P (1, 1)^2 - 4) = 2 * real.sqrt 5) :=
sorry

end equation_of_circle_min_area_of_quadrilateral_l354_354635


namespace vector_addition_l354_354444

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

-- State the problem as a theorem
theorem vector_addition : a + b = (-1, 5) := by
  -- the proof should go here
  sorry

end vector_addition_l354_354444


namespace victoria_money_returned_l354_354272

def cost_of_pizza (n : Nat) (price : Float) : Float := n * price

def cost_of_juice (n : Nat) (price : Float) : Float := n * price

def cost_of_chips (n : Nat) (price : Float) : Float := n * price

def cost_of_chocolate (n : Nat) (price : Float) : Float := n * price

def total_cost_before_discount_and_tax (pizza_cost juice_cost chip_cost chocolate_cost : Float) : Float :=
  pizza_cost + juice_cost + chip_cost + chocolate_cost

def apply_discount (cost : Float) (discount_rate : Float) : Float :=
  cost - (discount_rate * cost)

def apply_tax (cost : Float) (tax_rate : Float) : Float :=
  cost + (tax_rate * cost)

def total_money_returned (initial_amount : Float) (final_cost : Float) : Float :=
  if initial_amount < final_cost then 0 else initial_amount - final_cost

theorem victoria_money_returned :
  let pizza_cost := cost_of_pizza 3 15
  let juice_cost := cost_of_juice 4 4
  let chip_cost := cost_of_chips 2 3.5
  let chocolate_cost := cost_of_chocolate 5 1.25
  let total_cost := total_cost_before_discount_and_tax pizza_cost juice_cost chip_cost chocolate_cost
  let discounted_pizza_cost := apply_discount pizza_cost 0.10
  let new_total_cost := total_cost_before_discount_and_tax discounted_pizza_cost juice_cost chip_cost chocolate_cost
  let final_cost := apply_tax new_total_cost 0.05
  total_money_returned 50 final_cost = 0 :=
by
  sorry

end victoria_money_returned_l354_354272


namespace domino_grid_satisfies_conditions_l354_354881

def Cell : Type := ℤ -- assuming points can be integers (it doesn't specify non-negative but usually, points are).
def Grid := List (List Cell) -- representation of the grid as lists of lists

/-- The sum of points in each row equals 21. --/
def rowSum (grid : Grid) (r : ℕ) : Prop :=
  (grid.nth r).sum = 21

/-- The sum of points in each column equals 24. --/
def colSum (grid : Grid) (c : ℕ) : Prop :=
  (List.map (fun row => row.nth c) grid).sum = 24

/-- The 28 dominoes configuration where each domino covers two adjacent cells 
 (either horizontally or vertically), with each cell containing a point. --/
def validDominoConfiguration (grid : Grid) : Prop :=
  ∀ dom, List.contains grid dom ->  -- ensuring that domino tiles exist within the grid
    (dom.length = 2) ∧ -- domino covers two cells
    ((List.nth grid dom[0]) = Cell ∧ (List.nth grid dom[1]) = Cell ∨  -- horizontally adjacent
     (List.nth grid dom[0]) = Cell ∧ (List.nth grid dom[1]) = Cell)    -- vertically adjacent

theorem domino_grid_satisfies_conditions (grid : Grid) 
  (h_domino_conf : validDominoConfiguration grid) : 
  (∀ r, rowSum grid r) ∧ 
  (∀ c, colSum grid c) :=
sorry

end domino_grid_satisfies_conditions_l354_354881


namespace solveForY_l354_354213

variable (y : ℕ)

theorem solveForY (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
sorry

end solveForY_l354_354213


namespace root_zero_implies_m_eq_6_l354_354470

theorem root_zero_implies_m_eq_6 (m : ℝ) (h : ∃ x : ℝ, 3 * (x^2) + m * x + m - 6 = 0) : m = 6 := 
by sorry

end root_zero_implies_m_eq_6_l354_354470


namespace manufacturer_cost_price_l354_354323

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end manufacturer_cost_price_l354_354323


namespace t_sequence_correct_l354_354832

noncomputable def a_sequence (n : ℕ) : ℕ :=
  2 * n

noncomputable def b_sequence (n : ℕ) : ℕ :=
  a_sequence (n * (n + 1) / 2)

noncomputable def T_sequence (n : ℕ) : ℕ :=
  if even n then n * (n + 2) / 2
  else -((n + 1) ^ 2) / 2

theorem t_sequence_correct (n : ℕ) (d : ℕ := 2) (h_geometric_mean : a_sequence (2) ^ 2 = a_sequence (1) * a_sequence (4)) :
  T_sequence n = if even n then n * (n + 2) / 2
                 else -((n + 1) ^ 2) / 2 := sorry

end t_sequence_correct_l354_354832


namespace correct_drainage_time_l354_354624

-- Definitions for given values
def pool_width : ℝ := 80
def pool_length : ℝ := 150
def pool_depth : ℝ := 10
def hose_removal_rate : ℝ := 60

-- Calculate volume of the pool
def pool_volume : ℝ := pool_length * pool_width * pool_depth

-- Calculation of time needed to drain the pool in minutes
def drainage_time_minutes : ℝ := pool_volume / hose_removal_rate

-- Conversion of time from minutes to hours
def drainage_time_hours : ℝ := drainage_time_minutes / 60

-- The target values
def expected_drainage_time_minutes : ℝ := 2000
def expected_drainage_time_hours : ℝ := 2000 / 60

-- Prove that the computed values match the expected values
theorem correct_drainage_time : drainage_time_minutes = expected_drainage_time_minutes ∧ drainage_time_hours ≈ 33.33 := by
  sorry

end correct_drainage_time_l354_354624


namespace parts_per_hour_equality_l354_354201

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l354_354201


namespace compute_sum_l354_354030

theorem compute_sum : 
    (3 * (∑ i in (finset.range 25), (2 * (2*i + 1)*(2*i + 3)))) = 7650 :=
by
  -- Sum from i = 0 to 24, corresponding to the pairs in the given sequence.
  sorry

end compute_sum_l354_354030


namespace unit_digit_of_expression_l354_354347
-- Lean 4 statement

theorem unit_digit_of_expression : 
    (∃ (n : ℕ), n = (∏ i in (Finset.range 6), (2^(2^i) + 1)) ∧ n % 10 = 5) := 
by 
  sorry

end unit_digit_of_expression_l354_354347


namespace candles_burning_l354_354596

theorem candles_burning:
  ∀ (T1 T2 T3 x z: ℕ), T1 = 30 → T2 = 40 → T3 = 50 → z = 10 → x = 20 →
  (x + 2 * (T1 + T2 + T3 - x - 3 * z) / 2 + 3 * z = T1 + T2 + T3) →
  (∃ y: ℕ, y = (T1 + T2 + T3 - x - 3 * z) / 2 ∧ y = 35) :=
by
  intros T1 T2 T3 x z h1 h2 h3 h4 h5 h6
  use (T1 + T2 + T3 - x - 3 * z) / 2
  split
  case left => exact h6
  case right => sorry

end candles_burning_l354_354596


namespace exists_integer_square_with_three_identical_digits_l354_354024

theorem exists_integer_square_with_three_identical_digits:
  ∃ x: ℤ, (x^2 % 1000 = 444) := by
  sorry

end exists_integer_square_with_three_identical_digits_l354_354024


namespace distinct_real_numbers_inequalities_l354_354854

theorem distinct_real_numbers_inequalities
  (k : ℕ)
  (ω : Fin k → ℝ)
  (h_distinct : Function.Injective ω)
  (h_sum_nonzero : ∑ i, ω i ≠ 0) :
  ∃ n : Fin k → ℤ, (∑ i, n i * ω i > 0) ∧ ∀ (π : Equiv.Perm (Fin k)), π ≠ Equiv.refl (Fin k) → (∑ i, n i * ω (π i) < 0) :=
sorry

end distinct_real_numbers_inequalities_l354_354854


namespace singer_winner_l354_354933

/--
Let singers be A, B, C, and D with the following statements.
A: "B or C won the prize."
B: "A and C did not win the prize."
C: "I won the prize."
D: "B won the prize."
Only two of these statements are true.
Prove that C won the prize.
--/
theorem singer_winner (A_win B_win C_win D_win : Prop)
(HA : A_win = (B_win ∨ C_win)) -- A's statement
(HB : B_win = ¬(A_win ∨ C_win)) -- B's statement
(HC : C_win = C_win) -- C’s statement (which is tautologically true)
(HD : D_win = B_win) -- D's statement
(H_true_statements : [HA, HB, HC, HD].count (λ x => x = true) = 2) -- Exactly two statements are true
: C_win := 
sorry

end singer_winner_l354_354933


namespace smallest_n_l354_354677

-- Definitions for arithmetic sequences with given conditions
def arithmetic_sequence_a (n : ℕ) (x : ℕ) : ℕ := 1 + (n-1) * x
def arithmetic_sequence_b (n : ℕ) (y : ℕ) : ℕ := 1 + (n-1) * y

-- Main theorem statement
theorem smallest_n (x y n : ℕ) (hxy : x < y) (ha1 : arithmetic_sequence_a 1 x = 1) (hb1 : arithmetic_sequence_b 1 y = 1) 
  (h_sum : arithmetic_sequence_a n x + arithmetic_sequence_b n y = 2556) : n = 3 :=
sorry

end smallest_n_l354_354677


namespace range_of_a_l354_354756

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

def condition (x a : ℝ) : Prop := f x a ≤ abs (x - 4)

def A (a : ℝ) : set ℝ := { x | condition x a }

def included_interval (a : ℝ) : Prop := Icc 1 2 ⊆ A a

theorem range_of_a :
  ∀ a : ℝ, included_interval a → -3 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l354_354756


namespace ellipse_hyperbola_tangent_l354_354042

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m) → m = 2 :=
by sorry

end ellipse_hyperbola_tangent_l354_354042


namespace lion_cub_turtle_time_difference_l354_354966

theorem lion_cub_turtle_time_difference:
  (∀ (x : ℝ) (d : ℝ) (t1 t2 : ℝ), 
    (d = 6 * x) ∧ 
    (t1 = (6 * x - 1) / (x - (1/32))) ∧ 
    (t2 = 3.2) ∧ 
    (t2 = 1 / (1.5 * x + (1/32))) →
  (3.2 - (6 * x - 1) / (x - (1/32)) = 2.4)) :=
begin
  sorry
end

end lion_cub_turtle_time_difference_l354_354966


namespace jaclyns_constant_term_l354_354874

theorem jaclyns_constant_term :
  ∃ P Q : Polynomial ℝ, 
    P.monic ∧ Q.monic ∧
    P.degree = 4 ∧ Q.degree = 4 ∧
    (∀ z, P.coeff z = Q.coeff z ∧ 
          P.coeff 0 = Q.coeff 0 ∧ 
          P.coeff 0 = Q.coeff 0) ∧ 
    (P * Q = Polynomial.C 1 * Polynomial.X ^ 8 + 
                 Polynomial.C 2 * Polynomial.X ^ 7 + 
                 Polynomial.C 3 * Polynomial.X ^ 6 + 
                 Polynomial.C 9 * Polynomial.X ^ 5 + 
                 Polynomial.C 6 * Polynomial.X ^ 4 + 
                 Polynomial.C 6 * Polynomial.X ^ 3 + 
                 Polynomial.C 3 * Polynomial.X ^ 2 + 
                 Polynomial.C 6 * Polynomial.X + 
                 Polynomial.C 9) →
    P.coeff 0 = 3 :=
by
  sorry

end jaclyns_constant_term_l354_354874


namespace monotonic_intervals_range_of_k_l354_354090

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem monotonic_intervals {k : ℝ} :
  (k ≤ 0 → ∀ x > 0, f x k.monotoneOn (0, +∞)) ∧
  (k > 0 → 
    (∀ x ∈ (0, 1/k), f x k.monotoneOn (0, 1/k)) ∧ 
    (∀ x ∈ (1/k, +∞), f x k.monotoneOn (1/k, +∞))) :=
sorry

theorem range_of_k (h : ∀ x > 0, f x k ≤ 0) : k ≥ 1 :=
sorry

end monotonic_intervals_range_of_k_l354_354090


namespace find_g_two_l354_354230

noncomputable def g : ℝ → ℝ := λ x, -- we define g as a noncomputable function
  sorry -- the exact definition is provided by conditions

axiom g_condition (x : ℝ) (h : x ≠ 0) : 4 * g x - 3 * g (1 / x) = x^2

theorem find_g_two : g 2 = (65 : ℚ) / 28 := 
by sorry

end find_g_two_l354_354230


namespace total_packs_l354_354158

-- Definitions based on the conditions
def students_per_class : Nat := 30
def number_of_classes : Nat := 6
def packs_per_student : Nat := 2

-- The statement to be proved
theorem total_packs (students_per_class number_of_classes packs_per_student : Nat) : 
    (packs_per_student * (students_per_class * number_of_classes)) = 360 := 
by 
    -- Use the given values directly because they are defined
    have h1 : students_per_class = 30 := rfl
    have h2 : number_of_classes = 6 := rfl
    have h3 : packs_per_student = 2 := rfl
    calc
        packs_per_student * (students_per_class * number_of_classes)
            = 2 * (30 * 6) : by rw [h1, h2, h3]
        ... = 2 * 180 : by norm_num
        ... = 360 : by norm_num

end total_packs_l354_354158


namespace initial_ants_count_l354_354647

theorem initial_ants_count (n : ℕ) (h1 : ∀ x : ℕ, x ≠ n - 42 → x ≠ 42) : n = 42 :=
sorry

end initial_ants_count_l354_354647


namespace polynomial_power_degree_l354_354609

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

theorem polynomial_power_degree : 
  polynomial_degree ((5 * X^3 - 4 * X + 7)^10) = 30 := by
  sorry

end polynomial_power_degree_l354_354609


namespace perimeter_bounds_l354_354481

theorem perimeter_bounds (n : ℕ) (p : ℝ) :
  (n = 100) →
  (∀ i : ℕ, i < n → p = (1.0 + (4.0 / n))) →
  (14.0 / 10.0 < p ∧ p < 15.0 / 10.0) :=
by
  assume h₁ h₂
  sorry

end perimeter_bounds_l354_354481


namespace stack_height_of_coins_l354_354490

theorem stack_height_of_coins (n q : ℕ) (hn : 2.05 * n + 1.65 * q = 16.5) : n + q = 9 :=
by
  have : 205 * n + 165 * q = 1650 := by sorry
  have : 41 * n + 33 * q = 330 := by sorry
  sorry

end stack_height_of_coins_l354_354490


namespace spotlight_illumination_l354_354909

/-
The arena is illuminated with n spotlights. Each spotlight illuminates a convex area.
If any one spotlight is turned off, the arena is still fully illuminated.
If any two spotlights are turned off, the arena will not be fully illuminated.
This is only possible for n >= 2.
-/
theorem spotlight_illumination (n : ℕ) (h1 : n ≥ 2)
    (h2 : ∀ i : fin n, is_convex (spotlight i))
    (h3 : ∀ i : fin n, illuminated_with_one_off i)
    (h4 : ∀ i j : fin n, i ≠ j → ¬illuminated_with_two_off i j) :
    n ≥ 2 :=
by
  exact h1

end spotlight_illumination_l354_354909


namespace arithmetic_seq_product_of_first_two_terms_l354_354227

theorem arithmetic_seq_product_of_first_two_terms
    (a d : ℤ)
    (h1 : a + 4 * d = 17)
    (h2 : d = 2) :
    (a * (a + d) = 99) := 
by
    -- Proof to be done
    sorry

end arithmetic_seq_product_of_first_two_terms_l354_354227


namespace trailing_zeros_100_factorial_l354_354794

-- Define the function to count the number of times a number can be divided by p
def count_prime_factors (n p : Nat) : Nat :=
  if n < p then 0
  else n / p + count_prime_factors (n / p) p

-- Define the problem statement
theorem trailing_zeros_100_factorial : count_prime_factors 100 5 = 24 := by
  sorry

end trailing_zeros_100_factorial_l354_354794


namespace customers_sampling_candy_l354_354291

theorem customers_sampling_candy (total_customers caught fined not_caught : ℝ) 
    (h1 : total_customers = 100) 
    (h2 : caught = 0.22 * total_customers) 
    (h3 : not_caught / (caught / 0.9) = 0.1) :
    (not_caught + caught) / total_customers = 0.2444 := 
by sorry

end customers_sampling_candy_l354_354291


namespace tangent_line_at_one_range_of_a_l354_354183

noncomputable def f (x a : ℝ) : ℝ := ((1 - a) * x^2 - a * x + a) / real.exp x

theorem tangent_line_at_one (a : ℝ) (ha : a = 1) :
  let tangent_line := λ x y : ℝ, x + real.exp(1) * y - 1
  in tangent_line 1 (f 1 1) = 0 := 
begin
  sorry
end

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x → (1 - a) * x^2 - a * x + a ≤ a * real.exp x) :
  a ∈ Set.Ici (4 / (real.exp 2 + 5)) :=
begin
  sorry
end

end tangent_line_at_one_range_of_a_l354_354183


namespace f_eq_g_if_g_monotone_l354_354034

-- Define the conditions
variables {f g : ℝ → ℝ}
variable h_f_continuous : Continuous f
variable h_lim_exists : ∀ (a b c : ℝ), a < b → b < c → ∃ (x_n : ℕ → ℝ), (∀ n, x_n n ∈ ℝ) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, |x_n n - b| < ε) ∧ (∃ L, (tendsto (λ n, g(x_n n)) at_top (𝓝 L)) ∧ (f(a) < L ∧ L < f(c)))

-- The theorem to prove
theorem f_eq_g_if_g_monotone (h_g_monotone : Monotone g) : f = g :=
by
  sorry

end f_eq_g_if_g_monotone_l354_354034


namespace sin_double_angle_l354_354734

variables (α : ℝ)

-- Defining the condition that α is an acute angle
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Given conditions
def given_conditions : Prop :=
  is_acute α ∧ cos (α + π / 6) = 4 / 5

-- The theorem to prove
theorem sin_double_angle (α : ℝ) (h : given_conditions α) : 
  sin (2 * α + π / 3) = 24 / 25 := 
by 
  -- proof omitted
  sorry

end sin_double_angle_l354_354734


namespace luke_total_points_l354_354528

/-- Luke gained 327 points in each round of a trivia game. 
    He played 193 rounds of the game. 
    How many points did he score in total? -/
theorem luke_total_points : 327 * 193 = 63111 :=
by
  sorry

end luke_total_points_l354_354528


namespace count_n_grids_correct_l354_354725

variable {m k n : ℕ}

-- Define the grid and n-grid conditions
def grid (m n : ℕ) := matrix (fin m) (fin n) bool

def is_n_grid (g : grid m n) (n : ℕ) : Prop :=
  ∃ reds : fin m → fin n, function.injective reds ∧
  (∀ i, reds i < n) ∧
  (∀ (i : fin (m*n - (k-1))), 
    ∑ j in finset.range k, ite (g (fin.floor (i + j).val) (fin.mod (i+j).val) = ff) 1 0 < k) ∧
  (∀ (i : fin (m*n - (m-1))), 
    ∑ j in finset.range m, ite (g (fin.floor (i + j).val) (fin.mod (i+j).val) = ff) 1 0 < m)

-- Function f(n) to count n-grids
def count_n_grids (n : ℕ) : ℕ := n!

theorem count_n_grids_correct (n m k : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_pos_k : 0 < k) : 
  (∀ g : grid m k, is_n_grid g n) → count_n_grids n = n! :=
sorry

end count_n_grids_correct_l354_354725


namespace Andrew_has_5_more_goats_than_twice_Adam_l354_354653

-- Definitions based on conditions
def goats_Adam := 7
def goats_Ahmed := 13
def goats_Andrew := goats_Ahmed + 6
def twice_goats_Adam := 2 * goats_Adam

-- Theorem statement
theorem Andrew_has_5_more_goats_than_twice_Adam :
  goats_Andrew - twice_goats_Adam = 5 :=
by
  sorry

end Andrew_has_5_more_goats_than_twice_Adam_l354_354653


namespace find_matrix_M_l354_354971
    
open Matrix

-- Given definitions and assumptions
variables {M : Matrix (Fin 2) (Fin 2) ℝ}
variables {e : Fin 2 → ℝ} {v : Fin 2 → ℝ} {w : Fin 2 → ℝ}

def eigenvector_and_value_of_M (eigenval : ℝ) : Prop :=
  M.mul_vec e = eigenval • e

def transformed_point (point : Fin 2 → ℝ) (result : Fin 2 → ℝ) : Prop :=
  M.mul_vec point = result

-- Vectors given in the conditions
def e : Fin 2 → ℝ := ![1, 1]
def point : Fin 2 → ℝ := ![-1, 2]
def result : Fin 2 → ℝ := ![9, 15]

-- Proof that M satisfies the given conditions
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  eigenvector_and_value_of_M 3 ∧ transformed_point point result →
  M = !![-1, 4, -3, 6] :=
begin
  sorry
end

end find_matrix_M_l354_354971


namespace inverse_function_symmetry_l354_354112

theorem inverse_function_symmetry (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x = y ↔ x = log a y) :=
by sorry

end inverse_function_symmetry_l354_354112


namespace min_expense_l354_354774

noncomputable def tile_area (diagonal_length: ℝ) : ℝ :=
  let side := diagonal_length / real.sqrt 2 in
  side * side

noncomputable def total_cost (num_large_tiles num_small_tiles: ℕ) (large_tile_price small_tile_price: ℝ) (large_discount small_discount: ℝ) : ℝ :=
  let large_cost := if num_large_tiles > 700 then num_large_tiles * large_tile_price * large_discount else num_large_tiles * large_tile_price in
  let small_cost := if num_small_tiles > 1000 then num_small_tiles * small_tile_price * small_discount else num_small_tiles * small_tile_price in
  large_cost + small_cost

theorem min_expense (floor_area: ℝ) (diag_large diag_small large_price small_price: ℝ) (l_discount s_discount: ℝ)
                    (comp_A_small_discount more_than_1000 : ℝ) (comp_B_total_discount more_than_700 : ℝ) : 
  floor_area = 100 →
  diag_large = 0.5 →
  diag_small = 0.4 →
  large_price = 0.8 →
  small_price = 0.6 →
  l_discount = 0.9 → 
  s_discount = 0.7 →
  comp_A_small_discount = 128 → -- additional area covered by large tiles
  comp_B_total_discount = 0.8 → 
  (purchase_plan: string) : (purchase_plan = "C") :=
by
  let area_large := tile_area diag_large
  let area_small := tile_area diag_small
  let num_large := floor_area / area_large
  let num_small := floor_area / area_small
  let cost_A := total_cost (floor_area / area_small) 1000 large_price small_price l_discount s_discount
  let cost_B := total_cost (floor_area / 0.125) 1167 large_price small_price 1.0 comp_B_total_discount
  if cost_A < cost_B then
    exact purchase_plan = "C"
  else
    sorry -- Handling of remaining potential comparisons


end min_expense_l354_354774


namespace mudit_age_l354_354533

theorem mudit_age :
    ∃ x : ℤ, x + 16 = 3 * (x - 4) ∧ x = 14 :=
by
  use 14
  sorry -- Proof goes here

end mudit_age_l354_354533


namespace isosceles_triangle_base_length_l354_354921

theorem isosceles_triangle_base_length (s a b : ℕ) (h1 : 3 * s = 45)
  (h2 : 2 * a + b = 40) (h3 : a = s) : b = 10 :=
by
  sorry

end isosceles_triangle_base_length_l354_354921


namespace actual_total_discount_discount_difference_l354_354001

variable {original_price : ℝ}
variable (first_discount second_discount claimed_discount actual_discount : ℝ)

-- Definitions based on the problem conditions
def discount_1 (p : ℝ) : ℝ := (1 - first_discount) * p
def discount_2 (p : ℝ) : ℝ := (1 - second_discount) * discount_1 first_discount p

-- Statements we need to prove
theorem actual_total_discount (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70) :
  actual_discount = 1 - discount_2 first_discount second_discount original_price := 
by 
  sorry

theorem discount_difference (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70)
  (actual_discount : ℝ := 0.58) :
  claimed_discount - actual_discount = 0.12 := 
by 
  sorry

end actual_total_discount_discount_difference_l354_354001


namespace cosine_of_angle_BHD_l354_354129

-- Definitions of the problem conditions
variables (solid : Type) [rectangular_solid solid]
variables (D H G F B : solid)
variable (angle_DHG : angle D H G = 30)
variable (angle_FHB : angle F H B = 45)

-- Lean statement to prove the cosine of angle_BHD
theorem cosine_of_angle_BHD :
  cos (angle B H D) = (√2) / 12 := 
sorry

end cosine_of_angle_BHD_l354_354129


namespace integer_solutions_system_eqns_l354_354053

theorem integer_solutions_system_eqns (x y z t : ℤ) :
  (xz - 2yt = 3 ∧ xt + yz = 1) ↔
  (x, y, z, t) = (1, 0, 3, 1) ∨
  (x, y, z, t) = (-1, 0, -3, -1) ∨
  (x, y, z, t) = (3, 1, 1, 0) ∨
  (x, y, z, t) = (-3, -1, -1, 0) :=
by sorry

end integer_solutions_system_eqns_l354_354053


namespace square_area_12_5_l354_354487

noncomputable def area_square (s : ℝ) : ℝ := s * s

theorem square_area_12_5
    (P Q R : EuclideanGeometry.Point ℝ)
    (A B C D : EuclideanGeometry.Point ℝ)
    (BR PR : ℝ)
    (BR_angle: EuclideanGeometry.angle B R A = EuclideanGeometry.angle.Bisect R A D 45)
    (h1: EuclideanGeometry.is_square A B C D)
    (h2: EuclideanGeometry.is_on_line P (EuclideanGeometry.line A D))
    (h3: EuclideanGeometry.is_on_line Q (EuclideanGeometry.line A B))
    (h4: EuclideanGeometry.orthogonal (EuclideanGeometry.line B P) (EuclideanGeometry.line C Q))
    (h5: EuclideanGeometry.distance B R = BR)
    (h6: EuclideanGeometry.distance P R = PR)
    (h7: BR = 3)
    (h8: PR = 4) :
  area_square (EuclideanGeometry.distance A B) = 12.5 :=
by
  sorry

end square_area_12_5_l354_354487


namespace problem_100th_term_of_seq_l354_354661

theorem problem_100th_term_of_seq (f : ℕ → ℕ) (n : ℕ) :
  (∀ n, ∃ k : ℕ, f n = k) ∧ (∀ m < n, f m < f n) →
  f 100 = 981 :=
by
  sorry

end problem_100th_term_of_seq_l354_354661


namespace find_a_of_complex_roots_and_abs_sum_l354_354435

open Complex

theorem find_a_of_complex_roots_and_abs_sum (a : ℝ) (x1 x2 : ℂ)
  (h1 : x1^2 - 2 * x1 * a + (a^2 - 4 * a + 4) = 0)
  (h2 : x2^2 - 2 * x2 * a + (a^2 - 4 * a + 4) = 0)
  (h3 : x1 ≠ x2) -- ensures distinct but still conjugates automatically by being complex
  (h4 : |x1| + |x2| = 3) : a = 1 / 2 := 
sorry

end find_a_of_complex_roots_and_abs_sum_l354_354435


namespace max_m_circumcenter_m_le_half_BC_le_M_l354_354396

-- Part (1)
theorem max_m_circumcenter (A B C : Point) (T : Point) [Inside_Triangle T A B C] : 
  T = circumcenter A B C → m(T) := 
  sorry

-- Part (2)
theorem m_le_half_BC_le_M (A B C : Point) (T : Point) [Inside_Triangle T A B C] :
  ∠BAC ≥ 90° → m(T) ≤ BC/2 ∧ BC/2 ≤ M(T) :=
  sorry

end max_m_circumcenter_m_le_half_BC_le_M_l354_354396


namespace solve_for_a_l354_354326

theorem solve_for_a (x a : ℝ) (hx_pos : 0 < x) (hx_sqrt1 : x = (a+1)^2) (hx_sqrt2 : x = (a-3)^2) : a = 1 :=
by
  sorry

end solve_for_a_l354_354326


namespace wax_needed_l354_354773

theorem wax_needed (total_wax_required wax_available : ℕ) (h_total : total_wax_required = 574) (h_available : wax_available = 557) : total_wax_required - wax_available = 17 :=
by {
  rw [h_total, h_available],
  norm_num
}

end wax_needed_l354_354773


namespace center_circle_sum_l354_354705

theorem center_circle_sum (h k : ℝ) :
  (∃ h k, (h, k) = (3, -4)) → h + k = -1 :=
by
  intro h k
  sorry

end center_circle_sum_l354_354705


namespace tenth_term_sequence_is_9_l354_354563

-- Definition of the sequence
def a : ℕ → ℕ
| 0       := 3^2012
| (n + 1) := (a n).digits.sum

-- Condition: each term is divisible by 9
def divBy9 (n : ℕ) : Prop :=
  a n % 9 = 0

-- Theorem: The 10th term of the sequence is 9
theorem tenth_term_sequence_is_9 : a 10 = 9 :=
sorry

end tenth_term_sequence_is_9_l354_354563


namespace solve_for_x_l354_354454

theorem solve_for_x (x : ℝ) (hx : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354454


namespace hexagon_perimeter_l354_354294

theorem hexagon_perimeter (side_length : ℝ) (sides : ℕ) (h_sides : sides = 6) (h_side_length : side_length = 10) :
  sides * side_length = 60 :=
by
  rw [h_sides, h_side_length]
  norm_num

end hexagon_perimeter_l354_354294


namespace log_expression_equals_range_of_m_l354_354306

-- Statement for the first part: calculating the given logarithmic expression
theorem log_expression_equals :
  logb 3 (427 / 3) + logb 10 25 + logb 10 4 + logb 7 (7^2) + (logb 2 3) * (logb 3 4) = 23 / 4 :=
  sorry

-- Definition of set A
def A := {x : ℝ | (1 / 32) ≤ 2^(-x) ∧ 2^(-x) ≤ 4}

-- Definition of set B
def B (m : ℝ) := {x : ℝ | m - 1 < x ∧ x < 2 * m + 1}

-- Statement for the second part: range of m for which A ∪ B = A
theorem range_of_m (m : ℝ) :
  (A ∪ B m = A) ↔ (m ≤ -2 ∨ (-1 ≤ m ∧ m ≤ 3 / 2)) :=
  sorry

end log_expression_equals_range_of_m_l354_354306


namespace coloring_impossible_l354_354840

theorem coloring_impossible :
  ¬ ∃ (coloring : ℕ × ℕ → Prop),
    (∀ i j, 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 → 
      (∑ x in finset.range 3, ∑ y in finset.range 3, if coloring (i + x, j + y) then 1 else 0) = 5) ∧
    (∀ i' j', 0 ≤ i' ∧ i' ≤ 6 ∧ 0 ≤ j' ∧ j' ≤ 7 → 
      (∑ x in finset.range 2, ∑ y in finset.range 4, if coloring (i' + x, j' + y) then 1 else 0) = 4) ∧
    (∀ i'' j'', 0 ≤ i'' ∧ i'' ≤ 7 ∧ 0 ≤ j'' ∧ j'' ≤ 6 → 
      (∑ x in finset.range 4, ∑ y in finset.range 2, if coloring (i'' + x, j'' + y) then 1 else 0) = 4) := 
  sorry

end coloring_impossible_l354_354840


namespace quadratic_function_expression_l354_354223

theorem quadratic_function_expression :
    ∃ a b c : ℝ, 
    (∀ x : ℝ, (-1/3)*(x - 3)^2 - 1 = a*x^2 + b*x + c) ∧ 
    (c = -4) ∧ 
    (a = (-1/3)) ∧ 
    b = 2 :=
begin
  sorry
end

end quadratic_function_expression_l354_354223


namespace proof_problem_l354_354075

noncomputable def real_numbers (a x y : ℝ) (h₁ : 0 < a ∧ a < 1) (h₂ : a^x < a^y) : Prop :=
  x^3 > y^3

-- The theorem statement
theorem proof_problem (a x y : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a^x < a^y) : x^3 > y^3 :=
by
  sorry

end proof_problem_l354_354075


namespace compound_interest_rate_l354_354253

def SI (P r t : ℝ) : ℝ := (P * r * t) / 100

def CI (P R t : ℝ) : ℝ := P * (1 + R / 100) ^ t - P

theorem compound_interest_rate (
  P1 P2 SI_val CI_factor SI_coef t_rate : ℝ
) (hP1 : P1 = 2625.0000000000027)
  (hP2 : P2 = 4000)
  (hSI_val : SI_val = 420)
  (hSI : SI P1 8 2 = SI_val)
  (hCI : CI P2 t_rate 2 = CI_factor)
  (h_condition : SI_val = 1 / 2 * CI_factor) :
  t_rate = 10 := 
sorry

end compound_interest_rate_l354_354253


namespace pq_over_ef_l354_354544

/-- Helper definitions for points and distances in a plane -/
structure Point where
  x : ℝ
  y : ℝ

def distance (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def line_eqn (P Q : Point) (x : ℝ) : ℝ :=
  let slope := (Q.y - P.y) / (Q.x - P.x)
  slope * (x - P.x) + P.y

/-- Main problem statement -/
theorem pq_over_ef :
  let A := Point.mk 0 3
  let B := Point.mk 6 3
  let C := Point.mk 6 0
  let D := Point.mk 0 0
  let E := Point.mk 4 3
  let F := Point.mk 1 0
  let G := Point.mk 6 1
  let P := Point.mk (8/3) (5/3)
  let Q := Point.mk 3 2
  let EF := distance E F
  let PQ := distance P Q
  PQ / EF = 1 / (9 * real.sqrt 2) := by
  sorry

end pq_over_ef_l354_354544


namespace find_b_magnitude_l354_354406

open EuclideanSpace.RealInnerProductSpace

variables (a b : ℝ^n)
variables (θ : ℝ)

axiom condition1 : ‖a‖ = 1
axiom condition2 : θ = real.pi / 3
axiom condition3 : inner (a + (2 : ℝ) • b) a = 3

noncomputable def cos_pi_over_3 : ℝ := real.cos (real.pi / 3)

theorem find_b_magnitude : (∀ a b : ℝ^n, θ = real.pi / 3 → ‖a‖ = 1 → inner (a + (2:ℝ)•b) a = 3 → ‖b‖ = 2) :=
by
  intros a b θ hθ h_norm_a h_inner_ab
  sorry

end find_b_magnitude_l354_354406


namespace smallest_z_l354_354956

theorem smallest_z 
  (x y z : ℕ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x + y = z) 
  (h2 : x * y < z^2) 
  (ineq : (27^z) * (5^x) > (3^24) * (2^y)) :
  z = 10 :=
by
  sorry

end smallest_z_l354_354956


namespace number_of_correct_propositions_l354_354593

noncomputable theory

variables {V : Type*} [normed_add_comm_group V] [inner_product_space ℝ V]
variables (a b c : V) (AB BC : V)
variables {ABC : Type*} [triangle ABC AB BC]

def proposition1 (a b c : V) : Prop :=
  (a • b = b • c ∧ b ≠ 0) → a = c

def proposition2 (a b : V) : Prop :=
  (a • b = 0) → (a = 0 ∨ b = 0)

def proposition3 (AB BC : V) : Prop :=
  (AB • BC > 0) → ∃ (ABC : triangle ABC AB BC), is_obtuse ABC

def proposition4 (AB BC : V) : Prop :=
  (AB • BC = 0) → ∃ (ABC : triangle ABC AB BC), is_right ABC

-- Main statement
theorem number_of_correct_propositions : (count_correct_propositions [proposition1 a b c, proposition2 a b, proposition3 AB BC, proposition4 AB BC]) = 2 :=
sorry

end number_of_correct_propositions_l354_354593


namespace simplify_expression_l354_354355

theorem simplify_expression (a b : ℤ) : 4 * a + 5 * b - a - 7 * b = 3 * a - 2 * b :=
by
  sorry

end simplify_expression_l354_354355


namespace bisection_method_root_interval_l354_354172

def f : ℝ → ℝ := λ x, 3 * x ^ 2 + 3 * x - 8

theorem bisection_method_root_interval :
  (f 1 < 0) →
  (f 1.5 > 0) →
  (f 1.25 < 0) →
  ∃ c ∈ (1.25, 1.5), f c = 0 := by
  intros h1 h1_5 h1_25
  use [sorry, sorry]
  sorry

end bisection_method_root_interval_l354_354172


namespace people_on_bus_before_stop_l354_354014

variable (P_before P_after P_got_on : ℕ)
variable (h1 : P_got_on = 13)
variable (h2 : P_after = 17)

theorem people_on_bus_before_stop : P_before = 4 :=
by
  -- Given that P_after = 17 and P_got_on = 13
  -- We need to prove P_before = P_after - P_got_on = 4
  sorry

end people_on_bus_before_stop_l354_354014


namespace quadratic_complete_square_l354_354581

theorem quadratic_complete_square :
  ∃ d e, (∀ x : ℝ, x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ (e / d = -1298) :=
by
  use 1300, -1687400
  split
  { intro x
    sorry }
  { sorry }

end quadratic_complete_square_l354_354581


namespace g_cycle_l354_354865

def g (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem g_cycle : g(g(g(g(g(g(0)))))) = 1 :=
by
  sorry

end g_cycle_l354_354865


namespace max_value_sin2A_tan2C_l354_354085

theorem max_value_sin2A_tan2C (A B C a b c : ℝ)
  (h1 : B ∈ (0, π))
  (h2 : A + C = π - B)
  (h3 : -c * cos B = (sqrt 2 * a * cos B + sqrt 2 * b * cos A) / 2) :
∃ x, x = 3 - 2 * sqrt 2 ∧ (∀ y, y = sin (2 * A) * (tan C) ^ 2 → y ≤ x) := 
begin
  sorry
end

end max_value_sin2A_tan2C_l354_354085


namespace minimize_distance_sum_l354_354243

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 |> Real.sqrt

def A : point := (3, 6)
def B : point := (6, 2)
def C (k : ℝ) : point := (k, 0)

theorem minimize_distance_sum : 
  ∃ k : ℝ, k = 6.75 ∧ 
    ∀ k' : ℝ, (distance A (C k) + distance B (C k)) ≤ (distance A (C k') + distance B (C k')) := 
sorry

end minimize_distance_sum_l354_354243


namespace narration_per_disc_l354_354313

theorem narration_per_disc (total_time : ℕ) (disc_capacity : ℕ) (num_discs : ℕ) (even_distribution : ℕ) :
  total_time = 480 ∧ disc_capacity = 70 ∧ num_discs = nat.ceil (480 / 70) ∧ even_distribution = 480 / num_discs → 
  even_distribution = 480 / 7 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end narration_per_disc_l354_354313


namespace find_smallest_n_l354_354055

def graph (V : Type*) := V → V → Prop

noncomputable def smallest_n : ℕ := 16

theorem find_smallest_n (n : ℕ) (G : graph (fin n)) :
  n > 4 →
  (∀ (u v w : fin n), G u v → G v w → G u w → false) →
  (∀ (u v : fin n), ¬ G u v → ∃ (x y : fin n), G u x ∧ G u y ∧ G v x ∧ G v y) →
  n = smallest_n :=
sorry

end find_smallest_n_l354_354055


namespace ellipse_parameters_sum_l354_354007

theorem ellipse_parameters_sum (h k a b : ℝ) (H_h : h = 3) (H_k : k = -2) (H_a : a = 7) (H_b : b = 4) :
  h + k + a + b = 12 :=
by
  rw [H_h, H_k, H_a, H_b]
  simp
  rfl

end ellipse_parameters_sum_l354_354007


namespace A_beats_B_by_distance_l354_354972

def distanceCoveredByA := 4.5 -- kilometers
def timeTakenByA := (1 * 60) + 30 -- seconds
def timeTakenByB := 3 * 60 -- seconds
def timeDifference := timeTakenByB - timeTakenByA

theorem A_beats_B_by_distance :
  (distanceCoveredByA / timeTakenByA) * timeDifference = 4.5 :=
by
  sorry

end A_beats_B_by_distance_l354_354972


namespace complex_addition_l354_354453

theorem complex_addition : 
  let A := Complex.mk 2 1
  let O := Complex.mk (-2) 2
  let P := Complex.mk 0 3
  let S := Complex.mk 1 3
in (A - O + P + S) = Complex.mk 5 5 := 
by
  sorry

end complex_addition_l354_354453


namespace correct_options_l354_354420

-- Define the conditions
def condition1 := ∀ x, x ≥ 1 → (y = (x - 2) / (2 * x + 1)) → (y ≥ -1/3 ∧ y < 1/2)
def condition2 := ∀ f, (∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ f_x, f_x = f(2*x - 1)) →
  (∀ y, y = f(x - 1) / (Mathlib.sqrt (x - 1)) → 1 < x ∧ x ≤ 2)
def condition3 := ∀ A, A ⊆ ℝ → (∀ f, (∀ x ∈ A, f x = x^2) → (∃ B, B = {4})) →
  (∃ f1 f2 f3, (f1 = λ x, x = 2) ∧ (f2 = λ x, x = -2) ∧ (f3 = λ x, x = 2 ∨ x = -2))
def condition4 := ∀ f, (∀ x, f (x + 1/x) = x^2 + 1/x^2) → (f m = 4 → m = Mathlib.sqrt 6)

-- The final theorem statement combining all conditions
theorem correct_options:
  ∀ A B C D, 
    (condition1 A) ∧ 
    (condition2 B) ∧
    (condition3 C) ∧
    (condition4 D) → 
    (A ∧ B ∧ C ∧ ¬D) := 
begin
  intros,
  sorry
end

end correct_options_l354_354420


namespace common_chord_of_circles_l354_354561

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x + 2 * y = 0) :=
by
  sorry

end common_chord_of_circles_l354_354561


namespace cos_alpha_in_second_quadrant_l354_354804

-- Definitions of the conditions
variables {α : Real} -- α is a real number representing an angle
def is_second_quadrant (α : ℝ) := π/2 < α ∧ α < π 

def tan_alpha (α : ℝ) := Real.tan α = -5/12

-- Statement to be proven
theorem cos_alpha_in_second_quadrant (α : ℝ) (h1 : is_second_quadrant α) (h2 : tan_alpha α) : Real.cos α = -12/13 := 
by
  sorry

end cos_alpha_in_second_quadrant_l354_354804


namespace total_students_end_of_year_l354_354346

def M := 50
def E (M : ℕ) := 4 * M - 3
def H (E : ℕ) := 2 * E

def E_end (E : ℕ) := E + (E / 10)
def M_end (M : ℕ) := M - (M / 20)
def H_end (H : ℕ) := H + ((7 * H) / 100)

def total_end (E_end M_end H_end : ℕ) := E_end + M_end + H_end

theorem total_students_end_of_year : 
  total_end (E_end (E M)) (M_end M) (H_end (H (E M))) = 687 := sorry

end total_students_end_of_year_l354_354346


namespace min_expression_value_l354_354867

theorem min_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (z : ℝ) (h3 : x^2 + y^2 = z) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) = -2040200 :=
  sorry

end min_expression_value_l354_354867


namespace range_of_a_l354_354098

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 1 > 0) ∧ (∃ x : ℝ, x > 0 ∧ x^2 + x⁻² < a) → a > 2 := 
by
  sorry

end range_of_a_l354_354098


namespace rhombus_diagonal_difference_l354_354363

theorem rhombus_diagonal_difference (a d : ℝ) (h_a_pos : a > 0) (h_d_pos : d > 0):
  (∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2) ↔ d < 2 * a :=
sorry

end rhombus_diagonal_difference_l354_354363


namespace exist_two_people_with_same_number_of_friends_l354_354590

theorem exist_two_people_with_same_number_of_friends
  (n : ℕ) (h : n ≥ 2)
  (friendship : Fin n → Fin n → Prop)
  (symmetric_friendship : ∀ {x y : Fin n}, friendship x y → friendship y x)
  (no_self_friendship : ∀ {x : Fin n}, ¬ friendship x x) :
  ∃ (x y : Fin n), x ≠ y ∧ (∑ i, if friendship x i then 1 else 0) = ∑ i, if friendship y i then 1 else 0 :=
sorry

end exist_two_people_with_same_number_of_friends_l354_354590


namespace solve_for_x_l354_354430

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2 * x

theorem solve_for_x (x : ℝ) (h : f x = 5) : x = -2 ∨ x = 5 / 2 := by
  sorry

end solve_for_x_l354_354430


namespace maximum_non_attacking_rooks_on_50_cube_l354_354248

   theorem maximum_non_attacking_rooks_on_50_cube :
     ∃ (x y z : ℕ), 
       x + y + z = 75 ∧ 
       x + y ≤ 50 ∧ 
       x + z ≤ 50 ∧ 
       y + z ≤ 50 :=
   begin
     sorry
   end
   
end maximum_non_attacking_rooks_on_50_cube_l354_354248


namespace largest_angle_of_right_triangle_l354_354130

theorem largest_angle_of_right_triangle (a b : ℝ) (h1 : ∠A = 90) (h2 : 7 * b = 2 * a) :
  ∃ angle, angle = 90 :=
by
  -- Proof goes here
  sorry

end largest_angle_of_right_triangle_l354_354130


namespace Tyler_has_200_puppies_l354_354601

-- Define the number of dogs
def numDogs : ℕ := 25

-- Define the number of puppies per dog
def puppiesPerDog : ℕ := 8

-- Define the total number of puppies
def totalPuppies : ℕ := numDogs * puppiesPerDog

-- State the theorem we want to prove
theorem Tyler_has_200_puppies : totalPuppies = 200 := by
  exact (by norm_num : 25 * 8 = 200)

end Tyler_has_200_puppies_l354_354601


namespace solve_for_x_l354_354456

theorem solve_for_x (x : ℝ) (hx : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354456


namespace net_salary_change_l354_354927

variable (S : ℝ)

theorem net_salary_change : 
  let increased_salary := S * 1.2 in
  let final_salary := increased_salary * 0.8 in
  final_salary - S = -0.04 * S :=
by
  sorry

end net_salary_change_l354_354927


namespace shekar_average_marks_l354_354976

theorem shekar_average_marks :
  let marks := [76, 65, 82, 47, 85] in
  (marks.sum / marks.length) = 71 :=
by
  sorry

end shekar_average_marks_l354_354976


namespace moles_CO2_formed_l354_354450

def chemical_equation : Prop :=
  "NaHCO₃ + HCl → NaCl + H₂O + CO₂"

structure Reaction :=
  (NaHCO3 : ℕ) -- moles of Sodium bicarbonate
  (HCl : ℕ) -- moles of Hydrochloric acid
  (NaCl : ℕ) -- moles of Sodium chloride
  (H2O : ℕ) -- moles of Water
  (CO2 : ℕ) -- moles of Carbon dioxide

axiom balanced_reaction :
  ∀ r : Reaction, 
  r.NaHCO3 = r.HCl → -- NaHCO₃ and HCl react in 1:1 ratio
  r.NaCl = r.NaHCO3 ∧ -- NaCl forms in 1:1 ratio with NaHCO₃
  r.H2O = r.NaHCO3 ∧ -- H2O forms in 1:1 ratio with NaHCO₃
  r.CO2 = r.NaHCO3 -- CO₂ forms in 1:1 ratio with NaHCO₃

theorem moles_CO2_formed (r : Reaction) (h1 : r.NaHCO3 = 2) (h2 : r.HCl = 2) : r.CO2 = 2 :=
by
  apply balanced_reaction
  exact h1
  exact h2
  sorry

end moles_CO2_formed_l354_354450


namespace three_digit_numbers_divisible_by_5_l354_354790

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l354_354790


namespace min_value_a5_l354_354797

noncomputable def geometric_sequence (a: ℕ → ℝ) : Prop :=
∀ n m, ∃ q, q ≠ 0 ∧ a (n + 1) = a n * q 

theorem min_value_a5 (a: ℕ → ℝ) (h1: a 2 = 1) (h2: a 4 = 9) : 
  ∃ q : ℝ, q < 0 ∧ a 5 = -27 :=
by {
  -- Statements and further definitions necessary for the proof can be placed here
  -- This might involve defining how the indices of a affect the properties of q
  sorry
}

end min_value_a5_l354_354797


namespace length_of_first_race_l354_354477

theorem length_of_first_race (L : ℕ) :
  (∀ B_dist C_dist, B_dist = L - 20 ∧ C_dist = L - 38 → 
   ∀ B_C600, B_C600 = 60 → 
   ∀ B600 C600, B600 = 600 ∧ C600 = 540 → 
   (B_dist / C_dist = B600 / C600)) → 
  L = 200 :=
begin
  sorry
end

end length_of_first_race_l354_354477


namespace total_books_l354_354892

-- Define the conditions
def books_per_shelf : ℕ := 9
def mystery_shelves : ℕ := 6
def picture_shelves : ℕ := 2

-- The proof problem statement
theorem total_books : 
  (mystery_shelves * books_per_shelf) + 
  (picture_shelves * books_per_shelf) = 72 := 
sorry

end total_books_l354_354892


namespace polynomial_min_value_l354_354282

theorem polynomial_min_value (x : ℝ) : x = -3 → x^2 + 6 * x + 10 = 1 :=
by
  intro h
  sorry

end polynomial_min_value_l354_354282


namespace count_irreducible_fractions_even_l354_354207

theorem count_irreducible_fractions_even (n : ℕ) (h : n > 2) :
  (finset.card (finset.filter (λ k, Nat.gcd k n = 1) (finset.range n))) % 2 = 0 := 
sorry

end count_irreducible_fractions_even_l354_354207


namespace johnny_2008th_l354_354502

-- Define a function that checks if a number contains the digit 2
def contains_two (n : ℕ) : Bool :=
  n.digits 10 |> List.any (λ d => d = 2)

-- Define a function that generates the nth number in Johnny's sequence
-- Skipping numbers that contain a 2
def johnnys_sequence (n : ℕ) : ℕ :=
  (List.filter (λ x => ¬ contains_two x) (List.range (n*10))).nth_le n sorry

-- Define the statement proving that the 2008th number in Johnny's sequence is 3781
theorem johnny_2008th : johnnys_sequence 2008 = 3781 := 
  sorry

end johnny_2008th_l354_354502


namespace find_a_value_l354_354092

theorem find_a_value : 
  ∃ (a : ℝ), a > 0 ∧ (∀ (t : ℝ),
  let f := λ x => a * x^2 - 2014 * x + 2015,
      M := max (f (t - 1)) (f (t + 1)),
      N := min (f (t - 1)) (f (t + 1))
  in M - N = 1
) ∧ a = 403 := 
sorry

end find_a_value_l354_354092


namespace coffee_consumption_l354_354573

theorem coffee_consumption (h1 h2 g1 h3: ℕ) (k : ℕ) (g2 : ℕ) :
  (k = h1 * g1) → (h1 = 9) → (g1 = 2) → (h2 = 6) → (k / h2 = g2) → (g2 = 3) :=
by
  sorry

end coffee_consumption_l354_354573


namespace required_packages_for_school_lockers_l354_354648

def count_digit_usage (start end : Nat) (digit : Nat) : Nat :=
  (List.range' start (end - start + 1)).map (fun n => Nat.digits 10 n).join.count (= digit)

theorem required_packages_for_school_lockers : 
  let lockers_students_start := 250;
  let lockers_students_end := 325;
  let lockers_staff_start := 400;
  let lockers_staff_end := 425;
  let most_frequent_digit_usage := max 
    (max (count_digit_usage lockers_students_start lockers_students_end 2)
         (count_digit_usage lockers_students_start lockers_students_end 3))
    (max (count_digit_usage lockers_students_start lockers_students_end 4)
         (count_digit_usage lockers_staff_start lockers_staff_end 2));
  most_frequent_digit_usage = 50 :=
by
  sorry

end required_packages_for_school_lockers_l354_354648


namespace angle_MON_right_iff_angle_60_l354_354164

-- Let O be the circumcenter, M be the centroid, and N be the Nagel point of a triangle.
variables {O M N : Type}
variables (circumcenter : triangle → O)
variables (centroid : triangle → M)
variables (nagel_point : triangle → N)
variables (angle_MON_right : ∀ (triangle : Type), angle (centroid triangle) (circumcenter triangle) (nagel_point triangle) = 90)
variables (angle_60 : ∀ (triangle : Type), ∃ (A B C : Type), one_of_angle_60 A B C)

theorem angle_MON_right_iff_angle_60 (triangle : Type) :
  angle (centroid triangle) (circumcenter triangle) (nagel_point triangle) = 90 ↔
  ∃ (A B C : Type), one_of_angle_60 A B C :=
by sorry

end angle_MON_right_iff_angle_60_l354_354164


namespace geometric_prog_y_90_common_ratio_l354_354102

theorem geometric_prog_y_90_common_ratio :
  ∀ (y : ℝ), y = 90 → ∃ r : ℝ, r = (90 + y) / (30 + y) ∧ r = (180 + y) / (90 + y) ∧ r = 3 / 2 :=
by
  intros
  sorry

end geometric_prog_y_90_common_ratio_l354_354102


namespace min_val_16x_minus_4x_plus_4_l354_354377

theorem min_val_16x_minus_4x_plus_4 : ∃ x : ℝ, 16^x - 4^x + 4 = (15/4) :=
by
  sorry

end min_val_16x_minus_4x_plus_4_l354_354377


namespace find_S4_l354_354167

def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (2 * a(0) + (n - 1) * (a(1) - a(0) ))) / 2

theorem find_S4 {a : ℕ → ℝ} (a1_eq : a 1 = 3) (S3_eq : arithmetic_sequence_sum a 3 = 3) :
  arithmetic_sequence_sum a 4 = 0 :=
sorry

end find_S4_l354_354167


namespace volume_of_pyramid_l354_354546

-- Given: ABCDEF is a regular hexagon, which is the base of the right pyramid PABCDEF.
-- Given: PAD is an equilateral triangle with side length 8.
-- Prove: The volume of the pyramid PABCDEF is 96 cubic units.

theorem volume_of_pyramid (ABCDEF : Type) [IsRegularHexagon ABCDEF] (P : Point)
  (equilateral_triangle_PAD : EquilateralTriangle P (corner A) (corner D)) (hPAD : side_length P (corner A) = 8) :
  volume (PABCDEF) = 96 :=
sorry

end volume_of_pyramid_l354_354546


namespace evaluate_modulus_l354_354686

-- Define the complex number
def complex_number : ℂ := (7 / 4) + complex.I * 3

-- State the theorem to evaluate its modulus
theorem evaluate_modulus : abs complex_number = real.sqrt 193 / 4 :=
by
sory

end evaluate_modulus_l354_354686


namespace maximum_intersections_of_segments_l354_354598

theorem maximum_intersections_of_segments :
  let x_points := 12
  let y_points := 6
  let segments := x_points * y_points
  (∑ (i j : Fin x_points) in (Finset.offDiag (Finset.fin x_points)), ∑ (k l : Fin y_points) in (Finset.offDiag (Finset.fin y_points)), true) = 990 := sorry

end maximum_intersections_of_segments_l354_354598


namespace league_games_count_l354_354296

theorem league_games_count (n : ℕ) (h : n = 12) : (n * (n - 1) / 2) = 66 :=
by
  rw h
  simp
  sorry

end league_games_count_l354_354296


namespace shoes_count_l354_354632

def numberOfShoes (numPairs : Nat) (matchingPairProbability : ℚ) : Nat :=
  let S := numPairs * 2
  if (matchingPairProbability = 1 / (S - 1))
  then S
  else 0

theorem shoes_count 
(numPairs : Nat)
(matchingPairProbability : ℚ)
(hp : numPairs = 9)
(hq : matchingPairProbability = 0.058823529411764705) :
numberOfShoes numPairs matchingPairProbability = 18 := 
by
  -- definition only, the proof is not required
  sorry

end shoes_count_l354_354632


namespace no_real_solution_ineq_l354_354905

theorem no_real_solution_ineq (x : ℝ) (h : x ≠ 5) : ¬ (x^3 - 125) / (x - 5) < 0 := 
by
  sorry

end no_real_solution_ineq_l354_354905


namespace total_oranges_in_box_l354_354261

noncomputable def initial_oranges : ℝ := 55.0
noncomputable def added_oranges : ℝ := 35.0

theorem total_oranges_in_box : initial_oranges + added_oranges = 90.0 :=
by
  have total_oranges := initial_oranges + added_oranges
  show total_oranges = 90.0
  sorry

end total_oranges_in_box_l354_354261


namespace wall_height_eq_l354_354312

-- Define the dimensions of the brick
def brick_length := 50 -- in cm
def brick_width := 11.25 -- in cm
def brick_height := 6 -- in cm

-- Define the dimensions of the wall
def wall_length := 800 -- in cm
def wall_width := 22.5 -- in cm

-- Define the number of bricks
def num_bricks := 3200

-- Calculate the volume of one brick
def brick_volume := brick_length * brick_width * brick_height -- in cm³

-- Calculate the total volume of all bricks
def total_brick_volume := num_bricks * brick_volume -- in cm³

-- Theorem statement: Prove the height of the wall
theorem wall_height_eq :
  ∃ h : ℝ, total_brick_volume = wall_length * wall_width * h ∧ h = 600 :=
sorry -- we skip the proof for now

end wall_height_eq_l354_354312


namespace number_at_100th_position_is_2050_l354_354240

noncomputable def number_at_100th_position : ℕ :=
let sequence := list.range' 1951 (1982 - 1951 + 1) in
let larger_in_first_99 := list.foldl max (sequence.head!) (list.take 99 sequence) in
let smaller_in_last_1882 := list.foldl min (sequence.head!) (list.drop 99 sequence) in
if (larger_in_first_99 ≤ sequence.nth 99 ∧ sequence.nth 99 ≤ smaller_in_last_1882) then
  sequence.nth 99
else 
  0

theorem number_at_100th_position_is_2050 : number_at_100th_position = 2050 :=
by {
    sorry
}

end number_at_100th_position_is_2050_l354_354240


namespace percentage_students_received_A_l354_354260

def total_students : ℕ := 32
def failed_students : ℕ := 18
def not_failed_students : ℕ := total_students - failed_students

def A : ℝ := 
let eq := λ a : ℝ, a + (1 / 4) * (not_failed_students - a) = not_failed_students in
classical.some (Exists.intro 14 (by {dsimp [not_failed_students], norm_num, linarith }))

theorem percentage_students_received_A : (A / total_students) * 100 = 43.75 := 
by {
  -- Since A = 14 as calculated previously
  dsimp [A, total_students],
  norm_num,
}

end percentage_students_received_A_l354_354260


namespace function_value_l354_354627

theorem function_value (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f(x + 3) = -f(x + 1)) (h2 : f(3) = 2015) : f(f - 2) + 1 = -2014 :=
sorry

end function_value_l354_354627


namespace time_for_grid_5x5_l354_354482

-- Definition for the 3x7 grid conditions
def grid_3x7_minutes := 26
def grid_3x7_total_length := 4 * 7 + 8 * 3
def time_per_unit_length := grid_3x7_minutes / grid_3x7_total_length

-- Definition for the 5x5 grid total length
def grid_5x5_total_length := 6 * 5 + 6 * 5

-- Theorem stating that the time it takes to trace all lines of a 5x5 grid is 30 minutes
theorem time_for_grid_5x5 : (time_per_unit_length * grid_5x5_total_length) = 30 := by
  sorry

end time_for_grid_5x5_l354_354482


namespace maximum_value_expression_l354_354520

theorem maximum_value_expression (x y : ℝ) (h : x + y = 5) :
  ∃ p, p = x * y ∧ (4 * p^3 - 92 * p^2 + 754 * p) = 441 / 2 :=
by {
  sorry
}

end maximum_value_expression_l354_354520


namespace frequency_of_group_5_l354_354822

/-- Let the total number of data points be 50, number of data points in groups 1, 2, 3, and 4 be
  2, 8, 15, and 5 respectively. Prove that the frequency of group 5 is 0.4. -/
theorem frequency_of_group_5 :
  let total_data_points := 50
  let group1_data_points := 2
  let group2_data_points := 8
  let group3_data_points := 15
  let group4_data_points := 5
  let group5_data_points := total_data_points - group1_data_points - group2_data_points - group3_data_points - group4_data_points
  let frequency_group5 := (group5_data_points : ℝ) / total_data_points
  frequency_group5 = 0.4 := 
by
  sorry

end frequency_of_group_5_l354_354822


namespace find_angle_CDB_l354_354000

-- Define the entities involved
variables (C D B : Type) -- Points C, D, and B

-- Define angles
variable (angle_BCD : ℕ) -- Angle at BCD
variable (angle_DBC : ℕ) -- Angle at DBC
variable (angle_CDB : ℕ) -- Angle at CDB

-- Define values of angles given in the problem
def angle_D_eq_90 : Prop := angle_BCD = 90
def angle_B_eq_60 : Prop := angle_DBC = 60

-- Given conditions
axiom h1 : angle_D_eq_90
axiom h2 : angle_B_eq_60

-- Prove m∠CDB = 30 degrees
theorem find_angle_CDB : angle_CDB = 30 :=
by
  -- Use the given angles and solve by the triangle sum property.
  have h_sum : angle_BCD + angle_DBC + angle_CDB = 180 := by sorry
  sorry

end find_angle_CDB_l354_354000


namespace profit_per_unit_l354_354997

variable (a : ℝ)

def cost_price : ℝ := a
def set_price : ℝ := 1.5 * a
def selling_price : ℝ := 1.2 * a

theorem profit_per_unit : (selling_price - cost_price) = 0.2 * a :=
by
  sorry

end profit_per_unit_l354_354997


namespace sin_2_gamma_l354_354204

variables {E F G H Q : Type} [Point E] [Point F] [Point G] [Point H] [Point Q]

variables (EF FG GH : ℝ) (α β γ : ℝ)
variables (cos_EQG : ℝ) (cos_FQH : ℝ)

-- define the conditions
def equal_spacing : Prop := EF = FG ∧ FG = GH
def cos_angles : Prop :=
  cos_EQG = 3 / 5 ∧ cos_FQH = 5 / 13

-- final statement to prove
theorem sin_2_gamma {EF FG GH : ℝ} {cos_EQG cos_FQH : ℝ}
  (h1 : equal_spacing EF FG GH)
  (h2 : cos_angles cos_EQG cos_FQH) :
  sin (2 * γ) = 216 / 169 :=
sorry

end sin_2_gamma_l354_354204


namespace cups_of_flour_needed_l354_354531

-- Definitions and conditions
variables (total_scoops : ℕ) (scoops_per_cup : ℕ) (sugar_cups : ℕ)

-- Specific problem conditions
def total_scoops := 15
def scoops_per_cup := 3
def sugar_cups := 2

-- The proof statement
theorem cups_of_flour_needed :
  let sugar_scoops := sugar_cups * scoops_per_cup in
  let flour_scoops := total_scoops - sugar_scoops in
  (flour_scoops : ℚ) * (1 / 3) = 3 :=
by
  let sugar_scoops := sugar_cups * scoops_per_cup
  let flour_scoops := total_scoops - sugar_scoops
  have h1 : sugar_scoops = 6 := by
    unfold sugar_scoops
    unfold scoops_per_cup
    linarith
  have h2 : flour_scoops = 9 := by
    unfold flour_scoops
    rw [h1]
    linarith
  have h3 := (flour_scoops : ℚ) * (1 / 3)
  rw [←h2]
  norm_num
  sorry

end cups_of_flour_needed_l354_354531


namespace remainder_division_l354_354704

theorem remainder_division :
  (∀ x : ℝ, ∃ q : ℝ[X], (x^4 + 3 * x + 2) = ((x - 2)^2) * q + (32 * x - 30)) := by
  sorry

end remainder_division_l354_354704


namespace harvey_sold_17_steaks_l354_354772

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l354_354772


namespace frustum_lateral_area_l354_354926

def frustum_upper_base_radius : ℝ := 3
def frustum_lower_base_radius : ℝ := 4
def frustum_slant_height : ℝ := 6

theorem frustum_lateral_area : 
  (1 / 2) * (frustum_upper_base_radius + frustum_lower_base_radius) * 2 * Real.pi * frustum_slant_height = 42 * Real.pi :=
by
  sorry

end frustum_lateral_area_l354_354926


namespace trisha_spent_on_meat_l354_354944

theorem trisha_spent_on_meat :
  let initial_amount := 167
  let chicken_amount := 22
  let veggies_amount := 43
  let eggs_amount := 5
  let dog_food_amount := 45
  let amount_left := 35
  let total_other_expenses := chicken_amount + veggies_amount + eggs_amount + dog_food_amount
  let total_spent := initial_amount - amount_left
  total_spent - total_other_expenses = 17 := by
  let initial_amount := 167
  let chicken_amount := 22
  let veggies_amount := 43
  let eggs_amount := 5
  let dog_food_amount := 45
  let amount_left := 35
  let total_other_expenses := chicken_amount + veggies_amount + eggs_amount + dog_food_amount
  let total_spent := initial_amount - amount_left
  have : total_other_expenses = 115 := by
    sorry
  have : total_spent = 132 := by
    sorry
  show total_spent - total_other_expenses = 17 from calc
    total_spent - total_other_expenses = 132 - 115 : by sorry
                              ... = 17 : by sorry

end trisha_spent_on_meat_l354_354944


namespace number_of_perfect_square_factors_of_1200_l354_354776

theorem number_of_perfect_square_factors_of_1200 : 
  let factors_2 := {0, 2, 4}
  let factors_3 := {0}
  let factors_5 := {0, 2}
  3 * 1 * 2 = 6 := by {
    sorry
  }

end number_of_perfect_square_factors_of_1200_l354_354776


namespace cosine_of_acute_angle_l354_354317

theorem cosine_of_acute_angle 
  (line1 : ℝ × ℝ → ℝ × ℝ) 
  (line2 : ℝ × ℝ → ℝ × ℝ) 
  (dir1 : ℝ × ℝ := (4, 5)) 
  (dir2 : ℝ × ℝ := (2, 7))
  (p1 : ℝ × ℝ := (1, -1))
  (p2 : ℝ × ℝ := (-3, 9)) :
  ∃ θ : ℝ, 
  ∃ (cos_θ : ℝ), 
  (line1 = λ t, (p1.1 + t * dir1.1, p1.2 + t * dir1.2)) → 
  (line2 = λ u, (p2.1 + u * dir2.1, p2.2 + u * dir2.2)) → 
  cos_θ = (dir1.1 * dir2.1 + dir1.2 * dir2.2) / 
          (real.sqrt (dir1.1 ^ 2 + dir1.2 ^ 2) * real.sqrt (dir2.1 ^ 2 + dir2.2 ^ 2)) ∧
  cos_θ = 43 / (real.sqrt 41 * real.sqrt 53) := sorry

end cosine_of_acute_angle_l354_354317


namespace hospital_cost_minimization_l354_354639

theorem hospital_cost_minimization :
  ∃ (x y : ℕ), (5 * x + 6 * y = 50) ∧ (10 * x + 20 * y = 140) ∧ (2 * x + 3 * y = 23) :=
by
  sorry

end hospital_cost_minimization_l354_354639


namespace xiaoming_correct_answers_l354_354588

-- Define each statement's correctness as given in the conditions
-- Let’s make definitions to represent correctness assertions
def statement1 : Prop := ¬(opposite (-2/5) = -5/2)
def statement2 : Prop := ¬(reciprocal (-1) = 1)
def statement3 : Prop := ¬(4 - 7 = -3 → (4 = -3))
def statement4 : Prop := ¬((-3) + (-1/3) = 1)
def statement5 : Prop := ¬((-3)^2 = (-3)^2)

-- Main theorem stating that with the given conditions, Xiao Ming correctly answered two questions
theorem xiaoming_correct_answers : statement1 ∧ statement2 ∧ statement3 ∧ statement4 ∧ statement5 → 2 = 2 := by
  intro h,
  sorry

end xiaoming_correct_answers_l354_354588


namespace num_fish_l354_354298

theorem num_fish (num_fishbowls : ℕ) (fish_per_bowl : ℕ) (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) :
  num_fishbowls * fish_per_bowl = 6003 :=
by
  rw [h1, h2]
  norm_num
  sorry

end num_fish_l354_354298


namespace count_even_strictly_increasing_three_digit_integers_l354_354448

theorem count_even_strictly_increasing_three_digit_integers : 
  (∃ n : ℕ, n = 34 ∧
    (∃ a b c : ℕ, 
    (100 * a + 10 * b + c) = n ∧ 
    (1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9) ∧
    (c % 2 = 0))) :=
begin
  -- We know the answer is 34.
  use 34,
  split,
  { reflexivity },
  {
    -- We can now outline the possible values for (a, b, c) with c being 4, 6, or 8 and no zeros in digits.
    existsi [1, 2, 4],
    split,
    { 
      -- Check for a valid n
      norm_num,
      split,
      repeat { split },
      dec_trivial,
      dec_trivial,
    },
    -- Prove that the number has strictly increasing digits and that it's even (c = 4)
    sorry
  }
end

end count_even_strictly_increasing_three_digit_integers_l354_354448


namespace B_is_not_happy_M_is_happy_quadrant_l354_354738

noncomputable theory

def is_happy_point (m n : ℝ) : Prop :=
  2 * m = 8 + n

def B_is_happy (Bx By : ℝ) : Prop :=
  let n := By - 2 in
  2 * Bx = 8 + n

def M_is_happy (a : ℝ) : Prop :=
  let n := a - 3 in 
  2 * a = 8 + n

def M_quadrant (a : ℝ) : ℝ × ℝ :=
  (a, a - 1)

theorem B_is_not_happy : ¬ B_is_happy 4 5 :=
  sorry

theorem M_is_happy_quadrant (a : ℝ) (h : M_is_happy a) : 
  let p := M_quadrant a in
  p.1 > 0 ∧ p.2 > 0 :=
  sorry

end B_is_not_happy_M_is_happy_quadrant_l354_354738


namespace shortest_chord_through_M_l354_354409

noncomputable def point_M : ℝ × ℝ := (1, 0)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

theorem shortest_chord_through_M :
  (∀ x y : ℝ, circle_C x y → x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_l354_354409


namespace function_inequality_l354_354907

theorem function_inequality (f : ℝ+ → ℝ+) (h : ∀ x : ℝ+, f (f x) + x = f (2 * x)) :
  ∀ x : ℝ+, f x ≥ x :=
sorry

end function_inequality_l354_354907


namespace unit_digit_expression_l354_354349

theorem unit_digit_expression : 
  let expr := (2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) * (2^64+1)
  in (expr % 10) = 5 :=
by
  sorry

end unit_digit_expression_l354_354349


namespace find_center_radius_sum_l354_354057

theorem find_center_radius_sum :
    let x := x
    let y := y
    let a := 2
    let b := 3
    let r := 2 * Real.sqrt 6
    (x^2 - 4 * x + y^2 - 6 * y = 11) →
    (a + b + r = 5 + 2 * Real.sqrt 6) :=
by
  intros x y a b r
  sorry

end find_center_radius_sum_l354_354057


namespace radius_of_semicircle_l354_354242

theorem radius_of_semicircle (P : ℝ) (π_val : ℝ) (h1 : P = 162) (h2 : π_val = Real.pi) : 
  ∃ r : ℝ, r = 162 / (π + 2) :=
by
  use 162 / (Real.pi + 2)
  sorry

end radius_of_semicircle_l354_354242


namespace permutation_formula_l354_354324

theorem permutation_formula (n k : ℕ) (h : k ≤ n) :
  ∃ A : ℕ, A = List.prod (List.range' (n-k+1) k) ∧ A = ∏ i in finset.range k, (n - i) :=
sorry

end permutation_formula_l354_354324


namespace find_angle_A_l354_354495

variable (A B C : Type) [Triangle A B C]
variable (a b : ℝ) (B_angle: ℝ)

-- Conditions from the given problem
axiom a_value : a = sqrt 2
axiom b_value : b = sqrt 3
axiom B_value : B_angle = π / 3

-- Defining triangle ABC with sides a and b, opposite angles A and B respectively.
noncomputable def angle_A_in_triangle 
  (a b : ℝ) (B_angle: ℝ) : ℝ := 
  (λ a b B_angle, λ (_ : B_angle = π / 3) (_ : a = sqrt 2) (_ : b = sqrt 3), π/4) a b B_angle

theorem find_angle_A : angle_A_in_triangle a b B_angle = π / 4 := sorry

end find_angle_A_l354_354495


namespace base_of_second_fraction_l354_354114

theorem base_of_second_fraction (base : ℝ) (h1 : (1/2) ^ 16 * (1/base) ^ 8 = 1 / (18 ^ 16)): base = 81 :=
sorry

end base_of_second_fraction_l354_354114


namespace sequence_monotonic_increasing_iff_a1_lt_a2_l354_354152

theorem sequence_monotonic_increasing_iff_a1_lt_a2 {λ : ℝ} :
  (∀ n : ℕ, n > 0 → (n + 1)^2 + λ * (n + 1) > n^2 + λ * n) ↔ (1 + λ < 4 + 2 * λ) := 
by
  sorry

end sequence_monotonic_increasing_iff_a1_lt_a2_l354_354152


namespace area_of_parallelogram_l354_354168

-- Define the vectors
def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (14, -4)

-- Prove the area of the parallelogram
theorem area_of_parallelogram : 
  abs (v.1 * w.2 - v.2 * w.1) = 42 :=
by
  sorry

end area_of_parallelogram_l354_354168


namespace triangle_AXY_is_obtuse_l354_354916

-- Define the points A, B, C, D forming the triangular pyramid ABCD
variables (A B C D X Y : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X] [Nonempty Y]

-- Define the condition that the inscribed sphere touches face BCD at point X
axiom inscribed_sphere_touches_BCD_at_X : touches_insphere (face BCD) X

-- Define the condition that the exscribed sphere touches face BCD at point Y
axiom exscribed_sphere_touches_BCD_at_Y : touches_exsphere (face BCD) Y

-- Prove the triangle AXY is obtuse
theorem triangle_AXY_is_obtuse : obtuse_triangle A X Y :=
sorry

end triangle_AXY_is_obtuse_l354_354916


namespace smallest_percentage_boys_correct_l354_354846

noncomputable def smallest_percentage_boys (B : ℝ) : ℝ :=
  if h : 0 ≤ B ∧ B ≤ 1 then B else 0

theorem smallest_percentage_boys_correct :
  ∃ B : ℝ,
    0 ≤ B ∧ B ≤ 1 ∧
    (67.5 / 100 * B * 200 + 25 / 100 * (1 - B) * 200) ≥ 101 ∧
    B = 0.6 :=
by
  sorry

end smallest_percentage_boys_correct_l354_354846


namespace problem_l354_354522

def m (x : ℝ) : ℝ := (x + 2) * (x + 3)
def n (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 9

theorem problem (x : ℝ) : m x < n x :=
by sorry

end problem_l354_354522


namespace area_BDE_l354_354203

noncomputable def A : ℝ × ℝ × ℝ := (sqrt 2, sqrt 5, -2)
noncomputable def B : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def C : ℝ × ℝ × ℝ := (sqrt 2, sqrt 5, 2)
noncomputable def D : ℝ × ℝ × ℝ := (0, 0, 2)
noncomputable def E : ℝ × ℝ × ℝ := (0, 0, -2)

-- Define distance function in 3D space
def dist (P Q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Define the area of triangle given three vertices using the Heron's formula
def area_triangle (P Q R : ℝ × ℝ × ℝ) : ℝ :=
  let a := dist P Q
  let b := dist Q R
  let c := dist R P
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

-- Prove that the area of triangle BDE is 9
theorem area_BDE : area_triangle B D E = 9 :=
by
  sorry

end area_BDE_l354_354203


namespace count_3_digit_numbers_divisible_by_5_l354_354788

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354788


namespace divisor_iff_even_l354_354162

noncomputable def hasDivisor (k : ℕ) : Prop := 
  ∃ n : ℕ, n > 0 ∧ (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2

theorem divisor_iff_even (k : ℕ) (h : k > 0) : hasDivisor k ↔ (k % 2 = 0) :=
by
  sorry

end divisor_iff_even_l354_354162


namespace line_equation_l354_354641

theorem line_equation (x y : ℝ) : 
  ((y = 1 → x = 2) ∧ ((x,y) = (1,1) ∨ (x,y) = (3,5)))
  → (2 * x - y - 3 = 0) ∨ (x = 2) :=
sorry

end line_equation_l354_354641


namespace sum_of_primes_is_prime_l354_354244

theorem sum_of_primes_is_prime (P Q : ℕ) (hP : prime P) (hQ : prime Q) (hPQ1 : prime (P - Q)) (hPQ2 : prime (P + Q)) : prime (P + Q + (P - Q) + (P + Q)) :=
by
  sorry

end sum_of_primes_is_prime_l354_354244


namespace angle_ADC_eq_90_l354_354070

open EuclideanGeometry

variables {A B C D K N M : Point}

-- Assume A, B, C, D lie on a circle
variable (h_cyclic : CyclicQuadrilateral A B C D)

-- Assume rays AB and DC intersect at K
variable (h_intersect : ∃ K', Collinear A B K' ∧ Collinear D C K')

-- Assume N is the midpoint of KC, and M is the midpoint of AC
variable (h_mid_N : Midpoint K C N)
variable (h_mid_M : Midpoint A C M)

-- Assume B, D, N, and M lie on the same circle
variable (h_circ_NMBD : CyclicQuadrilateral B D N M)

theorem angle_ADC_eq_90 :
  ∠ A D C = 90 :=
sorry

end angle_ADC_eq_90_l354_354070


namespace c_not_parallel_b_l354_354744

-- Definitions of the conditions
def skew_lines (a b : Type) [has_no_intersections a b] [not_in_same_plane a b] := true
def parallel (x y : Type) [in_same_plane x y] [parallel_to x y] := true

-- Given conditions
variables (a b c : Type)
variable [skew_lines a b]
variable [parallel c a]

-- Statement to prove
theorem c_not_parallel_b : ¬ parallel c b :=
by sorry

end c_not_parallel_b_l354_354744


namespace august_has_five_thursdays_l354_354553

theorem august_has_five_thursdays (N : ℕ) (is_leap_year : Bool)
  (july_has_five_tuesdays : ∃ t : ℕ → Prop, (∀ d ∈ {1, 2, 3}, t d)) :
  ∃ f : ℕ → Prop, (∀ d ∈ {1, 2, 3}, f (d + 3)) := 
  sorry

end august_has_five_thursdays_l354_354553


namespace S_40_l354_354398

noncomputable def sequence_a : ℕ → ℤ
| 0       := 0
| 1       := 0
| (n + 2) := if n % 2 = 0 then sequence_a (n + 1) + (n + 2) else sequence_a (n + 1) - (n + 2)

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ k, sequence_a (k + 1))

theorem S_40 : S 40 = 440 := by
  sorry

end S_40_l354_354398


namespace expected_value_sum_of_three_marbles_l354_354796

def bag := {1, 2, 3, 4, 5, 6, 7}
def selected_subsets := {s : Set ℕ | s.card = 3 ∧ s ⊆ bag}

theorem expected_value_sum_of_three_marbles : 
  (∑ s in selected_subsets, (s.sum : ℚ)) / (selected_subsets.card : ℚ) = 12 :=
by sorry

end expected_value_sum_of_three_marbles_l354_354796


namespace distribute_awards_l354_354550

-- Definitions of the problem conditions
variable (A : Type) [Fintype A] [DecidableEq A]  -- Awards are of a finite type
variable (S : Type) [Fintype S] [DecidableEq S]  -- Students are of a finite type

-- Define the problem conditions
axiom six_awards : Fintype.card A = 6
axiom three_students : Fintype.card S = 3

-- Problem statement: There are 180 ways to distribute the awards.
theorem distribute_awards :
  ∑' (f : A → S), (∀ (s : S), 1 ≤ (f ⁻¹' {s}).toFinset.card) = 180 :=
sorry

end distribute_awards_l354_354550


namespace Dave_needs_31_gallons_l354_354680

noncomputable def numberOfGallons (numberOfTanks : ℕ) (height : ℝ) (diameter : ℝ) (coveragePerGallon : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_surface_area := lateral_surface_area * numberOfTanks
  let gallons_needed := total_surface_area / coveragePerGallon
  Nat.ceil gallons_needed

theorem Dave_needs_31_gallons :
  numberOfGallons 20 24 8 400 = 31 :=
by
  sorry

end Dave_needs_31_gallons_l354_354680


namespace projection_area_rectangular_board_l354_354012

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board_l354_354012


namespace min_frac_sum_l354_354402

theorem min_frac_sum (x y : ℝ) (hx : x > y) (hy : y > 0) (hxy : x + y = 1) : 
  ∃ c, c = 9 / 2 ∧ (∀ a, a = ∑ a in (λ z, z = 4 / (x + 3 * y) + 1 / (x - y)) → a ≥ c) :=
by
  sorry

end min_frac_sum_l354_354402


namespace returning_players_l354_354583

-- Definitions of conditions
def num_groups : Nat := 9
def players_per_group : Nat := 6
def new_players : Nat := 48

-- Definition of total number of players
def total_players : Nat := num_groups * players_per_group

-- Theorem: Find the number of returning players
theorem returning_players :
  total_players - new_players = 6 :=
by
  sorry

end returning_players_l354_354583


namespace porter_monthly_earnings_l354_354886

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end porter_monthly_earnings_l354_354886


namespace problem1_problem2_problem3_l354_354095

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

-- Prove f(x) + f(1-x) = 1/2
theorem problem1 (x : ℝ) : f(x) + f(1 - x) = 1 / 2 := sorry

-- Define sequence a_n = f(n/m)
noncomputable def a (m n : ℕ) (hm : m > 0) : ℝ := f(n / m)

-- Sum of the first m terms S_m
noncomputable def S_m (m : ℕ) : ℝ := ∑ k in finset.range m, a m (k+1) (nat.pos_of_ne_zero (nat.succ_pos k))

-- Prove S_m = (3m-1) / 12
theorem problem2 (m : ℕ) (hm : m > 0) : S_m m = (3 * m - 1) / 12 := sorry

-- Sequence b_n
noncomputable def b : ℕ → ℝ
| 0     := 1/3
| (n+1) := b n * (b n + 1)

-- Sum T_n
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (b (i + 1) + 1)

-- Maximum value of m such that S_m < T_n for all n ≥ 2
theorem problem3 (m n : ℕ) (hm : m > 0) (hn : n ≥ 2) (h : S_m m < T_n n) : m ≤ 6 := sorry

end problem1_problem2_problem3_l354_354095


namespace sum_of_arithmetic_sequence_l354_354858

noncomputable theory

variable (S : ℕ → ℤ)
variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := 
  ∀ n m : ℕ, a (n + m) = a n + m * d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem sum_of_arithmetic_sequence :
  ∃ d : ℤ, a 1 = -2015 ∧ is_arithmetic_sequence a d ∧
  (S 6 - 2 * S 3 = 18) ∧ 
  (S 2017 = 2017) := 
sorry

end sum_of_arithmetic_sequence_l354_354858


namespace ellipse_eccentricity_l354_354729

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = (4/3) * c) (h4 : a^2 - b^2 = c^2) : 
  c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l354_354729


namespace primes_less_than_200_with_ones_digit_3_l354_354451

theorem primes_less_than_200_with_ones_digit_3 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, Prime n ∧ n < 200 ∧ n % 10 = 3) ∧ S.card = 12 := 
by
  sorry

end primes_less_than_200_with_ones_digit_3_l354_354451


namespace find_range_of_a_l354_354871

-- Definition of p: For all x in ℝ, ax^2 > -ax - 1 holds
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a ≠ 0 → a * x^2 > -a * x - 1

-- Definition of q: The circles x^2 + y^2 = a^2 and (x + 3)^2 + (y - 4)^2 = 4 are externally disjoint
def distance( x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
def circle_externally_disjoint (r1 r2 d : ℝ) : Prop := d > r1 + r2
def prop_q (a : ℝ) : Prop :=
  let d := distance 0 0 (-3) 4 in circle_externally_disjoint a 2 d

-- Given p ∨ q is true and p ∧ q is false, find range of a
theorem find_range_of_a (a : ℝ) (p_or_q : prop_p a ∨ prop_q a) (p_and_q_false : ¬(prop_p a ∧ prop_q a)) :
  a ∈ Set.Icc (-3 : ℝ) (0 : ℝ) ∪ Set.Ico (3 : ℝ) (4 : ℝ) :=
sorry

end find_range_of_a_l354_354871


namespace porter_monthly_earnings_l354_354885

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end porter_monthly_earnings_l354_354885


namespace angle_terminal_side_equivalence_l354_354218

theorem angle_terminal_side_equivalence (k : ℤ) : 
    ∃ k : ℤ, 405 = k * 360 + 45 :=
by
  sorry

end angle_terminal_side_equivalence_l354_354218


namespace range_of_m_l354_354440

theorem range_of_m (m : ℝ) :
  (m + 4 - 4)*(2 + 2 * m - 4) < 0 → 0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l354_354440


namespace avg_hamburgers_per_day_l354_354646

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 49) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 7 :=
by {
  sorry
}

end avg_hamburgers_per_day_l354_354646


namespace tangency_proof_l354_354884

open EuclideanGeometry

variables {A B C D E M : Point}
variables [Triangle ABC] [IsAcute ABC]

def condition_1 (D : Point) := Angle D A C = Angle A C B
def condition_2 (D : Point) := Angle B D C = 90 + Angle B A C
def condition_3 (E : Point) (D : Point) := OnRay B D E ∧ Distance A E = Distance E C
def condition_4 (M : Point) := Midpoint M B C

theorem tangency_proof (h1 : Triangle ABC)
  (h2 : condition_1 D)
  (h3 : condition_2 D)
  (h4 : condition_3 E D)
  (h5 : condition_4 M) :
  Tangent (Line A B) (Circumcircle (Triangle B E M))
  sorry

end tangency_proof_l354_354884


namespace problem1_problem2_l354_354445

-- Definitions
def vec_a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)
def sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Problem 1: Prove the value of m such that vec_a ⊥ (vec_a - b(m))
theorem problem1 (m : ℝ) (h_perp: dot vec_a (sub vec_a (b m)) = 0) : m = -4 := sorry

-- Problem 2: Prove the value of k such that k * vec_a + b(-4) is parallel to vec_a - b(-4)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def parallel (u v : ℝ × ℝ) := ∃ (k : ℝ), scale k u = v

theorem problem2 (k : ℝ) (h_parallel: parallel (add (scale k vec_a) (b (-4))) (sub vec_a (b (-4)))) : k = -1 := sorry

end problem1_problem2_l354_354445


namespace triangle_count_l354_354065

theorem triangle_count (A B C : Point) (points : set Point) 
  (hABC : ∆ ABC)
  (h_points : points.card = 2005)
  (h_not_collinear : ∀ p ∈ points, ¬Collinear A B p ∧ ¬Collinear B C p ∧ ¬Collinear C A p):
  ∃ (triangles : set (Triangle)), triangles.card = (2008.choose 3) :=
by
  sorry

end triangle_count_l354_354065


namespace product_of_positive_c_for_rational_solutions_l354_354368

theorem product_of_positive_c_for_rational_solutions : 
  (∃ c₁ c₂ : ℕ, c₁ > 0 ∧ c₂ > 0 ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₁ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₂ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   c₁ * c₂ = 8) :=
sorry

end product_of_positive_c_for_rational_solutions_l354_354368


namespace distance_A_B_l354_354673

/-- Define a point in 3D Cartesian coordinate system -/
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- Distance formula in 3D space -/
noncomputable def distance (A B : Point3D) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2 + (B.z - A.z)^2)

/-- Define points A and B -/
def A : Point3D := {x := 1, y := 2, z := 3}
def B : Point3D := {x := -1, y := 3, z := -2}

/-- Prove the distance between A and B is sqrt(30) -/
theorem distance_A_B :
  distance A B = real.sqrt 30 :=
begin
  sorry
end

end distance_A_B_l354_354673


namespace max_knights_l354_354266

-- Definitions for the conditions
def Person : Type := ℕ -- Representing each person by a natural number
def people_count : ℕ := 30

-- Define knights and liars
def is_knight (p : Person) : Prop := -- to be defined
def is_liar (p : Person) : Prop := -- to be defined

-- Define neighbors relation
def are_neighbors (p q : Person) : Prop := abs p - q = 1 ∨ abs p - q = people_count - 1

-- Definitions for responses
def response_one (p : Person) : Prop := -- Person p answered "one"
def response_two (p : Person) : Prop := -- Person p answered "two"
def response_none (p : Person) : Prop := -- Person p answered "none"

-- Count responses
def count_responses (cond : Person → Prop) : ℕ :=
  Finset.filter cond (Finset.range people_count).card

-- The given conditions
axiom ten_response_one : count_responses response_one = 10
axiom ten_response_two : count_responses response_two = 10
axiom ten_response_none : count_responses response_none = 10

-- The theorem stating the formalized question
theorem max_knights : ∃ max_k : ℕ, max_k = 22 ∧ 
  (∀ p, (1 ≤ count_responses (λ p, is_knight p)) ≤ max_k) ∧ 
  count_responses (λ p, ¬is_knight p) = people_count - max_k :=
by
  sorry

end max_knights_l354_354266


namespace rotated_a_l354_354538

noncomputable def rotate_point_90_clockwise (origin : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
let translated_point := (point.1 - origin.1, point.2 - origin.2) in
let rotated_point := (translated_point.2, -translated_point.1) in
(rotated_point.1 + origin.1, rotated_point.2 + origin.2)

theorem rotated_a :
  rotate_point_90_clockwise (-2, 2) (-3, 2) = (-2, 3) :=
by
  -- need to provide proof here
  sorry

end rotated_a_l354_354538


namespace negation_of_existence_statement_l354_354571

theorem negation_of_existence_statement :
  ¬ (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 ≤ 0)) ↔ ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 > 0) :=
by
  sorry

end negation_of_existence_statement_l354_354571


namespace problem1_problem2_l354_354674

-- Problem 1
theorem problem1 (x : ℤ) : (x - 2) ^ 2 - (x - 3) * (x + 3) = -4 * x + 13 := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h₁ : x ≠ 1) : 
  (x^2 + 2 * x) / (x^2 - 1) / (x + 1 + (2 * x + 1) / (x - 1)) = 1 / (x + 1) := by 
  sorry

end problem1_problem2_l354_354674


namespace total_black_dots_in_2012_circles_l354_354333

-- Define the pattern of circles
def pattern (n : ℕ) : list ℕ := 
  (list.range (n + 1)).map (λ k => k + 1)

-- Define the function to count the number of ● in the first n circles
def count_black_dots (n : ℕ) : ℕ := 
  (list.range (n+1)).foldl (λ acc k => if (pattern k).sum ≤ 2012 then acc + 1 else acc) 0

-- Theorem stating that the number of ● in the first 2012 circles is 61
theorem total_black_dots_in_2012_circles : count_black_dots 2012 = 61 :=
sorry

end total_black_dots_in_2012_circles_l354_354333


namespace find_a6_l354_354142

open Nat

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def a2 := 4
def a4 := 2

theorem find_a6 (a1 d : ℤ) (h_a2 : arith_seq a1 d 2 = a2) (h_a4 : arith_seq a1 d 4 = a4) : 
  arith_seq a1 d 6 = 0 := by
  sorry

end find_a6_l354_354142


namespace matches_length_l354_354562

-- Definitions and conditions
def area_shaded_figure : ℝ := 300 -- given in cm^2
def num_small_squares : ℕ := 8
def large_square_area_coefficient : ℕ := 4
def area_small_square (a : ℝ) : ℝ := num_small_squares * a + large_square_area_coefficient * a

-- Question and answer to be proven
theorem matches_length (a : ℝ) (side_length: ℝ) :
  area_shaded_figure = 300 → 
  area_small_square a = area_shaded_figure →
  (a = 25) →
  (side_length = 5) →
  4 * 7 * side_length = 140 :=
by
  intros h1 h2 h3 h4
  sorry

end matches_length_l354_354562


namespace depreciation_rate_l354_354642

theorem depreciation_rate (initial_value final_value : ℝ) (years : ℕ) (r : ℝ)
  (h_initial : initial_value = 128000)
  (h_final : final_value = 54000)
  (h_years : years = 3)
  (h_equation : final_value = initial_value * (1 - r) ^ years) :
  r = 0.247 :=
sorry

end depreciation_rate_l354_354642


namespace safe_patterns_count_l354_354991

def num_legs : ℕ := 6
def legs_on_each_side : ℕ := 3

-- Define the states as per the problem:
def state : Type := vector ℕ num_legs

def initial_state : state := ⟨[1, 1, 1, 1, 1, 1], sorry⟩ -- all legs on the ground
def final_state : state := ⟨[2, 2, 2, 2, 2, 2], sorry⟩ -- all legs back on the ground

def valid_state (s : state) : Prop :=
  (s.to_list.count (λ x, x = 0) ≤ 3) ∧
  ((s.to_list.take legs_on_each_side).count (λ x, x = 0) ≠ legs_on_each_side) ∧
  ((s.to_list.drop legs_on_each_side).count (λ x, x = 0) ≠ legs_on_each_side)

-- This recursive function would count the number of valid sequences
noncomputable def pangzi : state → ℕ :=
λ s, if s = final_state
     then 1
     else if ¬valid_state s
     then 0
     else ∑ i in finset.range num_legs, if (s.nth i) = 1 -- raise leg
                                  then pangzi (s.update i 0)
                                  else if (s.nth i) = 0 -- lower leg
                                  then pangzi (s.update i 2)
                                  else 0

-- Initial call
noncomputable def number_of_safe_patterns : ℕ := pangzi initial_state

-- Assert the correct answer
theorem safe_patterns_count : number_of_safe_patterns = 1416528 := sorry

end safe_patterns_count_l354_354991


namespace speed_of_each_train_l354_354977

theorem speed_of_each_train (v : ℝ) (train_length time_cross : ℝ) (km_pr_s : ℝ) 
  (h_train_length : train_length = 120)
  (h_time_cross : time_cross = 8)
  (h_km_pr_s : km_pr_s = 3.6)
  (h_relative_speed : 2 * v = (2 * train_length) / time_cross) :
  v * km_pr_s = 54 := 
by sorry

end speed_of_each_train_l354_354977


namespace wall_width_l354_354622

theorem wall_width
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (l_eq : l = 7 * h)
  (volume_eq : w * h * l = 6804) :
  w = 3 :=
by
  sorry

end wall_width_l354_354622


namespace not_or_implies_both_false_l354_354800

-- The statement of the problem in Lean
theorem not_or_implies_both_false {p q : Prop} (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end not_or_implies_both_false_l354_354800


namespace sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l354_354076

variable (α β : ℝ)
variable (hα : α < π/2) (hβ : β < π/2) -- acute angles
variable (h1 : Real.cos (α + π/6) = 3/5)
variable (h2 : Real.cos (α + β) = -Real.sqrt 5 / 5)

theorem sin_2alpha_plus_pi_over_3 :
  Real.sin (2 * α + π/3) = 24 / 25 :=
by
  sorry

theorem cos_beta_minus_pi_over_6 :
  Real.cos (β - π/6) = Real.sqrt 5 / 5 :=
by
  sorry

end sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l354_354076


namespace greatest_of_consecutive_integers_l354_354625

theorem greatest_of_consecutive_integers (x y z : ℤ) (h1: y = x + 1) (h2: z = x + 2) (h3: x + y + z = 21) : z = 8 :=
by
  sorry

end greatest_of_consecutive_integers_l354_354625


namespace problem_statement_l354_354198

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l354_354198


namespace Roy_time_to_bake_l354_354968

theorem Roy_time_to_bake (R : ℝ) (hJ : ∀ t : ℝ, t > 0 → Jane_rate = 1 / 4)
  (hR : ∀ t : ℝ, t > 0 → Roy_rate = 1 / R)
  (hTogether : ∀ t : ℝ, t > 0 → together_rate t = t * (1 / 4 + 1 / R))
  (hJaneAlone : ∀ t : ℝ, t > 0 → jane_alone_rate t = t * (1 / 4))
  (hTotal : ∀ t : ℝ, t = 2 → ∀ s : ℝ, s = 0.4 → together_rate t + jane_alone_rate s = 1):
  R = 5 :=
by
  sorry

end Roy_time_to_bake_l354_354968


namespace connected_settlement_l354_354534

theorem connected_settlement {N : ℕ} (settlements : fin 2004) (tunnels : finset (fin 2004 × fin 2004)) : 
  (∀ s1 s2, s1 ≠ s2 → ∃ path, s1 ∈ path ∧ s2 ∈ path) → N = 2003 :=
sorry

end connected_settlement_l354_354534


namespace find_FC_l354_354390

theorem find_FC 
  (DC CB AD : ℕ)
  (h1 : DC = 9)
  (h2 : CB = 7)
  (h3 : AD ≠ 0)
  (h4 : ∃ (k : ℚ), k = (1/3) ∧ k * AD = ?m)
  (h5 : ∃ (l : ℚ), l = (2/3) ∧ l * AD = ?n)
  : ∃ (FC : ℚ), FC = 10 :=
begin
  sorry
end

end find_FC_l354_354390


namespace sum_primes_between_10_and_20_l354_354958

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter (λ n, is_prime n ∧ a ≤ n ∧ n ≤ b) (List.range (b + 1))

def sum_of_primes_between_10_and_20 : ℕ :=
  (primes_between 10 20).sum

theorem sum_primes_between_10_and_20 :
  sum_of_primes_between_10_and_20 = 60 :=
by
  sorry

end sum_primes_between_10_and_20_l354_354958


namespace find_A_and_B_l354_354694

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (∀ x : ℝ, x ≠ 10 ∧ x ≠ -3 → 5*x + 2 = A * (x + 3) + B * (x - 10)) ∧ 
    A = 4 ∧ B = 1 :=
  sorry

end find_A_and_B_l354_354694


namespace incorrect_conclusion_about_cos_sin_function_l354_354043

theorem incorrect_conclusion_about_cos_sin_function (f : ℝ → ℝ)
  (h₀ : ∀ x, f(x) = Real.cos x * Real.sin (2 * x))
  (h₁ : ∀ x, f(-x) = -f(x))
  (h₂ : ∀ x, f(x + 2 * Real.pi) = f(x))
  (h₃ : ∀ t : ℝ, (t ∈ Set.Icc (-1) 1) → ∃ y, y = 2 * t - 2 * t^3 ∧ ∀ y', y' = 2 - 6 * t^2 → y = f(Real.arcsin t))
  (maximum_value : ∃ x ∈ Set.Icc (-1) 1, f(x) = 4 * Real.sqrt 3 / 9)
  (h₄ : ∀ x, f(Real.pi - x) = -f(x))
  (h₅ : ∀ x, f(2 * Real.pi - x) + f(x) ≠ 0)
  : ¬ (∀ x, f(Real.pi - x) + f(x) = 0) :=
sorry

end incorrect_conclusion_about_cos_sin_function_l354_354043


namespace length_CD_l354_354539

theorem length_CD (AB AC BD CD : ℝ) (hAB : AB = 2) (hAC : AC = 5) (hBD : BD = 6) :
    CD = 3 :=
by
  sorry

end length_CD_l354_354539


namespace speed_downstream_l354_354319

variables (V_m V_s V_u V_d : ℕ)
variables (h1 : V_u = 12)
variables (h2 : V_m = 25)
variables (h3 : V_u = V_m - V_s)

theorem speed_downstream (h1 : V_u = 12) (h2 : V_m = 25) (h3 : V_u = V_m - V_s) :
  V_d = V_m + V_s :=
by
  -- The proof goes here
  sorry

end speed_downstream_l354_354319


namespace cos_is_even_and_has_zero_points_sin_is_not_even_ln_is_not_even_and_has_one_zero_point_x_squared_plus_one_is_even_and_has_no_zero_points_correct_answer_is_cos_l354_354655

theorem cos_is_even_and_has_zero_points :
  (∀ x : ℝ, cos (-x) = cos x) ∧ (∃ x : ℝ, cos x = 0) :=
begin
  sorry
end

theorem sin_is_not_even :
  ∀ x : ℝ, sin (-x) ≠ sin x :=
begin
  sorry
end

theorem ln_is_not_even_and_has_one_zero_point :
  (∀ x : ℝ, x > 0 → ln (-x) ≠ ln x) ∧ (∃ x : ℝ, x > 0 ∧ ln x = 0) :=
begin
  sorry
end

theorem x_squared_plus_one_is_even_and_has_no_zero_points :
  (∀ x : ℝ, (x^2 + 1) = ((-x)^2 + 1)) ∧ (∀ x : ℝ, (x^2 + 1) ≠ 0) :=
begin
  sorry
end

theorem correct_answer_is_cos :
  (∀ x : ℝ, cos (-x) = cos x) ∧ (∃ x : ℝ, cos x = 0) ∧
  ¬((∀ x : ℝ, sin (-x) = sin x) ∧ (∃ x : ℝ, sin x = 0)) ∧
  ¬((∀ x : ℝ, x > 0 → ln (-x) = ln x) ∧ (∃ x : ℝ, x > 0 ∧ ln x = 0)) ∧
  ¬((∀ x : ℝ, (x^2 + 1) = ((-x)^2 + 1)) ∧ (∃ x : ℝ, x^2 + 1 = 0)) :=
begin
  sorry
end

end cos_is_even_and_has_zero_points_sin_is_not_even_ln_is_not_even_and_has_one_zero_point_x_squared_plus_one_is_even_and_has_no_zero_points_correct_answer_is_cos_l354_354655


namespace Kevin_lost_cards_l354_354505

theorem Kevin_lost_cards (initial_cards final_cards : ℝ) (h1 : initial_cards = 47.0) (h2 : final_cards = 40) :
  initial_cards - final_cards = 7 :=
by
  sorry

end Kevin_lost_cards_l354_354505


namespace quadratic_root_unique_l354_354739

theorem quadratic_root_unique (m : ℝ) :
  (∀ (x : ℝ), (x = 1 → (m - 2) * x^2 + 4 * x - m^2 = 0)) → m = -1 :=
begin
  intro h,
  specialize h 1 (eq.refl 1),
  sorry
end

end quadratic_root_unique_l354_354739


namespace determine_m_l354_354507

theorem determine_m (p : ℕ) [Nat.Prime p] :
  ∀ m : ℕ, (∃ a : Fin p → ℤ, m ∣ (Finset.univ.sum (λ i => (a i)^p)) - (p + 1)) ↔
  (∃ q : ℕ, Nat.Prime q ∧ ∃ a : ℕ, m = q ^ a) ∧ p ≠ 2 ∧ p ≠ 3 := 
sorry

end determine_m_l354_354507


namespace manufacturer_cost_price_l354_354320

theorem manufacturer_cost_price (final_price : ℝ) (m_profit r1 r2 r3 : ℝ) : 
  final_price = 30.09 → 
  m_profit = 0.18 → 
  r1 = 1.20 → 
  r2 = 1.25 → 
  let C := final_price / ((1 + m_profit) * r1 * r2) in 
  C ≈ 17 :=
by sorry

end manufacturer_cost_price_l354_354320


namespace max_parts_two_planes_l354_354600

theorem max_parts_two_planes : 
  ∀ (P1 P2 : AffineSubspace ℝ (EuclideanSpace ℝ 3)), 
  (P1.direction = P2.direction ∨ ∃ x, x ∈ P1 ∧ x ∈ P2) → 
  ∃ n, (n = 4 ∨ n = 3) ∧ ∀ s ∈ refine_partition P1 P2, s.card ≤ n :=
sorry

end max_parts_two_planes_l354_354600


namespace fixed_points_of_logarithmic_graph_l354_354231

theorem fixed_points_of_logarithmic_graph (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∀ x y : ℝ, y = log a ((x - 1) ^ 2) + 2 → (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 2)) :=
by
  sorry

end fixed_points_of_logarithmic_graph_l354_354231


namespace base2_representation_95_l354_354276

theorem base2_representation_95 : nat.toDigits 2 95 = [1, 0, 1, 1, 1, 1, 1] := by
  sorry

end base2_representation_95_l354_354276


namespace complement_intersection_l354_354184

-- Definitions
def U : Set ℕ := {x | x ≤ 4 ∧ 0 < x}
def A : Set ℕ := {1, 4}
def B : Set ℕ := {2, 4}
def complement (s : Set ℕ) := {x | x ∈ U ∧ x ∉ s}

-- The theorem to prove
theorem complement_intersection :
  complement (A ∩ B) = {1, 2, 3} :=
by
  sorry

end complement_intersection_l354_354184


namespace factorization_of_5_1985_minus_1_l354_354690

theorem factorization_of_5_1985_minus_1 :
  ∃ (a b c : ℕ),
    a = 5^397 - 1 ∧
    b = 5^794 - 5^596 + 3 * 5^397 - 5^199 + 1 ∧
    c = 5^794 + 5^596 + 3 * 5^397 + 5^199 + 1 ∧
    5^1985 - 1 = a * b * c ∧
    a > 5^100 ∧ b > 5^100 ∧ c > 5^100 :=
begin
  sorry
end

end factorization_of_5_1985_minus_1_l354_354690


namespace symmetric_points_coords_l354_354410

theorem symmetric_points_coords (a b : ℝ) :
    let N := (a, -b)
    let P := (-a, -b)
    let Q := (b, a)
    N = (a, -b) ∧ P = (-a, -b) ∧ Q = (b, a) →
    Q = (b, a) :=
by
  intro h
  sorry

end symmetric_points_coords_l354_354410


namespace students_at_year_end_l354_354135

theorem students_at_year_end (initial_students left_students new_students end_students : ℕ)
  (h_initial : initial_students = 31)
  (h_left : left_students = 5)
  (h_new : new_students = 11)
  (h_end : end_students = initial_students - left_students + new_students) :
  end_students = 37 :=
by
  sorry

end students_at_year_end_l354_354135


namespace range_of_a_for_monotonicity_l354_354429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 :=
by sorry

end range_of_a_for_monotonicity_l354_354429


namespace f_eq_l354_354761

def a (n : ℕ) : ℚ := 1 / (n + 1)^2

def f (n : ℕ) : ℚ := ∏ k in Finset.range (n + 1), (1 - a k)

theorem f_eq (n : ℕ) : f n = (n + 2) / (2 * (n + 1)) := 
sorry

end f_eq_l354_354761


namespace factorial_simplify_l354_354900

theorem factorial_simplify : 
  (11.factorial / (9.factorial + 2 * 8.factorial)) = 90 := by sorry

end factorial_simplify_l354_354900


namespace max_area_dog_roam_l354_354893

theorem max_area_dog_roam (r : ℝ) (s : ℝ) (half_s : ℝ) (midpoint : Prop) :
  r = 10 → s = 20 → half_s = s / 2 → midpoint → 
  r > half_s → 
  π * r^2 = 100 * π :=
by 
  intros hr hs h_half_s h_midpoint h_rope_length
  sorry

end max_area_dog_roam_l354_354893


namespace op_example_l354_354038

def myOp (c d : Int) : Int :=
  c * (d + 1) + c * d

theorem op_example : myOp 5 (-2) = -15 := 
  by
    sorry

end op_example_l354_354038


namespace problem_solution_l354_354180

noncomputable def g (x : ℝ) : ℝ := 5 / (16^x + 5)

theorem problem_solution :
  ∑ k in finset.range 2001, g (↑k / 2002) = 1001 := by
  sorry

end problem_solution_l354_354180


namespace min_ratio_cyl_inscribed_in_sphere_l354_354340

noncomputable def min_surface_area_to_volume_ratio (R r : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (R^2 - r^2)
  let A := 2 * Real.pi * r * (h + r)
  let V := Real.pi * r^2 * h
  A / V

theorem min_ratio_cyl_inscribed_in_sphere (R : ℝ) :
  ∃ r h, h = 2 * Real.sqrt (R^2 - r^2) ∧
         min_surface_area_to_volume_ratio R r = (Real.sqrt (Real.sqrt 4 + 1))^3 / R := 
by {
  sorry
}

end min_ratio_cyl_inscribed_in_sphere_l354_354340


namespace value_of_a_l354_354753

noncomputable def f : ℝ → ℝ 
| x => if x > 0 then 2^x else x + 1

theorem value_of_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 :=
by
  sorry

end value_of_a_l354_354753


namespace sum_of_last_four_coeffs_l354_354281

theorem sum_of_last_four_coeffs (a : ℝ) (ha : a ≠ 0) : 
  let expr := (1 - (1 / a))^8 in
  let coeffs := (-56) + 28 + (-8) + 1 in
  expr.expand_binom_expr.last_four_coeff_sum = coeffs :=
begin
  sorry
end

end sum_of_last_four_coeffs_l354_354281


namespace distance_center_circle_to_line_l354_354758

theorem distance_center_circle_to_line :
  let C := { p : ℝ × ℝ | ∃ (θ : ℝ), (p.1)^2 + (p.2)^2 - 2 * p.1 = 0 }
  let l := { p : ℝ × ℝ | p.1 - p.2 + 1 = 0 }
  let center_C := (1, 0)
  let distance := λ (p : ℝ × ℝ) (l : ℝ × ℝ → Prop),
    (abs (p.1 - p.2 + 1)) / (real.sqrt ((1:ℝ)^2 + (-1:ℝ)^2))
  in distance center_C (λ p, p.1 - p.2 + 1 = 0) = real.sqrt 2 :=
sorry

end distance_center_circle_to_line_l354_354758


namespace integer_solutions_to_inequality_l354_354449

noncomputable def count_integer_solutions (π : ℝ) : ℕ :=
  let bound := Real.sqrt (4 * π)
  Finset.filter (λ x : ℤ, (x : ℝ)^2 < 4 * π) (Finset.Icc (-Int.floor bound) (Int.floor bound)).card

theorem integer_solutions_to_inequality (π : ℝ) (hπ : π ≈ 3.141592653589793) : 
  count_integer_solutions π = 7 :=
by
  sorry

end integer_solutions_to_inequality_l354_354449


namespace tangent_line_equation_find_a_l354_354721

-- Part 1
theorem tangent_line_equation (f : ℝ → ℝ) (a : ℝ) (h_f : ∀ x, f x = x^3 - 2/x - a) :
  a = 0 → ∃ m b, (m = 5) ∧ (b = -6) ∧ (∀ x, f 1 = -1) := 
  sorry

-- Part 2
theorem find_a (f : ℝ → ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ) (a : ℝ) (h_f : ∀ x, f x = x^3 - 2/x - a) (h_P : P = (-1, 0)) 
  (h_tangent : ∀ x, tangent_at l f x ↔ l x = f x) :
  (∃ x, x = 1 ∧ l x = f x ∧  ∀ x, f 1 = -1) → a = -11 :=
  sorry

end tangent_line_equation_find_a_l354_354721


namespace mr_karan_borrowed_years_l354_354876

theorem mr_karan_borrowed_years:
  let P := 5461.04
  let R := 0.06
  let A := 8410
  let SI := A - P
  let T := SI / (P * R)
  in T ≈ 9 := by
  sorry

end mr_karan_borrowed_years_l354_354876


namespace area_of_ABCD_l354_354828

-- Definitions based on the conditions outlined in the problem
def AB := 10
def BC := 6
def CD := 12
def DA := 12
def angle_CDA := 90

theorem area_of_ABCD (a b c : ℕ) (ha : a = 144) (hb : b = 72) (hc : c = 1) :
    ∃ (area : ℝ), area = real.sqrt a + b * real.sqrt c ∧ a + b + c = 217 := by
  use (real.sqrt 144 + 72 * real.sqrt 1)
  split
  . exact sorry -- Prove the area in this step
  . exact sorry -- Prove that a + b + c = 217 in this step

end area_of_ABCD_l354_354828


namespace slope_of_line_pi_div_3_trajectory_of_moving_point_P_l354_354759

-- Define the parametric line
def line (a b : ℝ) (α t : ℝ) : ℝ × ℝ := (a + t * real.sin α, b + t * real.cos α)

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- The first proof goal: slope of the line when α = π/3
theorem slope_of_line_pi_div_3 (a b t : ℝ) : real.cos (π / 3) / real.sin (π / 3) = √3 / 3 :=
by sorry

-- The second proof goal: trajectory equation of the moving point P
theorem trajectory_of_moving_point_P (a b t_A t_B : ℝ) (hα : a^2 + b^2 < 4)
  (h_eq : (a + t_A * real.sin (π / 3))^2 + (b + t_A * real.cos (π / 3))^2 = 4)
  (h_geometric : t_A * t_B = -(a^2 + b^2 - 4))
  : a^2 + b^2 = 2 :=
by sorry

end slope_of_line_pi_div_3_trajectory_of_moving_point_P_l354_354759


namespace max_valid_committees_l354_354259

-- Define the conditions
def community_size : ℕ := 20
def english_speakers : ℕ := 10
def german_speakers : ℕ := 10
def french_speakers : ℕ := 10
def total_subsets : ℕ := Nat.choose community_size 3
def invalid_subsets_per_language : ℕ := Nat.choose 10 3

-- Lean statement to verify the number of valid committees
theorem max_valid_committees :
  total_subsets - 3 * invalid_subsets_per_language = 1020 :=
by
  simp [community_size, total_subsets, invalid_subsets_per_language]
  sorry

end max_valid_committees_l354_354259


namespace find_g_60_l354_354914

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_func_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y^2
axiom g_45 : g 45 = 15

theorem find_g_60 : g 60 = 8.4375 := sorry

end find_g_60_l354_354914


namespace intersection_x_value_l354_354682

theorem intersection_x_value :
  (∃ x y, y = 3 * x - 7 ∧ y = 48 - 5 * x) → x = 55 / 8 :=
by
  sorry

end intersection_x_value_l354_354682


namespace find_a_and_b_intervals_of_monotonicity_l354_354718

-- Defining the problem conditions
variables (a b x : ℝ) (k : ℤ)
constant f : ℝ → ℝ
constant g : ℝ → ℝ

-- Given conditions
axiom a_positive : a > 0
axiom f_definition : ∀ x, f x = -2 * a * (Real.sin (2 * x + Real.pi / 6)) + 2 * a + b
axiom f_bounds : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → -5 ≤ f x ∧ f x ≤ 1
axiom g_definition : ∀ x, g x = f (x + Real.pi / 2)
axiom log_g_positive : ∀ x, Real.log(g x) > 0

-- Proof obligations
theorem find_a_and_b : a = 2 ∧ b = -5 := sorry

theorem intervals_of_monotonicity :
  (∀ k : ℤ, ∀ x, k * Real.pi < x ∧ x ≤ k * Real.pi + Real.pi / 6 → g (2 * x + Real.pi / 6) -1 > 1) ∧
  (∀ k : ℤ, ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x < k * Real.pi + Real.pi / 3 → g (2 * x + Real.pi / 6) - 1 < 1) :=
sorry

end find_a_and_b_intervals_of_monotonicity_l354_354718


namespace sum_a3_a4_a5_eq_84_l354_354147

-- Definitions based on the conditions in the problem
def is_geometric_sequence (a : ℕ → ℕ) :=
  ∃ q : ℕ, ∀ n : ℕ, a (n+1) = a n * q

variables {a : ℕ → ℕ}

-- Given conditions
axiom q_eq_2 : ∃ q, q = 2
axiom sum_first_three_eq_21 : (a 1) + (a 2) + (a 3) = 21

-- Required proof statement
theorem sum_a3_a4_a5_eq_84 (h : is_geometric_sequence a) (hq : q_eq_2) (h_sum : sum_first_three_eq_21) : 
  let q := 2 in (a 3) + (a 4) + (a 5) = 84 :=
by
  sorry

end sum_a3_a4_a5_eq_84_l354_354147


namespace sequence_converges_to_one_l354_354208

noncomputable def z (n : ℕ) : ℂ := (n : ℂ) - complex.I) / ((n : ℂ) + 1)

theorem sequence_converges_to_one :
  ∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (n : ℕ), n > N → abs (z n - 1) < ε :=
by sorry

end sequence_converges_to_one_l354_354208


namespace distribution_nuts_into_pockets_l354_354484

theorem distribution_nuts_into_pockets : 
  (finset.card ((finset.range 12).powerset.filter (λ s, s.card = 2))) = 55 :=
sorry

end distribution_nuts_into_pockets_l354_354484


namespace lines_meet_on_altitude_of_A_l354_354882

theorem lines_meet_on_altitude_of_A 
  (A B C D: Point)
  (h1 : on_euler_line D A B C)
  (E F X P: Point)
  (h2: intersection (line_through_points B D) (line_through_points C D) E A C)
  (h3: intersection (line_through_points C D) (line_through_points B D) F A B)
  (h4: ∠EXF = 180 - ∠A)
  (h5: same_side A X E F)
  (h6: second_intersection_of_circumcircles (circumcircle C X F) (circumcircle B X E) P) :
  intersects_on_altitude XP EF A :=
begin
  sorry
end

end lines_meet_on_altitude_of_A_l354_354882


namespace min_positive_m_value_l354_354488

noncomputable def P (m y : ℝ) : ℝ × ℝ := (1 - m * y, y)

def dist (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def satisfies_condition (m : ℝ) : Prop :=
  ∃ y : ℝ, dist (P m y) (1, 0) = 2 * dist (P m y) (4, 0)

theorem min_positive_m_value : ∃ (m : ℝ), m = Real.sqrt 3 ∧ 0 < m ∧ satisfies_condition m :=
sorry

end min_positive_m_value_l354_354488


namespace three_digit_numbers_divisible_by_5_l354_354792

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l354_354792


namespace nth_derivative_l354_354389

noncomputable def f (x : ℝ) : ℝ :=
  x / Real.exp x

noncomputable def f₁ (x : ℝ) : ℝ :=
  (f x)'

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := (f_n n)'

theorem nth_derivative (n : ℕ) (x : ℝ) :
  f_n n x = (-1)^n * (x - n) / Real.exp x :=
sorry

end nth_derivative_l354_354389


namespace coordinates_of_C_l354_354268

structure Point :=
  (x : Int)
  (y : Int)

def reflect_over_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

def reflect_over_y_axis (p : Point) : Point :=
  {x := -p.x, y := p.y}

def C : Point := {x := 2, y := 2}

noncomputable def C'_reflected_x := reflect_over_x_axis C
noncomputable def C''_reflected_y := reflect_over_y_axis C'_reflected_x

theorem coordinates_of_C'' : C''_reflected_y = {x := -2, y := -2} :=
by
  sorry

end coordinates_of_C_l354_354268


namespace product_gcd_lcm_equals_2910600_l354_354954

theorem product_gcd_lcm_equals_2910600 :
  let a := 210
  let b := 4620
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * (3 * lcm_ab) = 2910600 :=
by
  let a := 210
  let b := 4620
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  have h1 : gcd_ab = 210 := by 
    -- Calculation of gcd_ab
    sorry
  have h2 : lcm_ab = 4620 := by 
    -- Calculation of lcm_ab
    sorry
  calc
    gcd_ab * (3 * lcm_ab)
        = 210 * (3 * 4620) : by rw [h1, h2]
    ... = 2910600 : by norm_num

end product_gcd_lcm_equals_2910600_l354_354954


namespace area_of_octagon_l354_354064

-- Given a square of side length 2m, prove that the area of the octagon formed by cutting away the corners is 4(√2 - 1)m²

theorem area_of_octagon (m : ℝ) (h : 0 < m) : 
  (let s := 2 * m in
   let octagon_area := 4 * (sqrt 2 - 1) * (m ^ 2) in
   (octagon_area) = (4 * (sqrt 2 - 1) * (m ^ 2))) := 
begin
  sorry
end

end area_of_octagon_l354_354064


namespace marble_problem_l354_354124

theorem marble_problem {r b : ℕ} 
  (h1 : 9 * r - b = 27) 
  (h2 : 3 * r - b = 3) : r + b = 13 := 
by
  sorry

end marble_problem_l354_354124


namespace bologna_sandwich_count_l354_354287

theorem bologna_sandwich_count :
  ∃ (C B P : ℕ), (C + B + P = 80) ∧ (B = 7 * C) ∧ (P = 8 * C) ∧ (B = 35) :=
by
  -- defining C, B, P based on given conditions
  let C := 5
  let B := 7 * C
  let P := 8 * C
  -- ensuring conditions and the required result
  use C, B, P
  split
  case goal_1 => show C + B + P = 80 from by
    calc
      C + B + P = 5 + 7 * 5 + 8 * 5 : by rfl
      -- Computation steps to ensure conditions
      _ = 80 : by norm_num
  split
  case goal_2 => show B = 7 * C by rfl
  split
  case goal_3 => show P = 8 * C by rfl
  show B = 35 by 
    calc
      B = 7 * 5 : by rfl
      _ = 35 : by norm_num

end bologna_sandwich_count_l354_354287


namespace shaded_area_value_l354_354767

noncomputable def line_through (p1 p2 : ℝ × ℝ) : ℝ → ℝ := 
  λ x, (p2.2 - p1.2) / (p2.1 - p1.1) * x + p1.2

def first_line : ℝ → ℝ := line_through (0, 5) (10, 2)

def second_line : ℝ → ℝ := line_through (2, 6) (8, 1)

theorem shaded_area_value :
  ∫ x in 2..8, (second_line x - first_line x) = 13.33 :=
by
  sorry

end shaded_area_value_l354_354767


namespace billy_laundry_loads_l354_354025

-- Define constants based on problem conditions
def sweeping_minutes_per_room := 3
def washing_minutes_per_dish := 2
def laundry_minutes_per_load := 9

def anna_rooms := 10
def billy_dishes := 6

-- Calculate total time spent by Anna and the time Billy spends washing dishes
def anna_total_time := sweeping_minutes_per_room * anna_rooms
def billy_dishwashing_time := washing_minutes_per_dish * billy_dishes

-- Define the time difference Billy needs to make up with laundry
def time_difference := anna_total_time - billy_dishwashing_time
def billy_required_laundry_loads := time_difference / laundry_minutes_per_load

-- The theorem to prove
theorem billy_laundry_loads : billy_required_laundry_loads = 2 := by 
  sorry

end billy_laundry_loads_l354_354025


namespace triangle_angle_l354_354608

theorem triangle_angle {a b c s T : ℝ} (h1 : 2 * s = a + b + c) (h2 : T + (a * b) / 2 = s * (s - c)) :
  ∃ γ : ℝ, γ = 45 ∧ γ = real.arcsin (real.sin (real.pi / 4)) :=
by
  sorry

end triangle_angle_l354_354608


namespace total_students_in_class_l354_354478

-- Define the conditions
def students_taking_french : ℕ := 41
def students_taking_german : ℕ := 22
def students_taking_both : ℕ := 9
def students_not_in_courses : ℕ := 40

-- Define the question and the proof goal
theorem total_students_in_class : 
  let students_only_french := students_taking_french - students_taking_both in
  let students_only_german := students_taking_german - students_taking_both in
  let total_students := students_only_french + students_only_german + students_taking_both + students_not_in_courses in
  total_students = 94 :=
by
  sorry

end total_students_in_class_l354_354478


namespace asymptotes_of_hyperbola_are_correct_l354_354374

def hyperbola {x y : ℝ} (a b : ℝ) : Prop := x^2 / a^2 - y^2 = 1

theorem asymptotes_of_hyperbola_are_correct 
  (a b x y : ℝ) 
  (h : hyperbola x y 3 1) :
  x = 3 * y ∨ x = -3 * y := by
  sorry

end asymptotes_of_hyperbola_are_correct_l354_354374


namespace entire_wall_area_l354_354973

-- The conditions in Lean definitions
def regular_tile_length : ℝ := _
def regular_tile_width : ℝ := _
def Lw : ℝ := regular_tile_length * regular_tile_width -- Area of one regular tile
def R : ℝ := 90 / Lw -- Number of regular tiles
def jumbo_tile_length := 3 * regular_tile_length
def jumbo_tile_width := regular_tile_width
def R_jumbo := R / 2 -- Number of jumbo tiles
def jumbo_tile_area := jumbo_tile_length * jumbo_tile_width
def total_area := 90 + (3 / 2) * 90

-- Proof statement
theorem entire_wall_area :
  total_area = 225 := by
  sorry

end entire_wall_area_l354_354973


namespace lindas_initial_candies_l354_354526

theorem lindas_initial_candies (candies_given : ℝ) (candies_left : ℝ) (initial_candies : ℝ) : 
  candies_given = 28 ∧ candies_left = 6 → initial_candies = candies_given + candies_left → initial_candies = 34 := 
by 
  sorry

end lindas_initial_candies_l354_354526


namespace sum_of_sequence_l354_354519

def a : ℕ → ℕ
| 0       := 1
| (2*i+1) := a i
| (2*i+2) := a i + a (i+1)

def b (n : ℕ) : ℕ := (Finset.range (2^n)).sum a

theorem sum_of_sequence (n : ℕ) : b n = (3^n + 1) / 2 := by
  sorry

end sum_of_sequence_l354_354519


namespace problem_statement_l354_354722

open Complex

noncomputable def a : ℂ := 5 - 3 * I
noncomputable def b : ℂ := 2 + 4 * I

theorem problem_statement : 3 * a - 4 * b = 7 - 25 * I :=
by { sorry }

end problem_statement_l354_354722


namespace correct_statements_true_l354_354419

theorem correct_statements_true :
  (∀ x : ℝ, x ≥ 1 → (let y := (x - 2)/(2*x + 1) in y ∈ [-1/3, 1/2)) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, 2*x - 1 ∈ [-1, 1] → f (2*x -1) ∈ ℝ) →
    let domain_y := {x | 1 < x ∧ x ≤ 2} in ∀ x ∈ domain_y, f (x - 1) / sqrt (x - 1) ∈ ℝ ) ∧
  (∃! f, ∀ x, (x ∈ {-2, 2}) → f x = x^2) ∧
  (∀ f : ℝ → ℝ, (∀ x : ℝ, x ≥ 1 → f (x + 1/x) = x^2 + 1/x^2) →
    let m := sqrt 6 in f m = 4) :=
begin
  sorry
end

end correct_statements_true_l354_354419


namespace geometric_series_sum_l354_354706

theorem geometric_series_sum : 
  (finset.range 2012).sum (λ n, 2^n) = 2^2012 - 1 :=
by
  sorry

end geometric_series_sum_l354_354706


namespace suitable_storage_temp_l354_354930

theorem suitable_storage_temp : -5 ≤ -1 ∧ -1 ≤ 1 := by {
  sorry
}

end suitable_storage_temp_l354_354930


namespace problem_l354_354407

variables {f : ℝ → ℝ}

def differentiable_on_real_line (f : ℝ → ℝ) := ∀ x : ℝ, differentiable ℝ f

def condition_lt_derivative (f : ℝ → ℝ) := ∀ x : ℝ, f x < (deriv f x)

theorem problem 
  (h1 : differentiable_on_real_line f) 
  (h2 : condition_lt_derivative f) : 
  f 2 > real.exp 2 * f 0 ∧ f 2017 > real.exp 2017 * f 0 :=
sorry

end problem_l354_354407


namespace minimum_number_of_weights_l354_354649

-- Define the conditions for the weights set
variable (W : Type) [DecidableEq W] (mass : W → ℝ)
variable (S : Finset W)

-- Conditions
-- 1. There are 5 weights, each with a different mass.
def five_different_weights (S : Finset W) : Prop :=
  S.card = 5 ∧
  (∀ w1 w2 ∈ S, w1 ≠ w2 → mass w1 ≠ mass w2)

-- 2. For any 2 weights, it is possible to find another 2 weights such that the sum of their masses is equal.
def valid_pairs (S : Finset W) : Prop :=
  ∀ w1 w2 ∈ S, w1 ≠ w2 → ∃ w3 w4 ∈ S, w1 ≠ w3 ∧ w1 ≠ w4 ∧ w2 ≠ w3 ∧ w2 ≠ w4 ∧ mass w1 + mass w2 = mass w3 + mass w4

-- The Lean 4 statement for the problem
theorem minimum_number_of_weights (S : Finset W) (mass : W → ℝ) :
  five_different_weights S mass → valid_pairs S mass → S.card ≥ 13 :=
by
  -- Proof is omitted
  sorry

end minimum_number_of_weights_l354_354649


namespace polygon_sides_arithmetic_progression_l354_354683

theorem polygon_sides_arithmetic_progression:
  ∃ (n : ℕ), 
  (∀ (a_i : ℕ → ℝ), 
    (a_i 1 = 150) ∧ 
    (∀ k : ℕ, 1 ≤ k → a_i (k + 1) = a_i k + 3) ∧ 
    (∑ i in finset.range n, a_i i) = 180 * (n - 2)
  ) → 
  n = 25 :=
begin
  sorry
end

end polygon_sides_arithmetic_progression_l354_354683


namespace hyperbola_eccentricity_l354_354395

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote_angle : Real.arctan (b / a) = π / 6) : 
  sqrt (1 + (b / a) ^ 2) = 2 * sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l354_354395


namespace greatest_a_eq_sqrt_2_l354_354703

theorem greatest_a_eq_sqrt_2 :
  (∃ a : ℝ, (7 * real.sqrt((2 * a) ^ 2 + 1 ^ 2) - 4 * a ^ 2 - 1) / (real.sqrt(1 + 4 * a ^ 2) + 3) = 2) ∧
  (∀ a' : ℝ, (7 * real.sqrt((2 * a') ^ 2 + 1 ^ 2) - 4 * a' ^ 2 - 1) / (real.sqrt(1 + 4 * a' ^ 2) + 3) = 2 → a' ≤ real.sqrt 2) :=
sorry

end greatest_a_eq_sqrt_2_l354_354703


namespace matilda_card_value_unique_l354_354503

theorem matilda_card_value_unique (x : ℝ) (h1 : 30 < x ∧ x < 120)
  (hcos : cos x ≠ 0) (hsec : cos x ≠ 0) (htan : cos x ≠ 0) :
  (sin x = 1) → sin x = 1 ∧ cos x ≠ 0 ∧ tan x ≠ 0 ∧ sec x ≠ 0 :=
by
  sorry

end matilda_card_value_unique_l354_354503


namespace arithmetic_sequence_product_l354_354728

theorem arithmetic_sequence_product :
  ∃ (a : ℕ → ℕ) (q : ℕ) (p : ℕ),
    (∀ n, a n > 0) ∧
    q = 2 ∧
    (∏ i in finset.range 30, a i) = 2 ^ 30 ∧
    (∏ k in finset.range 10, a (3*k + 3)) =  2 ^ 20 :=
by
  sorry

end arithmetic_sequence_product_l354_354728


namespace probability_even_sum_of_selected_envelopes_l354_354592

theorem probability_even_sum_of_selected_envelopes :
  let face_values := [5, 6, 8, 10]
  let possible_sum_is_even (s : ℕ) : Prop := s % 2 = 0
  let num_combinations := Nat.choose 4 2
  let favorable_combinations := 3
  (favorable_combinations / num_combinations : ℚ) = 1 / 2 :=
by
  sorry

end probability_even_sum_of_selected_envelopes_l354_354592


namespace average_marbles_of_other_colors_l354_354537

theorem average_marbles_of_other_colors
  (clear_percentage : ℝ) (black_percentage : ℝ) (total_marbles_taken : ℕ)
  (h1 : clear_percentage = 0.4) (h2 : black_percentage = 0.2) :
  (total_marbles_taken : ℝ) * (1 - clear_percentage - black_percentage) = 2 :=
by
  sorry

end average_marbles_of_other_colors_l354_354537


namespace total_cost_is_correct_l354_354018

-- Definitions to reflect the conditions
def small_birdhouse_planks := 7
def small_birdhouse_nails := 20
def large_birdhouse_planks := 10
def large_birdhouse_nails := 36
def nail_cost := 0.05
def small_plank_cost := 3
def large_plank_cost := 5

-- Function to calculate the cost of one small birdhouse
def small_birdhouse_cost : Float :=
  (small_birdhouse_planks * small_plank_cost) + (small_birdhouse_nails * nail_cost)

-- Function to calculate the cost of one large birdhouse
def large_birdhouse_cost : Float :=
  (large_birdhouse_planks * large_plank_cost) + (large_birdhouse_nails * nail_cost)

-- Function to calculate the total cost for multiple birdhouses
def total_cost (num_small num_large : Nat) : Float :=
  (num_small * small_birdhouse_cost) + (num_large * large_birdhouse_cost)

-- Proof statement
theorem total_cost_is_correct :
  total_cost 3 2 = 169.60 := 
by
  -- Proof omitted
  sorry

end total_cost_is_correct_l354_354018


namespace number_of_men_l354_354807

variable (M : ℕ)

-- Define the first condition: M men reaping 80 hectares in 24 days.
def first_work_rate (M : ℕ) : ℚ := (80 : ℚ) / (M * 24)

-- Define the second condition: 36 men reaping 360 hectares in 30 days.
def second_work_rate : ℚ := (360 : ℚ) / (36 * 30)

-- Lean 4 statement: Prove the equivalence given conditions.
theorem number_of_men (h : first_work_rate M = second_work_rate) : M = 45 :=
by
  sorry

end number_of_men_l354_354807


namespace triangles_similar_l354_354175

-- Definitions
variables {A B C G M X Y Q P : Type*}
          [inhabited A] [inhabited B] [inhabited C] [inhabited G]
          [inhabited M] [inhabited X] [inhabited Y] [inhabited Q] [inhabited P]
  
noncomputable def centroid_of_triangle (Δ : triangle A B C) : G := sorry
noncomputable def midpoint (A B : Type*) : Type* := sorry
noncomputable def collinear (P Q R : Type*) : Prop := sorry
noncomputable def parallel (L M : set (Type*)) : Prop := sorry

noncomputable def similar (Δ₁ Δ₂ : triangle X Y Z) : Prop := sorry

-- Hypotheses
variables (ΔABC : triangle A B C)
hypothesis G_centroid : centroid_of_triangle ΔABC = G
hypothesis M_midpoint : midpoint B C = M
hypothesis X_on_AB : X ∈ segment A B
hypothesis Y_on_AC : Y ∈ segment A C
hypothesis collinear_XYG : collinear X Y G
hypothesis parallel_XY_BC : parallel (line_through X Y) (line_through B C)
hypothesis intersect_XC_GB : intersect (line_through X C) (line_through G B) = Q
hypothesis intersect_YB_GC : intersect (line_through Y B) (line_through G C) = P

-- Theorem to be proven
theorem triangles_similar : similar (triangle M P Q) (triangle A B C) := 
  sorry

end triangles_similar_l354_354175


namespace common_area_of_two_triangles_l354_354879

theorem common_area_of_two_triangles :
  ∃ (triangles : Type) (T1 T2 : triangles), 
  (∀ (a b c : ℕ), a = 18 ∧ b = 24 ∧ c = 30 ∧ 
    (a^2 + b^2 = c^2)) ∧ -- Right-angled property
  (Circumcircle T1 = Circumcircle T2) ∧
  (InscribedCircle T1 = InscribedCircle T2) ∧
  (Area T1 + Area T2 = 2 * 216 - 84) →
  ∃ (common_area : ℕ), common_area = 132 := 
sorry

end common_area_of_two_triangles_l354_354879


namespace find_input_number_l354_354935

noncomputable def digit_reverse (n : ℕ) : ℕ :=
  n.toString.reverse.toNat

theorem find_input_number (x : ℕ) (h₁ : 1000 ≤ x) (h₂ : x < 10000) :
  let y := digit_reverse (3 * x)
  y + 2 = 2015 → x = 1034 :=
by
  intros h
  sorry

end find_input_number_l354_354935


namespace library_book_count_l354_354849

theorem library_book_count (fiction_purchased_last_year non_fiction_purchased_last_year total_purchased_last_year total_purchased_this_year books_lost this_year_books_remaining : ℕ)
(
    h1 : fiction_purchased_last_year = 50,
    h2 : non_fiction_purchased_last_year = 2.5 * fiction_purchased_last_year,
    h3 : total_purchased_last_year = fiction_purchased_last_year + non_fiction_purchased_last_year,
    h4 : total_purchased_this_year = 3 * total_purchased_last_year,
    h5 : books_lost = 12,
    h6 : this_year_books_remaining = total_purchased_this_year - books_lost,
    h7 : initial_books_fiction = 60,
    h8 : initial_books_non_fiction = 100 - initial_books_fiction,
    h9 : total_books_before_this_year = initial_books_fiction + initial_books_non_fiction + total_purchased_last_year
) :
    this_year_books_remaining + total_books_before_this_year = 788 :=
sorry

end library_book_count_l354_354849


namespace tangents_concur_isotomic_conjugates_l354_354856

open scoped Classical

-- Define the setup
variables {P Q : Point} {X Y Z : Point}
variables {r_A r_B r_C : Ray P}
variables {w_A w_B w_C : Circle}
variables {s_A s_B s_C : Line}

-- Conditions
axiom tangent_wA_rB : tangent w_A r_B
axiom tangent_wA_rC : tangent w_A r_C
axiom tangent_wB_rA : tangent w_B rA
axiom tangent_wB_rC : tangent w_B r_C
axiom tangent_wC_rA : tangent w_C rA
axiom tangent_wC_rB : tangent w_C rB
axiom P_in_triangle_XYZ : inside P (triangle X Y Z)
axiom internal_tangent_sA : internal_tangent s_A w_B w_C
axiom internal_tangent_sB : internal_tangent s_B w_A w_C
axiom internal_tangent_sC : internal_tangent s_C w_A w_B

-- Prove concurrency and isotomic conjugacy
theorem tangents_concur_isotomic_conjugates :
    (∃ Q, concurrentAt Q s_A s_B s_C) ∧ (isotomic_conjugates P Q) :=
by
  sorry

end tangents_concur_isotomic_conjugates_l354_354856


namespace solve_for_x_l354_354459

theorem solve_for_x (x : ℚ) (h : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354459


namespace angle_CBA_l354_354662

theorem angle_CBA (α β γ : ℝ) (AD DC CB : ℝ)
  (h1 : AD = DC) (h2 : DC = CB)
  (h3 : ∠ADB + ∠ACB = 180)
  (h4 : ∠DAB = α) :
  ∠CBA = 120 - α := 
sorry

end angle_CBA_l354_354662


namespace longest_shortest_chord_l354_354818

variable {C : Type} [MetricSpace C]
variable {p M : C} (hM : M ∈ metric.sphere p r)  -- Given point M is on the circle

def is_diameter (chord : set C) : Prop := ∃ O, O ∈ metric.sphere p r ∧ ∀ x ∈ chord, ∃ y ∈ chord, dist x y = 2 * r
def is_shortest_chord (chord : set C) : Prop := chord ⊆ metric.sphere p r ∧ ∀ x y ∈ chord, dist x y = 0

theorem longest_shortest_chord (C : Type) [MetricSpace C] (p M : C) (r : ℝ) (hM : M ∈ metric.sphere p r) :
  ∃ d sc, is_diameter d ∧ is_shortest_chord sc := 
sorry

end longest_shortest_chord_l354_354818


namespace area_of_PQRS_is_one_l354_354143

variable (PQRS WXYZ : Type) [Square PQRS] [Square WXYZ]
variable (PS WZ : Line) (x : ℝ)
variable (shaded_area : ℝ)
variable (congruent_squares : congruent PQRS WXYZ) (parallel_sides : parallel PS WZ)

-- Given the conditions, the shaded area is defined as 1 cm^2
def shaded_area_is_one : shaded_area = 1 := by sorry

-- Problem statement: Prove the area of square PQRS is 1 cm^2
theorem area_of_PQRS_is_one :
  shaded_area = 1 → (side_length PQRS) ^ 2 = 1 := 
by sorry

end area_of_PQRS_is_one_l354_354143


namespace num_3_digit_div_by_5_l354_354782

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l354_354782


namespace increasing_interval_f_l354_354565

-- Define the function f(x) = lg(x^2 - 1)
def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

-- State the conditions
def domain_condition (x : ℝ) : Prop := x^2 - 1 > 0

-- The main statement to prove
theorem increasing_interval_f (x : ℝ) : domain_condition x → (f(x) > f(1) → x > 1) :=
by
  sorry

end increasing_interval_f_l354_354565


namespace correct_mark_l354_354220

theorem correct_mark
  (n : ℕ)
  (initial_avg : ℝ)
  (wrong_mark : ℝ)
  (correct_avg : ℝ)
  (correct_total_marks : ℝ)
  (actual_total_marks : ℝ)
  (final_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  correct_total_marks = (n * correct_avg) →
  actual_total_marks = (n * initial_avg - wrong_mark + final_mark) →
  correct_total_marks = actual_total_marks →
  final_mark = 10 :=
by
  intros h_n h_initial_avg h_wrong_mark h_correct_avg h_correct_total_marks h_actual_total_marks h_eq
  sorry

end correct_mark_l354_354220


namespace prime_divides_30_l354_354799

theorem prime_divides_30 (p : ℕ) (h_prime : Prime p) (h_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
  sorry

end prime_divides_30_l354_354799


namespace Emir_saved_money_l354_354047

theorem Emir_saved_money :
  let dictionary := 5
  let dinosaur_book := 11
  let cookbook := 5
  let total_cost := dictionary + dinosaur_book + cookbook
  let more_needed := 2
  total_cost - more_needed = 19 :=
by
  simp
  exact (by norm_num : 21 - 2 = 19)
  -- sorry

end Emir_saved_money_l354_354047


namespace neznaika_claim_is_incorrect_l354_354345

theorem neznaika_claim_is_incorrect (s1 s2 : ℝ) :
  let initial_distance : ℝ := 90
  let time_hours : ℝ := 2
  (s1 + s2) * time_hours - initial_distance = 180 →
  ¬ ((s1 + s2) = 60) :=
by 
  intros h1 h2 
  have : (s1 + s2) * time_hours = 180 + initial_distance := h1 
  rw [(show (s1 + s2) = 90, by nlinarith)] at h2 
  exact absurd (show 90 = 60, by nlinarith) h2

end neznaika_claim_is_incorrect_l354_354345


namespace no_Spanish_Couple_on_NN_Spanish_Couple_exists_on_S_l354_354863

def isStrictlyIncreasing {S : Type} [Preorder S] (f : S → S) : Prop :=
  ∀ {x y : S}, x < y → f x < f y

structure SpanishCouple (S : Type) [Preorder S] :=
  (f g : S → S)
  (f_strictly_increasing : isStrictlyIncreasing f)
  (g_strictly_increasing : isStrictlyIncreasing g)
  (spanish_condition : ∀ x : S, f (g (g x)) < g (f x))

-- Part (a)
theorem no_Spanish_Couple_on_NN :
  ¬ ∃ (f g : ℕ → ℕ), SpanishCouple ℕ :=
sorry

-- Define the specific set for part (b)
def S : Set ℝ := {x | ∃ a b : ℕ, x = a - 1 / b}

-- f and g on the specific set
noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := x - 1 / 3

-- Prove these form a Spanish Couple on S
theorem Spanish_Couple_exists_on_S :
  ∃ (f g : ℝ → ℝ), 
  (∀ x ∈ S, f x ∈ S ∧ g x ∈ S) ∧ -- Ensure f and g map S to S
  (∀ {x y : ℝ}, x < y → x ∈ S → y ∈ S → f x < f y ∧ g x < g y) ∧
  (∀ x : ℝ, x ∈ S → f (g (g x)) < g (f x)) :=
sorry

end no_Spanish_Couple_on_NN_Spanish_Couple_exists_on_S_l354_354863


namespace triangle_ABC_is_acute_l354_354837

noncomputable def arithmeticSeqTerm (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def geometricSeqTerm (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r^(n - 1)

def tanA_condition (a1 d : ℝ) :=
  arithmeticSeqTerm a1 d 3 = -4 ∧ arithmeticSeqTerm a1 d 7 = 4

def tanB_condition (a1 r : ℝ) :=
  geometricSeqTerm a1 r 3 = 1/3 ∧ geometricSeqTerm a1 r 6 = 9

theorem triangle_ABC_is_acute {A B : ℝ} (a1a da a1b rb : ℝ) 
  (hA : tanA_condition a1a da) 
  (hB : tanB_condition a1b rb) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ (A + B) < π :=
  sorry

end triangle_ABC_is_acute_l354_354837


namespace animal_shelter_dogs_l354_354814

theorem animal_shelter_dogs (D C R : ℕ) 
  (h₁ : 15 * C = 7 * D)
  (h₂ : 15 * R = 4 * D)
  (h₃ : 15 * (C + 20) = 11 * D)
  (h₄ : 15 * (R + 10) = 6 * D) : 
  D = 75 :=
by
  -- Proof part is omitted
  sorry

end animal_shelter_dogs_l354_354814


namespace distinct_numerators_count_l354_354509

-- Define the set S as described
def S : Set ℚ := {x : ℚ | ∃ a b : ℕ, 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ x = (10 * a + b) / 99 ∧ 0 < x ∧ x < 1}

-- Define the main theorem
theorem distinct_numerators_count : (Finset.map (λ r : ℚ, r.num) (Finset.filter (λ r : ℚ, r.denom.gcd (r.num.gcd 99) = 1) (Finset.image (λ r : ℚ, Rat.mk' r.num r.denom) (Finset.filter (∈ S) Finset.univ)))).card = 53 := 
sorry

end distinct_numerators_count_l354_354509


namespace geometric_sequence_constant_l354_354304

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1) (h2 : ∀ n, a (n + 1) = q * a n) (c : ℝ) :
  (∀ n, a (n + 1) + c = q * (a n + c)) → c = 0 := sorry

end geometric_sequence_constant_l354_354304


namespace balloons_lost_l354_354501

theorem balloons_lost (initial remaining : ℕ) (h_initial : initial = 9) (h_remaining : remaining = 7) : initial - remaining = 2 := by
  sorry

end balloons_lost_l354_354501


namespace radius_B_is_one_l354_354357

variables (rA rB rC rD : ℝ) (E H F G : ℝ)

-- Conditions
def circle_A_radius : ℝ := 2
def circle_BC_congruent : Prop := rB = rC
def circle_D_radius : ℝ := 2 * circle_A_radius
def center_D_external_point_of_tangency_circle_A : Prop := E = 2
def externally_tangent_circles : Prop := true  -- Placeholder; more complex geometrical relationships needed

-- Pythagorean theorem relationships
def pythagorean_egH (y x : ℝ) := (circle_A_radius + y)^2 = (circle_A_radius + x)^2 + y^2
def pythagorean_fgH (y x : ℝ) := (circle_D_radius - y)^2 = x^2 + y^2

-- Proof statement
theorem radius_B_is_one
  (h1 : circle_A_radius = 2)
  (h2 : circle_BC_congruent)
  (h3 : circle_D_radius = 2 * circle_A_radius)
  (h4 : externally_tangent_circles)
  (h5 : center_D_external_point_of_tangency_circle_A)
  (h6 : ∀ y x, pythagorean_egH y x ∧ pythagorean_fgH y x ):
  rB = 1 :=
sorry

end radius_B_is_one_l354_354357


namespace factor_x4_minus_8x2_plus_4_factor_4a4_minus_12a2_plus_1_factor_b5_plus_b_plus_1_factor_ab_a_minus_b_ac_a_plus_c_cb_2a_plus_c_minus_b_l354_354691

-- Define the first factorization problem
theorem factor_x4_minus_8x2_plus_4 (x : ℝ) : 
  x^4 - 8 * x^2 + 4 = (x^2 + 2 * x - 2) * (x^2 - 2 * x - 2) :=
by
  sorry

-- Define the second factorization problem
theorem factor_4a4_minus_12a2_plus_1 (a : ℝ) : 
  4 * a^4 - 12 * a^2 + 1 = (2 * a^2 - 3 - 2 * Real.sqrt 2) * (2 * a^2 - 3 + 2 * Real.sqrt 2) :=
by
  sorry

-- Define the third polynomial problem (not further factorizable)
theorem factor_b5_plus_b_plus_1 (b : ℝ) : 
  b^5 + b + 1 = b^5 + b + 1 :=
by
  refl

-- Define the fourth factorization problem
theorem factor_ab_a_minus_b_ac_a_plus_c_cb_2a_plus_c_minus_b (a b c : ℝ) : 
  ab * (a - b) - ac * (a + c) + cb * (2a + c - b) = a * (b^2 - c^2) + bc * (a + c) :=
by
  sorry

end factor_x4_minus_8x2_plus_4_factor_4a4_minus_12a2_plus_1_factor_b5_plus_b_plus_1_factor_ab_a_minus_b_ac_a_plus_c_cb_2a_plus_c_minus_b_l354_354691


namespace tg_pi_over_12_eq_exists_two_nums_l354_354305

noncomputable def tg (x : ℝ) := Real.tan x

theorem tg_pi_over_12_eq : tg (Real.pi / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

theorem exists_two_nums (a : Fin 13 → ℝ) (h_diff : Function.Injective a) :
  ∃ x y, 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

end tg_pi_over_12_eq_exists_two_nums_l354_354305


namespace quiz_common_difference_l354_354815

theorem quiz_common_difference 
  (x d : ℕ) 
  (h1 : x + 2 * d = 39) 
  (h2 : 8 * x + 28 * d = 360) 
  : d = 4 := 
  sorry

end quiz_common_difference_l354_354815


namespace solve_for_x_l354_354961

theorem solve_for_x (x : ℝ) (h : 8 / x + 6 = 8) : x = 4 :=
sorry

end solve_for_x_l354_354961


namespace fraction_equality_l354_354715

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l354_354715


namespace geometric_sequence_sum_l354_354145

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end geometric_sequence_sum_l354_354145


namespace sum_sequence_is_arithmetic_l354_354111

variable {a : ℕ → ℝ} -- sequence of real numbers indexed by natural numbers
variable {d : ℝ} -- common difference of the arithmetic sequence

-- Condition: Sequence 'a' is an arithmetic sequence with common difference 'd'
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- The sums to be formed
def sum_of_triple (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a (3*n + 1) + a (3*n + 2) + a (3*n + 3)

-- Define the sequence formed by sums of three terms of the original sequence
def sum_sequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, sum_of_triple a n

-- Theorem: If {a_n} is an arithmetic sequence, then the sequence formed by the sums
theorem sum_sequence_is_arithmetic (h : is_arithmetic_sequence a d) : 
  is_arithmetic_sequence (sum_sequence a) (9 * d) := 
sorry

end sum_sequence_is_arithmetic_l354_354111


namespace find_sets_A_B_union_and_intersection_range_of_a_l354_354437

namespace MathProof

def setA (x : ℝ) : Prop := (x - 2) / (x - 7) < 0

def setB (x : ℝ) : Prop := x^2 - 12 * x + 20 < 0

def setC (a : ℝ) (x : ℝ) : Prop := 5 - a < x ∧ x < a

theorem find_sets_A_B : 
  ( (λ x, setA x) = { x : ℝ | 2 < x ∧ x < 7 }) ∧
  ( (λ x, setB x) = { x : ℝ | 2 < x ∧ x < 10 }) := 
by
  sorry

theorem union_and_intersection :
  ( ∀ x, setA x ∨ setB x ↔ { x : ℝ | 2 < x ∧ x < 10 } x ) ∧
  ( ∀ x, (¬ setA x ∧ setB x) ↔ { x : ℝ | 7 ≤ x ∧ x < 10 } x ) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  ( ∀ x, setC a x → setB x ) → 
  a ∈ set.Iic 3 :=
by 
  sorry

end MathProof

end find_sets_A_B_union_and_intersection_range_of_a_l354_354437


namespace sequence_an_l354_354099

noncomputable section

open Nat

def sequence_s (n : ℕ) (a : ℕ → ℚ) : ℚ :=
1 - n * a n

theorem sequence_an (a : ℕ → ℚ) (n : ℕ) (h₁ : a 1 = 1 / 2)
(h_ind : ∀ k : ℕ, k > 0 → a k = 1 / (k * (k + 1)) → a (k + 1) = 1 / ((k + 1) * (k + 2))) :
  ∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1)) :=
begin
  intros n hn,
  induction n with k hk,
  { exfalso, exact Nat.lt_asymm hn hn },
  { cases k,
    { exact h₁ },
    { exact h_ind k (Nat.succ_pos k) hk }}
end

#check sequence_an

end sequence_an_l354_354099


namespace midpoint_construction_l354_354974

-- Define the setup, using type Point and ensuring necessary conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def dist (A B : Point) : ℝ := real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

noncomputable def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2 }

-- Define the main proof problem
theorem midpoint_construction (A B M' : Point) (P Q : Point)
  (hAB : dist A B > 0)
  (hAP : dist A P = dist A B)
  (hAQ : dist A Q = dist A B)
  (hPQ : dist P Q = 2 * dist A B)
  (hPAQ : dist P M' = dist Q M')
  (hM'Midpoint : dist A M' = dist M' B) :
  M' = midpoint A B :=
by
  sorry

end midpoint_construction_l354_354974


namespace cannot_determine_exact_order_l354_354263

def weight : Type := string -- or whatever representative type for weights
variable A B C D E : weight
def mass (w : weight) : ℝ := sorry -- a placeholder function for mass
variable (questions_answer : list (weight × weight × weight) → Prop)

theorem cannot_determine_exact_order (distinct_masses : ∀ (x y : weight), x ≠ y → mass x ≠ mass y)
  (q_allowed : ∀ (q : weight × weight × weight), questions_answer (list.repeat q 9)) :
  (∃! (perm : list weight), (list.perm [A, B, C, D, E] perm)) → false := 
sorry

end cannot_determine_exact_order_l354_354263


namespace probability_zero_l354_354577

-- Definitions of grid, numbers, etc.
def numbers := Finset.range 17  -- 1 to 16
def is_odd (n : ℤ) : Prop := n % 2 = 1

-- Assuming each number is used exactly once in the grid
def used_once (grid : Fin₄ → Fin₄ → ℤ) : Prop :=
  ∀ n ∈ numbers, ∃ i j, grid i j = n

-- Summing rows, columns, and diagonals to be odd
def row_sum_odd (grid : Fin₄ → Fin₄ → ℤ) : Prop :=
  ∀ i, is_odd (∑ j, grid i j)

def col_sum_odd (grid : Fin₄ → Fin₄ → ℤ) : Prop :=
  ∀ j, is_odd (∑ i, grid i j)

def main_diag_sum_odd (grid : Fin₄ → Fin₄ → ℤ) : Prop :=
  is_odd (∑ i, grid i i) ∧ is_odd (∑ i, grid i (3 - i))

-- Main statement to prove
theorem probability_zero (grid : Fin₄ → Fin₄ → ℤ) :
  used_once grid →
  row_sum_odd grid →
  col_sum_odd grid →
  main_diag_sum_odd grid →
  false :=
by sorry

end probability_zero_l354_354577


namespace unique_solution_l354_354049

def Z_star := { n : ℕ // n > 0 }

def f (n : Z_star) : ℝ := sorry

theorem unique_solution (f : Z_star → ℝ) (h : ∀ (n m : Z_star), n.val ≥ m.val → f ⟨n.val + m.val, by linarith⟩ + f ⟨n.val - m.val, by linarith⟩ = f ⟨3 * n.val, by linarith⟩) : 
  (∀ (n : Z_star), f n = 0) := 
by sorry

end unique_solution_l354_354049


namespace additional_male_students_l354_354636

variable (a : ℕ)

theorem additional_male_students (h : a > 20) : a - 20 = (a - 20) := 
by 
  sorry

end additional_male_students_l354_354636


namespace total_unique_games_played_l354_354476

theorem total_unique_games_played (teams : ℕ) (games_per_pair : ℕ) (teams = 30) (games_per_pair = 15) : 
  let unique_games := (teams * (teams - 1) * games_per_pair) / 2 in
  unique_games = 6525 :=
by
  sorry

end total_unique_games_played_l354_354476


namespace ceil_floor_difference_l354_354696

theorem ceil_floor_difference : 
  (Int.ceil ((15 : ℚ) / 8 * ((-34 : ℚ) / 4)) - Int.floor (((15 : ℚ) / 8) * Int.ceil ((-34 : ℚ) / 4)) = 0) :=
by 
  sorry

end ceil_floor_difference_l354_354696


namespace triangle_inequality_l354_354497

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (∑ i in ({A, B, C} : Finset ℝ), (cos i)^2 / (1 + cos i)) ≥ 1 / 2 := 
sorry

end triangle_inequality_l354_354497


namespace domain_of_k_l354_354610

noncomputable def k (x : ℝ) : ℝ :=
  (1 / (x + 9)) + (1 / (x ^ 2 + 9 * x + 20)) + (1 / (x ^ 3 + 9))

theorem domain_of_k :
  ∀ x, x ∈ (-∞, -9) ∪ (-9, -5) ∪ (-5, -4) ∪ (-4, -Real.cbrt 9) ∪ (-Real.cbrt 9, ∞) ↔
  k x = k x := sorry

end domain_of_k_l354_354610


namespace fewest_coach_handshakes_l354_354004

theorem fewest_coach_handshakes (n k : ℕ) (h1 : 0 < n) (h2 : k < n) 
  (h3 : nat.choose n 2 + 2 * k = 496) : 2 * k = 0 :=
by
  sorry

end fewest_coach_handshakes_l354_354004


namespace chord_length_on_x_axis_l354_354235

-- Define the circle equation
def circle (x y : ℝ) := (x - 1)^2 + (y + 2)^2 = 20

-- Condition: intersection with x-axis (y = 0)
def on_x_axis (x : ℝ) := circle x 0

-- Theorem: Length of the chord intercepted on the x-axis
theorem chord_length_on_x_axis : ∃ x1 x2 : ℝ, on_x_axis x1 ∧ on_x_axis x2 ∧ |x1 - x2| = 8 :=
by
  sorry

end chord_length_on_x_axis_l354_354235


namespace even_integer_count_between_300_and_800_l354_354107

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def all_digits_different (n : ℕ) : Prop :=
  let digits := List.ofArray (n.digits 10)
  List.Nodup digits

def digits_from_set (n : ℕ) (s : Set ℕ) : Prop :=
  let digits := List.ofArray (n.digits 10)
  ∀ d ∈ digits, d ∈ s

def is_valid_integer (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 800 ∧ is_even n ∧ all_digits_different n ∧ digits_from_set n {1, 2, 3, 5, 7, 8}

theorem even_integer_count_between_300_and_800 : ∃ (count : ℕ), count = 24 ∧ (∃ l : List ℕ, 
  (∀ n ∈ l, is_valid_integer n) ∧ l.length = count) :=
by
  sorry

end even_integer_count_between_300_and_800_l354_354107


namespace infinite_pairs_binom_equiv_not_zero_l354_354891

theorem infinite_pairs_binom_equiv_not_zero :
  ∃ᶠ p in filter.at_top.even (λ p : ℕ, p.prime), 
  let a := p + 1,
      b := p - 1 in
  a > b ∧
  (∃ k : ℕ, 
    nat.choose (a + b) a % (a + b) = k ∧ 
    (nat.choose (a + b) a ≠ 0 ∧ nat.choose (a + b) b ≠ 0 ∧
     nat.choose (a + b) a = nat.choose (a + b) b)) :=
sorry

end infinite_pairs_binom_equiv_not_zero_l354_354891


namespace volume_of_cone_l354_354119

-- Definitions of the problem's conditions
variable (C : Type) -- Cone type
variable [HasVolume C] -- Cone has volume
variable r_inscribed : ℝ -- radius 
variable volume : C → ℝ -- volume of cone

-- Given conditions:
axiom radius_inscribed (c : C) : r_inscribed = 1
axiom center_coincide (c : C) : -- axiom indicating the centers coincide
  ∃ O₁ O₂ : Point,
    Sphere.inscribed_center c O₁ ∧ Sphere.circumscribed_center c O₂ ∧ O₁ = O₂

-- The problem statement:
theorem volume_of_cone (c : C) (h1 : r_inscribed = 1) (h2 : ∃ O₁ O₂ : Point, Sphere.inscribed_center c O₁ ∧ Sphere.circumscribed_center c O₂ ∧ O₁ = O₂) :
    volume c = 3 * pi := 
begin
  -- Proof goes here
  sorry,
end

end volume_of_cone_l354_354119


namespace compare_neg_rationals_l354_354028

theorem compare_neg_rationals : - (3 / 4 : ℚ) > - (6 / 5 : ℚ) :=
by sorry

end compare_neg_rationals_l354_354028


namespace sum_a_b_l354_354803

theorem sum_a_b (a b : ℚ) (h1 : a + 3 * b = 27) (h2 : 5 * a + 2 * b = 40) : a + b = 161 / 13 :=
  sorry

end sum_a_b_l354_354803


namespace factor_expression_l354_354373

-- Problem Statement
theorem factor_expression (x y : ℝ) : 60 * x ^ 2 + 40 * y = 20 * (3 * x ^ 2 + 2 * y) :=
by
  -- Proof to be provided
  sorry

end factor_expression_l354_354373


namespace transformation_identity_l354_354191

theorem transformation_identity (n : Nat) (h : 2 ≤ n) : 
  n * Real.sqrt (n / (n ^ 2 - 1)) = Real.sqrt (n + n / (n ^ 2 - 1)) := 
sorry

end transformation_identity_l354_354191


namespace triangle_is_isosceles_with_base_BC_l354_354182

-- Definitions of points and vectors
variables (O A B C : Point)

-- Non-collinearity condition
axiom non_collinear : ¬Collinear A B C

-- Condition involving vectors
axiom vector_condition :
  (vector B O - vector C O) • (vector B O + vector C O - (2 : ℝ) • vector A O) = 0

-- Type of the triangle
theorem triangle_is_isosceles_with_base_BC :
  IsoscelesTriangleWithBaseBC A B C :=
by
  -- Sorry indicates the proof is omitted
  sorry

end triangle_is_isosceles_with_base_BC_l354_354182


namespace solve_for_x_l354_354455

theorem solve_for_x (x : ℝ) (hx : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354455


namespace prize_distribution_l354_354059

theorem prize_distribution 
  (total_winners : ℕ)
  (score1 score2 score3 : ℕ)
  (total_points : ℕ) 
  (winners1 winners2 winners3 : ℕ) :
  total_winners = 5 →
  score1 = 20 →
  score2 = 19 →
  score3 = 18 →
  total_points = 94 →
  score1 * winners1 + score2 * winners2 + score3 * winners3 = total_points →
  winners1 + winners2 + winners3 = total_winners →
  winners1 = 1 ∧ winners2 = 2 ∧ winners3 = 2 :=
by
  intros
  sorry

end prize_distribution_l354_354059


namespace lockers_count_l354_354566

theorem lockers_count 
(TotalCost : ℝ) 
(first_cents : ℝ) 
(additional_cents : ℝ) 
(locker_start : ℕ) 
(locker_end : ℕ) : 
  TotalCost = 155.94 
  → first_cents = 0 
  → additional_cents = 0.03 
  → locker_start = 2 
  → locker_end = 1825 := 
by
  -- Declare the number of lockers as a variable and use it to construct the proof
  let num_lockers := locker_end - locker_start + 1
  -- The cost for labeling can be calculated and matched with TotalCost
  sorry

end lockers_count_l354_354566


namespace distance_from_Washington_to_Idaho_l354_354358

-- Definitions for the given conditions
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Washington_to_Idaho : ℝ := 80
def speed_Idaho_to_Nevada : ℝ := 50
def total_time : ℝ := 19

-- Definition for time taken using the distances and speeds
def time_Washington_to_Idaho (distance_WI : ℝ) : ℝ := distance_WI / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- The theorem to prove
theorem distance_from_Washington_to_Idaho : 
  ∃ (distance_WI : ℝ), 
  time_Washington_to_Idaho distance_WI + time_Idaho_to_Nevada = total_time ∧
  distance_WI = 640 :=
begin
  sorry
end

end distance_from_Washington_to_Idaho_l354_354358


namespace trigonometric_identity_l354_354654

theorem trigonometric_identity :
  (cos 15 * cos 15 - sin 15 * sin 15) = (real.sqrt 3 / 2) :=
sorry

end trigonometric_identity_l354_354654


namespace restaurant_bill_l354_354554

theorem restaurant_bill
    (t : ℝ)
    (h1 : ∀ k : ℝ, k = 9 * (t / 10 + 3)) :
    t = 270 :=
by
    sorry

end restaurant_bill_l354_354554


namespace seahawks_touchdowns_l354_354267

theorem seahawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) (points_per_field_goal : ℕ) (field_goals : ℕ) (touchdowns : ℕ) :
  total_points = 37 →
  points_per_touchdown = 7 →
  points_per_field_goal = 3 →
  field_goals = 3 →
  total_points = (touchdowns * points_per_touchdown) + (field_goals * points_per_field_goal) →
  touchdowns = 4 :=
by
  intros h_total_points h_points_per_touchdown h_points_per_field_goal h_field_goals h_equation
  sorry

end seahawks_touchdowns_l354_354267


namespace sum_n_l354_354542

theorem sum_n (n p : ℕ) :
  (∑ k in finset.range n, ∏ j in finset.range p, (k + j + 1 : ℕ)) = 
  (n * (n+1) * (n+2) * ... * (n+p)) / (p+1) :=
begin
  sorry
end

end sum_n_l354_354542


namespace find_math_books_l354_354978

theorem find_math_books 
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) : 
  M = 10 := 
by 
  sorry

end find_math_books_l354_354978


namespace complex_roots_equilateral_l354_354868

-- Definitions according to conditions
variables {p q z₁ z₂ : ℂ}

-- Given that z₁ and z₂ are roots of the quadratic equation
def is_root (z : ℂ) (p q : ℂ) : Prop := z^2 + p*z + q = 0

-- Definition of forming an equilateral triangle
def is_equilateral (z₁ z₂ : ℂ) : Prop :=
∃ ω : ℂ, (ω = complex.exp (2 * complex.pi * complex.I / 3)) ∧ (z₂ = ω * z₁)

theorem complex_roots_equilateral (p q z₁ z₂ : ℂ)
  (h1 : is_root z₁ p q) (h2 : is_root z₂ p q) (h3 : is_equilateral z₁ z₂) :
  p^2 / q = 1 :=
sorry

end complex_roots_equilateral_l354_354868


namespace equilateral_centers_of_equilateral_triangles_l354_354196

theorem equilateral_centers_of_equilateral_triangles
  (A B C O₁ O₂ O₃: Type*) 
  [has_center (triangle (B, C)) O₁]
  [has_center (triangle (C, A)) O₂]
  [has_center (triangle (A, B)) O₃]
  (h₁ : equilateral (triangle (B, C))) 
  (h₂ : equilateral (triangle (C, A))) 
  (h₃ : equilateral (triangle (A, B))) :
  equilateral (triangle (O₁, O₂, O₃)) := sorry

end equilateral_centers_of_equilateral_triangles_l354_354196


namespace find_a_l354_354412

theorem find_a (a : ℝ) (h : (1 - 2016 * a) = 2017) : a = -1 := by
  -- proof omitted
  sorry

end find_a_l354_354412


namespace possible_teams_count_l354_354548

-- Defining the problem
def team_group_division : Prop :=
  ∃ (g1 g2 g3 g4 : ℕ), (g1 ≥ 2) ∧ (g2 ≥ 2) ∧ (g3 ≥ 2) ∧ (g4 ≥ 2) ∧
  (66 = (g1 * (g1 - 1) / 2) + (g2 * (g2 - 1) / 2) + (g3 * (g3 - 1) / 2) + 
       (g4 * (g4 - 1) / 2)) ∧ 
  ((g1 + g2 + g3 + g4 = 21) ∨ (g1 + g2 + g3 + g4 = 22) ∨ 
   (g1 + g2 + g3 + g4 = 23) ∨ (g1 + g2 + g3 + g4 = 24) ∨ 
   (g1 + g2 + g3 + g4 = 25))

-- Theorem statement to prove
theorem possible_teams_count : team_group_division :=
sorry

end possible_teams_count_l354_354548


namespace adam_bought_dog_food_packages_l354_354336

-- Define the constants and conditions
def num_cat_food_packages : ℕ := 9
def cans_per_cat_food_package : ℕ := 10
def cans_per_dog_food_package : ℕ := 5
def additional_cat_food_cans : ℕ := 55

-- Define the variable for dog food packages and our equation
def num_dog_food_packages (d : ℕ) : Prop :=
  (num_cat_food_packages * cans_per_cat_food_package) = (d * cans_per_dog_food_package + additional_cat_food_cans)

-- The theorem statement representing the proof problem
theorem adam_bought_dog_food_packages : ∃ d : ℕ, num_dog_food_packages d ∧ d = 7 :=
sorry

end adam_bought_dog_food_packages_l354_354336


namespace galya_number_l354_354388

theorem galya_number (N k : ℤ) (h : (k - N + 1 = k - 7729)) : N = 7730 := 
by
  sorry

end galya_number_l354_354388


namespace trapezium_other_parallel_side_length_l354_354701

theorem trapezium_other_parallel_side_length
  (a h A : ℝ)
  (h_a : a = 20)
  (h_h : h = 12)
  (h_A : A = 228) :
  ∃ x : ℝ, (A = 0.5 * (a + x) * h ∧ x = 18) :=
by 
  have hx : ∃ x : ℝ, A = 0.5 * (a + x) * h := sorry,
  cases hx with x hx_proof,
  use x,
  split,
  exact hx_proof,
  have eq_228 : A = (20 + x) * 6 := sorry,
  have eq_108 : 108 = 6 * x := sorry,
  have x_val : x = 18 := sorry,
  exact x_val

end trapezium_other_parallel_side_length_l354_354701


namespace simplify_power_of_power_l354_354902

variable (x : ℝ)

theorem simplify_power_of_power : (3 * x^4)^4 = 81 * x^16 := 
by 
sorry

end simplify_power_of_power_l354_354902


namespace relationship_among_numbers_l354_354597

theorem relationship_among_numbers
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₀ : a = 6 ^ 0.7)
  (h₁ : b = 0.7 ^ 6)
  (h₂ : c = Real.logOfBase 0.7 6) :
  a > b ∧ b > c :=
by
  sorry

end relationship_among_numbers_l354_354597


namespace cloth_length_in_first_scenario_l354_354628

/-
Given:
1. 6 women can color a certain length of cloth in 3 days.
2. 5 women can color 200 meters of cloth in 4 days.

Show:
The length of the cloth in the first scenario is 180 meters.
-/

theorem cloth_length_in_first_scenario :
  (6 * 3 * 200) / (5 * 4) = 180 :=
by
  calc
    (6 * 3 * 200) / (5 * 4) = (18 * 200) / 20 : by sorry
    ... = 3600 / 20 : by sorry
    ... = 180 : by sorry

end cloth_length_in_first_scenario_l354_354628


namespace magnitude_of_linear_combination_l354_354101

def a : ℝ × ℝ × ℝ := (1, -3, 2)
def b : ℝ × ℝ × ℝ := (-2, 1, 1)

def vector_add (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vector_scale (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem magnitude_of_linear_combination :
  magnitude (vector_add (vector_scale 2 a) b) = 5 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_linear_combination_l354_354101


namespace monotonically_increasing_implies_interval_a_l354_354808

theorem monotonically_increasing_implies_interval_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x - (1/3) * sin (2 * x) + a * sin x) < (y - (1/3) * sin (2 * y) + a * sin y)) →
  -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end monotonically_increasing_implies_interval_a_l354_354808


namespace eggshell_percentage_nearest_to_32_l354_354013

theorem eggshell_percentage (total_weight : ℕ) (yolk_percentage white_percentage : ℕ) (h₁ : total_weight = 60) (h₂ : yolk_percentage = 32) (h₃ : white_percentage = 53) :
  100 - (yolk_percentage + white_percentage) = 15 :=
by sorry

theorem nearest_to_32 (total_weight : ℕ) (yolk_percentage white_percentage shell_percentage : ℕ)
  (h₁ : total_weight = 60) (h₂ : yolk_percentage = 32) (h₃ : white_percentage = 53) (h₄ : shell_percentage = 15) :
  let yolk_mass := total_weight * yolk_percentage / 100,
      white_mass := total_weight * white_percentage / 100,
      shell_mass := total_weight * shell_percentage / 100 in
  if abs (32 - yolk_mass) < abs (32 - white_mass) ∧ abs (32 - yolk_mass) < abs (32 - shell_mass) then "Egg yolk"
  else if abs (32 - white_mass) < abs (32 - yolk_mass) ∧ abs (32 - white_mass) < abs (32 - shell_mass) then "Egg white"
  else "Eggshell" = "Egg white" :=
by sorry

end eggshell_percentage_nearest_to_32_l354_354013


namespace largest_divisor_of_m_l354_354620

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 36 ∣ m :=
sorry

end largest_divisor_of_m_l354_354620


namespace bus_speed_is_correct_l354_354712

noncomputable def Speed_of_Bus 
  (d_AB : ℝ) (t_meet_C : ℝ) (d_CD : ℝ) (v1 v2 : ℝ) (h₁ : d_AB = 4) 
  (h₂ : t_meet_C = 1/6) (h₃ : d_CD = 2/3) 
  (h₄ : v1 + v2 = 48)
  (h₅ : v2 * 1/6 = v1 * 1/6 + d_AB) : ℝ := 40

theorem bus_speed_is_correct :
  ∀ (d_AB t_meet_C d_CD v1 v2 : ℝ),
    d_AB = 4 →
    t_meet_C = 1/6 →
    d_CD = 2/3 →
    v1 + v2 = 48 →
    v2 * 1/6 = v1 * 1/6 + d_AB →
    Speed_of_Bus d_AB t_meet_C d_CD v1 v2 d_AB t_meet_C d_CD v1 v2 = 40 :=
by
  intros
  rw [Speed_of_Bus, h₁, h₂, h₃, h₄, h₅]
  sorry

end bus_speed_is_correct_l354_354712


namespace correct_options_l354_354422

-- Define the conditions
def condition1 := ∀ x, x ≥ 1 → (y = (x - 2) / (2 * x + 1)) → (y ≥ -1/3 ∧ y < 1/2)
def condition2 := ∀ f, (∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ f_x, f_x = f(2*x - 1)) →
  (∀ y, y = f(x - 1) / (Mathlib.sqrt (x - 1)) → 1 < x ∧ x ≤ 2)
def condition3 := ∀ A, A ⊆ ℝ → (∀ f, (∀ x ∈ A, f x = x^2) → (∃ B, B = {4})) →
  (∃ f1 f2 f3, (f1 = λ x, x = 2) ∧ (f2 = λ x, x = -2) ∧ (f3 = λ x, x = 2 ∨ x = -2))
def condition4 := ∀ f, (∀ x, f (x + 1/x) = x^2 + 1/x^2) → (f m = 4 → m = Mathlib.sqrt 6)

-- The final theorem statement combining all conditions
theorem correct_options:
  ∀ A B C D, 
    (condition1 A) ∧ 
    (condition2 B) ∧
    (condition3 C) ∧
    (condition4 D) → 
    (A ∧ B ∧ C ∧ ¬D) := 
begin
  intros,
  sorry
end

end correct_options_l354_354422


namespace probability_of_B_l354_354743

theorem probability_of_B :
  ∀ (A B : Type) [probability_space A] [probability_space B],
  mutually_exclusive A B →
  P(A) = 1 / 5 →
  P(A ∪ B) = 8 / 15 →
  P(B) = 1 / 3 :=
by
  intros A B h_me h_PA h_PAU
  -- Proof would go here
  sorry

end probability_of_B_l354_354743


namespace max_product_is_600_l354_354850

/-
Given:
m is a four-digit number of the form abab
n is a four-digit number of the form cdcd
m + n is a perfect square

Prove that the largest value of a * b * c * d is 600.
-/

noncomputable def max_product (a b c d : ℕ) : ℕ :=
  if (10 * a + b) + (10 * c + d) = 101 then
    a * b * c * d
  else
    0

theorem max_product_is_600 :
  ∃ (a b c d : ℕ), (m = 101 * (10 * a + b)) ∧ (n = 101 * (10 * c + d)) ∧
  (m + n) % (int.sqrt (m + n)) = 0 ∧
  max_product a b c d = 600 :=
begin
  sorry
end

end max_product_is_600_l354_354850


namespace gas_cost_l354_354058

theorem gas_cost 
  (x : ℝ)
  (h1 : 5 * (x / 5) = x)
  (h2 : 8 * (x / 8) = x)
  (h3 : (x / 5) - 15.50 = (x / 8)) : 
  x = 206.67 :=
by
  sorry

end gas_cost_l354_354058


namespace constant_temperature_l354_354676

def stable_system (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), T (a, b, c) = (1 / 6) * (T (a + 1, b, c) + T (a - 1, b, c) + T (a, b + 1, c) + T (a, b - 1, c) + T (a, b, c + 1) + T (a, b, c - 1))

theorem constant_temperature (T : ℤ × ℤ × ℤ → ℝ) 
    (h1 : ∀ (x : ℤ × ℤ × ℤ), 0 ≤ T x ∧ T x ≤ 1)
    (h2 : stable_system T) : 
  ∃ c : ℝ, ∀ x : ℤ × ℤ × ℤ, T x = c := 
sorry

end constant_temperature_l354_354676


namespace area_of_square_l354_354922

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def is_square (A B C D : (ℝ × ℝ)) : Prop :=
  let dAB := distance A.1 A.2 B.1 B.2 in
  let dBC := distance B.1 B.2 C.1 C.2 in
  let dCD := distance C.1 C.2 D.1 D.2 in
  let dDA := distance D.1 D.2 A.1 A.2 in
  dAB = dBC ∧ dBC = dCD ∧ dCD = dDA

def adjacent_vertices (A B : (ℝ × ℝ)) : Prop :=
  ∃ C D, is_square A B C D

theorem area_of_square :
  adjacent_vertices (0, 3) (3, -4) →
  ∃ side : ℝ, side = distance 0 3 3 (-4) ∧ side ^ 2 = 58 :=
by
  intro h
  use distance 0 3 3 (-4)
  split
  . refl
  . sorry

end area_of_square_l354_354922


namespace books_loaned_out_correct_l354_354617

-- Define the conditions
def total_books : ℕ := 75
def returned_percentage : ℝ := 0.70
def remaining_books : ℕ := 63
def missing_books : ℕ := total_books - remaining_books

-- Define the variable representing the number of books loaned out
def books_loaned_out : ℝ := missing_books / (1 - returned_percentage)

-- The theorem to prove
theorem books_loaned_out_correct : books_loaned_out = 40 := by
  -- Calculate missing books
  have h1 : missing_books = 12 := by
    unfold missing_books
    rw [Nat.sub_add_cancel, rfl]
    sorry
  
  -- Calculate the loaned out books
  have h2 : books_loaned_out = 12 / 0.30 := by
    unfold books_loaned_out
    rw [h1, rfl]
    sorry
  
  -- Validate the resulting calculation
  have h3 : 12 / 0.30 = 40 := by
    calc 12 / 0.30 = 12 / (3 / 10) : rfl
                _ = 12 * (10 / 3) : by sorry
                _ = 40 : by sorry

  -- Complete the proof
  exact Eq.trans h2 h3

end books_loaned_out_correct_l354_354617


namespace sum_a3_a4_a5_eq_84_l354_354148

-- Definitions based on the conditions in the problem
def is_geometric_sequence (a : ℕ → ℕ) :=
  ∃ q : ℕ, ∀ n : ℕ, a (n+1) = a n * q

variables {a : ℕ → ℕ}

-- Given conditions
axiom q_eq_2 : ∃ q, q = 2
axiom sum_first_three_eq_21 : (a 1) + (a 2) + (a 3) = 21

-- Required proof statement
theorem sum_a3_a4_a5_eq_84 (h : is_geometric_sequence a) (hq : q_eq_2) (h_sum : sum_first_three_eq_21) : 
  let q := 2 in (a 3) + (a 4) + (a 5) = 84 :=
by
  sorry

end sum_a3_a4_a5_eq_84_l354_354148


namespace ratio_a_b_l354_354428

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (a * Real.sin x + b * Real.cos x) * Real.exp x

theorem ratio_a_b (a b : ℝ) (h : ∃ x : ℝ, x = Real.pi / 3 ∧ (f x a b).deriv = 0) : a / b = 2 - Real.sqrt 3 :=
by
  sorry

end ratio_a_b_l354_354428


namespace product_calculation_l354_354354

theorem product_calculation : 
  (∏ (k : ℕ) in (finset.range (98)).filter (λ k, k % 2 = 1), (k * (k + 4)) / ((k + 2) * (k + 2))) = (101 / 297) := 
sorry

end product_calculation_l354_354354


namespace collinear_vector_sum_zero_l354_354401

open_locale classical
noncomputable theory

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (A B C O : V)
variables (p q r : ℝ)

theorem collinear_vector_sum_zero 
  (h_collinear : ∃ (l : V → Prop), l A ∧ l B ∧ l C)
  (h_not_on_l : ∀ (l : V → Prop), l O → (¬ l A ∨ ¬ l B ∨ ¬ l C))
  (h_vector_eq : p • (O - A) + q • (O - B) + r • (O - C) = 0) :
  p + q + r = 0 :=
sorry

end collinear_vector_sum_zero_l354_354401


namespace winnie_keeps_balloons_l354_354285

theorem winnie_keeps_balloons : 
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  remainder = 4 :=
by
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  show remainder = 4
  sorry

end winnie_keeps_balloons_l354_354285


namespace sqrt_x_minus_1_meaningful_l354_354463

theorem sqrt_x_minus_1_meaningful (x : ℝ) (h : ∃ y : ℝ, y = sqrt (x - 1)) : x ≥ 1 :=
by sorry

end sqrt_x_minus_1_meaningful_l354_354463


namespace problem_1_problem_2_l354_354023

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end problem_1_problem_2_l354_354023


namespace solve_for_x_l354_354457

theorem solve_for_x (x : ℚ) (h : sqrt ((3 / x) + 3) = 4 / 3) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354457


namespace max_area_tan_theta_l354_354666

theorem max_area_tan_theta (PA AB AE AF PC PB BC AC θ : ℝ) (H1 : PA = 2) (H2 : AB = 2) 
  (H3 : PA^2 + AB^2 = AE^2) (H4 : PA ≠ 0) (H5 : AB ≠ 0) (H6 : PA ≠ AB) 
  (H7 : AF^2 + (AE * AE/AB)^2 = AE^2) 
  (H8 : PA ⟂ BC) (H9 : AC ⟂ BC) (H10 : PA ⟂ ABC) 
  (H11 : AE ⟂ PB) (H12 : AF ⟂ PC) (H13 : ∠BPC = θ) : 
  tan θ = (Real.sqrt 2 / 2) :=
by 
  -- Proof omitted
  sorry

end max_area_tan_theta_l354_354666


namespace problem_1_problem_2_l354_354022

variable {m n x : ℝ}

theorem problem_1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m ^ 2 + 4 * m * n := 
by
  sorry

theorem problem_2 (x : ℝ) (h : x ≠ 0) : ((x + 3) / x - 2) / ((x ^ 2 - 9) / (4 * x)) = -(4  / (x + 3)) :=
by
  sorry

end problem_1_problem_2_l354_354022


namespace excellent_student_receives_under_3_l354_354334

theorem excellent_student_receives_under_3 :
  ∃ k, k ≤ 3 ∧ (∃ e : bool, e → excellent_student e k) :=
by
  -- Given conditions
  let x := 0 -- number of excellent students
  let y := 0 -- number of poor students
  let A := 0 -- number of grades 2 received
  let B := 0 -- number of grades 3 received (B = A - 10 by condition)
  let C := 0 -- number of grades 4 received
  let D := 0 -- number of grades 5 received (D = 3C by condition)
  sorry

end excellent_student_receives_under_3_l354_354334


namespace sum_is_correct_l354_354380

statement 
def sum_of_largest_and_smallest_two_digit_numbers : ℕ :=
  let digits := {0, 3, 5, 7, 8}
  let largest := 87
  let smallest := 30
  largest + smallest

theorem sum_is_correct :
  sum_of_largest_and_smallest_two_digit_numbers = 117 :=
by
  let digits := {0, 3, 5, 7, 8}
  have h1: 87 ∈ {x | ∃ a b, a ≠ b ∧ a ∈ digits ∧ b ∈ digits ∧ x = a * 10 + b ∧ a ≠ 0} :=
    by use [8, 7]; simp
  have h2: 30 ∈ {x | ∃ a b, a ≠ b ∧ a ∈ digits ∧ b ∈ digits ∧ x = a * 10 + b ∧ a ≠ 0} :=
    by use [3, 0]; simp
  have largest_smallest_sum : 87 + 30 = 117 := by norm_num
  exact largest_smallest_sum

end sum_is_correct_l354_354380


namespace reconstruction_impossible_l354_354660

noncomputable def original_triangle (A B C A1 B1 C1 : Point) : Prop :=
  acute_triangle A B C ∧
  foot_altitude A B C A1 ∧  -- A1 is the foot of the altitude from A
  foot_altitude B A C B1 ∧  -- B1 is the foot of the altitude from B
  midpoint A B C1           -- C1 is the midpoint of AB

theorem reconstruction_impossible (A B C A1 B1 C1 : Point) :
  original_triangle A B C A1 B1 C1 →
  ¬ ∃ (A' B' C' : Point), original_triangle A' B' C' A1 B1 C1 ∧ (A = A' ∧ B = B' ∧ C = C') :=
begin
  intro h,
  sorry
end

end reconstruction_impossible_l354_354660


namespace max_value_sqrt_sum_l354_354514

theorem max_value_sqrt_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 6) :
  sqrt (x + 3) + sqrt (y + 3) + sqrt (z + 3) ≤ 3 * sqrt 5 :=
sorry

end max_value_sqrt_sum_l354_354514


namespace count_3_digit_numbers_divisible_by_5_l354_354781

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354781


namespace count_3_digit_numbers_divisible_by_5_l354_354786

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354786


namespace choir_members_correct_l354_354559

def choir_members_condition (n : ℕ) : Prop :=
  150 < n ∧ n < 250 ∧ 
  n % 3 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3

theorem choir_members_correct : ∃ n, choir_members_condition n ∧ (n = 195 ∨ n = 219) :=
by {
  sorry
}

end choir_members_correct_l354_354559


namespace problem1_problem2_l354_354764

noncomputable def setA (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def setB : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}

theorem problem1 (a : ℝ) (h : a = 1) : setA a ∪ setB = {x | 0 ≤ x ∧ x ≤ 4} := by
  sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ setA a → x ∈ setB) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end problem1_problem2_l354_354764


namespace choose_4_from_25_l354_354189

theorem choose_4_from_25 :
  ∃ n : ℕ, n = 12650 ∧ n = nat.choose 25 4 :=
by
  use nat.choose 25 4
  split
  . 
    -- First part of split to show n = 12650
    sorry

  . 
    -- Second part of split nat.choose 25 4 = nat.choose 25 4
    refl
 
end choose_4_from_25_l354_354189


namespace calculate_fraction_l354_354352

theorem calculate_fraction :
  let a := 7
  let b := 5
  let c := -2
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 + c^2) = 460 / 43 :=
by
  sorry

end calculate_fraction_l354_354352


namespace inequality_of_a_seq_l354_354928

noncomputable def a_seq : ℕ → ℝ 
| 1     := 1 / 2
| (k+1) := -a_seq k + 1 / (2 - a_seq k)

theorem inequality_of_a_seq 
  (n : ℕ) (h_n_pos : 0 < n) : 
  (n / (2 * (finset.range n).sum (λi, a_seq i + 1)) - 1)^n ≤ 
  ((finset.range n).sum (λi, a_seq i + 1) / n)^n * 
  ((finset.range n).prod (λi, 1 / (a_seq i + 1) - 1)) :=
sorry

end inequality_of_a_seq_l354_354928


namespace correct_calculation_l354_354283

variable (a b : ℚ)

theorem correct_calculation :
  (a / b) ^ 4 = a ^ 4 / b ^ 4 := 
by
  sorry

end correct_calculation_l354_354283


namespace smallest_divisor_of_repeated_number_l354_354643

theorem smallest_divisor_of_repeated_number (abc : ℕ) (h1 : 100 ≤ abc) (h2 : abc < 1000) :
  ∃ k : ℕ, k > 1 ∧ (1001001 * abc) % k = 0 ∧ ∀ m : ℕ, m > 1 → (1001001 * abc) % m = 0 → k ≤ m :=
begin
  use 101,
  split,
  { norm_num },
  split,
  { sorry }, -- Proof that 1001001 * abc is divisible by 101
  { intros m h3 h4,
    sorry } -- Proof that 101 is the smallest such divisor
end

end smallest_divisor_of_repeated_number_l354_354643


namespace pear_price_is_6300_l354_354587

def price_of_pear (P : ℕ) : Prop :=
  P + (P + 2400) = 15000

theorem pear_price_is_6300 : ∃ (P : ℕ), price_of_pear P ∧ P = 6300 :=
by
  sorry

end pear_price_is_6300_l354_354587


namespace sqrt_pi_squared_minus_six_pi_plus_nine_l354_354982

noncomputable def pi : ℝ := Real.pi

theorem sqrt_pi_squared_minus_six_pi_plus_nine : sqrt (pi^2 - 6 * pi + 9) = pi - 3 :=
by
  sorry

end sqrt_pi_squared_minus_six_pi_plus_nine_l354_354982


namespace collinearity_iff_symmetry_l354_354862

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def P : Point := sorry
noncomputable def Q : Point := sorry
noncomputable def M : Point := sorry
noncomputable def N : Point := sorry
noncomputable def midpoint (A B : Point) : Point := sorry

def symmetric_about (P Q A' : Point) : Prop :=
  let A' := midpoint B C
  vector_length (P - A') = vector_length (Q - A')

def collinear (A M N : Point) : Prop := sorry

theorem collinearity_iff_symmetry :
  (P_on_BC : P ∈ line_segment B C) → 
  (Q_on_BC : Q ∈ line_segment B C) → 
  collinear A M N ↔ symmetric_about P Q (midpoint B C) := 
sorry

end collinearity_iff_symmetry_l354_354862


namespace vasya_claim_is_true_l354_354602

noncomputable def possible_to_form_cube (P : Type) [polyhedron P] (convex : convex P) 
  (triagonal : ∀ f, face f P → triangular f) 
  (hexagonal : ∀ f, face f P → hexagonal f): Prop :=
∃ (parts : list P), (glued parts cube)

theorem vasya_claim_is_true {P : Type} [polyhedron P] 
  (convex : convex P) 
  (triagonal : ∀ f, face f P → triangular f)
  (hexagonal : ∀ f, face f P → hexagonal f) : 
  possible_to_form_cube P convex triagonal hexagonal :=
begin
  sorry
end

end vasya_claim_is_true_l354_354602


namespace problem_1_problem_2_l354_354751

noncomputable def f (x a : ℝ) := x^2 + 2*a*x + 2

theorem problem_1 {x : ℝ} (h₀ : x ∈ set.Icc (-5 : ℝ) 5) :
  (f x 1).minimum = 1 ∧ (f x 1).maximum = 37 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-5 : ℝ) 5 → monotone (λ x, f x a)) ↔ a ∈ set.Iic (-5) ∪ set.Ici 5 :=
sorry

end problem_1_problem_2_l354_354751


namespace limit_ad_div_5d_l354_354855

-- Define the number of digits in the positive integer n
variable (d : ℕ)

-- Define the notion of n_k as described in the problem
def n_k (n : ℕ) (k : ℕ) : ℕ := sorry  -- Number obtained by moving the last k digits to the beginning

-- Define s_m(n)
def s_m (n m : ℕ) : ℕ := sorry  -- Number of values k such that n_k is a multiple of m

-- Define a_d
def a_d (d : ℕ) : ℕ :=
  ∑ n in {n : ℕ | has_d_digits_no_zero (n d) ∧ s_m n 2 + s_m n 3 + s_m n 5 = 2 * d}, 1

-- Prove the limit
theorem limit_ad_div_5d : 
  tendsto (λ d, (a_d d : ℝ) / (5 ^ d)) at_top (𝓝 (1 / 3)) :=
by
  sorry

end limit_ad_div_5d_l354_354855


namespace weekly_milk_consumption_l354_354918

theorem weekly_milk_consumption (total_students : ℕ) (num_girls num_boys monitors : ℕ) 
  (cartons_per_girl cartons_per_boy daily_carton_limit weekly_school_days : ℕ) 
  (girls_ratio boys_ratio : ℝ) 
  (H1 : girls_ratio = 0.4)
  (H2 : boys_ratio = 0.6)
  (H3 : 2 * total_students = 15 * monitors)
  (H4 : monitors = 8)
  (H5 : cartons_per_girl = 2)
  (H6 : cartons_per_boy = 1)
  (H7 : daily_carton_limit = 200)
  (H8 : weekly_school_days = 5)
  (H9 : num_girls = floor (girls_ratio * total_students))
  (H10 : num_boys = floor (boys_ratio * total_students))
  (H11 : num_girls + num_boys = total_students)
  : (num_girls * cartons_per_girl + num_boys * cartons_per_boy) * weekly_school_days = 420 :=
by
  sorry

end weekly_milk_consumption_l354_354918


namespace even_and_odd_implies_zero_l354_354071

theorem even_and_odd_implies_zero (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = -f x) (h2 : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f x = 0 :=
by
  sorry

end even_and_odd_implies_zero_l354_354071


namespace cos_half_angle_l354_354735

open Real

theorem cos_half_angle (α : ℝ) (h_sin : sin α = (4 / 9) * sqrt 2) (h_obtuse : π / 2 < α ∧ α < π) :
  cos (α / 2) = 1 / 3 :=
by
  sorry

end cos_half_angle_l354_354735


namespace sequence_sum_neq_l354_354979

noncomputable def sequenceSum : Nat :=
  sorry  -- Definition to compute sum of the sequence

theorem sequence_sum_neq (a : Fin 200 → ℕ) 
  (h1 : ∀ i : Fin 199, a i.succ = 9 * a i ∨ a i.succ = a i / 2) :
  sequenceSum a ≠ 24^2022 :=
by
  sorry

end sequence_sum_neq_l354_354979


namespace lions_and_turtle_l354_354965

noncomputable def speed_of_first_lion (d t : ℝ) : ℝ := d / t
noncomputable def speed_of_second_lion (s1: ℝ) : ℝ := s1 * 1.5
noncomputable def speed_of_turtle (d t : ℝ) : ℝ := d / t

noncomputable def time_to_catch_up (s1 st d1 d2: ℝ) : ℝ :=
  (d1 - d2) / (s1 - st)

theorem lions_and_turtle :
  ∀ (d1 d2: ℝ),
  (d2 = 32) →
  (d1 = 6) →
  (time_to_catch_up (speed_of_first_lion d1 1) (speed_of_turtle 1 d2) d1 d2) = 2.4 :=
by
  assume d1 d2 hd2 hd1,
  let first_lion_speed := speed_of_first_lion d1 1 in
  let turtle_speed := speed_of_turtle 1 d2 in
  have time_catch := time_to_catch_up first_lion_speed turtle_speed d1 d2,
  sorry

end lions_and_turtle_l354_354965


namespace max_integers_greater_than_10_l354_354255

theorem max_integers_greater_than_10 (s : Fin 9 → ℤ) (h_sum : (∑ i, s i) = 7) : ∃ t, (∑ i, (if s i > 10 then 1 else 0)) ≤ t ∧ t = 8 :=
by
  sorry

end max_integers_greater_than_10_l354_354255


namespace porter_monthly_earnings_l354_354888

/--
Porter earns $8 per day and works 5 times a week. He is promised an extra
50% on top of his daily rate for an extra day each week. There are 4 weeks in a month.
Prove that Porter will earn $208 in a month if he works the extra day every week.
-/
theorem porter_monthly_earnings :
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  in monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208 :=
by
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  show Prop, from monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208

end porter_monthly_earnings_l354_354888


namespace sigma_algebra_cardinality_not_countable_l354_354899

theorem sigma_algebra_cardinality_not_countable (S : Type) 
    [sigma_algebra S] (hs : #S = ω) 
    (hσ : #σ S ≤ ℵ₀) : false := 
sorry

end sigma_algebra_cardinality_not_countable_l354_354899


namespace cell_contains_3_l354_354986

noncomputable def sudoku_probability (get_entry : Fin 9 → Fin 9 → Fin 9) (valid_sudoku : Prop) : ℚ := sorry

-- conditions
def valid_sudoku : Prop :=
∀ i j k l, (0 ≤ i < 9) ∧ (0 ≤ j < 9) ∧ (0 ≤ k < 9) ∧ (0 ≤ l < 9) →
  ((get_entry i j = get_entry k l) → (i = k ∧ j = l)) ∧
  (∀ m, 0 ≤ m < 3 → ∀ n, 0 ≤ n < 3 →
    (get_entry (3 * m) (3 * n) ≠ 0) →
    ∃ a b, get_entry (3 * m + a) (3 * n + b) = get_entry (3 * m) (3 * n))

def predefined_entries : Prop :=
  (get_entry 0 0 = 1) ∧ (get_entry 1 1 = 2)

-- question and final probability answer
theorem cell_contains_3 (get_entry : Fin 9 → Fin 9 → Fin 9) :
  valid_sudoku ∧ predefined_entries →
  sudoku_probability get_entry valid_sudoku = 2 / 21 := sorry

end cell_contains_3_l354_354986


namespace cannot_leave_only_piles_of_three_l354_354265

noncomputable def initial_stones : ℕ := 1001

def is_valid_move (current_stones : ℕ) : Prop :=
  current_stones > 1

def make_move (current_stones : ℕ) : ℕ :=
  (current_stones - 1) + 1

theorem cannot_leave_only_piles_of_three :
  ∀ piles, (∀ pile ∈ piles, pile = 3) →
  (∃ n, n ∈ piles → n ≠ 3) ∨
  (piles_sum = 1002)
  (initial_stones = 1001) :=
sorry

end cannot_leave_only_piles_of_three_l354_354265


namespace point_in_third_quadrant_of_tangent_and_secant_l354_354806

theorem point_in_third_quadrant_of_tangent_and_secant
  (α : ℝ)
  (hα : π / 2 < α ∧ α < π) :
  let P := (Real.tan α, Real.sec α) in
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end point_in_third_quadrant_of_tangent_and_secant_l354_354806


namespace bond_face_value_l354_354848

theorem bond_face_value
  (F : ℝ)
  (S : ℝ)
  (hS : S = 3846.153846153846)
  (hI1 : I = 0.05 * F)
  (hI2 : I = 0.065 * S) :
  F = 5000 :=
by
  sorry

end bond_face_value_l354_354848


namespace no_necessary_100_rotations_l354_354872

/-- 
Given a regular 2n-gon, a pattern consisting of vertices whose numbers, 
when expressed in base 102, contain at least one zero. 
We aim to show that there do not necessarily exist 100 rotations 
of the 2n-gon such that the images of the pattern cover the entire set of 2n vertices.
-/
theorem no_necessary_100_rotations (n : ℕ) (h : n > 50) :
  ¬ (∃ rotations : fin 100 → fin (2 * n), ∀ v : fin (2 * n), ∃ r ∈ rotations, 
      (∃ i : fin (2 * n), i ∈ pattern ∧ rotate_by r i = v)) :=
sorry

/--
Definition of the pattern in terms of vertices whose numbers, when expressed 
in base 102, contain at least one zero.
-/
def pattern (n : ℕ) : set (fin (2 * n)) :=
{ v | ∃ digit, in_base_102(v) digit = 0 }

-- Helper function to check the base 102 representation condition
def in_base_102 {n : ℕ} (v : fin (2 * n)) (digit : ℕ) : ℕ := sorry

-- Helper function to rotate the vertices by r steps
def rotate_by {n : ℕ} (r : fin (100)) (i : fin (2 * n)) := sorry


end no_necessary_100_rotations_l354_354872


namespace largest_integer_in_set_l354_354436

def set_contains (x : ℝ) : Prop := 
  abs (x - 55) <= 11 / 2

theorem largest_integer_in_set :
  ∀ x ∈ {x : ℝ | set_contains x},
  ∃ y : ℕ, y ∈ {n : ℤ | set_contains (n : ℝ)} ∧ (∀ z : ℤ, set_contains (z : ℝ) → z ≤ n) → y = 60 :=
by
  sorry

end largest_integer_in_set_l354_354436


namespace product_of_05_and_2_3_is_1_3_l354_354351

theorem product_of_05_and_2_3_is_1_3 : (0.5 * (2 / 3) = 1 / 3) :=
by sorry

end product_of_05_and_2_3_is_1_3_l354_354351


namespace bridge_length_correct_l354_354623

-- Define the essential parameters
def train_length : ℝ := 150 -- length of the train in meters
def train_speed_kmph : ℝ := 45 -- speed of train in km/hr
def time_to_cross_bridge : ℝ := 30 -- time to cross in seconds
def bridge_length : ℝ := 600 -- correct answer: length of the bridge in meters

-- Conversion factor from km/hr to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Speed of train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Distance train covers in the given time (should be train_length + bridge_length)
def distance_covered : ℝ := train_speed_mps * time_to_cross_bridge

-- Mathematical statement: Prove that the length of the bridge is equal to 600 meters
theorem bridge_length_correct :
  distance_covered - train_length = bridge_length :=
by
  sorry

end bridge_length_correct_l354_354623


namespace difference_max_min_eq_2log2_minus_1_l354_354122

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem difference_max_min_eq_2log2_minus_1 :
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  M - N = 2 * Real.log 2 - 1 :=
by
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  sorry

end difference_max_min_eq_2log2_minus_1_l354_354122


namespace chord_length_of_CD_on_circle_l354_354813

-- Define the conditions and prove the length of the chord intercepted on the circle by line CD
theorem chord_length_of_CD_on_circle :
  (∀ {AB CD : ℝ → ℝ → Prop},
    (AB x y ↔ x - 2 * y - 2 = 0) →
    (CD x y ↔ x - 2 * y + 6 = 0) → 
    (∃ d : ℝ, d = abs (1 - 2 * 1 + 6) / real.sqrt (1 + (-2)^2) ∧
     2 * real.sqrt (9 - d^2) = 4)) :=
by
  sorry

end chord_length_of_CD_on_circle_l354_354813


namespace alley_width_l354_354480

theorem alley_width (a k h : ℝ) (w : ℝ)
  (h₁ : ∀ x, 0 < x) -- Ladder length condition and non-zero
  (h₂ : 0 < k) -- Height from ground to Q is positive
  (h₃ : 0 < h) -- Height from ground to R is positive
  (leng : a > 0) -- Length of ladder is positive
  (angleQ : ∀ θ, θ = 60) -- Angle at Q
  (angleR : ∀ θ, θ = 45) -- Angle at R
  (angleP : ∀ θ, θ = 75) -- Angle between Q and R
  : w = (sqrt 3 - sqrt 2) * (a / 2) := 
sorry

end alley_width_l354_354480


namespace solution_set_l354_354906

-- Given conditions
variable (x : ℝ)

def inequality1 := 2 * x + 1 > 0
def inequality2 := (x + 1) / 3 > x - 1

-- The proof statement
theorem solution_set (h1 : inequality1 x) (h2 : inequality2 x) :
  -1 / 2 < x ∧ x < 2 :=
sorry

end solution_set_l354_354906


namespace establish_model_steps_correct_l354_354221

-- Define each step as a unique identifier
inductive Step : Type
| observe_pose_questions
| propose_assumptions
| express_properties
| test_or_revise

open Step

-- The sequence of steps to establish a mathematical model for population change
def correct_model_steps : List Step :=
  [observe_pose_questions, propose_assumptions, express_properties, test_or_revise]

-- The correct answer is the sequence of steps in the correct order
theorem establish_model_steps_correct :
  correct_model_steps = [observe_pose_questions, propose_assumptions, express_properties, test_or_revise] :=
  by sorry

end establish_model_steps_correct_l354_354221


namespace restaurant_cooks_l354_354668

theorem restaurant_cooks
  (C W : ℕ)
  (h1 : C * 11 = 3 * W)
  (h2 : C * 5 = (W + 12))
  : C = 9 :=
  sorry

end restaurant_cooks_l354_354668


namespace exists_nonidentity_element_with_equal_images_l354_354173

variable {G H : Type} [Group G] [Group H]
variable (ϕ ψ : G →* H)
variable [Fintype G] [Fintype H]

theorem exists_nonidentity_element_with_equal_images (hϕ_surj : Function.Surjective ϕ) (hϕ_noninj : ¬ Function.Injective ϕ) (hψ_surj : Function.Surjective ψ) (hψ_noninj : ¬ Function.Injective ψ) :
  ∃ g ∈ (Set.univ \ {1 : G}), ϕ g = ψ g :=
by
  sorry

end exists_nonidentity_element_with_equal_images_l354_354173


namespace all_S_positive_l354_354870

def a (n : ℕ) : ℝ := (1 / n) * Real.sin (n * Real.pi / 25)
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem all_S_positive : ∀ n : ℕ, n ≤ 100 → S n > 0 := by
  sorry

end all_S_positive_l354_354870


namespace sum_of_entries_eq_l354_354709

noncomputable def table_entry (i j n : ℕ) : ℤ := Int.floor (i * j / (n + 1 : ℤ))

def table_sum (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), ∑ j in Finset.range (n + 1), table_entry i j n

theorem sum_of_entries_eq (n : ℕ) (hn : 1 ≤ n) : table_sum n = 1 / 4 * n^2 * (n - 1) ↔ Nat.Prime (n + 1) :=
sorry

end sum_of_entries_eq_l354_354709


namespace least_value_expression_l354_354611

theorem least_value_expression : ∃ x : ℝ, ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094
∧ ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 := by
  sorry

end least_value_expression_l354_354611


namespace math_problem_l354_354671

theorem math_problem :
  (let a := 6 + (3 / 5);
       b := 8.5 - (1 / 3);
       c := b / 3.5;
       d := 2 + (5 / 18);
       e := (11 / 12)
   in (a - c) * (d + e) = 368 / 27) := by
    let a := 6 + (3 / 5)
    let b := 8.5 - (1 / 3)
    let c := b / 3.5
    let d := 2 + (5 / 18)
    let e := (11 / 12)
    sorry

end math_problem_l354_354671


namespace actual_distance_traveled_l354_354805

theorem actual_distance_traveled 
    (t : ℝ) 
    (D : ℝ)
    (h1 : D = 12 * t)
    (h2 : D + 20 = 16 * t) : 
    D = 60 := 
begin
  sorry
end

end actual_distance_traveled_l354_354805


namespace locus_hyperbola_l354_354185

# Check if noncomputable is required
noncomputable section

open Real

theorem locus_hyperbola (l l' : Type) (pi : Type) [AffineSpace ℝ pi] [AffineSpace ℝ l] [AffineSpace ℝ l']
  (h1 : ∃ (a b c : ℝ), l = line (origin, ⟨a, b, c⟩))
  (h2 : ∃ (d e f : ℝ), l' = line (origin, ⟨d, e, f⟩))
  (h3 : ∃ k : ℝ, l.perpendicular l')
  (h4 : ∃ m : ℝ, l.parallel (plane (origin, ⟨m, 0, 0⟩)) pi)
  (h5 : ∃ v : ℝ, l'.lies_in_plane pi) :
  ∃ (a : ℝ), ∀ (M : pi), distance_to_line M l = distance_to_line M l' → (M.y)^2 - (M.x)^2 = a^2 :=
sorry

end locus_hyperbola_l354_354185


namespace ball_selection_probability_l354_354819

open_locale big_operators

noncomputable def ball_jar_probability : ℚ :=
let total_balls := 15 in
let total_select := nat.choose total_balls 5 in
let select_three_blue := nat.choose 4 3 in
let select_one_green := nat.choose 3 1 in
let select_one_yellow := nat.choose 2 1 in
(select_three_blue * select_one_green * select_one_yellow : ℚ) / total_select

theorem ball_selection_probability :
  ball_jar_probability = 8 / 1001 := by
begin
  unfold ball_jar_probability,
  norm_num,
  rw ←nat.cast_mul,
  norm_cast,
  rw mul_div_mul_right,
  linarith,
  exact dec_trivial,
end


end ball_selection_probability_l354_354819


namespace find_abc_l354_354273

open_locale big_operators

noncomputable def sum_k := ∑ k in finset.range 100, (-1)^(k+1) * (k^3 + k^2 + k + 1) / k.factorial

theorem find_abc : 
  sum_k = 1010101 / 100.factorial - 0 ∧ 1010101 + 100 + 0 = 1010201 :=
by { sorry }

end find_abc_l354_354273


namespace only_valid_M_l354_354386

def digit_sum (n : ℕ) : ℕ :=
  -- definition of digit_sum as a function summing up digits of n
  sorry 

def is_valid_M (M : ℕ) := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digit_sum (M * k) = digit_sum M

theorem only_valid_M (M : ℕ) :
  is_valid_M M ↔ ∃ n : ℕ, ∀ m : ℕ, M = 10^n - 1 :=
by
  sorry

end only_valid_M_l354_354386


namespace remainder_of_m_div_5_l354_354467

theorem remainder_of_m_div_5 (m n : ℕ) (h1 : m = 15 * n - 1) (h2 : n > 0) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l354_354467


namespace area_of_triangle_PQR_l354_354275

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-3, 2)
def Q : (ℝ × ℝ) := (1, 7)
def R : (ℝ × ℝ) := (3, -1)

-- Function to compute the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- Statement of the proof problem
theorem area_of_triangle_PQR : triangle_area P Q R = 16 :=
by
  sorry

end area_of_triangle_PQR_l354_354275


namespace solution_set_l354_354084

variable (f : ℝ → ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom increasing_interval : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f(x) < f(y)
axiom f_of_neg_two : f (-2) = -1
axiom f_of_one : f (1) = 0
axiom functional_equation : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 * x2) = f (x1) + f (x2)

theorem solution_set :
  {x : ℝ | log 2 (abs (f x + 1)) < 0} =
  {x : ℝ | (-4 < x ∧ x < -2) ∨ (-2 < x ∧ x < -1) ∨ (1/4 < x ∧ x < 1/2) ∨ (1/2 < x ∧ x < 1)} :=
by
  sorry

end solution_set_l354_354084


namespace number_of_outfits_l354_354614

variable (num_red_shirts num_green_shirts num_blue_pants num_black_pants num_green_hats num_red_hats : ℕ)
variable (condition_1 : num_red_shirts = 7)
variable (condition_2 : num_green_shirts = 7)
variable (condition_3 : num_blue_pants = 4)
variable (condition_4 : num_black_pants = 5)
variable (condition_5 : num_green_hats = 10)
variable (condition_6 : num_red_hats = 10)
variable (shirt_hat_diff_color : ∀ red_shirt green_hat, red_shirt ∈ {1,...,num_red_shirts} → green_hat ∈ {1,...,num_green_hats} → shirt_hat_diff_color red_shirt green_hat)
variable (blue_pants_with_red_shirt : ∀ red_shirt blue_pants, red_shirt ∈ {1,...,num_red_shirts} → blue_pants ∈ {1,...,num_blue_pants} → blue_pants_with_red_shirt red_shirt blue_pants)

theorem number_of_outfits (h₁ : num_red_shirts = 7)
                          (h₂ : num_green_shirts = 7)
                          (h₃ : num_blue_pants = 4)
                          (h₄ : num_black_pants = 5)
                          (h₅ : num_green_hats = 10)
                          (h₆ : num_red_hats = 10) :
                          ∃ (num_outfits : ℕ), num_outfits = 910 :=
by simp [h₁, h₂, h₃, h₄, h₅, h₆]; exact ⟨910, rfl⟩

end number_of_outfits_l354_354614


namespace conjugate_of_z_is_1_plus_I_find_a_and_b_l354_354068

/-- Define complex number z -/
def z : ℂ := (3 - I) / (2 + I)

/-- Define the conjugate of z -/
def z_conjugate : ℂ := conj z

/-- 1. Prove that the conjugate of z is 1 + I -/
theorem conjugate_of_z_is_1_plus_I : z_conjugate = 1 + I := sorry

/-- Define real variables a and b -/
variable (a b : ℝ)

/-- Define function f(z) -/
def f (z : ℂ) : ℂ := z^2 + a * z + b

/-- Condition: f(z) = conj(z) -/
theorem find_a_and_b (z : ℂ) (hz : z = 1 - I) (hc : f z = conj z) : a = -3 ∧ b = 4 :=
sorry

end conjugate_of_z_is_1_plus_I_find_a_and_b_l354_354068


namespace B_squared_eq_313_l354_354695

noncomputable def g (x : ℝ) : ℝ := sqrt 31 + 48 / x

theorem B_squared_eq_313 :
  let eq := λ x, x = g (g (g (g (g x))))
  let roots := [ (sqrt 31 + sqrt 313) / 2, (sqrt 31 - sqrt 313) / 2 ]
  let B := |roots.head| + |roots.tail.head|
  B^2 = 313 :=
by
  sorry

end B_squared_eq_313_l354_354695


namespace function_B_is_decreasing_on_positive_reals_l354_354612

def f_A (x : ℝ) : ℝ := 3 * x - 2
def f_B (x : ℝ) : ℝ := 9 - x^2
def f_C (x : ℝ) : ℝ := 1 / (x - 1)
def f_D (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem function_B_is_decreasing_on_positive_reals (x : ℝ) (hx : 0 < x) :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 < x2 → f_B x2 < f_B x1 :=
by
  sorry

end function_B_is_decreasing_on_positive_reals_l354_354612


namespace triangle_midline_intercept_l354_354154

noncomputable def proof_triangle_properties (ABC : Triangle)
    (BC : ℝ) (angleC : ℝ) (circumradius : ℝ)
    (H1 : BC = 4)
    (H2 : angleC = 30)
    (H3 : circumradius = 6) : Prop :=
(∃ MN PQ, 
 MN = (sqrt 3 + 2 * sqrt 2) ∧ 
 PQ = (4 * sqrt 3))

theorem triangle_midline_intercept : 
  ∀ (ABC : Triangle)
    (BC : ℝ) (angleC : ℝ) (circumradius : ℝ),
    BC = 4 →
    angleC = 30 →
    circumradius = 6 →
  ∃ MN PQ, 
    MN = sqrt 3 + 2 * sqrt 2 ∧ 
    PQ = 4 * sqrt 3 :=
by 
  intros ABC BC angleC circumradius H1 H2 H3
  apply proof_triangle_properties
  exact H1
  exact H2
  exact H3
  sorry -- Proof skipped as requested


end triangle_midline_intercept_l354_354154


namespace problem_solution_l354_354251

noncomputable def sequence_x : ℕ → ℤ
| 0     := 11
| (n+1) := 3 * sequence_x n + 2 * sequence_y n

noncomputable def sequence_y : ℕ → ℤ
| 0     := 7
| (n+1) := 4 * sequence_x n + 3 * sequence_y n

theorem problem_solution : 
  (sequence_y 1854)^(2018) - 2 * (sequence_x 1854)^(2018) % 2018 = 1825 := 
sorry

end problem_solution_l354_354251


namespace cone_surface_area_l354_354809

theorem cone_surface_area (l : ℝ) (C : ℝ) (h₁ : l = 2) (h₂ : C = 2 * Real.pi) : 
  ∃ S : ℝ, S = 3 * Real.pi ∧ S = Mathlib.pi * (C / (2 * Real.pi)) * ((C / (2 * Real.pi)) + l) := 
by 
  sorry

end cone_surface_area_l354_354809


namespace total_number_of_arithmetic_sets_l354_354367

-- Define what it means for four digits to form an arithmetic sequence
def is_arithmetic_sequence (a b c d : ℕ) (d' : ℕ) : Prop :=
  b = a + d' ∧ c = b + d' ∧ d = c + d'

-- Define the set of digits from 0 to 9
def digits : set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Count the total number of valid sets with the given common difference
def count_sets (d' : ℕ) : ℕ :=
  finset.card { s : finset ℕ | ∃ (a b c d : ℕ), s = {a, b, c, d} ∧ 
                  is_arithmetic_sequence a b c d d' ∧ s ⊆ digits }

-- The theorem to prove
theorem total_number_of_arithmetic_sets : 
  count_sets 1 + count_sets 2 = 11 :=
sorry

end total_number_of_arithmetic_sets_l354_354367


namespace unit_digit_expression_l354_354350

theorem unit_digit_expression : 
  let expr := (2+1) * (2^2+1) * (2^4+1) * (2^8+1) * (2^16+1) * (2^32+1) * (2^64+1)
  in (expr % 10) = 5 :=
by
  sorry

end unit_digit_expression_l354_354350


namespace tangent_ratio_eq_l354_354669

/-- Given circle O with diameter AB, and tangents at A and B. Another tangent at M intersects
the tangents at A and B at points K and L, and intersects the diameter AB at point N. -/
theorem tangent_ratio_eq {O A B M K L N : Point} {c : Circle O}
    (h1 : diameter O A B)
    (h2 : tangent c A K)
    (h3 : tangent c B L)
    (h4 : tangent c M)
    (h5 : M ∈ c)
    (h6 : inter_tangent_at_points K L N A B M) :
  KM / ML = KN / NL :=
sorry

end tangent_ratio_eq_l354_354669


namespace sum_of_possible_x_values_l354_354108

theorem sum_of_possible_x_values (x : ℝ) (h : (x - 2) * (x + 5) = 28) : x ∈ {λ x₀, x₀ = -3 ∨ x₀ = -3} :=
sorry

end sum_of_possible_x_values_l354_354108


namespace quadratic_y_at_x_5_l354_354564

-- Define the quadratic function
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions and question as part of a theorem
theorem quadratic_y_at_x_5 (a b c : ℝ) 
  (h1 : ∀ x, quadratic a b c x ≤ 10) -- Maximum value condition (The maximum value is 10)
  (h2 : (quadratic a b c (-2)) = 10) -- y = 10 when x = -2 (maximum point)
  (h3 : quadratic a b c 0 = -8) -- The first point (0, -8)
  (h4 : quadratic a b c 1 = 0) -- The second point (1, 0)
  : quadratic a b c 5 = -400 / 9 :=
sorry

end quadratic_y_at_x_5_l354_354564


namespace polar_to_cartesian_equation_l354_354141

theorem polar_to_cartesian_equation
  (ρ θ x y : ℝ)
  (h1 : ρ = 2 * cos θ - 4 * sin θ)
  (hx : x = ρ * cos θ)
  (hy : y = ρ * sin θ) :
  (x - 2)^2 - 15 * y^2 = 68 - (y + 8)^2 :=
sorry

end polar_to_cartesian_equation_l354_354141


namespace solve_for_x_l354_354460

noncomputable def x_solution (x : ℝ) : Prop := sqrt (3 / x + 3) = 4 / 3

theorem solve_for_x (x : ℝ) (h : x_solution x) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354460


namespace solve_for_x_l354_354462

noncomputable def x_solution (x : ℝ) : Prop := sqrt (3 / x + 3) = 4 / 3

theorem solve_for_x (x : ℝ) (h : x_solution x) : x = -27 / 11 :=
by
  sorry

end solve_for_x_l354_354462


namespace identify_quadratic_equation_l354_354963

theorem identify_quadratic_equation :
  (∀ b c d : Prop, ∀ (f : ℕ → Prop), f 0 → ¬ f 1 → ¬ f 2 → ¬ f 3 → b ∧ ¬ c ∧ ¬ d) →
  (∀ x y : ℝ,  (x^2 + 2 = 0) = (b ∧ ¬ b → c ∧ ¬ c → d ∧ ¬ d)) :=
by
  intros;
  sorry

end identify_quadratic_equation_l354_354963


namespace part_one_part_two_l354_354510

noncomputable theory

open Complex -- To work with complex numbers

variables {α : ℂ} {a : ℕ → ℂ} {n : ℕ}

-- Define the polynomial with complex coefficients
def polynomial (a : ℕ → ℂ) (n : ℕ) (x : ℂ) : ℂ :=
  ∑ i in Finset.range (n + 1), a i * x^i

-- Define the maximum value used in the problem
def M (a : ℕ → ℂ) (n : ℕ) : ℝ :=
  max 0 (max (finset.image (λ i => complex.abs (a(i) / a(n))) (finset.range n)))

-- The first part of the proof
theorem part_one (α : ℂ) (a : ℕ → ℂ) (n : ℕ) (hα : polynomial a n α = 0) (ha_n : a n ≠ 0) :
  abs α ≤ 1 + M a n :=
sorry

-- The second part of the proof
theorem part_two (α : ℂ) (a : ℕ → ℂ) (n : ℕ) (h1 : polynomial a n α = 0) (ha_n : a n ≠ 0)
  (h2 : ∀ (k : ℕ), k ∈ (finset.range (n + 1)).erase n → complex.abs (a k) ≤ 1) :
  abs α > (complex.abs (a 0)) / (1 + complex.abs (a 0)) :=
sorry

end part_one_part_two_l354_354510


namespace box_depth_is_3_feet_l354_354895

noncomputable def fill_rate : ℝ := 3
noncomputable def fill_time : ℝ := 20
noncomputable def length : ℝ := 5
noncomputable def width : ℝ := 4

def volume : ℝ := fill_rate * fill_time

def box_depth (l w v : ℝ) : ℝ := v / (l * w)

theorem box_depth_is_3_feet :
  box_depth length width volume = 3 := by
  sorry

end box_depth_is_3_feet_l354_354895


namespace mushroom_mistake_l354_354202

theorem mushroom_mistake (p k v : ℝ) (hk : k = p + v - 10) (hp : p = k + v - 7) : 
  ∃ p k : ℝ, ∀ v : ℝ, (p = k + v - 7) ∧ (k = p + v - 10) → false :=
by
  sorry

end mushroom_mistake_l354_354202


namespace geometric_sequence_sum_l354_354146

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end geometric_sequence_sum_l354_354146


namespace evaluate_expression_l354_354618

noncomputable def given_expression : ℝ :=
  |8 - 8 * (3 - 12)^2| - |5 - Real.sin 11| + |2^(4 - 2 * 3) / ((3^2) - 7)|

theorem evaluate_expression : given_expression = 634.125009794 := 
  sorry

end evaluate_expression_l354_354618


namespace line_passes_through_fixed_point_l354_354416

theorem line_passes_through_fixed_point (m : ℝ) : 
  (2 + m) * (-1) + (1 - 2 * m) * (-2) + 4 - 3 * m = 0 :=
by
  sorry

end line_passes_through_fixed_point_l354_354416


namespace correct_options_l354_354421

-- Define the conditions
def condition1 := ∀ x, x ≥ 1 → (y = (x - 2) / (2 * x + 1)) → (y ≥ -1/3 ∧ y < 1/2)
def condition2 := ∀ f, (∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ f_x, f_x = f(2*x - 1)) →
  (∀ y, y = f(x - 1) / (Mathlib.sqrt (x - 1)) → 1 < x ∧ x ≤ 2)
def condition3 := ∀ A, A ⊆ ℝ → (∀ f, (∀ x ∈ A, f x = x^2) → (∃ B, B = {4})) →
  (∃ f1 f2 f3, (f1 = λ x, x = 2) ∧ (f2 = λ x, x = -2) ∧ (f3 = λ x, x = 2 ∨ x = -2))
def condition4 := ∀ f, (∀ x, f (x + 1/x) = x^2 + 1/x^2) → (f m = 4 → m = Mathlib.sqrt 6)

-- The final theorem statement combining all conditions
theorem correct_options:
  ∀ A B C D, 
    (condition1 A) ∧ 
    (condition2 B) ∧
    (condition3 C) ∧
    (condition4 D) → 
    (A ∧ B ∧ C ∧ ¬D) := 
begin
  intros,
  sorry
end

end correct_options_l354_354421


namespace find_k_l354_354318

theorem find_k 
  (k : ℝ) 
  (h1 : 3 * 4 + 4 * -7 = 12) 
  (parallel : ∀ x1 y1 x2 y2 m, ((x2 - x1) ≠ 0) ∧ (m = (y2 - y1) / (x2 - x1)) → (3 - 4 * m = 0))
  : k = -116/3 :=
begin
  sorry
end

end find_k_l354_354318


namespace collinear_midpoints_l354_354166

-- Define A, B, C, D, E, F as points
variables {A B C D E F : Type*} [incidence_geometry A B C D E F]

-- Assume C and D are points on a semicircle with the diameter AB
def on_semicircle (C D A B : Type*) [incidence_geometry C D A B] : Prop :=
  let AB := line_through A B in
  let O := midpoint A B in
  C ∈ semicircle O ∧ D ∈ semicircle O

-- Define the conditions
def conditions (C D A B E F : Type*) [incidence_geometry C D A B E F] : Prop :=
  on_semicircle C D A B ∧
  (∃ E, line_through A C ∩ line_through B D = E) ∧
  (∃ F, line_through A D ∩ line_through B C = F)

-- Define the problem: show the collinearity of the midpoints
theorem collinear_midpoints (C D A B E F : Type*) [incidence_geometry C D A B E F] (h : conditions C D A B E F) : 
  collinear (midpoint A B) (midpoint C D) (midpoint E F) :=
sorry

end collinear_midpoints_l354_354166


namespace find_fraction_l354_354717

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l354_354717


namespace rotation_matrix_determinant_l354_354859

theorem rotation_matrix_determinant (θ : ℝ) (hθ : θ = 58) :
  let R := ![
    #[real.cos (θ), -real.sin (θ)],
    #[real.sin (θ),  real.cos (θ)]
  ] in
  matrix.det R = 1 :=
by
  sorry

end rotation_matrix_determinant_l354_354859


namespace arithmetic_sequence_eighth_term_l354_354491

theorem arithmetic_sequence_eighth_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 8 = 15 := by
  sorry

end arithmetic_sequence_eighth_term_l354_354491


namespace part_I_part_II_l354_354755

-- Definition of functions
def f (x a : ℝ) := |3 * x - a|
def g (x : ℝ) := |x + 1|

-- Part (I): Solution set for f(x) < 3 when a = 4
theorem part_I (x : ℝ) : f x 4 < 3 ↔ (1 / 3 < x ∧ x < 7 / 3) :=
by 
  sorry

-- Part (II): Range of a such that f(x) + g(x) > 1 for all x in ℝ
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x > 1) ↔ (a < -6 ∨ a > 0) :=
by 
  sorry

end part_I_part_II_l354_354755


namespace min_cubes_6_l354_354638

noncomputable def minimum_cubes (front_view side_view : nat → nat) : ℕ :=
if front_view 0 = 3 ∧ side_view 0 = 3 ∧ horizontal_symmetry then 6 else 0

constant front_view : nat → nat
constant side_view : nat → nat
constant horizontal_symmetry : Prop

theorem min_cubes_6 :
  (∀ n, front_view n = 3) →
  (∀ n, side_view n = 3) →
  horizontal_symmetry →
  minimum_cubes front_view side_view = 6 :=
begin
  intros,
  unfold minimum_cubes,
  split_ifs,
  exact h,
  contradiction,
end

end min_cubes_6_l354_354638


namespace expand_polynomial_l354_354372

theorem expand_polynomial (x : ℝ) : (5 * x + 3) * (6 * x ^ 2 + 2) = 30 * x ^ 3 + 18 * x ^ 2 + 10 * x + 6 :=
by
  sorry

end expand_polynomial_l354_354372


namespace choir_members_l354_354911

theorem choir_members (n k c : ℕ) (h1 : n = k^2 + 11) (h2 : n = c * (c + 5)) : n = 300 :=
sorry

end choir_members_l354_354911


namespace cost_of_article_l354_354465

-- Definitions for conditions
def gain_340 (C G : ℝ) : Prop := 340 = C + G
def gain_360 (C G : ℝ) : Prop := 360 = C + G + 0.05 * C

-- Theorem to be proven
theorem cost_of_article (C G : ℝ) (h1 : gain_340 C G) (h2 : gain_360 C G) : C = 400 :=
by sorry

end cost_of_article_l354_354465


namespace sin_sum_of_acute_l354_354541

open Real

theorem sin_sum_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) ≤ sin α + sin β := 
by
  sorry

end sin_sum_of_acute_l354_354541


namespace luigi_pizza_cost_l354_354873

theorem luigi_pizza_cost (num_pizzas pieces_per_pizza cost_per_piece : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : pieces_per_pizza = 5) 
  (h3 : cost_per_piece = 4) :
  num_pizzas * pieces_per_pizza * cost_per_piece / pieces_per_pizza = 80 := by
  sorry

end luigi_pizza_cost_l354_354873


namespace remaining_money_is_correct_l354_354898

def initial_amount : ℕ := 53
def cost_toy_car : ℕ := 11
def number_toy_cars : ℕ := 2
def cost_scarf : ℕ := 10
def cost_beanie : ℕ := 14
def remaining_money : ℕ := 
  initial_amount - (cost_toy_car * number_toy_cars) - cost_scarf - cost_beanie

theorem remaining_money_is_correct : remaining_money = 7 := by
  sorry

end remaining_money_is_correct_l354_354898


namespace ratio_of_areas_l354_354493

variables {A B C D E F G H : Type}

-- Assume all necessary properties of equilateral triangles and the respective points
def equilateral_triangle (t : Type) : Prop := 
sorry -- Define properties of being an equilateral triangle

def is_median (D G : Type) (t : Type) : Prop := 
sorry -- Define properties of D to G being the median of triangle t

def is_intersection_point (H : Type) (l1 l2 : Type) : Prop :=
sorry -- Define properties of H being the intersection point of lines l1 and l2

-- Problem statement in terms of Lean
theorem ratio_of_areas (ABC DEF : Type) (HGC BED : Type) 
  (h1 : equilateral_triangle ABC)
  (h2 : equilateral_triangle DEF)
  (h3 : ∃ D E F, ∃ AB BC AC, D ∈ AB ∧ E ∈ BC ∧ F ∈ AC ∧ 
                      is_median D G DEF ∧ is_intersection_point H DG BC) :
  area HGC / area BED = 1 / 4 :=
sorry

end ratio_of_areas_l354_354493


namespace axis_of_symmetry_l354_354942

theorem axis_of_symmetry:
  let f := λ x: Real, sqrt 3 * sin x * cos x + sin x ^ 2,
      g := λ x: Real, sin(x - π / 3) + 1 / 2 in
  ∃ k : Int, (x = k * π + π / 3 + π / 2) = -π / 6 :=
sorry

end axis_of_symmetry_l354_354942


namespace vector_coordinates_l354_354104

def vec := ℕ → ℤ          -- Define the type of a vector as a function from index to integer

def a : vec := λ i, if i = 0 then 3 else if i = 1 then 5 else if i = 2 then 1 else 0
def b : vec := λ i, if i = 0 then 2 else if i = 1 then 2 else if i = 2 then 3 else 0
def c : vec := λ i, if i = 0 then 4 else if i = 1 then -1 else if i = 2 then -3 else 0

theorem vector_coordinates :
  (2 * a 0 - 3 * b 0 + 4 * c 0 = 16) ∧
  (2 * a 1 - 3 * b 1 + 4 * c 1 = 0) ∧
  (2 * a 2 - 3 * b 2 + 4 * c 2 = -19) :=
by
  simp [a, b, c]
  split
  case left =>
    simp
    ring
  case right =>
    split
    case left =>
      simp
      ring
    case right =>
      simp
      ring

end vector_coordinates_l354_354104


namespace smallest_integer_in_odd_set_l354_354239

theorem smallest_integer_in_odd_set (is_odd: ℤ → Prop)
  (median: ℤ) (greatest: ℤ) (smallest: ℤ) 
  (h1: median = 126)
  (h2: greatest = 153) 
  (h3: ∀ x, is_odd x ↔ ∃ k: ℤ, x = 2*k + 1)
  (h4: ∀ a b c, median = (a+b) / 2 → c = a → a ≤ b)
  : 
  smallest = 100 :=
sorry

end smallest_integer_in_odd_set_l354_354239


namespace time_both_pipes_opened_l354_354271

def fill_rate_p := 1 / 10
def fill_rate_q := 1 / 15
def total_fill_rate := fill_rate_p + fill_rate_q -- Combined fill rate of both pipes

def remaining_fill_rate := 10 * fill_rate_q -- Fill rate of pipe q in 10 minutes

theorem time_both_pipes_opened (t : ℝ) :
  (t / 6) + (2 / 3) = 1 → t = 2 :=
by
  sorry

end time_both_pipes_opened_l354_354271


namespace problem_1_problem_2_l354_354302

-- Problem 1 Lean Statement
def f (b : ℝ) := (3 - b) * b^2

theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  ∃ c : ℝ, c ∈ Icc 0 3 ∧ has_deriv_at (λ b, (3 - b) * b^2) (-3 * c^2 + 6 * c) c := sorry

-- Problem 2 Lean Statement
def f (x : ℝ) := abs (2 * x + 1) - abs (x - 2)

theorem problem_2 (x : ℝ) :
  f x > 2 ↔ x < -5 ∨ 1 < x ∧ x < 2 ∨ x ≥ 2 := sorry

end problem_1_problem_2_l354_354302


namespace tiled_floor_area_l354_354384

namespace TiledFloor

def side_length_cm : ℕ := 30
def num_rows : ℕ := 5
def num_tiles_per_row : ℕ := 8

def tile_area_m² : ℝ := (side_length_cm * side_length_cm) / 10000
def total_tiles : ℕ := num_rows * num_tiles_per_row

def total_area_m² : ℝ := tile_area_m² * total_tiles

theorem tiled_floor_area : total_area_m² = 3.6 := by
  sorry

end TiledFloor

end tiled_floor_area_l354_354384


namespace fn_leq_a_div_1_add_n_minus_1_a_sum_fn_div_k_plus_1_lt_1_l354_354399

variable (f : ℕ → ℝ) (a : ℝ)
variable (n : ℕ)

axiom h₀ : 0 < a ∧ a < 1
axiom h₁ : f 1 = a
axiom h₂ : ∀ n, f (n + 1) ≤ f n / (1 + f n)

theorem fn_leq_a_div_1_add_n_minus_1_a (n : ℕ) : f n ≤ a / (1 + (n - 1) * a) :=
sorry

theorem sum_fn_div_k_plus_1_lt_1 (n : ℕ) : (∑ k in Finset.range n, f (k + 1) / (k + 2)) < 1 :=
sorry

end fn_leq_a_div_1_add_n_minus_1_a_sum_fn_div_k_plus_1_lt_1_l354_354399


namespace range_of_a_l354_354427

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 3 then (a + 1) * x - 2 * a else log 3 x

theorem range_of_a :
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a > -1 :=
by
  intro h
  -- Since we are not required to provide the proof steps, we skip detailed steps.
  -- The statement and structure will build successfully in Lean 4.
  sorry

end range_of_a_l354_354427


namespace sqrt_x_minus_2_range_l354_354471

theorem sqrt_x_minus_2_range (x : ℝ) : (↑0 ≤ (x - 2)) ↔ (x ≥ 2) := sorry

end sqrt_x_minus_2_range_l354_354471


namespace find_13x2_22xy_13y2_l354_354404

variable (x y : ℝ)

theorem find_13x2_22xy_13y2 
  (h1 : 3 * x + 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) 
: 13 * x^2 + 22 * x * y + 13 * y^2 = 184 := 
sorry

end find_13x2_22xy_13y2_l354_354404


namespace cos_is_correct_answer_l354_354657

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def has_zero_point (f : ℝ → ℝ) := ∃ x : ℝ, f x = 0

def cos_has_even_and_zero_point : 
  is_even (λ x : ℝ, Real.cos x) ∧ has_zero_point (λ x : ℝ, Real.cos x) :=
by sorry

def sin_is_not_even : 
  ¬ is_even (λ x : ℝ, Real.sin x) :=
by sorry

def ln_is_not_even : 
  ¬ is_even (λ x : ℝ, Real.log x) :=
by sorry

def ln_has_a_zero_point : 
  has_zero_point (λ x : ℝ, Real.log x) :=
by sorry

def x2plus1_has_no_zero_point : 
  ¬ has_zero_point (λ x : ℝ, x^2 + 1) :=
by sorry

def x2plus1_is_even : 
  is_even (λ x : ℝ, x^2 + 1) :=
by sorry

theorem cos_is_correct_answer : 
  cos_has_even_and_zero_point ∧ sin_is_not_even ∧ ln_is_not_even ∧ ln_has_a_zero_point ∧ x2plus1_is_even ∧ x2plus1_has_no_zero_point :=
by sorry

end cos_is_correct_answer_l354_354657


namespace lion_cub_turtle_time_difference_l354_354967

theorem lion_cub_turtle_time_difference:
  (∀ (x : ℝ) (d : ℝ) (t1 t2 : ℝ), 
    (d = 6 * x) ∧ 
    (t1 = (6 * x - 1) / (x - (1/32))) ∧ 
    (t2 = 3.2) ∧ 
    (t2 = 1 / (1.5 * x + (1/32))) →
  (3.2 - (6 * x - 1) / (x - (1/32)) = 2.4)) :=
begin
  sorry
end

end lion_cub_turtle_time_difference_l354_354967


namespace blue_to_yellow_ratio_is_half_l354_354192

noncomputable section

def yellow_fish := 12
def blue_fish : ℕ := by 
  have total_fish := 42
  have green_fish := 2 * yellow_fish
  exact total_fish - (yellow_fish + green_fish)
def fish_ratio (x y : ℕ) := x / y

theorem blue_to_yellow_ratio_is_half : fish_ratio blue_fish yellow_fish = 1 / 2 := by
  sorry

end blue_to_yellow_ratio_is_half_l354_354192


namespace part1_part2_l354_354736

-- Define all given conditions
variable {A B C AC BC : ℝ}
variable (A_in_range : 0 < A ∧ A < π/2)
variable (B_in_range : 0 < B ∧ B < π/2)
variable (C_in_range : 0 < C ∧ C < π/2)
variable (m_perp_n : (Real.cos (A + π/3) * Real.cos B) + (Real.sin (A + π/3) * Real.sin B) = 0)
variable (cos_B : Real.cos B = 3/5)
variable (AC_value : AC = 8)

-- First part: Prove A - B = π/6
theorem part1 : A - B = π / 6 :=
by
  sorry

-- Second part: Prove BC = 4√3 + 3 given additional conditions
theorem part2 : BC = 4 * Real.sqrt 3 + 3 :=
by
  sorry

end part1_part2_l354_354736


namespace existence_of_a_l354_354720

noncomputable def fx (x : ℝ) : ℝ := x * Real.exp x

noncomputable def gx (x a : ℝ) : ℝ := -(x + 1)^2 + a

theorem existence_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, fx x1 ≤ gx x2 a) → a ∈ Set.Ici (- (1 / Real.exp 1)) :=
begin
  sorry
end

end existence_of_a_l354_354720


namespace count_lines_ax_by_0_l354_354222

/-- The coefficients A and B of the line Ax + By = 0 can take values from the set {0, 1, 2, 3, 5, 7}. 
We want to prove that the number of different lines that can be represented by these equations is 23. -/
theorem count_lines_ax_by_0 :
  let values := {0, 1, 2, 3, 5, 7}
  (∑ A in values, ∑ B in values, if A = 0 ∧ B = 0 then 0
                             else if A = B then 1
                             else 1)
  = 23 := sorry

end count_lines_ax_by_0_l354_354222


namespace polynomial_degree_3_l354_354362

noncomputable def f : Polynomial ℚ := 1 - 12 * X + 3 * X^2 - 4 * X^3 + 7 * X^4
noncomputable def g : Polynomial ℚ := 5 - 2 * X - 6 * X^3 + 15 * X^4

theorem polynomial_degree_3 : ∃ c : ℚ, (f + c • g).degree = 3 ∧ c = -7/15 :=
by
  sorry

end polynomial_degree_3_l354_354362


namespace problem1_problem2_l354_354021

section problems
variable (a x y : ℝ)

-- Definition and theorem for Problem 1
def problem1_expr (a : ℝ) : ℝ := 2 * a^5 + a^7 / a^2

theorem problem1 (a : ℝ) : problem1_expr a = 3 * a^5 :=
by
  sorry

-- Definition and theorem for Problem 2
def problem2_expr (x y : ℝ) : ℝ := (x + y) * (x - y) + x * (2 * y - x)

theorem problem2 (x y : ℝ) : problem2_expr x y = 2 * x * y - y^2 :=
by
  sorry

end problems

end problem1_problem2_l354_354021


namespace Tony_science_degree_years_l354_354941

theorem Tony_science_degree_years (X : ℕ) (Total : ℕ)
  (h1 : Total = 14)
  (h2 : Total = X + 2 * X + 2) :
  X = 4 :=
by
  sorry

end Tony_science_degree_years_l354_354941


namespace parallelogram_midpoints_area_ratio_l354_354665

noncomputable def proof_equivalence (A B C D E F G H I: (ℝ × ℝ)) (midpoint : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)) : Prop :=
  let E := midpoint A D in  -- E is the midpoint of AD
  let F := midpoint C D in  -- F is the midpoint of CD
  let G := midpoint A B in  -- G is the midpoint of AB
  let H := midpoint B C in  -- H is the midpoint of BC
  let I := midpoint E F in  -- I is the midpoint of EF
  area_ratio_triangle_quadrilateral (triangle_area (G, H, I)) (quadrilateral_area (A, E, I, G)) = 1

axiom midpoint_definition (A B : (ℝ × ℝ)): (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def area_ratio_triangle_quadrilateral (area_triangle area_quadrilateral : ℝ): ℝ :=
  area_triangle / area_quadrilateral

noncomputable def triangle_area (v : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

noncomputable def quadrilateral_area (v : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

theorem parallelogram_midpoints_area_ratio (A B C D : ℝ × ℝ) :
  proof_equivalence A B C D midpoint_definition :=
sorry

end parallelogram_midpoints_area_ratio_l354_354665


namespace lateral_surface_area_proof_l354_354558

noncomputable def lateral_surface_area_of_pyramid (A B C S O : Point)
  (hABC_right : is_right_triangle A B C)
  (C_vertex : is_right_angle_vertex C)
  (inclination_angle : has_inclined_angle (arcsin (5 / 13)))
  (SO_height : altitude SO A B C)
  (AO_length : AO.length = 1)
  (BO_length : BO.length = 3 * sqrt(2)) : Real :=
if h : lateral_surface_area A B C S O = 911 / 25 then
  lateral_surface_area A B C S O
else
  sorry

theorem lateral_surface_area_proof
  (A B C S O : Point)
  (hABC_right : is_right_triangle A B C)
  (C_vertex : is_right_angle_vertex C)
  (inclination_angle : has_inclined_angle (arcsin (5 / 13)))
  (SO_height : altitude SO A B C)
  (AO_length : AO.length = 1)
  (BO_length : BO.length = 3 * sqrt(2)) :
  lateral_surface_area_of_pyramid A B C S O hABC_right C_vertex inclination_angle SO_height AO_length BO_length = 911 / 25 :=
sorry

end lateral_surface_area_proof_l354_354558


namespace odd_primes_pq_division_l354_354866

theorem odd_primes_pq_division (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(hp_odd : ¬Even p) (hq_odd : ¬Even q) (hp_gt_hq : p > q) (hm_pos : 0 < m) : ¬(p * q ∣ m ^ (p - q) + 1) :=
by 
  sorry

end odd_primes_pq_division_l354_354866


namespace lions_and_turtle_l354_354964

noncomputable def speed_of_first_lion (d t : ℝ) : ℝ := d / t
noncomputable def speed_of_second_lion (s1: ℝ) : ℝ := s1 * 1.5
noncomputable def speed_of_turtle (d t : ℝ) : ℝ := d / t

noncomputable def time_to_catch_up (s1 st d1 d2: ℝ) : ℝ :=
  (d1 - d2) / (s1 - st)

theorem lions_and_turtle :
  ∀ (d1 d2: ℝ),
  (d2 = 32) →
  (d1 = 6) →
  (time_to_catch_up (speed_of_first_lion d1 1) (speed_of_turtle 1 d2) d1 d2) = 2.4 :=
by
  assume d1 d2 hd2 hd1,
  let first_lion_speed := speed_of_first_lion d1 1 in
  let turtle_speed := speed_of_turtle 1 d2 in
  have time_catch := time_to_catch_up first_lion_speed turtle_speed d1 d2,
  sorry

end lions_and_turtle_l354_354964


namespace eval_fraction_expression_simplify_log_expression_l354_354301

theorem eval_fraction_expression :
  (1 : ℚ) * ((13 : ℚ) / 5) ^ 0 + 2 ^ -2 * ((9 : ℚ) / 4) ^ (1/2) + (25 / 36) ^ (1/2) + real.sqrt((-2 : ℚ)^2) = 3 + 5/24 :=
by sorry

theorem simplify_log_expression :
  (real.log 2) ^ 2 + (real.log 20 / real.log 5) * real.log 2 + real.log 100 = 9 + 7 * real.log 5 :=
by sorry

end eval_fraction_expression_simplify_log_expression_l354_354301


namespace triangle_count_l354_354106

theorem triangle_count (a b : ℕ) : 
  (a + b = 8) ∧ (a + 5 > b) ∧ (b + 5 > a) → 
  ∃ (s : set (ℕ × ℕ)), s = {p | p.1 + p.2 = 8 ∧ p.1 + 5 > p.2 ∧ p.2 + 5 > p.1} ∧ s.finite ∧ s.card = 5 :=
by
  sorry

end triangle_count_l354_354106


namespace midpoint_of_line_and_parabola_l354_354569

def line (x : ℝ) : ℝ := x - 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

theorem midpoint_of_line_and_parabola :
  ∃ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ parabola x2 y2 ∧ y1 = line x1 ∧ y2 = line x2 ∧ midpoint x1 y1 x2 y2 = (3, 2) :=
sorry

end midpoint_of_line_and_parabola_l354_354569


namespace angle_AD_CB1_is_45_l354_354494

def is_parallel (l₁ l₂ : Type) : Prop := sorry
def is_perpendicular (l₁ l₂ : Type) : Prop := sorry
def angle_between_lines (l₁ l₂ : Type) : ℝ := sorry

noncomputable def cube := sorry
variables (A B C D A1 B1 C1 D1 : cube)

theorem angle_AD_CB1_is_45 :
  angle_between_lines A D C B1 = 45 :=
sorry

end angle_AD_CB1_is_45_l354_354494


namespace coplanar_lines_implies_k_eq_neg2_l354_354269

def vector1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + 3 * s, -1 - 2 * k * s, 4 + k * s)
def vector2 (t : ℝ) : ℝ × ℝ × ℝ := (-t, 2 + 3 * t, 5 - 2 * t)

def direction_vector1 (k : ℝ) : ℝ × ℝ × ℝ := (3, -2 * k, k)
def direction_vector2 : ℝ × ℝ × ℝ := (-1, 3, -2)

def is_coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  (u.1 * (v.2 * w.3 - v.3 * w.2) - u.2 * (v.1 * w.3 - v.3 * w.1) + u.3 * (v.1 * w.2 - v.2 * w.1) = 0)

theorem coplanar_lines_implies_k_eq_neg2 :
    ∀ (s t : ℝ) (k : ℝ), is_coplanar (vector1 s k) (vector2 t) (0, 0, 0) → k = -2 :=
by
  sorry

end coplanar_lines_implies_k_eq_neg2_l354_354269


namespace sin_sum_difference_l354_354689

-- The proof problem statement in Lean 4
theorem sin_sum_difference (a b : ℝ) : 
  (sin (a + b) - sin (a - b)) = 2 * cos(a) * sin(b) :=
by
  -- The statement of the needed trigonometric identity
  have sum_to_product_identity : ∀ x y : ℝ, sin(x) - sin(y) = 2 * cos((x + y) / 2) * sin((x - y) / 2),
    -- Proof of the identity is not required, we assume it to be true
    sorry,
  
  -- Let's use the identity to prove the main statement
  sorry

end sin_sum_difference_l354_354689


namespace peter_wins_in_5_rounds_l354_354397

noncomputable def minimumRoundsToWin (original_triangle : Triangle) : ℕ :=
  5

theorem peter_wins_in_5_rounds :
  ∀ (triangle : Triangle), scalene triangle → ∃ min_rounds, min_rounds = minimumRoundsToWin triangle :=
by
  intros triangle h_scalene
  use minimumRoundsToWin triangle
  simp only [minimumRoundsToWin]
  exact ⟨5, rfl⟩

end peter_wins_in_5_rounds_l354_354397


namespace students_play_ball_or_cricket_or_both_l354_354975

theorem students_play_ball_or_cricket_or_both :
  ∀ (B C: ℕ) (B_inter_C: ℕ), 
    B = 7 → C = 8 → B_inter_C = 5 → B + C - B_inter_C = 10 :=
by
  intros B C B_inter_C hB hC hB_inter_C
  rw [hB, hC, hB_inter_C]
  sorry

end students_play_ball_or_cricket_or_both_l354_354975


namespace type_2004_A_least_N_type_B_diff_2004_l354_354176

def game_type_A (N : ℕ) : Prop :=
  ∀ n, (1 ≤ n ∧ n ≤ N) → (n % 2 = 0 → false) 

def game_type_B (N : ℕ) : Prop :=
  ∃ n, (1 ≤ n ∧ n ≤ N) ∧ (n % 2 = 0 → true)


theorem type_2004_A : game_type_A 2004 :=
sorry

theorem least_N_type_B_diff_2004 : ∀ N, N > 2004 → game_type_B N → N = 2048 :=
sorry

end type_2004_A_least_N_type_B_diff_2004_l354_354176


namespace line_not_pass_third_quadrant_l354_354118

theorem line_not_pass_third_quadrant (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
    ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ bx + ay = ab) :=
sorry

end line_not_pass_third_quadrant_l354_354118


namespace find_prime_pair_l354_354052

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair_l354_354052


namespace veg_eaters_l354_354292

variable (n_veg_only n_both : ℕ)

theorem veg_eaters
  (h1 : n_veg_only = 15)
  (h2 : n_both = 11) :
  n_veg_only + n_both = 26 :=
by sorry

end veg_eaters_l354_354292


namespace fractional_eq_solution_l354_354256

-- Define the problem statement
theorem fractional_eq_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
    (frac x (x - 1) + frac 2 (1 - x) = 2) → x = 0 :=
begin
  intro h_eq,
  have h1 : x - 1 ≠ 0, -- x ≠ 1
    sorry,
  have h2 : 1 - x ≠ 0, -- x ≠ -1
    sorry,
  
  -- original equation manipulation and simplification 
  rw ← add_sub_cancel' 2 (frac 2 (1 - x)) at h_eq,
  rw ← sub_add_cancel' 0 (x / (x - 1)) at h_eq,

  have h3 : 1 - x = -(x - 1), 
    sorry,
  
  rw ← h3 at h_eq,
  
  linarith,
end

end fractional_eq_solution_l354_354256


namespace marching_band_total_weight_l354_354831

def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drummer : ℕ := 15
def weight_percussionist : ℕ := 8

def uniform_trumpet : ℕ := 3
def uniform_clarinet : ℕ := 3
def uniform_trombone : ℕ := 4
def uniform_tuba : ℕ := 5
def uniform_drummer : ℕ := 6
def uniform_percussionist : ℕ := 3

def count_trumpet : ℕ := 6
def count_clarinet : ℕ := 9
def count_trombone : ℕ := 8
def count_tuba : ℕ := 3
def count_drummer : ℕ := 2
def count_percussionist : ℕ := 4

def total_weight_band : ℕ :=
  (count_trumpet * (weight_trumpet + uniform_trumpet)) +
  (count_clarinet * (weight_clarinet + uniform_clarinet)) +
  (count_trombone * (weight_trombone + uniform_trombone)) +
  (count_tuba * (weight_tuba + uniform_tuba)) +
  (count_drummer * (weight_drummer + uniform_drummer)) +
  (count_percussionist * (weight_percussionist + uniform_percussionist))

theorem marching_band_total_weight : total_weight_band = 393 :=
  by
  sorry

end marching_band_total_weight_l354_354831


namespace josh_pencils_left_proof_l354_354504

def percent (n : ℕ) (p : ℝ) : ℝ := (p / 100) * n

axiom number_of_pencils_josh_had : ℕ := 142
axiom percentage_given_away : ℝ := 40

noncomputable def pencils_given_away : ℕ :=
  (percent number_of_pencils_josh_had percentage_given_away).toInt

def pencils_left : ℕ :=
  number_of_pencils_josh_had - pencils_given_away

theorem josh_pencils_left_proof : pencils_left = 86 :=
by
  -- proof to be provided
  sorry

end josh_pencils_left_proof_l354_354504


namespace range_of_a_l354_354229

-- Function definition
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else x^2 - 2*x

-- Main theorem statement
theorem range_of_a
  (a : ℝ)
  (h : f (-a) + f(a) ≤ 2 * f 3) :
  -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l354_354229


namespace sports_league_total_games_l354_354330

theorem sports_league_total_games (n_teams divs size_div games_inner games_outer : ℕ) 
  (h_n_teams : n_teams = 16)
  (h_divs : divs = 2)
  (h_size_div : size_div = n_teams / divs)
  (h_games_inner : games_inner = 3)
  (h_games_outer : games_outer = 2)
  : let total_games := (size_div - 1) * games_inner + size_div * games_outer in
    (n_teams * total_games) / 2 = 296 :=
by
  sorry

end sports_league_total_games_l354_354330


namespace general_term_formula_sum_of_first_n_terms_of_b_sum_of_reciprocals_of_b_l354_354747

variable (n : ℕ)
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Given conditions
axiom sum_first_nine_terms : ∑ i in finset.range 9, a (i+1) = 153
axiom point_on_line : ∀ n, a (n + 1) = a n + 3

-- 1. Prove the general term formula for the sequence a_n
theorem general_term_formula : a n = 3 * n + 2 :=
sorry

-- Assuming the definition of sequence b and its sum S
def b (n : ℕ) : ℕ := 3 * n * (2 ^ n) + 2
def S (n : ℕ) : ℕ := ∑ i in finset.range n, b (i + 1)

-- 2. Prove the sum of the first n terms of sequence b
theorem sum_of_first_n_terms_of_b : S n = 3 * (n - 1) * (2 ^ (n + 1)) + 2 * n + 6 :=
sorry

-- 3. Prove the inequality for the sum of reciprocals of sequence b
theorem sum_of_reciprocals_of_b (n : ℕ) : ∑ i in finset.range n, 1 / (b (i + 1)) < 1 / 3 :=
sorry

end general_term_formula_sum_of_first_n_terms_of_b_sum_of_reciprocals_of_b_l354_354747


namespace segment_equality_l354_354742

noncomputable section

open EuclideanGeometry

-- Definitions for Points and Lines
variable (A B C D E F P Q R T : Point)
variable (O1 O2 : Circle)

-- Conditions
variables (h1 : circle O1 ∩ circle O2 = {A, B})
variables (h2 : R ∈ (arc O1 A B))
variables (h3 : T ∈ (arc O2 A B))
variables (h4 : ∃ C, C ∈ line A R ∧ C ∈ circle O2 ∧ C ≠ A ∧ C ≠ R)
variables (h5 : ∃ D, D ∈ line B R ∧ D ∈ circle O2 ∧ D ≠ B ∧ D ≠ R)
variables (h6 : ∃ Q, Q ∈ line A T ∧ Q ∈ circle O1 ∧ Q ≠ A ∧ Q ≠ T)
variables (h7 : ∃ P, P ∈ line B T ∧ P ∈ circle O1 ∧ P ≠ B ∧ P ≠ T)
variables (h8 : ∃ E, E ∈ line P R ∧ E ∈ line T D ∧ E ≠ P ∧ E ≠ R ∧ E ≠ T ∧ E ≠ D)
variables (h9 : ∃ F, F ∈ line T C ∧ F ∈ line R Q ∧ F ≠ T ∧ F ≠ C ∧ F ≠ R ∧ F ≠ Q)

-- Theorem
theorem segment_equality
  (h1 : circle O1 ∩ circle O2 = {A, B})
  (h2 : R ∈ (arc O1 A B))
  (h3 : T ∈ (arc O2 A B))
  (h4 : ∃ C, C ∈ line A R ∧ C ∈ circle O2 ∧ C ≠ A ∧ C ≠ R)
  (h5 : ∃ D, D ∈ line B R ∧ D ∈ circle O2 ∧ D ≠ B ∧ D ≠ R)
  (h6 : ∃ Q, Q ∈ line A T ∧ Q ∈ circle O1 ∧ Q ≠ A ∧ Q ≠ T)
  (h7 : ∃ P, P ∈ line B T ∧ P ∈ circle O1 ∧ P ≠ B ∧ P ≠ T)
  (h8 : ∃ E, E ∈ line P R ∧ E ∈ line T D ∧ E ≠ P ∧ E ≠ R ∧ E ≠ T ∧ E ≠ D)
  (h9 : ∃ F, F ∈ line T C ∧ F ∈ line R Q ∧ F ≠ T ∧ F ≠ C ∧ F ≠ R ∧ F ≠ Q) :
  segment_length A E * segment_length B T * segment_length B R =
  segment_length B F * segment_length A T * segment_length A R := sorry

end segment_equality_l354_354742


namespace infinite_rational_points_l354_354920

theorem infinite_rational_points (x y : ℚ) (h_pos : 0 < x ∧ 0 < y) (h_ineq : x + y ≤ 5) : 
  set.infinite { p : ℚ × ℚ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 ≤ 5 } :=
sorry

end infinite_rational_points_l354_354920


namespace sqrt_pi_expr_equals_pi_sub_3_l354_354984

theorem sqrt_pi_expr_equals_pi_sub_3 : sqrt (π^2 - 6 * π + 9) = π - 3 := by
  -- The proof would go here
  sorry

end sqrt_pi_expr_equals_pi_sub_3_l354_354984


namespace exterior_angle_regular_octagon_l354_354821

theorem exterior_angle_regular_octagon : 
  (∀ (n : ℕ), n = 8 → (∑ i in (finset.range n), exterior_angle i n) = 45) := 
by {
  sorry
}

end exterior_angle_regular_octagon_l354_354821


namespace find_current_l354_354258

theorem find_current :
  ∀ (V R : ℕ), V = 48 → R = 12 → (∃ I : ℕ, I = 48 / R) → ∃ I : ℕ, I = 4 :=
by
  intros V R hV hR hI
  rw [hV, hR] at hI
  cases hI with I hI
  use I
  linarith
  sorry

end find_current_l354_354258


namespace num_3_digit_div_by_5_l354_354784

theorem num_3_digit_div_by_5 : 
  ∃ (n : ℕ), 
  let a := 100 in let d := 5 in let l := 995 in
  (l = a + (n-1) * d) ∧ n = 180 :=
by
  sorry

end num_3_digit_div_by_5_l354_354784


namespace intersection_eq_l354_354468

def A : Set Int := { -1, 0, 1 }
def B : Set Int := { 0, 1, 2 }

theorem intersection_eq :
  A ∩ B = {0, 1} := 
by 
  sorry

end intersection_eq_l354_354468


namespace original_cost_price_l354_354616

theorem original_cost_price 
  (C SP SP_new C_new : ℝ)
  (h1 : SP = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : SP_new = SP - 8)
  (h4 : SP_new = 1.045 * C_new) :
  C = 1600 :=
by
  sorry

end original_cost_price_l354_354616


namespace triangle_inequality_obtuse_angle_l354_354157

variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

def is_point_on_line_segment (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] (a b : P) (p : Q) :=
  sorry  -- Placeholder for the actual definition of a point on a line segment.

theorem triangle_inequality_obtuse_angle (A B C D E: Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  (h1 : angle A B C > 90)
  (h2 : is_point_on_line_segment A B D)
  (h3 : is_point_on_line_segment A C E) :
  dist C D + dist B E > dist B D + dist D E + dist E C :=
sorry

end triangle_inequality_obtuse_angle_l354_354157


namespace function_monotonically_increasing_range_l354_354228

theorem function_monotonically_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ 1 ∧ y ≤ 1 ∧ x ≤ y → ((4 - a / 2) * x + 2) ≤ ((4 - a / 2) * y + 2)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≤ y → a^x ≤ a^y) ∧
  (∀ x : ℝ, if x = 1 then a^1 ≥ (4 - a / 2) * 1 + 2 else true) ↔
  4 ≤ a ∧ a < 8 :=
sorry

end function_monotonically_increasing_range_l354_354228


namespace students_seating_no_adjacent_l354_354932

theorem students_seating_no_adjacent (seats : ℕ) (students : ℕ) : 
  seats = 6 → students = 3 → 
  (number_of_ways : ℕ) = 24 := by
  -- Given conditions
  assume h1 : seats = 6,
  assume h2 : students = 3,
  
  -- Ensure number of ways = 24 proves
  have number_of_ways := 24,
  sorry

end students_seating_no_adjacent_l354_354932


namespace length_range_AB_l354_354006

noncomputable def triangle_inscribed_unit_circle (A B C : Point) : Prop :=
  triangle ABC ∧ acute_triangle ABC ∧ inscribed_in_unit_circle A B C

noncomputable def interior_points_PQ_on_AB_AC (P Q A B C : Point) : Prop :=
  on_segment P A B ∧ on_segment Q A C

noncomputable def ratios_and_collinearity
  (P Q A B C O : Point) : Prop :=
  (dist A P / dist P B = 2) ∧ (dist A Q / dist Q C = 1) ∧ collinear P O Q

theorem length_range_AB 
  (A B C P Q O : Point)
  (h1: triangle_inscribed_unit_circle A B C)
  (h2: interior_points_PQ_on_AB_AC P Q A B C)
  (h3: ratios_and_collinearity P Q A B C O) :
  sqrt 3 < dist A B ∧ dist A B < 2 :=
sorry

end length_range_AB_l354_354006


namespace range_a_implies_not_purely_imaginary_l354_354469

def is_not_purely_imaginary (z : ℂ) : Prop :=
  z.re ≠ 0

theorem range_a_implies_not_purely_imaginary (a : ℝ) :
  ¬ is_not_purely_imaginary ⟨a^2 - a - 2, abs (a - 1) - 1⟩ ↔ a ≠ -1 :=
by
  sorry

end range_a_implies_not_purely_imaginary_l354_354469


namespace relationship_among_a_b_c_l354_354719

-- Define the conditions
def a : ℝ := Real.log 0.3 / Real.log 2
def b : ℝ := 2 ^ 0.1
def c : ℝ := (0.2 : ℝ) ^ 1.3

-- State the theorem
theorem relationship_among_a_b_c : a < c ∧ c < b := sorry

end relationship_among_a_b_c_l354_354719


namespace ellipse_range_m_l354_354474

theorem ellipse_range_m (m : ℝ) :
    (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2 → 
    ∃ (c : ℝ), c = x^2 + (y + 1)^2 ∧ m > 5) :=
sorry

end ellipse_range_m_l354_354474


namespace power_mod_l354_354379

theorem power_mod (h1: 5^2 % 17 = 8) (h2: 5^4 % 17 = 13) (h3: 5^8 % 17 = 16) (h4: 5^16 % 17 = 1):
  5^2024 % 17 = 16 :=
by
  sorry

end power_mod_l354_354379


namespace amount_of_the_bill_l354_354295

-- Define the given conditions as constants
constant TD : ℝ := 150
constant t : ℝ := 9/12 -- time in years
constant r : ℝ := 0.16 -- rate of interest per annum

-- Define the calculation of PV from the given formula
def PV : ℝ := TD / (r * t / (1 + r * t))

-- Define the calculation of FV from PV and TD
def FV : ℝ := PV + TD

-- Prove that FV equals the correct answer given the conditions
theorem amount_of_the_bill : FV = 1550 :=
  by
    -- Lean will expect a proof here, so we put sorry to skip the proof for now
    sorry

end amount_of_the_bill_l354_354295


namespace triple_nested_g_l354_354039

def g : ℤ → ℤ := 
  λ n, if n < 3 then n^2 + 1 else 2 * n + 3

theorem triple_nested_g : g(g(g(1))) = 13 := by
  sorry

end triple_nested_g_l354_354039


namespace geom_seq_S8_value_l354_354741

variables {A n : ℕ}

-- Definition of the geometric sequence sum.
def Sn (A : ℕ) (n : ℕ) : ℤ := 2 - A * 2^(n-1)

-- Geometric sequence property leading to the specific sum.
theorem geom_seq_S8_value (A : ℕ) (h : A = 4) : Sn A 8 = -510 :=
by
  rw [h, Sn]
  norm_num
  sorry

end geom_seq_S8_value_l354_354741


namespace jogging_days_in_second_week_l354_354160

theorem jogging_days_in_second_week
  (daily_jogging_time : ℕ) (first_week_days : ℕ) (total_jogging_time : ℕ) :
  daily_jogging_time = 30 →
  first_week_days = 3 →
  total_jogging_time = 240 →
  ∃ second_week_days : ℕ, second_week_days = 5 :=
by
  intros
  -- Conditions
  have h1 := daily_jogging_time = 30
  have h2 := first_week_days = 3
  have h3 := total_jogging_time = 240
  -- Calculations
  have first_week_time := first_week_days * daily_jogging_time
  have second_week_time := total_jogging_time - first_week_time
  have second_week_days := second_week_time / daily_jogging_time
  -- Conclusion
  use second_week_days
  sorry

end jogging_days_in_second_week_l354_354160


namespace num_valid_assignments_l354_354603

open Finset

def volunteers : Fin 7 := ⟨0, 6⟩

def positions : Fin 4 := ⟨0, 3⟩

def valid_assignment (assign : Fin 7 → Fin 4) : Prop :=
  (∀ i, ∃ j, assign j = i) ∧ 
  (assign 0 ≠ assign 1)

theorem num_valid_assignments :
  (∃ (assign : Fin 7 → Fin 4), valid_assignment assign) = 216 := by
  sorry

end num_valid_assignments_l354_354603


namespace minimum_area_of_triangle_PBC_l354_354663

-- Define the problem conditions
noncomputable def P_on_parabola (P : ℝ × ℝ) := P.2^2 = 2 * P.1
def point_on_y_axis (B C : ℝ × ℝ) := B.1 = 0 ∧ C.1 = 0 ∧ B.2 > C.2
def circle_inscribed (P B C : ℝ × ℝ) := (P.1 - 1)^2 + P.2^2 = 1

-- Define the area function of triangle PBC
def triangle_area (P B C : ℝ × ℝ) : ℝ :=
  |P.1 * (B.2 - C.2)| / 2

-- The proof problem
theorem minimum_area_of_triangle_PBC :
  ∃ P B C : ℝ × ℝ, 
    P_on_parabola P ∧
    point_on_y_axis B C ∧
    circle_inscribed P B C ∧ 
    triangle_area P B C = 8 :=
sorry

end minimum_area_of_triangle_PBC_l354_354663


namespace triangle_divided_by_line_equal_area_l354_354335

theorem triangle_divided_by_line_equal_area (m : ℝ) :
  let A := (0, 0)
  let B := (2, 2)
  let C := (6 * m, 0)
  let L := λ x, 2 * m * x
  let area := λ A B C, 1 / 2 * abs (A.fst * (B.snd - C.snd) + B.fst * (C.snd - A.snd) + C.fst * (A.snd - B.snd))
  ∃ m, L B.fst = B.snd ∧ area A B C / 2 = area A B (2 * m * B.fst, 2 * m * B.snd) ∧ m = 11 / 30 :=
by sorry

end triangle_divided_by_line_equal_area_l354_354335


namespace triangle_df_length_proof_l354_354529

-- Definitions and conditions based on the given problem
variables (D E F P Q R G : Type)  -- Points in the triangle DEF and other relevant points
variables [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P] [MetricSpace Q]
variables [MetricSpace R] [MetricSpace G]

-- Given conditions using lengths
variables (DP EQ ER : ℝ)
variables (DP_len : DP = 27) (EQ_len : EQ = 36) (ER_len : ER = 15)

-- Given the perpendicular medians and the existence of altitude
variables (medians_perpendicular : ∀ (D E F P Q : Type), is_perpendicular DP EQ)
variables (alt_from_E : ∀ (E R D F : Type), is_altitude ER)

-- We need to prove the length of DF is 45
noncomputable def length_DF (DF : ℝ) : Prop :=
  is_equal DF 45

-- The main statement in Lean 4
theorem triangle_df_length_proof : length_DF D F :=
  sorry

end triangle_df_length_proof_l354_354529


namespace total_value_of_coins_l354_354795

theorem total_value_of_coins (num_quarters num_nickels : ℕ) (val_quarter val_nickel : ℝ)
  (h_quarters : num_quarters = 8) (h_nickels : num_nickels = 13)
  (h_total_coins : num_quarters + num_nickels = 21) (h_val_quarter : val_quarter = 0.25)
  (h_val_nickel : val_nickel = 0.05) :
  num_quarters * val_quarter + num_nickels * val_nickel = 2.65 := 
sorry

end total_value_of_coins_l354_354795


namespace triangle_areas_l354_354749

-- Define the problem as a Lean 4 statement
theorem triangle_areas (a x_0 : ℝ) (h : a > 0) (h_area_ABP : (1/4) * a = (1/2)) :
  let f := λ x, x + a / x
  let P := (x_0, f x_0)
  let A := (x_0 + a / (2 * x_0), x_0 + a / (2 * x_0))
  let B := (0, x_0 + a / x_0)
  let slope_tangent := 1 - a / (x_0^2)
  let N := (0, 2 * a / x_0)
  let M := (2 * x_0, 2 * x_0)
  in (1 / 2) * |2 * a / x_0| * |2 * x_0| = 4 :=
by
  sorry

end triangle_areas_l354_354749


namespace ellipse_equation_oa_ob_product_range_l354_354411

theorem ellipse_equation :
  (origin : ℝ × ℝ) 
  (a b c : ℝ)
  (E F D : ℝ × ℝ)
  (h_origin : origin = (0, 0))
  (h_focus_on_x_axis : ∃ x : ℝ, F = (x, 0))
  (h_minor_axis_end_point : D = (0, 2))
  (h_distance : dist D F = 3 * dist E F)
  (h_extension : ∃ t : ℝ, E = (D.1 + t * (F.1 - D.1), D.2 + t * (F.2 - D.2)))
  (ellipse_eq : ∀ x y : ℝ, (x, y) ∈ ellipse_eq ↔ (x^2 / 8 + y^2 / 4 = 1)) :
  True := 
sorry

theorem oa_ob_product_range :
  (A B : ℝ × ℝ)
  (k : ℝ)
  (h_product_of_slopes : slope origin A * slope origin B = -3 / 2)
  (h_coordinates : A = (x1, y1) ∧ B = (x2, y2) ∧ x1 ≠ 0 ∧ x2 ≠ 0)
  (slope_oa : y1 / x1 = k)
  (slope_ob : y2 / x2 = - 3 / (2 * k))
  (dot_product_oa_ob : x1 * x2 + y1 * y2) :
  dot_product_oa_ob ∈ set.Icc (-1 : ℝ) 0 ∪ set.Icc 0 1 ∧ dot_product_oa_ob ≠ 0 :=
sorry

end ellipse_equation_oa_ob_product_range_l354_354411


namespace sector_angle_l354_354557

theorem sector_angle (l S : ℝ) (r α : ℝ) 
  (h_arc_length : l = 6)
  (h_area : S = 6)
  (h_area_formula : S = 1/2 * l * r)
  (h_arc_formula : l = r * α) : 
  α = 3 :=
by
  sorry

end sector_angle_l354_354557


namespace triangle_proof_l354_354156

-- Assume the given conditions for the triangle ABC
variable (A B C : Real)
variable (AB AC BC : ℝ)
variable (angle_A : ℝ)

-- Given conditions
axiom h1 : AB = 2
axiom h2 : AC = 3
axiom h3 : angle_A = π / 3

-- Proof requirements:
-- 1. Prove that BC = sqrt(7)
-- 2. Prove that cos(A - C) = 5 * sqrt(7) / 14

theorem triangle_proof (A B C : Real) (AB AC BC : ℝ) (angle_A : ℝ) 
  (h1 : AB = 2) (h2 : AC = 3) (h3 : angle_A = π / 3) : 
  (BC = sqrt 7) ∧ (cos (angle_A - angle (A - C)) = 5 * sqrt 7 / 14) :=
by
  sorry

end triangle_proof_l354_354156


namespace count_3_digit_numbers_divisible_by_5_l354_354778

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354778


namespace difference_of_cubes_not_div_by_twice_diff_l354_354890

theorem difference_of_cubes_not_div_by_twice_diff (a b : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_neq : a ≠ b) :
  ¬ (2 * (a - b)) ∣ ((a^3) - (b^3)) := 
sorry

end difference_of_cubes_not_div_by_twice_diff_l354_354890


namespace speed_on_second_day_l354_354633

noncomputable def speed_first_day : ℝ := 5  -- km/hr
noncomputable def distance_home_school : ℝ := 2.5  -- km
noncomputable def time_late_first_day : ℝ := 7 / 60  -- convert minutes to hours
noncomputable def time_early_second_day : ℝ := 8 / 60  -- convert minutes to hours

theorem speed_on_second_day (v : ℝ) : 
  let correct_time := (distance_home_school / speed_first_day) - time_late_first_day in
  let second_day_time := correct_time - time_early_second_day in
  distance_home_school / second_day_time = v →
  v = 10 := 
by
  sorry

end speed_on_second_day_l354_354633


namespace path_count_l354_354309

def is_valid_step (p q : ℤ × ℤ) : Prop :=
  (p.1 = q.1 ∧ abs (p.2 - q.2) = 1) ∨ (p.2 = q.2 ∧ abs (p.1 - q.1) = 1)

def in_boundary (p : ℤ × ℤ) : Prop :=
  ¬(-3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3)

def valid_path (path : List (ℤ × ℤ)) : Prop :=
  path.length = 21 ∧
  (path.head = some (-5, -5)) ∧
  (path.last = some (5, 5)) ∧
  (∀ (i : ℕ), i < 20 → is_valid_step (path.nth_le i sorry) (path.nth_le (i + 1) sorry)) ∧
  (∀ (p : ℤ × ℤ), p ∈ path → in_boundary p)

theorem path_count : ∀ (paths : Finset (List (ℤ × ℤ))), paths.card = 4252 :=
sorry

end path_count_l354_354309


namespace area_ratio_triangle_quadrilateral_l354_354836
noncomputable theory

-- Definition of the problem conditions
def XY : ℚ := 20
def YZ : ℚ := 28
def XZ : ℚ := 32
def XP : ℚ := 8
def XQ : ℚ := 5

-- Definition of points P and Q
axiom on_XY (P : XY ≠ 0) : XP / XY < 1
axiom on_XZ (Q : XZ ≠ 0) : XQ / XZ < 1

-- Main theorem
theorem area_ratio_triangle_quadrilateral : 
  (area_ratio_of_triangle_XPQ_to_quadrilateral_PQZY XP P XQ Q XY YZ XZ = 924 / 5175) : sorry

end area_ratio_triangle_quadrilateral_l354_354836


namespace identify_increasing_function_on_R_l354_354659

open Real

theorem identify_increasing_function_on_R 
    (f_A : ℝ → ℝ) (f_B : ℝ → ℝ) (f_C : ℝ → ℝ) (f_D : ℝ → ℝ) 
    (domA : ∀ x : ℝ, True) (decA : ∀ x y : ℝ, x < y → f_A(x) > f_A(y)) 
    (domB : ∀ x : ℝ, True) (incB : ∀ x y : ℝ, x < y → f_B(x) < f_B(y)) 
    (domC : ∀ x : ℝ, x > 0 → True) (incC : ∀ x y : ℝ, x > 0 → y > 0 → x < y → f_C(x) < f_C(y))
    (domD : ∀ x : ℝ, True) 
    (incD_pos : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_D(x) < f_D(y)) 
    (decD_neg: ∀ x y : ℝ, x < 0 → y < 0 → x < y → f_D(x) > f_D(y)) :
    ∃ f : ℝ → ℝ, (domB = ∀ x, True) ∧ (f = f_B) :=
begin
    use f_B,
    split,
    { exact domB },
    { refl }
end

end identify_increasing_function_on_R_l354_354659


namespace simplify_fraction_l354_354903

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 2 + 1) + 1 / (Real.sqrt 5 - 2))) =
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) :=
by
  sorry

end simplify_fraction_l354_354903


namespace new_person_weight_l354_354621

-- Define the initial conditions
def initial_average_weight (w : ℕ) := 6 * w -- The total weight of 6 persons

-- Define the scenario where the average weight increases by 2 kg
def total_weight_increase := 6 * 2 -- The total increase in weight due to an increase of 2 kg in average weight

def person_replaced := 75 -- The weight of the person being replaced

-- Define the expected condition on the weight of the new person
theorem new_person_weight (w_new : ℕ) :
  initial_average_weight person_replaced + total_weight_increase = initial_average_weight (w_new / 6) →
  w_new = 87 :=
sorry

end new_person_weight_l354_354621


namespace edward_initial_amount_l354_354045

theorem edward_initial_amount (spent received final_amount : ℤ) 
  (h_spent : spent = 17) 
  (h_received : received = 10) 
  (h_final : final_amount = 7) : 
  ∃ initial_amount : ℤ, (initial_amount - spent + received = final_amount) ∧ (initial_amount = 14) :=
by
  sorry

end edward_initial_amount_l354_354045


namespace ratio_of_values_l354_354008

-- Define the geometric sequence with first term and common ratio
def geom_seq_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def geom_seq_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Sum of the first n terms for given sequence
noncomputable def S_n (n : ℕ) : ℚ :=
  geom_seq_sum (3/2) (-1/2) n

-- Define the function f(t) = t - 1/t
def f (t : ℚ) : ℚ := t - 1 / t

-- Define the maximum and minimum values of f(S_n) and their ratio
noncomputable def ratio_max_min_values : ℚ :=
  let max_val := f (3/2)
  let min_val := f (3/4)
  max_val / min_val

-- The theorem to prove the ratio of the maximum and minimum values
theorem ratio_of_values :
  ratio_max_min_values = -10/7 := by
  sorry

end ratio_of_values_l354_354008


namespace min_period_is_pi_interval_monotonic_increase_max_min_values_on_interval_l354_354094

noncomputable def f (x : ℝ) : ℝ := 2 * (√3 * Real.cos x - Real.sin x) * Real.sin x

theorem min_period_is_pi : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem interval_monotonic_increase (k : ℤ) : 
  -Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi → 
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2 :=
sorry

theorem max_min_values_on_interval : 
  (∃ x_min x_max ∈ (Set.Icc 0 (Real.pi / 4)), f x_min = 0 ∧ f x_max = 1) := 
sorry

end min_period_is_pi_interval_monotonic_increase_max_min_values_on_interval_l354_354094


namespace mirror_symmetric_polyhedra_equal_volume_l354_354209
-- Import the necessary library

-- Define the problem statement in Lean 4
theorem mirror_symmetric_polyhedra_equal_volume 
    (P Q : Polyhedron)
    (mirror_symmetric : P.symmetry = Q.symmetry)
    (decomposable_into_symmetric_tetrahedrons : ∀ P Q, (P.symmetry = Q.symmetry) → (P.decomposable Q → Symmetric_tetrahedrons))
    : P.volume = Q.volume := 
by
  sorry

end mirror_symmetric_polyhedra_equal_volume_l354_354209


namespace inequality_AM_GM_HM_l354_354115

variable {x y k : ℝ}

-- Define the problem conditions
def is_positive (a : ℝ) : Prop := a > 0
def is_unequal (a b : ℝ) : Prop := a ≠ b
def positive_constant_lessthan_two (c : ℝ) : Prop := c > 0 ∧ c < 2

-- State the theorem to be proven
theorem inequality_AM_GM_HM (h₁ : is_positive x) 
                             (h₂ : is_positive y) 
                             (h₃ : is_unequal x y) 
                             (h₄ : positive_constant_lessthan_two k) :
  ( ( ( (x + y) / 2 )^k > ( (x * y)^(1/2) )^k ) ∧ 
    ( ( (x * y)^(1/2) )^k > ( ( 2 * x * y ) / ( x + y ) )^k ) ) :=
by
  sorry

end inequality_AM_GM_HM_l354_354115


namespace perpendicular_line_exists_l354_354702

-- Definitions based on conditions
def point := (1 : ℝ, 1 : ℝ)
def line1 (x y : ℝ) : Prop := 2 * x - y = 0

-- The statement to be proved: 
theorem perpendicular_line_exists (x y : ℝ) :
  (x, y) = point → line1 x y → ∃ (m c : ℝ), y = m * x + c ∧ m = -1/2 ∧ c = 3/2 :=
sorry

end perpendicular_line_exists_l354_354702


namespace no_max_value_if_odd_and_symmetric_l354_354413

variable (f : ℝ → ℝ)

-- Definitions:
def domain_is_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_symmetric_about_1_1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 - x) = 2 - f x

-- The theorem stating that under the given conditions there is no maximum value.
theorem no_max_value_if_odd_and_symmetric :
  domain_is_R f → is_odd_function f → is_symmetric_about_1_1 f → ¬∃ M : ℝ, ∀ x : ℝ, f x ≤ M := by
  sorry

end no_max_value_if_odd_and_symmetric_l354_354413


namespace arrange_numbers_in_circle_l354_354969

theorem arrange_numbers_in_circle (n : ℕ) 
  (h : ∀ (circle : list ℕ), list.perm circle (list.range (n+1))) 
  (swap_condition : ∀ (a b : ℕ), |a - b| > 1 ↔ true): 
  ∃ (swapped_circle : list ℕ), swapped_circle = list.range (n + 1) :=
sorry

end arrange_numbers_in_circle_l354_354969


namespace tyson_distance_at_meet_l354_354844

-- Definitions of the conditions according to the problem statement.

def distance_between_A_and_B : ℝ := 80
def jenna_start_time_lead : ℝ := 1.5
def jenna_speed : ℝ := 3.5
def andre_speed : ℝ := 4.7
def tyson_speed : ℝ := 2.8

theorem tyson_distance_at_meet :
  let t := (distance_between_A_and_B - jenna_speed * jenna_start_time_lead) / (jenna_speed + tyson_speed)
  in tyson_speed * t = 33.25 := sorry

end tyson_distance_at_meet_l354_354844


namespace three_digit_numbers_divisible_by_5_l354_354793

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l354_354793


namespace exists_similar_triangle_with_double_area_l354_354356

open Real

noncomputable def transformation (p : ℝ × ℝ) : ℝ × ℝ := 
  (p.1 - p.2, p.1 + p.2)

theorem exists_similar_triangle_with_double_area (S : ℝ) (vertices_X : List (ℝ × ℝ))
  (h_vertex : ∀ v ∈ vertices_X, ∃ x y : ℤ, v = (x, y))
  (h_area : ∃ X, triangle X ∧ area X = S) :
  ∃ X', triangle X' ∧
  (∀ v ∈ X'.vertices, ∃ x y : ℤ, v = (x, y)) ∧
  area X' = 2 * S :=
sorry

end exists_similar_triangle_with_double_area_l354_354356


namespace probability_black_and_redKing_blackQueen_l354_354331

-- Definitions related to the deck and properties
inductive Suit 
| spades 
| hearts 
| diamonds 
| clubs 

inductive Rank 
| ace
| r2
| r3
| r4
| r5
| r6
| r7
| r8
| r9
| r10
| jack
| queen
| king

structure Card where
  rank : Rank
  suit : Suit

def is_black (c : Card) : Prop :=
  c.suit = Suit.spades ∨ c.suit = Suit.clubs

def is_red_king (c : Card) : Prop :=
  c.rank = Rank.king ∧ (c.suit = Suit.hearts ∨ c.suit = Suit.diamonds)

def is_black_queen (c : Card) : Prop :=
  c.rank = Rank.queen ∧ (c.suit = Suit.spades ∨ c.suit = Suit.clubs)

def card_deck := List.range 52 -- Assume 52 unique Cards

-- Main theorem statement
theorem probability_black_and_redKing_blackQueen : 
  (let total_black := 26 in
   let total_redKing_blackQueen := 4 in
   (total_black * total_redKing_blackQueen) / (52 * 51) = (2 / 51)) := 
by 
  sorry

end probability_black_and_redKing_blackQueen_l354_354331


namespace p_more_than_q_l354_354574

def stamps (p q : ℕ) : Prop :=
  p / q = 7 / 4 ∧ (p - 8) / (q + 8) = 6 / 5

theorem p_more_than_q (p q : ℕ) (h : stamps p q) : p - 8 - (q + 8) = 8 :=
by {
  sorry
}

end p_more_than_q_l354_354574


namespace midpoint_product_l354_354508

theorem midpoint_product (x y : ℝ) :
  (∃ B : ℝ × ℝ, B = (x, y) ∧ 
  (4, 6) = ( (2 + B.1) / 2, (9 + B.2) / 2 )) → x * y = 18 :=
by
  -- Placeholder for the proof
  sorry

end midpoint_product_l354_354508


namespace game_termination_l354_354540

def binary_val (n : ℕ) : ℕ :=
  Nat.find_greatest (fun k => 2^k ∣ n) n

theorem game_termination (a : ℕ) (S : List ℕ) (h1 : 0 < a)
  (h2 : ∀ n ∈ S, 0 < n) (h3 : ∀ n ∈ S, binary_val n < binary_val a) : 
  ∃ k, ∀ n ∈ S, ∀ m ≤ k, n = 2^m * (n / 2^m) → m = 0 :=
by
  sorry

end game_termination_l354_354540


namespace set_intersection_l354_354765

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem set_intersection : A ∩ B = {-1, 0} := by
  sorry

end set_intersection_l354_354765


namespace trapezium_area_ratio_l354_354003

noncomputable def trapezium {O A B C D : Type*} [Circle O] (AB CD : LineSegment) (AOB COD : Angle) 
  (h1 : ABCD.isTrapezium)
  (h2 : AB.isParallelTo CD)
  (h3 : COD = 3 * AOB) : Prop :=
  area (BOC : Triangle) / area (AOB : Triangle) = 3 / 2

theorem trapezium_area_ratio {O A B C D : Type*} [Circle O] (AB CD : LineSegment) (AOB COD : Angle)
  (h1 : ABCD.isTrapezium)
  (h2 : AB.isParallelTo CD)
  (h3 : COD = 3 * AOB)
  (h4 : AB.length = 2 / 5 * CD.length) : trapezium AB CD AOB COD h1 h2 h3 :=
sorry

end trapezium_area_ratio_l354_354003


namespace expected_number_of_inspections_is_seven_halves_l354_354087

open ProbabilityTheory

noncomputable def expected_value_of_number_of_inspections
  (inspection_results : List Bool)
  (defective_count : ℕ) : ℚ :=
let total_items := inspection_results.length in
let inspections := (1:ℚ) + (1:ℚ) / 2 + (1:ℚ) / 2 * (defective_count:ℚ) * (total_items-defective_count:ℚ) in
if total_items = 5 ∧ defective_count = 2 then inspections else sorry

theorem expected_number_of_inspections_is_seven_halves :
  expected_value_of_number_of_inspections [true, true, false, false, false] 2 = 7 / 2 :=
sorry

end expected_number_of_inspections_is_seven_halves_l354_354087


namespace first_term_of_arithmetic_progression_l354_354615

noncomputable def arithmetic_progression_smallest_sum (a n : ℕ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a_n n = a + ↑(n - 1)

theorem first_term_of_arithmetic_progression (a_1 : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) (sum_nat: ℕ):
  (arithmetic_progression_smallest_sum a_1 sum_nat a_n) →
  S 2022 < S n ∀ (n : ℕ) →  (-2022 < a_1 ∧ a_1 < -2021) :=
by 
  sorry

end first_term_of_arithmetic_progression_l354_354615


namespace length_of_MN_l354_354257

theorem length_of_MN :
  ∀ (A B C B' D M N : ℝ × ℝ)
    (h1 : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 2022^2)
    (h2 : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2023^2)
    (h3 : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 2024^2)
    (h4 : B'.1 = -B.2 + A.1 ∧ B'.2 = B.1 - A.2)
    (h5 : D.1 = ((A.1 - C.1) * ((B'.1 * (A.1 - C.1)) + (B'.2 * (A.2 - C.2)))) 
          / ((A.1 - C.1)^2 + (A.2 - C.2)^2) + A.1 ∧
          D.2 = ((A.2 - C.2) * ((B'.1 * (A.1 - C.1)) + (B'.2 * (A.2 - C.2)))) 
          / ((A.1 - C.1)^2 + (A.2 - C.2)^2) + A.2)
    (h6 : M.1 = (B.1 + B'.1) / 2 ∧ M.2 = (B.2 + B'.2) / 2)
    (h7 : ∃ P : ℝ × ℝ, circumcenter P C D M N ∧ P ≠ M ∧ on_ray P B M),
  dist M N = 2 * real.sqrt 2 := by
  sorry

end length_of_MN_l354_354257


namespace seq_arithmetic_sum_formula_arithmetic_subsequence_l354_354151

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 1 / 3
| (n+1) := (1 / 3) * (a n) - (2 / 3 ^ (n + 1))

-- Define the sequence 3^n * a_n
def b (n : ℕ) : ℝ := 3 ^ n * a n

-- Define the sum S_n of the first n terms of the sequence a
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Theorem 1: Prove that the sequence {3^n * a_n} is an arithmetic sequence.
theorem seq_arithmetic : ∃ d, ∀ n, b (n + 1) - b n = d := sorry

-- Theorem 2: Prove that S_n = n / 3^n.
theorem sum_formula (n : ℕ) : S n = n / 3 ^ n := sorry

-- Theorem 3: Prove that there exist positive integers p, q, r (p < q < r) 
-- such that S_p, S_q, S_r form an arithmetic sequence with p = 1, q = 2, r = 3.
theorem arithmetic_subsequence : ∃ p q r : ℕ, p < q ∧ q < r ∧ S p + S r = 2 * S q ∧ p = 1 ∧ q = 2 ∧ r = 3 := sorry

end seq_arithmetic_sum_formula_arithmetic_subsequence_l354_354151


namespace repeating_decimal_sum_l354_354241

theorem repeating_decimal_sum :
  let y := 4.78787878 -- representing the repeating decimal
  in let fraction := (474 / 99) -- derived from the repeating decimal
  in let reduced_fraction := (158 / 33) -- reduced to lowest terms
  in (158 + 33 = 191) :=
by
  sorry

end repeating_decimal_sum_l354_354241


namespace translate_upwards_l354_354489

theorem translate_upwards (x : ℝ) : (2 * x^2) + 2 = 2 * x^2 + 2 := by
  sorry

end translate_upwards_l354_354489


namespace prob_B_solve_l354_354579

theorem prob_B_solve :
  let P_A := 2 / 3 in
  let P_A_or_B := 0.9166666666666666 in
  let P_A_and_B (P_B : ℝ) := P_A * P_B in
  ∃ P_B : ℝ, P_A_or_B = P_A + P_B - P_A_and_B P_B ∧ P_B = 3 / 4 := 
by
  let P_A : ℝ := 2 / 3
  let P_A_or_B : ℝ := 0.9166666666666666
  let P_B : ℝ := 3 / 4
  let P_A_and_B := P_A * P_B
  use P_B
  split
  { -- showing P_A_or_B = P_A + P_B - P_A_and_B
    norm_num,
    ring
  },
  trivial

end prob_B_solve_l354_354579


namespace sunflower_cans_l354_354842

theorem sunflower_cans (total_seeds seeds_per_can : ℕ) (h_total_seeds : total_seeds = 54) (h_seeds_per_can : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end sunflower_cans_l354_354842


namespace z_eq_conj_then_norm_zero_or_six_point_in_first_quadrant_then_a_range_l354_354062

-- Define z as given in the problem
def z (a : ℝ) : ℂ := ((1 - ℂ.I) * a^2 - 3 * a + 2 + ℂ.I)

-- Prove that if z = conj(z), then |z| = 0 or 6
theorem z_eq_conj_then_norm_zero_or_six (a : ℝ) (h : z a = conj (z a)) : abs (z a) = 0 ∨ abs (z a) = 6 :=
by
  sorry

-- Prove the range of a if the point corresponding to z is in the first quadrant
theorem point_in_first_quadrant_then_a_range (a : ℝ) (h1 : 0 < re (z a)) (h2 : 0 < im (z a)) : -1 < a ∧ a < 1 :=
by
  sorry

end z_eq_conj_then_norm_zero_or_six_point_in_first_quadrant_then_a_range_l354_354062


namespace area_smallest_triangle_l354_354685

-- Define the side length of the equilateral triangle
def side_length_triangle := 2

-- Define the side lengths of the external squares
def side_length_square := 2

-- Define the function that calculates the area of an equilateral triangle
def equilateral_triangle_area (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- Assumption that the minimum containing triangle is aligned correctly
axiom minimal_containing_triangle : 
  ∃ (s : ℝ), s = 4 + 2 * sqrt 3

-- Proof statement
theorem area_smallest_triangle : 
  equilateral_triangle_area (4 + 2 * sqrt 3) = 12 + 7 * sqrt 3 :=
sorry

end area_smallest_triangle_l354_354685


namespace problem_statement_l354_354199

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l354_354199


namespace find_length_of_AB_l354_354123

def unit_square (A E F D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ E = (1, 0) ∧ F = (1, 1) ∧ D = (0, 1)

def is_solution (W : ℝ) : Prop :=
  W = (1 + Real.sqrt 5) / 2

theorem find_length_of_AB 
  (A E F D B C : ℝ × ℝ)
  (unit_square (A E F D)) 
  (BCFE : Rect (BC : C ≠ F ∧ F ≠ E))
  (W : ℝ) :
  ((W - 1) * W = 1) ∧ (W > 0) ->
  is_solution W :=
by
  sorry

end find_length_of_AB_l354_354123


namespace integral_f_values_l354_354179

open Set Function IntervalIntegral

theorem integral_f_values (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Icc 0 1)) 
  (h_range : ∀ x ∈ Icc 0 1, f x ∈ Icc 0 1)
  (h_prop : ∀ x ∈ Icc 0 1, f (f x) = 1) : 
  (3/4 : ℝ) < ∫ x in (0 : ℝ)..(1 : ℝ), f x ∧ ∫ x in (0 : ℝ)..(1 : ℝ), f x ≤ (1 : ℝ) := 
sorry

end integral_f_values_l354_354179


namespace solution_l354_354100

open Real EuclideanSpace

-- Definitions for the vectors and their properties
def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, -1)

-- The conditions from the problem
def condition1 : a = (2, 0) := rfl
def condition2 : a - b = (3, 1) := 
  by 
    simp [a, b]
    exact rfl

-- A Lean definition to express the correct conclusion
def correct_answer : Prop := 
  dot_product b (a + b) = 0

-- The goal to prove based on the conditions
theorem solution : correct_answer :=
  by 
    sorry

end solution_l354_354100


namespace Dasha_single_digit_l354_354679

open Nat

theorem Dasha_single_digit (n : ℕ) (d : ℕ → ℕ) (H : n = (d 0) + 10 * (d 1) + 100 * (d 2) + ... + 10^N * (d N)) (H1 : n ≤ (d 0) * (d 1) * (d 2) * ... * (d N)) : (n < 10) := by
  sorry

end Dasha_single_digit_l354_354679


namespace common_ratio_l354_354077

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geometric : ∀ n : ℕ, a (n+1) = q * a n) 
    (h_condition : ∀ n : ℕ, a n * a (n+4) = 9^(n+1)) : q = 3 ∨ q = -3 :=
begin
  sorry
end

end common_ratio_l354_354077


namespace minimum_AM_plus_MF_l354_354082

open Classical
open Real

noncomputable def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 8 * p.1 }
noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1 - 3)^2 + (p.2 + 1)^2 = 1 }
noncomputable def focus : (ℝ × ℝ) := (2, 0)

def AM (A M : ℝ × ℝ) : ℝ := sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2)
def MF (M : ℝ × ℝ) : ℝ := sqrt ((M.1 - focus.1)^2 + (M.2 - focus.2)^2)

theorem minimum_AM_plus_MF :
  ∃ A M : ℝ × ℝ, A ∈ circle_C ∧ M ∈ parabola ∧ (∀ A' ∈ circle_C, ∀ M' ∈ parabola, AM A' M' + MF M' ≥ 4) :=
sorry

end minimum_AM_plus_MF_l354_354082


namespace cos_is_even_and_has_zero_points_sin_is_not_even_ln_is_not_even_and_has_one_zero_point_x_squared_plus_one_is_even_and_has_no_zero_points_correct_answer_is_cos_l354_354656

theorem cos_is_even_and_has_zero_points :
  (∀ x : ℝ, cos (-x) = cos x) ∧ (∃ x : ℝ, cos x = 0) :=
begin
  sorry
end

theorem sin_is_not_even :
  ∀ x : ℝ, sin (-x) ≠ sin x :=
begin
  sorry
end

theorem ln_is_not_even_and_has_one_zero_point :
  (∀ x : ℝ, x > 0 → ln (-x) ≠ ln x) ∧ (∃ x : ℝ, x > 0 ∧ ln x = 0) :=
begin
  sorry
end

theorem x_squared_plus_one_is_even_and_has_no_zero_points :
  (∀ x : ℝ, (x^2 + 1) = ((-x)^2 + 1)) ∧ (∀ x : ℝ, (x^2 + 1) ≠ 0) :=
begin
  sorry
end

theorem correct_answer_is_cos :
  (∀ x : ℝ, cos (-x) = cos x) ∧ (∃ x : ℝ, cos x = 0) ∧
  ¬((∀ x : ℝ, sin (-x) = sin x) ∧ (∃ x : ℝ, sin x = 0)) ∧
  ¬((∀ x : ℝ, x > 0 → ln (-x) = ln x) ∧ (∃ x : ℝ, x > 0 ∧ ln x = 0)) ∧
  ¬((∀ x : ℝ, (x^2 + 1) = ((-x)^2 + 1)) ∧ (∃ x : ℝ, x^2 + 1 = 0)) :=
begin
  sorry
end

end cos_is_even_and_has_zero_points_sin_is_not_even_ln_is_not_even_and_has_one_zero_point_x_squared_plus_one_is_even_and_has_no_zero_points_correct_answer_is_cos_l354_354656


namespace f1_f2_independent_l354_354342

noncomputable def convolution_product (f g : ℕ → ℝ) : ℕ → ℝ :=
λ n, ∑ i in (finset.divisors n), f i * g (n / i)

def f_star (f : ℕ → ℝ) : ℕ → ℕ → ℝ
| f 0 := λ n, if n = 1 then 1 else 0
| f (nat.succ k) := convolution_product f (f_star f k)

def dependent (f g : ℕ → ℝ) : Prop :=
∃ (P : (ℝ → ℝ → ℝ)), ∃ (poly : ℕ × ℕ → ℝ),
  (P f g = λ n, ∑ i j, poly (i, j) * f_star f i * f_star g j n) ∧
  (P f g = 0)

def independent (f g : ℕ → ℝ) : Prop :=
¬ dependent f g

def f1 (p : ℕ) : ℕ → ℝ :=
λ n, if n = p then 1 else 0

def f2 (q : ℕ) : ℕ → ℝ :=
λ n, if n = q then 1 else 0

-- Given p and q are distinct primes
variables {p q : ℕ}
hypothesis h_p_prime : nat.prime p
hypothesis h_q_prime : nat.prime q
hypothesis h_p_q_distinct : p ≠ q

-- Prove f1 and f2 are independent
theorem f1_f2_independent : independent (f1 p) (f2 q) :=
sorry

end f1_f2_independent_l354_354342


namespace three_digit_numbers_divisible_by_5_l354_354791

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l354_354791


namespace rowers_voted_l354_354131

variable (R : ℕ)

/-- Each rower votes for exactly 4 coaches out of 50 coaches,
and each coach receives exactly 7 votes.
Prove that the number of rowers is 88. -/
theorem rowers_voted (h1 : 50 * 7 = 4 * R) : R = 88 := by 
  sorry

end rowers_voted_l354_354131


namespace min_value_expr_l354_354376

theorem min_value_expr (x : ℝ) (hx : x > 0) :
  2 * real.sqrt (real.sqrt x) + 1 / x ≥ 3 :=
by
  sorry

example : 2 * real.sqrt (real.sqrt 1) + 1 / 1 = 3 := by norm_num

end min_value_expr_l354_354376


namespace one_person_consumption_l354_354637

theorem one_person_consumption (bucket_volume : ℝ) (days_6_people : ℝ) (people_6 : ℕ) 
  (days_7_people : ℝ) (people_7 : ℕ) :
  bucket_volume = 18.9 → days_6_people = 4 → people_6 = 6 → days_7_people = 2 → people_7 = 7 →
  let daily_consumption_6 := bucket_volume / (days_6_people * people_6) in
  let daily_consumption_7 := (bucket_volume / (days_6_people * people_6)) * people_6 * days_7_people in
  let x := (bucket_volume - (daily_consumption_6 * people_6 * days_7_people)) /
            (daily_consumption_6 * days_7_people) in
  x = 6 :=
by
  intros bucket_volume_eq days_6_people_eq people_6_eq days_7_people_eq people_7_eq
  simp only at *
  sorry

end one_person_consumption_l354_354637


namespace num_of_digits_sum_l354_354109

theorem num_of_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let x := 98765
  let y := 1000 * A + 432
  let z := 10 * B + 2
  (Nat.log10 (x + y + z) + 1) = 6 := 
by
  sorry

end num_of_digits_sum_l354_354109


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354425

theorem option_A_correct (x : ℝ) (hx : x ≥ 1) : 
  let y := (x - 2) / (2 * x + 1)
  in y ∈ set.Ico (-1/3 : ℝ) (1/2 : ℝ) :=
sorry

theorem option_B_correct (f : ℝ → ℝ) (domain_of_f : set.Icc (-1 : ℝ) 1) :
  let y := λ x, f(x - 1) / real.sqrt(x - 1)
  in ∀ x, x ∈ set.Ioo (1 : ℝ) (2 : ℝ) → x - 1 ∈ domain_of_f :=
sorry

theorem option_C_correct (f : ℝ → ℝ) :
  (∀ x, f x = x^2) ∧ ∃ A : set ℝ, (A = {-2}) ∨ (A = {2}) ∨ (A = {-2, 2}) :=
sorry

theorem option_D_incorrect (f : ℝ → ℝ) (m : ℝ) (hyp1 : ∀ x, f(x + 1/x) = x^2 + 1/x^2) (hyp2 : f m = 4) :
  m ≠ real.sqrt 6 :=
sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354425


namespace Parallelogram_with_equal_diagonals_is_rectangle_l354_354284

-- Definitions for a quadrilateral, parallelogram, and rectangle
structure Quadrilateral where
  a b c d : Point
  h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d

structure Parallelogram extends Quadrilateral where
  parallel_ab_cd : Parallel (a, b) (c, d)
  parallel_ad_bc : Parallel (a, d) (b, c)

structure Rectangle extends Parallelogram where
  equal_diagonals : Diagonal (a, c) = Diagonal (b, d)

-- Declaration of our proof goal
theorem Parallelogram_with_equal_diagonals_is_rectangle (p : Parallelogram) (h : Diagonal p.a p.c = Diagonal p.b p.d) : Rectangle := sorry

end Parallelogram_with_equal_diagonals_is_rectangle_l354_354284


namespace rudolph_stop_signs_per_mile_l354_354193

theorem rudolph_stop_signs_per_mile :
  let distance := 5 + 2
  let stop_signs := 17 - 3
  (stop_signs / distance) = 2 :=
by
  let distance := 5 + 2
  let stop_signs := 17 - 3
  calc
    (stop_signs / distance) = (14 / 7) : by rw [stop_signs, distance]
                          ... = 2 : by norm_num

end rudolph_stop_signs_per_mile_l354_354193


namespace n_digit_number_divisible_by_sum_of_digits_l354_354206

theorem n_digit_number_divisible_by_sum_of_digits :
  ∀ n : ℕ, n > 0 → ∃ z : ℕ, (z.to_digits.all (λ d, d ≠ 0)) ∧ (z.to_digits.length = n) ∧ (z % (z.to_digits.sum) = 0) :=
by
  sorry

end n_digit_number_divisible_by_sum_of_digits_l354_354206


namespace problem1_problem2_l354_354869

-- Given conditions
def f (x a b : ℝ) : ℝ := (exp x - 1) / x - a * x - b

-- Problem 1.
theorem problem1 (a b : ℝ) (H : tangent_line (λ x, f x a b) 1 (1, f 1 a b) = λ x, -x/2 - 2 : ℝ → ℝ) : 
  a = 3 / 2 ∧ b = 2 := 
sorry

-- Problem 2.
theorem problem2 (a m : ℝ) (H : b = 1)
  (H1 : m < 0) (H2 : ∀ x ∈ set.Icc m 0, f x a 1 < 0) : 
  a ∈ set.Iic (exp m - 1 - m) / (m * m) :=
sorry

end problem1_problem2_l354_354869


namespace cos_is_correct_answer_l354_354658

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def has_zero_point (f : ℝ → ℝ) := ∃ x : ℝ, f x = 0

def cos_has_even_and_zero_point : 
  is_even (λ x : ℝ, Real.cos x) ∧ has_zero_point (λ x : ℝ, Real.cos x) :=
by sorry

def sin_is_not_even : 
  ¬ is_even (λ x : ℝ, Real.sin x) :=
by sorry

def ln_is_not_even : 
  ¬ is_even (λ x : ℝ, Real.log x) :=
by sorry

def ln_has_a_zero_point : 
  has_zero_point (λ x : ℝ, Real.log x) :=
by sorry

def x2plus1_has_no_zero_point : 
  ¬ has_zero_point (λ x : ℝ, x^2 + 1) :=
by sorry

def x2plus1_is_even : 
  is_even (λ x : ℝ, x^2 + 1) :=
by sorry

theorem cos_is_correct_answer : 
  cos_has_even_and_zero_point ∧ sin_is_not_even ∧ ln_is_not_even ∧ ln_has_a_zero_point ∧ x2plus1_is_even ∧ x2plus1_has_no_zero_point :=
by sorry

end cos_is_correct_answer_l354_354658


namespace smallest_positive_period_maximum_value_and_set_l354_354426

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * Real.sin x ^ 2

theorem smallest_positive_period : ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
begin
  use π,
  split,
  { apply Real.pi_pos, },
  { split,
    { intro x,
      sorry },
    { intros T' T'_pos h_T',
      sorry } }
end

theorem maximum_value_and_set (k : ℤ) : 
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x, f x = M ↔ ∃ k : ℤ, x = π / 8 + k * π :=
begin
  use sqrt 2 - 1,
  split,
  { refl },
  { intro x,
    split,
    { intro h,
      sorry },
    { rintro ⟨k, rfl⟩,
      sorry } }
end

end smallest_positive_period_maximum_value_and_set_l354_354426


namespace all_elements_same_color_l354_354181

-- Define conditions
variables {n : ℕ} {k : ℤ}
hypothesis (hn_pos : 0 < n)
hypothesis (hk_bounds : 0 < k ∧ k < n)
hypothesis (h_gcd : Int.gcd k n = 1)

noncomputable def M : Set ℤ := {i | 1 ≤ i ∧ i < n}

-- Color function to represent the color of each element
def color : ℤ → Bool := sorry

-- Define the conditions on colors
hypothesis (color_symmetry : ∀ i ∈ M, color i = color (n - i))
hypothesis (color_cond : ∀ i ∈ M, i ≠ k → color i = color (abs (k - i)))

-- Statement to be proved
theorem all_elements_same_color :
  ∀ i j ∈ M, color i = color j := sorry

end all_elements_same_color_l354_354181


namespace count_3_digit_numbers_divisible_by_5_l354_354789

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354789


namespace max_area_triangle_after_t_seconds_l354_354195

-- Define the problem conditions and question
def second_hand_rotation_rate : ℝ := 6 -- degrees per second
def minute_hand_rotation_rate : ℝ := 0.1 -- degrees per second
def perpendicular_angle : ℝ := 90 -- degrees

theorem max_area_triangle_after_t_seconds : 
  ∃ (t : ℝ), (second_hand_rotation_rate - minute_hand_rotation_rate) * t = perpendicular_angle ∧ t = 15 + 15 / 59 :=
by
  -- This is a statement of the proof problem; the proof itself is omitted.
  sorry

end max_area_triangle_after_t_seconds_l354_354195


namespace high_fever_temperature_l354_354572

theorem high_fever_temperature (T t : ℝ) (h1 : T = 36) (h2 : t > 13 / 12 * T) : t > 39 :=
by
  sorry

end high_fever_temperature_l354_354572


namespace find_hyperbola_eq_and_sum_of_slopes_l354_354432

noncomputable def hyperbola_equation (x y : ℝ) (b : ℝ) : Prop :=
  x^2 - y^2 / b^2 = 1

def point_on_hyperbola (P : ℝ × ℝ) (b : ℝ) : Prop :=
  hyperbola_equation P.1 P.2 b

def line_through (Q : ℝ × ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x, k * x - 1

def intersect_hyperbola_line (k : ℝ) : Prop :=
  let l := line_through (0, -1) k in
  ∃ x1 x2 : ℝ, x1 ≠ -2 ∧ x2 ≠ -2 ∧ hyperbola_equation x1 (l x1) 3 ∧ hyperbola_equation x2 (l x2) 3

theorem find_hyperbola_eq_and_sum_of_slopes :
  ∃ b > 0, point_on_hyperbola (-2, -3) b ∧ 
  (∀ (k : ℝ), intersect_hyperbola_line k → 
     let PA_slope := (λ x y, (y + 3) / (x + 2)) in
     ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
     hyperbola_equation x1 y1 3 ∧ hyperbola_equation x2 y2 3 ∧ 
     PA_slope x1 y1 + PA_slope x2 y2 = 3) :=
begin
  sorry
end

end find_hyperbola_eq_and_sum_of_slopes_l354_354432


namespace socks_tshirt_probability_l354_354237

theorem socks_tshirt_probability :
  let sock_colors := {red, green, blue},
      tshirt_colors := {red, yellow, green, blue, white},
      total_combinations := 3 * 5,
      favorable_combinations := 4 + 4 + 4 + 1 in
  ∃ probability : ℚ,
  probability = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧
  probability = 13 / 15 :=
by
  sorry

end socks_tshirt_probability_l354_354237


namespace evaluate_expression_l354_354708

variable (x y : ℝ)

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x * y) :
  (1 / x^2) - (1 / y^2) = - (1 / (x * y)) :=
sorry

end evaluate_expression_l354_354708


namespace player_A_has_winning_strategy_l354_354394

theorem player_A_has_winning_strategy (n : ℕ) (h_ge_5 : n ≥ 5) :
  ∃ (strategy : strategy_for_A), winning_strategy strategy :=
sorry

end player_A_has_winning_strategy_l354_354394


namespace maximum_infernal_rooks_l354_354605

-- Define the problem conditions
def chessboard : Type := fin 8 × fin 8
def color (pos : chessboard) : bool := (pos.1.1 + pos.2.1) % 2 = 0

-- Define the infernal rook attack relationships
def row_attack (pos1 pos2 : chessboard) : Prop := 
  pos1.1 = pos2.1 ∧ color pos1 = color pos2

def column_attack (pos1 pos2 : chessboard) : Prop := 
  pos1.2 = pos2.2 ∧ color pos1 ≠ color pos2

def attacks (pos1 pos2 : chessboard) : Prop :=
  row_attack pos1 pos2 ∨ column_attack pos1 pos2

-- Define "safe placement" as no two infernal rooks attack each other
def safe_placement (positions : finset chessboard) : Prop :=
  ∀ p1 p2 ∈ positions, p1 ≠ p2 → ¬ attacks p1 p2

-- The maximum number of infernal rooks that can be placed on an
-- 8×8 chessboard without them attacking each other
theorem maximum_infernal_rooks :
  ∃ (positions : finset chessboard), safe_placement positions ∧ positions.card = 16 :=
sorry

end maximum_infernal_rooks_l354_354605


namespace total_profit_l354_354332

noncomputable def radio := (cp : ℝ, sp : ℝ)
noncomputable def tv := (cp : ℝ, sp : ℝ, tax_rate : ℝ)
noncomputable def speaker := (cp : ℝ, sp : ℝ, discount_rate : ℝ)
noncomputable def second_radio := (cp : ℝ, sp : ℝ, shipping_cost : ℝ)

noncomputable def total_cp (r : radio) (t : tv) (s : speaker) (sr : second_radio) : ℝ :=
  r.1 + t.1 + s.1 + sr.1

noncomputable def total_sp (r : radio) (t : tv) (s : speaker) (sr : second_radio) : ℝ :=
  r.2 + (t.2 + t.2 * t.3) + (s.2 - s.2 * s.3) + (sr.2 + sr.3)

theorem total_profit :
  let r := radio (490 : ℝ) (465.50 : ℝ),
      t := tv (12000 : ℝ) (11400 : ℝ) (0.10 : ℝ),
      s := speaker (1200 : ℝ) (1150 : ℝ) (0.05 : ℝ),
      sr := second_radio (600 : ℝ) (550 : ℝ) (50 : ℝ) in
  total_cp r t s sr - total_sp r t s sr = -408 :=
by
  sorry

end total_profit_l354_354332


namespace find_b_in_triangle_l354_354498

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a = 3)
variable (h2 : c = 2 * Real.sqrt 3)
variable (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6))

-- The proof goal
theorem find_b_in_triangle (h1 : a = 3) (h2 : c = 2 * Real.sqrt 3) (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6)) : b = Real.sqrt 3 :=
sorry

end find_b_in_triangle_l354_354498


namespace part1_part2_l354_354434

noncomputable def P (n : ℕ) : ℝ × ℝ
| 0       := (1, -1)
| (n + 1) := let (a_n, b_n) := P n in (a_n * b(n + 1), b(n + 1))
  where b (n : ℕ) : ℝ :=
         match P n with
         | (_, b_n) := b_n / (1 - 4 * (P n).fst^2)

def lies_on_line (P1 P2 Pn : ℝ × ℝ) : Prop :=
  let m := (P2.snd - P1.snd) / (P2.fst - P1.fst) in
  Pn.snd = m * (Pn.fst - P1.fst) + P1.snd

theorem part1 (n : ℕ) : lies_on_line (P 1) (P 2) (P n) :=
sorry

theorem part2 (k : ℝ) : 
  (∀ n ∈ ℕ+, (∏ i in finset.range n, (1 + (P i).fst)) ≥ k / (real.sqrt (∏ i in finset.range n, (P (i + 1)).snd))) →
  k ≤ (2 * real.sqrt 3) / 3 :=
sorry

end part1_part2_l354_354434


namespace rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l354_354830

theorem rhombus_diagonal_BD_equation (A C : ℝ × ℝ) (AB_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 1 ∧ b = 6 ∧ ∀ x y : ℝ, x + y - 6 = 0 := by
  sorry

theorem rhombus_diagonal_AD_equation (A C : ℝ × ℝ) (AB_eq BD_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0 ∧ x + y - 6 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 3 ∧ b = 14 ∧ ∀ x y : ℝ, x - 3 * y + 14 = 0 := by
  sorry

end rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l354_354830


namespace find_root_interval_l354_354948

noncomputable def f : ℝ → ℝ := sorry

theorem find_root_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 < 0 ∧ f 2.75 > 0 ∧ f 2.625 > 0 ∧ f 2.5625 > 0 →
  ∃ x, 2.5 < x ∧ x < 2.5625 ∧ f x = 0 := sorry

end find_root_interval_l354_354948


namespace poly_value_at_two_l354_354946

def f (x : ℝ) : ℝ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem poly_value_at_two : f 2 = 216 :=
by
  unfold f
  norm_num
  sorry

end poly_value_at_two_l354_354946


namespace choose_n_subsets_bounded_union_l354_354931

theorem choose_n_subsets_bounded_union (n : ℕ) (h : n ≥ 1) :
  ∃ (S : Finset (Finset (Fin n))), 
    S.card = n ∧ 
    (∀ ss ∈ S, ss.card = 2) ∧ 
    (S.bUnion id).card ≤ (2 * n / 3) + 1 :=
by
  sorry

end choose_n_subsets_bounded_union_l354_354931


namespace school_tour_teachers_students_school_tour_rental_options_l354_354634

/-- A certain school plans to organize a collective study tour. There are 14 teachers and 176 students. 
    Prove the following conditions:

    1: The number of teachers and students
    2: Number of rental options with minimum rental cost -/

/-- Proof 1: Given the conditions, prove the number of teachers and students -/
theorem school_tour_teachers_students 
  (x y : ℕ)
  (h1 : y = 12 * x + 8)
  (h2 : y = 13 * x - 6)
  : x = 14 ∧ y = 176 := sorry

/-- Proof 2: Given 14 teachers and 176 students, prove the number of rental options 
    and minimum rental cost under capacity and cost constraints -/
theorem school_tour_rental_options 
  (total_teachers total_students : ℕ) (total_buses num_typeA_buses num_typeB_buses : ℕ)
  (cost_typeA cost_typeB budget total_cost : ℕ)
  (capacity_typeA capacity_typeB : ℕ)
  (h1 : total_teachers = 14)
  (h2 : total_students = 176)
  (h3 : total_buses = 6)
  (h4 : cost_typeA = 400)
  (h5 : cost_typeB = 320)
  (h6 : budget = 2300)
  (h7 : capacity_typeA = 40)
  (h8 : capacity_typeB = 30)
  (h9 : total_cost = cost_typeA * num_typeA_buses + cost_typeB * num_typeB_buses)
  (h10 : total_teachers + total_students ≤ capacity_typeA * num_typeA_buses + capacity_typeB * num_typeB_buses)
  (h11 : total_cost ≤ budget)
  : ∃ num_typeA_buses num_typeB_buses, 
      1 ≤ num_typeA_buses ∧ num_typeA_buses ≤ 4 ∧
      2000 = cost_typeA * 1 + cost_typeB * (6 - 1) ∧
      40 * num_typeA_buses + 30 * (6 - num_typeA_buses) ≥ 190 := sorry

end school_tour_teachers_students_school_tour_rental_options_l354_354634


namespace liam_comic_books_l354_354525

theorem liam_comic_books (cost_per_book : ℚ) (total_money : ℚ) (n : ℚ) : cost_per_book = 1.25 ∧ total_money = 10 → n = 8 :=
by
  intros h
  cases h
  have h1 : 1.25 * n ≤ 10 := by sorry
  have h2 : n ≤ 10 / 1.25 := by sorry
  have h3 : n ≤ 8 := by sorry
  have h4 : n = 8 := by sorry
  exact h4

end liam_comic_books_l354_354525


namespace find_a_correct_l354_354381

noncomputable def find_a : ℕ :=
  let expr := (2 + ∑ i in range (5 * n), 2 ^ (i + 1))
  ∀ n : ℕ, expr + a ≡ 3 [MOD 31] → a := 4

theorem find_a_correct (n : ℕ) : 
  ∃ a : ℕ, ∑ i in range (5 * n), 2^(i + 1) + a ≡ 3 [MOD 31] :=
by
  sorry

end find_a_correct_l354_354381


namespace cost_price_of_book_l354_354670

theorem cost_price_of_book (Selling_Price : ℝ) (Profit_Percentage : ℝ) (h1 : Selling_Price = 270) (h2 : Profit_Percentage = 0.2) : 
  (C : ℝ) (hC : C = Selling_Price / (1 + Profit_Percentage)) : C = 225 :=
by
  have h : C = 270 / (1 + 0.2) := by
    rw [h1, h2]
  rw [h] at hC
  norm_num at hC
  exact hC

end cost_price_of_book_l354_354670


namespace brendan_weekly_taxes_correct_l354_354017

def brendan_waiter_hourly_wage : ℝ := 6
def brendan_barista_hourly_wage : ℝ := 8
def waiter_shifts_hours : ℝ := 2 * 8 + 1 * 12
def barista_shift_hours : ℝ := 1 * 6
def waiter_tips_per_hour : ℝ := 12
def barista_tips_per_hour : ℝ := 5
def waiter_tax_rate : ℝ := 0.20
def barista_tax_rate : ℝ := 0.25
def waiter_reported_tip_fraction : ℝ := 1 / 3
def barista_reported_tip_fraction : ℝ := 1 / 2

def waiter_total_wages : ℝ := waiter_shifts_hours * brendan_waiter_hourly_wage
def waiter_total_tips : ℝ := waiter_shifts_hours * waiter_tips_per_hour
def waiter_reported_tips : ℝ := waiter_total_tips * waiter_reported_tip_fraction
def waiter_reported_income : ℝ := waiter_total_wages + waiter_reported_tips

def barista_total_wages : ℝ := barista_shift_hours * brendan_barista_hourly_wage
def barista_total_tips : ℝ := barista_shift_hours * barista_tips_per_hour
def barista_reported_tips : ℝ := barista_total_tips * barista_reported_tip_fraction
def barista_reported_income : ℝ := barista_total_wages + barista_reported_tips

def total_reported_income : ℝ := waiter_reported_income + barista_reported_income

def waiter_taxes : ℝ := waiter_reported_income * waiter_tax_rate
def barista_taxes : ℝ := barista_reported_income * barista_tax_rate
def total_taxes : ℝ := waiter_taxes + barista_taxes

theorem brendan_weekly_taxes_correct : total_taxes = 71.75 := by
  sorry

end brendan_weekly_taxes_correct_l354_354017


namespace problem1_intersection_problem1_union_complements_problem2_evaluation_l354_354303

universe u

-- Problem 1
noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < -4 ∨ x > 1 }
noncomputable def B : Set ℝ := { x : ℝ | -3 ≤ x-1 ∧ x-1 ≤ 2 }

theorem problem1_intersection :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 3 } :=
sorry

theorem problem1_union_complements :
  (U \ A) ∪ (U \ B) = { x : ℝ | x ≤ 1 ∨ x > 3 } :=
sorry

-- Problem 2
noncomputable def expr (x : ℝ) :=
  (2 * x ^ (1 / 4) + 3 ^ (3 / 2)) * (2 * x ^ (1 / 4) - 3 ^ (3 / 2)) - 4 * x ^ (-1 / 2) * (x - x ^ (1 / 2))

theorem problem2_evaluation (x : ℝ) (hx : x > 0) :
  expr x = -23 :=
sorry

end problem1_intersection_problem1_union_complements_problem2_evaluation_l354_354303


namespace normal_expectation_variance_l354_354375

theorem normal_expectation_variance (X : ℝ → ℝ) (a σ : ℝ) (hX : IsNormalDistribution X a σ²) :
  (expectation X = a) ∧ (variance X = σ²) :=
sorry

end normal_expectation_variance_l354_354375


namespace segment_PQ_length_l354_354027

-- Define the circles and their properties
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the points A, B, P, Q and their relevant properties
variables (A B P Q : ℝ × ℝ)
variables (C1 C2 : Circle)

-- Given conditions
variable h1 : C1.radius = 3
variable h2 : C2.radius = 7
variable h3 : P ≠ C2.center
variable h4 : dist P (C1.center) = 5
variable h5 : dist P (C2.center) > C2.radius
variable h6 : tangent P Q C2

-- Prove that length of segment PQ is 4
theorem segment_PQ_length : dist P Q = 4 :=
sorry

end segment_PQ_length_l354_354027


namespace cost_equal_at_x_20_l354_354938

-- Step a): Define the costs and conditions
def cost_store_A (x : ℕ) := 30 * 5 + 5 * (x - 5)
def cost_store_B (x : ℕ) := (30 * 5 + 5 * x) * 0.9

-- Step c): State the final proof problem
theorem cost_equal_at_x_20 (x : ℕ) (hx : x ≥ 5) :
  cost_store_A x = cost_store_B x ↔ x = 20 :=
by
  sorry

end cost_equal_at_x_20_l354_354938


namespace ten_adult_worms_from_one_in_one_hour_l354_354234

-- Definitions for the problem conditions
def adult_worm_length : ℝ := 1

def cut_worm (length : ℝ) : ℝ × ℝ := (length / 2, length / 2)

def grow_worm (length : ℝ) (time : ℝ) : ℝ := length + time

def is_adult (length : ℝ) : Prop := length = adult_worm_length

-- Main theorem statement
theorem ten_adult_worms_from_one_in_one_hour :
  ∃ t : ℝ, t < 1 ∧ ∀ n : ℕ, n < 10 → ∃ length : ℝ, is_adult (grow_worm length t) :=
sorry

end ten_adult_worms_from_one_in_one_hour_l354_354234


namespace oblique_asymptote_l354_354366

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x - 2)

theorem oblique_asymptote : ∃ m b : ℝ, (∀ (x : ℝ), f x = m * x + b + o(1)) ∧ m = 3 ∧ b = -2 :=
by
  use 3, -2
  split
  sorry
  split
  refl
  refl

end oblique_asymptote_l354_354366


namespace bus_speed_is_correct_l354_354713

noncomputable def Speed_of_Bus 
  (d_AB : ℝ) (t_meet_C : ℝ) (d_CD : ℝ) (v1 v2 : ℝ) (h₁ : d_AB = 4) 
  (h₂ : t_meet_C = 1/6) (h₃ : d_CD = 2/3) 
  (h₄ : v1 + v2 = 48)
  (h₅ : v2 * 1/6 = v1 * 1/6 + d_AB) : ℝ := 40

theorem bus_speed_is_correct :
  ∀ (d_AB t_meet_C d_CD v1 v2 : ℝ),
    d_AB = 4 →
    t_meet_C = 1/6 →
    d_CD = 2/3 →
    v1 + v2 = 48 →
    v2 * 1/6 = v1 * 1/6 + d_AB →
    Speed_of_Bus d_AB t_meet_C d_CD v1 v2 d_AB t_meet_C d_CD v1 v2 = 40 :=
by
  intros
  rw [Speed_of_Bus, h₁, h₂, h₃, h₄, h₅]
  sorry

end bus_speed_is_correct_l354_354713


namespace evaluate_floor_ceil_l354_354371

def floor (x : ℝ) : ℤ := Int.floor x
def ceil (x : ℝ) : ℤ := Int.ceil x

theorem evaluate_floor_ceil :
  floor (-0.123) + ceil (4.567) = 4 := 
by
  sorry

end evaluate_floor_ceil_l354_354371


namespace range_of_function_l354_354955

theorem range_of_function : ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2)/(x^2 + x + 1) :=
by
  sorry

end range_of_function_l354_354955


namespace AB_twelve_l354_354829

noncomputable def AB_of_rectangle_tangent (ABCD : Type) [rectangle ABCD]
  (P : ABCD → ABCD) (B C : ABCD)  (BP CP : ℝ) 
  (tan_angle_APD : ℝ) : ℝ :=
if h1 : BP = 12 ∧ CP = 6 ∧ tan_angle_APD = 2 then 12 else 0

theorem AB_twelve (ABCD : Type) [rectangle ABCD]
  (P B C : ABCD) :
  ∀ (BP CP : ℝ) (tan_angle_APD : ℝ),
  BP = 12 ∧ CP = 6 ∧ tan_angle_APD = 2 → AB_of_rectangle_tangent ABCD P B C BP CP tan_angle_APD = 12 :=
by
  intros _ _ _ h_cond
  simp [AB_of_rectangle_tangent]
  rw if_pos h_cond
  rfl

end AB_twelve_l354_354829


namespace fraction_equality_l354_354714

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 :=
by
  sorry

end fraction_equality_l354_354714


namespace find_a_plus_b_l354_354584

variable {R : Type} [LinearOrderedField R]

def quadratic_inequality_solution_set (a x b : R) : Prop :=
  ∀ x, (1 < x ∧ x < 2) ↔ (ax^2 + x + b > 0)

theorem find_a_plus_b (a b : R) 
  (ineq_sol_set : quadratic_inequality_solution_set a x b) : 
  a + b = -1 :=
sorry

end find_a_plus_b_l354_354584


namespace binom_sum_mod_9_l354_354041

theorem binom_sum_mod_9 (S : ℕ) (hS : S = ∑ k in finset.range 28, if k = 0 then 0 else nat.choose 27 k) : S % 9 = 7 :=
by sorry

end binom_sum_mod_9_l354_354041


namespace dean_taller_than_ron_l354_354337

theorem dean_taller_than_ron (d h r : ℕ) (h1 : d = 15 * h) (h2 : r = 13) (h3 : d = 255) : h - r = 4 := 
by 
  sorry

end dean_taller_than_ron_l354_354337


namespace triangle_ABC_is_right_triangle_l354_354838

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (h1 : cos A = √3 / 2) (h2 : tan B = √3) :
  A + B + C = π → C = π / 2 :=
sorry

end triangle_ABC_is_right_triangle_l354_354838


namespace elberta_money_l354_354769

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) 
  (h1 : granny_smith = 120) 
  (h2 : anjou = granny_smith / 4) 
  (h3 : elberta = anjou + 5) : 
  elberta = 35 :=
by {
  sorry
}

end elberta_money_l354_354769


namespace find_angle_CBO_l354_354341

def triangle : Type := {A B C O : Type} 

def angle_eq (α β : ℝ) : Prop := α = β

def triangle_angles (A B C O : triangle) : Prop :=
  ∃ (BAO CAO CBO ABO ACO BCO : ℝ),
    angle_eq BAO CAO ∧ -- condition 1
    angle_eq CBO ABO ∧ -- condition 2
    angle_eq ACO BCO ∧ -- condition 3
    angle_eq (BAO + CAO + (180 - (BAO + CAO))) 180 -- sum of angles in triangle

theorem find_angle_CBO (A B C O : triangle) (AO_C : ℝ) (BAO CAO CBO ABO ACO BCO : ℝ) :
  triangle_angles A B C O ∧ AO_C = 110 →
  CBO = 20 :=
  by sorry

end find_angle_CBO_l354_354341


namespace geometric_body_is_cylinder_l354_354970

def top_view_is_circle : Prop := sorry

def is_prism_or_cylinder : Prop := sorry

theorem geometric_body_is_cylinder 
  (h1 : top_view_is_circle) 
  (h2 : is_prism_or_cylinder) 
  : Cylinder := 
sorry

end geometric_body_is_cylinder_l354_354970


namespace min_value_sub_abs_l354_354723

theorem min_value_sub_abs 
  (x y : ℝ) 
  (h1 : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) : 
    ∃ t : ℝ, t = x - |y| ∧ t = sqrt 3 :=
by
  sorry

end min_value_sub_abs_l354_354723


namespace log_inequality_l354_354178

theorem log_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
    (Real.log (c ^ 2) / Real.log (a + b) + Real.log (a ^ 2) / Real.log (b + c) + Real.log (b ^ 2) / Real.log (c + a)) ≥ 3 :=
sorry

end log_inequality_l354_354178


namespace a2_value_l354_354740

def z : ℂ := (1 / 2 : ℂ) + (sqrt 3 / 2 : ℂ) * complex.I
def polynomial := ∑ i in (finset.range 5), (λ j, j • x^(4-j)) z

theorem a2_value :
  ( (x - z)^4 ).coeff 2 = (-3 : ℂ) + 3 * (sqrt 3 : ℂ) * complex.I :=
  sorry

end a2_value_l354_354740


namespace fourth_quadrant_range_l354_354066

theorem fourth_quadrant_range (m : ℝ) (z : ℂ) (h : z = (m + 3) + (m - 1) * (complex.I)) :
  (m + 3 > 0) ∧ (m - 1 < 0) ↔ -3 < m ∧ m < 1 :=
by
  sorry

end fourth_quadrant_range_l354_354066


namespace count_3_digit_numbers_divisible_by_5_l354_354780

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354780


namespace find_pairs_satisfying_system_l354_354051

theorem find_pairs_satisfying_system :
  let φ := (x y : ℝ) → 6 * (1 - x)^2 = 1 / y ∧ 6 * (1 - y)^2 = 1 / x
  (∃ x y : ℝ, φ x y ∧ ((x = 3 / 2 ∧ y = 2 / 3) ∨ (x = 2 / 3 ∧ y = 3 / 2) ∨ 
                        (x = 1 / 3 * (2 + (2:ℝ).cbrt + 1 / (2:ℝ).cbrt) ∧ y = 1 / 3 * (2 + (2:ℝ).cbrt + 1 / (2:ℝ).cbrt)))) :=
  sorry

end find_pairs_satisfying_system_l354_354051


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354423

theorem option_A_correct (x : ℝ) (hx : x ≥ 1) : 
  let y := (x - 2) / (2 * x + 1)
  in y ∈ set.Ico (-1/3 : ℝ) (1/2 : ℝ) :=
sorry

theorem option_B_correct (f : ℝ → ℝ) (domain_of_f : set.Icc (-1 : ℝ) 1) :
  let y := λ x, f(x - 1) / real.sqrt(x - 1)
  in ∀ x, x ∈ set.Ioo (1 : ℝ) (2 : ℝ) → x - 1 ∈ domain_of_f :=
sorry

theorem option_C_correct (f : ℝ → ℝ) :
  (∀ x, f x = x^2) ∧ ∃ A : set ℝ, (A = {-2}) ∨ (A = {2}) ∨ (A = {-2, 2}) :=
sorry

theorem option_D_incorrect (f : ℝ → ℝ) (m : ℝ) (hyp1 : ∀ x, f(x + 1/x) = x^2 + 1/x^2) (hyp2 : f m = 4) :
  m ≠ real.sqrt 6 :=
sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l354_354423


namespace count_3_digit_numbers_divisible_by_5_l354_354787

theorem count_3_digit_numbers_divisible_by_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_5 := {n : ℕ | n % 5 = 0}
  let count := {n : ℕ | n ∈ three_digit_numbers ∧ n ∈ divisible_by_5}.card
  count = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l354_354787


namespace twenty_fifth_occurrence_of_digit3_l354_354186

theorem twenty_fifth_occurrence_of_digit3
  (seq : ℕ → ℕ)
  (h_seq : ∀ n, seq n = n + 1) :
  ∃ n, digit_occurrence seq 3 n = 25 ∧ seq n = 134 := 
sorry

-- Defining a function to count occurrences of a given digit up to a certain number in a sequence
def digit_occurrence (seq : ℕ → ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  let occurrences := λ (x : ℕ), seq x in
  -- Count occurrences of digit d in all numbers up to seq(n).
  occurrences.count (λ x, (seq x).to_digits.contains d)
  
-- A helper function to convert a number to its digits (from Decimal representation)
def to_digits (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else let rec := n.digits 10 in rec.reverse

#eval to_digits 134  -- To verify our digit extraction logic

end twenty_fifth_occurrence_of_digit3_l354_354186


namespace coprime_gcd_interval_l354_354724

theorem coprime_gcd_interval (n : ℕ) (h : n > 6) :
  let s := { k | n * (n - 1) < k ∧ k < n^2 ∧ Nat.gcd k (n * (n - 1)) = 1 }
  in s.nonempty → ∃ k ∈ s, Nat.gcd k (n * (n - 1)) = 1
 := sorry

end coprime_gcd_interval_l354_354724


namespace shade_entire_square_with_three_folds_l354_354133

noncomputable def fold_and_shade (grid : List (List Bool)) (folds : List (String × Nat)) : List (List Bool) :=
  sorry -- Assume this function models the folding and shading process.

def is_fully_shaded (grid : List (List Bool)) : Prop :=
  ∀ row, ∀ cell, row.get cell = some true

theorem shade_entire_square_with_three_folds (initial_grid : List (List Bool)) : 
  ∃ folds : List (String × Nat), folds.length ≤ 3 ∧ is_fully_shaded (fold_and_shade initial_grid folds) :=
sorry

end shade_entire_square_with_three_folds_l354_354133


namespace midpoint_locus_l354_354650

theorem midpoint_locus (c : ℝ) (H : 0 < c ∧ c ≤ Real.sqrt 2) :
  ∃ L, L = "curvilinear quadrilateral with arcs forming transitions" :=
sorry

end midpoint_locus_l354_354650


namespace correct_propositions_l354_354746

variable {f : ℝ → ℝ}

-- Condition: Domain of f is ℝ and f is not constant
axiom f_domain : ∀ x, f x ∈ ℝ
axiom f_not_constant : ∃ x y, x ≠ y ∧ f x ≠ f y

-- Proposition 1: g(x) = f(x) + f(-x) is always even
def g : ℝ → ℝ := λ x, f x + f (-x)
axiom prop1 : ∀ x, g(-x) = g(x)

-- Proposition 2: If f(x) + f(2 - x) = 0 for any x ∈ ℝ, then f(x) is periodic with period 2
axiom f_periodic2 : ∀ x, f x + f (2 - x) = 0 → ¬∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Proposition 3: If f is odd and f(x) + f(2 + x) = 0 for any x ∈ ℝ, then the axis of symmetry of f is x = 2n + 1 (n ∈ ℝ)
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_symmetry : ∀ x, f(x) + f(2 + x) = 0 → ∃ n : ℤ, ∀ x, f (x + (2 * n + 1)) = f (2 * n + 1 - x)

-- Proposition 4: For any x₁, x₂ ∈ ℝ, x₁ ≠ x₂, if (f(x₁) - f(x₂)) / (x₁ - x₂) > 0, then f is increasing on ℝ
axiom f_increasing : ∀ x₁ x₂, x₁ ≠ x₂ ∧ (f(x₁) - f(x₂)) / (x₁ - x₂) > 0 → ∀ x y, x < y → f x < f y

-- The theorem that we ultimately need to prove
theorem correct_propositions
  (h1 : ∀ x, g (-x) = g x)
  (h3 : ∀ x, (f x + f (2 + x) = 0 → ∃ n : ℤ, ∀ x, f (x + (2 * n + 1)) = f (2 * n + 1 - x)))
  (h4 : ∀ x₁ x₂, x₁ ≠ x₂ ∧ (f(x₁) - f(x₂)) / (x₁ - x₂) > 0 → ∀ x y, x < y → f x < f y) :
  true := 
sorry

end correct_propositions_l354_354746


namespace mary_total_time_spent_l354_354187

theorem mary_total_time_spent :
  let mac_download := 10
  let windows_download := 3 * mac_download
  let ny_audio_glitches := 6 * 2
  let ny_video_glitch := 8
  let ny_glitch_time := ny_audio_glitches + ny_video_glitch
  let ny_without_glitches := 3 * ny_glitch_time
  let ny_total := ny_glitch_time + ny_without_glitches
  let berlin_audio_glitches := 4 * 3
  let berlin_video_glitches := 5 * 2
  let berlin_glitch_time := berlin_audio_glitches + berlin_video_glitches
  let berlin_without_glitches := 2 * berlin_glitch_time
  let berlin_total := berlin_glitch_time + berlin_without_glitches
  let tokyo_audio_glitch := 7
  let tokyo_video_glitches := 9 * 2
  let tokyo_glitch_time := tokyo_audio_glitch + tokyo_video_glitches
  let tokyo_without_glitches := 4 * tokyo_glitch_time
  let tokyo_total := tokyo_glitch_time + tokyo_without_glitches
  let sydney_audio_glitches := 6 * 2
  let sydney_video_glitches := 0
  let sydney_glitch_time := sydney_audio_glitches + sydney_video_glitches
  let sydney_without_glitches := 5 * sydney_glitch_time
  let sydney_total := sydney_glitch_time + sydney_without_glitches
  let total := mac_download + windows_download +
               ny_total + berlin_total + tokyo_total + sydney_total
  in total = 383 := sorry

end mary_total_time_spent_l354_354187


namespace max_binomial_coefficient_term_l354_354750

theorem max_binomial_coefficient_term :
    ∃ T : ℕ → Type*, ∀ (n : ℕ) (x : ℝ), 
    n = 7 → 
    ((C(n, 4) * (2 : ℝ)^4 * (x : ℝ)^(4 / 2) = 560 * (x : ℝ) ^ 2) ∧ 
    (C(n, 3) * (2 : ℝ)^3 * (x : ℝ)^(3 / 2) = 280 * (x : ℝ)^(3 / 2))) :=
by
  sorry

-- Auxiliary function to compute binomial coefficient
def C (n k : ℕ) : ℕ :=
  nat.choose n k

end max_binomial_coefficient_term_l354_354750


namespace area_of_feasible_region_l354_354219

theorem area_of_feasible_region : 
  let S := {p : ℝ × ℝ | p.2 ≤ -p.1 + 2 ∧ p.2 ≤ p.1 - 1 ∧ p.2 ≥ 0} in
  let area_triangle (P Q R : ℝ × ℝ) := (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2)) in
  S = {(x, 0) | (x = 1) ∨ (x = 2)} ∪ {(3 / 2, 1 / 2)} → 
  area_triangle (1, 0) (2, 0) (3 / 2, 1 / 2) = 1 / 4 :=
by sorry

end area_of_feasible_region_l354_354219


namespace tom_books_total_l354_354940

theorem tom_books_total :
  (2 + 6 + 10 + 14 + 18) = 50 :=
by {
  -- Proof steps would go here.
  sorry
}

end tom_books_total_l354_354940


namespace probability_car_X_wins_l354_354128

/-- 
In a race where 16 cars are running, the chance that car X will win is some probability, 
that Y will win is 1/12, and that Z will win is 1/7. Assuming that a dead heat is impossible, 
the chance that one of them will win is 0.47619047619047616. What is the probability that car X will win?
--/
theorem probability_car_X_wins :
  let P_Y := (1 : ℚ) / 12,
      P_Z := (1 : ℚ) / 7,
      P_combined := 0.47619047619047616 in
  P_combined = P_X + P_Y + P_Z → P_Y = 1 / 12 → P_Z = 1 / 7 → P_X = 1 / 4 :=
begin
  sorry, -- proof goes here
end

end probability_car_X_wins_l354_354128


namespace vector_equivalence_l354_354110

variables {Point Vector : Type} [AddCommGroup Vector] [Module ℝ Vector]

-- Let O be the center of the parallelogram ABCD
variables {O A B C D : Point}
-- Define e1 and e2 as basis vectors
variables (e1 e2 : Vector)

-- Define the conditions of the problem
axiom AB_eq_4e1 : A - B = 4 • e1
axiom BC_eq_6e2 : B - C = 6 • e2
axiom O_center_parallelogram : O = (A + B + C + D) / 4

-- Define vector BO
def BO : Vector := O - B

-- The proof problem
theorem vector_equivalence :
  3 • e2 - 2 • e1 = BO :=
sorry

end vector_equivalence_l354_354110


namespace magnitude_of_vector_l354_354441

def a : ℝ × ℝ := (-1, 2)

def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vector (k : ℝ)
    (b : ℝ × ℝ := (2, k))
    (h_parallel : is_parallel a b) :
  magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 4 * real.sqrt 5 :=
sorry

end magnitude_of_vector_l354_354441


namespace sum_possible_values_of_y_l354_354361

theorem sum_possible_values_of_y :
  let y_values := [17, 53 / 13] in
  ∃ y : ℝ, y ∈ y_values ∧
  (∑ i in y_values, i) = 17 + 53 / 13 :=
by
  sorry

end sum_possible_values_of_y_l354_354361


namespace sum_of_squares_of_consecutive_integers_l354_354580

theorem sum_of_squares_of_consecutive_integers (a : ℕ) (h : (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2))) :
  (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2 = 86 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l354_354580


namespace exists_continuous_function_l354_354853

noncomputable def g (x : ℝ) : ℝ :=
  if h : x = 0 then 0.5 else x^2 * Real.sin (1 / x) + 0.5

theorem exists_continuous_function (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1)) (h1 : ∀ m c : ℝ, ∀ᶠ x in (nhdsWithin 0 (Set.Icc 0 1)), f x ≠ m * x + c)
  (h2 : ∀ n : ℕ+, ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c ∧ countable {x | f x = m * x + c} ∧ {x | f x = m * x + c}.infinite) : 
  ∃ g : ℝ → ℝ, ContinuousOn g (Set.Icc 0 1) ∧ (∀ m c : ℝ, ¬ (Set.Iic 0 ∩ {x | g x = m * x + c}).infinite) ∧ (∀ n : ℕ+, ∃ m c : ℝ, (Set.Icc 0 1 ∩ {x | g x = m * x + c}).infinite) :=
  sorry

end exists_continuous_function_l354_354853


namespace constant_term_expansion_l354_354745

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : (x + 2 + 1/x)^n = 20) : n = 3 :=
by
sorry

end constant_term_expansion_l354_354745


namespace widgets_purchasable_at_new_price_l354_354877

variable (m n r : ℕ)
variable (p np k : ℕ)

-- Given conditions
def m := 24
def n := 6
def r := 1

-- Original widget price
def orig_price := m / n

-- New widget price after reduction
def new_price := orig_price - r

-- Number of widgets purchasable at new price
theorem widgets_purchasable_at_new_price : (24 / (4 - 1) = 8) :=
by
  have orig_price := m / n
  have new_price := orig_price - r
  exact m / new_price == 8
sorry

end widgets_purchasable_at_new_price_l354_354877


namespace hyperbola_min_value_l354_354433

open Real

-- Define all the necessary conditions and variables
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def left_focus (F1 : ℝ × ℝ) : Prop := F1 = (-√8, 0)
def right_branch (P : ℝ × ℝ) : Prop := ∃ x y, hyperbola x y ∧ x > 0 ∧ P = (x, y)
def left_focus_distance (F1 P : ℝ × ℝ) : ℝ := dist F1 P
def point_distance (P1 P2 : ℝ × ℝ) : ℝ := dist P1 P2

-- Define the statement to be proved
theorem hyperbola_min_value (F1 : ℝ × ℝ) (P1 P2 : ℝ × ℝ)
    (hf : left_focus F1) (hP1 : right_branch P1) (hP2 : right_branch P2) :
    ∃ C, C = 8 ∧ 
    ∃ F2, (P1.1 > 0 ∧ P2.1 > 0 ∧ 
           F1 = (-√8, 0) ∧
           F2 = (√8, 0) ∧
           left_focus_distance F1 P1 + left_focus_distance F1 P2 - point_distance P1 P2 ≥ C) := 
sorry

end hyperbola_min_value_l354_354433


namespace common_chord_length_l354_354560

noncomputable def first_circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 4*p.1 - 4*p.2 = 0}
noncomputable def second_circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 + 2*p.1 - 12 = 0}
noncomputable def common_chord_line := {p : ℝ × ℝ | p.1 - 2*p.2 + 6 = 0}
noncomputable def first_circle_center : ℝ × ℝ := (-2, 2)
noncomputable def first_circle_radius : ℝ := 2*Real.sqrt 2

theorem common_chord_length : 
  let length := 4*Real.sqrt 2 in
  True
:=by
  sorry

end common_chord_length_l354_354560


namespace sum_gt_6_is_random_event_l354_354339

theorem sum_gt_6_is_random_event :
  ∃ (S : set ℕ), (∀ x ∈ {1, 2, 3, ..., 10}, x ∈ S) ∧ 
  ∃ (a b c : ℕ), 
  {a, b, c} ⊆ S ∧ 
  6 < a + b + c ∧ 
  1 ≤ a ∧ a ≤ 10 ∧ 
  1 ≤ b ∧ b ≤ 10 ∧ 
  1 ≤ c ∧ c ≤ 10 ∧ 
  ((a + b + c = 6) → {a, b, c} = {1, 2, 3}) :=
sorry

end sum_gt_6_is_random_event_l354_354339


namespace units_digit_of_2_pow_1501_5_pow_1502_11_pow_1503_l354_354020

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_2_pow_1501_5_pow_1502_11_pow_1503 :
  units_digit (2 ^ 1501 * 5 ^ 1502 * 11 ^ 1503) = 0 :=
by
  let n := 1501
  let m := 1502
  let l := 1503
  have h2 : units_digit (2 ^ n) = units_digit 2 := sorry
  have h5 : units_digit (5 ^ m) = units_digit 5 := sorry
  have h11 : units_digit (11 ^ l) = units_digit 1 := sorry
  calc
  units_digit (2 ^ n * 5 ^ m * 11 ^ l) = units_digit (2 * 5 * 1) := by
    rw [h2, h5, h11]
  ... = units_digit 10 := by norm_num
  ... = 0 := by norm_num

end units_digit_of_2_pow_1501_5_pow_1502_11_pow_1503_l354_354020


namespace uncovered_cells_less_than_fraction_uncovered_cells_less_than_fraction_p5_l354_354594

theorem uncovered_cells_less_than_fraction (m n : ℕ) (domino_placement : ℕ → ℕ → bool) 
  (h_coverage : ∀ (i j : ℕ), (domino_placement i j ↔ domino_placement (i+1) j ∨ domino_placement i (j+1))) :
  (∑ i in range m, ∑ j in range n, if ¬domino_placement i j then 1 else 0) < (m * n) / 4 :=
by sorry

theorem uncovered_cells_less_than_fraction_p5 (m n : ℕ) (domino_placement : ℕ → ℕ → bool) 
  (h_coverage : ∀ (i j : ℕ), (domino_placement i j ↔ domino_placement (i+1) j ∨ domino_placement i (j+1))) :
  (∑ i in range m, ∑ j in range n, if ¬domino_placement i j then 1 else 0) < (m * n) / 5 :=
by sorry

end uncovered_cells_less_than_fraction_uncovered_cells_less_than_fraction_p5_l354_354594


namespace jelly_beans_per_bag_l354_354485

theorem jelly_beans_per_bag :
  (∃ n : ℕ, 10 ≤ n ∧ n ≤ 20 ∧ (∑ i in finset.range(n), 2*i + 1) = 225) →
  (∃ k : ℕ, 225 = 5 * k ∧ k = 45) :=
by
  intro h
  obtain ⟨n, hn1, hn2, hn3⟩ := h
  use 45
  split
  · exact hn3.symm.trans (by sorry)
  · refl

end jelly_beans_per_bag_l354_354485


namespace part1_solution_part2_solution_l354_354438

def A (x : ℝ) : Prop := (1 / 2) < 2^(x + 1) ∧ 2^(x + 1) < 8
def B (a x : ℝ) : Prop := 3 * a - 2 < x ∧ x < 2 * a + 1

theorem part1_solution (x : ℝ) : A x ∨ (x ≤ 1 ∨ 3 ≤ x) ↔ x < 2 ∨ 3 ≤ x := sorry

theorem part2_solution (a : ℝ) : (∀ x, A x → B a x → B a x) ↔ (a ∈ (Icc 0 (1 / 2) ∪ Ioc 3 ⊤)) := sorry

end part1_solution_part2_solution_l354_354438


namespace min_mn_arith_seq_l354_354483

theorem min_mn_arith_seq :
  ∃ (m n : ℕ), 
  (∀ (a : ℕ → ℕ) (d : ℕ),
    (∀ k, a k = a 0 + k * d) ∧ a 0 = 1919 ∧ a m = 1949 ∧ a n = 2019 → m + n = 15) :=
begin
  sorry
end

end min_mn_arith_seq_l354_354483


namespace problem_statement_l354_354338

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f₁ (x : ℝ) : ℝ := Real.log x
def f₂ (x : ℝ) : ℝ := x^2
def f₃ (x : ℝ) : ℝ := x^3
def f₄ (x : ℝ) : ℝ := x + 1

theorem problem_statement : is_even f₂ :=
by
  -- Proof goes here
  sorry

end problem_statement_l354_354338


namespace conjugate_of_z_l354_354088

noncomputable def imaginary_unit : ℂ := complex.i

def z : ℂ := (imaginary_unit ^ 2018) / ((1 - imaginary_unit) ^ 2)

theorem conjugate_of_z : complex.conj z = 1 / 2 * imaginary_unit := sorry

end conjugate_of_z_l354_354088


namespace base_video_card_cost_l354_354845

theorem base_video_card_cost
    (cost_computer : ℕ)
    (fraction_monitor_peripherals : ℕ → ℕ → ℕ)
    (twice : ℕ → ℕ)
    (total_spent : ℕ)
    (cost_monitor_peripherals_eq : fraction_monitor_peripherals cost_computer 5 = 300)
    (twice_eq : ∀ x, twice x = 2 * x)
    (eq_total : ∀ (base_video_card : ℕ), cost_computer + fraction_monitor_peripherals cost_computer 5 + twice base_video_card = total_spent)
    : ∃ x, total_spent = 2100 ∧ cost_computer = 1500 ∧ x = 150 :=
by
  sorry

end base_video_card_cost_l354_354845


namespace a_seq_formula_S_n_sum_l354_354083

noncomputable def a (n : ℕ) : ℚ := (2 / 3 : ℚ) * n + (1 / 3 : ℚ)

axiom a_4_a_7_eq_fifteen : a 4 * a 7 = 15
axiom a_3_a_8_eq_eight : a 3 + a 8 = 8

theorem a_seq_formula : a n = (2 / 3 : ℚ) * n + (1 / 3 : ℚ) :=
sorry

noncomputable def b (n : ℕ) [fact (n ≥ 2)] : ℚ := 1 / (9 * a (n - 1) * a n)
noncomputable def b₁ : ℚ := 1 / 3

noncomputable def S (n : ℕ) : ℚ :=
if n = 1 then b₁ else finset.sum (finset.range (n + 1)) (λ i, b (i + 1))

axiom b_eq : ∀ n : ℕ, n ≥ 2 → b n = 1 / (2 * (n * 2 - 1) * (n * 2 + 1))
axiom S_n_formula : ∀ n : ℕ, S n = n / (2 * n + 1)

theorem S_n_sum (n : ℕ) : S n = n / (2 * n + 1) :=
sorry

end a_seq_formula_S_n_sum_l354_354083


namespace percentage_gain_on_selling_price_l354_354567

def manufacturing_cost : ℝ := 200
def transportation_cost_per_100_shoes : ℝ := 500
def selling_price_per_shoe : ℝ := 246

theorem percentage_gain_on_selling_price :
  let transportation_cost_per_shoe := transportation_cost_per_100_shoes / 100 in
  let total_cost_per_shoe := manufacturing_cost + transportation_cost_per_shoe in
  let gain_per_shoe := selling_price_per_shoe - total_cost_per_shoe in
  (gain_per_shoe / selling_price_per_shoe) * 100 = 16.67 :=
by
  sorry

end percentage_gain_on_selling_price_l354_354567


namespace surface_area_of_sphere_50_pi_l354_354328

noncomputable def surface_area_of_sphere {a b c : ℕ} (h: {x // x = a} ) (i: {x // x = b} ) (j: {x // x = c}): ℝ :=
  let radius := (Real.sqrt ((a:ℝ)^2 + (b:ℝ)^2 + (c:ℝ)^2)) / 2
  in 4 * Real.pi * radius^2

theorem surface_area_of_sphere_50_pi :
  surface_area_of_sphere {a := 3} {b := 4} {c := 5} = 50 * Real.pi :=
by
  sorry

end surface_area_of_sphere_50_pi_l354_354328


namespace sequence_bound_l354_354264

theorem sequence_bound
  (a : ℕ → ℕ)
  (h_base0 : a 0 < a 1)
  (h_base1 : 0 < a 0 ∧ 0 < a 1)
  (h_recur : ∀ n, 2 ≤ n → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 :=
by
  sorry

end sequence_bound_l354_354264


namespace smallest_class_size_l354_354126

theorem smallest_class_size (n : ℕ) 
  (eight_students_scored_120 : 8 * 120 ≤ n * 92)
  (three_students_scored_115 : 3 * 115 ≤ n * 92)
  (min_score_70 : 70 * n ≤ n * 92)
  (mean_score_92 : (8 * 120 + 3 * 115 + 70 * (n - 11)) / n = 92) :
  n = 25 :=
by
  sorry

end smallest_class_size_l354_354126


namespace complex_number_solution_l354_354439

open Complex

theorem complex_number_solution (z1 z2 : ℂ)
  (h1 : z1 = conj z2)
  (h2 : (z1 + z2) - z1 * z2 * Complex.I = 4 - 6 * Complex.I) :
  (z1 = 2 + sqrt 2 * Complex.I ∧ z2 = 2 - sqrt 2 * Complex.I) ∨
  (z1 = 2 - sqrt 2 * Complex.I ∧ z2 = 2 + sqrt 2 * Complex.I) :=
sorry

end complex_number_solution_l354_354439


namespace polymial_no_positive_real_roots_example_l354_354074

noncomputable def polymial_no_positive_real_roots
    (a : ℕ → ℕ) (n k M : ℕ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) →
  (∑ i in finset.range n, 1 / (a i : ℝ) = k) →
  (finset.prod (finset.range n) (λ i, a i) = M) →
  (1 < M) →
  ∀ x > 0, M * (x + 1)^k < (∏ i in finset.range n, (x + a i) : ℝ)

theorem polymial_no_positive_real_roots_example
    (a : ℕ → ℕ) (n k M : ℕ) :
  polymial_no_positive_real_roots a n k M :=
by
  intros h1 h2 h3 h4 x hx
  sorry

end polymial_no_positive_real_roots_example_l354_354074


namespace sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l354_354995

def avg_daily_production := 400
def weekly_planned_production := 2800
def daily_deviations := [15, -5, 21, 16, -7, 0, -8]
def total_weekly_deviation := 80

-- Calculation for sets produced on Saturday
def sat_production_exceeds_plan := total_weekly_deviation - (daily_deviations.take (daily_deviations.length - 1)).sum
def sat_production := avg_daily_production + sat_production_exceeds_plan

-- Calculation for the difference between the max and min production days
def max_deviation := max sat_production_exceeds_plan (daily_deviations.maximum.getD 0)
def min_deviation := min sat_production_exceeds_plan (daily_deviations.minimum.getD 0)
def highest_lowest_diff := max_deviation - min_deviation

-- Calculation for the weekly wage for each worker
def workers := 20
def daily_wage := 200
def basic_weekly_wage := daily_wage * 7
def additional_wage := (15 + 21 + 16 + sat_production_exceeds_plan) * 10 - (5 + 7 + 8) * 15
def total_bonus := additional_wage / workers
def total_weekly_wage := basic_weekly_wage + total_bonus

theorem sat_production_correct : sat_production = 448 := by
  sorry

theorem highest_lowest_diff_correct : highest_lowest_diff = 56 := by
  sorry

theorem total_weekly_wage_correct : total_weekly_wage = 1435 := by
  sorry

end sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l354_354995


namespace ellipse_triangle_perimeter_is_18_l354_354521

open Real

-- Define the ellipse with given parameters 
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 9 = 1

-- Define the foci of the ellipse
def focal_distance (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Define the distance between foci points
def foci_distance (c : ℝ) : ℝ := 2 * c

-- The perimeter of triangle formed by point on ellipse and foci
def triangle_perimeter (a c : ℝ) : ℝ := 2 * (a + c)

-- Given conditions as definitions
axiom a_eq_5 : 5 = a
axiom b_eq_3 : 3 = b
axiom c_eq_4 : 4 = c

-- Proof statement
theorem ellipse_triangle_perimeter_is_18 (x y : ℝ) (hx : ellipse x y) : 
  triangle_perimeter 5 4 = 18 :=
by
  have ha : a = 5 := by rw [← a_eq_5]
  have hc : c = 4 := by rw [← c_eq_4]
  sorry

end ellipse_triangle_perimeter_is_18_l354_354521


namespace alex_produces_more_pages_l354_354535

variable (p h : ℕ)

theorem alex_produces_more_pages (p h : ℕ) (hp : p = 3 * h) :
  let pages_first_day := p * h,
      pages_second_day := (p - 5) * (h + 3)
  in pages_second_day - pages_first_day = 4 * h - 15 :=
by
  intros
  sorry

end alex_produces_more_pages_l354_354535


namespace set_of_lines_exists_l354_354044

noncomputable def exists_set_of_lines (M : Set (ℝ × ℝ × ℝ → Prop)) : Prop :=
  (∀ p : ℝ × ℝ × ℝ, ∃ l1 l2 ∈ M, l1 p ∧ l2 p ∧ l1 ≠ l2)
  ∧ 
  (∀ p q : ℝ × ℝ × ℝ, ∃ l1 l2 ... ln ∈ M, connected_by_polygonal_line l1 l2 ... ln p q)

theorem set_of_lines_exists : 
  ∃ M : Set (ℝ × ℝ × ℝ → Prop), exists_set_of_lines M :=
sorry

end set_of_lines_exists_l354_354044


namespace positive_real_y_sin_arccos_eq_y_l354_354934

theorem positive_real_y_sin_arccos_eq_y (y : ℝ) (h_pos : y > 0) (h_eq : sin (arccos y) = y) : y^2 = 1 / 2 :=
by
  -- We will insert the proof steps here if needed
  sorry

end positive_real_y_sin_arccos_eq_y_l354_354934


namespace fixed_charge_l354_354369

theorem fixed_charge(
  F C : ℝ,
  h1 : F + C = 52,
  h2 : F + 2 * C = 76
) : F = 28 := 
by {
  sorry
}

end fixed_charge_l354_354369


namespace spinner_probabilities_l354_354987

theorem spinner_probabilities (pA pB pC pD : ℚ) (h1 : pA = 1/4) (h2 : pB = 1/3) (h3 : pA + pB + pC + pD = 1) :
  pC + pD = 5/12 :=
by
  -- Here you would construct the proof (left as sorry for this example)
  sorry

end spinner_probabilities_l354_354987


namespace candidate_marks_secured_l354_354990

-- Conditions
def max_marks : ℕ := 150
def passing_percentage : ℝ := 0.40
def failed_by : ℕ := 20
def passing_marks : ℕ := (passing_percentage * max_marks).to_nat

-- Statement
theorem candidate_marks_secured (x : ℕ) (h : x + failed_by = passing_marks) : x = 40 :=
  sorry

end candidate_marks_secured_l354_354990


namespace coeff_sum_l354_354464

theorem coeff_sum (a : ℕ → ℤ) (x : ℤ) :
    (3 - (1 + x))^7 = ∑ k in (Finset.range 8), a k * (1 + x)^k → 
    (∑ k in (Finset.range 7), a k) = 129 :=
by
  sorry

end coeff_sum_l354_354464


namespace correct_options_l354_354382

-- Definitions based on conditions
def balls : List ℕ := [1, 2, 3, 4, 5]
def boxes : List ℕ := [1, 2, 3, 4]

-- Propositions for each option
def option_A : Prop := (4^5 = 1024)
def option_B : Prop := False -- Since C_4^3 ways is incorrect
def option_C : Prop := ((Nat.choose 5 4) * (Nat.choose 4 1) = 5 * 4)
def option_D : Prop := ((Nat.choose 5 2) * Nat.perm 4 4 = 10 * 24)

-- The main theorem stating the correct options
theorem correct_options : option_A ∧ option_C ∧ option_D :=
by {
    split,
    { exact rfl },
    split;
    { exact rfl }
}

#check correct_options

end correct_options_l354_354382


namespace tangent_line_equation_l354_354081

theorem tangent_line_equation :
  ∃ (P : ℝ × ℝ) (m : ℝ), 
  P = (-2, 15) ∧ m = 2 ∧ 
  (∀ (x y : ℝ), (y = x^3 - 10 * x + 3) → (y - 15 = 2 * (x + 2))) :=
sorry

end tangent_line_equation_l354_354081


namespace solve_prime_triplet_exists_l354_354214

theorem solve_prime_triplet_exists (p q r : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) 
    (prime_r : Nat.Prime r) (h : p ≠ q) : 
    (\frac{p}{q} - \frac{4}{r + 1} = 1) ↔ ((p = 7 ∧ q = 3 ∧ r = 2) ∨ (p = 5 ∧ q = 3 ∧ r = 5) ∨ (p = 3 ∧ q = 2 ∧ r = 7)) := 
by
  sorry

end solve_prime_triplet_exists_l354_354214


namespace f_five_is_eleven_l354_354392

def f (x : ℕ) : ℕ :=
if x >= 10 then x - 2 else f (f (x + 6))

theorem f_five_is_eleven : f 5 = 11 :=
by
  sorry

end f_five_is_eleven_l354_354392


namespace area_square_C_l354_354692

theorem area_square_C (s t : ℕ) (hA : 4 * s = 16) (hB : 4 * t = 32) :
  let side_length_C := s + t in (side_length_C * side_length_C = 144) :=
by 
  sorry

end area_square_C_l354_354692


namespace city_miles_per_tank_proof_l354_354993

def highway_miles_per_tank := 420
def city_miles_per_gallon := 24
def highway_miles_per_gallon := city_miles_per_gallon + 6

theorem city_miles_per_tank_proof (h_t: nat) : 
  h_t = highway_miles_per_tank / highway_miles_per_gallon →
  city_miles_per_tank = h_t * city_miles_per_gallon := 
sorry

end city_miles_per_tank_proof_l354_354993


namespace perpendicular_line_eq_l354_354912

noncomputable def slope_of_line (m : ℝ) : ℝ := 1 / m

theorem perpendicular_line_eq (m : ℝ) (h : m = 1): 
  ∃ L2: ℝ × ℝ → Prop, (∀ P: ℝ × ℝ, P = (2, 1) → L2 P) ∧ 
  (∀ P1 P2: ℝ × ℝ, (m * P1.fst - m^2 * P1.snd = 1) → (L2 P1) → P2.fst + P2.snd - 3 = 0) := 
by 
  use fun P => P.fst + P.snd - 3 = 0
  intro P hP
  simp [hP]
  sorry

end perpendicular_line_eq_l354_354912


namespace boat_problem_l354_354308

theorem boat_problem (x y : ℕ) (h : 12 * x + 5 * y = 99) :
  (x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3) :=
sorry

end boat_problem_l354_354308


namespace porter_monthly_earnings_l354_354887

/--
Porter earns $8 per day and works 5 times a week. He is promised an extra
50% on top of his daily rate for an extra day each week. There are 4 weeks in a month.
Prove that Porter will earn $208 in a month if he works the extra day every week.
-/
theorem porter_monthly_earnings :
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  in monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208 :=
by
  let daily_rate := 8
  let days_per_week := 5
  let weeks_per_month := 4
  let overtime_extra_rate := 0.5
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_without_overtime := weekly_earnings * weeks_per_month
  let overtime_earnings_per_day := daily_rate * (1 + overtime_extra_rate)
  let total_overtime_earnings_per_month := overtime_earnings_per_day * weeks_per_month
  show Prop, from monthly_earnings_without_overtime + total_overtime_earnings_per_month = 208

end porter_monthly_earnings_l354_354887


namespace Ryan_correct_percentage_l354_354894

theorem Ryan_correct_percentage 
  (score1 : ℕ) (score2 : ℕ) (score3 : ℕ) (score4 : ℕ)
  (test1 : ℕ) (test2 : ℕ) (test3 : ℕ) (test4 : ℕ) 
  (h1 : score1 = 75 * 20 / 100)
  (h2 : score2 = 85 * 50 / 100)
  (h3 : score3 = 60 * 30 / 100)
  (h4 : score4 = 0 * 15 / 100)
  (total_problems : ℕ := 20 + 50 + 30 + 15) 
  (total_correct : ℕ := score1 + score2 + score3 + score4) : 
  total_correct / total_problems * 100 = 65.65 := 
sorry

end Ryan_correct_percentage_l354_354894


namespace fraction_of_clever_div_by_27_l354_354159

def is_clever_integer (n : ℕ) : Prop :=
  even n ∧ n > 20 ∧ n < 120 ∧ (n.digits 10).sum = 9

def clever_integers : List ℕ :=
  List.filter is_clever_integer (List.range 120).drop 20

def divisible_by_27 (n : ℕ) : Prop :=
  n % 27 = 0

def clever_div_by_27 : List ℕ :=
  List.filter divisible_by_27 clever_integers

theorem fraction_of_clever_div_by_27 :
  (clever_div_by_27.length : ℚ) / (clever_integers.length : ℚ) = 2 / 5 :=
by
  sorry

end fraction_of_clever_div_by_27_l354_354159


namespace rearrange_2055_is_9_l354_354446

theorem rearrange_2055_is_9 : 
  let digits := [2, 0, 5, 5],
    count := Multiset.card digits,
    n_occur_2 := Multiset.count 2 digits,
    n_occur_0 := Multiset.count 0 digits,
    n_occur_5 := Multiset.count 5 digits,
    permutations := Nat.factorial count / (Nat.factorial n_occur_2 * Nat.factorial n_occur_0 * Nat.factorial n_occur_5) in
  permutations - (Nat.factorial (count - 1) / Nat.factorial n_occur_5) = 9 :=
by
  sorry

end rearrange_2055_is_9_l354_354446


namespace smallest_possible_other_integer_l354_354962

theorem smallest_possible_other_integer (n : ℕ) (h1 : Nat.lcm 60 n / Nat.gcd 60 n = 84) : n = 35 :=
sorry

end smallest_possible_other_integer_l354_354962


namespace new_age_ratio_l354_354582

theorem new_age_ratio (F S : ℕ) (I : ℝ) :
  (F / S = 7 / 3) →
  (F * S = 756) →
  (F = 0.4 * ↑I) →
  (F + 6) / (S + 6) = 2 / 1 :=
by
  intro h1 h2 h3
  sorry

end new_age_ratio_l354_354582


namespace bikers_meet_in_approx_4_19_hours_l354_354599

open Real

-- Conditions
def speed_a : ℝ := 16
def speed_b : ℝ := 14
def diameter : ℝ := 40

-- Calculate the circumference
def circumference : ℝ := π * diameter

-- Relative speed
def relative_speed : ℝ := speed_a + speed_b

-- Time to meet
noncomputable def time_to_meet : ℝ := circumference / relative_speed

-- Statement to be proved
theorem bikers_meet_in_approx_4_19_hours : time_to_meet = 125.66370614359174 / 30 := by
  sorry

end bikers_meet_in_approx_4_19_hours_l354_354599


namespace compute_x_y_power_sum_l354_354513

noncomputable def pi : ℝ := Real.pi

theorem compute_x_y_power_sum
  (x y : ℝ)
  (h1 : 1 < x)
  (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^pi + y^pi = 2^(pi * (16:ℝ)^(1/5)) + 3^(pi * (16:ℝ)^(1/5)) :=
by
  sorry

end compute_x_y_power_sum_l354_354513


namespace concentric_circle_chord_ratio_l354_354103

theorem concentric_circle_chord_ratio 
  (r R : ℝ) (h1 : 0 < r) (h2 : 0 < R) (h3 : r < R) :
  ∃ (A B C D: ℝ × ℝ), 
    (A ≠ B ∧ C ≠ D) ∧ 
    (dist A B = 2 * dist C D) := sorry

end concentric_circle_chord_ratio_l354_354103


namespace common_root_equations_l354_354472

theorem common_root_equations (a b : ℝ) 
  (h : ∃ x₀ : ℝ, (x₀ ^ 2 + a * x₀ + b = 0) ∧ (x₀ ^ 2 + b * x₀ + a = 0)) 
  (hc : ∀ x₁ x₂ : ℝ, (x₁ ^ 2 + a * x₁ + b = 0 ∧ x₂ ^ 2 + bx₀ + a = 0) → x₁ = x₂) :
  a + b = -1 :=
sorry

end common_root_equations_l354_354472


namespace circumcenter_ADE_on_circumcircle_ABC_l354_354943

/-!
  Given triangle ABC with centroid G and circumcenter O such that GO ⊥ AG, 
  and A' is the second intersection of AG with the circumcircle of triangle ABC.
  Let D be the intersection of lines CA' and AB, and E the intersection of 
  lines BA' and AC.
  Prove that the circumcenter of triangle ADE lies on the circumcircle of triangle ABC.
-/
theorem circumcenter_ADE_on_circumcircle_ABC
  (A B C G O A' D E : Point)
  (h1 : Triangle A B C)
  (h2 : Centroid G A B C)
  (h3 : Circumcenter O A B C)
  (h4 : GO_perp_AG : Perpendicular (Line.mk G O) (Line.mk A G))
  (h5 : A'_intersection : Second_intersection (AG_intersection (Circumcircle_ABC A B C)) = A')
  (h6 : D_intersection : Intersection (Line.mk C A') (Line.mk A B) = D)
  (h7 : E_intersection : Intersection (Line.mk B A') (Line.mk A C) = E)
  : lies_on (Circumcenter (Triangle.mk A D E)) (Circumcircle_ABC A B C) :=
sorry

end circumcenter_ADE_on_circumcircle_ABC_l354_354943


namespace equal_segments_imply_equal_arcs_l354_354852

open EuclideanGeometry

variable {A B C K L M P : Point} (Γ : Circle A B C) (S : Point)

-- Definitions
def IsInteriorPointOfTriangle (P A B C : Point) : Prop := sorry
def LinesIntersectCircleAt (A P K : Point) (Γ : Circle A B C) : Prop := sorry
def TangentAtPoint (C S : Point) (Γ : Circle A B C) : Prop := sorry

-- Problem Statement
theorem equal_segments_imply_equal_arcs
    (h1 : IsInteriorPointOfTriangle P A B C)
    (h2 : LinesIntersectCircleAt A P K Γ)
    (h3 : LinesIntersectCircleAt B P L Γ)
    (h4 : LinesIntersectCircleAt C P M Γ)
    (h5 : TangentAtPoint C S Γ)
    (h6 : Line AB S)
    (h7 : SC = SP) :
  MK = ML := 
sorry

end equal_segments_imply_equal_arcs_l354_354852


namespace angle_at_3_25_l354_354700

def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours * 30) + (minutes * 30 / 60)

def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

def angle_between_hands (hours : ℕ) (minutes : ℕ) : ℝ :=
  let h_angle := hour_hand_angle hours minutes
  let m_angle := minute_hand_angle minutes
  abs (m_angle - h_angle)

theorem angle_at_3_25 : angle_between_hands 3 25 = 47.5 :=
by
  sorry -- The detailed proof is skipped using 'sorry'

end angle_at_3_25_l354_354700


namespace monotonicity_intervals_lambda_range_ineq_l354_354431

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln (1 + a * x) - x ^ 2

theorem monotonicity_intervals (a : ℝ) (h_a : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x < (sqrt (2 * a ^ 2 + 1) - 1) / (2 * a) → 0 < f' a x) ∧ 
  (∀ x : ℝ, (sqrt (2 * a ^ 2 + 1) - 1) / (2 * a) < x ∧ x ≤ 1 → f' a x < 0) :=
sorry

theorem lambda_range_ineq (λ : ℝ) :
  (∀ n : ℕ, 0 < n → (1 / (n ^ 2 : ℝ)) + λ ≥ ln (1 + 2 / n)) → 
  λ ≥ ln 2 - 1 / 4 :=
sorry

end monotonicity_intervals_lambda_range_ineq_l354_354431


namespace travel_time_K_l354_354980

theorem travel_time_K (d x : ℝ) (h_pos_d : d > 0) (h_x_pos : x > 0) (h_time_diff : (d / (x - 1/2)) - (d / x) = 1/2) : d / x = 40 / x :=
by
  sorry

end travel_time_K_l354_354980


namespace dice_product_sum_impossible_l354_354950

theorem dice_product_sum_impossible (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (hprod : d1 * d2 * d3 * d4 = 180) :
  (d1 + d2 + d3 + d4 ≠ 14) ∧ (d1 + d2 + d3 + d4 ≠ 17) :=
by
  sorry

end dice_product_sum_impossible_l354_354950


namespace probability_at_least_one_shows_1_l354_354766

def prob_at_least_one (s : Set (ℕ × ℕ)) (E : Event s) : ℝ :=
  let die1 : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let die2 : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let outcomes := die1.product die2
  let favorable_outcomes := outcomes.filter (λ (x : ℕ × ℕ), x.fst = 1 ∨ x.snd = 1)
  (favorable_outcomes.card : ℝ) / (outcomes.card : ℝ)

theorem probability_at_least_one_shows_1 : prob_at_least_one = 11 / 36 :=
sorry

end probability_at_least_one_shows_1_l354_354766


namespace proof_problem_l354_354096

open Real

noncomputable def f (x : ℝ) : ℝ := 2^x + x
noncomputable def g (x : ℝ) : ℝ := log x / log 2 + x
noncomputable def h (x : ℝ) : ℝ := log x / log 2 - 2

lemma zero_of_f_lt_zero (a : ℝ) (hfa : f a = 0) : a < 0 := 
sorry

lemma zero_of_g_between_zero_and_one (b : ℝ) (hgb : g b = 0) : 0 < b ∧ b < 1 := 
sorry

lemma zero_of_h_eq_four (c : ℝ) (hhc : h c = 0) : c = 4 := 
sorry

theorem proof_problem (a b c : ℝ) 
  (hfa : f a = 0) 
  (hgb : g b = 0) 
  (hhc : h c = 0) : a < b ∧ b < c :=
begin
  have ha : a < 0 := zero_of_f_lt_zero a hfa,
  have hb : 0 < b ∧ b < 1 := zero_of_g_between_zero_and_one b hgb,
  have hc : c = 4 := zero_of_h_eq_four c hhc,
  split,
  { exact ha },
  { cases hb with hb_left hb_right,
    linarith }
end

end proof_problem_l354_354096


namespace log_arith_prog_iff_eqn_l354_354586

section
variables {x y z : ℝ}

/-- The numbers lg x, lg y, and lg z are in arithmetic progression if and only if y^2 = xz. -/
theorem log_arith_prog_iff_eqn (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  (∃ d : ℝ, log x + 2*d = log y ∧ log y + d = log z) ↔ y^2 = x * z :=
sorry
end

end log_arith_prog_iff_eqn_l354_354586


namespace find_intervals_of_monotonicity_find_max_min_values_l354_354752

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem find_intervals_of_monotonicity :
  (∀ x, (x > 1 ∨ x < -1) → f' x > 0) ∧ (∀ x, (-1 < x ∧ x < 1) → f' x < 0) := sorry

theorem find_max_min_values :
  (∃ (x_min x_max : ℝ), 
    x_min ∈ set.Icc (-3 : ℝ) 2 ∧ f x_min = -18 ∧ 
    x_max ∈ set.Icc (-3 : ℝ) 2 ∧ f x_max = 2 
  ) := sorry

end find_intervals_of_monotonicity_find_max_min_values_l354_354752


namespace triangle_ABC_properties_l354_354155

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 13)
  (h3 : c = 3)
  (h_angle_range : A > 0 ∧ A < Real.pi) : 
  A = Real.pi / 3 ∧ (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 := 
by
  sorry

end triangle_ABC_properties_l354_354155


namespace smallest_constant_N_l354_354056

-- Given that a, b, c are sides of a triangle and in arithmetic progression, prove that
-- (a^2 + b^2 + c^2) / (ab + bc + ca) ≥ 1.

theorem smallest_constant_N
  (a b c : ℝ)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality
  (hap : ∃ d : ℝ, b = a + d ∧ c = a + 2 * d) -- Arithmetic progression
  : (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ≥ 1 := 
sorry

end smallest_constant_N_l354_354056


namespace largest_angle_of_pentagon_l354_354820

theorem largest_angle_of_pentagon (x : ℕ) (hx : 5 * x + 100 = 540) : x + 40 = 128 := by
  sorry

end largest_angle_of_pentagon_l354_354820


namespace clock_angle_at_330_l354_354344

theorem clock_angle_at_330 :
  let h := 3 in
  let m := 30 in
  let angle := | (60 * h - 11 * m) / 2 | in
  angle = 75 :=
by
  sorry

end clock_angle_at_330_l354_354344


namespace closest_fraction_to_whole_l354_354245

-- Define the given ratio
def ratio := (10^2000 + 10^2002) / (2 * 10^2001)

-- Define the goal: the closest integer to the ratio
def closestWholeNumber := 5

-- State the theorem
theorem closest_fraction_to_whole :
  Int.round ratio = closestWholeNumber :=
by
  sorry

end closest_fraction_to_whole_l354_354245


namespace train_cross_bridge_time_l354_354619

variables (l_t l_b v : ℝ) (t : ℝ)

def speed_in_m_s (v_kmph : ℝ) : ℝ := v_kmph * 1000 / 3600

theorem train_cross_bridge_time 
  (h1 : l_t = 100)
  (h2 : l_b = 200)
  (h3 : v = 36)
  (h4 : t = 30) :
  (l_t + l_b) / (speed_in_m_s v) = t :=
sorry

end train_cross_bridge_time_l354_354619


namespace max_distance_in_intersection_of_spheres_l354_354878

theorem max_distance_in_intersection_of_spheres (T : set ℝ³) (s : set (set ℝ³)) (e : ℝ) 
  (hT : regular_tetrahedron T 1) 
  (hs : ∀ edge ∈ edges T, ∃ sphere ∈ s, sphere = {x ∈ ℝ³ | dist x (midpoint edge) ≤ e / 2}) :
  (∀ (p q ∈ ⋂₀ s), dist p q ≤ 1 / real.sqrt 6) :=
sorry

end max_distance_in_intersection_of_spheres_l354_354878


namespace sum_of_roots_of_quadratic_eq_l354_354959

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  (((-b + real.sqrt(b ^ 2 - 4 * a * c)) / (2 * a)),
   ((-b - real.sqrt(b ^ 2 - 4 * a * c)) / (2 * a)))

theorem sum_of_roots_of_quadratic_eq :
  let f := λ x : ℝ => (x - 3) ^ 2 - 16
  let a := 1
  let b := -6
  let c := -7
  let roots := quadratic_roots a b c
  roots.fst + roots.snd = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l354_354959


namespace sum_of_interior_numbers_eighth_row_l354_354499

def sum_of_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem sum_of_interior_numbers_eighth_row : sum_of_interior_numbers 8 = 126 :=
by
  sorry

end sum_of_interior_numbers_eighth_row_l354_354499


namespace num_integers_n_cubed_plus_eight_divisors_l354_354777

theorem num_integers_n_cubed_plus_eight_divisors :
  {n : ℤ | (nat.factors (int.natAbs (n^3 + 8))).length ≤ 3 }.finite.toFinset.card = 2 :=
by
  sorry

end num_integers_n_cubed_plus_eight_divisors_l354_354777


namespace parameter_range_l354_354063

theorem parameter_range (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≤ 5 ∧ x2 ≤ 5 ∧ (x1^2 - a * x1 + 2 * x1 - 2 * a) * sqrt (5 - x1) ≤ 0
    ∧ (x2^2 - a * x2 + 2 * x2 - 2 * a) * sqrt (5 - x2) ≤ 0 ∧ abs (x2 - x1) = 5) ↔
    a ∈ Set.Iic (-7) ∪ Set.Ici 0 := 
sorry

end parameter_range_l354_354063


namespace rudolph_stop_signs_per_mile_l354_354194

theorem rudolph_stop_signs_per_mile :
  let distance := 5 + 2
  let stop_signs := 17 - 3
  (stop_signs / distance) = 2 :=
by
  let distance := 5 + 2
  let stop_signs := 17 - 3
  calc
    (stop_signs / distance) = (14 / 7) : by rw [stop_signs, distance]
                          ... = 2 : by norm_num

end rudolph_stop_signs_per_mile_l354_354194


namespace find_a_l354_354032

noncomputable def A_coords (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 4 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

noncomputable def B_coords (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 6 * a^2 * x - 2 * a^3 * y + 4 * a * y + a^4 + 4 = 0

noncomputable def opposite_sides (a : ℝ) (yA yB : ℝ) : Prop :=
  (yA - 1) * (yB - 1) < 0

theorem find_a (a xA yA xB yB : ℝ) :
  A_coords a xA yA →
  B_coords a xB yB →
  yA = a →
  xA = a →
  yB = a - 2 / a →
  xB = 3 →
  (yA ≠ 1) →
  (yB ≠ 1) →
  (opposite_sides a yA yB) ↔ a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo (1 : ℝ) 2 := sorry

end find_a_l354_354032
