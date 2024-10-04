import Mathlib

namespace exists_m_n_l555_555672

noncomputable def f (x a : ℝ) := - (1 / 2) * x^2 + x + a

theorem exists_m_n (a : ℝ) (h_a : a ≤ 5 / 2) : 
  (∃ m n : ℝ, m < n ∧ (∀ x, m ≤ x ∧ x ≤ n → f x a = 3 * x)) ↔ a ∈ Ioo (-2) (5 / 2) :=
sorry

end exists_m_n_l555_555672


namespace solution_set_of_inequality_l555_555555

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem solution_set_of_inequality :
  {x : ℝ | f (2 * x + 1) + f (1) ≥ 0} = {x : ℝ | -1 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l555_555555


namespace length_AX_l555_555771

open Real

theorem length_AX {A B C D X : Point}
  (h_circle : Circle (diameter AD = 1))
  (h_X_on_AD : on_diameter X A D)
  (h_BX_CX : dist B X = dist C X)
  (h_angles : 3 * angle A B C = angle B X C = 36) :
  dist A X = cos 6 * sin 12 * csc 18 :=
sorry

end length_AX_l555_555771


namespace perimeter_of_regular_polygon_l555_555412

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555412


namespace seating_arrangements_l555_555346

def numWaysCircularSeating (democrats republicans independents : Nat) : Nat :=
  Nat.factorial (democrats + republicans + independents - 1)

theorem seating_arrangements :
  numWaysCircularSeating 4 4 3 = 10! :=
by
  unfold numWaysCircularSeating
  sorry

end seating_arrangements_l555_555346


namespace common_difference_is_power_of_10_l555_555343

theorem common_difference_is_power_of_10 (A : ℕ → ℕ) (D : ℕ) :
  (∀ n, ∃ digits : list ℕ, (list.join digits).take n = list.range n ∧ A(n) = A(0) + n * D) →
  (∃ k : ℕ, D = 10 ^ k) :=
by sorry

end common_difference_is_power_of_10_l555_555343


namespace find_x_l555_555165

variables (x : ℝ)

def a : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (3, -1)

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (h : dot_prod (vec_add a b) (vec_sub a (3 • b)) = 0) : x = -6 :=
sorry

end find_x_l555_555165


namespace sum_of_dimensions_eq_18_sqrt_1_5_l555_555021

theorem sum_of_dimensions_eq_18_sqrt_1_5 (P Q R : ℝ) (h1 : P * Q = 30) (h2 : P * R = 50) (h3 : Q * R = 90) :
  P + Q + R = 18 * Real.sqrt 1.5 :=
sorry

end sum_of_dimensions_eq_18_sqrt_1_5_l555_555021


namespace isosceles_triangle_base_length_l555_555273

theorem isosceles_triangle_base_length (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 5) (h₂ : a + b + c = 17) : c = 7 :=
by
  -- proof would go here
  sorry

end isosceles_triangle_base_length_l555_555273


namespace product_B_original_price_l555_555450

variable (a b : ℝ)

theorem product_B_original_price (h1 : a = 1.2 * b) (h2 : 0.9 * a = 198) : b = 183.33 :=
by
  sorry

end product_B_original_price_l555_555450


namespace limit_of_u_l555_555096

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Definition of the sequence u_n
def u (n : ℕ) : ℚ := ∑ i in Finset.range n, 1 / (fib i * fib (i + 2))

-- The main limit statement we want to prove
theorem limit_of_u : tendsto (λ n, u n) at_top (𝓝 1) :=
by
  sorry

end limit_of_u_l555_555096


namespace number_of_students_l555_555361

-- Conditions
def avg_original := 72
def avg_correct := 71.71
def reema_incorrect := 46
def reema_correct := 56
def correction := reema_correct - reema_incorrect

-- Hypothesis: The incorrect average gives us the expression
def incorrect_total (n : ℕ) := avg_original * n
def correct_total (n : ℕ) := incorrect_total n + correction

-- Statement to prove
theorem number_of_students (n : ℕ) :
  (correct_total n) / n = avg_correct ↔ n = Nat.floor (10 / 0.29) :=
by
  sorry

end number_of_students_l555_555361


namespace regular_polygon_perimeter_l555_555429

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555429


namespace find_c_l555_555980

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l555_555980


namespace hyperbola_eccentricity_l555_555131

-- Conditions: the parabola, focus, and distance relations.
def parabola_eq (x y : ℝ) := y^2 = 8 * x
def focus : ∀ {x y : ℝ}, parabola_eq x y → (x - 2)^2 + y^2 = 16

-- Prove the eccentricity of the hyperbola.
theorem hyperbola_eccentricity (x y : ℝ) (hx : parabola_eq x y) 
  : ( ∃ e : ℝ, e = real.sqrt 5 ∨ e = real.sqrt 5 / 2 ) :=
begin
  sorry
end

end hyperbola_eccentricity_l555_555131


namespace solution_inequality_l555_555941

noncomputable def f : ℝ → ℝ := sorry

theorem solution_inequality (h_f_odd: ∀ x, f (-x) = -f x)
  (h_f_deriv: ∀ x, deriv f x < 2) :
  {x : ℝ | f(x + 1) - log(x + 2) - 2 > exp(x + 1) + 3 * x} = set.Ioo (-2) (-1) :=
  sorry

end solution_inequality_l555_555941


namespace gcd_ab_is_22_l555_555761

def a : ℕ := 198
def b : ℕ := 308

theorem gcd_ab_is_22 : Nat.gcd a b = 22 := 
by { sorry }

end gcd_ab_is_22_l555_555761


namespace hyperbola_properties_l555_555503

-- Define the ellipse given in the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 144) + (y^2 / 169) = 1

-- Define the hyperbola that shares a common focus with the ellipse and passes through (0,2)
def hyperbola (x y : ℝ) : Prop := (y^2 / 4) - (x^2 / 21) = 1

-- Prove that the hyperbola shares common aspects with the given conditions
theorem hyperbola_properties :
    (∃ (x y : ℝ), ellipse x y ∧ ((0,2) ∈ set_of (λ (p : ℝ × ℝ), hyperbola p.1 p.2))
    ∧ ellipse 0 5
    ∧ ellipse 0 (-5)
    ∧ hyperbola 0 5
    ∧ hyperbola 0 (-5)
    ∧ real_axis_length = 4
    ∧ focal_distance = 10
    ∧ eccentricity = 5 / 2
    ∧ (∀ (x : ℝ), y = (2 * sqrt 21 / 21) * x ∨ y = (-2 * sqrt 21 / 21) * x)) :=
by
  sorry -- Proof steps are skipped here

-- Define additional terms for readability (not required but helpful for understanding)
def real_axis_length : ℝ := 2 * 2
def focal_distance : ℝ := 2 * 5
def eccentricity : ℝ := 5 / 2


end hyperbola_properties_l555_555503


namespace two_point_three_six_as_fraction_l555_555759

theorem two_point_three_six_as_fraction : (236 : ℝ) / 100 = (59 : ℝ) / 25 := 
by
  sorry

end two_point_three_six_as_fraction_l555_555759


namespace regular_polygon_perimeter_is_28_l555_555397

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555397


namespace cos_double_beta_l555_555908

noncomputable def sin_alpha := real.sin α = (real.sqrt 5) / 5
noncomputable def sin_alpha_minus_beta := real.sin (α - β) = -(real.sqrt 10) / 10
noncomputable def acute_angles := 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2

theorem cos_double_beta (α β : ℝ) (h₁ : sin_alpha) (h₂ : sin_alpha_minus_beta) (h₃ : acute_angles) :
  real.cos (2 * β) = 0 :=
sorry

end cos_double_beta_l555_555908


namespace double_integral_value_l555_555468

noncomputable def integrand (x y : ℝ) : ℝ := x / (y^5)

def region_D (x y : ℝ) : Prop :=
  1 ≤ (x^2)/16 + y^2 ∧ (x^2)/16 + y^2 ≤ 3 ∧ y ≥ x/4 ∧ x ≥ 0

open MeasureTheory

theorem double_integral_value :
  ∫∫ (x y : ℝ) in { p : ℝ × ℝ | region_D p.1 p.2 } , integrand x y 
  = 4 :=
  sorry

end double_integral_value_l555_555468


namespace max_groups_possible_l555_555185

theorem max_groups_possible
  (b : Fin 5 → ℕ) (g : Fin 5 → ℕ)
  (hb : (∑ i, b i) = 300)
  (hg : (∑ i, g i) = 300)
  (h_class_students : ∀ i, b i + g i = 120)
  (h_min_boys_in_class : ∀ i, b i ≥ 33)
  (h_min_girls_in_class : ∀ i, g i ≥ 33) :
  ∃ max_groups : ℕ, max_groups = 192 :=
sorry

end max_groups_possible_l555_555185


namespace table_value_l555_555679

theorem table_value (n m : ℕ) : 
    ∃ a : ℕ, (∀ i j : ℕ, a = ((i - 1) + (j - 1)) + 1) → 
    (a = (n + 1) + 1 * (m - 1) = m + n) := sorry

end table_value_l555_555679


namespace min_F_value_F_eq_g_range_l555_555653

-- Condition for 'a'
variable (a : ℝ) (ha : a ≥ 3)

-- Defining the function F
def F (x : ℝ) := min (2 * abs (x - 1)) (x^2 - 2 * a * x + 4 * a - 2)

-- Question 1: Minimum value of F
theorem min_F_value : 
  let m := if 3 ≤ a ∧ a ≤ 2 + Real.sqrt 2 then 0 else -a^2 + 4 * a - 2 in
  ∃ m, F m = if 3 ≤ a ∧ a ≤ 2 + Real.sqrt 2 then 0 else -a^2 + 4 * a - 2 :=
sorry

-- Question 2: Range for F(x) = g(x)
theorem F_eq_g_range : 
  ∀ x, F(x) = x^2 - 2 * a * x + 4 * a - 2 ↔ 2 ≤ x ∧ x ≤ 2 * a :=
sorry

end min_F_value_F_eq_g_range_l555_555653


namespace unattainable_y_l555_555060

theorem unattainable_y (x : ℝ) (h : 4 * x + 5 ≠ 0) : 
  (y = (3 - x) / (4 * x + 5)) → (y ≠ -1/4) :=
sorry

end unattainable_y_l555_555060


namespace intersection_of_sets_l555_555673

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ x : ℝ, y = 2^x - 1 }
def C : Set ℝ := { m | -1 < m ∧ m < 2 }

theorem intersection_of_sets : A ∩ B = C := 
by sorry

end intersection_of_sets_l555_555673


namespace angle_bxy_l555_555196

theorem angle_bxy (AB CD: Line) (AXE CYX BXY: Angle) (par : AB ∥ CD)
  (h : AXE = 4 * CYX - 120) : 
  BXY = 40 :=
by
  sorry

end angle_bxy_l555_555196


namespace trajectory_and_slopes_l555_555938

open_locale classical

-- Define the necessary points and the circle equation
variables {F1 F2 C M N P A B : Point}
variables (k1 k2 : ℝ)

-- Assume the given conditions
axiom cond1 : circle_equation F1 (λ x y, (x + 2)^2 + y^2 = 32)
axiom cond2 : F2 = (2, 0)
axiom cond3 : C ∈ circle F1
axiom cond4 : perp_bisector_inter F2C F1C M
axiom cond5 : N = (0, 2)
axiom cond6 : line_through P (-1, -2) intersects trajectory_of_M A B
axiom cond7 : slope N A = k1
axiom cond8 : slope N B = k2

-- State the theorem
theorem trajectory_and_slopes :
  (trajectory_equation M = (λ x y, x^2 / 8 + y^2 / 4 = 1)) ∧ (k1 + k2 = 4) :=
sorry

end trajectory_and_slopes_l555_555938


namespace number_of_paths_in_MATHEMATICIAN_diagram_l555_555868

theorem number_of_paths_in_MATHEMATICIAN_diagram : ∃ n : ℕ, n = 8191 :=
by
  -- Define necessary structure
  -- Number of rows and binary choices
  let rows : ℕ := 12
  let choices_per_position : ℕ := 2
  -- Total paths calculation
  let total_paths := choices_per_position ^ rows
  -- Including symmetry and subtracting duplicate
  let final_paths := 2 * total_paths - 1
  use final_paths
  have : final_paths = 8191 :=
    by norm_num
  exact this

end number_of_paths_in_MATHEMATICIAN_diagram_l555_555868


namespace interest_rate_l555_555735
-- Importing necessary library

-- Define the conditions
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) := (P * R * T) / 100

-- Main theorem statement
theorem interest_rate (P SI : ℝ) (T : ℝ) : 
    (SI = P - 1920) → (P = 2400) → SI = simple_interest P R T → R = 4 :=
by
  sorry

end interest_rate_l555_555735


namespace transformed_curve_is_circle_l555_555753

theorem transformed_curve_is_circle:
  ∀ (ρ θ : ℝ),
    ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2) ->
    let x := ρ * Real.cos θ in
    let y := ρ * Real.sin θ in
    ∀ (x' y' : ℝ),
      x' = (1/2) * x ->
      y' = (Real.sqrt 3 / 3) * y ->
      (x'^2 + y'^2 = 12) :=
sorry

end transformed_curve_is_circle_l555_555753


namespace excenter_distance_inequality_l555_555253

variables (A B C I_a I_b I_c : Type)
variables [acute_triangle : ∀ (ABC : A ∧ B ∧ C), acute_triangle ABC]

def acute_angle (α β γ : ℝ) := α ≥ β ∧ β ≥ γ

noncomputable def semi_perimeter (p : ℝ) := p

-- conditions
variable (p : ℝ)
variable (α β γ : ℝ)
variable (angles : ∀ (A B C : ℝ), (∠ A = 2 * α) ∧ (∠ B = 2 * β) ∧ (∠ C = 2 * γ))
variable (exincenters : ∀ (A B C I_a I_b I_c : ℝ), (A = I_a) ∧ (B = I_b) ∧ (C = I_c))
variable (distance_relation : ∀ (A I_a B I_b C I_c p : ℝ), (A * cos α = B * cos β ∧ B * cos β = C * cos γ ∧ C * cos γ = p))
variable (distance_inequality : ∀ (A I_a B I_b : ℝ), (A * I_a ≥ B * I_b ∧ B * I_b ≥ C * I_c))

-- goal
theorem excenter_distance_inequality (h : acute_triangle ABC) (ha : acute_angle α β γ)
  (hp : semi_perimeter p) (ha : angles A B C)
  (hex : exincenters A B C I_a I_b I_c)
  (hdr : distance_relation A I_a B I_b C I_c p)
  (hdi : distance_inequality A I_a B I_b) :
  A * I_a < A * C + B * C :=
sorry

end excenter_distance_inequality_l555_555253


namespace angle_between_a_and_b_is_2pi_over_3_l555_555542

open Real

variables (a b c : ℝ × ℝ)

-- Given conditions
def condition1 := a.1^2 + a.2^2 = 2  -- |a| = sqrt(2)
def condition2 := b = (-1, 1)        -- b = (-1, 1)
def condition3 := c = (2, -2)        -- c = (2, -2)
def condition4 := a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 1  -- a · (b + c) = 1

-- Prove the angle θ between a and b is 2π/3
theorem angle_between_a_and_b_is_2pi_over_3 :
  condition1 a → condition2 b → condition3 c → condition4 a b c →
  ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = -(1/2) ∧ θ = 2 * π / 3 :=
by
  sorry

end angle_between_a_and_b_is_2pi_over_3_l555_555542


namespace Martha_blocks_end_l555_555257

variable (Ronald_blocks : ℕ) (Martha_start_blocks : ℕ) (Martha_found_blocks : ℕ)
variable (Ronald_has_blocks : Ronald_blocks = 13)
variable (Martha_has_start_blocks : Martha_start_blocks = 4)
variable (Martha_finds_more_blocks : Martha_found_blocks = 80)

theorem Martha_blocks_end : Martha_start_blocks + Martha_found_blocks = 84 :=
by
  have Martha_start_blocks := Martha_has_start_blocks
  have Martha_found_blocks := Martha_finds_more_blocks
  sorry

end Martha_blocks_end_l555_555257


namespace regular_polygon_perimeter_l555_555444

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555444


namespace perimeter_of_polygon_l555_555437

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555437


namespace daily_rental_cost_l555_555798

theorem daily_rental_cost
  (daily_rent : ℝ)
  (cost_per_mile : ℝ)
  (max_budget : ℝ)
  (miles : ℝ)
  (H1 : cost_per_mile = 0.18)
  (H2 : max_budget = 75)
  (H3 : miles = 250)
  (H4 : daily_rent + (cost_per_mile * miles) = max_budget) : daily_rent = 30 :=
by sorry

end daily_rental_cost_l555_555798


namespace find_expression_l555_555534

variables {x y : ℝ}

theorem find_expression
  (h1: 3 * x + y = 5)
  (h2: x + 3 * y = 6)
  : 10 * x^2 + 13 * x * y + 10 * y^2 = 97 :=
by
  sorry

end find_expression_l555_555534


namespace parabola_b_coefficient_l555_555728

theorem parabola_b_coefficient {a b c q : ℝ} (h_vertex : ∀ x : ℝ, y = a * (x - q) ^ 2 - q)
  (h_y_intercept : y = q when x = 0) (h_q_nonzero : q ≠ 0) : b = -4 :=
by sorry

end parabola_b_coefficient_l555_555728


namespace angle_between_altitudes_proof_l555_555278

-- Define conditions
def sum_external_angles (a b c : ℝ) : Prop :=
  a + b + c = 360

def proportional_external_angles (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 7 * x ∧ c = 8 * x

-- Define the theorem to be proved
theorem angle_between_altitudes_proof (a b c : ℝ) (ha : sum_external_angles a b c)
  (hp : proportional_external_angles a b c) : 
  ∀ (α β : ℝ), α = 54 ∧ β = 36 → angle_between_altitudes α β = 90 :=
by sorry

end angle_between_altitudes_proof_l555_555278


namespace range_f_when_a_is_1_find_a_for_minimum_value_l555_555552

namespace ProblemSolutions

-- Problem 1
def f (x a : ℝ) := 4*x^2 - 4*a*x + (a^2 - 2*a + 2)

theorem range_f_when_a_is_1 : 
  (set.range (λ x, f x 1)) ∩ (set.Icc 0 2) = set.Icc 0 9 :=
sorry

-- Problem 2
theorem find_a_for_minimum_value :
  ∃ a : ℝ, (f 0 a = 3 ∨ f (a / 2) a = 3 ∨ f 2 a = 3) ∧ (a = 1 - real.sqrt 2 ∨ a = 5 + real.sqrt 10) :=
sorry

end ProblemSolutions

end range_f_when_a_is_1_find_a_for_minimum_value_l555_555552


namespace regular_polygon_perimeter_l555_555447

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555447


namespace min_blackened_squares_l555_555480

theorem min_blackened_squares (grid : Fin 4 × Fin 4 → Bool) :
  (∃ (S : Finset (Fin 4 × Fin 4)), (S.card = 7) ∧
    (∀ (rows : Finset (Fin 4)), (rows.card = 2) → ∀ (cols : Finset (Fin 4)), (cols.card = 2) →
      ¬((S.filter (λ p, p.1 ∉ rows ∧ p.2 ∉ cols)).card = 0))) :=
begin
  sorry
end

end min_blackened_squares_l555_555480


namespace find_a_l555_555557

noncomputable def f (a x : ℝ) := a * (x - 1 / x) - 2 * Real.log x
noncomputable def g (a x : ℝ) := -a / x

theorem find_a (a : ℝ) (h : ∃ x0 ∈ Set.Icc (1:ℝ) Real.exp 1, f a x0 > g a x0) : 0 < a :=
sorry

end find_a_l555_555557


namespace ellipse_formula_area_of_triangle_l555_555930

-- Definitions and conditions
def is_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (p.1^2 / a^2) + (p.2^2 / b^2) = 1

def is_midpoint (p m f : ℝ × ℝ) : Prop :=
  2 * m.1 = p.1 + f.1 ∧ 2 * m.2 = p.2 + f.2

def is_right_angle (p1 p2 f1 f2 : ℝ × ℝ) : Prop :=
  let v1 := (p1.1 - f1.1, p1.2 - f1.2)
  let v2 := (p2.1 - f2.1, p2.2 - f2.2)
  (v1.1 * v2.1 + v1.2 * v2.2) = 0

-- First part: Prove the equation of ellipse
theorem ellipse_formula (a b : ℝ) :
  ∀ F1 F2 Q, is_ellipse a b Q → is_midpoint Q (0, Q.2) F2 → F2 = (√2, 0) → 
  a^2 = 4 → b^2 = 2 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + (y^2 / 2) = 1) := 
by
  intros F1 F2 Q h_ellipse h_midpoint eq_F2 ha2 hb2 x y
  sorry

-- Second part: Prove the area of triangle
theorem area_of_triangle (P F1 F2 : ℝ × ℝ) :
  is_ellipse 2 √2 P → is_right_angle P P (F1, (0, 0)) (F2, (√2, 0)) →
  ∃ (S : ℝ), S = 2 :=
by
  intros h_ellipse h_angle
  use 2
  sorry

end ellipse_formula_area_of_triangle_l555_555930


namespace f_difference_l555_555663

noncomputable def f (n : ℕ) : ℕ :=
  ∑ r in Finset.range (n + 1), (r / 2)

theorem f_difference (m n : ℕ) (h1 : m > n) (h2 : n > 0) : 
  f (m + n) - f (m - n) = m * n :=
  sorry

end f_difference_l555_555663


namespace hyperbola_equation_l555_555849

def is_origin_center (C : Type) : Prop :=
  ∃ (O : C), O = (0,0)

def foci_on_x_axis (C : Type) : Prop :=
  ∃ (F₁ F₂ : C), F₁ = (c, 0) ∧ F₂ = (-c, 0) ∧ c > 0

def asymptote_angle_60 (C : Type) : Prop :=
  ∃ (θ : ℝ), θ = 60 * (π / 180)

def right_focus (C F : Type) : Prop :=
  F = (c, 0) ∧ c > 0

def point_on_hyperbola_first_quadrant (C M : Type) : Prop :=
  ∃ (x y : ℝ), M = (x, y) ∧ x > 0 ∧ y > 0

def midpoint_property (C M F N : Type) : Prop :=
  N = ((fst M + fst F) / 2, (snd M + snd F) / 2)

def magnitude_condition (O N F : (ℝ × ℝ)) : Prop :=
  dist O N = dist N F + 1

theorem hyperbola_equation (C M F N O : Type) [is_origin_center C] [foci_on_x_axis C] [asymptote_angle_60 C] 
  [right_focus C F] [point_on_hyperbola_first_quadrant C M] [midpoint_property C M F N] [magnitude_condition O N F] :
  ∃ (a b : ℝ), (a = 1 ∧ b = √3) ∧ ∀ (x y : ℝ), ((x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2)) = 1 :=
by
  sorry

end hyperbola_equation_l555_555849


namespace algebraic_expression_identity_l555_555592

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l555_555592


namespace num_funcs_satisfy_condition_l555_555509

theorem num_funcs_satisfy_condition : 
∀ (a b c d : ℝ), 
  (∀ x : ℝ, (a * x^3 + b * x^2 + c * x + d) * (-a * x^3 + b * x^2 - c * x + d) = a * x^9 + b * x^6 + c * x^3 + d)
  → (a = 0) ∧ (b = 0) ∧ (c = 0) ∧ (d = 1 ∨ d = -1) → 
  ({if (a = 0) ∧ (b = 0) ∧ (c = 0) then if (d = 1 ∨ d = -1) then 1 else 0 else 0}.size = 2) := 
sorry

end num_funcs_satisfy_condition_l555_555509


namespace find_x_l555_555297

theorem find_x :
  ∃ x : Real, abs (x - 0.052) < 1e-3 ∧
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + x^2 + 0.0035^2) = 100 :=
by
  sorry

end find_x_l555_555297


namespace area_of_set_R_is_1006point5_l555_555646

-- Define the set of points R as described in the problem
def isPointInSetR (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y ≤ 2013 ∧ ⌈x⌉ * ⌊y⌋ = ⌊x⌋ * ⌈y⌉

noncomputable def computeAreaOfSetR : ℝ :=
  1006.5

theorem area_of_set_R_is_1006point5 :
  (∃ x y : ℝ, isPointInSetR x y) → computeAreaOfSetR = 1006.5 := by
  sorry

end area_of_set_R_is_1006point5_l555_555646


namespace find_value_of_a_l555_555533

theorem find_value_of_a : ∀ (a : ℝ), (a > 1) → 
  (∀ t : ℝ, (x = (sqrt 3 / 2) * t + a) ∧ (y = (1 / 2) * t)) → 
  (ρ = 2 * cos θ) :=
  ∃ (x y : ℝ), (x^2 + y^2 = 2 * x) →
  (∃ A B : ℝ, (|PA| * |PB| = 1)) → 
  (a = 1 + sqrt 2) :=
sorry

end find_value_of_a_l555_555533


namespace percent_area_of_triangle_is_correct_l555_555807

theorem percent_area_of_triangle_is_correct (a : ℝ) :
  ( ( (1 / 2) / (Real.sqrt 2 + (1 / 2)) ) * 100 ) ≈ 29.29 :=
by
  sorry

end percent_area_of_triangle_is_correct_l555_555807


namespace cost_of_500_pencils_is_15_dollars_l555_555711

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l555_555711


namespace dot_product_eq_l555_555084

def u : ℝ × ℝ × ℝ := (7, -3, 2)
def v : ℝ × ℝ × ℝ := (-4, 6, -3)

theorem dot_product_eq : (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) = -52 :=
by
  -- Skipping actual proof for now
  sorry

end dot_product_eq_l555_555084


namespace cone_slant_height_l555_555118

open Real

theorem cone_slant_height {r l : ℝ} (h₁ : r = sqrt 2)
  (h₂ : π * r * l = 4 * π * 1^2) : l = 2 * sqrt 2 :=
by {
  rw [h₁, pow_two, mul_one, mul_comm 4 π, mul_assoc] at h₂,
  field_simp [π_ne_zero] at h₂,
  rw [← mul_assoc, mul_comm (sqrt 2) _, ← mul_assoc] at h₂,
  rw [mul_inv_cancel (sqrt_ne_zero 2), mul_comm _ π, mul_inv_cancel π_ne_zero] at h₂,
  simpa using h₂,
}

end cone_slant_height_l555_555118


namespace reflection_addition_l555_555724

theorem reflection_addition (m b : ℝ) :
  (∀ x y : ℝ, (x, y) = (2, 2) → (x', y') = (10, 6) → reflects_across_line (x, y) (x', y') (m, b)) → 
  m + b = 14 :=
by
  sorry

end reflection_addition_l555_555724


namespace angle_AFE_170_l555_555041

noncomputable theory

-- Define the basic structure and points
def square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  angle A B C = 90 ∧ angle B C D = 90 ∧ angle C D A = 90 ∧ angle D A B = 90

-- Define the given conditions
variable (A B C D E F : Point)
variable h1 : square A B C D
variable h2 : on_opposite_half_plane D C E A
variable h3 : angle C D E = 110
variable h4 : collinear F A D
variable h5 : dist D E = dist D F

-- State the final proof declaration
theorem angle_AFE_170 : angle A F E = 170 :=
by 
  sorry

end angle_AFE_170_l555_555041


namespace extreme_value_sum_l555_555153

noncomputable def f (m n x : ℝ) : ℝ := x^3 + 3 * m * x^2 + n * x + m^2

theorem extreme_value_sum (m n : ℝ) (h1 : f m n (-1) = 0) (h2 : (deriv (f m n)) (-1) = 0) : m + n = 11 := 
sorry

end extreme_value_sum_l555_555153


namespace sequences_with_both_properties_are_constant_l555_555070

-- Definitions according to the problem's conditions
def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = seq (n + 2) / seq (n + 1)

-- Definition of the sequence properties combined
def arithmetic_and_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  arithmetic_sequence seq ∧ geometric_sequence seq

-- Problem to prove
theorem sequences_with_both_properties_are_constant (seq : ℕ → ℝ) :
  arithmetic_and_geometric_sequence seq → ∀ n m : ℕ, seq n = seq m :=
sorry

end sequences_with_both_properties_are_constant_l555_555070


namespace exists_points_with_irrational_dist_and_rational_area_l555_555219

theorem exists_points_with_irrational_dist_and_rational_area (n : ℕ) (hn : n ≥ 3) :
  ∃ (S : finset (ℤ × ℤ)), S.card = n ∧
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ S → p2 ∈ S → p1 ≠ p2 → (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 ≠ (some_rational_square : ℚ)^2) ∧
  (∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ S → p2 ∈ S → p3 ∈ S → p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → 
     is_rational (triangle_area p1 p2 p3)) :=
sorry

-- Helper function to determine the area of a triangle given three points in ℤ × ℤ.
def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
  1 / 2 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Helper function to check if a number is rational.
def is_rational (x : ℚ) : Prop := true  -- Placeholder definition, as every ℚ is inherently rational.

end exists_points_with_irrational_dist_and_rational_area_l555_555219


namespace find_a_l555_555875

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l555_555875


namespace sequence_sum_l555_555123

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = (1 / 4) ^ n

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1) * 4 ^ i

theorem sequence_sum (a : ℕ → ℝ) (n : ℕ) (h_seq : sequence a) :
  5 * S_n a n - (4^n) * a n = n :=
sorry

end sequence_sum_l555_555123


namespace angle_bisector_le_median_in_triangle_l555_555690

theorem angle_bisector_le_median_in_triangle 
  (a b c : ℝ) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  let m_a := sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4),
      l_a := sqrt (a * b * ((a + b)^2 - c^2) / ((a + b)^2)) in
  l_a <= m_a :=
by
  sorry

end angle_bisector_le_median_in_triangle_l555_555690


namespace cyclic_B_E_C_K_l555_555239

variable (A B C D E F G H K M : Point) (ω : Circle A B C)

-- Conditions
variables 
  (hD_on_BC : D ∈ LineSegment B C)
  (hE_on_AD : E ∈ LineSegment A D)
  (hF_on_ADω : F ∈ RayAD ∧ F ∈ ω ∧ F ≠ A)
  (hM_bisects_AF : M ∈ ω ∧ M ∈ Midpoint A F ∧ M ∉ Line A C F)
  (hG_on_MEω : G ∈ RayME ∧ G ∈ ω ∧ G ≠ M)
  (hH_on_GDω : H ∈ RayGD ∧ H ∈ ω ∧ H ≠ G)
  (hK_on_MHAD : K ∈ LineMH ∧ K ∈ LineAD)

-- Goal: B, E, C, K are cyclic.
theorem cyclic_B_E_C_K :
  CyclicQueue B E C K := 
  sorry

end cyclic_B_E_C_K_l555_555239


namespace sum_of_remaining_is_product_l555_555752

theorem sum_of_remaining_is_product (x y : ℕ) (h₀ : x ≠ y) (h₁ : x ∈ finset.range 38) (h₂ : y ∈ finset.range 38)
  (h₃ : finset.sum (finset.range 38) id - x - y = x * y) : y - x = 10 := by 
  sorry

end sum_of_remaining_is_product_l555_555752


namespace arithmetic_progression_rth_term_l555_555902

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 2 * n + 3 * n^2

theorem arithmetic_progression_rth_term : (S r) - (S (r - 1)) = 6 * r - 1 :=
by
  sorry

end arithmetic_progression_rth_term_l555_555902


namespace problem_61_a_problem_61_b_problem_62_a_problem_62_b_l555_555786

-- Problem 61 Part a
theorem problem_61_a (P : Type) [polygon P] [convex_shape P] (h : area P = 1) :
  ∃ q : parallelogram, encloses q P ∧ area q = 2 :=
sorry

-- Problem 61 Part b
theorem problem_61_b (T : Type) [triangle T] (h : area T = 1) :
  ¬∃ p : parallelogram, area p < 2 ∧ encloses p T :=
sorry

-- Problem 62 Part a
theorem problem_62_a (P : Type) [polygon P] [convex_shape P] (h : area P = 1) :
  ∃ t : triangle, encloses t P ∧ area t = 2 :=
sorry

-- Problem 62 Part b
theorem problem_62_b (P : Type) [parallelogram P] (h : area P = 1) :
  ¬∃ t : triangle, area t < 2 ∧ encloses t P :=
sorry

end problem_61_a_problem_61_b_problem_62_a_problem_62_b_l555_555786


namespace true_statements_count_l555_555059

theorem true_statements_count :
  let converse := ∀ (T1 T2 : Triangle), (T1 ≠ T2 → area T1 ≠ area T2)
  let negation := ∀ (a b : ℕ), (a * b ≠ 0 → a ≠ 0)
  let contrapositive := ∀ (T : Triangle), (∃ angle, angle ≠ 60 → ¬is_equilateral T)
  (to_bool converse + to_bool negation + to_bool contrapositive) = 2 :=
by
  sorry

end true_statements_count_l555_555059


namespace interval_contains_root_l555_555281

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem interval_contains_root : ∃ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
by
  have f_neg : f (-1) < 0 := by sorry
  have f_zero : f 0 > 0 := by sorry
  sorry

end interval_contains_root_l555_555281


namespace equation_of_latus_rectum_l555_555085

theorem equation_of_latus_rectum (y x : ℝ) : (x = -1/4) ∧ (y^2 = x) ↔ (2 * (1 / 2) = 1) ∧ (l = - (1 / 2) / 2) := sorry

end equation_of_latus_rectum_l555_555085


namespace area_of_triangle_bounded_by_lines_l555_555325

theorem area_of_triangle_bounded_by_lines :
  let line1 (x : ℝ) : ℝ := 3 * x + 6 in
  let line2 (x : ℝ) : ℝ := -2 * x + 12 in
  let intersection_x : ℝ := 6 / 5 in
  let intersection_y : ℝ := line1 intersection_x in
  let y_intercept1 := (0, 6 : ℝ) in
  let y_intercept2 := (0, 12 : ℝ) in
  let base : ℝ := y_intercept2.2 - y_intercept1.2 in
  let height : ℝ := intersection_x in
  (1 / 2) * base * height = 18 / 5 := 
by
  sorry

end area_of_triangle_bounded_by_lines_l555_555325


namespace compute_expression_l555_555058

theorem compute_expression : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end compute_expression_l555_555058


namespace holly_initial_amount_l555_555969

variable (breakfast_lunch_dinner_consumed : ℕ)
variable (end_of_day_ounces : ℕ)

def total_consumed (consumed : ℕ) :=
  3 * consumed

theorem holly_initial_amount :
  ∀ (consumed : ℕ) (end_amount : ℕ),
  total_consumed consumed = breakfast_lunch_dinner_consumed →
  end_amount = end_of_day_ounces →
  (total_consumed consumed + end_amount = 80) :=
by
  intros consumed end_amount
  assume h1 h2
  rw [h1, h2]
  sorry

end holly_initial_amount_l555_555969


namespace quadratic_root_range_l555_555562

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range (a : ℝ) (h : a ≠ 0) (h_distinct_roots : ∃ x1 x2 : ℝ, quadratic_function a x1 = 0 ∧ quadratic_function a x2 = 0 ∧ x1 ≠ x2 ∧ x1 < 1 ∧ x2 > 1) :
    -(2 / 11) < a ∧ a < 0 :=
sorry

end quadratic_root_range_l555_555562


namespace proof_of_problem1_proof_of_problem2_l555_555478

noncomputable def problem1 : Prop := 
  sqrt (25 / 9) - (8 / 27)^(1 / 3) - (Real.pi + Real.exp 1)^0 + (1 / 4)^(-1 / 2) = 2 

noncomputable def problem2 : Prop := 
  ((log 2 2)^2 + (log 2 2) * (log 2 5) + sqrt ((log 2 2)^2 - (log 2 4) + 1) ) ≈ 3.66096

theorem proof_of_problem1 : problem1 :=
sorry

theorem proof_of_problem2 : problem2 :=
sorry

end proof_of_problem1_proof_of_problem2_l555_555478


namespace symmetric_line_x_0_l555_555115

def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem symmetric_line_x_0 (ϕ : ℝ) :
  (∀ x, f (x + ϕ) = f (-x + ϕ)) ↔ ϕ = π / 6 := by
  sorry

end symmetric_line_x_0_l555_555115


namespace find_m_l555_555975

/-- Given the sets {3, 4, m^2 - 3m - 1} and {2m, -3}, show that if their intersection equals {-3}, then m = 1. -/
theorem find_m (m : ℤ) (h : {3, 4, m^2 - 3m - 1} ∩ {2m, -3} = {-3}) : m = 1 :=
sorry

end find_m_l555_555975


namespace cube_surface_area_specific_cube_surface_area_l555_555352

theorem cube_surface_area (e : ℝ) (S : ℝ) (h : e = 8) : S = 6 * e^2 := by
  sorry

theorem specific_cube_surface_area : ∃ S : ℝ, cube_surface_area 8 S 8 = 384 := by
  use 384
  -- Here we would do the local proof, but inserting sorry satisfies the requirement
  sorry

end cube_surface_area_specific_cube_surface_area_l555_555352


namespace angle_AFE_is_170_l555_555043

-- Definitions of the geometric configurations
def square (A B C D : Point) : Prop := 
  (distance A B = distance B C) ∧ (distance B C = distance C D) ∧ 
  (distance C D = distance D A) ∧ (angle A B C = 90) ∧ 
  (angle B C D = 90) ∧ (angle C D A = 90) ∧ (angle D A B = 90)

def isosceles_triangle (D E F : Point) : Prop := distance D E = distance D F

-- Variables for points in the problem
variables (A B C D E F : Point)

-- Hypotheses for the given conditions in the problem
hypothesis (sq : square A B C D)
hypothesis (angle_CDE : angle C D E = 110)
hypothesis (isosceles : isosceles_triangle D E F)

-- The goal statement
theorem angle_AFE_is_170 :
  angle A F E = 170 := 
sorry

end angle_AFE_is_170_l555_555043


namespace A_cap_B_correct_complement_U_A_cap_B_correct_complement_A_cap_B_correct_l555_555962

variable {ℝ : Type} [LinearOrder ℝ] -- If not predefined in Mathlib

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 3

def B (x : ℝ) : Prop := (1 ≤ x) ∨ (x < -1)

def A_cap_B (x : ℝ) : Prop := A x ∧ B x

def complement_U (s : ℝ → Prop) (x : ℝ) : Prop := ¬ s x

theorem A_cap_B_correct :
  A_cap_B = { x : ℝ | (-2 ≤ x ∧ x < -1) ∨ (1 ≤ x ∧ x ≤ 3) } :=
sorry

theorem complement_U_A_cap_B_correct :
  complement_U A_cap_B = { x : ℝ | (x < -2) ∨ (-1 ≤ x ∧ x < 1) ∨ (x > 3) } :=
sorry

def complement_A (x : ℝ) : Prop := complement_U A x

def complement_A_cap_B (x : ℝ) : Prop := complement_A x ∧ B x

theorem complement_A_cap_B_correct :
  complement_A_cap_B = { x : ℝ | (x < -2) ∨ (x > 3) } :=
sorry

end A_cap_B_correct_complement_U_A_cap_B_correct_complement_A_cap_B_correct_l555_555962


namespace approx_values_relationship_l555_555324

theorem approx_values_relationship : 
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a = b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a > b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a < b) :=
by sorry

end approx_values_relationship_l555_555324


namespace remainder_polynomial_division_l555_555069

theorem remainder_polynomial_division :
  ∃ (Q R : Polynomial ℝ), let p := Polynomial.C 1 * Polynomial.X ^ 50 in
  let d := Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X + Polynomial.C 3 in
  let r := Polynomial.C ((3^50 - 1) / 2) * Polynomial.X + Polynomial.C ((3 - 3^50) / 2) in
  p = d * Q + r ∧ r.degree < 2 :=
by
  sorry

end remainder_polynomial_division_l555_555069


namespace fifth_term_expansion_l555_555333

theorem fifth_term_expansion (a x : ℂ) : 
  let expr := (a / Complex.sqrt x) + (x / (a ^ 3)) 
  in let fifth_term := 70 * (x ^ 2) / (a ^ 8) 
  in (expr ^ 8).term (4) = fifth_term := 
by 
  sorry

end fifth_term_expansion_l555_555333


namespace a_5_value_l555_555631

variable (a_n : ℕ → ℝ) (r : ℝ) (a_1: ℝ)
variable (h1: a_4a_6_plus_a_5_squared = 50)
variable (positive_terms : ∀ n, a_n n > 0)

def geometric_sequence_term (a_1 r : ℝ) (n : ℕ) : ℝ :=
  a_1 * r ^ n

noncomputable def a_5 := geometric_sequence_term a_1 r 4

theorem a_5_value : a_5 = 5 :=
by
  have h_geom : geometric_sequence_term a_1 r 3 * geometric_sequence_term a_1 r 5 + (geometric_sequence_term a_1 r 4)^2 = 50:= h1
  sorry

end a_5_value_l555_555631


namespace number_of_factors_correct_l555_555067

def number_of_factors (n : ℕ) : ℕ :=
  let factors := [(2, 5), (3, 4), (5, 3), (7, 2), (11, 1)]
  factors.map (λ p => p.2 + 1).prod

theorem number_of_factors_correct :
  number_of_factors (2^5 * 3^4 * 5^3 * 7^2 * 11^1) = 720 :=
by
  -- Calculation based on the factors list:
  let factors := [(2, 5), (3, 4), (5, 3), (7, 2), (11, 1)]
  let num_factors := factors.map (λ p => p.2 + 1).prod
  show num_factors = 720
  sorry

end number_of_factors_correct_l555_555067


namespace min_factor_difference_of_2025_l555_555973

theorem min_factor_difference_of_2025 : ∃ a b : ℕ, a * b = 2025 ∧ |a - b| = 0 :=
by
  sorry

end min_factor_difference_of_2025_l555_555973


namespace trailing_zeros_30_factorial_l555_555467

theorem trailing_zeros_30_factorial : 
  (nat.factorial 30).trailing_zeros = 7 := 
by
  -- Proof is omitted
  sorry

end trailing_zeros_30_factorial_l555_555467


namespace regular_polygon_perimeter_l555_555372

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555372


namespace angle_bisector_le_median_l555_555686

noncomputable def angle_bisector (A B C : ℝ) (C : ℝ) : ℝ := sorry
noncomputable def median (A B C : ℝ) (C : ℝ) : ℝ := sorry

theorem angle_bisector_le_median (A B C : ℝ) :
  ∀ (CD CM : ℝ), angle_bisector A B C = CD → median A B C = CM → CD ≤ CM :=
by
  intros CD CM hCD hCM
  rw [angle_bisector, median] at hCD hCM
  sorry

end angle_bisector_le_median_l555_555686


namespace find_F_l555_555580

theorem find_F (F C : ℝ) (h1 : C = 30) (h2 : C = (5 / 9) * (F - 30)) : F = 84 := by
  sorry

end find_F_l555_555580


namespace reflection_image_l555_555722

theorem reflection_image (m b : ℝ) :
  (∃ m b, ∀ (P Q : ℝ × ℝ), P = (2, 2) → Q = (10, 6) →
    let x_m : ℝ := (P.1 + Q.1) / 2
        y_m : ℝ := (P.2 + Q.2) / 2
        m := -2
        b := y_m - m * x_m
    in y_m = m * x_m + b) → m + b = 14 :=
by
  sorry

end reflection_image_l555_555722


namespace other_root_of_quadratic_l555_555247

theorem other_root_of_quadratic (p x : ℝ) (h : 7 * x^2 + p * x - 9 = 0) (root1 : x = -3) : 
  x = 3 / 7 :=
by
  sorry

end other_root_of_quadratic_l555_555247


namespace angle_bisector_le_median_l555_555687

noncomputable def angle_bisector (A B C : ℝ) (C : ℝ) : ℝ := sorry
noncomputable def median (A B C : ℝ) (C : ℝ) : ℝ := sorry

theorem angle_bisector_le_median (A B C : ℝ) :
  ∀ (CD CM : ℝ), angle_bisector A B C = CD → median A B C = CM → CD ≤ CM :=
by
  intros CD CM hCD hCM
  rw [angle_bisector, median] at hCD hCM
  sorry

end angle_bisector_le_median_l555_555687


namespace cost_of_500_pencils_is_15_dollars_l555_555710

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l555_555710


namespace max_clique_size_bound_l555_555221

open Finset

variable {V : Type} [Fintype V] (G : SimpleGraph V)

noncomputable def degrees (v : V) : ℕ :=
  G.degree v

def max_clique_size (G : SimpleGraph V) : ℕ :=
  ∃ S : Finset V, G.IsClique S ∧ S.card = clique_size

theorem max_clique_size_bound (G : SimpleGraph V) :
  max_clique_size G ≥ ∑ v in Finset.univ, (1 / (Fintype.card V - ∑ v, degrees v)) :=
sorry

end max_clique_size_bound_l555_555221


namespace min_value_a_b_c_l555_555514

def A_n (a : ℕ) (n : ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (b : ℕ) (n : ℕ) : ℕ := b * ((10^n - 1) / 9)
def C_n (c : ℕ) (n : ℕ) : ℕ := c * ((10^(2*n) - 1) / 9)

theorem min_value_a_b_c (a b c : ℕ) (Ha : 0 < a ∧ a < 10) (Hb : 0 < b ∧ b < 10) (Hc : 0 < c ∧ c < 10) :
  (∃ n1 n2 : ℕ, (n1 ≠ n2) ∧ (C_n c n1 - A_n a n1 = B_n b n1 ^ 2) ∧ (C_n c n2 - A_n a n2 = B_n b n2 ^ 2)) →
  a + b + c = 5 :=
by
  sorry

end min_value_a_b_c_l555_555514


namespace necessary_and_sufficient_condition_l555_555112

def M (a : ℝ) : set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}
def N (a : ℝ) : set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  (M a ∩ N a = N a) ↔ a ≥ 5 / 4 :=
by sorry

end necessary_and_sufficient_condition_l555_555112


namespace basketball_points_distinct_numbers_l555_555793

/-- A basketball player made 7 baskets during a game. Each basket was worth either 2 or 3 points.
Prove that the number of different numbers that could represent the total points scored by the
player is 8. -/
theorem basketball_points_distinct_numbers : 
  ∃ S : Finset ℕ, (∀ P ∈ S, ∃ x : ℕ, x ≤ 7 ∧ P = 3 * x + 2 * (7 - x)) ∧ S.card = 8 :=
by 
  sorry

end basketball_points_distinct_numbers_l555_555793


namespace range_of_m_l555_555143

noncomputable def circleC : set (ℝ × ℝ) :=
  {p | (p.1 - 4)^2 + (p.2 - 3)^2 = 4}

def pointA (m : ℝ) (h_m : m > 0) : ℝ × ℝ :=
  (-m, 0)

def pointB (m : ℝ) (h_m : m > 0) : ℝ × ℝ :=
  (m, 0)

def angle90 (A B P : ℝ × ℝ) : Prop :=
  (P.1 + A.1) * (P.1 + B.1) + P.2^2 = 0

theorem range_of_m (m : ℝ) (h_m : m > 0) :
  (∃ P : ℝ × ℝ, P ∈ circleC ∧ angle90 (pointA m h_m) (pointB m h_m) P) →
  3 ≤ m ∧ m ≤ 7 :=
sorry

end range_of_m_l555_555143


namespace opinion_change_difference_l555_555047

variables (initial_enjoy final_enjoy initial_not_enjoy final_not_enjoy : ℕ)
variables (n : ℕ) -- number of students in the class

-- Given conditions
def initial_conditions :=
  initial_enjoy = 40 * n / 100 ∧ initial_not_enjoy = 60 * n / 100

def final_conditions :=
  final_enjoy = 80 * n / 100 ∧ final_not_enjoy = 20 * n / 100

-- The theorem to prove
theorem opinion_change_difference :
  initial_conditions n initial_enjoy initial_not_enjoy →
  final_conditions n final_enjoy final_not_enjoy →
  (40 ≤ initial_enjoy + 20 ∧ 40 ≤ initial_not_enjoy + 20 ∧
  max_change = 60 ∧ min_change = 40 → max_change - min_change = 20) := 
  sorry

end opinion_change_difference_l555_555047


namespace ben_eggs_remaining_l555_555048

def initial_eggs : ℕ := 75

def ben_day1_morning : ℝ := 5
def ben_day1_afternoon : ℝ := 4.5
def alice_day1_morning : ℝ := 3.5
def alice_day1_evening : ℝ := 4

def ben_day2_morning : ℝ := 7
def ben_day2_evening : ℝ := 3
def alice_day2_morning : ℝ := 2
def alice_day2_afternoon : ℝ := 4.5
def alice_day2_evening : ℝ := 1.5

def ben_day3_morning : ℝ := 4
def ben_day3_afternoon : ℝ := 3.5
def alice_day3_evening : ℝ := 6.5

def total_eggs_eaten : ℝ :=
  (ben_day1_morning + ben_day1_afternoon + alice_day1_morning + alice_day1_evening) +
  (ben_day2_morning + ben_day2_evening + alice_day2_morning + alice_day2_afternoon + alice_day2_evening) +
  (ben_day3_morning + ben_day3_afternoon + alice_day3_evening)

def remaining_eggs : ℝ :=
  initial_eggs - total_eggs_eaten

theorem ben_eggs_remaining : remaining_eggs = 26 := by
  -- proof goes here
  sorry

end ben_eggs_remaining_l555_555048


namespace triangle_ratio_a_divides_triangle_ratio_b_divides_l555_555782

theorem triangle_ratio_a_divides (BC CA AB : ℝ) (X Y Z : ℝ) (h1 : BC/3 = X) (h2 : CA/3 = Y) (h3 : AB/3 = Z)
  (h4 : area (triangle AZY) / area (triangle ABC) = 2 / a) 
  : a = 9 :=
sorry

theorem triangle_ratio_b_divides (BC CA AB : ℝ) (X Y Z : ℝ) (h1 : BC/3 = X) (h2 : CA/3 = Y) (h3 : AB/3 = Z)
  (h5 : area (triangle AZY) / area (triangle XYZ) = 2 / b) 
  : b = 3 :=
sorry

end triangle_ratio_a_divides_triangle_ratio_b_divides_l555_555782


namespace two_cos_45_eq_sqrt_two_l555_555295

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l555_555295


namespace partial_fraction_sum_zero_l555_555050

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l555_555050


namespace mango_rate_l555_555025

theorem mango_rate (x : ℕ) : 
  (sells_rate : ℕ) = 3 → 
  (profit_percent : ℕ) = 50 → 
  (buying_price : ℚ) = 2 := by
  sorry

end mango_rate_l555_555025


namespace magnitude_of_inverse_complex_l555_555784

theorem magnitude_of_inverse_complex : abs(1 / (1 - 2 * Complex.I)) = (Real.sqrt 5) / 5 := by
  sorry

end magnitude_of_inverse_complex_l555_555784


namespace find_roots_l555_555512

noncomputable def has_roots : Prop :=
  ∃ (z : ℂ), (z = 3 - complex.i ∨ z = -2 + complex.i) ∧ z^2 - z = 5 - 5 * complex.i

theorem find_roots : has_roots :=
  sorry

end find_roots_l555_555512


namespace quadrilateral_intersection_analysis_l555_555748

-- Defines the problem
def problem : Type := 
  Σ (circle : Type) (points : set circle), points.card = 13 ∧ 
    (∀ (p₁ p₂ p₃ p₄ : circle), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ 
        p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ → ¬(∃ (c : circle), 
        c ∈ points ∧ c = p₁ ∧ c = p₂ ∧ c = p₃ ∧ c = p₄))

-- States the theorem
theorem quadrilateral_intersection_analysis :=
  ∀ (c : problem), 
  ∃ (requires_further_analysis : Prop), requires_further_analysis :=
  sorry

end quadrilateral_intersection_analysis_l555_555748


namespace regular_polygon_perimeter_is_28_l555_555401

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555401


namespace golden_section_DC_length_l555_555012

theorem golden_section_DC_length :
  let A : ℝ := 0
  let B : ℝ := 1
  let C : ℝ := B - (sqrt 5 - 1) / 2
  let D : ℝ := A + (sqrt 5 - 1) / 2
  B - A = 1 →
  C = B - (sqrt 5 - 1) / 2 →
  D = A + (sqrt 5 - 1) / 2 →
  D < C →
  abs(C - D) = 1/2 :=
by
  sorry

end golden_section_DC_length_l555_555012


namespace arithmetic_sequence_problem_l555_555191

variable {a : ℕ → ℝ} {d : ℝ} -- Declare the sequence and common difference

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 5 + a 10 = 12 ∧ arithmetic_sequence a d

-- Main theorem statement
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h : given_conditions a d) :
  3 * a 7 + a 9 = 24 :=
sorry

end arithmetic_sequence_problem_l555_555191


namespace find_smaller_number_l555_555742

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l555_555742


namespace shoebox_height_l555_555266

theorem shoebox_height (h : ℕ) : 6 * h = 24 → h = 4 := by
  intro h_eq
  exact Nat.eq_of_mul_eq_mul_right (by norm_num) h_eq

end shoebox_height_l555_555266


namespace binary_to_decimal_101101_l555_555856

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end binary_to_decimal_101101_l555_555856


namespace avg_of_7_consecutive_integers_l555_555103

theorem avg_of_7_consecutive_integers (a b : ℕ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 5 := 
  sorry

end avg_of_7_consecutive_integers_l555_555103


namespace clock_strikes_six_times_in_thirty_seconds_l555_555744

noncomputable def interval_between_strikes (strikes : ℕ) (total_time : ℕ) : ℕ :=
  (total_time / (strikes - 1))

noncomputable def time_for_six_strikes (interval : ℕ) : ℕ :=
  5 * interval

theorem clock_strikes_six_times_in_thirty_seconds :
  interval_between_strikes 3 12 = 6 →
  time_for_six_strikes 6 = 30 :=
by
  intros h1
  rw interval_between_strikes at h1
  simp at h1
  rw time_for_six_strikes
  exact h1
  sorry

end clock_strikes_six_times_in_thirty_seconds_l555_555744


namespace length_of_faster_train_l555_555788

theorem length_of_faster_train (speed_faster_train : ℝ) (speed_slower_train : ℝ) (elapsed_time : ℝ) (relative_speed : ℝ) (length_train : ℝ)
  (h1 : speed_faster_train = 50) 
  (h2 : speed_slower_train = 32) 
  (h3 : elapsed_time = 15) 
  (h4 : relative_speed = (speed_faster_train - speed_slower_train) * (1000 / 3600)) 
  (h5 : length_train = relative_speed * elapsed_time) :
  length_train = 75 :=
sorry

end length_of_faster_train_l555_555788


namespace part1_equation_part2_equation_l555_555189

-- Define the slope and the point through which the line passes for part 1
def slope (l : ℝ → ℝ) := ∀ x y, l y = 2 * x
def point_A (A : ℝ × ℝ) := A = (-1, 3)

-- Define the slope and the intercept condition for part 2
def intercept_condition (l : ℝ → ℝ) := ∀ b, (l 0 = b ∧ l (-b / 2) = 0) → b - (b / 2) = 4

-- Part 1: Prove Equation of Line
theorem part1_equation (l : ℝ → ℝ) : (slope l) → (point_A A) → ∀ x y, (2 * x - y + 5 = 0) :=
by sorry

-- Part 2: Prove Equation of Line with intercept condition
theorem part2_equation (l : ℝ → ℝ) : (slope l) → (intercept_condition l) → ∀ x y, (2 * x - y + 8 = 0) :=
by sorry

end part1_equation_part2_equation_l555_555189


namespace centroid_coor_l555_555311

noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def P : ℝ × ℝ := (7, 5)
def Q : ℝ × ℝ := (1, -3)
def R : ℝ × ℝ := (4, 4)
def S : ℝ × ℝ := centroid P Q R

theorem centroid_coor {x y : ℝ} (hS : S = (x, y)) : 12 * x + 3 * y = 54 :=
by
  -- The proof would normally go here
  sorry

end centroid_coor_l555_555311


namespace fraction_twins_l555_555003

variables (P₀ I E P_f f : ℕ) (x : ℚ)

def initial_population := P₀ = 300000
def immigrants := I = 50000
def emigrants := E = 30000
def pregnant_fraction := f = 1 / 8
def final_population := P_f = 370000

theorem fraction_twins :
  initial_population P₀ ∧ immigrants I ∧ emigrants E ∧ pregnant_fraction f ∧ final_population P_f →
  x = 1 / 4 :=
by
  sorry

end fraction_twins_l555_555003


namespace minimum_ab_collinear_l555_555584

theorem minimum_ab_collinear (a b : ℝ) (h1 : a * b > 0) (h2 : collinear ℝ ![(a, 0), (0, b), (-2, -2)]) : a * b ≥ 16 :=
sorry

end minimum_ab_collinear_l555_555584


namespace sum_of_all_possible_values_of_d_l555_555080

def d_is_digit (d : ℕ) : Prop := d < 10
def e_is_even_digit (e : ℕ) : Prop := e < 10 ∧ (e % 2 = 0)
def divisibility_by_3 (d e : ℕ) : Prop := (12 + d + e) % 3 = 0
def divisibility_by_11 (d e : ℕ) : Prop := d + e = 10

theorem sum_of_all_possible_values_of_d : 
  (∑ d in {d | ∃ e, e_is_even_digit e ∧ divisibility_by_11 d e ∧ divisibility_by_3 d e}.to_finset, d) = 18 :=
sorry

end sum_of_all_possible_values_of_d_l555_555080


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555313

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555313


namespace five_minus_a_l555_555579

theorem five_minus_a (a b : ℚ) (h1 : 5 + a = 3 - b) (h2 : 3 + b = 8 + a) : 5 - a = 17/2 :=
by
  sorry

end five_minus_a_l555_555579


namespace perimeter_of_regular_polygon_l555_555407

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555407


namespace cos_theta_value_l555_555907

theorem cos_theta_value (θ : ℝ) (h : sin θ = 1 / 3) (h₁ : θ > π / 2) (h₂ : θ < π) : cos θ = - (2 * real.sqrt 2) / 3 :=
sorry

end cos_theta_value_l555_555907


namespace perimeter_of_regular_polygon_l555_555416

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555416


namespace number_of_divisors_l555_555488

def positive_divisors (n : ℕ) : ℕ := sorry

variables (a n : ℕ)

theorem number_of_divisors (h1 : a > 1) (h2 : n > 0) (h3 : prime (a ^ n + 1)) : positive_divisors (a ^ n - 1) ≥ n :=
sorry

end number_of_divisors_l555_555488


namespace polygon_perimeter_l555_555378

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555378


namespace intersecting_lines_l555_555726

-- Definitions for the conditions
def line1 (x y a : ℝ) : Prop := x = (1/3) * y + a
def line2 (x y b : ℝ) : Prop := y = (1/3) * x + b

-- The theorem we need to prove
theorem intersecting_lines (a b : ℝ) (h1 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line1 x y a) 
                           (h2 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line2 x y b) : 
  a + b = 10 / 3 :=
sorry

end intersecting_lines_l555_555726


namespace solve_equation_l555_555699

theorem solve_equation 
: ∃ x1 x2 : ℝ, (x1 - 3)^6 + (x1 - 5)^6 = 72 ∧ x1 = 4 + real.sqrt(real.cbrt(50) - 5) ∧ (x2 - 3)^6 + (x2 - 5)^6 = 72 ∧ x2 = 4 - real.sqrt(real.cbrt(50) - 5) :=
by
  sorry

end solve_equation_l555_555699


namespace sum_of_inverses_of_lengths_l555_555955

theorem sum_of_inverses_of_lengths {a p q : ℝ} (h_pos_a : a > 0)
  (h_parabola : ∀ (x y : ℝ), y = ax^2 → ∃ F, F = (0, 1 / (4 * a)))
  (line_intersects : ∃ F P Q, line_through F ∧ line_intersects_parabola ax^2 P Q
  ∧ (length_segment F P = p) ∧ (length_segment F Q = q)) :
  1 / p + 1 / q = 4 * a :=
sorry

end sum_of_inverses_of_lengths_l555_555955


namespace greatest_integer_a_l555_555328

theorem greatest_integer_a (a : ℤ) : a * a < 44 → a ≤ 6 :=
by
  intros h
  sorry

end greatest_integer_a_l555_555328


namespace proof_angle_C_proof_side_c_l555_555565

noncomputable def angle_C (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (h1 : m = (Real.sin A, Real.sin B))
  (h2 : n = (Real.cos B, Real.cos A))
  (h3 : (m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C)) :
  ℝ :=
  if h : Real.cos C = 1 / 2 then 60 else 0

theorem proof_angle_C (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin C = Real.sin A + Real.sin B)
  (h2 : ∀ (CA AB AC : ℝ × ℝ), CA • (AB - AC) = 18) :
  angle_C A B C a b c (Real.sin A, Real.sin B) (Real.cos B, Real.cos A) ((Real.sin A) * (Real.cos B) + (Real.sin B) * (Real.cos A) = Real.sin (2 * C)) = 60 :=
sorry

noncomputable def side_c (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * Real.sin C = Real.sin A + Real.sin B)
  (h2 : (∀ (CA AB AC: ℝ × ℝ), CA • (AB - AC) = 18))
  (h3 : a * b = 36) : ℝ :=
  if h : a + b = 2 * c then 6 else 0

theorem proof_side_c (A B C a b c : ℝ)
  (h1 : 2 * Real.sin C = Real.sin A + Real.sin B)
  (h2 : ∀ (CA AB AC : ℝ × ℝ), CA • (AB - AC) = 18) 
  (h3 : a * b = 36) :
  side_c A B C a b c h1 h2 h3 = 6 :=
sorry

end proof_angle_C_proof_side_c_l555_555565


namespace find_a_l555_555888

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l555_555888


namespace subset_condition_l555_555669

variable {U : Type}
variables (P Q : Set U)

theorem subset_condition (h : P ∩ Q = P) : ∀ x : U, x ∉ Q → x ∉ P :=
by {
  sorry
}

end subset_condition_l555_555669


namespace series_sum_eq_neg50_l555_555853

theorem series_sum_eq_neg50 :
  let seq := List.range' 2 101 in
  let series := seq.enum.filter (λ (n : ℕ × ℕ), n.1 % 2 = 0).map (λ (n : ℕ × ℕ), n.2) --
                seq.enum.filter (λ (n : ℕ × ℕ), n.1 % 2 = 1).map (λ (n : ℕ × ℕ), n.2) in
  series.sum = -50 := by
  sorry

end series_sum_eq_neg50_l555_555853


namespace find_first_discount_percentage_l555_555639

theorem find_first_discount_percentage
  (final_price original_price : ℝ)
  (additional_discount : ℝ)
  (h_final : final_price = 17)
  (h_original : original_price = 30.22)
  (h_additional : additional_discount = 0.25) :
  ∃ x : ℝ, (17 / (30.22 * (1 - x / 100) * 0.75)) = 1 ∧ x ≈ 24.97 :=
by
  sorry

end find_first_discount_percentage_l555_555639


namespace problem_l555_555539

variables {f : ℝ → ℝ}
variables {x x1 x2 : ℝ}

-- Definition of an even function
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = f(x)

-- Definition of an increasing function on positive numbers
def increasing_on_positive (f : ℝ → ℝ) := ∀ ⦃x1 x2 : ℝ⦄, 0 < x1 → x1 < x2 → f(x1) < f(x2)

-- Definitions of the conditions in Lean
axiom h1 : even_function f
axiom h2 : ∀ x : ℝ, x ∈ ℝ
axiom h3 : increasing_on_positive f
axiom h4 : x1 < 0
axiom h5 : x2 > 0
axiom h6 : |x1| < |x2|

-- Proof that f(-x1) < f(-x2) given the conditions
theorem problem : f(-x1) < f(-x2) :=
by sorry

end problem_l555_555539


namespace max_positive_integers_on_circle_l555_555523

theorem max_positive_integers_on_circle (a : ℕ → ℕ) (h: ∀ k : ℕ, 2 < k → a k > a (k-1) + a (k-2)) :
  ∃ n : ℕ, (∀ i < 2018, a i > 0 -> n ≤ 1009) :=
  sorry

end max_positive_integers_on_circle_l555_555523


namespace line_passes_through_P_and_minimum_value_of_PM_PN_l555_555158

theorem line_passes_through_P_and_minimum_value_of_PM_PN (m : ℝ) :
  let l := λ (x y : ℝ), (3 * m + 1) * x + (2 + 2 * m) * y - 8 = 0,
      l1 := λ (x : ℝ), x = -1,
      l2 := λ (y : ℝ), y = -1,
      P := (-4, 6),
      distance := λ (A B : ℝ × ℝ), (A.1 - B.1)^2 + (A.2 - B.2)^2,
      point_M := (-1, (3 * m + 6)),
      point_N := (-7/m - 4, -1) in
  (l (-4) 6) ∧ (distance P point_M) * (distance P point_N) ≥ 42 := 
begin
  sorry
end

end line_passes_through_P_and_minimum_value_of_PM_PN_l555_555158


namespace sum_of_divisors_form_l555_555291

theorem sum_of_divisors_form (i j k : ℕ) (h : (finset.Ico 0 (i+1)).sum (λ n, 2^n) * 
    (finset.Ico 0 (j+1)).sum (λ n, 3^n) * (finset.Ico 0 (k+1)).sum (λ n, 5^n) = 1200) : i + j + k = 7 :=
by abstract
  sorry

end sum_of_divisors_form_l555_555291


namespace sufficient_condition_l555_555910

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

noncomputable def log_base_a (a x : ℝ) : ℝ := log x / log a

theorem sufficient_condition 
  {a : ℝ} 
  (h1 : 0 < a ∧ a < 1) :
  ∃ x : ℝ, 3^x + a - 1 = 0 :=
by
  sorry

end sufficient_condition_l555_555910


namespace unique_false_inequality_l555_555521

theorem unique_false_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) : 
  xz > yz → z > 0 :=
by {
  sorry
}

end unique_false_inequality_l555_555521


namespace third_day_water_collected_l555_555822

theorem third_day_water_collected:
  ∀ (total_capacity : ℕ) (initial_fraction : ℚ) 
  (first_day_collected : ℕ) (second_day_extra : ℕ),
  total_capacity = 100 →
  initial_fraction = 2/5 →
  first_day_collected = 15 →
  second_day_extra = 5 →
  let initial_amount := (initial_fraction * total_capacity).toNat in
  let after_first_day := initial_amount + first_day_collected in
  let second_day_collected := first_day_collected + second_day_extra in
  let after_second_day := after_first_day + second_day_collected in
  total_capacity - after_second_day = 25 :=
begin
  sorry
end

end third_day_water_collected_l555_555822


namespace ball_prob_ratio_l555_555496

   noncomputable def p : ℝ :=
     let N := (24).choose(24); 
     let numerator := (6).choose(2) * 13;
     numerator / N

   noncomputable def q : ℝ :=
     let numerator := (24).choose(4) * (20).choose(4) * (16).choose(4) * (12).choose(4) * (8).choose(4) * (4).choose(4); 
     let N := (24).choose(24);
     numerator / N

   theorem ball_prob_ratio : p / q = 12 := by
     sorry
   
end ball_prob_ratio_l555_555496


namespace polynomial_evaluation_l555_555114

noncomputable def a : ℝ := Real.sqrt 17 - 1

theorem polynomial_evaluation :
  let p := a^5 + 2 * a^4 - 17 * a^3 - a^2 + 18 * a - 17
  in p = -1 :=
by
  let A := Real.sqrt 17 - 1
  have h : A ^ 2 + 2 * A - 16 = 0 := by
    calc
      A ^ 2 + 2 * A
        = (Real.sqrt 17 - 1) ^ 2 + 2 * (Real.sqrt 17 - 1) : by rfl
    ... = 17 - 2 * Real.sqrt 17 + 1 + 2 * (Real.sqrt 17 - 1) : by ring
    ... = 17 + 0 - 16 : by ring
  show a^5 + 2*a^4 - 17*a^3 - a^2 + 18*a - 17 = -1
  sorry

end polynomial_evaluation_l555_555114


namespace bisect_CD_l555_555660

structure Pentagon (α : Type*) [EuclideanGeometry α] :=
  (A B C D E : α)
  (convex : ConvexPentagon A B C D E)
  (angle_cond_1 : ∠ BAC = ∠ CAD)
  (angle_cond_2 : ∠ CAD = ∠ DAE)
  (angle_cond_3 : ∠ ABC = ∠ ACD)
  (angle_cond_4 : ∠ ACD = ∠ ADE)

theorem bisect_CD (α : Type*) [EuclideanGeometry α] (pent : Pentagon α) :
  let P : α := intersectionOfDiagonals pent.B pent.D pent.C pent.E in
  let M : α := intersectionOfLineAndSegment pent.A P pent.C pent.D in
  distance pent.C M = distance M pent.D :=
begin
  -- proof goes here
  sorry
end

end bisect_CD_l555_555660


namespace regular_polygon_perimeter_is_28_l555_555396

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555396


namespace verify_triangle_statements_l555_555206

noncomputable def triangle_statements : Prop :=
  let π := Real.pi in
  ∀ (A B C : ℝ) (a b c : ℝ),
  (a = 1 ∧ b = 2 ∧ A = π / 6) →
  (a / Real.cos A = b / Real.sin B → A = π / 4) ∧
  (Real.sin (2 * A) = Real.sin (2 * B) → (A = B ∨ A + B = π / 2 → false)) ∧
  (∀ {x : ℝ}, x = Real.sin A → x + (Real.sin B) > Real.cos A + (Real.cos B)) →
  (π / 6 = Real.pi / 6)

theorem verify_triangle_statements : triangle_statements :=
  sorry

end verify_triangle_statements_l555_555206


namespace oranges_packed_in_a_week_l555_555803

open Nat

def oranges_per_box : Nat := 15
def boxes_per_day : Nat := 2150
def days_per_week : Nat := 7

theorem oranges_packed_in_a_week : oranges_per_box * boxes_per_day * days_per_week = 225750 :=
  sorry

end oranges_packed_in_a_week_l555_555803


namespace balls_in_boxes_l555_555578

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l555_555578


namespace matches_in_eighth_matchbox_l555_555747

def matchbox_indices : Type := Fin 27

def matches (i : matchbox_indices) : ℕ := sorry  -- representing the number of matches in the i-th matchbox

def sum_in_four_consecutive_boxes (i : matchbox_indices) : Prop :=
  matches i + matches ⟨i.1 + 1, sorry⟩ + matches ⟨i.1 + 2, sorry⟩ + matches ⟨i.1 + 3, sorry⟩ = 25

def total_sum_is_165 : Prop :=
  (Finset.univ.sum matches) = 165

theorem matches_in_eighth_matchbox :
  (∀ i:Fin (24), sum_in_four_consecutive_boxes i) → total_sum_is_165 → matches ⟨7, sorry⟩ = 10 := 
by
  intros
  sorry

end matches_in_eighth_matchbox_l555_555747


namespace factor_adjustment_l555_555287

theorem factor_adjustment (a b : ℝ) (h : a * b = 65.08) : a / 100 * (100 * b) = 65.08 :=
by
  sorry

end factor_adjustment_l555_555287


namespace sum_x_coordinates_Q3_equals_512_l555_555811

noncomputable def Q1_sum_x_coordinates := 512

theorem sum_x_coordinates_Q3_equals_512 
    (Q1 : Type) 
    [HasVertices Q1] 
    [Regular Q1] 
    [HasMidpoints Q1]
    (sum_x_coords_Q1 : SumXCoords Q1 = Q1_sum_x_coordinates) 
    (Q2 : Octagon := midpoints Q1)
    (Q3 : Octagon := midpoints Q2) :
  SumXCoords Q3 = 512 :=
sorry

end sum_x_coordinates_Q3_equals_512_l555_555811


namespace original_speed_of_car_l555_555351

theorem original_speed_of_car (d t1 t2 v2 v1 : ℝ) 
  (ht1 : t1 = 12) 
  (ht2 : t2 = 4) 
  (hv2 : v2 = 30) 
  (dist_eq : d = v2 * t2)
  : v1 = d / t1 := 
sorry

# The problem states the initial time is 12 hours, the new time is 4 hours, and the new speed required to cover the distance is 30 mph.
# We include a condition that the original speed is the distance (calculated from new speed and time) divided by the initial time.

end original_speed_of_car_l555_555351


namespace minimum_distance_between_curve_and_line_l555_555858

def mapping_x (n : ℝ) (hn : 0 ≤ n) : ℝ := -Real.sqrt n
def mapping_y (m : ℝ) (hm : 0 ≤ m) : ℝ := -Real.sqrt m / 2

def segment_condition (m n : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n) : Prop := m + n = 4

def ellipse_condition (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def minimum_distance (E : set (ℝ × ℝ)) (L : set (ℝ × ℝ)) : ℝ := 
  Inf { d | ∃ (p ∈ E) (q ∈ L), dist p q = d }

theorem minimum_distance_between_curve_and_line :
  let curve_E := {p : ℝ × ℝ | ∃ (m n : ℝ), segment_condition m n 0 0
                  ∧ p.1 = mapping_x n 0 ∧ p.2 = mapping_y m 0 ∧ ellipse_condition p.1 p.2}
  let line_AB := {p : ℝ × ℝ | segment_condition p.1 p.2 0 0}
  minimum_distance curve_E line_AB = 5 * Real.sqrt 2 / 2 :=
sorry

end minimum_distance_between_curve_and_line_l555_555858


namespace largest_constant_ineq_l555_555094

theorem largest_constant_ineq (a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
    sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e))
    + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e))
    + sqrt (e / (a + b + c + d)) > 2 :=
sorry

end largest_constant_ineq_l555_555094


namespace quadratic_solution1_quadratic_solution2_l555_555264

theorem quadratic_solution1 (x : ℝ) :
  (x^2 + 4 * x - 4 = 0) ↔ (x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) :=
by sorry

theorem quadratic_solution2 (x : ℝ) :
  ((x - 1)^2 = 2 * (x - 1)) ↔ (x = 1 ∨ x = 3) :=
by sorry

end quadratic_solution1_quadratic_solution2_l555_555264


namespace coloring_count_l555_555298

-- Definition of colors
inductive Color
| red | white | blue

-- Define the figure: three connected equilateral triangles and an additional connecting segment
structure Figure :=
  (dots : Fin 9 → Color)
  (condition_1 : ∀ (i : Fin 9), Color) -- condition 1: each triangle is colored
  (condition_2 : dots 3 ≠ dots 6)      -- condition 2: additional segment connecting corresponding vertices

-- No two dots connected by a segment may be the same color
def valid_coloring (f : Figure) : Prop :=
  ∀ (i j : Fin 9), 
    -- Assume adj(i,j) indicates adjacency in the figure
    adj(i,j) → f.dots i ≠ f.dots j

-- The coloring count theorem
theorem coloring_count : ∃ n : ℕ, n = 18 ∧ (Figure 18 valid_coloring)
  sorry

end coloring_count_l555_555298


namespace volume_increase_factor_l555_555998

-- Defining the initial volume of the cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Defining the modified height and radius
def new_height (h : ℝ) : ℝ := 3 * h
def new_radius (r : ℝ) : ℝ := 2.5 * r

-- Calculating the new volume with the modified dimensions
def new_volume (r h : ℝ) : ℝ := volume (new_radius r) (new_height h)

-- Proof statement to verify the volume factor
theorem volume_increase_factor (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  new_volume r h = 18.75 * volume r h :=
by
  sorry

end volume_increase_factor_l555_555998


namespace simplify_and_evaluate_l555_555696

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = -1) (h2 : b = 1) :
  (4/5 * a * b - (2 * a * b^2 - 4 * (-1/5 * a * b + 3 * a^2 * b)) + 2 * a * b^2) = 12 :=
by
  have ha : a = -1 := h1
  have hb : b = 1 := h2
  sorry

end simplify_and_evaluate_l555_555696


namespace range_of_m_l555_555952

def f (x : ℝ) : ℝ := x^2 - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 3 ≤ x → (f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m)) → 
  m ≤ -Real.sqrt(2) / 2 ∨ m ≥ Real.sqrt(2) / 2 :=
by
  sorry

end range_of_m_l555_555952


namespace perimeter_of_polygon_l555_555439

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555439


namespace cost_of_500_pencils_in_dollars_l555_555713

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l555_555713


namespace perimeter_of_polygon_l555_555433

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555433


namespace interest_difference_correct_l555_555778

noncomputable def principal : ℝ := 1000
noncomputable def rate : ℝ := 0.10
noncomputable def time : ℝ := 4

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

noncomputable def interest_difference (P r t : ℝ) : ℝ := 
  compound_interest P r t - simple_interest P r t

theorem interest_difference_correct :
  interest_difference principal rate time = 64.10 :=
by
  sorry

end interest_difference_correct_l555_555778


namespace count_possible_k_l555_555516

theorem count_possible_k : 
  let k_factors := λ k : Nat, ∃ (a b : Nat), k = 2^a * 3^b in
  let lcm_991616 := Nat.lcm (9^9) (16^16) in
  let target_lcm := 18^18 in
  let valid_k := λ k : Nat, k_factors k ∧ Nat.lcm lcm_991616 k = target_lcm in
  (∃ (k_values : Finset Nat), 
    ∀ k ∈ k_values, valid_k k ∧ 
    k_values.card = 19) :=
by
  sorry

end count_possible_k_l555_555516


namespace max_area_rectangle_is_square_l555_555682

theorem max_area_rectangle_is_square (d : ℝ) : ∀ (R : Type) [rect R], (∀ (r : R), diagonal r = d) → 
(show ∃ (r : R), (∀ (s : R), area s ≤ area r) ∧ is_square r) :=
begin
  sorry
end

end max_area_rectangle_is_square_l555_555682


namespace number_of_subsets_sum_13_l555_555567

theorem number_of_subsets_sum_13 : 
  let s := {1, 2, 3, ..., 12}
  (∑ k in Finset.range 6, 2^(2 * k)) = 1365 := by
  let s := {1, 2, 3, ..., 12}
  have h1: 1 ∈ s, from Finset.mem_insert_self _
  have h2: 12 ∈ s, from Finset.mem_insert_of_mem h1
  sorry

end number_of_subsets_sum_13_l555_555567


namespace trip_length_l555_555353

theorem trip_length 
  (total_time : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (hike_time_eq : hike_distance / hike_speed = 5.4) 
  (canoe_time_eq : total_time - hike_distance / hike_speed = 0.1)
  (canoe_distance_eq : canoe_speed * (total_time - hike_distance / hike_speed) = 1.2)
  (total_time_val : total_time = 5.5)
  (canoe_speed_val : canoe_speed = 12)
  (hike_speed_val : hike_speed = 5)
  (hike_distance_val : hike_distance = 27) :
  total_time = 5.5 → canoe_speed = 12 → hike_speed = 5 → hike_distance = 27 → hike_distance + canoe_speed * (total_time - hike_distance / hike_speed) = 28.2 := 
by
  intro h_total_time h_canoe_speed h_hike_speed h_hike_distance
  rw [h_total_time, h_canoe_speed, h_hike_speed, h_hike_distance]
  sorry

end trip_length_l555_555353


namespace division_remainder_l555_555102

def p (x : ℝ) := x^5 + 2 * x^3 - x + 4
def a : ℝ := 2
def remainder : ℝ := 50

theorem division_remainder :
  p a = remainder :=
sorry

end division_remainder_l555_555102


namespace tangent_eq_range_a_for_zeros_sum_of_zeros_l555_555947

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - (a + 1) * x^2 - 2 * a * x + 1

theorem tangent_eq (a x : ℝ) (h : a = 1) : 
  let f_x := f x 1 in
  let f_prime := (2/x - 4*x - 2) in
  ∃ y m, m = -4 ∧ y = -3 ∧ f' 1 = m ∧ y = m * (x - 1) + 1 → 
  true := 
by {sorry}

theorem range_a_for_zeros : {a : ℝ | -1 < a ∧ a < 0} = {a : ℝ | True} :=
by {sorry}

theorem sum_of_zeros (a x1 x2 : ℝ) (h : -1 < a ∧ a < 0) (hx : f x1 a = 0 ∧ f x2 a = 0) : 
  x1 + x2 > 2 / (a + 1) :=
by {sorry}

end tangent_eq_range_a_for_zeros_sum_of_zeros_l555_555947


namespace number_of_nonempty_proper_subsets_l555_555959

theorem number_of_nonempty_proper_subsets (M : Set ℤ) (hM : M = {-1, 0, 1}) :
  ∃ n, n = 6 ∧ (∀ (s : Set ℤ), s ⊆ M ∧ s ≠ ∅ ∧ s ≠ M → n = (2^3 - 2)) :=
by {
  have h_subsets : 2^3 = 8 := by norm_num,
  have h_nonempty_properf : 8 - 2 = 6 := by norm_num,
  use 6,
  split,
  exact h_nonempty_properf,
  intros s hs,
  cases hs with h_ss hs_nonempty,
  cases hs_nonempty with h_s_eq_empty h_s_eq_M,
  simp [h_ss, h_s_eq_empty, h_s_eq_M] at *,
  linarith,
  sorry
}

end number_of_nonempty_proper_subsets_l555_555959


namespace eval_expression_l555_555055

theorem eval_expression : (-3 : ℝ) ^ 0 + real.sqrt 8 + (-3 : ℝ) ^ 2 - 4 * (real.sqrt 2 / 2) = 10 := 
by 
  sorry

end eval_expression_l555_555055


namespace find_a_l555_555876

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l555_555876


namespace equation_represent_two_parallel_lines_l555_555860

theorem equation_represent_two_parallel_lines :
  ∀ x y : ℝ, x^2 - 9 * y^2 + 3 * x = 0 ↔ (x = 3 * y ∨ x = -3 * y - 3) :=
by
  intros x y
  split
  · intro h
    -- Insert the proof here
    sorry
  · intro h
    -- Insert the proof here
    sorry

end equation_represent_two_parallel_lines_l555_555860


namespace satisfy_inequality_l555_555489

theorem satisfy_inequality (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end satisfy_inequality_l555_555489


namespace part1_part2_l555_555545

theorem part1 (a b c : ℝ) (A B : ℝ) (h1 : a * Real.sin B + b * Real.cos A = c) : B = Real.pi / 4 :=
sorry

theorem part2 (c : ℝ)
  (h1 : ∀ (a b : ℝ) (A B : ℝ), a = √2 * c → b = 2 → a * Real.sin B + b * Real.cos A = c → B = Real.pi / 4)
  (h2 : √2 * c * Real.sin (Real.pi / 4) + 2 * Real.cos (Real.pi / 4) = c) : c = 2 :=
sorry

end part1_part2_l555_555545


namespace find_range_of_m_l555_555151

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def piecewise_function (m : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then sqrt (3 - m * x) / m else (1 / m) * x - 1

theorem find_range_of_m (m : ℝ) :
  is_monotonically_decreasing (piecewise_function m) 0 ∞ → m ≤ -1 :=
  sorry

end find_range_of_m_l555_555151


namespace max_f_A_find_b_l555_555634

noncomputable def f (A : Real) : Real :=
  2 * cos (A / 2) * sin (π - A / 2) + sin (A / 2) ^ 2 - cos (A / 2) ^ 2

theorem max_f_A (A : Real) (hA : 0 < A ∧ A < π) :
  ∃ M, M = √2 ∧ ∀ x, f(x) ≤ M :=
sorry

theorem find_b (A B C a b : Real) (hC : C = 5 * π / 12)
  (ha : a = √6) (hf : f(A) = 0) (hABC : A + B + C = π) :
  b = 3 :=
sorry

end max_f_A_find_b_l555_555634


namespace x1_x2_multiple_of_pi_l555_555218

-- definitions of the conditions
def f (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑ k in finset.range n, (1 / (2:ℝ)^(k : ℝ)) * real.cos (a k + x)

-- The proof statement
theorem x1_x2_multiple_of_pi (n : ℕ) (a : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : f n a x1 = 0) (h2 : f n a x2 = 0) : 
  ∃ k : ℤ, x1 - x2 = k * real.pi :=
sorry

end x1_x2_multiple_of_pi_l555_555218


namespace triangle_construction_exists_l555_555483

theorem triangle_construction_exists
  (A B C: Type)
  (AB: Real)
  (m_a : Real)
  (angle_CBA : Real)
  (angle_BAC : Real)
  (h1 : angle_CBA = 2 * angle_BAC) :
  ∃ (triangle: Type), 
    (side_length triangle A B AB) ∧
    (altitude triangle m_a) ∧
    (angle_measure triangle C B A angle_CBA) ∧
    (angle_measure triangle B A C angle_BAC) :=
sorry

end triangle_construction_exists_l555_555483


namespace malfatti_circles_not_maximal_in_equilateral_triangle_l555_555461

theorem malfatti_circles_not_maximal_in_equilateral_triangle :
  let r_malfatti := (Real.sqrt 3 - 1) / 4 in
  let area_malfatti := fun r => Real.pi * r ^ 2 in
  let total_area_malfatti := 3 * area_malfatti r_malfatti in
  let total_area_alternative := 11 * Real.pi / 108 in
  total_area_malfatti < total_area_alternative :=
by
  let r_malfatti := (Real.sqrt 3 - 1) / 4
  let area_malfatti := fun r => Real.pi * r ^ 2
  let total_area_malfatti := 3 * area_malfatti r_malfatti
  let total_area_alternative := 11 * Real.pi / 108
  calc
    total_area_malfatti
      = 3 * (Real.pi * ((Real.sqrt 3 - 1) / 4) ^ 2) : by sorry
      < 11 * Real.pi / 108 : by sorry

end malfatti_circles_not_maximal_in_equilateral_triangle_l555_555461


namespace count_desired_multiples_l555_555568

def is_multiple (x n : ℕ) : Prop := ∃ k : ℕ, x = k * n

lemma count_multiples_six_not_eighteen (n : ℕ) (h1 : n < 350)
  (h2 : is_multiple n 6) (h3 : ¬ is_multiple n 18) : ℕ :=
∑ x in finset.range 350, if is_multiple x 6 ∧ ¬ is_multiple x 18 then 1 else 0

theorem count_desired_multiples : count_multiples_six_not_eighteen 350 350 sorry sorry = 39 :=
sorry

end count_desired_multiples_l555_555568


namespace tangent_length_l555_555110

theorem tangent_length (P : Point) (O : Point) (r : ℝ) (A B : Point) 
  (h1 : dist O A = r) 
  (h2 : dist O B = r)
  (h3 : PA ⟂ PB)
  (h4: ∃ (PA PB : Line), tangent_to_circle PA ∧ tangent_to_circle PB) :
  dist P A = 10 := 
sorry

end tangent_length_l555_555110


namespace algebraic_expression_value_l555_555588

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l555_555588


namespace find_a_for_binomial_square_l555_555886

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l555_555886


namespace initial_production_rate_l555_555457

theorem initial_production_rate 
  (x : ℝ)
  (h1 : 60 <= (60 * x) / 30 - 60 + 1800)
  (h2 : 60 <= 120)
  (h3 : 30 = (120 / (60 / x + 1))) : x = 20 := by
  sorry

end initial_production_rate_l555_555457


namespace pencil_price_l555_555738

variable (P N : ℕ) -- This assumes the price of a pencil (P) and the price of a notebook (N) are natural numbers (non-negative integers).

-- Define the conditions
def conditions : Prop :=
  (P + N = 950) ∧ (N = P + 150)

-- The theorem to prove
theorem pencil_price (h : conditions P N) : P = 400 :=
by
  sorry

end pencil_price_l555_555738


namespace intersection_complement_l555_555241

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {1, 2}

def B : Set ℕ := {2, 3}

def C_U (S : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ S}

theorem intersection_complement :
  A ∩ C_U B = {1} :=
by
  have : C_U B = {1, 4, 5} := by sorry
  rw this
  sorry

end intersection_complement_l555_555241


namespace perimeter_of_regular_polygon_l555_555414

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555414


namespace angle_bisector_le_median_l555_555688

noncomputable def angle_bisector (A B C : ℝ) (C : ℝ) : ℝ := sorry
noncomputable def median (A B C : ℝ) (C : ℝ) : ℝ := sorry

theorem angle_bisector_le_median (A B C : ℝ) :
  ∀ (CD CM : ℝ), angle_bisector A B C = CD → median A B C = CM → CD ≤ CM :=
by
  intros CD CM hCD hCM
  rw [angle_bisector, median] at hCD hCM
  sorry

end angle_bisector_le_median_l555_555688


namespace perimeter_of_regular_polygon_l555_555419

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555419


namespace antonella_purchase_l555_555458

theorem antonella_purchase
  (total_coins : ℕ)
  (coin_value : ℕ → ℕ)
  (num_toonies : ℕ)
  (initial_loonies : ℕ)
  (initial_toonies : ℕ)
  (total_value : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (H1 : total_coins = 10)
  (H2 : coin_value 1 = 1)
  (H3 : coin_value 2 = 2)
  (H4 : initial_toonies = 4)
  (H5 : initial_loonies = total_coins - initial_toonies)
  (H6 : total_value = initial_loonies * coin_value 1 + initial_toonies * coin_value 2)
  (H7 : amount_spent = 3)
  (H8 : amount_left = total_value - amount_spent)
  (H9 : amount_left = 11) :
  ∃ (used_loonies used_toonies : ℕ), used_loonies = 1 ∧ used_toonies = 1 ∧ (used_loonies * coin_value 1 + used_toonies * coin_value 2 = amount_spent) :=
by
  sorry

end antonella_purchase_l555_555458


namespace semicircle_perimeter_approx_l555_555339

theorem semicircle_perimeter_approx (r : ℝ) (π_approx : ℝ) (h_r : r = 9) (h_π : π_approx = 3.14) :
  let diameter := 2 * r
  let full_circumference := 2 * π_approx * r
  let semicircle_perimeter := (1 / 2) * full_circumference + diameter
  semicircle_perimeter ≈ 46.26 :=
by
  have h1 : diameter = 2 * 9, by rw [h_r]
  have h2 : full_circumference = 2 * 3.14 * 9, by rw [h_r, h_π]
  have h3 : semicircle_perimeter = (1 / 2) * (2 * 3.14 * 9) + 2 * 9, by rw [h2, h1]
  exact Real.rat_cast_eq_cast.mpr (46.26 : ℚ)

end semicircle_perimeter_approx_l555_555339


namespace tangent_secent_theorem_l555_555111

theorem tangent_secent_theorem
  {circle : Type*} [metric_space circle]
  {A M C B : circle}
  (hA_outside : ∀ (X ∈ circle), A ≠ X)
  (h_tangent : tangent_line A M M)
  (h_secent : secant_line A C B)
  (h_order : C ∈ seg A B) :
  distance A C * distance A B = (distance A M)^2 :=
sorry

end tangent_secent_theorem_l555_555111


namespace balls_in_boxes_ways_l555_555570

theorem balls_in_boxes_ways : 
  (∃ (ways : ℕ), ways = 21 ∧
    ∀ {k : ℕ}, (5 + k - 1).choose (5) = ways) := 
begin
  sorry,
end

end balls_in_boxes_ways_l555_555570


namespace circles_touch_each_other_l555_555751

-- Define the circles and tangents
variables (K1 K2 : Type) [circle K1] [circle K2]
variables (L1 L2 : Line)

-- Define the conditions given in the problem
def no_common_interior (K1 K2 : Type) [circle K1] [circle K2] :=
  ∀ p : Point, ¬(p ∈ interior K1 ∧ p ∈ interior K2)

def circumscribed_quadrilateral (K1 K2 : Type) [circle K1] [circle K2] (L1 L2 : Line) : Prop :=
  (∃ A B C D : Point, L1.tangent K1 A ∧ L1.tangent K2 B ∧ L2.tangent K1 C ∧ L2.tangent K2 D ∧ 
   quadrilateral A B C D)

-- The proof problem, stating that given the conditions, the circles touch each other
theorem circles_touch_each_other (K1 K2 : Type) [circle K1] [circle K2] (L1 L2 : Line)
  (h1 : no_common_interior K1 K2)
  (h2 : circumscribed_quadrilateral K1 K2 L1 L2) :
  tangent K1 K2 :=
sorry

end circles_touch_each_other_l555_555751


namespace angle_AFE_170_l555_555040

noncomputable theory

-- Define the basic structure and points
def square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  angle A B C = 90 ∧ angle B C D = 90 ∧ angle C D A = 90 ∧ angle D A B = 90

-- Define the given conditions
variable (A B C D E F : Point)
variable h1 : square A B C D
variable h2 : on_opposite_half_plane D C E A
variable h3 : angle C D E = 110
variable h4 : collinear F A D
variable h5 : dist D E = dist D F

-- State the final proof declaration
theorem angle_AFE_170 : angle A F E = 170 :=
by 
  sorry

end angle_AFE_170_l555_555040


namespace quadratic_no_real_roots_l555_555928

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c)
  (h6 : p ≠ q)
  (h7 : a^2 = p * q)
  (h8 : b + c = p + q)
  (h9 : b = (2 * p + q) / 3)
  (h10 : c = (p + 2 * q) / 3) :
  (∀ x : ℝ, ¬ (b * x^2 - 2 * a * x + c = 0)) := 
by
  sorry

end quadratic_no_real_roots_l555_555928


namespace initial_oranges_l555_555474

variable (x : ℕ)
variable (total_oranges : ℕ := 8)
variable (oranges_from_joyce : ℕ := 3)

theorem initial_oranges (h : total_oranges = x + oranges_from_joyce) : x = 5 := by
  sorry

end initial_oranges_l555_555474


namespace total_time_taken_l555_555736

-- Definitions for the conditions
def speed_boat_still : ℝ := 15
def speed_stream : ℝ := 3
def distance_place : ℝ := 180

-- The time calculation based on downstream and upstream speeds
theorem total_time_taken:
  let downstream_speed := speed_boat_still + speed_stream in
  let upstream_speed := speed_boat_still - speed_stream in
  let time_downstream := distance_place / downstream_speed in
  let time_upstream := distance_place / upstream_speed in
  time_downstream + time_upstream = 25 :=
by
  sorry

end total_time_taken_l555_555736


namespace retail_price_of_machine_l555_555337

theorem retail_price_of_machine 
  (wholesale_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) 
  (P : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.10)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = wholesale_price * (1 + profit_rate))
  (h5 : (P * (1 - discount_rate)) = selling_price) : 
  P = 120 := by
  sorry

end retail_price_of_machine_l555_555337


namespace geometric_progression_common_ratio_l555_555549

variable {a : ℕ → ℤ}
variable (q : ℤ)
variable h_arith_seq : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
variable h_geom_seq : (a 1 - 1) * q = a 3 - 3 ∧ (a 3 - 3) * q = a 5 - 5

theorem geometric_progression_common_ratio :
  q = 1 :=
by
  sorry

end geometric_progression_common_ratio_l555_555549


namespace calculate_f_f_2_l555_555554

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - 3 else x^2 + x - 6

theorem calculate_f_f_2 : f (f 2) = -3 := by
  sorry

end calculate_f_f_2_l555_555554


namespace find_f_2_2_l555_555064

def A : Set (ℝ × ℝ) := { p | ∃ m n, p = (m, n) }
def B : Set ℝ := Set.univ

def f : ℝ × ℝ → ℝ
| (m, 1) := 1
| (m, n + 1) := if n + 1 > m then 0 else (n + 1) * (f (m, n + 1) + f (m, n))
| (m + 1, n) := n * (f (m, n) + f (m, n - 1))
| _ := 0

theorem find_f_2_2 : f (2, 2) = 2 := by
  sorry

end find_f_2_2_l555_555064


namespace even_sum_probability_l555_555108

noncomputable def probability_even_sum : ℚ :=
  let cards : Finset ℕ := {1, 2, 3, 4}
  let total_outcomes : ℚ := (cards.card ^ 2 : ℚ)
  let even_favorable : Finset (ℕ × ℕ) :=
    (cards.product cards).filter (λ pair, (pair.1 + pair.2) % 2 = 0)
  let favorable_count : ℚ := (even_favorable.card : ℚ)
  favorable_count / total_outcomes

theorem even_sum_probability :
  probability_even_sum = 1 / 2 :=
by
  sorry

end even_sum_probability_l555_555108


namespace shape_is_cylinder_l555_555513

open Real

def cylindrical_shape (c : ℝ) (h : 0 < c) : Prop :=
  ∀ (ρ ϕ z: ℝ), ρ = c ↔ (sqrt (ρ^2 + z^2) = c ∧ ∀ θ : ℝ, θ ∈ Icc 0 (2*π))

theorem shape_is_cylinder (c : ℝ) (h : 0 < c) :
  cylindrical_shape c h ↔ 
  (∃ ρ ϕ z: ℝ, (ρ = c ∧ 0 ≤ ϕ ∧ ϕ < 2*π ∧ z ∈ ℝ)) :=
sorry

end shape_is_cylinder_l555_555513


namespace regular_polygon_perimeter_l555_555391

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555391


namespace largest_c_such_that_sum_of_squares_geq_cM_squared_l555_555895

theorem largest_c_such_that_sum_of_squares_geq_cM_squared :
  ∀ (x : Fin 201 → ℝ),
  (∑ i, x i = 0) →
  let M := x ⟨100, by simp⟩ in
  ∑ i, (x i)^2 ≥ (20401 / 100) * M^2 :=
begin
  sorry
end

end largest_c_such_that_sum_of_squares_geq_cM_squared_l555_555895


namespace abs_div_sign_possible_values_l555_555976

def abs_div_sign (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ℝ :=
  (|a| / a) + (b / |b|)

theorem abs_div_sign_possible_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  abs_div_sign a b ha hb ∈ {-2, 0, 2} :=
sorry

end abs_div_sign_possible_values_l555_555976


namespace tile_covering_problem_l555_555022

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end tile_covering_problem_l555_555022


namespace distributing_balls_into_boxes_l555_555575

-- Define the parameters for the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Statement of the problem in Lean
theorem distributing_balls_into_boxes :
  (finset.card (finset.univ.filter (λ (f : fin num_boxes → ℕ), fin.sum univ f = num_balls))) = 21 := 
sorry

end distributing_balls_into_boxes_l555_555575


namespace find_x_in_sequence_l555_555787

theorem find_x_in_sequence
  (x d1 d2 : ℤ)
  (h1 : d1 = x - 1370)
  (h2 : d2 = 1070 - x)
  (h3 : -180 - 1070 = -1250)
  (h4 : -6430 - (-180) = -6250)
  (h5 : d2 - d1 = 5000) :
  x = 3720 :=
by
-- Proof omitted
sorry

end find_x_in_sequence_l555_555787


namespace largest_neg_root_eq_l555_555505

theorem largest_neg_root_eq {x : ℝ} (h : (sin x) / (1 + cos x) = 2 - (cos x) / (sin x)) :
  let x0 := if -7 * Real.pi / 6 < -11 * Real.pi / 6 then -7 * Real.pi / 6 else -11 * Real.pi / 6
  in x0 * (3 / Real.pi) = -3.5 :=
by
  sorry

end largest_neg_root_eq_l555_555505


namespace log2_sum_sequence_4_l555_555240

noncomputable def sequence (n : ℕ) : ℕ
| 0     := 1
| (n+1) := 3 * (finset.range (n+1)).sum sequence

noncomputable def sum_sequence (n : ℕ) : ℕ :=
(finset.range n).sum sequence

theorem log2_sum_sequence_4 : real.log 2 (sum_sequence 4) = 6 :=
sorry

end log2_sum_sequence_4_l555_555240


namespace more_wins_than_losses_prob_l555_555475

-- Given conditions
def num_matches : ℕ := 5

def win_prob : ℚ := 1 / 3
def lose_prob : ℚ := 1 / 3
def tie_prob : ℚ := 1 / 3

-- Question is proving that the probability of having more wins than losses
-- P = 193 / 486 is correct, and given m and n relatively prime, m + n = 679
theorem more_wins_than_losses_prob : ∃ m n : ℕ, m.coprime n ∧ m + n = 679 ∧ (193 / 486 : ℚ) = 193 / 486 :=
by
  sorry

end more_wins_than_losses_prob_l555_555475


namespace minimum_a_such_that_quadratic_nonnegative_l555_555175

theorem minimum_a_such_that_quadratic_nonnegative : 
  (∀ x ∈ set.Ioc 0 (1 / 2 : ℝ), x^2 + a * x + 1 ≥ 0) → a ≥ - (5 / 2) :=
sorry

end minimum_a_such_that_quadratic_nonnegative_l555_555175


namespace horseback_riding_trip_l555_555464

theorem horseback_riding_trip : ∃ x : ℚ,
  (5 * 7) +
  (6 * 6) +
  (x * 6 * 3) +
  (7 * 5) = 115 ∧
  x = 1 / 2 :=
begin
  use 1 / 2,
  split,
  { norm_num },
  { refl }
end

end horseback_riding_trip_l555_555464


namespace domain_shift_l555_555995

theorem domain_shift (f : ℝ → ℝ) (h : set.Icc 1 2 ⊆ set.univ) :
    ∃ (d : set ℝ), d = set.Icc 0 1 ∧ ∀ x ∈ d, ∃ y, y = x + 1 ∧ y ∈ set.Icc 1 2 :=
begin
  use set.Icc 0 1,
  split,
  { refl, },
  { intros x hx,
    use x + 1,
    split,
    { refl, },
    { exact set.mem_Icc.mpr ⟨add_nonneg hx.1 zero_le_one, add_le_add_right hx.2 1⟩, }
  }
end

end domain_shift_l555_555995


namespace convert_3241_quinary_to_septenary_l555_555062

/-- Convert quinary number 3241_(5) to septenary number, yielding 1205_(7). -/
theorem convert_3241_quinary_to_septenary : 
  let quinary := 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0
  let septenary := 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0
  quinary = 446 → septenary = 1205 :=
by
  intros
  -- Quinary to Decimal
  have h₁ : 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0 = 446 := by norm_num
  -- Decimal to Septenary
  have h₂ : 446 = 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 := by norm_num
  exact sorry

end convert_3241_quinary_to_septenary_l555_555062


namespace product_of_digits_of_largest_integer_l555_555898

-- Definition of conditions
def sum_of_squares_eq_65 (n : ℕ) : Prop :=
  let digits := Integer.digits 10 n
  digits.sum (λ d, d*d) = 65

def digits_doubled_condition (n : ℕ) : Prop :=
  let digits := Integer.digits 10 n
  ∀ i ∈ List.range (digits.length - 1), digits.get i.succ = 2 * digits.get i

-- Main theorem statement
theorem product_of_digits_of_largest_integer (n : ℕ) (h1 : sum_of_squares_eq_65 n) (h2 : digits_doubled_condition n) : 
  List.product (Integer.digits 10 n) = 8 := sorry

end product_of_digits_of_largest_integer_l555_555898


namespace find_plot_width_l555_555214

def sidewalk_area (sidewalk_width sidewalk_length : ℝ) : ℝ :=
  sidewalk_width * sidewalk_length

def flower_bed_area (a b c d e f : ℝ) : ℝ :=
  (a * b) * 2 + (c * d) + (e * f)

def total_non_sodded_area (sidewalk flower_beds : ℝ) : ℝ :=
  sidewalk + flower_beds

def total_sodded_area (total_yard non_sodded : ℝ) : ℝ :=
  total_yard - non_sodded

def plot_width (sodded_area length : ℝ) : ℝ :=
  sodded_area / length

theorem find_plot_width :
  let total_yard := 9474
  let length := 50
  let sidewalk := sidewalk_area 3 50
  let flower_beds := flower_bed_area 4 25 10 12 7 8
  let non_sodded_area := total_non_sodded_area sidewalk flower_beds
  let sodded_area := total_sodded_area total_yard non_sodded_area
  plot_width sodded_area length = 178.96 := by
  sorry

end find_plot_width_l555_555214


namespace simplify_expression_l555_555862

theorem simplify_expression : (16 : ℝ) ^ (-((2 : ℝ) ^ (-1))) = 1 / 4 :=
by
  sorry

end simplify_expression_l555_555862


namespace polygon_perimeter_l555_555382

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555382


namespace find_a_l555_555874

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l555_555874


namespace tangent_line_equation_l555_555718

noncomputable def f (x : ℝ) : ℝ := (x + 1) * log x - 4 * (x - 1)
noncomputable def f_prime (x : ℝ) : ℝ := log x + (x + 1) / x - 4

theorem tangent_line_equation :
  let point := 1
  let slope := f_prime point
  let tangent_point := (point, f point)
  slope = -2 ∧ tangent_point = (1, 0) →
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 2 ∧ b = 1 ∧ c = -2 :=
by
  intros
  sorry

end tangent_line_equation_l555_555718


namespace part1_inverse_function_part2_g_monotonicity_R_part2_h_range_l555_555105

noncomputable def f (x : ℝ) : ℝ := (8 + x^3) / (8 - x^3)
def D : Set ℝ := {x | x^2 ≠ 4}

theorem part1_inverse_function : ∀ x ∈ D, f(x) * f(-x) = 1 :=
sorry

-- Assumptions for part 2
variable (g : ℝ → ℝ)
axiom g_continuous : ∀ x : ℝ, Continuous (g x)
axiom g_monotone_increasing_neg : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → g x₁ < g x₂
axiom g_pos_neg_interval : ∀ x : ℝ, x < 0 → g x > 0
axiom g_at_neg2 : g(-2) = 1 / 2

-- We translate Part 2 as per the given conditions and conclusions:
theorem part2_g_monotonicity_R : ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂ :=
sorry

noncomputable def h (x : ℝ) : ℝ := (g x)^2 + (g (-x))^2 - g x - g (-x)

theorem part2_h_range : ∀ y ∈ Icc (-2 : ℝ) 2, h(y) ∈ Icc 0 (7 / 4) :=
sorry

end part1_inverse_function_part2_g_monotonicity_R_part2_h_range_l555_555105


namespace intersection_complement_l555_555243

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end intersection_complement_l555_555243


namespace frequencies_and_quality_difference_l555_555318

theorem frequencies_and_quality_difference 
  (A_first_class A_second_class B_first_class B_second_class : ℕ)
  (total_A total_B : ℕ)
  (total_first_class total_second_class total : ℕ)
  (critical_value_99 confidence_level : ℕ)
  (freq_A freq_B : ℚ)
  (K_squared : ℚ) :
  A_first_class = 150 →
  A_second_class = 50 →
  B_first_class = 120 →
  B_second_class = 80 →
  total_A = 200 →
  total_B = 200 →
  total_first_class = 270 →
  total_second_class = 130 →
  total = 400 →
  critical_value_99 = 10.828 →
  confidence_level = 99 →
  freq_A = 3 / 4 →
  freq_B = 3 / 5 →
  K_squared = 400 * ((150 * 80 - 50 * 120) ^ 2) / (270 * 130 * 200 * 200) →
  K_squared < critical_value_99 →
  freq_A = 3 / 4 ∧ 
  freq_B = 3 / 5 ∧ 
  confidence_level = 99 := 
by
  intros; 
  sorry

end frequencies_and_quality_difference_l555_555318


namespace exists_subset_no_double_l555_555072

theorem exists_subset_no_double (s : Finset ℕ) (h₁ : s = Finset.range 3000) :
  ∃ t : Finset ℕ, t.card = 2000 ∧ (∀ x ∈ t, ∀ y ∈ t, x ≠ 2 * y ∧ y ≠ 2 * x) :=
by
  sorry

end exists_subset_no_double_l555_555072


namespace impossible_option_B_l555_555733

theorem impossible_option_B (a : ℕ → ℝ) (c : ℝ) (h1 : ∀ n : ℕ, a (n + 3) = a n)
(h2 : ∀ n : ℕ, a n * a (n + 3) - a(n + 1) * a(n + 2) = c) : 
¬ (a 1 = 2 ∧ c = 2) :=
by
  sorry

end impossible_option_B_l555_555733


namespace triangle_integral_y_difference_l555_555201

theorem triangle_integral_y_difference :
  ∀ (y : ℕ), (3 ≤ y ∧ y ≤ 15) → (∃ y_min y_max : ℕ, y_min = 3 ∧ y_max = 15 ∧ (y_max - y_min = 12)) :=
by
  intro y
  intro h
  -- skipped proof
  sorry

end triangle_integral_y_difference_l555_555201


namespace find_a_l555_555887

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l555_555887


namespace factor_expression_l555_555079

theorem factor_expression (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) :=
by sorry

end factor_expression_l555_555079


namespace length_of_AD_l555_555203

theorem length_of_AD (A B C D : ℝ) (AB BC AC : ℝ) (hAB : AB = 8) (hBC : BC = 10) (hAC : AC = 6)
  (hAD_bisects_angle : bisects_angle A B C D) : length A D = 64 / sqrt 7 := 
by
  sorry

end length_of_AD_l555_555203


namespace rubbleShortBy8_75_l555_555258

def rubblePocket : Float := 45.0
def costNotebook : Float := 4.0
def costPen : Float := 1.5
def costEraser : Float := 2.25
def costPencilCase : Float := 7.5

def notebooksBought : Int := 5
def pensBought : Int := 8
def erasersBought : Int := 3
def pencilCasesBought : Int := 2

def totalSpent : Float := 
  (notebooksBought * costNotebook) + 
  (pensBought * costPen) + 
  (erasersBought * costEraser) + 
  (pencilCasesBought * costPencilCase)

def moneyLeft : Float := rubblePocket - totalSpent

theorem rubbleShortBy8_75 : moneyLeft = -8.75 := by
  sorry

end rubbleShortBy8_75_l555_555258


namespace areas_of_triangles_PTS_and_PSU_are_30_l555_555256

theorem areas_of_triangles_PTS_and_PSU_are_30
    (PQ RS QR : ℝ)
    (PQ_perpendicular_to_PS QR_perpendicular_to_PS : ∀ {P Q R S T U : ℝ}, T = U → ∃ x y, T = PQ ∨ U = QR) :
    (PQ = 5) → (QR = 12) → (RS = sqrt (PQ ^ 2 + QR ^ 2)) →
    let PS := sqrt (PQ ^ 2 + QR ^ 2)
    in (1 / 2 * PS  * (60 / PS) = 30)
    ∧ (1 / 2 * PS * (60 / PS) = 30) := 
by
  intros PQ QR RS PQ_eq_5 QR_eq_12 RS_eq_sqrt_PQ2_QR2
  let PS := sqrt (PQ ^ 2 + QR ^ 2)
  have PS_eq : PS = sqrt (5 ^ 2 + 12 ^ 2) := 
    by rw [PQ_eq_5, QR_eq_12]; apply sqrt_inj; norm_num
  show (1 / 2 * PS  * (60 / PS) = 30) 
    ∧ (1 / 2 * PS * (60 / PS) = 30) := 
    by
      rw [PS_eq, sqrt_sqr (show 0 ≤ 13, by norm_num)]
      split
      all_goals norm_num
      sorry

end areas_of_triangles_PTS_and_PSU_are_30_l555_555256


namespace find_a5_l555_555708

-- Define the geometric sequence and the given conditions
def geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Define the conditions for our problem
def conditions (a : ℕ → ℝ) :=
  geom_sequence a 2 ∧ (∀ n, 0 < a n) ∧ a 3 * a 11 = 16

-- Our goal is to prove that a_5 = 1
theorem find_a5 (a : ℕ → ℝ) (h : conditions a) : a 5 = 1 := 
by 
  sorry

end find_a5_l555_555708


namespace terminal_side_of_960_deg_in_third_quadrant_l555_555743

theorem terminal_side_of_960_deg_in_third_quadrant :
  (960 : ℝ) % 360 > 180 ∧ (960 : ℝ) % 360 < 270 :=
by
  norm_num[960 % 360, (960 % 360 : ℝ)]
  sorry

end terminal_side_of_960_deg_in_third_quadrant_l555_555743


namespace max_val_f_l555_555104

/-- Definition of the function f(n) which returns the smallest integer k such that n / k is not an integer. -/
def f (n : ℕ) : ℕ :=
  Nat.find (λ k, ¬ n % k = 0)

/-- Theorem stating the maximum value of f(n) as n ranges from 1 to 3000 is 11. -/
theorem max_val_f : ∃ n ∈ (Finset.range 3001).val, f n = 11 :=
sorry

end max_val_f_l555_555104


namespace compute_expression_l555_555605

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l555_555605


namespace mul_65_35_l555_555848

theorem mul_65_35 : (65 * 35) = 2275 := by
  -- define a and b
  let a := 50
  let b := 15
  -- use the equivalence (a + b) and (a - b)
  have h1 : 65 = a + b := by rfl
  have h2 : 35 = a - b := by rfl
  -- use the difference of squares formula
  have h_diff_squares : (a + b) * (a - b) = a^2 - b^2 := by sorry
  -- calculate each square
  have ha_sq : a^2 = 2500 := by sorry
  have hb_sq : b^2 = 225 := by sorry
  -- combine the results
  have h_result : a^2 - b^2 = 2500 - 225 := by sorry
  -- finish the proof
  have final_result : (65 * 35) = 2275 := by sorry
  exact final_result

end mul_65_35_l555_555848


namespace min_distance_point_coordinates_l555_555924

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x
noncomputable def line (x : ℝ) : ℝ := x - 2

def point_p (x : ℝ) : Prop := (curve x) = x^2 - Real.log x
def distance_from_p_min (x : ℝ) : Prop := x = 1

theorem min_distance_point_coordinates :
  ∃ x : ℝ, point_p x ∧ distance_from_p_min x :=
by { use 1, split, sorry, sorry }

end min_distance_point_coordinates_l555_555924


namespace sum_series_eq_l555_555229

theorem sum_series_eq (x : ℝ) (hx : x > 1) : 
  ∑' n, 1 / (x^(3^n) - x^(-3^n)) = 1 / (x - 1) :=
sorry

end sum_series_eq_l555_555229


namespace linear_equation_with_two_variables_l555_555193

def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3 * a) * x + (a + 1) * y + 3 * a

theorem linear_equation_with_two_variables (a : ℝ) :
  (equation a x y = 0) ∧ (a^2 - 4 = 0) ∧ (2 - 3 * a ≠ 0) ∧ (a + 1 ≠ 0) →
  (a = 2 ∨ a = -2) :=
by sorry

end linear_equation_with_two_variables_l555_555193


namespace distributing_balls_into_boxes_l555_555573

-- Define the parameters for the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Statement of the problem in Lean
theorem distributing_balls_into_boxes :
  (finset.card (finset.univ.filter (λ (f : fin num_boxes → ℕ), fin.sum univ f = num_balls))) = 21 := 
sorry

end distributing_balls_into_boxes_l555_555573


namespace regular_polygon_perimeter_l555_555390

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555390


namespace problem_1_problem_2_problem_3_problem_4_l555_555208

variable (A B C a b c : ℝ)
variable (triangle_ABC : A + B + C = Real.pi)
variable (acute_triangle : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)

-- Problem 1
theorem problem_1 (h : a / Real.cos A = b / Real.sin B) : A = Real.pi / 4 := sorry

-- Problem 2
theorem problem_2 (h : Real.sin (2 * A) = Real.sin (2 * B)) : ¬ ∀ (A B : ℝ), A = B := sorry

-- Problem 3
theorem problem_3 (ha : a = 1) (hb : b = 2) (hA : A = Real.pi / 6) : ¬ (∃ C, True) := sorry

-- Problem 4
theorem problem_4 (h_acute : acute_triangle) : Real.sin A + Real.sin B > Real.cos A + Real.cos B := sorry

end problem_1_problem_2_problem_3_problem_4_l555_555208


namespace max_area_PAOB_l555_555626

-- Definitions
variables {R : Type*} [linear_ordered_field R] -- specify a real number type

structure Point :=
(x : R)
(y : R)

def distance (p1 p2 : Point) : R :=
real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Conditions
def P_in_first_quadrant (P : Point) : Prop :=
P.x > 0 ∧ P.y > 0

def A_on_x_axis (A : Point) : Prop :=
A.y = 0

def B_on_y_axis (B : Point) : Prop :=
B.x = 0

def PA_PB_eq_2 (P A B : Point) : Prop :=
distance P A = 2 ∧ distance P B = 2

def area_PAOB (P A B O : Point) : R :=
0.5 * abs (A.x * B.y)

-- Main theorem statement
theorem max_area_PAOB (P A B : Point) (hP : P_in_first_quadrant P)
    (hA : A_on_x_axis A) (hB : B_on_y_axis B) (hPA_PB : PA_PB_eq_2 P A B) :
    ∃ x, area_PAOB P A B O = 2 + 2 * real.sqrt 2 :=
sorry

end max_area_PAOB_l555_555626


namespace profit_percentage_with_discount_l555_555817

-- Define the cost price (CP) of the book
def CP : ℝ := 100

-- Define the selling price (SP) with no discount
def SP_no_discount : ℝ := CP + 1.5 * CP

-- Define the selling price (SP) with a 5% discount on the no-discount selling price
def SP_with_discount : ℝ := 0.95 * SP_no_discount

-- Define the profit with a 5% discount
def profit_with_discount : ℝ := SP_with_discount - CP

-- Define the percentage of profit with a 5% discount
def percentage_profit_with_discount : ℝ := (profit_with_discount / CP) * 100

-- The theorem to prove
theorem profit_percentage_with_discount : percentage_profit_with_discount = 137.5 := 
by
  sorry

end profit_percentage_with_discount_l555_555817


namespace congruence_problem_l555_555990

theorem congruence_problem {x : ℤ} (h : 4 * x + 5 ≡ 3 [ZMOD 20]) : 3 * x + 8 ≡ 2 [ZMOD 10] :=
sorry

end congruence_problem_l555_555990


namespace folded_pentagon_smaller_perimeter_l555_555524

theorem folded_pentagon_smaller_perimeter {A B C D: ℝ} (h_rect: quadrilateral A B C D) 
  (h_non_square: ¬(AB = AD) ∧ ¬(BC = AD) ∧ ¬(CD = AB)) :
  folded_pentagon_smaller_perimeter A B C D :=
by
  -- Let’s specify the vertices and hypothesis declarations here:
  -- h_rect: quadrilateral A B C D - stating that this is a quadrilateral
  -- h_non_square: ¬(AB = AD) ∧ ¬(BC = AD) ∧ ¬(CD = AB) - stating that it is not a square
  sorry

end folded_pentagon_smaller_perimeter_l555_555524


namespace h_is_even_l555_555658

variable {α : Type*} [LinearOrderedCommRing α]
variable (g h : α → α)

def is_odd_function (f : α → α) : Prop :=
∀ x, f (-x) = -f x

def is_even_function (f : α → α) : Prop :=
∀ x, f (-x) = f x

def h_definition (g : α → α) (x : α) : α :=
| g (x^5) |

theorem h_is_even (hg : is_odd_function g) : is_even_function (h_definition g) :=
sorry

end h_is_even_l555_555658


namespace sufficient_and_necessary_condition_l555_555030

theorem sufficient_and_necessary_condition (x : ℝ) :
  (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 :=
by sorry

end sufficient_and_necessary_condition_l555_555030


namespace perimeter_of_regular_polygon_l555_555409

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555409


namespace max_green_beads_l555_555363

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end max_green_beads_l555_555363


namespace Grant_scored_higher_l555_555968

noncomputable def Hunter_score : ℕ := 45
noncomputable def John_score : ℕ := 2 * Hunter_score
noncomputable def Grant_score : ℕ := 100

theorem Grant_scored_higher : Grant_score - John_score = 10 :=
by 
  -- Grant's score is 100
  have hGrant : Grant_score = 100 := rfl 
  -- Hunter's score is 45
  have hHunter : Hunter_score = 45 := rfl
  -- John's score is twice Hunter's score
  have hJohn : John_score = 2 * Hunter_score := rfl
  -- Combining the above, we get
  calc
    Grant_score - John_score = 100 - (2 * 45) : by rw [hGrant, hJohn, hHunter]
                        ... = 10             : by norm_num

end Grant_scored_higher_l555_555968


namespace pedestrian_impossible_walk_l555_555017

theorem pedestrian_impossible_walk {V : Type*} (streets : V → V → Prop) (u: V) :
  (∃ p : list (V × V), ∀ e ∈ p, streets e.1 e.2) → 
  (∃ v: V, 6 = cardinal.mk ↥{ w | streets u w }) →
  ¬(∃ p : list (V × V), ∀ e ∈ p, streets e.1 e.2) :=
sorry

end pedestrian_impossible_walk_l555_555017


namespace find_angle_C_l555_555204

-- Define the variables and conditions
variables {a b C : ℝ} {S : ℝ}
def is_triangle (a b S : ℝ) := a = 2 * sqrt 3 ∧ b = 2 ∧ S = sqrt 3

-- State the main theorem to prove the value of angle C
theorem find_angle_C (h : is_triangle a b S) : C = π / 6 :=
sorry

end find_angle_C_l555_555204


namespace elizabeth_initial_bottles_l555_555073

theorem elizabeth_initial_bottles (B : ℕ) (H1 : B - 2 - 1 = (3 * X) → 3 * (B - 3) = 21) : B = 10 :=
by
  sorry

end elizabeth_initial_bottles_l555_555073


namespace sum_of_fractions_l555_555254

theorem sum_of_fractions (n : ℕ) :
  (∑ r in Finset.filter (λ rs : Finset (ℕ × ℕ), 
    let (r, s) := rs in r < s ∧ s ≤ n ∧ r + s > n ∧ Nat.coprime r s) (Finset.range n).offDiag,
    (1 : ℝ) / (r.1 * r.2)) = 1 / 2 := sorry

end sum_of_fractions_l555_555254


namespace reflection_addition_l555_555723

theorem reflection_addition (m b : ℝ) :
  (∀ x y : ℝ, (x, y) = (2, 2) → (x', y') = (10, 6) → reflects_across_line (x, y) (x', y') (m, b)) → 
  m + b = 14 :=
by
  sorry

end reflection_addition_l555_555723


namespace minimum_colored_cells_l555_555763

theorem minimum_colored_cells (n : ℕ) (h : n = 6) : 
  ∃ (m : ℕ), m = 12 ∧ (∃ (covering : (ℕ × ℕ) → Prop), 
    (∀ (L_tetromino : fin 6 × fin 6 → Prop) (hL : is_l_tetromino L_tetromino),
      ∃ (i j : fin 6), covering (i, j) = true → (i, j) ∈ L_tetromino) →
    (∑ (i : fin 6) (j : fin 6), if covering (i, j) then 1 else 0) = m) :=
by 
  use 12
  split
  . rfl
  . sorry


end minimum_colored_cells_l555_555763


namespace min_sin_cos_over_all_A_l555_555507

theorem min_sin_cos_over_all_A : 
  ∃ A : ℝ, sin (A / 2) + cos (A / 2) = -2 :=
by
  -- Omitted: Proof will come here
  sorry

end min_sin_cos_over_all_A_l555_555507


namespace volume_of_region_l555_555900

theorem volume_of_region :
  ∀ (x y z : ℝ), 
    0 <= x ∧ 0 <= y ∧ 0 <= z ∧ 
    abs (x + y + z) + abs (x + y - z) + abs (x - y + z) + abs (-x + y + z) <= 8 
    → volume ({ p : ℝ × ℝ × ℝ | 0 <= p.1 ∧ 0 <= p.2 ∧ 0 <= p.3 ∧ p.1 + p.2 + p.3 <= 4 }) = 32 / 3 :=
sorry

end volume_of_region_l555_555900


namespace wire_in_corridor_l555_555038

theorem wire_in_corridor : 
  ∃ (x : ℝ), (width : ℝ) (h_width : width = 1), let r := width + x in 
  (x = 1 + Real.sqrt 2) ∧ (width = 1) → 2 * r > 4 := by 
  sorry

end wire_in_corridor_l555_555038


namespace discriminant_of_quadratic_is_321_l555_555327

-- Define the quadratic equation coefficients
def a : ℝ := 4
def b : ℝ := -9
def c : ℝ := -15

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The proof statement
theorem discriminant_of_quadratic_is_321 : discriminant a b c = 321 := by
  sorry

end discriminant_of_quadratic_is_321_l555_555327


namespace find_a_for_square_binomial_l555_555881

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l555_555881


namespace basketball_scores_distinct_count_l555_555795

theorem basketball_scores_distinct_count :
  ∀ (x : ℕ), 0 ≤ x ∧ x ≤ 7 → ∃! (P : ℕ), (∃ y : ℕ, y = 7 - x ∧ P = 3*x + 2*y) ∧ (P ∈ {14, 15, 16, 17, 18, 19, 20, 21}) :=
by sorry

end basketball_scores_distinct_count_l555_555795


namespace is_inverse_g1_is_inverse_g2_l555_555329

noncomputable def f (x : ℝ) := 3 + 2*x - x^2

noncomputable def g1 (x : ℝ) := -1 + Real.sqrt (4 - x)
noncomputable def g2 (x : ℝ) := -1 - Real.sqrt (4 - x)

theorem is_inverse_g1 : ∀ x, f (g1 x) = x :=
by
  intro x
  sorry

theorem is_inverse_g2 : ∀ x, f (g2 x) = x :=
by
  intro x
  sorry

end is_inverse_g1_is_inverse_g2_l555_555329


namespace count_with_consecutive_ones_l555_555970

noncomputable def countValidIntegers : ℕ := 512
noncomputable def invalidCount : ℕ := 89

theorem count_with_consecutive_ones :
  countValidIntegers - invalidCount = 423 :=
by
  sorry

end count_with_consecutive_ones_l555_555970


namespace binary_to_decimal_101101_l555_555857

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end binary_to_decimal_101101_l555_555857


namespace probability_of_composite_in_first_50_l555_555341

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def count_composites (N : ℕ) : ℕ :=
  (Finset.range (N + 1)).filter is_composite |> Finset.card

theorem probability_of_composite_in_first_50 :
  (count_composites 50 : ℝ) / 50 = 0.68 :=
by {
  sorry
}

end probability_of_composite_in_first_50_l555_555341


namespace sequence_contains_perfect_square_l555_555235

def f (x : ℕ) : ℕ := x + (Int.sqrt x : ℕ)

theorem sequence_contains_perfect_square (m : ℕ) : 
  ∃ n, ∃ k : ℕ, f^[n] m = k * k :=
sorry

end sequence_contains_perfect_square_l555_555235


namespace find_a_for_square_binomial_l555_555879

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l555_555879


namespace number_of_integer_exponents_l555_555630

noncomputable def binomial_expansion_integer_exponents (n : ℕ) : ℕ :=
  -- Function that calculates the number of terms with integer exponents in the binomial expansion
  -- The specific implementation of this calculation can be complex and needs to be formalized.
  sorry

theorem number_of_integer_exponents (n : ℕ) : 
  -- Condition: The coefficients of the first three terms in the expansion form an arithmetic sequence.
  -- Proof Problem: Prove the number of terms with integer exponents is 3.
  (coeff (λ k, (binomial n k) * (sqrt x)^(n - k) * (1 / (2 * sqrt (sqrt x)))^k) 0)  + 
  (coeff (λ k, (binomial n k) * (sqrt x)^(n - k) * (1 / (2 * sqrt (sqrt x)))^k) 1) +
  (coeff (λ k, (binomial n k) * (sqrt x)^(n - k) * (1 / (2 * sqrt (sqrt x)))^k) 2) =>
  binomial_expansion_integer_exponents n = 3 :=
by sorry

end number_of_integer_exponents_l555_555630


namespace time_A_problems_60_l555_555616

variable (t : ℕ) -- time in minutes per type B problem

def time_per_A_problem := 2 * t
def time_per_C_problem := t / 2
def total_time_for_A_problems := 20 * time_per_A_problem

theorem time_A_problems_60 (hC : 80 * time_per_C_problem = 60) : total_time_for_A_problems = 60 := by
  sorry

end time_A_problems_60_l555_555616


namespace annual_average_expenditure_relation_minimum_annual_average_expenditure_minimum_annual_average_year_l555_555010

variables (x : ℝ) (y : ℝ)

def initial_purchase : ℝ := 120000
def annual_fees : ℝ := 10500 / 1000  -- converting to thousand yuan
def maintenance_costs (x : ℝ) : ℝ := 0.05 * x^2 + 0.15 * x
def sedan_value (x : ℝ) : ℝ := 10.75 - 0.8 * x
def total_cost (x : ℝ) : ℝ := initial_purchase / 1000 + annual_fees * x + maintenance_costs x - sedan_value x

def annual_average_expenditure (x : ℝ) : ℝ := total_cost x / x

theorem annual_average_expenditure_relation (hx : x > 0) :
  annual_average_expenditure x = 0.05 * x + 2 + 1.25 / x :=
by
  sorry

theorem minimum_annual_average_expenditure :
  annual_average_expenditure 5 = 2.5 :=
by
  sorry

theorem minimum_annual_average_year :
  ∀ x > 0, annual_average_expenditure x ≥ 2.5 :=
by
  sorry

end annual_average_expenditure_relation_minimum_annual_average_expenditure_minimum_annual_average_year_l555_555010


namespace find_distance_l555_555019

-- Definitions and assumptions from the problem
def speed : ℝ := 3.6 -- Speed in km/h
def time : ℝ := 12   -- Time in minutes

-- Conversion factors
def kilometers_to_meters (km : ℝ) : ℝ := km * 1000
def hours_to_minutes (hrs : ℝ) : ℝ := hrs * 60

-- Theorem stating the distance calculation
theorem find_distance (S : ℝ) (T : ℝ) (D : ℝ) (h₁ : S = 3.6) (h₂ : T = 12) : 
  D = kilometers_to_meters (S) / hours_to_minutes (1) * T :=
by
  -- Proof will be provided later
  sorry

-- Goal: Prove that the length of the street is 720 meters.
example : ∃ D, find_distance speed time D (rfl) (rfl) ∧ D = 720 :=
by
  -- Proof will be provided later
  sorry

end find_distance_l555_555019


namespace sam_hourly_rate_l555_555693

theorem sam_hourly_rate
  (first_month_earnings : ℕ)
  (second_month_earnings : ℕ)
  (total_hours : ℕ)
  (h1 : first_month_earnings = 200)
  (h2 : second_month_earnings = first_month_earnings + 150)
  (h3 : total_hours = 55) :
  (first_month_earnings + second_month_earnings) / total_hours = 10 := 
  by
  sorry

end sam_hourly_rate_l555_555693


namespace total_length_of_ropes_l555_555300

theorem total_length_of_ropes 
  (L : ℕ)
  (first_used second_used : ℕ)
  (h1 : first_used = 42) 
  (h2 : second_used = 12) 
  (h3 : (L - second_used) = 4 * (L - first_used)) :
  2 * L = 104 :=
by
  -- We skip the proof for now
  sorry

end total_length_of_ropes_l555_555300


namespace derivative_value_l555_555144

theorem derivative_value :
  (∀ (x : ℝ), f (x) = x ^ 2 + 4 * x * f' (2) + Real.log x) →
  (∀ (x : ℝ), f' (x) = 2 * x + 4 * f' (2) + 1 / x) →
  f' (3) = 1 / 3 :=
by
  intros h₁ h₂
  sorry

end derivative_value_l555_555144


namespace find_angle_C_l555_555173

variable {A B C : ℝ}
variable (m n : ℝ × ℝ)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def vector1 := (Real.sqrt 3 * Real.sin A, Real.sin B)
noncomputable def vector2 := (Real.cos B, Real.sqrt 3 * Real.cos A)

theorem find_angle_C :
  dot_product vector1 vector2 = 1 + Real.cos (A + B) →
  C = 120 := sorry

end find_angle_C_l555_555173


namespace triangle_side_ratio_l555_555636

variable {A B C a b c : ℝ}
variable (h1 : tan A * tan B = 4 * (tan A + tan B) * tan C)
variable (h2 : a = sin A / cos A * c / sin C)
variable (h3 : b = sin B / cos B * c / sin C)

theorem triangle_side_ratio (h1 : tan A * tan B = 4 * (tan A + tan B) * tan C)
  (h2 : a = sin A / cos A * c / sin C) 
  (h3 : b = sin B / cos B * c / sin C) : 
  (a^2 + b^2) / c^2 = 9 := 
by sorry

end triangle_side_ratio_l555_555636


namespace regular_polygon_perimeter_l555_555446

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555446


namespace identify_incorrect_statement_l555_555245

theorem identify_incorrect_statement :
  ¬(
    (∀ (a b c : ℝ), (a > b) → (c > 0) → (a + c > b + c) ∧ (a - c > b - c) ∧ (a * c > b * c) ∧ (a / c > b / c)) ∧
    (∀ (a b : ℝ), (a > 0) → (b > 0) → ¬ ( (a ≠ b) → ( (a + b) / 2 < (2 * a * b) / (a + b) ) )) ∧
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x * y = k) → (x + y ≠ 2 * (x + y) / 2) ) ∧
    (forall a b : ℝ, (a > 0) → (b > 0) → (a = b) → ((1 / 2) * (a ^ 2 + b ^ 2) = (1 / 2) * (a + b) ^ 2)) ∧
    (∀ (x y s : ℝ), (x + y = s) → (x ≠ y) → (x * y ≠ min))
  ) := sorry

end identify_incorrect_statement_l555_555245


namespace average_velocity_interval_l555_555277

theorem average_velocity_interval (Δt : ℝ) (hΔt : Δt ≠ 0) : 
  average_velocity_interval (s : ℝ → ℝ) := s = 2 * t^2 - 2 → 
  (average_velocity [2, 2 + Δt]) = 8 + 2 * Δt :=
by sorry

end average_velocity_interval_l555_555277


namespace find_a_for_binomial_square_l555_555882

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l555_555882


namespace perimeter_of_regular_polygon_l555_555411

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555411


namespace peter_wrote_on_board_l555_555250

theorem peter_wrote_on_board : ∃ n : ℕ, ∃ (seq : Fin n → ℕ), 
  (∀ i, 10 ≤ seq i ∧ seq i < 100) ∧
  ∀ i ∈ (Finset.range n).erase (n-1), seq i > seq (i+1) ∧ 
  ¬ (seq (n-1) / 10 = 7 ∨ seq (n-1) % 10 = 7) ∧
  let x := seq.to_list.foldl (λ acc d, acc * 100 + d) 0 in
  ∃ a b : ℕ, nat.prime a ∧ nat.prime b ∧ (b = a + 4) ∧ (x = a * b) ∧ x = 221 := 
sorry

end peter_wrote_on_board_l555_555250


namespace alphaMoreAdvantageousRegular_betaMoreAdvantageousMood_l555_555755

/-
  Definitions based on conditions
-/
def alphaCostPerMonth : ℕ := 999
def betaCostPerMonth : ℕ := 1299
def monthsInYear : ℕ := 12
def weeksInMonth : ℕ := 4
def regularVisitsPerWeek : ℕ := 2
def moodPatternVisitsPerYear : ℕ := 56  -- Derived from mood pattern calculation

/-
  Hypotheses based on question interpretation
-/
def alphaYearlyCost : ℕ := alphaCostPerMonth * monthsInYear
def betaYearlyCost : ℕ := betaCostPerMonth * monthsInYear

def regularVisitsPerYear : ℕ := regularVisitsPerWeek * weeksInMonth * monthsInYear
def alphaCostPerVisitRegular : ℕ := alphaYearlyCost / regularVisitsPerYear
def betaCostPerVisitRegular : ℕ := betaYearlyCost / regularVisitsPerYear

def alphaCostPerVisitMood : ℕ := alphaYearlyCost / moodPatternVisitsPerYear
def betaCostDuringMood : ℕ := betaCostPerMonth * 8  -- Only 8 months visited
def betaCostPerVisitMood : ℕ := betaCostDuringMood / moodPatternVisitsPerYear

/-
  Theorems to be proven.
-/
theorem alphaMoreAdvantageousRegular : alphaCostPerVisitRegular < betaCostPerVisitRegular := 
sorry

theorem betaMoreAdvantageousMood : betaCostPerVisitMood < alphaCostPerVisitMood := 
sorry

end alphaMoreAdvantageousRegular_betaMoreAdvantageousMood_l555_555755


namespace computation_of_expression_l555_555596

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l555_555596


namespace roots_situation_depends_on_k_l555_555289

theorem roots_situation_depends_on_k (k : ℝ) : 
  let a := 1
  let b := -3
  let c := 2 - k
  let Δ := b^2 - 4 * a * c
  (Δ > 0) ∨ (Δ = 0) ∨ (Δ < 0) :=
by
  intros
  sorry

end roots_situation_depends_on_k_l555_555289


namespace arithmetic_prog_leq_l555_555124

def t3 (s : List ℤ) : ℕ := 
  sorry -- Placeholder for function calculating number of 3-term arithmetic progressions

theorem arithmetic_prog_leq (a : List ℤ) (k : ℕ) (h_sorted : a = List.range k)
  : t3 a ≤ t3 (List.range k) :=
sorry -- Proof here

end arithmetic_prog_leq_l555_555124


namespace prob_one_product_successful_l555_555004

open ProbabilityTheory

noncomputable def probAtLeastOneSuccess (probA probB : ℝ) (hA: 0 ≤ probA ∧ probA ≤ 1) (hB: 0 ≤ probB ∧ probB ≤ 1) (indep: Independent (fun _ => ℕ) id) : ℝ :=
  1 - ((1 - probA) * (1 - probB))

theorem prob_one_product_successful :
  let probA := 2 / 3 in
  let probB := 3 / 5 in
  let prob_none := (1 - probA) * (1 - probB) in
  probAtLeastOneSuccess probA probB (by norm_num) (by norm_num) (by apply_instance) = 13 / 15 :=
by {
  unfold probAtLeastOneSuccess,
  norm_num
}

end prob_one_product_successful_l555_555004


namespace intersect_or_parallel_l555_555233

variables {α : Type*} [LinearOrderedField α]
variables {A B C O M N P Q R T : α} -- Consider these as points in the plane

-- Definitions as per conditions
def is_point (x : α) := true -- Simplified for the context of this problem

axiom O_is_arbitrary : is_point O
axiom M_foot_perpendicular_A : is_point M -- and perpendicular from O to internal angle bisector at A
axiom N_foot_perpendicular_A : is_point N -- and perpendicular from O to external angle bisector at A
axiom P_foot_perpendicular_B : is_point P -- and perpendicular from O to internal angle bisector at B
axiom Q_foot_perpendicular_B : is_point Q -- and perpendicular from O to external angle bisector at B
axiom R_foot_perpendicular_C : is_point R -- and perpendicular from O to internal angle bisector at C
axiom T_foot_perpendicular_C : is_point T -- and perpendicular from O to external angle bisector at C

-- Theorem to prove that lines MN, PQ, and RT either intersect at a single point or are parallel
theorem intersect_or_parallel 
  (hM : M_foot_perpendicular_A)
  (hN : N_foot_perpendicular_A)
  (hP : P_foot_perpendicular_B)
  (hQ : Q_foot_perpendicular_B)
  (hR : R_foot_perpendicular_C)
  (hT : T_foot_perpendicular_C) :
  (MN ∩ PQ ≠ ∅ ∧ MN ∩ RT ≠ ∅ ∧ PQ ∩ RT ≠ ∅) ∨ MN ∥ PQ ∨ MN ∥ RT ∨ PQ ∥ RT :=
sorry

end intersect_or_parallel_l555_555233


namespace _l555_555692

variables (a b c : ℝ)
-- Conditionally define the theorem giving the constraints in the context.
example (h1 : a < 0) (h2 : b < 0) (h3 : c > 0) : 
  abs a - abs (a + b) + abs (c - a) + abs (b - c) = 2 * c - a := by 
sorry

end _l555_555692


namespace smallest_positive_period_and_monotonically_increasing_interval_maximum_area_of_triangle_ABC_l555_555519

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos (x + π / 4) ^ 2

theorem smallest_positive_period_and_monotonically_increasing_interval :
  ∀ (k : ℤ), (∃ T : ℝ, T > 0 ∧ T = π ∧
  (∀ x, f (x) = f (x + T)) ∧
  (∀ x ∈ Icc (-π/4 + k * π) (π/4 + k * π), ∀ y ∈ Icc x (π/4 + k * π), f x ≤ f y)) :=
sorry

theorem maximum_area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧
  A + B + C = π ∧ a = 1 ∧ f (A / 2) = 0 →
  ∀ b c, (b > 0 ∧ c > 0 ∧ a^2 = b^2 + c^2 - 2 * b * c * cos A) →
  ∃ area : ℝ, area = (2 + real.sqrt 3) / 4 :=
sorry

end smallest_positive_period_and_monotonically_increasing_interval_maximum_area_of_triangle_ABC_l555_555519


namespace evaluate_root_power_l555_555076

theorem evaluate_root_power : (Real.sqrt (Real.sqrt 9))^12 = 729 := 
by sorry

end evaluate_root_power_l555_555076


namespace arithmetic_seq_problem_l555_555922

-- Define the arithmetic sequence and the sum of the first n terms
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Given conditions
def a2 : Prop := arithmetic_seq a₁ d 2 = 1
def S4 : Prop := sum_arithmetic_seq a₁ d 4 = 8

-- Proof goal
theorem arithmetic_seq_problem (a₁ d : ℤ) (h1 : a2) (h2 : S4) : 
  arithmetic_seq a₁ d 5 = 7 ∧ sum_arithmetic_seq a₁ d 10 = 80 :=
by
  sorry

end arithmetic_seq_problem_l555_555922


namespace bound_stride_difference_l555_555473

-- Definitions of conditions
def n_strides : ℕ := 55
def n_bounds : ℕ := 15
def n_trees : ℕ := 51
def dist_feet : ℝ := 6336

-- Proposition statement using Lean
theorem bound_stride_difference : 
  let gaps := n_trees - 1,
      total_strides := n_strides * gaps,
      total_bounds := n_bounds * gaps,
      stride_length := dist_feet / total_strides,
      bound_length := dist_feet / total_bounds in
  bound_length - stride_length ≈ 6 := 
by sorry

end bound_stride_difference_l555_555473


namespace pond_90_percent_free_on_day_29_l555_555453

/-
Define the conditions in a) as mathematical definitions and variables.
1. Algae triples each day
2. Pond is fully covered on day 30
-/
noncomputable def algae_ratio_day : ℕ → ℝ
| 30     := 1               -- Day 30, 100% coverage
| (n+1)  := algae_ratio_day n / 3 -- The area triples backward

/-
The problem now asks to prove the pond was 90% free of algae on day 29,
which means 10% covered on day 29.
-/
theorem pond_90_percent_free_on_day_29 :
  algae_ratio_day 29 = 0.1 :=
sorry

end pond_90_percent_free_on_day_29_l555_555453


namespace filled_circles_count_l555_555451

theorem filled_circles_count (n : ℕ) : 
  ∑ i in range n, (i + 1) < 2009 → n ≤ 63 := sorry

end filled_circles_count_l555_555451


namespace largest_constant_inequality_l555_555089

variable {a b c d e : ℝ}
variable (h : ∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)

theorem largest_constant_inequality : (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d)) > 2) :=
by
  sorry

end largest_constant_inequality_l555_555089


namespace regular_polygon_perimeter_l555_555392

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555392


namespace problem_solution_l555_555625

open Real

variable (a : ℝ) (t : ℝ)

-- Definition of curve C1 with parametric equations
def C1_x : ℝ := a + √2 * t
def C1_y : ℝ := 1 + √2 * t

-- Standard equation of curve C1
def C1_standard_eq : Prop := ∀ t, C1_x a t - C1_y a t = a - 1

-- Polar equation of curve C2 and its Cartesian form
def C2_polar_eq (ρ θ : ℝ) : Prop := ρ * cos θ ^ 2 + 4 * cos θ - ρ = 0
def C2_cartesian_eq : Prop := ∀ (x y : ℝ), y^2 = 4 * x

-- Intersection condition
def intersection_condition (P A B : ℝ × ℝ) : Prop := (| P.1 - A.1 + P.2 - A.2 |) = 2 * (| P.1 - B.1 + P.2 - B.2 |)

-- Main statement in Lean
theorem problem_solution :
  C1_standard_eq a ∧ C2_cartesian_eq ∧
  (∀ (P A B : ℝ × ℝ), intersection_condition P A B → (a = 1 / 36 ∨ a = 9 / 4)) :=
by sorry

end problem_solution_l555_555625


namespace minimum_s_value_l555_555929

theorem minimum_s_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  ∃ (s : ℝ), s = 8 * Real.sqrt 6 ∧ ∀ (x' y' z' : ℝ), (0 < x' ∧ 0 < y' ∧ 0 < z' ∧ 3 * x'^2 + 2 * y'^2 + z'^2 = 1) → 
      s ≤ (1 + z') / (x' * y' * z') :=
sorry

end minimum_s_value_l555_555929


namespace find_value_of_c_l555_555750

-- Given: The transformed linear regression equation and the definition of z
theorem find_value_of_c (z : ℝ) (y : ℝ) (x : ℝ) (c : ℝ) (k : ℝ) (h1 : z = 0.4 * x + 2) (h2 : z = Real.log y) (h3 : y = c * Real.exp (k * x)) : 
  c = Real.exp 2 :=
by
  sorry

end find_value_of_c_l555_555750


namespace computation_of_expression_l555_555599

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l555_555599


namespace real_root_quadratic_l555_555065

theorem real_root_quadratic (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := 
sorry

end real_root_quadratic_l555_555065


namespace max_pieces_on_chessboard_l555_555762

theorem max_pieces_on_chessboard : 
  ∃ (n : ℕ), 
  (∀ (A : fin 8 → fin 8 → Prop), 
    (∀ i j, A i j → A i j = true) → 
    (∀ d, ∃ k, ∀ A : fin 8 → fin 8 → Prop, (∑ x in finset.univ, if A x (x + d) = true then 1 else 0) ≤ 3) → 
    ∑ i j, if A i j = true then 1 else 0 = n) ∧ n = 38 := 
begin
  sorry
end

end max_pieces_on_chessboard_l555_555762


namespace computation_of_expression_l555_555598

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l555_555598


namespace stratified_sampling_and_probability_l555_555940

open Nat

def total_students_A := 240
def total_students_B := 160
def total_students_C := 160

def selected_students := 7

noncomputable def ratio_A := 3
noncomputable def ratio_B := 2
noncomputable def ratio_C := 2

def selected_A := (selected_students * ratio_A) / (ratio_A + ratio_B + ratio_C)
def selected_B := (selected_students * ratio_B) / (ratio_A + ratio_B + ratio_C)
def selected_C := (selected_students * ratio_C) / (ratio_A + ratio_B + ratio_C)

-- Set of all 2-student combinations from 7 students
def combinations := {s : Finset (Fin 7) | s.card = 2}

-- Favorable outcomes
def from_same_grade (s : Finset (Fin 7)) : Prop :=
  (s = {0, 1} ∨ s = {0, 2} ∨ s = {1, 2} ∨ s = {3, 4} ∨ s = {5, 6})

def event_M := {s : Finset (Fin 7) | from_same_grade s}

def probability_M := (Finset.card event_M) / (Finset.card combinations)

theorem stratified_sampling_and_probability :
  selected_A = 3 ∧ selected_B = 2 ∧ selected_C = 2 ∧ probability_M = 5 / 21 :=
by
  sorry

end stratified_sampling_and_probability_l555_555940


namespace solve_hyperbola_problem_l555_555358

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def line (x y : ℝ) : Prop :=
  y = sqrt 3 * x - 4 * sqrt 3

theorem solve_hyperbola_problem (a b c : ℝ) 
  (h_hyperbola : ∀ x y, hyperbola a b x y)
  (h_line : ∀ x y, line x y)
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_focus : ∀ (x : ℝ), x ≥ 0 → x * sqrt 3 = 4 * sqrt 3 → x = c)
  (h_asymptote : b / a = sqrt 3)
  (h_focal_property : a^2 + b^2 = c^2)
  : (2 * c = 6) ∧ (c / a = 2) :=
by sorry

end solve_hyperbola_problem_l555_555358


namespace price_reduction_after_markup_l555_555452

theorem price_reduction_after_markup (p : ℝ) (x : ℝ) (h₁ : 0 < p) (h₂ : 0 ≤ x ∧ x < 1) :
  (1.25 : ℝ) * (1 - x) = 1 → x = 0.20 := by
  sorry

end price_reduction_after_markup_l555_555452


namespace percentage_is_12_l555_555986

variable (x : ℝ) (p : ℝ)

-- Given the conditions
def condition_1 : Prop := 0.25 * x = (p / 100) * 1500 - 15
def condition_2 : Prop := x = 660

-- We need to prove that the percentage p is 12
theorem percentage_is_12 (h1 : condition_1 x p) (h2 : condition_2 x) : p = 12 := by
  sorry

end percentage_is_12_l555_555986


namespace problem_statement_l555_555994

definition M : Set ℕ := {0, 1}
definition I : Set ℕ := {0, 1, 2, 3, 4, 5}

def C_I (s : Set ℕ) : Set ℕ := I \ s -- Complimentary set of s in I

theorem problem_statement : C_I M = {2, 3, 4, 5} :=
by
  -- Definitions and conditions
  have hM : M = {0, 1} := rfl
  have hI : I = {0, 1, 2, 3, 4, 5} := rfl
  -- Conclusion
  sorry

end problem_statement_l555_555994


namespace Eval_trig_exp_l555_555869

theorem Eval_trig_exp :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end Eval_trig_exp_l555_555869


namespace volume_increase_factor_l555_555999

-- Defining the initial volume of the cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Defining the modified height and radius
def new_height (h : ℝ) : ℝ := 3 * h
def new_radius (r : ℝ) : ℝ := 2.5 * r

-- Calculating the new volume with the modified dimensions
def new_volume (r h : ℝ) : ℝ := volume (new_radius r) (new_height h)

-- Proof statement to verify the volume factor
theorem volume_increase_factor (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  new_volume r h = 18.75 * volume r h :=
by
  sorry

end volume_increase_factor_l555_555999


namespace A_B_relation_l555_555238

-- Define propositions A and B
def proposition_A (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 ≤ 0
def proposition_B (a : ℝ) : Prop := ∀ x ∈ (1 : ℝ, ∞), log a (x + a - 2) > 0

-- Define a condition to check the relationship between A and B
def A_is_necessary_but_not_sufficient_for_B : Prop :=
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧ ¬(∀ a : ℝ, proposition_A a → proposition_B a)

theorem A_B_relation : A_is_necessary_but_not_sufficient_for_B :=
sorry

end A_B_relation_l555_555238


namespace quadratic_function_solution_l555_555100

noncomputable def g (x : ℝ) : ℝ := x^2 + 44 * x + 50

theorem quadratic_function_solution (c d : ℝ)
  (h : ∀ x, (g (g x + x)) / (g x) = x^2 + 44 * x + 50) :
  c = 44 ∧ d = 50 :=
by
  sorry

end quadratic_function_solution_l555_555100


namespace regular_polygon_perimeter_l555_555428

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555428


namespace relationship_of_P_Q_R_l555_555172

variables (θ : ℝ) (P Q R : ℝ)

-- Definition of the conditions
def condition1 : Prop := π/2 < θ ∧ θ < π
def condition2 : Prop := P = 3^(cos θ)
def condition3 : Prop := Q = (cos θ)^3
def condition4 : Prop := R = (cos θ)^(1/3)

-- Theorem statement: prove that R < Q < P
theorem relationship_of_P_Q_R
  (h1 : condition1 θ)
  (h2 : condition2 θ P)
  (h3 : condition3 θ Q)
  (h4 : condition4 θ R)
  : R < Q ∧ Q < P :=
by
  sorry

end relationship_of_P_Q_R_l555_555172


namespace problem_statement_l555_555667

noncomputable def a : ℕ → ℝ
| 0       := -3
| (n + 1) := a n + b n + Real.sqrt (a n ^ 2 + b n ^ 2)

noncomputable def b : ℕ → ℝ
| 0       := 2
| (n + 1) := a n + b n - Real.sqrt (a n ^ 2 + b n ^ 2)

theorem problem_statement : (1 / a 10) + (1 / b 10) = 1 / 3 :=
sorry

end problem_statement_l555_555667


namespace xiaohong_test_number_l555_555768

theorem xiaohong_test_number (x : ℕ) :
  (88 * x - 85 * (x - 1) = 100) → x = 5 :=
by
  intro h
  sorry

end xiaohong_test_number_l555_555768


namespace solve_equation_number_of_real_solutions_l555_555972

theorem solve_equation : (∃ x : ℝ, (2 * x^2 - 7)^2 - 8 * x = 48) :=
begin
  sorry
end

theorem number_of_real_solutions : { x : ℝ // (2 * x^2 - 7)^2 - 8 * x = 48 }.card = 4 :=
begin
  sorry
end

end solve_equation_number_of_real_solutions_l555_555972


namespace distance_at_half_speed_is_5_total_distance_is_10_l555_555814

noncomputable section

def mass : ℝ := 0.5
def initial_speed : ℝ := 10
def friction_coeff : ℝ := 0.5

def velocity (t : ℝ) : ℝ := initial_speed * real.exp (-friction_coeff * t / mass)

def distance (t : ℝ) : ℝ := initial_speed * mass / friction_coeff * (1 - real.exp (-friction_coeff * t / mass))

def t_half : ℝ := mass * real.log 2 / friction_coeff

def distance_at_half_speed := distance t_half
def total_distance : ℝ := initial_speed * mass / friction_coeff

theorem distance_at_half_speed_is_5 : distance_at_half_speed = 5 := by
  sorry

theorem total_distance_is_10 : total_distance = 10 := by
  sorry

end distance_at_half_speed_is_5_total_distance_is_10_l555_555814


namespace bethany_saw_16_portraits_l555_555838

variable (P S : ℕ)

def bethany_conditions : Prop :=
  S = 4 * P ∧ P + S = 80

theorem bethany_saw_16_portraits (P S : ℕ) (h : bethany_conditions P S) : P = 16 := by
  sorry

end bethany_saw_16_portraits_l555_555838


namespace perimeter_of_triangle_ABF2_l555_555917

def hyperbola_conditions
  (F1 F2 : Point)
  (chord_length : ℝ)
  (a : ℝ)
  (A B : Point)
  (AB : LineSegment A B)
  (AF1 : LineSegment A F1)
  (BF1 : LineSegment B F1)
  (AF2 : LineSegment A F2)
  (BF2 : LineSegment B F2) : Prop :=
  2 * a = 8 ∧ 
  chord_length = 5 ∧ 
  (|AF2.length - AF1.length| = 8) ∧
  (|BF2.length - BF1.length| = 8)

theorem perimeter_of_triangle_ABF2
  (F1 F2 : Point)
  (chord_length : ℝ)
  (a : ℝ)
  (A B : Point)
  (AB : LineSegment A B)
  (AF1 : LineSegment A F1)
  (BF1 : LineSegment B F1)
  (AF2 : LineSegment A F2)
  (BF2 : LineSegment B F2)
  (h : hyperbola_conditions F1 F2 chord_length a A B AB AF1 BF1 AF2 BF2) :
  (AF2.length + BF2.length + chord_length = 26) :=
sorry

end perimeter_of_triangle_ABF2_l555_555917


namespace find_a9_l555_555943

def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  finset.sum (finset.range n) a

variables (a : ℕ → ℚ)

def sequence_condition_1 := ∀ n, 2 * a (n + 1) = a n + a (n + 2)
def sequence_condition_2 := a 5 - a 3 = 2
def sequence_condition_3 := sum_first_n_terms a 2 = 2

theorem find_a9
  (h1 : sequence_condition_1 a)
  (h2 : sequence_condition_2 a)
  (h3 : sequence_condition_3 a) :
  a 9 = 17 / 2 :=
sorry

end find_a9_l555_555943


namespace domain_f_l555_555083

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_f :
  {x : ℝ | f x ≠ f x} = {x : ℝ | (x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x)} :=
by sorry

end domain_f_l555_555083


namespace find_c_of_binomial_square_l555_555981

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l555_555981


namespace max_min_sum_reciprocals_l555_555255

theorem max_min_sum_reciprocals (x y : ℝ) (S : ℝ) (p q : ℝ) 
  (h_eq : 4 * x^2 - 5 * x * y + 4 * y^2 = 5)
  (h_S : S = x^2 + y^2)
  (h_max_min : ∀ θ : ℝ, p = max (S : ℝ) | θ ∈ (real.range (λ θ, √(S) * cos θ)) 
                          ∧ q = min (S : ℝ) | θ ∈ (real.range (λ θ, √(S) * sin θ))
  (h_max_min_val : ∀ θ : ℝ, p = max (S θ) ∧ q = min (S θ)) :
  (1 / p) + (1 / q) = 8 / 5 :=
sorry

end max_min_sum_reciprocals_l555_555255


namespace find_y_of_collinear_vectors_l555_555142

theorem find_y_of_collinear_vectors (y : ℝ) 
  (h : ∃ k : ℝ, (2, 3) = k • (-4, y)) : y = -6 := 
sorry

end find_y_of_collinear_vectors_l555_555142


namespace perimeter_of_regular_polygon_l555_555410

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555410


namespace tan_cot_eq_sin_cos_conditions_l555_555772

theorem tan_cot_eq_sin_cos_conditions {x k n : ℝ} (h1 : tan x ≠ cot x) (h2 : cos x ≠ 0) (h3 : sin x ≠ 0) :
  x = (-π / 8 + k * π / 2) ∨ x = (1 / 2 * arctan 5 + n * π / 2) → 
  (tan x + cot x) / (cot x - tan x) = 6 * cos (2 * x) + 4 * sin (2 * x) :=
sorry

end tan_cot_eq_sin_cos_conditions_l555_555772


namespace range_of_a_l555_555958

open Real

noncomputable def conditions (a : ℝ) : Prop :=
  (a x^2 + (a + 1) * x + 6 * a = 0) ∧
  (∃ x_1 x_2 : ℝ, x_1 ≠ x_2 ∧ x_1 < 1 ∧ 1 < x_2)

theorem range_of_a (a : ℝ) : conditions a → -((1 / 8) : ℝ) < a ∧ a < 0 := 
begin
  sorry
end

end range_of_a_l555_555958


namespace unique_8_tuple_real_numbers_l555_555508

theorem unique_8_tuple_real_numbers : 
  ∃! (x : ℕ → ℝ) (h : ∀ n, 1 ≤ n ∧ n ≤ 8 → (1 - x 1)^2 + ∑ i in Finset.range (8 - 1), (x (i + 1) - x i)^2 + x 8^2 = 1 / 9) := 
sorry

end unique_8_tuple_real_numbers_l555_555508


namespace perimeter_of_polygon_l555_555432

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555432


namespace Emilee_earnings_l555_555213

theorem Emilee_earnings (J R_j T R_t E R_e : ℕ) :
  (R_j * J = 35) → 
  (R_t * T = 30) → 
  (R_j * J + R_t * T + R_e * E = 90) → 
  (R_e * E = 25) :=
by
  intros h1 h2 h3
  sorry

end Emilee_earnings_l555_555213


namespace sufficient_but_not_necessary_condition_l555_555911

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * |x| + b

-- Theorem stating "a = 0" is a sufficient but not necessary condition for "f(x) = x^2 + a|x| + b to be an even function"
theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a = 0 → is_even_function (λ x, f x a b)) ∧ ¬(∀ a, is_even_function (λ x, f x a b) → (a = 0)) :=
begin
  split,
  { intro ha, 
    rw ha, 
    unfold is_even_function f,
    intro x,
    exact sorry },
  { intro h,
    unfold is_even_function f at h,
    exact sorry }
end

end sufficient_but_not_necessary_condition_l555_555911


namespace largest_constant_m_l555_555088

theorem largest_constant_m (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) +
   sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d))) > 2 := 
sorry

end largest_constant_m_l555_555088


namespace triangle_GCD_area_l555_555701

-- Definitions
variables {A B C D E F G : Type}
variables [Square ABCD]
variables [Point E on BC]
variables [Midpoint F of AE]
variables [Midpoint G of DE]

-- Given conditions
axiom square_area : area ABCD = 200
axiom BEGF_area : area (quadrilateral B E G F) = 34

-- Goal: Prove the area of triangle GCD is 41
theorem triangle_GCD_area : area (triangle G C D) = 41 :=
by
  sorry

end triangle_GCD_area_l555_555701


namespace parallelogram_area_eq_80_l555_555760

-- Define the base, height, and angle of the parallelogram
def base : ℝ := 20
def height : ℝ := 4
def angle : ℝ := 60 -- in degrees, though angle is not used in area calculation, includes it as per problem condition

-- Define the area of the parallelogram
def area (b h : ℝ) : ℝ := b * h

-- Theorem stating the area of the parallelogram with given conditions
theorem parallelogram_area_eq_80 (b h : ℝ) (hb : b = 20) (hh : h = 4) : 
  area b h = 80 := by
  sorry

end parallelogram_area_eq_80_l555_555760


namespace smallest_n_for_simplest_form_l555_555904

-- Definitions and conditions
def simplest_form_fractions (n : ℕ) :=
  ∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + 2) = 1

-- Problem statement
theorem smallest_n_for_simplest_form :
  ∃ n : ℕ, simplest_form_fractions (n) ∧ ∀ m : ℕ, m < n → ¬ simplest_form_fractions (m) := 
by 
  sorry

end smallest_n_for_simplest_form_l555_555904


namespace max_reflections_theorem_l555_555347

-- Problem definitions
def angle_CDA := 10
def angle_incidence := 15

-- Definition of maximum reflections function
def max_reflections (angle_CDA angle_incidence : ℕ) : ℕ :=
(max_n : ℕ) (h_max_n : 10 * max_n + 5 ≤ 90) : max_n

-- Theorem to determine the maximum number of reflections
theorem max_reflections_theorem : max_reflections angle_CDA angle_incidence = 8 := 
by 
    sorry

end max_reflections_theorem_l555_555347


namespace min_shift_value_sin_l555_555720

theorem min_shift_value_sin (φ : ℝ) (hφ : φ > 0) :
  (∀ x, sin (3 * (x + φ)) = sin (3 * (2 * π / 4 - x + φ))) → φ = π / 4 :=
by
  sorry

end min_shift_value_sin_l555_555720


namespace binary_to_decimal_101101_l555_555854

theorem binary_to_decimal_101101 :
  let b := [1, 0, 1, 1, 0, 1] in
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0 in
  decimal = 45 :=
by
  let b := [1, 0, 1, 1, 0, 1];
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0;
  show decimal = 45;
  sorry

end binary_to_decimal_101101_l555_555854


namespace sum_of_max_marks_l555_555028

theorem sum_of_max_marks :
  ∀ (M S E : ℝ),
  (30 / 100 * M = 180) ∧
  (50 / 100 * S = 200) ∧
  (40 / 100 * E = 120) →
  M + S + E = 1300 :=
by
  intros M S E h
  sorry

end sum_of_max_marks_l555_555028


namespace regular_polygon_perimeter_l555_555430

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555430


namespace angle_DFB_is_100_l555_555177

-- Definitions for the geometrical problem
structure TriangleABC :=
  (A B C : Point) -- Points of the triangle
  (angle_A angle_B angle_C : ℝ) -- Angles of the triangle
  (sum_angles : angle_A + angle_B + angle_C = 180)

structure PerpendicularBisector (A D : Point) (BC : Line) :=
  (perpendicular : is_perpendicular (Segment AD) BC)
  (bisects : bisects BC D)

structure Altitude (A F : Point) (BC : Line) :=
  (perpendicular : is_perpendicular (Segment AF) BC)

def find_angle_DFB (T : TriangleABC) (D F B : Point) [P : PerpendicularBisector T.A D (Line.bc T.B T.C)]
  [A : Altitude T.A F (Line.bc T.B T.C)] : ℝ :=
  sorry -- Definition ends with the answer of the angle measure

-- Theorem Statement
theorem angle_DFB_is_100 (T : TriangleABC) (D F B : Point)
  (hB : TriangleABC.angle_B T = 70)
  (hA : TriangleABC.angle_A T = 80)
  (hC : TriangleABC.angle_C T = 30)
  [P : PerpendicularBisector T.A D (Line.bc T.B T.C)]
  [A : Altitude T.A F (Line.bc T.B T.C)] :
  find_angle_DFB T D F B = 100 :=
sorry -- Proof will be provided later

end angle_DFB_is_100_l555_555177


namespace fibonacci_sum_l555_555268

noncomputable def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

def S : ℚ :=
  ∑' n, (fibonacci n)^2 / (9 : ℚ)^n

theorem fibonacci_sum :
  S = 207 / 220 ∧ nat.gcd 207 220 = 1 →
  207 + 220 = 427 :=
by {
  sorry
}

end fibonacci_sum_l555_555268


namespace correct_discount_rate_l555_555020

def purchase_price : ℝ := 200
def marked_price : ℝ := 300
def desired_profit_percentage : ℝ := 0.20

theorem correct_discount_rate :
  ∃ (x : ℝ), 300 * x = 240 ∧ x = 0.80 := 
by
  sorry

end correct_discount_rate_l555_555020


namespace find_value_of_n_l555_555493

theorem find_value_of_n (n : ℤ) : 
    n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 :=
by 
  intro h
  sorry

end find_value_of_n_l555_555493


namespace empty_subset_zero_l555_555766

theorem empty_subset_zero : ∅ ⊆ {0} :=
sorry

end empty_subset_zero_l555_555766


namespace from_458_to_14_l555_555186

def double_or_erase (n : ℕ) : List ℕ :=
  [n * 2, n / 10]

def reachable (start target : ℕ) : Prop :=
  ∃ moves : List ℕ, moves.head = start ∧ moves.last = target ∧ ∀ (i : ℕ), i < moves.length - 1 → moves.nthLe i sorry ∈ double_or_erase (moves.nthLe i sorry)

theorem from_458_to_14 : reachable 458 14 :=
  sorry

end from_458_to_14_l555_555186


namespace range_of_g_l555_555101

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x / 3))^2 + 2 * Real.pi * Real.arcsin (x / 3) -
  (Real.arcsin (x / 3))^2 + (Real.pi^2 / 18) * (x^2 + 12 * x + 27)

lemma arccos_arcsin_identity (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.arccos x + Real.arcsin x = Real.pi / 2 := sorry

theorem range_of_g : ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, g x = y ∧ y ∈ Set.Icc (Real.pi^2 / 4) (5 * Real.pi^2 / 2) :=
sorry

end range_of_g_l555_555101


namespace dwarfs_multiple_of_four_l555_555495

axiom dwarf_voting : Prop
-- The number of dwarfs at the table
def num_dwarfs : ℕ
-- Each dwarf has neighbors that affect their voting
axiom neighbors_voting_affect (dwarves : ℕ) : Prop
-- If neighbors vote the same way
axiom unanimous_neighbors (dwarves : ℕ) : Prop
-- If neighbors vote differently
axiom different_neighbors (dwarves : ℕ) : Prop

axiom unanimous_vote_on_gold : Prop := ∀ d, d < num_dwarfs → dwarf_voting
axiom thorin_abstains_on_dragon : Prop := dwarf_voting

theorem dwarfs_multiple_of_four (dwarves : ℕ) (hv : dwarf_voting) (hn : neighbors_voting_affect dwarves)
  (hun : unanimous_neighbors dwarves) (hdn : different_neighbors dwarves)
  (hg : unanimous_vote_on_gold) (ht : thorin_abstains_on_dragon) :
  ∃ k : ℕ, num_dwarfs = 4 * k :=
sorry

end dwarfs_multiple_of_four_l555_555495


namespace no_function_satisfies_condition_l555_555504

theorem no_function_satisfies_condition :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f(f(n)) = n + 1 :=
by
  sorry

end no_function_satisfies_condition_l555_555504


namespace g_2187_value_l555_555288

-- Define the function properties and the goal
theorem g_2187_value (g : ℕ → ℝ) (h : ∀ x y m : ℕ, x + y = 3^m → g x + g y = m^3) :
  g 2187 = 343 :=
sorry

end g_2187_value_l555_555288


namespace tip_customers_count_l555_555829

-- Definitions and given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def no_tip_customers : ℕ := 34

-- Total customers computation
def total_customers : ℕ := initial_customers + added_customers

-- Lean 4 statement for proof problem
theorem tip_customers_count : (total_customers - no_tip_customers) = 15 := by
  sorry

end tip_customers_count_l555_555829


namespace computation_of_expression_l555_555595

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l555_555595


namespace two_point_three_six_as_fraction_l555_555758

theorem two_point_three_six_as_fraction : (236 : ℝ) / 100 = (59 : ℝ) / 25 := 
by
  sorry

end two_point_three_six_as_fraction_l555_555758


namespace team_selection_l555_555678

-- Define the number of boys and girls in the club
def boys : Nat := 10
def girls : Nat := 12

-- Define the number of boys and girls to be selected for the team
def boys_team : Nat := 4
def girls_team : Nat := 4

-- Calculate the number of combinations using Nat.choose
noncomputable def choosing_boys : Nat := Nat.choose boys boys_team
noncomputable def choosing_girls : Nat := Nat.choose girls girls_team

-- Calculate the total number of ways to form the team
noncomputable def total_combinations : Nat := choosing_boys * choosing_girls

-- Theorem stating the total number of combinations equals the correct answer
theorem team_selection :
  total_combinations = 103950 := by
  sorry

end team_selection_l555_555678


namespace jeremy_is_40_l555_555737

-- Definitions for Jeremy (J), Sebastian (S), and Sophia (So)
def JeremyCurrentAge : ℕ := 40
def SebastianCurrentAge : ℕ := JeremyCurrentAge + 4
def SophiaCurrentAge : ℕ := 60 - 3

-- Assertion properties
axiom age_sum_in_3_years : (JeremyCurrentAge + 3) + (SebastianCurrentAge + 3) + (SophiaCurrentAge + 3) = 150
axiom sebastian_older_by_4 : SebastianCurrentAge = JeremyCurrentAge + 4
axiom sophia_age_in_3_years : SophiaCurrentAge + 3 = 60

-- The theorem to prove that Jeremy is currently 40 years old
theorem jeremy_is_40 : JeremyCurrentAge = 40 := by
  sorry

end jeremy_is_40_l555_555737


namespace trajectory_is_semicircle_scaled_l555_555344

-- Defining points A, B, C and the semicircle overline(AB)
structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  ⟨ (A.x + B.x) / 2, (A.y + B.y) / 2 ⟩

def semicircle (A B : Point) : Set Point :=
  { P | P.x^2 + P.y^2 = (midpoint A B).x^2 + (midpoint A B).y^2 ∧ P.y ≥ 0 }

noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

noncomputable def problem (A B : Point) (h : ∃ C, C = midpoint A B) :
  Set Point :=
  { Q | ∃ P ∈ semicircle A B,
       ∃ C, C = midpoint A B ∧
       Q ∈ segment C P ∧
       distance Q P = (distance P A - distance P B) / 2 }

theorem trajectory_is_semicircle_scaled (A B : Point) (h : ∃ C, C = midpoint A B) :
  ∃ r, ∀ C, C = midpoint A B →
    ∀ Q ∈ problem A B h,
    ∃ P ∈ semicircle A B, distance Q C = r * distance P C := sorry

end trajectory_is_semicircle_scaled_l555_555344


namespace eccentricity_of_ellipse_through_C_l555_555187

theorem eccentricity_of_ellipse_through_C {A B C : Type} [metric_space A] [metric_space B] [metric_space C] 
  (h_right_triangle : ∀ (A B C : A), ∠C = 90° ∧ ∠A = 30°) 
  (h_dist : dist B C = 1 ∧ dist A C = √3 ∧ dist A B = 2) :
  let e := 1 / (√3 + 1) / 2 in e = √3 - 1 :=
by
  sorry

end eccentricity_of_ellipse_through_C_l555_555187


namespace measure_angle_BAO_l555_555627

-- Given conditions translated to Lean
variables {CD : Type} [C : has_length CD]
variables {O A E B D : PointType}
variables {semi_circle : semicircle CD O}
variables (H1 : A lies_on_extension_of DC past C)
variables (H2 : E lies_on semi_circle)
variables (H3 : B = intersection (line_segment AE) semi_circle)
variables (H4 : length (line_segment AB) = 2 * (length (line_segment OD)))
variables (H5 : angle_measure (angle EOD) = 60)

-- The proof problem to be stated
theorem measure_angle_BAO :
  angle_measure (angle BAO) = 30 := sorry

end measure_angle_BAO_l555_555627


namespace build_days_eq_a_squared_div_x_l555_555977

-- Definitions
variables {a n x : ℕ}

-- Condition: a people can build x meters of road in n days
def build_efficiency (a n x : ℕ) : ℚ := x / (a * n)

-- Theorem: The number of days it takes for n people to build a meters of road
theorem build_days_eq_a_squared_div_x (ha : 0 < a) (hn : 0 < n) (hx : 0 < x) :
  ∀ {d : ℚ}, (n * build_efficiency a n x * d = a) → d = a^2 / x :=
begin
  intro d,
  intro h,
  sorry
end

end build_days_eq_a_squared_div_x_l555_555977


namespace distance_point_to_plane_l555_555134

-- Define the plane using the normal vector and a point on the plane
variable (a : ℝ³ := (-1, 1, 2)) (A P : ℝ³ := (2, 1, 7), (1, -2, 2))

-- Define a proof that calculates the distance
theorem distance_point_to_plane (a : ℝ³) (A P : ℝ³) (normal_a : a = (-1, 1, 2))
    (point_A : A = (2, 1, 7)) (point_P : P = (1, -2, 2)) :
    distance P (plane_through_point_normal A a) = 7 * real.sqrt 6 / 3 := 
sorry

end distance_point_to_plane_l555_555134


namespace circle_diameter_l555_555082
open Real

theorem circle_diameter (A : ℝ) (hA : A = 50.26548245743669) : ∃ d : ℝ, d = 8 :=
by
  sorry

end circle_diameter_l555_555082


namespace arithmetic_sequence_a7_l555_555628

/--
In an arithmetic sequence {a_n}, it is known that a_1 = 2 and a_3 + a_5 = 10.
Then, we need to prove that a_7 = 8.
-/
theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (h3 : ∀ n, a n = 2 + (n - 1) * d) : 
  a 7 = 8 := by
  sorry

end arithmetic_sequence_a7_l555_555628


namespace find_m_l555_555127

theorem find_m (m : ℤ): m < (real.sqrt 11 - 1) / 2 ∧ (real.sqrt 11 - 1) / 2 < m + 1 → m = 1 :=
by sorry

end find_m_l555_555127


namespace A_and_B_together_time_l555_555349

-- Definitions based on given conditions
def A_rate : ℝ := 1 / 20
def B_rate : ℝ := 1 / 15

-- Theorem statement for the proof problem
theorem A_and_B_together_time : 1 / (A_rate + B_rate) = 60 / 7 := by
  sorry

end A_and_B_together_time_l555_555349


namespace math_problem_l555_555532

open Real

theorem math_problem
  (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h : x^2 + y^2 + z^2 = 3) :
  sqrt (3 - ( (x + y) / 2) ^ 2) + sqrt (3 - ( (y + z) / 2) ^ 2) + sqrt (3 - ( (z + x) / 2) ^ 2) ≥ 3 * sqrt 2 :=
by 
  sorry

end math_problem_l555_555532


namespace frequencies_and_confidence_level_l555_555321

namespace MachineQuality

-- Definitions of the given conditions
def productsA := 200
def firstClassA := 150
def secondClassA := 50

def productsB := 200
def firstClassB := 120
def secondClassB := 80

def totalProducts := productsA + productsB
def totalFirstClass := firstClassA + firstClassB
def totalSecondClass := secondClassA + secondClassB

-- 1. Frequencies of first-class products
def frequencyFirstClassA := firstClassA / productsA
def frequencyFirstClassB := firstClassB / productsB

-- 2. \( K^2 \) calculation
def n := 400
def a := 150
def b := 50
def c := 120
def d := 80

def K_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- The theorem to prove the frequencies and the confidence level
theorem frequencies_and_confidence_level : 
    frequencyFirstClassA = (3 / 4) ∧ frequencyFirstClassB = (3 / 5) ∧ K_squared > 6.635 := 
    by {
        sorry -- Proof steps go here
    }

end MachineQuality

end frequencies_and_confidence_level_l555_555321


namespace polygon_perimeter_l555_555377

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555377


namespace regular_polygon_perimeter_l555_555394

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555394


namespace football_club_initial_balance_l555_555354

noncomputable def initial_balance (final_balance income expense : ℕ) : ℕ :=
  final_balance + income - expense

theorem football_club_initial_balance :
  initial_balance 60 (2 * 10) (4 * 15) = 20 := by
sorry

end football_club_initial_balance_l555_555354


namespace sum_geometric_series_l555_555236

theorem sum_geometric_series (x : ℝ) (n : ℕ) : 
  (∑ k in Finset.range (n + 1), x^k) = 
    if x = 1 then 
      n + 1 
    else 
      (x^(n+1) - 1)/(x - 1) := 
by
 sorry

end sum_geometric_series_l555_555236


namespace shanghai_mock_exam_problem_l555_555785

noncomputable def a_n : ℕ → ℝ := sorry -- Defines the arithmetic sequence 

theorem shanghai_mock_exam_problem 
  (a_is_arithmetic : ∃ d a₀, ∀ n, a_n n = a₀ + n * d)
  (h₁ : a_n 1 + a_n 3 + a_n 5 = 9)
  (h₂ : a_n 2 + a_n 4 + a_n 6 = 15) :
  a_n 3 + a_n 4 = 8 := 
  sorry

end shanghai_mock_exam_problem_l555_555785


namespace tourist_in_circular_forest_tourist_in_half_plane_l555_555031

theorem tourist_in_circular_forest :
  ∀ (d : ℝ) (A B : ℝ × ℝ), 
    (0 < d) →
    (dist A B < d) →
    ¬(∀ (direction : ℝ), ∃ (P : ℝ × ℝ), dist A P = d ∧ dist P B = dist A B) :=
by sorry

theorem tourist_in_half_plane :
  ∀ (d : ℝ) (A B1 B2 C1 C2 D1 E : ℝ × ℝ), 
    (0 < d) →
    (dist A B1 = (2 * sqrt 3 / 3) * d) →
    (dist B1 C1 = (sqrt 3 / 3) * d) →
    (dist C1 D1 = (7 / 6) * π * d) →
    (dist D1 E = d) →
    (dist A E < (1 + 2 * π) * d) :=
by sorry

end tourist_in_circular_forest_tourist_in_half_plane_l555_555031


namespace max_val_neg_5000_l555_555228

noncomputable def max_val_expression (x y : ℝ) : ℝ :=
  (x^2 + (1 / y^2)) * (x^2 + (1 / y^2) - 100) + (y^2 + (1 / x^2)) * (y^2 + (1 / x^2) - 100)

theorem max_val_neg_5000 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y, x > 0 ∧ y > 0 ∧ max_val_expression x y = -5000 :=
by
  sorry

end max_val_neg_5000_l555_555228


namespace compute_expression_l555_555606

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l555_555606


namespace find_a_l555_555889

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l555_555889


namespace subset_of_sum_elements_l555_555665

theorem subset_of_sum_elements (n : ℕ) (hn : n > 1) (S : Finset ℕ) (hS : S ⊆ Finset.range (2 * n + 1)) (h_size : S.card = n + 2) : 
  ∃ a b c ∈ S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b = c :=
sorry

end subset_of_sum_elements_l555_555665


namespace algebraic_expression_value_l555_555587

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l555_555587


namespace billy_distance_l555_555840

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem billy_distance :
  distance 0 0 (7 + 4 * Real.sqrt 2) (4 * (Real.sqrt 2 + 1)) = Real.sqrt (129 + 88 * Real.sqrt 2) :=
by
  -- proof goes here
  sorry

end billy_distance_l555_555840


namespace mary_average_speed_l555_555210

noncomputable def average_speed (D T : ℝ) : ℝ := D / T

def uphill_distance : ℝ := 1.5
def uphill_time : ℝ := 45 / 60
def rest_time : ℝ := 15 / 60
def downhill_time : ℝ := 15 / 60
def total_distance : ℝ := 2 * uphill_distance
def total_time : ℝ := uphill_time + rest_time + downhill_time

theorem mary_average_speed :
  average_speed total_distance total_time = 2.4 :=
by
  sorry

end mary_average_speed_l555_555210


namespace triangle_area_increases_l555_555635

noncomputable def heron_area (a b c : ℝ) : ℝ :=
let s := (a + b + c) / 2 in
real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_increases
    (XY XZ YZ : ℝ)
    (hXY : XY = 8)
    (hXZ : XZ = 5)
    (hYZ : YZ = 6)
    (new_XZ : ℝ)
    (hnew_XZ : new_XZ = 10) :
    heron_area XY new_XZ YZ > heron_area XY XZ YZ
      :=
by
  rw [hXY, hXZ, hYZ, hnew_XZ]
  apply sorry

end triangle_area_increases_l555_555635


namespace initial_deck_card_count_l555_555008

-- Define the initial conditions
def initial_red_probability (r b : ℕ) : Prop := r * 4 = r + b
def added_black_probability (r b : ℕ) : Prop := r * 5 = 4 * r + 6

theorem initial_deck_card_count (r b : ℕ) (h1 : initial_red_probability r b) (h2 : added_black_probability r b) : r + b = 24 := 
by sorry

end initial_deck_card_count_l555_555008


namespace increasing_interval_of_even_function_l555_555996

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m-2) * x^2 + (m-1) * x + 2

theorem increasing_interval_of_even_function :
  ∀ m : ℝ, (∀ x : ℝ, f m (-x) = f m x) → (m = 1 → (∀ x, x ∈ Iic (0:ℝ) ↔ deriv (λ x, f m x) x ≥ 0)) :=
begin
  intros m h_even h_m1,
  sorry
end

end increasing_interval_of_even_function_l555_555996


namespace find_n_l555_555547

theorem find_n {n : ℕ} (H : 2 * nat.choose n 9 = nat.choose n 8 + nat.choose n 10) :
  n = 14 ∨ n = 23 :=
by
  sorry

end find_n_l555_555547


namespace regular_polygon_perimeter_l555_555370

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555370


namespace variance_remaining_scores_l555_555180

noncomputable def scores : List ℝ := [90, 89, 90, 95, 93, 94, 93]

def remaining_scores (s : List ℝ) : List ℝ :=
s.erase 95 |>.erase 89

def mean (l : List ℝ) : ℝ :=
(l.sum / l.length)

def variance (l : List ℝ) : ℝ :=
let m := mean l in
(sum (l.map (λ x, (x - m) ^ 2)) / l.length)

theorem variance_remaining_scores :
  variance (remaining_scores scores) = 2.8 :=
sorry

end variance_remaining_scores_l555_555180


namespace sector_area_l555_555942

-- Definitions based on the conditions
def radius : ℝ := 3
def central_angle : ℝ := 120 / 360
def pi := Real.pi

-- Theorems based on the question and correct answer
theorem sector_area : (central_angle * pi * radius^2) = 3 * pi :=
by
  sorry

end sector_area_l555_555942


namespace donut_combinations_l555_555049

namespace DonutShop

/-- Define the total number of donuts Bill must buy --/
def total_donuts : ℕ := 10

/-- Define the number of kinds of donuts in the shop --/
def kinds_of_donuts : ℕ := 4

/-- Prove that there are 10 ways for Bill to purchase the donuts according to the given conditions --/
theorem donut_combinations : 
  (∃ purchases : (fin kinds_of_donuts → ℕ), 
      (∀ i, 2 ≤ purchases i) ∧ 
      (finset.univ.sum purchases = total_donuts)) → 
  ∃ count, count = 10 :=
sorry

end DonutShop

end donut_combinations_l555_555049


namespace geometric_prod_eight_l555_555129

theorem geometric_prod_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : ∀ n, a n ≠ 0)
  (h_eq : a 4 + 3 * a 8 = 2 * (a 7)^2)
  (h_geom : ∀ {m n : ℕ}, b m * b (m + n) = b (2 * m + n))
  (h_b_eq_a : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 :=
sorry

end geometric_prod_eight_l555_555129


namespace problem_proof_l555_555601

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l555_555601


namespace P_lt_Q_lt_R_l555_555535

theorem P_lt_Q_lt_R (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  let P := real.sqrt (real.log a * real.log b)
  let Q := (1 / 2) * (real.log a + real.log b)
  let R := real.log ((a + b) / 2) 
  in P < Q ∧ Q < R :=
by
  let P := real.sqrt (real.log a * real.log b)
  let Q := (1 / 2) * (real.log a + real.log b)
  let R := real.log ((a + b) / 2)
  sorry

end P_lt_Q_lt_R_l555_555535


namespace numerator_of_fraction_l555_555617

-- Definition of conditions
variable (y : ℝ) (h_pos : y > 0)

-- Definition of the equation in the problem
def equation := y / 20 + (x / y) * y = 0.35 * y   -- Rewrite (x / y) * y as x

-- The theorem to prove
theorem numerator_of_fraction (x : ℝ) (h_eq : y / 20 + x = 0.35 * y) : x = 3 :=
by
  sorry

end numerator_of_fraction_l555_555617


namespace check_polynomial_c_is_square_of_binomial_l555_555335

def is_square_of_binomial (poly : ℚ) : Prop := 
  ∃ (a b : ℚ), poly = (a + b) * (a - b)

def polynomial_c := -((1 / 4) * x^2) + (1 / 9) * y^2

theorem check_polynomial_c_is_square_of_binomial (x y : ℚ) :
  is_square_of_binomial polynomial_c :=
sorry

end check_polynomial_c_is_square_of_binomial_l555_555335


namespace find_multiple_of_son_age_l555_555731

-- Definition of the conditions
variables (S k : ℕ)
def father_age := 27
def condition1 := father_age = k * S + 3
def condition2 := father_age + 3 = 2 * (S + 3) + 8

-- Statement to prove
theorem find_multiple_of_son_age
  (h1 : condition1)
  (h2 : condition2)
  : k = 3 :=
  sorry

end find_multiple_of_son_age_l555_555731


namespace polynomial_nonzero_coeff_l555_555252

open Polynomials

theorem polynomial_nonzero_coeff (P Q : Poly) (a : ℝ) (k : ℕ) (hP : P = (X - C a) ^ k * Q) (hk : k > 0) (ha : a ≠ 0) (hQ : Q ≠ 0) : 
  P.nonzero_coeff_count ≥ k + 1 := 
sorry

end polynomial_nonzero_coeff_l555_555252


namespace prob_b_eq_c_prob_roots_real_l555_555671

def set_P (b : ℕ) : Set ℕ := {b, 1}
def set_Q (c : ℕ) : Set ℕ := {c, 1, 2}
def possible_values : Set ℕ := {2, 3, 4, 5, 6}

def P_subset_Q (b c : ℕ) : Prop :=
  set_P(b) ⊆ set_Q(c)

def probability_b_eq_c (b c : ℕ) (h : P_subset_Q b c) : ℚ :=
  if b = c then 1 / 2 else 0

def discriminant (b c : ℕ) : ℤ :=
  b^2 - 4 * c

def prob_real_roots (b c : ℕ) (h : P_subset_Q b c) : ℚ :=
  if discriminant b c ≥ 0 then 3 / 8 else 0

theorem prob_b_eq_c (b c : ℕ) (h : P_subset_Q b c) : probability_b_eq_c b c h = 1 / 2 := 
  sorry

theorem prob_roots_real (b c : ℕ) (h : P_subset_Q b c) : prob_real_roots b c h = 3 / 8 := 
  sorry

end prob_b_eq_c_prob_roots_real_l555_555671


namespace probability_correct_l555_555808

namespace ProbabilitySongs

/-- Define the total number of ways to choose 2 out of 4 songs -/ 
def total_ways : ℕ := Nat.choose 4 2

/-- Define the number of ways to choose 2 songs such that neither A nor B is chosen (only C and D can be chosen) -/
def ways_without_AB : ℕ := Nat.choose 2 2

/-- The probability of playing at least one of A and B is calculated via the complementary rule -/
def probability_at_least_one_AB_played : ℚ := 1 - (ways_without_AB / total_ways)

theorem probability_correct : probability_at_least_one_AB_played = 5 / 6 := sorry
end ProbabilitySongs

end probability_correct_l555_555808


namespace total_money_spent_is_29_point_25_l555_555789

-- Define the conditions
variables (n : ℕ) (eighths_expenditure each_expenditure ninth_expenditure avg_expenditure total_expenditure : ℝ)

-- Given conditions
def condition1 : Prop := eighths_expenditure = 8 * 3
def condition2 : Prop := ninth_expenditure = avg_expenditure + 2
def condition3 : Prop := total_expenditure = eighths_expenditure + ninth_expenditure
def condition4 : Prop := 9 * avg_expenditure = total_expenditure

-- Prove that the total expenditure is Rs. 29.25
theorem total_money_spent_is_29_point_25 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) :
  total_expenditure = 29.25 :=
by
  sorry

end total_money_spent_is_29_point_25_l555_555789


namespace calculate_c10_l555_555486

noncomputable def sequence_c : ℕ → ℝ
| 1 := 3
| 2 := 4
| (n + 1) => 3 * sequence_c n * sequence_c (n - 1)

theorem calculate_c10 :
  sequence_c 10 = 3^(d_10) :=
sorry

end calculate_c10_l555_555486


namespace total_students_l555_555033

theorem total_students (a b c d e f : ℕ)  (h : a + b = 15) (h1 : a = 5) (h2 : b = 10) 
(h3 : c = 15) (h4 : d = 10) (h5 : e = 5) (h6 : f = 0) (h_total : a + b + c + d + e + f = 50) : a + b + c + d + e + f = 50 :=
by {exact h_total}

end total_students_l555_555033


namespace number_of_books_on_third_shelf_l555_555299

-- Definitions for the statements in the conditions
variables (x1 x2 x3 : ℕ)

-- Total number of books
def total_books := x1 + x2 + x3 = 275

-- Number of books on the third shelf
def third_shelf_books := x3 = 3 * x2 + 8

-- Number of books on the first shelf
def first_shelf_books := x1 = x2 / 2 - 3

-- Proof statement that the number of books on the third shelf is 188
theorem number_of_books_on_third_shelf
  (h1 : total_books)
  (h2 : third_shelf_books)
  (h3 : first_shelf_books) :
  x3 = 188 :=
sorry

end number_of_books_on_third_shelf_l555_555299


namespace measure_of_angle_B_sin_angle_BAC_l555_555178

-- Define the given conditions for the first question
variables (a b c : ℝ) (A B C : ℝ)
hypothesis h1 : 2 * b * Real.sin (C + (Real.pi / 6)) = a + c

-- Prove the measure of angle B is π / 3
theorem measure_of_angle_B (h1 : 2 * b * Real.sin (C + (Real.pi / 6)) = a + c) : 
  B = Real.pi / 3 :=
sorry

-- Define additional conditions for the second question
variables (M : point) (AM AC : ℝ)
hypothesis h2 : midpoint M B C
hypothesis h3 : distance AM = distance AC

-- Prove sin(angle BAC) = sqrt(21) / 7
theorem sin_angle_BAC (h1 : 2 * b * Real.sin (C + (Real.pi / 6)) = a + c)
  (h2 : midpoint M B C) (h3 : distance AM = distance AC) : 
  Real.sin A = Real.sqrt 21 / 7 :=
sorry

end measure_of_angle_B_sin_angle_BAC_l555_555178


namespace find_C2_l555_555125

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define the symmetry line
def sym_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the second circle C2 based on the symmetry
def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- The main theorem
theorem find_C2 (x y : ℝ) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, symm_point (x, y) (x', y') ∧ sym_line x' y') →
  C2 x y :=
sorry

end find_C2_l555_555125


namespace jake_sister_weight_ratio_l555_555989

theorem jake_sister_weight_ratio :
  ∀ (Jake_present_weight : ℕ) (total_weight : ℕ) (Jake_loss : ℕ),
  Jake_present_weight = 108 →
  total_weight = 156 →
  Jake_loss = 12 →
  let Jake_after_loss := Jake_present_weight - Jake_loss in
  let Sister_weight := total_weight - Jake_after_loss in
  (Jake_after_loss : ℚ) / Sister_weight = 8 / 5 :=
by
  intros Jake_present_weight total_weight Jake_loss h1 h2 h3
  simp [h1, h2, h3]
  let Jake_after_loss := 108 - 12
  let Sister_weight := 156 - Jake_after_loss
  have hJake_after_loss : Jake_after_loss = 96 := by rfl
  have hSister_weight : Sister_weight = 60 := by rfl
  simp [hJake_after_loss, hSister_weight]
  norm_num
  sorry

end jake_sister_weight_ratio_l555_555989


namespace reservoir_percentage_l555_555832

variable (C : ℝ)  -- C represents the total capacity of the reservoir
variable (original_contents added_water : ℝ)
variable (percentage_full_after : ℝ)

noncomputable def percentage_full_before_storm : ℝ :=
  let new_contents := original_contents + added_water
  let total_capacity := new_contents / percentage_full_after
  (original_contents / total_capacity) * 100

theorem reservoir_percentage (h1 : added_water = 120) (h2 : original_contents = 220) (h3 : percentage_full_after = 0.85) :
  percentage_full_before_storm C original_contents added_water percentage_full_after = 55 :=
by
  simp [percentage_full_before_storm, h1, h2, h3]
  norm_num
  sorry

end reservoir_percentage_l555_555832


namespace imaginary_part_of_i_l555_555117

open Complex

theorem imaginary_part_of_i : Im i = 1 :=
by sorry

end imaginary_part_of_i_l555_555117


namespace angle_AFE_is_170_l555_555042

-- Definitions of the geometric configurations
def square (A B C D : Point) : Prop := 
  (distance A B = distance B C) ∧ (distance B C = distance C D) ∧ 
  (distance C D = distance D A) ∧ (angle A B C = 90) ∧ 
  (angle B C D = 90) ∧ (angle C D A = 90) ∧ (angle D A B = 90)

def isosceles_triangle (D E F : Point) : Prop := distance D E = distance D F

-- Variables for points in the problem
variables (A B C D E F : Point)

-- Hypotheses for the given conditions in the problem
hypothesis (sq : square A B C D)
hypothesis (angle_CDE : angle C D E = 110)
hypothesis (isosceles : isosceles_triangle D E F)

-- The goal statement
theorem angle_AFE_is_170 :
  angle A F E = 170 := 
sorry

end angle_AFE_is_170_l555_555042


namespace radius_of_convergence_zeta_eq_e_l555_555652

noncomputable def xi (n : ℕ) : ℝ → ℝ := sorry -- Define the sequence of random variables

noncomputable def zeta (n : ℕ) : ℝ := ∏ i in finset.range (n+1), xi i 0

def radius_of_convergence (a : ℕ → ℝ) : ℝ :=
  1 / real.limsup (λ n, (real.abs (a n)) ^ (1 / (n:ℝ)))

theorem radius_of_convergence_zeta_eq_e :
  ∀ (ξ : ℕ → ℝ) (h_indep : ∀ i j, i ≠ j → probabilistic_independence (ξ i) (ξ j)) 
  (h_uniform : ∀ n, uniform ξ n),
  radius_of_convergence (λ n, ∏ i in finset.range (n + 1), ξ i) = real.exp 1 :=
by sorry

end radius_of_convergence_zeta_eq_e_l555_555652


namespace percentage_increase_area_is_21_l555_555991

variable (L W : ℝ)

def percentage_increase_area (L W : ℝ) : ℝ :=
  let A_original := L * W
  let L_new := 1.10 * L
  let W_new := 1.10 * W
  let A_new := L_new * W_new
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase

theorem percentage_increase_area_is_21 (L W : ℝ) : percentage_increase_area L W = 21 := by
  unfold percentage_increase_area
  sorry

end percentage_increase_area_is_21_l555_555991


namespace number_of_tables_l555_555182

theorem number_of_tables (total_legs chairs_per_table chair_legs table_legs : ℕ) (total_legs = 352) (chairs_per_table = 4) (chair_legs = 4) (table_legs = 4) :
  ∃ t : ℕ, 4 * (chairs_per_table * t) + 4 * t = total_legs ∧ t = 18 := by
sorry

end number_of_tables_l555_555182


namespace PE_tangent_to_Omega_l555_555645

noncomputable def is_midpoint (E A C : Point) : Prop :=
  dist A E = dist E C

noncomputable def circumcircle (A B E : Point) : Circle :=
  -- Assuming a function that defines the circumcircle from three points
  sorry

noncomputable def is_tangent (P E : Point) (Ω : Circle) : Prop :=
  -- Assuming a predicate that checks if a point P is tangent to a circle Ω
  sorry

theorem PE_tangent_to_Omega
  {A B C D E P : Point}
  (ABCD_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
  (AB_parallel_CD : is_parallel A B C D)
  (E_is_midpoint_AC : is_midpoint E A C)
  (omega : Circle := circumcircle A B E)
  (Omega : Circle := circumcircle C D E)
  (P_is_intersection : is_intersection P (tangent_line omega A) (tangent_line Omega D)) :
  is_tangent P E Omega :=
sorry

end PE_tangent_to_Omega_l555_555645


namespace infinite_series_value_l555_555479

open BigOperators

noncomputable def infinite_series := 
  ∑' (n : ℕ) in set.Ici 3, (n^4 + 5 * n^2 + 8 * n + 12) / (2^n * (n^4 + 9))

theorem infinite_series_value :
  infinite_series = 1 / 4 := 
sorry

end infinite_series_value_l555_555479


namespace find_a_for_binomial_square_l555_555883

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l555_555883


namespace parabola_symmetric_points_l555_555560

theorem parabola_symmetric_points (a : ℝ) (x1 y1 x2 y2 m : ℝ) 
  (h_parabola : ∀ x, y = a * x^2)
  (h_a_pos : a > 0)
  (h_focus_directrix : 1 / (2 * a) = 1 / 4)
  (h_symmetric : y1 = a * x1^2 ∧ y2 = a * x2^2 ∧ ∃ m, y1 = m + (x1 - m))
  (h_product : x1 * x2 = -1 / 2) :
  m = 3 / 2 := 
sorry

end parabola_symmetric_points_l555_555560


namespace find_a_for_square_binomial_l555_555880

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l555_555880


namespace value_of_m_l555_555984

theorem value_of_m (m : ℤ) : (∃ (f : ℤ → ℤ), ∀ x : ℤ, x^2 + m * x + 16 = (f x)^2) ↔ (m = 8 ∨ m = -8) := 
by
  sorry

end value_of_m_l555_555984


namespace angle_bisector_le_median_in_triangle_l555_555689

theorem angle_bisector_le_median_in_triangle 
  (a b c : ℝ) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  let m_a := sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4),
      l_a := sqrt (a * b * ((a + b)^2 - c^2) / ((a + b)^2)) in
  l_a <= m_a :=
by
  sorry

end angle_bisector_le_median_in_triangle_l555_555689


namespace prob_event_A_prob_event_not_B_l555_555179

-- Define number of factories
def num_factories := 5

-- Define number of days in the week
def num_days := 7

-- 1. Define the event A: all 5 factories choose Sunday to shut down
def event_A :=
  ∀ (choices : Fin num_factories → Fin num_days), choices = fun _ => 0

-- Prove the probability of event A
theorem prob_event_A :
  (Prob.event_A = 1 / num_days ^ num_factories) :=
by
  sorry

-- 2. Define the event B: all 5 factories choose different days 
def event_B :=
  ∃ (choices : Fin num_factories → Fin num_days), Function.injective choices

-- Define the complement of event B: at least two factories choose the same day
def event_not_B :=
  ¬event_B

-- Prove the probability of event_not_B
theorem prob_event_not_B :
  (Prob.event_not_B = 1 - A(num_days, num_factories) / num_days ^ num_factories) :=
by
  sorry

end prob_event_A_prob_event_not_B_l555_555179


namespace find_general_formula_l555_555200

section sequence

variables {R : Type*} [LinearOrderedField R]
variable (c : R)
variable (h_c : c ≠ 0)

def seq (a : Nat → R) : Prop :=
  a 1 = 1 ∧ ∀ n : Nat, n > 0 → a (n + 1) = c * a n + c^(n + 1) * (2 * n + 1)

def general_formula (a : Nat → R) : Prop :=
  ∀ n : Nat, n > 0 → a n = (n^2 - 1) * c^n + c^(n - 1)

theorem find_general_formula :
  ∃ a : Nat → R, seq c a ∧ general_formula c a :=
by
  sorry

end sequence

end find_general_formula_l555_555200


namespace max_green_beads_l555_555362

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end max_green_beads_l555_555362


namespace polar_to_rectangular_coordinates_l555_555061

theorem polar_to_rectangular_coordinates 
  (r θ : ℝ) 
  (hr : r = 7) 
  (hθ : θ = 7 * Real.pi / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2) := 
by
  sorry

end polar_to_rectangular_coordinates_l555_555061


namespace regular_polygon_perimeter_l555_555375

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555375


namespace unique_solution_iff_a_values_l555_555859

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 5 * a

theorem unique_solution_iff_a_values (a : ℝ) :
  (∃! x : ℝ, |f x a| ≤ 3) ↔ (a = 3 / 4 ∨ a = -3 / 4) :=
by
  sorry

end unique_solution_iff_a_values_l555_555859


namespace problem1_problem2_problem3_problem4_l555_555842

theorem problem1 : (70.8 - 1.25 - 1.75 = 67.8) := sorry

theorem problem2 : ((8 + 0.8) * 1.25 = 11) := sorry

theorem problem3 : (125 * 0.48 = 600) := sorry

theorem problem4 : (6.7 * (9.3 * (6.2 + 1.7)) = 554.559) := sorry

end problem1_problem2_problem3_problem4_l555_555842


namespace correct_propositions_count_l555_555964

-- Defining the planes and lines
variables (a b : Plane) (m n : Line)

-- Proposition 1: If m is parallel to n and m is perpendicular to a, then n is perpendicular to a
def prop1 := m.parallel n → m.perpendicular_to a → n.perpendicular_to a

-- Proposition 2: If m is perpendicular to a and m is perpendicular to b, then a is parallel to b
def prop2 := m.perpendicular_to a → m.perpendicular_to b → a.parallel b

-- Proposition 3: If m is perpendicular to a, m is parallel to n, and n intersects b, then a is perpendicular to b
def prop3 := m.perpendicular_to a → m.parallel n → n.intersects b → a.perpendicular_to b

-- Proposition 4: If m is parallel to a, and the intersection of a and b is n, then m is parallel to n
def prop4 := m.parallel_to a → (a ∩ b = n) → m.parallel_to n

-- Proposition list and count of correct propositions.
def correct_props := [prop1, prop2, prop3, ¬prop4]

-- Proof that exactly three propositions are correct
theorem correct_propositions_count :
  count_correct correct_props = 3 :=
sorry

end correct_propositions_count_l555_555964


namespace circles_divide_plane_l555_555184

def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  ∀ (i j : Fin n), i ≠ j → (∃ p : ℝ × ℝ, p ∈ circle i ∧ p ∈ circle j) ∧
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ (∃ p : ℝ × ℝ, p ∈ circle i ∧ p ∈ circle j ∧ p ∈ circle k)

theorem circles_divide_plane (n : ℕ) (h : satisfies_conditions n) : 
  ℕ :=
begin
  sorry
end

end circles_divide_plane_l555_555184


namespace find_ac_l555_555664

variables {R : Type*} [linear_ordered_field R]

def f (a b x : R) : R := x^2 + a * x + b
def g (c d x : R) : R := x^2 + c * x + d

theorem find_ac (a b c d : R) (h1 : b = -2650 - 50 * a)
                               (h2 : d = -2650 - 50 * c)
                               (h3 : - (a^2 / 4) + b = -9)
                               (h4 : - (c^2 / 4) + d = -4)
                               (h5 : g c d (-a / 2) = 0)
                               (h6 : f a b (-c / 2) = 0)
                               (h7 : f a b 50 = -150)
                               (h8 : g c d 50 = -150) :
    a + c = -800 :=
sorry

end find_ac_l555_555664


namespace factorial_expression_equals_l555_555466

theorem factorial_expression_equals :
  7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end factorial_expression_equals_l555_555466


namespace perimeter_of_regular_polygon_l555_555415

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555415


namespace solution_I_solution_II_l555_555790

noncomputable def problem_I : Prop :=
  let white_ball_prob := (2 : ℝ) / 5
  let black_ball_prob := (3 : ℝ) / 5
  let diff_color_prob := white_ball_prob * black_ball_prob + black_ball_prob * white_ball_prob
  diff_color_prob = 12 / 25

noncomputable def problem_II : Prop :=
  let P_xi_0 := (3 : ℝ) / 5 * (2 / 4)
  let P_xi_1 := (3 / 5) * (2 / 4) + (2 / 5) * (3 / 4)
  let P_xi_2 := (2 / 5) * (1 / 4)
  let E_xi := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2
  let Var_xi := (0 - E_xi) ^ 2 * P_xi_0 + (1 - E_xi) ^ 2 * P_xi_1 + (2 - E_xi) ^ 2 * P_xi_2
  E_xi = 4 / 5 ∧ Var_xi = 9 / 25

theorem solution_I : problem_I := 
by sorry

theorem solution_II : problem_II := 
by sorry

end solution_I_solution_II_l555_555790


namespace retail_price_eq_120_l555_555338

noncomputable def retail_price : ℝ :=
  let W := 90
  let P := 0.20 * W
  let SP := W + P
  SP / 0.90

theorem retail_price_eq_120 : retail_price = 120 := by
  sorry

end retail_price_eq_120_l555_555338


namespace original_proposition_true_converse_proposition_false_l555_555561

theorem original_proposition_true (a b : ℝ) : 
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := 
sorry

theorem converse_proposition_false : 
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_proposition_false_l555_555561


namespace designed_height_correct_l555_555332
noncomputable def designed_height_of_lower_part (H : ℝ) (L : ℝ) : Prop :=
  H = 2 ∧ (H - L) / L = L / H

theorem designed_height_correct : ∃ L, designed_height_of_lower_part 2 L ∧ L = Real.sqrt 5 - 1 :=
by
  sorry

end designed_height_correct_l555_555332


namespace binary_to_decimal_101101_l555_555855

theorem binary_to_decimal_101101 :
  let b := [1, 0, 1, 1, 0, 1] in
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0 in
  decimal = 45 :=
by
  let b := [1, 0, 1, 1, 0, 1];
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0;
  show decimal = 45;
  sorry

end binary_to_decimal_101101_l555_555855


namespace sine_shift_left_symmetry_l555_555260

theorem sine_shift_left_symmetry :
  ∀ x, f x = sin (x + (π / 2)) → f (x + π) = - f x :=
begin
  intro x,
  intro hx,
  rw hx,
  rw [sin_add, cos_add],
  have hπ : sin (π / 2) = 1 := by sorry,
  have h0 : cos (π / 2) = 0 := by sorry,
  simp [hπ, h0],
  sorry -- Further steps needed to finalize proof
end

end sine_shift_left_symmetry_l555_555260


namespace find_c_of_binomial_square_l555_555982

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l555_555982


namespace total_amount_paid_correct_l555_555016

/--
Given:
1. The marked price of each article is $17.5.
2. A discount of 30% was applied to the total marked price of the pair of articles.

Prove:
The total amount paid for the pair of articles is $24.5.
-/
def total_amount_paid (marked_price_each : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price_pair := marked_price_each * 2
  let discount := discount_rate * marked_price_pair
  marked_price_pair - discount

theorem total_amount_paid_correct :
  total_amount_paid 17.5 0.30 = 24.5 :=
by
  sorry

end total_amount_paid_correct_l555_555016


namespace find_value_l555_555936

theorem find_value (
  a b c d e f : ℝ) 
  (h1 : a * b * c = 65) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 1000) 
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := 
sorry

end find_value_l555_555936


namespace new_sequence_after_removal_is_geometric_l555_555909

theorem new_sequence_after_removal_is_geometric (a : ℕ → ℝ) (a₁ q : ℝ) (k : ℕ)
  (h_geo : ∀ n, a n = a₁ * q ^ n) :
  ∀ n, (a (n + k)) = a₁ * q ^ (n + k) :=
by
  sorry

end new_sequence_after_removal_is_geometric_l555_555909


namespace regular_polygon_perimeter_l555_555443

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555443


namespace holiday_not_on_22nd_l555_555809

def isThirdWednesday (d : ℕ) : Prop :=
  d = 15 ∨ d = 16 ∨ d = 17 ∨ d = 18 ∨ d = 19 ∨ d = 20 ∨ d = 21

theorem holiday_not_on_22nd :
  ¬ isThirdWednesday 22 :=
by
  intro h
  cases h
  repeat { contradiction }

end holiday_not_on_22nd_l555_555809


namespace line_conversion_circle_conversion_min_distance_from_circle_to_line_l555_555188

noncomputable def line_parametric_equations (t : ℝ) : ℝ × ℝ :=
  (2 + t, sqrt 3 - sqrt 3 * t)

noncomputable def line_standard (x y : ℝ) : Prop :=
  sqrt 3 * x + y = 3 * sqrt 3

noncomputable def circle_polar (ρ θ : ℝ) : Prop :=
  ρ + 2 * cos θ = 0

noncomputable def circle_cartesian (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + y ^ 2 = 1

noncomputable def center_of_circle : ℝ × ℝ :=
  (-1, 0)

noncomputable def point_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a ^ 2 + b ^ 2)

noncomputable def minimum_distance : ℝ :=
  2 * sqrt 3 - 1

theorem line_conversion (t : ℝ) (x y : ℝ) :
  (line_parametric_equations t = (x, y)) ↔ line_standard x y :=
sorry

theorem circle_conversion (ρ θ x y : ℝ) :
  (circle_polar ρ θ) ↔ (circle_cartesian x y) :=
sorry

theorem min_distance_from_circle_to_line :
  point_line_distance (sqrt 3) 1 (-3 * sqrt 3) (-1) 0 = minimum_distance :=
sorry

end line_conversion_circle_conversion_min_distance_from_circle_to_line_l555_555188


namespace find_ST_length_l555_555826

-- Definitions for the problem
variables {P Q R S T U V W X Y : Type} -- Defining points

-- Given conditions
variables (QR_length : ℝ) (QR_length_eq_fifteen : QR_length = 15)
variables (projected_area_fraction : ℝ) (projected_area_fraction_eq_point_two_five : projected_area_fraction = 0.25)
variables (height_ratio : ℝ) (height_ratio_eq_half : height_ratio = 0.5)

-- The statement to prove
theorem find_ST_length (QR_length_eq_fifteen : QR_length = 15) 
                       (projected_area_fraction_eq_point_two_five : projected_area_fraction = 0.25) 
                       (height_ratio_eq_half : height_ratio = 0.5) : 
  let ST_length := 0.75 * QR_length in 
  ST_length = 11.25 :=
by
  -- main proof starts here
  sorry -- proof goes here

end find_ST_length_l555_555826


namespace divides_5n_4n_iff_n_is_multiple_of_3_l555_555905

theorem divides_5n_4n_iff_n_is_multiple_of_3 (n : ℕ) (h : n > 0) : 
  61 ∣ (5^n - 4^n) ↔ ∃ k : ℕ, n = 3 * k :=
by
  sorry

end divides_5n_4n_iff_n_is_multiple_of_3_l555_555905


namespace angle_bisector_le_median_in_triangle_l555_555691

theorem angle_bisector_le_median_in_triangle 
  (a b c : ℝ) (triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  let m_a := sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4),
      l_a := sqrt (a * b * ((a + b)^2 - c^2) / ((a + b)^2)) in
  l_a <= m_a :=
by
  sorry

end angle_bisector_le_median_in_triangle_l555_555691


namespace max_different_numbers_in_rect_l555_555804

-- Definitions and assumptions for the problem
def is_magic_square (a b c d e f g h i : ℕ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧ a + b + c = a + d + g ∧ a + d + g = c + f + i ∧ a + e + i = c + e + g

def is_magic_square_in_rect (rect : ℕ → ℕ → ℕ) (x y : ℕ) : Prop :=
  is_magic_square (rect x y) (rect (x+1) y) (rect (x+2) y)
                  (rect x (y+1)) (rect (x+1) (y+1)) (rect (x+2) (y+1))
                  (rect x (y+2)) (rect (x+1) (y+2)) (rect (x+2) (y+2))

def all_3x3_sub_squares_magic (m n : ℕ) (rect : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, x + 2 < m → y + 2 < n → is_magic_square_in_rect rect x y

-- Main theorem statement
theorem max_different_numbers_in_rect (m n : ℕ) (rect : ℕ → ℕ → ℕ) :
  (m = 3 ∧ n = 3 → ∃ f : ℕ → ℕ → ℕ → bool, ∀ i j : ℕ, f rect i j = true → i ≠ j) ∧
  ((m > 3 ∨ n > 3) → ∀ i j, rect i j = rect 0 0)
:= by
  sorry

end max_different_numbers_in_rect_l555_555804


namespace limit_problem_l555_555053

open Real

noncomputable def limit_cos_cos_tan_squared (x : ℝ) : Prop :=
  tendsto (λ x, (cos (3 * x) - cos x) / (tan (2 * x))^2) (𝓝 π) (𝓝 1)

theorem limit_problem : limit_cos_cos_tan_squared π :=
sorry

end limit_problem_l555_555053


namespace perimeter_of_polygon_l555_555434

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555434


namespace f_decreasing_l555_555648

variable {n : ℕ} (a : Fin n → ℝ)

def f (x : ℝ) : ℝ :=
  ∑ i in Finset.range n, (a i + x) / (a ((i + 1) % n) + x)

theorem f_decreasing (hn : 0 < n) (ha : ∀ i, 0 < a i) : ∀ x y, 0 ≤ x → x < y → f a x ≥ f a y :=
by
  sorry

end f_decreasing_l555_555648


namespace new_train_distance_l555_555806

theorem new_train_distance (old_train_distance : ℕ) (additional_factor : ℕ) (h₀ : old_train_distance = 300) (h₁ : additional_factor = 50) :
  let new_train_distance := old_train_distance + (additional_factor * old_train_distance / 100)
  new_train_distance = 450 :=
by
  sorry

end new_train_distance_l555_555806


namespace five_spheres_iff_regular_l555_555251

variable {S A B C : Point}
variable (tetrahedron : Tetrahedron S A B C)

axiom five_spheres_touch_edges (tetrahedron : Tetrahedron S A B C) :
  ∃ (insphere : Sphere) (exsphere1 exsphere2 exsphere3 exsphere4 : Sphere), 
  (∀ (edge : Edge), touching_sphere insphere edge) ∧
  (∀ (i : Fin 4), (∃ (vertex : Point), touching_sphere (exsphere i) (face_opposite tetrahedron vertex)))

noncomputable def is_regular_tetrahedron (tetrahedron : Tetrahedron S A B C) : Prop :=
  tetrahedron.edge_lengths.equal

theorem five_spheres_iff_regular :
  five_spheres_touch_edges tetrahedron ↔ is_regular_tetrahedron tetrahedron :=
sorry

end five_spheres_iff_regular_l555_555251


namespace expansion_eq_coeff_sum_l555_555551

theorem expansion_eq_coeff_sum (a : ℕ → ℤ) (m : ℤ) 
  (h : (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7)
  (h_coeff : a 4 = -35) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1 ∧ a 1 + a 3 + a 5 + a 7 = 26 := 
by 
  sorry

end expansion_eq_coeff_sum_l555_555551


namespace cost_price_one_meter_l555_555032

variable (SP80 : ℝ) (ppmeter : ℝ)

theorem cost_price_one_meter (h1 : SP80 = 6900) (h2 : ppmeter = 20) :
  let profit_total := ppmeter * 80 in
  let CP80 := SP80 - profit_total in
  CP80 / 80 = 66.25 :=
by
  sorry

end cost_price_one_meter_l555_555032


namespace regular_polygon_perimeter_l555_555388

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555388


namespace find_c_l555_555979

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l555_555979


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555315

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555315


namespace proof_x_value_l555_555612

theorem proof_x_value : (∃ x : ℕ, (2^4)*(3^6) = 9*(6^x)) → x = 4 :=
by
  intro h
  cases h with x hx
  sorry

end proof_x_value_l555_555612


namespace largest_pies_without_any_ingredients_l555_555641

-- Define the conditions
def total_pies : ℕ := 60
def pies_with_strawberries : ℕ := total_pies / 4
def pies_with_bananas : ℕ := total_pies * 3 / 8
def pies_with_cherries : ℕ := total_pies / 2
def pies_with_pecans : ℕ := total_pies / 10

-- State the theorem to prove
theorem largest_pies_without_any_ingredients : (total_pies - pies_with_cherries) = 30 := by
  sorry

end largest_pies_without_any_ingredients_l555_555641


namespace pete_ten_dollar_bills_l555_555249

theorem pete_ten_dollar_bills (owes dollars bills: ℕ) (bill_value_per_bottle : ℕ) (num_bottles : ℕ) (ten_dollar_bills : ℕ):
  owes = 90 →
  dollars = 40 →
  bill_value_per_bottle = 5 →
  num_bottles = 20 →
  dollars + (num_bottles * bill_value_per_bottle) + (ten_dollar_bills * 10) = owes →
  ten_dollar_bills = 4 :=
by
  sorry

end pete_ten_dollar_bills_l555_555249


namespace shared_vertex_triangles_l555_555642

theorem shared_vertex_triangles :
  ∀ (n m : ℕ), (n = 10) → (m = 7) → ∀ (triangles : Finset (Finset (ℕ × ℕ))), 
  (triangles.card = 30) → 
  (∀ t ∈ triangles, t.card = 3) → 
  ∃ t₁ t₂ ∈ triangles, t₁ ≠ t₂ ∧ (∃ v, v ∈ t₁ ∧ v ∈ t₂) :=
begin
  intros n m hn hm triangles hcard htriangle,
  have n_lines := n + 1,
  have m_lines := m + 1,
  have inter_points := n_lines * m_lines,
  have vertices_needed := 30 * 3,
  have h1 : inter_points < vertices_needed := by {
    have n_lines_eq : n_lines = 11 := by linarith,
    have m_lines_eq : m_lines = 8 := by linarith,
    have calc1 : inter_points = 11 * 8 := by {rw [n_lines_eq, m_lines_eq]},
    have calc2 : vertices_needed = 90 := by linarith,
    rw calc1,
    rw calc2,
    linarith,
  },
  exfalso,
  apply nat.not_lt_of_ge,
  sorry
end

end shared_vertex_triangles_l555_555642


namespace fraction_sum_squares_eq_sixteen_l555_555170

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l555_555170


namespace bijective_condition1_bijective_condition2_neither_injective_nor_surjective_condition3_neither_injective_nor_surjective_condition4_l555_555515

-- 1. Bijectivity from f(f(x) - 1) = x + 1
theorem bijective_condition1 (f : ℝ → ℝ) (h : ∀ x, f (f x - 1) = x + 1) : function.bijective f := sorry

-- 2. Bijectivity from f(x + f(y)) = f(x) + y^5
theorem bijective_condition2 (f : ℝ → ℝ) (h : ∀ x y, f (x + f y) = f x + y^5) : function.bijective f := sorry

-- 3. Neither injective nor surjective from f(f(x)) = sin x
theorem neither_injective_nor_surjective_condition3 (f : ℝ → ℝ) (h : ∀ x, f (f x) = sin x) : ¬ (function.injective f) ∧ ¬ (function.surjective f) := sorry

-- 4. Neither injective nor surjective from Δf(x + y²) = f(x)f(y) + x f(y) - y³ f(x)
theorem neither_injective_nor_surjective_condition4 (f : ℝ → ℝ) (h : ∀ x y, Δf(x + y^2) = f(x) * f(y) + x * f(y) - y^3 * f(x)) : ¬ (function.injective f) ∧ ¬ (function.surjective f) := sorry

end bijective_condition1_bijective_condition2_neither_injective_nor_surjective_condition3_neither_injective_nor_surjective_condition4_l555_555515


namespace regular_polygon_perimeter_l555_555368

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555368


namespace find_f_neg_2_l555_555934

variable (f g : ℝ → ℝ)

-- Conditions
def f_def (x : ℝ) : Prop := f x = g x + 2
def g_odd : Prop := ∀ x, g (-x) = -g x
def f_at_2 : Prop := f 2 = 3

-- Theorem statement
theorem find_f_neg_2 (h1 : f_def f g) (h2 : g_odd g) (h3 : f_at_2 f) : f (-2) = 1 :=
by
  sorry

end find_f_neg_2_l555_555934


namespace regular_polygon_perimeter_l555_555376

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555376


namespace reflection_image_l555_555721

theorem reflection_image (m b : ℝ) :
  (∃ m b, ∀ (P Q : ℝ × ℝ), P = (2, 2) → Q = (10, 6) →
    let x_m : ℝ := (P.1 + Q.1) / 2
        y_m : ℝ := (P.2 + Q.2) / 2
        m := -2
        b := y_m - m * x_m
    in y_m = m * x_m + b) → m + b = 14 :=
by
  sorry

end reflection_image_l555_555721


namespace new_concentration_is_37_percent_l555_555773

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end new_concentration_is_37_percent_l555_555773


namespace smallest_portion_quantity_l555_555270

-- Define the conditions for the problem
def conditions (a1 a2 a3 a4 a5 d : ℚ) : Prop :=
  a2 = a1 + d ∧
  a3 = a1 + 2 * d ∧
  a4 = a1 + 3 * d ∧
  a5 = a1 + 4 * d ∧
  5 * a1 + 10 * d = 100 ∧
  (a3 + a4 + a5) = (1/7) * (a1 + a2)

-- Lean theorem statement
theorem smallest_portion_quantity : 
  ∃ (a1 a2 a3 a4 a5 d : ℚ), conditions a1 a2 a3 a4 a5 d ∧ a1 = 5 / 3 :=
by
  sorry

end smallest_portion_quantity_l555_555270


namespace snowfall_difference_l555_555839

def baldMountainSnowfallMeters : ℝ := 1.5
def billyMountainSnowfallMeters : ℝ := 3.5
def mountPilotSnowfallCentimeters : ℝ := 126
def cmPerMeter : ℝ := 100

theorem snowfall_difference :
  billyMountainSnowfallMeters * cmPerMeter + mountPilotSnowfallCentimeters - baldMountainSnowfallMeters * cmPerMeter = 326 :=
by
  sorry

end snowfall_difference_l555_555839


namespace sin_alpha_sub_beta_cos_beta_l555_555130

variables (α β : ℝ)
variables (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
variables (h1 : Real.sin α = 3 / 5)
variables (h2 : Real.tan (α - β) = -1 / 3)

theorem sin_alpha_sub_beta : Real.sin (α - β) = - Real.sqrt 10 / 10 :=
by
  sorry

theorem cos_beta : Real.cos β = 9 * Real.sqrt 10 / 50 :=
by
  sorry

end sin_alpha_sub_beta_cos_beta_l555_555130


namespace perimeter_of_regular_polygon_l555_555405

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555405


namespace find_second_expression_l555_555707

theorem find_second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 84) (h₂ : a = 32) : x = 88 :=
  sorry

end find_second_expression_l555_555707


namespace sum_of_number_and_square_is_306_l555_555615

theorem sum_of_number_and_square_is_306 (n : ℕ) (h : n = 17) : n + n^2 = 306 :=
by
  sorry

end sum_of_number_and_square_is_306_l555_555615


namespace sin_105_deg_l555_555477

theorem sin_105_deg : 
  sin (105 * (real.pi / 180)) = 
  (real.sqrt 2 + real.sqrt 6) / 4 :=
by
  -- Using the angle addition formula for sine
  have h1 : 105 = 45 + 60 := by norm_num
  rw [h1, real.sin_add]
  -- Use known values
  have h2 : sin (45 * (real.pi / 180)) = real.sqrt 2 / 2 := by rw [real.sin_pi_div_four]
  have h3 : cos (45 * (real.pi / 180)) = real.sqrt 2 / 2 := by rw [real.cos_pi_div_four]
  have h4 : cos (60 * (real.pi / 180)) = 1 / 2 := by rw [real.cos_pi_div_three]
  have h5 : sin (60 * (real.pi / 180)) = real.sqrt 3 / 2 := by rw [real.sin_pi_div_three]
  rw [h2, h4, h3, h5]
  -- Simplify
  sorry

end sin_105_deg_l555_555477


namespace largest_constant_ineq_l555_555092

theorem largest_constant_ineq (a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
    sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e))
    + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e))
    + sqrt (e / (a + b + c + d)) > 2 :=
sorry

end largest_constant_ineq_l555_555092


namespace length_PQ_slope_PQ_max_min_MQ_l555_555522

noncomputable def circle : set (ℝ × ℝ) := { p | let ⟨x, y⟩ := p in x^2 + y^2 - 4 * x - 14 * y + 45 = 0 }
def Q : ℝ × ℝ := (-2, 3)

theorem length_PQ (P : ℝ × ℝ) (A : ℝ) (hP : P = (A, A + 1)) (hP_on_circle : P ∈ circle) : 
  dist P Q = 2 * real.sqrt 10 := sorry

theorem slope_PQ (P : ℝ × ℝ) (A : ℝ) (hP : P = (A, A + 1)) (hP_on_circle : P ∈ circle) :
  (P.2 - Q.2) / (P.1 - Q.1) = 1 / 3 := sorry

theorem max_min_MQ (M : ℝ × ℝ) (hM : M ∈ circle) : 
  let r := 2 * real.sqrt 2
  let CQ := dist (2, 7) Q
  CQ = 4 * real.sqrt 2 →
  ∃ max_min : ℝ × ℝ, max_min = (6 * real.sqrt 2, 2 * real.sqrt 2) := sorry

end length_PQ_slope_PQ_max_min_MQ_l555_555522


namespace inverse_matrix_correct_l555_555894

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1, 2, 3],
    ![0, -1, 2],
    ![3, 0, 7]
  ]

def A_inv_correct : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-1/2, -1, 1/2],
    ![3/7, -1/7, -1/7],
    ![3/14, 3/7, -1/14]
  ]

theorem inverse_matrix_correct : A⁻¹ = A_inv_correct := by
  sorry

end inverse_matrix_correct_l555_555894


namespace parallelogram_area_l555_555851

-- Definitions
def equation1 (z : ℂ) : Prop := z^2 = 8 + 8 * real.sqrt 7 * complex.i
def equation2 (z : ℂ) : Prop := z^2 = 3 + 3 * real.sqrt 3 * complex.i

-- Theorem to prove the area of the parallelogram formed by the solutions
theorem parallelogram_area :
  (∃ z1 z2 z3 z4 : ℂ, equation1 z1 ∧ equation1 z2 ∧ equation2 z3 ∧ equation2 z4) →
  abs (-(real.sqrt 146) + 2 * real.sqrt 63) = 2 * real.sqrt 63 - real.sqrt 146 := sorry

end parallelogram_area_l555_555851


namespace geometry_problem_l555_555234

variables {A B C D E F G X Y Z P Q : Type*}
variables [triangle ABC] [midpoints D E F] [intersection A D X] [circle_tangent Omega]
variables [tangents B C circumcircle] [line PG]

-- Definitions for midpoints
def midpoint (X Y : Point) : Point := sorry

-- Definitions for intersection
def intersection (l₁ l₂ : Line) : Point := sorry

-- Definitions for circle and tangent
def circle_tangent (Ω : Circle) (P Q : Point) : Prop := sorry
def circumcircle (ABC : Triangle) : Circle := sorry

-- Definitions for centroid
def centroid (ABC : Triangle) : Point := sorry

-- Define perpendicular bisectors and intersection points
def is_perpendicular_bisector (l : Line) (X Y : Point) : Prop := sorry
def tangent (Ω : Circle) (X : Point) : Line := sorry

-- Main theorem
theorem geometry_problem : 
  ∃ Q : Point,
  tangents B C circumcircle ∧ line P G ∧ concurrent :=
sorry

end geometry_problem_l555_555234


namespace find_a_for_square_binomial_l555_555877

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l555_555877


namespace abigail_collected_43_l555_555034

noncomputable def cans_needed : ℕ := 100
noncomputable def collected_by_alyssa : ℕ := 30
noncomputable def more_to_collect : ℕ := 27
noncomputable def collected_by_abigail : ℕ := cans_needed - (collected_by_alyssa + more_to_collect)

theorem abigail_collected_43 : collected_by_abigail = 43 := by
  sorry

end abigail_collected_43_l555_555034


namespace surface_area_ratio_l555_555007

theorem surface_area_ratio (a : ℝ) : let a' := 1.5 * a in
  let r := (a' * Real.sqrt 3) / 2 in
  let S_cube := 6 * a' ^ 2 in
  let S_sphere := 4 * Real.pi * r ^ 2 in
  S_sphere / S_cube = Real.pi / 2 :=
by
  -- Variables and relationships are already defined in the let bindings above.
  -- The proof is not required as per the instructions.
  sorry

end surface_area_ratio_l555_555007


namespace purchasing_methods_count_l555_555802

theorem purchasing_methods_count :
  {p : ℕ × ℕ // (60 * p.1 + 70 * p.2 ≤ 500) ∧ (p.1 ≥ 3) ∧ (p.2 ≥ 2)}.card = 7 :=
by
  -- Proof skipped
  sorry

end purchasing_methods_count_l555_555802


namespace total_pints_l555_555865

variables (Annie Kathryn Ben Sam : ℕ)

-- Conditions
def condition1 := Annie = 16
def condition2 (Annie : ℕ) := Kathryn = 2 * Annie + 2
def condition3 (Kathryn : ℕ) := Ben = Kathryn / 2 - 3
def condition4 (Ben Kathryn : ℕ) := Sam = 2 * (Ben + Kathryn) / 3

-- Statement to prove
theorem total_pints (Annie Kathryn Ben Sam : ℕ) 
  (h1 : condition1 Annie) 
  (h2 : condition2 Annie Kathryn) 
  (h3 : condition3 Kathryn Ben) 
  (h4 : condition4 Ben Kathryn Sam) : 
  Annie + Kathryn + Ben + Sam = 96 :=
sorry

end total_pints_l555_555865


namespace negation_of_proposition_l555_555956

theorem negation_of_proposition (p : Prop) : 
  (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0) ↔ ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end negation_of_proposition_l555_555956


namespace log_product_l555_555077

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_product (x y : ℝ) (hx : 0 < x) (hy : 1 < y) :
  log_base (y^3) x * log_base (x^4) (y^3) * log_base (y^5) (x^2) * log_base (x^2) (y^5) * log_base (y^3) (x^4) =
  (1/3) * log_base y x :=
by
  sorry

end log_product_l555_555077


namespace isosceles_triangle_base_length_l555_555729

theorem isosceles_triangle_base_length (P B : ℕ) (hP : P = 13) (hB : B = 3) :
    ∃ S : ℕ, S ≠ 3 ∧ S = 3 :=
by
    sorry

end isosceles_triangle_base_length_l555_555729


namespace algebraic_expression_value_l555_555589

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l555_555589


namespace problem_statement_l555_555927

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 0 }

-- Define a point P which moves such that |PA| = √2|PB|
def moving_point (P : Point) : Prop :=
  dist P A = real.sqrt 2 * dist P B

-- Define the equation of the trajectory curve
def curve_C (P : Point) : Prop :=
  (P.x - 3)^2 + P.y^2 = 8

-- Parabola equation
def parabola (Q : Point) : Prop :=
  Q.y^2 = Q.x

-- Center of symmetry of curve C
def center_of_symmetry : Point := { x := 3, y := 0 }

-- Function to calculate distance between two points
noncomputable def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Function to find the shortest distance from a point on the parabola to the center of symmetry
noncomputable def shortest_distance : ℝ :=
  let m := 5 / 2 in
  let Q := { x := m, y := real.sqrt m } in
  dist Q center_of_symmetry

-- Theorem statement
theorem problem_statement :
  (∃ P, moving_point P ∧ curve_C P) ∧ shortest_distance = real.sqrt 11 / 2 :=
by
  sorry

end problem_statement_l555_555927


namespace inradius_of_isosceles_triangle_l555_555225

theorem inradius_of_isosceles_triangle (BC : ℝ) (AB AC IC : ℝ) (h_iso : AB = AC) (BC_eq : BC = 40) (IC_eq : IC = 24) : ∃ r : ℝ, r = 2 * Real.sqrt 44 :=
by
  -- Define the conditions
  let CD := BC / 2
  have CO_eq : CD = 20 := by linarith

  -- Compute the inradius r using Pythagorean theorem in triangle IDC
  let r := Real.sqrt (IC ^ 2 - CD ^ 2)
  have r_eq : r = Real.sqrt 176 := by sorry
  
  -- Conclude that r = 2 * sqrt 44
  use r
  rw [r_eq]
  sorry

end inradius_of_isosceles_triangle_l555_555225


namespace angle_of_cone_is_correct_l555_555681

-- Define the setup of the problem
def sphere (r : ℝ) (cntr : ℝ × ℝ × ℝ) := {p : ℝ × ℝ × ℝ // dist cntr p = r}

noncomputable def sphere1 := sphere 2 (0, 0, 0)
noncomputable def sphere2 := sphere 2 (4, 0, 0)
noncomputable def sphere3 := sphere 1 (2, 2 * sqrt 3, 0)

-- Define the position of the cone's tip
noncomputable def cone_tip : ℝ × ℝ × ℝ := (2, sqrt 3, sqrt (8))

-- Define the function to calculate the angle at the tip of the cone
noncomputable def angle_at_tip (p1 p2 : ℝ × ℝ × ℝ) (cone_tip : ℝ × ℝ × ℝ) : ℝ :=
  2 * Real.arctan (8)

-- Problem statement to prove the angle
theorem angle_of_cone_is_correct :
  angle_at_tip (0, 0, 0) (4, 0, 0) cone_tip = 2 * Real.arccos (1 / sqrt 65) :=
sorry

end angle_of_cone_is_correct_l555_555681


namespace brody_calculator_battery_life_l555_555841

theorem brody_calculator_battery_life (h : ∃ t : ℕ, (3 / 4) * t + 2 + 13 = t) : ∃ t : ℕ, t = 60 :=
by
  -- Define the quarters used by Brody and the remaining battery life after the exam.
  obtain ⟨t, ht⟩ := h
  -- Simplify the equation (3/4) * t + 2 + 13 = t to get t = 60
  sorry

end brody_calculator_battery_life_l555_555841


namespace eccentricity_min_value_l555_555945

variable (a b x y : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
variable (point_on_ellipse : (x, y) = (3, 2))

theorem eccentricity_min_value :
  (x = 3) → (y = 2) → (a > b) → (a > 0) → (b > 0) → 
  ∀ (e : ℝ), a^2 + b^2 = 25 → e = Real.sqrt (1 - (b^2 / a^2)) → 
  e = (Real.sqrt 3) / 3 := 
begin
  intros,
  sorry
end

end eccentricity_min_value_l555_555945


namespace sequence_geometric_l555_555528

theorem sequence_geometric {a_n : ℕ → ℕ} (S : ℕ → ℕ) (a1 a2 a3 : ℕ) 
(hS : ∀ n, S n = 2 * a_n n - a_n 1) 
(h_arith : 2 * (a_n 2 + 1) = a_n 3 + a_n 1) : 
  ∀ n, a_n n = 2 ^ n :=
sorry

end sequence_geometric_l555_555528


namespace two_cos_45_eq_sqrt_two_l555_555296

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l555_555296


namespace count_special_numbers_l555_555704

theorem count_special_numbers : 
  let digits := { d : ℕ | 1 ≤ d ∧ d ≤ 9 },
      tens := { t : ℕ | 0 ≤ t ∧ t ≤ 9 } in
  ∃ n : ℕ, 
    (∀ h u t : ℕ, h ∈ digits ∧ u ∈ digits ∧ t ∈ tens → 
      n = 100 * h + 10 * t + u → n = 100 * u + 10 * t + h) → 
    n = 90 := 
by
  sorry

end count_special_numbers_l555_555704


namespace smallest_acute_angle_right_triangle_l555_555023

theorem smallest_acute_angle_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a / b = 3 / 4) (h2 : a / c = 3 / 5) :
  (real.arctan (3 / 4)) = 36.87 :=
by
  sorry

end smallest_acute_angle_right_triangle_l555_555023


namespace range_of_f_symmetry_axis_and_monotonic_intervals_l555_555164

noncomputable def f (ω x : ℝ) : ℝ :=
  (sin (ω * x) + cos (ω * x)) * (sin (ω * x) - cos (ω * x)) + 2 * sqrt 3 * sin (ω * x) * cos (ω * x) + 1

theorem range_of_f (ω : ℝ) (hω : 0 < ω ∧ ω < 2) :
  ω = 1 → (∀ x, x ∈ set.Icc 0 (π / 2) → 0 ≤ f ω x ∧ f ω x ≤ 3) :=
begin
  sorry
end

theorem symmetry_axis_and_monotonic_intervals (ω : ℝ) (hω : 0 < ω ∧ ω < 2) :
  ω = 1 →
  (∀ k : ℤ, ∃ x : ℝ,
  (x = (k * π + π / 6) / 2) ∧
  (∀ x, x ∈ set.Ioo (k * π - π / 6) (k * π + π / 3) → f ω x < f ω (x + ε) + 1) ∧
  (∀ x, x ∈ set.Ioo (k * π + π / 3) (k * π + 5 * π / 6) → f ω x > f ω (x + ε) - 1)
  ) :=
begin
  sorry
end

end range_of_f_symmetry_axis_and_monotonic_intervals_l555_555164


namespace regular_polygon_perimeter_l555_555386

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555386


namespace smallest_x_undefined_l555_555764

theorem smallest_x_undefined :
  ∀ (x : ℝ), (6 * x^2 - 55 * x + 35 = 0) → (∃ (y : ℝ), y = 0.6875 ∧ is_less_than y x) := sorry

end smallest_x_undefined_l555_555764


namespace general_form_of_equation_l555_555484

theorem general_form_of_equation (x : ℝ) : 2 * x ^ 2 - 1 = 6 * x → 2 * x ^ 2 - 6 * x - 1 = 0 :=
by
  assume h : 2 * x ^ 2 - 1 = 6 * x
  sorry

end general_form_of_equation_l555_555484


namespace area_of_triangle_is_correct_l555_555706

-- Define the curve as a function
def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + x

-- Define the point where tangent is calculated
def point : ℝ × ℝ := (1, 4 / 3)

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := x^2 + 1

-- Define the equation of the tangent line at the given point
def tangent_line (x : ℝ) : ℝ := 2 * x - 2 / 3

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := 1 / 3

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := -2 / 3

-- Proof problem statement to prove that the area of the triangle formed is 1/9
theorem area_of_triangle_is_correct : 
  (1 / 2) * abs (1 - (1 / 3)) * abs ((-2 / 3) - (4 / 3)) = 1 / 9 :=
by
  sorry

end area_of_triangle_is_correct_l555_555706


namespace basketball_scores_distinct_count_l555_555794

theorem basketball_scores_distinct_count :
  ∀ (x : ℕ), 0 ≤ x ∧ x ≤ 7 → ∃! (P : ℕ), (∃ y : ℕ, y = 7 - x ∧ P = 3*x + 2*y) ∧ (P ∈ {14, 15, 16, 17, 18, 19, 20, 21}) :=
by sorry

end basketball_scores_distinct_count_l555_555794


namespace geometric_series_sum_eq_l555_555054

-- Definitions of the geometric series properties
def a : ℚ := 2
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Statement of the theorem to prove the sum of the series
theorem geometric_series_sum_eq :
  let S_5 := a * (1 - r^n) / (1 - r) in
  S_5 = 341 / 128 :=
by
  -- Proof omitted
  sorry

end geometric_series_sum_eq_l555_555054


namespace number_of_primes_in_list_l555_555169

-- Define the list of numbers
def numbers : List ℕ := [11, 12, 13, 14, 15, 16, 17]

-- Check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Count the number of primes in the list
def prime_count (lst : List ℕ) : ℕ :=
  (lst.filter is_prime).length

theorem number_of_primes_in_list :
  prime_count numbers = 3 :=
by
  sorry

end number_of_primes_in_list_l555_555169


namespace time_to_fix_shirt_l555_555211

theorem time_to_fix_shirt :
  ∃ (h : ℝ), 
    let n_s := 10 in
    let n_p := 12 in
    let rate := 30 in
    let total_cost := 1530 in
    let cost_shirts := n_s * h * rate in
    let cost_pants := n_p * 2 * h * rate in
    cost_shirts + cost_pants = total_cost ∧ h = 1.5 :=
begin
  existsi (1.5 : ℝ),
  let n_s := 10,
  let n_p := 12,
  let rate := 30,
  let total_cost := 1530,
  let cost_shirts := n_s * 1.5 * rate,
  let cost_pants := n_p * 2 * 1.5 * rate,
  have h1 : cost_shirts = 10 * 1.5 * 30 := by norm_num,
  have h2 : cost_pants = 12 * 2 * 1.5 * 30 := by norm_num,
  have h3 : cost_shirts + cost_pants = 1530 := by norm_num,
  exact ⟨h3, by norm_num⟩,
  sorry,
end

end time_to_fix_shirt_l555_555211


namespace find_general_term_l555_555161

theorem find_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a (n + 1) = 2 * a n + n - 1) :
  ∀ n, a n = 2^n - n :=
begin
  -- proof goes here
  sorry
end

end find_general_term_l555_555161


namespace regular_polygon_perimeter_l555_555440

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555440


namespace exists_prime_infinitely_many_n_l555_555301

theorem exists_prime_infinitely_many_n (m : ℕ) : ∃ p, nat.prime p ∧ ∃n : ℕ, (∃ k : ℕ, n = k^2) ∧ (√(p + n) + √n) ∈ ℤ := 
sorry

end exists_prime_infinitely_many_n_l555_555301


namespace measure_angle_BDC_l555_555113

-- We have a triangle ABC and line CE bisects ∠C
variables {A B C E D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq E] [DecidableEq D]
variables (triangle_ABC : Triangle A B C)
variables (CE_bisects_angle_C : Bisects E C (Angle A C B))
variables (AB_extended_to_D : Extends A B D)
variables (angle_EBC_30 : Measures (Angle E B C) (30 : ℝ))
variables (angle_C_90 : Measures (Angle A C B) (90 : ℝ))

-- We need to prove the measure of angle BDC is 150 degrees
theorem measure_angle_BDC : Measures (Angle B D C) (150 : ℝ) :=
sorry

end measure_angle_BDC_l555_555113


namespace value_of_a5_l555_555525

/-- Define the sequence recursively -/
def a : ℕ → ℚ
| 0 := 1/2
| (n+1) := 1 - 1 / a n

/-- Prove the value of a_5 -/
theorem value_of_a5 : a 4 = -1 :=
by
  unfold a,
  norm_num,
  have a2 : a 1 = -1 := by norm_num,
  have a3 : a 2 = 2 := by norm_num [a2],
  have a4 : a 3 = 1/2 := by norm_num [a3],
  exact a4
-- sorry

end value_of_a5_l555_555525


namespace perimeter_of_regular_polygon_l555_555420

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555420


namespace maximize_abs_diff_l555_555926
-- Import the necessary library

-- Define the points A, B, and the x-axis condition for point P
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 3 }
def B : Point := { x := 5, y := -2 }
def P : Point := { x := 13, y := 0 }

-- Define the distance function
def dist (P1 P2 : Point) : ℝ :=
  real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

-- Define the absolute difference function to be maximized
def abs_diff (A B P : Point) : ℝ :=
  |dist A P - dist B P|

-- Define the theorem that states the condition to be proven
theorem maximize_abs_diff : 
  ∀ (P : Point), P.y = 0 → (P = { x := 13, y := 0 } ↔ abs_diff A B P = abs_diff A B { x := 13, y := 0 }) := 
by 
  intros P Px0
  sorry

end maximize_abs_diff_l555_555926


namespace average_coins_collected_per_day_l555_555259

noncomputable def average_coins (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (a + (a + (n - 1) * d)) / 2

theorem average_coins_collected_per_day :
  average_coins 10 5 7 = 25 := by
  sorry

end average_coins_collected_per_day_l555_555259


namespace algebraic_expression_value_l555_555585

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l555_555585


namespace pentagon_perimeter_ratio_l555_555779

-- Definitions
def is_equilateral_triangle (T : Type) [triangle T] : Prop :=
  ∀ (a b c : segment T), a.length = b.length ∧ b.length = c.length ∧ c.length = a.length

def is_equilateral_pentagon (P : Type) [pentagon P] : Prop :=
  ∀ (p1 p2 p3 p4 p5 : segment P), 
    p1.length = p2.length ∧ p2.length = p3.length ∧ p3.length = p4.length ∧ p4.length = p5.length ∧ p5.length = p1.length

-- The main theorem statement
theorem pentagon_perimeter_ratio (T : Type) [triangle T] [is_equilateral_triangle T] 
  (P : Type) [pentagon P] [is_equilateral_pentagon P]
  (a b : ℝ) (ha : a = 5 * (2 * real.sqrt 3 - 3)) (hb : b = 5 * (1 / 2)) : 
  a / b = 4 * real.sqrt 3 - 6 :=
by 
  sorry

end pentagon_perimeter_ratio_l555_555779


namespace minor_premise_is_statement1_l555_555499

-- Define the statements
def statement1 : Prop := "A square is a parallelogram"
def statement2 : Prop := "A parallelogram has opposite sides equal"
def statement3 : Prop := "A square has opposite sides equal"

-- Define the propositions for the major, minor premise and conclusion
def is_major_premise (p: Prop) : Prop := p = statement2
def is_minor_premise (p: Prop) : Prop := p = statement1
def is_conclusion (p: Prop) : Prop := p = statement3

-- Main theorem
theorem minor_premise_is_statement1 : is_minor_premise statement1 :=
by
  unfold is_minor_premise
  apply Eq.refl

end minor_premise_is_statement1_l555_555499


namespace balls_in_boxes_l555_555577

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l555_555577


namespace train_length_250_meters_l555_555825

open Real

noncomputable def speed_in_ms (speed_km_hr: ℝ): ℝ :=
  speed_km_hr * (1000 / 3600)

noncomputable def length_of_train (speed: ℝ) (time: ℝ): ℝ :=
  speed * time

theorem train_length_250_meters (speed_km_hr: ℝ) (time_seconds: ℝ) :
  speed_km_hr = 40 → time_seconds = 22.5 → length_of_train (speed_in_ms speed_km_hr) time_seconds = 250 :=
by
  intros
  sorry

end train_length_250_meters_l555_555825


namespace unbiased_efficiency_binomial_dist_l555_555651

noncomputable def unbiased_estimator (ξ : ℕ) (n : ℕ) : ℝ :=
  ξ / n

def E_theta_T (θ : ℝ) (n : ℕ) : Prop :=
  ∀ (ξ : ℕ), 0 ≤ ξ ∧ ξ ≤ n → (θ * n + (1 - θ) * n - ξ = 0)

def efficient_unbiased_estimator_class (ξ : ℕ) (θ : ℝ) (n : ℕ) : Prop :=
  (∀ (T : ℕ → ℝ) (∀ ξ : ℕ, unbiased_estimator ξ n = T ξ → (θ * θ - ξ = 0))) ∧
  (∀ (T' : ℕ → ℝ) ((∀ ξ : ℕ, unbiased_estimator ξ n = T' ξ) → 
  ∃ T : ℕ → ℝ, unbiased_estimator ξ n = T ξ ∧ (θ * θ - T ξ = 0)))

theorem unbiased_efficiency_binomial_dist 
  (ξ : ℕ) (θ : ℝ) (n : ℕ) (h1 : E_theta_T θ n) 
  (h2 : θ ∈ set.Icc (1 / 4) (3 / 4)) (h3 : n = 3) :
  ∀ (ξ : ℕ), 
  θ * θ < unbiased_estimator ξ n :=
sorry

end unbiased_efficiency_binomial_dist_l555_555651


namespace regular_polygon_perimeter_l555_555424

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555424


namespace find_a_for_binomial_square_l555_555885

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l555_555885


namespace sin_double_angle_l555_555140

theorem sin_double_angle (α : ℝ) 
  (h : ∃ P : ℝ × ℝ, P = (-4, -3) ∧ terminal_side α P) : 
  sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l555_555140


namespace peanut_butter_candy_count_l555_555304

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l555_555304


namespace find_b_l555_555292

theorem find_b (a b c : ℤ) (h1 : a + b + c = 120) (h2 : a + 4 = b - 12) (h3 : a + 4 = 3 * c) : b = 60 :=
sorry

end find_b_l555_555292


namespace final_position_total_fuel_consumption_main_l555_555013

-- Define the team's travel records and fuel consumption rate
def records : List ℤ := [-4, 7, -9, 8, 6, -5, -2]
def fuel_rate : ℚ := 0.3

-- Theorem to prove the final position from point A
theorem final_position (recs: List ℤ) : recs.sum = 1 := 
by
  -- given recs = [-4, 7, -9, 8, 6, -5, -2]
  have h : List.sum recs = List.sum [-4, 7, -9, 8, 6, -5, -2] := by rfl
  rw [h]
  have := by norm_num
  -- We know that the sum of the elements [-4, 7, -9, 8, 6, -5, -2] is 1
  rw [this]
  exact rfl

-- Theorem to prove the total fuel consumption
theorem total_fuel_consumption (recs: List ℤ) : recs.map Int.natAbs.sum * fuel_rate = 12.3 :=
by
  -- given recs = [-4, 7, -9, 8, 6, -5, -2]
  have h : recs.map Int.natAbs = [-4, 7, -9, 8, 6, -5, -2].map Int.natAbs := by rfl
  rw [h]
  calc
    List.sum ([4, 7, 9, 8, 6, 5, 2] : List ℕ) * fuel_rate
        = (4 + 7 + 9 + 8 + 6 + 5 + 2) * fuel_rate : rfl
    ... = 41 * fuel_rate : by norm_num
    ... = 12.3 : by norm_num

-- Define the main proof problem combining both theorems
theorem main : records.sum = 1 ∧ records.map Int.natAbs.sum * fuel_rate = 12.3 :=
by 
  constructor
  · apply final_position
  · apply total_fuel_consumption
  done


end final_position_total_fuel_consumption_main_l555_555013


namespace sample_size_correct_l555_555800

theorem sample_size_correct :
  ∃ n,
    let full_professors := 120
    let associate_professors := 100
    let lecturers := 80
    let teaching_assistants := 60
    let selected_lecturers := 16
    let total_teachers := full_professors + associate_professors + lecturers + teaching_assistants
    let probability_per_lecturer := selected_lecturers / lecturers
    n = total_teachers * probability_per_lecturer 
    n = 72 :=
sorry

end sample_size_correct_l555_555800


namespace ship_length_l555_555074

theorem ship_length (E S L : ℕ) (h1 : 150 * E = L + 150 * S) (h2 : 90 * E = L - 90 * S) : 
  L = 24 :=
by
  sorry

end ship_length_l555_555074


namespace min_elements_even_subset_l555_555029

theorem min_elements_even_subset (n : ℕ) : ∃ k, (∀ subset : Finset (Fin n × Fin n),
  subset.card = k → (∃ even_subset : Finset (Fin n × Fin n),
    even_subset ⊆ subset ∧
    (∀ i : Fin n, even_subset.filter (λ p, p.1 = i).card % 2 = 0) ∧
    (∀ j : Fin n, even_subset.filter (λ p, p.2 = j).card % 2 = 0))) ∧ k = 2 * n - 1 :=
by
  sorry

end min_elements_even_subset_l555_555029


namespace frequencies_and_confidence_level_l555_555320

namespace MachineQuality

-- Definitions of the given conditions
def productsA := 200
def firstClassA := 150
def secondClassA := 50

def productsB := 200
def firstClassB := 120
def secondClassB := 80

def totalProducts := productsA + productsB
def totalFirstClass := firstClassA + firstClassB
def totalSecondClass := secondClassA + secondClassB

-- 1. Frequencies of first-class products
def frequencyFirstClassA := firstClassA / productsA
def frequencyFirstClassB := firstClassB / productsB

-- 2. \( K^2 \) calculation
def n := 400
def a := 150
def b := 50
def c := 120
def d := 80

def K_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- The theorem to prove the frequencies and the confidence level
theorem frequencies_and_confidence_level : 
    frequencyFirstClassA = (3 / 4) ∧ frequencyFirstClassB = (3 / 5) ∧ K_squared > 6.635 := 
    by {
        sorry -- Proof steps go here
    }

end MachineQuality

end frequencies_and_confidence_level_l555_555320


namespace angle_ACB_is_25_l555_555197

theorem angle_ACB_is_25 (angle_ABD angle_BAC : ℝ) (is_supplementary : angle_ABD + (180 - angle_BAC) = 180) (angle_ABC_eq : angle_BAC = 95) (angle_ABD_eq : angle_ABD = 120) :
  180 - (angle_BAC + (180 - angle_ABD)) = 25 :=
by
  sorry

end angle_ACB_is_25_l555_555197


namespace prove_values_l555_555155

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 1/x + b

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem prove_values (a b : ℝ) (h1 : a > 0) (h2 : is_integer b) :
  (f a b (Real.log a) = 6 ∧ f a b (Real.log (1 / a)) = 2) ∨
  (f a b (Real.log a) = -2 ∧ f a b (Real.log (1 / a)) = 2) :=
sorry

end prove_values_l555_555155


namespace set_C_is_basis_l555_555035

def is_basis (v1 v2 : ℝ × ℝ) : Prop :=
  ¬ ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

def set_A : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 0), (1, -2))
def set_B : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, -2), (3, 6))
def set_C : (ℝ × ℝ) × (ℝ × ℝ) := ((3, -5), (6, 10))
def set_D : (ℝ × ℝ) × (ℝ × ℝ) := ((2, -3), (-2, 3))

theorem set_C_is_basis : is_basis (3, -5) (6, 10) :=
by {
  sorry
}

end set_C_is_basis_l555_555035


namespace find_constant_term_l555_555501

noncomputable def constant_term_expansion : ℕ := 581

theorem find_constant_term :
  (constant_term ((x + (2 / x) + 1) ^ 6)) = 581 := 
  by
    sorry

end find_constant_term_l555_555501


namespace survey_total_parents_l555_555819

theorem survey_total_parents (P : ℝ)
  (h1 : 0.15 * P + 0.60 * P + 0.20 * 0.25 * P + 0.05 * P = P)
  (h2 : 0.05 * P = 6) : 
  P = 120 :=
sorry

end survey_total_parents_l555_555819


namespace cost_to_open_store_l555_555644

-- Define the conditions as constants
def revenue_per_month : ℕ := 4000
def expenses_per_month : ℕ := 1500
def months_to_payback : ℕ := 10

-- Theorem stating the cost to open the store
theorem cost_to_open_store : (revenue_per_month - expenses_per_month) * months_to_payback = 25000 :=
by
  sorry

end cost_to_open_store_l555_555644


namespace sum_first_8_terms_of_geom_seq_l555_555133

-- Definitions: the sequence a_n, common ratio q, and the fact that specific terms form an arithmetic sequence.
def geom_seq (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) := ∀ n, a n = a1 * q^(n-1)
def arith_seq (b c d : ℕ) := 2 * b + (c - 2 * b) = d

-- Conditions
variables {a : ℕ → ℕ} {a1 : ℕ} {q : ℕ}
variables (h1 : geom_seq a a1 q) (h2 : q = 2)
variables (h3 : arith_seq (2 * a 4) (a 6) 48)

-- Goal: sum of the first 8 terms of the sequence equals 255
def sum_geometric_sequence (a1 : ℕ) (q : ℕ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem sum_first_8_terms_of_geom_seq : 
  sum_geometric_sequence a1 q 8 = 255 :=
by
  sorry

end sum_first_8_terms_of_geom_seq_l555_555133


namespace max_green_beads_l555_555365

/-- Define the problem conditions:
  1. A necklace consists of 100 beads of red, blue, and green colors.
  2. Among any five consecutive beads, there is at least one blue bead.
  3. Among any seven consecutive beads, there is at least one red bead.
  4. The beads in the necklace are arranged cyclically (the last one is adjacent to the first one).
--/
def necklace_conditions := 
  ∀ (beads : ℕ → ℕ) (n : ℕ), 
    (∀ i, 0 ≤ beads i ∧ beads i < 3) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 5 ∧ beads j = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 7 ∧ beads j = 0)

/-- Prove the maximum number of green beads that can be in this necklace is 65. --/
theorem max_green_beads : ∃ beads : (ℕ → ℕ), necklace_conditions beads → (∑ p, if beads p = 2 then 1 else 0) = 65 :=
by sorry

end max_green_beads_l555_555365


namespace find_max_value_l555_555558

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x + a

theorem find_max_value (a x : ℝ) (h_min : f 1 a = 1) : 
  ∃ x : ℝ, f (-1/3) 2 = 59/27 :=
by {
  sorry
}

end find_max_value_l555_555558


namespace ball_bounce_height_l555_555791

theorem ball_bounce_height (b : ℕ) (h₀: ℝ) (r: ℝ) (h_final: ℝ) :
  h₀ = 200 ∧ r = 3 / 4 ∧ h_final = 25 →
  200 * (3 / 4) ^ b < 25 ↔ b ≥ 25 := by
  sorry

end ball_bounce_height_l555_555791


namespace problem_proof_l555_555600

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l555_555600


namespace regular_polygon_perimeter_l555_555374

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555374


namespace factorization_problem1_factorization_problem2_l555_555871

-- Define the first problem: Factorization of 3x^2 - 27
theorem factorization_problem1 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) :=
by
  sorry 

-- Define the second problem: Factorization of (a + 1)(a - 5) + 9
theorem factorization_problem2 (a : ℝ) : (a + 1) * (a - 5) + 9 = (a - 2) ^ 2 :=
by
  sorry

end factorization_problem1_factorization_problem2_l555_555871


namespace fgf_of_3_l555_555654

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 4 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

-- The statement we need to prove
theorem fgf_of_3 : f (g (f 3)) = 332 := by
  sorry

end fgf_of_3_l555_555654


namespace algebraic_expression_identity_l555_555591

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l555_555591


namespace max_value_of_f_l555_555506

noncomputable def f : ℝ → ℝ := λ x, 6 * x - 2 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 / 2 :=
by
  sorry

end max_value_of_f_l555_555506


namespace problem_i_l555_555345

theorem problem_i (n : ℕ) (h : n ≥ 1) : n ∣ 2^n - 1 ↔ n = 1 := by
  sorry

end problem_i_l555_555345


namespace center_square_side_length_l555_555272

theorem center_square_side_length (s : ℝ) :
    let total_area := 120 * 120
    let l_shape_area := (5 / 24) * total_area
    let l_shape_total_area := 4 * l_shape_area
    let center_square_area := total_area - l_shape_total_area
    s^2 = center_square_area → s = 49 :=
by
  intro total_area l_shape_area l_shape_total_area center_square_area h
  sorry

end center_square_side_length_l555_555272


namespace regular_polygon_perimeter_is_28_l555_555395

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555395


namespace centroid_sum_reciprocal_squares_eq_l555_555223

theorem centroid_sum_reciprocal_squares_eq
  (α β γ : ℝ)
  (O : ℝ × ℝ × ℝ := (0, 0, 0))
  (A : ℝ × ℝ × ℝ := (α, 0, 0))
  (B : ℝ × ℝ × ℝ := (0, β, 0))
  (C : ℝ × ℝ × ℝ := (0, 0, γ))
  (hαβγ : α + β + γ = 6)
  (h_dist : (1 / (sqrt (1 / α^2 + 1 / β^2 + 1 / γ^2))) = 2)
  : (let p := α / 3; let q := β / 3; let r := γ / 3 
     in 1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 :=
by
  sorry

end centroid_sum_reciprocal_squares_eq_l555_555223


namespace simplify_expression_l555_555263

variable (y : ℝ)

theorem simplify_expression : 3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := 
  by
  sorry

end simplify_expression_l555_555263


namespace essay_count_problem_l555_555107

noncomputable def eighth_essays : ℕ := sorry
noncomputable def seventh_essays : ℕ := sorry

theorem essay_count_problem (x : ℕ) (h1 : eighth_essays = x) (h2 : seventh_essays = (1/2 : ℚ) * x - 2) (h3 : eighth_essays + seventh_essays = 118) : 
  seventh_essays = 38 :=
sorry

end essay_count_problem_l555_555107


namespace regular_polygon_perimeter_l555_555442

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555442


namespace geometric_figure_properties_l555_555837

noncomputable def surface_area_after_cylinder_removed (cube_edge_length radius_cylinder height_cylinder : ℕ) (π : ℝ) : ℝ :=
  let cube_surface_area := 6 * (cube_edge_length ^ 2)
  let cylinder_end_area := 2 * π * (radius_cylinder ^ 2)
  let cylinder_lateral_area := 2 * π * radius_cylinder * height_cylinder
  cube_surface_area - cylinder_end_area + cylinder_lateral_area

noncomputable def volume_after_cylinder_removed (cube_edge_length radius_cylinder height_cylinder : ℕ) (π : ℝ) : ℝ :=
  let cube_volume := cube_edge_length ^ 3
  let cylinder_volume := π * (radius_cylinder ^ 2) * height_cylinder
  cube_volume - cylinder_volume

theorem geometric_figure_properties :
  surface_area_after_cylinder_removed 10 2 10 3 = 696 ∧ volume_after_cylinder_removed 10 2 10 3 = 880 :=
by
  sorry

end geometric_figure_properties_l555_555837


namespace smallest_abs_value_l555_555226

variable (a b c : ℤ) (ω : ℂ)
variable (h1 : a ≠ b)
variable (h2 : a ≠ c)
variable (h3 : b ≠ c)
variable (hω1 : ω^4 = 1)
variable (hω2 : ω ≠ 1)

theorem smallest_abs_value (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_omega_fourth : ω^4 = 1) (h_omega_not_one : ω ≠ 1) :
  ∃ (m : ℝ), m = |a + b*ω + c*ω^3| ∧ m = 1 :=
sorry

end smallest_abs_value_l555_555226


namespace minimum_value_fraction_l555_555719

theorem minimum_value_fraction (m n : ℝ) (h1 : m + 4 * n = 1) (h2 : m > 0) (h3 : n > 0): 
  (1 / m + 4 / n) ≥ 25 :=
sorry

end minimum_value_fraction_l555_555719


namespace isosceles_largest_angle_eq_60_l555_555623

theorem isosceles_largest_angle_eq_60 :
  ∀ (A B C : ℝ), (
    -- Condition: A triangle is isosceles with two equal angles of 60 degrees.
    ∀ (x y : ℝ), A = x ∧ B = x ∧ C = y ∧ x = 60 →
    -- Prove that
    max A (max B C) = 60 ) :=
by
  intros A B C h
  -- Sorry denotes skipping the proof.
  sorry

end isosceles_largest_angle_eq_60_l555_555623


namespace length_of_angle_bisector_leq_median_l555_555685

-- Define the entities and conditions
variables {A B C D M : Type} [in_triangl ABC]
variables (a b : ℝ)

-- Define the Length Bisector and Median properties
def AngleBisectorTheorem (A B C D : Type) :=
  ∀ (b : ℝ) (a : ℝ), (AD / DB) = (b / a)

def MedianBisects (A B C : Type) :=
  ∀ (A B : ℝ), A = B

-- The main theorem to prove
theorem length_of_angle_bisector_leq_median (A B C D M : Type) [in_triangl ABC] [AngleBisectorTheorem A B C D (b : ℝ) (a : ℝ)] [MedianBisects C M (A : ℝ) (B : ℝ)] : 
  CD ≤ CM :=
sorry

end length_of_angle_bisector_leq_median_l555_555685


namespace range_of_x_l555_555106

theorem range_of_x {x : ℝ} (h : ∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → (1 / 8) * (2 * x - x^2) ≤ t^2 - 3 * t + 2 ∧ t^2 - 3 * t + 2 ≤ 3 - x^2) : 
  -1 ≤ x ∧ x ≤ 1 - real.sqrt 3 :=
by
  sorry

end range_of_x_l555_555106


namespace triangle_sequence_relation_l555_555966

theorem triangle_sequence_relation (b d c k : ℤ) (h₁ : b % d = 0) (h₂ : c % k = 0) (h₃ : b^2 + (b + 2*d)^2 = (c + 6*k)^2) :
  c = 0 :=
sorry

end triangle_sequence_relation_l555_555966


namespace semicircle_triangle_t_squared_l555_555783

theorem semicircle_triangle_t_squared {R : ℝ} {D E F : Point} (hDE_diameter : dist D E = 2 * R) 
  (hDEF_right_triangle : ∠DEF = π / 2) (hF_on_semicircle : ∀ F, dist F (midpoint D E) = R ∧ F ≠ D ∧ F ≠ E) :
  ∀ (t : ℝ), t = dist D F + dist E F -> t^2 = 8 * R^2 :=
by
  sorry

end semicircle_triangle_t_squared_l555_555783


namespace min_value_inequality_l555_555157

theorem min_value_inequality (a b : ℝ) (h : ∀ x : ℝ, (ln (x + 1)) - (a + 2) * x ≤ b - 2) : 
  ∃ (t > 0), (a + 2 = t ∧ t = 1 / exp 1 ∧ (b - 3) / (a + 2) = 1 - exp 1) :=
sorry

end min_value_inequality_l555_555157


namespace total_bike_route_length_l555_555275

theorem total_bike_route_length : 
  let horizontal_segments := [4, 7, 2] in
  let vertical_segments := [6, 7] in
  let total_length := 2 * (horizontal_segments.sum + vertical_segments.sum) in
  total_length = 52 :=
by
  let horizontal_segments := [4, 7, 2]
  let vertical_segments := [6, 7]
  let total_length := 2 * (horizontal_segments.sum + vertical_segments.sum)
  have h_sum_horizontal : horizontal_segments.sum = 13 :=
    by sorry  -- Detailed sum calculation skipped
  have h_sum_vertical : vertical_segments.sum = 13 :=
    by sorry  -- Detailed sum calculation skipped
  calc
    total_length
      = 2 * (horizontal_segments.sum + vertical_segments.sum) : by rfl
  ... = 2 * (13 + 13) : by rw [h_sum_horizontal, h_sum_vertical]
  ... = 2 * 26 : by rfl
  ... = 52 : by rfl

end total_bike_route_length_l555_555275


namespace polygon_perimeter_l555_555379

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555379


namespace volume_of_cube_l555_555279

theorem volume_of_cube
  (s : ℝ) 
  (h : s * sqrt 2 = 5 * sqrt 2) :
  s^3 = 125 :=
by
  sorry

end volume_of_cube_l555_555279


namespace perimeter_of_regular_polygon_l555_555404

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555404


namespace inverse_logarithm_function_l555_555913

variable (x : ℝ)

def f (a x : ℝ) : ℝ := Real.logb a x

def inv_f (a : ℝ) (y : ℝ) : ℝ := a ^ y

theorem inverse_logarithm_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : inv_f a (-1) = 2) : inv_f a x = (1/2) ^ x :=
by
  sorry

end inverse_logarithm_function_l555_555913


namespace min_value_at_3_l555_555331

-- Define the function f(x) = x^2 - 6x + 13
def f (x : ℝ) : ℝ := x^2 - 6*x + 13

-- State the theorem that the minimum value of f(x) is achieved at x = 3
theorem min_value_at_3 : ∃ c : ℝ, c = 3 ∧ ∀ x : ℝ, f(c) ≤ f(x) := 
sorry

end min_value_at_3_l555_555331


namespace packets_of_gum_is_eight_l555_555677

-- Given conditions
def pieces_left : ℕ := 2
def pieces_chewed : ℕ := 54
def pieces_per_packet : ℕ := 7

-- Given he chews all the gum except for pieces_left pieces, and chews pieces_chewed pieces at once
def total_pieces_of_gum (pieces_chewed pieces_left : ℕ) : ℕ :=
  pieces_chewed + pieces_left

-- Calculate the number of packets
def number_of_packets (total_pieces pieces_per_packet : ℕ) : ℕ :=
  total_pieces / pieces_per_packet

-- The final theorem asserting the number of packets is 8
theorem packets_of_gum_is_eight : number_of_packets (total_pieces_of_gum pieces_chewed pieces_left) pieces_per_packet = 8 :=
  sorry

end packets_of_gum_is_eight_l555_555677


namespace regular_polygon_perimeter_is_28_l555_555399

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555399


namespace problem_proof_l555_555602

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l555_555602


namespace apple_weights_standard_deviation_l555_555109

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length)

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

noncomputable def standard_deviation (data : List ℝ) : ℝ :=
  real.sqrt (variance data)

theorem apple_weights_standard_deviation :
  standard_deviation [125, 124, 121, 123, 127] = 2 := by
  sorry

end apple_weights_standard_deviation_l555_555109


namespace quadratic_vertex_axis_of_symmetry_l555_555502

theorem quadratic_vertex_axis_of_symmetry :
  let y := λ x : ℝ, x^2 - 2 * x - 3
  in (y 1 = -4) ∧ (∀ x : ℝ, (x = 1 ↔ (y x = (x - 1)^2 - 4))) :=
by
  -- Define the quadratic function
  let y := λ x : ℝ, x^2 - 2 * x - 3
  -- Prove the vertex is at (1, -4) and the axis of symmetry is x = 1
  sorry

end quadratic_vertex_axis_of_symmetry_l555_555502


namespace domain_of_f_l555_555276

noncomputable def f (x : ℝ) : ℝ := Real.log (sqrt (1 - x^2))

theorem domain_of_f : 
  { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | ∀ y : ℝ, y = f x → 0 < y } :=
by
  sorry

end domain_of_f_l555_555276


namespace sin_390_eq_1_over_2_sin_30_eq_1_over_2_sin_390_l555_555471

example : ∀ x : ℝ, sin (x + 2 * π) = sin x := by sorry

theorem sin_390_eq_1_over_2 :
  sin (390 * (π / 180)) = sin (30 * (π / 180)) :=
by sorry

theorem sin_30_eq_1_over_2 :
  sin (30 * (π / 180)) = 1 / 2 :=
by sorry

theorem sin_390 : sin (390 * (π / 180)) = 1 / 2 :=
by {
  rw sin_390_eq_1_over_2,
  exact sin_30_eq_1_over_2
}

end sin_390_eq_1_over_2_sin_30_eq_1_over_2_sin_390_l555_555471


namespace maria_traveled_portion_of_distance_l555_555864

theorem maria_traveled_portion_of_distance (total_distance first_stop remaining_distance_to_destination : ℝ) 
  (h1 : total_distance = 560) 
  (h2 : first_stop = total_distance / 2) 
  (h3 : remaining_distance_to_destination = 210) : 
  ((first_stop - (first_stop - (remaining_distance_to_destination + (first_stop - total_distance / 2)))) / (total_distance - first_stop)) = 1 / 4 :=
by
  sorry

end maria_traveled_portion_of_distance_l555_555864


namespace valid_functions_eq_ordered_set_partitions_l555_555121

-- Define the set [n]
def finset_n (n : ℕ) : Type := { i // i ∈ finset.range (n+1) }

-- Define the number of functions f: [n] → [n] such that f(f(i)) ≥ i ∀ i
def num_valid_functions (n : ℕ) : ℕ :=
  cardinal.mk { f : finset_n n → finset_n n // ∀ i, f (f i) ≥ i }

-- Define the number of ordered set partitions of [n]
def num_ordered_set_partitions (n : ℕ) : ℕ :=
  cardinal.mk { k // ∃ (A : fin n → finset { i // i < n }) (H : ∀ (m : fin n), A m ≠ ∅) (disjoint_all : ∀ (i j : fin n), i ≠ j → disjoint A i A j), (⋃ i, A i) = finset.univ }

-- Main statement to be proven
theorem valid_functions_eq_ordered_set_partitions (n : ℕ) : num_valid_functions n = num_ordered_set_partitions n :=
sorry

end valid_functions_eq_ordered_set_partitions_l555_555121


namespace smallest_possible_difference_l555_555716

theorem smallest_possible_difference : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    {a, b, c, d} = {2, 4, 6, 8} ∧ 
    (∃ (x y : ℕ), 10*a + b = x ∧ 10*c + d = y ∧ abs (x - y) = 14) :=
begin
  sorry

end smallest_possible_difference_l555_555716


namespace parallelepiped_from_midpoints_l555_555749

-- Define the structures and conditions
structure Segment (P : Type) [Point P] :=
  (start : P)
  (end : P)

variable {P : Type} [Point P]

def midpoint (s : Segment P) (O : P) : Prop :=
  dist s.start O = dist O s.end

def not_coplanar (s1 s2 s3 : Segment P) : Prop :=
  ¬ coplanar s1.start s1.end s2.start s2.end s3.start s3.end

-- Main statement: proof problem from the given conditions
theorem parallelepiped_from_midpoints (s1 s2 s3 : Segment P) (O : P) :
  midpoint s1 O →
  midpoint s2 O →
  midpoint s3 O →
  not_coplanar s1 s2 s3 →
  -- Prove the endpoints form the vertices of a parallelepiped
  parallelepiped [s1.start, s1.end, s2.start, s2.end, s3.start, s3.end] :=
by
  sorry

end parallelepiped_from_midpoints_l555_555749


namespace probability_of_grade_A_l555_555621

open ProbabilityTheory

noncomputable def prob_grade_A (X : MeasureTheory.MeasureSpace Real) := 
  sorry

theorem probability_of_grade_A :
  prob_grade_A (MeasureTheory.gaussian 80 (real.sqrt 25)) = 0.15865 :=
by
  sorry

end probability_of_grade_A_l555_555621


namespace prob_union_of_mutually_exclusive_l555_555992

-- Let's denote P as a probability function
variable {Ω : Type} (P : Set Ω → ℝ)

-- Define the mutually exclusive condition
def mutually_exclusive (A B : Set Ω) : Prop :=
  (A ∩ B) = ∅

-- State the theorem that we want to prove
theorem prob_union_of_mutually_exclusive (A B : Set Ω) 
  (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B :=
sorry

end prob_union_of_mutually_exclusive_l555_555992


namespace total_distance_walked_in_6_days_l555_555212

theorem total_distance_walked_in_6_days :
  let group_route_dist := 3
  let days_per_week := 6
  let jamies_additional_per_day := 2
  let sues_ratio := 0.5
  let total_group_dist := group_route_dist * days_per_week
  let total_jamie_dist := jamies_additional_per_day * days_per_week
  let sue_ratio_dist := total_jamie_dist * sues_ratio
  total_group_dist + total_jamie_dist + sue_ratio_dist = 36 :=
by
-- Definitions
let group_route_dist := 3
let days_per_week := 6
let jamies_additional_per_day := 2
let sues_ratio := 0.5
let total_group_dist := group_route_dist * days_per_week
let total_jamie_dist := jamies_additional_per_day * days_per_week
let sue_ratio_dist := total_jamie_dist * sues_ratio

-- Assertion
have : total_group_dist + total_jamie_dist + sue_ratio_dist = 36 := by sorry
exact this

end total_distance_walked_in_6_days_l555_555212


namespace parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l555_555850

noncomputable def parabola (p x : ℝ) : ℝ := (p-1) * x^2 + 2 * p * x + 4

-- 1. Prove that if \( p = 2 \), the parabola \( g_p \) is tangent to the \( x \)-axis.
theorem parabola_tangent_xaxis_at_p2 : ∀ x, parabola 2 x = (x + 2)^2 := 
by 
  intro x
  sorry

-- 2. Prove that if \( p = 0 \), the vertex of the parabola \( g_p \) lies on the \( y \)-axis.
theorem parabola_vertex_yaxis_at_p0 : ∃ x, parabola 0 x = 4 := 
by 
  sorry

-- 3. Prove the parabolas for \( p = 2 \) and \( p = 0 \) are symmetric with respect to \( M(-1, 2) \).
theorem parabolas_symmetric_m_point : ∀ x, 
  (parabola 2 x = (x + 2)^2) → 
  (parabola 0 x = -x^2 + 4) → 
  (-1, 2) = (-1, 2) := 
by 
  sorry

-- 4. Prove that the points \( (0, 4) \) and \( (-2, 0) \) lie on the curve for all \( p \).
theorem parabola_familiy_point_through : ∀ p, 
  parabola p 0 = 4 ∧ 
  parabola p (-2) = 0 :=
by 
  sorry

end parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l555_555850


namespace airplane_average_speed_l555_555037

-- Define the conditions
def miles_to_kilometers (miles : ℕ) : ℝ :=
  miles * 1.60934

def distance_miles : ℕ := 1584
def time_hours : ℕ := 24

-- Define the problem to prove
theorem airplane_average_speed : 
  (miles_to_kilometers distance_miles) / (time_hours : ℝ) = 106.24 :=
by
  sorry

end airplane_average_speed_l555_555037


namespace regular_polygon_perimeter_l555_555373

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555373


namespace question1_question2_l555_555537

variable (a : ℝ)
def z1 : ℂ := complex.mk 2 (-1)
def z2 : ℂ := complex.mk a 1

-- Prove that if a = 1, then z1 + conj(z2) is in the fourth quadrant
theorem question1 (h : a = 1) : 
  complex.quadrant (z1 + complex.conj z2) = 4 := 
  sorry

-- Prove that if z1 * z2 is purely imaginary, then a = -1/2
theorem question2 (h : complex.is_purely_imaginary (z1 * z2)) : 
  a = -1 / 2 := 
  sorry

end question1_question2_l555_555537


namespace sum_series_evaluation_l555_555078

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (if k = 0 then 0 else (2 * k) / (4 : ℝ) ^ k)

theorem sum_series_evaluation : sum_series = 8 / 9 := by
  sorry

end sum_series_evaluation_l555_555078


namespace part_I_part_II_l555_555951

noncomputable def f (x : ℝ) (a : ℝ) := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem part_I (a : ℝ) (h : a = 3 / 2) : 
  (∀ x, 0 < x ∧ x < 1 → f x a < 0) ∧ (∀ x, 1 < x ∧ x < 2 → f x a > 0) ∧ (∀ x, 2 < x → f x a < 0) := sorry

theorem part_II (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ 0) → a ≤ 1 := sorry

end part_I_part_II_l555_555951


namespace dave_apps_left_l555_555063

theorem dave_apps_left (A : ℕ) 
  (h1 : 24 = A + 22) : A = 2 :=
by
  sorry

end dave_apps_left_l555_555063


namespace problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l555_555036

-- Problem G6.1
theorem problem_G6_1 (n : ℕ) (h: (n * (n - 1)) / 2 - n = 20) : n = 8 :=
sorry

-- Problem G6.2
theorem problem_G6_2 (k : ℕ) (h: (∀ x y: ℕ, x > 0 ∧ x < 7 ∧ y > 0 ∧ y < 7 ∧ x + y = 8 → 1) = k) : k = 5 :=
sorry

-- Problem G6.3
theorem problem_G6_3 (u : ℕ) (h : (3 * 25 + 2 * 50) / 5 = u) : u = 35 :=
sorry

-- Problem G6.4
theorem problem_G6_4 (a : ℤ) (h: (3 * (2 * a + 1) + 1 = 10)) : a = 1 :=
sorry

end problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l555_555036


namespace cost_of_500_pencils_in_dollars_l555_555712

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l555_555712


namespace regular_polygon_perimeter_l555_555393

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555393


namespace symmetric_patterns_8x8_grid_l555_555366

theorem symmetric_patterns_8x8_grid :
  let n : ℕ := 8,
      colors : Finset (Fin 3) := {0, 1, 2}, -- Representing Red, Blue, White as 0, 1, 2
      is_symmetric_pattern (grid : Matrix (Fin n) (Fin n) (Fin 3)) : Prop :=
        ∀ i j : Fin n, grid i j = grid (n - 1 - i) (n - 1 - j) ∧
                       grid i j = grid j i ∧
                       grid i j = grid (n - 1 - i) (n - 1 - i),
      count_symmetric_patterns : ℕ :=
        Finset.card (Finset.filter
          (λ grid : Matrix (Fin n) (Fin n) (Fin 3),
            is_symmetric_pattern grid ∧
            ∀ c ∈ colors, ∃ i j : Fin n, grid i j = c)
          (finset.univ : Finset (Matrix (Fin n) (Fin n) (Fin 3))))
  in count_symmetric_patterns = 5559060566555523 :=
sorry

end symmetric_patterns_8x8_grid_l555_555366


namespace investment_years_l555_555099

noncomputable def A := 1120
noncomputable def P := 933.3333333333334
noncomputable def r := 0.05
noncomputable def n := 1

theorem investment_years :
  ∃ t : ℕ, A = P * (1 + r / n)^(t * n) ∧ t = 4 :=
begin
  sorry
end

end investment_years_l555_555099


namespace number_of_integers_satisfy_inequality_l555_555168

theorem number_of_integers_satisfy_inequality : 
  {m : ℤ // m ≠ 0 ∧ 0 < |m| < 10}.card = 18 := 
sorry

end number_of_integers_satisfy_inequality_l555_555168


namespace problem_l555_555957

-- Define proposition p: for all x in ℝ, x^2 + 1 ≥ 1
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q: for angles A and B in a triangle, A > B ↔ sin A > sin B
def q : Prop := ∀ {A B : ℝ}, A > B ↔ Real.sin A > Real.sin B

-- The problem definition: prove that p ∨ q is true
theorem problem (hp : p) (hq : q) : p ∨ q := sorry

end problem_l555_555957


namespace sum_of_three_distinct_members_l555_555633

def set : List ℤ := [2, 5, 8, 11, 14, 17, 20, 23]

def sum_of_three_distinct (s : List ℤ) : List ℤ :=
  List.foldl (λ accum a, List.foldl (λ accum b, 
    if b ≠ a then 
      List.foldl (λ accum c, if c ≠ a ∧ c ≠ b then (a + b + c) :: accum else accum) accum s 
    else accum) accum s) [] s

theorem sum_of_three_distinct_members :
  (sum_of_three_distinct set).toFinset.card = 16 := by
  sorry

end sum_of_three_distinct_members_l555_555633


namespace distinct_values_l555_555482

-- Define the standard evaluation of the expression
def standard_eval : ℕ := 3 ^ 27

-- Define the different possible parenthesized values of the expression
def value1 : ℕ := 3 ^ (3 ^ (3 ^ 3))
def value2 : ℕ := 3 ^ ((3 ^ 3) ^ 3)
def value3 : ℕ := ((3 ^ 3) ^ 3) ^ 3
def value4 : ℕ := (3 ^ (3 ^ 3)) ^ 3
def value5 : ℕ := (3 ^ 3) ^ (3 ^ 3)

-- Prove that there is exactly one other distinct value different from standard_eval
theorem distinct_values : (finset.find (λ x, x ≠ standard_eval)
  (finset.filter (λ x, x ≠ standard_eval) (finset.of_list [value1, value2, value3, value4, value5]))).length = 1 := by
  sorry

end distinct_values_l555_555482


namespace perimeter_of_polygon_l555_555438

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555438


namespace det_projection_matrix_zero_l555_555650

theorem det_projection_matrix_zero :
  let Q := let v : ℝ × ℝ := (3, 5) in
           let norm_v := real.sqrt ((v.1)^2 + (v.2)^2) in
           let cos_phi := v.1 / norm_v in
           let sin_phi := v.2 / norm_v in
           matrix.of_vec_transpose 
             [cos_phi^2, cos_phi * sin_phi, cos_phi * sin_phi, sin_phi^2] 
             2 2 in
  matrix.det Q = 0 :=
by
  let v : ℝ × ℝ := (3, 5)
  let norm_v := real.sqrt ((v.1)^2 + (v.2)^2)
  let cos_phi := v.1 / norm_v
  let sin_phi := v.2 / norm_v
  let Q := matrix.of_vec_transpose [cos_phi^2, cos_phi * sin_phi, cos_phi * sin_phi, sin_phi^2] 2 2
  show matrix.det Q = 0
  sorry

end det_projection_matrix_zero_l555_555650


namespace math_problem_l555_555775

theorem math_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 35) : L = 1631 := 
by
  sorry

end math_problem_l555_555775


namespace amanda_days_needed_to_meet_goal_l555_555834

def total_tickets : ℕ := 80
def first_day_friends : ℕ := 5
def first_day_per_friend : ℕ := 4
def first_day_tickets : ℕ := first_day_friends * first_day_per_friend
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_days_needed_to_meet_goal : 
  first_day_tickets + second_day_tickets + third_day_tickets = total_tickets → 
  3 = 3 :=
by
  intro h
  sorry

end amanda_days_needed_to_meet_goal_l555_555834


namespace valid_pairings_count_l555_555745

theorem valid_pairings_count :
  let people := Fin 12
  let knows (p q : Fin 12) : Prop := 
    abs (p.val - q.val) = 1 ∨ abs (p.val - q.val) = 2 ∨ abs (p.val - q.val) = 10 
    ∨ abs (p.val - q.val) = 11
  ∃! pairs : List (Fin 12 × Fin 12), 
      pairs.length = 6 ∧ 
      ∀ (p : Fin 12), ∃ (q : Fin 12), (p, q) ∈ pairs ∨ (q, p) ∈ pairs ∧ knows p q := 2 
:= sorry

end valid_pairings_count_l555_555745


namespace distributing_balls_into_boxes_l555_555574

-- Define the parameters for the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Statement of the problem in Lean
theorem distributing_balls_into_boxes :
  (finset.card (finset.univ.filter (λ (f : fin num_boxes → ℕ), fin.sum univ f = num_balls))) = 21 := 
sorry

end distributing_balls_into_boxes_l555_555574


namespace solve_for_A_l555_555906

theorem solve_for_A (a b : ℝ) (A : ℝ) (h : 3 * a * b * A = 6 * a ^ 2 * b - 9 * a * b ^ 2) : 
  A = 2 * a - 3 * b :=
begin
  sorry
end

end solve_for_A_l555_555906


namespace find_divisor_l555_555095

theorem find_divisor (n : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : d = 10) (h3 : ∃ k, n - d = k * d) : d = 3 :=
by
  sorry

end find_divisor_l555_555095


namespace sarah_stamp_collection_value_l555_555694

theorem sarah_stamp_collection_value :
  ∀ (stamps_owned total_value_for_4_stamps : ℝ) (num_stamps_single_series : ℕ), 
  stamps_owned = 20 → 
  total_value_for_4_stamps = 10 → 
  num_stamps_single_series = 4 → 
  (stamps_owned / num_stamps_single_series) * (total_value_for_4_stamps / num_stamps_single_series) = 50 :=
by
  intros stamps_owned total_value_for_4_stamps num_stamps_single_series 
  intro h_stamps_owned
  intro h_total_value_for_4_stamps
  intro h_num_stamps_single_series
  rw [h_stamps_owned, h_total_value_for_4_stamps, h_num_stamps_single_series]
  sorry

end sarah_stamp_collection_value_l555_555694


namespace find_length_segment_XY_l555_555309

-- Define the given conditions as variables
variable (A B C D X Y : Point)
variable (BX DY : ℝ)
variable [hAB : Side AB] [hBC : Side BC]

-- Define additional required properties and constraints
def rectangle (A B C D : Point) : Prop := 
(square A B C D)

def perpendicular (P Q : Point) (ℓ : Line) : Prop :=
(PQ ⊥ ℓ)

def ratio (XY : ℝ) (AB CD : Point) : ℝ := 
(∃ r, XY / AB = r ∧ XY / CD = 1 / r)

variable (ℓ : Line)
axiom perpendiculars : BX = 4 ∧ DY = 10
axiom BA_ratio : BC = 2 * AB
axiom similarity_ratio : ∀ (XY AB AD : ℝ), similar(XAB YDA) -> (AB / AD = 1 / 2)

theorem find_length_segment_XY :
  length XY = 13 :=
by 
  sorry

end find_length_segment_XY_l555_555309


namespace number_of_lines_through_P_forming_30_with_a_and_b_l555_555564

variables (a b : line) (P : point)

-- Conditions
axiom skew_lines_form_angle : ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ θ = 50 * (π / 180) ∧ skew_lines a b
axiom fixed_point : ∃ P : point, true

-- The theorem to prove
theorem number_of_lines_through_P_forming_30_with_a_and_b :
  ∃! (l1 l2 : line), (l1 ∈ lines_through P) ∧ (l2 ∈ lines_through P) ∧ 
  (angle_between l1 a = 30 * (π / 180)) ∧ (angle_between l2 a = 30 * (π / 180)) ∧
  (angle_between l1 b = 30 * (π / 180)) ∧ (angle_between l2 b = 30 * (π / 180)) :=
sorry

end number_of_lines_through_P_forming_30_with_a_and_b_l555_555564


namespace difference_of_squares_divisibility_l555_555052

theorem difference_of_squares_divisibility (a b : ℤ) :
  ∃ m : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 8 * m ∧ 
           ¬∃ n : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 16 * n :=
by
  sorry

end difference_of_squares_divisibility_l555_555052


namespace encounter_on_weekday_l555_555624

-- Definitions: FirstBrother and SecondBrother behavior
def lies_on (day : String) : Prop :=
  day = "Saturday" ∨ day = "Sunday"

def first_brother_lies (day : String) : Prop :=
  day = "Saturday" ∨ day = "Sunday"

def second_brother_lies_tomorrow (today tomorrow : String) : Prop :=
  tomorrow = "Saturday" ∨ tomorrow = "Sunday"

-- Theorem to prove: the encounter happens on a regular weekday
theorem encounter_on_weekday (day : String) (tomorrow : String) :
  ¬(first_brother_lies day) →
  ¬(second_brother_lies_tomorrow day tomorrow) →
  ¬(day = "Saturday" ∨ day = "Sunday") :=
by
  intros
  intro H
  cases H
  case inl H1 => contradiction
  case inr H2 => contradiction
  sorry

end encounter_on_weekday_l555_555624


namespace income_difference_l555_555863

theorem income_difference
  (D W : ℝ)
  (hD : 0.08 * D = 800)
  (hW : 0.08 * W = 840) :
  (W + 840) - (D + 800) = 540 := 
  sorry

end income_difference_l555_555863


namespace find_k_arithmetic_sequence_l555_555530

noncomputable def arithmetic_sequence_condition (d k : ℕ) (hk : k > 0) (h1 : d ≠ 0) : Prop :=
  ∃ a : ℕ → ℤ, (∀ n, a n = a 1 + (n - 1) * d) ∧ a 1 = 2 * d ∧ ∃ n, a n = nat.sqrt (a 1 * a (2 * k + 1)) ∧ n = k

theorem find_k_arithmetic_sequence (d : ℕ) (h1 : d ≠ 0) : 
  ∃ k : ℕ, k > 0 ∧ arithmetic_sequence_condition d k (by sorry) h1 :=
  by
    use 3
    split
    · exact nat.succ_pos'
    · unfold arithmetic_sequence_condition
      sorry

end find_k_arithmetic_sequence_l555_555530


namespace option_a_is_correct_l555_555536

variable (a b : ℝ)
variable (ha : a < 0)
variable (hb : b < 0)
variable (hab : a < b)

theorem option_a_is_correct : (a < abs (3 * a + 2 * b) / 5) ∧ (abs (3 * a + 2 * b) / 5 < b) :=
by
  sorry

end option_a_is_correct_l555_555536


namespace angle_B_triangle_area_l555_555176

variables (A B C : ℝ) (a b c : ℝ)
variables (area : ℝ)
theorem angle_B :
  a * sin B * cos C + c * sin B * cos A = (1 / 2) * b →
  a > b →
  B = π / 6 :=
by
  sorry

theorem triangle_area :
  b = Real.sqrt 13 →
  a + c = 4 →
  B = π / 6 →
  area = (1 / 2) * a * c * sin B →
  area = (6 - 3 * Real.sqrt 3) / 4 :=
by
  sorry

end angle_B_triangle_area_l555_555176


namespace flower_garden_mystery_value_l555_555198

/-- Prove the value of "花园探秘" given the arithmetic sum conditions and unique digit mapping. -/
theorem flower_garden_mystery_value :
  ∀ (shu_hua_hua_yuan : ℕ) (wo_ai_tan_mi : ℕ),
  shu_hua_hua_yuan + 2011 = wo_ai_tan_mi →
  (∃ (hua yuan tan mi : ℕ),
    0 ≤ hua ∧ hua < 10 ∧
    0 ≤ yuan ∧ yuan < 10 ∧
    0 ≤ tan ∧ tan < 10 ∧
    0 ≤ mi ∧ mi < 10 ∧
    hua ≠ yuan ∧ hua ≠ tan ∧ hua ≠ mi ∧
    yuan ≠ tan ∧ yuan ≠ mi ∧ tan ≠ mi ∧
    shu_hua_hua_yuan = hua * 1000 + yuan * 100 + tan * 10 + mi ∧
    wo_ai_tan_mi = 9713) := sorry

end flower_garden_mystery_value_l555_555198


namespace distance_between_intersections_l555_555490

def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

theorem distance_between_intersections : ∀ (x1 y1 x2 y2 : ℝ), 
  curve1 x1 y1 → curve2 x1 y1 → curve1 x2 y2 → curve2 x2 y2 →
  (x1, y1) ≠ (x2, y2) → 
  dist (x1, y1) (x2, y2) = real.sqrt 2 :=
begin
  sorry
end

end distance_between_intersections_l555_555490


namespace cot_alpha_minus_pi_div_2_l555_555931

variable (α : Real)
variable (h1 : cos (π - α) = 1 / 3)
variable (h2 : π < α ∧ α < 3 * π / 2)

theorem cot_alpha_minus_pi_div_2 : cot (α - π / 2) = -2 * sqrt 2 := by
  sorry

end cot_alpha_minus_pi_div_2_l555_555931


namespace area_of_LOSE_sector_l555_555002

-- Definitions based on conditions
def radius : ℝ := 12
def prob_win : ℝ := 1 / 3

-- Area of the spinner (entire circle)
def area_circle : ℝ := π * radius ^ 2

-- Area of the WIN sector
def area_win : ℝ := (prob_win * area_circle)

-- The required property to prove (area of the LOSE sector)
def area_lose : ℝ := area_circle - area_win

theorem area_of_LOSE_sector : area_lose = 96 * π := by
  sorry

end area_of_LOSE_sector_l555_555002


namespace perimeter_of_regular_polygon_l555_555421

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555421


namespace sum_product_smallest_number_l555_555739

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l555_555739


namespace sum_of_distinct_prime_factors_of_882_l555_555330

theorem sum_of_distinct_prime_factors_of_882 : (∑ p in {2, 3, 7}, p) = 12 :=
by sorry

end sum_of_distinct_prime_factors_of_882_l555_555330


namespace rowing_distance_l555_555831

theorem rowing_distance :
  let boat_speed := 300 -- in kmph
  let current_speed := 42 -- in kmph
  let time_minutes := 1.9998400127989762
  let time_hours := time_minutes / 60 -- convert time to hours
  let effective_speed := boat_speed + current_speed -- effective speed downstream
  let distance_covered := effective_speed * time_hours -- distance formula
  distance_covered ≈ 11.399168 :=
by sorry

end rowing_distance_l555_555831


namespace part_a_l555_555774

theorem part_a (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, (n = 10 * k) ∧ (∃ m : ℕ, (n * n - (n * n % 100)) / 100 = m * m) :=
by {
  use 10 * k,
  split,
  { ring },
  { use k,
    rw [mul_assoc, Nat.pow_two, Nat.mul_sub_div_of_dvd, Nat.div_self],
    { ring },
    { use 10 } }
}

end part_a_l555_555774


namespace transform_graph_sinx_l555_555310

theorem transform_graph_sinx :
  (∀ (x : ℝ), sin (3 * x - π / 6) = sin 3 * (x - π / 18)) ∧
  (∀ (x : ℝ), sin (3 * (x - π / 6)) = sin 3 * (x - π / 18)) :=
sorry

end transform_graph_sinx_l555_555310


namespace polygon_perimeter_l555_555384

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555384


namespace total_money_divided_l555_555001

theorem total_money_divided (A B C T : ℝ) 
    (h1 : A = (2/5) * (B + C)) 
    (h2 : B = (1/5) * (A + C)) 
    (h3 : A = 600) :
    T = A + B + C →
    T = 2100 :=
by 
  sorry

end total_money_divided_l555_555001


namespace regular_polygon_perimeter_l555_555441

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555441


namespace trisect_ratio_l555_555835

variables {A B C D E : Point}
variables {AB BC AC BD BE AD EC : ℝ}

-- Add the necessary geometric structures, e.g., points, triangles, bisectors, etc.
-- Define the conditions that BD and BE trisect the angle at B, and D, E lie on AC.
def trisects (T : Triangle) (BD BE : Line) (B D E : Point) : Prop :=
  IsTrisection T B BD BE D E -- assuming IsTrisection is a predicate abstracting the trisect condition

theorem trisect_ratio
  (T : Triangle)
  (BD BE : Line)
  (htrisect : trisects T BD BE $ T.BD)
  (htrisect : trisects T BD BE $ T.BE) :
  (AD / EC = (AB * BD) / (BE * BC)) :=
by
  sorry

end trisect_ratio_l555_555835


namespace work_days_for_A_alone_l555_555350

/-- A proof that the number of days A takes to complete the work alone is 4,
    given that B can finish the work in 12 days and together they can finish the work in 3 days. -/
theorem work_days_for_A_alone : ∀ (A_days : ℝ),
  ((1 / A_days) + (1 / 12) = 1 / 3) → A_days = 4 :=
by 
  intros,
  field_simp [div_eq_mul_inv] at *,
  linarith,
  sorry

end work_days_for_A_alone_l555_555350


namespace probability_more_1s_than_6s_l555_555988

theorem probability_more_1s_than_6s :
  let outcomes := (list.range 6).repeats 4 in
  let valid_outcomes := list.filter (λ (xs : list ℕ), xs.countp (= 1) > xs.countp (= 6)) outcomes in
  (valid_outcomes.length / outcomes.length : ℚ) = 421 / 1296 :=
by
  sorry

end probability_more_1s_than_6s_l555_555988


namespace largest_quantity_l555_555491

theorem largest_quantity 
  (A := (2010 / 2009) + (2010 / 2011))
  (B := (2012 / 2011) + (2010 / 2011))
  (C := (2011 / 2010) + (2011 / 2012)) : C > A ∧ C > B := 
by {
  sorry
}

end largest_quantity_l555_555491


namespace perimeter_of_regular_polygon_l555_555413

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555413


namespace h_even_if_g_odd_l555_555656

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end h_even_if_g_odd_l555_555656


namespace number_of_people_l555_555494

-- Definitions based on conditions
def per_person_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 100 else max 72 (100 - 2 * (x - 30))

def total_cost (x : ℕ) : ℕ :=
  x * per_person_cost x

-- Main theorem statement
theorem number_of_people (x : ℕ) (h1 : total_cost x = 3150) (h2 : x > 30) : x = 35 :=
by {
  sorry
}

end number_of_people_l555_555494


namespace find_parabola_equation_l555_555544

noncomputable def parabola_constant (m : ℝ) : Prop := 
  m^2 + 4 * m - 4.5 = 0

def parabola_equation (m : ℝ) : Prop :=
  ∃ k : ℝ, m = -4 + k ∧ y^2 = ( -4 + k ) * x 
  ∨ m = -4 - k ∧ y^2 = ( -4 - k ) * x

theorem find_parabola_equation (m : ℝ) (h : m ≠ 0) 
  (h_chord : ∃ A B : (ℝ × ℝ), ((A.2 = A.1 - 4) ∧ (B.2 = B.1 - 4)) ∧
                                 ((y^2 = 2 * m * A.1) ∧ (y^2 = 2 * m * B.1)) ∧
                                 (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * sqrt 2)) : 
  ∃ k : ℝ, y^2 = ( -4 + sqrt 34 ) * x 
  ∨ y^2 = ( -4 - sqrt 34 ) * x := 
begin
  sorry
end

end find_parabola_equation_l555_555544


namespace tangent_intersects_y_axis_at_point_l555_555944

noncomputable def curve (n : ℕ) (x : ℝ) : ℝ := x^(n+1)

theorem tangent_intersects_y_axis_at_point (n : ℕ) (h : 0 < n) :
  let y' := (n + 1 : ℝ) * 1^n in
  let tangent_line (x : ℝ) := (n + 1) * (x - 1) + 1 in
  tangent_line 0 = -n :=
  by
    sorry

end tangent_intersects_y_axis_at_point_l555_555944


namespace sin40_tan10_minus_sqrt3_eq_neg1_l555_555360

theorem sin40_tan10_minus_sqrt3_eq_neg1 :
  sin (40 * real.pi / 180) * (tan (10 * real.pi / 180) - real.sqrt 3) = -1 := 
by
  sorry

end sin40_tan10_minus_sqrt3_eq_neg1_l555_555360


namespace eq_g_i_l555_555983

noncomputable def complex_i : ℂ := complex.I

def g (x : ℂ) : ℂ := (x^5 - x^3 + x) / (x^2 - 1)

theorem eq_g_i : g complex_i = - (3 * complex_i) / 2 := by
  sorry

end eq_g_i_l555_555983


namespace range_of_a_for_monotonicity_l555_555548

noncomputable def f (a x : ℝ) := sqrt (x^2 + 1) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : 0 < a) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≤ x2 → f a x1 ≤ f a x2) → 1 ≤ a :=
by
  sorry

end range_of_a_for_monotonicity_l555_555548


namespace find_x_l555_555899

theorem find_x (x: ℝ) (h: sqrt (x + 16) = 5) : x = 9 := by
  sorry

end find_x_l555_555899


namespace calculate_expression_l555_555470

theorem calculate_expression : 12 * (1 / (2 / 3 - 1 / 4 + 1 / 6)) = 144 / 7 :=
by
  sorry

end calculate_expression_l555_555470


namespace find_cookbooks_stashed_in_kitchen_l555_555566

-- Definitions of the conditions
def total_books := 99
def books_in_boxes := 3 * 15
def books_in_room := 21
def books_on_table := 4
def books_picked_up := 12
def current_books := 23

-- Main statement
theorem find_cookbooks_stashed_in_kitchen :
  let books_donated := books_in_boxes + books_in_room + books_on_table
  let books_left_initial := total_books - books_donated
  let books_left_before_pickup := current_books - books_picked_up
  books_left_initial - books_left_before_pickup = 18 := by
  sorry

end find_cookbooks_stashed_in_kitchen_l555_555566


namespace fraction_comparison_l555_555767

theorem fraction_comparison : (5555553 / 5555557 : ℚ) > (6666664 / 6666669 : ℚ) :=
  sorry

end fraction_comparison_l555_555767


namespace peanut_butter_candy_count_l555_555307

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l555_555307


namespace probability_f_leq_0_l555_555150

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem probability_f_leq_0 (x_0 : ℝ) (hx0 : x_0 ∈ Icc (-5 : ℝ) 5) :
  (∃ x_0 ∈ Icc (2 : ℝ) 3, f x_0 ≤ 0) →
  (intervalLength (Icc (2 : ℝ) 3)) / (intervalLength (Icc (-5 : ℝ) 5)) = 1 / 10 :=
sorry

end probability_f_leq_0_l555_555150


namespace largest_n_divisible_103_l555_555066

theorem largest_n_divisible_103 (n : ℕ) (h1 : n < 103) (h2 : 103 ∣ (n^3 - 1)) : n = 52 :=
sorry

end largest_n_divisible_103_l555_555066


namespace calculate_f_f_sqrt5_l555_555948

def f : ℝ → ℝ :=
  λ x, if x ≤ 2 then exp (x - 1) else log 2 (x ^ 2 - 1)

theorem calculate_f_f_sqrt5 : f (f (sqrt 5)) = exp 1 := by
  sorry

end calculate_f_f_sqrt5_l555_555948


namespace probability_winning_game_show_l555_555356

open ProbabilityTheory
open Finset

noncomputable def probability_of_winning : ℚ :=
  let p_correct_one := (1 / 4) in
  let p_incorrect_one := (3 / 4) in
  let cases_4_correct := (p_correct_one ^ 4) in
  let cases_3_correct := (4.choose 3) * (p_correct_one ^ 3) * p_incorrect_one in
  cases_4_correct + cases_3_correct

theorem probability_winning_game_show :
  probability_of_winning = 13 / 256 :=
by
  simp [probability_of_winning]
  sorry

end probability_winning_game_show_l555_555356


namespace find_smaller_number_l555_555741

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l555_555741


namespace percentage_increase_is_approx_22_22_l555_555216

noncomputable def percentage_increase_in_average_contribution
  (initial_avg : ℝ) (initial_contributors : ℕ) (johns_donation : ℝ) : ℝ :=
let initial_total := initial_avg * initial_contributors in
let new_total := initial_total + johns_donation in
let new_contributors := initial_contributors + 1 in
let new_avg := new_total / new_contributors in
let increase := new_avg - initial_avg in
(increase / initial_avg) * 100

theorem percentage_increase_is_approx_22_22
  (initial_avg : real := 75) (initial_contributors : ℕ := 2) (johns_donation : real := 125) :
  abs (percentage_increase_in_average_contribution initial_avg initial_contributors johns_donation - 22.22) < 0.01 :=
by sorry

end percentage_increase_is_approx_22_22_l555_555216


namespace statements_correctness_l555_555659

open Complex

theorem statements_correctness (z1 z2 z3 : ℂ) :
  (z1 * z2 = 0 → z1 = 0 ∨ z2 = 0) ∧
  (z1 * z2 = z1 * z3 ∧ z1 ≠ 0 → z2 = z3) ∧
  (abs z1 = abs z2 → z1 * conj(z1) = z2 * conj(z2)) ∧
  ¬(abs z1 = abs z2 → z1^2 = z2^2) :=
by
  sorry

end statements_correctness_l555_555659


namespace complex_quadrant_l555_555520

open Complex

theorem complex_quadrant (z : ℂ) (h : z = 2 * Complex.i / (1 + Complex.i)) :
    z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l555_555520


namespace balance_difference_7286_l555_555847

def P := 12000
def r_cedric := 0.06
def r_daniel := 0.08
def n := 20

def Cedric_balance : ℝ := P * (1 + r_cedric) ^ n
def Daniel_balance : ℝ := P * (1 + n * r_daniel)
def balance_difference : ℝ := |Cedric_balance - Daniel_balance|

theorem balance_difference_7286 : balance_difference = 7286 := by
  sorry

end balance_difference_7286_l555_555847


namespace sequence_sum_l555_555265

noncomputable def x (k : ℕ) : ℚ :=
if k = 1 then 1 else x (k - 1) + (k - 1)/10

def sum_sequence (n : ℕ) : ℚ :=
∑ k in finset.range.n succ, x k

theorem sequence_sum (n : ℕ) : sum_sequence n = (2 * n^3 - n^2 + 58 * n) / 60 := by
  sorry

end sequence_sum_l555_555265


namespace tea_consumption_on_dayB_l555_555727

variable (p t k : ℝ)

-- Given inverse proportionality condition
def inverse_proportional (t p k : ℝ) : Prop :=
  t * p = k

-- Conditions from Day A
def dayA_conds : Prop :=
  inverse_proportional 3 8 24

-- Calculate the cups of tea on Day B with 5 hours of programming
def cups_of_tea_on_dayB : ℝ :=
  24 / 5

-- Theorem to prove the result on Day B
theorem tea_consumption_on_dayB :
  dayA_conds →
  inverse_proportional (24 / 5) 5 24 :=
sorry

end tea_consumption_on_dayB_l555_555727


namespace proof_problem_l555_555916

noncomputable theory
open BigOperators

def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) : ℕ := (3 * n) / 2 + 1 / 2

def c_n (n : ℕ) : ℕ := (2 * a_n n) / ((a_n (n + 1) - 1) * (a_n (n + 2) - 1)) + b_n n / a_n n

def S_n (n : ℕ) : ℕ := ∑ i in finset.range(n + 1), c_n i

theorem proof_problem (n : ℕ) : S_n n = 8 - (1 / (2^(n + 1) - 1)) - (3 * n + 7) / 2^n :=
sorry

end proof_problem_l555_555916


namespace dishwasher_manager_ratio_l555_555044

noncomputable def dishwasher_wage (D C M : ℝ) :=
  M = 7.50 ∧
  C = M - 3 ∧
  C = 1.20 * D

theorem dishwasher_manager_ratio (D C M : ℝ) 
  (h : dishwasher_wage D C M) : 
  (D / M) = 0.5 :=
by
  cases h with
  | intro h₁ h₂ h₃ =>
  sorry

end dishwasher_manager_ratio_l555_555044


namespace balls_in_boxes_ways_l555_555571

theorem balls_in_boxes_ways : 
  (∃ (ways : ℕ), ways = 21 ∧
    ∀ {k : ℕ}, (5 + k - 1).choose (5) = ways) := 
begin
  sorry,
end

end balls_in_boxes_ways_l555_555571


namespace semicircle_circumference_approx_l555_555777

noncomputable def semicircle_circumference (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let diameter_semicircle := side_square
  let π := Real.pi
  let circumference_semicircle := (π * diameter_semicircle) / 2 + diameter_semicircle
  circumference_semicircle

-- Proof problem statement
theorem semicircle_circumference_approx : 
  (Real.floor (semicircle_circumference 18 14 * 100) / 100) = 41.13 :=
sorry

end semicircle_circumference_approx_l555_555777


namespace check_ratio_l555_555769

theorem check_ratio (initial_balance check_amount new_balance : ℕ) 
  (h1 : initial_balance = 150) (h2 : check_amount = 50) (h3 : new_balance = initial_balance + check_amount) :
  (check_amount : ℚ) / new_balance = 1 / 4 := 
by { 
  sorry 
}

end check_ratio_l555_555769


namespace y_intercept_of_line_l555_555136

theorem y_intercept_of_line : 
  (∃ t : ℝ, 4 - 4 * t = 0) → (∃ y : ℝ, y = -2 + 3 * 1) := 
by
  sorry

end y_intercept_of_line_l555_555136


namespace range_of_a_l555_555510

theorem range_of_a (x : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  (∃ a : ℝ, ∀ x : ℝ, ∀ θ ∈ set.Icc 0 (π / 2), 
    (x + 3 + 2 * Math.sin θ * Math.cos θ)^2 + (x + a * Math.sin θ + a * Math.cos θ)^2 ≥ 1 / 8) ↔ 
    (a ≤ Real.sqrt 6 ∨ a ≥ 7 / 2) :=
by
  sorry

end range_of_a_l555_555510


namespace sequence_a3_equals_neg8_l555_555919

-- Conditions
def S (n : ℕ) : ℕ → ℤ := λ n, 2 - 2^(n+1)

-- Sequence definition derived from the condition
def a (n : ℕ) : ℕ → ℤ := λ n, S n - S (n - 1)

-- The statement to be proved
theorem sequence_a3_equals_neg8 : a 3 = -8 := by
  sorry

end sequence_a3_equals_neg8_l555_555919


namespace min_value_funct_proof_l555_555662

noncomputable def min_value_funct {a b : ℂ} (h1: a ≠ 0) (h2: b ≠ 0) (h3 : (b / a) ∈ ℝ) (h4 : |a| > |b|) : ℝ :=
  (|a * conj b - conj a * b|) / (2 * |a - b|)

theorem min_value_funct_proof {a b : ℂ} (h1: a ≠ 0) (h2: b ≠ 0) (h3 : (b / a) ∈ ℝ) (h4 : |a| > |b|) :
  ∃ m : ℝ, (∀ t : ℝ, t ≠ -1 → (abs ((a * t + b) / (t + 1)) ≥ m)) ∧ m = min_value_funct h1 h2 h3 h4 :=
sorry

end min_value_funct_proof_l555_555662


namespace box_height_l555_555810

theorem box_height (m n : ℕ) (hmn : Nat.coprime m n) (h_area : Real.sqrt (((m / 2)^2 + 100)) * 17.5 / 2 = 35) : m + n = 9 :=
sorry

end box_height_l555_555810


namespace water_collected_on_third_day_l555_555823

open_locale classical

variable (tank_capacity : ℕ)
variable (initial_fill_fraction : ℚ)
variable (first_day_collection : ℕ)
variable (additional_second_day_collection : ℕ)
variable (final_day_filled : tank_capacity = 100)

theorem water_collected_on_third_day :
  initial_fill_fraction = (2/5 : ℚ) →
  first_day_collection = 15 →
  additional_second_day_collection = 5 →
  tank_capacity = 100 →
  let initial_volume := (initial_fill_fraction * tank_capacity : ℚ).to_nat in
  let first_day_volume := initial_volume + first_day_collection in
  let second_day_collection := first_day_collection + additional_second_day_collection in
  let second_day_volume := first_day_volume + second_day_collection in
  let third_day_collection := tank_capacity - second_day_volume in
  third_day_collection = 25 :=
begin
  intros,
  sorry
end

end water_collected_on_third_day_l555_555823


namespace polygon_perimeter_l555_555385

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555385


namespace exists_prime_divisor_l555_555918
open Nat

theorem exists_prime_divisor (n : ℕ) (h1 : n > 1) (k : ℕ) (hk : k ∈ (Finset.range n).map Finset.succEmb) :
  ∃ p : ℕ, Prime p ∧ p ∣ (factorial n + k) ∧ ∀ j ∈ (Finset.range n).map Finset.succEmb, j ≠ k → ¬ (p ∣ (factorial n + j)) :=
by
  sorry

end exists_prime_divisor_l555_555918


namespace Jen_current_height_jen_current_height_l555_555463

theorem Jen_current_height (h : ℝ) (g : ℝ) (h_baki_now : ℝ) (growth_baki : ℝ) (growth_jen : ℝ) :
  Baki and Jen were originally the same height at h,
  Baki's growth by g is 25%,
  Baki's current height h_baki_now is 75 inches,
  Jen's growth by growth_jen equals two-thirds of growth_baki,
  implies Jen's current height is 70 inches.

theorem jen_current_height
  (h : ℝ)
  (h_baki_now : ℝ)
  (growth_baki : ℝ)
  (growth_jen : ℝ)
  (Hbaki_growth : growth_baki = 0.25 * h)
  (Hjen_growth : growth_jen = 2 / 3 * growth_baki)
  (Hbaki_height : h_baki_now = 75)
  (Hbaki_current : h_baki_now = h + growth_baki)
  : h_baki_now = 75 → growth_baki = 15 → growth_jen = 10 → h = 60 → jen_current_height = h + growth_jen :=
sorry.

end Jen_current_height_jen_current_height_l555_555463


namespace log_x2y2_l555_555974

variable {R : Type*} [LinearOrderedField R] [Nontrivial R]

def log_eq1 (x y : R) : Prop := log (x * y^5) = 1
def log_eq2 (x y : R) : Prop := log (x^3 * y) = 1

theorem log_x2y2 {x y : R} (h1 : log_eq1 x y) (h2 : log_eq2 x y) : log (x^2 * y^2) = 6 / 7 :=
by
  sorry

end log_x2y2_l555_555974


namespace problem_a_b_min_f_log2_x_l555_555937

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 - x + b

theorem problem_a_b (a b : ℝ) (h₁ : f (Real.log2 a) b = b) (h₂ : Real.log2 (f a b) = 2) (h₃ : a > 0) (h₄ : a ≠ 1) :
  a = 2 ∧ b = 2 :=
sorry

theorem min_f_log2_x (x : ℝ) (b : ℝ) (h₁ : b = 2) :
  (∀ y, y > 0 → (f (Real.log2 y) b) ≥ (f (Real.log2 (Real.sqrt 2)) b)) ∧ (∃ x, x = Real.sqrt 2 ∧ f (Real.log2 x) b = 7 / 4) :=
sorry

end problem_a_b_min_f_log2_x_l555_555937


namespace sin_B_in_triangle_ABC_l555_555638

-- Define the properties and conditions
def angle_A : ℝ := 30 -- angle A in degrees
def AC : ℝ := 2       -- length of side AC
def BC : ℝ := Real.sqrt 2 -- length of side BC

-- Theorem statement to prove that sin B equals sqrt(2)/2 given the conditions
theorem sin_B_in_triangle_ABC (A_eq_30 : angle_A = 30) (AC_eq_2 : AC = 2) (BC_eq_sqrt2 : BC = Real.sqrt 2) : 
  ∃ B : ℝ, Real.sin B = Real.sqrt 2 / 2 := 
sorry

end sin_B_in_triangle_ABC_l555_555638


namespace count_valid_ys_l555_555770

theorem count_valid_ys : 
  ∃ ys : Finset ℤ, ys.card = 4 ∧ ∀ y ∈ ys, (y - 3 > 0) ∧ ((y + 3) * (y - 3) * (y^2 + 9) < 2000) :=
by
  sorry

end count_valid_ys_l555_555770


namespace sufficient_but_not_necessary_l555_555538

theorem sufficient_but_not_necessary (a b c : ℝ) (h : c^2 > 0) :
  (ac^2 > bc^2 -> a > b) ∧ (a > b -> ∃c:ℝ, ac^2 > bc^2 → c^2 = 0 ∨ ¬ac^2 > bc^2) :=
by
  sorry

end sufficient_but_not_necessary_l555_555538


namespace coprime_integers_exist_l555_555126

theorem coprime_integers_exist (a b c : ℚ) (t : ℤ) (h1 : a + b + c = t) (h2 : a^2 + b^2 + c^2 = t) (h3 : t ≥ 0) : 
  ∃ (u v : ℤ), Int.gcd u v = 1 ∧ abc = (u^2 : ℚ) / (v^3 : ℚ) :=
by sorry

end coprime_integers_exist_l555_555126


namespace problem_proof_l555_555604

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l555_555604


namespace tangent_line_at_1_extreme_values_in_interval_l555_555553

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6 * x + 5

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 6

-- Tangent line equation at x = 1
theorem tangent_line_at_1 : ∀ (x y : ℝ), y = f_prime 1 * (x - 1) + f 1 ↔ 3 * x - y - 3 = 0 := by
  sorry

-- Maximum and minimum values of f(x) in [-2, 2]
theorem extreme_values_in_interval : 
  ∀ x ∈ set.Icc (-2 : ℝ) 2, ∃ (y_max y_min : ℝ), 
  y_max = 5 + 4 * real.sqrt 2 ∧ 
  y_min = 5 - 4 * real.sqrt 2 ∧ 
  (∀ x ∈ set.Icc (-2 : ℝ) 2, f x ≤ y_max ∧ f x ≥ y_min) := by
  sorry

end tangent_line_at_1_extreme_values_in_interval_l555_555553


namespace fraction_sum_squares_eq_sixteen_l555_555171

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l555_555171


namespace container_capacity_l555_555006

-- Define the given conditions
def initially_full (x : ℝ) : Prop := (1 / 4) * x + 300 = (3 / 4) * x

-- Define the proof problem to show that the total capacity is 600 liters
theorem container_capacity : ∃ x : ℝ, initially_full x → x = 600 := sorry

end container_capacity_l555_555006


namespace alpha_perp_beta_l555_555925

variables (m n : Line) (α β : Plane)

-- Given conditions as definitions
def parallel_lines : Prop := m ∥ n
def line_m_perp_β : Prop := m ⊆ α ∧ m ∥ n ∧ n ⟂ β

-- The theorem
theorem alpha_perp_beta (hmn : parallel_lines m n) (h1 : m ⊆ α) (h2 : n ⟂ β) :
  α ⟂ β :=
sorry

end alpha_perp_beta_l555_555925


namespace action_figure_price_l555_555867

theorem action_figure_price (x : ℝ) (h1 : 2 + 4 * x = 30) : x = 7 :=
by
  -- The proof is provided here
  sorry

end action_figure_price_l555_555867


namespace coefficient_x4_in_expansion_l555_555194

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
  if h : k ≤ n then nat.choose n k else 0

theorem coefficient_x4_in_expansion :
  let f := λ x : ℤ, (1 - x) ^ 5 * (2 * x + 1) in
  let x := 1 in -- dummy x to evaluate coefficients differently
  (coeff f 4) = -15 :=
by
  sorry

end coefficient_x4_in_expansion_l555_555194


namespace triangle_area_546_l555_555827

theorem triangle_area_546 :
  ∀ (a b c : ℕ), a = 13 ∧ b = 84 ∧ c = 85 ∧ a^2 + b^2 = c^2 →
  (1 / 2 : ℝ) * (a * b) = 546 :=
by
  intro a b c
  intro h
  sorry

end triangle_area_546_l555_555827


namespace eustace_age_in_3_years_l555_555075

variable (E M : ℕ)

theorem eustace_age_in_3_years
  (h1 : E = 2 * M)
  (h2 : M + 3 = 21) :
  E + 3 = 39 :=
sorry

end eustace_age_in_3_years_l555_555075


namespace umbrella_numbers_count_l555_555613

-- A three-digit number is represented as a tuple of three digits.
def is_three_digit_num (n : ℕ × ℕ × ℕ) : Prop :=
  n.1 < 10 ∧ n.2 < 10 ∧ n.3 < 10 ∧ n.1 ≠ n.2 ∧ n.2 ≠ n.3 ∧ n.1 ≠ n.3

-- An "umbrella number" means the tens digit > (both hundreds and units digits)
def is_umbrella_number (n : ℕ × ℕ × ℕ) : Prop :=
  is_three_digit_num n ∧ n.2 > n.1 ∧ n.2 > n.3

-- The set of all digits to consider
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}

-- All three-digit numbers formed by selecting 3 different digits from the set {0, 1, 2, 3, 4, 5, 6}
def all_three_digit_numbers : Finset (ℕ × ℕ × ℕ) :=
  (digits.product digits).product digits |>.filter is_three_digit_num

-- The set of all "umbrella numbers" from the above set
def umbrella_numbers : Finset (ℕ × ℕ × ℕ) :=
  all_three_digit_numbers.filter is_umbrella_number

-- Prove that the number of "umbrella numbers" is 55
theorem umbrella_numbers_count : umbrella_numbers.card = 55 :=
  sorry

end umbrella_numbers_count_l555_555613


namespace two_cos_45_eq_sqrt_2_l555_555293

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l555_555293


namespace shuttle_bus_waiting_probability_l555_555005

theorem shuttle_bus_waiting_probability 
  (shuttle_times : set ℝ := {7.5, 8, 8.5})
  (arrival_time : ℝ) 
  (arrival_time_random : true) -- Random arrival is implied in probability space
  (arrival_interval : 7.8333 ≤ arrival_time ∧ arrival_time ≤ 8.5) : 
  (probability (λ t, abs (arrival_time - t) < 0.1667) = 1 / 2) :=
sorry

end shuttle_bus_waiting_probability_l555_555005


namespace number_of_three_star_reviews_l555_555705

theorem number_of_three_star_reviews:
  ∀ (x : ℕ),
  (6 * 5 + 7 * 4 + 1 * 2 + x * 3) / 18 = 4 →
  x = 4 :=
by
  intros x H
  sorry  -- Placeholder for the proof

end number_of_three_star_reviews_l555_555705


namespace cheryl_not_same_color_l555_555797

noncomputable def box_probability : ℚ := 
  let total_ways : ℕ := 2520
  let favorable_outcomes : ℕ := 1080
  in favorable_outcomes / total_ways 

theorem cheryl_not_same_color :
  box_probability = 5 / 12 := by
  simp [box_probability]
  sorry

end cheryl_not_same_color_l555_555797


namespace winner_more_votes_than_second_place_l555_555045

theorem winner_more_votes_than_second_place :
  ∃ (W S T F : ℕ), 
    F = 199 ∧
    W = S + (W - S) ∧
    W = T + 79 ∧
    W = F + 105 ∧
    W + S + T + F = 979 ∧
    W - S = 53 :=
by
  sorry

end winner_more_votes_than_second_place_l555_555045


namespace expression_eval_l555_555057

theorem expression_eval :
  (Real.sqrt 63 / Real.sqrt 7 - Real.abs (-4)) = -1 := by
sorry

end expression_eval_l555_555057


namespace polygon_perimeter_l555_555380

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555380


namespace difference_in_x_coordinates_is_constant_l555_555670

variable {a x₀ y₀ k : ℝ}

-- Define the conditions
def point_on_x_axis (a : ℝ) : Prop := true

def passes_through_fixed_point_and_tangent (a : ℝ) : Prop :=
  a = 1

def equation_of_curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

def tangent_condition (a x₀ y₀ : ℝ) (k : ℝ) : Prop :=
  a > 2 ∧ y₀ > 0 ∧ y₀^2 = 4 * x₀ ∧ 
  (4 * x₀ - 2 * y₀ * y₀ + y₀^2 = 0)

-- The statement
theorem difference_in_x_coordinates_is_constant (a x₀ y₀ k : ℝ) :
  point_on_x_axis a →
  passes_through_fixed_point_and_tangent a →
  equation_of_curve_C x₀ y₀ →
  tangent_condition a x₀ y₀ k → 
  a - x₀ = 2 :=
by
  intro h1 h2 h3 h4 
  sorry

end difference_in_x_coordinates_is_constant_l555_555670


namespace A_marks_is_360_l555_555046

-- Define full marks and the percentages involved
def full_marks := 500
def percentage_80 := 0.80
def percentage_25 := 0.25
def percentage_20 := 0.20
def percentage_10 := 0.10

-- Define marks obtained by D
def D_marks := percentage_80 * full_marks

-- Define marks obtained by C as 20% less than D
def C_marks := D_marks - (percentage_20 * D_marks)

-- Define marks obtained by B as 25% more than C
def B_marks := C_marks + (percentage_25 * C_marks)

-- Define marks obtained by A as 10% less than B
def A_marks := B_marks - (percentage_10 * B_marks)

-- The theorem to prove that A_marks is 360
theorem A_marks_is_360 : A_marks = 360 := by
  sorry

end A_marks_is_360_l555_555046


namespace frequencies_and_quality_difference_l555_555316

theorem frequencies_and_quality_difference 
  (A_first_class A_second_class B_first_class B_second_class : ℕ)
  (total_A total_B : ℕ)
  (total_first_class total_second_class total : ℕ)
  (critical_value_99 confidence_level : ℕ)
  (freq_A freq_B : ℚ)
  (K_squared : ℚ) :
  A_first_class = 150 →
  A_second_class = 50 →
  B_first_class = 120 →
  B_second_class = 80 →
  total_A = 200 →
  total_B = 200 →
  total_first_class = 270 →
  total_second_class = 130 →
  total = 400 →
  critical_value_99 = 10.828 →
  confidence_level = 99 →
  freq_A = 3 / 4 →
  freq_B = 3 / 5 →
  K_squared = 400 * ((150 * 80 - 50 * 120) ^ 2) / (270 * 130 * 200 * 200) →
  K_squared < critical_value_99 →
  freq_A = 3 / 4 ∧ 
  freq_B = 3 / 5 ∧ 
  confidence_level = 99 := 
by
  intros; 
  sorry

end frequencies_and_quality_difference_l555_555316


namespace value_of_expression_l555_555912

open Real

theorem value_of_expression {a : ℝ} (h : a^2 + 4 * a - 5 = 0) : 3 * a^2 + 12 * a = 15 :=
by sorry

end value_of_expression_l555_555912


namespace algebraic_expression_identity_l555_555594

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l555_555594


namespace cody_cooked_dumplings_l555_555476

theorem cody_cooked_dumplings (dumplings_eaten dumplings_left : ℕ) (h1 : dumplings_eaten = 7) (h2 : dumplings_left = 7) : 
  dumplings_eaten + dumplings_left = 14 :=
by
  rw [h1, h2]
  sorry

end cody_cooked_dumplings_l555_555476


namespace number_add_thrice_number_eq_twenty_l555_555015

theorem number_add_thrice_number_eq_twenty (x : ℝ) (h : x + 3 * x = 20) : x = 5 :=
sorry

end number_add_thrice_number_eq_twenty_l555_555015


namespace regular_polygon_perimeter_l555_555387

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555387


namespace parabola_line_intersection_l555_555135

variables (a m n k b : ℝ) (a_nonzero : a ≠ 0) (intersect1 : y₁ = a * (1 - m) * (1 - n) = k + b)
  (intersect2 : y₂ = a * (6 - m) * (6 - n) = 6 * k + b)

theorem parabola_line_intersection (a m n : ℝ) (a_nonzero : a ≠ 0) (h1 : m + n < 7) (h2 : a < 0) :
  ∃ (k : ℝ), k = a * (7 - m - n) ∧ k < 0 :=
by
  use a * (7 - m - n)
  split
  · sorry -- Proof that k = a * (7 - m - n)
  · sorry -- Proof that k < 0

end parabola_line_intersection_l555_555135


namespace decimal_to_fraction_l555_555756

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l555_555756


namespace compute_expression_l555_555609

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l555_555609


namespace area_of_triangle_F1PF2_l555_555232

noncomputable def sqrt5 : ℝ := Real.sqrt 5
noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem area_of_triangle_F1PF2 (P F1 F2 : ℝ × ℝ)
  (hP : P ∈ {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 / 4 = 1})
  (hF1F2_foci : F1 = (1, 0) ∧ F2 = (-1, 0))
  (h_angle : ∠F1 P F2 = 30) :
  let S := (8 - 4 * sqrt3) in
  triangle_area F1 P F2 = S := by
  sorry

end area_of_triangle_F1PF2_l555_555232


namespace monotonic_increasing_minimum_value_l555_555949

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := log x + (1 - x) / (a * x)

-- Theorem 1: Monotonicity of f(x) on [1, +∞) requires a ≥ 1
theorem monotonic_increasing {a : ℝ} (h_a : 0 < a) : 
  (∀ x ∈ set.Ici 1, 0 ≤ deriv (λ x, f x a) x) ↔ (1 ≤ a) :=
sorry

-- Theorem 2: Minimum value of f(x) in the interval [1, 2]
theorem minimum_value {a : ℝ} (h_a : 0 < a) :
  (∀ x ∈ set.Icc 1 2, ∃ y ∈ set.Icc 1 2, f y a ≤ f x a) ∧
  if h1 : a ≥ 1 then 
    (∃ x ∈ set.Icc 1 2, f x a = 0)
  else if h2 : a > 0 ∧ a ≤ (1 / 2) then 
    (∃ x ∈ set.Icc 1 2, f x a = log 2 - (1 / (2 * a)))
  else
    (∃ x ∈ set.Icc 1 2, x = 1 / a ∧ f x a = log (1 / a) + 1 - (1 / a)) :=
sorry

end monotonic_increasing_minimum_value_l555_555949


namespace sum_reciprocal_Q_l555_555559

def S (σ : List ℕ) (k : ℕ) : ℕ :=
  (List.take k σ).sum

def Q (σ : List ℕ) (n : ℕ) : ℕ :=
  List.foldl (λ acc k => acc * S σ k) 1 (List.range n)

theorem sum_reciprocal_Q (n : ℕ) :
  let L := List.range (n + 1)
  let permutations := List.permutations L
  L = List.map (λ i => 2 ^ i) L →
  (∑ σ in permutations, 1 / (Q σ n)) = 1 / (List.foldl (*) 1 L) :=
by
  sorry

end sum_reciprocal_Q_l555_555559


namespace curve_equation_line_intersection_inequality_l555_555915

-- Problem (Ⅰ)
theorem curve_equation (x y : ℝ) (h : sqrt ((x - 1)^2 + y^2) - x = 1) (hx : x > 0) : y^2 = 4 * x := by
  sorry

-- Problem (Ⅱ)
theorem line_intersection_inequality (m : ℝ) (hm: 3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2)
  (A B : ℝ × ℝ) (hA : A.1 = (A.2^2) / 4 ∧ A.2 ∈ {y : ℝ | y * y - 4 * t * y - 4 * m = 0})
  (hB : B.1 = (B.2^2) / 4 ∧ B.2 ∈ {y : ℝ | y * y - 4 * t * y - 4 * m = 0})
  (F : ℝ × ℝ := (1, 0)) : 
  let FA := (A.1 - F.1, A.2 - F.2), FB := (B.1 - F.1, B.2 - F.2) in 
  FA.1 * FB.1 + FA.2 * FB.2 < 0 := 
by
  sorry

end curve_equation_line_intersection_inequality_l555_555915


namespace range_of_g_l555_555511

noncomputable def g (x : Real) : Real := 
  (Real.sin x) ^ 3 + 5 * (Real.sin x) ^ 2 + 2 * (Real.sin x) + 
  3 * (Real.cos x) ^ 2 - 9) / (Real.sin x + 1)

theorem range_of_g :
  ∀ (x : Real), Real.sin x ≠ -1 → g x ∈ Set.Icc (-6 : Real) (-4 : Real) :=
by
  sorry

end range_of_g_l555_555511


namespace probability_y_lt_3x_l555_555163

open Set

/-- Definition of the set of possible values for the pair (x, y) according to the condition. -/
def interval_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | -1 ≤ p.1 ∧ p.1 ≤ 0 ∧ -1 ≤ p.2 ∧ p.2 ≤ 0}

/-- Definition of the event where y < 3x within the given interval. -/
def event_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | -1 ≤ p.1 ∧ p.1 ≤ 0 ∧ -1 ≤ p.2 ∧ p.2 ≤ 0 ∧ p.2 < 3 * p.1}

/--
The probability that y < 3x given x and y are randomly selected
from the interval [-1, 0] is 1/6.
-/
theorem probability_y_lt_3x : (event_set \ interval_set).measure_space / interval_set.measure_space = 1 / 6 :=
sorry

end probability_y_lt_3x_l555_555163


namespace Carolina_mailed_five_letters_l555_555283

-- Definitions translating the given conditions into Lean
def cost_of_mail (cost_letters cost_packages : ℝ) (num_letters num_packages : ℕ) : ℝ :=
  cost_letters * num_letters + cost_packages * num_packages

-- The main theorem to prove the desired answer
theorem Carolina_mailed_five_letters (P L : ℕ)
  (h1 : L = P + 2)
  (h2 : cost_of_mail 0.37 0.88 L P = 4.49) :
  L = 5 := 
sorry

end Carolina_mailed_five_letters_l555_555283


namespace num_subsets_of_P_l555_555162

open Finset

theorem num_subsets_of_P :
  let M := {0, 1, 2, 3, 4}
  let N := {1, 3, 5}
  let P := M ∩ N
  (P.card = 2) → card (powerset P) = 4 :=
by
  intros
  rw [card_powerset, card_eq_two]
  sorry

end num_subsets_of_P_l555_555162


namespace area_of_triangle_PF1F2_l555_555246

noncomputable def ellipse := {P : ℝ × ℝ // (P.1 ^ 2) / 49 + (P.2 ^ 2) / 24 = 1}
noncomputable def F1 := (-5, 0 : ℝ)
noncomputable def F2 := (5, 0 : ℝ)

noncomputable def perpendicular_condition (P : ℝ × ℝ) : Prop :=
  let m := P.1 in let n := P.2 in
  (n / (m + 5)) * (n / (m - 5)) = -1

noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * 10 * abs P.2

theorem area_of_triangle_PF1F2 {P : ellipse} (h : perpendicular_condition P.1) :
  area_of_triangle P.1 = 24 :=
sorry

end area_of_triangle_PF1F2_l555_555246


namespace perimeter_of_polygon_l555_555435

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555435


namespace set_difference_l555_555224

-- Definitions of P and Q
def P : set ℝ := {x | real.log x / real.log 2 < 1}
def Q : set ℝ := {x | abs (x - 2) < 1}

-- Proof statement
theorem set_difference :
  P \ Q = {x | 0 < x ∧ x ≤ 1} :=
sorry

end set_difference_l555_555224


namespace bisect_AH_l555_555661

variables {A B C H D E F P Q N : Type}
variables [triangle A B C] [orthocenter A B C H]
  [foot_of_altitude A B C D] [foot_of_altitude B C A E] 
  [foot_of_altitude C A B F] [intersection DF B P] 
  [perpendicular P B C] [line AB Q] [intersection EQ A N]

theorem bisect_AH (h1 : altitude B C ∠ P H)
  (h2 : perpendicular (line_through P) (line_through B C))
  (h3 : foot_of_altitude E Q N) :
  midpoint N A H :=
sorry

end bisect_AH_l555_555661


namespace necessary_but_not_sufficient_l555_555914

-- Define conditions P and Q
def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

-- Statement to prove
theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x ∧ ¬ (Q x → P x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l555_555914


namespace sum_of_80_consecutive_integers_l555_555303

-- Definition of the problem using the given conditions
theorem sum_of_80_consecutive_integers (n : ℤ) (h : (80 * (n + (n + 79))) / 2 = 40) : n = -39 := by
  sorry

end sum_of_80_consecutive_integers_l555_555303


namespace find_a_l555_555116

theorem find_a (x y a : ℝ) (hx_pos_even : x > 0 ∧ ∃ n : ℕ, x = 2 * n) (hx_le_y : x ≤ y) 
  (h_eq_zero : |3 * y - 18| + |a * x - y| = 0) : 
  a = 3 ∨ a = 3 / 2 ∨ a = 1 :=
sorry

end find_a_l555_555116


namespace base_of_isosceles_triangle_l555_555776

theorem base_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : 3 * a = 45) 
  (h₂ : 2 * b + c = 40) 
  (h₃ : b = a ∨ b = a) : c = 10 := 
sorry

end base_of_isosceles_triangle_l555_555776


namespace parallelogram_height_l555_555893

theorem parallelogram_height (A B H : ℝ) 
    (h₁ : A = 96) 
    (h₂ : B = 12) 
    (h₃ : A = B * H) :
  H = 8 := 
by {
  sorry
}

end parallelogram_height_l555_555893


namespace largest_constant_inequality_l555_555091

variable {a b c d e : ℝ}
variable (h : ∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)

theorem largest_constant_inequality : (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d)) > 2) :=
by
  sorry

end largest_constant_inequality_l555_555091


namespace problem_proof_l555_555603

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l555_555603


namespace tetrahedron_inequality_l555_555220

theorem tetrahedron_inequality (x1 x2 x3 x4 h1 h2 h3 h4 : ℝ) :
  (x1 + x2 + x3 + x4 = h1 + h2 + h3 + h4) →
  (0 ≤ x1) → (0 ≤ x2) → (0 ≤ x3) → (0 ≤ x4) →
  sqrt (h1 + h2 + h3 + h4) ≥ sqrt x1 + sqrt x2 + sqrt x3 + sqrt x4 :=
by {
  intros h_sum h_x1 h_x2 h_x3 h_x4,
  sorry
}

end tetrahedron_inequality_l555_555220


namespace find_x_approx_value_l555_555098

theorem find_x_approx_value :
  ∃ x : ℝ, (log 3 (x - 3) + (log 3 (x^2 - 3) / log 3 (real.sqrt 3)) + (log 3 (x - 3) / log 3 (1/3)) = 5) ∧ (x ≈ 4.311) :=
sorry

end find_x_approx_value_l555_555098


namespace sum_alternating_sequence_l555_555845

theorem sum_alternating_sequence :
  let seq := List.range' 1 102 -- List of integers from 1 through 101
  let alternating_seq := seq.enum.map (λ p, if p.1 % 2 = 0 then p.2 else - p.2)
  (alternating_seq.sum) = 51 :=
by
  sorry

end sum_alternating_sequence_l555_555845


namespace average_physics_chemistry_l555_555818

theorem average_physics_chemistry (P C M : ℕ) 
  (h1 : (P + C + M) / 3 = 80)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 80) :
  (P + C) / 2 = 70 := 
sorry

end average_physics_chemistry_l555_555818


namespace cone_volume_half_sector_rolled_l555_555357

theorem cone_volume_half_sector_rolled {r slant_height h V : ℝ}
  (radius_given : r = 3)
  (height_calculated : h = 3 * Real.sqrt 3)
  (slant_height_given : slant_height = 6)
  (arc_length : 2 * Real.pi * r = 6 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * (r^2) * h) :
  V = 9 * Real.pi * Real.sqrt 3 :=
by {
  sorry
}

end cone_volume_half_sector_rolled_l555_555357


namespace real_solution_to_abs_equation_l555_555569

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end real_solution_to_abs_equation_l555_555569


namespace probability_round_robin_l555_555901

-- Define the probability function p_N for the given conditions
def probability_all_matches_played (N : ℕ) : ℚ :=
  if N < 3 then 0
  else if N = 3 then 1 / 12
  else if N = 4 then 1 / 3
  else if N = 5 then 5 / 7
  else if N = 6 then 20 / 21
  else if N >= 7 then 1
  else 0

-- The theorem to prove
theorem probability_round_robin (N : ℕ) (h1 : 3 ≤ N) (h2 : N ≤ 10) :
  probability_all_matches_played N =
  match N with
  | 3 := 1 / 12
  | 4 := 1 / 3
  | 5 := 5 / 7
  | 6 := 20 / 21
  | 7 := 1
  | 8 := 1
  | 9 := 1
  | 10 := 1
  | _ := 0 := sorry

end probability_round_robin_l555_555901


namespace series_sum_l555_555056

theorem series_sum : (1 - 2 + 3 - 4 + ... - 2022 + 2023) = 1012 := by
  sorry

end series_sum_l555_555056


namespace concave_function_m_range_l555_555487

noncomputable def f (m x : ℝ) : ℝ :=
  exp x + (1/6)*(1-m)*(x^3) - (1/2)*(x^2)*(log x + log m - 3/2)

theorem concave_function_m_range {m : ℝ} :
  (∀ x > 0, (deriv (deriv (λ x, f m x)) x) > 0) →
  0 < m ∧ m < exp 1 :=
by
  -- placeholder for the eventual proof
  sorry

end concave_function_m_range_l555_555487


namespace hare_speed_in_20_foot_race_l555_555675

theorem hare_speed_in_20_foot_race (turtle_speed distance head_start_time : ℕ) 
    (h1 : turtle_speed = 1) 
    (h2 : distance = 20)
    (h3 : head_start_time = 18) :
    let hare_speed := distance / (head_start_time + (distance - turtle_speed * head_start_time) / turtle_speed)
    hare_speed = 10 :=
by
  have hs : hare_speed = 20 / (18 + (20 - 18) / 1) :=
    by rw [h1, h2, h3]
  simp at hs
  exact hs

end hare_speed_in_20_foot_race_l555_555675


namespace sample_size_is_13_l555_555009

noncomputable def stratified_sample_size : ℕ :=
  let A := 120
  let B := 80
  let C := 60
  let total_units := A + B + C
  let sampled_C_units := 3
  let sampling_fraction := sampled_C_units / C
  let n := sampling_fraction * total_units
  n

theorem sample_size_is_13 :
  stratified_sample_size = 13 := by
  sorry

end sample_size_is_13_l555_555009


namespace tan_a_pi_div_6_l555_555137

theorem tan_a_pi_div_6 (a : ℝ) (h : (a, 81) ∈ setOf (λ p : ℝ × ℝ, p.2 = 3 ^ p.1)) : 
  tan (a * π / 6) = -real.sqrt 3 :=
sorry

end tan_a_pi_div_6_l555_555137


namespace compute_expression_l555_555608

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l555_555608


namespace hexadecagon_area_l555_555367

theorem hexadecagon_area (s : ℝ) : 
  let θ := 22.5 / 180 * Real.pi,
      areaOfOneTriangle := 1 / 2 * s^2 * Real.sin θ,
      totalArea := 16 * areaOfOneTriangle
  in totalArea = 3.061 * s^2 :=
by
  let θ := 22.5 / 180 * Real.pi
  let areaOfOneTriangle := 1 / 2 * s^2 * Real.sin θ
  let totalArea := 16 * areaOfOneTriangle
  have h : Real.sin (22.5 / 180 * Real.pi) = 0.382683432
  -- The following will need a rigorous proof, but we will assert the known value for brevity here.
  sorry
  show totalArea = 3.061 * s^2
  -- This will need more rigorous tying back to the exact steps, but we'll assert correctness for now.
  sorry

end hexadecagon_area_l555_555367


namespace find_x_l555_555614

noncomputable theory

def matrix_A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  λ i j, match (i, j) with
        | (0, 0) => 2^(x - 1)
        | (0, 1) => 4
        | (1, 0) => 1
        | (1, 1) => 2
        | _ => 0  -- this case won't occur, added for exhaustiveness 

open Matrix

-- Defining the problem
theorem find_x (x : ℝ) (h : det (matrix_A x) = 0) : x = 2 := by
  sorry

end find_x_l555_555614


namespace calculate_m_plus_b_l555_555861

-- Define the points through which the line passes.
def point1 := (1, 3)
def point2 := (3, 7)
def point3 := (5, 11)

-- Define the slope of the line
def slope (p1 p2 : ℕ × ℕ) : ℚ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept
def intercept (p1 : ℕ × ℕ) (m : ℚ) : ℚ :=
  p1.2 - m * p1.1

-- Check if a point is on the line
def is_on_line (m b : ℚ) (p : ℕ × ℕ) : Prop :=
  p.2 = m * p.1 + b

-- The theorem to prove
theorem calculate_m_plus_b
  (m : ℚ := slope point1 point2)
  (b : ℚ := intercept point1 m) :
  is_on_line m b point3 → m + b = 3 :=
by
  simp [point1, point2, point3, slope, intercept, is_on_line]
  sorry

end calculate_m_plus_b_l555_555861


namespace integer_exponent_terms_count_l555_555629

theorem integer_exponent_terms_count :
  let expr := (λ (x : ℝ), (sqrt x + 1 / cbrt x) ^ 24)
  ∃ n : ℕ, n = 5 ∧ 
    ∀ t ∈ finset.range 25, 
      let term_exp := 12 - 5 * t / 6 in 
      (∃ k : ℤ, term_exp = k → t = 0 ∨ t = 6 ∨ t = 12 ∨ t = 18 ∨ t = 24) :=
begin
  sorry
end

end integer_exponent_terms_count_l555_555629


namespace a₁₀_eq_1000_l555_555160

-- Define the sequence
def odd (n : ℕ) : ℕ := 2 * n + 1

def sum_first_n_odds (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, odd i

-- Prove that the 10th term in the defined sequence is 1000
theorem a₁₀_eq_1000 : sum_first_n_odds 10 = 1000 :=
sorry

end a₁₀_eq_1000_l555_555160


namespace line_always_passes_fixed_point_l555_555359

theorem line_always_passes_fixed_point (m : ℝ) :
  m * 1 + (1 - m) * 2 + m - 2 = 0 :=
by
  sorry

end line_always_passes_fixed_point_l555_555359


namespace h_even_if_g_odd_l555_555655

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end h_even_if_g_odd_l555_555655


namespace perimeter_of_polygon_l555_555436

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555436


namespace Aubriella_water_problem_l555_555462

variable (tank_capacity rate_pouring rate_leaking time_poured time_interval_pour time_interval_leak : ℝ)
variable (water_poured water_leaked net_water additional_water : ℝ)

theorem Aubriella_water_problem
  (hc1 : tank_capacity = 150)
  (hc2 : rate_pouring = 1 / 15)
  (hc3 : rate_leaking = 0.1 / 30)
  (hc4 : time_poured = 525)
  (hc5 : time_interval_pour = 15)
  (hc6 : time_interval_leak = 30) :
  (additional_water = tank_capacity - (water_poured - water_leaked)) :=
by {
  let water_poured := time_poured / time_interval_pour * rate_pouring,
  let num_intervals_leak := time_poured / time_interval_leak,
  let full_intervals_leak := real.floor num_intervals_leak,
  let water_leaked := full_intervals_leak * rate_leaking,
  let net_water := water_poured - water_leaked,
  let additional_water := tank_capacity - net_water,
  sorry
}

end Aubriella_water_problem_l555_555462


namespace sum_first_21_terms_l555_555139

-- Definitions of the conditions
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def f (x : ℝ) : ℝ :=
  sin (2 * x) + 2 * (cos x)^2

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  f (a n)

-- Lean theorem statement to prove the given condition
theorem sum_first_21_terms (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) 
  (ha : is_arithmetic_seq a) (ha11 : a 11 = (3 * Real.pi) / 8) 
  (hf : ∀ n, b n = f (a n)) :
  (Finset.range 21).sum (λ n, b n) = 21 := 
sorry

end sum_first_21_terms_l555_555139


namespace balls_in_boxes_ways_l555_555572

theorem balls_in_boxes_ways : 
  (∃ (ways : ℕ), ways = 21 ∧
    ∀ {k : ℕ}, (5 + k - 1).choose (5) = ways) := 
begin
  sorry,
end

end balls_in_boxes_ways_l555_555572


namespace range_of_y_given_x_l555_555546

theorem range_of_y_given_x (x : ℝ) (h₁ : x > 3) : 0 < (6 / x) ∧ (6 / x) < 2 :=
by 
  sorry

end range_of_y_given_x_l555_555546


namespace parallel_lines_l555_555965

theorem parallel_lines
  (C₁ C₂ : set (ℝ × ℝ))
  (A B : ℕ → ℝ × ℝ)
  (h1 : ∀ n, A n ∈ C₁)
  (h2 : ∀ n, B n ∈ C₂)
  (h3 : ∀ i : ℕ, A i.1, 1 < i <= 2*n → (A i.1+1 - A i.1) = (B i.1+1 - B i.1)) :
  let A0 := A 0 in
  let A2n := A (2 * n) in
  let B2n := B (2 * n) in
  let B0 := B 0 in
  (A0.2 - B2n.2) / (A0.1 - B2n.1) = (A2n.2 - B0.2) / (A2n.1 - B0.1) := sorry

end parallel_lines_l555_555965


namespace handshakes_total_count_l555_555308

/-
Statement:
There are 30 gremlins and 20 imps at a Regional Mischief Meet. Only half of the imps are willing to shake hands with each other.
All cooperative imps shake hands with each other. All imps shake hands with each gremlin. Gremlins shake hands with every
other gremlin as well as all the imps. Each pair of creatures shakes hands at most once. Prove that the total number of handshakes is 1080.
-/

theorem handshakes_total_count (gremlins imps cooperative_imps : ℕ)
  (H1 : gremlins = 30)
  (H2 : imps = 20)
  (H3 : cooperative_imps = imps / 2) :
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_cooperative_imps := cooperative_imps * (cooperative_imps - 1) / 2
  let handshakes_imps_gremlins := imps * gremlins
  handshakes_gremlins + handshakes_cooperative_imps + handshakes_imps_gremlins = 1080 := 
by {
  sorry
}

end handshakes_total_count_l555_555308


namespace regular_polygon_perimeter_l555_555371

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555371


namespace monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l555_555148

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - 2 * a * x

theorem monotonicity_of_f_prime :
  ∀ (x : ℝ) (a : ℝ), 
  (a ≤ 0 → f_prime x a > 0) ∧ 
  (a > 0 → (x < Real.log(2 * a) → f_prime x a < 0) ∧ (x > Real.log(2 * a) → f_prime x a > 0)) :=
sorry

theorem range_of_a_for_real_roots_in_interval :
  1 < a ∧ a < Real.exp 1 - 1 ↔
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ f x a + f_prime x a = 2 - a * x^2 :=
sorry

end monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l555_555148


namespace angle_mul_add_proof_solve_equation_proof_l555_555469

-- For (1)
def angle_mul_add_example : Prop :=
  let a := 34 * 3600 + 25 * 60 + 20 -- 34°25'20'' to seconds
  let b := 35 * 60 + 42 * 60        -- 35°42' to total minutes
  let result := a * 3 + b * 60      -- Multiply a by 3 and convert b to seconds
  let final_result := result / 3600 -- Convert back to degrees
  final_result = 138 + (58 / 60)

-- For (2)
def solve_equation_example : Prop :=
  ∀ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9

theorem angle_mul_add_proof : angle_mul_add_example := sorry

theorem solve_equation_proof : solve_equation_example := sorry

end angle_mul_add_proof_solve_equation_proof_l555_555469


namespace verify_triangle_statements_l555_555205

noncomputable def triangle_statements : Prop :=
  let π := Real.pi in
  ∀ (A B C : ℝ) (a b c : ℝ),
  (a = 1 ∧ b = 2 ∧ A = π / 6) →
  (a / Real.cos A = b / Real.sin B → A = π / 4) ∧
  (Real.sin (2 * A) = Real.sin (2 * B) → (A = B ∨ A + B = π / 2 → false)) ∧
  (∀ {x : ℝ}, x = Real.sin A → x + (Real.sin B) > Real.cos A + (Real.cos B)) →
  (π / 6 = Real.pi / 6)

theorem verify_triangle_statements : triangle_statements :=
  sorry

end verify_triangle_statements_l555_555205


namespace monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l555_555149

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - 2 * a * x

theorem monotonicity_of_f_prime :
  ∀ (x : ℝ) (a : ℝ), 
  (a ≤ 0 → f_prime x a > 0) ∧ 
  (a > 0 → (x < Real.log(2 * a) → f_prime x a < 0) ∧ (x > Real.log(2 * a) → f_prime x a > 0)) :=
sorry

theorem range_of_a_for_real_roots_in_interval :
  1 < a ∧ a < Real.exp 1 - 1 ↔
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ f x a + f_prime x a = 2 - a * x^2 :=
sorry

end monotonicity_of_f_prime_range_of_a_for_real_roots_in_interval_l555_555149


namespace regular_polygon_perimeter_l555_555422

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555422


namespace find_x_l555_555987

-- Define the condition as a Lean equation
def equation (x : ℤ) : Prop :=
  45 - (28 - (37 - (x - 19))) = 58

-- The proof statement: if the equation holds, then x = 15
theorem find_x (x : ℤ) (h : equation x) : x = 15 := by
  sorry

end find_x_l555_555987


namespace cyclic_points_l555_555529

variables (A B C P C1 A1 B1 C2 A2 B2 : Type) [PlaneGeometry A B C P C1 A1 B1 C2 A2 B2] 

theorem cyclic_points
  (hABC : is_triangle A B C)
  (hP: is_point P)
  (hl: is_line l)
  (hl_inter_C1: intersects l A B C C1)
  (hl_inter_A1: intersects l B C A A1)
  (hl_inter_B1: intersects l C A B B1)
  (PC1_circumcircle: intersects_circumcircle P C1 (circumcircle (triangle P A B)) C2)
  (PA1_circumcircle: intersects_circumcircle P A1 (circumcircle (triangle P B C)) A2)
  (PB1_circumcircle: intersects_circumcircle P B1 (circumcircle (triangle P C A)) B2):
  cyclic P A2 B2 C2 :=
by
  sorry

end cyclic_points_l555_555529


namespace min_candies_l555_555348

theorem min_candies : ∃ n : ℕ, (∀ d ∈ {2, 3, 4, 5, 6, 7, 9}, d ∣ n) ∧ n = 1260 :=
by
  use 1260
  sorry

end min_candies_l555_555348


namespace find_common_ratio_l555_555132

noncomputable def common_ratio_of_geometric_sequence (a: ℕ → ℝ) [fact (a 1) ≠ 0] (q: ℝ) : Prop :=
a 3 * a 9 = 2 * (a 5)^2 ∧ q > 0

theorem find_common_ratio {a : ℕ → ℝ} (q : ℝ) [fact (a 1) ≠ 0] :
  common_ratio_of_geometric_sequence a q → q = real.sqrt 2 :=
by
  intro h
  sorry

end find_common_ratio_l555_555132


namespace frequencies_and_confidence_level_l555_555319

namespace MachineQuality

-- Definitions of the given conditions
def productsA := 200
def firstClassA := 150
def secondClassA := 50

def productsB := 200
def firstClassB := 120
def secondClassB := 80

def totalProducts := productsA + productsB
def totalFirstClass := firstClassA + firstClassB
def totalSecondClass := secondClassA + secondClassB

-- 1. Frequencies of first-class products
def frequencyFirstClassA := firstClassA / productsA
def frequencyFirstClassB := firstClassB / productsB

-- 2. \( K^2 \) calculation
def n := 400
def a := 150
def b := 50
def c := 120
def d := 80

def K_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- The theorem to prove the frequencies and the confidence level
theorem frequencies_and_confidence_level : 
    frequencyFirstClassA = (3 / 4) ∧ frequencyFirstClassB = (3 / 5) ∧ K_squared > 6.635 := 
    by {
        sorry -- Proof steps go here
    }

end MachineQuality

end frequencies_and_confidence_level_l555_555319


namespace focus_to_asymptote_distance_eq_sqrt_5_l555_555156

noncomputable def distance_focus_to_asymptote (F : ℝ × ℝ) (A B : ℝ) (a : ℝ) : ℝ :=
  let d := A * F.1 + B * F.2
  abs d / (sqrt (A^2 + B^2))

theorem focus_to_asymptote_distance_eq_sqrt_5
  (a : ℝ) (b : ℝ) (c : ℝ) (h_a : a^2 = 4) (h_c : c = 3)
  (eq_c : c^2 = a^2 + b^2)
  (hf : (3, 0))
  (asymptote_eq : ∀ x, ∃ y, y = (sqrt (5) / 2) * x) :
  distance_focus_to_asymptote (3, 0) (sqrt (5) / 2) (-1) = sqrt 5 := by sorry

end focus_to_asymptote_distance_eq_sqrt_5_l555_555156


namespace lloyd_total_hours_worked_l555_555674

noncomputable def total_hours_worked (daily_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier: ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_hours := 7.5
  let regular_pay := regular_hours * regular_rate
  if total_earnings <= regular_pay then daily_hours else
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / (regular_rate * overtime_multiplier)
  regular_hours + overtime_hours

theorem lloyd_total_hours_worked :
  total_hours_worked 7.5 5.50 1.5 66 = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l555_555674


namespace isabella_haircut_length_l555_555209

-- Define the original length of Isabella's hair.
def original_length : ℕ := 18

-- Define the length of hair cut off.
def cut_off_length : ℕ := 9

-- The length of Isabella's hair after the haircut.
def length_after_haircut : ℕ := original_length - cut_off_length

-- Statement of the theorem we want to prove.
theorem isabella_haircut_length : length_after_haircut = 9 :=
by
  sorry

end isabella_haircut_length_l555_555209


namespace sequence_sum_57_l555_555563

-- Define the sequence using the given conditions.
def sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) = 3 * n

-- The value we want to prove.
theorem sequence_sum_57 (a : ℕ → ℕ) (h : sequence a) :
  a 2 + a 4 + a 6 + a 8 + a 10 + a 12 = 57 := by
  -- Sequence defined by the conditions
  have h1 : a 2 = 2 := h.1
  have h2 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) = 3 * n := h.2
  sorry

end sequence_sum_57_l555_555563


namespace max_value_cos_cos_l555_555284

theorem max_value_cos_cos (x : ℝ) : ∃ y, y = cos (cos x) ∧ y ≤ 1 :=
by 
  -- Since the proof is not required, we leave this part as sorry
  sorry

end max_value_cos_cos_l555_555284


namespace product_of_conversions_l555_555870

theorem product_of_conversions : 
  let b1 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0,
      t1 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  in b1 * t1 = 442 :=
by
  sorry

end product_of_conversions_l555_555870


namespace largest_constant_ineq_l555_555093

theorem largest_constant_ineq (a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
    sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e))
    + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e))
    + sqrt (e / (a + b + c + d)) > 2 :=
sorry

end largest_constant_ineq_l555_555093


namespace find_a_l555_555890

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l555_555890


namespace find_a_l555_555993

theorem find_a (a : ℝ) :
  let A := {-1, 0, 1}
  B := {a - 1, a + (1 / a)}
  in A ∩ B = {0} → a = 1 :=
by
  intros
  sorry

end find_a_l555_555993


namespace simplify_expression_l555_555967

theorem simplify_expression (x y z : ℝ) (h : x * y * z = 1) : (5 ^ (x + y + z)^2) / (5 ^ (x - y + 2 * z)) = 625 :=
by sorry

end simplify_expression_l555_555967


namespace hyperbola_asymptotes_l555_555120

-- Define the hyperbola and its properties 
structure Hyperbola (a b : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)

-- Define conditions for line l intersecting points A and B on hyperbola C to give |AB| = 4a and exactly 3 such lines exist
def satisfies_conditions (C : Hyperbola) (line : ℝ → ℝ) : Prop :=
  ∃ A B : ℝ, 
  |A - B| = 4 * C.a ∧ -- Length condition
  line 0 = 0 ∧        -- Line passing through the origin
  ∃ num_lines, num_lines = 3 -- Exactly three lines

noncomputable def asymptotes_equation (C : Hyperbola) : Prop :=
  ∀ y x : ℝ, y = sqrt 2 * x ∨ y = - sqrt 2 * x

-- Problem statement in Lean 4
theorem hyperbola_asymptotes (a b : ℝ) (C : Hyperbola a b)
  (line : ℝ → ℝ) :
  satisfies_conditions C line →
  asymptotes_equation C :=
sorry

end hyperbola_asymptotes_l555_555120


namespace composite_proposition_l555_555071

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 5 ≤ 4

noncomputable def q : Prop := ∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → ¬ (∀ v : ℝ, v = (Real.sin x + 4 / Real.sin x) → v = 4)

theorem composite_proposition : p ∧ ¬q := 
by 
  sorry

end composite_proposition_l555_555071


namespace tan_alpha_value_l555_555543

variables (α β : ℝ)

theorem tan_alpha_value
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) :
  Real.tan α = 13 / 16 :=
sorry

end tan_alpha_value_l555_555543


namespace largest_base4_to_base10_l555_555282

theorem largest_base4_to_base10 : 
  (3 * 4^2 + 3 * 4^1 + 3 * 4^0) = 63 := 
by
  -- sorry to skip the proof steps
  sorry

end largest_base4_to_base10_l555_555282


namespace clock_strikes_interval_l555_555312

theorem clock_strikes_interval (strikes_1 : ℕ → Prop) (strikes_2 : ℕ → Prop) :
  (∀ n, strikes_1 (2 * n)) ∧ (∀ m, strikes_2 (3 * m)) ∧ 
  (count_strikes (λ t, strikes_1 t ∨ strikes_2 t) = 13) → 
  interval_between_first_and_last_strike (λ t, strikes_1 t ∨ strikes_2 t) = 18 := 
sorry

end clock_strikes_interval_l555_555312


namespace find_like_term_l555_555454

-- Definition of the problem conditions
def monomials : List (String × String) := 
  [("A", "-2a^2b"), 
   ("B", "a^2b^2"), 
   ("C", "ab^2"), 
   ("D", "3ab")]

-- A function to check if two terms can be combined (like terms)
def like_terms(a b : String) : Prop :=
  a = "a^2b" ∧ b = "-2a^2b"

-- The theorem we need to prove
theorem find_like_term : ∃ x, x ∈ monomials ∧ like_terms "a^2b" (x.2) ∧ x.2 = "-2a^2b" :=
  sorry

end find_like_term_l555_555454


namespace max_value_is_one_l555_555231

noncomputable def max_value (x y z : ℝ) : ℝ :=
  (x^2 - 2 * x * y + y^2) * (x^2 - 2 * x * z + z^2) * (y^2 - 2 * y * z + z^2)

theorem max_value_is_one :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 →
  max_value x y z ≤ 1 :=
by sorry

end max_value_is_one_l555_555231


namespace simplify_fraction_120_1800_l555_555697

theorem simplify_fraction_120_1800 :
  (120 : ℚ) / 1800 = (1 : ℚ) / 15 := by
  sorry

end simplify_fraction_120_1800_l555_555697


namespace arithmetic_sequence_sum_of_reciprocals_l555_555852

noncomputable def seq (n : ℕ) : ℝ := sorry -- Define the sequence according to the given conditions

axiom recurring_relation (n : ℕ) : seq n.succ.succ - 2 * seq n.succ + seq n = 1
axiom initial_conditions : seq 1 = 1 ∧ seq 2 = 3

theorem arithmetic_sequence : ∀ n : ℕ, (seq (n + 1) - seq n) = (seq (n + 2) - seq (n + 1)) :=
by
  sorry -- Proof that the sequence {a_{n+1} - a_n} is an arithmetic sequence.

theorem sum_of_reciprocals (n : ℕ) : (∑ i in finset.range n, 1 / seq (i + 1)) = 2 * n / (n + 1) :=
by
  sorry -- Proof that the sum of the first n terms of the sequence {1 / a_n} is 2n / (n + 1).

end arithmetic_sequence_sum_of_reciprocals_l555_555852


namespace num_distinct_points_in_quadrants_l555_555960

def M := {1, -2, 3}
def N := {-4, 5, 6, -7}

theorem num_distinct_points_in_quadrants :
  (M.product N).count (λ p => 0 < p.1 ∧ 0 < p.2) + (M.product N).count (λ p => p.1 < 0 ∧ 0 < p.2) +
  (N.product M).count (λ p => 0 < p.1 ∧ 0 < p.2) + (N.product M).count (λ p => p.1 < 0 ∧ 0 < p.2) = 14 :=
begin
  sorry
end

end num_distinct_points_in_quadrants_l555_555960


namespace quadratic_inequality_solution_l555_555290

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x ∈ set.Ioo (-1 : ℝ) 3) ↔ (x^2 - 2 * x - 3 < 0) :=
by
  intro x
  split
  sorry

end quadratic_inequality_solution_l555_555290


namespace initial_cards_l555_555244

-- The number of cards Nell gave away.
def cards_given_away : ℕ := 28 

-- The number of cards Nell has left.
def cards_left : ℕ := 276 

-- Prove that the initial number of cards is 304.
theorem initial_cards (given_away : ℕ) (left : ℕ) : given_away = cards_given_away → left = cards_left → given_away + left = 304 :=
by
  intros h1 h2
  rw [h1, h2]
  rfl
  sorry

end initial_cards_l555_555244


namespace largest_constant_inequality_l555_555090

variable {a b c d e : ℝ}
variable (h : ∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)

theorem largest_constant_inequality : (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d)) > 2) :=
by
  sorry

end largest_constant_inequality_l555_555090


namespace area_union_square_circle_l555_555026

theorem area_union_square_circle (side_length : ℝ) (radius : ℝ) (h_side : side_length = 12) (h_radius : radius = 6) :
  let A_square := side_length^2,
      A_circle := Real.pi * radius^2,
      A_union := A_square + A_circle - A_circle
  in A_union = 144 :=
by {
  subst h_side,
  subst h_radius,
  let A_square := 12^2,
  let A_circle := Real.pi * 6^2,
  let A_union := A_square + A_circle - A_circle,
  show A_union = 144,
  sorry
}

end area_union_square_circle_l555_555026


namespace total_days_on_jury_duty_l555_555215

-- Define the conditions
def jury_selection_days : ℕ := 2
def trial_duration_factor : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_per_day : ℕ := 24

-- Calculate the trial duration in days
def trial_days : ℕ := trial_duration_factor * jury_selection_days

-- Calculate the total deliberation time in days
def deliberation_total_hours : ℕ := deliberation_days * deliberation_hours_per_day
def deliberation_days_converted : ℕ := deliberation_total_hours / hours_per_day

-- Statement that John spends a total of 14 days on jury duty
theorem total_days_on_jury_duty : jury_selection_days + trial_days + deliberation_days_converted = 14 :=
sorry

end total_days_on_jury_duty_l555_555215


namespace perimeter_of_polygon_l555_555431

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l555_555431


namespace side_length_greater_than_green_segments_l555_555680

-- Define the main structure of the problem
structure EquilateralTriangleProblem where
  a : ℝ -- Side length of the triangle
  r1 r2 r3 : ℝ -- Radii of the circles
  h : ℝ -- Height of the triangle
  yellow_area green_area blue_area : ℝ -- Areas of the colored regions
  triangle_height : h = (a * (Math.sqrt 3)) / 2 -- Height in terms of side length
  r_lt_h : r1 < h ∧ r2 < h ∧ r3 < h -- Radius conditions
  areas : yellow_area = 1000 ∧ green_area = 100 ∧ blue_area = 1 -- Area conditions

theorem side_length_greater_than_green_segments 
  (problem : EquilateralTriangleProblem) : 
  problem.a > sum_green_segments problem :=
sorry

end side_length_greater_than_green_segments_l555_555680


namespace distance_covered_at_40_kmph_l555_555000

theorem distance_covered_at_40_kmph (x : ℝ) : 
  (x / 40 + (250 - x) / 60 = 5.4) → (x = 148) :=
by
  intro h
  sorry

end distance_covered_at_40_kmph_l555_555000


namespace median_segments_intersect_l555_555024

-- Define the necessary structures and conditions
def is_collinear (P Q R : Point) : Prop :=
  ∃ l : Line, (P ∈ l ∧ Q ∈ l ∧ R ∈ l)

def is_bisector (H : Set Point) (PQ : Segment) : Prop :=
  let (P, Q) := PQ in
  (count_points_on_one_side (P, Q, H) = H.card / 2) ∧
  (count_points_on_the_other_side (P, Q, H) = H.card / 2)

-- Define the core theorem
theorem median_segments_intersect
  (H : Set Point) (n : ℕ)
  (h1 : H.card = 2 * n)
  (h2 : ∀ P Q R : Point, P ∈ H → Q ∈ H → R ∈ H → ¬ is_collinear P Q R)
  (h3 : ∃ bisectors : Set Segment, bisectors.card = n ∧ ∀ b ∈ bisectors, is_bisector H b) :
  ∀ b1 b2 ∈ bisectors, b1 ≠ b2 → segments_intersect b1 b2 :=
by
  sorry

end median_segments_intersect_l555_555024


namespace first_discount_calculation_l555_555815

-- Define the given conditions and final statement
theorem first_discount_calculation (P : ℝ) (D : ℝ) :
  (1.35 * (1 - D / 100) * 0.85 = 1.03275) → (D = 10.022) :=
by
  -- Proof is not provided, to be done.
  sorry

end first_discount_calculation_l555_555815


namespace angle_bisector_b_c_sum_l555_555828

theorem angle_bisector_b_c_sum (A B C : ℝ × ℝ)
  (hA : A = (4, -3))
  (hB : B = (-6, 21))
  (hC : C = (10, 7)) :
  ∃ b c : ℝ, (3 * x + b * y + c = 0) ∧ (b + c = correct_answer) :=
by
  sorry

end angle_bisector_b_c_sum_l555_555828


namespace analyze_monotonicity_and_find_a_range_l555_555147

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end analyze_monotonicity_and_find_a_range_l555_555147


namespace part1_tangent_from_origin_part2_monotonic_F_l555_555128

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x - log x
noncomputable def g (x : ℝ) : ℝ := exp x
noncomputable def F (x a : ℝ) : ℝ := (f x a) / (g x)

theorem part1_tangent_from_origin (a : ℝ) :
  let x0 := 1 in
  ∃ (y0 : ℝ), y0 = f x0 a ∧
  (∀ (x : ℝ), f x a - f x0 a = (2 * x0 + a - 1 / x0) * (x - x0)) :=
sorry

theorem part2_monotonic_F (a : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x ≤ 1 → F' x a ≤ 0) → a ≤ 2 :=
sorry

end part1_tangent_from_origin_part2_monotonic_F_l555_555128


namespace double_counted_page_number_l555_555285

theorem double_counted_page_number (n x : ℕ) 
  (h1: 1 ≤ x ∧ x ≤ n)
  (h2: (n * (n + 1) / 2) + x = 1997) : 
  x = 44 := 
by
  sorry

end double_counted_page_number_l555_555285


namespace exists_nat_pair_l555_555500

theorem exists_nat_pair 
  (k : ℕ) : 
  let a := 2 * k
  let b := 2 * k * k + 2 * k + 1
  (b - 1) % (a + 1) = 0 ∧ (a * a + a + 2) % b = 0 := by
  sorry

end exists_nat_pair_l555_555500


namespace a7_equals_21_l555_555632

-- Define the sequence {a_n} recursively
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n + 2) => seq n + seq (n + 1)

-- Statement to prove that a_7 = 21
theorem a7_equals_21 : seq 6 = 21 := 
  sorry

end a7_equals_21_l555_555632


namespace average_of_first_17_even_numbers_l555_555326

noncomputable def average_first_n_even_numbers (n : ℕ) : ℤ :=
  2 * n

theorem average_of_first_17_even_numbers : 
  let n := 17 in
  let sum := (n * (2 + (2 * (n-1)))) / 2 in
  sum / n = 18 :=
by
  sorry

end average_of_first_17_even_numbers_l555_555326


namespace find_a_l555_555891

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l555_555891


namespace peanut_butter_candy_count_l555_555306

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l555_555306


namespace multiple_solutions_no_solution_2891_l555_555481

theorem multiple_solutions (n : ℤ) (x y : ℤ) (h1 : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (u v : ℤ), u ≠ x ∧ v ≠ y ∧ u^3 - 3 * u * v^2 + v^3 = n :=
  sorry

theorem no_solution_2891 (x y : ℤ) (h2 : x^3 - 3 * x * y^2 + y^3 = 2891) :
  false :=
  sorry

end multiple_solutions_no_solution_2891_l555_555481


namespace analyze_monotonicity_and_find_a_range_l555_555146

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end analyze_monotonicity_and_find_a_range_l555_555146


namespace sum_of_commutators_is_zero_l555_555844

variables {α : Type*} [Ring α] {A B C : Matrix α}

/-- Define the commutator of two matrices -/
def commutator (X Y : Matrix α) : Matrix α := X * Y - Y * X

/-- Main theorem stating that the sum of certain commutators is zero -/
theorem sum_of_commutators_is_zero :
  commutator A (commutator B C) + commutator B (commutator C A) + commutator C (commutator A B) = 0 :=
sorry

end sum_of_commutators_is_zero_l555_555844


namespace wall_blocks_required_l555_555830

theorem wall_blocks_required (length feet : ℕ) (height feet : ℕ) (block_height : ℕ) (block_length_1 : ℕ) (block_length_2 : ℕ)
  (rows : ℕ) (odd_row_blocks : ℕ) (even_row_blocks : ℕ) : 
  (height feet = 8) → 
  (block_height = 1) → 
  (length feet = 150) → 
  (block_length_1 = 3) → 
  (block_length_2 = 2) → 
  (rows = 8) → 
  (odd_row_blocks = 50) → 
  (even_row_blocks = 51) → 
  (4 * odd_row_blocks + 4 * even_row_blocks = 404) :=
by 
  intros h_height h_bh l_length b1 b2 r h_odd h_even
  sorry

end wall_blocks_required_l555_555830


namespace find_value_l555_555610

theorem find_value (x : ℝ) (h : 0.20 * x = 80) : 0.40 * x = 160 := 
by
  sorry

end find_value_l555_555610


namespace regular_polygon_perimeter_l555_555369

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l555_555369


namespace general_formula_for_a_sum_of_first_n_terms_of_b_l555_555526

variable {n : ℕ}

-- Conditions
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

-- Proving the general formula for the n-th term of the sequence {a_n}
theorem general_formula_for_a (n : ℕ) : a_n = 2^n := sorry

-- Proving the sum of the first n terms of the sequence {b_n}
theorem sum_of_first_n_terms_of_b (n : ℕ) : 
  T_n = ∑ k in finset.range (n + 1), (S k + real.logb 2 (1 / 2^k)) := 2^(n + 2) - 4 - 2n - (n * (n + 1)) / 2 := sorry

end general_formula_for_a_sum_of_first_n_terms_of_b_l555_555526


namespace coin_difference_l555_555248

/-- 
  Given that Paul has 5-cent, 20-cent, and 15-cent coins, 
  prove that the difference between the maximum and minimum number of coins
  needed to make exactly 50 cents is 6.
-/
theorem coin_difference :
  ∃ (coins : Nat → Nat),
    (coins 5 + coins 20 + coins 15) = 6 ∧
    (5 * coins 5 + 20 * coins 20 + 15 * coins 15 = 50) :=
sorry

end coin_difference_l555_555248


namespace sphere_volume_in_cone_l555_555813

theorem sphere_volume_in_cone :
  let d := 24
  let theta := 90
  let r := 24 * (Real.sqrt 2 - 1)
  let V := (4 / 3) * Real.pi * r^3
  ∃ (R : ℝ), r = R ∧ V = (4 / 3) * Real.pi * R^3 := by
  sorry

end sphere_volume_in_cone_l555_555813


namespace part1_part2_l555_555152

noncomputable def f (a x : ℝ) : ℝ := a - 1/x - Real.log x

theorem part1 (a : ℝ) :
  a = 2 → ∃ m b : ℝ, (∀ x : ℝ, f a x = x * m + b) ∧ (∀ y : ℝ, f a 1 = y → b = y ∧ m = 0) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) → a = 1 :=
by
  sorry

end part1_part2_l555_555152


namespace length_of_AD_l555_555730
noncomputable def unit_circle : Type := sorry   -- Assuming a definition for unit circle

variables (A B C D : unit_circle)
variables (h1: dist A B = 2) -- AB is diameter
variables (h2: ∃ k, dist A C = 3*k ∧ dist C B = 4*k ∧ k ≠ 0) -- |AC| / |CB| = 3 / 4

theorem length_of_AD : dist A D = 5 / 13 :=
by
  sorry

end length_of_AD_l555_555730


namespace fewer_blue_than_green_l555_555820

-- Definitions for given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def total_buttons : ℕ := 275
def blue_buttons : ℕ := total_buttons - (green_buttons + yellow_buttons)

-- Theorem statement to be proved
theorem fewer_blue_than_green : green_buttons - blue_buttons = 5 :=
by
  -- Proof is omitted as per the instructions
  sorry

end fewer_blue_than_green_l555_555820


namespace exists_N_in_V_with_multiple_factorizations_l555_555666

variable (p : ℕ) (hp : p.prime) (h5 : 5 < p)

def V (p : ℕ) : set ℕ :=
  {n : ℕ | ∃ k : ℕ, n = k * p + 1 ∨ n = k * p - 1}

def indecomposable_in_V (n : ℕ) : Prop :=
  n ∈ V p ∧ ∀ k l : ℕ, k ∈ V p → l ∈ V p → n ≠ k * l

theorem exists_N_in_V_with_multiple_factorizations :
  ∃ N : ℕ, N ∈ V p ∧
  ∃ a₁ b₁ a₂ b₂ : ℕ, 
    a₁ ∈ V p ∧ b₁ ∈ V p ∧ indecomposable_in_V p a₁ ∧ indecomposable_in_V p b₁ ∧ 
    a₂ ∈ V p ∧ b₂ ∈ V p ∧ indecomposable_in_V p a₂ ∧ indecomposable_in_V p b₂ ∧ 
    N = a₁ * b₁ ∧ N = a₂ * b₂ ∧ (a₁ ≠ a₂ ∨ b₁ ≠ b₂) :=
sorry

end exists_N_in_V_with_multiple_factorizations_l555_555666


namespace john_pre_lunch_drive_l555_555640

def drive_before_lunch (h : ℕ) : Prop :=
  45 * h + 45 * 3 = 225

theorem john_pre_lunch_drive : ∃ h : ℕ, drive_before_lunch h ∧ h = 2 :=
by
  sorry

end john_pre_lunch_drive_l555_555640


namespace loss_per_metre_is_5_l555_555816

-- Definitions
def selling_price (total_meters : ℕ) : ℕ := 18000
def cost_price_per_metre : ℕ := 65
def total_meters : ℕ := 300

-- Loss per meter calculation
def loss_per_metre (selling_price : ℕ) (cost_price_per_metre : ℕ) (total_meters : ℕ) : ℕ :=
  ((cost_price_per_metre * total_meters) - selling_price) / total_meters

-- Theorem statement
theorem loss_per_metre_is_5 : loss_per_metre (selling_price total_meters) cost_price_per_metre total_meters = 5 :=
by
  sorry

end loss_per_metre_is_5_l555_555816


namespace find_a_l555_555873

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l555_555873


namespace sum_a100_l555_555527

noncomputable def sequence (a : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = C

theorem sum_a100 
  (a : ℕ → ℤ)
  (C : ℤ)
  (h_seq : sequence a)
  (h_a5 : a 5 = 2)
  (h_a7 : a 7 = -3)
  (h_a9 : a 9 = 4) : 
  (∑ i in range 1 101, a i) = 102 := 
sorry

end sum_a100_l555_555527


namespace find_a_for_binomial_square_l555_555884

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l555_555884


namespace regular_polygon_perimeter_l555_555423

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555423


namespace associate_professor_charts_l555_555460

theorem associate_professor_charts (A B C : ℕ) : 
  A + B = 8 → 
  2 * A + B = 10 → 
  C * A + 2 * B = 14 → 
  C = 1 := 
by 
  intros h1 h2 h3 
  sorry

end associate_professor_charts_l555_555460


namespace compute_expression_l555_555607

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l555_555607


namespace sum_G_div_five_pow_eq_35_div_13_l555_555222

noncomputable def G : ℕ → ℕ 
| 0     := 1
| 1     := 2
| (n+2) := G (n+1) + 2 * G n

theorem sum_G_div_five_pow_eq_35_div_13 :
  (∑' n : ℕ, (G n : ℚ) / 5^n) = 35 / 13 := by
  sorry

end sum_G_div_five_pow_eq_35_div_13_l555_555222


namespace problem1_problem2_l555_555846

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (m^2 - 4) / (4 + 4 * m + m^2) / ((m - 2) / (2 * m - 2)) * ((m + 2) / (m - 1)) = 2 := 
sorry

end problem1_problem2_l555_555846


namespace nth_term_of_arithmetic_seq_sum_first_n_terms_sequence_l555_555923

-- Part (1) definitions
def is_arithmetic_seq (a : ℕ → ℕ) :=
  ∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (finset.range n).sum a

-- Given conditions
def condition1 := sum_first_n_terms (λ n, 1 + (n - 1)) 3 = 6
def condition2 := sum_first_n_terms (λ n, 1 + (n - 1)) 6 = 21

-- Question and answer for Part (1)
theorem nth_term_of_arithmetic_seq (a : ℕ → ℕ) (n : ℕ) (h1 : is_arithmetic_seq a)
  (h2 : condition1) (h3 : condition2) : a n = n :=
sorry

-- Part (2) definitions
def sum_first_n_terms_exp (a : ℕ → ℕ) (n : ℕ) :=
  (finset.range n).sum (λ k, a (k + 1) * 2^(k + 1))

-- Given definitions for sequence {a_n} 
def a_n (n : ℕ) : ℕ := n

-- Question and answer for Part (2)
theorem sum_first_n_terms_sequence (n : ℕ) :
  sum_first_n_terms_exp a_n n = 2 + (n - 1) * 2^(n + 1) :=
sorry

end nth_term_of_arithmetic_seq_sum_first_n_terms_sequence_l555_555923


namespace length_of_angle_bisector_leq_median_l555_555684

-- Define the entities and conditions
variables {A B C D M : Type} [in_triangl ABC]
variables (a b : ℝ)

-- Define the Length Bisector and Median properties
def AngleBisectorTheorem (A B C D : Type) :=
  ∀ (b : ℝ) (a : ℝ), (AD / DB) = (b / a)

def MedianBisects (A B C : Type) :=
  ∀ (A B : ℝ), A = B

-- The main theorem to prove
theorem length_of_angle_bisector_leq_median (A B C D M : Type) [in_triangl ABC] [AngleBisectorTheorem A B C D (b : ℝ) (a : ℝ)] [MedianBisects C M (A : ℝ) (B : ℝ)] : 
  CD ≤ CM :=
sorry

end length_of_angle_bisector_leq_median_l555_555684


namespace perimeter_of_regular_polygon_l555_555417

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555417


namespace total_emeralds_in_boxes_l555_555725

-- Define the problem conditions and goal
theorem total_emeralds_in_boxes
    (D R E : ℕ)
    (total_boxes : nat)
    (boxes_diamonds : nat)
    (boxes_emeralds : nat)
    (boxes_rubies : nat)
    (rubies_more_than_diamonds : R = D + 15)
    (total_boxes = 6)
    (boxes_diamonds = 2)
    (boxes_emeralds = 2)
    (boxes_rubies = 2) : 
    E = 12 := 
sorry

end total_emeralds_in_boxes_l555_555725


namespace find_n_l555_555932

theorem find_n (x : ℝ) (n : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -1)
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 n - 2)) : 
  n = 120 :=
by
  sorry

end find_n_l555_555932


namespace frequencies_and_quality_difference_l555_555317

theorem frequencies_and_quality_difference 
  (A_first_class A_second_class B_first_class B_second_class : ℕ)
  (total_A total_B : ℕ)
  (total_first_class total_second_class total : ℕ)
  (critical_value_99 confidence_level : ℕ)
  (freq_A freq_B : ℚ)
  (K_squared : ℚ) :
  A_first_class = 150 →
  A_second_class = 50 →
  B_first_class = 120 →
  B_second_class = 80 →
  total_A = 200 →
  total_B = 200 →
  total_first_class = 270 →
  total_second_class = 130 →
  total = 400 →
  critical_value_99 = 10.828 →
  confidence_level = 99 →
  freq_A = 3 / 4 →
  freq_B = 3 / 5 →
  K_squared = 400 * ((150 * 80 - 50 * 120) ^ 2) / (270 * 130 * 200 * 200) →
  K_squared < critical_value_99 →
  freq_A = 3 / 4 ∧ 
  freq_B = 3 / 5 ∧ 
  confidence_level = 99 := 
by
  intros; 
  sorry

end frequencies_and_quality_difference_l555_555317


namespace main_l555_555237

def prop_p (x0 : ℝ) : Prop := x0 > -2 ∧ 6 + abs x0 = 5
def p : Prop := ∃ x : ℝ, prop_p x

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, abs x + abs y ≤ 1 → abs y / (abs x + 2) ≤ 1 / 2
def not_r : Prop := ∃ x y : ℝ, abs x + abs y > 1 ∧ abs y / (abs x + 2) > 1 / 2

theorem main : ¬ p ∧ ¬ p ∨ r ∧ (p ∧ q) := by
  sorry

end main_l555_555237


namespace rhombus_perimeter_l555_555715

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  (4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))) = 52 := by
  sorry

end rhombus_perimeter_l555_555715


namespace moles_of_C2H5Cl_formed_l555_555896

theorem moles_of_C2H5Cl_formed (n_C2H6 : ℕ) (n_Cl2 : ℕ) 
    (h1 : n_C2H6 = 3) (h2 : n_Cl2 = 3) : 
    let n_C2H5Cl := n_C2H6 in
    n_C2H5Cl = 3 := by
  intros
  rw [h1]
  rw [h2]
  rfl

end moles_of_C2H5Cl_formed_l555_555896


namespace alfred_wins_sixth_game_l555_555833

noncomputable def compute_probability : ℚ :=
  let a₀ := 2 / 3 in
  let b₀ := 1 / 3 in
  let rec step (a b : ℚ) (n : ℕ) : ℚ × ℚ :=
    if n = 0 then (a, b)
    else 
      let (a', b') := step a b (n - 1)
      (1 / 3 * a' + 2 / 3 * b', 2 / 3 * a' + 1 / 3 * b')
  in
  fst (step a₀ b₀ 6)

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem alfred_wins_sixth_game :
  last_three_digits (364 + 729) = 93 :=
by
  have hp : compute_probability = 364 / 729, from sorry,
  have hn : 364 + 729 = 1093, from sorry,
  show last_three_digits 1093 = 93, from sorry

end alfred_wins_sixth_game_l555_555833


namespace monotonically_decreasing_range_l555_555280

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, x < 1 → (deriv (λ x, x^2 - 2 * a * x + a^2)) x ≤ 0) ↔ a ≥ 1 :=
by
  sorry

end monotonically_decreasing_range_l555_555280


namespace normal_probability_l555_555138

noncomputable def normal_prob {μ σ : ℝ} (x : ℝ) : ℝ :=
  (Real.to_nnreal (1 / (σ * Real.sqrt (2 * Real.pi))) * 
  Real.exp (-(x - μ)^2 / (2 * σ^2))).toReal

theorem normal_probability (σ : ℝ)
  (h1 : ∀ ξ, ∫ x in -∞..ξ, normal_prob 2 σ x = 0.84) :
  ∫ x in -∞..0, normal_prob 2 2 x = 0.16 :=
by
  sorry

end normal_probability_l555_555138


namespace diagonal_length_of_closet_l555_555274

theorem diagonal_length_of_closet :
  ∃ (d : ℝ), (∀ (a b : ℝ), a = 4 ∧ a * b = 27 → d = Real.sqrt (a^2 + b^2)) :=
begin
  sorry
end

end diagonal_length_of_closet_l555_555274


namespace shift_parabola_l555_555271

theorem shift_parabola :
  ∀ x : ℝ, (∃ y : ℝ, y = 2 * (x - 3)^2 + 2) → 
           (∃ y' : ℝ, y' = 2 * x^2) :=
begin
  sorry
end

end shift_parabola_l555_555271


namespace lucas_numbers_with_digit_one_2016_l555_555269

def contains_digit_one (n : ℕ) : Prop :=
  ∃ k, '1' = (n.toString.data.get ⟨k, sorry⟩)

def lucas_number : ℕ → ℕ
| 0     := 2
| 1     := 1
| (n+2) := lucas_number (n+1) + lucas_number n

def count_lucas_numbers_with_digit_one : ℕ → ℕ
| 0     := 0
| (n+1) := count_lucas_numbers_with_digit_one n + if contains_digit_one (lucas_number (n+1)) then 1 else 0

theorem lucas_numbers_with_digit_one_2016 :
  count_lucas_numbers_with_digit_one 2016 = 1984 :=
sorry

end lucas_numbers_with_digit_one_2016_l555_555269


namespace sum_product_smallest_number_l555_555740

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l555_555740


namespace percentage_orange_juice_l555_555011

-- Definitions based on conditions
def total_volume : ℝ := 120
def watermelon_percentage : ℝ := 0.60
def grape_juice_volume : ℝ := 30
def watermelon_juice_volume : ℝ := watermelon_percentage * total_volume
def combined_watermelon_grape_volume : ℝ := watermelon_juice_volume + grape_juice_volume
def orange_juice_volume : ℝ := total_volume - combined_watermelon_grape_volume

-- Lean 4 statement to prove the percentage of orange juice
theorem percentage_orange_juice : (orange_juice_volume / total_volume) * 100 = 15 := by
  -- sorry to skip the proof
  sorry

end percentage_orange_juice_l555_555011


namespace two_cos_45_eq_sqrt_2_l555_555294

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l555_555294


namespace value_of_power_l555_555985

theorem value_of_power (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2014 = 1 :=
by
  sorry

end value_of_power_l555_555985


namespace eulers_formula_l555_555119

structure PlanarGraph :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)
(connected : Prop)

theorem eulers_formula (G: PlanarGraph) (H_conn: G.connected) : G.vertices - G.edges + G.faces = 2 :=
sorry

end eulers_formula_l555_555119


namespace retail_price_before_discount_l555_555812

variable (R : ℝ) -- Let R be the retail price of each machine before the discount

theorem retail_price_before_discount :
    let wholesale_price := 126
    let machines := 10
    let bulk_discount_rate := 0.05
    let profit_margin := 0.20
    let sales_tax_rate := 0.07
    let discount_rate := 0.10

    -- Calculate wholesale total price
    let wholesale_total := machines * wholesale_price

    -- Calculate bulk purchase discount
    let bulk_discount := bulk_discount_rate * wholesale_total

    -- Calculate total amount paid
    let amount_paid := wholesale_total - bulk_discount

    -- Calculate profit per machine
    let profit_per_machine := profit_margin * wholesale_price
    
    -- Calculate total profit
    let total_profit := machines * profit_per_machine

    -- Calculate sales tax on profit
    let tax_on_profit := sales_tax_rate * total_profit

    -- Calculate total amount after paying tax
    let total_amount_after_tax := (amount_paid + total_profit) - tax_on_profit

    -- Express total selling price after discount
    let total_selling_after_discount := machines * (0.90 * R)

    -- Total selling price after discount is equal to total amount after tax
    (9 * R = total_amount_after_tax) →
    R = 159.04 :=
by
  sorry

end retail_price_before_discount_l555_555812


namespace max_AB_value_l555_555217

-- Define the given conditions of the quadrilateral
variables
  (A B C D : Type)
  (AB AD BC CD : ℕ)
  (angle_A angle_B : ℝ)
  (h1 : angle_A = 120)
  (h2 : angle_B = 120)
  (h3 : |AD - BC| = 42)
  (h4 : CD = 98)
  (is_convex_quad : convex_quadrilateral A B C D)

-- State the theorem to prove AB = 69 under given conditions
theorem max_AB_value
  [positive_integer_sidelengths : ∀ x, x = AB ∨ x = AD ∨ x = BC ∨ x = CD → x > 0]
  [convex_quadrilateral_sides : quadrilateral_side_lengths AB AD BC CD]
  (h : convex_quadrilateral_sides = is_convex_quad)
  : AB = 69 := 
sorry

end max_AB_value_l555_555217


namespace club_members_after_four_years_l555_555181

theorem club_members_after_four_years
  (b : ℕ → ℕ)
  (h_initial : b 0 = 20)
  (h_recursive : ∀ k, b (k + 1) = 3 * (b k) - 10) :
  b 4 = 1220 :=
sorry

end club_members_after_four_years_l555_555181


namespace extreme_values_a3_increasing_function_range_of_a_range_of_a_under_conditions_l555_555556

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

-- Part I: Extreme values for a = 3
theorem extreme_values_a3 :
  let a := 3
  ∃ x1 x2 : ℝ, f x1 3 = (-5/4) - Real.log 2 ∧ f x2 3 = -2 := by
  sorry

-- Part II: Range of values for a when f(x) is increasing
theorem increasing_function_range_of_a :
  (∀ x : ℝ, x > 0 → f' x 0 + 2 * x - a ≥ 0) →
  ∀ a : ℝ, a ≤ 2 * Real.sqrt 2 := by
  sorry

-- Part III: Range of values for a under the given conditions
theorem range_of_a_under_conditions (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x a ≤ (3 * x^2 + (1 / x^2) - 6 * x) / 2) →
  2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end extreme_values_a3_increasing_function_range_of_a_range_of_a_under_conditions_l555_555556


namespace even_perfect_square_factors_count_l555_555166

theorem even_perfect_square_factors_count:
  let n := (2^6 * 3^4 * 5^2)
  let is_even_perfect_square (m : ℕ) := 
    ∃ a b c : ℕ, 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 4 ∧ 0 ≤ c ∧ c ≤ 2 ∧
      a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ 
      m = 2^a * 3^b * 5^c ∧ a ≥ 1
  in
  (finset.filter is_even_perfect_square (finset.range (n + 1))).card = 18 := 
sorry

end even_perfect_square_factors_count_l555_555166


namespace distance_from_P_to_l_l555_555145

variables (A P : EuclideanSpace ℝ (Fin 3))
variables (a : (Fin 3) → ℝ)

def line_l_through_point (A : EuclideanSpace ℝ (Fin 3)) (a : (Fin 3) → ℝ) : EuclideanSpace ℝ (Fin 3) → Prop :=
  λ x, ∃ t : ℝ, x = A + t • a

axiom point_A_def : A = ![1, 0, -1]
axiom point_P_def : P = ![-1, 2, 0]
axiom dir_vec_a_def : a = ![1, 2, 2]

noncomputable def distance_point_to_line (A P : EuclideanSpace ℝ (Fin 3)) (a : (Fin 3) → ℝ) : ℝ :=
  let AP := P - A in
  let n_norm := ‖a‖ in
  ‖AP - ((AP ⬝ a) / (n_norm * n_norm)) • a‖

theorem distance_from_P_to_l : distance_point_to_line A P a = (Real.sqrt 65) / 3 :=
sorry

end distance_from_P_to_l_l555_555145


namespace regular_polygon_perimeter_l555_555448

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555448


namespace ratio_of_areas_l555_555622

theorem ratio_of_areas (h a b R : ℝ) (h_triangle : a^2 + b^2 = h^2) (h_circumradius : R = h / 2) :
  (π * R^2) / (1/2 * a * b) = π * h / (4 * R) :=
by sorry

end ratio_of_areas_l555_555622


namespace decimal_to_fraction_l555_555757

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l555_555757


namespace solve_quadratic_equation_l555_555700

theorem solve_quadratic_equation (x : ℂ) :
    (3 * x^2 - 10 * x + 6 = 0) ->
    (x = (5 / 3) + Complex.sqrt 7 / 3 ∨ x = (5 / 3) - Complex.sqrt 7 / 3) :=
by
  intro h
  sorry

end solve_quadratic_equation_l555_555700


namespace find_power_l555_555799

noncomputable def x : Real := 14.500000000000002
noncomputable def target : Real := 126.15

theorem find_power (n : Real) (h : (3/5) * x^n = target) : n = 2 :=
sorry

end find_power_l555_555799


namespace positive_solution_for_x_l555_555961

/-- Given the system of equations:
  xy = 8 - 2x - 3y
  yz = 8 - 4y - 2z
  xz = 40 - 4x - 3z
 prove that the positive solution for x is x = (7 * real.sqrt 13 - 6) / 2. -/
theorem positive_solution_for_x
  (x y z : ℝ)
  (h1 : x * y = 8 - 2 * x - 3 * y)
  (h2 : y * z = 8 - 4 * y - 2 * z)
  (h3 : x * z = 40 - 4 * x - 3 * z) :
  x = (7 * real.sqrt 13 - 6) / 2 := 
sorry

end positive_solution_for_x_l555_555961


namespace all_diameters_and_radii_equal_in_same_circle_l555_555199

theorem all_diameters_and_radii_equal_in_same_circle
  (C : set Point) -- Let C be the set of all points on the circle
  (r : ℝ) -- Let r be the radius of the circle
  (center : Point) -- Let center be the center of the circle
  (H1 : ∀ p ∈ C, dist center p = r)  -- All points p in C are at distance r from center
  (diameter : ℝ) -- Let diameter be the diameter of the circle
  (H2 : diameter = 2 * r)  -- The diameter of a circle is twice the radius
  : (∀ d1 d2 : ℝ, (∃ p1 p2 ∈ C, dist p1 p2 = d1 ∧ d1 = diameter) → (∃ p3 p4 ∈ C, dist p3 p4 = d2 ∧ d2 = diameter) → d1 = d2)
    ∧ (∀ r1 r2 : ℝ, (∃ p ∈ C, dist center p = r1) → (∃ p ∈ C, dist center p = r2) → r1 = r2) :=
by
  sorry

end all_diameters_and_radii_equal_in_same_circle_l555_555199


namespace collinear_points_l555_555459

variable (α β γ δ E : Type)
variables {A B C D K L P Q : α}
variables (convex : α → α → α → α → Prop)
variables (not_parallel : α → α → Prop)
variables (internal_bisector : α → α → α → Prop)
variables (external_bisector : α → α → α → Prop)
variables (collinear : α → α → α → α → Prop)

axiom convex_quad : convex A B C D
axiom AD_not_parallel_BC : not_parallel A D ∧ not_parallel B C

axiom internal_bisectors :
  internal_bisector A B K ∧ internal_bisector B A K ∧ internal_bisector C D P ∧ internal_bisector D C P

axiom external_bisectors :
  external_bisector A B L ∧ external_bisector B A L ∧ external_bisector C D Q ∧ external_bisector D C Q

theorem collinear_points : collinear K L P Q := 
sorry

end collinear_points_l555_555459


namespace average_reading_time_l555_555497

theorem average_reading_time (t_Emery t_Serena : ℕ) (h1 : t_Emery = 20) (h2 : t_Serena = 5 * t_Emery) : 
  (t_Emery + t_Serena) / 2 = 60 := 
by
  sorry

end average_reading_time_l555_555497


namespace polynomial_remainder_l555_555732

theorem polynomial_remainder :
  ∀ x : ℝ, (x^2 - 1) ∣ (x^12 - x^6 + 1) % (x^2 - 1) = 1 :=
by
  intro x
  sorry

end polynomial_remainder_l555_555732


namespace regular_polygon_perimeter_l555_555425

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555425


namespace perimeter_of_regular_polygon_l555_555408

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555408


namespace player_max_scores_l555_555322

structure Game :=
(player1_cards : Finset ℕ)
(player2_cards : Finset ℕ)
(turns : ℕ)
(player1_starts : Bool)

def game_max_score (g : Game) (player1_score player2_score : ℕ) : Prop :=
player1_score = 999 ∧ player2_score = 1

theorem player_max_scores :
  ∀ (g : Game),
    g.player1_cards = {2, 4, ..., 2000} ∧ g.player2_cards = {1, 3, ..., 2001} ∧ g.turns = 1000 ∧ g.player1_starts →
    ∃ (player1_score player2_score : ℕ), game_max_score g player1_score player2_score :=
by
  intro g
  sorry

end player_max_scores_l555_555322


namespace real_solutions_condition_l555_555780

theorem real_solutions_condition (a b : ℝ) (x y : ℝ) 
  (h1 : x + y = 2 * a) 
  (h2 : x * y * (x^2 + y^2) = 2 * b^4) : 
  a = 10 → b^4 = 9375 → a^2 ≥ b^2 :=
by
  intros ha hb
  have ha_eq : a = 10 := ha
  have hb_pow : b^4 = 9375 := hb
  have ha_square : a^2 = 10^2 := by rw ha_eq; norm_num
  have hb_square : b^2 = Real.sqrt (9375 : ℝ ) := by sorry
  have result : 10^2 ≥ Real.sqrt (9375) := sorry
  exact result

end real_solutions_condition_l555_555780


namespace div_factorial_l555_555668

theorem div_factorial (n q : ℕ) 
  (h₁ : n ≥ 5) 
  (h₂ : 2 ≤ q) 
  (h₃ : q ≤ n) : 
  q - 1 ∣ (Nat.div (n - 1)! q) := 
sorry

end div_factorial_l555_555668


namespace find_a_for_square_binomial_l555_555878

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l555_555878


namespace percentage_of_bags_not_sold_l555_555643

theorem percentage_of_bags_not_sold
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : sold_monday = 25)
  (h_tuesday : sold_tuesday = 70)
  (h_wednesday : sold_wednesday = 100)
  (h_thursday : sold_thursday = 110)
  (h_friday : sold_friday = 145) : 
  (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) * 100 / initial_stock = 25 :=
by
  sorry

end percentage_of_bags_not_sold_l555_555643


namespace length_of_angle_bisector_leq_median_l555_555683

-- Define the entities and conditions
variables {A B C D M : Type} [in_triangl ABC]
variables (a b : ℝ)

-- Define the Length Bisector and Median properties
def AngleBisectorTheorem (A B C D : Type) :=
  ∀ (b : ℝ) (a : ℝ), (AD / DB) = (b / a)

def MedianBisects (A B C : Type) :=
  ∀ (A B : ℝ), A = B

-- The main theorem to prove
theorem length_of_angle_bisector_leq_median (A B C D M : Type) [in_triangl ABC] [AngleBisectorTheorem A B C D (b : ℝ) (a : ℝ)] [MedianBisects C M (A : ℝ) (B : ℝ)] : 
  CD ≤ CM :=
sorry

end length_of_angle_bisector_leq_median_l555_555683


namespace length_segment_EC_l555_555781

structure Trapezoid (AB CD : ℝ) (E : Type) :=
  (parallel : AB ≠ CD)
  (AB_eq_3CD : AB = 3 * CD)
  (diags_intersect : E)
  (diags_perpendicular : Angle E = 90)
  (BD_length : BD = 15)

theorem length_segment_EC {AB CD : ℝ} {E : Type} (T : Trapezoid AB CD E) : 
  segment_length EC = 15 / 4 :=
by
  sorry

end length_segment_EC_l555_555781


namespace volume_of_parallelepiped_l555_555581

-- Define a unit vector and angle between vectors
variables (a b : ℝ^3) (θ : ℝ)
-- Define the conditions that a and b are unit vectors, and the angle between them is π/4
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hθ : θ = Real.pi / 4)

-- Define cross product, dot product, and the volume (scalar triple product)
def volume_parallelepiped := |a • ((b + b × a) × b)|

-- Theorem stating the volume of the parallelepiped
theorem volume_of_parallelepiped (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hθ : θ = Real.pi / 4) :
  volume_parallelepiped a b = 1 / 2 :=
by
  sorry

end volume_of_parallelepiped_l555_555581


namespace minimal_polynomial_characterization_l555_555897

noncomputable def minimal_polynomial (x : ℝ) : ℝ :=
  x^4 - 6 * x^3 + 8 * x^2 - 4 * x - 4

theorem minimal_polynomial_characterization :
  ∃ (P : ℝ → ℝ), (∀ x, P x = x^4 - 6 * x^3 + 8 * x^2 - 4 * x - 4) ∧
  (P (2 + Real.sqrt 2) = 0) ∧
  (P (2 - Real.sqrt 2) = 0) ∧
  (P (1 + Real.sqrt 3) = 0) ∧
  (P (1 - Real.sqrt 3) = 0) ∧
  (∀ (Q : ℝ → ℝ), (leading_coeff Q = 1) ∧ 
     (Q (2 + Real.sqrt 2) = 0) ∧
     (Q (2 - Real.sqrt 2) = 0) ∧
     (Q (1 + Real.sqrt 3) = 0) ∧
     (Q (1 - Real.sqrt 3) = 0) → (degree Q ≥ 4)) :=
by
  use minimal_polynomial
  split
  { intro x,
    refl },
  -- Here's where you'd continue to prove the roots and minimum degree
  sorry

end minimal_polynomial_characterization_l555_555897


namespace only_fourth_is_rational_l555_555336

def is_rational (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem only_fourth_is_rational :
  ¬ is_rational (Real.sqrt (Real.pi ^ 2)) ∧
  ¬ is_rational (Real.cbrt 0.8) ∧
  is_rational (Real.root 4 0.00016) ∧
  is_rational (Real.cbrt (-1) * Real.sqrt ((0.09)⁻¹)) :=
by
  sorry

end only_fourth_is_rational_l555_555336


namespace points_on_circle_l555_555517

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  x^2 + y^2 = 1 := 
by 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  sorry

end points_on_circle_l555_555517


namespace valid_distributions_l555_555302

def adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∧ i / n = j / n) ∨ (i = j - 1 ∧ i / n = j / n) ∨
  (i = j + n ∧ i % n = j % n) ∨ (i = j - n ∧ i % n = j % n)

noncomputable def num_valid_distributions (n : ℕ) : ℕ :=
if n = 1 then 1 else
if n = 2 then 1 else
if n % 2 = 0 then 3 else 8

theorem valid_distributions (n : ℕ) :
  ∃ d, num_valid_distributions n = d ∧
  (∀ i j, adjacent i j n → adjacent (d i) (d j) n ∧
  (0 ∈ d ∨ (n-1) ∈ d ∨ (n*(n-1)) ∈ d ∨ (n*n-1) ∈ d)) :=
sorry

end valid_distributions_l555_555302


namespace sum_of_five_consecutive_squares_not_perfect_square_l555_555261

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ (k : ℤ), k^2 = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 := 
by
  sorry

end sum_of_five_consecutive_squares_not_perfect_square_l555_555261


namespace regular_polygon_perimeter_l555_555445

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l555_555445


namespace relationship_abc_l555_555583

-- Definitions of the variables
variables (a b c : ℝ)

-- Given conditions
def a_def : Prop := a = 1.01^(0.5)
def b_def : Prop := b = 1.01^(0.6)
def c_def : Prop := c = 0.6^(0.5)

-- The theorem statement we need to prove
theorem relationship_abc : a_def → b_def → c_def → b > a ∧ a > c :=
by
  intros
  sorry

end relationship_abc_l555_555583


namespace original_price_l555_555805

theorem original_price (P : ℝ) (h1 : ∃ P : ℝ, (120 : ℝ) = P + 0.2 * P) : P = 100 :=
by
  obtain ⟨P, h⟩ := h1
  sorry

end original_price_l555_555805


namespace min_k_exists_sequences_l555_555097

theorem min_k_exists_sequences : 
  ∃ (a b : Fin 1997 → ℕ), 
    (∀ i, a i ∈ {1996^n | n : ℕ} ∧ b i ∈ {1996^n | n : ℕ}) ∧
    (∀ i, a i ≠ b i) ∧
    (∀ i < 1996, a i ≤ a (i + 1) ∧ b i ≤ b (i + 1)) ∧
    (∑ i, a i = ∑ i, b i) :=
sorry

end min_k_exists_sequences_l555_555097


namespace number_of_integer_values_satisfying_abs_inequality_l555_555167

-- Define the condition for the proof: the absolute value inequality
def abs_inequality (x : Int) : Prop :=
  abs x < 4

-- Define the set of integers satisfying the condition
def satisfying_set : Set Int :=
  {x : Int | abs_inequality x}

-- The theorem: proving the number of elements in the set
theorem number_of_integer_values_satisfying_abs_inequality :
  set.finite satisfying_set ∧ (set.toFinset satisfying_set).card = 7 :=
by
  sorry

end number_of_integer_values_satisfying_abs_inequality_l555_555167


namespace tan_theta_eq_four_thirds_l555_555649

variable (k θ : ℝ)
variable (h₁ : k > 0)
variable (D : Matrix (Fin 2) (Fin 2) ℝ := ![[k, 0], [0, k]])
variable (R : Matrix (Fin 2) (Fin 2) ℝ := ![[Real.cos θ, -Real.sin θ], [Real.sin θ, Real.cos θ]])
variable (h₂ : R ⬝ D = ![[15, -20], [20, 15]])

theorem tan_theta_eq_four_thirds (k > 0) (θ : ℝ) (h₁ : D = ![[k, 0], [0, k]]) (h₂ : R = ![[Real.cos θ, -Real.sin θ], [Real.sin θ, Real.cos θ]]) (h₃ : R ⬝ D = ![[15, -20], [20, 15]]): 
  Real.tan θ = 4 / 3 :=
sorry

end tan_theta_eq_four_thirds_l555_555649


namespace find_f_2_l555_555540

variable (f : ℝ → ℝ)

-- Defining the conditions
axiom increasing_function : ∀ x y : ℝ, x < y → f(x) < f(y)
axiom function_property : ∀ x : ℝ, f(f(x) - 3^x) = 4

-- The goal
theorem find_f_2 : f(2) = 10 := by
  sorry

end find_f_2_l555_555540


namespace tangent_addition_l555_555174

theorem tangent_addition (y : ℝ) (h : Real.tan y = -1) : Real.tan (y + Real.pi / 3) = -1 :=
sorry

end tangent_addition_l555_555174


namespace part1_part2_l555_555921

open Real

-- Condition definitions
noncomputable def q := 1 / 3
noncomputable def a (n : ℕ) : ℝ := q^n
noncomputable def S (n : ℕ) : ℝ := n * (a (n + 1) - a 1) / 2
noncomputable def b (n : ℕ) : ℝ := 1 / log (1 / 3) (a n)
noncomputable def c (n : ℕ) : ℝ := b n * (b (n + 1) - b (n + 2))

-- Proving parts
theorem part1 (n : ℕ) : a n = (1 / 3) ^ n :=
by
-- The proof goes here
sorry

theorem part2 (n : ℕ) :
  (finset.range n).sum c = 1 / 4 - 1 / (2 * (n + 1) * (n + 2)) :=
by
-- The proof goes here
sorry

end part1_part2_l555_555921


namespace computation_of_expression_l555_555597

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l555_555597


namespace h_is_even_l555_555657

variable {α : Type*} [LinearOrderedCommRing α]
variable (g h : α → α)

def is_odd_function (f : α → α) : Prop :=
∀ x, f (-x) = -f x

def is_even_function (f : α → α) : Prop :=
∀ x, f (-x) = f x

def h_definition (g : α → α) (x : α) : α :=
| g (x^5) |

theorem h_is_even (hg : is_odd_function g) : is_even_function (h_definition g) :=
sorry

end h_is_even_l555_555657


namespace seating_arrangement_l555_555866

theorem seating_arrangement (n x : ℕ) (h1 : 7 * x + 6 * (n - x) = 53) : x = 5 :=
sorry

end seating_arrangement_l555_555866


namespace derivative_2013_equals_cos_l555_555714

noncomputable def f : ℝ → ℝ := λ x, Real.sin x

theorem derivative_2013_equals_cos (x : ℝ) : 
  iterated_deriv 2013 f x = Real.cos x :=
sorry

end derivative_2013_equals_cos_l555_555714


namespace regular_polygon_perimeter_is_28_l555_555398

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555398


namespace revenue_decrease_by_1_point_1_percent_l555_555342

theorem revenue_decrease_by_1_point_1_percent (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let T_new := T * 0.86
  let C_new := C * 1.15
  let R_initial := (T / 100) * C
  let R_new := (T_new / 100) * C_new
  (R_new - R_initial) / R_initial * 100 = -1.1 :=
by
  let T_new := T * 0.86
  let C_new := C * 1.15
  let R_initial := (T / 100) * C
  let R_new := (T_new / 100) * C_new
  have h : (R_new - R_initial) / R_initial * 100 = (T * 0.86 * C * 1.15 - T * C) / (T * C) * 100 :=
      by sorry
  rw [h]
  have h2 : (T * 0.86 * C * 1.15 - T * C) / (T * C) * 100 = (0.86 * 1.15 - 1) * 100 :=
      by sorry
  rw [h2]
  norm_num

end revenue_decrease_by_1_point_1_percent_l555_555342


namespace balls_in_boxes_l555_555576

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l555_555576


namespace regular_polygon_perimeter_l555_555389

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l555_555389


namespace peanut_butter_candy_count_l555_555305

theorem peanut_butter_candy_count (B G P : ℕ) 
  (hB : B = 43)
  (hG : G = B + 5)
  (hP : P = 4 * G) :
  P = 192 := by
  sorry

end peanut_butter_candy_count_l555_555305


namespace problem_1_problem_2_l555_555227

-- Sub-question 1: Prove a_0 + a_2 + a_4 + a_6 = 128 for f(x) = (1+x)^7 + (1+x)^7
theorem problem_1 (x : ℝ) : 
  let f (x : ℝ) := (1 + x)^7 + (1 + x)^7 in 
  (∃ a_7 a_6 a_5 a_4 a_3 a_2 a_1 a_0 : ℝ, 
    f x = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0 ∧ 
    a_0 + a_2 + a_4 + a_6 = 128) :=
sorry

-- Sub-question 2: Minimum value of coefficient of x^2 in f(x) = (1+x)^m + (1+x)^n given m+n=19
theorem problem_2 (m n : ℕ) (h : m + n = 19) : 
  let f (x : ℝ) := (1 + x)^m + (1 + x)^n in 
  ∃ a_2 : ℝ, (∃ a_7 a_6 a_5 a_4 a_3 a_1 a_0 : ℝ, 
    f x = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) ∧ 
    a_2 = (1/2) * (m * (m-1)) + (1/2) * (n * (n-1)) ∧ 
    a_2 = 81 :=
sorry

end problem_1_problem_2_l555_555227


namespace regular_polygon_perimeter_is_28_l555_555403

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555403


namespace integral_inequality_for_continuous_function_l555_555695

theorem integral_inequality_for_continuous_function 
  (f : ℝ → ℝ) 
  (hf : continuous_on f (Icc 0 1)) : 
  (∫ x in 0..1, ∫ y in 0..1, |f x + f y|) ≥ (∫ x in 0..1, |f x|) :=
by 
  sorry

end integral_inequality_for_continuous_function_l555_555695


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555314

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l555_555314


namespace tangent_circle_exists_l555_555323

theorem tangent_circle_exists 
  (S : Circle)  -- The given circle
  (l : Line)  -- The given line
  (A : Point)  -- The given point on the line
  (h_A_on_l : A ∈ l)  -- Point A is on line l
  : 
  ∃ (C : Circle), tangent C S ∧ tangent C l ∧ (A ∈ C) := 
sorry

end tangent_circle_exists_l555_555323


namespace min_trials_to_ensure_pass_l555_555027

theorem min_trials_to_ensure_pass (p : ℝ) (n : ℕ) (h₁ : p = 3 / 4) (h₂ : n ≥ 1): 
  (1 - (1 - p) ^ n) > 0.99 → n ≥ 4 :=
by sorry

end min_trials_to_ensure_pass_l555_555027


namespace auspicious_numbers_count_l555_555355

open Finset

theorem auspicious_numbers_count :
  ∑ s in ({0, 1, 2, 3, 4, 5}.powerset.filter (λ s, s.sum id = 8 ∧ s.card = 4)),
    (filter (λ n, n > 2015) (permutations_of_set s)).card = 23 :=
by
  sorry

end auspicious_numbers_count_l555_555355


namespace linear_function_passing_points_l555_555939

theorem linear_function_passing_points :
  ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b) ∧ (k * 0 + b = 3) ∧ (k * (-4) + b = 0)
  →
  (∃ a : ℝ, y = -((3:ℝ) / (4:ℝ)) * x + 3 ∧ (∀ x y : ℝ, y = -((3:ℝ) / (4:ℝ)) * a + 3 → y = 6 → a = -4)) :=
by sorry

end linear_function_passing_points_l555_555939


namespace profit_percentage_l555_555018

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 725) : 
  100 * (SP - CP) / CP = 45 :=
by
  sorry

end profit_percentage_l555_555018


namespace function_properties_l555_555903

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x * Real.exp x else x ^ 2 - 2 * x + 0.5

theorem function_properties :
  (∃ x, f(x) = 0.5) ∧  -- Conditions to check if the function meets the specified criteria
  (f'(-2) = -2/Real.exp 2) ∧  -- Check slope at x=-2
  (∀ x, f(x) ≥ -2/Real.exp 1) ∧  -- Verify minimum value
  (∀ x ∈ Icc (-∞, -1), f(x) ≤ f(-1)) ∧  -- Decreasing behavior over (-∞, -1]
  (∀ x ∈ Icc (0, 1), f(x) ≤ f(1))  -- Decreasing behavior over (0, 1]
  → 
  -- Assertion that the correct statements are ①, ②, and ④.
  true := sorry

end function_properties_l555_555903


namespace more_interesting_even_numbers_than_odd_l555_555765

-- Define the range of natural numbers we are interested in
def range := { n : ℕ | n ≥ 1 ∧ n ≤ 1000000 }

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for a number to be interesting based on the problem conditions
def is_interesting_even (n : ℕ) : Prop :=
  n % 2 = 0 ∧ sum_of_digits n % 2 = 1

def is_interesting_odd (n : ℕ) : Prop :=
  n % 2 = 1 ∧ sum_of_digits n % 2 = 0

-- Define the sets of interesting even and odd numbers
def interesting_even_numbers := { n ∈ range | is_interesting_even n }
def interesting_odd_numbers := { n ∈ range | is_interesting_odd n }

-- The statement to prove
theorem more_interesting_even_numbers_than_odd :
  interesting_even_numbers.card > interesting_odd_numbers.card :=
sorry

end more_interesting_even_numbers_than_odd_l555_555765


namespace rook_connected_partition_l555_555449

-- Define a rook-connected set and the necessary properties
structure RookConnectedSet (α : Type) [DecidableEq α] :=
  (cells : Set α)
  (is_rook_connected : ∀ {x y : α}, x ∈ cells → y ∈ cells → ∃ path : List α, path.head = x ∧ path.last = y 
    ∧ ∀ p ∈ path, p ∈ cells 
    ∧ ∀ i < path.length - 1, (path.get i.1 = path.get (i.1 + 1).1))

-- Define the problem statement that needs to be proven
theorem rook_connected_partition (S : RookConnectedSet (ℕ × ℕ)) (hS : S.cells.card = 100) : 
  ∃ (pairs : Set (Set (ℕ × ℕ))), (∀ p ∈ pairs, p.card = 2) ∧ (∀ p ∈ pairs, ∃ (i : ℕ), ∀ x y ∈ p, x.1 = y.1 ∨ x.2 = y.2) ∧ (∀ p₁ p₂ ∈ pairs, p₁ ≠ p₂ → p₁ ∩ p₂ = ∅) :=
by
sory

end rook_connected_partition_l555_555449


namespace algebraic_expression_value_l555_555586

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l555_555586


namespace distance_between_foci_l555_555892

def ellipse_foci_distance : ℝ :=
  let F1 := (4 : ℝ, 5 : ℝ)
  let F2 := (-6 : ℝ, 9 : ℝ)
  (Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2))

theorem distance_between_foci :
  ∀ (x y : ℝ), (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24) →
     ellipse_foci_distance = 2 * Real.sqrt (29) :=
by
  intros x y h
  -- Proof goes here
  sorry

end distance_between_foci_l555_555892


namespace polygon_perimeter_l555_555383

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555383


namespace basketball_points_distinct_numbers_l555_555792

/-- A basketball player made 7 baskets during a game. Each basket was worth either 2 or 3 points.
Prove that the number of different numbers that could represent the total points scored by the
player is 8. -/
theorem basketball_points_distinct_numbers : 
  ∃ S : Finset ℕ, (∀ P ∈ S, ∃ x : ℕ, x ≤ 7 ∧ P = 3 * x + 2 * (7 - x)) ∧ S.card = 8 :=
by 
  sorry

end basketball_points_distinct_numbers_l555_555792


namespace second_offset_length_l555_555081

theorem second_offset_length (d h1 area : ℝ) (h_diagonal : d = 28) (h_offset1 : h1 = 8) (h_area : area = 140) :
  ∃ x : ℝ, area = (1/2) * d * (h1 + x) ∧ x = 2 :=
by
  sorry

end second_offset_length_l555_555081


namespace isosceles_triangle_count_l555_555754

theorem isosceles_triangle_count (n : ℕ) (h_n : n = 14) : 
  ∃ (triangles : ℕ), 
  (triangles = 3) ∧ 
  (∀ (a b c : ℕ), 
    (a + b + c = n) ∧ 
    ((a = b ∨ b = c ∨ c = a) ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (a - b).natAbs < c ∧ 
    (b - c).natAbs < a ∧ 
    (c - a).natAbs < b)) ↔ 
  ((a = 4 ∧ b = 4 ∧ c = 6) ∨ 
   (a = 5 ∧ b = 5 ∧ c = 4) ∨ 
   (a = 6 ∧ b = 6 ∧ c = 2)) sorry

end isosceles_triangle_count_l555_555754


namespace algebraic_expression_identity_l555_555593

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l555_555593


namespace largest_constant_m_l555_555086

theorem largest_constant_m (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) +
   sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d))) > 2 := 
sorry

end largest_constant_m_l555_555086


namespace max_green_beads_l555_555364

/-- Define the problem conditions:
  1. A necklace consists of 100 beads of red, blue, and green colors.
  2. Among any five consecutive beads, there is at least one blue bead.
  3. Among any seven consecutive beads, there is at least one red bead.
  4. The beads in the necklace are arranged cyclically (the last one is adjacent to the first one).
--/
def necklace_conditions := 
  ∀ (beads : ℕ → ℕ) (n : ℕ), 
    (∀ i, 0 ≤ beads i ∧ beads i < 3) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 5 ∧ beads j = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 7 ∧ beads j = 0)

/-- Prove the maximum number of green beads that can be in this necklace is 65. --/
theorem max_green_beads : ∃ beads : (ℕ → ℕ), necklace_conditions beads → (∑ p, if beads p = 2 then 1 else 0) = 65 :=
by sorry

end max_green_beads_l555_555364


namespace sequence_property_l555_555920

noncomputable def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 else if n = 1 then 1 else 2^n - 1

def S_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∑ i in finset.range n, a_n (i + 1)

theorem sequence_property (n : ℕ) (hn : n ≥ 1) : 
  S_n n = 2 * a_n n - n := by sorry

end sequence_property_l555_555920


namespace find_a_l555_555872

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l555_555872


namespace solution_set_eq_inequality_proof_l555_555953

def f (x : ℝ) : ℝ :=
  abs (x + 1) + abs (x - 1)

theorem solution_set_eq :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -3 ≤ x ∧ x ≤ 3} :=
sorry

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a^2 ∧ a^2 ≤ 3) (hb : 0 ≤ b^2 ∧ b^2 ≤ 3) :
  sqrt 3 * abs (a + b) ≤ abs (a * b + 3) :=
sorry

end solution_set_eq_inequality_proof_l555_555953


namespace card_probability_l555_555267

def binom (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Finset.card (Finset.powersetLen k (Finset.range (n + 1))) else 0

def q (a : ℕ) : ℚ := (binom (40 - a) 2 + binom (a - 1) 2) / 1225

theorem card_probability:
  ∃ (a : ℕ), q(a) = 91 / 175 ∧ q(a) ≥ 1 / 2 := 
by
  sorry

end card_probability_l555_555267


namespace average_not_necessarily_closer_l555_555340

def average (s : List ℝ) : ℝ := s.sum / s.length

theorem average_not_necessarily_closer (A B : List ℝ) : 
  (∃ avgA avgB : ℝ, avgA = average A ∧ avgB = average B) ∧ 
  (length A ≠ length B) → ¬ (average (A ++ B) ≠ ((average A + average B) / 2)) :=
by 
  intros,
  sorry

end average_not_necessarily_closer_l555_555340


namespace vector_problem_l555_555963

theorem vector_problem
  (OA OB OM : ℝ × ℝ) (PA PB OP : ℝ × ℝ)
  (t : ℝ) 
  (hOA : OA = (-1, -3))
  (hOB : OB = (5, 3))
  (hOM : OM = (2, 2))
  (hP_on_OM : ∃ λ : ℝ, OP = (2 * λ, 2 * λ))
  (hPA_PB_dot : PA • PB = -16)
  (hOP_coords : OP = (1, 1))
  (hPA_def : PA = (-1, -3) - OP)
  (hPB_def : PB = (5, 3) - OP) :
  OP = (1, 1) ∧
  ∀ (PA PB : ℝ × ℝ), PA = (-2, -4) ∧ PB = (4, 2) → real.cos (vector.angle PA PB) = -4 / 5 ∧
  (∃ (t : ℝ), real.sqrt ((-1 + t)^2 + (-3 + t)^2) = real.sqrt 2) :=
by
  sorry

end vector_problem_l555_555963


namespace not_right_triangle_set_C_l555_555455

theorem not_right_triangle_set_C :
  ¬((3^2 + 5^2 = 7^2) ∧
    ((3^2 + 4^2 = 5^2) ∨
    (5^2 + 12^2 = 13^2) ∨
    (1^2 + (real.sqrt 3)^2 = 2^2))) :=
by {
  -- Prove by contradiction
  intro H,
  cases H with H1 H2,
  -- Evaluate the calculations
  have H1_eval : 3^2 + 5^2 ≠ 7^2 := by norm_num,
  exact H1_eval H1,
}

end not_right_triangle_set_C_l555_555455


namespace find_variance_l555_555717

variable {α : Type*} (ξ : α → ℝ) (P : α → ℝ)

-- Hypotheses
def is_arithmetic_seq (a b c : ℝ) : Prop :=
  2 * b = a + c

def prob_sum_to_one (a b c : ℝ) : Prop :=
  a + b + c = 1

def expected_value (a b c : ℝ) : Prop :=
  c - a = 1 / 3

-- Main statement to prove
theorem find_variance (a b c : ℝ)
    (h₁ : is_arithmetic_seq a b c)
    (h₂ : prob_sum_to_one a b c)
    (h₃ : expected_value a b c) :
    let Eξ := -a + c in
    let Dξ := ((-1 - Eξ) ^ 2 * a) + ((0 - Eξ) ^ 2 * b) + ((1 - Eξ) ^ 2 * c) in
    Dξ = 5 / 9 :=
by
  sorry

end find_variance_l555_555717


namespace magnitude_A7_A8_coordinates_OA_coordinates_OB_l555_555933

open Classical

noncomputable def i : EuclideanSpace ℝ (Fin 2) := ![1, 0]
noncomputable def j : EuclideanSpace ℝ (Fin 2) := ![0, 1]

def OA (n : ℕ+) : EuclideanSpace ℝ (Fin 2) :=
  if n = 1 then j else if n = 2 then 5 • j else sorry -- To be filled in for full definition

def OB : EuclideanSpace ℝ (Fin 2) := 3 • i + 3 • j

def A (n : ℕ+) (m : ℕ+) :=
  if (m = n + 1) then 2 • OA n - OA m else sorry -- To be filled in for full definition

def B (n : ℕ+) (m : ℕ+) :=
  if (m = n + 1) then 2 • i + 2 • j else sorry -- To be filled in for full definition

theorem magnitude_A7_A8 :
  ‖A 7 8‖ = 1 / 16 := sorry

theorem coordinates_OA (n : ℕ+) :
  OA n = (if n = 1 then j else if n ≥ 2 then ![0, 9 - 2^(4 - n)] else sorry) := sorry

theorem coordinates_OB (n : ℕ+) :
  OB = ![2*n+1, 2*n+1] := sorry

end magnitude_A7_A8_coordinates_OA_coordinates_OB_l555_555933


namespace police_coverage_l555_555195

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define each street as a set of intersections
def horizontal_streets : List (List Intersection) :=
  [[A, B, C, D], [E, F, G], [H, I, J, K]]

def vertical_streets : List (List Intersection) :=
  [[A, E, H], [B, F, I], [D, G, J]]

def diagonal_streets : List (List Intersection) :=
  [[H, F, C], [C, G, K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the set of police officers' placements
def police_officers : List Intersection := [B, G, H]

-- Check if each street is covered by at least one police officer
def is_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i, i ∈ street ∧ i ∈ officers

-- Define the proof problem statement
theorem police_coverage :
  ∀ street ∈ all_streets, is_covered street police_officers :=
by sorry

end police_coverage_l555_555195


namespace angle_value_is_140_l555_555192

-- Definitions of conditions
def angle_on_straight_line_degrees (x y : ℝ) : Prop := x + y = 180

-- Main statement in Lean
theorem angle_value_is_140 (x : ℝ) (h₁ : angle_on_straight_line_degrees 40 x) : x = 140 :=
by
  -- Proof is omitted (not required as per instructions)
  sorry

end angle_value_is_140_l555_555192


namespace cos_B_half_l555_555619

theorem cos_B_half {a b c A B C : ℝ} (h1 : ∠A a = ∠B b) (h2 : ∠B b = ∠C c) 
  (h : b / a = (sqrt 3 * cos B) / sin A) : cos B = 1 / 2 :=
by
  sorry

end cos_B_half_l555_555619


namespace average_speed_is_correct_l555_555456
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end average_speed_is_correct_l555_555456


namespace mark_bought_food_for_three_weeks_l555_555676

theorem mark_bought_food_for_three_weeks
  (cost_puppy : ℕ)
  (daily_food_intake : ℚ)
  (bag_cost : ℕ)
  (bag_capacity : ℚ)
  (total_cost : ℕ)
  (num_days_in_week : ℕ)
  (spent_amount_on_food : ℕ)
  (num_bags : ℕ)
  (total_food_cups : ℚ)
  (total_days : ℕ)
(by 
  -- Given conditions
  (h1 : cost_puppy = 10)
  (h2 : daily_food_intake = 1/3)
  (h3 : bag_cost = 2)
  (h4 : bag_capacity = 3.5)
  (h5 : total_cost = 14)
  (h6 : num_days_in_week = 7)
  -- Intermediate calculations
  (h7 : spent_amount_on_food = total_cost - cost_puppy)
  (h8 : num_bags = spent_amount_on_food / bag_cost)
  (h9 : total_food_cups = num_bags * bag_capacity)
  (h10 : total_days = total_food_cups / daily_food_intake)
):
  -- Desired result
  total_days / num_days_in_week = 3 := 
sorry

end mark_bought_food_for_three_weeks_l555_555676


namespace degree_polynomial_rational_l555_555702

-- Definitions based on given conditions
def a := 3 - Real.sqrt 2
def b := 5 + Real.sqrt 3
def c := 8 - 2 * Real.sqrt 7
def d := -2 * Real.sqrt 2

theorem degree_polynomial_rational (p : ℝ[X]) (h : p ≠ 0) :
  (p.eval a  = 0) ∧ (p.eval b  = 0) ∧ (p.eval c  = 0) ∧ (p.eval d  = 0)
  ∧ (∀ x : ℝ, (p.eval x = 0) → (∃ q : ℚ[X], x = q)) → p.degree = 8 :=
by
  sorry

end degree_polynomial_rational_l555_555702


namespace find_AB_l555_555620

namespace TriangleProblem

-- Definitions of the conditions
variables {A B C : Type}
variables [has_cos B] [has_cos A]

/-- Right-angled triangle as given in the problem -/
def is_right_angled_triangle (A B C : Type) [has_angle_right A] :=
  ∠A = 90

-- Cosine of angle B
def cos_B (A B C : Type) [has_cos B] := cos(B) = 4/5

-- Side AC
def side_AC (A B C : Type) := (distance A C : ℝ) = 40

-- The final proof problem statement
theorem find_AB (A B C : Type) [has_angle_right A] [has_cos B] [has_distance A C] :
  is_right_angled_triangle A B C →
  cos_B A B C →
  side_AC A B C →
  (distance A B : ℝ) = 32 :=
by
  sorry

end TriangleProblem

end find_AB_l555_555620


namespace simplify_expression_l555_555262

theorem simplify_expression : 
  (20 * (9 / 14) * (1 / 18) : ℚ) = (5 / 7) := 
by 
  sorry

end simplify_expression_l555_555262


namespace range_of_m_l555_555122

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end range_of_m_l555_555122


namespace find_f2_l555_555946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem find_f2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) : f a 2 = 7 := 
by 
  sorry

end find_f2_l555_555946


namespace cylindrical_coordinates_conversion_l555_555485

def point_rect := (3, -3 * Real.sqrt 3, 2)

-- Defining cylindrical coordinates conversion
def to_cylindrical (x y z : ℝ) : (r : ℝ) × (θ : ℝ) × (z' : ℝ) :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan2 y x
  (r, θ, z)

-- Theorem statement
theorem cylindrical_coordinates_conversion :
  let (r, θ, z') := to_cylindrical 3 (-3 * Real.sqrt 3) 2
  r = 6 ∧ θ = (5 * Real.pi) / 3 ∧ z' = 2 := 
  sorry

end cylindrical_coordinates_conversion_l555_555485


namespace closest_to_sqrt_101_minus_sqrt_99_is_0_2_l555_555334

theorem closest_to_sqrt_101_minus_sqrt_99_is_0_2 :
  (0.18 < sqrt 101 - sqrt 99) ∧ (sqrt 101 - sqrt 99 < 0.21) → sqrt 101 - sqrt 99 ≈ 0.2 :=
by 
  sorry

end closest_to_sqrt_101_minus_sqrt_99_is_0_2_l555_555334


namespace regular_polygon_perimeter_l555_555426

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555426


namespace third_day_water_collected_l555_555821

theorem third_day_water_collected:
  ∀ (total_capacity : ℕ) (initial_fraction : ℚ) 
  (first_day_collected : ℕ) (second_day_extra : ℕ),
  total_capacity = 100 →
  initial_fraction = 2/5 →
  first_day_collected = 15 →
  second_day_extra = 5 →
  let initial_amount := (initial_fraction * total_capacity).toNat in
  let after_first_day := initial_amount + first_day_collected in
  let second_day_collected := first_day_collected + second_day_extra in
  let after_second_day := after_first_day + second_day_collected in
  total_capacity - after_second_day = 25 :=
begin
  sorry
end

end third_day_water_collected_l555_555821


namespace regular_polygon_perimeter_is_28_l555_555402

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555402


namespace polygon_perimeter_l555_555381

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l555_555381


namespace problem_1_problem_2_problem_3_problem_4_l555_555207

variable (A B C a b c : ℝ)
variable (triangle_ABC : A + B + C = Real.pi)
variable (acute_triangle : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)

-- Problem 1
theorem problem_1 (h : a / Real.cos A = b / Real.sin B) : A = Real.pi / 4 := sorry

-- Problem 2
theorem problem_2 (h : Real.sin (2 * A) = Real.sin (2 * B)) : ¬ ∀ (A B : ℝ), A = B := sorry

-- Problem 3
theorem problem_3 (ha : a = 1) (hb : b = 2) (hA : A = Real.pi / 6) : ¬ (∃ C, True) := sorry

-- Problem 4
theorem problem_4 (h_acute : acute_triangle) : Real.sin A + Real.sin B > Real.cos A + Real.cos B := sorry

end problem_1_problem_2_problem_3_problem_4_l555_555207


namespace f_log2_3_eq_1_24_l555_555997

noncomputable def f : ℝ → ℝ
| x :=
  if x ≥ 4 then (1 / 2)^x
  else f (x + 1)

theorem f_log2_3_eq_1_24 : f (Real.log2 3) = 1 / 24 :=
  sorry

end f_log2_3_eq_1_24_l555_555997


namespace arrange_abc_l555_555978

def a : ℝ := (-2/3)^(-2)
def b : ℝ := (-1)^(-1)
def c : ℝ := (-π/2)^0

theorem arrange_abc : b < c ∧ c < a := by
  rw [b, c, a]
  -- Placeholder for the proof
  sorry

end arrange_abc_l555_555978


namespace largest_constant_m_l555_555087

theorem largest_constant_m (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + sqrt (c / (a + b + d + e)) +
   sqrt (d / (a + b + c + e)) + sqrt (e / (a + b + c + d))) > 2 := 
sorry

end largest_constant_m_l555_555087


namespace option_d_l555_555472

variable {R : Type*} [LinearOrderedField R]

theorem option_d (a b c d : R) (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by 
  sorry

end option_d_l555_555472


namespace basketball_subs_mod_100_l555_555796

-- Definitions based on the conditions
def players := 15
def starters := 5
def substitutes := 10
def max_subs := 4

-- Proof problem translated and stated in Lean 4
theorem basketball_subs_mod_100 :
  let ways_0 := 1
  let ways_1 := 5 * 10
  let ways_2 := ways_1 * (4 * 9)
  let ways_3 := ways_2 * (3 * 8)
  let ways_4 := ways_3 * (2 * 7)
  let total_ways := ways_0 + ways_1 + ways_2 + ways_3 + ways_4
  total_ways % 100 = 51 := by 
  -- Big steps performed in the initial solution are directly reduced to mathematical chunk here
  let ways_0 := 1
  let ways_1 := 50
  let ways_2 := 1800
  let ways_3 := 43200
  let ways_4 := 604800
  let total_ways := ways_0 + ways_1 + ways_2 + ways_3 + ways_4
  calc total_ways % 100
       = 648851 % 100 := by sorry
       = 51 := by sorry

end basketball_subs_mod_100_l555_555796


namespace club_last_names_l555_555746

theorem club_last_names :
  ∃ A B C D E F : ℕ,
    A + B + C + D + E + F = 21 ∧
    A^2 + B^2 + C^2 + D^2 + E^2 + F^2 = 91 :=
by {
  sorry
}

end club_last_names_l555_555746


namespace right_angled_isosceles_triangle_third_side_length_l555_555286

theorem right_angled_isosceles_triangle_third_side_length (a b c : ℝ) (h₀ : a = 50) (h₁ : b = 50) (h₂ : a + b + c = 160) : c = 60 :=
by
  -- TODO: Provide proof
  sorry

end right_angled_isosceles_triangle_third_side_length_l555_555286


namespace integral_circle_quarter_area_l555_555498

theorem integral_circle_quarter_area :
  ∫ x in 0..2, sqrt (4 - (x - 2)^2) = π :=
by
  sorry

end integral_circle_quarter_area_l555_555498


namespace find_side_length_of_triangle_l555_555734

theorem find_side_length_of_triangle
  (a b c : ℝ)
  (cosA : ℝ)
  (h₁ : a = real.sqrt 5)
  (h₂ : c = 2)
  (h₃ : cosA = 2 / 3) :
  b = 3 :=
by
  sorry

end find_side_length_of_triangle_l555_555734


namespace number_of_valid_sets_l555_555068

def digits : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def is_arithmetic (a b c : ℤ) : Prop := b - a = c - b

def sum_is_zero (a b c : ℤ) : Prop := a + b + c = 0

theorem number_of_valid_sets : 
  (Finset.card ((Finset.filter (λ s : Finset ℤ, 
    (∃ (a b c : ℤ), 
        (a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ 
         is_arithmetic a b c ∧ sum_is_zero a b c) ∧
         (a ≠ b ∧ b ≠ c ∧ a ≠ c))
  ) (Finset.powersetLen 3 (digits.toFinset)))) : ℕ) = 4 := 
sorry

end number_of_valid_sets_l555_555068


namespace sqrt_product_simplification_l555_555051

variable (p : ℝ)

theorem sqrt_product_simplification (hp : 0 ≤ p) :
  (Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p)) = 42 * p * Real.sqrt (7 * p) :=
sorry

end sqrt_product_simplification_l555_555051


namespace regular_polygon_perimeter_l555_555427

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l555_555427


namespace probability_of_B_given_A_l555_555550

-- Conditions
def total_bottles : ℕ := 6
def qualified_bottles : ℕ := 4
def unqualified_bottles : ℕ := 2
def draw_without_replacement : Prop := true

-- Event A: Drawing an unqualified disinfectant the first time
def event_A : Prop := true
-- Event B: Drawing an unqualified disinfectant the second time
def event_B : Prop := true

-- The probability of drawing an unqualified disinfectant the second time given that an unqualified disinfectant was drawn the first time
def P_B_given_A : ℚ := 1 / 5

theorem probability_of_B_given_A (htotal : total_bottles = 6)
                                 (hqual : qualified_bottles = 4)
                                 (hunqual : unqualified_bottles = 2)
                                 (hdraw : draw_without_replacement) 
                                 (hA : event_A) 
                                 (hB : event_B) : 
  P_B_given_A = 1 / 5 :=
sorry

end probability_of_B_given_A_l555_555550


namespace perimeter_of_regular_polygon_l555_555418

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l555_555418


namespace buckets_needed_for_milk_l555_555014

-- defining the problem and given conditions
def milk_per_bucket : ℕ := 15
def total_milk : ℕ := 147
def approximate_buckets : ℕ := 10

-- the statement we want to prove
theorem buckets_needed_for_milk :
  ⌊(total_milk : ℚ) / (milk_per_bucket : ℚ)⌋ ≈ approximate_buckets := 
sorry

end buckets_needed_for_milk_l555_555014


namespace find_a2019_l555_555154

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then x + 1/2
  else if x < 1 then 2 * x - 1
  else x - 1

def a₁ : ℝ := 7 / 3

def a : ℕ → ℝ
| 0     := a₁
| (n+1) := f (a n)

theorem find_a2019 : a 2019 = 1 / 3 :=
by
  sorry

end find_a2019_l555_555154


namespace complex_in_second_quadrant_l555_555935

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant (z : ℂ) (h : (1 + complex.i) * z = -1) : 
  is_second_quadrant z :=
sorry

end complex_in_second_quadrant_l555_555935


namespace iPhones_sold_l555_555836

theorem iPhones_sold (x : ℕ) (h1 : (1000 * x + 18000 + 16000) / (x + 100) = 670) : x = 100 :=
by
  sorry

end iPhones_sold_l555_555836


namespace contrapositive_l555_555709

-- Define odd and even
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The original proposition
def original_proposition (a b : ℕ) : Prop :=
  is_odd a ∧ is_odd b → is_odd (a * b)

-- The contrapositive proposition to prove
theorem contrapositive (a b : ℕ) (h : is_odd a ∧ is_odd b) : original_proposition a b :=
begin
  sorry
end

end contrapositive_l555_555709


namespace union_sets_l555_555242

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {-1, 0, 1}

theorem union_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  -- Proof goes here
  sorry

end union_sets_l555_555242


namespace minimize_variance_correct_l555_555141

noncomputable def minimize_variance (a b : ℝ) : Prop :=
  let values := [2, 3, 3, 7, a, b, 12, 13.7, 18.3, 20]
  let median := (values.nth_le 4 sorry + values.nth_le 5 sorry) / 2
  median = 10.5 ∧
  ∀ u v : ℝ, 
    (median (values.set_nth 4 u).sort = 10.5) →
    (median (values.set_nth 5 v).sort = 10.5) →
    (let m := (2 + 3 + 3 + 7 + a + b + 12 + 13.7 + 18.3 + 20) / 10 in
     ∑ x in [2, 3, 3, 7, a, b, 12, 13.7, 18.3, 20], (x - m) ^ 2) ≤
    (let m := (2 + 3 + 3 + 7 + u + v + 12 + 13.7 + 18.3 + 20) / 10 in
     ∑ x in [2, 3, 3, 7, u, v, 12, 13.7, 18.3, 20], (x - m) ^ 2)

theorem minimize_variance_correct : minimize_variance 10.5 10.5 :=
by
  sorry

end minimize_variance_correct_l555_555141


namespace books_left_over_l555_555183

theorem books_left_over (n_boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) :
  n_boxes = 1575 → books_per_box = 45 → new_box_capacity = 46 →
  (n_boxes * books_per_box) % new_box_capacity = 15 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  -- Actual proof steps would go here
  sorry

end books_left_over_l555_555183


namespace water_collected_on_third_day_l555_555824

open_locale classical

variable (tank_capacity : ℕ)
variable (initial_fill_fraction : ℚ)
variable (first_day_collection : ℕ)
variable (additional_second_day_collection : ℕ)
variable (final_day_filled : tank_capacity = 100)

theorem water_collected_on_third_day :
  initial_fill_fraction = (2/5 : ℚ) →
  first_day_collection = 15 →
  additional_second_day_collection = 5 →
  tank_capacity = 100 →
  let initial_volume := (initial_fill_fraction * tank_capacity : ℚ).to_nat in
  let first_day_volume := initial_volume + first_day_collection in
  let second_day_collection := first_day_collection + additional_second_day_collection in
  let second_day_volume := first_day_volume + second_day_collection in
  let third_day_collection := tank_capacity - second_day_volume in
  third_day_collection = 25 :=
begin
  intros,
  sorry
end

end water_collected_on_third_day_l555_555824


namespace taxi_fare_80_miles_l555_555039

theorem taxi_fare_80_miles (fare_60 : ℝ) (flat_rate : ℝ) (proportional_rate : ℝ) (d : ℝ) (charge_60 : ℝ) 
  (h1 : fare_60 = 150) (h2 : flat_rate = 20) (h3 : proportional_rate * 60 = charge_60) (h4 : charge_60 = (fare_60 - flat_rate)) 
  (h5 : proportional_rate * 80 = d - flat_rate) : d = 193 := 
by
  sorry

end taxi_fare_80_miles_l555_555039


namespace problem_statement_l555_555582

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) : b < 0 ∧ |b| > |a| :=
by
  sorry

end problem_statement_l555_555582


namespace missing_bricks_is_26_l555_555971

-- Define the number of bricks per row and the number of rows
def bricks_per_row : Nat := 10
def number_of_rows : Nat := 6

-- Calculate the total number of bricks for a fully completed wall
def total_bricks_full_wall : Nat := bricks_per_row * number_of_rows

-- Assume the number of bricks currently present
def bricks_currently_present : Nat := total_bricks_full_wall - 26

-- Define a function that calculates the number of missing bricks
def number_of_missing_bricks (total_bricks : Nat) (bricks_present : Nat) : Nat :=
  total_bricks - bricks_present

-- Prove that the number of missing bricks is 26
theorem missing_bricks_is_26 : 
  number_of_missing_bricks total_bricks_full_wall bricks_currently_present = 26 :=
by
  sorry

end missing_bricks_is_26_l555_555971


namespace trapezoid_WY_length_l555_555202

theorem trapezoid_WY_length (W Z X Y O Q : Type) [LinearOrder X] [LinearOrder Y]
  [LinearOrder W] [LinearOrder Z] 
  (WZ_parallel_XY : W ≠ Z ∧ X ≠ Y ∧ ∀ {A B : Y → X}, A = B → W = Z)
  (WX_eq_YZ : WX = 24 ∧ YZ = 24)
  (WY_perpendicular_XY : WY ⟂ XY)
  (midpoint_Q : ∀ {A : X → X}, is_median Q A XY)
  (OQ_eq_9 : OQ = 9) :
  let a := 6 in let b := 65 in a + b = 71 := by
  sorry

end trapezoid_WY_length_l555_555202


namespace triangle_side_b_l555_555637

theorem triangle_side_b 
  (A B C : ℝ) (a b c : ℝ) 
  (ha : a = sqrt 6 + sqrt 2)
  (hc : c = sqrt 6 + sqrt 2)
  (hA : A = 75) :
  b = 2 :=
sorry

end triangle_side_b_l555_555637


namespace pepper_remaining_l555_555465

/-- Brennan initially had 0.25 grams of pepper. He used 0.16 grams for scrambling eggs. 
His friend added x grams of pepper to another dish. Given y grams are remaining, 
prove that y = 0.09 + x . --/
theorem pepper_remaining (x y : ℝ) (h1 : 0.25 - 0.16 = 0.09) (h2 : y = 0.09 + x) : y = 0.09 + x := 
by
  sorry

end pepper_remaining_l555_555465


namespace sum_of_fractions_l555_555843

theorem sum_of_fractions :
  (3 / 20 : ℝ) +  (7 / 200) + (8 / 2000) + (3 / 20000) = 0.1892 :=
by 
  sorry

end sum_of_fractions_l555_555843


namespace percent_second_question_correct_l555_555611

variable {Ω : Type} -- Ω represents the sample space
variable {P : Ω → Prop → ℝ}  -- A probability measure

def P_A := P Ω (λ ω, ω ∈ A) = 0.7
def P_A_and_B := P Ω (λ ω, ω ∈ A ∧ ω ∈ B) = 0.45
def P_not_A_and_not_B := P Ω (λ ω, ¬(ω ∈ A) ∧ ¬(ω ∈ B)) = 0.2

theorem percent_second_question_correct (A B : Ω → Prop) 
  (h1 : P Ω (λ ω, ω ∈ A) = 0.7) 
  (h2 : P Ω (λ ω, ω ∈ A ∧ ω ∈ B) = 0.45) 
  (h3 : P Ω (λ ω, ¬(ω ∈ A) ∧ ¬(ω ∈ B)) = 0.2) : 
  P Ω (λ ω, ω ∈ B) = 0.55 := 
sorry

end percent_second_question_correct_l555_555611


namespace suraj_avg_after_10th_inning_l555_555703

theorem suraj_avg_after_10th_inning (A : ℝ) 
  (h1 : ∀ A : ℝ, (9 * A + 200) / 10 = A + 8) :
  ∀ A : ℝ, A = 120 → (A + 8 = 128) :=
by
  sorry

end suraj_avg_after_10th_inning_l555_555703


namespace intersection_points_always_distinct_shortest_chord_length_l555_555531

theorem intersection_points_always_distinct (k : ℝ) :
  ∀ k : ℝ, ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
    (p1.1^2 + p1.2^2 - 4 * p1.1 - 6 * p1.2 - 3 = 0) ∧
    (k * p1.1 - p1.2 - 4 * k + 2 = 0) ∧
    (p2.1^2 + p2.2^2 - 4 * p2.1 - 6 * p2.2 - 3 = 0) ∧
    (k * p2.1 - p2.2 - 4 * k + 2 = 0) :=
by
  intro k
  sorry

theorem shortest_chord_length :
  ∃ l : ℝ, ∃ k : ℝ, k = 2 ∧ l = 2 * Real.sqrt(11) :=
by
  sorry

end intersection_points_always_distinct_shortest_chord_length_l555_555531


namespace S_equals_X_l555_555647

noncomputable def X_set (h : ℕ) : Set ℕ := { n | n ≥ 2 * h }

def S_set (h : ℕ) (S : Set ℕ) : Prop :=
  ∃ (a b : ℕ), (a + b ∈ S ∧ a ≥ h ∧ b ≥ h → a * b ∈ S) ∧
               (a * b ∈ S ∧ a ≥ h ∧ b ≥ h → a + b ∈ S)

theorem S_equals_X (h : ℕ) (S : Set ℕ) :
  h ≥ 3 →
  S.nonempty →
  (∀ a b, a + b ∈ S ∧ a ≥ h ∧ b ≥ h → a * b ∈ S) →
  (∀ a b, a * b ∈ S ∧ a ≥ h ∧ b ≥ h → a + b ∈ S) →
  S = X_set h :=
begin
  sorry
end

end S_equals_X_l555_555647


namespace min_distance_MN_l555_555190

-- Definitions
def C1 (x y : ℝ) : Prop := (x ^ 2 / 16) + (y ^ 2 / 8) = 1
def C2 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 2

-- Theorem statement
theorem min_distance_MN : 
  ∀ M N : ℝ × ℝ, 
  C1 M.fst M.snd → C2 N.fst N.snd → 
  ∃ α θ : ℝ, 
  (M = (4 * Real.cos α, 2 * Real.sqrt 2 * Real.sin α)) ∧ 
  (N = (-1 + Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)) ∧
  ∀ d : ℝ, (d = Real.sqrt ((M.fst - N.fst) ^ 2 + (M.snd - N.snd) ^ 2)) → d ≥ Real.sqrt 7 - Real.sqrt 2 :=
begin
  sorry
end

end min_distance_MN_l555_555190


namespace perimeter_of_regular_polygon_l555_555406

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l555_555406


namespace liquid_X_percentage_36_l555_555698

noncomputable def liquid_X_percentage (m : ℕ) (pX : ℕ) (m_evaporate : ℕ) (m_add : ℕ) (p_add : ℕ) : ℕ :=
  let m_X_initial := (pX * m / 100)
  let m_water_initial := ((100 - pX) * m / 100)
  let m_X_after_evaporation := m_X_initial
  let m_water_after_evaporation := m_water_initial - m_evaporate
  let m_X_additional := (p_add * m_add / 100)
  let m_water_additional := ((100 - p_add) * m_add / 100)
  let m_X_new := m_X_after_evaporation + m_X_additional
  let m_water_new := m_water_after_evaporation + m_water_additional
  let m_total_new := m_X_new + m_water_new
  (m_X_new * 100 / m_total_new)

theorem liquid_X_percentage_36 :
  liquid_X_percentage 10 30 2 2 30 = 36 := by
  sorry

end liquid_X_percentage_36_l555_555698


namespace algebraic_expression_identity_l555_555590

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l555_555590


namespace tan_inverse_least_positive_l555_555230

variables (a b x : ℝ)

-- Condition 1: tan(x) = a / (2*b)
def condition1 : Prop := Real.tan x = a / (2 * b)

-- Condition 2: tan(2*x) = 2*b / (a + 2*b)
def condition2 : Prop := Real.tan (2 * x) = (2 * b) / (a + 2 * b)

-- The theorem stating the least positive value of x is arctan(0)
theorem tan_inverse_least_positive (h1 : condition1 a b x) (h2 : condition2 a b x) : ∃ k : ℝ, Real.arctan k = 0 :=
by
  sorry

end tan_inverse_least_positive_l555_555230


namespace circle_divides_rectangle_sides_l555_555801

theorem circle_divides_rectangle_sides :
  ∀ (r : ℝ) (a b : ℝ), r = 26 ∧ a = 36 ∧ b = 60 → 
    (∃ (s1 s2 s3 s4 : ℝ), s1 = 26 ∧ s2 = 34 ∧ s3 = 2 ∧ s4 = 48 ∧ (s1 + s2 = a ∧ s3 + s4 = b)) ∧ 
    (∃ (t1 t2 : ℝ), t1 = 26 ∧ t2 = 10 ∧ (t1 + t2 = a)) :=
by 
  intros r a b h
  have hr : r = 26 := h.1 
  have ha : a = 36 := h.2.1 
  have hb : b = 60 := h.2.2 
  exists 26, 34, 2, 48
  constructor
  split 
  repeat {exact rfl}
  split 
  exact rfl
  ringer
  exists 26, 10
  constructor
  split 
  repeat {exact rfl}
  split 
  exact rfl
  ringer
  repeat {split; repeat {exact rfl}}
  sorry

end circle_divides_rectangle_sides_l555_555801


namespace regular_polygon_perimeter_is_28_l555_555400

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l555_555400


namespace minimum_value_of_fraction_l555_555518

theorem minimum_value_of_fraction (a b : ℝ) (h1 : a > 2 * b) (h2 : 2 * b > 0) :
  (a^4 + 1) / (b * (a - 2 * b)) >= 16 :=
sorry

end minimum_value_of_fraction_l555_555518


namespace find_ff2_eq_two_l555_555954

def f (x : ℝ) : ℝ :=
  if 1 < x then real.log x else real.exp x

theorem find_ff2_eq_two : f (f 2) = 2 := by
  sorry

end find_ff2_eq_two_l555_555954


namespace range_of_a_l555_555541

variable {α : Type}
variable (f : α → α) (a : α)

noncomputable def decreasing_function_on_domain (f : α → α) (domain : Set α) : Prop :=
  ∀ (x y : α), x ∈ domain → y ∈ domain → x < y → f y < f x

variable [LinearOrderedField α]

theorem range_of_a (h₁ : decreasing_function_on_domain f (set.Ioo (-1 : α) 1))
  (h₂ : f (1 - a) < f (2 * a - 1)) : 0 < a ∧ a < 2 / 3 :=
  sorry

end range_of_a_l555_555541


namespace log_base_3_of_9_l555_555492

theorem log_base_3_of_9 : log 3 9 = 2 :=
by
  sorry

end log_base_3_of_9_l555_555492


namespace quadratic_roots_l555_555159

theorem quadratic_roots (a b c : ℝ)
  (h1 : a * (-3)^2 + b * (-3) + c = 0)
  (h2 : a * (-2)^2 + b * (-2) + c = -3)
  (h3 : a * 0^2 + b * 0 + c = -3) : 
  (x1 x2 : ℝ) (hx1 : x1 = -3) (hx2 : x2 = 1) → a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by
  sorry

end quadratic_roots_l555_555159


namespace monotonicity_valid_a_l555_555950

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x - x

theorem monotonicity (a : ℝ) :
  (∀ a ≤ 0, ∀ x > 0, deriv (λ x, f a x) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Ioo 0 a, deriv (λ x, f a x) x > 0) ∧
  (∀ a > 0, ∀ x ∈ Ioi a, deriv (λ x, f a x) x < 0) :=
sorry

theorem valid_a (a : ℝ) (h : ∀ x > 0, x * (exp x - a - 1) - f a x ≥ 1) : a = 1 :=
sorry

end monotonicity_valid_a_l555_555950


namespace ilya_triplet_invariant_l555_555618

theorem ilya_triplet_invariant (a b c : ℕ) (n : ℕ) (h : a = 70 ∧ b = 61 ∧ c = 20 ∧ n = 1989) :
  let transform := λ t : (ℕ × ℕ × ℕ), (t.2 + t.3, t.1 + t.3, t.1 + t.2) in
  ∀ k, k ≤ n → (transform^[k] (a, b, c)).fst - (transform^[k] (a, b, c)).snd = 50 :=
by 
  sorry

end ilya_triplet_invariant_l555_555618
