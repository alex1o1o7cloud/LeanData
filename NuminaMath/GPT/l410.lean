import Mathlib

namespace compare_a_b_c_l410_410909

noncomputable def a : ℝ := Real.log 0.4 / Real.log 4
noncomputable def b : ℝ := Real.log 0.2 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 0.2

theorem compare_a_b_c : b > c ∧ c > a :=
by
  sorry

end compare_a_b_c_l410_410909


namespace largest_prime_factor_of_10201_l410_410383

theorem largest_prime_factor_of_10201 : ∃ p, p ∈ {7, 13, 3, 37} ∧ p = 37 ∧ Prime p :=
by
  sorry

end largest_prime_factor_of_10201_l410_410383


namespace max_value_sin_cos_expression_l410_410889

theorem max_value_sin_cos_expression (x y z : ℝ) 
  (h1 : sin x^2 + cos x^2 = 1)
  (h2 : sin (2 * y)^2 + cos (2 * y)^2 = 1)
  (h3 : sin (3 * z)^2 + cos (3 * z)^2 = 1) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l410_410889


namespace area_of_ellipse_l410_410526

variable (x y a b : ℝ)

def ellipse_area (x₁ y₁ x₂ y₂ hx hy : ℝ) (p₁ p₂ : ℝ) : ℝ := 
  let cₓ := (x₁ + x₂) / 2
  let c_y := (y₁ + y₂) / 2
  let a := (Real.sqrt ((x₂ - cₓ)^2 + (y₂ - c_y)^2))
  let b_sq := ((p₁ - cₓ)^2 / a^2) + ((p₂ - c_y)^2 / b)^2
  π * a * (Real.sqrt (b_sq))

theorem area_of_ellipse :
  ellipse_area (-9) 5 11 5 6 9 = (80 * Real.sqrt 3 * π) / 3 :=
by
  sorry

end area_of_ellipse_l410_410526


namespace third_side_exists_l410_410479

theorem third_side_exists (a b : ℝ) (h : 0 < a) (k : 0 < b) :
  (b < a ∧ a < 2 * b) ∨ (a < b ∧ b < 2 * a) → 
  ∃ c, c = 3 * |a - b| ∧ ∀ x y z : ℝ, x = c / 3 → y = c / 3 → z = c / 3 → x + y + z = c :=
by 
  intro h_cond
  use 3 * |a - b|
  split
  {
    refl
  }
  {
     intros x y z hx hy hz
     rw [hx, hy, hz]
     linarith
  }

end third_side_exists_l410_410479


namespace factorial_div_eq_l410_410848

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410848


namespace angle_XYZ_90_degrees_l410_410159

-- Define a regular octagon and a square inside it such that they share one side.
def regular_octagon : Prop := sorry
def square_inside_octagon : Prop := sorry

-- Define points X, Y, Z where Y and Z are consecutive vertices of the square,
-- and X is a vertex of the octagon.
def points_consecutive_vertices_square (X Y Z : Prop) : Prop := sorry

theorem angle_XYZ_90_degrees (X Y Z : Prop) :
  regular_octagon ∧ square_inside_octagon ∧ points_consecutive_vertices_square X Y Z →
  angle XYZ = 90 :=
sorry

end angle_XYZ_90_degrees_l410_410159


namespace log_inequality_l410_410779

theorem log_inequality : ∀ (base x y : ℝ), 0 < base ∧ base < 1 ∧ 0 < x ∧ 0 < y → x < y → log base y < log base x :=
by sorry

# Example instance proving log_inequality for the given numbers
example : log (0.4 : ℝ) 6 < log (0.4 : ℝ) 4 :=
log_inequality 0.4 4 6 (by norm_num) (by norm_num)

end log_inequality_l410_410779


namespace range_of_a_l410_410074

theorem range_of_a (a m : ℝ) (h_a_pos : a > 0) :
  (m^2 - 7 * m * a + 12 * a^2 < 0) -> (1 < m ∧ m < (3/2)) ->
  (3*a ≥ 1) ∧ (4*a ≤ (3/2)) ∧ ((1/3) ≤ a ∧ a ≤ (3/8)) :=
by
  intros h_p h_q
  have h_a_ineq1 : 3*a ≥ 1,
  sorry
  have h_a_ineq2 : 4*a ≤ (3/2),
  sorry
  exact ⟨h_a_ineq1, h_a_ineq2, ⟨(1/3) ≤ a, a ≤ (3/8)⟩⟩

end range_of_a_l410_410074


namespace smallest_n_l410_410683

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410683


namespace regular_polygon_diagonals_l410_410375

theorem regular_polygon_diagonals (n : ℕ) (h1 : ∀ (n : ℕ), 180 * (n - 2) = 120 * n) : (n = 6) → (n * (n - 3)) / 2 = 9 :=
by
  intro hn
  rw hn
  sorry

end regular_polygon_diagonals_l410_410375


namespace myopia_relation_l410_410786

def myopia_data := 
  [(1.00, 100), (0.50, 200), (0.25, 400), (0.20, 500), (0.10, 1000)]

noncomputable def myopia_function (x : ℝ) : ℝ :=
  100 / x

theorem myopia_relation (h₁ : 100 = (1.00 : ℝ) * 100)
    (h₂ : 100 = (0.50 : ℝ) * 200)
    (h₃ : 100 = (0.25 : ℝ) * 400)
    (h₄ : 100 = (0.20 : ℝ) * 500)
    (h₅ : 100 = (0.10 : ℝ) * 1000) :
  (∀ x > 0, myopia_function x = 100 / x) ∧ (myopia_function 250 = 0.4) :=
by
  sorry

end myopia_relation_l410_410786


namespace ap_intersects_at_midpoint_l410_410191

variables {A B C D E P M : Type*}
variables [ConvexPentagon ABCDE]
variables (angle : Type*) (BAC CAD DAE ABC ACD ADE : angle)
variables (BD CE : A → A → Prop) (intersects_at : A → A → A → Prop)

noncomputable def TriangleSimilarity (angle1 angle2 : angle) : Prop :=
angle1 = angle2

theorem ap_intersects_at_midpoint 
  (convex : ConvexPentagon ABCDE)
  (h1 : TriangleSimilarity BAC CAD)
  (h2 : TriangleSimilarity CAD DAE)
  (h3 : TriangleSimilarity ABC ACD)
  (h4 : TriangleSimilarity ACD ADE)
  (h5 : intersects_at BD P)
  (h6 : intersects_at CE P)
  (h7 : ∃ M, intersects_at (AP) M) :
  M = (midpoint C D) :=
sorry

end ap_intersects_at_midpoint_l410_410191


namespace part1_part2_l410_410858

noncomputable def z1 (a : ℝ) : ℂ := ⟨√3, -(a + 1)⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨(a^2 + √3) / 4, (a^2 - a - 2) / 8⟩

-- Part 1: Prove that if z1 > z2, then a = -1
theorem part1 (a : ℝ) : z1 a > z2 a → a = -1 :=
sorry

-- Part 2: Given a = 0 and the specific values of z1 and z2, prove that α = 5π / 3
noncomputable def z1_0 : ℂ := z1 0
noncomputable def z2_0 : ℂ := z2 0
noncomputable def alpha : ℝ := (5 * π) / 3

theorem part2 : (z1_0 * z2_0 = complex.exp (complex.I * alpha)) → alpha = (5 * π) / 3 :=
sorry

end part1_part2_l410_410858


namespace honey_container_ounces_l410_410614

-- Define the conditions
variable (serving_per_cup : ℕ) (cups_per_night : ℕ) (servings_per_ounce : ℕ) (nights : ℕ) 

-- Hypotheses based on the conditions
def conditions : Prop :=
  serving_per_cup = 1 ∧ 
  cups_per_night = 2 ∧
  servings_per_ounce = 6 ∧
  nights = 48

-- Define the statement to prove
theorem honey_container_ounces
          (serving_per_cup cups_per_night servings_per_ounce nights : ℕ)
          (h : conditions serving_per_cup cups_per_night servings_per_ounce nights) :
  let total_servings := nights * (serving_per_cup * cups_per_night)
  in total_servings / servings_per_ounce = 16 := by
  sorry

end honey_container_ounces_l410_410614


namespace smallest_n_l410_410681

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410681


namespace find_t_vals_l410_410064

theorem find_t_vals (x t : ℂ) (x1 x2 : ℂ) (h1 : x^2 + t*x + 2 = 0)
  (h2 : |x1 - x2| = 2*sqrt(2)) (h3 : t ∈ ℝ) :
  t = -4 ∨ t = 0 ∨ t = 4 :=
sorry

end find_t_vals_l410_410064


namespace total_customers_l410_410747

open Set

variable (A B C : Set Nat)
variable (h1 : (A ∩ B).card = 80)
variable (h2 : (A ∩ C).card = 90)
variable (h3 : (B ∩ C).card = 100)
variable (h4 : (A ∩ B ∩ C).card = 20)

theorem total_customers : (A ∪ B ∪ C).card = 230 := by
  have hAB := h1
  have hAC := h2
  have hBC := h3
  have hABC := h4
  -- applying the principle of inclusion-exclusion
  calc
    (A ∪ B ∪ C).card = (A.card + B.card + C.card) - (A ∩ B).card - (A ∩ C).card - (B ∩ C).card + (A ∩ B ∩ C).card : sorry
    ... = 230 : sorry

end total_customers_l410_410747


namespace elizabeth_fruits_l410_410376

def total_fruits (initial_bananas initial_apples initial_grapes eaten_bananas eaten_apples eaten_grapes : Nat) : Nat :=
  let bananas_left := initial_bananas - eaten_bananas
  let apples_left := initial_apples - eaten_apples
  let grapes_left := initial_grapes - eaten_grapes
  bananas_left + apples_left + grapes_left

theorem elizabeth_fruits : total_fruits 12 7 19 4 2 10 = 22 := by
  sorry

end elizabeth_fruits_l410_410376


namespace distance_from_point_to_line_is_correct_l410_410869

-- Definitions of the given points and line
def point_a : (ℝ × ℝ × ℝ) := (3, -3, 4)
def line_point1 : (ℝ × ℝ × ℝ) := (1, 0, -2)
def line_point2 : (ℝ × ℝ × ℝ) := (0, 4, 0)

-- Function to compute the distance from a point to a line determined by two points
noncomputable def distance_point_to_line (a b c : ℝ × ℝ × ℝ) : ℝ := 
  let (x1, y1, z1) := b in
  let (x2, y2, z2) := c in
  let (x0, y0, z0) := a in
  let dir := (x2 - x1, y2 - y1, z2 - z1) in
  let param t := (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)) in
  let v_param := param ((x0 - x1) * (x1 - x2) + (y0 - y1) * (y1 - y2) + (z0 - z1) * (z1 - z2)) / ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) in
  let v := (fst v_param - x0, snd v_param - y0, trd v_param - z0) in
  (v.0^2 + v.1^2 + v.2^2).sqrt

-- Proof statement of the distance from point (3, -3, 4) to the described line
theorem distance_from_point_to_line_is_correct : distance_point_to_line point_a line_point1 line_point2 = 85/7 :=
by 
  sorry

end distance_from_point_to_line_is_correct_l410_410869


namespace find_fake_coin_with_two_weighings_l410_410654

-- Assume we have four coins and one of them is fake
axiom coin : Type
axiom is_real_coin : coin → Prop
axiom is_fake_coin : coin → Prop
axiom A B C D : coin

-- Assumptions about the coins
axiom three_real_one_fake : (is_real_coin A ∧ is_real_coin B ∧ is_real_coin C ∧ is_fake_coin D) ∨
                            (is_real_coin A ∧ is_real_coin B ∧ is_fake_coin C ∧ is_real_coin D) ∨
                            (is_real_coin A ∧ is_fake_coin B ∧ is_real_coin C ∧ is_real_coin D) ∨
                            (is_fake_coin A ∧ is_real_coin B ∧ is_real_coin C ∧ is_real_coin D)

-- Two-pan balance scale assumptions:
axiom balance_scale_result : coin → coin → Prop -- balance_scale_result x y means x and y are balanced
axiom balance_scale_unbalanced_result : coin → coin → Prop -- balance_scale_unbalanced_result x y means x and y are unbalanced

-- Theorem that states, we can determine the fake coin using only two weighings
theorem find_fake_coin_with_two_weighings :
  ∃ fake_coin : coin, 
  (balance_scale_result A B ∧ (balance_scale_result A C → fake_coin = D) ∧ (balance_scale_unbalanced_result A C → fake_coin = C)) ∨
  (balance_scale_unbalanced_result A B ∧ (balance_scale_result B C → fake_coin = A) ∧ (balance_scale_unbalanced_result B C → fake_coin = B)) :=
sorry

end find_fake_coin_with_two_weighings_l410_410654


namespace line_AB_minimized_condition_l410_410421

theorem line_AB_minimized_condition :
  ∀ (x y : ℝ) (P : ℝ × ℝ),
  (x^2 + y^2 - 2*x - 2*y - 2 = 0) →
  (2*P.1 + P.2 + 2 = 0) →
  let M: ℝ × ℝ := (1, 1),
      PM := dist P M,
      AB := dist P (0, 1),
      minimized : PM * AB → (2*x + y + 1 = 0)
  (2 * x + y + 1 = 0) :=
begin
  sorry
end

end line_AB_minimized_condition_l410_410421


namespace problem1_problem2_l410_410796

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l410_410796


namespace find_slope_of_bisector_l410_410235

theorem find_slope_of_bisector 
  (m1 m2 : ℝ) 
  (h_m1 : m1 = 2) 
  (h_m2 : m2 = 4) 
  (h_def : ∀ m1 m2, (k = (m1 + m2 + √(1 + m1^2 + m2^2)) / (1 - m1 * m2)) 
                  ∨ (k = (m1 + m2 - √(1 + m1^2 + m2^2)) / (1 - m1 * m2))) : 
  k = (√21 - 6) / 7 :=
by
  sorry

end find_slope_of_bisector_l410_410235


namespace part_1_part_2_l410_410158

namespace CoordinatePlane

noncomputable def C (x : ℝ) : ℝ := (1/2)*x + sqrt ((1/4)*x^2 + 2)

theorem part_1
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁ = C x₁)
  (h₂ : y₂ = C x₂) :
  let H₁ := (y₁, y₁), H₂ := (y₂, y₂) in
  let area_triangle (P : ℝ × ℝ) (H : ℝ × ℝ) : ℝ := 
    (1/2) * |P.1 * (H.2 - 0) + H.1 * (0 - P.2)| in
  area_triangle (x₁, y₁) H₁ = area_triangle (x₂, y₂) H₂ := 
sorry

theorem part_2 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁ = C x₁)
  (h₂ : y₂ = C x₂)
  (hx : x₁ < x₂) :
  let bounded_area := 2 * real.log (y₂ / y₁) in
  bounded_area = 2 * real.log (y₂ / y₁) := 
sorry

end CoordinatePlane

end part_1_part_2_l410_410158


namespace solve_natural_numbers_system_l410_410224

theorem solve_natural_numbers_system :
  ∃ a b c : ℕ, (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (a + b + c)) ∧
  ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end solve_natural_numbers_system_l410_410224


namespace orthogonal_projection_area_l410_410216

-- Definitions from the problem conditions
variables {P : Type} [plane_polygon P]
variables {S : ℝ} -- area of the polygon
variables {S' : ℝ} -- area of the orthogonal projection
variables {φ : ℝ} -- angle between the projection plane and the plane of the polygon

-- The theorem to prove
theorem orthogonal_projection_area (S : ℝ) (φ : ℝ) (S' : ℝ) :
  S' = S * cos φ :=
sorry

end orthogonal_projection_area_l410_410216


namespace dealer_selling_price_above_cost_l410_410712

variable (cost_price : ℝ := 100)
variable (discount_percent : ℝ := 20)
variable (profit_percent : ℝ := 20)

theorem dealer_selling_price_above_cost :
  ∀ (x : ℝ), 
  (0.8 * x = 1.2 * cost_price) → 
  x = cost_price * (1 + profit_percent / 100) :=
by
  sorry

end dealer_selling_price_above_cost_l410_410712


namespace nancy_total_spending_l410_410736

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l410_410736


namespace count_squares_below_line_l410_410250

theorem count_squares_below_line (units : ℕ) :
  let intercept_x := 221;
  let intercept_y := 7;
  let total_squares := intercept_x * intercept_y;
  let diagonal_squares := intercept_x - 1 + intercept_y - 1 + 1; 
  let non_diag_squares := total_squares - diagonal_squares;
  let below_line := non_diag_squares / 2;
  below_line = 660 :=
by
  sorry

end count_squares_below_line_l410_410250


namespace number_of_ways_to_put_7_balls_in_2_boxes_l410_410498

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l410_410498


namespace smallest_n_45_l410_410689

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410689


namespace find_b_l410_410898

theorem find_b (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : b = 4 :=
  sorry

end find_b_l410_410898


namespace contracting_arrangements_1680_l410_410356

def num_contracting_arrangements (n a b c d : ℕ) : ℕ :=
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c

theorem contracting_arrangements_1680 : num_contracting_arrangements 8 3 1 2 2 = 1680 := by
  unfold num_contracting_arrangements
  simp
  sorry

end contracting_arrangements_1680_l410_410356


namespace smallest_n_for_sum_of_square_roots_inequality_l410_410028

theorem smallest_n_for_sum_of_square_roots_inequality
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  sqrt (a / (b + c + d)) + sqrt (b / (a + c + d)) + sqrt (c / (a + b + d)) + sqrt (d / (a + b + c)) < 4 :=
by sorry

end smallest_n_for_sum_of_square_roots_inequality_l410_410028


namespace harold_shared_with_five_friends_l410_410482

theorem harold_shared_with_five_friends 
  (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) (shared : ℕ) (friends : ℕ)
  (H1 : total_marbles = 100)
  (H2 : kept_marbles = 20)
  (H3 : marbles_per_friend = 16)
  (H4 : shared = total_marbles - kept_marbles)
  (H5 : friends = shared / marbles_per_friend) :
  friends = 5 :=
by
  sorry

end harold_shared_with_five_friends_l410_410482


namespace nancy_total_spent_l410_410740

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l410_410740


namespace boys_without_calculators_l410_410197

theorem boys_without_calculators :
    ∀ (total_boys students_with_calculators girls_with_calculators : ℕ),
    total_boys = 16 →
    students_with_calculators = 22 →
    girls_with_calculators = 13 →
    total_boys - (students_with_calculators - girls_with_calculators) = 7 :=
by
  intros
  sorry

end boys_without_calculators_l410_410197


namespace base4_division_l410_410863

/-- Given in base 4:
2023_4 div 13_4 = 155_4
We need to prove the quotient is equal to 155_4.
-/
theorem base4_division (n m q r : ℕ) (h1 : n = 2 * 4^3 + 0 * 4^2 + 2 * 4^1 + 3 * 4^0)
    (h2 : m = 1 * 4^1 + 3 * 4^0)
    (h3 : q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0)
    (h4 : n = m * q + r)
    (h5 : 0 ≤ r ∧ r < m):
  q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0 := 
by
  sorry

end base4_division_l410_410863


namespace problem1_problem2_l410_410797

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l410_410797


namespace probability_negative_roots_probability_of_neg_roots_l410_410511

open Real

theorem probability_negative_roots (a : ℝ) (h : 0 ≤ a ∧ a ≤ 5) :
  let Δ := 4 * a^2 - 12 * a + 8 in
  let sum_roots_neg := -2 * a < 0 in
  let prod_roots_pos := 3 * a - 2 > 0 in
  Δ ≥ 0 ∧ sum_roots_neg ∧ prod_roots_pos ↔ a ∈ Icc (2 : ℝ) 5 :=
by {
  sorry
}

theorem probability_of_neg_roots :
  ∃ (a : ℝ → Prop), (∀ a, a ∈ Icc (0 : ℝ) 5 → a ∈ Icc (2 : ℝ) 5) →
  (∫ a in Icc (0 : ℝ) 5, if (a ∈ Icc (2 : ℝ) 5) then 1 else 0) / (5 - 0) = (3 / 5 : ℝ) :=
by {
  sorry
}

end probability_negative_roots_probability_of_neg_roots_l410_410511


namespace xyz_product_neg4_l410_410977

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l410_410977


namespace find_fifth_term_of_GP_l410_410247

noncomputable def first_term : ℝ := real.rpow 2 (1/4)
noncomputable def second_term : ℝ := real.rpow 2 (1/2)
noncomputable def third_term : ℝ := real.rpow 2 (3/4)

theorem find_fifth_term_of_GP :
  let r := second_term / first_term,
      fourth_term := third_term * r,
      fifth_term := fourth_term * r
  in fifth_term = real.rpow 2 (5/4) :=
by
  let r := second_term / first_term
  let fourth_term := third_term * r
  let fifth_term := fourth_term * r
  have hr: r = real.rpow 2 (1/4) := by sorry
  have hf: fourth_term = real.rpow 2 1 := by sorry
  have result: fifth_term = real.rpow 2 (5/4) := by sorry
  exact result

end find_fifth_term_of_GP_l410_410247


namespace intersect_at_one_point_l410_410369

theorem intersect_at_one_point (a : ℝ) : 
  (a * (4 * 4) + 4 * 4 * 6 = 0) -> a = 2 / (3: ℝ) :=
by sorry

end intersect_at_one_point_l410_410369


namespace angle_BFC_is_right_l410_410914

noncomputable theory
open EuclideanGeometry

-- Definitions of the entities involved
variables {A B C P Q F B_1 C_1 : Point}
variables [acute_angle_triangle A B C]
variables [is_altitude A P Q B_1]
variables [is_altitude A P Q C_1]
variables [is_on_extension B B_1 P]
variables [is_on_extension C C_1 Q]
variables [angle (A P Q) = 90]

-- The theorem to prove
theorem angle_BFC_is_right : 
  ∀ {A B C P Q F B_1 C_1: Point}, 
    acute_angle_triangle A B C → 
    is_altitude A P Q B_1 → 
    is_altitude A P Q C_1 →
    is_on_extension B B_1 P → 
    is_on_extension C C_1 Q → 
    angle (A P Q) = 90 → 
    angle (B F C) = 90 := 
sorry

end angle_BFC_is_right_l410_410914


namespace population_increase_l410_410997

theorem population_increase (initial_population final_population: ℝ) (r: ℝ) : 
  initial_population = 14000 →
  final_population = 16940 →
  final_population = initial_population * (1 + r) ^ 2 →
  r = 0.1 :=
by
  intros h_initial h_final h_eq
  sorry

end population_increase_l410_410997


namespace samantha_probability_l410_410604

noncomputable def probability_of_selecting_yellow_apples 
  (total_apples : ℕ) (yellow_apples : ℕ) (selection_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_apples selection_size
  let yellow_ways := Nat.choose yellow_apples selection_size
  yellow_ways / total_ways

theorem samantha_probability : 
  probability_of_selecting_yellow_apples 10 5 3 = 1 / 12 := 
by 
  sorry

end samantha_probability_l410_410604


namespace number_of_true_propositions_l410_410459

theorem number_of_true_propositions {k : ℤ} {a : ℝ} (P1: (k = 1 ↔ ∀ x, (cos (2 * k * x) - sin (2 * k * x)) = cos (2 * x)))
  (P2: (∀ x, sin (2 * (x - π / 6) - π / 6) = cos (2 * x)))
  (P3: (∀ x, ax^2 - 2ax + 1 > 0) ↔ 0 < a ∧ a < 1)
  (P4: (O : ℝ × ℝ) (A B C : ℝ × ℝ), (O.1 + A.1, O.2 + A.2) + (O.1 + C.1, O.2 + C.2) = (B.1 + B.1, B.2 + B.2)
    → (area {O, A, B} / area {O, A, C}) = 1 / 2)
  : nat := sorry

end number_of_true_propositions_l410_410459


namespace at_least_one_ge_one_l410_410565

theorem at_least_one_ge_one (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 :=
by 
  sorry

end at_least_one_ge_one_l410_410565


namespace max_value_sin_cos_expression_l410_410881

-- Define the variables and the expressions

def max_trig_expression (x y z : ℝ) : ℝ :=
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))

-- State the theorem to find the maximum value of the given expression.

theorem max_value_sin_cos_expression : ∀ x y z : ℝ, max_trig_expression x y z ≤ 4.5 :=
by {
  sorry -- This is where the proof would go
}

end max_value_sin_cos_expression_l410_410881


namespace no_sum_of_three_squares_l410_410220

theorem no_sum_of_three_squares (n : ℤ) (h : n % 8 = 7) : 
  ¬ ∃ a b c : ℤ, a^2 + b^2 + c^2 = n :=
by 
sorry

end no_sum_of_three_squares_l410_410220


namespace isosceles_right_triangle_leg_length_l410_410630

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l410_410630


namespace inequality_proof_l410_410080

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d :=
  sorry

end inequality_proof_l410_410080


namespace bowling_ball_weight_l410_410588

theorem bowling_ball_weight (b c : ℝ) (h1 : 9 * b = 6 * c) (h2 : 4 * c = 120) : b = 20 :=
sorry

end bowling_ball_weight_l410_410588


namespace sample_mean_properties_sample_distribution_confidence_level_l410_410726

noncomputable def sample_mean (n : ℕ) (ξ : ℕ → ℝ) : ℝ :=
  (1 / n) * (∑ i in range n, ξ i)

def unbiased_sample_mean (a σ0 : ℝ) (n : ℕ) (ξ : ℕ → ℝ) : Prop :=
  ∀ (i : ℕ), ξ i ~ 𝓝 (a, σ0^2) →
  ℰ (sample_mean n ξ) = a ∧
  D (sample_mean n ξ) = σ0^2 / n

def efficient_estimator (σ0 : ℝ) (n : ℕ) (T : (ℕ → ℝ) → ℝ): Prop :=
  ∀ (ξ : ℕ → ℝ), ∀ (a : ℝ), ∀ (i : ℕ),
  ξ i ~ 𝓝 (a, σ0^2) → unbiased_estimator T ξ → 
  V(T ξ) >= σ0^2 / n

theorem sample_mean_properties {a σ0 : ℝ} {n : ℕ} {ξ : ℕ → ℝ} :
  unbiased_sample_mean a σ0 n ξ →
  efficient_estimator σ0 n (sample_mean n)

theorem sample_distribution {a σ0 : ℝ} {n : ℕ} (ξ : ℕ → ℝ) :
  ∀ (i : ℕ), ξ i ~ 𝓝 (a, σ0^2) →
  (sqrt n * (sample_mean n ξ - a) / σ0) ~ 𝓝 (0, 1)

noncomputable def confidence_interval (a σ0 : ℝ) (n : ℕ) (z_alpha : ℝ) (ξ : ℕ → ℝ) : Set ℝ :=
  {x | sample_mean n ξ - σ0 / sqrt n * z_alpha ≤ x ∧ x ≤ sample_mean n ξ + σ0 / sqrt n * z_alpha}

theorem confidence_level {a σ0 : ℝ} {n : ℕ} {z_alpha : ℝ} {ξ : ℕ → ℝ} :
  ∀ (i : ℕ), ξ i ~ 𝓝 (a, σ0^2) →
  ∀ α (0 < α < 1), ∀ (z_alpha : ℝ), 
  (1 - α = (1 / sqrt (2 * π)) * ∫ (t : ℝ) in (-z_alpha)..z_alpha, e^(-(t^2) / 2)) →
  P ((λ x, sample_mean n ξ - σ0 / sqrt n * z_alpha ≤ x ∧ x ≤ sample_mean n ξ + σ0 / sqrt n * z_alpha)) = λ (i : ℕ), 1 - α

end sample_mean_properties_sample_distribution_confidence_level_l410_410726


namespace product_xyz_equals_zero_l410_410967

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l410_410967


namespace area_of_circle_defined_by_equation_l410_410664

def circle_area : ℝ :=
  let r := 4 in
  π * r^2

theorem area_of_circle_defined_by_equation :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 * y = 3 ↔ (x + 3)^2 + (y - 2)^2 = 16) →
  circle_area = 16 * π :=
by
  intros h
  have h_radius : (∃ r, ∀ x y, (x + 3)^2 + (y - 2)^2 = r^2 → r = 4) := sorry
  rw [circle_area]
  exact h_radius sorry
  sorry

end area_of_circle_defined_by_equation_l410_410664


namespace product_xyz_l410_410974

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l410_410974


namespace triangle_segment_sum_l410_410554

theorem triangle_segment_sum (A B C D : Point)
  (h_triangle : triangle A B C)
  (h_angle_B : ∠B = 40)
  (h_angle_C : ∠C = 40)
  (h_BD_angle_bisector : BD.is_angle_bisector ∠B) :
  BD.length + DA.length = BC.length :=
by
  sorry

end triangle_segment_sum_l410_410554


namespace isosceles_right_triangle_leg_length_l410_410635

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l410_410635


namespace B_and_C_complementary_l410_410052

def EventA (selected : List String) : Prop :=
  selected.count "boy" = 1

def EventB (selected : List String) : Prop :=
  selected.count "boy" ≥ 1

def EventC (selected : List String) : Prop :=
  selected.count "girl" = 2

theorem B_and_C_complementary :
  ∀ selected : List String,
    (selected.length = 2 ∧ (EventB selected ∨ EventC selected)) ∧ 
    (¬ (EventB selected ∧ EventC selected)) →
    (EventB selected → ¬ EventC selected) ∧ (EventC selected → ¬ EventB selected) :=
  sorry

end B_and_C_complementary_l410_410052


namespace mark_donates_cans_l410_410195

-- Definitions coming directly from the conditions
def num_shelters : ℕ := 6
def people_per_shelter : ℕ := 30
def cans_per_person : ℕ := 10

-- The final statement to be proven
theorem mark_donates_cans : (num_shelters * people_per_shelter * cans_per_person) = 1800 :=
by sorry

end mark_donates_cans_l410_410195


namespace max_height_of_container_l410_410329

-- Define the conditions
def steel_bar_length := 6 -- total length of the steel bar in meters
def ratio_base_adjacents := (3, 4) -- the ratio of adjacent sides of the base (3:4)

-- The theorem to prove
theorem max_height_of_container (h : ℝ) (x : ℝ) 
  (h_positive : x > 0)
  (h_upper_bound : x < 3 / 14) 
  (base_length_condition : 
    2 * (3 * x + 4 * x + (1.5 - 7 * x)) = steel_bar_length) 
  (volume : ℝ := 12 * x^2 * (1.5 - 7 * x)) 
  (critical_point : x = 1 / 7) 
  : h = 0.5 :=
begin
  -- place the proof here
  sorry
end

end max_height_of_container_l410_410329


namespace cos_90_eq_zero_l410_410009

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410009


namespace factorial_div_eq_l410_410847

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410847


namespace arithmetic_geometric_sequence_common_difference_l410_410783

theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (a 1 = 1) →
  (d ≠ 0) →
  (∀ n, a (n + 1) = a n + d) →
  (a 1, a 2, a 5) forms a geometric sequence →
  d = 2 :=
by
  intros a d h1 hd ha hgeom
  -- Proof omitted 
  sorry

end arithmetic_geometric_sequence_common_difference_l410_410783


namespace problem_statement_l410_410193

variable (x : ℝ)
def S (m : ℕ) := x^m + (1 / x)^m

theorem problem_statement (h : x + 1/x = 4) : S x 6 = 2700 := 
  sorry

end problem_statement_l410_410193


namespace parallelogram_diagonal_intersection_l410_410658

noncomputable def triangle (A B C : Point) : Prop :=
  ¬ (collinear A B C)

noncomputable def parallelogram_with_diagonal (A B C D : Point) : Prop :=
  ¬ collinear A B C ∧ collinear A C D ∧ collinear B C D

theorem parallelogram_diagonal_intersection
  (A B C : Point)
  (⊓: ∀ (P : Point), triangle A B C)
  (⊓ P_1 P_2 P_3 D_1 D_2 D_3 : Point)
  (⊓ : parallelogram_with_diagonal B P_1 C A)
  (⊓ : parallelogram_with_diagonal C P_2 A B)
  (⊓ : parallelogram_with_diagonal A P_3 B C) :
  (∃ P : Point, collinear D_1 P D_2 ∧ collinear D_2 P D_3) :=
  sorry

end parallelogram_diagonal_intersection_l410_410658


namespace negation_of_prop_p_l410_410955

noncomputable def neg_of_p (p : Prop) : Prop := 
  ∀ x, x > 5 → ¬(2 * x^2 - x + 1 > 0)

theorem negation_of_prop_p :
  (∃ x, x > 5 ∧ 2 * x^2 - x + 1 > 0) ↔ (∀ x, x > 5 → 2 * x^2 - x + 1 ≤ 0) :=
begin
  sorry,
end

end negation_of_prop_p_l410_410955


namespace rachel_lamp_placement_l410_410218

theorem rachel_lamp_placement :
  let plants := {basil, aloe, cactus}
  let white_lamps := {w1, w2}
  let red_lamps := {r1, r2}
  ∃ (placement : plants → (white_lamps ∪ red_lamps)),
    card {placement | ∀ p ∈ plants, placement p ∈ (white_lamps ∪ red_lamps)} = 20 := 
sorry

end rachel_lamp_placement_l410_410218


namespace correct_operation_l410_410707

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^4 ≠ a^6) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  ((a^2 * b)^3 = a^6 * b^3) ∧
  (a^6 / a^6 ≠ a) :=
by
  sorry

end correct_operation_l410_410707


namespace original_candle_length_l410_410267

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (h_current : current_length = 48) (h_factor : factor = 1.33) :
  (current_length * factor = 63.84) :=
by
  -- The proof goes here
  sorry

end original_candle_length_l410_410267


namespace median_of_sequence_between_1_and_2018_l410_410231

theorem median_of_sequence_between_1_and_2018 : 
  let seq := (list.range' 23 (2018 / 105).to_nat).map (λ k, 23 + k * 105) in 
  (seq = [23, 128, 233, 338, 443, 548, 653, 758, 863, 968, 1073, 1178, 1283, 1388, 1493, 1598, 1703, 1808, 1913, 2018]) → 
  let sorted_seq := seq.sort (≤) in 
  (sorted_seq.nth (sorted_seq.length / 2) + sorted_seq.nth (sorted_seq.length / 2 - 1)) / 2 = 1020.5 :=
by
  sorry

end median_of_sequence_between_1_and_2018_l410_410231


namespace circumference_of_back_wheel_l410_410240

theorem circumference_of_back_wheel
  (C_f : ℝ) (C_b : ℝ) (D : ℝ) (N_b : ℝ)
  (h1 : C_f = 30)
  (h2 : D = 1650)
  (h3 : (N_b + 5) * C_f = D)
  (h4 : N_b * C_b = D) :
  C_b = 33 :=
sorry

end circumference_of_back_wheel_l410_410240


namespace range_of_a_l410_410919

noncomputable
def proposition_p (x : ℝ) : Prop := abs (x - (3 / 4)) <= (1 / 4)
noncomputable
def proposition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) <= 0

theorem range_of_a :
  (∀ x : ℝ, proposition_p x → ∃ x : ℝ, proposition_q x a) ∧
  (∃ x : ℝ, ¬(proposition_p x → proposition_q x a )) →
  0 ≤ a ∧ a ≤ (1 / 2) :=
sorry

end range_of_a_l410_410919


namespace central_angle_of_sector_l410_410938

theorem central_angle_of_sector (r : ℝ) (A : ℝ) (h1 : r = 6) (h2 : A = 6 * Real.pi) : 
  let n := 360 * A / (Real.pi * r^2) 
  in n = 60 :=
by
  sorry

end central_angle_of_sector_l410_410938


namespace rose_tom_profit_difference_l410_410719

def investment_months (amount: ℕ) (months: ℕ) : ℕ :=
  amount * months

def total_investment_months (john_inv: ℕ) (rose_inv: ℕ) (tom_inv: ℕ) : ℕ :=
  john_inv + rose_inv + tom_inv

def profit_share (investment: ℕ) (total_investment: ℕ) (total_profit: ℕ) : ℤ :=
  (investment * total_profit) / total_investment

theorem rose_tom_profit_difference
  (john_inv rs_per_year: ℕ := 18000 * 12)
  (rose_inv rs_per_9_months: ℕ := 12000 * 9)
  (tom_inv rs_per_8_months: ℕ := 9000 * 8)
  (total_profit: ℕ := 4070):
  profit_share rose_inv (total_investment_months john_inv rose_inv tom_inv) total_profit -
  profit_share tom_inv (total_investment_months john_inv rose_inv tom_inv) total_profit = 370 := 
by
  sorry

end rose_tom_profit_difference_l410_410719


namespace coefficient_sum_expression_l410_410077

theorem coefficient_sum_expression :
  ∀ (x : ℝ) (a : ℕ → ℝ),
  (x + 1)^2 * (x + 2) ^ 2016 = (∑ i in Finset.range 2019, a i * (x + 2) ^ i) →
  (∑ i in Finset.range 2019, a i * (1 / 2) ^ i) = 1 / 2 ^ 2018 :=
by
  intros x a h
  sorry

end coefficient_sum_expression_l410_410077


namespace problem1_problem2_l410_410799

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l410_410799


namespace find_theta_l410_410866

-- Define the relevant angles and values
def cos_15 := real.cos (real.pi / 12)
def sin_15 := real.sin (real.pi / 12)
def sin_45 := real.sin (real.pi / 4)
def theta := real.pi / 12

-- Assign the target equation based on the given conditions
theorem find_theta (h : cos_15 = sin_45 + real.sin theta) : theta = real.pi / 12 :=
by
  sorry  -- Proof not required

end find_theta_l410_410866


namespace volume_of_tetrahedron_l410_410563

def volume (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

theorem volume_of_tetrahedron (A B C D P: ℝ × ℝ × ℝ) :
  let S_A := (1/4 : ℝ) • (P + B + C + D),
      S_B := (1/4 : ℝ) • (P + C + D + A),
      S_C := (1/4 : ℝ) • (P + D + A + B),
      S_D := (1/4 : ℝ) • (P + A + B + C) in
  volume S_A S_B S_C S_D = (1/64 : ℝ) * volume A B C D :=
begin
  sorry
end

end volume_of_tetrahedron_l410_410563


namespace factorial_division_identity_l410_410829

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410829


namespace males_who_dont_listen_l410_410265

-- Define the conditions
variables
  (l_m : ℕ) -- number of males who listen
  (d_f : ℕ) -- number of females who don't listen
  (l_t : ℕ) -- total number of listeners
  (d_t : ℕ) -- total number of people who don't listen

-- Given specific values for conditions
def conditions := (l_m = 45 ∧ d_f = 87 ∧ l_t = 115 ∧ d_t = 160)

-- The statement to prove
theorem males_who_dont_listen (h : conditions):
  ∃ n : ℕ, n = 73 :=
by
  intro h,
  sorry

end males_who_dont_listen_l410_410265


namespace range_of_a_l410_410450
noncomputable def f (x : ℝ) : ℝ := real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - 1/2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : ∀ x > 0, f x < g a x) : a > 1/2 :=
by
  sorry

end range_of_a_l410_410450


namespace factorize_x2_plus_2x_l410_410040

theorem factorize_x2_plus_2x (x : ℝ) : x^2 + 2*x = x * (x + 2) :=
by sorry

end factorize_x2_plus_2x_l410_410040


namespace smallest_n_satisfies_conditions_l410_410701

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410701


namespace part1_part2_l410_410950

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Statement for part 1
theorem part1 (a b : ℝ) (h1 : f a b (-1) = 0) (h2 : f a b 3 = 0) (h3 : a ≠ 0) :
  a = -1 ∧ b = 4 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h1 : f a b 1 = 2) (h2 : a + b = 1) (h3 : a > 0) (h4 : b > 0) :
  (∀ x > 0, 1 / a + 4 / b ≥ 9) :=
sorry

end part1_part2_l410_410950


namespace probability_point_above_curve_l410_410227

-- Define a and b as single-digit positive integers
def single_digit_positive_integer (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

-- Define the probability computation problem
theorem probability_point_above_curve :
  (∑ a in (Finset.filter single_digit_positive_integer (Finset.range 10)), 
   (∑ b in (Finset.filter single_digit_positive_integer (Finset.range 10)), 
    if b > a^3 / (a + 1) then 1 else 0) / 9) / 9 = 19 / 81 :=
by 
  sorry

end probability_point_above_curve_l410_410227


namespace pentagon_exists_with_divisible_cut_l410_410031

noncomputable def exists_pentagon_with_conditions : Prop :=
  ∃ (ABCDE : Type) (A B C D E X Y : Point) (ABXY : Rectangle) (CDX EDY : Triangle),
  aspect_ratio ABXY 2 1 ∧
  congruent_triangles CDX EDY ∧
  (∃ cut : Line, divides_into_three_parts cut ABCDE ∧ 
    (∃ parts : List Part, length parts = 3 ∧
      can_combine_two_to_form_third parts))

theorem pentagon_exists_with_divisible_cut :
  exists_pentagon_with_conditions :=
sorry

end pentagon_exists_with_divisible_cut_l410_410031


namespace combination_formula_l410_410808

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410808


namespace angles_cos_inequality_l410_410555

-- Define the conditions
def triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B > C ∧ A + C > B ∧ B + C > A

-- Define the theorem
theorem angles_cos_inequality (A B C : ℝ) (h_triangle : triangle A B C) (h_angles : A < B < C) :
  cos (2 * A) > cos (2 * B) > cos (2 * C) ↔ A < B < C :=
by
  sorry

end angles_cos_inequality_l410_410555


namespace savings_per_month_l410_410017

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end savings_per_month_l410_410017


namespace min_lambda_l410_410434

theorem min_lambda (n : ℕ) (hn : n ≥ 2) : 
  (∃ λ, ∀ (a b : Fin n → ℝ), (∀ i, 0 < a i) → (∀ i, 0 < b i) →
    (∏ i, a i = ∏ i, b i) → (∑ i, a i ≤ λ * ∑ i, b i)) ∧ 
  (∀ λ, (∀ (a b : Fin n → ℝ), (∀ i, 0 < a i) → (∀ i, 0 < b i) →
    (∏ i, a i = ∏ i, b i) → (∑ i, a i ≤ λ * ∑ i, b i)) → (n-1 ≤ λ)) := 
sorry

end min_lambda_l410_410434


namespace geometric_progression_general_formula_range_of_a_l410_410575

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
| 0     => a
| (n+1) => 2 * (Finset.range (n+1)).sum sequence_a + 4^n

def sequence_S (a : ℝ) : ℕ → ℝ
| 0     => sequence_a a 0
| (n+1) => (Finset.range (n+2)).sum (sequence_a a)

def sequence_b (a : ℝ) : ℕ → ℝ
| n => sequence_S a n - 4^n

-- (I) Prove that the sequence {b_n} is a geometric progression
theorem geometric_progression (a : ℝ) (h : a ≠ 4) : 
  ∀ n : ℕ, sequence_b a (n+1) = 3 * sequence_b a n := sorry

-- (II) Find the general formula for the sequence {a_n}
theorem general_formula (a : ℝ) (h : a ≠ 4) : 
  ∀ n ≥ 1, 
    sequence_a a n = if n = 1 then a 
                     else 3 * 4^(n-1) + 2 * (a - 4) * 3^(n-2) := sorry

-- (III) Determine the range of values for the real number a
theorem range_of_a (a : ℝ) : (∀ n : ℕ, sequence_a a (n+1) ≥ sequence_a a n) ↔ 
  (a ∈ Set.Icc (-4) 4 ∪ Set.Ici 4) := sorry

end geometric_progression_general_formula_range_of_a_l410_410575


namespace apple_equation_l410_410562

-- Conditions directly from a)
def condition1 (x : ℕ) : Prop := (x - 1) % 3 = 0
def condition2 (x : ℕ) : Prop := (x + 2) % 4 = 0

theorem apple_equation (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : 
  (x - 1) / 3 = (x + 2) / 4 := 
sorry

end apple_equation_l410_410562


namespace y_is_even_and_monotonically_decreasing_l410_410780

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

def is_monotonically_decreasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ (x y : ℝ), x ∈ s → y ∈ s → x < y → f x ≥ f y

def y (x : ℝ) : ℝ := -x^2 + 1

theorem y_is_even_and_monotonically_decreasing :
  is_even_function y ∧ is_monotonically_decreasing_on y {x : ℝ | 0 < x} :=
sorry

end y_is_even_and_monotonically_decreasing_l410_410780


namespace minimum_inlets_needed_l410_410341

noncomputable def waterInflow (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := x * a - b

theorem minimum_inlets_needed (a b : ℝ) (ha : a = b)
  (h1 : (4 * a - b) * 5 = (2 * a - b) * 15)
  (h2 : (a * 9 - b) * 2 ≥ 1) : 
  ∃ n : ℕ, 2 * (a * n - b) ≥ (4 * a - b) * 5 := 
by 
  sorry

end minimum_inlets_needed_l410_410341


namespace lambda_range_l410_410437

theorem lambda_range (λ : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + λ * n) : (∀ n : ℕ, a (n+1) > a n) → λ > -3 :=
by
  intros h_inc
  have h_ineq : ∀ n : ℕ, 2 * n + 1 + λ > 0, from sorry
  exact sorry

end lambda_range_l410_410437


namespace max_value_sin_cos_expression_l410_410877

-- Define the variables and the expressions

def max_trig_expression (x y z : ℝ) : ℝ :=
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))

-- State the theorem to find the maximum value of the given expression.

theorem max_value_sin_cos_expression : ∀ x y z : ℝ, max_trig_expression x y z ≤ 4.5 :=
by {
  sorry -- This is where the proof would go
}

end max_value_sin_cos_expression_l410_410877


namespace dawn_monthly_savings_l410_410015

-- Definitions for the conditions
def annual_salary := 48000
def months_in_year := 12
def saving_rate := 0.10

-- Define the monthly salary
def monthly_salary := annual_salary / months_in_year

-- Define the monthly savings
def monthly_savings := monthly_salary * saving_rate

-- The theorem to prove
theorem dawn_monthly_savings : monthly_savings = 400 :=
by
  -- Proof details skipped
  sorry

end dawn_monthly_savings_l410_410015


namespace domain_of_f_l410_410461

-- Define the function
def f (x : ℝ) : ℝ := (Real.sqrt (x - 2)) / (x - 1)

-- The domain is the set of all x such that both conditions are satisfied
def domain_f : Set ℝ := {x | x ≥ 2 ∧ x ≠ 1}

-- The main statement to prove
theorem domain_of_f : ∀ x : ℝ, (f x) = f x → x ∈ domain_f := by
  sorry

end domain_of_f_l410_410461


namespace max_value_l410_410885

/-- 
Proof of the maximum value of the expression 
(sin x + sin 2y + sin 3z) * (cos x + cos 2y + cos 3z)
-/
theorem max_value (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_l410_410885


namespace petya_wins_l410_410203

-- Defining the game conditions and the board
structure Game :=
  (n : ℕ)                         -- size of the board
  (board : Matrix ℕ (n+1) (n+1))  -- representing the board; initially should be all white except one black cell
  (player_turn : bool)            -- true for Petya's turn, false for Vasya's turn

-- Initial condition: All cells are white except one corner which is black
def initial_condition (g: Game) : Prop := 
  g.board 0 0 = 1 ∧               -- (0, 0) is black
  (∀ i j, (i ≠ 0 ∨ j ≠ 0) → g.board i j = 0) -- All other cells are white

-- The rook's movement constraints
inductive Move : Type
| horizontal (steps : ℕ) : Move
| vertical (steps : ℕ) : Move

open Move

-- Definition of valid move under the game rules (cannot move through/onto black cells)
def valid_move (g : Game) (r c : ℕ) (m : Move) : Prop :=
  match m with
  | horizontal steps => r < g.n ∧ c + steps ≤ g.n ∧ (∀ k, 1 ≤ k ∧ k ≤ steps → g.board r (c + k) = 0)
  | vertical steps => r + steps ≤ g.n ∧ c < g.n ∧ (∀ k, 1 ≤ k ∧ k ≤ steps → g.board (r + k) c = 0)
  end

-- Transition function: make a move and update the board
def make_move (g : Game) (r c : ℕ) (m : Move) : Game :=
  match m with
  | horizontal steps => ⟨g.n, (Matrix.copy g.board).update_range (λ x y, x = r ∧ c < y ∧ y ≤ c + steps) 1, !g.player_turn⟩
  | vertical steps => ⟨g.n, (Matrix.copy g.board).update_range (λ x y, c = y ∧ r < x ∧ x ≤ r + steps) 1, !g.player_turn⟩
  end

-- Theorem: Petya always wins with optimal play
theorem petya_wins : ∀ (n : ℕ), ∃ (g : Game), g.n = n → initial_condition g → (∀ g', valid_move g' r c m → 
                        (make_move g' r c m).player_turn = false → ∃ m', valid_move (make_move g' r c m) r' c' m' → false) :=
sorry

end petya_wins_l410_410203


namespace solve_abs_eq_l410_410865

theorem solve_abs_eq (x : ℝ) : |2*x - 6| = 3*x + 6 ↔ x = 0 :=
by 
  sorry

end solve_abs_eq_l410_410865


namespace smallest_n_satisfies_conditions_l410_410700

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410700


namespace smallest_n_for_2007_l410_410257

/-- The smallest number of positive integers \( n \) such that their product is 2007 and their sum is 2007.
Given that \( n > 1 \), we need to show 1337 is the smallest such \( n \).
-/
theorem smallest_n_for_2007 (n : ℕ) (H : n > 1) :
  (∃ s : Finset ℕ, (s.sum id = 2007) ∧ (s.prod id = 2007) ∧ (s.card = n)) → (n = 1337) :=
sorry

end smallest_n_for_2007_l410_410257


namespace problem_solution_l410_410189

theorem problem_solution (x1 x2 : ℝ) (h1 : x1^2 + x1 - 4 = 0) (h2 : x2^2 + x2 - 4 = 0) (h3 : x1 + x2 = -1) : 
  x1^3 - 5 * x2^2 + 10 = -19 := 
by 
  sorry

end problem_solution_l410_410189


namespace distinct_real_roots_range_l410_410902

-- Define the logarithmic equation
def equation (k x : ℝ) : ℝ := k * x^2 - 2 * Real.log x - k

-- The problem statement
theorem distinct_real_roots_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation k x₁ = 0 ∧ equation k x₂ = 0 ∧ x₁ > 0 ∧ x₂ > 0) →
  k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1) := 
sorry

end distinct_real_roots_range_l410_410902


namespace probability_first_ball_odd_and_in_range_is_correct_l410_410309
noncomputable theory

def probability_first_ball_odd_and_in_range : ℝ :=
  let total_balls := 200
  let odd_balls_in_1_49 := 24
  in (odd_balls_in_1_49.to_real / total_balls.to_real)

theorem probability_first_ball_odd_and_in_range_is_correct (total_balls : ℕ) (odd_balls_in_1_49 : ℕ) :
  total_balls = 200 → odd_balls_in_1_49 = 24 → probability_first_ball_odd_and_in_range = 0.12 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  done 

end probability_first_ball_odd_and_in_range_is_correct_l410_410309


namespace intriguing_quadruples_count_l410_410366

theorem intriguing_quadruples_count :
  (∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + 2 * d > b + c) :=
begin
  -- The solution shows that the number of such quadruples is 420
  have h : ∃ quadruples, ∃ (n : ℕ), quadruples.card = n ∧ n = 420, from sorry,
  exact h,
end

end intriguing_quadruples_count_l410_410366


namespace cost_of_ice_cream_scoop_l410_410037

theorem cost_of_ice_cream_scoop
  (num_meals : ℕ) (meal_cost : ℕ) (total_money : ℕ)
  (total_meals_cost : num_meals * meal_cost = 30)
  (remaining_money : total_money - 30 = 15)
  (num_ice_cream_scoops : ℕ) (cost_per_scoop : ℕ)
  (total_cost : 30 + 15 = total_money)
  (total_ice_cream_cost : num_ice_cream_scoops * cost_per_scoop = remaining_money) :
  cost_per_scoop = 5 :=
by
  have h_num_meals : num_meals = 3 := by sorry
  have h_meal_cost : meal_cost = 10 := by sorry
  have h_total_money : total_money = 45 := by sorry
  have h_num_ice_cream_scoops : num_ice_cream_scoops = 3 := by sorry
  exact sorry

end cost_of_ice_cream_scoop_l410_410037


namespace chickens_at_stacy_farm_l410_410655
-- Importing the necessary library

-- Defining the provided conditions and correct answer in Lean 4.
theorem chickens_at_stacy_farm (C : ℕ) (piglets : ℕ) (goats : ℕ) : 
  piglets = 40 → 
  goats = 34 → 
  (C + piglets + goats) = 2 * 50 → 
  C = 26 :=
by
  intros h_piglets h_goats h_animals
  sorry

end chickens_at_stacy_farm_l410_410655


namespace min_value_of_A2_minus_B2_nonneg_l410_410187

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)

theorem min_value_of_A2_minus_B2_nonneg (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z) ^ 2 - (B x y z) ^ 2 ≥ 36 :=
by
  sorry

end min_value_of_A2_minus_B2_nonneg_l410_410187


namespace length_of_room_l410_410242

theorem length_of_room (cost_per_metre : ℝ) (total_cost : ℝ) (width_cm : ℝ) (breadth : ℝ) (length : ℝ) : 
  cost_per_metre = 4.50 → total_cost = 810 → width_cm = 75 → breadth = 7.5 → 
  length = (total_cost / cost_per_metre) * (width_cm / 100) / breadth :=
by
  intros h1 h2 h3 h4
  let total_length_of_carpet := total_cost / cost_per_metre
  let width_m := width_cm / 100
  let total_area_of_carpet := total_length_of_carpet * width_m
  have h : length = total_area_of_carpet / breadth := by sorry
  rw [h1, h2, h3, h4] at h
  exact h

-- The specific value inferred from conditions implies that 'length' evaluates to the expected value, 18 meters.

end length_of_room_l410_410242


namespace arithmetic_sequence_third_term_l410_410536

theorem arithmetic_sequence_third_term :
  ∀ (a₁ a₆ : ℕ) (n : ℕ) (d : ℕ), 
    n = 6 → a₁ = 11 → a₆ = 51 → 
    d = (a₆ - a₁) / (n - 1) →
    ∀ e, e = a₁ + 2 * d → e = 27 :=
by
  intros a₁ a₆ n d hn ha₁ ha₆ hd e he
  rw [hn, ha₁, ha₆, hd] at *
  rw he
  sorry

end arithmetic_sequence_third_term_l410_410536


namespace range_of_c_l410_410131

variable (μ σ : ℝ) (ξ : ℝ)
variable (P : set ℝ → ℝ) (f : ℝ → ℝ)

def normal_distribution (μ σ : ℝ) : Prop :=
  P {x | μ - σ < x ∧ x < μ + σ} = 0.6826 ∧ P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544

def line_circle_distance_condition (c : ℝ) : Prop :=
  |c| / 13 < 1

theorem range_of_c
  (h1 : ξ ~ N(1, σ^2))
  (h2 : normal_distribution 1 σ)
  (d : ∀ x y, x^2 + y^2 = σ^2 → |12 * x - 5 * y + c| / sqrt (12^2 + 5^2) = 1) :
  |c| / 13 < 1 :=
sorry

end range_of_c_l410_410131


namespace problem1_problem2_l410_410798

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l410_410798


namespace largest_5_digit_integer_congruent_to_19_mod_26_l410_410665

theorem largest_5_digit_integer_congruent_to_19_mod_26 :
  ∃ n : ℕ, 10000 ≤ 26 * n + 19 ∧ 26 * n + 19 < 100000 ∧ (26 * n + 19 ≡ 19 [MOD 26]) ∧ 26 * n + 19 = 99989 :=
by
  sorry

end largest_5_digit_integer_congruent_to_19_mod_26_l410_410665


namespace polynomial_roots_and_sum_l410_410186

theorem polynomial_roots_and_sum (p q r s : ℤ) (m1 m2 m3 m4 : ℤ) 
  (hroots : ∀ i ∈ {m1, m2, m3, m4}, (i % 2 = 1) ∧ (i > 0))
  (hpoly : ∀ x : ℤ, (Polynomial.eval x (Polynomial.C s + Polynomial.C r * x + Polynomial.C q * x^2 + Polynomial.C p * x^3 + Polynomial.C x^4)) = (x + m1) * (x + m2) * (x + m3) * (x + m4))
  (hsum : p + q + r + s = 2023) : 
  s = 624 :=
begin
  sorry,
end

end polynomial_roots_and_sum_l410_410186


namespace sum_of_squares_of_consecutive_integers_l410_410899

theorem sum_of_squares_of_consecutive_integers :
  ∃ x : ℕ, x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) ∧ (x^2 + (x + 1)^2 + (x + 2)^2 = 77) :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l410_410899


namespace number_of_ways_to_distribute_balls_l410_410504

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l410_410504


namespace factorial_division_identity_l410_410830

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410830


namespace find_b_minus_a_2023_l410_410065

theorem find_b_minus_a_2023 (a b : ℝ) 
  (h : |a + b - 1| + real.sqrt (2 * a + b - 2) = 0) : (b - a)^2023 = -1 := 
sorry

end find_b_minus_a_2023_l410_410065


namespace find_lambda_n_l410_410569

theorem find_lambda_n (n : ℕ) (h : n > 1) (z : ℕ → ℂ) 
  (nz : ∀ k, k > 0 ∧ k ≤ n → z k ≠ 0) 
  (z_nil : z (n + 1) = z 1) : 
  ∑ k in Finset.range n, |z k+1|^2 ≥ 
  (if (n % 2 = 0) then (n / 4) else (n / (4 * ((Real.cos (Real.pi / (2 * n)))^2)))) * 
  (Finset.min' (Finset.range n) (by simp) (λ k, |z (k+1) - z k|^2)) :=
sorry -- Proof to be filled in


end find_lambda_n_l410_410569


namespace problem_statement_l410_410382

noncomputable def circle_equation_with_symmetric_reflection : Prop := 
  let original_circle : (ℝ × ℝ) → Prop := λ p, (p.1 + 2)^2 + p.2^2 = 5
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)
  let symmetric_circle : (ℝ × ℝ) → Prop := λ p, p.1^2 + (p.2 + 2)^2 = 5
  ∀ p : ℝ × ℝ, original_circle p ↔ symmetric_circle (symmetric_point p)

theorem problem_statement : circle_equation_with_symmetric_reflection :=
  sorry

end problem_statement_l410_410382


namespace wages_for_both_l410_410775

-- Given definitions
variables {A B S : ℝ}

-- Conditions
def condition1 : Prop := S = 20 * A
def condition2 : Prop := S = 30 * B
def condition3 : Prop := A / B = 3 / 2

-- Proof statement
theorem wages_for_both (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  (∃ (d : ℝ), d * ((5 / 3) * A) = S) :=
by {
  use 12,
  sorry
}

end wages_for_both_l410_410775


namespace nested_composition_l410_410178

def N (x : ℝ) : ℝ := 2 * Real.sqrt x
def O (x : ℝ) : ℝ := x ^ 2

theorem nested_composition :
  N (O (N (O (N (O 3))))) = 24 := by
  sorry

end nested_composition_l410_410178


namespace length_AF_l410_410171

theorem length_AF
  (A B C D E : Type)
  (AF CE : A → B)
  (m : B → ℝ)
  (square_ABCD : A)
  (point_outside : E)
  (angle_BEC : m (AF E) = 90)
  (F_on_CE : C)
  (AF_perp_CE : ∃ x, AF x = CE x)
  (AB_25 : ∀ x, |A x - B x| = 25)
  (BE_7 : ∀ x, |B x - E x| = 7) :
  (∀ x, |AF x| = 33) := by
  sorry

end length_AF_l410_410171


namespace find_ratio_of_hyperbola_asymptotes_l410_410901

theorem find_ratio_of_hyperbola_asymptotes (a b : ℝ) (h : a > b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → |(2 * b / a)| = 1) : 
  a / b = 2 := 
by 
  sorry

end find_ratio_of_hyperbola_asymptotes_l410_410901


namespace Tony_exercises_time_l410_410269

-- Conditions
def distance_walked : ℝ := 3  -- miles
def distance_run : ℝ := 10  -- miles
def speed_walk : ℝ := 3  -- miles per hour
def speed_run : ℝ := 5  -- miles per hour
def days_per_week : ℕ := 7  -- days

-- The time spent walking each morning
def time_walk : ℝ := distance_walked / speed_walk

-- The time spent running each morning
def time_run : ℝ := distance_run / speed_run

-- Total time spent exercising each morning
def total_time_per_morning : ℝ :=  time_walk + time_run

-- Total time spent exercising each week
def total_time_per_week : ℝ := total_time_per_morning * days_per_week

-- Prove that Tony spends 21 hours each week exercising
theorem Tony_exercises_time :
  total_time_per_week = 21 := by
  sorry

end Tony_exercises_time_l410_410269


namespace factorial_div_eq_l410_410843

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410843


namespace arrangement_count_is_4_l410_410051

theorem arrangement_count_is_4 :
  let grid_size := 3
  let num_A := 4
  let num_B := 3
  let num_C := 3
  let fixed_A_position := (1, 1)
  let valid_arrangement (grid : Array (Array (Option Char))) : Prop :=
    grid.size = grid_size ∧
    ∀ i, (grid[i].size = grid_size ∧ ∑ j, if grid[i][j] = some 'A' then 1 else 0 = 1) ∧
    ∑ i, if grid[i][fixed_A_position.2 - 1] = some 'A' then 1 else 0 = 1 ∧
    ∑ j, if grid[fixed_A_position.1 - 1][j] = some 'A' then 1 else 0 = 1 ∧
    ∀ r c, (r ≠ fixed_A_position.1 - 1 ∨ c ≠ fixed_A_position.2 - 1) → grid[r][c] ≠ some 'A' →
    ∀ i, if grid[r][i] = some 'B' then ∀ k, grid[i][k] ≠ some 'B' ∧ grid[k][i] ≠ some 'B' ∧
         if grid[r][i] = some 'C' then ∀ k, grid[i][k] ≠ some 'C' ∧ grid[k][i] ≠ some 'C' 
  in
  ∃ (grid : Array (Array (Option Char))),
    grid[fixed_A_position.1 - 1][fixed_A_position.2 - 1] = some 'A' ∧
    valid_arrangement grid ∧
    ∑ i j, if grid[i][j] = some 'A' then 1 else 0 = num_A ∧
    ∑ i j, if grid[i][j] = some 'B' then 1 else 0 = num_B ∧
    ∑ i j, if grid[i][j] = some 'C' then 1 else 0 = num_C ∧
    (∃ n, ∑' n, (count_valid_arrangements grid)) = 4 :=
sorry

end arrangement_count_is_4_l410_410051


namespace max_value_sin_cos_expression_l410_410890

theorem max_value_sin_cos_expression (x y z : ℝ) 
  (h1 : sin x^2 + cos x^2 = 1)
  (h2 : sin (2 * y)^2 + cos (2 * y)^2 = 1)
  (h3 : sin (3 * z)^2 + cos (3 * z)^2 = 1) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l410_410890


namespace sum_binomial_coeffs_sum_all_coeffs_sum_odd_coeffs_sum_abs_vals_coef_l410_410161

-- Sum of binomial coefficients
theorem sum_binomial_coeffs (x y : ℕ) : 
  ∑ i in finset.range (10), binomial 9 i = 2^9 :=
sorry

-- Sum of all coefficients
theorem sum_all_coeffs :
  (2 - 3)^9 = -1 :=
sorry

-- Sum of all odd-numbered coefficients
theorem sum_odd_coeffs :
  (5^9 - 1) / 2 = (2 - 3)^9 - 
sorry

-- Sum of absolute values of coefficients in (2x + 3y)^9
theorem sum_abs_vals_coef :
  (2 + 3)^9 = 5^9 :=
sorry

end sum_binomial_coeffs_sum_all_coeffs_sum_odd_coeffs_sum_abs_vals_coef_l410_410161


namespace vectors_parallel_l410_410481

def vector_1 : ℝ × ℝ := (-1, 3)
def vector_2 (t : ℝ) : ℝ × ℝ := (1, t)

def slope (v : ℝ × ℝ) : ℝ := v.2 / v.1

theorem vectors_parallel (t : ℝ) : 
  (slope vector_1 = slope (vector_2 t)) ↔ t = -3 := by
  sorry

end vectors_parallel_l410_410481


namespace cos_90_eq_zero_l410_410008

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410008


namespace factorial_quotient_l410_410814

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410814


namespace floor_T_eq_31_l410_410572

def T : ℝ := √(∑ i in finset.range 1 1001, √(1 + 1/(i:ℝ)^3 + 1/((i+1):ℝ)^3))

theorem floor_T_eq_31 : ⌊T⌋ = 31 := 
by 
  sorry

end floor_T_eq_31_l410_410572


namespace sum_of_ks_with_distinct_real_roots_eq_45_l410_410030

theorem sum_of_ks_with_distinct_real_roots_eq_45 : 
  (∑ k in Finset.filter (λ k : ℕ, 196 - 20 * k > 0) (Finset.range 10)) = 45 :=
by
  sorry

end sum_of_ks_with_distinct_real_roots_eq_45_l410_410030


namespace sine_minus_cosine_l410_410078

variable {α : ℝ}

theorem sine_minus_cosine (h1 : sin α + cos α = 7 / 5) (h2 : π / 4 < α) (h3 : α < π / 2) :
  sin α - cos α = 1 / 5 :=
sorry

end sine_minus_cosine_l410_410078


namespace arrange_five_cubes_arrange_six_cubes_l410_410714

-- Define some basic concepts we need
def cube : Type := { shared_faces : ℕ }  -- Assume a cube type with a number of shared faces property
def arrangement (n : ℕ) : Type := list cube -- An arrangement is a list of cubes

-- Assume polygons are represented in a typical way
def polygonal_face (a b : cube) : Prop := a.shared_faces ≥ 1 ∧ b.shared_faces ≥ 1

-- State that it is possible to arrange five wooden cubes
theorem arrange_five_cubes :
  ∃ (cubes : arrangement 5), ∀ (a b : cube), a ∈ cubes ∧ b ∈ cubes → polygonal_face(a, b) :=
sorry

-- State that it is possible to arrange six wooden cubes
theorem arrange_six_cubes :
  ∃ (cubes : arrangement 6), ∀ (a b : cube), a ∈ cubes ∧ b ∈ cubes → polygonal_face(a, b) :=
sorry

end arrange_five_cubes_arrange_six_cubes_l410_410714


namespace paint_red_faces_of_octahedral_die_l410_410787

theorem paint_red_faces_of_octahedral_die :
  (finset.univ.subset (finset.univ : finset (fin (8)))) ∧ -- Die has 8 faces
  (card ((finset.univ : finset (fin 8)).powerset.filter (λ s, s.card = 2)) = 28) ∧
  (card (finset.filter (λ (s : finset (fin 8)), set.sum s.val = 9) ((finset.univ : finset (fin 8)).powerset.filter (λ s, s.card = 2))) = 4) →
  24 :=
begin
  sorry,
end

end paint_red_faces_of_octahedral_die_l410_410787


namespace smallest_positive_n_l410_410675

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410675


namespace matching_shoes_probability_l410_410770

-- Definitions and conditions:
def total_pairs : ℕ := 500
def distinct_pairs : ℕ := 300
def sizes : ℕ := 3
def shoes_per_pair : ℕ := 2

-- Problem statement: Prove the probability is 1/499.
theorem matching_shoes_probability :
  (total_pairs * shoes_per_pair = 1000) ∧ 
  (distinct_pairs / sizes = 100) ∧ 
  (size_assign : ∀ s:ℕ, shoes_per_size = 100 * 2) ∧ 
  (total_shoes = 500) ∧ 
  (∃ probability : ℚ, probability = 1 / 499) :=
begin
  -- Proof steps will be required here.
  sorry,
end

end matching_shoes_probability_l410_410770


namespace fraction_power_times_base_l410_410301

theorem fraction_power_times_base (a b c : ℕ) (h : a = 100) (hb : b = 101) (hc : c = 3) :
  (1 / 3 : ℝ) ^ a * (3 : ℝ) ^ b = 3 :=
by
  rw [h, hb, hc]
  -- Proof steps would follow here
  sorry

end fraction_power_times_base_l410_410301


namespace planes_perpendicular_l410_410918

-- Given different planes α, β, γ and different lines m, n,
variable (α β γ : Type) [plane α] [plane β] [plane γ]
variable (m n : Type) [line m] [line n]

-- If m⊥α
variable [perpendicular m α]

-- and n⊥β
variable [perpendicular n β]

-- and m⊥n
variable [perpendicular m n]

-- Then α⊥β
theorem planes_perpendicular (h1 : perpendicular m α) (h2 : perpendicular n β) (h3 : perpendicular m n) : perpendicular α β := sorry

end planes_perpendicular_l410_410918


namespace probability_even_diagonals_l410_410734

theorem probability_even_diagonals :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (grid : ℕ → ℕ → ℕ),
    (∀ i j, i ∈ {1, 2, 3} → j ∈ {1, 2, 3} → grid i j ∈ nums)
    ∧ (∀ n ∈ nums, ∃! (i, j), i ∈ {1, 2, 3} ∧ j ∈ {1, 2, 3} ∧ grid i j = n)
    → (even (grid 1 1 + grid 2 2 + grid 3 3) ∧ even (grid 1 3 + grid 2 2 + grid 3 1)) :=
by
  sorry

end probability_even_diagonals_l410_410734


namespace x_squared_minus_y_squared_l410_410991

theorem x_squared_minus_y_squared (x y : ℚ) (h₁ : x + y = 9 / 17) (h₂ : x - y = 1 / 51) : x^2 - y^2 = 1 / 289 :=
by
  sorry

end x_squared_minus_y_squared_l410_410991


namespace purchase_costs_10_l410_410054

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10_l410_410054


namespace product_of_constants_l410_410396

theorem product_of_constants (t : ℤ) (a b : ℤ) (h1 : a * b = 12)
  (h2 : t = a + b) :
  ∃ p : ℤ, (∀ t, (∃ a b : ℤ, a * b = 12 ∧ t = a + b) → t) ∧ p = -527776 :=
sorry

end product_of_constants_l410_410396


namespace ways_to_choose_lineup_l410_410308

-- Define the number of team members
def team_members : ℕ := 12

-- Define the number of members that can play center
def centers : ℕ := 4

-- Define the number of members that can play any position
def versatile_members : ℕ := team_members - centers

-- Define the distinct position constraints
def valid_lineup_count (total: ℕ) (center_count: ℕ) : ℕ :=
  let non_centers := total - center_count in
  center_count * (non_centers) * (non_centers - 1) * (non_centers - 2) * (non_centers - 3)

-- Prove the number of ways to choose the lineup is 31680
theorem ways_to_choose_lineup : valid_lineup_count team_members centers = 31680 :=
  by
    sorry

end ways_to_choose_lineup_l410_410308


namespace factorial_div_combination_l410_410826

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410826


namespace solve_N_l410_410995

noncomputable def N (a b c d : ℝ) := (a + b) / c - d

theorem solve_N : 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  N a b c d = -1 :=
by 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  let n := N a b c d
  sorry

end solve_N_l410_410995


namespace find_roses_in_november_l410_410627

noncomputable def roses_in_month : ℕ → ℕ
| 1 := 108    -- October
| 2 := 120    -- November (to be proved)
| 3 := 132    -- December
| 4 := 144    -- January
| 5 := 156    -- February
| _ := 0

theorem find_roses_in_november :
  roses_in_month 2 = 120 :=
by
  sorry

end find_roses_in_november_l410_410627


namespace intersection_of_sets_l410_410956

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2}) (hB : B = {1, 2, 3, 4}) :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l410_410956


namespace cost_of_ice_cream_l410_410034

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end cost_of_ice_cream_l410_410034


namespace cevian_ratio_l410_410996

variables {A B C O D E F : Type} [EuclideanGeometry A B C O D E F]

def inside_triangle (O : Point) (A B C D E F : Point) :=
  (segment A O).intersects_at (segment B C) = D ∧
  (segment B O).intersects_at (segment C A) = E ∧
  (segment C O).intersects_at (segment A B) = F

theorem cevian_ratio (O A B C D E F : Point)
  (h : inside_triangle O A B C D E F) :
  (dist O D / dist A D) + 
  (dist O E / dist B E) + 
  (dist O F / dist C F) = 1 :=
sorry

end cevian_ratio_l410_410996


namespace annie_barrettes_l410_410788

def total_decorations (B : ℕ) : ℕ := B + 2 * B + (B - 3)
def bobby_pins (B : ℕ) : ℕ := B - 3
def fourteen_percent_of_total (B : ℕ) : ℕ := nat.floor (0.14 * float.of_int (total_decorations B))

theorem annie_barrettes : ∃ (B : ℕ), bobby_pins B = fourteen_percent_of_total B ∧ B = 6 :=
begin
  use 6,
  -- these are the conditions we would verify
  have h1 : total_decorations 6 = 4 * 6 - 3,
  have h2 : bobby_pins 6 = nat.floor (0.14 * float.of_int (total_decorations 6)),
  sorry
end

end annie_barrettes_l410_410788


namespace no_odd_positive_factor_of_144_is_multiple_of_18_l410_410113

theorem no_odd_positive_factor_of_144_is_multiple_of_18 : 
  ∀ n, n ∣ 144 ∧ odd n ∧ 18 ∣ n → false :=
by 
  intros n h,
  obtain ⟨_, ⟨n_odd,_⟩⟩ := h,
  have := coprime.coprime_iff_gcd_eq_one 18 2,
  simp [coprime] at this,
  exact this (gcd_coprime 18 (odd_iff_one_not_zero.1 n_odd).trans (nat.prime_2.gcd_eq_one 18))
  sorry

end no_odd_positive_factor_of_144_is_multiple_of_18_l410_410113


namespace eccentricity_of_hyperbola_l410_410127

-- Given definitions and conditions
variables {a b : ℝ}

-- Assumptions
axioms (h₁ : a > 0) (h₂ : b > 0)
noncomputable def hyperbola := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)
noncomputable def asymptote := (3, -4)  -- Point the asymptote passes through.

-- Hypothesis based on the given conditions
axiom hyp_assume : (3 * b = 4 * a)

-- Goal to prove
theorem eccentricity_of_hyperbola : 
  ∃ e : ℝ, e = (5 / 3) ∧ (hyperbola ∧ asymptote ∧ h₁ ∧ h₂ ∧ hyp_assume) :=
begin
  sorry
end

end eccentricity_of_hyperbola_l410_410127


namespace equation_of_line_AB_l410_410422

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

noncomputable def line_equation (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

noncomputable def on_line (P : ℝ × ℝ) : Prop :=
  line_equation P.1 P.2

noncomputable def the_point : ℝ × ℝ :=
  (-1, 0)

theorem equation_of_line_AB :
  (circle_equation 1 1) ∧ (line_equation 1 1) ∧ on_line the_point →
  (2*the_point.1 + the_point.2 + 1 = 0) :=
begin
  sorry
end

end equation_of_line_AB_l410_410422


namespace trig_cos_sum_l410_410731

open Real

theorem trig_cos_sum :
  cos (37 * (π / 180)) * cos (23 * (π / 180)) - sin (37 * (π / 180)) * sin (23 * (π / 180)) = 1 / 2 :=
by
  sorry

end trig_cos_sum_l410_410731


namespace line_AB_minimized_condition_l410_410420

theorem line_AB_minimized_condition :
  ∀ (x y : ℝ) (P : ℝ × ℝ),
  (x^2 + y^2 - 2*x - 2*y - 2 = 0) →
  (2*P.1 + P.2 + 2 = 0) →
  let M: ℝ × ℝ := (1, 1),
      PM := dist P M,
      AB := dist P (0, 1),
      minimized : PM * AB → (2*x + y + 1 = 0)
  (2 * x + y + 1 = 0) :=
begin
  sorry
end

end line_AB_minimized_condition_l410_410420


namespace intersection_M_N_l410_410102

open Set

-- Definitions based on conditions
def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

-- Theorem statement based on the solution and equivalence proof problem
theorem intersection_M_N : M ∩ N = {1} :=
by
  sorry  -- proof is omitted for now

end intersection_M_N_l410_410102


namespace problem_statement_l410_410076

theorem problem_statement :
  let f (x : ℝ) := (1 - 2 * x) ^ 2004,
      a : ℕ → ℝ := coeff x (f x) in
  (finset.sum (finset.range 2005) (λ i, a 0 + a i) = 2004) :=
by
  sorry

end problem_statement_l410_410076


namespace factorial_div_eq_l410_410846

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410846


namespace xyz_product_value_l410_410988

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l410_410988


namespace max_value_sin_cos_expression_l410_410888

theorem max_value_sin_cos_expression (x y z : ℝ) 
  (h1 : sin x^2 + cos x^2 = 1)
  (h2 : sin (2 * y)^2 + cos (2 * y)^2 = 1)
  (h3 : sin (3 * z)^2 + cos (3 * z)^2 = 1) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l410_410888


namespace problem1_problem2_l410_410800

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l410_410800


namespace find_a4_l410_410083

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
if k ≤ n then
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))
else 0

theorem find_a4 (a0 a1 a2 a3 a4 a5 x : ℤ)
    (h : x^5 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5) :
  a4 = -5 :=
sorry

end find_a4_l410_410083


namespace bounded_sequence_l410_410439

variable (a : ℕ → ℝ)

def pos_seq (a: ℕ → ℝ) : Prop :=
  ∀ n, 0 < a n

def recurrence_relation (a: ℕ → ℝ) : Prop :=
  ∀ n, a(n+2) = 2 / (a(n+1) + a(n))

theorem bounded_sequence (a : ℕ → ℝ) (h_pos : pos_seq a) (h_recur : recurrence_relation a) :
  ∃ s t : ℝ, 0 < s ∧ 0 < t ∧ ∀ n, s ≤ a n ∧ a n ≤ t := 
sorry

end bounded_sequence_l410_410439


namespace a_n_formula_T_n_less_than_6_l410_410438

-- Given sequence a_n and the sum S_n
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions
axiom a_1 : a 1 = 1
axiom a_non_zero : ∀ n : ℕ, n > 0 → a n ≠ 0
axiom sum_condition : ∀ n : ℕ, n > 0 → a n * a (n + 1) = 4 * S n - 1

-- Finding the general formula for a_n
theorem a_n_formula : ∀ n : ℕ, n > 0 → a n = 2 * n - 1 :=
sorry

-- Given sequence b_n with a_n / b_n = 2^(n-1)
variable {b : ℕ → ℕ}
axiom b_relation : ∀ n : ℕ, n > 0 → a n = b n * 2^(n-1)

-- Define T_n as the sum of the first n terms of b_n
def T (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), b i

-- Prove T_n < 6
theorem T_n_less_than_6 : ∀ n : ℕ, T n < 6 :=
sorry

end a_n_formula_T_n_less_than_6_l410_410438


namespace product_is_120_l410_410860

theorem product_is_120 (n : ℕ) (h : n = 3) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  rw [h]
  norm_num
  sorry

end product_is_120_l410_410860


namespace problem_l410_410982

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l410_410982


namespace tunnel_length_equivalence_l410_410735

-- Definitions based on the conditions
def train_length : ℝ := 2   -- The train is 2 miles long
def train_speed : ℝ := 60   -- The train speed is 60 miles per hour
def tunnel_exit_time : ℝ := 4 / 60  -- The train exits the tunnel 4 minutes (4/60 hours) after the front enters the tunnel

-- The length of the tunnel given the conditions
theorem tunnel_length_equivalence (train_length : ℝ) (train_speed : ℝ) (tunnel_exit_time : ℝ) : 
  let front_distance := train_speed * tunnel_exit_time in
  let tunnel_length := front_distance - train_length in
  tunnel_length = 2 := 
by
  -- Definitions based on the problem
  let front_distance : ℝ := 60 * (4 / 60)  -- Convert 4 minutes to hours and calculate distance
  let tunnel_length : ℝ := front_distance - 2 -- Calculating tunnel length
  show tunnel_length = 2 from rfl -- Proving the tunnel length is 2 miles
  sorry  -- Skip the actual proof steps

end tunnel_length_equivalence_l410_410735


namespace factorial_div_eq_l410_410836
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410836


namespace min_val_of_a2_plus_b2_l410_410153

variable (a b : ℝ)

def condition := 3 * a - 4 * b - 2 = 0

theorem min_val_of_a2_plus_b2 : condition a b → (∃ a b : ℝ, a^2 + b^2 = 4 / 25) := by 
  sorry

end min_val_of_a2_plus_b2_l410_410153


namespace ephraim_one_more_head_than_keiko_l410_410170

-- Define probability function given the total number of outcomes and favorable outcomes
def prob {α : Type} [Fintype α] (s : Finset α) : ℚ :=
  (s.card : ℚ) / Fintype.card α

-- Define the event space for the coin tosses
def outcomes (n : ℕ) : Finset (Vector Bool n) :=
  Finset.univ

-- Define the condition Ephraim gets exactly one more head than Keiko
def condition (k_outcomes : Vector Bool 2) (e_outcomes : Vector Bool 3) : Bool :=
  let k_heads := k_outcomes.toList.count (λ b => b)
  let e_heads := e_outcomes.toList.count (λ b => b)
  e_heads = k_heads + 1

-- Define the set of favorable outcomes where Ephraim has exactly one more head than Keiko
def favorable_outcomes : Finset (Vector Bool 2 × Vector Bool 3) :=
  (outcomes 2).product (outcomes 3).filter (λ p => condition p.1 p.2)

-- Prove the probability of Ephraim getting exactly one more head than Keiko is 1/4.
theorem ephraim_one_more_head_than_keiko : prob favorable_outcomes = 1 / 4 :=
by
  -- proof goes here
  sorry

end ephraim_one_more_head_than_keiko_l410_410170


namespace smallest_positive_n_l410_410678

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410678


namespace locus_on_segment_AB_locus_inside_triangle_OAB_l410_410166

-- Define the conditions
variables {O A B M P Q H : Type*} [euclidean_geometry O A B M P Q H]

-- Define angle inequalities and orthocenter properties
axiom angle_AOB_lt_90 : angle A O B < 90
axiom is_perpendicular_MP_OA : is_perpendicular M P O A
axiom is_perpendicular_MQ_OB : is_perpendicular M Q O B
axiom is_orthocenter_H_OPQ : is_orthocenter H O P Q

-- The locus when M lies on segment AB
theorem locus_on_segment_AB (hM_on_AB : on_line_segment A B M)
    : locus H = line_segment C D :=
sorry

-- The locus when M lies inside triangle OAB
theorem locus_inside_triangle_OAB (hM_inside_OAB : inside_triangle O A B M)
    : locus H = interior_triangle O C D :=
sorry

end locus_on_segment_AB_locus_inside_triangle_OAB_l410_410166


namespace probability_three_digit_integer_divisible_by_4_l410_410343

open Nat

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem probability_three_digit_integer_divisible_by_4 :
  ∀ M : ℕ, nat.digits 10 M = [d1, d2, 5] → 
  (100 * d1 + 10 * d2 + 5) % 100 = 5 → 
  (nat.digits 10 (10 * d2 + 5)).length = 2 → 
    (∃ d2 : ℕ, d2 ∈ [0,1,2,3,4,5,6,7,8,9] ∧ divisible_by_4 (10 * d2 + 5)) = 2 / 10 :=
by
  sorry

end probability_three_digit_integer_divisible_by_4_l410_410343


namespace positive_difference_of_roots_l410_410388

open Polynomial

theorem positive_difference_of_roots (a b c : ℤ) (h_eq : a = 5 ∧ b = -11 ∧ c = -14)
  (h_quad : a ≠ 0) :
  let Δ := b^2 - 4 * a * c,
      root_diff := (Real.sqrt (Δ)) / (a * 2)
  in Δ = 401 ∧ root_diff = Real.sqrt 401 / 5 → 406 = 401 + 5 :=
by
  sorry

end positive_difference_of_roots_l410_410388


namespace count_irrationals_l410_410347

-- Define the list of numbers in the conditions
def numbers : List Real := [
  (Real.sqrt 2) / 2,
  0.2,
  22 / 7,
  Real.sqrt (4 / 9),
  0,
  Real.pi,
  4.211.repeating(1), -- representation of 4.2 overline 1, may be approximated differently
  - (Real.cbrt 8),
  2.020020002 -- increasing zeros pattern, may be approximated differently
]

-- Define the property of being irrational
def is_irrational (r : Real) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ r = p / q

-- The theorem statement that verifies the problem
theorem count_irrationals : (numbers.count is_irrational = 3) :=
by sorry

end count_irrationals_l410_410347


namespace point_not_on_graph_and_others_on_l410_410288

theorem point_not_on_graph_and_others_on (y : ℝ → ℝ) (h₁ : ∀ x, y x = x / (x - 1))
  : ¬ (1 = (1 : ℝ) / ((1 : ℝ) - 1)) 
  ∧ (2 = (2 : ℝ) / ((2 : ℝ) - 1)) 
  ∧ ((-1 : ℝ) = (1/2 : ℝ) / ((1/2 : ℝ) - 1)) 
  ∧ (0 = (0 : ℝ) / ((0 : ℝ) - 1)) 
  ∧ (3/2 = (3 : ℝ) / ((3 : ℝ) - 1)) := 
sorry

end point_not_on_graph_and_others_on_l410_410288


namespace total_distance_Cincinnati_to_NYC_l410_410602

theorem total_distance_Cincinnati_to_NYC :
  let distance_day1 := 20 in
  let distance_day2 := (distance_day1 / 2) - 6 in
  let distance_day3 := 10 in
  let distance_remaining := 36 in
  distance_day1 + distance_day2 + distance_day3 + distance_remaining = 70 :=
by
  sorry

end total_distance_Cincinnati_to_NYC_l410_410602


namespace correct_statements_l410_410949

-- Define the function f
def f (x : ℝ) : ℝ := sqrt 2 * sin (4 * x + π / 6)

-- Define the problem statement with conditions and correct answers
theorem correct_statements : 
  (¬ (∀ (x : ℝ), f (x + π / 3) = f (x - π / 3))) ∧          -- ① is false
  (∃ (count : ℕ), count = 8 ∧ ∀ (x : ℝ),                     -- ② is true
    x ∈ Ioo -π π → tan (4 * x + π / 6) = 0 →
    (x = π / 8 ∨ x = -π / 8 ∨ x = π / 3 ∨ x = -π / 3 ∨ -- list all extreme points
     x = π / 2 ∨ x = -π / 2 ∨ x = 2 * π / 3 ∨ x = -2 * π / 3)) ∧
  (∀ (x ∈ Icc (-π / 8) (π / 8)), f(x) ≤ sqrt 2) ∧              -- ③ is true
  (¬ (∀ (x y : ℝ), x < y → x ∈ Ioo (-π / 4) (π / 4) →         -- ④ is false
      y ∈ Ioo (-π / 4) (π / 4) → f(x) ≤ f(y))) :=
by
  sorry  -- The proof is omitted.

end correct_statements_l410_410949


namespace arm_wrestling_qualification_l410_410143

theorem arm_wrestling_qualification :
  ∀ (n : Nat),
  n = 896 →
  ∃ k : Nat, 
  k = 10 ∧ 
  ∀ (a w : Nat), 
  a = 1 ∨ a = 0 → 
  w = 1 ∨ w = 0 → 
  (∀ (m p : Nat), 
  m = (if p % 2 = 0 then p / 2 else p / 2 + 1) → 
  (if m < n then m * (m + 1) else m = k)) :=
by
  -- proof omitted
  sorry

end arm_wrestling_qualification_l410_410143


namespace serena_mother_age_l410_410605

theorem serena_mother_age {x : ℕ} (h : 39 + x = 3 * (9 + x)) : x = 6 := 
by
  sorry

end serena_mother_age_l410_410605


namespace smallest_n_l410_410679

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410679


namespace solution_set_product_positive_l410_410926

variable {R : Type*} [LinearOrderedField R]

def is_odd (f : R → R) : Prop := ∀ x : R, f (-x) = -f (x)

variable (f g : R → R)

noncomputable def solution_set_positive_f : Set R := { x | 4 < x ∧ x < 10 }
noncomputable def solution_set_positive_g : Set R := { x | 2 < x ∧ x < 5 }

theorem solution_set_product_positive :
  is_odd f →
  is_odd g →
  (∀ x, f x > 0 ↔ x ∈ solution_set_positive_f) →
  (∀ x, g x > 0 ↔ x ∈ solution_set_positive_g) →
  { x | f x * g x > 0 } = { x | (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) } :=
by
  sorry

end solution_set_product_positive_l410_410926


namespace smallest_x_for_ffx_l410_410512

def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 3 : ℝ)

theorem smallest_x_for_ffx :
  ∀ x : ℝ, x >= 2 → (f(f(x)) = f((x - 2) ^ (1 / 3 : ℝ))) → x >= 10 := by sorry

end smallest_x_for_ffx_l410_410512


namespace number_of_incorrect_statements_l410_410781

def prop1 (P1 P2 L : Type) [plane P1] [plane P2] [line L] : Prop := 
  (parallel P1 L) ∧ (parallel P2 L) → (parallel P1 P2)

def prop2 (P1 P2 P3 : Type) [plane P1] [plane P2] [plane P3] : Prop :=
  (parallel P1 P3) ∧ (parallel P2 P3) → (parallel P1 P2)

def prop3 (P1 P2 P3 : Type) [plane P1] [plane P2] [plane P3] : Prop := 
  (parallel P1 P2) ∧ (intersects P3 P1) ∧ (intersects P3 P2) → 
  ∃ L1 L2 : line, (intersection_line P3 P1 = L1) ∧ (intersection_line P3 P2 = L2) ∧ (parallel L1 L2)

def prop4 (L : Type) (P1 P2 : Type) [line L] [plane P1] [plane P2] : Prop :=
  (parallel P1 P2) ∧ (intersects L P1) → (intersects L P2)

theorem number_of_incorrect_statements : 
  (¬ prop1 ∨ ¬ prop1) ∧ prop2 ∧ prop3 ∧ prop4 → 
  (count_false [prop1, prop2, prop3, prop4] = 1) :=
sorry

end number_of_incorrect_statements_l410_410781


namespace number_of_ways_to_distribute_balls_l410_410503

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l410_410503


namespace product_of_constants_l410_410392

theorem product_of_constants :
  ∃ (a b : ℤ), ab = 12 ∧
  let t := a + b in 
  (∏ (p : (Σ a b : ℤ, a * b = 12)), p.2.1 + p.2.2) = -531441 :=
by
  sorry

end product_of_constants_l410_410392


namespace arithmetic_sequence_terms_l410_410791

theorem arithmetic_sequence_terms (a d : ℤ) (h_a : a = 102) (h_d : d = -6) :
  let a_n := a + (n-1) * d
  (a_n = 0) -> n - 1 = 17 := 
by 
  intro h_eq
  have h1 : 0 = 102 + (n-1) * (-6), from h_eq
  have h2 : 0 = 102 - 6 * (n-1), by rw [h1, h_a, h_d]
  have h3 : 6 * (n-1) = 102, from eq.symm (by linarith)
  have h4 : n - 1 = 17, from int.eq_of_mul_eq_mul_left (by norm_num) h3
  exact h4


end arithmetic_sequence_terms_l410_410791


namespace hyperbola_problems_l410_410099

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * y^2 - 16 * x^2 = 144

-- Define conversion to standard form
def standard_form_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the lengths, coordinations, and eccentricity for the hyperbola
def hyperbola_properties : Prop :=
  (length_transverse_axis = 8) ∧
  (length_conjugate_axis = 6) ∧
  (foci = set.of (λ y, (0, y)) ∧ ((0, 5) ∨ (0, -5))) ∧
  (eccentricity = 5/4)

-- Define another hyperbola's equation passing through point P(6, 4)
def new_hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 48 - x^2 / 27 = 1

theorem hyperbola_problems :
  (∀ x y : ℝ, hyperbola_eq x y ↔ standard_form_eq x y) →
  hyperbola_properties →
  (new_hyperbola_eq 6 4) :=
by sorry

end hyperbola_problems_l410_410099


namespace number_of_ways_to_distribute_balls_l410_410501

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l410_410501


namespace magnitude_of_sum_l410_410109

variable {V : Type*} [InnerProductSpace ℝ V] (a b : V)

-- Given conditions
axiom ha : ∥ a ∥ = 1
axiom hb : ∥ b ∥ = 2
axiom hab : ∥ a - b ∥ = 2

-- Statement to prove
theorem magnitude_of_sum : ∥ a + b ∥ = Real.sqrt 6 := by
  sorry

end magnitude_of_sum_l410_410109


namespace matrix_proof_l410_410175

open Complex Matrix

noncomputable def matrix_proof_statement (p : ℕ) (hp : p ≥ 3) (A : Matrix (Fin p) (Fin p) ℂ) : Prop :=
  Tr A = 0 ∧ det (A - 1) ≠ 0 → A^p ≠ 1

theorem matrix_proof (p : ℕ) (hp : Nat.Prime p) (h : p ≥ 3) (A : Matrix (Fin p) (Fin p) ℂ) :
  Tr A = 0 ∧ det (A - 1) ≠ 0 → A^p ≠ 1 := by
  sorry

end matrix_proof_l410_410175


namespace problem_1_problem_2_l410_410435

-- Definition of the sequence a_n, sum S_n and the equation
def seq (a : ℕ → ℝ) := ∀ n : ℕ, a n > 0
def sum_seq (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = ∑ i in range n, a i
def equation (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) := ∀ n : ℕ, λ ≠ 0 → 
  a n * S (n + 1) - a (n + 1) * S n + a n - a (n + 1) = λ * a n * a (n + 1)

-- Problem 1: Given that a_1, a_2, a_3 form a geometric sequence, prove that λ = 1
theorem problem_1 (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) (q : ℝ) 
  (h_geom : a 2 = q * a 1 ∧ a 3 = q * a 2) 
  (h_seq : seq a) 
  (h_sum : sum_seq a S) 
  (h_eq : equation a S λ) : 
  λ = 1 := 
sorry

-- Problem 2: Given λ = 1/2 and the sequence condition, find S_n
noncomputable def S_val (n : ℕ) := (n * (n + 5)) / 6

theorem problem_2 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_seq : seq a) 
  (h_sum : sum_seq a S) 
  (h_eq : equation a S (1/2)) : 
  ∀ n : ℕ, S n = S_val n := 
sorry

end problem_1_problem_2_l410_410435


namespace evaluate_series_l410_410377

theorem evaluate_series : 
  let T := ∑ k in Finset.range 50, (-1)^k * Nat.choose 99 (2*k+1) in
  T = -2^49 :=
by
  sorry

end evaluate_series_l410_410377


namespace gift_card_amount_l410_410201

theorem gift_card_amount (original_price final_price : ℝ) 
  (discount1 discount2 : ℝ) 
  (discounted_price1 discounted_price2 : ℝ) :
  original_price = 2000 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  discounted_price1 = original_price - (discount1 * original_price) →
  discounted_price2 = discounted_price1 - (discount2 * discounted_price1) →
  final_price = 1330 →
  discounted_price2 - final_price = 200 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end gift_card_amount_l410_410201


namespace number_of_combinations_l410_410386

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations_l410_410386


namespace problem_statement_l410_410124

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2002 + a ^ 2001 = 2 := 
by 
  sorry

end problem_statement_l410_410124


namespace smallest_n_satisfies_conditions_l410_410698

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410698


namespace possible_remainder_degrees_l410_410284

theorem possible_remainder_degrees (f : Polynomial ℝ) :
  ∃ d, (degree (X ^ 7 - 2 * X ^ 3 + X - 8) = 7) ∧ 
  (0 ≤ d ∧ d < 7) → ∃ r, degree r = d :=
begin
  sorry
end

end possible_remainder_degrees_l410_410284


namespace integral_sqrt_minus_x_l410_410793

theorem integral_sqrt_minus_x :
  ∫ x in 0..2, (sqrt (4 - x^2) - 2 * x) = Real.pi - 4 := by
  sorry

end integral_sqrt_minus_x_l410_410793


namespace line_AB_equation_l410_410418

theorem line_AB_equation : 
  ∀ (M P : Point) (l : Line), 
    Circle M (x^2 + y^2 - 2*x - 2*y - 2 = 0) ∧ 
    Line l (2*x + y + 2 = 0) ∧ 
    P ∈ l ∧
    (∃ A B : Point, Tangent A P ∧ Tangent B P ∧ 
                    A ∈ Circle M ∧ B ∈ Circle M ∧
                    (∃ AB : Line, Minimize |PM| * |AB|)) → 
  is_equation_of_line AB (2*x + y + 1 = 0) := 
sorry

end line_AB_equation_l410_410418


namespace plane_divided_into_four_regions_l410_410850

/-- Given the equations y = 3 * x and y = (1/3) * x. 
    The two lines divide the plane into exactly 4 regions. -/
theorem plane_divided_into_four_regions :
  let line1 := {p : ℝ × ℝ | p.2 = 3 * p.1}
  let line2 := {p : ℝ × ℝ | p.2 = (1/3) * p.1}
  (card (plane \ (line1 ∪ line2))) = 4 :=
sorry

end plane_divided_into_four_regions_l410_410850


namespace tan_sub_pi_over_4_l410_410455

theorem tan_sub_pi_over_4 (α : ℝ) (hCond1 : α ∈ Ioo (3 * π / 2) (2 * π)) 
  (hCond2 : 2 * sin (2 * α) + 1 = cos (2 * α)) : tan (α - π / 4) = 3 :=
by
  sorry

end tan_sub_pi_over_4_l410_410455


namespace profit_share_difference_l410_410291

noncomputable def ratio (x y : ℕ) : ℕ := x / Nat.gcd x y

def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1900
def total_parts : ℕ := 15  -- Sum of the ratio parts (4 for A, 5 for B, 6 for C)
def part_amount : ℕ := profit_share_B / 5  -- 5 parts of B

def profit_share_A : ℕ := 4 * part_amount
def profit_share_C : ℕ := 6 * part_amount

theorem profit_share_difference :
  (profit_share_C - profit_share_A) = 760 := by
  sorry

end profit_share_difference_l410_410291


namespace find_y_l410_410378

theorem find_y (y : ℝ) (h : (y^2 - 11 * y + 24) / (y - 1) + (4 * y^2 + 20 * y - 25) / (4*y - 5) = 5) :
  y = 3 ∨ y = 4 :=
sorry

end find_y_l410_410378


namespace arithmetic_sequence_properties_l410_410916

variable {a : ℕ → ℕ}
variable {n : ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_properties 
  (a_3_eq_7 : a 3 = 7)
  (a_5_plus_a_7_eq_26 : a 5 + a 7 = 26) :
  (∃ a1 d, (a 1 = a1) ∧ (∀ n, a n = a1 + (n - 1) * d) ∧ d = 2) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ S_n, S_n = n^2 + 2 * n) ∧ 
  ∀ T_n n, (∃ b : (ℕ → ℕ) → ℕ → ℕ, b a n = 1 / (a n ^ 2 - 1)) 
  → T_n = n / (4 * (n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l410_410916


namespace simplify_exponent_fraction_l410_410280

theorem simplify_exponent_fraction : (3 ^ 2015 + 3 ^ 2013) / (3 ^ 2015 - 3 ^ 2013) = 5 / 4 := by
  sorry

end simplify_exponent_fraction_l410_410280


namespace even_function_value_l410_410428

-- Define the function condition
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the main problem with given conditions
theorem even_function_value (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : ∀ x : ℝ, x < 0 → f x = x * (x + 1)) 
  (x : ℝ) (hx : x > 0) : f x = x * (x - 1) :=
  sorry

end even_function_value_l410_410428


namespace vasya_can_place_checkers_l410_410204

theorem vasya_can_place_checkers 
(board : ℕ → ℕ → Bool)
(placed_checkers : ∀ i j, board i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50)
: ∃ (new_checkers : ℕ → ℕ → Bool), (∀ i j, new_checkers i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
       (∑ i j, if new_checkers i j then 1 else 0 ≤ 99) ∧ 
       (∀ i, (∑ j, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) ∧ 
       (∀ j, (∑ i, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) := 
sorry

end vasya_can_place_checkers_l410_410204


namespace flour_usage_percentage_l410_410310

def total_flour : ℚ := 2.5
def recipe_flour : ℚ := 4 / 3
def recipe_count : ℚ := 1.5

theorem flour_usage_percentage :
  let used_flour := recipe_count * recipe_flour in
  let percentage := (used_flour / total_flour) * 100 in
  percentage = 80 := by
  sorry

end flour_usage_percentage_l410_410310


namespace nancy_total_spent_l410_410739

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l410_410739


namespace fib_subsequence_fib_l410_410595

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fib_subsequence_fib (p : ℕ) (hp : p > 0) :
  ∀ n : ℕ, fibonacci ((n - 1) * p) + fibonacci (n * p) = fibonacci ((n + 1) * p) := 
by
  sorry

end fib_subsequence_fib_l410_410595


namespace number_of_ways_to_put_7_balls_in_2_boxes_l410_410496

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l410_410496


namespace factorial_div_eq_l410_410838
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410838


namespace algorithm_correct_l410_410262

def algorithm_output (x : Int) : Int :=
  let y := Int.natAbs x
  (2 ^ y) - y

theorem algorithm_correct : 
  algorithm_output (-3) = 5 :=
  by sorry

end algorithm_correct_l410_410262


namespace arc_length_greater_than_diameter_l410_410213

noncomputable def circle (O : Point) (R : ℝ): Set Point := sorry

def diameter {O : Point} {R : ℝ}: ℝ := 2 * R

noncomputable def arc_length (A B : Point) (S2 : Set Point) : ℝ := sorry

theorem arc_length_greater_than_diameter
  (O : Point) (R : ℝ) (A B : Point)
  (S1 : Set Point) (S2 : Set Point)
  (h1 : A ∈ circle O R)
  (h2 : B ∈ circle O R)
  (h3 : ∀ P ∈ circle O R, ∃ Q ∈ S2, arc_length A Q + arc_length Q B = arc_length A B)
  (h4 : arc_length A B (circle O R) / 2 = arc_length A B S2) :
  arc_length A B S2 > diameter O R := sorry

end arc_length_greater_than_diameter_l410_410213


namespace factorial_div_combination_l410_410822

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410822


namespace symmetry_point_l410_410936

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x) + π / 6

theorem symmetry_point (x₀ y₀ : ℝ) (h_symm : ∀ x, f (2 * x₀ - x) = 2 * y₀ - f x)
  (h_range : x₀ > π / 2 ∧ x₀ < π) : x₀ + y₀ = π :=
begin
  sorry
end

end symmetry_point_l410_410936


namespace quadratic_inequality_solution_interval_l410_410367

theorem quadratic_inequality_solution_interval (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 8 * x + c < 0 → real.sq x) → 0 < c ∧ c < 8 :=
by
  sorry

end quadratic_inequality_solution_interval_l410_410367


namespace circles_tangent_internally_l410_410025

noncomputable def circle1 : set (ℝ × ℝ) := {p | let (x, y) := p in x^2 + y^2 - 8 * x + 6 * y + 16 = 0}
noncomputable def circle2 : set (ℝ × ℝ) := {p | let (x, y) := p in x^2 + y^2 = 64}

theorem circles_tangent_internally :
  let A := (4 : ℝ, -3 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let r := 3
  let R := 8
  let d := Real.sqrt ((4 - 0)^2 + (-3 - 0)^2)
  d = R - r :=
by {
  sorry
}

end circles_tangent_internally_l410_410025


namespace common_root_condition_rational_root_if_not_identical_l410_410303

theorem common_root_condition (p1 q1 p2 q2 : ℚ) :
  (∃ x : ℚ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) ↔
  (p1 - p2) * (p1 * q2 - p2 * q1) + (q1 - q2)^2 = 0 :=
by sorry

theorem rational_root_if_not_identical (p1 q1 p2 q2 : ℚ) (h : p1 ≠ p2 ∨ q1 ≠ q2) :
  (∃ x : ℚ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) → 
  (∀ x1 x2 : ℚ, (x1^2 + p1 * x1 + q1 = 0 → x2^2 + p2 * x2 + q2 = 0) → x1 ∈ ℚ ∧ x2 ∈ ℚ) :=
by sorry

end common_root_condition_rational_root_if_not_identical_l410_410303


namespace farthings_in_a_pfennig_l410_410906

theorem farthings_in_a_pfennig (x : ℕ) (h : 54 - 2 * x = 7 * x) : x = 6 :=
by
  sorry

end farthings_in_a_pfennig_l410_410906


namespace problem_statement_l410_410087

variable {n : ℕ} (S : ℕ → ℕ) (a : ℕ → ℤ)

-- Problem conditions
def condition1 : Prop := S 2 = 11
def condition2 : Prop := S 5 = 50

-- Define the arithmetic sequence sum
def Sn (n : ℕ) : Prop := S n = n * (a 1 + a (n + 1)) / 2

-- Define the specific terms of the arithmetic sequence
def a_terms (S : ℕ → ℕ) (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, a 2 = a 1 + d ∧ S 2 = 2 * a 1 + d ∧
  S 5 = 5 * a 1 + 10 * d

-- Define the direction vector for the points P and Q
def direction_vector (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, 
  let an := λ n, a1 + (n - 1) * d
  in let P := (n, an n) let Q := (n + 2, an (n + 2))
  in (Q.1 - P.1, Q.2 - P.2) ∈ { (x, y) | ∃ k : ℤ, (x, y) = (-1 * k, -3 * k) }

theorem problem_statement (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℤ) :
  condition1 S ∧ condition2 S ∧ a_terms S a → direction_vector a :=
by
  sorry

end problem_statement_l410_410087


namespace domain_of_function_l410_410243

theorem domain_of_function :
  {x : ℝ | 0 ≤ x ∧ x < 2} = set.Ico 0 2 :=
by sorry

end domain_of_function_l410_410243


namespace sales_tax_difference_l410_410381

theorem sales_tax_difference :
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  difference = 0.375 :=
by
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  exact sorry

end sales_tax_difference_l410_410381


namespace AM_GM_Inequality_l410_410728

theorem AM_GM_Inequality (n : ℕ) (a : Fin n → ℝ) (h1 : ∀ i, a i > 0) (h2 : (∀ i : Fin n, a i) = 1) :
  ∏ i, (1 + a i) ≥ 2^n := 
by
  sorry

end AM_GM_Inequality_l410_410728


namespace vasya_can_place_99_checkers_l410_410211

theorem vasya_can_place_99_checkers (board : Fin 50 × Fin 50 → Prop) :
  (∀ i j, board i j → ¬ board i j) → ∃ (new_checkers : Fin 50 × Fin 50 → Prop),
  (∀ i j, board i j → ¬ new_checkers i j) ∧
  (∑ i, if new_checkers i then 1 else 0 ≤ 99) ∧
  (∀ i : Fin 50, even (∑ j, if new_checkers (i, j) then 1 else 0)) ∧
  (∀ j : Fin 50, even (∑ i, if new_checkers (i, j) then 1 else 0)) := sorry

end vasya_can_place_99_checkers_l410_410211


namespace smallest_n_45_l410_410685

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410685


namespace total_earnings_of_a_b_c_l410_410716

theorem total_earnings_of_a_b_c 
  (days_a days_b days_c : ℕ)
  (ratio_a ratio_b ratio_c : ℕ)
  (wage_c : ℕ) 
  (h_ratio : ratio_a * wage_c = 3 * (3 + 4 + 5))
  (h_ratio_a_b : ratio_b = 4 * wage_c / 5 * ratio_a / 60)
  (h_ratio_b_c : ratio_b = 4 * wage_c / 5 * ratio_c / 60):
  (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c) = 1480 := 
  by
    sorry

end total_earnings_of_a_b_c_l410_410716


namespace polygonal_line_revolutions_odd_l410_410724

-- Define the conditions
def closed_polygonal_line (P : Set (ℝ × ℝ)) (n : ℕ) : Prop :=
  -- P is a collection of n points in the plane forming a closed polygonal line
  ∃ (vertices : Fin n → ℝ × ℝ), (∀ i, vertices (i + 1) = vertices ((i + 1) % n)) ∧ P = Set.range vertices

def symmetric_with_respect_to (P : Set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
  ∀ p ∈ P, (2 • O - p) ∈ P

def not_on_line (O : ℝ × ℝ) (P : Set (ℝ × ℝ)) : Prop :=
  O ∉ P

-- Define the goal
theorem polygonal_line_revolutions_odd (P : Set (ℝ × ℝ)) (O : ℝ × ℝ) (n : ℕ) :
  closed_polygonal_line P n →
  symmetric_with_respect_to P O →
  not_on_line O P →
  (∑ i in Finset.range n, oriented_angle (vertex i) O (vertex ((i + 1) % n))) / (2 * π) % 2 = 1 :=
begin
  sorry  -- Proof to be added
end

end polygonal_line_revolutions_odd_l410_410724


namespace cost_price_of_article_l410_410722

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 :=
by sorry

end cost_price_of_article_l410_410722


namespace product_xyz_l410_410973

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l410_410973


namespace t_shirt_cost_l410_410578

theorem t_shirt_cost
    (packs_white : ℕ) (packs_blue : ℕ)
    (shirts_per_pack_white : ℕ) (shirts_per_pack_blue : ℕ)
    (total_spent : ℕ)
    (total_white : ℕ := packs_white * shirts_per_pack_white)
    (total_blue : ℕ := packs_blue * shirts_per_pack_blue)
    (total_shirts : ℕ := total_white + total_blue)
    (x : ℕ)
    (cost_equation : total_shirts * x = total_spent)
    (packs_white_val : packs_white = 2)
    (packs_blue_val : packs_blue = 4)
    (shirts_per_pack_white_val : shirts_per_pack_white = 5)
    (shirts_per_pack_blue_val : shirts_per_pack_blue = 3)
    (total_spent_val : total_spent = 66) :
    x = 3 :=
by 
  rw [packs_white_val, packs_blue_val, shirts_per_pack_white_val, shirts_per_pack_blue_val, total_spent_val] at cost_equation
  rw [total_white, total_blue, total_shirts] at cost_equation
  exact (Eq.trans cost_equation (Eq.refl (22 * 3)))

end t_shirt_cost_l410_410578


namespace increasing_interval_lemma_l410_410256

def f (x: ℝ) := sin (π / 3 - (1 / 2) * x)

theorem increasing_interval_lemma :
  ∀ x: ℝ,
    -2 * real.pi ≤ x ∧ x ≤ 2 * real.pi →
    (((-2 * real.pi ≤ x) ∧ (x ≤ - real.pi / 3)) ∨ ((5 * real.pi / 3 ≤ x) ∧ (x ≤ 2 * real.pi))) :=
by sorry

end increasing_interval_lemma_l410_410256


namespace fraction_of_states_l410_410229

theorem fraction_of_states (t s : ℕ) (ht : t = 22) (hs : s = 7) : s / t = 7 / 22 := by
  rw [ht, hs]
  exact rfl

end fraction_of_states_l410_410229


namespace slant_height_of_cone_l410_410717

theorem slant_height_of_cone
  (h : ℝ) (d : ℝ) (r : ℝ) (l : ℝ)
  (h_eq : h = 3)
  (d_eq : d = 8)
  (r_eq : r = d / 2)
  (l_eq : l = real.sqrt (h^2 + r^2)) :
  l = 5 :=
by
  sorry

end slant_height_of_cone_l410_410717


namespace exists_fixed_point_sequence_l410_410172

theorem exists_fixed_point_sequence (N : ℕ) (hN : 0 < N) (a : ℕ → ℕ)
  (ha_conditions : ∀ i < N, a i % 2^(N+1) ≠ 0) :
  ∃ M, ∀ n ≥ M, a n = a M :=
sorry

end exists_fixed_point_sequence_l410_410172


namespace inequality_solution_l410_410642

theorem inequality_solution (x : ℝ) (h : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1)) : x > 1 := 
sorry

end inequality_solution_l410_410642


namespace find_λ_l410_410457

variables {V : Type*} [inner_product_space ℝ V]

-- Unit vectors e1 and e2
variables (e1 e2 : V) (λ : ℝ)

-- The angle between e1 and e2 is π/3
def angle (e1 e2 : V) : ℝ := real.arccos ((⟪e1, e2⟫) / (‖e1‖ * ‖e2‖))

-- The vector a is defined as e1 + λ * e2
def a := e1 + λ • e2

-- The magnitude of a is √3 / 2
def mag_a := ‖a‖ = (sqrt 3) / 2

-- The problem statement
theorem find_λ (h1 : angle e1 e2 = π / 3)
  (h2 : ‖e1‖ = 1) 
  (h3 : ‖e2‖ = 1)
  (h4 : mag_a e1 e2 λ):
  λ = -1 / 2 :=
sorry

end find_λ_l410_410457


namespace xyz_product_value_l410_410987

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l410_410987


namespace number_of_months_l410_410290

def holidays_per_month : ℕ := 2
def total_holidays : ℕ := 24

theorem number_of_months (hpm : ℕ) (th : ℕ) : (th / hpm) = 12 :=
by
  have hpm_eq : holidays_per_month = hpm := rfl
  have th_eq : total_holidays = th := rfl
  rw [←hpm_eq, ←th_eq]
  have months : ℕ := th / hpm
  have months_eq : months = 12 := by
    unfold weeks
    simp [total_holidays, holidays_per_month]
    sorry
  exact months_eq

end number_of_months_l410_410290


namespace number_of_ways_to_put_7_balls_in_2_boxes_l410_410500

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l410_410500


namespace solve_congruence_l410_410225

theorem solve_congruence : ∃ n : ℕ, n < 29 ∧ 8 * n % 29 = 5 :=
by {
    use 26,
    split,
    { exact dec_trivial }, -- 26 < 29 is true
    { exact dec_trivial }, -- 8 * 26 % 29 = 5 is true
    sorry -- placeholder for detailed proof steps
}

end solve_congruence_l410_410225


namespace evaluate_piecewise_function_l410_410945

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else 3^x

theorem evaluate_piecewise_function : f (f (1 / 4)) = 1 / 9 :=
by
  sorry

end evaluate_piecewise_function_l410_410945


namespace xy_equals_x_l410_410521

theorem xy_equals_x (x y : ℝ) (h : y = 1) (h_eq : (6 ^ ((x + y) ^ 2)) / (6 ^ ((x - y) ^ 2)) = 1296) : x * y = x :=
by
  sorry

end xy_equals_x_l410_410521


namespace sin_B_in_right_triangle_l410_410532

theorem sin_B_in_right_triangle 
  {a b c : ℝ} -- Denote the sides of the triangle
  (h₁ : (a = 1/2 * c)) -- BC = 1/2 * AC
  (h₂ : b^2 = a^2 + c^2) -- Pythagorean theorem: AB^2 = BC^2 + AC^2
  (hc : c ≠ 0) -- Ensure AC is non-zero to avoid division by zero
  (ha : a ≠ 0) -- Ensure BC is non-zero to avoid division by zero
  (t : ∃ x : ℝ, x = a / c)
  : ( (2 * sqrt 5) / 5 = a / b) :=
sorry

end sin_B_in_right_triangle_l410_410532


namespace smallest_n_exists_l410_410360

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 3 else 3 * sequence (n - 1) - sequence (n - 2)

theorem smallest_n_exists : ∃ n : ℕ, 2 ^ 2016 ∣ sequence n ∧ ¬ 2 ^ 2017 ∣ sequence n ∧ n = 3 * 2 ^ 2013 := 
by 
  sorry

end smallest_n_exists_l410_410360


namespace vasya_can_place_99_checkers_l410_410209

/-- 
  Given a 50x50 board where Petya has placed some checkers, with at most one per cell,
  prove that Vasya can place at most 99 new checkers such that each row and each column 
  of the board contains an even number of checkers.
-/
theorem vasya_can_place_99_checkers (board : Fin 50 → Fin 50 → Bool) (h_checker_placed : ∀ i j, board i j = true → True) : 
  ∃ newCheckers : Fin 50 → Fin 50 → Bool,
    (∑ i j, if newCheckers i j then 1 else 0 ≤ 99) ∧
    (∀ i, (∑ j, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) ∧
    (∀ j, (∑ i, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) := 
by
  sorry

end vasya_can_place_99_checkers_l410_410209


namespace final_tv_price_l410_410116

-- Define constants
def original_price : ℝ := 1700
def first_discount_rate : ℝ := 0.10
def membership_discount_rate : ℝ := 0.05
def vat_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def recycling_fee : ℝ := 25

-- Define a function to calculate the final price
def final_price (original_price : ℝ) (first_discount_rate : ℝ) (membership_discount_rate : ℝ) (vat_rate : ℝ) (sales_tax_rate : ℝ) (recycling_fee : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - membership_discount_rate)
  let vat := price_after_second_discount * vat_rate
  let sales_tax := price_after_second_discount * sales_tax_rate
  let total_tax := vat + sales_tax
  let price_before_fee := price_after_second_discount + total_tax
  let final_price := price_before_fee + recycling_fee
  (Float.round (final_price * 100) / 100 : Float).toReal -- rounding to the nearest cent

-- Theorem statement
theorem final_tv_price : final_price original_price first_discount_rate membership_discount_rate vat_rate sales_tax_rate recycling_fee = 1812.81 := by
  sorry

end final_tv_price_l410_410116


namespace min_pos_period_y_equals_one_sub_two_sin_sq_2x_l410_410255

theorem min_pos_period_y_equals_one_sub_two_sin_sq_2x :
  ∀ x : ℝ, 
  (∃ T > 0, ∀ x : ℝ, 1 - 2 * (sin (2 * x))^2 = 1 - 2 * (sin (2 * (x + T)))^2) ↔ T = π / 2 :=
sorry

end min_pos_period_y_equals_one_sub_two_sin_sq_2x_l410_410255


namespace general_term_a_n_b_n_less_than_b_n_minus_1_b_n_less_than_7_l410_410767

-- Problem 1: General term of |a_n|
theorem general_term_a_n (n : ℕ) : 
  ∀ (a : ℕ → ℝ), 
    (a 1 = sqrt 2) ∧ 
    (∀ (k : ℕ), k ≥ 2 → a (k - 1) = sqrt (2 - sqrt (1 - (a k) ^ 2))) 
    → a n = 2 * sin (π / 2 ^ (n + 1)) :=
sorry

-- Problem 2: Prove b_n < b_{n-1}
theorem b_n_less_than_b_n_minus_1 (n : ℕ) : 
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ), 
    (a 1 = sqrt 2) ∧ 
    (∀ (k : ℕ), k ≥ 2 → a (k - 1) = sqrt (2 - sqrt (1 - (a k) ^ 2))) ∧ 
    (∀ (k : ℕ), b k = 2^(k + 1) * a k) 
    → b n < b (n - 1) :=
sorry

-- Problem 3: Prove b_n < 7
theorem b_n_less_than_7 (n : ℕ) : 
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ), 
    (a 1 = sqrt 2) ∧ 
    (∀ (k : ℕ), k ≥ 2 → a (k - 1) = sqrt (2 - sqrt (1 - (a k) ^ 2))) ∧ 
    (∀ (k : ℕ), b k = 2^(k + 1) * a k) 
    → b n < 7 :=
sorry

end general_term_a_n_b_n_less_than_b_n_minus_1_b_n_less_than_7_l410_410767


namespace find_m_in_interval_l410_410365

def sequence (x : ℕ → ℝ) (h0 : x 0 = 6) (hn : ∀ n, x (n + 1) = (x n ^ 2 + 6 * x n + 7) / (x n + 7)) : Prop :=
  ∃ m : ℕ, 4 + 1 / 2 ^ 25 ≥ x m ∧ 151 ≤ m ∧ m ≤ 300

theorem find_m_in_interval [noncomputable] : ∃ x : ℕ → ℝ, 
  sequence x 
    (by sorry) -- Proof that x 0 = 6
    (by sorry) -- Proof of the recursive sequence property
:= 
sorry

end find_m_in_interval_l410_410365


namespace distinct_equilateral_triangles_in_polygon_l410_410440

noncomputable def num_distinct_equilateral_triangles (P : Finset (Fin 10)) : Nat :=
  90

theorem distinct_equilateral_triangles_in_polygon (P : Finset (Fin 10)) :
  P.card = 10 →
  num_distinct_equilateral_triangles P = 90 :=
by
  intros
  sorry

end distinct_equilateral_triangles_in_polygon_l410_410440


namespace functional_equation_and_injectivity_l410_410042

noncomputable def f : ℤ → ℤ := sorry

theorem functional_equation_and_injectivity :
  (∀ m n : ℤ, f(f m + n) + f m = f n + f (3 * m) + 2014) ∧ 
  (∀ x y : ℤ, f x = f y → x = y) :=
sorry

end functional_equation_and_injectivity_l410_410042


namespace find_smallest_n_l410_410454

theorem find_smallest_n 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (h1 : a_n 1 + a_n 2 = 9 / 2)
    (h2 : S_n 4 = 45 / 8)
    (h3 : ∀ n, S_n n = (1 / 2) * n * (a_n 1 + a_n n)) :
    ∃ n : ℕ, a_n n < 1 / 10 ∧ ∀ m : ℕ, m < n → a_n m ≥ 1 / 10 := 
sorry

end find_smallest_n_l410_410454


namespace perpendicular_points_constant_l410_410152

open Real

-- Parametric equations of the curve C
def C (φ : ℝ) : ℝ × ℝ :=
  (3 * cos φ, 2 * sin φ)

-- Polar equations for points on the curve
def polar_equation (θ : ℝ) : ℝ :=
  real.sqrt(36 / (4 * cos(θ) ^ 2 + 9 * sin(θ) ^ 2))

-- Prove the given statement
theorem perpendicular_points_constant (φ1 φ2 : ℝ) 
  (h_eq : (C φ1).fst * (C φ2).fst + (C φ1).snd * (C φ2).snd = 0) : 
  1 / (polar_equation φ1) ^ 2 + 1 / (polar_equation (φ1 + π / 2)) ^ 2 = 13 / 36 :=
by 
  sorry

end perpendicular_points_constant_l410_410152


namespace find_V_y_l410_410222

-- Define the volumes and percentages given in the problem
def V_x : ℕ := 300
def percent_x : ℝ := 0.10
def percent_y : ℝ := 0.30
def desired_percent : ℝ := 0.22

-- Define the alcohol volumes in the respective solutions
def alcohol_x := percent_x * V_x
def total_volume (V_y : ℕ) := V_x + V_y
def desired_alcohol (V_y : ℕ) := desired_percent * (total_volume V_y)

-- Define our main statement
theorem find_V_y : ∃ (V_y : ℕ), alcohol_x + (percent_y * V_y) = desired_alcohol V_y ∧ V_y = 450 :=
by
  sorry

end find_V_y_l410_410222


namespace max_n_value_l410_410425

-- Define the arithmetic sequence
variable {a : ℕ → ℤ} (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)

-- Given conditions
variable (h1 : a 1 + a 3 + a 5 = 105)
variable (h2 : a 2 + a 4 + a 6 = 99)

-- Goal: Prove the maximum integer value of n is 10
theorem max_n_value (n : ℕ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) : n ≤ 10 → 
  (∀ m, (0 < m ∧ m ≤ n) → a (2 * m) ≥ 0) → n = 10 := 
sorry

end max_n_value_l410_410425


namespace usual_time_16_l410_410723

theorem usual_time_16 (S T : ℕ) (h1 : ∀ T, S / (0.4 * S) = (T + 24) / T) : T = 16 := 
by sorry

end usual_time_16_l410_410723


namespace insurance_premium_increases_l410_410507

-- Define the insurance premium and conditions
variable (current_premium future_premium : ℝ)
variable (has_accident : Bool)
variable (records_claims considers_risk increases_premium : Bool)

-- State the theorem
theorem insurance_premium_increases 
    (h1 : has_accident = true)
    (h2 : records_claims = true) 
    (h3 : considers_risk = true) 
    (h4 : increases_premium = true) 
    (current_premium < future_premium) 
    : future_premium > current_premium :=
by 
  sorry

end insurance_premium_increases_l410_410507


namespace ellipse_equation_l410_410069

-- Define the problem conditions
def is_ellipse (a b : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1
def line_eq (x y : ℝ) : Prop := y = x + 3
def one_intersection (a b : ℝ) : Prop := ∃ xy : ℝ, 
  (line_eq xy (xy + 3) ∧ is_ellipse a b)

-- Define the eccentricity condition
def eccentricity_of (a b : ℝ) (e : ℝ) : Prop := e = (Real.sqrt (a^2 - b^2)) / a

-- Build the final theorem
theorem ellipse_equation (a b : ℝ) (h1 : 1 > sqrt 2 > a) 
  (h2 : sqrt 2 > b > 0) (h3: eccentricity_of a b (sqrt 5 / 5)) (h4: one_intersection a b) :
  ∃ eq : Prop, eq = (λ x y : ℝ, (x^2 / 5) + (y^2 / 4) = 1) :=
by 
  exists (λ x y : ℝ, (x^2 / 5) + (y^2 / 4) = 1)
  sorry

end ellipse_equation_l410_410069


namespace heavy_rain_duration_l410_410603

-- Define the conditions as variables and constants
def initial_volume := 100 -- Initial volume in liters
def final_volume := 280   -- Final volume in liters
def flow_rate := 2        -- Flow rate in liters per minute

-- Define the duration query as a theorem to be proved
theorem heavy_rain_duration : 
  (final_volume - initial_volume) / flow_rate = 90 := 
by
  sorry

end heavy_rain_duration_l410_410603


namespace trader_cheats_l410_410345

theorem trader_cheats (c: ℝ) (x: ℝ):
  ∀ (claimed_weight actual_weight profit_percentage: ℝ),
  claimed_weight = 1100 →
  profit_percentage = 65 →
  actual_weight = 666.67 →
  (claimed_weight - actual_weight) / actual_weight * 100 = profit_percentage :=
  
begin
  intros,
  sorry
end

end trader_cheats_l410_410345


namespace constant_expression_orthocenter_l410_410177

-- Define the problem in terms of Lean types and constants
variables (d e f S : Real)
variables (D E F Q H : Point)  -- Points in the geometry

-- Define the conditions
def ortho_center (H : Point) (D E F : Point) : Prop := sorry
def circumcircle_contains_point (Q : Point) : Prop := sorry
def side_lengths (D E F : Point) (d e f : Real) : Prop := sorry
def circumradius (S : Real) : Prop := sorry

-- Define the theorem statement
theorem constant_expression_orthocenter (H D E F Q : Point)
  (h_ortho : ortho_center H D E F)
  (h_circumcircle : circumcircle_contains_point Q)
  (h_side_lengths : side_lengths D E F d e f)
  (h_circumradius : circumradius S) :
  QD^2 + QE^2 + QF^2 - QH^2 = d^2 + e^2 + f^2 - 4S^2 := sorry

end constant_expression_orthocenter_l410_410177


namespace triangle_area_l410_410276

theorem triangle_area (AC BK BC : ℝ) (hAC : AC = 10) (hBK : BK = 8) (hBC : BC = 15)
    (hAK : ∃ K : ℝ, K = BK ∧ ∃ AK : ℝ, AK = real.sqrt (AC^2 - K^2) ∧ AK > 0) :
    ∃ area : ℝ, area = (15 * real.sqrt 51) / 2 :=
by
  sorry

end triangle_area_l410_410276


namespace common_difference_of_arithmetic_sequence_l410_410156

variable {a : ℕ → ℝ} (a2 a5 : ℝ)
variable (h1 : a 2 = 9) (h2 : a 5 = 33)

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l410_410156


namespace students_left_in_classroom_l410_410140

theorem students_left_in_classroom :
  ∀ (total_students : ℕ), total_students = 50 →
  (painting_ratio playing_ratio : ℚ), painting_ratio = 3/5 → playing_ratio = 1/5 →
  (students_painting students_playing : ℕ), students_painting = (painting_ratio * total_students).toNat → students_playing = (playing_ratio * total_students).toNat →
  (students_away students_left : ℕ), students_away = students_painting + students_playing → students_left = total_students - students_away →
  students_left = 10 :=
by
  intros total_students h_total students_painting_ratio h_painting students_playing_ratio h_playing students_painting h_students_painting students_playing h_students_playing students_away h_students_away students_left h_students_left
  sorry

end students_left_in_classroom_l410_410140


namespace number_of_ways_to_put_7_balls_in_2_boxes_l410_410497

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l410_410497


namespace parabola_line_area_l410_410622

noncomputable def parabolaArea : ℝ := 
  ∫ x in ↑(-2) .. ↑4, (x + 4 - (1 / 2) * x^2)

theorem parabola_line_area :
  parabolaArea = 18 :=
by
  sorry

end parabola_line_area_l410_410622


namespace Gwen_books_total_l410_410299

theorem Gwen_books_total 
  (shelves_mystery shelves_picture books_per_shelf : ℕ) 
  (h1 : shelves_mystery = 5) 
  (h2 : shelves_picture = 3) 
  (h3 : books_per_shelf = 4) 
  : (shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf) = 32 :=
  by 
  rw [h1, h2, h3]
  sorry

end Gwen_books_total_l410_410299


namespace max_sin_cos_expr_l410_410894

theorem max_sin_cos_expr (x y z : ℝ) :
  let expr := (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))
  in ∀ (x y z : ℝ), expr ≤ 4.5 :=
sorry

end max_sin_cos_expr_l410_410894


namespace find_a_l410_410447

noncomputable def value_of_a (a : ℝ) : Prop :=
  let f := λ x : ℝ, a * x^2 in
  let f_prime := deriv f in
  let slope_tangent := f_prime 2 in
  let slope_given_line := 4 in  -- Slope of the given line 4x - y + 4 = 0
  slope_tangent = -1 / slope_given_line ∧ slope_tangent = 4 * a

theorem find_a (a : ℝ) : value_of_a a → a = -1 / 16 :=
begin
  intro h,
  sorry
end

end find_a_l410_410447


namespace solve_eq_l410_410472

-- Parametric equation of line l
def line_l_param (t : ℝ) (a : ℝ) : (ℝ × ℝ) :=
  (4 * t, 4 * t + a)

-- Standard equation of line l
def line_l_standard (x y a : ℝ) : Prop :=
  x - y + a = 0

-- Polar to Cartesian conversion for the circle
def circle_cartesian (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y + 2) ^ 2 = 8

-- Condition for exactly three points at a distance of sqrt(2) from the line
def distance_condition (a : ℝ) : Prop :=
  a = -6 ∨ a = -2

-- Main theorem to be proven
theorem solve_eq (a : ℝ) : ∃ x y, line_l_standard x y a ∧ circle_cartesian x y ∧ distance_condition a :=
begin
  sorry
end

end solve_eq_l410_410472


namespace definite_integral_sin_l410_410998

theorem definite_integral_sin (a : ℝ) (h : (coeff (x : ℝ) 9 ((x^2 - 1/(a * x))^9)) = -21/2) : 
  ∫ x in 0..a, sin x = 1 - cos 2 :=
by 
  sorry  -- proof steps not required

end definite_integral_sin_l410_410998


namespace no_sport_members_count_l410_410147

theorem no_sport_members_count (n B T B_and_T : ℕ) (h1 : n = 27) (h2 : B = 17) (h3 : T = 19) (h4 : B_and_T = 11) : 
  n - (B + T - B_and_T) = 2 :=
by
  sorry

end no_sport_members_count_l410_410147


namespace polar_line_equation_l410_410543

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem polar_line_equation (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) :
  let P := polar_to_rectangular ρ θ in
  (∃ x, P = (x, 1) ∧ ∀ ρ θ, ρ * Real.sin θ = 1) :=
begin
  sorry
end

end polar_line_equation_l410_410543


namespace factorial_quotient_l410_410818

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410818


namespace vasya_can_place_checkers_l410_410206

theorem vasya_can_place_checkers 
(board : ℕ → ℕ → Bool)
(placed_checkers : ∀ i j, board i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50)
: ∃ (new_checkers : ℕ → ℕ → Bool), (∀ i j, new_checkers i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
       (∑ i j, if new_checkers i j then 1 else 0 ≤ 99) ∧ 
       (∀ i, (∑ j, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) ∧ 
       (∀ j, (∑ i, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) := 
sorry

end vasya_can_place_checkers_l410_410206


namespace percent_sales_other_l410_410232

theorem percent_sales_other (percent_notebooks : ℕ) (percent_markers : ℕ) (h1 : percent_notebooks = 42) (h2 : percent_markers = 26) :
    100 - (percent_notebooks + percent_markers) = 32 := by
  sorry

end percent_sales_other_l410_410232


namespace exponential_difference_l410_410965

theorem exponential_difference (f : ℕ → ℕ) (x : ℕ) (h : f x = 3^x) : f (x + 2) - f x = 8 * f x :=
by sorry

end exponential_difference_l410_410965


namespace more_squares_with_17th_digit_7_than_8_l410_410782

theorem more_squares_with_17th_digit_7_than_8 :
  let S : ℕ → ℕ → ℕ := λ f (d : ℕ), (f * f) % 10^(d+1) / 10^d
  (count (λ n, n ≤ 10^20 ∧ S n 16 = 7 ∧ is_square n) (range (10^10 + 1))) >
  (count (λ n, n ≤ 10^20 ∧ S n 16 = 8 ∧ is_square n) (range (10^10 + 1))) := 
sorry

end more_squares_with_17th_digit_7_than_8_l410_410782


namespace intersection_A_B_l410_410075

noncomputable def A := set.Ioo (-1 : ℝ) 1
noncomputable def B := set.Icc 0 3

theorem intersection_A_B : A ∩ B = set.Ico 0 1 :=
by {
  sorry
}

end intersection_A_B_l410_410075


namespace max_value_sine_cosine_expression_l410_410873

theorem max_value_sine_cosine_expression (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sine_cosine_expression_l410_410873


namespace wave_number_probability_l410_410912

def is_wave_number (a b c d e : ℕ) : Prop :=
  a < b ∧ b > c ∧ c < d ∧ d > e

def no_repeats (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e

theorem wave_number_probability :
  let digits := {1, 2, 3, 4, 5}
  let total_count := 5 * 4 * 3 * 2 * 1
  let wave_number_count := 16
  (wave_number_count : ℚ) / total_count = 2 / 15 :=
by {
  sorry
}

end wave_number_probability_l410_410912


namespace calculate_f2013_fneg2014_l410_410061

noncomputable def f : ℝ → ℝ
| x := if x ∈ set.Icc 0 2 then real.exp x - 1 else sorry

-- conditions: 1) f(x) is an odd function, 2) f(x+2) = f(x) for x ≥ 0, 3) f(x) = e^x - 1 for x ∈ [0, 2]
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic (f : ℝ → ℝ) : ∀ x, x ≥ 0 → f (x + 2) = f x 
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = real.exp x - 1

-- goal: prove that f(2013) + f(-2014) = e - 1
theorem calculate_f2013_fneg2014 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_periodic : ∀ x, x ≥ 0 → f (x + 2) = f x) 
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = real.exp x - 1) 
  : f 2013 + f (-2014) = real.exp 1 - 1 := 
sorry

end calculate_f2013_fneg2014_l410_410061


namespace simplify_expression_l410_410221

theorem simplify_expression : 
  1 / (1 / ((1 / 2) ^ (-1)) + 1 / ((1 / 2) ^ (-2)) + 1 / ((1 / 2) ^ (-3))) = 8 / 7 := 
  sorry

end simplify_expression_l410_410221


namespace value_of_d_minus_2_times_e_minus_2_l410_410183

-- Define the given quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- State that our equation is 3x² + 4x - 7 = 0
def given_quadratic_eq := quadratic_eq 3 4 (-7)

-- Using Vieta's formulas
def sum_roots (d e : ℝ) := d + e = - (4 / 3)
def product_roots (d e : ℝ) := d * e = - (7 / 3)

-- Prove the required value
theorem value_of_d_minus_2_times_e_minus_2 (d e : ℝ) (h1 : sum_roots d e) (h2 : product_roots d e) :
  (d - 2) * (e - 2) = 13 / 3 := by
  sorry

end value_of_d_minus_2_times_e_minus_2_l410_410183


namespace probability_of_interval_l410_410281

noncomputable def geometric_probability (a b m n : ℝ) : ℝ :=
  if h : b - a ≠ 0 then (n - m) / (b - a) else 0

theorem probability_of_interval (x : ℝ) (h₁ : x ∈ Icc (-1 : ℝ) (2 : ℝ)) :
  geometric_probability (-1) 2 0 1 = 1 / 3 :=
by
  sorry

end probability_of_interval_l410_410281


namespace line_plane_intersection_l410_410515

variables {Point : Type} {Plane : Type} [linear_ordered_field Plane] [inhabited Point]

-- Definitions of line, plane, common_point, parallel, contained_in, intersects
def line (l : Type) : Type := sorry
def plane (α : Type) : Type := sorry
def common_point (l : Type) (α : Type) (p : Point) : Prop := sorry
def parallel (l : Type) (α : Type) : Prop := sorry
def contained_in (l : Type) (α : Type) : Prop := sorry
def intersects (l : Type) (α : Type) : Prop := sorry

-- The main problem statement
theorem line_plane_intersection (l : Type) (α : Type) (p : Point) :
  common_point l α p → (contained_in l α ∨ intersects l α) :=
by sorry

end line_plane_intersection_l410_410515


namespace dot_product_eq_neg_one_l410_410959

variables (m n : ℝ^3) -- Assume vectors in 3-dimensional real space
variables (theta : ℝ)
variables (h1 : ‖m‖ = 1) (h2 : ‖n‖ = sqrt 2) (h3 : theta = 3*real.pi / 4)

theorem dot_product_eq_neg_one (h_angle : angle m n = theta) : dot_product m n = -1 :=
by {
  sorry -- Placeholder for the proof
}

end dot_product_eq_neg_one_l410_410959


namespace pyramid_surface_area_l410_410342

-- Definition of a right square pyramid with specific dimensions
def right_square_pyramid (side height : ℝ) :=
  ∃ peak base_corner : EuclideanSpace ℝ (Fin 3), 
    dist base_corner peak = height ∧ 
    ∃ other_corner₁ other_corner₂ other_corner₃ : EuclideanSpace ℝ (Fin 3), 
    dist base_corner other_corner₁ = side ∧ 
    dist base_corner other_corner₂ = side ∧ 
    dist base_corner other_corner₃ = side ∧
    dist other_corner₁ other_corner₂ = side ∧
    dist other_corner₂ other_corner₃ = side ∧
    dist other_corner₃ other_corner₁ = side

-- Statement of the problem
theorem pyramid_surface_area : 
  ∀ (side height : ℝ), side = 8 → height = 10 → 
  right_square_pyramid side height →
  total_surface_area = 64 + 10 * real.sqrt 164 + 8 * real.sqrt 228 := 
by
  intros side height hside hheight hpyramid
  sorry

end pyramid_surface_area_l410_410342


namespace stock_percent_change_l410_410594

theorem stock_percent_change (x : ℝ) :
  let first_day_value := 0.75 * x,
      second_day_value := 1.35 * first_day_value in
  (second_day_value - x) / x * 100 = 1.25 :=
by
  let first_day_value := 0.75 * x
  let second_day_value := 1.35 * first_day_value
  sorry

end stock_percent_change_l410_410594


namespace sum_of_possible_digit_counts_in_base_2_l410_410323

theorem sum_of_possible_digit_counts_in_base_2 :
  (∑ d in ({13, 14, 15} : Finset ℕ), d) = 42 :=
by 
  sorry

end sum_of_possible_digit_counts_in_base_2_l410_410323


namespace prove_cos_eq_half_range_f_A_l410_410110

section Problem1

variables {x : ℝ}

def m : ℝ × ℝ := (sin (x / 4), cos (x / 4))
def n : ℝ × ℝ := (sqrt 3 * cos (x / 4), cos (x / 4))
def f (x : ℝ) : ℝ := m.1 * n.1 + m.2 * n.2

-- Given: f(x) = 1
-- Prove: cos (x + π / 3) = 1 / 2

theorem prove_cos_eq_half (h : f x = 1) : cos (x + π / 3) = 1 / 2 := sorry

end Problem1

section Problem2

variables {A B C a b c : ℝ}

-- Triangle ABC with sides opposite to angles a, b, c respectively
-- Given: (2a - c) * cos B = b * cos C
-- Prove: The range of values for f(A) is (1, 3/2)

hypothesis h : (2 * a - c) * cos B = b * cos C

def f_A (A : ℝ) : ℝ := sin (A / 2 + π / 6) + 1 / 2

theorem range_f_A : Set.Ioo 1 (3 / 2) = {y : ℝ | ∃ A, f_A A = y} := sorry

end Problem2

end prove_cos_eq_half_range_f_A_l410_410110


namespace sum_possible_values_base2_l410_410313

theorem sum_possible_values_base2 (d_8 : ℕ) (h1 : d_8 >= 8^4) (h2 : d_8 < 8^5) :
  d_8.bits ≠ 0 → (d_8.bits.card = 13 ∨ d_8.bits.card = 14 ∨ d_8.bits.card = 15) →
  d_8.bits.card = 13 + d_8.bits.card = 14 + d_8.bits.card = 15 := sorry

end sum_possible_values_base2_l410_410313


namespace smallest_n_l410_410680

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410680


namespace max_value_sin_cos_expression_l410_410891

theorem max_value_sin_cos_expression (x y z : ℝ) 
  (h1 : sin x^2 + cos x^2 = 1)
  (h2 : sin (2 * y)^2 + cos (2 * y)^2 = 1)
  (h3 : sin (3 * z)^2 + cos (3 * z)^2 = 1) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l410_410891


namespace trajectory_of_M_max_min_area_l410_410448

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem trajectory_of_M
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ := (4, 0))
  (H : (P.1 + 2) ^ 2 + P.2 ^ 2 = 4)
  (M := midpoint P Q) :
  (M.1 - 1) ^ 2 + M.2 ^ 2 = 1 :=
sorry

noncomputable def area (k1 k2 : ℝ) : ℝ :=
  18 / (k1 - k2)

theorem max_min_area
  (t : ℝ)
  (Ht : -5 ≤ t ∧ t ≤ -2)
  (k1 := (1 - t^2) / (2 * t))
  (k2 := (1 - (t + 6)^2) / (2 * (t + 6)))
  (S := area k1 k2) :
  27 / 4 ≤ S ∧ S ≤ 15 / 2 :=
sorry

end trajectory_of_M_max_min_area_l410_410448


namespace arm_wrestling_qualification_l410_410144

theorem arm_wrestling_qualification :
  ∀ (n : Nat),
  n = 896 →
  ∃ k : Nat, 
  k = 10 ∧ 
  ∀ (a w : Nat), 
  a = 1 ∨ a = 0 → 
  w = 1 ∨ w = 0 → 
  (∀ (m p : Nat), 
  m = (if p % 2 = 0 then p / 2 else p / 2 + 1) → 
  (if m < n then m * (m + 1) else m = k)) :=
by
  -- proof omitted
  sorry

end arm_wrestling_qualification_l410_410144


namespace multiplicative_inverse_185_mod_341_l410_410013

theorem multiplicative_inverse_185_mod_341 :
  ∃ (b: ℕ), b ≡ 74466 [MOD 341] ∧ 185 * b ≡ 1 [MOD 341] :=
sorry

end multiplicative_inverse_185_mod_341_l410_410013


namespace slope_of_line_l410_410027

theorem slope_of_line : 
  (∃ x y : ℝ, (x / 4) + (y / 3) = 1) → 
  ∃ m : ℝ, m = -3 / 4 :=
by
  intro h,
  use -3 / 4,
  sorry

end slope_of_line_l410_410027


namespace part_a_part_b_l410_410715

-- Part (a)

theorem part_a (ABC : Triangle) (M : Point) (O : Point) (O_b : Point) (A C : Point) :
  is_intersection_of_angle_bisector_and_circumcircle M ABC B ∧
  is_incenter O ABC ∧
  is_excenter_tangent_to_side_ac O_b ABC AC →
  ∃ (circle : Circle), is_center M circle ∧ on_circle circle A ∧ on_circle circle C ∧ on_circle circle O ∧ on_circle circle O_b := 
sorry

-- Part (b)

theorem part_b (ABC : Triangle) (O : Point) :
  (∀ P, P ∈ circumcircle(B, O, C) ∧ P ∈ circumcircle(A, O, C) ∧ P ∈ circumcircle(A, B, O)) →
  is_incenter O ABC :=
sorry

end part_a_part_b_l410_410715


namespace vasya_can_place_99_checkers_l410_410210

theorem vasya_can_place_99_checkers (board : Fin 50 × Fin 50 → Prop) :
  (∀ i j, board i j → ¬ board i j) → ∃ (new_checkers : Fin 50 × Fin 50 → Prop),
  (∀ i j, board i j → ¬ new_checkers i j) ∧
  (∑ i, if new_checkers i then 1 else 0 ≤ 99) ∧
  (∀ i : Fin 50, even (∑ j, if new_checkers (i, j) then 1 else 0)) ∧
  (∀ j : Fin 50, even (∑ i, if new_checkers (i, j) then 1 else 0)) := sorry

end vasya_can_place_99_checkers_l410_410210


namespace sum_possible_values_base2_l410_410314

theorem sum_possible_values_base2 (d_8 : ℕ) (h1 : d_8 >= 8^4) (h2 : d_8 < 8^5) :
  d_8.bits ≠ 0 → (d_8.bits.card = 13 ∨ d_8.bits.card = 14 ∨ d_8.bits.card = 15) →
  d_8.bits.card = 13 + d_8.bits.card = 14 + d_8.bits.card = 15 := sorry

end sum_possible_values_base2_l410_410314


namespace line_intersects_hyperbola_l410_410133

theorem line_intersects_hyperbola (k : Real) : 
  (∃ x y : Real, y = k * x ∧ (x^2) / 9 - (y^2) / 4 = 1) ↔ (-2 / 3 < k ∧ k < 2 / 3) := 
sorry

end line_intersects_hyperbola_l410_410133


namespace xyz_product_value_l410_410986

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l410_410986


namespace integral_cos_4x_l410_410794

open Real

theorem integral_cos_4x : ∫ x in 0..(2 * π), (1 - 8 * x^2) * cos (4 * x) = -2 * π :=
by
  sorry

end integral_cos_4x_l410_410794


namespace remaining_credit_to_be_paid_l410_410582

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l410_410582


namespace max_rectangles_l410_410651

def num_matches : ℕ := 20

def valid_rectangle (l w : ℕ) : Prop :=
  2 * (l + w) ≤ num_matches ∧ (l - w).natAbs > 3

theorem max_rectangles : ∃ r, r = 6 ∧ ∀ l w : ℕ, valid_rectangle l w → l ≠ w → ∃! (l, w) ∈ [(5,1), (6,1), (7,1), (8,1), (6,2), (7,2)] :=
by
  sorry

end max_rectangles_l410_410651


namespace makenna_garden_larger_by_132_l410_410561

-- Define the dimensions of Karl's garden
def length_karl : ℕ := 22
def width_karl : ℕ := 50

-- Define the dimensions of Makenna's garden including the walking path
def length_makenna_total : ℕ := 30
def width_makenna_total : ℕ := 46
def walking_path_width : ℕ := 1

-- Define the area calculation functions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

-- Calculate the areas
def area_karl : ℕ := area length_karl width_karl
def area_makenna : ℕ := area (length_makenna_total - 2 * walking_path_width) (width_makenna_total - 2 * walking_path_width)

-- Define the theorem to prove
theorem makenna_garden_larger_by_132 :
  area_makenna = area_karl + 132 :=
by
  -- We skip the proof part
  sorry

end makenna_garden_larger_by_132_l410_410561


namespace factorial_division_identity_l410_410833

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410833


namespace sin_C_equals_four_fifths_l410_410137

noncomputable def triangle_sin_c (B : Real) (b c : Real) (sin : Real → Real) :=
  B = 30 ∧ b = 10 ∧ c = 16 → sin B = 1/2 → sin C = 4/5

theorem sin_C_equals_four_fifths (B : Real) (b c : Real) (sin : Real → Real) 
  (h1 : B = 30) (h2 : b = 10) (h3 : c = 16) (h4 : sin B = 1/2) 
  : sin C = 4/5 
:= sorry

end sin_C_equals_four_fifths_l410_410137


namespace shaded_area_of_floor_l410_410305

/-- A 16-foot by 20-foot floor is tiled with square tiles of size 2 feet by 2 feet. Each tile has a pattern consisting of four white quarter circles of radius 1 foot centered at each corner of the tile. The remaining portion of the tile is shaded. -/
theorem shaded_area_of_floor:
  let tile_area := 4
  let quarter_circle_area := π
  let shaded_area_per_tile := tile_area - quarter_circle_area
  let number_of_tiles := 80
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  in total_shaded_area = 320 - 80 * π :=
by sorry

end shaded_area_of_floor_l410_410305


namespace fixed_point_of_inverse_l410_410122

-- Define an odd function f on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f (x)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) := f (x + 1) - 2

-- Define the condition for a point to be on the inverse of a function
def inv_contains (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = f p.1

-- The theorem statement
theorem fixed_point_of_inverse (f : ℝ → ℝ) 
  (Hf_odd : odd_function f) :
  inv_contains (λ y => g f (y)) (-2, -1) :=
sorry

end fixed_point_of_inverse_l410_410122


namespace combination_formula_l410_410809

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410809


namespace max_profit_l410_410531

def fixed_cost : ℕ := 20000
def variable_cost (x : ℕ) : ℕ :=
  if x < 80 then
    (1 / 3 : ℝ) * (x ^ 2) + 2 * x
  else
    7 * x + 100 / x - 37

def selling_price : ℕ := 6

def revenue (x : ℕ) : ℕ := selling_price * x

def profit (x : ℕ) : ℕ :=
  if x < 80 then
    revenue x - (variable_cost x + fixed_cost)
  else
    revenue x - (variable_cost x + fixed_cost)

theorem max_profit :
  ∃ x : ℕ, profit x = 15000 ∧ ∀ y : ℕ, profit y ≤ 15000 :=
sorry

end max_profit_l410_410531


namespace sum_of_primitive_roots_mod_11_is_23_l410_410029

def is_primitive_root (a p : ℕ) : Prop :=
  let ord := order_of a % p in
  ord = p - 1

def sum_of_primitive_roots_modulo_11 : ℕ :=
  let roots := (Finset.filter (λ x, is_primitive_root x 11) (Finset.range 11)).val in
  roots.sum id

theorem sum_of_primitive_roots_mod_11_is_23 :
  sum_of_primitive_roots_modulo_11 = 23 := sorry

end sum_of_primitive_roots_mod_11_is_23_l410_410029


namespace isosceles_right_triangle_leg_length_l410_410637

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l410_410637


namespace percentage_increase_l410_410344

variables (P : ℝ) (buy_price : ℝ := 0.60 * P) (sell_price : ℝ := 1.08000000000000007 * P)

theorem percentage_increase (h: (0.60 : ℝ) * P = buy_price) (h1: (1.08000000000000007 : ℝ) * P = sell_price) :
  ((sell_price - buy_price) / buy_price) * 100 = 80.00000000000001 :=
  sorry

end percentage_increase_l410_410344


namespace factorial_division_identity_l410_410831

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410831


namespace length_of_second_train_is_90_l410_410306

-- Define the conditions based on the problem
def length_first_train := 360 -- in meters
def speed_first_train := 120 / 3.6 -- converted to m/s
def speed_second_train := 150 / 3.6 -- converted to m/s
def crossing_time := 6 -- in seconds

-- Define the relative speed calculation
def relative_speed := speed_first_train + speed_second_train

-- The total distance covered when the trains cross each other
def total_distance := relative_speed * crossing_time

-- Define the length of the second train
def length_second_train := total_distance - length_first_train

-- Main theorem to prove
theorem length_of_second_train_is_90 : length_second_train = 90 :=
by
  -- Leaving the proof as sorry
  sorry

end length_of_second_train_is_90_l410_410306


namespace range_a_l410_410460

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + 2 * x else Real.log (x + 1)

theorem range_a {a : ℝ} :
  (∀ x : ℝ, |f x| ≥ a * x) ↔ a ∈ set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end range_a_l410_410460


namespace convert_minus_330_deg_to_radians_l410_410361

def degrees_to_radians (deg : ℤ) : ℝ := deg * (Real.pi / 180)

theorem convert_minus_330_deg_to_radians : degrees_to_radians (-330) = -11 * Real.pi / 6 :=
by
  sorry

end convert_minus_330_deg_to_radians_l410_410361


namespace distinct_digits_solution_l410_410252

theorem distinct_digits_solution (A B C : ℕ)
  (h1 : A + B = 10)
  (h2 : C + A = 9)
  (h3 : B + C = 9)
  (h4 : A ≠ B)
  (h5 : B ≠ C)
  (h6 : C ≠ A)
  (h7 : 0 < A)
  (h8 : 0 < B)
  (h9 : 0 < C)
  : A = 1 ∧ B = 9 ∧ C = 8 := 
  by sorry

end distinct_digits_solution_l410_410252


namespace graph_not_pass_through_second_quadrant_l410_410251

theorem graph_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = 2 * x - 3 ∧ x < 0 ∧ y > 0 :=
by sorry

end graph_not_pass_through_second_quadrant_l410_410251


namespace sequence_98th_term_l410_410230

-- Definitions of the rules
def rule1 (n : ℕ) : ℕ := n * 9
def rule2 (n : ℕ) : ℕ := n / 2
def rule3 (n : ℕ) : ℕ := n - 5

-- Function to compute the next term in the sequence based on the current term
def next_term (n : ℕ) : ℕ :=
  if n < 10 then rule1 n
  else if n % 2 = 0 then rule2 n
  else rule3 n

-- Function to compute the nth term of the sequence starting with the initial term
def nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate next_term n start

-- Theorem to prove that the 98th term of the sequence starting at 98 is 27
theorem sequence_98th_term : nth_term 98 98 = 27 := by
  sorry

end sequence_98th_term_l410_410230


namespace number_of_pebbles_l410_410270

theorem number_of_pebbles (P : ℕ) : 
  (P * (1/4 : ℝ) + 3 * (1/2 : ℝ) + 2 * 2 = 7) → P = 6 := by
  sorry

end number_of_pebbles_l410_410270


namespace oatmeal_baggies_example_l410_410590

noncomputable def oatmeal_baggies (total_cookies : ℝ) (chocolate_chip_cookies : ℝ) (cookies_per_bag : ℝ) : ℕ := 
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ 

theorem oatmeal_baggies_example :
  oatmeal_baggies 41 13 9 = 3 := 
by 
  sorry

end oatmeal_baggies_example_l410_410590


namespace inequality_one_inequality_two_l410_410079

-- Definitions of the three positive real numbers and their sum of reciprocals squared is equal to 1
variables {a b c : ℝ}
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1)

-- First proof that (1/a + 1/b + 1/c) <= sqrt(3)
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (1 / a) + (1 / b) + (1 / c) ≤ Real.sqrt 3 :=
sorry

-- Second proof that (a^2/b^4) + (b^2/c^4) + (c^2/a^4) >= 1
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (a^2 / b^4) + (b^2 / c^4) + (c^2 / a^4) ≥ 1 :=
sorry

end inequality_one_inequality_two_l410_410079


namespace vasya_can_place_99_checkers_l410_410207

/-- 
  Given a 50x50 board where Petya has placed some checkers, with at most one per cell,
  prove that Vasya can place at most 99 new checkers such that each row and each column 
  of the board contains an even number of checkers.
-/
theorem vasya_can_place_99_checkers (board : Fin 50 → Fin 50 → Bool) (h_checker_placed : ∀ i j, board i j = true → True) : 
  ∃ newCheckers : Fin 50 → Fin 50 → Bool,
    (∑ i j, if newCheckers i j then 1 else 0 ≤ 99) ∧
    (∀ i, (∑ j, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) ∧
    (∀ j, (∑ i, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) := 
by
  sorry

end vasya_can_place_99_checkers_l410_410207


namespace distance_lines_eq_2_l410_410100

-- Define the first line in standard form
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the second line in standard form, established based on the parallel condition
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 14 = 0

-- Define the condition for parallel lines which gives m
axiom parallel_lines_condition : ∀ (x y : ℝ), (line1 x y) → (line2 x y)

-- Define the distance between two parallel lines formula
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / (Real.sqrt (a ^ 2 + b ^ 2))

-- Prove the distance between the given lines is 2
theorem distance_lines_eq_2 : distance_between_parallel_lines 3 4 (-3) 7 = 2 :=
by
  -- Details of proof are omitted, but would show how to manipulate and calculate distances
  sorry

end distance_lines_eq_2_l410_410100


namespace cost_of_mozzarella_cheese_l410_410761

-- Define the problem conditions as Lean definitions
def blendCostPerKg : ℝ := 696.05
def romanoCostPerKg : ℝ := 887.75
def weightMozzarella : ℝ := 19
def weightRomano : ℝ := 18.999999999999986  -- Practically the same as 19 in context
def totalWeight : ℝ := weightMozzarella + weightRomano

-- Define the expected result for the cost per kilogram of mozzarella cheese
def expectedMozzarellaCostPerKg : ℝ := 504.40

-- Theorem statement to verify the cost of mozzarella cheese
theorem cost_of_mozzarella_cheese :
  weightMozzarella * (expectedMozzarellaCostPerKg : ℝ) + weightRomano * romanoCostPerKg = totalWeight * blendCostPerKg := by
  sorry

end cost_of_mozzarella_cheese_l410_410761


namespace athletes_qualify_l410_410146

-- Definitions representing conditions
def athletes : ℕ := 896

def points_for_win : ℕ := 1
def points_for_loss : ℕ := 0

-- Consider we have a function to form pairs and simulate a round
def form_pairs : list ℕ → list (ℕ × ℕ) := sorry
def simulate_round (pairs : list (ℕ × ℕ)) : list ℕ := sorry

-- The condition that a player is eliminated after a second loss
def eliminated (losses : ℕ) : bool := losses ≥ 2

-- Let's assume we have a tuple where fst is the count of athletes without defeats
-- and snd is the count of athletes with one defeat
def tournament_progress : ℕ → ℕ × ℕ := sorry

-- The final condition proving 10 athletes remain in the qualification round
theorem athletes_qualify : (tournament_progress 10).fst + (tournament_progress 10).snd = 10 := sorry

end athletes_qualify_l410_410146


namespace factorial_div_combination_l410_410827

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410827


namespace equilateral_triangle_midpoints_l410_410777

/-- Given a hexagon ABCDEF with vertices on a circle of radius r,
     and sides AB, CD, EF having length r, 
     prove that the midpoints of BC, DE, FA form an equilateral triangle. -/
theorem equilateral_triangle_midpoints 
  (r : ℝ)
  (A B C D E F : EuclideanSpace ℝ (Fin 2))
  (h_circle : dist A (0 : EuclideanSpace ℝ (Fin 2)) = r)
  (h_AB : dist A B = r)
  (h_CD : dist C D = r)
  (h_EF : dist E F = r) :
  let P := (B + C) / 2
  let Q := (D + E) / 2
  let R := (F + A) / 2
  in dist P Q = dist Q R ∧ dist Q R = dist R P ∧ dist R P = dist P Q :=
sorry

end equilateral_triangle_midpoints_l410_410777


namespace problem_1_problem_2_l410_410441

theorem problem_1 (A B C : ℝ) (S : ℝ) 
  (h1 : ∀ (AB AC : ℝ), AB * AC = S) :
  ∃ tan2A : ℝ, tan2A = -4/3 := 
by
  sorry

theorem problem_2 (A B C : ℝ) (S : ℝ)
  (h1 : ∀ (AB AC : ℝ), AB * AC = S)
  (h2 : cos C = 3/5)
  (h3 : | AC - AB | = 2) :
  S = 8/5 := 
by
  sorry

end problem_1_problem_2_l410_410441


namespace fastest_growing_function_l410_410705

theorem fastest_growing_function (x : ℝ) (hx : x > 0):
  (∀ x, x > 0 → (λ x, e^[x]) x > (λ x, log x) x ∧
                  (λ x, e^[x]) x > (λ x, x^2) x ∧
                  (λ x, e^[x]) x > (λ x, 2^[x]) x ∧
                  e > 2) → 
  (λ x, e^[x]) x = (λ x, e^[x]) x :=
by
  -- Proof is omitted.
  sorry

end fastest_growing_function_l410_410705


namespace problem_l410_410981

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l410_410981


namespace consecutive_subsequence_sum_l410_410573

theorem consecutive_subsequence_sum
  (seq : Fin 51 → ℕ)
  (h_sum : (∑ i, seq i) = 100) :
  ∀ k : ℕ, 1 ≤ k → k < 100 → 
  (∃ l u : ℕ, l < u ∧ (∑ i in Finset.Ico l u, seq i) = k) ∨ 
  (∃ l u : ℕ, l < u ∧ (∑ i in Finset.Ico l u, seq i) = 100 - k) :=
sorry

end consecutive_subsequence_sum_l410_410573


namespace graded_worksheets_before_l410_410776

-- Definitions based on conditions
def initial_worksheets : ℕ := 34
def additional_worksheets : ℕ := 36
def total_worksheets : ℕ := 63

-- Equivalent proof problem statement
theorem graded_worksheets_before (x : ℕ) (h₁ : initial_worksheets - x + additional_worksheets = total_worksheets) : x = 7 :=
by sorry

end graded_worksheets_before_l410_410776


namespace problem_l410_410983

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l410_410983


namespace area_of_triangle_BEC_is_six_l410_410548

open EuclideanGeometry

variables (A B C D E : Point)
variables (AD DC : LineSegment) 

-- Conditions
variables (AD_perpendicular_DC : IsPerpendicular AD DC)
variables (AD_length : length AD = 3)
variables (AB_length : length (lineSegment A B) = 3)
variables (DC_length : length DC = 7)
variables (E_on_DC : E ∈ DC)
variables (BE_parallel_AD : IsParallel (lineSegment B E) AD)
variables (DE_length : length (lineSegment D E) = 3)

-- Proof Goal
theorem area_of_triangle_BEC_is_six :
  area (triangle B E C) = 6 :=
  sorry

end area_of_triangle_BEC_is_six_l410_410548


namespace problem_statement_l410_410534

noncomputable def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2) in
  let θ := atan y x in
  (r, θ)

theorem problem_statement :
  (∀ t : ℝ, y - x - 2 = 0) ∧
  ((x - 2)^2 + (y - 1)^2 = 5) ∧
  ((l_translated : ℝ → ℝ × ℝ) → 
    (∀ x y : ℝ, l_translated = cartesian_to_polar (x+2) y)) →
  (polar equation of curve : (r θ : ℝ) → r = 4 * cos θ + 2 * sin θ) →
  (∀ A B : ℝ → ℝ, l_translated_translates_2_units_right_to x y l_translated) →
  (polars_of (A ≠ B) : 
    (length_AB : ℝ, length_AB = 3*sqrt(2)) ∧
    (area : ℝ, area = 6)) :=
begin
  sorry
end

end problem_statement_l410_410534


namespace triangle_incenter_inequality_l410_410107

noncomputable theory

variables {A B C I A' B' C' : Type*}

def incenter_of_triangle (A B C I : Type*) : Prop := sorry
def angle_bisector_intersection (A B C A' B' C' : Type*) : Prop := sorry
def geometric_inequality (AI BI CI A'A B'B C'C : ℝ) : Prop :=
  (1/4 < (AI * BI * CI) / (A'A * B'B * C'C)) ∧ ((AI * BI * CI) / (A'A * B'B * C'C) ≤ 8/27)

theorem triangle_incenter_inequality
  (h_abc_incenter : incenter_of_triangle A B C I)
  (h_angle_bisector : angle_bisector_intersection A B C A' B' C')
  (AI BI CI A'A B'B C'C : ℝ) :
  geometric_inequality AI BI CI A'A B'B C'C :=
sorry

end triangle_incenter_inequality_l410_410107


namespace intersection_set_cardinality_l410_410953

variable {α β : Type*}
variables (f : α → β) (a b : α)

theorem intersection_set_cardinality : 
  (x y : α) → y = f x ∧ a ≤ x ∧ x ≤ b ∧ x = 2 → 
  ∃ n : ℕ, n = 0 ∨ n = 1 ∧ 
  #({(x, y) | y = f x ∧ a ≤ x ∧ x ≤ b} ∩ {(x, y) | x = 2}) = n := 
by sorry

end intersection_set_cardinality_l410_410953


namespace max_sin_cos_expr_l410_410893

theorem max_sin_cos_expr (x y z : ℝ) :
  let expr := (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))
  in ∀ (x y z : ℝ), expr ≤ 4.5 :=
sorry

end max_sin_cos_expr_l410_410893


namespace simplify_equation_l410_410541

theorem simplify_equation (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) -> 
  (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by 
  sorry

end simplify_equation_l410_410541


namespace axis_of_symmetry_exists_l410_410952

theorem axis_of_symmetry_exists (ω φ : ℝ) (hφ : |φ| < π / 2) (hω : 2 = ω) 
(h_pass : 3 * sin (ω * (-π / 6) + φ) = 0) : 
∃ k : ℤ, - (5 * π) / 12 = (π / 12) + (1 / 2) * k * π :=
by
  -- Insert the detailed proof here
  sorry

end axis_of_symmetry_exists_l410_410952


namespace leg_length_of_isosceles_right_triangle_l410_410632

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * real.sqrt 2

def median_to_hypotenuse (a b c m : ℝ) : Prop :=
  is_isosceles_right_triangle a b c ∧ m = c / 2

-- The proof problem statement
theorem leg_length_of_isosceles_right_triangle (a b c m : ℝ) (h1 : is_isosceles_right_triangle a b c)
  (h2 : median_to_hypotenuse a b c m) (h3 : m = 12) : a = 12 * real.sqrt 2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l410_410632


namespace value_of_a_minus_b_l410_410510

theorem value_of_a_minus_b (a b : ℝ) (h : (a + 1)^2 + sqrt (b - 2) = 0) : a - b = -3 := sorry

end value_of_a_minus_b_l410_410510


namespace balls_in_boxes_l410_410490

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l410_410490


namespace pipes_fill_cistern_together_in_15_minutes_l410_410660

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

end pipes_fill_cistern_together_in_15_minutes_l410_410660


namespace international_society_pigeonhole_l410_410785

theorem international_society_pigeonhole (members : Finset ℕ) (m : ℕ)
  (h_mem_sizes : members.card = 1978)
  (countries : Finset (Finset ℕ)) 
  (h_country_sizes : countries.card = 6)
  (h_members_in_countries : (∀ c ∈ countries, (c ⊆ members) ∧ (c.nonempty))):
  ∃ a ∈ members, ∃ b c ∈ members, (b ≠ c ∧ (a = b + c ∨ a = 2 * b)) :=
by
  sorry

end international_society_pigeonhole_l410_410785


namespace train_speed_l410_410346

theorem train_speed
  (length : ℕ)
  (time : ℕ)
  (h_length : length = 140)
  (h_time : time = 8) :
  (length / time) * 3.6 = 63 :=
by
  sorry

end train_speed_l410_410346


namespace marathon_time_l410_410334

theorem marathon_time (distance_first_10 : ℕ) (time_first_10 : ℕ) (total_distance : ℕ) (pace_reduction : ℚ) : 
  total_distance = 26 → 
  distance_first_10 = 10 → 
  time_first_10 = 1 →
  pace_reduction = 0.8 →
  ∃ total_time : ℚ, total_time = 3 :=
by 
  intros h_total_distance h_distance_first_10 h_time_first_10 h_pace_reduction
  let pace_first_10 := distance_first_10 / time_first_10
  let pace_reduced := pace_first_10 * pace_reduction
  let remaining_distance := total_distance - distance_first_10
  let remaining_time := remaining_distance / pace_reduced
  let total_time := time_first_10 + remaining_time
  use total_time
  rw [h_total_distance, h_distance_first_10, h_time_first_10, h_pace_reduction]
  norm_num
  sorry

end marathon_time_l410_410334


namespace angle_ABF_90_degrees_l410_410244

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b > 0) := (√5 - 1) / 2

theorem angle_ABF_90_degrees (a b : ℝ) (h : a > b > 0) (e : ℝ)
  (h_e : e = ellipse_eccentricity a b h) :
  ∃ (A B F : point) (angle_ABF : ℝ), angle_ABF = 90 :=
begin
  -- definitions and proof go here
  sorry
end

end angle_ABF_90_degrees_l410_410244


namespace eq_line_AD_CD_parallel_BE_l410_410458

open Set

-- Define the ellipse Γ
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the points C(1, 0) and D(2, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)

-- Define that a line passing through C intersects the ellipse at points A and B
def passes_through_C (A B : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∃ k : ℝ, (A = (C.1 + k, C.2 + m * k) ∧ ellipse A.1 A.2) ∧ (B = (C.1 - k, C.2 - m * k) ∧ ellipse B.1 B.2)

-- Define that AB is perpendicular to the x-axis
def AB_perpendicular_to_x_axis (A B : ℝ × ℝ) : Prop := A.1 = B.1

-- Define the line x = 3
def line_x_eq_3 : (ℝ × ℝ) → Prop := λ P, P.1 = 3

-- Prove that the equation of line AD is y = ± √6 / 3(x - 2) when AB is perpendicular to the x-axis
theorem eq_line_AD (A : ℝ × ℝ) (B : ℝ × ℝ) (h1 : passes_through_C A B)
  (h2 : AB_perpendicular_to_x_axis A B) : 
  (line_x_eq_3 (A.1, A.2 * (A.1 - 2))) := 
sorry

-- Define the points E such that E is the intersection of line AD and x = 3
def E (A : ℝ × ℝ) : ℝ × ℝ := (3, A.2 * (3 - 2))

-- Prove that CD is parallel to BE
theorem CD_parallel_BE (A B : ℝ × ℝ) (h1 : passes_through_C A B)
  (h2 : AB_perpendicular_to_x_axis A B) : 
  ∃ E : ℝ × ℝ, E = (3, A.2 * (3 - 2)) ∧ ((D.2 - C.2) / (D.1 - C.1) = (E.2 - B.2) / (E.1 - B.1)) :=
sorry

end eq_line_AD_CD_parallel_BE_l410_410458


namespace ellipse_properties_l410_410068

theorem ellipse_properties 
  (a b : ℝ) (h1 : a > b) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : ∃ c : ℝ, 2 * c = a ∧ ∀ (x y : ℝ), (x, y) = (2, 0) ∨ (∃ c, c = 1 ∧ x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1))
  (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (A B : ℝ × ℝ) 
  (hP : P = (4, 3)) 
  (hQ : Q = (1, 0))
  (hAB : ∃ k : ℝ, ((A = (1, 3 / 2)) ∧ (B = (1, -3 / 2))) ∨ (A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1) ∧ A ≠ B)) :
  (x y : ℝ) (ha : a = 2) (hb : b^2 = 4 - 1) :
  x^2 / a^2 + y^2 / b^2 = 1 ∧ 
  (k₁ k₂ : ℝ) (h_k₀ : k₁ = (3 - 3 / 2) / (4 - 1)) (h_k₁ : k₂ = (3 + 3 / 2) / (4 - 1)) :
  k₁ + k₂ = 2 :=
by sorry

end ellipse_properties_l410_410068


namespace rational_terms_count_l410_410475

theorem rational_terms_count :
  let a (n : ℕ) := 1 / ((n + 1) * Real.sqrt n + n * Real.sqrt (n + 1))
  let S (n : ℕ) := ∑ k in Finset.range (n + 1), a k
  (Finset.filter (fun n => Real.is_rational (S n)) (Finset.range 2009)).card = 43 :=
by
  let a := fun (n : ℕ) => 1 / ((n + 1) * Real.sqrt n + n * Real.sqrt (n + 1))
  let S := fun (n : ℕ) => ∑ k in Finset.range (n + 1), a k
  exact sorry

end rational_terms_count_l410_410475


namespace find_angle_A_find_side_a_l410_410139

variable {α : Type*}
noncomputable
def triangle (a b c : ℝ) (A B C : ℝ) : Prop := 
  A + B + C = π

noncomputable
def area (a b c : ℝ) (A : ℝ) : ℝ :=
  1/2 * b * c * sin A

theorem find_angle_A 
(a b c A B C : ℝ) 
(h_triangle : triangle a b c A B C)
(h_condition : sqrt 3 * a * sin C = c * cos A) 
: A = π / 6 := sorry

theorem find_side_a
(a b c A B C : ℝ)
(h_triangle : triangle a b c A B C)
(h_area : area a b c A = sqrt 3)
(h_sum_bc : b + c = 2 + 2 * sqrt 3)
: a = 2 := sorry

end find_angle_A_find_side_a_l410_410139


namespace magnitude_difference_l410_410108

noncomputable def vector_magnitude_difference (a b : ℝ) : ℝ :=
  let norm_sum := real.sqrt 20
  let dot_product := 4
  real.sqrt (norm_sum^2 - 4 * dot_product)

theorem magnitude_difference (a b : ℝ) 
    (h1 : (real.sqrt ((a + b)^2) = real.sqrt 20)) 
    (h2 : (a * b = 4)) : |a - b| = 2 :=
by 
  rw abs_sub
  sorry

end magnitude_difference_l410_410108


namespace correct_propositions_l410_410927

-- Definitions of the conditions
variables {α β : Plane} {l : Line} {P : Point}

-- Given conditions
axiom α_perp_β : α ⊥ β
axiom α_cap_β_eq_l : α ∩ β = l
axiom P_in_α : P ∈ α
axiom P_notin_l : P ∉ l

-- Assertions
theorem correct_propositions :
  (PlaneThrough P ⊥ l) ⊥ β ∧       -- ①
  (LineThrough P ⊥ α) ∥ β ∧         -- ③
  (LineThrough P ⊥ β) ∈ α :=        -- ④
sorry

end correct_propositions_l410_410927


namespace polygon_sides_l410_410263

theorem polygon_sides (h1 : 1260 - 360 = 900) (h2 : (n - 2) * 180 = 900) : n = 7 :=
by 
  sorry

end polygon_sides_l410_410263


namespace geometric_seq_b_sum_T_n_harmonic_inequality_l410_410436

def seq_a (n : ℕ) : ℕ
| 1       := 2
| (n + 1) := 2 * (Seq.sum (λ k, seq_a k) n + n + 1) -- This definition is not directly needed for the proof but represents the original condition

def seq_b (n : ℕ) := seq_a n + 1

def is_geometric (seq : ℕ → ℕ) (r : ℕ) : Prop :=
∀ n, seq (n + 1) = r * seq n

theorem geometric_seq_b : is_geometric seq_b 3 :=
sorry

noncomputable def T_n (n : ℕ) : ℕ :=
∑ k in Finset.range n, (k + 1) * seq_b (k + 1)

theorem sum_T_n (n : ℕ) : T_n n = (2 * n - 1) / 4 * 3 ^ (n + 1) + 3 / 4 :=
sorry

noncomputable def harmonic_sum (n : ℕ) : ℝ :=
∑ k in Finset.range n, 1 / seq_a (k + 1)

theorem harmonic_inequality (n : ℕ) :
  1/2 - 1/(2 * 3 ^ n) < harmonic_sum n ∧ harmonic_sum n < 11/16 :=
sorry

end geometric_seq_b_sum_T_n_harmonic_inequality_l410_410436


namespace ellipse_problem_l410_410070

-- Conditions for the ellipse
def is_ellipse (E : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ ∀ x y : ℝ, E x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) 

-- Given specific conditions for solving the ellipse equation
noncomputable def EccentricityCondition (a b : ℝ) : Prop :=
 ∀ c : ℝ, (c = (sqrt 3 / 2) * a) ∧ (1 / a^2 + 3 / (4 * b^2) = 1)

-- Check if ellipse matches provided point
def PassThroughPoint (E : ℝ → ℝ → Prop) : Prop :=
E 1 (sqrt 3 / 2)

-- Line intersection and tangent to the circle checks
def LineConditions (E : ℝ → ℝ → Prop) (k m : ℝ) : Prop :=
∀ P Q : ℝ × ℝ, 
(P ≠ Q ∧ E P.1 P.2 ∧ E Q.1 Q.2 →
((P.1 + Q.1 = -8 * k * m / (1 + 4 * k^2)) ∧ 
 (P.1 * Q.1 = (4 * (m^2 - 1)) / (1 + 4 * k^2)) ∧ 
 ((P.2 / P.1) + (Q.2 / Q.1) = 2)) ∧ 
(m^2 + k = 1)) ∧ (abs m / sqrt (k^2 + 1) = 1)

theorem ellipse_problem (a b k m: ℝ) : 
  ∃ E : ℝ → ℝ → Prop,
  is_ellipse E a b ∧ 
  EccentricityCondition a b ∧ 
  PassThroughPoint E ∧ 
  LineConditions E k m ↔ 
  (k = -1) ∧ (m = sqrt 2 ∨ m = -sqrt 2) :=
sorry

end ellipse_problem_l410_410070


namespace surface_area_of_rectangular_solid_l410_410373

theorem surface_area_of_rectangular_solid :
  ∃ (a b c : ℕ), (nat.prime a) ∧ (nat.prime b) ∧ (nat.prime c) ∧ (a * b * c = 1001) ∧ (2 * (a * b + b * c + c * a) = 622) :=
by {
  use [7, 11, 13],
  split, { exact nat.prime_7 },
  split, { exact nat.prime_11 },
  split, { exact nat.prime_13 },
  split, {
    norm_num,
  },
  {
    norm_num,
  },
}

end surface_area_of_rectangular_solid_l410_410373


namespace solve_arithmetic_seq_l410_410609

theorem solve_arithmetic_seq (x : ℝ) (h : x > 0) (hx : x^2 = (4 + 16) / 2) : x = Real.sqrt 10 :=
sorry

end solve_arithmetic_seq_l410_410609


namespace isosceles_right_triangle_leg_length_l410_410629

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l410_410629


namespace inequality_solution_set_l410_410261

theorem inequality_solution_set (x : ℝ) : 
  (0 < x ∧ x < 2) ↔ (| (x - 2) / x | > (x - 2) / x) :=
by
  sorry

end inequality_solution_set_l410_410261


namespace smallest_n_satisfies_conditions_l410_410671

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410671


namespace log_sum_equals_zero_l410_410426

theorem log_sum_equals_zero {a b : ℝ} (h1 : 1 < a) (h2 : 1 < b) (h3 : Real.log10 (a + b) = Real.log10 a + Real.log10 b) : 
  Real.log10 (a - 1) + Real.log10 (b - 1) = 0 := 
by
  sorry

end log_sum_equals_zero_l410_410426


namespace max_magnitude_z3_plus_3z_plus_2i_l410_410574

open Complex

theorem max_magnitude_z3_plus_3z_plus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ M, M = 3 * Real.sqrt 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z^3 + 3 * z + 2 * Complex.I) ≤ M :=
by
  sorry

end max_magnitude_z3_plus_3z_plus_2i_l410_410574


namespace divide_triangle_in_half_l410_410913

def triangle_vertices : Prop :=
  let A := (0, 2)
  let B := (0, 0)
  let C := (10, 0)
  let base := 10
  let height := 2
  let total_area := (1 / 2) * base * height

  ∀ (a : ℝ),
  (1 / 2) * a * height = total_area / 2 → a = 5

theorem divide_triangle_in_half : triangle_vertices := 
  sorry

end divide_triangle_in_half_l410_410913


namespace total_fence_needed_l410_410904

theorem total_fence_needed (a1 a2 b1 b2 c1 c2 : ℝ) (areaA areaB areaC : ℝ) : 
  a1 = 40 → areaA = 320 → b1 = 60 → areaB = 480 → c1 = 80 → areaC = 720 → 
  a2 = areaA / a1 → b2 = areaB / b1 → c2 = areaC / c1 → 
  3 * (a1 + b1 + c1) + 2 * (a2 + b2 + c2) = 410 :=
by
  intros hA1 hAreaA hB1 hAreaB hC1 hAreaC hA2 hB2 hC2
  rw [hA1, hAreaA, hB1, hAreaB, hC1, hAreaC, hA2, hB2, hC2]
  exact sorry

end total_fence_needed_l410_410904


namespace smallest_n_45_l410_410688

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410688


namespace form_seven_hollow_squares_l410_410662

-- Define a domino as a pair of natural numbers representing the points on its ends
structure Domino :=
  (first : ℕ)
  (second : ℕ)

-- The main theorem statement
theorem form_seven_hollow_squares (dominoes : List Domino) 
  (h_length : dominoes.length = 28) :
  ∃ (squares : List (List Domino)),
  squares.length = 7 ∧ 
  ∀ (square : List Domino) (h_square : square ∈ squares) (S : ℕ),
    ((∑ d in square, d.first + d.second) = S) :=
sorry

end form_seven_hollow_squares_l410_410662


namespace range_of_omega_l410_410094

noncomputable def f (ω x : ℝ) : ℝ := 
  sin^2 (ω * x / 2) + 1 / 2 * sin (ω * x) - 1 / 2

theorem range_of_omega (ω : ℝ) (h₁ : ω > 0) : 
  (∃ x : ℝ, x ∈ set.Ioo π (2 * π) ∧ f ω x = 0) ↔ 
  ω ∈ set.Ioo (1 / 8) (1 / 4) ∪ set.Ioi (5 / 8) := 
by 
  sorry

end range_of_omega_l410_410094


namespace smallest_y_value_smallest_y_value_is_neg6_l410_410703

theorem smallest_y_value :
  ∀ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) → (y = -3 ∨ y = -6) :=
by
  sorry

theorem smallest_y_value_is_neg6 :
  ∃ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧ (y = -6) :=
by
  have H := smallest_y_value
  sorry

end smallest_y_value_smallest_y_value_is_neg6_l410_410703


namespace log_x_y_pi_l410_410750

theorem log_x_y_pi (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h1 : ∀ r, r = log 10 (x^3))
  (h2 : ∀ C, C = log 10 (y^6))
  (h3 : ∀ r C, C = 2 * π * r):
  log x y = π :=
by sorry

end log_x_y_pi_l410_410750


namespace smallest_positive_period_symmetry_about_line_zero_at_pi_over_3_max_value_not_sqrt3_plus_1_l410_410466

def f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := by
  sorry

theorem symmetry_about_line : ∀ x, f (\frac{π}{12} - x) = f (\frac{π}{12} + x) := by
  sorry

theorem zero_at_pi_over_3 : f (Real.pi / 3) = 0 := by
  sorry

theorem max_value_not_sqrt3_plus_1 : ∀ x, f x ≤ 2 := by
  sorry

end smallest_positive_period_symmetry_about_line_zero_at_pi_over_3_max_value_not_sqrt3_plus_1_l410_410466


namespace stream_speed_l410_410648

variables (v_s t_d t_u : ℝ)
variables (D : ℝ) -- Distance is not provided in the problem but assumed for formulation.

theorem stream_speed (h1 : t_u = 2 * t_d) (h2 : v_s = 54 + t_d / t_u) :
  v_s = 18 := 
by
  sorry

end stream_speed_l410_410648


namespace smallest_n_45_l410_410690

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410690


namespace max_min_XY_XZ_diff_zero_l410_410550

theorem max_min_XY_XZ_diff_zero (YZ : ℝ) (XM : ℝ) (M : ℝ → ℝ) (x : ℝ) (h : ℝ):
  (YZ = 10) →
  (XM = 6) →
  (∀ y z : ℝ, M y = M z) →
  ∃ N n : ℝ, N = n ∧ N - n = 0 :=
by
  intro YZ_eq XM_eq M_midpoint
  use 122
  use 122
  split
  · sorry -- Proof that N = n = 122
  · sorry -- Proof that N - n = 0

end max_min_XY_XZ_diff_zero_l410_410550


namespace cos_90_eq_zero_l410_410002

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410002


namespace product_of_constants_l410_410391

theorem product_of_constants :
  ∃ (a b : ℤ), ab = 12 ∧
  let t := a + b in 
  (∏ (p : (Σ a b : ℤ, a * b = 12)), p.2.1 + p.2.2) = -531441 :=
by
  sorry

end product_of_constants_l410_410391


namespace decimals_not_align_ends_l410_410286

theorem decimals_not_align_ends :
  ¬(∀ (a b : Real), true → align_ends a b = true) := sorry

end decimals_not_align_ends_l410_410286


namespace sample_size_l410_410326

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ) (elderly_employees : ℕ) (young_in_sample : ℕ)

theorem sample_size (h1 : total_employees = 750) (h2 : young_employees = 350) (h3 : middle_aged_employees = 250) (h4 : elderly_employees = 150) (h5 : young_in_sample = 7) :
  ∃ sample_size, young_in_sample * total_employees / young_employees = sample_size ∧ sample_size = 15 :=
by
  sorry

end sample_size_l410_410326


namespace unique_a_for_set_l410_410576
-- Import necessary library

-- Define the main theorem
theorem unique_a_for_set (a : ℝ) (U M : Set ℝ)
  (hU : U = {1, 3, a^3 + 3 * a^2 + 2 * a})
  (hM : M = {1, |2 * a - 1|})
  (hC : C_U_U = {0}) :
  a = -1 :=
by sorry

end unique_a_for_set_l410_410576


namespace f_a11_eq_6_l410_410930

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x * (1 - x) else x * (1 + x)

def a : ℕ → ℝ
| 0     := 1 / 2
| (n+1) := 1 / (1 - a n)

theorem f_a11_eq_6 : f (a 11) = 6 :=
by
  sorry

end f_a11_eq_6_l410_410930


namespace intersection_is_1_l410_410477

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {y | ∃ x ∈ M, y = x ^ 2}
theorem intersection_is_1 : M ∩ N = {1} := by
  sorry

end intersection_is_1_l410_410477


namespace minimum_slope_of_MN_l410_410940

noncomputable def min_slope_MN (a b : ℝ) (y0 : ℝ) : ℝ :=
  -2 * y0 / (3 * y0^2 + 16)

theorem minimum_slope_of_MN : 
  ∃ y0 : ℝ, y0 > 0 ∧ min_slope_MN 2 (sqrt 3) y0 = - (sqrt 3) / 12 := 
begin
  sorry
end

end minimum_slope_of_MN_l410_410940


namespace binomial_expansion_coefficient_x2_l410_410538

theorem binomial_expansion_coefficient_x2 :
  let x := (λ x : ℝ, (sqrt x / 2 - 2 / sqrt x) ^ 6) in
    (∃ c : ℝ, c * x ^ 2 = (binomialCoef 6 1 *
    (1/2) ^ 5 * (-2) * x ^ 2) ∧ c = -3/8) :=
sorry

end binomial_expansion_coefficient_x2_l410_410538


namespace isosceles_right_triangle_leg_length_l410_410631

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l410_410631


namespace max_sin_cos_expr_l410_410895

theorem max_sin_cos_expr (x y z : ℝ) :
  let expr := (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))
  in ∀ (x y z : ℝ), expr ≤ 4.5 :=
sorry

end max_sin_cos_expr_l410_410895


namespace find_a_l410_410163

theorem find_a (A point B C : ℝ × ℝ) (theta a : ℝ) 
  (h_theta_acute : theta > 0 ∧ theta < π/2) 
  (h_tan_theta : tan theta = 2) 
  (h_a_pos : a > 0)
  (h_A : A = (2 * sqrt 5, π + theta)) 
  (h_parallel_line : ∀ (ρ : ℝ), B ≠ C → B ≠ A → C ≠ A → 
    (A.2 - B.2) / (A.1 - B.1) = (A.2 - C.2) / (A.1 - C.1) = tan (π/4) ∨ tan (5*π/4)) 
  (curve_L : ∀ (θ : ℝ) (ρ : ℝ), ρ * sin(θ)^2 = 2 * a * cos(θ)) 
  (h_geometric_prog : (dist A B) ^ 2 = (dist B C) ^ 2 ∧ (dist C A) ^ 2 = (dist A B) * (dist B C))
  : a = 1 := by
  sorry

end find_a_l410_410163


namespace tan_eq_sin_solutions_count_l410_410484

theorem tan_eq_sin_solutions_count :
  set.count
    {x | x ∈ set.Icc 0 (2 * Real.pi / 3) ∧ Real.tan (3 * x) = Real.sin (3 * x / 2)}
    = 3 := 
sorry

end tan_eq_sin_solutions_count_l410_410484


namespace corresponding_side_of_larger_triangle_l410_410624

theorem corresponding_side_of_larger_triangle
  (A1 A2 : ℝ)
  (side_smaller larger_side : ℝ)
  (h_diff : A1 - A2 = 27)
  (h_ratio : A1 / A2 = 9)
  (h_integer : ∃ n : ℕ, A2 = n)
  (h_side : side_smaller = 4)
  (larger_side = side_smaller * 3) : 
  larger_side = 12 :=
by
  sorry

end corresponding_side_of_larger_triangle_l410_410624


namespace children_to_add_l410_410589

def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def desired_children := 30

theorem children_to_add : (desired_children - children) = 10 := by
  sorry

end children_to_add_l410_410589


namespace range_of_a_l410_410470

-- Defining the function y = a * (x^3 - 3 * x)
def f (a x : ℝ) : ℝ := a * (x^3 - 3 * x)

-- Defining the derivative of the function
def f' (a x : ℝ) : ℝ := a * (3 * x^2 - 3)

-- The mathematical condition that the function is increasing on the interval (-1, 1)
def is_increasing_on (a : ℝ) : Prop :=
  ∀ x ∈ set.Ioo (-1 : ℝ) 1, f' a x ≥ 0

-- The main proof problem statement
theorem range_of_a (a : ℝ) : 
  is_increasing_on a → a < 0 :=
begin
  sorry
end

end range_of_a_l410_410470


namespace nancy_total_spent_l410_410741

def crystal_cost : ℕ := 9
def metal_cost : ℕ := 10
def total_crystal_cost : ℕ := crystal_cost
def total_metal_cost : ℕ := 2 * metal_cost
def total_cost : ℕ := total_crystal_cost + total_metal_cost

theorem nancy_total_spent : total_cost = 29 := by
  sorry

end nancy_total_spent_l410_410741


namespace function_single_intersection_l410_410129

theorem function_single_intersection (a : ℝ) : 
  (∃ x : ℝ, ax^2 - x + 1 = 0 ∧ ∀ y : ℝ, (ax^2 - x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1/4) :=
sorry

end function_single_intersection_l410_410129


namespace find_expression_value_l410_410964

theorem find_expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end find_expression_value_l410_410964


namespace sqrt_three_is_achievable_l410_410639

def initial_number : ℝ := 0

def operations := 
  { f : ℝ → ℝ | 
      f = sin ∨ f = cos ∨ f = tan ∨ f = λ x, 1 / tan x ∨ 
      f = λ x, real.arcsin x ∨ f = λ x, real.arccos x ∨ 
      f = λ x, real.arctan x ∨ f = λ x, real.arccot x }

noncomputable def achievable_numbers : set ℝ := 
  {x | ∃ f (g : ℝ → ℝ) (a b : ℝ),
    f ∈ operations ∧ g ∈ operations ∧ a ∈ achievable_numbers ∧ 
    b ∈ achievable_numbers ∧ (x = f a ∨ x = g b ∨ x = a * b ∨ x = a / b)}

theorem sqrt_three_is_achievable : (√3 : ℝ) ∈ achievable_numbers := 
  sorry

end sqrt_three_is_achievable_l410_410639


namespace xy_sum_is_one_l410_410219

theorem xy_sum_is_one (x y : ℝ) (h : x^2 + y^2 + x * y = 12 * x - 8 * y + 2) : x + y = 1 :=
sorry

end xy_sum_is_one_l410_410219


namespace speed_of_man_cycling_l410_410644

theorem speed_of_man_cycling (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : B = 3 * L)
  (h3 : L * B = 30000) (h4 : ∀ t : ℝ, t = 4 / 60): 
  ( (2 * L + 2 * B) / (4 / 60) ) = 12000 :=
by
  -- Assume given conditions
  sorry

end speed_of_man_cycling_l410_410644


namespace marathon_time_l410_410336

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end marathon_time_l410_410336


namespace calculate_f_values_l410_410019

noncomputable def f : ℝ → ℝ := sorry

axiom f_conditions :
  (f(0) = 0) ∧
  (∀ x, f(x) + f(1 - x) = 1) ∧
  (∀ x, f(x / 3) = (1 / 2) * f(x)) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f(x1) ≤ f(x2))

theorem calculate_f_values :
  f (1 / 3) + f (1 / 8) = 3 / 4 :=
by {
  sorry
}

end calculate_f_values_l410_410019


namespace intersection_A_B_l410_410103

def set_A (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def set_B (x : ℝ) : Prop := 2 < x ∧ x < 4

theorem intersection_A_B (x : ℝ) :
  (set_A x ∧ set_B x) ↔ 2 < x ∧ x < 4 :=
by sorry

end intersection_A_B_l410_410103


namespace incorrect_judgment_l410_410910

open Classical

variable (P : Prop) (Q : Prop) 

theorem incorrect_judgment (hP : P = (2 + 2 = 5)) (hQ : Q = 3 > 2)
  (hA : (P ∨ Q = True) ∧ (¬Q = False))
  (hB : (P ∧ Q = False) ∧ (¬P = True))
  (hC : (P ∧ Q = False) ∧ (¬P = False))
  (hD : (P ∧ Q = False) ∧ (P ∨ Q = True)) :
  hC = False := 
sorry

end incorrect_judgment_l410_410910


namespace number_of_ways_to_distribute_balls_l410_410502

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l410_410502


namespace product_of_constants_l410_410398

theorem product_of_constants (t : ℤ) (a b : ℤ) (h1 : a * b = 12)
  (h2 : t = a + b) :
  ∃ p : ℤ, (∀ t, (∃ a b : ℤ, a * b = 12 ∧ t = a + b) → t) ∧ p = -527776 :=
sorry

end product_of_constants_l410_410398


namespace minimum_elements_l410_410645

structure relation_on (X : Type) :=
(wedge : X → X → Prop)
(irreflexive : ∀ x : X, ¬ wedge x x)
(asymmetric : ∀ x y : X, x ≠ y → wedge x y ∨ wedge y x)
(transitive : ∀ x y, wedge x y → ∃ z : X, wedge x z ∧ wedge z y)

theorem minimum_elements (X : Type) [fintype X] (R : relation_on X) (h : fintype.card X > 1) :
  7 ≤ fintype.card X :=
sorry

end minimum_elements_l410_410645


namespace sum_from_1_to_15_fractions_l410_410806

def sum_of_fractions (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (k : ℚ) / 7

theorem sum_from_1_to_15_fractions :
  sum_of_fractions 15 = 120 / 7 :=
by
  sorry

end sum_from_1_to_15_fractions_l410_410806


namespace smallest_positive_period_l410_410646

noncomputable def given_function (x : ℝ) : ℝ := (√3 / 2) * Real.sin (2 * x) + Real.cos (x) ^ 2

theorem smallest_positive_period : ∃ T > 0, ∀ x, given_function (x + T) = given_function x ∧ T = Real.pi :=
by sorry

end smallest_positive_period_l410_410646


namespace geom_series_exists_l410_410101

noncomputable def infinite_geom_series_subsequence : Prop :=
  ∃ (a q : ℝ) (seq : ℕ → ℝ),
  (∀ n, seq n = 1 / 2 ^ (n + 1)) ∧
  (S = ∑' (n : ℕ), a * q ^ n) ∧
  (S = 1 / 7) ∧
  (a = 1 / 8) ∧
  (q = 1 / 8)

theorem geom_series_exists : infinite_geom_series_subsequence :=
begin
  sorry
end

end geom_series_exists_l410_410101


namespace traffic_officer_distribution_l410_410856

-- Statement of the problem in Lean 4
theorem traffic_officer_distribution (total_officers : ℕ)
  (intersections : ℕ)
  (conditions : total_officers = 5 ∧ intersections = 3 ∧ total_officers ≥ intersections)
  (at_least_one_per_intersection : ∀ (i : ℕ), i < intersections → at_least_one_at i total_officers intersections)
  : (number_of_arrangements_with_officers_A_and_B_together total_officers intersections = 36) := by
  sorry

-- Assume a definition for at_least_one_at and number_of_arrangements_with_officers_A_and_B_together to match the conditions
def at_least_one_at (i : ℕ) (total_officers : ℕ) (intersections : ℕ) : Prop := sorry

def number_of_arrangements_with_officers_A_and_B_together (total_officers : ℕ) (intersections : ℕ) : ℕ := sorry

end traffic_officer_distribution_l410_410856


namespace xyz_product_value_l410_410989

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l410_410989


namespace sum_even_binomial_coefficients_rational_terms_l410_410933

-- (I) Prove the sum of the binomial coefficients of all the even terms.
theorem sum_even_binomial_coefficients (n : ℕ) (h1 : (∑ r in range (n // 2 + 1), (n.choose (2 * r))) = 256) :
    n = 9 → 
    2^8 = 256 := 
sorry

-- (II) Prove the rational terms in the expansion.
theorem rational_terms (n : ℕ) (h1 : n = 9)
    (t3 : (_ : ℕ) (Cn : BinomialCoeff 9 n ) ((9.choose 0)) x ^ 3 = x^3)
    (t4 : (_ : ℕ) (Cn : BinomialCoeff 9 n ) ((9.choose 3)) (-2)^3 x ^ -1 = -672 * x ^ -1)
    (t7 : (_ : ℕ) (Cn : BinomialCoeff 9 n ) ((9.choose 6)) (-2)^6 x ^ -5 = 5376 * x ^ -5)
    (t10 : (_ : ℕ) (Cn : BinomialCoeff 9 n ) ((9.choose 9)) (-2)^9 x ^ -9 = -512 * x ^ -9) :
    ∀ (r : ℕ), r ∈ [0, 3, 6, 9] →   
    BinomialCoeff 9 r := sorry

end sum_even_binomial_coefficients_rational_terms_l410_410933


namespace cost_of_ice_cream_scoop_l410_410036

theorem cost_of_ice_cream_scoop
  (num_meals : ℕ) (meal_cost : ℕ) (total_money : ℕ)
  (total_meals_cost : num_meals * meal_cost = 30)
  (remaining_money : total_money - 30 = 15)
  (num_ice_cream_scoops : ℕ) (cost_per_scoop : ℕ)
  (total_cost : 30 + 15 = total_money)
  (total_ice_cream_cost : num_ice_cream_scoops * cost_per_scoop = remaining_money) :
  cost_per_scoop = 5 :=
by
  have h_num_meals : num_meals = 3 := by sorry
  have h_meal_cost : meal_cost = 10 := by sorry
  have h_total_money : total_money = 45 := by sorry
  have h_num_ice_cream_scoops : num_ice_cream_scoops = 3 := by sorry
  exact sorry

end cost_of_ice_cream_scoop_l410_410036


namespace circular_garden_radius_l410_410751

theorem circular_garden_radius (r : ℝ) (h1 : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) : r = 12 :=
by sorry

end circular_garden_radius_l410_410751


namespace vector_parallel_solution_l410_410480

theorem vector_parallel_solution (λ : ℝ) :
  let a := (-4, 5)
  let b := (λ, 1)
  (a.1 - b.1, a.2 - b.2) = (c * b.1, c * b.2) for some c : ℝ 
  ∧ b ≠ (0, 0)
  →
  λ = -4 / 5 :=
by
  sorry

end vector_parallel_solution_l410_410480


namespace arithmetic_sequence_condition_l410_410300

theorem arithmetic_sequence_condition (a : ℕ → ℝ) :
  (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2)) ↔
  (∀ n ∈ {k : ℕ | k > 0}, a (n+1) - a n = a (n+2) - a (n+1)) ∧ ¬ (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2) → a (n+1) = a n) :=
sorry

end arithmetic_sequence_condition_l410_410300


namespace number_of_two_digit_integers_twice_perfect_square_sum_l410_410485

def is_two_digit_nat (N : ℕ) := N >= 10 ∧ N < 100

def reverse_digits (N : ℕ) : ℕ :=
  let t := N / 10
  let u := N % 10
  10 * u + t

def is_twice_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, 2 * k^2 = x

theorem number_of_two_digit_integers_twice_perfect_square_sum : 
  {N : ℕ | is_two_digit_nat N ∧ is_twice_perfect_square (N + reverse_digits N) }.card = 7 :=
sorry

end number_of_two_digit_integers_twice_perfect_square_sum_l410_410485


namespace max_k_value_l410_410920

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∃ k : ℝ, (∀ m, 0 < m → m < 1/2 → (1/m + 2/(1-2*m) ≥ k)) ∧ k = 8) := 
sorry

end max_k_value_l410_410920


namespace magpie_cooked_40_grams_l410_410647

-- Definitions based on conditions
variables (m n : ℕ)

-- Condition definitions
def third_chick := m + n
def fourth_chick := n + third_chick
def fifth_chick := third_chick + fourth_chick
lemma fifth_chick_received_ten : fifth_chick m n = 10 :=
  by sorry

def sixth_chick := fourth_chick + fifth_chick

-- Total porridge cooked
def total_porridge (m n : ℕ) :=
  m + n + third_chick m n + fourth_chick m n + fifth_chick m n + sixth_chick m n 

-- Theorem to prove
theorem magpie_cooked_40_grams : ∀ m n : ℕ, 2 * m + 3 * n = 10 → total_porridge m n = 40 :=
by sorry  

end magpie_cooked_40_grams_l410_410647


namespace greatest_expression_l410_410055

theorem greatest_expression 
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : x1 < x2) 
  (hx12 : x1 + x2 = 1) 
  (hy1 : 0 < y1) 
  (hy2 : y1 < y2) 
  (hy12 : y1 + y2 = 1) : 
  x1 * y1 + x2 * y2 > max (x1 * x2 + y1 * y2) (max (x1 * y2 + x2 * y1) (1/2)) := 
sorry

end greatest_expression_l410_410055


namespace cheryl_initial_mms_l410_410802

theorem cheryl_initial_mms (lunch_mms : ℕ) (dinner_mms : ℕ) (sister_mms : ℕ) (total_mms : ℕ) 
  (h1 : lunch_mms = 7) (h2 : dinner_mms = 5) (h3 : sister_mms = 13) (h4 : total_mms = lunch_mms + dinner_mms + sister_mms) : 
  total_mms = 25 := 
by 
  rw [h1, h2, h3] at h4
  exact h4

end cheryl_initial_mms_l410_410802


namespace leg_length_of_isosceles_right_triangle_l410_410633

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * real.sqrt 2

def median_to_hypotenuse (a b c m : ℝ) : Prop :=
  is_isosceles_right_triangle a b c ∧ m = c / 2

-- The proof problem statement
theorem leg_length_of_isosceles_right_triangle (a b c m : ℝ) (h1 : is_isosceles_right_triangle a b c)
  (h2 : median_to_hypotenuse a b c m) (h3 : m = 12) : a = 12 * real.sqrt 2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l410_410633


namespace n_squared_plus_one_divides_n_plus_one_l410_410390

theorem n_squared_plus_one_divides_n_plus_one (n : ℕ) (h : n^2 + 1 ∣ n + 1) : n = 1 :=
by
  sorry

end n_squared_plus_one_divides_n_plus_one_l410_410390


namespace remaining_credit_l410_410579

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l410_410579


namespace conjugate_z_in_first_quadrant_l410_410089

noncomputable def z : ℂ := (2 + I) / (1 + I)

def conjugate_z : ℂ := conj z

theorem conjugate_z_in_first_quadrant (h : (1 + I) * z = 2 + I) : 
  0 < conjugate_z.re ∧ 0 < conjugate_z.im :=
by {
  -- Proof skipped
  sorry
}

end conjugate_z_in_first_quadrant_l410_410089


namespace cube_cut_off_edges_l410_410032

theorem cube_cut_off_edges :
  let original_edges := 12
  let new_edges_per_vertex := 3
  let vertices := 8
  let new_edges := new_edges_per_vertex * vertices
  (original_edges + new_edges) = 36 :=
by
  sorry

end cube_cut_off_edges_l410_410032


namespace factorial_div_eq_l410_410839
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410839


namespace xyz_product_neg4_l410_410980

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l410_410980


namespace female_A_stand_end_both_female_not_at_ends_female_students_not_adjacent_female_A_right_of_B_l410_410733

open Nat
noncomputable theory
open_locale big_operators

-- Define the conditions
def num_male_students := 5
def num_female_students := 2

-- Define the first proof statement
theorem female_A_stand_end : 
  let total_ways := 2 * (num_male_students + num_female_students - 1)! in 
  total_ways = 1440 := by
  sorry

-- Define the second proof statement
theorem both_female_not_at_ends : 
  let choose_positions := (num_male_students - 1).choose num_female_students in
  let remaining_ways := (num_male_students - 1)! in
  let total_ways := choose_positions * remaining_ways in 
  total_ways = 2400 := by
  sorry

-- Define the third proof statement
theorem female_students_not_adjacent : 
  let male_permutations := num_male_students! in
  let gap_choices := (num_male_students + 1).choose num_female_students in
  let total_ways := male_permutations * gap_choices in 
  total_ways = 3600 := by
  sorry

-- Define the fourth proof statement
theorem female_A_right_of_B : 
  let total_permutations := (num_male_students + num_female_students)! in 
  let valid_arrangements := total_permutations / 2 in 
  valid_arrangements = 2520 := by
  sorry

end female_A_stand_end_both_female_not_at_ends_female_students_not_adjacent_female_A_right_of_B_l410_410733


namespace fourth_person_height_l410_410653

variable (H : ℤ)

-- Conditions
def height1 := H
def height2 := H + 2
def height3 := H + 4
def height4 := H + 10

-- Average height condition
def avg_height := (height1 + height2 + height3 + height4) / 4 = 76

-- Proof statement
theorem fourth_person_height
    (H : ℤ)
    (h1 : height1 = H)
    (h2 : height2 = H + 2)
    (h3 : height3 = H + 4)
    (h4 : height4 = H + 10)
    (avg : avg_height H) :
    height4 = 82 :=
by
    sorry

end fourth_person_height_l410_410653


namespace parabola_correctness_l410_410942

noncomputable def parabola_equation (focus : ℝ × ℝ) (directrix : ℝ → Prop) (point : ℝ × ℝ) : Prop :=
  let p := 4 in -- derived from focus
  let eq1 := (y : ℝ) ^ 2 = 4 * (x : ℝ) in
  let eq2 := (x : ℝ) ^ 2 = (1 / 2) * (y : ℝ) in
  (focus = (-2, 0)) ∧ directrix (-1) ∧ (point = (1, 2)) → (eq1 ∨ eq2)

-- Example theorem statement using the above definition
theorem parabola_correctness :
  parabola_equation (-2, 0) (λ y, y = -1) (1, 2) :=
by
  sorry

end parabola_correctness_l410_410942


namespace car_travel_distance_l410_410312

theorem car_travel_distance :
  let original_mpg := 300 / 10
  let efficiency_decrease := 0.10
  let new_mpg := original_mpg * (1 - efficiency_decrease)
  let gallons := 15
  (new_mpg * gallons) = 405 :=
by {
  let original_mpg := 300 / 10
  let efficiency_decrease := 0.10
  let new_mpg := original_mpg * (1 - efficiency_decrease)
  let gallons := 15
  have h1 : original_mpg = 30 := by norm_num
  have h2 : new_mpg = 27 := by norm_num
  have h3 : (new_mpg * gallons) = 405 := by norm_num
  exact h3
}

end car_travel_distance_l410_410312


namespace average_expenditure_decrease_l410_410657

-- Define the conditions given in the problem
def original_students : ℕ := 35
def new_students : ℕ := 42
def expense_increase : ℕ := 84
def original_total_expense : ℕ := 630

-- Define the proof statement
theorem average_expenditure_decrease :
  let A := original_total_expense / original_students,
      new_total_expense := original_total_expense + expense_increase,
      new_average_expense := new_total_expense / new_students
  in A - new_average_expense = 1 := by
  sorry

end average_expenditure_decrease_l410_410657


namespace sum_of_p_and_q_l410_410958

-- Definitions for points and collinearity condition
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := {x := 1, y := 3, z := -2}
def B : Point3D := {x := 2, y := 5, z := 1}
def C (p q : ℝ) : Point3D := {x := p, y := 7, z := q - 2}

def collinear (A B C : Point3D) : Prop :=
  ∃ (k : ℝ), B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) ∧ B.z - A.z = k * (C.z - A.z)

theorem sum_of_p_and_q (p q : ℝ) (h : collinear A B (C p q)) : p + q = 9 := by
  sorry

end sum_of_p_and_q_l410_410958


namespace problem_1_problem_2_l410_410948

-- Define the function f
def f (a x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Prove that the interval of monotonic decrease for f(x) given f(1) = 0 is (1, +∞)
theorem problem_1 (a : ℝ) (h : f a 1 = 0) : { x : ℝ | 1 < x } ⊆ { x | ∀ y, f a y < f a x } :=
sorry

-- Prove that the minimum value of integer a such that f(x) ≤ ax - 1 always holds for x is 2
theorem problem_2 : ∀ (a : ℝ), (∀ x : ℝ, f a x ≤ a * x - 1) → ∃ i : ℤ, (a = i) ∧ (i ≥ 2) :=
sorry

end problem_1_problem_2_l410_410948


namespace sequence_formula_l410_410067

noncomputable def sequence (n : ℕ) : ℕ := 3 * n - 1

theorem sequence_formula (a_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n > 0) 
  (S_n : ℕ → ℕ)
  (h2 : ∀ n, S_n n > 1)
  (h3 : ∀ n, 6 * S_n n = (a_n n + 1) * (a_n n + 2))
  (n : ℕ) (pos_n : 0 < n):
  a_n n = 3 * n - 1 := by
    sorry

end sequence_formula_l410_410067


namespace product_of_all_possible_t_l410_410401

-- Define the set of possible integer pairs (a, b) such that ab = 12.
def possible_pairs : Set (Int × Int) :=
  {(a, b) | a * b = 12}

-- Define the set of possible values of t = a + b for those pairs (a, b).
def possible_values_of_t : Set Int :=
  {t | ∃ (a b : Int), (a, b) ∈ possible_pairs ∧ t = a + b}

-- Lean 4 theorem to prove the product of all possible values of t.
theorem product_of_all_possible_t : 
  ∏ t in possible_values_of_t, t = 532224 :=
sorry

end product_of_all_possible_t_l410_410401


namespace cot_minus_tan_eq_five_sixths_l410_410125

theorem cot_minus_tan_eq_five_sixths (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 4) 
    (h3 : 1 / Real.sin θ - 1 / Real.cos θ = Real.sqrt 13 / 6) : 
    Real.cot θ - Real.tan θ = 5 / 6 := 
by 
  sorry

end cot_minus_tan_eq_five_sixths_l410_410125


namespace proof_problem_l410_410363

noncomputable def double_factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := (n + 2) * double_factorial n

theorem proof_problem :
  let S := ∑ i in finset.range 11 (λ i, 
    let n := i + 1 in
    let term := (double_factorial (2 * n - 1)) / ((2^n) * (n!)) in
    term) in
  let a := 8 in
  let b := 1 in
  (S.denom_factors.primes_product = 2^a * b) ∧ (b % 2 = 1) →
  (a * b / 10 = 0.8) :=
begin
  sorry
end

end proof_problem_l410_410363


namespace non_intersecting_segments_l410_410911

theorem non_intersecting_segments (n : ℕ) (points : Finset (ℝ × ℝ)) 
    (h_points : points.card = 2 * n) (h_collinear : ∀ p₁ p₂ p₃ ∈ points, 
    (p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) → 
    ¬ collinear ℝ (λ a : (ℝ × ℝ), (a.1, a.2)) ⟨p₁, p₂, p₃⟩)
    (colors : Finset (ℝ × ℝ) → Fin (2 * n) → Prop) 
    (h_colors : ∀ x, x ∈ points → (colors points x = 0 ∨ colors points x = 1)) :
    ∃ (segments : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)), 
    (segments.card = n) ∧ (∀ s ∈ segments, s.fst ∈ points ∧ s.snd ∈ points ∧ colors points s.fst ≠ colors points s.snd) ∧ 
    (∀ s₁ s₂ ∈ segments, s₁ ≠ s₂ → ¬(segments_intersect s₁ s₂)) :=
sorry

/-
Definitions needed:
- collinear ℝ f: a function that checks if three points are collinear in a 2D space.
- segments_intersect s1 s2: a function that checks if two line segments (defined by pairs of points) intersect.
- colors points x: a function that assigns colors to points, returning either 0 (red) or 1 (blue).
-/

end non_intersecting_segments_l410_410911


namespace average_next_five_consecutive_even_numbers_l410_410056

theorem average_next_five_consecutive_even_numbers (a b : ℕ) (h1 : b = 2a + 4) :
  let seq_b := [4a + 8, 4a + 10, 4a + 12, 4a + 14, 4a + 16]
  in (seq_b.sum / 5) = 4a + 12 :=
by sorry

end average_next_five_consecutive_even_numbers_l410_410056


namespace mask_arrangement_count_l410_410372

/-
Problem Statement: There are 7 types of masks and 4 people. Each person needs to collect data on at least one type of mask and a mask cannot be assigned to more than one person. Prove that the total number of different arrangements for this task is equal to 8400.
-/

noncomputable def num_arrangements : ℕ :=
  -- Calculation steps can be encapsulated in auxiliary definitions if necessary
  let ways_to_divide_masks := 
    -- Different ways to divide masks among people where each gets at least one mask
    (Nat.choose 7 4 + Nat.choose 7 3 * Nat.choose 4 2 + Nat.choose 7 1 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 / 3!)
  in ways_to_divide_masks * 4!

theorem mask_arrangement_count :
  num_arrangements = 8400 := by
  sorry

end mask_arrangement_count_l410_410372


namespace number_of_elements_in_A_inter_Z_l410_410517

open Set Int

def A : Set ℝ := {x | (x - 1)^2 < 3 * x + 7}

theorem number_of_elements_in_A_inter_Z :
  ∃ (elems : Finset ℤ), (∀ x, x ∈ elems ↔ x ∈ A ∩ Icc (-1) 6) ∧ elems.card = 6 :=
by
  sorry
 
end number_of_elements_in_A_inter_Z_l410_410517


namespace dawn_monthly_savings_l410_410016

-- Definitions for the conditions
def annual_salary := 48000
def months_in_year := 12
def saving_rate := 0.10

-- Define the monthly salary
def monthly_salary := annual_salary / months_in_year

-- Define the monthly savings
def monthly_savings := monthly_salary * saving_rate

-- The theorem to prove
theorem dawn_monthly_savings : monthly_savings = 400 :=
by
  -- Proof details skipped
  sorry

end dawn_monthly_savings_l410_410016


namespace max_value_sin_cos_l410_410385

theorem max_value_sin_cos (x : ℝ) : 
  let y := sin x * cos x + sin x + cos x in 
  y ≤ (1/2 + sqrt 2) := by 
  sorry

end max_value_sin_cos_l410_410385


namespace sum_c_n_is_3_pow_2016_l410_410915

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n - 1

-- Define the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 3^(n - 1)

-- Define the sequence {c_n} with given conditions
def c_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2 * 3^(n - 1)

-- The main proof problem
theorem sum_c_n_is_3_pow_2016 :
  (Finset.range 2016).sum (λ n, c_n (n + 1)) = 3^(2016) :=
by
  sorry

end sum_c_n_is_3_pow_2016_l410_410915


namespace xyz_product_neg4_l410_410979

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l410_410979


namespace percent_increase_stock_price_l410_410294

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem percent_increase_stock_price :
  percent_increase 28 29 ≈ 3.57 :=
by
  sorry

end percent_increase_stock_price_l410_410294


namespace product_xyz_l410_410972

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l410_410972


namespace factorial_division_identity_l410_410828

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410828


namespace smallest_positive_n_l410_410676

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410676


namespace factorial_div_eq_l410_410835
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410835


namespace local_symmetry_point_quadratic_range_of_m_l410_410096

-- 1. Prove that f(x) = ax^2 + bx - a has local symmetry points x = ±1 given a, b ∈ ℝ and a ≠ 0
theorem local_symmetry_point_quadratic (a b : ℝ) (h : a ≠ 0) : 
  ∃ x₀, x₀ ∈ { 1, -1 } ∧ ∀ x, f x = ax^2 + bx - a ∧ f (-x₀) = -f x₀ := sorry

-- 2. Determine the range of m such that the function f(x) = 4^x - m * 2^(n + 1) + m - 3 has a local symmetry point, yielding 1 - √3 ≤ m ≤ 2√2.
theorem range_of_m (m n x : ℝ) :
  (∃ x₀ ∈ ℝ, ∀ x, f x = 4^x - m * 2^(n + 1) + m - 3 ∧ f (-x₀) = -f x₀) ↔ 1 - real.sqrt 3 ≤ m ∧ m ≤ 2 * real.sqrt 2 := sorry

end local_symmetry_point_quadratic_range_of_m_l410_410096


namespace train_crossing_time_l410_410718

def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def crossing_time (length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph_to_mps speed_kmph
  length / speed_mps

theorem train_crossing_time (length : ℝ) (speed_kmph : ℝ) (length_95 : length = 95) (speed_214 : speed_kmph = 214) :
  crossing_time length speed_kmph ≈ 1.598 :=
by
  rw [length_95, speed_214]
  sorry

end train_crossing_time_l410_410718


namespace find_lambda_l410_410138

theorem find_lambda
  (λ : ℝ)
  (A B C : ℝ × ℝ)
  (hB : ∡BAC = 90)
  (hAB : A = (1, 2))
  (hAC : C = (3, λ)) :
  λ = 1 :=
by
  sorry

end find_lambda_l410_410138


namespace factorial_quotient_l410_410816

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410816


namespace parabola_properties_l410_410649

variable (P : ℝ × ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) (x y x1 y1 : ℝ)

noncomputable def parabola_eqn : Prop :=
  y^2 = 4 * x
  
noncomputable def focus_coords : Prop :=
  F = (1, 0)
  
noncomputable def trajectory_eqn : Prop :=
  y^2 = 2 * x - 1
  
theorem parabola_properties (P_on_parabola : P = (4, 4))
    (vertex_at_origin : (0, 0))
    (focus_on_x_axis : F = (1, 0)) :
  parabola_eqn 4 4 ∧
  focus_coords (1, 0) ∧
  trajectory_eqn ((x = (x1 + 1) / 2) ∧ (y = y1 / 2)) :=
by
  sorry

end parabola_properties_l410_410649


namespace sum_of_interior_numbers_in_eighth_row_l410_410523

theorem sum_of_interior_numbers_in_eighth_row :
  (let sum_in_row (n : Nat) := 2^(n - 1) - 2 in
  sum_in_row 6 = 30)
  → (let sum_in_row (n : Nat) := 2^(n - 1) - 2 in
  sum_in_row 8 = 126) :=
by
  intros
  sorry

end sum_of_interior_numbers_in_eighth_row_l410_410523


namespace max_value_2x_plus_y_l410_410063

def max_poly_value : ℝ :=
  sorry

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 
  2 * x + y ≤ 6 :=
sorry

example (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 2 * x + y = 6 
  ↔ x = 3 ∧ y = 0 :=
by exact sorry

end max_value_2x_plus_y_l410_410063


namespace triangle_side_length_l410_410549

theorem triangle_side_length (A B C : Type) [triangle A B C] 
  (angle_A : angle_A = 60)
  (BC : BC = 3)
  (AB_lt_AC : AB < AC) 
  : AB = 2 := 
by
  sorry

end triangle_side_length_l410_410549


namespace inventors_of_arabic_numerals_l410_410640

def arabic_numerals : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem inventors_of_arabic_numerals : "Indians" :=
by
  -- Numerical digits currently in use
  have digits : Set ℕ := arabic_numerals

  -- These numerical digits are referred to as Arabic numerals
  have arabic_digits := digits

  -- (Condition 1) Known as Arabic numerals
  have known_as_arabic : "Arabic numerals" = "Indians' invention" :=
    sorry 

  -- (Condition 2) Introduced to Europe by Arabic Mathematicians
  have introduced_by_arabs : "introduced to Europe by Arabic Mathematicians" := sorry

  -- The numerals were originated earlier in history
  have originated_earlier := "invented by ancient Indians"

  -- Verify the identification of inventors
  exact "Indians"

end inventors_of_arabic_numerals_l410_410640


namespace knights_switch_places_in_16_moves_l410_410591

-- Define conditions of the problem
def initial_placement_on_3x3_chessboard : Prop :=
  -- Define the initial placement of knights in corners as per the problem
  true -- Placeholder, detailed definitions should replace

def knight_moves_in_L_shape : Prop :=
  -- The knights move in a standard L-shape
  ∀ (start_pos end_pos : Position),
    is_knight_move start_pos end_pos → valid_move start_pos end_pos

def one_piece_per_square : Prop :=
  -- Only one piece can occupy a square at a time
  ∀ (pos : Position),
    piece_at pos → ∀ (pos2 : Position), pos ≠ pos2 → ¬(piece_at pos2)

-- Main theorem to prove the answer
theorem knights_switch_places_in_16_moves :
  initial_placement_on_3x3_chessboard ∧
  knight_moves_in_L_shape ∧
  one_piece_per_square →
  min_moves_to_switch_places = 16 := 
by
  sorry

end knights_switch_places_in_16_moves_l410_410591


namespace find_smallest_n_l410_410692

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410692


namespace initial_cupcakes_l410_410411

   theorem initial_cupcakes (X : ℕ) (condition : X - 20 + 20 = 26) : X = 26 :=
   by
     sorry
   
end initial_cupcakes_l410_410411


namespace perimeter_shaded_region_l410_410160

theorem perimeter_shaded_region (r: ℝ) (circumference: ℝ) (h1: circumference = 36) (h2: {x // x = 3 * (circumference / 6)}) : x = 18 :=
by
  sorry

end perimeter_shaded_region_l410_410160


namespace box_ratio_l410_410169

theorem box_ratio (h : ℤ) (l : ℤ) (w : ℤ) (v : ℤ)
  (H_height : h = 12)
  (H_length : l = 3 * h)
  (H_volume : l * w * h = 3888)
  (H_length_multiple : ∃ m, l = m * w) :
  l / w = 4 := by
  sorry

end box_ratio_l410_410169


namespace find_theta_l410_410093

noncomputable def f (A θ x : ℝ) : ℝ :=
  A * sin (x + θ) - cos (x / 2) * cos ((π / 6) - (x / 2))

theorem find_theta 
  (A : ℝ) 
  (θ : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : θ ∈ Ioo (-π) 0) 
  (h2 : x1 < x2) 
  (h3 : x2 < x3) 
  (h4 : x3 - x1 < 2 * π) 
  (h5 : f A θ x1 = f A θ x2) 
  (h6 : f A θ x2 = f A θ x3) : 
  θ = -2 * π / 3 :=
sorry

end find_theta_l410_410093


namespace random_variable_prob_l410_410194

theorem random_variable_prob (n : ℕ) (h : (3 : ℝ) / n = 0.3) : n = 10 :=
sorry

end random_variable_prob_l410_410194


namespace max_area_circle_between_parallel_lines_l410_410871

theorem max_area_circle_between_parallel_lines : 
  ∀ (l₁ l₂ : ℝ → ℝ → Prop), 
    (∀ x y, l₁ x y ↔ 3*x - 4*y = 0) → 
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 20 = 0) → 
  ∃ A, A = 4 * Real.pi :=
by 
  sorry

end max_area_circle_between_parallel_lines_l410_410871


namespace magician_red_marbles_taken_l410_410755

theorem magician_red_marbles_taken:
  ∃ R : ℕ, (20 - R) + (30 - 4 * R) = 35 ∧ R = 3 :=
by
  sorry

end magician_red_marbles_taken_l410_410755


namespace net_rate_of_pay_l410_410328

-- Define the conditions as constants and parameters
def hours_driven : ℝ := 3
def speed_mph : ℝ := 50
def miles_per_gallon : ℝ := 25
def pay_per_mile : ℝ := 0.60
def cost_per_gallon : ℝ := 2.50

-- Define total distance, gasoline usage, earnings, cost, net earnings, and net rate of pay
def total_distance := speed_mph * hours_driven
def gallons_used := total_distance / miles_per_gallon
def total_earnings := total_distance * pay_per_mile
def total_cost := gallons_used * cost_per_gallon
def net_earnings := total_earnings - total_cost
def net_rate_per_hour := net_earnings / hours_driven

-- Establish the theorem statement
theorem net_rate_of_pay : net_rate_per_hour = 25 := by
  sorry

end net_rate_of_pay_l410_410328


namespace glove_problem_l410_410652

theorem glove_problem : 
  let total_ways := nat.choose 8 4
  let no_pair_ways := 2 * 2 * 2 * 2
  total_ways - no_pair_ways = 54 :=
by
  sorry

end glove_problem_l410_410652


namespace sin2theta_eq_three_fourths_l410_410058

theorem sin2theta_eq_three_fourths (θ : ℝ) (h : sin θ + cos θ = sqrt 7 / 2) : sin (2 * θ) = 3 / 4 :=
sorry

end sin2theta_eq_three_fourths_l410_410058


namespace total_selling_price_l410_410760

theorem total_selling_price 
  (cost_tv : ℝ) 
  (cost_dvd : ℝ) 
  (profit_percentage : ℝ) 
  (total_cost_price : ℝ := cost_tv + cost_dvd) 
  (profit_amount : ℝ := profit_percentage / 100 * total_cost_price) 
  (total_selling_price : ℝ := total_cost_price + profit_amount) :
  cost_tv = 16000 →
  cost_dvd = 6250 →
  profit_percentage = 60.00000000000001 →
  total_selling_price ≈ 35600 :=
by
  intros h_cost_tv h_cost_dvd h_profit_percentage
  rw [h_cost_tv, h_cost_dvd, h_profit_percentage]
  norm_num
  sorry

end total_selling_price_l410_410760


namespace surface_area_of_circumscribed_sphere_l410_410518

theorem surface_area_of_circumscribed_sphere (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 2 * Real.sqrt 6) :
  let r := Real.sqrt ((a^2 + b^2 + c^2) / 4) in
  4 * Real.pi * r^2 = 49 * Real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_l410_410518


namespace find_z_l410_410136

/-
Given:
x = 22
y = 13
The ratio of boys who went down the slide to boys who watched is 5:3.

Prove that the number of boys who watched but didn’t go down the slide, z, is 21.
-/

def num_boys_went_down_slide (x y : Nat) : Nat := x + y

def ratio_watched_slide (W Z : Nat) : Prop := 5 * Z = 3 * W

theorem find_z (x y : Nat) (h1 : x = 22) (h2 : y = 13) (h3 : ratio_watched_slide (num_boys_went_down_slide x y) Z) : Z = 21 :=
by
  have W := num_boys_went_down_slide x y
  rw [h1, h2] at W
  have h4 : 5 * Z = 3 * W := h3
  -- Apply given values
  rw [W] at h4
  -- skipping proof with sorry
  sorry

end find_z_l410_410136


namespace volume_of_solid_of_revolution_l410_410296

def f (x : ℝ) : ℝ := x ^ 2

theorem volume_of_solid_of_revolution :
  let a := 1
  let b := 2
  let V := Real.pi * (∫ x in a..b, (f x) ^ 2 - 1) in
  V = (26 / 5) * Real.pi :=
by
  sorry

end volume_of_solid_of_revolution_l410_410296


namespace smallest_n_satisfies_conditions_l410_410670

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410670


namespace line_AB_equation_l410_410416

theorem line_AB_equation : 
  ∀ (M P : Point) (l : Line), 
    Circle M (x^2 + y^2 - 2*x - 2*y - 2 = 0) ∧ 
    Line l (2*x + y + 2 = 0) ∧ 
    P ∈ l ∧
    (∃ A B : Point, Tangent A P ∧ Tangent B P ∧ 
                    A ∈ Circle M ∧ B ∈ Circle M ∧
                    (∃ AB : Line, Minimize |PM| * |AB|)) → 
  is_equation_of_line AB (2*x + y + 1 = 0) := 
sorry

end line_AB_equation_l410_410416


namespace max_value_3x_4y_l410_410430

noncomputable def y_geom_mean (x y : ℝ) : Prop :=
  y^2 = (1 - x) * (1 + x)

theorem max_value_3x_4y (x y : ℝ) (h : y_geom_mean x y) : 3 * x + 4 * y ≤ 5 :=
sorry

end max_value_3x_4y_l410_410430


namespace cos_90_eq_zero_l410_410003

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410003


namespace cos_90_eq_zero_l410_410007

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410007


namespace no_solution_eqn_l410_410128

theorem no_solution_eqn (m : ℝ) :
  ¬ ∃ x : ℝ, (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) = -1 ↔ m = 1 :=
by
  sorry

end no_solution_eqn_l410_410128


namespace original_classes_l410_410527

theorem original_classes (x : ℕ) (h1 : 280 % x = 0) (h2 : 585 % (x + 6) = 0) : x = 7 :=
sorry

end original_classes_l410_410527


namespace standing_arrangements_l410_410268

   theorem standing_arrangements :
     let steps := 7 in
     let people := 3 in
     let max_per_step := 2 in
     (∑ (i in combinations (finset.range steps) people), 
      (if ((i.card = people) && (∀ step in i, 1 ≤ step ∧ step ≤ max_per_step)) then 
       1 
       else if ((i.card < people) && (∑ step in i, if i.count step = max_per_step then 2 else 1) = people) then 
       1 
       else 
       0)) = 336 :=
   by
     sorry
   
end standing_arrangements_l410_410268


namespace solve_for_a_l410_410086

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 1) else -x * (x + 1) - x

theorem solve_for_a : is_odd_function f → (∃ a : ℝ, f a = -2 ∧ a = -2) :=
by
  intros h_odd
  use -2
  split
  -- Proof that f(-2) = -2 skipped
  sorry
  -- Proof that a = -2 skipped
  refl

end solve_for_a_l410_410086


namespace distance_between_skew_medians_l410_410868

noncomputable def distance_skew_medians (a : ℝ) : ℝ × ℝ := 
(a * real.sqrt (2 / 35), a / real.sqrt 10)

theorem distance_between_skew_medians (a : ℝ) :
  ∃ d1 d2, d1 = a * real.sqrt (2 / 35) ∧ d2 = a / real.sqrt 10 :=
by
  let distances := distance_skew_medians a
  use distances.1, distances.2
  split
  { refl }
  { refl }

end distance_between_skew_medians_l410_410868


namespace shaded_region_area_l410_410598

-- Define the quarter circle pattern's radius
def radius : ℝ := 1.0
-- Define the length of the pattern
def length : ℝ := 3.0

-- Define the number of quarter circles
def num_of_quarter_circles : ℕ := 3

-- Define the correct area of the shaded region.
def expected_area : ℝ := (3/4) * Real.pi

-- Theorem statement
theorem shaded_region_area :
  let r := radius in
  let l := length in
  let n := num_of_quarter_circles in
  let area := expected_area in
  area = (n * (1/4) * Real.pi) :=
by sorry

end shaded_region_area_l410_410598


namespace mechanic_hours_l410_410337

theorem mechanic_hours (h : ℕ) (labor_cost_per_hour parts_cost total_bill : ℕ) 
  (H1 : labor_cost_per_hour = 45) 
  (H2 : parts_cost = 225) 
  (H3 : total_bill = 450) 
  (H4 : labor_cost_per_hour * h + parts_cost = total_bill) : 
  h = 5 := 
by
  sorry

end mechanic_hours_l410_410337


namespace triangle_ratio_l410_410162

variables (A B C D E F : Type)
variables (AB AC BC AD BD DC AE EB CF FA : ℝ)
variables (H1 : AB = AC + BC)
variables (H2 : D ∈ BC)
variables (H3 : E ∈ AB)
variables (H4 : F ∈ AC)
variables (H5 : DE / EB = AD / DB) -- Angle bisector theorem for angle ADB
variables (H6 : DF / FC = AD / DC) -- Angle bisector theorem for angle ADC

theorem triangle_ratio (H1 : AB = AC + BC) (H2 : D ∈ BC) (H3 : E ∈ AB)
                       (H4 : F ∈ AC) (H5 : DE / EB = AD / DB) (H6 : DF / FC = AD / DC) :
                       AE / EB * BD / DC * CF / FA = 1 :=
sorry

end triangle_ratio_l410_410162


namespace product_of_constants_l410_410397

theorem product_of_constants (t : ℤ) (a b : ℤ) (h1 : a * b = 12)
  (h2 : t = a + b) :
  ∃ p : ℤ, (∀ t, (∃ a b : ℤ, a * b = 12 ∧ t = a + b) → t) ∧ p = -527776 :=
sorry

end product_of_constants_l410_410397


namespace borrow_years_l410_410340

/-- A person borrows Rs. 5000 at 4% p.a simple interest and lends it at 6% p.a simple interest.
His gain in the transaction per year is Rs. 100. Prove that he borrowed the money for 1 year. --/
theorem borrow_years
  (principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain : ℝ)
  (interest_paid_per_year : ℝ)
  (interest_earned_per_year : ℝ) :
  (principal = 5000) →
  (borrow_rate = 0.04) →
  (lend_rate = 0.06) →
  (gain = 100) →
  (interest_paid_per_year = principal * borrow_rate) →
  (interest_earned_per_year = principal * lend_rate) →
  (interest_earned_per_year - interest_paid_per_year = gain) →
  1 = 1 := 
by
  -- Placeholder for the proof
  sorry

end borrow_years_l410_410340


namespace find_smallest_n_l410_410691

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410691


namespace dot_product_vectors_l410_410012

def vector1 : ℝ × ℝ × ℝ := (1/2, -3/4, -5/6)
def vector2 : ℝ × ℝ × ℝ := (-3/2, 1/4, 2/3)

theorem dot_product_vectors :
  let dot_product := 
    (vector1.1 * vector2.1) + (vector1.2 * vector2.2) + (vector1.3 * vector2.3)
  in dot_product = -215/144 :=
by 
  sorry

end dot_product_vectors_l410_410012


namespace line_AB_minimized_condition_l410_410419

theorem line_AB_minimized_condition :
  ∀ (x y : ℝ) (P : ℝ × ℝ),
  (x^2 + y^2 - 2*x - 2*y - 2 = 0) →
  (2*P.1 + P.2 + 2 = 0) →
  let M: ℝ × ℝ := (1, 1),
      PM := dist P M,
      AB := dist P (0, 1),
      minimized : PM * AB → (2*x + y + 1 = 0)
  (2 * x + y + 1 = 0) :=
begin
  sorry
end

end line_AB_minimized_condition_l410_410419


namespace parallel_sufficient_not_necessary_l410_410729

def line := Type
def parallel (l1 l2 : line) : Prop := sorry
def in_plane (l : line) : Prop := sorry

theorem parallel_sufficient_not_necessary (a β : line) :
  (parallel a β → ∃ γ, in_plane γ ∧ parallel a γ) ∧
  ¬( (∃ γ, in_plane γ ∧ parallel a γ) → parallel a β ) :=
by sorry

end parallel_sufficient_not_necessary_l410_410729


namespace largest_number_no_repetition_digits_1_2_3_l410_410663

theorem largest_number_no_repetition_digits_1_2_3 :
  ∃ n ∈ {321, 21^3, 3^21, 2^31}, n = 3^21 := by
  sorry

end largest_number_no_repetition_digits_1_2_3_l410_410663


namespace additional_movies_needed_l410_410338

-- Definitions based on given conditions
def current_movies : ℕ := 9
def target_odd_group_sum : ℕ := 10

-- Proof statement
theorem additional_movies_needed : current_movies % 2 = 1 ∧ ∃ k : ℕ, target_odd_group_sum = current_movies + k ∧ target_odd_group_sum % 2 = 0 ∧ (target_odd_group_sum / 2) % 2 = 1 :=
by
  existsi (1 : ℕ)
  split
  -- current_movies % 2 = 1 
  sorry
  split
  -- target_odd_group_sum = current_movies + 1
  sorry
  split
  -- target_odd_group_sum % 2 = 0
  sorry
  -- (target_odd_group_sum / 2) % 2 = 1
  sorry

end additional_movies_needed_l410_410338


namespace area_of_R2_l410_410601

-- Define the dimensions of the rectangles and triangle
def length_R1 : ℝ := 3
def width_R1 : ℝ := 9
def base_T : ℝ := 3
def height_T : ℝ := 4.5
def diagonal_R2 : ℝ := 18

-- Hypothesis for similarity of rectangles
def aspect_ratio (a b : ℝ) := a / b

-- Main theorem statement
theorem area_of_R2 :
  (∀ (R1 R2 : Type) (a b diag : ℝ), 
      length_R1 = a ∧ width_R1 = b ∧ diagonal_R2 = diag ∧ aspect_ratio b a = 3 → 
      width_R1 * length_R1 = 27 ∧ ((3 * diag / √(diag^2 / (a^2 + b^2))) * (diag / √(diag^2 / (a^2 + b^2)))) = 97.2) :=
begin
  -- Use given information to match assumptions for the proof
  sorry  -- The proof needs to be filled in
end

end area_of_R2_l410_410601


namespace Bretschneider_theorem_l410_410790

-- Definitions of the variables involved
variables {a b c d m n : ℝ}
variables {A C : ℝ}

-- Bretschneider's theorem statement in Lean 4
theorem Bretschneider_theorem (a b c d m n : ℝ) (A C : ℝ) :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * real.cos (A + C) := 
sorry

end Bretschneider_theorem_l410_410790


namespace range_of_m_l410_410465

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 1

theorem range_of_m (a x1 x2 m : ℝ) (h : a ≤ 1) (hx1x2 : x1 ≠ x2) (hderiv : f a x1 = f a x2) :
  m ∈ set.Iic 1 ↔ f a (x1 + x2) ≥ m :=
sorry

end range_of_m_l410_410465


namespace find_f6_l410_410628

variable {R : Type} [LinearOrderedField R]

def f : R → R := sorry

theorem find_f6 (h1 : ∀ x y : R, f (x - y) = f x * f y) (h2 : ∀ x : R, f x ≠ 0) : f 6 = 1 :=
sorry

end find_f6_l410_410628


namespace ω_trajectory_l410_410241

open Complex Real

noncomputable def z (θ : ℝ) : ℂ := cos θ - complex.I * (sin θ - 1)

def ω (θ : ℝ) := z θ ^ 2 - 2 * complex.I * z θ

theorem ω_trajectory (θ : ℝ) (hθ : θ ∈ Ioo (π / 2) π) : 
  ∃ (x y : ℝ), ω θ = x + complex.I * y ∧ 
  (x - 1)^2 + y^2 = 1 ∧ 
  x > 0 ∧ 
  y > 0 := 
sorry

end ω_trajectory_l410_410241


namespace no_partition_for_n_gt_2_l410_410371

theorem no_partition_for_n_gt_2 :
  ∀ n : ℕ, n > 2 → ¬ ∃ (A : fin n → set ℕ),
  (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧
  (∀ k : ℕ, (∑ i in finset.erase finset.univ k, (λ j, (A j).choose (λ x, x > 0) : ℕ)) ∈ A k) :=
by
  sorry

end no_partition_for_n_gt_2_l410_410371


namespace pure_imaginary_fraction_eq_two_l410_410446

theorem pure_imaginary_fraction_eq_two (a : ℝ) (h : (a + Complex.i) / (1 - 2 * Complex.i) = Complex.i * (2 * a + 1) / 5) : a = 2 :=
sorry

end pure_imaginary_fraction_eq_two_l410_410446


namespace percentage_of_books_returned_l410_410759

theorem percentage_of_books_returned
  (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) (returned_books_percentage : ℚ) 
  (h1 : initial_books = 75) 
  (h2 : end_books = 68) 
  (h3 : loaned_books = 20)
  (h4 : returned_books_percentage = (end_books - (initial_books - loaned_books)) * 100 / loaned_books):
  returned_books_percentage = 65 := 
by
  sorry

end percentage_of_books_returned_l410_410759


namespace number_of_ways_to_distribute_balls_l410_410505

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l410_410505


namespace min_length_M_inter_N_l410_410104

def setM (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3 / 4}
def setN (n : ℝ) : Set ℝ := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
def setP : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem min_length_M_inter_N (m n : ℝ) 
  (hm : 0 ≤ m ∧ m + 3 / 4 ≤ 1) 
  (hn : 1 / 3 ≤ n ∧ n ≤ 1) : 
  let I := (setM m ∩ setN n)
  ∃ Iinf Isup : ℝ, I = {x | Iinf ≤ x ∧ x ≤ Isup} ∧ Isup - Iinf = 1 / 12 :=
  sorry

end min_length_M_inter_N_l410_410104


namespace smallest_n_l410_410684

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410684


namespace number_of_cupcakes_l410_410339

theorem number_of_cupcakes (total gluten_free vegan gluten_free_vegan non_vegan : ℕ) 
    (h1 : gluten_free = total / 2)
    (h2 : vegan = 24)
    (h3 : gluten_free_vegan = vegan / 2)
    (h4 : non_vegan = 28)
    (h5 : gluten_free_vegan = gluten_free / 2) :
    total = 52 :=
by
  sorry

end number_of_cupcakes_l410_410339


namespace eighteen_is_abundant_l410_410757

def is_proper_divisor (n d : ℕ) : Prop :=
d < n ∧ n % d = 0

def sum_proper_divisors (n : ℕ) : ℕ :=
(nat.divisors n).filter (is_proper_divisor n) |> list.sum

def is_abundant (n : ℕ) : Prop :=
n < sum_proper_divisors n

theorem eighteen_is_abundant : is_abundant 18 :=
by sorry

end eighteen_is_abundant_l410_410757


namespace find_f7_l410_410081

noncomputable def f : ℝ → ℝ := sorry

-- The conditions provided in the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom function_in_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The final proof goal
theorem find_f7 : f 7 = -2 :=
by sorry

end find_f7_l410_410081


namespace max_distance_ellipse_line_l410_410254

theorem max_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 4) = 1}
  let line := {p : ℝ × ℝ | p.1 + 2 * p.2 = sqrt 2 }
  let distance := λ p q : ℝ × ℝ, sqrt ((p.1 - q .1)^2 + (p.2 - q.2)^2)
  let line_distance := λ l1 l2 : ℝ × ℝ → Prop, abs ((l1 ({p : ℝ × ℝ | p.1 = - 2 * p.2}).1 - l2( {p : ℝ × ℝ | p. 1 = - 2 * p.2 }).1 ) / sqrt (1^2 + 2^2)) 
  (line_distance {p : ℝ × ℝ | p.1 + 2 * p.2 = sqrt 2} {p : ℝ × ℝ | p.1 + 2 * p.2 = -4 * sqrt 2}) 
 = sqrt 10 := 
sorry

end max_distance_ellipse_line_l410_410254


namespace remaining_adults_fed_l410_410746

theorem remaining_adults_fed 
  (cans : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (remaining_cans : ℕ)
  (remaining_adults : ℕ) :
  (adults_per_can = 4) →
  (children_per_can = 6) →
  (initial_cans = 7) →
  (children_fed = 18) →
  (remaining_cans = initial_cans - children_fed / children_per_can) →
  (remaining_adults = remaining_cans * adults_per_can) →
  remaining_adults = 16 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end remaining_adults_fed_l410_410746


namespace complex_number_in_first_quadrant_l410_410082

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 1 + 2 * i
noncomputable def conjz : ℂ := complex.conj z

theorem complex_number_in_first_quadrant :
  let a := z + i * conjz in
  0 < a.re ∧ 0 < a.im :=
by 
  sorry

end complex_number_in_first_quadrant_l410_410082


namespace not_f_x_plus_2_eq_neg_f_x_f_is_odd_function_center_of_symmetry_f_not_center_of_symmetry_f_prime_l410_410934

-- Define the function and its conditions
def f : ℝ → ℝ := sorry
-- Condition 1: The function f(x) and its derivative f'(x) have a domain of ℝ, implicitly assumed.
-- Condition 2
axiom h1 : ∀ x : ℝ, f(x + 2) = - (1 / f(x))
-- Condition 3
axiom h2 : ∀ x : ℝ, f(x + 2) + f(-x + 6) = 0

-- Prove the statements 
theorem not_f_x_plus_2_eq_neg_f_x : ¬ (∀ x : ℝ, f(x + 2) = -f(x)) :=
sorry

theorem f_is_odd_function : ∀ x : ℝ, f(-x) = -f(x) :=
sorry

theorem center_of_symmetry_f : f(2) = 0 :=
sorry

theorem not_center_of_symmetry_f_prime : ¬ (f'(-2) = 0) :=
sorry

end not_f_x_plus_2_eq_neg_f_x_f_is_odd_function_center_of_symmetry_f_not_center_of_symmetry_f_prime_l410_410934


namespace ellipse_and_line_equation_l410_410917

-- Define the conditions of the ellipse with its foci and a passing point
def foci1 : ℝ × ℝ := (-(real.sqrt 3), 0)
def foci2 : ℝ × ℝ := (real.sqrt 3, 0)
def passing_point : ℝ × ℝ := (real.sqrt 3, 1 / 2)

-- Define the standard form of the ellipse equation
def ellipse_eq (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- Define the condition for the minimum area of triangle ABD with the line equation
def is_line_eq (k : ℝ) : Prop := (y = k * x)

theorem ellipse_and_line_equation (a b : ℝ) :
  a^2 = 4 → b^2 = 1 → 
  ellipse_eq (real.sqrt 3) (1 / 2) a b →
  ∃ k, (k = 1 ∨ k = -1) ∧ 
  (∀ D ∈ ellipse_eq, |AD| = |BD| → line_eq y x (k * x)) := 
sorry

end ellipse_and_line_equation_l410_410917


namespace initial_apples_l410_410111

theorem initial_apples (classmates : ℕ) (apples_each : ℕ) (leftovers : ℕ) : 
  (classmates = 3) → (apples_each = 5) → ∃ initial_apples : ℕ, initial_apples = (classmates * apples_each + leftovers) :=
by
  intros h1 h2
  use (3 * 5 + leftovers)
  rw [h1, h2]
  sorry

end initial_apples_l410_410111


namespace prime_divisors_of_17325_correct_sum_of_prime_divisors_of_17325_correct_l410_410115

noncomputable def prime_divisors_of_17325 : ℕ := 4
noncomputable def sum_of_prime_divisors_of_17325 : ℕ := 26

theorem prime_divisors_of_17325_correct :
  find_prime_divisors_count 17325 = prime_divisors_of_17325 :=
sorry

theorem sum_of_prime_divisors_of_17325_correct :
  find_sum_of_prime_divisors 17325 = sum_of_prime_divisors_of_17325 :=
sorry

end prime_divisors_of_17325_correct_sum_of_prime_divisors_of_17325_correct_l410_410115


namespace five_cubic_yards_to_cubic_meters_l410_410483

noncomputable def cubicMeters_in_five_cubicYards : ℝ :=
let yards_to_meters := 0.9144 in
let cubic_conversion := yards_to_meters ^ 3 in
5 * cubic_conversion

theorem five_cubic_yards_to_cubic_meters : 
  cubicMeters_in_five_cubicYards = 3.82277 :=
by
  sorry

end five_cubic_yards_to_cubic_meters_l410_410483


namespace cookie_radius_l410_410617

theorem cookie_radius (x y : ℝ) : 
    x^2 + y^2 + 36 = 6 * x + 12 * y → 
    ∃ r : ℝ, r = 3 :=
by
  intro h
  use 3
  sorry

end cookie_radius_l410_410617


namespace cos_90_eq_zero_l410_410000

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410000


namespace product_xyz_equals_zero_l410_410969

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l410_410969


namespace range_of_a_l410_410999

theorem range_of_a (a : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x <= 1, f x = x^2 - 2 * a * x + a + 2)
    (h2 : ∀ x > 1, f x = x^(2 * a - 6))
    (h3 : ∀ (x1 x2 : ℝ), x1 <= x2 → f x1 >= f x2) :
    1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l410_410999


namespace factorial_div_eq_l410_410837
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410837


namespace compute_f_at_5_l410_410121

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (10 ^ x) = x

theorem compute_f_at_5 : f 5 = Real.log 5 / Real.log 10 :=
by
  sorry

end compute_f_at_5_l410_410121


namespace sector_angle_removed_l410_410325

theorem sector_angle_removed (R r V : ℝ) (volume_eq : V = 500 * Real.pi)
  (radius_original_circle : R = 15)
  (radius_cone : r = 10) :
  let hieght_cone := 15 in
  let slant_height_cone := 5 * Real.sqrt 13 in
  let full_circumference := 30 * Real.pi in
  let base_circumference := 20 * Real.pi in
  let central_angle := (base_circumference / full_circumference) * 360 in
  360 - central_angle = 120 := 
by
  sorry

end sector_angle_removed_l410_410325


namespace complement_of_union_l410_410476

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) * (x + 1) ≤ 0 }
def B : Set ℝ := { x | 0 ≤ x ∧ x < 3 }

theorem complement_of_union :
  Set.compl (A ∪ B) = { x : ℝ | x < -1 } ∪ { x | x ≥ 3 } := by
  sorry

end complement_of_union_l410_410476


namespace brian_read_chapters_l410_410351

theorem brian_read_chapters :
  let book1 := 20
  let book2 := 15
  let book3 := 15
  let total_chapters := book1 + book2 + book3
  let book4 := total_chapters / 2
  total_chapters + book4 = 120 :=
by
  -- definitions
  let book1 := 20
  let book2 := 15
  let book3 := 15
  let total_chapters := book1 + book2 + book3
  let book4 := total_chapters / 2
  -- This is the point we need to prove
  have h : total_chapters + book4 = 120 := sorry,
  exact h

end brian_read_chapters_l410_410351


namespace alpha_second_quadrant_l410_410130

theorem alpha_second_quadrant (α : ℝ) (h1 : sin α > 0) (h2 : sin(2 * α) < 0) : 
    α ∈ set.Icc (π / 2) π :=
begin
  sorry
end

end alpha_second_quadrant_l410_410130


namespace reporters_percentage_l410_410861

-- Define the total number of reporters
def total_reporters : ℕ := 100

-- 30% of the reporters cover local politics
def cover_local_politics := 0.3 * (total_reporters : ℝ)

-- 60% of the reporters do not cover politics
def not_cover_politics := 0.6 * (total_reporters : ℝ)

-- Calculate the reporters covering politics
def cover_politics := (total_reporters : ℝ) - not_cover_politics

-- Calculate the reporters covering politics but not local politics
def politics_not_local_politics := cover_politics - cover_local_politics

-- Calculate the percentage of reporters covering politics but not local politics
def percentage_politics_not_local_politics := (politics_not_local_politics / cover_politics) * 100

-- Statement to prove the percentage of reporters who cover politics but do not cover local politics is 25%
theorem reporters_percentage :
  percentage_politics_not_local_politics = 25 := 
by
  sorry

end reporters_percentage_l410_410861


namespace factorial_division_identity_l410_410834

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410834


namespace symmetric_inverse_function_l410_410451

theorem symmetric_inverse_function (f : ℝ → ℝ) (h : ∀ x > 0, f (2^(x+1)) = x):
  ∀ x > 2, f x = log 2 x - 1 :=
by
  sorry

end symmetric_inverse_function_l410_410451


namespace dimensions_of_triangle_from_square_l410_410769

theorem dimensions_of_triangle_from_square :
  ∀ (a : ℝ) (triangle : ℝ × ℝ × ℝ), 
    a = 10 →
    triangle = (a, a, a * Real.sqrt 2) →
    triangle = (10, 10, 10 * Real.sqrt 2) :=
by
  intros a triangle a_eq triangle_eq
  -- Proof
  sorry

end dimensions_of_triangle_from_square_l410_410769


namespace product_of_factors_l410_410406

theorem product_of_factors :
  (∏ (t : ℤ) in { t | ∃ a b : ℤ, a * b = 12 ∧ t = a + b }, t) = -530144 :=
by sorry

end product_of_factors_l410_410406


namespace friend_charge_per_animal_l410_410585

-- Define the conditions.
def num_cats := 2
def num_dogs := 3
def total_payment := 65

-- Define the total number of animals.
def total_animals := num_cats + num_dogs

-- Define the charge per animal per night.
def charge_per_animal := total_payment / total_animals

-- State the theorem.
theorem friend_charge_per_animal : charge_per_animal = 13 := by
  -- Proof goes here.
  sorry

end friend_charge_per_animal_l410_410585


namespace train_cross_bridge_time_l410_410961

theorem train_cross_bridge_time :
  ∀ (train_length bridge_length : ℕ) (train_speed_kmph : ℚ),
  train_length = 110 →
  bridge_length = 140 →
  train_speed_kmph = 60 →
  let total_distance := train_length + bridge_length
  let conversion_factor := 1000 / 3600
  let train_speed_mps := train_speed_kmph * conversion_factor
  let time := total_distance / train_speed_mps
  abs (time - 15) < 1 :=
by
  intro train_length bridge_length train_speed_kmph
  intros h1 h2 h3
  let total_distance := train_length + bridge_length
  let conversion_factor := 1000 / 3600
  let train_speed_mps := train_speed_kmph * conversion_factor
  let time := total_distance / train_speed_mps
  have h4 : abs (time - 15) < 1, from sorry
  exact h4

end train_cross_bridge_time_l410_410961


namespace handshakes_among_6_people_l410_410513

theorem handshakes_among_6_people (n : ℕ) (h : n = 6) : (nat.choose n 2) = 15 := by
  -- Since we do not require the proof, we use sorry to indicate it
  sorry

end handshakes_among_6_people_l410_410513


namespace cos_90_eq_zero_l410_410010

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410010


namespace number_of_rotations_l410_410710

theorem number_of_rotations (circumference distance : ℝ) (h₁ : circumference = 1.5) (h₂ : distance = 900) :
  distance / circumference = 600 :=
by {
  rw [h₁, h₂],
  norm_num,
}

end number_of_rotations_l410_410710


namespace smallest_n_45_l410_410687

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410687


namespace sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410319

theorem sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer :
  (sum (λ d, d) (filter (λ d, 5 = nat.digits_length 8 d) (range 2^13 2^16))) = 58 := sorry

end sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410319


namespace linear_relationship_selling_price_maximize_profit_l410_410748

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end linear_relationship_selling_price_maximize_profit_l410_410748


namespace leg_length_of_isosceles_right_triangle_l410_410634

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * real.sqrt 2

def median_to_hypotenuse (a b c m : ℝ) : Prop :=
  is_isosceles_right_triangle a b c ∧ m = c / 2

-- The proof problem statement
theorem leg_length_of_isosceles_right_triangle (a b c m : ℝ) (h1 : is_isosceles_right_triangle a b c)
  (h2 : median_to_hypotenuse a b c m) (h3 : m = 12) : a = 12 * real.sqrt 2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l410_410634


namespace xyz_product_value_l410_410990

variables {x y z : ℝ}

def condition1 : Prop := x + 2 / y = 2
def condition2 : Prop := y + 2 / z = 2

theorem xyz_product_value 
  (h1 : condition1) 
  (h2 : condition2) : 
  x * y * z = -2 := 
sorry

end xyz_product_value_l410_410990


namespace part1_part2_l410_410925

variable (a b c : ℝ)
hypothesis (ha : a > 0)
hypothesis (hb : b > 0)
hypothesis (hc : c > 0)
hypothesis (habc : a + b + c = 4)

theorem part1 : a^2 + b^2 / 4 + c^2 / 9 >= 8 / 7 :=
by { sorry }

theorem part2 : 1 / (a + c) + 1 / (a + b) + 1 / (b + c) >= 9 / 8 :=
by { sorry }

end part1_part2_l410_410925


namespace max_value_sin_cos_expression_l410_410880

-- Define the variables and the expressions

def max_trig_expression (x y z : ℝ) : ℝ :=
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))

-- State the theorem to find the maximum value of the given expression.

theorem max_value_sin_cos_expression : ∀ x y z : ℝ, max_trig_expression x y z ≤ 4.5 :=
by {
  sorry -- This is where the proof would go
}

end max_value_sin_cos_expression_l410_410880


namespace field_ratio_l410_410638

theorem field_ratio (w : ℝ) (h : ℝ) (pond_len : ℝ) (field_len : ℝ) 
  (h1 : pond_len = 8) 
  (h2 : field_len = 112) 
  (h3 : w > 0) 
  (h4 : field_len = w * h) 
  (h5 : pond_len * pond_len = (1 / 98) * (w * h * h)) : 
  field_len / h = 2 := 
by 
  sorry

end field_ratio_l410_410638


namespace simple_interest_double_in_4_years_interest_25_percent_l410_410774

theorem simple_interest_double_in_4_years_interest_25_percent :
  ∀ {P : ℕ} (h : P > 0), ∃ (R : ℕ), R = 25 ∧ P + P * R * 4 / 100 = 2 * P :=
by
  sorry

end simple_interest_double_in_4_years_interest_25_percent_l410_410774


namespace factorial_div_eq_l410_410841
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410841


namespace max_min_values_f_l410_410092

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem max_min_values_f : 
  ∃ x_min x_max ∈ set.Icc 1 Real.exp 1, 
    (∀ x ∈ set.Icc 1 Real.exp 1, f x_min ≤ f x) ∧ 
    (∀ x ∈ set.Icc 1 Real.exp 1, f x ≤ f x_max) ∧ 
    f 1 = 1 ∧ 
    f Real.exp 1 = Real.exp 1 ^ 2 + 1 :=
by
  sorry

end max_min_values_f_l410_410092


namespace soda_cans_purchasable_l410_410226

theorem soda_cans_purchasable (S Q : ℕ) (t D : ℝ) (hQ_pos : Q > 0) :
    let quarters_from_dollars := 4 * D
    let total_quarters_with_tax := quarters_from_dollars * (1 + t)
    (total_quarters_with_tax / Q) * S = (4 * D * S * (1 + t)) / Q :=
sorry

end soda_cans_purchasable_l410_410226


namespace table_sum_row_col_eq_l410_410151

theorem table_sum_row_col_eq (M K : ℕ) (a : ℕ → ℕ → ℝ)
  (h_row_sum : ∀ i, (finset.range K).sum (λ j, a i j) = 1)
  (h_col_sum : ∀ j, (finset.range M).sum (λ i, a i j) = 1) :
  M = K :=
sorry

end table_sum_row_col_eq_l410_410151


namespace max_value_l410_410886

/-- 
Proof of the maximum value of the expression 
(sin x + sin 2y + sin 3z) * (cos x + cos 2y + cos 3z)
-/
theorem max_value (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_l410_410886


namespace final_expression_in_simplest_form_l410_410117

variable (x : ℝ)

theorem final_expression_in_simplest_form : 
  ((3 * x + 6 - 5 * x + 10) / 5) = (-2 / 5) * x + 16 / 5 :=
by
  sorry

end final_expression_in_simplest_form_l410_410117


namespace smallest_positive_n_l410_410674

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410674


namespace base_b_digits_l410_410302

theorem base_b_digits (b : ℕ) : b^4 ≤ 500 ∧ 500 < b^5 → b = 4 := by
  intro h
  sorry

end base_b_digits_l410_410302


namespace savings_per_month_l410_410018

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end savings_per_month_l410_410018


namespace xyz_product_neg4_l410_410978

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l410_410978


namespace limit_derivative_l410_410185

variable {α : Type*} [LinearOrderedField α] {f : α → α} {x₀ : α}

theorem limit_derivative (h : DifferentiableAt ℝ f x₀) :
  filter.tendsto (λ Δx, (f (x₀ + 2*Δx) - f x₀) / Δx) (nhds 0) (nhds (2 * (Deriv f x₀))) :=
sorry

end limit_derivative_l410_410185


namespace cos_90_eq_zero_l410_410011

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410011


namespace total_weight_in_grams_l410_410150

-- Definitions of the conversions and conditions

def ancient_liang_per_jin := 16
def modern_liang_per_jin := 10
def grams_per_ancient_jin := 600
def grams_per_modern_jin := 500

def total_jin := 5
def total_liang := 68

-- Total weight in grams
theorem total_weight_in_grams : (3 * grams_per_ancient_jin) + (2 * grams_per_modern_jin) = 2800 :=
  by
  have x := 3
  have y := 2
  calc
    (x * grams_per_ancient_jin) + (y * grams_per_modern_jin) = 3 * 600 + 2 * 500 := by rw [←grams_per_ancient_jin, ←grams_per_modern_jin]
    ... = 1800 + 1000 := rfl
    ... = 2800 := rfl

end total_weight_in_grams_l410_410150


namespace avg_runs_last_4_matches_l410_410295

variable (matches : Type)
variable (cricketer: matches → ℕ → ℕ)
variable (firstMatches avg12 avg8: ℕ)
variable (totalRuns12 totalRuns8 totalRuns4 avg4 : ℕ)

theorem avg_runs_last_4_matches :
  avg12 = 48 ∧ avg8 = 40 ∧ firstMatches = 8 ∧ cricketer matches totalRuns12 = 576 ∧ cricketer matches totalRuns8 = 320 ∧ totalRuns4 = totalRuns12 - totalRuns8 ∧ avg4 = totalRuns4 / 4
  → avg4 = 64 :=
by
  intros
  sorry

end avg_runs_last_4_matches_l410_410295


namespace f_f_neg_two_l410_410091

def f (x : ℝ) : ℝ :=
  if x < 0 then -x else x^2

theorem f_f_neg_two : f (f (-2)) = 4 :=
  by
    sorry

end f_f_neg_two_l410_410091


namespace correct_solution_l410_410202

theorem correct_solution : 
  ∀ (x y a b : ℚ), (a = 1) → (b = 1 / 2) → 
  (a * x + y = 2) → (2 * x - b * y = 1) → 
  (x = 4 / 5 ∧ y = 6 / 5) := 
by
  intros x y a b ha hb h1 h2
  sorry

end correct_solution_l410_410202


namespace product_of_all_possible_t_l410_410402

-- Define the set of possible integer pairs (a, b) such that ab = 12.
def possible_pairs : Set (Int × Int) :=
  {(a, b) | a * b = 12}

-- Define the set of possible values of t = a + b for those pairs (a, b).
def possible_values_of_t : Set Int :=
  {t | ∃ (a b : Int), (a, b) ∈ possible_pairs ∧ t = a + b}

-- Lean 4 theorem to prove the product of all possible values of t.
theorem product_of_all_possible_t : 
  ∏ t in possible_values_of_t, t = 532224 :=
sorry

end product_of_all_possible_t_l410_410402


namespace product_of_factors_l410_410403

theorem product_of_factors :
  (∏ (t : ℤ) in { t | ∃ a b : ℤ, a * b = 12 ∧ t = a + b }, t) = -530144 :=
by sorry

end product_of_factors_l410_410403


namespace triangle_sets_l410_410708

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets_l410_410708


namespace max_value_sin_cos_expression_l410_410879

-- Define the variables and the expressions

def max_trig_expression (x y z : ℝ) : ℝ :=
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))

-- State the theorem to find the maximum value of the given expression.

theorem max_value_sin_cos_expression : ∀ x y z : ℝ, max_trig_expression x y z ≤ 4.5 :=
by {
  sorry -- This is where the proof would go
}

end max_value_sin_cos_expression_l410_410879


namespace derek_savings_in_march_l410_410852

theorem derek_savings_in_march :
  (∀ n : ℕ, n > 0 → (∀ m : ℕ, m < n → (∀ k : ℕ, k < m → (k = 1 → m = 2 * k)))) →
  (∀ total : ℕ, total = 4096 → 
  (∀ a : ℕ, a = 2) →
  (∀ r : ℕ, r = 2) →
  (∀ n : ℕ, n = 12) →
  ∃ M : ℕ, (M = a * r^(3-1)) ∧ (M = 8) ∧ 
  (total = ∑ i in finset.range n, (a * r^i))) :=
sorry

end derek_savings_in_march_l410_410852


namespace sum_tan_squared_eq_sqrt2_l410_410570
-- Import the necessary library

-- Lean 4 statement of the problem
theorem sum_tan_squared_eq_sqrt2 (S : Set ℝ) :
  (∀ x, x ∈ S → 0 < x ∧ x < π ∧ 
        (let sin_x := Real.sin x,
             cos_x := Real.cos x,
             tan_x := Real.tan x in
         sin_x^2 + cos_x^2 = 1 ∧ 
         (tan_x^2 = 1 ∨ cos_x^2 = 1 / sqrt 2 - 1))) →
  (∑ x in S, (Real.tan x)^2 = Real.sqrt 2) :=
sorry

end sum_tan_squared_eq_sqrt2_l410_410570


namespace value_of_work_clothes_l410_410771

theorem value_of_work_clothes (x y : ℝ) (h1 : x + 70 = 30 * y) (h2 : x + 20 = 20 * y) : x = 80 :=
by
  sorry

end value_of_work_clothes_l410_410771


namespace multiples_of_4_between_200_and_500_l410_410112
-- Import the necessary library

open Nat

theorem multiples_of_4_between_200_and_500 : 
  ∃ n, n = (500 / 4 - 200 / 4) :=
by
  sorry

end multiples_of_4_between_200_and_500_l410_410112


namespace projection_non_ambiguity_l410_410165

theorem projection_non_ambiguity 
    (a b c : ℝ) 
    (theta : ℝ) 
    (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos theta) : 
    ∃ (c' : ℝ), c' = c * Real.cos theta ∧ a^2 = b^2 + c^2 + 2 * b * c' := 
sorry

end projection_non_ambiguity_l410_410165


namespace distance_sum_geq_1983_l410_410623

theorem distance_sum_geq_1983:
  ∀ (M : Fin 1983 → Point) (N₁ N₂ : Point) (dist : Point → Point → ℝ),
    (dist N₁ N₂ = 2) →
    (∀ i, dist (M i) N₁ + dist (M i) N₂ ≥ 2) →
    ∃ P : Point, ∑ i, dist P (M i) ≥ 1983 :=
by
  intros M N₁ N₂ dist h₁ h₂
  sorry

end distance_sum_geq_1983_l410_410623


namespace balls_in_boxes_l410_410486

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l410_410486


namespace d_squared_value_l410_410754

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * complex.I) * z

theorem d_squared_value (c d : ℝ) (z : ℂ) (h1 : ∀ z, complex.abs (g z c d - z) = complex.abs (g z c d)) (h2 : complex.abs (c + d * complex.I) = 5) :
  d ^ 2 = 99 / 4 :=
sorry

end d_squared_value_l410_410754


namespace throne_sitter_identity_l410_410258

-- Definitions
inductive Disposition
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Person  -- Can either tell the truth or lie

def is_truthful : Disposition → Prop
| Disposition.Knight := true
| Disposition.Liar := false
| Disposition.Person := true ∨ false  -- Person can either lie or tell the truth

def chair_claim (d : Disposition) : Prop :=
  (is_truthful d → "I am a liar and a monkey" = true) ∧
  (¬is_truthful d → "I am a liar and a monkey" = false)

-- Theorem to prove
theorem throne_sitter_identity : ∀ d : Disposition, chair_claim d → d = Disposition.Person ∧ ¬is_truthful Disposition.Person :=
by
  intro d
  sorry

end throne_sitter_identity_l410_410258


namespace solve_equation_l410_410610

-- Define the equation as a predicate
def equation (x y z : ℤ) : Prop :=
  x * y / z + y * z / x + z * x / y = 3

-- Define the solutions as a set of tuples
def solutions : set (ℤ × ℤ × ℤ) :=
  { (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1) }

-- The theorem stating that the integer solutions to the equation are exactly the predefined set
theorem solve_equation (x y z : ℤ) (h : equation x y z) : (x, y, z) ∈ solutions :=
sorry

end solve_equation_l410_410610


namespace external_bisector_triangle_largest_angle_l410_410253

theorem external_bisector_triangle_largest_angle (α β γ : ℝ) 
  (hα : α = 42) (hβ : β = 59) (hγ : γ = 79) 
  (sum_angles : α + β + γ = 180) :
  let A₁ := (180 - α) / 2 in
  let B₁ := (180 - β) / 2 in
  let C₁ := (180 - γ) / 2 in
  max A₁ (max B₁ C₁) = 69 :=
by 
  sorry

end external_bisector_triangle_largest_angle_l410_410253


namespace number_of_ways_to_distribute_balls_l410_410492

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l410_410492


namespace isosceles_triangle_perimeter_l410_410149

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 11) (h3 : a = b ∨ b = b) :
  (5 + 11 + 11 = 27) := 
by {
  sorry
}

end isosceles_triangle_perimeter_l410_410149


namespace sum_is_eight_l410_410613

theorem sum_is_eight (a b c d : ℤ)
  (h1 : 2 * (a - b + c) = 10)
  (h2 : 2 * (b - c + d) = 12)
  (h3 : 2 * (c - d + a) = 6)
  (h4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 :=
by
  sorry

end sum_is_eight_l410_410613


namespace _l410_410271

lemma power_of_a_point_theorem (AP BP CP DP : ℝ) (hAP : AP = 5) (hCP : CP = 2) (h_theorem : AP * BP = CP * DP) :
  BP / DP = 2 / 5 :=
by
  sorry

end _l410_410271


namespace parallelogram_area_and_vector_l410_410957

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x, y := p1.y - p2.y, z := p1.z - p2.z }

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def cross_product (v1 v2 : Point3D) : Point3D :=
  { x := v1.y * v2.z - v1.z * v2.y,
    y := v1.z * v2.x - v1.x * v2.z,
    z := v1.x * v2.y - v1.y * v2.x }

def area_of_parallelogram (v1 v2 : Point3D) : ℝ :=
  magnitude (cross_product v1 v2)

def is_perpendicular (v1 v2 : Point3D) : Prop :=
  dot_product v1 v2 = 0

def is_unit_vector (v : Point3D) (m : ℝ) : Prop :=
  magnitude v = m

noncomputable def vector_a :=
  { to_real : ∃ (x y z : ℝ), 
    is_perpendicular ⟨-2, -1, 3⟩ ⟨x, y, z⟩ ∧
    is_perpendicular ⟨1, -3, 2⟩ ⟨x, y, z⟩ ∧
    is_unit_vector ⟨x, y, z⟩ (Real.sqrt 3)
  }

theorem parallelogram_area_and_vector : 
  let A := ⟨0, 2, 3⟩,
      B := ⟨-2, 1, 6⟩,
      C := ⟨1, -1, 5⟩,
      AB := vector_sub B A,
      AC := vector_sub C A
  in
    area_of_parallelogram AB AC = 7 * Real.sqrt 3 ∧ 
    (vector_a.to_real = (1, 1, 1) ∨ vector_a.to_real = (-1, -1, -1)) :=
by sorry

end parallelogram_area_and_vector_l410_410957


namespace probability_of_at_least_one_red_ball_l410_410524

/-- In a bag containing one red ball and one blue ball of the same size,
if drawing a ball and recording its color is considered one experiment,
and the experiment is conducted three times with replacement,
the probability of drawing at least one red ball is 7/8. -/
theorem probability_of_at_least_one_red_ball :
  (3.times {x // x = red ∨ x = blue}).probability (λ seq, ∃ n, seq[n] = red) = 7 / 8 :=
by sorry

end probability_of_at_least_one_red_ball_l410_410524


namespace potluck_soda_consumption_l410_410050

theorem potluck_soda_consumption (total_bottles : ℕ) (bottles_taken_back : ℕ) (bottles_consumed : ℕ) : 
  total_bottles = 10 → bottles_taken_back = 2 → bottles_consumed = 8 := 
by
  intros h1 h2
  have : bottles_consumed = total_bottles - bottles_taken_back
  { rw [h1, h2]
    norm_num }
  exact this

#eval potluck_soda_consumption 10 2 8

end potluck_soda_consumption_l410_410050


namespace distance_between_skew_medians_l410_410867

noncomputable def distance_skew_medians (a : ℝ) : ℝ × ℝ := 
(a * real.sqrt (2 / 35), a / real.sqrt 10)

theorem distance_between_skew_medians (a : ℝ) :
  ∃ d1 d2, d1 = a * real.sqrt (2 / 35) ∧ d2 = a / real.sqrt 10 :=
by
  let distances := distance_skew_medians a
  use distances.1, distances.2
  split
  { refl }
  { refl }

end distance_between_skew_medians_l410_410867


namespace sought_circle_and_point_exists_l410_410533

-- Conditions
def quadratic_func_f (x : ℝ) : ℝ := (real.sqrt 3 / 3) * (x^2 + 2 * x - 3)

-- Standard equation of sought circle
def circle_eq (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 4

-- Point P satisfying PA = √2 PB
def PA_eq_sqrt_2_PB (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 2 * ((x - 2) ^ 2 + y ^ 2)

-- Main theorem
theorem sought_circle_and_point_exists :
  (∀ x y : ℝ, quadratic_func_f x = y → (circle_eq x y)) ∧ ∃ (x y : ℝ), circle_eq x y ∧ PA_eq_sqrt_2_PB x y :=
by
  sorry

end sought_circle_and_point_exists_l410_410533


namespace circumscribed_sphere_radius_of_pyramid_equilateral_base_l410_410239

section
variable {a : ℝ} (h₁ : ∀ h ≥ 0, \frac{3a}{2}) (hap : a > 0)

theorem circumscribed_sphere_radius_of_pyramid_equilateral_base (h₁ : ∀ (a : ℝ), a > 0 → circumscribed_sphere_radius (equilateral_triangle_base a) (\frac{3 * a}{2}) = (a * real.sqrt 7) / 3 :
  let base := equilateral_triangle_base a,
      height := 3 * a / 2, 
  in circumscribed_sphere_radius base height = (a * real.sqrt 7) / 3 := sorry
end

end circumscribed_sphere_radius_of_pyramid_equilateral_base_l410_410239


namespace largest_n_exists_l410_410023

theorem largest_n_exists :
  ∃ (n : ℕ), (∃ (x y z : ℕ), n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) ∧
    ∀ (m : ℕ), (∃ (x y z : ℕ), m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) →
    n ≥ m :=
  sorry

end largest_n_exists_l410_410023


namespace number_of_ways_to_put_7_balls_in_2_boxes_l410_410499

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l410_410499


namespace probability_is_9_over_50_l410_410350

noncomputable def probability_at_least_one_multiple_of_5_and_sum_even : ℚ :=
  let total_numbers := 60
  let multiples_of_5_count := 12
  let probability_not_multiple_of_5 := (total_numbers - multiples_of_5_count) / total_numbers
  let probability_at_least_one_multiple_of_5 := 1 - probability_not_multiple_of_5^2
  let probability_sum_even := 1 / 2
  probability_at_least_one_multiple_of_5 * probability_sum_even

theorem probability_is_9_over_50
  (h1 : probability_at_least_one_multiple_of_5_and_sum_even = 9 / 50) :
  h1 = 9 / 50 :=
by
  sorry

end probability_is_9_over_50_l410_410350


namespace cone_volume_proof_l410_410452

-- Definitions based on the conditions
def central_angle := 120
def sector_area : ℝ := 3 * Real.pi

noncomputable def slant_height : ℝ := 3
noncomputable def base_radius : ℝ := 1
noncomputable def cone_height : ℝ := Real.sqrt (slant_height^2 - base_radius^2)
noncomputable def cone_volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * cone_height

-- Theorem statement to prove the given question
theorem cone_volume_proof :
  (cone_volume = (2 * Real.sqrt 2 / 3) * Real.pi) :=
by
  sorry

end cone_volume_proof_l410_410452


namespace cost_of_ice_cream_l410_410035

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end cost_of_ice_cream_l410_410035


namespace inscribed_circle_equals_arc_length_l410_410615

open Real

theorem inscribed_circle_equals_arc_length 
  (R : ℝ) 
  (hR : 0 < R) 
  (θ : ℝ)
  (hθ : θ = (2 * π) / 3)
  (r : ℝ)
  (h_r : r = R / 2) 
  : 2 * π * r = 2 * π * R * (θ / (2 * π)) := by
  sorry

end inscribed_circle_equals_arc_length_l410_410615


namespace positive_difference_between_solutions_l410_410389

theorem positive_difference_between_solutions :
  let a := (6 - (x^2) / 4) in 
  a^(1/3) = -3 → 2 * Real.sqrt 132 = 
  let x1 := Real.sqrt 132 in 
  let x2 := -Real.sqrt 132 in 
  Real.abs (x1 - x2) := 
by {
  assume a : 6 - (x^2) / 4,
  assume h : a^(1/3) = -3,
  sorry -- Proof omitted
}

end positive_difference_between_solutions_l410_410389


namespace rhombus_area_l410_410721

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 11) (h2 : d2 = 16) : (d1 * d2) / 2 = 88 :=
by {
  -- substitution and proof are omitted, proof body would be provided here
  sorry
}

end rhombus_area_l410_410721


namespace new_alcohol_percentage_is_26_percent_l410_410331

-- Define the problem conditions
def initial_alcohol_percentage : ℝ := 0.40
def replaced_whisky_alcohol_percentage : ℝ := 0.19
def replaced_quantity_fraction : ℝ := 2/3

-- Define the Lean statement to prove
theorem new_alcohol_percentage_is_26_percent
  (V : ℝ) (hV: V > 0) : 
  let initial_alcohol_volume := initial_alcohol_percentage * V in
  let replaced_volume := replaced_quantity_fraction * V in
  let alcohol_in_replacement := replaced_whisky_alcohol_percentage * replaced_volume in
  let alcohol_removed := initial_alcohol_percentage * replaced_volume in
  let new_alcohol_volume := initial_alcohol_volume - alcohol_removed + alcohol_in_replacement in
  (new_alcohol_volume / V) * 100 = 26 := 
  by
    sorry

end new_alcohol_percentage_is_26_percent_l410_410331


namespace product_of_third_side_8_15_l410_410273

noncomputable def product_of_third_side (a b : ℝ) (hyp : a^2 + b^2 = c^2 ∨ hypotenuse = a^2 + side^2 = b^2) : ℝ :=
  match hyp with
  | or.inl h => (sqrt (a^2 + b^2)) * (sqrt (b^2 - a^2))
  | or.inr h => sqrt (b^2 - a^2) * (sqrt (a^2 + b^2))

theorem product_of_third_side_8_15 : product_of_third_side 8 15 ≈ 215.9 := sorry

end product_of_third_side_8_15_l410_410273


namespace count_two_digit_integers_congruent_mod_4_l410_410963

theorem count_two_digit_integers_congruent_mod_4 :
  let count_k := λ (n m : ℕ), (m - n + 1) in
  count_k 2 24 = 23 := 
by
  -- sorry is used as placeholder for the proof.
  sorry

end count_two_digit_integers_congruent_mod_4_l410_410963


namespace value_of_k_l410_410764

-- Conditions provided
variable (length_to_width_ratio : ℝ := 5 / 2)
variable (perimeter : ℝ := 42)
variable (diagonal : ℝ)

-- Variables for length and width
noncomputable def length (x : ℝ) := 5 * x
noncomputable def width (x : ℝ) := 2 * x

-- Define perimeter and solve for x.
def solve_for_x : ℝ := (perimeter / 14)

-- Define the diagonal using Pythagorean theorem.
def d_squared (x : ℝ) := (length x) ^ 2 + (width x) ^ 2

-- Define the area of the rectangle.
def area (x : ℝ) := (length x) * (width x)

-- Define the value of k such that area = k * d^2
noncomputable def k : ℝ := (area solve_for_x) / (d_squared solve_for_x)

-- The theorem stating the value of k.
theorem value_of_k : k = 10 / 29 := by
  sorry

end value_of_k_l410_410764


namespace find_value_of_3x_plus_y_l410_410932

variables {A B C D : Type}
variables [A : AffineSpace]
variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables [affA : AffineSpace V A]

noncomputable def point_on_BC (B C D : A) : Prop :=
∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (t • (C -ᵥ B) = D -ᵥ B)

variables (CD DB AB AC : V) (x y : ℝ)

theorem find_value_of_3x_plus_y
  (h1 : point_on_BC B C D)
  (h2 : (CD : V) = 4 • DB)
  (h3 : (CD : V) = x • AB + y • AC) :
  3 * x + y = 8 / 5 :=
by
  sorry

end find_value_of_3x_plus_y_l410_410932


namespace max_value_sine_cosine_expression_l410_410875

theorem max_value_sine_cosine_expression (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sine_cosine_expression_l410_410875


namespace pyramid_height_l410_410752

noncomputable def height_of_pyramid (h : ℝ) : Prop :=
  let cube_edge_length := 6
  let pyramid_base_edge_length := 12
  let V_cube := cube_edge_length ^ 3
  let V_pyramid := (1 / 3) * (pyramid_base_edge_length ^ 2) * h
  V_cube = V_pyramid → h = 4.5

theorem pyramid_height : height_of_pyramid 4.5 :=
by {
  sorry
}

end pyramid_height_l410_410752


namespace purchasing_plan_exists_l410_410330

-- Define the structure for our purchasing plan
structure PurchasingPlan where
  n3 : ℕ
  n6 : ℕ
  n9 : ℕ
  n12 : ℕ
  n15 : ℕ
  n19 : ℕ
  n21 : ℕ
  n30 : ℕ

-- Define the length function to sum up the total length of the purchasing plan
def length (p : PurchasingPlan) : ℕ :=
  3 * p.n3 + 6 * p.n6 + 9 * p.n9 + 12 * p.n12 + 15 * p.n15 + 19 * p.n19 + 21 * p.n21 + 30 * p.n30

-- Define the purchasing options
def options : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

-- Define the requirement
def requiredLength : ℕ := 50

-- State the theorem that there exists a purchasing plan that sums up to the required length
theorem purchasing_plan_exists : ∃ p : PurchasingPlan, length p = requiredLength :=
  sorry

end purchasing_plan_exists_l410_410330


namespace utensils_ratio_l410_410559

-- Definitions for conditions
variables (K F S : ℕ)

-- Conditions for the problem
def total_utensils (K F S : ℕ) : Prop :=
  K + F + S = 30

def spoons_per_pack (S : ℕ) : Prop :=
  S = 10

-- Mathematically equivalent proof problem
theorem utensils_ratio (K F : ℕ) (S : ℕ) (h1 : total_utensils K F S) (h2 : spoons_per_pack S) :
  K + F = 20 :=
begin
  -- Proof to be provided
  sorry
end

end utensils_ratio_l410_410559


namespace remainder_degrees_l410_410283

-- Declare the polynomial divisor
def divisor : Polynomial ℤ := 5 * (X^7) - 2 * (X^3) + X - 8

-- Problem statement: the possible degrees of the remainder when dividing by a specific polynomial divisor
theorem remainder_degrees (p : Polynomial ℤ) :
  ∃ r : Polynomial ℤ, (∀ d, d < 7 → r.degree = Option.some d) ∧ (p = q * divisor + r) :=
sorry

end remainder_degrees_l410_410283


namespace sandwiches_bought_l410_410278

-- Define constants for the problem:
def cost_per_sandwich := 1.49
def number_of_sodas := 4
def cost_per_soda := 0.87
def total_cost := 6.46

-- State the theorem to prove the number of sandwiches bought
theorem sandwiches_bought : 
  ∃ S : ℝ, (cost_per_sandwich * S + cost_per_soda * number_of_sodas = total_cost) ∧ S = 2 := 
by 
  sorry

end sandwiches_bought_l410_410278


namespace find_percentage_l410_410514

theorem find_percentage (x : ℝ) (h1 : x = 780) (h2 : ∀ P : ℝ, P / 100 * x = 225 - 30) : P = 25 :=
by
  -- Definitions and conditions here
  -- Recall: x = 780 and P / 100 * x = 195
  sorry

end find_percentage_l410_410514


namespace eval_piecewise_function_l410_410464

noncomputable def f : ℝ → ℝ := λ x, if x > 0 then Real.log x / Real.log 2 else 3^x

theorem eval_piecewise_function :
  f (f (1/2)) = 1/3 := by
  sorry

end eval_piecewise_function_l410_410464


namespace series_equality_l410_410217

theorem series_equality (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range n, 1 / (((2 * k + 1) * (2 * k + 2)) : ℚ)) = (∑ k in Finset.range n, 1 / (n + k + 1 : ℚ)) :=
sorry

end series_equality_l410_410217


namespace parallelogram_area_diagonal_l410_410792

-- Definitions for the given conditions
def base : ℝ := 10
def height : ℝ := 4
def slant_height : ℝ := 6

-- Proof statements
theorem parallelogram_area_diagonal :
  (base * height = 40) ∧
  (real.sqrt ((base + slant_height) ^ 2 + height ^ 2) = real.sqrt 272) :=
by
  -- We assume the base, height, and slant_height values from the conditions
  have h1 : base * height = 40,
  have h2 : real.sqrt ((base + slant_height) ^ 2 + height ^ 2) = real.sqrt 272,
  exact ⟨h1, h2⟩
  sorry

end parallelogram_area_diagonal_l410_410792


namespace sum_of_possible_digit_counts_in_base_2_l410_410321

theorem sum_of_possible_digit_counts_in_base_2 :
  (∑ d in ({13, 14, 15} : Finset ℕ), d) = 42 :=
by 
  sorry

end sum_of_possible_digit_counts_in_base_2_l410_410321


namespace equation_satisfied_except_seven_l410_410370

theorem equation_satisfied_except_seven (x : ℝ) :
  x ≠ 7 →
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) :=
by
  intro h
  have h1 : (x^2 - 9 * x + 14) ≠ 0, by calc
    x^2 - 9 * x + 14 = (x-7)*(x-2) : by ring
    ... ≠ 0 : by intro hz; cases hz; exact h hz.left
  have h2 : (x^2 - 4 * x - 21) ≠ 0, by calc
    x^2 - 4 * x - 21 = (x-7)*(x+3) : by ring
    ... ≠ 0 : by intro hz; cases hz; exact h hz.left
  calc
    (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14)
      = (x-4)*(x-2) / ((x-7)*(x-2)) : by ring
    ... = (x-4) / (x-7) : by simp [mul_comm, mul_assoc, mul_left_comm]
    ... = (x-6)*(x+3) / ((x-7)*(x+3)) : by ring
    ... = (x-6) / (x-7) : by simp [mul_comm, mul_assoc, mul_left_comm]
    ... = (x^2 - 3 * x - 18) / (x^2 - 4 * x - 21) : by ring
  sorry

end equation_satisfied_except_seven_l410_410370


namespace probability_of_two_draws_l410_410307

theorem probability_of_two_draws (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
    (total_balls = white_balls + black_balls) 
    (P_black_first : ℚ := black_balls / total_balls) 
    (remaining_balls_after_black : ℕ := total_balls - 1) 
    (P_white_second := white_balls / remaining_balls_after_black) :
     P_black_first * P_white_second = 7 / 30 :=
by
  have total_eq : total_balls = 7 + 3 := by simp [white_balls, black_balls]
  have prob_black_first : P_black_first = 3 / 10 := by simp [P_black_first, total_eq]
  have remaining_balls_eq : remaining_balls_after_black = 9 := by simp [remaining_balls_after_black, total_eq]
  have prob_white_second : P_white_second == 7 / remaining_balls_after_black := by simp [P_white_second, total_eq]
  have prob_white_second_final := 7 / 9 := by simp [prob_white_second, remaining_balls_eq]
  sorry

end probability_of_two_draws_l410_410307


namespace integral_of_h_eq_pi_over_16_l410_410928

-- Define the function g(x) as given in the condition
def g (x : ℝ) : ℝ := real.sqrt (x * (1 - x))

-- Define the function h(x) incorporating x g(x)
def h (x : ℝ) : ℝ := x * g x

-- State the theorem to prove the area under h(x) from 0 to 1 equals π/16
theorem integral_of_h_eq_pi_over_16 : (∫ x in 0..1, h x) = (real.pi / 16) :=
by
  sorry

end integral_of_h_eq_pi_over_16_l410_410928


namespace sine_cosine_series_sum_l410_410353

theorem sine_cosine_series_sum :
  (∑ x in finset.range 45, (sin (real.of_nat x + 1) * cos (real.of_nat x + 1) + cos (real.of_nat x + 1) * sin (real.of_nat x + 1))) = 0 :=
by
  sorry

end sine_cosine_series_sum_l410_410353


namespace true_proposition_l410_410444

-- Define the propositions p and q
def p : Prop := ∃ x0 : ℝ, x0 ^ 2 - x0 + 1 ≥ 0

def q : Prop := ∀ (a b : ℝ), a < b → 1 / a > 1 / b

-- Prove that p ∧ ¬q is true
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l410_410444


namespace find_cookies_per_tray_l410_410903

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray_l410_410903


namespace minimum_value_of_expression_l410_410047

theorem minimum_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
    (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 := 
sorry

end minimum_value_of_expression_l410_410047


namespace find_m_value_l410_410106

noncomputable def int_solution_exists (x y m : ℤ) : Prop :=
  x + 2 * y = 6 ∧ x - 2 * y + m * x = -5

theorem find_m_value (m : ℤ) 
  (H : ∃ x y, int_solution_exists x y m ∧ x ∈ Int ∧ m ∈ Int) : 
  m = -1 ∨ m = -3 :=
by sorry

end find_m_value_l410_410106


namespace quadrilateral_collinearity_and_ratio_l410_410596

section ConvexQuadrilateral

variables {k : Type*} [Field k]

structure Point (k : Type*) :=
(x : k)
(y : k)

variables (A B C D F1 F2 F3 F4 U V W: Point k)

def midpoint (P Q : Point k) : Point k :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

def centroid (P Q R : Point k) : Point k :=
{ x := (P.x + Q.x + R.x) / 3,
  y := (P.y + Q.y + R.y) / 3 }

def segment (P Q : Point k) : set (Point k) := λ R, ∃ a b : k, a + b = 1 ∧ R.x = a * P.x + b * Q.x ∧ R.y = a * P.y + b * Q.y

def collinear (P Q R : Point k) : Prop :=
∃ a b c : k, a • (P.x, P.y) + b • (Q.x, Q.y) + c • (R.x, R.y) = (0, 0) ∧ a + b + c = 0

def ratio (P Q R : Point k) : k :=
if Q.x = R.x then (Q.y - P.y) / (R.y - Q.y) else (Q.x - P.x) / (R.x - Q.x)

theorem quadrilateral_collinearity_and_ratio (hW : W = centroid k (centroid k A B D) (centroid k B C D))
  (hF1 : F1 = midpoint k A B) (hF2 : F2 = midpoint k B C) (hF3 : F3 = midpoint k C D) (hF4 : F4 = midpoint k D A)
  (hU : U ∈ segment k A C) (hU' : U ∈ segment k B D)
  (hV : V ∈ segment k F1 F3) (hV' : V ∈ segment k F2 F4) :
  collinear k U V W ∧ ratio k W V U = 1 / 3 :=
sorry

end ConvexQuadrilateral

end quadrilateral_collinearity_and_ratio_l410_410596


namespace room_width_l410_410248

theorem room_width (w : ℝ) (h1 : 21 > 0) (h2 : 2 > 0) 
  (h3 : (25 * (w + 4) - 21 * w = 148)) : w = 12 :=
by {
  sorry
}

end room_width_l410_410248


namespace value_of_b_minus_d_squared_l410_410993

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := 
by 
  sorry

end value_of_b_minus_d_squared_l410_410993


namespace max_value_of_product_l410_410188

theorem max_value_of_product (x y z w : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) (h_sum : x + y + z + w = 1) : 
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 :=
by
  sorry

end max_value_of_product_l410_410188


namespace max_value_l410_410884

/-- 
Proof of the maximum value of the expression 
(sin x + sin 2y + sin 3z) * (cos x + cos 2y + cos 3z)
-/
theorem max_value (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_l410_410884


namespace part1_part2_l410_410157

open_complex

-- Conditions for Part 1
def z1 : ℂ := 1 - 3 * I
def z2 (a : ℝ) : ℂ := a + I

-- Part 1: Prove that a = 3
theorem part1 (a : ℝ) (h : (z2 a / z1).re = 0) : a = 3 :=
sorry

-- Conditions for Part 2
def z1_conjugate : ℂ := 1 + 3 * I
def root_equation (p q : ℝ) := λ x : ℂ, x^2 + (p : ℂ) * x + (q : ℂ)

-- Part 2: Prove that p = 2 and q = 10
theorem part2 (p q : ℝ) (h : root_equation p q z1_conjugate = 0 ∧ root_equation p q (conj(z1)) = 0) : 
  p = 2 ∧ q = 10 :=
sorry

end part1_part2_l410_410157


namespace product_xyz_l410_410975

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l410_410975


namespace proof_arithmetic_progression_l410_410387

-- Define the terms and conditions of the arithmetic progression
variables {a₁ d : ℕ}
variables {n : ℕ}

-- Conditions given in the problem
def sum_of_terms (a₁ d n : ℕ) : Prop :=
  n * (2 * a₁ + (n - 1) * d) / 2 = 112

def second_term_product (a₁ d : ℕ) : Prop :=
  (a₁ + d) * d = 30

def third_fifth_sum (a₁ d : ℕ) : Prop :=
  a₁ + 2 * d + (a₁ + 4 * d) = 32

-- Define the sequence based on given conditions
def arithmetic_progression {a₁ d n : ℕ} (h_sum : sum_of_terms a₁ d n) (h_product : second_term_product a₁ d) (h_sum_third_fifth : third_fifth_sum a₁ d) : Prop :=
  n = 7 ∧ ((a₁ = 1 ∧ d = 5) ∨ (a₁ = 7 ∧ d = 3))

-- State the theorem in Lean
theorem proof_arithmetic_progression : ∃ (a₁ d n : ℕ), sum_of_terms a₁ d n ∧ second_term_product a₁ d ∧ third_fifth_sum a₁ d ∧ arithmetic_progression :=
sorry

end proof_arithmetic_progression_l410_410387


namespace triple_hash_90_l410_410364

def hash (N : ℝ) : ℝ := 0.3 * N + 2

theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 :=
by
  sorry

end triple_hash_90_l410_410364


namespace tan_ratio_is_7_over_3_l410_410182

open Real

theorem tan_ratio_is_7_over_3 (a b : ℝ) (h1 : sin (a + b) = 5 / 8) (h2 : sin (a - b) = 1 / 4) : (tan a / tan b) = 7 / 3 :=
by
  sorry

end tan_ratio_is_7_over_3_l410_410182


namespace dishonest_dealer_uses_correct_weight_l410_410753

noncomputable def dishonest_dealer_weight (profit_percent : ℝ) (true_weight : ℝ) : ℝ :=
  true_weight - (profit_percent / 100 * true_weight)

theorem dishonest_dealer_uses_correct_weight :
  dishonest_dealer_weight 11.607142857142861 1 = 0.8839285714285714 :=
by
  -- We skip the proof here
  sorry

end dishonest_dealer_uses_correct_weight_l410_410753


namespace product_xyz_equals_zero_l410_410970

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l410_410970


namespace painter_red_cells_count_l410_410656

open Nat

/-- Prove the number of red cells painted by the painter in the given 2000 x 70 grid. -/
theorem painter_red_cells_count :
  let rows := 2000
  let columns := 70
  let lcm_rc := Nat.lcm rows columns -- Calculate the LCM of row and column counts
  lcm_rc = 14000 := by
sorry

end painter_red_cells_count_l410_410656


namespace sales_tax_difference_l410_410352

theorem sales_tax_difference :
  let price := 50
  let rate1 := 0.0725
  let rate2 := 0.0675
  let tax1 := price * rate1
  let tax2 := price * rate2
  let difference := tax1 - tax2
  difference = 0.25 :=
by
  /-
  Using the conditions:
    price = 50
    rate1 = 0.0725
    rate2 = 0.0675
    tax1 = 50 * 0.0725
    tax2 = 50 * 0.0675
    difference = 50 * 0.0725 - 50 * 0.0675
  We need to prove:
    (50 * 0.0725 - 50 * 0.0675) = 0.25
  -/
  calc 50 * 0.0725 - 50 * 0.0675 = 3.625 - 3.375 : by norm_num
                          ... = 0.25 : by norm_num

end sales_tax_difference_l410_410352


namespace circumradius_ratio_of_triangles_l410_410443

theorem circumradius_ratio_of_triangles :
  ∀ {A B C A' B' C' O_A O_B O_C : Type} 
  [geometry_acute_angle_triangle A B C]
  [symmetric_wrt_line A' A B C]
  [symmetric_wrt_line B' B A C]
  [symmetric_wrt_line C' C A B]
  [is_center_circle_passing_through_midpoints O_A A A'B A'C]
  [is_center_circle_passing_through_midpoints O_B B B'A B'C]
  [is_center_circle_passing_through_midpoints O_C C C'A C'B]
  (circumradius_ABC : ℝ)
  (circumradius_OA_OB_OC : ℝ) :
  circumradius_ABC / circumradius_OA_OB_OC = 6 := 
sorry

end circumradius_ratio_of_triangles_l410_410443


namespace ways_A_to_C_via_B_l410_410134

def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 3

theorem ways_A_to_C_via_B : ways_A_to_B * ways_B_to_C = 6 := by
  sorry

end ways_A_to_C_via_B_l410_410134


namespace S_is_intersection_of_arithmetic_progression_l410_410176

section math_problem

variables {q r : ℤ} {A B : set ℝ}
variables (T : set ℝ) (S : set ℤ)

-- Assume conditions as given in the problem
axiom q_pos : q > 0
axiom q_int : q ∈ ℤ
axiom r_int : r ∈ ℤ
axiom A_interval : ∃ a₁ a₂ : ℝ, A = set.Icc a₁ a₂ -- A is an interval on the real line
axiom B_interval : ∃ b₁ b₂ : ℝ, B = set.Icc b₁ b₂ -- B is an interval on the real line
axiom T_def : ∀ (b m : ℤ), b ∈ B → b + m * q ∈ T -- T defined as given
axiom S_def : ∀ (a : ℤ), a ∈ A → (r * a ∈ T) → a ∈ S -- S defined as given
axiom length_product : ∀ (a₁ a₂ b₁ b₂ : ℝ), A = set.Icc a₁ a₂ → B = set.Icc b₁ b₂ → (a₂ - a₁) * (b₂ - b₁) < q -- product of lengths < q

-- The statement to be proved
theorem S_is_intersection_of_arithmetic_progression (h1 : q > 0) (h2 : ∃ a₁ a₂ : ℝ, A = set.Icc a₁ a₂) 
  (h3 : ∃ b₁ b₂ : ℝ, B = set.Icc b₁ b₂) (h4 : T = { b + m * q | b ∈ B, m ∈ ℤ }) 
  (h5 : S = { a ∈ ℤ | a ∈ A ∧ r * a ∈ T }) 
  (h6 : ∀a₁ a₂ b₁ b₂ : ℝ, A = set.Icc a₁ a₂ → B = set.Icc b₁ b₂ → (a₂ - a₁) * (b₂ - b₁) < q) : 
  ∃ d : ℕ, ∃ m n : ℤ, set_eq S (A ∩ {a : ℤ | a = m + n * d}) :=
sorry

end math_problem

end S_is_intersection_of_arithmetic_progression_l410_410176


namespace xyz_product_neg4_l410_410976

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l410_410976


namespace annual_interest_rate_l410_410745

theorem annual_interest_rate 
  (P A : ℝ) 
  (hP : P = 136) 
  (hA : A = 150) 
  : (A - P) / P = 0.10 :=
by sorry

end annual_interest_rate_l410_410745


namespace max_value_ellipse_l410_410433

theorem max_value_ellipse (x y : ℝ) (h : x^2 + 4 * y^2 = 4) : 
  (∃ x y, x^2 + 4 * y^2 = 4) → (x = 2) → (y = 0) → 
  ∀ x y, (x^2 + 2 * x - y^2 ≤ 7) := 
by 
  exist x y 
  assume h 
  have x = 2 := sorry 
  have y = 0 := sorry
  show x^2 + 2 * x - y^2 ≤ 7 from sorry 
  sorry 

end max_value_ellipse_l410_410433


namespace triangle_XYZ_median_inequalities_l410_410552

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end triangle_XYZ_median_inequalities_l410_410552


namespace find_smallest_n_l410_410696

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410696


namespace sum_even_indexed_phis_l410_410650

theorem sum_even_indexed_phis (n : ℕ) (hn : 4*n > 0) 
  (z : ℂ → ℂ → Prop)
  (hz : ∀ z, z^36 = z^12 + 1 ∧ |z| = 1)
  (φ : ℕ → ℝ)
  (hφ : ∀ k, (k < 4*n) → z k = (complex.cos (φ k)) + complex.sin (φ k)) 
  :
  ∑ i in finset.range (4*n), if i.even then φ i else 0 = 780 :=
begin
  sorry
end

end sum_even_indexed_phis_l410_410650


namespace eval_expression_l410_410859

open Real

noncomputable def e : ℝ := 2.71828

theorem eval_expression : abs (5 * e - 15) = 1.4086 := by
  sorry

end eval_expression_l410_410859


namespace minimum_tangent_sum_l410_410148

theorem minimum_tangent_sum (A B C a b c: ℝ) (hA: A + B + C = π) 
(h_acute: A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
(h_a: a = 2 * b * sin C) 
(h_sines: a / sin A = b / sin B = c / sin C):
  tan A + tan B + tan C = 8 :=
sorry  -- proof is omitted

end minimum_tangent_sum_l410_410148


namespace statement_A_statement_B_statement_C_l410_410471

variables {p : ℝ} (hp : p > 0) (x0 y0 x1 y1 x2 y2 : ℝ)
variables (h_parabola : ∀ x y, y^2 = 2*p*x) 
variables (h_point_P : ∀ k m, y0 ≠ 0 ∧ x0 = k*y0 + m)

-- Statement A
theorem statement_A (hy0 : y0 = 0) : y1 * y2 = -2 * p * x0 :=
sorry

-- Statement B
theorem statement_B (hx0 : x0 = 0) : 1 / y1 + 1 / y2 = 1 / y0 :=
sorry

-- Statement C
theorem statement_C : (y0 - y1) * (y0 - y2) = y0^2 - 2 * p * x0 :=
sorry

end statement_A_statement_B_statement_C_l410_410471


namespace vector_sum_zero_l410_410732

variable {V : Type} [AddCommGroup V]

variables (A B C D E F : V)

theorem vector_sum_zero :
  (\overrightarrow{A B}) + (\overrightarrow{B C}) + (\overrightarrow{C D}) + (\overrightarrow{D E}) + (\overrightarrow{E F}) + (\overrightarrow{F A}) = 0 :=
by
  sorry

end vector_sum_zero_l410_410732


namespace factorial_quotient_l410_410820

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410820


namespace increase_premium_after_accident_l410_410508

-- Definitions based on the problem conditions
def InsurancePremium : Type := ℝ

def InvolvesAccident (VasyaInsurance : InsurancePremium) : Prop :=
  ∃ accident, VasyaInsurance > 0  -- assuming Vasya had insurance and accident impacts premium

def PolicyRenewal (VasyaInsurance : InsurancePremium) (VasyaInsuranceRenewed : InsurancePremium) : Prop :=
  InvolvesAccident VasyaInsurance → VasyaInsuranceRenewed > VasyaInsurance

-- The theorem to prove based on the condition and correct answer
theorem increase_premium_after_accident
  (VasyaInsurance : InsurancePremium)
  (VasyaInsuranceRenewed : InsurancePremium)
  (h1 : InvolvesAccident VasyaInsurance)
  (h2 : PolicyRenewal VasyaInsurance VasyaInsuranceRenewed) :
  VasyaInsuranceRenewed > VasyaInsurance :=
by
  sorry

end increase_premium_after_accident_l410_410508


namespace evaluate_f_l410_410097

def f (x : ℚ) : ℚ := (2 * x - 3) / (3 * x ^ 2 - 1)

theorem evaluate_f :
  f (-2) = -7 / 11 ∧ f (0) = 3 ∧ f (1) = -1 / 2 :=
by
  sorry

end evaluate_f_l410_410097


namespace problem_l410_410985

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l410_410985


namespace problem_statement_l410_410706

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def decreasing_on_interval (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≥ f y

def f1 (x : ℝ) : ℝ := 1 / |x|
def f2 (x : ℝ) : ℝ := -x + 1 / x
def f3 (x : ℝ) : ℝ := 2 ^ |x|
def f4 (x : ℝ) : ℝ := |x + 1|

theorem problem_statement : even_function f3 ∧ decreasing_on_interval f3 {x : ℝ | x < 0} :=
by {
  sorry
}

end problem_statement_l410_410706


namespace binomial_pmf_rocket_launch_failure_l410_410453

section binomial_distribution

variables {n : ℕ} {p q : ℝ} (X : ℕ → ℝ)

def binomial_pmf (n : ℕ) (p q : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * q^(n-k)

theorem binomial_pmf_rocket_launch_failure :
  (∀ k : ℕ, binomial_pmf 10 0.01 0.99 k = (Nat.choose 10 k) * 0.01^k * 0.99^(10 - k)) :=
begin
  intro k,
  sorry
end

end binomial_distribution

end binomial_pmf_rocket_launch_failure_l410_410453


namespace cost_of_tuna_l410_410626

theorem cost_of_tuna (num_red_snappers : ℕ := 8) (num_tunas : ℕ := 14)
  (red_snapper_cost : ℕ := 3) (daily_earnings : ℕ := 52) :
  (14 * T = 52 - 8 * 3) → T = 2 :=
by
  intros h
  calc
    T = (52 - 8 * 3) / 14 : by sorry
    ... = 2 : by sorry

end cost_of_tuna_l410_410626


namespace tan_diff_l410_410059

theorem tan_diff (α β : ℝ) (hα : tan α = 1 / 2) (hβ : tan β = 1 / 3) : tan (α - β) = 1 / 7 := 
by 
sorry

end tan_diff_l410_410059


namespace max_value_sine_cosine_expression_l410_410876

theorem max_value_sine_cosine_expression (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sine_cosine_expression_l410_410876


namespace cost_of_washer_l410_410349

-- Variables for the problem
variables (W : ℝ) (cost_wash : ℝ) (cost_dry_per_quarter : ℝ)
          (loads : ℕ) (dryers : ℕ) (minutes_per_dryer : ℕ) 
          (total_time_every_10_minutes : ℕ) (total_cost : ℝ)

-- Initialize the values according to the conditions
def conditions :=
  cost_dry_per_quarter = 0.25 ∧
  loads = 2 ∧
  dryers = 3 ∧
  minutes_per_dryer = 40 ∧
  total_cost = 11

-- Calculate the total cost for washing and drying according to the conditions
def calc_total_cost :=
  2 * W + (3 * (40 / 10 * 0.25))

-- Theorem to prove the equivalent cost for a washer is $4
theorem cost_of_washer (h : conditions) : calc_total_cost = total_cost → W = 4 := by
  sorry

end cost_of_washer_l410_410349


namespace sin_cos_eq_cos_cos_implies_x_10_degrees_l410_410855

theorem sin_cos_eq_cos_cos_implies_x_10_degrees (x : ℝ) (h : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) (deg : x = 10 / 180 * π) :
  x = 10 / 180 * π :=
by 
  sorry

end sin_cos_eq_cos_cos_implies_x_10_degrees_l410_410855


namespace marks_for_correct_answer_l410_410530

theorem marks_for_correct_answer (x : ℕ) 
  (total_marks : ℤ) (total_questions : ℕ) (correct_answers : ℕ) 
  (wrong_mark : ℤ) (result : ℤ) :
  total_marks = result →
  total_questions = 70 →
  correct_answers = 27 →
  (-1) * (total_questions - correct_answers) = wrong_mark →
  total_marks = (correct_answers : ℤ) * (x : ℤ) + wrong_mark →
  x = 3 := 
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end marks_for_correct_answer_l410_410530


namespace balls_in_boxes_l410_410487

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l410_410487


namespace factorial_div_eq_l410_410840
-- Import the entire math library

-- Define the entities involved in the problem
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the given conditions
def given_expression : ℕ := factorial 10 / (factorial 7 * factorial 3)

-- State the main theorem that corresponds to the given problem and its correct answer
theorem factorial_div_eq : given_expression = 120 :=
by 
  -- Proof is omitted
  sorry

end factorial_div_eq_l410_410840


namespace product_xyz_equals_zero_l410_410966

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l410_410966


namespace product_xyz_equals_zero_l410_410968

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l410_410968


namespace num_divisors_with_more_than_three_factors_l410_410114

theorem num_divisors_with_more_than_three_factors :
  let n := 2728
  let factors := [1, 2, 4, 8, 11, 22, 31, 44, 62, 88, 124, 248, 341, 682, 1364, 2728]
  let factor_counts := [1, 2, 3, 4, 2, 4, 2, 4, 4, 4, 4, 6, 4, 6, 6, 16]
  let more_than_three_factors := filter (fun m => m > 3) factor_counts
  in more_than_three_factors.length = 11 := by
  -- proof goes here
  sorry

end num_divisors_with_more_than_three_factors_l410_410114


namespace shortest_chord_length_l410_410384

-- Define the properties of the ellipse
def ellipse (x y : ℝ) : Prop := 
  (x^2 / 16 + y^2 / 9 = 1)

-- Introduce the definitions necessary for our proof
def focus : ℝ := sqrt (16 - 9)

theorem shortest_chord_length : 
  ∀ (x y : ℝ), ellipse x y → x = focus → y = 9 / 4 → 
  2 * abs y = 9 / 2 :=
by
  sorry

end shortest_chord_length_l410_410384


namespace increase_premium_after_accident_l410_410509

-- Definitions based on the problem conditions
def InsurancePremium : Type := ℝ

def InvolvesAccident (VasyaInsurance : InsurancePremium) : Prop :=
  ∃ accident, VasyaInsurance > 0  -- assuming Vasya had insurance and accident impacts premium

def PolicyRenewal (VasyaInsurance : InsurancePremium) (VasyaInsuranceRenewed : InsurancePremium) : Prop :=
  InvolvesAccident VasyaInsurance → VasyaInsuranceRenewed > VasyaInsurance

-- The theorem to prove based on the condition and correct answer
theorem increase_premium_after_accident
  (VasyaInsurance : InsurancePremium)
  (VasyaInsuranceRenewed : InsurancePremium)
  (h1 : InvolvesAccident VasyaInsurance)
  (h2 : PolicyRenewal VasyaInsurance VasyaInsuranceRenewed) :
  VasyaInsuranceRenewed > VasyaInsurance :=
by
  sorry

end increase_premium_after_accident_l410_410509


namespace nine_digit_sum_l410_410661

def is_product_of_single_digits (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a * b = n ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

theorem nine_digit_sum (A B C D E F G H I : ℕ) 
  (h₁ : {A, B, C, D, E, F, G, H, I} = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h₂ : is_product_of_single_digits (10 * A + B))
  (h₃ : is_product_of_single_digits (10 * B + C))
  (h₄ : is_product_of_single_digits (10 * C + D))
  (h₅ : is_product_of_single_digits (10 * D + E))
  (h₆ : is_product_of_single_digits (10 * E + F))
  (h₇ : is_product_of_single_digits (10 * F + G))
  (h₈ : is_product_of_single_digits (10 * G + H))
  (h₉ : is_product_of_single_digits (10 * H + I)) :
  (100 * A + 10 * B + C) + (100 * D + 10 * E + F) + (100 * G + 10 * H + I) = 1602 :=
sorry

end nine_digit_sum_l410_410661


namespace proof_fneg2017_l410_410463

def f (x : ℝ) := ((x + 1) ^ 2 + (Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x)) * (Real.cos x)) / (x ^ 2 + 1)

theorem proof_fneg2017 :
  f(2017) = 2016 →
  f(-2017) = -2014 :=
by
  sorry

end proof_fneg2017_l410_410463


namespace find_r_l410_410849

def cubic_function (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_r (p q r : ℝ) (h1 : cubic_function p q r (-1) = 0) :
  r = p - 2 :=
sorry

end find_r_l410_410849


namespace problem_l410_410931

open scoped Real

theorem problem {
  -- Given conditions
  θ : ℝ,
  M : ℝ × ℝ,
  (hM : M.1 = 1 + Real.cos θ ∧ M.2 = Real.sin θ),
  A B : ℝ × ℝ,
  -- Cartesian equation for line l (in polar form transformed to Cartesian)
  (hl : ∀ (ρ : ℝ), ρ * Real.cos (θ + π / 4) = 0 ↔ M.1 - M.2 = 0 )
  -- Defining curve C
}: 
  (∀ (x y : ℝ), (∃ θ : ℝ, x = 1 + Real.cos θ ∧ y = Real.sin θ) ↔ (x - 1)^2 + y^2 = 1 ) ∧
  (∀ (x y : ℝ), (∃ ρ θ : ℝ, ρ * Real.cos (θ + π / 4) = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ x - y = 0 ) ∧
  (∃ A B : ℝ × ℝ, A ∈ curve_C ∧ B ∈ curve_C ∧ on_line_L A ∧ on_line_L B ∧
    let d := |1 - 0| / Real.sqrt (1 + 1),
        max_m_dist := d + 1, 
        |AB| := 2 * Real.sqrt (1 - (d / 2) ^ 2) in
    ∀ M : ℝ × ℝ, 
    (M.1 = 1 + Real.cos θ ∧ M.2 = Real.sin θ) →
    |M - A| * |M - B| * Real.sin (angle A M B) = (√2 + 1) / 2 ) := sorry

end problem_l410_410931


namespace part1_monotonicity_l410_410432

theorem part1_monotonicity (a : ℝ) :
  let f := λ x : ℝ, (a * x - 3/4) * Real.exp x in
  (a < 0 → ∃ c ∈ ℝ, ∀ x, if x < c then f' x > 0 else f' x < 0 ∧ f' c = 0) ∧
  (a = 0 → ∀ x, f' x < 0) ∧
  (a > 0 → ∃ c ∈ ℝ, ∀ x, if x < c then f' x < 0 else f' x > 0 ∧ f' c = 0) := 
sorry

end part1_monotonicity_l410_410432


namespace blue_eyed_among_blondes_l410_410298

variable (l g b a : ℝ)

-- Given: The proportion of blondes among blue-eyed people is greater than the proportion of blondes among all people.
axiom given_condition : a / g > b / l

-- Prove: The proportion of blue-eyed people among blondes is greater than the proportion of blue-eyed people among all people.
theorem blue_eyed_among_blondes (l g b a : ℝ) (h : a / g > b / l) : a / b > g / l :=
by
  sorry

end blue_eyed_among_blondes_l410_410298


namespace aladdin_distance_difference_l410_410778

-- Define the angular position function φ(t) : ℝ → [0,1)
noncomputable def ϕ (t : ℝ) : ℝ := sorry

-- The problem statement to prove
theorem aladdin_distance_difference :
  (∀ t : ℝ, 0 ≤ ϕ(t) ∧ ϕ(t) < 1) → (∃ t₁ t₂ : ℝ, (ϕ(t₁) = 0 ∧ ϕ(t₂) = 1) ∨ (ϕ(t₁) = 1 ∧ ϕ(t₂) = 0)) →
  ∃ t₁ t₂ : ℝ, ϕ(t₁) - ϕ(t₂) ≥ 1 :=
begin
  sorry
end

end aladdin_distance_difference_l410_410778


namespace evaluate_f_2014_l410_410851

noncomputable def f : ℝ → ℝ :=
sorry

theorem evaluate_f_2014 :
  (∀ x, f(x) = f(4 - x)) →
  (∀ x, f(2 - x) + f(x - 2) = 0) →
  f(2) = 1 →
  f(2014) = -1 :=
by
  intros h1 h2 h3
  sorry

end evaluate_f_2014_l410_410851


namespace fifth_term_is_correct_l410_410358

noncomputable def arithmetic_sequence_fifth_term (x y : ℝ) :=
  let a1 := x + 2*y
  let a2 := x - 2*y
  let a3 := 2*x*y
  let a4 := x / y
  let d := a2 - a1
  let a5 := x + 2*y + 4*d
  a5

theorem fifth_term_is_correct (x y : ℝ) (h1 : x = 6 * y / (1 - 2*y))
                             (h2 : y = 0.65 ∨ y = -0.45) :
  arithmetic_sequence_fifth_term (-13) (0.65) = -27.7 :=
by
  unfold arithmetic_sequence_fifth_term
  have hx : x = -13 := sorry
  have hy : y = 0.65 := sorry
  have d : x - 6*y = 2*x*y := sorry
  have d_term : a2 - a1 = d := sorry
  simp [hx, hy, d, d_term]
  sorry

end fifth_term_is_correct_l410_410358


namespace athletes_qualify_l410_410145

-- Definitions representing conditions
def athletes : ℕ := 896

def points_for_win : ℕ := 1
def points_for_loss : ℕ := 0

-- Consider we have a function to form pairs and simulate a round
def form_pairs : list ℕ → list (ℕ × ℕ) := sorry
def simulate_round (pairs : list (ℕ × ℕ)) : list ℕ := sorry

-- The condition that a player is eliminated after a second loss
def eliminated (losses : ℕ) : bool := losses ≥ 2

-- Let's assume we have a tuple where fst is the count of athletes without defeats
-- and snd is the count of athletes with one defeat
def tournament_progress : ℕ → ℕ × ℕ := sorry

-- The final condition proving 10 athletes remain in the qualification round
theorem athletes_qualify : (tournament_progress 10).fst + (tournament_progress 10).snd = 10 := sorry

end athletes_qualify_l410_410145


namespace smallest_solution_l410_410408

theorem smallest_solution (x : ℝ) (h₀ : x ≠ 0) (h₃ : x ≠ 3) :
  (3 * x / (x - 3) + (3 * x^2 - 27 * x) / x = 14) → x = (-(41) - real.sqrt (4633)) / 12 :=
by
  sorry

end smallest_solution_l410_410408


namespace weights_lifting_equivalence_l410_410616

theorem weights_lifting_equivalence (w1 n1 w2 : ℕ) (H_w1 : w1 = 25) (H_n1 : n1 = 10) (H_w2 : w2 = 20) :
  60 * (750 / 60) = w1 * 3 * n1 :=
by
  rw [H_w1, H_n1, H_w2]
  exact rfl

end weights_lifting_equivalence_l410_410616


namespace solve_fraction_equation_l410_410803

theorem solve_fraction_equation :
  ∀ (x : ℚ), (5 * x + 3) / (7 * x - 4) = 4128 / 4386 → x = 115 / 27 := by
  sorry

end solve_fraction_equation_l410_410803


namespace find_smallest_n_l410_410694

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410694


namespace solve_for_y_l410_410223

theorem solve_for_y (y : ℝ): log 2 y - 4 * log 2 5 = -3 → y = 78.125 :=
by
  sorry

end solve_for_y_l410_410223


namespace max_det_A_l410_410046

open Real

-- Define the matrix and the determinant expression
noncomputable def A (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 1, 1],
    ![1, 1 + cos θ, 1],
    ![1 + sin θ, 1, 1]
  ]

-- Lean statement to prove the maximum value of the determinant of matrix A
theorem max_det_A : ∃ θ : ℝ, (Matrix.det (A θ)) ≤ 1/2 := by
  sorry

end max_det_A_l410_410046


namespace range_of_a_l410_410098

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1
noncomputable def g (x a : ℝ) : ℝ := 2^x - a

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ Icc (0:ℝ) 2, |f x1 - g x2 a| ≤ 2) ↔ (a ∈ Icc 2 5) :=
begin
  sorry
end

end range_of_a_l410_410098


namespace incorrect_conclusion_l410_410922

variable {a b : ℝ}

theorem incorrect_conclusion (h : (1 / a) < (1 / b) < 0) : ¬ (|a| + |b| > |a + b|) := 
by sorry

end incorrect_conclusion_l410_410922


namespace count_multiples_of_30_l410_410407

theorem count_multiples_of_30 (a b n : ℕ) (h1 : a = 900) (h2 : b = 27000) 
    (h3 : ∃ n, 30 * n = a) (h4 : ∃ n, 30 * n = b) : 
    (b - a) / 30 + 1 = 871 := 
by
    sorry

end count_multiples_of_30_l410_410407


namespace find_ordered_pair_l410_410571

noncomputable def eq_ordered_pair (p : ℝ) (c : ℤ) : Prop :=
∀ x : ℝ, 0 < x ∧ x < 10^(-100) → ↑c - 0.1 < x^p * (1 - (1 + x)^10) / (1 + (1 + x)^10) ∧ x^p * (1 - (1 + x)^10) / (1 + (1 + x)^10) < ↑c + 0.1

theorem find_ordered_pair (p : ℝ) (c : ℤ) (h : eq_ordered_pair p c) : (p, c) = (-1, -5) := sorry

end find_ordered_pair_l410_410571


namespace intersection_point_l410_410368

theorem intersection_point (x y : ℚ) 
  (h1 : 3 * y = -2 * x + 6) 
  (h2 : 2 * y = 7 * x - 4) :
  x = 24 / 25 ∧ y = 34 / 25 :=
sorry

end intersection_point_l410_410368


namespace angle_quadrant_l410_410289

theorem angle_quadrant (θ : ℝ) : θ = -2015 → 90 < θ % 360 ∧ θ % 360 < 180 :=
by
  intros h1 h2
  sorry

end angle_quadrant_l410_410289


namespace estimated_probability_is_correct_l410_410558

/-- A function that determines whether a group of 4 shots results 
    in hitting the target at least 3 times -/
def hits_target_at_least_three_times (shots: List ℕ) : Bool :=
  shots.filter (λ x => x ≥ 2).length ≥ 3

/-- Simulated groups of shots -/
def simulated_shots : List (List ℕ) :=
  [[7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7], [0, 3, 4, 7], [4, 3, 7, 3],
   [8, 6, 3, 6], [6, 9, 4, 7], [1, 4, 1, 7], [4, 6, 9, 8], [0, 3, 7, 1], [6, 2, 3, 3],
   [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1], [3, 6, 6, 1], [9, 5, 9, 7], [7, 4, 2, 4],
   [7, 6, 1, 0], [4, 2, 8, 1]] 

/-- Function to estimate the probability of hitting the target at least 3 times in 4 shots -/
def estimate_probability : List (List ℕ) → ℕ :=
  λ shots_groups =>
    (shots_groups.filter hits_target_at_least_three_times).length / shots_groups.length

theorem estimated_probability_is_correct :
  estimate_probability simulated_shots = 0.75 := by
  sorry

end estimated_probability_is_correct_l410_410558


namespace sum_of_areas_of_triangles_l410_410540

theorem sum_of_areas_of_triangles 
  (AB AC AD : ℝ)
  (hAB : AB = 30)
  (hAC : AC = 24)
  (hAD : AD = 18)
  (hAngleA : ∠A = 90) :
  (1/2 * AB * AC + 1/2 * AB * AD) = 630 :=
by
  sorry

end sum_of_areas_of_triangles_l410_410540


namespace marathon_time_l410_410335

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end marathon_time_l410_410335


namespace sum_from_1_to_15_fractions_l410_410805

def sum_of_fractions (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (k : ℚ) / 7

theorem sum_from_1_to_15_fractions :
  sum_of_fractions 15 = 120 / 7 :=
by
  sorry

end sum_from_1_to_15_fractions_l410_410805


namespace sum_of_repeating_decimals_l410_410862

theorem sum_of_repeating_decimals :
  (0.33333... : ℝ) + (0.004004004... : ℝ) + (0.0005000500050005... : ℝ) = 10099098 / 29970003 := 
  sorry

end sum_of_repeating_decimals_l410_410862


namespace max_n_l410_410088

def sum_first_n_terms (S n : ℕ) (a : ℕ → ℕ) : Prop :=
  S = 2 * a n - n

theorem max_n (S : ℕ) (a : ℕ → ℕ) :
  (∀ n, sum_first_n_terms S n a) → ∀ n, (2 ^ n - 1 ≤ 10 * n) → n ≤ 5 :=
by
  sorry

end max_n_l410_410088


namespace vinny_fifth_month_loss_l410_410275

theorem vinny_fifth_month_loss (start_weight : ℝ) (end_weight : ℝ) (first_month_loss : ℝ) (second_month_loss : ℝ) (third_month_loss : ℝ) (fourth_month_loss : ℝ) (total_loss : ℝ):
  start_weight = 300 ∧
  first_month_loss = 20 ∧
  second_month_loss = first_month_loss / 2 ∧
  third_month_loss = second_month_loss / 2 ∧
  fourth_month_loss = third_month_loss / 2 ∧
  (start_weight - end_weight) = total_loss ∧
  end_weight = 250.5 →
  (total_loss - (first_month_loss + second_month_loss + third_month_loss + fourth_month_loss)) = 12 :=
by
  sorry

end vinny_fifth_month_loss_l410_410275


namespace max_area_of_triangle_ABC_is_l410_410908

noncomputable def max_area_of_triangle 
  (A B C : ℝ) (a b c : ℝ) 
  (angle_A angle_B angle_C : Real.Angle) 
  (h1 : a = 3) 
  (h2 : (3 + b) * (Real.sin angle_A - Real.sin angle_B) = (c - b) * Real.sin angle_C)
  (h3 : a = (Real.sin angle_A * c) / (Real.sin angle_C)) 
  (h4 : b = (Real.sin angle_B * c) / (Real.sin angle_C)) : ℝ :=
  let area := (1 / 2) * b * c * Real.sin angle_A in
  if b = 3 ∧ c = 3 then 
    (9 * Real.sqrt 3) / 4 
  else 
    area 

theorem max_area_of_triangle_ABC_is :
  ∀ (A B C : ℝ) (a b c: ℝ)
  (angle_A angle_B angle_C : Real.Angle)
  (h1 : a = 3)
  (h2 : (3 + b) * (Real.sin angle_A - Real.sin angle_B) = (c - b) * Real.sin angle_C)
  (h3 : a = (Real.sin angle_A * c) / (Real.sin angle_C))
  (h4 : b = (Real.sin angle_B * c) / (Real.sin angle_C)),
  max_area_of_triangle A B C a b c angle_A angle_B angle_C h1 h2 h3 h4 = (9 * Real.sqrt 3) / 4 := 
sorry

end max_area_of_triangle_ABC_is_l410_410908


namespace solve_for_x_l410_410864

theorem solve_for_x (x : ℝ) :
    (8^x + 27^x) / (12^x + 18^x) = 8 / 7 ↔ 
    (x = Real.log (5 / 7) / Real.log (2 / 3) ∨ x = 0) :=
by
  sorry

end solve_for_x_l410_410864


namespace XY_squared_proof_l410_410180

theorem XY_squared_proof (ABC : Triangle) (ω: Circle) (B C T X Y : Point)
  (h₁: isosceles ABC A B C)
  (h₂: tangent ω B = T ∧ tangent ω C = T)
  (h₃: projection T (line_through A B) = X ∧ projection T (line_through A C) = Y)
  (h₄: BT = 18 ∧ CT = 18)
  (h₅: BC = 24)
  (h₆: TX^2 + TY^2 + XY^2 = 1288) :
  XY^2 = 864 := sorry

end XY_squared_proof_l410_410180


namespace max_value_sine_cosine_expression_l410_410872

theorem max_value_sine_cosine_expression (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sine_cosine_expression_l410_410872


namespace max_value_l410_410883

/-- 
Proof of the maximum value of the expression 
(sin x + sin 2y + sin 3z) * (cos x + cos 2y + cos 3z)
-/
theorem max_value (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_l410_410883


namespace remaining_credit_to_be_paid_l410_410581

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l410_410581


namespace find_brick_width_l410_410327

variable (length_courtyard : ℝ) (width_courtyard : ℝ) (num_bricks : ℕ) (length_brick : ℝ)

def width_of_brick (w : ℝ) : Prop :=
  let area_courtyard := length_courtyard * 100 * width_courtyard * 100
  let area_brick := length_brick * w
  let total_area_bricks := area_brick * num_bricks
  area_courtyard = total_area_bricks

theorem find_brick_width 
  (h_length_courtyard : length_courtyard = 18)
  (h_width_courtyard : width_courtyard = 16)
  (h_num_bricks : num_bricks = 14400)
  (h_length_brick : length_brick = 20) :
  width_of_brick 10 :=
by
  unfold width_of_brick
  rw [h_length_courtyard, h_width_courtyard, h_num_bricks, h_length_brick]
  simp
  sorry

end find_brick_width_l410_410327


namespace lim_sum_not_lim_diff_counter_example_l410_410577

-- Define fractional part
def fractional_part (x : ℝ) : ℝ := x - floor(x)

-- Define sequences x_n and y_n
axiom seq_x : ℕ → ℝ
axiom seq_y : ℕ → ℝ

-- Conditions given in the problem
axiom lim_x : tendsto (λ n, fractional_part (seq_x n)) at_top (𝓝 0)
axiom lim_y : tendsto (λ n, fractional_part (seq_y n)) at_top (𝓝 0)

-- Proofs to be provided (statements)
theorem lim_sum : tendsto (λ n, fractional_part (seq_x n + seq_y n)) at_top (𝓝 0) :=
sorry

theorem not_lim_diff : ¬tendsto (λ n, fractional_part (seq_x n - seq_y n)) at_top (𝓝 0) :=
sorry

-- Provide a counter-example for the second theorem
def seq_x_example (n : ℕ) := (0 : ℝ)
def seq_y_example (n : ℕ) := 1 / (n + 1)

theorem counter_example :
  tendsto (λ n, fractional_part (seq_x_example n - seq_y_example n)) at_top (𝓝 1) :=
sorry

end lim_sum_not_lim_diff_counter_example_l410_410577


namespace smallest_n_satisfies_conditions_l410_410667

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410667


namespace half_angle_in_quadrant_l410_410921

theorem half_angle_in_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 90 < α ∧ α < k * 360 + 180) :
  ∃ n : ℤ, (n * 360 + 45 < α / 2 ∧ α / 2 < n * 360 + 90) ∨ (n * 360 + 225 < α / 2 ∧ α / 2 < n * 360 + 270) :=
by sorry

end half_angle_in_quadrant_l410_410921


namespace smallest_n_satisfies_conditions_l410_410668

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410668


namespace f_sum_of_powers_l410_410190

variable {f : ℚ → ℝ} 

-- Assume the given conditions
axiom f_pos : ∀ {α : ℚ}, α ≠ 0 → f α > 0
axiom f_zero : f 0 = 0
axiom f_mul : ∀ {α β : ℚ}, f (α * β) = f α * f β

-- Statement to be proven
theorem f_sum_of_powers (x : ℤ) (n : ℕ) (hn : 0 < n) : 
  f (1 + ∑ i in finset.range (n + 1), (x : ℚ)^i) = 1 := 
sorry

end f_sum_of_powers_l410_410190


namespace equation_of_tangent_circle_l410_410929

theorem equation_of_tangent_circle (a : ℝ) :
  (∀ x y : ℝ, (a, 0) ≠ (3, sqrt 3) → (sqrt 3 / 3 * 3 = sqrt 3) →
    ((3 - a) ≠ 0 → (- sqrt 3 = sqrt 3 / (3 - a)) →
    ((3 - a) = 3 → a = 4 → (y - 0)^2 + (x - 4)^2 = 4))) := sorry

end equation_of_tangent_circle_l410_410929


namespace b_sequence_decreasing_l410_410725

-- Conditions: Define the sequences a_n and b_n
def a : ℕ → ℝ
| 0       := 1  -- using 0-indexing for convenience; a₀ = 1
| (n + 1) := real.sqrt (a n ^ 2 + 1 / a n)

def b (n : ℕ) : ℝ := a (n + 1) - a n

-- Theorem: Prove that the sequence b_n is strictly decreasing
theorem b_sequence_decreasing : ∀ n : ℕ, b n > b (n + 1) :=
by sorry

end b_sequence_decreasing_l410_410725


namespace combination_formula_l410_410807

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410807


namespace student_courses_last_year_l410_410772

variable (x : ℕ)
variable (courses_last_year : ℕ := x)
variable (avg_grade_last_year : ℕ := 100)
variable (courses_year_before : ℕ := 5)
variable (avg_grade_year_before : ℕ := 60)
variable (avg_grade_two_years : ℕ := 81)

theorem student_courses_last_year (h1 : avg_grade_last_year = 100)
                                   (h2 : courses_year_before = 5)
                                   (h3 : avg_grade_year_before = 60)
                                   (h4 : avg_grade_two_years = 81)
                                   (hc : ((5 * avg_grade_year_before) + (courses_last_year * avg_grade_last_year)) / (courses_year_before + courses_last_year) = avg_grade_two_years) :
                                   courses_last_year = 6 := by
  sorry

end student_courses_last_year_l410_410772


namespace geometric_sequence_problem_l410_410542

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem_l410_410542


namespace major_axis_length_ellipse_tangent_l410_410784

def elliptic_major_axis_length (f1 f2 : ℝ × ℝ) (tangentX tangentY : Prop) : ℝ :=
  if f1 = (1 + real.sqrt 3, 2) ∧ f2 = (1 - real.sqrt 3, 2) ∧ tangentX ∧ tangentY then 4 else 0

theorem major_axis_length_ellipse_tangent {f1 f2 : ℝ × ℝ}
  (tangentX tangentY : Prop)
  (h1 : f1 = (1 + real.sqrt 3, 2))
  (h2 : f2 = (1 - real.sqrt 3, 2))
  (hx : tangentX)
  (hy : tangentY) :
  elliptic_major_axis_length f1 f2 tangentX tangentY = 4 :=
begin
  simp [elliptic_major_axis_length, h1, h2, hx, hy],
  sorry
end

end major_axis_length_ellipse_tangent_l410_410784


namespace smallest_n_45_l410_410686

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l410_410686


namespace mass_of_man_l410_410292

theorem mass_of_man (L B h ρ : ℝ) (hL : L = 7) (hB : B = 2) (hh : h = 0.01) (hρ : ρ = 1000) :
  let V := L * B * h in
  let m := ρ * V in
  m = 140 :=
by
  sorry

end mass_of_man_l410_410292


namespace factorial_div_combination_l410_410823

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410823


namespace prove_line_PQ_through_circumcenter_of_ABC_l410_410409

-- Definitions for the points and the ratios given in the problem
variables {A B C P Q : Type}
variables [metric_space P]

-- Defining the conditions
def condition1 (A B C : P) : Prop := ¬collinear {A, B, C}

def condition2 (A B P : P) : Prop := (dist A P / dist B P) = 21 / 20

def condition3 (A B Q : P) : Prop := (dist A Q / dist B Q) = 21 / 20

def condition4 (B C P : P) : Prop := (dist B P / dist C P) = 20 / 19

def condition5 (B C Q : P) : Prop := (dist B Q / dist C Q) = 20 / 19

-- The theorem to be proven
theorem prove_line_PQ_through_circumcenter_of_ABC 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B P) 
  (h3 : condition3 A B Q) 
  (h4 : condition4 B C P) 
  (h5 : condition5 B C Q) : 
  ∃ O : P, is_circumcenter A B C O ∧ collinear {P, Q, O} :=
begin
  sorry
end

end prove_line_PQ_through_circumcenter_of_ABC_l410_410409


namespace smallest_n_satisfies_conditions_l410_410699

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410699


namespace inhabitable_fraction_l410_410516

theorem inhabitable_fraction 
  (total_land_fraction : ℚ)
  (inhabitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1 / 3)
  (h2 : inhabitable_land_fraction = 3 / 4):
  total_land_fraction * inhabitable_land_fraction = 1 / 4 := 
by
  sorry

end inhabitable_fraction_l410_410516


namespace smallest_n_satisfies_conditions_l410_410702

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410702


namespace number_of_boys_l410_410773

theorem number_of_boys (x : ℕ) (y : ℕ) (h1 : x + y = 8) (h2 : y > x) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end number_of_boys_l410_410773


namespace rescue_vehicle_accessible_area_l410_410765

-- Define the conditions
def speed_along_road : ℝ := 60  -- miles per hour
def speed_off_road : ℝ := 18    -- miles per hour
def max_time : ℝ := 1/6         -- hours (10 minutes)

-- Function to compute area (to be more formalized if needed)
noncomputable def accessible_area (time : ℝ) (speed_road : ℝ) (speed_desert : ℝ) : ℚ := 
  -- Placeholder quadratic function to simulate area for given conditions. To replace by actual derivations.
  (20 : ℚ) / (3 : ℚ)  -- Hypothetical result  

-- State the theorem
theorem rescue_vehicle_accessible_area : 
  ∃ p q : ℕ, (gcd p q = 1) ∧ (accessible_area max_time speed_along_road speed_off_road = (p : ℚ) / (q : ℚ)) ∧ (p + q = 23) :=
sorry

end rescue_vehicle_accessible_area_l410_410765


namespace remaining_credit_l410_410580

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l410_410580


namespace is_perfect_square_l410_410174

-- Definitions and conditions
variables {m n : ℕ}

-- Positive integers m and n such that m > n
def positive_integers (m n : ℕ) : Prop := m > 0 ∧ n > 0 ∧ m > n

-- m ≡ n (mod 2)
def mod_condition (m n : ℤ) : Prop := m % 2 = n % 2

-- (m^2 - n^2 + 1) divides (n^2 - 1)
def divisibility_condition (m n : ℤ) : Prop := (m ^ 2 - n ^ 2 + 1) ∣ (n ^ 2 - 1)

-- The theorem to be proven
theorem is_perfect_square (m n : ℤ) (h1: positive_integers m n) (h2: mod_condition m n) (h3: divisibility_condition m n) : 
  ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := sorry

end is_perfect_square_l410_410174


namespace sphere_radius_in_unit_cube_l410_410587

theorem sphere_radius_in_unit_cube :
  ∃ (r : ℝ), 
    (∃ center_sphere : ℝ × ℝ × ℝ, center_sphere = (1 / 2, 1 / 2, 1 / 2)) ∧ 
    (∀ i : fin 8, ∃ (sphere_i : ℝ × ℝ × ℝ), -- For each of the 8 surrounding spheres
      (sphere_i.1 = center_sphere.1 ∨ sphere_i.1 = 0 ∨ sphere_i.1 = 1) ∧  -- Tangent to one of the cube faces
      (sphere_i.2 = center_sphere.2 ∨ sphere_i.2 = 0 ∨ sphere_i.2 = 1) ∧
      (sphere_i.3 = center_sphere.3 ∨ sphere_i.3 = 0 ∨ sphere_i.3 = 1) ∧
      (dist (sphere_i.1, sphere_i.2, sphere_i.3) (center_sphere.1, center_sphere.2, center_sphere.3) = 2 * r)) ∧
    (√3 = 4 * r + 2 * √3 * r) ∧
    r = (2 * √3 - 3) / 2 :=
begin
  sorry
end

end sphere_radius_in_unit_cube_l410_410587


namespace combination_formula_l410_410813

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410813


namespace max_min_extremum_three_distinct_zeros_arithmetic_sequence_relationship_a_b_tangent_slopes_l410_410951

noncomputable def f (a b x : ℝ) : ℝ := x ^ 3 + a * x ^ 2 + b * x + 1

section part_1_1

variable (a b : ℝ)
variable (hx : (a^2 + b = 0))
variable (ha : a > 0)

theorem max_min_extremum :
  (f a b (-a) = 1 + a^3 ∧ f a b (a / 3) = 1 - 5*a^3/27) :=
  sorry
end part_1_1

section part_1_2

variable (a b : ℝ)
variable (hx : (a^2 + b = 0))
variable (h_distinct_zeros : f a b x has three distinct zeros)

theorem three_distinct_zeros_arithmetic_sequence :
  ∃ a, f a b x has three distinct zeros forming arithmetic sequence ↔ a = - (3 / (11^(1 / 3))) :=
  sorry
end part_1_2

section part_2

variable (a b : ℝ)
variable (hx : (a^2 + b = 0))
variable (k1 k2 : ℝ)
variable (h_slope : k2 = 4 * k1)

theorem relationship_a_b_tangent_slopes :
  ∃ a b, a^2 = 3 * b :=
  sorry
end part_2

end max_min_extremum_three_distinct_zeros_arithmetic_sequence_relationship_a_b_tangent_slopes_l410_410951


namespace intersection_length_AB_correct_l410_410544

variable {α : Type*}

/-- Parametric equations of the curve C. --/
def curve_C (α : ℝ) : ℝ × ℝ := 
  (2 * Real.cos α + Real.sqrt 3, 2 * Real.sin α)

/-- Cartesian equation of the line l from polar form. --/
def line_l (θ : ℝ) : ℝ × ℝ → Prop := 
  λ p, θ = π / 6 → p.2 = Real.sqrt 3 / 3 * p.1

/-- Distance formula for length AB. --/
noncomputable def length_AB : ℝ :=
  2 * Real.sqrt (4 - (Real.sqrt 3 / 2)^2)

/-- Proof statement: If line l intersects curve C at points A and B,
    then the length of segment AB is sqrt 13. --/
theorem intersection_length_AB_correct :
  ∀ (α1 α2 : ℝ), curve_C α1 = curve_C α2 →
  length_AB = Real.sqrt 13 :=
by
  intros
  sorry

end intersection_length_AB_correct_l410_410544


namespace product_of_all_possible_t_l410_410399

-- Define the set of possible integer pairs (a, b) such that ab = 12.
def possible_pairs : Set (Int × Int) :=
  {(a, b) | a * b = 12}

-- Define the set of possible values of t = a + b for those pairs (a, b).
def possible_values_of_t : Set Int :=
  {t | ∃ (a b : Int), (a, b) ∈ possible_pairs ∧ t = a + b}

-- Lean 4 theorem to prove the product of all possible values of t.
theorem product_of_all_possible_t : 
  ∏ t in possible_values_of_t, t = 532224 :=
sorry

end product_of_all_possible_t_l410_410399


namespace necessary_condition_for_inequality_l410_410944

noncomputable def f (x : ℝ) : ℝ :=
  abs x + (Real.sin x)^2

theorem necessary_condition_for_inequality (x1 x2 : ℝ) :
  f x1 > f x2 → abs x1 > x2 :=
begin
  sorry
end

end necessary_condition_for_inequality_l410_410944


namespace bank_discount_correctness_l410_410228

noncomputable def bankersDiscount (TD t T : ℝ) : ℝ :=
  TD + (TD * t) / (T - t)

def bill1 : ℝ := 8000
def bill2 : ℝ := 10000
def bill3 : ℝ := 12000
def bill4 : ℝ := 15000

def TD1 : ℝ := 360
def t1 : ℝ := 60
def TD2 : ℝ := 450
def t2 : ℝ := 90
def TD3 : ℝ := 480
def t3 : ℝ := 120
def TD4 : ℝ := 500
def t4 : ℝ := 150

def T : ℝ := 360

def BD1 := bankersDiscount TD1 t1 T
def BD2 := bankersDiscount TD2 t2 T
def BD3 := bankersDiscount TD3 t3 T
def BD4 := bankersDiscount TD4 t4 T

def TotalInvestment := bill1 + bill2 + bill3 + bill4

def SumProducts := (bill1 * BD1) + (bill2 * BD2) + (bill3 * BD3) + (bill4 * BD4)

def AvgBD := SumProducts / TotalInvestment

theorem bank_discount_correctness : BD1 = 432 ∧ BD2 = 600 ∧ BD3 = 720 ∧ BD4 = 857.14 ∧
                                     TotalInvestment = 44000 ∧ SumProducts = 30771100 ∧ 
                                     AvgBD = 699.34 :=
by 
  sorry

end bank_discount_correctness_l410_410228


namespace two_times_six_pow_n_plus_one_ne_product_of_consecutive_l410_410597

theorem two_times_six_pow_n_plus_one_ne_product_of_consecutive (n k : ℕ) :
  2 * (6 ^ n + 1) ≠ k * (k + 1) :=
sorry

end two_times_six_pow_n_plus_one_ne_product_of_consecutive_l410_410597


namespace vasya_can_place_99_checkers_l410_410208

/-- 
  Given a 50x50 board where Petya has placed some checkers, with at most one per cell,
  prove that Vasya can place at most 99 new checkers such that each row and each column 
  of the board contains an even number of checkers.
-/
theorem vasya_can_place_99_checkers (board : Fin 50 → Fin 50 → Bool) (h_checker_placed : ∀ i j, board i j = true → True) : 
  ∃ newCheckers : Fin 50 → Fin 50 → Bool,
    (∑ i j, if newCheckers i j then 1 else 0 ≤ 99) ∧
    (∀ i, (∑ j, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) ∧
    (∀ j, (∑ i, if board i j ∨ newCheckers i j then 1 else 0) % 2 = 0) := 
by
  sorry

end vasya_can_place_99_checkers_l410_410208


namespace factorial_div_eq_l410_410845

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410845


namespace angle_between_MF_and_AB_is_90_minimum_area_of_triangle_MAB_is_9_div_2_l410_410566

noncomputable def hyperbola := {p : ℝ × ℝ | p.1^2 - (p.2^2 / 3) = 1}

def left_focus : ℝ × ℝ := (-2, 0)

def line_through_focus (m : ℝ) : set (ℝ × ℝ) := {p | p.1 = m * p.2 - 2}

def points_on_hyperbola (m : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1, p.2) ∈ hyperbola ∧ p ∈ line_through_focus m}

def tangents_at_points (p : ℝ × ℝ) : set (ℝ × ℝ → ℝ) := 
  {l | ∀ q ∈ hyperbola, l (p.1) * q.1 - (q.2 * l (p.2) / 3) = 1}

def intersection_point_of_tangents (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  lattice.inf (tangents_at_points p1 ∩ tangents_at_points p2)

theorem angle_between_MF_and_AB_is_90:
  ∀ m A B M,
  A ∈ points_on_hyperbola m →
  B ∈ points_on_hyperbola m →
  M = intersection_point_of_tangents A B →
  let F := left_focus in (vector_angle (M - F) (B - A) = π / 2) := 
sorry

theorem minimum_area_of_triangle_MAB_is_9_div_2:
  ∀ m A B M,
  A ∈ points_on_hyperbola m →
  B ∈ points_on_hyperbola m →
  M = intersection_point_of_tangents A B →
  let area := (1 / 2) * abs ((B.1 - A.1) * (M.2 - A.2) - (B.2 - A.2) * (M.1 - A.1)) in 
  (∀ₘ, area m ≥ 9 / 2) := 
sorry

end angle_between_MF_and_AB_is_90_minimum_area_of_triangle_MAB_is_9_div_2_l410_410566


namespace appointment_schemes_count_l410_410905

/-- Given 5 male teachers and 4 female teachers, the total number of
different appointment schemes for selecting 3 teachers to serve as class
advisers (one adviser per class), with the requirement that both male and
female teachers must be included is 420. -/
theorem appointment_schemes_count :
  let male_teachers := 5
  let female_teachers := 4
  ∃ (ans : ℕ), 
  (ans = 420 ∧ 
    ((
      (nat.choose male_teachers 2) * 
      (nat.choose female_teachers 1) + 
      (nat.choose male_teachers 1) * 
      (nat.choose female_teachers 2)
    ) * nat.choose 3 3 = ans)) :=
by
  use 420
  sorry

end appointment_schemes_count_l410_410905


namespace nancy_total_spending_l410_410743

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l410_410743


namespace range_of_cos_squared_minus_cos_l410_410643

def cos_squared_minus_cos (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.cos x

theorem range_of_cos_squared_minus_cos : 
  ∃ a b, a = -1/4 ∧ b = 2 ∧ ∀ y, y = cos_squared_minus_cos x → y ∈ Set.Icc a b :=
sorry

end range_of_cos_squared_minus_cos_l410_410643


namespace probability_of_drawing_two_white_balls_l410_410525

-- Define the total number of balls and their colors
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

-- Define the total number of ways to draw 2 balls from 4
def total_draw_ways : ℕ := (total_balls.choose 2)

-- Define the number of ways to draw 2 white balls
def white_draw_ways : ℕ := (white_balls.choose 2)

-- Define the probability of drawing 2 white balls
def probability_white_draw : ℚ := white_draw_ways / total_draw_ways

-- The main theorem statement to prove
theorem probability_of_drawing_two_white_balls :
  probability_white_draw = 1 / 6 := by
  sorry

end probability_of_drawing_two_white_balls_l410_410525


namespace identify_pastry_eater_l410_410584

-- The bear cubs and tiger cub identifiers.
inductive Cub
| Bear1 | Bear2 | Bear3 | Tiger

-- Denote a pastry.
axiom Pastry : Type

-- Two pastries are missing
axiom pastries_missing : Fin 2 Pastry

-- Define the weighing results
inductive WeighingResult
| less | equal | greater

-- Function to represent the result of weighing two sides
axiom weigh : (Cub × Cub) → WeighingResult
axiom weigh_with_pastry : (Cub × (Cub × Pastry)) → WeighingResult
axiom weigh_with_pastry2 : ((Cub × Pastry) × Cub) → WeighingResult

-- Define conditions of the problem
axiom all_bears_weight_same : ∀ (b1 b2 : Cub), b1 ≠ b2 → 
                             (b1 = Cub.Bear1 ∨ b1 = Cub.Bear2 ∨ b1 = Cub.Bear3) → 
                             (b2 = Cub.Bear1 ∨ b2 = Cub.Bear2 ∨ b2 = Cub.Bear3) → 
                              weigh (b1, b2) = WeighingResult.equal
axiom tiger_diet : ∀ (b : Cub), b = Cub.Tiger → 
                   (∃ x, (x = Pastry) → (tiger_diet (Cub.Bear1) = 0 ∨ tiger_diet (Cub.Bear2) = 0 ∨ 
                   tiger_diet (Cub.Bear3) = 0))

-- Prove we can determine who ate the pastries in two weighings
theorem identify_pastry_eater : ∃ w1 w2,
  (w1 = weigh (Cub.Bear1, Cub.Bear2) ∧
   (w1 = WeighingResult.greater → (w2 = weigh_with_pastry (Cub.Bear1, (Cub.Bear3, Pastry)) ∧
     (w2 = WeighingResult.less → (Cub.Bear1 ∧ Cub.Bear3)) ∧
     (w2 = WeighingResult.equal → (Cub.Bear1 ∧ Cub.Tiger)) ∧
     (w2 = WeighingResult.greater → (Cub.Bear1)) )) ∨
   (w1 = WeighingResult.equal → (w2 = weigh_with_pastry2 ((Cub.Bear1, Pastry), Cub.Bear3) ∧
     (w2 = WeighingResult.less → (Cub.Bear3)) ∧
     (w2 = WeighingResult.equal → (Cub.Bear3 ∧ Cub.Tiger)) ∧
     (w2 = WeighingResult.greater → (Cub.Bear1 ∧ Cub.Bear2)) )))
:= sorry

end identify_pastry_eater_l410_410584


namespace func_increasing_l410_410022

noncomputable def func (x : ℝ) : ℝ :=
  x^3 + x + 1

theorem func_increasing : ∀ x : ℝ, deriv func x > 0 := by
  sorry

end func_increasing_l410_410022


namespace simplify_and_evaluate_expression_l410_410607

theorem simplify_and_evaluate_expression (a : ℕ) 
  (h1 : 2*a + 1 < 3*a + 3) 
  (h2 : (2/3) * (a - 1) ≤ (1/2) * (a + 1/3))
  (h3 : a ≠ 0) 
  (h4 : a ≠ 1) 
  (h5 : a ≠ 2) : 
    (a + 1 - (4 * a - 5)/(a - 1)) ÷ ((1/a) - (1/(a^2 - a))) = a * (a - 2) ∧ 
      (a = 3 ∨ a = 4 ∨ a = 5) →
      (a = 3 → a * (a-2) = 3) ∧ (a = 4 → a * (a-2) = 8) ∧ (a = 5 → a * (a-2) = 15) := 
by {
  sorry -- This skips the proof. 
}

end simplify_and_evaluate_expression_l410_410607


namespace number_of_ways_to_distribute_balls_l410_410493

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l410_410493


namespace average_speed_l410_410311

theorem average_speed 
 (h1 : ∀ t : ℕ, t ≤ 4 → 45 = 45)
 (h2 : ∀ t : ℕ, t > 4 → 75 = 75)
 (total_time: ℕ := 12):
 (let distance_1 := 45 * 4,
      distance_2 := 75 * 8,
      total_distance := distance_1 + distance_2,
      average_speed := total_distance / 12 in average_speed = 65) := 
by
  sorry

end average_speed_l410_410311


namespace distinct_values_of_g_l410_410014

noncomputable def phi (n : ℕ) : ℕ := nat.totient n

noncomputable def g (x : ℝ) : ℕ :=
  (∑ k in finset.range 10 + 3, (real.floor (k * x) - (k + 1) * real.floor x))

theorem distinct_values_of_g : (finset.range 10 + 3).sum (λ k, phi k) + 1 = 45 :=
by calc
  (finset.range 10 + 3).sum (λ k, phi k) + 1 = 44 + 1 : by sorry
  ... = 45 : by sorry

end distinct_values_of_g_l410_410014


namespace nancy_total_spending_l410_410742

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l410_410742


namespace length_of_CD_l410_410198

theorem length_of_CD 
  (A B C D : Point)
  (h_AB : dist A B = 48)
  (h_mid_C : midpoint C A B)
  (h_AD : dist A D = (1/3) * dist A B) :
  dist C D = 8 := 
sorry

end length_of_CD_l410_410198


namespace arithmetic_sequence_problem_l410_410537

variable {d : Real} (h : d ≠ 0)

def a (n : ℕ) : Real := (n - 1) * d

theorem arithmetic_sequence_problem :
  let a := λ n => (n - 1) * d
  let a_m := a m
  let sum_a_1_to_9 := Finset.range 9 |> Finset.sum (λ n => a (n + 1))
  a_m = sum_a_1_to_9 → m = 37 :=
by
  sorry

end arithmetic_sequence_problem_l410_410537


namespace proof_problem_b_l410_410293

-- Definitions for the given problem
constants (S_1 S_2 S_3 : Type)  -- representing three concentric circles
constants (A B C : Type)        -- representing points of intersection
constant l : Type               -- representing the line intersecting the circles

-- Conditions
axiom intersection_S1_l : A
axiom intersection_S2_l : B
axiom intersection_S3_l : C

-- Given condition: AB = BC
axiom geometric_relation : (AB AC : ℝ) (h1 : AB / AC = 1 / 2)

-- Statement to prove
theorem proof_problem_b : AB = BC := by 
  sorry

end proof_problem_b_l410_410293


namespace determine_j_l410_410184

theorem determine_j (a b c x : ℤ) (h1: f(1) = 0) 
  (h2: 60 < f(7) ∧ f(7) < 70) 
  (h3: 80 < f(8) ∧ f(8) < 90) 
  (h4: 1000 * j < f(10) ∧ f(10) < 1000 * (j + 1)) :
  j = 0 := sorry

end determine_j_l410_410184


namespace max_min_XY_XZ_diff_zero_l410_410551

theorem max_min_XY_XZ_diff_zero (YZ : ℝ) (XM : ℝ) (M : ℝ → ℝ) (x : ℝ) (h : ℝ):
  (YZ = 10) →
  (XM = 6) →
  (∀ y z : ℝ, M y = M z) →
  ∃ N n : ℝ, N = n ∧ N - n = 0 :=
by
  intro YZ_eq XM_eq M_midpoint
  use 122
  use 122
  split
  · sorry -- Proof that N = n = 122
  · sorry -- Proof that N - n = 0

end max_min_XY_XZ_diff_zero_l410_410551


namespace cycle_selling_price_l410_410713

theorem cycle_selling_price
(C : ℝ := 1900)  -- Cost price of the cycle
(Lp : ℝ := 18)  -- Loss percentage
(S : ℝ := 1558) -- Expected selling price
: (S = C - (Lp / 100) * C) :=
by 
  sorry

end cycle_selling_price_l410_410713


namespace table_covered_with_three_layers_l410_410049

theorem table_covered_with_three_layers (A T X two_layers four_layers three_layers : ℝ) :
  -- Given conditions
  A = 360 →
  T = 250 →
  (0.9 * T) = 0.9 * 250 →
  two_layers = 35 →
  four_layers = 15 →
  -- Overlapping area relationships
  X + 2 * two_layers + 3 * three_layers + 4 * four_layers + (T - (X + 2 * two_layers + 3 * three_layers + 4 * four_layers)) = A →
  X + 2 * two_layers + 3 * three_layers + 4 * four_layers = (0.9 * T) →
  -- Required to show
  three_layers = 65 :=
begin
   sorry
end

end table_covered_with_three_layers_l410_410049


namespace joan_remaining_kittens_l410_410168

-- Definitions based on the given conditions
def original_kittens : Nat := 8
def kittens_given_away : Nat := 2

-- Statement to prove
theorem joan_remaining_kittens : original_kittens - kittens_given_away = 6 := 
by
  -- Proof skipped
  sorry

end joan_remaining_kittens_l410_410168


namespace average_age_l410_410238

theorem average_age (avg_age_students : ℝ) (num_students : ℕ) (avg_age_teachers : ℝ) (num_teachers : ℕ) :
  avg_age_students = 13 → 
  num_students = 40 → 
  avg_age_teachers = 42 → 
  num_teachers = 60 → 
  (num_students * avg_age_students + num_teachers * avg_age_teachers) / (num_students + num_teachers) = 30.4 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_l410_410238


namespace dice_sum_probability_l410_410272

theorem dice_sum_probability :
  let outcomes := {(x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6}}
  let favorable := {(x, y) | (x, y) ∈ outcomes ∧ (x + y = 4 ∨ x + y = 8 ∨ x + y = 12)}
  (favorable.card.to_rat / outcomes.card.to_rat) = 1 / 4 :=
by
  -- The proof would be filled here
  sorry

end dice_sum_probability_l410_410272


namespace integer_squares_l410_410379

theorem integer_squares (x y : ℤ) 
  (hx : ∃ a : ℤ, x + y = a^2)
  (h2x3y : ∃ b : ℤ, 2 * x + 3 * y = b^2)
  (h3xy : ∃ c : ℤ, 3 * x + y = c^2) : 
  x = 0 ∧ y = 0 := 
by { sorry }

end integer_squares_l410_410379


namespace right_angle_division_l410_410727

theorem right_angle_division (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
    (h5 : α + β = π / 2) : (sin α - sin β = sqrt 2 / 2) :=
sorry

end right_angle_division_l410_410727


namespace determine_a_l410_410937

theorem determine_a (h1: ∀ x : ℝ, exp (|x - 1|) - m > 0)
                    (h2: ∀ m : ℝ, m < a):
  a = 1 :=
sorry

end determine_a_l410_410937


namespace problem1_problem2_l410_410801

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l410_410801


namespace distance_BC_is_sqrt_30_l410_410939

noncomputable def intersection_distance (t : ℝ) : ℝ := 
  let sin_30 := 1 / 2 -- sin(30°) = 1/2
  let x := 2 - t * sin_30
  let y := -1 + t * sin_30
  if (x^2 + y^2 = 8 ∧ y = 1 - x) then
    real.dist (2 - (0 - sin_30)) (-1 + (1 - sin_30))
  else 0

theorem distance_BC_is_sqrt_30 : ∃ t : ℝ, intersection_distance t = sqrt 30 :=
sorry

end distance_BC_is_sqrt_30_l410_410939


namespace cos_90_eq_zero_l410_410004

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410004


namespace compute_expression_l410_410192

theorem compute_expression (zeta : ℂ) (h1 : zeta^4 = 1) (h2 : ¬ (zeta ∈ ℝ)) :
  ((1 - zeta + zeta^3)^4 + (1 + zeta - zeta^3)^4) = -14 - 48 * complex.I :=
by
  sorry

end compute_expression_l410_410192


namespace place_kings_on_board_l410_410413

-- Define the board structure and conditions
structure Chessboard (n : ℕ) where
  size : ℕ := n * n
  corners_removed : Fin n × Fin n

-- Define a king's attackable positions
def attacks (pos1 pos2 : Fin 13 × Fin 13) : Prop :=
  abs (pos1.1 - pos2.1) ≤ 1 ∧ abs (pos1.2 - pos2.2) ≤ 1

-- Define a board with 2 corners removed
def initial_board := { size := 13 * 13 - 2, corners_removed := ((0, 0), (12, 12)) }

-- Define the total number of kings
def Kings_num : ℕ := 47

-- Main statement of the problem
theorem place_kings_on_board :
  ∃ (marked : Fin 13 × Fin 13 → Prop) (num_kings : Fin 47 → Fin 13 × Fin 13),
  ∀ (pos : Fin 13 × Fin 13), (¬marked pos → (∃ k : Fin 47, attacks (num_kings k) pos)) :=
sorry

end place_kings_on_board_l410_410413


namespace circle_center_l410_410625

theorem circle_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = 3) (h3 : x2 = 8) (h4 : y2 = -5) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (5, -1) :=
by 
  rw [h1, h2, h3, h4]
  have hx : (2 + 8) / 2 = 5 := by norm_num
  have hy : (3 + (-5)) / 2 = -1 := by norm_num
  rw [hx, hy]
  exact ⟨rfl, rfl⟩

end circle_center_l410_410625


namespace vasya_can_place_99_checkers_l410_410212

theorem vasya_can_place_99_checkers (board : Fin 50 × Fin 50 → Prop) :
  (∀ i j, board i j → ¬ board i j) → ∃ (new_checkers : Fin 50 × Fin 50 → Prop),
  (∀ i j, board i j → ¬ new_checkers i j) ∧
  (∑ i, if new_checkers i then 1 else 0 ≤ 99) ∧
  (∀ i : Fin 50, even (∑ j, if new_checkers (i, j) then 1 else 0)) ∧
  (∀ j : Fin 50, even (∑ i, if new_checkers (i, j) then 1 else 0)) := sorry

end vasya_can_place_99_checkers_l410_410212


namespace angle_in_fourth_quadrant_l410_410026

def quadrant (θ : ℝ) : String :=
  if 0 ≤ θ ∧ θ < 90 then "First quadrant"
  else if 90 ≤ θ ∧ θ < 180 then "Second quadrant"
  else if 180 ≤ θ ∧ θ < 270 then "Third quadrant"
  else if 270 ≤ θ ∧ θ < 360 then "Fourth quadrant"
  else "Out of range"

def coterminal_angle (θ : ℝ) : ℝ :=
  θ % 360

theorem angle_in_fourth_quadrant : quadrant (coterminal_angle (-1120)) = "Fourth quadrant" :=
by
  sorry

end angle_in_fourth_quadrant_l410_410026


namespace square_root_ratio_area_l410_410612

theorem square_root_ratio_area (side_length_C side_length_D : ℕ) (hC : side_length_C = 45) (hD : side_length_D = 60) : 
  Real.sqrt ((side_length_C^2 : ℝ) / (side_length_D^2 : ℝ)) = 3 / 4 :=
by
  rw [hC, hD]
  sorry

end square_root_ratio_area_l410_410612


namespace problem_l410_410984

theorem problem (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -2 := 
by
  -- the proof will go here but is omitted
  sorry

end problem_l410_410984


namespace max_value_sine_cosine_expression_l410_410874

theorem max_value_sine_cosine_expression (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sine_cosine_expression_l410_410874


namespace probability_draw_add_oil_l410_410412

theorem probability_draw_add_oil :
  let balls := ["光", "山", "加", "油"]
  let outcomes := { (x, y) // x ≠ y ∧ x ∈ balls ∧ y ∈ balls }
  let favorable := { (x, y) ∈ outcomes | (x = "加" ∧ y = "油") ∨ (x = "油" ∧ y = "加") }
  let total_outcomes := finset.card outcomes
  let favorable_outcomes := finset.card favorable
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6
:= by {
  sorry
}

end probability_draw_add_oil_l410_410412


namespace equilateral_triangle_data_l410_410620

theorem equilateral_triangle_data
  (A : ℝ)
  (b : ℝ)
  (ha : A = 450)
  (hb : b = 25)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a) :
  ∃ (h P : ℝ), h = 36 ∧ P = 75 := by
  sorry

end equilateral_triangle_data_l410_410620


namespace polar_to_cartesian_and_chord_length_l410_410164

theorem polar_to_cartesian_and_chord_length :
  ∀ (ρ θ: ℝ),
  ρ * sin (θ + π / 4) = √2 →
  ∃ (x y t: ℝ),
  x = cos t ∧ y = 1 + sin t ∧
  (x + y = 2) ∧
  (x^2 + (y - 1)^2 = 1) →
  let d := 1 / √2 in 
  let chord_length := 2 * sqrt(1 - d^2) in 
  chord_length = sqrt 2 :=
by 
  intros ρ θ h1 x y t h2 h3 h4 h5;
  let d := 1 / sqrt 2;
  let chord_length := 2 * sqrt (1 - d^2);
  sorry

end polar_to_cartesian_and_chord_length_l410_410164


namespace solution_concentration_40_percent_l410_410762

variable (P : ℝ)
variable (x : ℝ := 0.7142857142857143)
variable (initial_concentration final_concentration : ℝ)
noncomputable def replaced_solution_concentration := P

theorem solution_concentration_40_percent :
  let initial_concentration := 0.9 in
  let final_concentration := 0.4 in
  (initial_concentration * (1 - x) + replaced_solution_concentration * x = final_concentration) ↔ (replaced_solution_concentration = 0.2) :=
by sorry

end solution_concentration_40_percent_l410_410762


namespace only_pair_dividing_power_sum_l410_410297

theorem only_pair_dividing_power_sum : 
  ∀ (p q : ℕ), p.prime ∧ q.prime → (3 * p ^ (q - 1) + 1) ∣ (11 ^ p + 17 ^ p) ↔ (p = 3 ∧ q = 3) := 
by
  sorry

end only_pair_dividing_power_sum_l410_410297


namespace balls_in_boxes_l410_410488

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l410_410488


namespace number_of_pop_albums_l410_410709

theorem number_of_pop_albums (P : ℕ) (h1 : ∀ c p, p = 6 c → c * 9 + p * 9 = 72) : P = 2 :=
by
  have h2 : 6 * 9 = 54, by norm_num -- calculates and confirms 54 songs from country albums
  have h3 : 72 - 54 = 18, by norm_num -- calculates and confirms 18 songs from pop albums
  have h4 : 18 / 9 = 2, by norm_num -- calculates and confirms 2 pop albums from 18 songs
  exact h4

end number_of_pop_albums_l410_410709


namespace sum_possible_values_base2_l410_410316

theorem sum_possible_values_base2 (d_8 : ℕ) (h1 : d_8 >= 8^4) (h2 : d_8 < 8^5) :
  d_8.bits ≠ 0 → (d_8.bits.card = 13 ∨ d_8.bits.card = 14 ∨ d_8.bits.card = 15) →
  d_8.bits.card = 13 + d_8.bits.card = 14 + d_8.bits.card = 15 := sorry

end sum_possible_values_base2_l410_410316


namespace cos_90_eq_zero_l410_410005

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410005


namespace number_of_factors_8100_l410_410279

/-- 8,100 has a total of 45 positive factors -/
theorem number_of_factors_8100 : ∃ n : ℕ, n = 8100 ∧ (number_of_factors n = 45) := 
sorry

end number_of_factors_8100_l410_410279


namespace polynomial_coefficients_l410_410992

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℕ)
  (h : (1 + 2 * x)^10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^{10}) :
  a_0 = 1 ∧ 
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = 3^10 - 1) ∧ 
  (∃ r, r = 5 ∧ a_r = 10 choose 5) ∧ 
  a_2 = 9 * a_1 := 
by
  -- Proof omitted.
  sorry

end polynomial_coefficients_l410_410992


namespace number_of_ways_to_distribute_balls_l410_410495

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l410_410495


namespace sum_of_all_alternating_sums_nine_l410_410900

-- Condition definitions
def S (n : ℕ) : Finset ℕ := {i | 1 ≤ i ∧ i ≤ n}.to_finset
def special_subsets (n : ℕ) : List (Finset ℕ) := S n.powerset.toList.filter (λ subset, subset ≠ ∅)
def alternating_sum (s : Finset ℕ) : ℤ :=
  let desc_list := s.sort (· ≥ ·)
  let alt_sum := desc_list.enum.filterMap (λ (idx, val), if idx % 2 = 0 then some val else some (-val))
  alt_sum.sum

def alternating_sum_with_9 (s : Finset ℕ) : ℤ :=
  if 9 ∈ s then alternating_sum s + 9 else alternating_sum s

-- Question and proof statement
theorem sum_of_all_alternating_sums_nine : 
  ∑ s in special_subsets 9, alternating_sum_with_9 s = 2304 :=
by
  sorry

end sum_of_all_alternating_sums_nine_l410_410900


namespace min_sum_l410_410024

theorem min_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    (a / (3 * b) + b / (6 * c) + c / (12 * a) + (a + b + c) / (5 * a * b * c)) 
    ≥ (4 / Real.root 360 4) :=
by
  sorry

end min_sum_l410_410024


namespace tan_sum_eq_l410_410118

theorem tan_sum_eq (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1/2 :=
by sorry

end tan_sum_eq_l410_410118


namespace exists_odd_k_l410_410564

noncomputable def f (n : ℕ) : ℕ :=
sorry

theorem exists_odd_k : 
  (∀ m n : ℕ, f (m * n) = f m * f n) → 
  (∀ m n : ℕ, (m + n) ∣ (f m + f n)) → 
  ∃ k : ℕ, (k % 2 = 1) ∧ (∀ n : ℕ, f n = n ^ k) :=
sorry

end exists_odd_k_l410_410564


namespace series_converges_to_4_l410_410039

def series_term (n : ℕ) : ℝ :=
  let k := ⌊Real.sqrt n⌋₊ + 1 in
  (3^k + 3^(-k)) / 3^n

noncomputable def series_sum : ℝ :=
  ∑' n, series_term n

theorem series_converges_to_4 :
  series_sum = 4 :=
sorry

end series_converges_to_4_l410_410039


namespace furniture_shop_cost_price_l410_410259

noncomputable def cost_price (SP : ℝ) (x : ℝ) : ℝ := SP / x

theorem furniture_shop_cost_price (SP : ℝ) (markup : ℝ) (CP : ℝ) 
  (h1 : SP = CP * markup)
  (h2 : SP = 8337)
  (h3 : markup = 1.20) : CP = 6947.5 :=
by
  rw [h2, h3] at h1
  rw ←h1
  sorry

end furniture_shop_cost_price_l410_410259


namespace total_miles_traveled_is_17_l410_410033

noncomputable def travel_distance_day (minutes_per_mile : ℕ) : ℕ :=
  60 / minutes_per_mile

def total_travel_distance : ℕ :=
  let times_per_mile := [6, 12, 30] -- valid days: 6, 12, 30 (minutes per mile)
  times_per_mile.map travel_distance_day |>.sum

theorem total_miles_traveled_is_17 :
  total_travel_distance = 17 :=
by
  sorry

end total_miles_traveled_is_17_l410_410033


namespace find_expression_value_l410_410445

-- We declare our variables x and y
variables (x y : ℝ)

-- We state our conditions as hypotheses
def h1 : 3 * x + y = 5 := sorry
def h2 : x + 3 * y = 8 := sorry

-- We prove the given mathematical expression
theorem find_expression_value (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 10 * x^2 + 19 * x * y + 10 * y^2 = 153 := 
by
  -- We intentionally skip the proof
  sorry

end find_expression_value_l410_410445


namespace frog_arrangement_l410_410606

def num_ways_to_arrange_frogs (total_frogs green red yellow blue : ℕ) (valid_positioning : set (list (ℕ × ℕ))) : ℕ :=
  if (total_frogs = 7 ∧ green = 2 ∧ red = 2 ∧ yellow = 2 ∧ blue = 1) then
    16
  else
    0

theorem frog_arrangement :
  num_ways_to_arrange_frogs 7 2 2 2 1 {
    l | ∀ i, (l[i].1 = 1 → l[i+1].1 ≠ 2) ∧ (l[i].1 = 2 → l[i+1].1 ≠ 1)
       ∧ (l[i].1 = 3 → l[i+1].1 ≠ 4) ∧ (l[i].1 = 4 → l[i+1].1 ≠ 3)
  } = 16 :=
by sorry

end frog_arrangement_l410_410606


namespace trapezoid_bc_squared_l410_410547

theorem trapezoid_bc_squared (BC AB CD AD : ℝ) 
  (h1 : BC ⊥ AB) 
  (h2 : BC ⊥ CD) 
  (h3 : (diagonal_BO AC ⊥ diagonal_BO BD))
  (h4 : AB = 5) 
  (h5 : AD = 9 * (Real.sqrt 5)) :
  BC^2 = 55 := 
sorry

end trapezoid_bc_squared_l410_410547


namespace non_neg_sequence_l410_410173

theorem non_neg_sequence (a : ℝ) (x : ℕ → ℝ) (h0 : x 0 = 0)
  (h1 : ∀ n, x (n + 1) = 1 - a * Real.exp (x n)) (ha : a ≤ 1) :
  ∀ n, x n ≥ 0 := 
  sorry

end non_neg_sequence_l410_410173


namespace smallest_n_satisfies_conditions_l410_410669

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410669


namespace triangle_area_is_194_97_l410_410132

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_is_194_97 :
  heron_area 30 28 14 = 194.97 := sorry

end triangle_area_is_194_97_l410_410132


namespace christen_peeled_potatoes_l410_410960

open Nat

theorem christen_peeled_potatoes :
  ∀ (total_potatoes homer_rate homer_time christen_rate : ℕ) (combined_rate : ℕ),
    total_potatoes = 60 →
    homer_rate = 4 →
    homer_time = 6 →
    christen_rate = 6 →
    combined_rate = homer_rate + christen_rate →
    Nat.ceil ((total_potatoes - (homer_rate * homer_time)) / combined_rate * christen_rate) = 21 :=
by
  intros total_potatoes homer_rate homer_time christen_rate combined_rate
  intros htp hr ht cr cr_def
  rw [htp, hr, ht, cr, cr_def]
  sorry

end christen_peeled_potatoes_l410_410960


namespace difference_proof_l410_410264

noncomputable def solve_difference (sum : ℕ) (divisible : ℕ → Prop) (erase : ℕ → ℕ) : ℕ :=
let a := sum / 101 in
100 * a - a

theorem difference_proof :
  ∀ (sum : ℕ) (divisible : ℕ → Prop) (erase : ℕ → ℕ),
    sum = 23540 →
    (∃ n, divisible n ∧ divisible 16 ∧ erase n = sum - n) →
    solve_difference sum divisible erase = 23067 :=
by
  intros sum divisible erase h_sum h_conditions
  sorry

end difference_proof_l410_410264


namespace sum_of_possible_digit_counts_in_base_2_l410_410322

theorem sum_of_possible_digit_counts_in_base_2 :
  (∑ d in ({13, 14, 15} : Finset ℕ), d) = 42 :=
by 
  sorry

end sum_of_possible_digit_counts_in_base_2_l410_410322


namespace max_value_of_y_l410_410084

theorem max_value_of_y (a b c : ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) :
  (∃ y_max : ℝ, ∀ a b c, y = (a * b + 2 * b * c) / (a^2 + b^2 + c^2) → y ≤ y_max) ∧ 
  (∀ ε > 0, ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ (a' * b' + 2 * b' * c') / (a'^2 + b'^2 + c'^2) > (sqrt 5 / 2) - ε) :=
sorry

end max_value_of_y_l410_410084


namespace sum_possible_values_base2_l410_410315

theorem sum_possible_values_base2 (d_8 : ℕ) (h1 : d_8 >= 8^4) (h2 : d_8 < 8^5) :
  d_8.bits ≠ 0 → (d_8.bits.card = 13 ∨ d_8.bits.card = 14 ∨ d_8.bits.card = 15) →
  d_8.bits.card = 13 + d_8.bits.card = 14 + d_8.bits.card = 15 := sorry

end sum_possible_values_base2_l410_410315


namespace simplify_expression_l410_410608

variables (a b c : ℝ)

theorem simplify_expression 
  (ha : 0 < a) 
  (hc : 0 < c) 
  (hden : 2 * (a - b^2)^2 + (2 * b * real.sqrt (2 * a))^2 ≠ 0) :
  (real.sqrt 3 * (a - b^2) + real.sqrt 3 * b * real.cbrt (8 * b^3)) / (real.sqrt (2 * (a - b^2)^2 + (2 * b * real.sqrt (2 * a))^2)) 
  * (real.sqrt (2 * a) - real.sqrt (2 * c)) / (real.sqrt (3 / a) - real.sqrt (3 / c)) = 
  -real.sqrt (a * c) :=
sorry

end simplify_expression_l410_410608


namespace insurance_premium_increases_l410_410506

-- Define the insurance premium and conditions
variable (current_premium future_premium : ℝ)
variable (has_accident : Bool)
variable (records_claims considers_risk increases_premium : Bool)

-- State the theorem
theorem insurance_premium_increases 
    (h1 : has_accident = true)
    (h2 : records_claims = true) 
    (h3 : considers_risk = true) 
    (h4 : increases_premium = true) 
    (current_premium < future_premium) 
    : future_premium > current_premium :=
by 
  sorry

end insurance_premium_increases_l410_410506


namespace problem_intersection_l410_410105

open Set

variable {x : ℝ}

def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def C : Set ℝ := {x | (5 / 2) ≤ x ∧ x < 3}

theorem problem_intersection : A ∩ B = C := by
  sorry

end problem_intersection_l410_410105


namespace vasya_can_place_checkers_l410_410205

theorem vasya_can_place_checkers 
(board : ℕ → ℕ → Bool)
(placed_checkers : ∀ i j, board i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50)
: ∃ (new_checkers : ℕ → ℕ → Bool), (∀ i j, new_checkers i j = true → 1 ≤ i ∧ i ≤ 50 ∧ 1 ≤ j ∧ j ≤ 50) ∧
       (∑ i j, if new_checkers i j then 1 else 0 ≤ 99) ∧ 
       (∀ i, (∑ j, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) ∧ 
       (∀ j, (∑ i, if board i j ∨ new_checkers i j then 1 else 0) % 2 = 0) := 
sorry

end vasya_can_place_checkers_l410_410205


namespace min_value_y_l410_410123

theorem min_value_y (x : ℝ) (h : x > 5 / 4) : 
  ∃ y, y = 4*x - 1 + 1 / (4*x - 5) ∧ y ≥ 6 :=
by
  sorry

end min_value_y_l410_410123


namespace nancy_total_spending_l410_410744

theorem nancy_total_spending :
  let crystal_bead_price := 9
  let metal_bead_price := 10
  let nancy_crystal_beads := 1
  let nancy_metal_beads := 2
  nancy_crystal_beads * crystal_bead_price + nancy_metal_beads * metal_bead_price = 29 := by
sorry

end nancy_total_spending_l410_410744


namespace proof_problem_l410_410768

noncomputable def problem_statement : Prop :=
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ 
    x1 + x2 + x3 + x4 = 8 ∧ 
    x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ 
    (x1 + x2) = 2 * 2 ∧ 
    (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 - 4 * 2 * (x1 + x2 + x3 + x4) + 4 * 4) = 4 ∧ 
    (x1 = 1 ∧ x2 = 1 ∧ x3 = 3 ∧ x4 = 3)

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l410_410768


namespace sequence_recurrence_l410_410474

theorem sequence_recurrence (a : ℕ → ℝ) (h₀ : a 1 = 1) (h : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  intro n hn
  exact sorry

end sequence_recurrence_l410_410474


namespace ab_value_l410_410120

theorem ab_value (a b : ℝ) (h₁ : b = (real.sqrt (1 - 2 * a)) + (real.sqrt (2 * a - 1)) + 3) (h₂ : a = 1 / 2) : a^b = 1 / 8 := 
by 
  sorry

end ab_value_l410_410120


namespace product_xyz_l410_410971

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end product_xyz_l410_410971


namespace evaluate_g_neg_1_l410_410095

noncomputable def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

theorem evaluate_g_neg_1 : g (-1) = -14 := 
by
  sorry

end evaluate_g_neg_1_l410_410095


namespace smallest_positive_n_l410_410673

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410673


namespace maximal_cross_sectional_area_of_cut_prism_l410_410763

-- Define the conditions
structure Prism :=
(vertical_edges_parallel_z : Prop)
(square_cross_section : ℝ)
(square_side_length : ℝ)

-- Define the plane equation
def cutting_plane (x y z : ℝ) := 4 * x - 7 * y + 4 * z = 25

-- Prove the maximal cross-sectional area
theorem maximal_cross_sectional_area_of_cut_prism (P : Prism)
  (h1: P.vertical_edges_parallel_z)
  (h2: P.square_cross_section = 10)
  (h3: P.square_side_length = 10):
  ∃ area : ℝ, area = 225 :=
begin
  use 225,
  sorry
end

end maximal_cross_sectional_area_of_cut_prism_l410_410763


namespace valid_starting_lineups_l410_410234

theorem valid_starting_lineups (players : Fin 15) (bob, yogi, danny : Fin 15) 
    (H_distinct : bob ≠ yogi ∧ bob ≠ danny ∧ yogi ≠ danny) :
    -- Lineup constraints
    (∀ lineup : Finset (Fin 15), lineup.card = 5 →
    (bob ∈ lineup → yogi ∉ lineup) ∧
    (yogi ∈ lineup → bob ∉ lineup) ∧
    (bob ∈ lineup → danny ∉ lineup) ∧
    (yogi ∈ lineup → danny ∉ lineup)) →
    -- Expected result
    ∑ lineup, (if lineup.card = 5 ∧ 
                ¬(bob ∈ lineup ∧ yogi ∈ lineup) ∧ 
                (∀ d, d ∈ lineup → (d ≠ danny ∨ (bob ∉ lineup ∧ yogi ∉ lineup))) 
               then 1 else 0) = 1782 :=
begin
  sorry
end

end valid_starting_lineups_l410_410234


namespace division_problem_l410_410854

theorem division_problem :
  ∃ x y, x ∈ { n : ℕ | 100 ≤ n ∧ n < 1000 } ∧ y = 97809 ∧ x * y = 12128316 :=
by
  use 124, 97809
  split
  · -- Prove that x is a 3-digit number
    split
    · exact Nat.le_refl 124 -- 100 ≤ 124
    · exact Nat.lt_of_le_of_lt (by norm_num) (by norm_num) -- 124 < 1000
  · split
    · rfl -- y = 97809
    · rfl -- x * y = 12128316
  sorry

end division_problem_l410_410854


namespace equation_of_line_AB_l410_410423

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

noncomputable def line_equation (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

noncomputable def on_line (P : ℝ × ℝ) : Prop :=
  line_equation P.1 P.2

noncomputable def the_point : ℝ × ℝ :=
  (-1, 0)

theorem equation_of_line_AB :
  (circle_equation 1 1) ∧ (line_equation 1 1) ∧ on_line the_point →
  (2*the_point.1 + the_point.2 + 1 = 0) :=
begin
  sorry
end

end equation_of_line_AB_l410_410423


namespace nonagon_angles_l410_410021

/-- Determine the angles of the nonagon given specified conditions -/
theorem nonagon_angles (a : ℝ) (x : ℝ) 
  (h_angle_eq : ∀ (AIH BCD HGF : ℝ), AIH = x → BCD = x → HGF = x)
  (h_internal_sum : 7 * 180 = 1260)
  (h_tessellation : x + x + x + (360 - x) + (360 - x) + (360 - x) = 1080) :
  True := sorry

end nonagon_angles_l410_410021


namespace max_sin_cos_expr_l410_410892

theorem max_sin_cos_expr (x y z : ℝ) :
  let expr := (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))
  in ∀ (x y z : ℝ), expr ≤ 4.5 :=
sorry

end max_sin_cos_expr_l410_410892


namespace smallest_n_satisfies_conditions_l410_410672

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l410_410672


namespace inequality_a_log_b_sin_theta_l410_410427

theorem inequality_a_log_b_sin_theta 
  (a b : ℝ) 
  (θ : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : 0 < θ ∧ θ < (Float.pi / 2)) : 
  a * Real.log b (Real.sin θ) < b * Real.log a (Real.sin θ) := 
sorry

end inequality_a_log_b_sin_theta_l410_410427


namespace fixed_point_of_f_l410_410249

noncomputable def f (a : Real) (x : Real) := log a (3 * x - 2) + 2

theorem fixed_point_of_f (a : Real) (h₁ : a > 0) (h₂ : a ≠ 1) :
  f a 1 = 2 := by
  sorry

end fixed_point_of_f_l410_410249


namespace shaded_area_is_28_l410_410237

theorem shaded_area_is_28 (A B : ℕ) (h1 : A = 64) (h2 : B = 28) : B = 28 := by
  sorry

end shaded_area_is_28_l410_410237


namespace correct_propositions_count_l410_410073

noncomputable def line := ℝ → ℝ × ℝ × ℝ  -- example representation for lines in 3D space
noncomputable def plane := {n : ℝ × ℝ × ℝ // n ≠ (0, 0, 0)} → ℝ  -- example representation for planes

variables {l m : line} {α β : plane}

-- Auxiliary definitions to express perpendicularity and parallelism
def perp (l : line) (p : plane) : Prop := sorry -- l is perpendicular to plane p
def parallel (p q : plane) : Prop := sorry -- planes p and q are parallel
def contains (p : plane) (l : line) : Prop := sorry -- plane p contains line l
def perp_lines (l m : line) : Prop := sorry -- lines l and m are perpendicular
def parallel_lines (l m : line) : Prop := sorry -- lines l and m are parallel

noncomputable def num_correct_propositions (l m : line) (α β : plane) 
  (hlα : perp l α) (hmβ : contains β m)
  (p₁ : if perp_lines l m then parallel α β else true)
  (p₂ : if parallel α β then perp_lines l m else true)
  (p₃ : if perp α β then parallel_lines l m else true) : ℕ :=
if (if perp_lines l m then parallel α β else true) && 
   (if parallel α β then perp_lines l m else true) && 
   (if perp α β then parallel_lines l m else true)
then 3
else if (if perp_lines l m then parallel α β else true) &&
        (if parallel α β then perp_lines l m else true)
then 2
else if (if perp_lines l m then parallel α β else true) ||
        (if parallel α β then perp_lines l m else true) ||
        (if perp α β then parallel_lines l m else true)
then 1
else 0

theorem correct_propositions_count (l m : line) (α β : plane)
  (hlα : perp l α) (hmβ : contains β m) :
  num_correct_propositions l m α β hlα hmβ
  (if perp_lines l m then parallel α β else true)
  (if parallel α β then perp_lines l m else true)
  (if perp α β then parallel_lines l m else true) = 1 :=
sorry

end correct_propositions_count_l410_410073


namespace range_of_x2_y2_l410_410935

-- Definitions based on conditions
def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def is_symmetric_about_1_0(f : ℝ → ℝ) : Prop :=
∀ x, f (x - 1) = -f (2 - x)

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
∀ x y, f (x^2 - 6x + 21) + f (y^2 - 8y) < 0

-- Main statement to prove
theorem range_of_x2_y2 (f : ℝ → ℝ)
  (h1 : is_increasing f)
  (h2 : is_symmetric_about_1_0 f)
  (h3 : satisfies_inequality f) :
  ∀ x y, x > 3 → 13 < x^2 + y^2 ∧ x^2 + y^2 < 49 :=
sorry

end range_of_x2_y2_l410_410935


namespace charles_drink_milk_l410_410354

/-- Define the conditions: the amounts of each ingredient required for one glass and the total amounts available --/
def milk_per_glass := 6
def syrup_per_glass := 1.5
def cream_per_glass := 0.5
def total_milk := 130
def total_syrup := 60
def total_cream := 25
def glass_size := 8

/-- Define total number of glasses possible from each ingredient --/
def max_glasses_milk : ℝ := total_milk / milk_per_glass
def max_glasses_syrup : ℝ := total_syrup / syrup_per_glass
def max_glasses_cream : ℝ := total_cream / cream_per_glass

/-- Define the maximum number of full glasses Charles can make --/
def max_glasses : ℕ := Real.to_nat (Real.floor (min (min max_glasses_milk max_glasses_syrup) max_glasses_cream))

/-- The total ounces of chocolate milk Charles will drink is the product of the number of full glasses he can make
and the size of each glass --/
def total_milk_consumed := max_glasses * glass_size

/-- Proof that the total ounces of chocolate milk Charles will drink is 168 ounces --/
theorem charles_drink_milk : total_milk_consumed = 168 := sorry

end charles_drink_milk_l410_410354


namespace max_edges_plane_intersection_l410_410666

-- Define the structure of a regular prism with a decagon base
structure RegularPrism :=
  (n : ℕ) -- number of sides of the base polygon
  (regular : ∀ i j, i ≠ j → distance_vertices i j = baseline) -- base polygon is regular
  (right : true) -- prism is right

-- Define the problem statement
theorem max_edges_plane_intersection (P : RegularPrism) (decagon : P.n = 10) : 
  ∃ (S : Plane), maximum_edges_intersected_by_plane P S = 12 := 
sorry

end max_edges_plane_intersection_l410_410666


namespace problem_solution_l410_410468

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x) + cos (ω * x)

theorem problem_solution (ω : ℝ) (hω : ω > 0) 
    (h_mono : ∀ x y: ℝ, -ω ≤ x ∧ x < y ∧ y ≤ ω → f ω x ≤ f ω y)
    (h_symm : ∀ x : ℝ, f ω (2 * ω - x) = f ω x) :
  ω = sqrt π / 2 :=
sorry

end problem_solution_l410_410468


namespace product_of_all_possible_t_l410_410400

-- Define the set of possible integer pairs (a, b) such that ab = 12.
def possible_pairs : Set (Int × Int) :=
  {(a, b) | a * b = 12}

-- Define the set of possible values of t = a + b for those pairs (a, b).
def possible_values_of_t : Set Int :=
  {t | ∃ (a b : Int), (a, b) ∈ possible_pairs ∧ t = a + b}

-- Lean 4 theorem to prove the product of all possible values of t.
theorem product_of_all_possible_t : 
  ∏ t in possible_values_of_t, t = 532224 :=
sorry

end product_of_all_possible_t_l410_410400


namespace cross_correlation_l410_410044

variable {Ω : Type*} -- Sample space
variable {P : MeasureTheory.ProbabilityMeasure Ω} -- Probability measure
variable (U : Ω → ℝ) -- Random variable

noncomputable def X (t : ℝ) := t^2 * U
noncomputable def Y (t : ℝ) := t^3 * U

axiom D_U : MeasureTheory.variance P U = 5

theorem cross_correlation (t1 t2 : ℝ) :
  MeasureTheory.esperance P ((X t1 - MeasureTheory.esperance P (X t1)) * (Y t2 - MeasureTheory.esperance P (Y t2))) = 5 * t1^2 * t2^3 := 
sorry

end cross_correlation_l410_410044


namespace ceil_and_floor_difference_l410_410923

theorem ceil_and_floor_difference (x : ℝ) (ε : ℝ) 
  (h_cond : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) (h_eps : 0 < ε ∧ ε < 1) :
  ⌈x + ε⌉ - (x + ε) = 1 - ε :=
sorry

end ceil_and_floor_difference_l410_410923


namespace sin_two_alpha_sub_pi_eq_24_div_25_l410_410907

noncomputable def pi_div_2 : ℝ := Real.pi / 2

theorem sin_two_alpha_sub_pi_eq_24_div_25
  (α : ℝ) 
  (h1 : pi_div_2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan (α + Real.pi / 4) = -1 / 7) : 
  Real.sin (2 * α - Real.pi) = 24 / 25 := 
sorry

end sin_two_alpha_sub_pi_eq_24_div_25_l410_410907


namespace trajectory_of_P_sum_of_λ_μ_l410_410535

-- Definitions and conditions for part (1)
variable (F : ℝ × ℝ) (l : ℝ → Prop) (P Q : ℝ × ℝ)
variable (OP OF FP FQ : ℝ × ℝ)
def vectors_equation_1 (P Q F : ℝ × ℝ) : Prop := 
  let OP := (P.1, P.2)
  let OF := (F.1, F.2)
  let FP := (P.1 - F.1, P.2 - F.2)
  let FQ := (Q.1, Q.2 - F.2)
  ⟪ OP, OF ⟫ = ⟪ FP, FQ ⟫

def trajectory_equation : Prop := 
  ∃ (x y : ℝ), P = (x, y) ∧ ((Q = (-2, y)) ∧ (vectors_equation_1 (x, y) (-2, y) (2, 0))) ∧ (y^2 = 8 * x)

-- Theorem to prove trajectory equation
theorem trajectory_of_P {F : ℝ × ℝ} {l : ℝ → Prop} {P Q : ℝ × ℝ} :
  F = (2, 0) ∧ 
  (∀ x, l x ↔ x = -2) ∧ 
  (P = (x, y) ∧ Q = (-2, y)) ∧ 
  (vectors_equation_1 P Q F) 
  → trajectory_equation := 
sorry

-- Definitions and conditions for part (2)
variable (A B M : ℝ × ℝ)
variable (λ μ : ℝ)

def line_intersects_C_and_l (F M A B : ℝ × ℝ) : Prop :=
  let slope_eq := (B.1 - A.1) / (B.2 - A.2)
  let inter_l := (M.1 = -2)
  let conditions := (A, B) ∈ (C : set (ℝ × ℝ))
  slope_eq ≠ 0 ∧ inter_l ∧ conditions ∧ 
  (A = (x_1, y_1) ∧ B = (x_2, y_2)) →
    λ = -1 - (4 / (slope_eq * y_1)) ∧
    μ = -1 - (4 / (slope_eq * y_2)) →

-- Theorem to prove λ + μ = 0
theorem sum_of_λ_μ {F M A B : ℝ × ℝ} :
  (F = (2, 0)) ∧ 
  (M = (-2, -(4 / t))) ∧ 
  (λ = -1 - (4 / (t * y_1)) ∧ 
  μ = -1 - (4 / (t * y_2))) ∧ 
  (y_1, y_2) solutions of y² - 8ty - 16 = 0 
  → λ + μ = 0 := 
sorry

end trajectory_of_P_sum_of_λ_μ_l410_410535


namespace problem_max_value_problem_min_positive_period_problem_monotonic_increase_interval_l410_410946

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * sin x + sqrt 3 * (2 * cos x ^ 2 - 1)

theorem problem_max_value : ∀ x : ℝ, f(x) ≤ 2 := sorry

noncomputable def f2 (x : ℝ) : ℝ := f(2 * x)

theorem problem_min_positive_period : ∃ T > 0, ∀ x : ℝ, f2(x + T) = f2(x) := 
  sorry

theorem problem_monotonic_increase_interval : ∀ k : ℤ, 
  ∃ l u : ℝ, 
  l = - (5 * π / 24) + k * (π / 2) ∧ 
  u = π / 24 + k * (π / 2) ∧ 
  ∀ x : ℝ, l ≤ x ∧ x ≤ u → f2(x) ≤ f2(x + ε) for 0 < ε := 
  sorry

end problem_max_value_problem_min_positive_period_problem_monotonic_increase_interval_l410_410946


namespace time_to_cross_trains_l410_410274

noncomputable def length_first_train : ℝ := 200 -- Length of the first train in meters
noncomputable def length_second_train : ℝ := 300 -- Length of the second train in meters
noncomputable def speed_first_train : ℝ := 70 -- Speed of the first train in kmph
noncomputable def speed_second_train : ℝ := 50 -- Speed of the second train in kmph

theorem time_to_cross_trains:
  let relative_speed_meters_per_second := (speed_first_train + speed_second_train) * 1000 / 3600 in
  let combined_length_meters := length_first_train + length_second_train in
  let time_to_cross := combined_length_meters / relative_speed_meters_per_second in
  time_to_cross ≈ 15 :=
by sorry

end time_to_cross_trains_l410_410274


namespace cos2theta_eq_zero_l410_410520

def a (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 1)
def b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)

theorem cos2theta_eq_zero (θ : ℝ) (h_collinear : (2 * Real.cos θ) * (Real.cos θ) - 1 * 1 = 0) : Real.cos (2 * θ) = 0 :=
by sorry

end cos2theta_eq_zero_l410_410520


namespace projection_is_orthocenter_l410_410519

-- Define a structure for a point in 3D space.
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

-- Define mutually perpendicular edges condition.
def mutually_perpendicular {α : Type} [Field α] (A B C D : Point α) :=
(A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) + (A.z - D.z) * (B.z - D.z) = 0 ∧
(A.x - D.x) * (C.x - D.x) + (A.y - D.y) * (C.y - D.y) + (A.z - D.z) * (C.z - D.z) = 0 ∧
(B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) + (B.z - D.z) * (C.z - D.z) = 0

-- The main theorem statement.
theorem projection_is_orthocenter {α : Type} [Field α]
    (A B C D : Point α) (h : mutually_perpendicular A B C D) :
    ∃ O : Point α, -- there exists a point O (the orthocenter)
    (O.x * (B.y - A.y) + O.y * (A.y - B.y) + O.z * (A.y - B.y)) = 0 ∧
    (O.x * (C.y - B.y) + O.y * (B.y - C.y) + O.z * (B.y - C.y)) = 0 ∧
    (O.x * (A.y - C.y) + O.y * (C.y - A.y) + O.z * (C.y - A.y)) = 0 := 
sorry

end projection_is_orthocenter_l410_410519


namespace student_incorrect_answer_l410_410141

theorem student_incorrect_answer (D I : ℕ) (h1 : D / 63 = I) (h2 : D / 36 = 42) : I = 24 := by
  sorry

end student_incorrect_answer_l410_410141


namespace part1_part2_l410_410467

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + Real.sqrt 3 * Real.tan θ / 8 - 3 / 2

-- Assumptions
variables (x : ℝ) (θ : ℝ)
hypothesis h1 : 0 ≤ x ∧ x ≤ π / 2
hypothesis h2 : 0 ≤ θ ∧ θ ≤ π / 3

-- Part 1: Maximum value when θ = π / 3
theorem part1 : (θ = π / 3) → (∃ x, f x θ = 15 / 8 ∧ x = 0) :=
by sorry

-- Part 2: Existence of θ such that maximum value of f(x) is -1/8
theorem part2 : (∃ θ, ∃ x, f x θ = - 1 / 8 ∧ θ = π / 6) :=
by sorry

end part1_part2_l410_410467


namespace find_number_l410_410994

variable (a n : ℝ)

theorem find_number (h1: 2 * a = 3 * n) (h2: a * n ≠ 0) (h3: (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 :=
sorry

end find_number_l410_410994


namespace circles_intersect_and_distance_l410_410245

noncomputable def circle_a (x y : ℝ) := x^2 + y^2 - 2 * x - 2 * y - 7 = 0
noncomputable def circle_b (x y : ℝ) := x^2 + y^2 + 2 * x + 2 * y - 2 = 0

theorem circles_intersect_and_distance :
  (∃ (x y : ℝ), circle_a x y ∧ circle_b x y) ∧
  (∀ (x y : ℝ), circle_a x y ∧ circle_b x y → 4 * x + 4 * y + 5 = 0) ∧
  (∃ (d : ℝ), d = sqrt 238 / 4) :=
by
  sorry

end circles_intersect_and_distance_l410_410245


namespace equidistant_points_from_midpoint_l410_410442

noncomputable def midpoint (C D : Point) : Point := sorry

theorem equidistant_points_from_midpoint
  (A B C D K L X Y : Point)
  (ωA ωB : Sphere)
  (M : Point)
  (h1 : tangent ωA (plane B C D))
  (h2 : tangent ωA (plane A C D))
  (h3 : tangent ωA (plane A B C))
  (h4 : tangent ωB (plane A C D))
  (h5 : tangent ωB (plane B C D))
  (h6 : tangent ωB (plane A B C))
  (hK : tangent_point ωA (plane A C D) = K)
  (hL : tangent_point ωB (plane B C D) = L)
  (hX : tangent_extension A K = X)
  (hY : tangent_extension B L = Y)
  (h_angle1 : ∠ C K D = ∠ C X D + ∠ C B D)
  (h_angle2 : ∠ C L D = ∠ C Y D + ∠ C A D) :
  dist X (midpoint C D) = dist Y (midpoint C D) :=
sorry

end equidistant_points_from_midpoint_l410_410442


namespace max_value_sin_cos_expression_l410_410887

theorem max_value_sin_cos_expression (x y z : ℝ) 
  (h1 : sin x^2 + cos x^2 = 1)
  (h2 : sin (2 * y)^2 + cos (2 * y)^2 = 1)
  (h3 : sin (3 * z)^2 + cos (3 * z)^2 = 1) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_sin_cos_expression_l410_410887


namespace infinite_product_eval_l410_410038

theorem infinite_product_eval :
    (\prod_{n : ℕ in (Set.Ioi 0), 3 ^ (n / (3 ^ n))}) = 3 ^ (3 / 4) :=
sorry

end infinite_product_eval_l410_410038


namespace linear_relationship_selling_price_maximize_profit_l410_410749

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end linear_relationship_selling_price_maximize_profit_l410_410749


namespace exam_statements_correct_l410_410287

theorem exam_statements_correct:
  let data := [2, 2, 3, 5, 6, 7, 7, 8, 10, 11]
  let is_B_correct := ∀ (x : List ℝ) (a b : ℝ) (sx : ℝ), stddev x = sx → stddev (x.map (λ xi, a * xi + b)) = abs a * sx
  let is_C_correct := ∀ {Ω : Type} {p : Ω → ℝ} (A B C : set Ω), pairwise (mutually_disjoint_on p) [A, B, C] → probability (A ∪ B ∪ C) = probability A + probability B + probability C
  let is_D_correct := ∀ {Ω : Type} {p : Ω → ℝ} (A B : set Ω), p (A ∩ B) = p A * p B → independent_events p A B
  is_B_correct ∧ is_C_correct ∧ is_D_correct :=
by
  sorry

end exam_statements_correct_l410_410287


namespace max_value_sin_cos_expression_l410_410878

-- Define the variables and the expressions

def max_trig_expression (x y z : ℝ) : ℝ :=
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))

-- State the theorem to find the maximum value of the given expression.

theorem max_value_sin_cos_expression : ∀ x y z : ℝ, max_trig_expression x y z ≤ 4.5 :=
by {
  sorry -- This is where the proof would go
}

end max_value_sin_cos_expression_l410_410878


namespace round_robin_tournament_l410_410586

-- Define the conditions
def condition1 (A : List ℕ) (n : ℕ) : Prop :=
  ∀ k, k < n → (A.subsets k).forall (λ s, s.sum ≥ k * (k - 1))

def condition2 (A : List ℕ) (n : ℕ) : Prop :=
  A.sum = n * (n - 1)

-- Prove the main theorem based on the conditions
theorem round_robin_tournament (A : List ℕ) (n : ℕ) 
  (h1 : condition1 A n) (h2 : condition2 A n) : 
  ∃ B : List ℕ, B = A := sorry

end round_robin_tournament_l410_410586


namespace factorial_quotient_l410_410817

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410817


namespace factorial_div_combination_l410_410824

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410824


namespace distance_from_point_on_C1_to_C2_l410_410154

-- Define the polar to Cartesian conversion of C1
def curve_C1_cartesian_equation (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

-- Define the general form of C2
def curve_C2_general_equation (x y : ℝ) : Prop :=
  √3 * x - y + √3 = 0

-- Define the range of distance from any point P on C1 to C2
def distance_range : set ℝ :=
  set.Icc 0 ((√3 + 1) / 2)

theorem distance_from_point_on_C1_to_C2 (x y : ℝ) :
  curve_C1_cartesian_equation x y →
  ∃ d : ℝ, curve_C2_general_equation x y ∧ d ∈ distance_range :=
sorry

end distance_from_point_on_C1_to_C2_l410_410154


namespace exists_coeff_less_than_neg_one_l410_410711

theorem exists_coeff_less_than_neg_one 
  (P : Polynomial ℤ)
  (h1 : P.eval 1 = 0)
  (h2 : P.eval 2 = 0) :
  ∃ i, P.coeff i < -1 := sorry

end exists_coeff_less_than_neg_one_l410_410711


namespace sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410320

theorem sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer :
  (sum (λ d, d) (filter (λ d, 5 = nat.digits_length 8 d) (range 2^13 2^16))) = 58 := sorry

end sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410320


namespace sibling_age_difference_l410_410266

theorem sibling_age_difference (Y : ℝ) (Y_eq : Y = 25.75) (avg_age_eq : (Y + (Y + 3) + (Y + 6) + (Y + x)) / 4 = 30) : (Y + 6) - Y = 6 :=
by
  sorry

end sibling_age_difference_l410_410266


namespace triangle_inequality_l410_410593

variables {A B C P : Type}
variables [metric_space P]
variables [metric_space A]
variables [metric_space B]
variables [metric_space C]

-- Definitions for distances
noncomputable def distance_PA (A P : P) : ℝ := dist A P
noncomputable def distance_PB (B P : P) : ℝ := dist B P
noncomputable def distance_PC (C P : P) : ℝ := dist C P 

noncomputable def AB_distance (A B : P) : ℝ := dist A B
noncomputable def BC_distance (B C : P) : ℝ := dist B C
noncomputable def CA_distance (C A : P) : ℝ := dist C A

-- Conditions
def k (P A B C : P) := min (distance_PA A P) (min (distance_PB B P) (distance_PC C P))

-- Statement of the problem
theorem triangle_inequality (P A B C : P) 
  (inside_triangle : True)  -- Assumed condition that P is inside the triangle ABC
  : k P A B C + distance_PA A P + distance_PB B P + distance_PC C P ≤ AB_distance A B + BC_distance B C + CA_distance C A :=
sorry

end triangle_inequality_l410_410593


namespace sequence_sum_l410_410795

theorem sequence_sum : 
  (∑ n in finset.range 2000, if n % 6 ∈ [0, 1, 2] then -((n + 1) : ℤ) else ((n + 1) : ℤ)) = -334 := 
by 
  sorry

end sequence_sum_l410_410795


namespace part1_l410_410469

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) ↔ a ≤ -2 := by
  sorry

end part1_l410_410469


namespace triangle_side_b_l410_410522

open Real

variable {a b c : ℝ} (A B C : ℝ)

theorem triangle_side_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin B = 6 * cos A * sin C) : b = 3 :=
sorry

end triangle_side_b_l410_410522


namespace solve_x_correct_l410_410043

noncomputable def solve_x_equation (x : ℝ) : Prop :=
  (25^x + 49^x) / (35^x + 40^x) = 5 / 4

theorem solve_x_correct : ∃ x : ℝ, solve_x_equation x ∧ x = - (Real.log 4 / Real.log (5 / 7)) :=
by {
  sorry
}

end solve_x_correct_l410_410043


namespace wheel_of_fortune_probability_l410_410592

theorem wheel_of_fortune_probability :
  let outcomes := ["Bankrupt", "$700$", "$600$", "$5000$", "$800$", "$300$"] in
  let total_spins := 6 ^ 3 in
  let favorable_combinations := [
    ["$700$", "$600$", "$700$"], 
    ["$700$", "$800$", "$500$"]
  ] in
  let ways_per_comb := 3 * 2 * 1 in
  let total_favorable := 6 + 6 in
  (total_favorable : ℚ) / (total_spins : ℚ) = 1 / 18 :=
by sorry

end wheel_of_fortune_probability_l410_410592


namespace students_per_group_l410_410260

-- Definitions for conditions
def number_of_boys : ℕ := 28
def number_of_girls : ℕ := 4
def number_of_groups : ℕ := 8
def total_students : ℕ := number_of_boys + number_of_girls

-- The Theorem we want to prove
theorem students_per_group : total_students / number_of_groups = 4 := by
  sorry

end students_per_group_l410_410260


namespace sufficient_not_necessary_condition_l410_410214

-- Definitions of propositions
def propA (x : ℝ) : Prop := (x - 1)^2 < 9
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Lean statement of the problem
theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, propA x → propB x a) ∧ (∃ x, ¬ propA x ∧ propB x a) ↔ a < -4 :=
sorry

end sufficient_not_necessary_condition_l410_410214


namespace log_eq_given_conditions_l410_410804

theorem log_eq_given_conditions (n : ℕ) (h : n > 3) : 
  log (n - 3)! / log 10 + log (n - 1)! / log 10 + 3 = 2 * log n! / log 10 ↔ n = 6 := by
  sorry

end log_eq_given_conditions_l410_410804


namespace smallest_positive_n_l410_410677

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l410_410677


namespace factorial_div_eq_l410_410842

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410842


namespace sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410318

theorem sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer :
  (sum (λ d, d) (filter (λ d, 5 = nat.digits_length 8 d) (range 2^13 2^16))) = 58 := sorry

end sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410318


namespace problem1_problem2_l410_410941

-- Statement for Problem 1
theorem problem1 :
  (∃ (x y : ℝ), x + 2 * y - 5 = 0 ∧ 3 * x - y - 1 = 0) ∧
  (∃ (l : Line), parallel l (Line.mk 5 (-1) 100) ∧ passes_through l (1, 2))
  → ∃ (l : Line), l.equation = 5 * x - y - 3 = 0 :=
sorry

-- Statement for Problem 2
theorem problem2 :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) ∧
  (exists_equal_intercepts {l : Line | passes_through l (3, 2)})
  → (equation (2 * x - 3 * y = 0) ∨ equation (x + y - 5 = 0)) :=
sorry

end problem1_problem2_l410_410941


namespace sufficient_but_not_necessary_l410_410567

-- Definitions for lines a and b, and planes alpha and beta
variables {a b : Type} {α β : Type}

-- predicate for line a being in plane α
def line_in_plane (a : Type) (α : Type) : Prop := sorry

-- predicate for line b being perpendicular to plane β
def line_perpendicular_plane (b : Type) (β : Type) : Prop := sorry

-- predicate for plane α being parallel to plane β
def plane_parallel_plane (α : Type) (β : Type) : Prop := sorry

-- predicate for line a being perpendicular to line b
def line_perpendicular_line (a : Type) (b : Type) : Prop := sorry

-- Proof of the statement: The condition of line a being in plane α, line b being perpendicular to plane β,
-- and plane α being parallel to plane β is sufficient but not necessary for line a being perpendicular to line b.
theorem sufficient_but_not_necessary
  (a b : Type) (α β : Type)
  (h1 : line_in_plane a α)
  (h2 : line_perpendicular_plane b β)
  (h3 : plane_parallel_plane α β) :
  line_perpendicular_line a b :=
sorry

end sufficient_but_not_necessary_l410_410567


namespace paint_brush_ratio_l410_410758

theorem paint_brush_ratio 
  (s w : ℝ) 
  (h1 : s > 0) 
  (h2 : w > 0) 
  (h3 : (1 / 2) * w ^ 2 + (1 / 2) * (s - w) ^ 2 = (s ^ 2) / 3) 
  : s / w = 3 + Real.sqrt 3 :=
sorry

end paint_brush_ratio_l410_410758


namespace find_smallest_n_l410_410695

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410695


namespace _l410_410539

-- Definitions for the conditions
variables {T U V W : Type}
variables (angle_UTW angle_TWU angle_UVW angle_TVW angle_TWV angle_UVT : ℝ)

-- Given conditions
def TUV_is_straight_line : Prop := angle_UTW + angle_TWV + angle_UVT = 180
def angle_UTW_is_60 : Prop := angle_UTW = 60
def angle_TWU_is_75 : Prop := angle_TWU = 75
def angle_UVW_is_30 : Prop := angle_UVW = 30

-- The theorem to prove
lemma measure_of_angle_TVW
  (h1 : TUV_is_straight_line)
  (h2 : angle_UTW_is_60)
  (h3 : angle_TWU_is_75)
  (h4 : angle_UVW_is_30) :
  angle_TVW = 15 :=
sorry

end _l410_410539


namespace smallest_n_l410_410682

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l410_410682


namespace cos_90_eq_zero_l410_410001

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410001


namespace sum_largest_odd_divisors_l410_410568

def largest_odd_divisor (n : ℕ) : ℕ :=
if h : n > 0 then Nat.find_greatest (λ m, m ≤ n ∧ Nat.Odd m) n else 0

theorem sum_largest_odd_divisors (n : ℕ) : 
  (∑ k in Finset.range(2^n + 1), largest_odd_divisor (k + 1)) = (4^n + 5) / 3 := 
sorry

end sum_largest_odd_divisors_l410_410568


namespace time_to_fill_is_correct_l410_410720

-- Definitions of rates
variable (R_1 : ℚ) (R_2 : ℚ)

-- Conditions given in the problem
def rate1 := (1 : ℚ) / 8
def rate2 := (1 : ℚ) / 12

-- The resultant rate when both pipes work together
def combined_rate := rate1 + rate2

-- Calculate the time taken to fill the tank
def time_to_fill_tank := 1 / combined_rate

theorem time_to_fill_is_correct (h1 : R_1 = rate1) (h2 : R_2 = rate2) :
  time_to_fill_tank = 24 / 5 := by
  sorry

end time_to_fill_is_correct_l410_410720


namespace factorial_quotient_l410_410819

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410819


namespace find_f3_l410_410041

theorem find_f3 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : f 3 = -25.5 :=
sorry

end find_f3_l410_410041


namespace dot_product_range_l410_410179

variables (a b c : ℝ^3)
variables (θ φ : ℝ)

def norm (v : ℝ^3) := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (x y : ℝ^3) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

theorem dot_product_range (h1 : norm a = 3)
                         (h2 : norm b = 4)
                         (h3 : norm c = 5) :
  -32 ≤ dot_product a b + dot_product b c ∧ dot_product a b + dot_product b c ≤ 32 :=
sorry

end dot_product_range_l410_410179


namespace product_of_factors_l410_410405

theorem product_of_factors :
  (∏ (t : ℤ) in { t | ∃ a b : ℤ, a * b = 12 ∧ t = a + b }, t) = -530144 :=
by sorry

end product_of_factors_l410_410405


namespace smallest_n_satisfies_conditions_l410_410697

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l410_410697


namespace roots_relationship_l410_410600

variable {a b c : ℝ} (h : a ≠ 0)

theorem roots_relationship (x y : ℝ) :
  (x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (y = (-b + Real.sqrt (b^2 - 4*a*c)) / 2 ∨ y = (-b - Real.sqrt (b^2 - 4*a*c)) / 2) →
  (x = y / a) :=
by
  sorry

end roots_relationship_l410_410600


namespace product_of_constants_l410_410393

theorem product_of_constants :
  ∃ (a b : ℤ), ab = 12 ∧
  let t := a + b in 
  (∏ (p : (Σ a b : ℤ, a * b = 12)), p.2.1 + p.2.2) = -531441 :=
by
  sorry

end product_of_constants_l410_410393


namespace matrix_condition_implies_odd_l410_410853

theorem matrix_condition_implies_odd (n : ℕ) :
  (∃ (M : matrix (fin n) (fin n) ℕ), 
    (∀ i, finset.card (finset.filter (λ x, ∃ j, M i j = x) (finset.range n + 1)) = n) ∧
    (∃ (row_sums : fin n → ℕ), (finset.card (finset.image (λ i, row_sums i % n) (finset.univ)) = n)) ∧
    (∃ (col_sums : fin n → ℕ), (finset.card (finset.image (λ j, col_sums j % n) (finset.univ)) = n))
  ) → odd n :=
by
  sorry

end matrix_condition_implies_odd_l410_410853


namespace equation_of_line_AB_l410_410424

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

noncomputable def line_equation (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

noncomputable def on_line (P : ℝ × ℝ) : Prop :=
  line_equation P.1 P.2

noncomputable def the_point : ℝ × ℝ :=
  (-1, 0)

theorem equation_of_line_AB :
  (circle_equation 1 1) ∧ (line_equation 1 1) ∧ on_line the_point →
  (2*the_point.1 + the_point.2 + 1 = 0) :=
begin
  sorry
end

end equation_of_line_AB_l410_410424


namespace area_of_triangle_AOB_is_correct_l410_410090

noncomputable def area_triangle_AOB : ℝ :=
let F2 := (0, real.sqrt 3) in
let line := λ x, -2 * x + real.sqrt 3 in
let ellipse (x y : ℝ) := y^2 / 4 + x^2 = 1 in
let distance_from_origin_to_line := real.sqrt 15 / 5 in
let AB_length := 5 / 2 in
(1 / 2) * AB_length * distance_from_origin_to_line

theorem area_of_triangle_AOB_is_correct :
  area_triangle_AOB = real.sqrt 15 / 4 := sorry

end area_of_triangle_AOB_is_correct_l410_410090


namespace tan_alpha_terminal_side_set_s_angles_terminal_side_l410_410456

theorem tan_alpha_terminal_side {α : ℝ} (h_terminal: ∃ (m : ℝ) (h : m ≠ 0), (m, -real.sqrt 3 * m) ∈ ({p : ℝ × ℝ | p.2 = -real.sqrt 3 * p.1} : set (ℝ × ℝ))) :
  real.tan α = -real.sqrt 3 :=
sorry

theorem set_s_angles_terminal_side {α : ℝ} :
  (∃ (m : ℝ) (h : m ≠ 0), (m, -real.sqrt 3 * m) ∈ ({p : ℝ × ℝ | p.2 = -real.sqrt 3 * p.1} : set (ℝ × ℝ))) →
  {β : ℝ | ∃ (k : ℤ), β = k * real.pi + 2 * real.pi / 3} = {β : ℝ | ∃ (k : ℤ), β = 2 * k * real.pi + 2 * real.pi / 3} ∪ {β : ℝ | ∃ (k : ℤ), β = 2 * k * real.pi + 5 * real.pi / 3} :=
sorry

end tan_alpha_terminal_side_set_s_angles_terminal_side_l410_410456


namespace smallest_n_for_2019_l410_410546

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
  Nat.recOn h 1 (λ n val, val + Int.floor (Real.sqrt (val)))

-- Define the main proof statement
theorem smallest_n_for_2019 : ∃ n : ℕ, seq n ≥ 2019 ∧ ∀ m < n, seq m < 2019 :=
  sorry

end smallest_n_for_2019_l410_410546


namespace probability_in_picture_approx_l410_410599

/-- Given Rachel and Robert's running times and the photographer's timing,
    the probability that both Rachel and Robert are in the picture at any given time
    within the specified window is approximately 0.486666... --/
theorem probability_in_picture_approx :
  ∀ (rachel_lap_time robert_lap_time start_time end_time pic_start_fraction pic_length_fraction : ℕ), 
  rachel_lap_time = 100 →
  robert_lap_time = 90 →
  start_time = 720 →
  end_time = 780 →
  pic_start_fraction = 1/12 →
  pic_length_fraction = 1/3 →
  |((end_time - start_time) -
    ((min (start_time + (pic_start_fraction + pic_length_fraction) * rachel_lap_time)
         (start_time + (pic_start_fraction + pic_length_fraction) * robert_lap_time))
    - 
     (max (start_time + pic_start_fraction * rachel_lap_time)
          (start_time + pic_start_fraction * robert_lap_time)))) / (end_time - start_time)| 
  ≈ 0.486666... :=
begin
  sorry
end

end probability_in_picture_approx_l410_410599


namespace greenville_height_of_boxes_l410_410704

theorem greenville_height_of_boxes:
  ∃ h : ℝ, 
    (20 * 20 * h) * (2160000 / (20 * 20 * h)) * 0.40 = 180 ∧ 
    400 * h = 2160000 / (2160000 / (20 * 20 * h)) ∧
    400 * h = 5400 ∧
    h = 12 :=
    sorry

end greenville_height_of_boxes_l410_410704


namespace line_AB_equation_l410_410417

theorem line_AB_equation : 
  ∀ (M P : Point) (l : Line), 
    Circle M (x^2 + y^2 - 2*x - 2*y - 2 = 0) ∧ 
    Line l (2*x + y + 2 = 0) ∧ 
    P ∈ l ∧
    (∃ A B : Point, Tangent A P ∧ Tangent B P ∧ 
                    A ∈ Circle M ∧ B ∈ Circle M ∧
                    (∃ AB : Line, Minimize |PM| * |AB|)) → 
  is_equation_of_line AB (2*x + y + 1 = 0) := 
sorry

end line_AB_equation_l410_410417


namespace nonneg_sol_count_l410_410962

theorem nonneg_sol_count (x : ℝ) : {x : ℝ | x^2 + 4 * x = 0 ∧ x ≥ 0}.to_finset.card = 1 := 
sorry

end nonneg_sol_count_l410_410962


namespace calculate_chord_length_l410_410236

noncomputable def chord_length_of_tangent (r1 r2 : ℝ) (c : ℝ) : Prop :=
  r1^2 - r2^2 = 18 ∧ (c / 2)^2 = 18

theorem calculate_chord_length (r1 r2 : ℝ) (h : chord_length_of_tangent r1 r2 (6 * Real.sqrt 2)) :
  (6 * Real.sqrt 2) = 6 * Real.sqrt 2 :=
by
  sorry

end calculate_chord_length_l410_410236


namespace razorback_shop_tshirt_price_l410_410233

theorem razorback_shop_tshirt_price :
  ∀ (T J : ℕ), J = 210 ∧ T = J + 30 → T = 240 :=
by
  intros T J h
  cases h with hJ hT
  rw [hJ] at hT
  exact hT

end razorback_shop_tshirt_price_l410_410233


namespace track_circumference_is_180_l410_410789

noncomputable def track_circumference : ℕ :=
  let brenda_first_meeting_dist := 120
  let sally_second_meeting_dist := 180
  let brenda_speed_factor : ℕ := 2
  -- circumference of the track
  let circumference := 3 * brenda_first_meeting_dist / brenda_speed_factor
  circumference

theorem track_circumference_is_180 :
  track_circumference = 180 :=
by 
  sorry

end track_circumference_is_180_l410_410789


namespace number_of_ways_to_distribute_balls_l410_410491

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l410_410491


namespace sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410317

theorem sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer :
  (sum (λ d, d) (filter (λ d, 5 = nat.digits_length 8 d) (range 2^13 2^16))) = 58 := sorry

end sum_of_possible_digit_lengths_in_base2_for_base8_five_digit_integer_l410_410317


namespace not_quasi_prime_1000_l410_410362

-- Definition of quasi-prime sequence
def is_qp (q : ℕ → Prop) : Prop :=
  q 1 = 2 ∧ ∀ n ≥ 2, (∀ m ≤ n - 1, q n > q m) ∧ (∀ i j, (1 ≤ i ∧ i ≤ n - 1 ∧ 1 ≤ j ∧ j ≤ n - 1) → (q n ≠ q i * q j))

-- Hypothesis: 1000 is a quasi-prime
noncomputable def q (n : ℕ) : ℕ := sorry

-- The proposition to prove
theorem not_quasi_prime_1000 : ¬(is_qp q) → q 1000 = 1000 := sorry

end not_quasi_prime_1000_l410_410362


namespace find_n_l410_410167

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = 6 * a (n + 1) - a n

noncomputable def a : ℕ → ℤ
| 0     := 0  -- Dummy value for 0, not used
| 1     := 1
| 2     := 7
| (n+3) := 6 * a (n+2) - a (n+1)

theorem find_n (h : recurrence_relation a) :
  ∀ n : ℕ, (∃ m : ℤ, a n = 2 * m^2 - 1) ↔ n = 1 ∨ n = 2 :=
by sorry

#check find_n

end find_n_l410_410167


namespace projection_correct_l410_410641

namespace ProjectionProblem

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (1, 2)

def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_sq := a.1^2 + a.2^2
  (dot_product / magnitude_sq * a.1, dot_product / magnitude_sq * a.2)

theorem projection_correct : projection vector_a vector_b = (-1 / 2, 1 / 2) :=
sorry

end ProjectionProblem

end projection_correct_l410_410641


namespace cost_per_tissue_l410_410348

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue_l410_410348


namespace combination_formula_l410_410812

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410812


namespace find_omega_l410_410947

/-- Given a function f(x) = A * sin^2(ωx + π/8) with A > 0 and ω > 0. 
The graph of the function is symmetric w.r.t the point (π/2, 1) 
and the minimum positive period T satisfies π/2 < T < 3π/2.
Prove that ω = 5/4. -/
theorem find_omega (A ω T: ℝ) (hA_pos : 0 < A) (hω_pos : 0 < ω) 
  (hsymmetric : ∀ x, f (π/2 - x) = 2 - f (π/2 + x))
  (hT_period : π/2 < T ∧ T < 3π/2) : ω = 5/4 := sorry

end find_omega_l410_410947


namespace marathon_time_l410_410333

theorem marathon_time (distance_first_10 : ℕ) (time_first_10 : ℕ) (total_distance : ℕ) (pace_reduction : ℚ) : 
  total_distance = 26 → 
  distance_first_10 = 10 → 
  time_first_10 = 1 →
  pace_reduction = 0.8 →
  ∃ total_time : ℚ, total_time = 3 :=
by 
  intros h_total_distance h_distance_first_10 h_time_first_10 h_pace_reduction
  let pace_first_10 := distance_first_10 / time_first_10
  let pace_reduced := pace_first_10 * pace_reduction
  let remaining_distance := total_distance - distance_first_10
  let remaining_time := remaining_distance / pace_reduced
  let total_time := time_first_10 + remaining_time
  use total_time
  rw [h_total_distance, h_distance_first_10, h_time_first_10, h_pace_reduction]
  norm_num
  sorry

end marathon_time_l410_410333


namespace combination_formula_l410_410811

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410811


namespace rectangle_proof_l410_410621

noncomputable def rectangle_area_proof : Prop :=
  ∀ (x : ℝ), x > 0 → let E := (0, 0)
                       let F := (0, 6)
                       let G := (x, 6)
                       let H := (x, 0)
                       in (6 * x = 48 ∧ ∀ (slope : ℝ), slope = (G.2 - E.2) / (G.1 - E.1) → slope ≠ 1 ) → x = 8 ∧ (∀ θ : ℝ, θ = real.arctan (3 / 4) → θ ≠ real.pi / 4)

theorem rectangle_proof : rectangle_area_proof :=
by sorry

end rectangle_proof_l410_410621


namespace isosceles_right_triangle_leg_length_l410_410636

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l410_410636


namespace maximize_Z_of_arg_l410_410085

noncomputable def maximize_Z (z : ℂ) : Prop :=
  ∃ z_max : ℂ, z_max = z ∧ (∀ w : ℂ, arg (w + 3) = real.pi * (3 / 4) →
    (|w + 6| + |w - 3i|) ≥ (|z + 6| + |z - 3i|))

theorem maximize_Z_of_arg (z : ℂ) (h : arg (z + 3) = real.pi * (3 / 4)) : maximize_Z (-4 + i) :=
  sorry

end maximize_Z_of_arg_l410_410085


namespace candidate_votes_needed_to_win_l410_410529

theorem candidate_votes_needed_to_win (total_votes : ℕ) (geoff_percent : ℝ) (additional_votes_needed : ℕ) :
    total_votes = 6000 →
    geoff_percent = 0.5 →
    additional_votes_needed = 3000 →
    let geoff_votes := total_votes * (geoff_percent / 100) in
    let votes_needed_to_win := geoff_votes + additional_votes_needed in
    let percent_votes_needed := (votes_needed_to_win / total_votes) * 100 in
    percent_votes_needed = 50.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let geoff_votes := (6000 : ℝ) * (0.5 / 100)
  let votes_needed_to_win := geoff_votes + 3000
  let percent_votes_needed := (votes_needed_to_win / 6000) * 100
  have : percent_votes_needed = 50.5 :=
    calc
      percent_votes_needed
          = ((30 + 3000) / 6000) * 100 : by
            field_simp [geoff_votes, votes_needed_to_win]
          ... = 50.5 : by norm_num
  exact this

end candidate_votes_needed_to_win_l410_410529


namespace product_of_constants_l410_410395

theorem product_of_constants (t : ℤ) (a b : ℤ) (h1 : a * b = 12)
  (h2 : t = a + b) :
  ∃ p : ℤ, (∀ t, (∃ a b : ℤ, a * b = 12 ∧ t = a + b) → t) ∧ p = -527776 :=
sorry

end product_of_constants_l410_410395


namespace man_receives_at_least_paid_cash_l410_410618

theorem man_receives_at_least_paid_cash (s : ℝ) (λ 97_triples : Finset (Fin 100 × Fin 100 × Fin 100)) :
  (∃ (P : Fin 100 → ℝ × ℝ), convex_hull P ∧ area (convex_hull P) = 100 
  ∧ (∑ (t ∈ 97_triples), area (triangle (P t.1) (P t.2) (P t.3)) ≤ s := 0) :=
begin
  sorry,
end

end man_receives_at_least_paid_cash_l410_410618


namespace ab_value_l410_410119

theorem ab_value (a b : ℝ) (h₁ : b = (real.sqrt (1 - 2 * a)) + (real.sqrt (2 * a - 1)) + 3) (h₂ : a = 1 / 2) : a^b = 1 / 8 := 
by 
  sorry

end ab_value_l410_410119


namespace distance_between_foci_of_hyperbola_l410_410359

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : x ^ 2 - 4 * y ^ 2 = 4) :
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_hyperbola_l410_410359


namespace stuffed_dogs_count_l410_410020

theorem stuffed_dogs_count (D : ℕ) (h1 : 14 + D % 7 = 0) : D = 7 :=
by {
  sorry
}

end stuffed_dogs_count_l410_410020


namespace common_intersection_of_circumcircles_l410_410857

-- Define the lines and points of intersection
variables (a b c d : Line)  -- Four lines
variables (P Q B1 B2 C1 C2 F : Point)

-- Define the conditions of the problem
axiom lines_intersect : ∀ {l1 l2 : Line}, l1 ≠ l2 → ∃ p : Point, p ∈ l1 ∧ p ∈ l2
axiom no_three_lines_intersect : ∀ p : Point, ¬ (p ∈ a ∧ p ∈ b ∧ p ∈ c) ∧ ¬ (p ∈ a ∧ p ∈ b ∧ p ∈ d) ∧ ¬ (p ∈ a ∧ p ∈ c ∧ p ∈ d) ∧ ¬ (p ∈ b ∧ p ∈ c ∧ p ∈ d)

-- Define the triangles and their circumscribed circles
axiom triangle_circumcircle : ∀ {A B C : Point}, ∃ (circle : Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle

-- State the theorem to be proven
theorem common_intersection_of_circumcircles :
  ∀ (a b c d : Line),
  (∀ {l1 l2 : Line}, l1 ≠ l2 → ∃ p : Point, p ∈ l1 ∧ p ∈ l2) →
  (∀ p : Point, ¬ (p ∈ a ∧ p ∈ b ∧ p ∈ c) ∧ ¬ (p ∈ a ∧ p ∈ b ∧ p ∈ d) ∧ ¬ (p ∈ a ∧ p ∈ c ∧ p ∈ d) ∧ ¬ (p ∈ b ∧ p ∈ c ∧ p ∈ d)) →
  ∃ F : Point,
    (∃ circle1 : Circle, P ∈ circle1 ∧ B1 ∈ circle1 ∧ B2 ∈ circle1 ∧ F ∈ circle1) ∧
    (∃ circle2 : Circle, P ∈ circle2 ∧ C1 ∈ circle2 ∧ C2 ∈ circle2 ∧ F ∈ circle2) ∧
    (∃ circle3 : Circle, Q ∈ circle3 ∧ B1 ∈ circle3 ∧ C1 ∈ circle3 ∧ F ∈ circle3) ∧
    (∃ circle4 : Circle, Q ∈ circle4 ∧ B2 ∈ circle4 ∧ C2 ∈ circle4 ∧ F ∈ circle4) :=
begin
  sorry
end

end common_intersection_of_circumcircles_l410_410857


namespace root_in_interval_find_k_l410_410943

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a ^ x - x + b

theorem root_in_interval (a b : ℝ) (h₁ : 3 ^ a = 2) (h₂ : 3 ^ b = 9 / 4) :
  ∃ x ∈ set.Ioo 1 2, f a b x = 0 :=
sorry

theorem find_k (a b : ℝ) (h₁ : 3 ^ a = 2) (h₂ : 3 ^ b = 9 / 4) :
  ∃ k : ℤ, k = 1 ∧ ∃ x ∈ set.Ioo (k : ℝ) (k + 1 : ℝ), f a b x = 0 :=
by
  use 1
  split
  · rfl
  · exact root_in_interval a b h₁ h₂

end root_in_interval_find_k_l410_410943


namespace range_of_m_l410_410429

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
by
  sorry

end range_of_m_l410_410429


namespace cos_alpha_of_given_conditions_l410_410057

theorem cos_alpha_of_given_conditions 
  (α : ℝ) 
  (h1 : sin (α + (π / 3)) = 2 / 3) 
  (h2 : π / 6 < α ∧ α < 2 * π / 3) :
  cos α = (2 * real.sqrt 3 - real.sqrt 5) / 6 :=
by sorry

end cos_alpha_of_given_conditions_l410_410057


namespace factorial_division_identity_l410_410832

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l410_410832


namespace find_smallest_n_l410_410693

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l410_410693


namespace value_of_a_unique_zero_h_in_interval_l410_410072

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 * Real.exp (-x)
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

-- 1) Prove that a = 1
theorem value_of_a (a x : ℝ) (ha1 : a ≤ 1) (hx : x > 0) :
  (Real.log x + a / x + 1 >= 2) → (a = 1) :=
begin
  sorry,
end

-- 2) Prove h(x) = f(x) - g(x) has a unique zero in (1, 2)
theorem unique_zero_h_in_interval (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  ∃! (c : ℝ), h 1 c = 0 :=
begin
  sorry,
end

end value_of_a_unique_zero_h_in_interval_l410_410072


namespace vehicle_sampling_is_systematic_l410_410659

-- Conditions:
def city_main_roads := Type
def vehicle (P : city_main_roads) := (license_plate : ℕ)

def is_selected (v : vehicle city_main_roads) : Prop :=
  v.license_plate % 10 = 8

-- Question:
def systematic_sampling := Prop

-- The Proof Statement:
theorem vehicle_sampling_is_systematic :
  (∀ (v : vehicle city_main_roads), is_selected v → systematic_sampling) :=
sorry

end vehicle_sampling_is_systematic_l410_410659


namespace infinitely_many_points_outside_plane_l410_410135

variable {Point Line Plane : Type}
variable (l : Line) (π : Plane) 
variable (P : Point)

-- Assume that Line contains points
-- Assume that Plane contains points
variables (Point_on_Line : l → Point)
variables (Point_on_Plane : π → Point)

-- Assumption: There exists a point P on the line that is outside the plane
axiom exists_point_on_line_outside_plane : ∃ (P : Point), Point_on_Line P ∧ ¬ Point_on_Plane P

-- Theorem: There are infinitely many points on the line that are outside the plane
theorem infinitely_many_points_outside_plane : 
  ∃ (infinitely_many_Ps : set Point), infinite (subtype set P) ∧ ∀ P ∈ infinitely_many_Ps, Point_on_Line P ∧ ¬ Point_on_Plane P :=
sorry

end infinitely_many_points_outside_plane_l410_410135


namespace factorial_div_eq_l410_410844

-- Define the factorial function.
def fact (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * fact (n - 1)

-- State the theorem for the given mathematical problem.
theorem factorial_div_eq : (fact 10) / ((fact 7) * (fact 3)) = 120 := by
  sorry

end factorial_div_eq_l410_410844


namespace compare_p_q_l410_410155

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Conditions
def is_acute_triangle (A B C : ℝ) : Prop := A < π / 2 ∧ B < π / 2 ∧ C < π / 2
def angle_order (A B C : ℝ) : Prop := A < B ∧ B < C
def side_order (a b c : ℝ) : Prop := a < b ∧ b < c
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
def cosine_sum (a b c A B C : ℝ) : ℝ := a * Real.cos A + b * Real.cos B + c * Real.cos C

-- Statement to prove
theorem compare_p_q (a b c A B C : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_angle_order : angle_order A B C) 
  (h_side_order : side_order a b c)
  : cosine_sum a b c A B C ≤ semi_perimeter a b c := sorry

end compare_p_q_l410_410155


namespace shot_cost_is_correct_l410_410355

/-- defining the conditions as parameters -/
def num_pregnant_dogs : Nat := 3
def puppies_per_dog : Nat := 4
def shots_per_puppy : Nat := 2
def total_cost : Nat := 120

/-- defining the total puppies, total shots and cost per shot -/
def total_puppies : Nat :=
  num_pregnant_dogs * puppies_per_dog

def total_shots : Nat :=
  total_puppies * shots_per_puppy

def cost_per_shot : Nat :=
  total_cost / total_shots

/-- statement asserting the correct cost per shot -/
theorem shot_cost_is_correct : cost_per_shot = 5 := by
  /-- inserting the proof here -/
  sorry

end shot_cost_is_correct_l410_410355


namespace smallest_sum_is_20_l410_410528

noncomputable def smallest_sum_of_estimates : ℕ :=
  let estR2 (x : ℕ) : ℕ := nat.ceil ((4 / 5 : ℚ) * x)
  let estR3 (x : ℕ) : ℕ := nat.ceil ((4 / 5 : ℚ) * (24 - x))
  nat.min (finset.image (λ x, estR2 x + estR3 x) (finset.range 24).filter (λ x, 0 < x ∧ x < 24))

theorem smallest_sum_is_20 :
  smallest_sum_of_estimates = 20 := sorry

end smallest_sum_is_20_l410_410528


namespace measure_angle_ABC_regular_octagon_square_l410_410199

-- Define the vertices of the regular octagon
def vertices : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
λ A B C D E F G H, (A, B, C, D, E, F, G, H)

-- Define a function to calculate the internal angle of a regular octagon
def internal_angle_regular_octagon (n : ℕ) : ℝ :=
(↑(n-2) / ↑n) * 180

-- Define the measure of angle ABC in terms of the properties given
def measure_angle_ABC (n : ℕ) (A B C : ℕ) : ℝ :=
if n = 8 then 22.5 else 0

-- Prove that the measure of angle ABC is 22.5 degrees given the conditions
theorem measure_angle_ABC_regular_octagon_square 
  (n : ℕ) (A B C D E F G H : ℕ)
  (h1 : n = 8)
  (h2 : ∀ (i j : ℕ), i ≠ j → i ∈ vertices A B C D E F G H → j ∈ vertices A B C D E F G H)
  (h3 : is_square_on_side (vertices A B C D E F G H) B)
  (h4 : is_diagonal_intersect_at_B (vertices A B C D E F G H) B) :
  measure_angle_ABC n A B C = 22.5 :=
by sorry

end measure_angle_ABC_regular_octagon_square_l410_410199


namespace original_price_of_television_l410_410557

theorem original_price_of_television
  (discount : ℝ := 0.05)
  (first_installment : ℝ := 150)
  (monthly_installment : ℝ := 102)
  (total_paid : ℝ := first_installment + 3 * monthly_installment)
  (paid_after_discount : ℝ := total_paid)
  (percentage_paid : ℝ := 0.95) :
  let P := paid_after_discount / percentage_paid in
  P = 480 := by
    -- The proof would go here
    sorry

end original_price_of_television_l410_410557


namespace product_of_positive_integral_roots_l410_410048

theorem product_of_positive_integral_roots :
  (∃ n : ℕ, (n^2 - 33 * n + 272 : ℤ) = 2) →
  ∏ n in finset.filter (λ n : ℕ, (n^2 - 33 * n + 272 : ℤ) = 2) (finset.range 34) = 270 :=
by
  intros h_exists
  sorry

end product_of_positive_integral_roots_l410_410048


namespace average_goals_per_game_l410_410196

-- Definitions based on the conditions
def total_slices (large_pizzas medium_pizzas : ℕ) : ℕ := (large_pizzas * 12) + (medium_pizzas * 8)
def total_slices_including_bonus (total_slices G : ℕ) : ℕ := total_slices + (5 * G)
def average_goals (total_slices G games : ℕ) : ℕ := (total_slices + (5 * G)) / games

-- The main theorem statement
theorem average_goals_per_game (large_pizzas medium_pizzas games G : ℕ) :
  large_pizzas = 4 →
  medium_pizzas = 6 →
  games = 10 →
  average_goals (total_slices large_pizzas medium_pizzas) G games = (96 + 5 * G) / 10 :=
  by {
    intro h1 h2 h3,
    rw [h1, h2, h3, total_slices, average_goals],
    sorry
  }

end average_goals_per_game_l410_410196


namespace log_a_b_equals_2_l410_410062

noncomputable def imaginary_unit : ℂ := complex.I

noncomputable def complex_number : ℂ := 2 * imaginary_unit * (2 - imaginary_unit)

theorem log_a_b_equals_2 :
  let a := 2 in
  let b := 4 in
  log a b = 2 := by
  sorry

end log_a_b_equals_2_l410_410062


namespace midpoints_form_regular_hexagon_l410_410357

noncomputable def is_regular_hexagon (vertices : Fin 6 → ℂ) : Prop := sorry
noncomputable def midpoint (z1 z2 : ℂ) : ℂ := (z1 + z2) / 2

noncomputable def new_hexagon_vertices (a b c d e f : ℂ) : Fin 6 → ℂ :=
  λ n, match n with
  | 0 => sorry -- Construction for vertex on AB
  | 1 => sorry -- Construction for vertex on BC
  | 2 => sorry -- Construction for vertex on CD
  | 3 => sorry -- Construction for vertex on DE
  | 4 => sorry -- Construction for vertex on EF
  | 5 => sorry -- Construction for vertex on FA
  end

noncomputable def new_hexagon_midpoints (a b c d e f : ℂ) : Fin 6 → ℂ :=
  λ n, match n with
  | 0 => midpoint (new_hexagon_vertices a b c d e f 0) (new_hexagon_vertices a b c d e f 1)
  | 1 => midpoint (new_hexagon_vertices a b c d e f 1) (new_hexagon_vertices a b c d e f 2)
  | 2 => midpoint (new_hexagon_vertices a b c d e f 2) (new_hexagon_vertices a b c d e f 3)
  | 3 => midpoint (new_hexagon_vertices a b c d e f 3) (new_hexagon_vertices a b c d e f 4)
  | 4 => midpoint (new_hexagon_vertices a b c d e f 4) (new_hexagon_vertices a b c d e f 5)
  | 5 => midpoint (new_hexagon_vertices a b c d e f 5) (new_hexagon_vertices a b c d e f 0)
  end

theorem midpoints_form_regular_hexagon (a b c d e f : ℂ) :
  true → is_regular_hexagon (new_hexagon_midpoints a b c d e f) :=
sorry

end midpoints_form_regular_hexagon_l410_410357


namespace cos_90_eq_zero_l410_410006

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l410_410006


namespace books_per_student_l410_410766

-- Define the initial conditions as constants
def total_books : ℕ := 120
def students_day1 : ℕ := 4
def students_day2 : ℕ := 5
def students_day3 : ℕ := 6
def students_day4 : ℕ := 9

-- Define the proof problem statement
theorem books_per_student :
  (students_day1 + students_day2 + students_day3 + students_day4 = 24) →
  (total_books / (students_day1 + students_day2 + students_day3 + students_day4) = 5) :=
by
sry

end books_per_student_l410_410766


namespace number_of_ways_to_distribute_balls_l410_410494

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l410_410494


namespace find_x_for_compositions_l410_410415

theorem find_x_for_compositions (x : ℝ)
  (delta : ℝ → ℝ := λ x, 5 * x + 2)
  (phi : ℝ → ℝ := λ x, 6 * x + 9)
  (h : delta (phi x) = 17) :
  x = -1 :=
by
  sorry

end find_x_for_compositions_l410_410415


namespace gcf_and_multiples_l410_410277

theorem gcf_and_multiples (a b gcf : ℕ) : 
  (a = 90) → (b = 135) → gcd a b = gcf → 
  (gcf = 45) ∧ (45 % gcf = 0) ∧ (90 % gcf = 0) ∧ (135 % gcf = 0) := 
by
  intros ha hb hgcf
  rw [ha, hb] at hgcf
  sorry

end gcf_and_multiples_l410_410277


namespace sum_of_possible_digit_counts_in_base_2_l410_410324

theorem sum_of_possible_digit_counts_in_base_2 :
  (∑ d in ({13, 14, 15} : Finset ℕ), d) = 42 :=
by 
  sorry

end sum_of_possible_digit_counts_in_base_2_l410_410324


namespace first_statement_second_statement_l410_410431

noncomputable def p (m : ℝ) : Prop := ∃ x > 0, x^2 - 2 * Real.exp(1) * Real.log x ≤ m

noncomputable def q (m : ℝ) : Prop := ∀ x ≥ 2, 2 * x^2 - m * x + 2 ≤ 2 * (x + 1)^2 - m * (x + 1) + 2

theorem first_statement (m : ℝ) : ¬(p m ∨ q m) → m ∈ ∅ := 
by sorry

theorem second_statement (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 8 ∨ m < 0) :=
by sorry

end first_statement_second_statement_l410_410431


namespace paper_length_calculation_l410_410756

theorem paper_length_calculation (strip_width : ℝ) (initial_diameter : ℝ) 
  (wraps : ℕ) (final_diameter : ℝ) (fin_length_m : ℝ) :
  strip_width = 2 ∧ initial_diameter = 4 ∧ wraps = 800 ∧ final_diameter = 16 ∧ fin_length_m = 80 * π
  → 
  let n := wraps
  let a := initial_diameter
  let l := final_diameter
  let S := (n / 2) * (a + l)
  let total_length_cm := π * S
  let total_length_m := total_length_cm / 100
  total_length_m = fin_length_m :=
by 
  intro h
  obtain ⟨hw, hi, hn, hf, hfl⟩ := h
  have ha : n = 800, from hn
  have hb : a = 4, from hi
  have hc : l = 16, from hf
  have hd : total_length_cm = 8000 * π, by 
      calc
      total_length_cm = π * S : by sorry
      ... = π * ((n / 2) * (a + l)) : by sorry
      ... = π * ((800 / 2) * (4 + 16)) : by rw [ha, hb, hc]
      ... = π * (400 * 20) : by norm_num
      ... = 8000 * π : by norm_num,
  have he : total_length_m = total_length_cm / 100, from rfl
  have hf : total_length_m = 80 * π, by
      calc
      total_length_m = 8000 * π / 100 : by rw [hd, he]
      ... = 80 * π : by norm_num,
  exact hf.symm ▸ hfl

end paper_length_calculation_l410_410756


namespace tim_income_percentage_less_than_juan_l410_410583

variables (T J M : ℝ)

-- Conditions
def mary_income_condition : Prop := M = 1.60 * T
def mary_juan_relationship : Prop := M = 1.44 * J

-- Problem
theorem tim_income_percentage_less_than_juan (h₁ : mary_income_condition T J M) (h₂ : mary_juan_relationship T J M) :
  0.9 * J = T :=
by sorry

end tim_income_percentage_less_than_juan_l410_410583


namespace remainder_degrees_l410_410282

-- Declare the polynomial divisor
def divisor : Polynomial ℤ := 5 * (X^7) - 2 * (X^3) + X - 8

-- Problem statement: the possible degrees of the remainder when dividing by a specific polynomial divisor
theorem remainder_degrees (p : Polynomial ℤ) :
  ∃ r : Polynomial ℤ, (∀ d, d < 7 → r.degree = Option.some d) ∧ (p = q * divisor + r) :=
sorry

end remainder_degrees_l410_410282


namespace volume_of_regular_triangular_pyramid_l410_410053

theorem volume_of_regular_triangular_pyramid (p α : ℝ) :
  ∃ V : ℝ, V = (9 * p^3 * (Real.tan (α / 2))^3) / (4 * Real.sqrt (3 * (Real.tan (α / 2))^2 - 1)) :=
by
  use (9 * p^3 * (Real.tan (α / 2))^3) / (4 * Real.sqrt (3 * (Real.tan (α / 2))^2 - 1))
  sorry

end volume_of_regular_triangular_pyramid_l410_410053


namespace radiator_water_fraction_l410_410304

theorem radiator_water_fraction :
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  fraction_remaining_per_replacement^4 = 81 / 256 := by
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  sorry

end radiator_water_fraction_l410_410304


namespace find_m_l410_410060

theorem find_m (m : ℝ) (h : (4 * (-1)^3 + 3 * m * (-1)^2 + 6 * (-1) = 2)) :
  m = 4 :=
by
  sorry

end find_m_l410_410060


namespace max_value_l410_410882

/-- 
Proof of the maximum value of the expression 
(sin x + sin 2y + sin 3z) * (cos x + cos 2y + cos 3z)
-/
theorem max_value (x y z : ℝ) : 
  (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z)) ≤ 4.5 :=
sorry

end max_value_l410_410882


namespace johns_journey_length_l410_410560

-- Define the total journey variable
variables (x : ℝ)

-- Define the conditions given in the problem
def gravel_road := x / 4
def highway := 30
def remaining_after_gravel := 3 * x / 4 - 30
def dirt_track := remaining_after_gravel / 3

-- The main theorem
theorem johns_journey_length : gravel_road + highway + dirt_track = x :=
by sorry

end johns_journey_length_l410_410560


namespace find_probability_l410_410473

noncomputable def problem_statement (X : ℝ → ℝ) [is_normal X 0 (σ^2)] : Prop :=
  let P := probability X in
  P (-2 ≤ X ≤ 0) = 0.4 → P (2 < X) = 0.1

-- The theorem with the conditions and the expected output
theorem find_probability (X : ℝ → ℝ) [is_normal X 0 (σ^2)] (h : probability X (-2 ≤ X ≤ 0) = 0.4) :
  probability X (2 < X) = 0.1 :=
begin
  sorry
end

end find_probability_l410_410473


namespace balls_in_boxes_l410_410489

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l410_410489


namespace Tigers_Sharks_min_games_l410_410619

open Nat

theorem Tigers_Sharks_min_games (N : ℕ) : 
  (let total_games := 3 + N
   let sharks_wins := 1 + N
   sharks_wins * 20 ≥ total_games * 19) ↔ N ≥ 37 := 
by
  sorry

end Tigers_Sharks_min_games_l410_410619


namespace moles_of_NaCl_formed_l410_410897

def NaOH : Type := sorry
def HCl : Type := sorry
def NaCl : Type := sorry
def H₂O : Type := sorry

axiom balanced_reaction : NaOH → HCl → NaCl → H₂O → Prop

theorem moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) (h1 : moles_NaOH = 1) (h2 : moles_HCl = 1)
: balanced_reaction NaOH HCl NaCl H₂O →
  (∃ moles_NaCl : ℕ, moles_NaCl = 1) := sorry

end moles_of_NaCl_formed_l410_410897


namespace no_diagonal_path_in_rubiks_cube_l410_410556

/-- A 3x3 Rubik's Cube has a total of 54 faceslets and 56 vertices on its surface.
    Given that each vertex may be part of 3 or 4 faceslets, it is not possible to
    draw a diagonal in each square on the surface such that a non-self-intersecting
    path is formed.
-/
theorem no_diagonal_path_in_rubiks_cube :
  ¬ ∃ path : ℕ → ℕ × ℕ, (∀ n, path n ∈ finset.univ.filter ... -- more specific condition details required
  sorry

end no_diagonal_path_in_rubiks_cube_l410_410556


namespace combination_formula_l410_410810

theorem combination_formula : (10! / (7! * 3!)) = 120 := 
by 
  sorry

end combination_formula_l410_410810


namespace train_length_l410_410332

theorem train_length (speed_jogger_kmh speed_train_kmh lead_jogger_m time_pass_s : ℝ)
  (hjog: speed_jogger_kmh = 9)
  (htrain: speed_train_kmh = 45)
  (hlead: lead_jogger_m = 250)
  (htime: time_pass_s = 37) : 
  let speed_jogger := speed_jogger_kmh * (1000 / 3600)
      speed_train := speed_train_kmh * (1000 / 3600)
      relative_speed := speed_train - speed_jogger
      total_distance := relative_speed * time_pass_s
  in lead_jogger_m + 120 = total_distance := 
by
  intros
  let speed_jogger := speed_jogger_kmh * (1000 / 3600)
  let speed_train := speed_train_kmh * (1000 / 3600)
  let relative_speed := speed_train - speed_jogger
  have h1 : speed_jogger = 2.5 := by sorry
  have h2 : speed_train = 12.5 := by sorry
  have h3 : relative_speed = 10 := by sorry
  have h4 : total_distance = 370 := by sorry
  show lead_jogger_m + 120 = total_distance := by
    rw [hlead, h4]
    norm_num

end train_length_l410_410332


namespace nancy_total_spending_l410_410737

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l410_410737


namespace nancy_total_spending_l410_410738

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l410_410738


namespace min_value_S1_plus_3S2_l410_410449

-- Definitions of the given conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus : ℝ × ℝ := (1, 0)

def line (m : ℝ) (x y : ℝ) : Prop := x = 5 + m * y

def point_on_parabola_intersects_line (x y m : ℝ) : Prop := 
  parabola x y ∧ line m x y

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  1 / 2 * abs (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1)

def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (5, 0)
def F := focus

def S1 (A B : ℝ × ℝ) : ℝ :=
  area_triangle A B O

def S2 (A : ℝ × ℝ) : ℝ :=
  area_triangle A F O

-- Theorem stating the desired property
theorem min_value_S1_plus_3S2 {m : ℝ} (A B : ℝ × ℝ) (y1 y2 : ℝ) (h1 : point_on_parabola_intersects_line (A.fst) y1 m)
   (h2 : point_on_parabola_intersects_line (B.fst) y2 m) : 
  S1 A B + 3 * S2 A = 20 * Real.sqrt 2 := 
sorry  -- Proof is omitted

end min_value_S1_plus_3S2_l410_410449


namespace smallest_angle_less_than_45_l410_410215

theorem smallest_angle_less_than_45
  (a b c m_a m_b m_c : ℝ)
  (h_med_a : m_a = sqrt ((2 * b ^ 2 + 2 * c ^ 2 - a ^ 2) / 4))
  (h_med_b : m_b = sqrt ((2 * a ^ 2 + 2 * c ^ 2 - b ^ 2) / 4))
  (h_med_c : m_c = sqrt ((2 * a ^ 2 + 2 * b ^ 2 - c ^ 2) / 4))
  (h_obtuse : m_a ^ 2 > m_b ^ 2 + m_c ^ 2) :
  ∃ A : ℝ, A < 45 ∧ (cos A = (b^2 + c^2 - a^2) / (2 * b * c)) :=
sorry

end smallest_angle_less_than_45_l410_410215


namespace surface_area_of_rectangular_solid_l410_410374

-- Conditions
variables {a b c : ℕ}
variables (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
variables (h_volume : a * b * c = 308)

-- Question and Proof Problem
theorem surface_area_of_rectangular_solid :
  2 * (a * b + b * c + c * a) = 226 :=
sorry

end surface_area_of_rectangular_solid_l410_410374


namespace tangent_angle_range_l410_410071

def curve (x : ℝ) : ℝ := x^3 - real.sqrt 3 * x + 3 / 5

def derivative (x : ℝ) : ℝ := 3 * x^2 - real.sqrt 3

def angle_of_inclination (k : ℝ) : ℝ := real.arctan k

def inclination_range (α : ℝ) : Prop :=
  (0 ≤ α ∧ α ≤ real.pi / 2) ∨ (2 * real.pi / 3 ≤ α ∧ α < real.pi)

theorem tangent_angle_range (x : ℝ) (k : ℝ) (α : ℝ)
  (h1 : k = derivative x)
  (h2 : α = angle_of_inclination k) :
  inclination_range α :=
sorry

end tangent_angle_range_l410_410071


namespace solve_system_eqns_l410_410611

theorem solve_system_eqns (x y : ℝ) (h1 : 4 * (Real.log 2 x)^2 + 1 = 2 * Real.log 2 y) (h2 : Real.log 2 (x^2) ≥ Real.log 2 y) (hx : x > 0) (hy : y > 0) : (x = Real.sqrt 2 ∧ y = 2) :=
sorry

end solve_system_eqns_l410_410611


namespace fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l410_410200

theorem fraction_area_of_shaded_square_in_larger_square_is_one_eighth :
  let side_larger_square := 4
  let area_larger_square := side_larger_square^2
  let side_shaded_square := Real.sqrt (1^2 + 1^2)
  let area_shaded_square := side_shaded_square^2
  area_shaded_square / area_larger_square = 1 / 8 := 
by 
  sorry

end fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l410_410200


namespace triangle_XYZ_median_inequalities_l410_410553

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end triangle_XYZ_median_inequalities_l410_410553


namespace gcd_of_6Tn2_and_nplus1_eq_2_l410_410410

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2_l410_410410


namespace factorial_div_combination_l410_410825

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410825


namespace constant_term_of_transformed_polynomial_l410_410380

theorem constant_term_of_transformed_polynomial (a b c : ℝ) (h1 : a^3 - 3*a^2 + 6 = 0) (h2 : b^3 - 3*b^2 + 6 = 0) (h3 : c^3 - 3*c^2 + 6 = 0) :
  let p := (fun (x : ℝ) => (x - (3 - c) / c^3) * (x - (3 - a) / a^3) * (x - (3 - b) / b^3)) in
  p 0 = 39 / 216 :=
by 
  sorry

end constant_term_of_transformed_polynomial_l410_410380


namespace find_x_satisfies_fx_eq_4_l410_410462

def f : ℝ → ℝ :=
λ x, if x < 2 then 2^x else (Real.logb 2 x)

theorem find_x_satisfies_fx_eq_4 :
  (∃ x: ℝ, f x = 4) ↔ (x = 16) :=
by
  split
  { intro h
    cases h with x hx
    by_cases h₁: x < 2
    { rw [if_pos h₁] at hx
      have h₂: x = log 2 4 := by sorry
      rw h₂ at h₁
      contradiction
    {
      rw [if_neg h₁] at hx
      have h₃: x = 16 := by sorry
      exact ⟨16, h₃⟩ }
  }
  { intro hx
    use 16
    rw hx
    sorry }

end find_x_satisfies_fx_eq_4_l410_410462


namespace track_circumference_l410_410730

variable (A B : Nat → ℝ)
variable (speedA speedB : ℝ)
variable (x : ℝ) -- half the circumference of the track
variable (y : ℝ) -- the circumference of the track

theorem track_circumference
  (x_pos : 0 < x)
  (y_def : y = 2 * x)
  (start_opposite : A 0 = 0 ∧ B 0 = x)
  (B_first_meet_150 : ∃ t₁, B t₁ = 150 ∧ A t₁ = x - 150)
  (A_second_meet_90 : ∃ t₂, A t₂ = 2 * x - 90 ∧ B t₂ = x + 90) :
  y = 720 := 
by 
  sorry

end track_circumference_l410_410730


namespace factorial_div_combination_l410_410821

theorem factorial_div_combination : nat.factorial 10 / (nat.factorial 7 * nat.factorial 3) = 120 := 
by 
  sorry

end factorial_div_combination_l410_410821


namespace product_of_factors_l410_410404

theorem product_of_factors :
  (∏ (t : ℤ) in { t | ∃ a b : ℤ, a * b = 12 ∧ t = a + b }, t) = -530144 :=
by sorry

end product_of_factors_l410_410404


namespace distance_AB_l410_410954

noncomputable def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem distance_AB {x : ℝ} (hx : x = 19 + 6 * real.sqrt 10 ∨ x = 19 - 6 * real.sqrt 10)
  (A B : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (h_sym : B.2 = - A.2)
  (F : ℝ × ℝ := (1, 0))
  (M := midpoint A F)
  (N := midpoint B F)
  (hAN_BM_perp : (A.1 - N.1) * (B.1 - M.1) + A.2 - N.2 * (B.2 - M.2) = 0) :
  |B.2 - A.2| = 4 * real.sqrt 10 + 12 ∨ |B.2 - A.2| = 4 * real.sqrt 10 - 12 :=
sorry

end distance_AB_l410_410954


namespace sum_of_coefficients_l410_410246

theorem sum_of_coefficients (x : ℝ) :
    let A := 5
    let B := 4
    let C := 25
    let D := -20
    let E := 16
    (125 * x^3 + 64 = (A * x + B) * (C * x^2 + D * x + E)) →
    (A + B + C + D + E = 30) :=
by 
  intros A B C D E
  rw [A, B, C, D, E]
  sorry

end sum_of_coefficients_l410_410246


namespace trader_loss_percent_l410_410126

noncomputable def CP1 : ℝ := 325475 / 1.13
noncomputable def CP2 : ℝ := 325475 / 0.87
noncomputable def TCP : ℝ := CP1 + CP2
noncomputable def TSP : ℝ := 325475 * 2
noncomputable def profit_or_loss : ℝ := TSP - TCP
noncomputable def profit_or_loss_percent : ℝ := (profit_or_loss / TCP) * 100

theorem trader_loss_percent : profit_or_loss_percent = -1.684 := by 
  sorry

end trader_loss_percent_l410_410126


namespace possible_remainder_degrees_l410_410285

theorem possible_remainder_degrees (f : Polynomial ℝ) :
  ∃ d, (degree (X ^ 7 - 2 * X ^ 3 + X - 8) = 7) ∧ 
  (0 ≤ d ∧ d < 7) → ∃ r, degree r = d :=
begin
  sorry
end

end possible_remainder_degrees_l410_410285


namespace beginner_trigonometry_probability_l410_410142

def BC := ℝ
def AC := ℝ
def IC := ℝ
def BT := ℝ
def AT := ℝ
def IT := ℝ
def T := 5000

theorem beginner_trigonometry_probability :
  ∀ (BC AC IC BT AT IT : ℝ),
  (BC + AC + IC = 0.60 * T) →
  (BT + AT + IT = 0.40 * T) →
  (BC + BT = 0.45 * T) →
  (AC + AT = 0.35 * T) →
  (IC + IT = 0.20 * T) →
  (BC = 1.25 * BT) →
  (IC + AC = 1.20 * (IT + AT)) →
  (BT / T = 1/5) :=
by
  intros
  sorry

end beginner_trigonometry_probability_l410_410142


namespace largest_integer_less_than_120_with_remainder_5_div_8_l410_410870

theorem largest_integer_less_than_120_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 120 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 120 → m % 8 = 5 → m ≤ n :=
sorry

end largest_integer_less_than_120_with_remainder_5_div_8_l410_410870


namespace part1_part2_l410_410478

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem part1 : A ∩ B = {3, 5} := by
  sorry

theorem part2 : (U \ A) ∪ B = {3, 4, 5, 6} := by
  sorry

end part1_part2_l410_410478


namespace problem_I_problem_II_problem_III_l410_410066

theorem problem_I (R W : ℕ) (hR : R = 4) (hW : W = 6) :
  ∑ r in {0..4}, ∑ w in {0..4}, if r + w = 4 ∧ r ≥ w then choose 4 r * choose 6 w else 0 = 115 :=
by sorry

theorem problem_II (R W : ℕ) (hR : R = 4) (hW : W = 6) :
  ∑ r in {0..5}, ∑ w in {0..5}, if r + w = 5 ∧ 2 * r + w ≥ 7 then choose 4 r * choose 6 w else 0 = 186 :=
by sorry

theorem problem_III (R W : ℕ) (hR : R = 4) (hW : W = 6) :
  ∑ r in {0..4}, ∑ w in {0..6}, if r = 3 ∧ w = 2 ∧ 2 * r + w = 8 then choose 4 r * choose 6 w * 3 * 3 * 2 * 1 * 2 * 1 else 0 = 4320 :=
by sorry

end problem_I_problem_II_problem_III_l410_410066


namespace product_of_constants_l410_410394

theorem product_of_constants :
  ∃ (a b : ℤ), ab = 12 ∧
  let t := a + b in 
  (∏ (p : (Σ a b : ℤ, a * b = 12)), p.2.1 + p.2.2) = -531441 :=
by
  sorry

end product_of_constants_l410_410394


namespace distance_from_P_to_AB_l410_410181

-- Let \(ABC\) be an isosceles triangle where \(AB\) is the base. 
-- An altitude from vertex \(C\) to base \(AB\) measures 6 units.
-- A line drawn through a point \(P\) inside the triangle, parallel to base \(AB\), 
-- divides the triangle into two regions of equal area.
-- The vertex angle at \(C\) is a right angle.
-- Prove that the distance from \(P\) to \(AB\) is 3 units.

theorem distance_from_P_to_AB :
  ∀ (A B C P : Type)
    (distance_AB distance_AC distance_BC : ℝ)
    (is_isosceles : distance_AC = distance_BC)
    (right_angle_C : distance_AC^2 + distance_BC^2 = distance_AB^2)
    (altitude_C : distance_BC = 6)
    (line_through_P_parallel_to_AB : ∃ (P_x : ℝ), 0 < P_x ∧ P_x < distance_BC),
  ∃ (distance_P_to_AB : ℝ), distance_P_to_AB = 3 :=
by
  sorry

end distance_from_P_to_AB_l410_410181


namespace part1_part2_l410_410545

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (c : ℕ → ℕ → ℝ)
variable (T : ℕ → ℝ)

axiom h1 : a 1 = 1
axiom h2 : a 2 = 3
axiom h3 : ∀ n ≥ 2, ∃ A B C : ℝ, A = S (n + 1) ∧ B = S n ∧ C = S (n - 1)
axiom h4 : ∀ n ≥ 2, ∃ u v : ℝ, u = B - A ∧ v = C - B ∧ u = ((2 * a n + 1) / a n) * v
axiom h5 : b 1 = 1
axiom h6 : ∀ n, b (n + 1) - b n = real.log 2 (a n + 1)
axiom h7 : ∀ n, c n = 4 ^ ((b (n + 1) - 1) / (n + 1)) / (a n * a (n + 1))
axiom h8 : ∀ n, T n = ∑ i in finset.range n, c i

theorem part1 (n : ℕ) : ∃ r, ∀ n ≥ 2, a n + 1 = 2 * (a (n - 1) + 1) := sorry

theorem part2 (n : ℕ) : T n < 1 := sorry

end part1_part2_l410_410545


namespace tan_alpha_beta_identity_l410_410924

noncomputable
def tan_alpha_beta_sum (α β : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + 6 * x + 7 = 0 ∧ x = tan α) ∧ (∃ y : ℝ, y^2 + 6 * y + 7 = 0 ∧ y = tan β) ∧ (tan α + tan β = -6) ∧ (tan α * tan β = 7)

theorem tan_alpha_beta_identity (α β : ℝ) (h : tan_alpha_beta_sum α β) : tan (α + β) = 1 :=
  sorry

end tan_alpha_beta_identity_l410_410924


namespace locus_is_line_segment_l410_410045

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def locus (M : ℝ × ℝ) : Prop := distance M F1 + distance M F2 = 2

theorem locus_is_line_segment :
  ∀ (M : ℝ × ℝ), locus M ↔ (M.1 = (F1.1 * (1 - M.2) + F2.1 * (M.2)) / (1 + M.2) ∧ F1.2 = 0 ∧ F2.2 = 0) :=
sorry

end locus_is_line_segment_l410_410045


namespace factorial_quotient_l410_410815

theorem factorial_quotient : (10! / (7! * 3!)) = 120 := by
  sorry

end factorial_quotient_l410_410815


namespace max_sin_cos_expr_l410_410896

theorem max_sin_cos_expr (x y z : ℝ) :
  let expr := (sin x + sin (2 * y) + sin (3 * z)) * (cos x + cos (2 * y) + cos (3 * z))
  in ∀ (x y z : ℝ), expr ≤ 4.5 :=
sorry

end max_sin_cos_expr_l410_410896


namespace intersection_A_B_eq_1_2_l410_410414

noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℤ := {x : ℤ | x^2 < 9}

theorem intersection_A_B_eq_1_2 : A ∩ B = {1, 2} :=
by sorry

end intersection_A_B_eq_1_2_l410_410414
