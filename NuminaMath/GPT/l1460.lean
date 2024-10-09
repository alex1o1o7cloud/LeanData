import Mathlib

namespace product_of_reals_condition_l1460_146089

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end product_of_reals_condition_l1460_146089


namespace inequality_proof_l1460_146011

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  (1 - 2 * x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2 * y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2 * z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) :=
by
  sorry

end inequality_proof_l1460_146011


namespace math_proof_problem_l1460_146045

theorem math_proof_problem
  (n m k l : ℕ)
  (hpos_n : n > 0)
  (hpos_m : m > 0)
  (hpos_k : k > 0)
  (hpos_l : l > 0)
  (hneq_n : n ≠ 1)
  (hdiv : n^k + m*n^l + 1 ∣ n^(k+l) - 1) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) :=
by 
  sorry

end math_proof_problem_l1460_146045


namespace mars_bars_count_l1460_146065

theorem mars_bars_count (total_candy_bars snickers butterfingers : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_butterfingers : butterfingers = 7) :
  total_candy_bars - (snickers + butterfingers) = 2 :=
by sorry

end mars_bars_count_l1460_146065


namespace convert_spherical_coords_l1460_146036

theorem convert_spherical_coords (ρ θ φ : ℝ) (hρ : ρ > 0) (hθ : 0 ≤ θ ∧ θ < 2 * π) (hφ : 0 ≤ φ ∧ φ ≤ π) :
  (ρ = 4 ∧ θ = 4 * π / 3 ∧ φ = π / 4) ↔ (ρ, θ, φ) = (4, 4 * π / 3, π / 4) :=
by { sorry }

end convert_spherical_coords_l1460_146036


namespace min_value_of_quadratic_l1460_146088

noncomputable def quadratic_min_value (x : ℕ) : ℝ :=
  3 * (x : ℝ)^2 - 12 * x + 800

theorem min_value_of_quadratic : (∀ x : ℕ, quadratic_min_value x ≥ 788) ∧ (quadratic_min_value 2 = 788) :=
by
  sorry

end min_value_of_quadratic_l1460_146088


namespace convert_base_9A3_16_to_4_l1460_146046

theorem convert_base_9A3_16_to_4 :
  let h₁ := 9
  let h₂ := 10 -- A in hexadecimal
  let h₃ := 3
  let b₁ := 21 -- h₁ converted to base 4
  let b₂ := 22 -- h₂ converted to base 4
  let b₃ := 3  -- h₃ converted to base 4
  9 * 16^2 + 10 * 16^1 + 3 * 16^0 = 2 * 4^5 + 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 0 * 4^1 + 3 * 4^0 :=
by
  sorry

end convert_base_9A3_16_to_4_l1460_146046


namespace exists_good_number_in_interval_l1460_146094

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≤ 5

theorem exists_good_number_in_interval (x : ℕ) (hx : x ≠ 0) :
  ∃ g : ℕ, is_good_number g ∧ x ≤ g ∧ g < ((9 * x) / 5) + 1 := 
sorry

end exists_good_number_in_interval_l1460_146094


namespace second_car_speed_correct_l1460_146040

noncomputable def first_car_speed : ℝ := 90

noncomputable def time_elapsed (h : ℕ) (m : ℕ) : ℝ := h + m / 60

noncomputable def distance_travelled (speed : ℝ) (time : ℝ) : ℝ := speed * time

def distance_ratio_at_832 (dist1 dist2 : ℝ) : Prop := dist1 = 1.2 * dist2
def distance_ratio_at_920 (dist1 dist2 : ℝ) : Prop := dist1 = 2 * dist2

noncomputable def time_first_car_832 : ℝ := time_elapsed 0 24
noncomputable def dist_first_car_832 : ℝ := distance_travelled first_car_speed time_first_car_832

noncomputable def dist_second_car_832 : ℝ := dist_first_car_832 / 1.2

noncomputable def time_first_car_920 : ℝ := time_elapsed 1 12
noncomputable def dist_first_car_920 : ℝ := distance_travelled first_car_speed time_first_car_920

noncomputable def dist_second_car_920 : ℝ := dist_first_car_920 / 2

noncomputable def time_second_car_travel : ℝ := time_elapsed 0 42

noncomputable def second_car_speed : ℝ := (dist_second_car_920 - dist_second_car_832) / time_second_car_travel

theorem second_car_speed_correct :
  second_car_speed = 34.2857 := by
  sorry

end second_car_speed_correct_l1460_146040


namespace infinite_subsets_exists_divisor_l1460_146086

-- Definition of the set M
def M : Set ℕ := { n | ∃ a b : ℕ, n = 2^a * 3^b }

-- Infinite family of subsets of M
variable (A : ℕ → Set ℕ)
variables (inf_family : ∀ i, A i ⊆ M)

-- Theorem statement
theorem infinite_subsets_exists_divisor :
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x := by
  sorry

end infinite_subsets_exists_divisor_l1460_146086


namespace triangle_side_b_length_l1460_146005

noncomputable def length_of_side_b (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) : Prop :=
  b = 21 / 13

theorem triangle_side_b_length (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) :
  length_of_side_b A B C a b c h1 h2 h3 :=
by
  sorry

end triangle_side_b_length_l1460_146005


namespace sufficient_but_not_necessary_condition_l1460_146098

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1460_146098


namespace proposition_a_is_true_l1460_146048

-- Define a quadrilateral
structure Quadrilateral (α : Type*) [Ring α] :=
(a b c d : α)

-- Define properties of a Quadrilateral
def parallel_and_equal_opposite_sides (Q : Quadrilateral ℝ) : Prop := sorry  -- Assumes parallel and equal opposite sides
def is_parallelogram (Q : Quadrilateral ℝ) : Prop := sorry  -- Defines a parallelogram

-- The theorem we need to prove
theorem proposition_a_is_true (Q : Quadrilateral ℝ) (h : parallel_and_equal_opposite_sides Q) : is_parallelogram Q :=
sorry

end proposition_a_is_true_l1460_146048


namespace part1_l1460_146052

theorem part1 (a b : ℝ) : 3*(a - b)^2 - 6*(a - b)^2 + 2*(a - b)^2 = - (a - b)^2 :=
by
  sorry

end part1_l1460_146052


namespace quadratic_bound_l1460_146069

theorem quadratic_bound (a b c : ℝ) :
  (∀ (u : ℝ), |u| ≤ 10 / 11 → ∃ (v : ℝ), |u - v| ≤ 1 / 11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2 := by
  sorry

end quadratic_bound_l1460_146069


namespace inequality_solution_l1460_146026

theorem inequality_solution (x : ℝ) :
  2 * (2 * x - 1) > 3 * x - 1 → x > 1 :=
by
  sorry

end inequality_solution_l1460_146026


namespace regular_polygons_from_cube_intersection_l1460_146033

noncomputable def cube : Type := sorry  -- Define a 3D cube type
noncomputable def plane : Type := sorry  -- Define a plane type

-- Define what it means for a polygon to be regular (equilateral and equiangular)
def is_regular_polygon (polygon : Type) : Prop := sorry

-- Define a function that describes the intersection of a plane with a cube,
-- resulting in a polygon
noncomputable def intersection (c : cube) (p : plane) : Type := sorry

-- Define predicates for the specific regular polygons: triangle, quadrilateral, and hexagon
def is_triangle (polygon : Type) : Prop := sorry
def is_quadrilateral (polygon : Type) : Prop := sorry
def is_hexagon (polygon : Type) : Prop := sorry

-- Ensure these predicates imply regular polygons
axiom triangle_is_regular : ∀ (t : Type), is_triangle t → is_regular_polygon t
axiom quadrilateral_is_regular : ∀ (q : Type), is_quadrilateral q → is_regular_polygon q
axiom hexagon_is_regular : ∀ (h : Type), is_hexagon h → is_regular_polygon h

-- The main theorem statement
theorem regular_polygons_from_cube_intersection (c : cube) (p : plane) :
  is_regular_polygon (intersection c p) →
  is_triangle (intersection c p) ∨ is_quadrilateral (intersection c p) ∨ is_hexagon (intersection c p) :=
sorry

end regular_polygons_from_cube_intersection_l1460_146033


namespace union_is_faction_l1460_146024

variable {D : Type} (is_faction : Set D → Prop)
variable (A B : Set D)

-- Define the complement
def complement (S : Set D) : Set D := {x | x ∉ S}

-- State the given condition
axiom faction_complement_union (A B : Set D) : 
  is_faction A → is_faction B → is_faction (complement (A ∪ B))

-- The theorem to prove
theorem union_is_faction (A B : Set D) :
  is_faction A → is_faction B → is_faction (A ∪ B) := 
by
  -- Proof goes here
  sorry

end union_is_faction_l1460_146024


namespace product_b1_b13_l1460_146080

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) := ∀ n m k : ℕ, m > 0 → k > 0 → a (n + m) - a n = a (n + k) - a (n + k - m)

-- Conditions for the geometric sequence
def is_geometric_seq (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

-- Given conditions
def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (a 3 - (a 7 ^ 2) / 2 + a 11 = 0) ∧ (b 7 = a 7)

theorem product_b1_b13 
  (ha : is_arithmetic_seq a)
  (hb : is_geometric_seq b)
  (h : conditions a b) :
  b 1 * b 13 = 16 :=
sorry

end product_b1_b13_l1460_146080


namespace acme_horseshoes_production_l1460_146061

theorem acme_horseshoes_production
  (profit : ℝ)
  (initial_outlay : ℝ)
  (cost_per_set : ℝ)
  (selling_price : ℝ)
  (number_of_sets : ℕ) :
  profit = selling_price * number_of_sets - (initial_outlay + cost_per_set * number_of_sets) →
  profit = 15337.5 →
  initial_outlay = 12450 →
  cost_per_set = 20.75 →
  selling_price = 50 →
  number_of_sets = 950 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end acme_horseshoes_production_l1460_146061


namespace fractional_part_zero_l1460_146030

noncomputable def fractional_part (z : ℝ) : ℝ := z - (⌊z⌋ : ℝ)

theorem fractional_part_zero (x : ℝ) :
  fractional_part (1 / 3 * (1 / 3 * (1 / 3 * x - 3) - 3) - 3) = 0 ↔ 
  ∃ k : ℤ, 27 * k + 9 ≤ x ∧ x < 27 * k + 18 :=
by
  sorry

end fractional_part_zero_l1460_146030


namespace greatest_integer_y_l1460_146034

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l1460_146034


namespace solve_abs_inequality_l1460_146007

theorem solve_abs_inequality (x : ℝ) (h : abs ((8 - x) / 4) < 3) : -4 < x ∧ x < 20 := 
  sorry

end solve_abs_inequality_l1460_146007


namespace first_sphere_weight_l1460_146075

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * (r ^ 2)

noncomputable def weight (r1 r2 : ℝ) (W2 : ℝ) : ℝ :=
  let A1 := surface_area r1
  let A2 := surface_area r2
  (W2 * A1) / A2

theorem first_sphere_weight :
  let r1 := 0.15
  let r2 := 0.3
  let W2 := 32
  weight r1 r2 W2 = 8 := 
by
  sorry

end first_sphere_weight_l1460_146075


namespace identity_holds_for_all_real_numbers_l1460_146047

theorem identity_holds_for_all_real_numbers (a b : ℝ) : 
  a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by sorry

end identity_holds_for_all_real_numbers_l1460_146047


namespace diameter_large_circle_correct_l1460_146070

noncomputable def diameter_of_large_circle : ℝ :=
  2 * (Real.sqrt 17 + 4)

theorem diameter_large_circle_correct :
  ∃ (d : ℝ), (∀ (r : ℝ), r = Real.sqrt 17 + 4 → d = 2 * r) ∧ d = diameter_of_large_circle := by
    sorry

end diameter_large_circle_correct_l1460_146070


namespace range_of_m_l1460_146090

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = -(m + 2) ∧ x1 * x2 = m + 5) : -5 < m ∧ m < -2 := 
sorry

end range_of_m_l1460_146090


namespace range_of_3a_minus_b_l1460_146082

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 4) (h2 : -1 ≤ a - b ∧ a - b ≤ 2) :
  -1 ≤ (3 * a - b) ∧ (3 * a - b) ≤ 8 :=
sorry

end range_of_3a_minus_b_l1460_146082


namespace sum_of_A_B_in_B_l1460_146010

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k }
def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def C : Set ℤ := { x | ∃ k : ℤ, x = 4 * k + 1 }

theorem sum_of_A_B_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_of_A_B_in_B_l1460_146010


namespace largest_solution_l1460_146031

-- Define the largest solution to the equation |5x - 3| = 28 as 31/5.
theorem largest_solution (x : ℝ) (h : |5 * x - 3| = 28) : x ≤ 31 / 5 := 
  sorry

end largest_solution_l1460_146031


namespace carpet_shaded_area_is_correct_l1460_146037

def total_shaded_area (carpet_side_length : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  let large_shaded_area := large_square_side * large_square_side
  let small_shaded_area := small_square_side * small_square_side
  large_shaded_area + 12 * small_shaded_area

theorem carpet_shaded_area_is_correct :
  ∀ (S T : ℝ), 
  12 / S = 4 →
  S / T = 4 →
  total_shaded_area 12 S T = 15.75 :=
by
  intros S T h1 h2
  sorry

end carpet_shaded_area_is_correct_l1460_146037


namespace point_on_opposite_sides_l1460_146015

theorem point_on_opposite_sides (y_0 : ℝ) :
  (2 - 2 * 3 + 5 > 0) ∧ (6 - 2 * y_0 < 0) → y_0 > 3 :=
by
  sorry

end point_on_opposite_sides_l1460_146015


namespace ratio_is_three_l1460_146029

-- Define the conditions
def area_of_garden : ℕ := 588
def width_of_garden : ℕ := 14
def length_of_garden : ℕ := area_of_garden / width_of_garden

-- Define the ratio
def ratio_length_to_width := length_of_garden / width_of_garden

-- The proof statement
theorem ratio_is_three : ratio_length_to_width = 3 := 
by sorry

end ratio_is_three_l1460_146029


namespace contribution_proof_l1460_146071

theorem contribution_proof (total : ℕ) (a_months b_months : ℕ) (a_total b_total a_received b_received : ℕ) :
  total = 3400 →
  a_months = 12 →
  b_months = 16 →
  a_received = 2070 →
  b_received = 1920 →
  (∃ (a_contributed b_contributed : ℕ), a_contributed = 1800 ∧ b_contributed = 1600) :=
by
  sorry

end contribution_proof_l1460_146071


namespace multiply_scientific_notation_l1460_146074

theorem multiply_scientific_notation (a b : ℝ) (e1 e2 : ℤ) 
  (h1 : a = 2) (h2 : b = 8) (h3 : e1 = 3) (h4 : e2 = 3) :
  (a * 10^e1) * (b * 10^e2) = 1.6 * 10^7 :=
by
  simp [h1, h2, h3, h4]
  sorry

end multiply_scientific_notation_l1460_146074


namespace function_equality_l1460_146001

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l1460_146001


namespace find_function_l1460_146009

variable (R : Type) [LinearOrderedField R]

theorem find_function
  (f : R → R)
  (h : ∀ x y : R, f (x + y) + y ≤ f (f (f x))) :
  ∃ c : R, ∀ x : R, f x = c - x :=
sorry

end find_function_l1460_146009


namespace construct_angle_approx_l1460_146021
-- Use a broader import to bring in the entirety of the necessary library

-- Define the problem 
theorem construct_angle_approx (α : ℝ) (m : ℕ) (h : ∃ l : ℕ, (l : ℝ) / 2^m * 90 ≤ α ∧ α ≤ ((l+1) : ℝ) / 2^m * 90) :
  ∃ β : ℝ, β ∈ { β | ∃ l : ℕ, β = (l : ℝ) / 2^m * 90} ∧ |α - β| ≤ 90 / 2^m :=
sorry

end construct_angle_approx_l1460_146021


namespace sum_of_squares_l1460_146039

theorem sum_of_squares (x y z : ℕ) (h1 : x + y + z = 30)
  (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  x^2 + y^2 + z^2 = 504 :=
by
  sorry

end sum_of_squares_l1460_146039


namespace recycling_weight_l1460_146055

theorem recycling_weight :
  let marcus_milk_bottles := 25
  let john_milk_bottles := 20
  let sophia_milk_bottles := 15
  let marcus_cans := 30
  let john_cans := 25
  let sophia_cans := 35
  let milk_bottle_weight := 0.5
  let can_weight := 0.025

  let total_milk_bottles_weight := (marcus_milk_bottles + john_milk_bottles + sophia_milk_bottles) * milk_bottle_weight
  let total_cans_weight := (marcus_cans + john_cans + sophia_cans) * can_weight
  let combined_weight := total_milk_bottles_weight + total_cans_weight

  combined_weight = 32.25 :=
by
  sorry

end recycling_weight_l1460_146055


namespace correct_system_of_equations_l1460_146051

theorem correct_system_of_equations (x y : ℝ) :
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) :=
sorry

end correct_system_of_equations_l1460_146051


namespace solve_for_b_l1460_146064

theorem solve_for_b (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x + 6 = 0 ↔ y = (2 / 3) * x - 2) → 
  (∀ x y, 4 * y + b * x + 3 = 0 ↔ y = -(b / 4) * x - 3 / 4) → 
  (∀ m1 m2, (m1 = (2 / 3)) → (m2 = -(b / 4)) → m1 * m2 = -1) → 
  b = 6 :=
sorry

end solve_for_b_l1460_146064


namespace largest_integer_in_mean_set_l1460_146017

theorem largest_integer_in_mean_set :
  ∃ (A B C D : ℕ), 
    A < B ∧ B < C ∧ C < D ∧
    (A + B + C + D) = 4 * 68 ∧
    A ≥ 5 ∧
    D = 254 :=
sorry

end largest_integer_in_mean_set_l1460_146017


namespace quadratic_residue_one_mod_p_l1460_146067

theorem quadratic_residue_one_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ) :
  (a^2 % p = 1 % p) ↔ (a % p = 1 % p ∨ a % p = (p-1) % p) :=
sorry

end quadratic_residue_one_mod_p_l1460_146067


namespace parking_average_cost_l1460_146035

noncomputable def parking_cost_per_hour := 
  let cost_two_hours : ℝ := 20.00
  let cost_per_excess_hour : ℝ := 1.75
  let weekend_surcharge : ℝ := 5.00
  let discount_rate : ℝ := 0.10
  let total_hours : ℝ := 9.00
  let excess_hours : ℝ := total_hours - 2.00
  let remaining_cost := cost_per_excess_hour * excess_hours
  let total_cost_before_discount := cost_two_hours + remaining_cost + weekend_surcharge
  let discount := discount_rate * total_cost_before_discount
  let discounted_total_cost := total_cost_before_discount - discount
  let average_cost_per_hour := discounted_total_cost / total_hours
  average_cost_per_hour

theorem parking_average_cost :
  parking_cost_per_hour = 3.725 := 
by
  sorry

end parking_average_cost_l1460_146035


namespace degree_of_divisor_polynomial_l1460_146097

theorem degree_of_divisor_polynomial (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15)
  (hq : q.degree = 9)
  (hr : r.degree = 4)
  (hfdqr : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_polynomial_l1460_146097


namespace union_of_A_B_l1460_146043

def A (p q : ℝ) : Set ℝ := {x | x^2 + p * x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - p * x - 2 * q = 0}

theorem union_of_A_B (p q : ℝ)
  (h1 : A p q ∩ B p q = {-1}) :
  A p q ∪ B p q = {-1, -2, 4} := by
sorry

end union_of_A_B_l1460_146043


namespace max_xyz_l1460_146003

theorem max_xyz (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : 5 * x + 8 * y + 3 * z = 90) : xyz ≤ 225 :=
by
  sorry

end max_xyz_l1460_146003


namespace simplify_fraction_150_div_225_l1460_146025

theorem simplify_fraction_150_div_225 :
  let a := 150
  let b := 225
  let gcd_ab := Nat.gcd a b
  let num_fact := 2 * 3 * 5^2
  let den_fact := 3^2 * 5^2
  gcd_ab = 75 →
  num_fact = a →
  den_fact = b →
  (a / gcd_ab) / (b / gcd_ab) = (2 / 3) :=
  by
    intros 
    sorry

end simplify_fraction_150_div_225_l1460_146025


namespace probability_is_two_thirds_l1460_146044

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l1460_146044


namespace mutually_exclusive_one_two_odd_l1460_146020

-- Define the event that describes rolling a fair die
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Event: Exactly one die shows an odd number -/
def exactly_one_odd (d1 d2 : ℕ) : Prop :=
  (is_odd d1 ∧ ¬ is_odd d2) ∨ (¬ is_odd d1 ∧ is_odd d2)

/-- Event: Exactly two dice show odd numbers -/
def exactly_two_odd (d1 d2 : ℕ) : Prop :=
  is_odd d1 ∧ is_odd d2

/-- Main theorem: Exactly one odd number and exactly two odd numbers are mutually exclusive but not converse-/
theorem mutually_exclusive_one_two_odd (d1 d2 : ℕ) :
  (exactly_one_odd d1 d2 ∧ ¬ exactly_two_odd d1 d2) ∧
  (¬ exactly_one_odd d1 d2 ∧ exactly_two_odd d1 d2) ∧
  (exactly_one_odd d1 d2 ∨ exactly_two_odd d1 d2) :=
by
  sorry

end mutually_exclusive_one_two_odd_l1460_146020


namespace quadratic_solution1_quadratic_solution2_l1460_146085

theorem quadratic_solution1 (x : ℝ) :
  (x^2 + 4 * x - 4 = 0) ↔ (x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) :=
by sorry

theorem quadratic_solution2 (x : ℝ) :
  ((x - 1)^2 = 2 * (x - 1)) ↔ (x = 1 ∨ x = 3) :=
by sorry

end quadratic_solution1_quadratic_solution2_l1460_146085


namespace cauchy_solution_l1460_146078

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end cauchy_solution_l1460_146078


namespace water_added_is_five_l1460_146027

theorem water_added_is_five :
  ∃ W x : ℝ, (4 / 3 = 10 / W) ∧ (4 / 5 = 10 / (W + x)) ∧ x = 5 := by
  sorry

end water_added_is_five_l1460_146027


namespace clock_correct_after_240_days_l1460_146004

theorem clock_correct_after_240_days (days : ℕ) (minutes_fast_per_day : ℕ) (hours_to_be_correct : ℕ) 
  (h1 : minutes_fast_per_day = 3) (h2 : hours_to_be_correct = 12) : 
  (days * minutes_fast_per_day) % (hours_to_be_correct * 60) = 0 :=
by 
  -- Proof skipped
  sorry

end clock_correct_after_240_days_l1460_146004


namespace total_worth_is_correct_l1460_146079

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l1460_146079


namespace mean_score_40_l1460_146095

theorem mean_score_40 (mean : ℝ) (std_dev : ℝ) (h_std_dev : std_dev = 10)
  (h_within_2_std_dev : ∀ (score : ℝ), score ≥ mean - 2 * std_dev)
  (h_lowest_score : ∀ (score : ℝ), score = 20 → score = mean - 20) :
  mean = 40 :=
by
  -- Placeholder for the proof
  sorry

end mean_score_40_l1460_146095


namespace tank_capacity_is_48_l1460_146076

-- Define the conditions
def num_4_liter_bucket_used : ℕ := 12
def num_3_liter_bucket_used : ℕ := num_4_liter_bucket_used + 4

-- Define the capacities of the buckets and the tank
def bucket_4_liters_capacity : ℕ := 4 * num_4_liter_bucket_used
def bucket_3_liters_capacity : ℕ := 3 * num_3_liter_bucket_used

-- Tank capacity
def tank_capacity : ℕ := 48

-- Statement to prove
theorem tank_capacity_is_48 : 
    bucket_4_liters_capacity = tank_capacity ∧
    bucket_3_liters_capacity = tank_capacity := by
  sorry

end tank_capacity_is_48_l1460_146076


namespace transformed_sum_l1460_146077

theorem transformed_sum (n : ℕ) (y : Fin n → ℝ) (s : ℝ) (h : s = (Finset.univ.sum (fun i => y i))) :
  Finset.univ.sum (fun i => 3 * (y i) + 30) = 3 * s + 30 * n :=
by 
  sorry

end transformed_sum_l1460_146077


namespace correct_statement_2_l1460_146019

-- Definitions of parallel and perpendicular relationships
variables (a b : line) (α β : plane)

-- Conditions
def parallel (x y : plane) : Prop := sorry -- definition not provided
def perpendicular (x y : plane) : Prop := sorry -- definition not provided
def line_parallel_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular (l1 l2 : line) : Prop := sorry -- definition not provided

-- Proof of the correct statement among the choices
theorem correct_statement_2 :
  line_perpendicular a b → line_perpendicular_plane a α → line_perpendicular_plane b β → perpendicular α β :=
by
  intros h1 h2 h3
  sorry

end correct_statement_2_l1460_146019


namespace geometric_sequence_sixth_term_l1460_146053

variable (a r : ℝ) 

theorem geometric_sequence_sixth_term (h1 : a * (1 + r + r^2 + r^3) = 40)
                                    (h2 : a * r^4 = 32) :
  a * r^5 = 1280 / 15 :=
by sorry

end geometric_sequence_sixth_term_l1460_146053


namespace inequality_one_inequality_two_l1460_146054

variable (a b c : ℝ)

-- First Inequality Proof Statement
theorem inequality_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := 
sorry

-- Second Inequality Proof Statement
theorem inequality_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c) ≥ 2 * (a + b + c) := 
sorry

end inequality_one_inequality_two_l1460_146054


namespace probability_top_two_same_suit_l1460_146016

theorem probability_top_two_same_suit :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let first_card_prob := (13 / 52 : ℚ)
  let remaining_cards := 51
  let second_card_same_suit_prob := (12 / 51 : ℚ)
  first_card_prob * second_card_same_suit_prob = (1 / 17 : ℚ) :=
by
  sorry

end probability_top_two_same_suit_l1460_146016


namespace max_cosine_value_l1460_146099

theorem max_cosine_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 1 ≥ Real.cos a :=
sorry

end max_cosine_value_l1460_146099


namespace sam_gave_joan_seashells_l1460_146050

variable (original_seashells : ℕ) (total_seashells : ℕ)

theorem sam_gave_joan_seashells (h1 : original_seashells = 70) (h2 : total_seashells = 97) :
  total_seashells - original_seashells = 27 :=
by
  sorry

end sam_gave_joan_seashells_l1460_146050


namespace parabola_problem_l1460_146022

theorem parabola_problem (a x1 x2 y1 y2 : ℝ)
  (h1 : y1^2 = a * x1)
  (h2 : y2^2 = a * x2)
  (h3 : x1 + x2 = 8)
  (h4 : (x2 - x1)^2 + (y2 - y1)^2 = 144) : 
  a = 8 := 
sorry

end parabola_problem_l1460_146022


namespace min_value_expression_l1460_146023

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem min_value_expression (a b : ℝ) (h1 : b > 0) (h2 : f a b 1 = 3) :
  ∃ x, x = (4 / (a - 1) + 1 / b) ∧ x = 9 / 2 :=
by
  sorry

end min_value_expression_l1460_146023


namespace total_gum_correct_l1460_146068

def num_cousins : ℕ := 4  -- Number of cousins
def gum_per_cousin : ℕ := 5  -- Pieces of gum per cousin

def total_gum : ℕ := num_cousins * gum_per_cousin  -- Total pieces of gum Kim needs

theorem total_gum_correct : total_gum = 20 :=
by sorry

end total_gum_correct_l1460_146068


namespace intersection_sets_l1460_146006

theorem intersection_sets :
  let M := {x : ℝ | (x + 3) * (x - 2) < 0 }
  let N := {x : ℝ | 1 ≤ x ∧ x ≤ 3 }
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_sets_l1460_146006


namespace least_pounds_of_sugar_l1460_146087

theorem least_pounds_of_sugar :
  ∃ s : ℝ, (∀ f : ℝ, (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s = 4) :=
by {
    use 4,
    sorry
}

end least_pounds_of_sugar_l1460_146087


namespace asymptotes_of_hyperbola_l1460_146018

-- Definition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 9 = 1

-- Definition of the equations of the asymptotes
def asymptote_eq (x y : ℝ) : Prop := y = (3/4)*x ∨ y = -(3/4)*x

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq x y :=
sorry

end asymptotes_of_hyperbola_l1460_146018


namespace tom_speed_from_A_to_B_l1460_146092

theorem tom_speed_from_A_to_B (D S : ℝ) (h1 : 2 * D = S * (3 * D / 36 - D / 20))
  (h2 : S * (3 * D / 36 - D / 20) = 3 * D / 36 ∨ 3 * D / 36 = S * (3 * D / 36 - D / 20))
  (h3 : D > 0) : S = 60 :=
by { sorry }

end tom_speed_from_A_to_B_l1460_146092


namespace system_solutions_a_l1460_146000

theorem system_solutions_a (x y z : ℝ) :
  (2 * x = (y + z) ^ 2) ∧ (2 * y = (z + x) ^ 2) ∧ (2 * z = (x + y) ^ 2) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_a_l1460_146000


namespace box_length_is_10_l1460_146084

theorem box_length_is_10
  (width height vol_cube num_cubes : ℕ)
  (h₀ : width = 13)
  (h₁ : height = 5)
  (h₂ : vol_cube = 5)
  (h₃ : num_cubes = 130) :
  (num_cubes * vol_cube) / (width * height) = 10 :=
by
  -- Proof steps will be filled here.
  sorry

end box_length_is_10_l1460_146084


namespace compute_z_pow_8_l1460_146093

noncomputable def z : ℂ := (1 - Real.sqrt 3 * Complex.I) / 2

theorem compute_z_pow_8 : z ^ 8 = -(1 + Real.sqrt 3 * Complex.I) / 2 :=
by
  sorry

end compute_z_pow_8_l1460_146093


namespace profit_per_meal_A_and_B_l1460_146062

theorem profit_per_meal_A_and_B (x y : ℝ) 
  (h1 : x + 2 * y = 35) 
  (h2 : 2 * x + 3 * y = 60) : 
  x = 15 ∧ y = 10 :=
sorry

end profit_per_meal_A_and_B_l1460_146062


namespace no_prime_divisible_by_77_l1460_146096

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l1460_146096


namespace weight_replacement_proof_l1460_146058

noncomputable def weight_of_replaced_person (increase_in_average_weight new_person_weight : ℝ) : ℝ :=
  new_person_weight - (5 * increase_in_average_weight)

theorem weight_replacement_proof (h1 : ∀ w : ℝ, increase_in_average_weight = 5.5) (h2 : new_person_weight = 95.5) :
  weight_of_replaced_person 5.5 95.5 = 68 := by
  sorry

end weight_replacement_proof_l1460_146058


namespace profit_percentage_l1460_146057

theorem profit_percentage (C S : ℝ) (h1 : C > 0) (h2 : S > 0)
  (h3 : S - 1.25 * C = 0.7023809523809523 * S) :
  ((S - C) / C) * 100 = 320 := by
sorry

end profit_percentage_l1460_146057


namespace mn_equals_neg16_l1460_146081

theorem mn_equals_neg16 (m n : ℤ) (h1 : m = -2) (h2 : |n| = 8) (h3 : m + n > 0) : m * n = -16 := by
  sorry

end mn_equals_neg16_l1460_146081


namespace triangle_RS_length_l1460_146008

theorem triangle_RS_length (PQ QR PS QS RS : ℝ)
  (h1 : PQ = 8) (h2 : QR = 8) (h3 : PS = 10) (h4 : QS = 5) :
  RS = 3.5 :=
by
  sorry

end triangle_RS_length_l1460_146008


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l1460_146012

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l1460_146012


namespace d_value_l1460_146072

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l1460_146072


namespace find_factor_l1460_146059

theorem find_factor (f : ℝ) : (120 * f - 138 = 102) → f = 2 :=
by
  sorry

end find_factor_l1460_146059


namespace angle_triple_supplementary_l1460_146066

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l1460_146066


namespace sequence_twice_square_l1460_146032

theorem sequence_twice_square (n : ℕ) (a : ℕ → ℕ) :
    (∀ i : ℕ, a i = 0) →
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
        ∀ i : ℕ, i % (2 * m) = 0 → 
            a i = if a i = 0 then 1 else 0) →
    (∀ i : ℕ, a i = 1 ↔ ∃ k : ℕ, i = 2 * k^2) :=
by
  sorry

end sequence_twice_square_l1460_146032


namespace largest_divisor_of_m_l1460_146056

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^3 = 847 * k) : ∃ d : ℕ, d = 77 ∧ ∀ x : ℕ, x > d → ¬ (x ∣ m) :=
sorry

end largest_divisor_of_m_l1460_146056


namespace math_problem_l1460_146041

theorem math_problem : ((3.6 * 0.3) / 0.6 = 1.8) :=
by
  sorry

end math_problem_l1460_146041


namespace decompose_five_eighths_l1460_146091

theorem decompose_five_eighths : 
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (5 : ℚ) / 8 = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) := 
by
  sorry

end decompose_five_eighths_l1460_146091


namespace remainder_of_3_pow_19_mod_10_l1460_146083

theorem remainder_of_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_mod_10_l1460_146083


namespace dave_hourly_wage_l1460_146038

theorem dave_hourly_wage :
  ∀ (hours_monday hours_tuesday total_money : ℝ),
  hours_monday = 6 → hours_tuesday = 2 → total_money = 48 →
  (total_money / (hours_monday + hours_tuesday) = 6) :=
by
  intros hours_monday hours_tuesday total_money h_monday h_tuesday h_money
  sorry

end dave_hourly_wage_l1460_146038


namespace profit_percentage_l1460_146073

-- Definitions and conditions
variable (SP : ℝ) (CP : ℝ)
variable (h : CP = 0.98 * SP)

-- Lean statement to prove the profit percentage is 2.04%
theorem profit_percentage (h : CP = 0.98 * SP) : (SP - CP) / CP * 100 = 2.04 := 
sorry

end profit_percentage_l1460_146073


namespace smallest_integer_to_multiply_y_to_make_perfect_square_l1460_146013

noncomputable def y : ℕ :=
  3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_multiply_y_to_make_perfect_square :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (k * y) = m^2) ∧ k = 3 := by
  sorry

end smallest_integer_to_multiply_y_to_make_perfect_square_l1460_146013


namespace y_value_l1460_146063

theorem y_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_eq1 : (1 / x) + (1 / y) = 3 / 2) (h_eq2 : x * y = 9) : y = 6 :=
sorry

end y_value_l1460_146063


namespace bobby_last_10_throws_successful_l1460_146049

theorem bobby_last_10_throws_successful :
    let initial_successful := 18 -- Bobby makes 18 successful throws out of his initial 30 throws.
    let total_throws := 30 + 10 -- Bobby makes a total of 40 throws.
    let final_successful := 0.64 * total_throws -- Bobby needs to make 64% of 40 throws to achieve a 64% success rate.
    let required_successful := 26 -- Adjusted to the nearest whole number.
    -- Bobby makes 8 successful throws in his last 10 attempts.
    required_successful - initial_successful = 8 := by
  sorry

end bobby_last_10_throws_successful_l1460_146049


namespace five_n_plus_3_composite_l1460_146014

theorem five_n_plus_3_composite (n : ℕ)
  (h1 : ∃ k : ℤ, 2 * n + 1 = k^2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m^2) :
  ¬ Prime (5 * n + 3) :=
by
  sorry

end five_n_plus_3_composite_l1460_146014


namespace lowest_test_score_dropped_l1460_146060

theorem lowest_test_score_dropped (A B C D : ℕ) 
  (h_avg_four : A + B + C + D = 140) 
  (h_avg_three : A + B + C = 120) : 
  D = 20 := 
by
  sorry

end lowest_test_score_dropped_l1460_146060


namespace ratio_neha_mother_age_12_years_ago_l1460_146002

variables (N : ℕ) (M : ℕ) (X : ℕ)

theorem ratio_neha_mother_age_12_years_ago 
  (hM : M = 60)
  (h_future : M + 12 = 2 * (N + 12)) :
  (12 : ℕ) * (M - 12) = (48 : ℕ) * (N - 12) :=
by
  sorry

end ratio_neha_mother_age_12_years_ago_l1460_146002


namespace triangle_angle_C_l1460_146028

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l1460_146028


namespace minimum_value_of_a_l1460_146042

variable (a x y : ℝ)

-- Condition
def condition (x y : ℝ) (a : ℝ) : Prop := 
  (x + y) * ((1/x) + (a/y)) ≥ 9

-- Main statement
theorem minimum_value_of_a : (∀ x > 0, ∀ y > 0, condition x y a) → a ≥ 4 :=
sorry

end minimum_value_of_a_l1460_146042
