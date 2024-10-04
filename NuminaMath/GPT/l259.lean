import Mathlib

namespace describe_difference_of_squares_l259_259235

def description_of_a_squared_minus_b_squared : Prop :=
  ∃ (a b : ℝ), (a^2 - b^2) = (a^2 - b^2)

theorem describe_difference_of_squares :
  description_of_a_squared_minus_b_squared :=
by sorry

end describe_difference_of_squares_l259_259235


namespace function_properties_y_l259_259258

open Real Set Function

noncomputable def domain_y : Set ℝ := {x : ℝ | ∃ k : ℤ, x ≠ (k * π / 5 + π / 20)}

noncomputable def monotonic_intervals_y : Set (Set ℝ) := 
  {I : Set ℝ | ∃ k : ℤ, I = Icc (k * π / 5 - 3 * π / 20) (k * π / 5 + π / 20)}

noncomputable def symmetry_center_y : Set (ℝ × ℝ) := 
  {(x, 0) | ∃ k : ℤ, x = k * π / 10 - π / 20}

theorem function_properties_y (f : ℝ → ℝ) (hf : f = (λ x, 1 / 2 * tan (5 * x + π / 4))) :
  (domain f = domain_y ∧ 
  ∀ k : ℤ, {x : ℝ | k * π / 5 - 3 * π / 20 < x ∧ x < k * π / 5 + π / 20} ⊆ monotonic_intervals_y ∧ 
  ∀ k : ℤ, (k * π / 10 - π / 20, 0) ∈ symmetry_center_y) :=
sorry

end function_properties_y_l259_259258


namespace product_of_ages_l259_259870

theorem product_of_ages (O Y : ℕ) (h1 : O - Y = 12) (h2 : O + Y = (O - Y) + 40) : O * Y = 640 := by
  sorry

end product_of_ages_l259_259870


namespace season_duration_l259_259899

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 323) (h2 : games_per_month = 19) :
  (total_games / games_per_month) = 17 :=
by
  sorry

end season_duration_l259_259899


namespace correct_proposition_l259_259974

theorem correct_proposition : 
  (¬ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 2 * x_0) ↔ (∀ x : ℝ, x^2 + 1 > 2 * x) := 
sorry

end correct_proposition_l259_259974


namespace janice_overtime_earnings_l259_259803

-- Definitions based on conditions
def days_worked : ℕ := 5
def daily_earnings : ℕ := 30
def overtime_shifts : ℕ := 3
def total_earnings : ℕ := 195

-- The theorem we want to prove
theorem janice_overtime_earnings :
  let regular_earnings := days_worked * daily_earnings,
      overtime_earnings := total_earnings - regular_earnings,
      earnings_per_overtime_shift := overtime_earnings / overtime_shifts
  in earnings_per_overtime_shift = 15 :=
by
  sorry

end janice_overtime_earnings_l259_259803


namespace vectors_coplanar_l259_259211

def a : ℝ × ℝ × ℝ := (2, 3, 2)
def b : ℝ × ℝ × ℝ := (4, 7, 5)
def c : ℝ × ℝ × ℝ := (2, 0, -1)

-- Define the determinant function for 3x3 matrix
def determinant_3x3 (a b c : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * (b.2 * c.3 - b.3 * c.2) -
  a.2 * (b.1 * c.3 - b.3 * c.1) +
  a.3 * (b.1 * c.2 - b.2 * c.1)

-- Assertion that the vectors are coplanar if the determinant is zero
theorem vectors_coplanar : determinant_3x3 a b c = 0 :=
  by
    -- Calculation steps can be inserted here if needed
    sorry

end vectors_coplanar_l259_259211


namespace distinct_five_topping_pizzas_l259_259955

def number_of_toppings := 8
def number_of_toppings_per_pizza := 5

theorem distinct_five_topping_pizzas :
  ∃ (n k : ℕ), n = number_of_toppings ∧ k = number_of_toppings_per_pizza ∧ nat.choose n k = 56 :=
by
  use number_of_toppings
  use number_of_toppings_per_pizza
  simp [number_of_toppings, number_of_toppings_per_pizza]
  sorry

end distinct_five_topping_pizzas_l259_259955


namespace ellipse_equation_and_line_intersection_l259_259692

theorem ellipse_equation_and_line_intersection 
  (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,2))
  (hB : B = (Real.sqrt 2, Real.sqrt 2))
  (hF : ∃ c : ℝ, F = (c, 0) ∧ Real.sqrt ((c - Real.sqrt 2)^2 + (0 - Real.sqrt 2)^2) = 2)
  (h_center : ∀ (x y : ℝ), (x, y) ↔ (x-center.x)^2 / a^2 + (y-center.y)^2 / b^2 = 1)
  (h_axes : center = (0, 0))
  (h_vertex : A = (0, b)) : 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 12 + y^2 / 4 = 1) ∧ 
  (line_eq : ∃ k : ℝ, k ≠ 0 ∧ (∀ x y : ℝ, y = k*x - 3 ∧ (Real.sqrt 6 / 3 = |k|))) := 
begin
  sorry
end

end ellipse_equation_and_line_intersection_l259_259692


namespace sum_of_positive_integers_for_quadratic_l259_259095

theorem sum_of_positive_integers_for_quadratic :
  (∑ k in {k : ℕ | (∃ α β : ℤ, α * β = 18 ∧ α + β = k)}.to_finset) = 39 :=
by
  sorry

end sum_of_positive_integers_for_quadratic_l259_259095


namespace complex_number_equality_l259_259750

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h1 : is_imaginary_unit i) (h2 : (a + 4 * i) * i = b + i) : a + b = -3 :=
sorry

end complex_number_equality_l259_259750


namespace greatest_multiple_of_4_l259_259626

/-- 
Given x is a positive multiple of 4 and x^3 < 2000, 
prove that x is at most 12 and 
x = 12 is the greatest value that satisfies these conditions. 
-/
theorem greatest_multiple_of_4 (x : ℕ) (hx1 : x % 4 = 0) (hx2 : x^3 < 2000) : x ≤ 12 ∧ x = 12 :=
by
  sorry

end greatest_multiple_of_4_l259_259626


namespace smallest_n_for_g_gt_21_l259_259033

def g (n : ℕ) : ℕ := sorry

theorem smallest_n_for_g_gt_21 (h : ∃ k, n = 21 * k) (hg : ∀ n, g n > 21 ↔ n = 483) :
  ∃ n, n = 483 :=
by
  use 483
  sorry

end smallest_n_for_g_gt_21_l259_259033


namespace cos_arith_prog_impossible_l259_259552

theorem cos_arith_prog_impossible
  (x y z : ℝ)
  (sin_arith_prog : 2 * Real.sin y = Real.sin x + Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by
  sorry

end cos_arith_prog_impossible_l259_259552


namespace number_of_arrangements_l259_259066

-- Definitions for the conditions
def balls := {A, B, C, D}
def boxes := {1, 2, 3}

-- Condition that each box must contain at least one ball
def boxes_nonempty (dist : balls → boxes) : Prop :=
  ∀ b : boxes, ∃ a : balls, dist a = b

-- Condition that balls A and B cannot be in the same box
def AB_not_in_same_box (dist : balls → boxes) : Prop :=
  dist A ≠ dist B

-- Main statement to prove
theorem number_of_arrangements :
  ∃ dist : balls → boxes, boxes_nonempty dist ∧ AB_not_in_same_box dist ∧ (count_distinct_arrangements dist = 18) :=
sorry

end number_of_arrangements_l259_259066


namespace polynomial_derivative_l259_259147

-- Definitions of the polynomials and their properties
variables {R : Type*} [CommRing R]

noncomputable def p (x : R) : R := sorry
noncomputable def q (x : R) : R := sorry

-- Setting up hypotheses
variables (n m : ℕ) (c : R)
hypothesis h1 : degree (p x) = n
hypothesis h2 : degree (q x) = m
hypothesis h3 : leadingCoeff (p x) = c
hypothesis h4 : leadingCoeff (q x) = c
hypothesis h5 : (p x) ^ 2 = (x ^ 2 - 1) * (q x) ^ 2 + 1

-- The theorem statement we aim to prove
theorem polynomial_derivative : p' x = n * q x :=
by sorry

end polynomial_derivative_l259_259147


namespace find_exponent_l259_259353

theorem find_exponent (y : ℕ) (b : ℕ) (h_b : b = 2)
  (h : 1 / 8 * 2 ^ 40 = b ^ y) : y = 37 :=
by
  sorry

end find_exponent_l259_259353


namespace overall_mean_daily_profit_l259_259192

theorem overall_mean_daily_profit 
  (mean_first_15_days : ℕ)
  (mean_last_15_days : ℕ)
  (days_in_month : ℕ)
  (days_each_period : ℕ)
  (total_profit_first_15_days : ℕ)
  (total_profit_last_15_days : ℕ)
  (total_profit_month : ℕ)
  (overall_mean : ℕ) : 
  (mean_first_15_days = 255) →
  (mean_last_15_days = 445) →
  (days_in_month = 30) →
  (days_each_period = 15) →
  (total_profit_first_15_days = days_each_period * mean_first_15_days) →
  (total_profit_last_15_days = days_each_period * mean_last_15_days) →
  (total_profit_month = total_profit_first_15_days + total_profit_last_15_days) →
  (overall_mean = total_profit_month / days_in_month) →
  (overall_mean = 350) := 
begin
  sorry
end

end overall_mean_daily_profit_l259_259192


namespace ratio_sum_odd_even_divisors_l259_259024

theorem ratio_sum_odd_even_divisors :
  let M := 24 * 36 * 49 * 125 in
  let b := (1 + 3 + 9 + 27) * (1 + 7 + 49) * (1 + 5 + 25 + 125) in
  let T := (1 + 2 + 4 + 8 + 16 + 32) * b in
  let even_divisors_sum := T - b in
  b / even_divisors_sum = 1 / 62 :=
by
  let M := 24 * 36 * 49 * 125
  let b := (1 + 3 + 9 + 27) * (1 + 7 + 49) * (1 + 5 + 25 + 125)
  let T := (1 + 2 + 4 + 8 + 16 + 32) * b
  let even_divisors_sum := T - b
  have h1 : T = 63 * b := by sorry
  have h2 : even_divisors_sum = 62 * b := by sorry
  have h3 : b / even_divisors_sum = b / (62 * b) := by sorry
  have h4 : b / (62 * b) = 1 / 62 := by
    rw [div_eq_div_iff]
    ring
    exact zero_ne' _
  rw [h4]
  refl

end ratio_sum_odd_even_divisors_l259_259024


namespace compare_negatives_l259_259911

-- Define the given numbers
def negative_number1 : ℚ := -7/2
def negative_number2 : ℚ := -7/3
def negative_number3 : ℚ := -3/4
def negative_number4 : ℚ := -4/5

-- Define the absolute value function for rational numbers
def abs (x : ℚ) : ℚ := if x >= 0 then x else -x

-- State the comparison using the absolute values
theorem compare_negatives :
  abs negative_number1 > abs negative_number2 → 
  negative_number1 < negative_number2 
  ∧
  abs negative_number3 < abs negative_number4 → 
  negative_number3 > negative_number4
:=
by {
  sorry
}

end compare_negatives_l259_259911


namespace triangle_smallest_angle_l259_259512

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l259_259512


namespace triangle_ratio_eq_diameter_of_circumscribed_circle_l259_259464

theorem triangle_ratio_eq_diameter_of_circumscribed_circle 
  (a b c h_a h_b h_c R : ℝ) 
  (cond1 : h_a = b * Math.sin c)
  (cond2 : h_b = c * Math.sin a)
  (cond3 : h_c = a * Math.sin b)
  (circumradius_def : a = 2 * R * Math.sin a ∧ b = 2 * R * Math.sin b ∧ c = 2 * R * Math.sin c) :
  (ab + bc + ac) / (h_a + h_b + h_c) = 2 * R := 
by
  sorry

end triangle_ratio_eq_diameter_of_circumscribed_circle_l259_259464


namespace area_of_annulus_l259_259582

-- Definitions based on conditions
def smaller_circle_circumference : ℝ := 18 * Real.pi
def larger_circle_radius_difference : ℝ := 10

-- Theorem statement: Given the conditions, prove the area of the annulus is 280 * π
theorem area_of_annulus : 
  ∃ (r R : ℝ), (2 * Real.pi * r = smaller_circle_circumference) ∧ 
               (R = r + larger_circle_radius_difference) ∧ 
               (Real.pi * R^2 - Real.pi * r^2 = 280 * Real.pi) := 
sorry

end area_of_annulus_l259_259582


namespace find_g_30_l259_259876

def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) = x * g y

axiom g_one : g 1 = 10

theorem find_g_30 : g 30 = 300 := by
  sorry

end find_g_30_l259_259876


namespace allison_roll_higher_probability_l259_259614

/-- 
Allison and Noah each have a 6-sided cube. Brian has a modified 6-sided cube.
Allison's cube’s faces all show the number 5. Noah’s cube’s faces are configured equally with three faces showing a 3 and the other three faces showing a 7. 
Brian's modified cube has the numbers 1, 1, 2, 2, 5, and 6. 
Prove that the probability that Allison's roll is higher than both Brian's and Noah's is 1/3.
-/
theorem allison_roll_higher_probability :
  let allison_roll := 5
  let noah_faces := [3, 3, 3, 7, 7, 7]
  let brian_faces := [1, 1, 2, 2, 5, 6]
  let prob_noah_less_than_5 := (3 : ℚ) / 6
  let prob_brian_less_than_5 := (4 : ℚ) / 6
  let prob_combined := prob_noah_less_than_5 * prob_brian_less_than_5
  prob_combined = 1 / 3 :=
by
  let allison_roll := 5
  let noah_faces := [3, 3, 3, 7, 7, 7]
  let brian_faces := [1, 1, 2, 2, 5, 6]
  let prob_noah_less_than_5 := (3 : ℚ) / 6
  let prob_brian_less_than_5 := (4 : ℚ) / 6
  let prob_combined := prob_noah_less_than_5 * prob_brian_less_than_5
  show prob_combined = 1 / 3 from sorry

end allison_roll_higher_probability_l259_259614


namespace vending_machine_change_sum_l259_259863

def solution1 (amount : ℕ) : Bool := 
  (amount < 100) ∧ ((amount % 5 = 4) ∧ (amount % 10 = 6))

theorem vending_machine_change_sum : 
  (∑ a in (Finset.filter solution1 (Finset.range 100)), a) = 506 :=
by
  sorry

end vending_machine_change_sum_l259_259863


namespace prove_sphere_projection_l259_259090

noncomputable def sphere_tangent_to_plane_projection (b : ℝ) (a : ℚ) (p q : ℕ) (hpq : a = (p : ℚ) / (q : ℚ)) : Prop :=
  let P : ℝ × ℝ × ℝ := (0, b, a)
  let conic_section := λ x : ℝ, x^2
  ∃ z : ℝ, z > 0 ∧ (∀ x : ℝ, conic_section x = x^2) ∧ (P.2.2 = a) ∧ (a = 1 / 2) ∧ (p + q = 3)

theorem prove_sphere_projection : ∃ b a p q, (sphere_tangent_to_plane_projection b a p q) :=
  sorry

end prove_sphere_projection_l259_259090


namespace polynomial_roots_sum_to_ten_and_B_is_neg88_l259_259200

theorem polynomial_roots_sum_to_ten_and_B_is_neg88
  (roots : Fin 6 → ℕ)
  (h_sum : ∑ i, roots i = 10)
  (h_all_pos : ∀ i, 0 < roots i)
  (h_poly : ∑ i in Finset.range 7, (-1)^i * (roots.prod (λ r, r^(6 - i))) = λ A B C D, (0^6 - 10 * 0^5 + A * 0^4 + B * 0^3 + C * 0^2 + D * 0 + 16)) :
  B = -88 := 
sorry

end polynomial_roots_sum_to_ten_and_B_is_neg88_l259_259200


namespace dirichlet_function_properties_l259_259496

def D (x : ℝ) : ℝ :=
  if rat_cast x then 1 else 0

theorem dirichlet_function_properties :
    (∃ x y : ℝ, D(x * y) = D(x) + D(y)) ∧ -- Proposition B
    (∀ x : ℝ, D(D(x)) = D(D (-x))) ∧ -- Proposition C
    (∀ a : ℚ, ∀ x : ℝ, D(a + x) = D(a - x)) := -- Proposition D
by
  sorry

end dirichlet_function_properties_l259_259496


namespace ratio_perimeters_is_sqrt3_l259_259600

theorem ratio_perimeters_is_sqrt3 (s_t s_h : ℝ)
  (h1 : (s_t^2 * real.sqrt 3) / 4 = 2 * (3 * s_h^2 * real.sqrt 3) / 2) :
  (3 * s_t) / (6 * s_h) = real.sqrt 3 :=
by
  sorry

end ratio_perimeters_is_sqrt3_l259_259600


namespace numberOfPlanes_through_P_l259_259864

noncomputable def numberOfPlanesFormingAngle (P : Point) (alpha beta : Plane) : Nat :=
  have angle_between_planes : Measure.angle alpha beta = 40 := by sorry
  ∃ (X : Plane), Measure.angle X alpha = 70 ∧ Measure.angle X beta = 70

theorem numberOfPlanes_through_P
  (P : Point) (alpha beta : Plane)
  (h_angle : Measure.angle alpha beta = 40) :
  numberOfPlanesFormingAngle P alpha beta = 4 := sorry

end numberOfPlanes_through_P_l259_259864


namespace maximum_distance_l259_259609

variable (highway_mpg : ℝ)
variable (gallons : ℝ)

theorem maximum_distance (highway_mpg = 12.2) (gallons = 24) : 
  ∃ distance, distance = 12.2 * 24 := by
  sorry

end maximum_distance_l259_259609


namespace nondegenerate_triangles_count_l259_259660

theorem nondegenerate_triangles_count :
  let points := {(s, t) | 0 ≤ s ∧ s ≤ 4 ∧ 0 ≤ t ∧ t ≤ 4} in
  let total_points := 5 * 5 in
  let total_triangles := Nat.choose total_points 3 in
  let horizontal_degenerate := 5 * Nat.choose 5 3 in
  let vertical_degenerate := 5 * Nat.choose 5 3 in
  let diagonal_degenerate := 2 * (Nat.choose 5 3 + Nat.choose 4 3 + Nat.choose 3 3) in
  let total_degenerate := horizontal_degenerate + vertical_degenerate + diagonal_degenerate in
  total_triangles - total_degenerate = 2170 :=
by
  sorry

end nondegenerate_triangles_count_l259_259660


namespace ellipse_with_foci_on_x_axis_l259_259354

theorem ellipse_with_foci_on_x_axis {a : ℝ} (h1 : a - 5 > 0) (h2 : 2 > 0) (h3 : a - 5 > 2) :
  a > 7 :=
by
  sorry

end ellipse_with_foci_on_x_axis_l259_259354


namespace Kath_payment_l259_259773

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l259_259773


namespace correct_statements_l259_259145

noncomputable def lineB (m : ℝ) : AffineMap ℝ ℝ := 
  { toFun := λ p, (p.1 * m - p.2 + 1 - m) }

noncomputable def lineC : (ℝ × ℝ) := 
  (0, -2)

noncomputable def lineD (c : ℝ) : AffineMap ℝ ℝ := 
  { toFun := λ p, (p.1 + p.2 - c) }

theorem correct_statements :
  (∀ (m : ℝ), lineB m (1, 1) = 0) ∧
  lineC = (0, -2) ∧
  (lineD 4 (1, 3) = 0) := sorry

end correct_statements_l259_259145


namespace problem_complex_number_l259_259570

theorem problem_complex_number (h : Complex.i^2 = -1) : (5 * Complex.i) / (1 - 2 * Complex.i) = -2 + Complex.i :=
by
  sorry

end problem_complex_number_l259_259570


namespace sqrt_domain_l259_259377

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259377


namespace cost_of_senior_ticket_l259_259591

theorem cost_of_senior_ticket (x : ℤ) (total_tickets : ℤ) (cost_regular_ticket : ℤ) (total_sales : ℤ) (senior_tickets_sold : ℤ) (regular_tickets_sold : ℤ) :
  total_tickets = 65 →
  cost_regular_ticket = 15 →
  total_sales = 855 →
  senior_tickets_sold = 24 →
  regular_tickets_sold = total_tickets - senior_tickets_sold →
  total_sales = senior_tickets_sold * x + regular_tickets_sold * cost_regular_ticket →
  x = 10 :=
by
  sorry

end cost_of_senior_ticket_l259_259591


namespace part1_part2_part3_l259_259244

open Classical

-- Define the relationship y = kx + b with given points
variables (x y : ℤ) (k b : ℤ)

theorem part1 (h1: y = 9 * k + b) (h2: y = 11 * k + b) : y = -5 * x + 150 := 
sorry

-- Define the profit function and conditions given (8 ≤ x ≤ 15)
noncomputable def profit (x y : ℤ) := (x - 8) * y

variables {profit : ℤ → ℤ}
axiom h3 : ∀ x y ∈ (set.range profit), 8 ≤ x ∧ x ≤ 15

theorem part2 (x : ℤ) (hx : 8 ≤ x ∧ x ≤ 15) (hprofit : profit x (-5 * x + 150) = 425) : x = 13 :=
sorry

-- Define maximum profit calculations
noncomputable def max_profit (x : ℤ) := -5 * (x - 19)^2 + 605

theorem part3 (x : ℤ) (hx : 8 ≤ x ∧ x ≤ 15) (y := max_profit x) : 
  (hx : x = 15 ∧ y = 525 ∧ ∀ x', 8 ≤ x' ∧ x' ≤ 15 → (-5 * (x' - 19)^2 + 605 ≤ 525)) :=
sorry

end part1_part2_part3_l259_259244


namespace BG_eq_4_DF_l259_259799

-- Definitions of the points, lines, and their relationships
variables {A B C D E F G : Type} [AddZeroClass A]
variables (b c : real) (AB AC : real) (altitude AD : Line) (angleBisector BE : Line)
variables (perpendicular1 EF : Line) (perpendicular2 EG : Line) 
variables (BC : Line)

-- Definitions of the conditions
def condition1 : Prop := AB = AC
def condition2 : Prop := altitude AD F
def condition3 : Prop := angleBisector BE B
def condition4 : Prop := perpendicular1 EF BC
def condition5 : Prop := perpendicular2 EG BE

-- Theorem to be proved
theorem BG_eq_4_DF 
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3)
    (h4 : condition4)
    (h5 : condition5) : 
    BG = 4 * DF :=
sorry

end BG_eq_4_DF_l259_259799


namespace total_wheels_in_garage_l259_259515

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end total_wheels_in_garage_l259_259515


namespace restaurant_chili_paste_needs_l259_259961

theorem restaurant_chili_paste_needs:
  let large_can_volume := 25
  let small_can_volume := 15
  let large_cans_required := 45
  let total_volume := large_cans_required * large_can_volume
  let small_cans_needed := total_volume / small_can_volume
  small_cans_needed - large_cans_required = 30 :=
by
  sorry

end restaurant_chili_paste_needs_l259_259961


namespace solution_set_of_f_inequality_l259_259289

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_deriv : ∀ x, f' x < f x)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_initial : f 0 = Real.exp 4)

theorem solution_set_of_f_inequality :
  {x : ℝ | f x < Real.exp x} = {x : ℝ | x > 4} := 
sorry

end solution_set_of_f_inequality_l259_259289


namespace zero_in_interval_l259_259261

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 2

theorem zero_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
by
  have h1 : f 2 > 0 := by sorry
  have h2 : f 3 < 0 := by sorry
  exact IntermediateValueTheorem _ _ _ sorry -- using the Bolzano theorem

end zero_in_interval_l259_259261


namespace platform_length_l259_259576

theorem platform_length (train_length : ℝ) (time_cross_pole : ℝ) (time_cross_platform : ℝ) (speed : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_cross_pole = 18) 
  (h3 : time_cross_platform = 54)
  (h4 : speed = train_length / time_cross_pole) :
  train_length + (speed * time_cross_platform) - train_length = 600 := 
by
  sorry

end platform_length_l259_259576


namespace part_1_part_2_part_3_l259_259246

noncomputable def cost_per_item : ℕ := 8
def selling_price_range (x : ℕ) : Prop := 8 ≤ x ∧ x ≤ 15
def data_points : List (ℕ × ℕ) := [(9, 105), (11, 95), (13, 85)]

def y (x : ℕ) : ℕ := -5 * x + 150
def profit (x : ℕ) : ℤ := (x - cost_per_item) * (y x - cost_per_item)
def profit_formula (x : ℕ) : ℤ := -5 * (x - 19)^2 + 605

theorem part_1 : ∀ x, (x, y x) ∈ data_points :=
  by sorry

theorem part_2 : profit 13 = 425 :=
  by sorry

theorem part_3 : 
  (∀ x, x ∈ {₁ ... ₅} → profit x ≤ profit 15) ∧ profit 15 = 525 :=
  by sorry

end part_1_part_2_part_3_l259_259246


namespace find_sum_of_digits_l259_259008

theorem find_sum_of_digits (a b c d : ℕ) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : a = 1)
  (h3 : 1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100)
  : a + b + c + d = 2 := 
sorry

end find_sum_of_digits_l259_259008


namespace exercise_l259_259519

variables (V : Type) [Fintype V] [DecidableEq V]

def is_pentagon (s : Finset V) := (Fintype.card V = 5) ∧ (s.card = 5)

def is_triangle (t : Finset V) := (t.card = 3) 

def is_acute (t : Finset V) : Prop := sorry -- Definition of acute triangle

def number_of_triangles (V : Type) [Fintype V] : ℕ := 
  Fintype.card { t : Finset V // is_triangle t }

def number_of_acute_triangles (V : Type) [Fintype V] : ℕ :=
  Fintype.card { t : Finset V // is_triangle t ∧ is_acute t }

def probability_of_acute_triangle (V : Type) [Fintype V] : ℚ :=
  number_of_acute_triangles V / number_of_triangles V

theorem exercise (V : Type) [Fintype V] [DecidableEq V] (s : Finset V) :
  is_pentagon s →
  10 * probability_of_acute_triangle V = 5 :=
begin
  intros h,
  -- convert conditions for lean proof
  have h1 : number_of_triangles V = 10, 
  { sorry, },
  have h2 : number_of_acute_triangles V = 5, 
  { sorry, },
  have p_eq : probability_of_acute_triangle V = 1/2,
  { simp [probability_of_acute_triangle, h2, h1], },
  simp [p_eq],
end

end exercise_l259_259519


namespace cube_root_59319_cube_root_103823_l259_259635

theorem cube_root_59319 : ∃ x : ℕ, x ^ 3 = 59319 ∧ x = 39 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

theorem cube_root_103823 : ∃ x : ℕ, x ^ 3 = 103823 ∧ x = 47 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

end cube_root_59319_cube_root_103823_l259_259635


namespace ratio_d_s_proof_l259_259196

noncomputable def ratio_d_s (n : ℕ) (s d : ℝ) : ℝ :=
  d / s

theorem ratio_d_s_proof : ∀ (n : ℕ) (s d : ℝ), 
  (n = 30) → 
  ((n ^ 2 * s ^ 2) / (n * s + 2 * n * d) ^ 2 = 0.81) → 
  ratio_d_s n s d = 1 / 18 :=
by
  intros n s d h_n h_area
  sorry

end ratio_d_s_proof_l259_259196


namespace problem_solution_l259_259342

-- Define the set X and the constraints on A
def X : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5 * 10^6}

-- Define the properties of subsets A
def valid_subset (A : Set ℕ) : Prop :=
  (A ⊆ X) ∧ (A.card = 2015) ∧ ∀ (B : Finset ℕ), (B ⊆ A) ∧ (B ≠ ∅) → (∑ i in B, i) % 2016 ≠ 0

-- Define the function to count the number of valid subsets
noncomputable def count_valid_subsets : ℕ :=
  92 * (Nat.choose 2480 2015) + 484 * (Nat.choose 2488 2015)

-- The main theorem statement
theorem problem_solution :
  ∃ (A : Finset (Finset ℕ)), (A.card = count_valid_subsets) ∧ ∀ x, x ∈ A ↔ valid_subset x :=
sorry

end problem_solution_l259_259342


namespace part_1_part_2_part_3_l259_259245

noncomputable def cost_per_item : ℕ := 8
def selling_price_range (x : ℕ) : Prop := 8 ≤ x ∧ x ≤ 15
def data_points : List (ℕ × ℕ) := [(9, 105), (11, 95), (13, 85)]

def y (x : ℕ) : ℕ := -5 * x + 150
def profit (x : ℕ) : ℤ := (x - cost_per_item) * (y x - cost_per_item)
def profit_formula (x : ℕ) : ℤ := -5 * (x - 19)^2 + 605

theorem part_1 : ∀ x, (x, y x) ∈ data_points :=
  by sorry

theorem part_2 : profit 13 = 425 :=
  by sorry

theorem part_3 : 
  (∀ x, x ∈ {₁ ... ₅} → profit x ≤ profit 15) ∧ profit 15 = 525 :=
  by sorry

end part_1_part_2_part_3_l259_259245


namespace probability_different_colors_proof_l259_259896

noncomputable def probability_different_colors (red green : ℕ) (total_chips : ℕ) : ℚ :=
  (red * green / (total_chips * (total_chips - 1) / 2))

theorem probability_different_colors_proof :
  probability_different_colors 7 5 12 = 35 / 66 :=
by
  -- Definition of the number of red chips
  let red := 7 : ℕ
  -- Definition of the number of green chips
  let green := 5 : ℕ
  -- Total number of chips
  let total_chips := 12 : ℕ
  -- RHS as fraction
  let val := (35 / 66) : ℚ
  -- Probability calculation
  have prob := probability_different_colors red green total_chips
  -- Comparison
  show prob = val
  sorry

end probability_different_colors_proof_l259_259896


namespace find_x_l259_259695

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 4

theorem find_x (x : ℝ) : 2 * f(x) - 21 = f(x - d) → x = 8 :=
by
  sorry

end find_x_l259_259695


namespace election_winners_l259_259367

-- Definitions for groups and their sizes
def group1_votes : ℕ := 33000
def group2_votes : ℕ := 18000
def group3_votes : ℕ := 12000
def group4_votes : ℕ := 37000

-- Definition of total voters
def total_voters : ℕ := group1_votes + group2_votes + group3_votes + group4_votes

-- Each candidate
inductive Candidate
| Montoran
| AjudaPinto
| VidameOfOussel

open Candidate

-- One-round voting system winner
def one_round_winner : Candidate :=
  if group4_votes > group1_votes && group4_votes > group2_votes then VidameOfOussel
  else if group1_votes > group2_votes then Montoran
  else AjudaPinto

-- Two-round voting system winner
def two_round_winner : Candidate :=
  let first_round_winner :=
    if group1_votes > group2_votes && group1_votes > group4_votes then Montoran
    else if group2_votes + group3_votes > group4_votes then AjudaPinto
    else VidameOfOussel in
  if first_round_winner = Montoran then
    if group3_votes > group4_votes then Montoran else VidameOfOussel
  else first_round_winner

-- Three-round voting system winner
def three_round_winner : Candidate :=
  let m_vs_a_winner := if group1_votes > group2_votes + group3_votes + group4_votes then Montoran else AjudaPinto in
  let a_vs_v_winner := if group4_votes > group2_votes + group3_votes then VidameOfOussel else AjudaPinto in
  if m_vs_a_winner = AjudaPinto && a_vs_v_winner = VidameOfOussel then AjudaPinto
  else VidameOfOussel

-- Theorem to state the winners
theorem election_winners : 
  (one_round_winner = VidameOfOussel) ∧ (two_round_winner = Montoran) ∧ (three_round_winner = AjudaPinto) :=
by
  -- The proof is left as sorry as per instructions
  sorry

end election_winners_l259_259367


namespace ricardo_coin_difference_l259_259851

theorem ricardo_coin_difference (p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ 2299) :
  (11500 - 4 * p) - (11500 - 4 * (2300 - p)) = 9192 :=
by
  sorry

end ricardo_coin_difference_l259_259851


namespace chord_length_squared_l259_259636

theorem chord_length_squared
  (r5 r10 r15 : ℝ) 
  (externally_tangent : r5 = 5 ∧ r10 = 10)
  (internally_tangent : r15 = 15)
  (common_external_tangent : r15 - r10 - r5 = 0) :
  ∃ PQ_squared : ℝ, PQ_squared = 622.44 :=
by
  sorry

end chord_length_squared_l259_259636


namespace price_per_apple_l259_259213

variable (cost_bushel : ℕ) 
variable (apples_per_bushel : ℕ)
variable (profit_per_100_apples : ℕ)

theorem price_per_apple {cost_bushel apples_per_bushel profit_per_100_apples : ℕ} 
(h1 : cost_bushel = 12)
(h2 : apples_per_bushel = 48)
(h3 : profit_per_100_apples = 15) :
  (cost_bushel / apples_per_bushel * 100 + profit_per_100_apples) / 100 = 0.40 := 
by 
  sorry

end price_per_apple_l259_259213


namespace ellipse_eccentricity_l259_259625

def ellipse_eqn (a b x y : ℝ) : Prop :=
  (a > b) ∧ (0 < b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def circle_eqn (a x y : ℝ) : Prop :=
  x^2 + y^2 = a^2

def point_B (a : ℝ) : (ℝ × ℝ) :=
  (0, a)

def point_A (a : ℝ) : (ℝ × ℝ) :=
  ((a * Real.sqrt 3 / 2), (a / 2))

def slope_AB (a : ℝ) : ℝ :=
  - (Real.sqrt 3 / 3)

def line_eqn (a x : ℝ) : ℝ :=
  - (Real.sqrt 3 / 3) * x + a

def is_tangent (a b x y : ℝ) : Prop :=
  let line_y := line_eqn a x
  in x^2 / a^2 + line_y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity (a b: ℝ) (h1 : ellipse_eqn a b 0 a) (h2 : circle_eqn a (a * Real.sqrt 3 / 2) (a / 2)) 
(h3 : is_tangent a b (a * Real.sqrt 3 / 2) (a / 2)) (h4: ∃ B A, (B = (point_B a)) ∧ (A = (point_A a)) ∧ (\angle AOB = 60)):
  eccentricity a b = Real.sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l259_259625


namespace generatrix_length_l259_259486

section cone_problem

variables {r h l : ℝ} (V : ℝ)
-- Given conditions
def sector_area_twice_base_area (r l : ℝ) : Prop := (π * r * l = 2 * π * r^2)

def cone_volume (r h : ℝ) (V : ℝ) : Prop := (1 / 3) * π * r^2 * h = V

-- Theorem to prove
theorem generatrix_length (A_twice : sector_area_twice_base_area r l) (vol : cone_volume r (sqrt(3) * r) 9 * sqrt 3 * π) : l = 6 :=
begin
  sorry
end

end cone_problem

end generatrix_length_l259_259486


namespace triangle_circle_property_l259_259000

theorem triangle_circle_property 
  (A B C : Type)
  [LinearOrderedField A] [LinearOrderedField B] 
  (m n : A)
  (L : LinearOrderedField (B → A))
  (hypotenuse_length : m)
  (diameter_length : n)
  (right_angle_triangle : ∀ a b c : B, a^2 + b^2 = c^2)
  (O_midpoint : B → B → B)
  (circle_center : O_midpoint B B = (B → A))
  (circle_diameter : (B → A → A) = n)
  (n_lt_m_div_2 : n < m / 2)
  (BC_intersects : ∀ P Q : B, circle_center A (B → A → A)) : 
  |AP|^2 + |AQ|^2 + |PQ|^2 = m^2 / 2 + 6 * n^2 :=
sorry

end triangle_circle_property_l259_259000


namespace number_of_outfits_l259_259366

theorem number_of_outfits (red_shirts blue_shirts pairs_of_pants green_hats red_hats : ℕ) 
  (distinct_clothing : ∀ (s : ℕ),  s ∈ {red_shirts, blue_shirts, pairs_of_pants, green_hats, red_hats}):
  red_shirts = 7 → blue_shirts = 7 → pairs_of_pants = 10 → green_hats = 9 → red_hats = 9 →
  (red_shirts * pairs_of_pants * green_hats + blue_shirts * pairs_of_pants * red_hats = 1260) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end number_of_outfits_l259_259366


namespace range_of_sqrt_x_plus_3_l259_259382

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259382


namespace overtime_hourly_rate_l259_259209

theorem overtime_hourly_rate
  (hourly_rate_first_40_hours: ℝ)
  (hours_first_40: ℝ)
  (gross_pay: ℝ)
  (overtime_hours: ℝ)
  (total_pay_first_40: ℝ := hours_first_40 * hourly_rate_first_40_hours)
  (pay_overtime: ℝ := gross_pay - total_pay_first_40)
  (hourly_rate_overtime: ℝ := pay_overtime / overtime_hours)
  (h1: hourly_rate_first_40_hours = 11.25)
  (h2: hours_first_40 = 40)
  (h3: gross_pay = 622)
  (h4: overtime_hours = 10.75) :
  hourly_rate_overtime = 16 := 
by
  sorry

end overtime_hourly_rate_l259_259209


namespace analytical_and_max_value_l259_259234

noncomputable def isOddFun (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc (-a) a, f (-x) = -f x

noncomputable def f_analytical (x : ℝ) : ℝ := 
  -1 / (Real.exp (2 * x)) + 1 / (Real.exp x)

theorem analytical_and_max_value (a : ℝ) (ha : 0 < a) :
  (∀ x ∈ Icc (0 : ℝ) a, f_analytical x = Real.exp (2 * x) - Real.exp x) ∧
  (∃ M, M = Real.exp a * (Real.exp a - 1) ∧ ∀ x ∈ Icc (0 : ℝ) a, f_analytical x ≤ M) :=
by
  sorry

end analytical_and_max_value_l259_259234


namespace vasya_guarantee_win_l259_259842

theorem vasya_guarantee_win (x : Fin 10 → ℝ) (cards : Finset (Finset (Fin 10)))
  (h1 : ∀ card ∈ cards, card.card = 5)
  (h2 : ∀ (i j : Fin 10), i ≤ j → x i ≤ x j) :
  ∃ s : Finset (Finset (Fin 10)), s.card = cards.card / 2 ∧
    (∑ card in s, ∏ i in card, x i) > (∑ card in cards \ s, ∏ i in card, x i) :=
by
  sorry

end vasya_guarantee_win_l259_259842


namespace solution_set_of_absolute_value_inequality_l259_259893

theorem solution_set_of_absolute_value_inequality :
  { x : ℝ | |x + 1| - |x - 2| > 1 } = { x : ℝ | 1 < x } :=
by 
  sorry

end solution_set_of_absolute_value_inequality_l259_259893


namespace sum_possible_cupcakes_l259_259915

def satisfies_conditions (N : ℕ) : Prop :=
  (N % 5 = 3) ∧ (N % 7 = 4) ∧ (N < 60)

theorem sum_possible_cupcakes :
  -- Sum of all N such that N ≡ 3 (mod 5), N ≡ 4 (mod 7), and N < 60 is 71 
  (Finset.sum (Finset.filter satisfies_conditions (Finset.range 60)) (λ x, x)) = 71 :=
  by
  -- skipping the proof
  sorry

end sum_possible_cupcakes_l259_259915


namespace odd_function_iff_b_zero_l259_259722

variable (b : ℝ)
def f (x : ℝ) : ℝ := 3 * x + b * Real.cos x

theorem odd_function_iff_b_zero : 
  (∀ x : ℝ, f b (-x) = -f b x) ↔ b = 0 :=
by sorry

end odd_function_iff_b_zero_l259_259722


namespace optimal_travel_time_minimization_l259_259127

theorem optimal_travel_time_minimization :
  ∀ (M N : ℝ) (d : ℝ) (v_walk v_bike : ℝ)
    (A B C : ℝ → ℝ),
  M = 0 → N = 15 →
  d = 15 →
  v_walk = 6 →
  v_bike = 15 →
  ∀ (t : ℝ),
  C(t) = (t - \frac{3}{11} * 15) →
  ∃ (D t : ℝ), t = \frac{5}{7} ∧
  (B(t) = 15 * \frac{5}{7} + 15 * (1 - \frac{5}{7}) / 6) :=
sorry

end optimal_travel_time_minimization_l259_259127


namespace odell_kershaw_meeting_count_l259_259834

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def angular_speed (v : ℝ) (r : ℝ) : ℝ := v / circumference r * 2 * Real.pi

def relative_angular_speed (w1 w2 : ℝ) : ℝ := w1 + w2

def meeting_time (relative_w : ℝ) : ℝ := 2 * Real.pi / relative_w

def total_meeting_count (time : ℝ) (k : ℝ) : ℝ := Real.floor (time / k)

noncomputable def odell_meeting_count : ℝ :=
  let v_O := 260
  let r_O := 55
  let v_K := 280
  let r_K := 65
  let t := 30
  let w_O := angular_speed v_O r_O
  let w_K := angular_speed v_K r_K
  let relative_w := relative_angular_speed w_O w_K
  let k := meeting_time relative_w
  total_meeting_count t k

theorem odell_kershaw_meeting_count :
  odell_meeting_count = 126 := by
  sorry

end odell_kershaw_meeting_count_l259_259834


namespace solution_to_equation_l259_259114

theorem solution_to_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ x ≠ -5 → (2 / (x + 5) = 1 / x ↔ x = 5) := 
by
  intro x
  intro h
  have h1 : x ≠ 0 := h.1
  have h2 : x ≠ -5 := h.2
  apply Iff.intro
  {
    intro heq
    -- To be filled with proof steps
    sorry
  }
  {
    intro heq
    rw heq
    -- To be filled with proof steps
    sorry
  }

end solution_to_equation_l259_259114


namespace spring_outing_students_l259_259649

variable (x y : ℕ)

theorem spring_outing_students (hx : x % 10 = 0) (hy : y % 10 = 0) (h1 : x + y = 1008) (h2 : y - x = 133) :
  x = 437 ∧ y = 570 :=
by
  sorry

end spring_outing_students_l259_259649


namespace mike_toys_l259_259619

theorem mike_toys (M A T : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : T = A + 2)
  (h3 : M + A + T = 56) 
  : M = 6 := 
by 
  sorry

end mike_toys_l259_259619


namespace find_number_l259_259623

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l259_259623


namespace exactly_one_proposition_is_true_l259_259282

/-- Given m and n are two different lines and α and β are two different planes,
prove that exactly one of the following propositions is true:
1. If α and β are perpendicular to the same plane, then α and β are parallel.
2. If m and n are parallel to the same plane, then m and n are parallel.
3. If α and β are not parallel, then there is no line in α that is parallel to β.
4. If m and n are not parallel, then m and n cannot be perpendicular to the same plane. -/

theorem exactly_one_proposition_is_true
  {m n : Line}
  {α β : Plane}
  (h_mn_diff : m ≠ n)
  (h_ab_diff : α ≠ β) :
  ( if (h1 : (∃ p : Plane, α.perpendicular_to p ∧ β.perpendicular_to p)) then α.parallel_to β else true ) ∧
  ( if (h2 : (∃ p : Plane, m.parallel_to p ∧ n.parallel_to p)) then m.parallel_to n else true ) ∧
  ( if (h3 : ¬ α.parallel_to β ) then ∀ l ∈ α, ¬ l.parallel_to β else true ) ∧
  ( if (h4 : ¬ m.parallel_to n ) then ∀ p : Plane, ¬ (m.perpendicular_to p ∧ n.perpendicular_to p) else true ) :=
sorry

end exactly_one_proposition_is_true_l259_259282


namespace sum_of_roots_l259_259539

-- Define the polynomial
def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X

-- Statement of the problem: Prove the sum of the roots is 3
theorem sum_of_roots : (Polynomial.roots poly).sum = 3 :=
by sorry

end sum_of_roots_l259_259539


namespace ellipse_problem_l259_259296

theorem ellipse_problem (O F E : Point)
    (hO : O = (0,0))
    (hF : F = (1,0))
    (hE : E = (2,0))
    (hl : ∀ x, Line x = 2)
    (hFE_O : Vector.ofPoints F E = Vector.ofPoints O F)
    (points_A_B : ∃ A B : Point, Line.through F A ∧ Line.through F B ∧ Ellipse.point A ∧ Ellipse.point B)
    (points_C_D : ∃ C D : Point, C ∈ Line 2 ∧ D ∈ Line 2 ∧ Parallel AD BC ∧ Parallel AD (xAxis))
    : 
    -- Part (I)
    (∃ a b: ℝ, a > b > 0 ∧ Ellipse.equation = (x^2)/a^2 + (y^2)/b^2 = 1 ∧ 2*b = 2 ∧ Ellipse.eccentricity = c/a ∧ a = sqrt 2 ∧ c = 1 ∧ Ellipse.equation = (x^2)/2 + y^2 = 1 ∧ Ellipse.eccentricity = sqrt 2 / 2)
    ∧
    -- Part (II)
    (Line.passesThrough AC (midpoint (segment EF)) = True)
    :=
sorry

end ellipse_problem_l259_259296


namespace toby_initial_photos_l259_259527

-- Defining the problem conditions and proving the initial number of photos Toby had.
theorem toby_initial_photos (X : ℕ) 
  (h1 : ∃ n, X = n - 7) 
  (h2 : ∃ m, m = (n - 7) + 15) 
  (h3 : ∃ k, k = m) 
  (h4 : (k - 3) = 84) 
  : X = 79 :=
sorry

end toby_initial_photos_l259_259527


namespace solve_n_m_equation_l259_259859

theorem solve_n_m_equation : 
  ∃ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ∧ ((n, m) = (3, 5) ∨ (n, m) = (3, -5) ∨ (n, m) = (-3, 5) ∨ (n, m) = (-3, -5)) :=
by { sorry }

end solve_n_m_equation_l259_259859


namespace geometric_sequence_arithmetic_sequence_l259_259444

def seq₃ := 7
def rec_rel (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + a 2 - 2

-- Problem Part 1: Prove that {a_n+1} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℕ) (h_rec_rel : rec_rel a) :
  ∃ r, ∀ n, n ≥ 1 → (a n + 1) = r * (a (n - 1) + 1) :=
sorry

-- Problem Part 2: Given a general formula, prove n, a_n, and S_n form an arithmetic sequence
def general_formula (a : ℕ → ℕ) := ∀ n, a n = 2^n - 1
def sum_formula (S : ℕ → ℕ) := ∀ n, S n = 2^(n+1) - n - 2

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_general : general_formula a) (h_sum : sum_formula S) :
  ∀ n, n + S n = 2 * a n :=
sorry

end geometric_sequence_arithmetic_sequence_l259_259444


namespace num_mappings_A_to_B_num_injective_mappings_A_to_B_exists_surjective_mapping_A_to_B_l259_259816

open Finset

variables (m n : ℕ) (h : m ≤ n)
def A : Finset ℕ := range m
def B : Finset ℕ := range n

-- Number of all mappings from A to B
theorem num_mappings_A_to_B : A.card = m → B.card = n → ∃ f : Fin m → Fin n, true := 
by
  intros
  sorry

-- Number of all injective mappings from A to B
theorem num_injective_mappings_A_to_B : A.card = m → B.card = n → ∃ f : Fin m ↪ Fin n, finset.card (univ.injective_maps (Fin m) (Fin n)) = n.choose m := 
by
  intros
  sorry

-- Existence of surjective mappings from A to B
theorem exists_surjective_mapping_A_to_B : A.card = m → B.card = n → (m ≥ n ↔ ∃ f : Fin m → Fin n, function.surjective f) := 
by
  intros
  sorry

end num_mappings_A_to_B_num_injective_mappings_A_to_B_exists_surjective_mapping_A_to_B_l259_259816


namespace sum_inverses_mod_17_l259_259220

theorem sum_inverses_mod_17 :
  (2^(-7 : ℤ) + 2^(-8 : ℤ) + 2^(-9 : ℤ) + 2^(-10 : ℤ) + 2^(-11 : ℤ) + 2^(-12 : ℤ)) % 17 = 14 :=
by
  -- The proof goes here, which is skipped
  sorry

end sum_inverses_mod_17_l259_259220


namespace neg_exists_exp_l259_259734

theorem neg_exists_exp (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x < 0)) = (∀ x : ℝ, Real.exp x ≥ 0) :=
by
  sorry

end neg_exists_exp_l259_259734


namespace evaluate_expr_l259_259155

noncomputable def expr : ℚ :=
  2013 * (5.7 * 4.2 + (21 / 5) * 4.3) / ((14 / 73) * 15 + (5 / 73) * 177 + 656)

theorem evaluate_expr : expr = 126 := by
  sorry

end evaluate_expr_l259_259155


namespace last_two_digits_of_binom_200_100_l259_259657

open Nat

theorem last_two_digits_of_binom_200_100 :
  (nat.binomial 200 100) % 100 = 20 :=
by
  have h_mod_4 : (nat.binomial 200 100) % 4 = 0 := sorry
  have h_mod_25 : (nat.binomial 200 100) % 25 = 20 := sorry
  exact Nat.modeq.chinese_remainder h_mod_4 h_mod_25 sorry

end last_two_digits_of_binom_200_100_l259_259657


namespace distance_focus_asymptote_hyperbola_eq_sqrt2_l259_259727

theorem distance_focus_asymptote_hyperbola_eq_sqrt2 :
  ∀ x y, x^2 - y^2 = 2 → distance_from_focus_to_asymptote x y = √2 :=
sorry

end distance_focus_asymptote_hyperbola_eq_sqrt2_l259_259727


namespace common_ratio_geometric_sequence_l259_259314

variables {a : ℕ → ℝ} -- 'a' is a sequence of positive real numbers
variable {q : ℝ} -- 'q' is the common ratio of the geometric sequence

-- Definition of a geometric sequence with common ratio 'q'
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition from the problem statement
def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  2 * a 5 - 3 * a 4 = 2 * a 3

-- Main theorem: If the sequence {a_n} is a geometric sequence with positive terms and satisfies the condition, 
-- then the common ratio q = 2
theorem common_ratio_geometric_sequence :
  (∀ n, 0 < a n) → geometric_sequence a q → condition a q → q = 2 :=
by
  intro h_pos h_geom h_cond
  sorry

end common_ratio_geometric_sequence_l259_259314


namespace function_intersects_x_axis_l259_259845

theorem function_intersects_x_axis (y : ℝ → ℝ) : ∃ x : ℝ, y = λ x, x + 1 ∧ y x = 0 :=
by
  refine ⟨ -1, _, _ ⟩
  have h : y = λ x, x + 1 := sorry
  rw [h]
  norm_num
  exact hm
  sorry

end function_intersects_x_axis_l259_259845


namespace boat_speed_in_still_water_l259_259894

def speed_of_stream : ℝ := 8
def downstream_distance : ℝ := 64
def upstream_distance : ℝ := 32

theorem boat_speed_in_still_water (x : ℝ) (t : ℝ) 
  (HS_downstream : t = downstream_distance / (x + speed_of_stream)) 
  (HS_upstream : t = upstream_distance / (x - speed_of_stream)) :
  x = 24 := by
  sorry

end boat_speed_in_still_water_l259_259894


namespace smallest_n_for_g_greater_than_21_l259_259039

noncomputable def g (n : ℕ) : ℕ :=
  Nat.find (λ k, k.factorial % n = 0)

theorem smallest_n_for_g_greater_than_21 {r : ℕ} (hr : r ≥ 23) :
  let n := 21 * r in g n > 21 :=
by
  let n := 21 * r
  have hr23 : r ≥ 23 := hr
  have hdiv : 23 ∣ n := by
    sorry
  have gn : g n ≥ 23 := by
    sorry
  exact Nat.lt_of_le_of_lt 21 gn

end smallest_n_for_g_greater_than_21_l259_259039


namespace line_through_A_max_area_triangle_CPQ_l259_259693

namespace problem

/-- Part (I): Given circle C: (x-3)^2+(y-4)^2=4 and line l passing through point A(2,3)
show that the equation of line l is either x=2 or y=3. -/
theorem line_through_A (x y : ℝ) :
  (x - 3)^2 + (y - 4)^2 = 4 →
  (∃ m b : ℝ, y = m * (x - 2) + 3 ∧ ∀ x y, y = m * (x - 2) + 3 → (x - 3)^2 + (y - 4)^2 = 4) →
  ∃ l : ℝ, (l = 2 ∨ l = 3) :=
  sorry

/-- Part (II): Given the circle C: (x-3)^2+(y-4)^2=4, line l passing through B(1,0),
and line l intersects circle C at points P and Q, show that the maximum area of CPQ is 2
and the equation of l is either x - y - 1 = 0 or 7x - y - 7 = 0. -/
theorem max_area_triangle_CPQ (x y : ℝ) :
  (x - 3)^2 + (y - 4)^2 = 4 →
  (∃ m b, y = m * (x - 1) + 0 ∧ ∀ x y, y = m * (x - 1) + 0 → (x - 3)^2 + (y - 4)^2 = 4) →
  ∃ s : ℝ, (s = 2 ∧ ((1 - y = x - 1) ∨ (7 * (1 - y) = x - 7))) :=
  sorry

end problem

end line_through_A_max_area_triangle_CPQ_l259_259693


namespace find_t_l259_259301

theorem find_t
  (x y t : ℝ)
  (h1 : 2 ^ x = t)
  (h2 : 5 ^ y = t)
  (h3 : 1 / x + 1 / y = 2)
  (h4 : t ≠ 1) : 
  t = Real.sqrt 10 := 
by
  sorry

end find_t_l259_259301


namespace cone_volume_semicircle_unfolded_l259_259110

theorem cone_volume_semicircle_unfolded (r h : ℝ) (volume : ℝ) (rad : r = 1) :
  volume = (π * (1 / 2)^2 * (sqrt (1 - (1 / 2)^2)) / 3) → volume = sqrt(3) * π / 24 :=
by
  assume h_rad : rad
  assume vol_calc : volume = (π * (1 / 2)^2 * (sqrt (1 - (1 / 2)^2)) / 3)
  show volume = sqrt(3) * π / 24, from
    sorry

end cone_volume_semicircle_unfolded_l259_259110


namespace cot_identity_triangle_l259_259434

theorem cot_identity_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h_triangle : a^2 + b^2 + 32 * c^2 = 2021 * c^2) :
  ∀ (h_angles_sum : α + β + γ = π) (h_law_sines : a / sin α = b / sin β = c / sin γ),
  (cot γ / (cot α + cot β) = 994) :=
begin
  sorry
end

end cot_identity_triangle_l259_259434


namespace hyperbola_eccentricity_correct_l259_259731

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (F : ℝ × ℝ) (l : ℝ → ℝ) (A B P : ℝ × ℝ)
  (m n : ℝ) (h_mn : m * n = 2 / 9) (O : ℝ × ℝ)
  (h_hyperbola : ∀ x y, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1) 
  (h_l : l F.1 = F.2) 
  (h_AB : ∃ c : ℝ, A = (c, b * c / a) ∧ B = (c, -b * c / a)) 
  (h_P_on_l : l P.1 = P.2) 
  (h_OP : ∃ m n, P = (m + n, (m - n) * b / a) ∧ m * n = 2 / 9) 
: real := 
  (3 * real.sqrt 2) / 4

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
    (F : ℝ × ℝ) (l : ℝ → ℝ) (A B P : ℝ × ℝ)
    (m n : ℝ) (h_mn : m * n = 2 / 9) (O : ℝ × ℝ)
    (h_hyperbola : ∀ x y, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)
    (h_l : l F.1 = F.2)
    (h_AB : ∃ c : ℝ, A = (c, b * c / a) ∧ B = (c, -b * c / a))
    (h_P_on_l : l P.1 = P.2)
    (h_OP : ∃ m n, P = (m + n, (m - n) * b / a) ∧ m * n = 2 / 9), 
    hyperbola_eccentricity a b h_a h_b F l A B P m n h_mn O h_hyperbola h_l h_AB h_P_on_l h_OP = (3 * real.sqrt 2) / 4 :=
by {
  sorry
}


end hyperbola_eccentricity_correct_l259_259731


namespace coins_in_box_l259_259942

theorem coins_in_box (n : ℕ) 
    (h1 : n % 8 = 7) 
    (h2 : n % 7 = 5) : 
    n = 47 ∧ (47 % 9 = 2) :=
sorry

end coins_in_box_l259_259942


namespace sequences_properties_l259_259685

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) - a n = 2)

def geometric_seq (b : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ b 4 = 8

def combined_seq_sum (a b : ℕ → ℕ) (c : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = ∑ i in finset.range n, c (i + 1))

theorem sequences_properties (a b c S : ℕ → ℕ) (ha : arithmetic_seq a) (hb : geometric_seq b) (hcs: combined_seq_sum a b c S) :
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, b n = 2 ^ (n - 1)) ∧
  (∀ n : ℕ, S n = 2 ^ n + n ^ 2 - 1) :=
sorry

end sequences_properties_l259_259685


namespace find_train_parameters_l259_259172

-- Definitions based on the problem statement
def bridge_length : ℕ := 1000
def time_total : ℕ := 60
def time_on_bridge : ℕ := 40
def speed_train (x : ℕ) := (40 * x = bridge_length)
def length_train (x y : ℕ) := (60 * x = bridge_length + y)

-- Stating the problem to be proved
theorem find_train_parameters (x y : ℕ) (h₁ : speed_train x) (h₂ : length_train x y) :
  x = 20 ∧ y = 200 :=
sorry

end find_train_parameters_l259_259172


namespace meryll_remaining_questions_l259_259449

variables (total_mc total_ps total_tf : ℕ)
variables (frac_mc frac_ps frac_tf : ℚ)

-- Conditions as Lean definitions:
def written_mc (total_mc : ℕ) (frac_mc : ℚ) := (frac_mc * total_mc).floor
def written_ps (total_ps : ℕ) (frac_ps : ℚ) := (frac_ps * total_ps).floor
def written_tf (total_tf : ℕ) (frac_tf : ℚ) := (frac_tf * total_tf).floor

def remaining_mc (total_mc : ℕ) (frac_mc : ℚ) := total_mc - written_mc total_mc frac_mc
def remaining_ps (total_ps : ℕ) (frac_ps : ℚ) := total_ps - written_ps total_ps frac_ps
def remaining_tf (total_tf : ℕ) (frac_tf : ℚ) := total_tf - written_tf total_tf frac_tf

def total_remaining (total_mc total_ps total_tf : ℕ) (frac_mc frac_ps frac_tf : ℚ) :=
  remaining_mc total_mc frac_mc + remaining_ps total_ps frac_ps + remaining_tf total_tf frac_tf

-- The statement to prove:
theorem meryll_remaining_questions :
  total_remaining 50 30 40 (5/8) (7/12) (2/5) = 56 :=
by
  sorry

end meryll_remaining_questions_l259_259449


namespace cookie_cost_difference_l259_259104

theorem cookie_cost_difference 
    (total_items : ℕ)
    (cookie_ratio_oreos : ℕ)
    (cookie_ratio_choco_chip : ℕ)
    (cookie_ratio_sugar : ℕ)
    (price_oreo : ℕ)
    (price_choco_chip : ℕ)
    (price_sugar : ℕ)
    (h1 : cookie_ratio_oreos = 4)
    (h2 : cookie_ratio_choco_chip = 5)
    (h3 : cookie_ratio_sugar = 6)
    (h4 : price_oreo = 2)
    (h5 : price_choco_chip = 3)
    (h6 : price_sugar = 4)
    (h7 : total_items = 90) :
    let x := total_items / (cookie_ratio_oreos + cookie_ratio_choco_chip + cookie_ratio_sugar),
        num_oreos := cookie_ratio_oreos * x,
        num_choco_chip := cookie_ratio_choco_chip * x,
        num_sugar := cookie_ratio_sugar * x,
        cost_oreos := num_oreos * price_oreo,
        cost_choco_chip := num_choco_chip * price_choco_chip,
        cost_sugar := num_sugar * price_sugar,
        total_cost_choco_chip_sugar := cost_choco_chip + cost_sugar
    in total_cost_choco_chip_sugar - cost_oreos = 186 :=
by
  sorry

end cookie_cost_difference_l259_259104


namespace sum_of_positive_ks_l259_259097

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end sum_of_positive_ks_l259_259097


namespace angle_sum_quadrilateral_l259_259650

noncomputable def quadrilateral (A B C D: Type) : Prop :=
  ∃ (B_prime C_prime: Type),
    (equilateral_triangle A C B_prime ∧ 
     equilateral_triangle B D C_prime) ∧
    (same_side B B_prime A C) ∧
    (same_side C C_prime B D) ∧
    (distance B_prime C_prime = distance A B + distance C D)

theorem angle_sum_quadrilateral {A B C D : Type} 
  (H1: quadrilateral A B C D) :
  ∠BAD + ∠CDA = 120 :=
sorry

end angle_sum_quadrilateral_l259_259650


namespace regular_polygon_sides_l259_259959

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ m : ℕ, m = 360 / n → n ≠ 0 → m = 30) : n = 12 :=
  sorry

end regular_polygon_sides_l259_259959


namespace edge_length_of_regular_tetrahedron_l259_259317

theorem edge_length_of_regular_tetrahedron
  (sum_of_edges_of_face : ℝ) 
  (h : sum_of_edges_of_face = 18) 
  : let edge_length := sum_of_edges_of_face / 3 
  in edge_length = 6 := 
by 
  let edge_length := sum_of_edges_of_face / 3
  have : edge_length = 6, from calc
    edge_length = 18 / 3 : by rw h
             ... = 6    : by norm_num
  exact this

end edge_length_of_regular_tetrahedron_l259_259317


namespace trig_identity_l259_259567

theorem trig_identity : sin (10 * Real.pi / 180) * cos (40 * Real.pi / 180) - cos (50 * Real.pi / 180) * cos (10 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end trig_identity_l259_259567


namespace determine_translation_vector_l259_259877

-- Definitions of the functions
def f (x : ℝ) : ℝ := sin (2 * x)
def g (x : ℝ) : ℝ := sin (2 * x + π / 3) + 2

-- Statement that verifies the translation
theorem determine_translation_vector :
  ∃ a b : ℝ, (∀ x : ℝ, g (x - a) = f x + b) ∧ (a = -π / 6 ∧ b = 2) :=
by {
  use [-π / 6, 2],
  split,
  -- Here is where we would prove the properties if a proof were needed
  sorry,
  split,
  -- Proving the values correspond
  refl,
  refl
}

end determine_translation_vector_l259_259877


namespace incorrect_statement_l259_259198

-- Define the relationship between the length of the spring and the mass of the object
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

-- Formalize statements A, B, C, and D
def statementA : Prop := spring_length 0 = 10

def statementB : Prop :=
  ¬ ∃ (length : ℝ) (mass : ℝ), (spring_length mass = length ∧ mass = (length - 10) / 2.5)

def statementC : Prop :=
  ∀ m : ℝ, spring_length (m + 1) = spring_length m + 2.5

def statementD : Prop := spring_length 4 = 20

-- The Lean statement to prove that statement B is incorrect
theorem incorrect_statement (hA : statementA) (hC : statementC) (hD : statementD) : ¬ statementB := by
  sorry

end incorrect_statement_l259_259198


namespace binary_div_mul_eq_l259_259215

open nat

-- Definitions for binary literals
noncomputable def bin_1101110 : ℕ := nat.of_digits 2 [1,1,0,1,1,1,0]
noncomputable def bin_100 : ℕ := nat.of_digits 2 [1,0,0]
noncomputable def bin_1101 : ℕ := nat.of_digits 2 [1,1,0,1]
noncomputable def bin_10010001 : ℕ := nat.of_digits 2 [1,0,0,1,0,0,0,1]

theorem binary_div_mul_eq : 
  (bin_1101110 / bin_100) * bin_1101 = bin_10010001 := 
by 
  sorry

end binary_div_mul_eq_l259_259215


namespace problem_a_problem_b_l259_259555

variable {n : ℕ}

theorem problem_a (Cn : ℕ -> ℕ) :
  (C n 0 - C (n-1) 1 * (1/4) + C (n-2) 2 * (1/4^2) - 
    ... + (-1) ^ i * C (n-i) i * (1/4^i) + ...) = (n + 1)/2^n := by
  sorry

theorem problem_b (Cn : ℕ -> ℕ) (p q : ℚ) (h : p + q = 1) :
  (C n 0 - C (n-1) 1 * p * q + C (n-2) 2 * p^2 * q^2 - 
    ... + (-1) ^ i * C (n-i) i * p^i * q^i + ...) = (p^(n+1) - q^(n+1)) / (p - q) := by
  sorry

end problem_a_problem_b_l259_259555


namespace fixed_point_always_l259_259874

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always_l259_259874


namespace function_domain_l259_259407

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259407


namespace range_of_x_in_sqrt_x_plus_3_l259_259401

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259401


namespace francie_remaining_money_l259_259673

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end francie_remaining_money_l259_259673


namespace euclidean_remainder_of_P_div_X2_X1_l259_259815

noncomputable def P : ℝ[X] := sorry
def Q (X : ℝ) : ℝ := sorry -- This is just a placeholder for the quotient polynomial.

theorem euclidean_remainder_of_P_div_X2_X1 :
  let R := (λ X, 2 * X + 1) in
  P.eval 2 = 5 ∧ P.eval 1 = 3 →
  ∃ Q : ℝ[X], P = Q * (X - 2) * (X - 1) + (X + 1) :=
begin
  -- Proof goes here
  sorry
end

end euclidean_remainder_of_P_div_X2_X1_l259_259815


namespace triangle_altitude_problem_l259_259204

theorem triangle_altitude_problem
  (H_trio_acute : is_acute_triangle ABC)
  (H_altitudes : altitudes_intersect_at H)
  (H_HP : HP = 3)
  (H_HQ : HQ = 4)
  (H_similarity1 : similar_triangles BPH APC)
  (H_similarity2 : similar_triangles AQH BQC)
  (H_ratio_BP_PC : ratio_eq BP PC 2 3)
  (H_ratio_AQ_QC : ratio_eq AQ QC 1 4) :
  (BP * PC) - (AQ * QC) = -10 :=
sorry

end triangle_altitude_problem_l259_259204


namespace g_decreasing_on_1_2_l259_259052

open Function

/-- Define the function f(x) and g(x) as given -/
def f (x : ℝ) := 1 / x
def g (x : ℝ) := x^2 * (f (x - 1))

/-- The main theorem that g(x) is decreasing on the interval (1, 2) -/
theorem g_decreasing_on_1_2 : ∃ (I : Set ℝ), I = Set.Ioo 1 2 ∧ ∀ (x : ℝ), x ∈ I → ∃ (g' : ℝ), deriv g x = g' ∧ g' < 0 := 
sorry

end g_decreasing_on_1_2_l259_259052


namespace polygon_isosceles_triangle_exists_l259_259363

def isosceles_triangle (vertices : finset ℕ) : Prop :=
  ∃ v1 v2 v3 : ℕ, v1 ∈ vertices ∧ v2 ∈ vertices ∧ v3 ∈ vertices ∧
  (v2 - v1) % 5000 = (v3 - v2) % 5000

theorem polygon_isosceles_triangle_exists (poly : finset ℕ) (h1 : poly.card = 5000)
(h2 : (poly.filter (λ v, v ∈ painted)).card = 2001) :
∃ tri ⊆ painted, isosceles_triangle tri :=
sorry

end polygon_isosceles_triangle_exists_l259_259363


namespace smallest_angle_in_triangle_l259_259511

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l259_259511


namespace least_possible_integral_BC_l259_259569

theorem least_possible_integral_BC :
  ∃ (BC : ℕ), (BC > 0) ∧ (BC ≥ 15) ∧ 
    (7 + BC > 15) ∧ (25 + 10 > BC) ∧ 
    (7 + 15 > BC) ∧ (25 + BC > 10) := by
    sorry

end least_possible_integral_BC_l259_259569


namespace find_a_such_that_perfect_square_l259_259653

theorem find_a_such_that_perfect_square :
  {a : ℕ | a ∈ ({1, 2, 3, 4} : Set ℕ) ∧ ∃ᶠ n in at_top, ∃ k : ℕ, (10^n - 1) * (underbrace 1 k + underbrace 1 k) = k * k} = {1, 4} :=
sorry

end find_a_such_that_perfect_square_l259_259653


namespace bisects_DX_EF_l259_259047

-- Definitions used in Lean 4 statement from the conditions
open EuclideanGeometry

variables {A B C G D E F X : Point} -- Points in the geometric setting
variables (ABC : Triangle)
variables (G : Point) -- Centroid of triangle ABC
variables (circum_BCG : IsCircumcenter D (Triangle.mk B C G))
variables (circum_CAG : IsCircumcenter E (Triangle.mk C A G))
variables (circum_ABG : IsCircumcenter F (Triangle.mk A B G))
variables (X_is_intersection : IsIntersectionPerpendiculars X E F A B C) -- X is intersection of the perpendiculars

-- Theorem statement
theorem bisects_DX_EF (hG : Centroid G ABC) (hD : circum_BCG) (hE : circum_CAG) (hF : circum_ABG) (hX : X_is_intersection) : 
  Bisects (Line.mk D X) (Segment.mk E F) :=
sorry

end bisects_DX_EF_l259_259047


namespace find_number_l259_259508

theorem find_number (N : ℕ) (hN1 : 10 ≤ N) (hN2 : N ≤ 99) (h_sqrt_form : ∃ a : ℕ, (√N).toString.startsWith (toString a ++ "." ++ toString a ++ toString a ++ toString a)) : N = 79 :=
sorry

end find_number_l259_259508


namespace francie_has_3_dollars_remaining_l259_259676

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l259_259676


namespace find_second_discount_l259_259972

def second_discount (D1 : ℝ) (P : ℝ) (F : ℝ) : ℝ :=
  100 * (1 - F / (P * (1 - D1 / 100)))

theorem find_second_discount:
  let P : ℝ := 150
  let F : ℝ := 105
  let D1 : ℝ := 19.954259576901087
  second_discount D1 P F = 12.552 :=
by {
  -- The proof steps would go here
  sorry
}

end find_second_discount_l259_259972


namespace problem_l259_259640

theorem problem (x : ℝ) (p : Prop := x < -1 ∨ x > 1) (q : Prop := x < -2) :
  (q → p) ∧ ¬ (p → q) := by
  split
  {
    -- First part: q implies p
    assume hq : q,
    cases hq
    sorry
  },
  {
    -- Second part: Not (p implies q)
    assume hp : p,
    cases hp
    sorry
  }

end problem_l259_259640


namespace even_sum_probability_l259_259118

-- Conditions
def prob_even_first_wheel : ℚ := 1 / 4
def prob_odd_first_wheel : ℚ := 3 / 4
def prob_even_second_wheel : ℚ := 2 / 3
def prob_odd_second_wheel : ℚ := 1 / 3

-- Statement: Theorem that the probability of the sum being even is 5/12
theorem even_sum_probability : 
  (prob_even_first_wheel * prob_even_second_wheel) + 
  (prob_odd_first_wheel * prob_odd_second_wheel) = 5 / 12 :=
by
  -- Proof steps would go here
  sorry

end even_sum_probability_l259_259118


namespace cookies_per_person_l259_259921

theorem cookies_per_person (total_cookies : ℕ) (people : ℕ) (h1 : total_cookies = 24) (h2 : people = 6) : total_cookies / people = 4 := by
  rw [h1, h2]
  norm_num
  sorry

end cookies_per_person_l259_259921


namespace score_order_l259_259805

-- Definitions that come from the problem conditions
variables (M Q S K : ℝ)
variables (hQK : Q = K) (hMK : M > K) (hSK : S < K)

-- The theorem to prove
theorem score_order (hQK : Q = K) (hMK : M > K) (hSK : S < K) : S < Q ∧ Q < M :=
by {
  sorry
}

end score_order_l259_259805


namespace division_problem_l259_259351

theorem division_problem (x : ℝ) (h1 : x = 1) : 4 / (1 + 3 / x) = 1 := by
  rw [h1]
  norm_num
  sorry

end division_problem_l259_259351


namespace num_ordered_pairs_satisfying_systems_l259_259263

theorem num_ordered_pairs_satisfying_systems : 
  (∃! (xy : ℝ × ℝ), (xy.1 = (xy.2)^2 + 2*(xy.1)^2) ∧ (xy.2 = 3*(xy.1)*xy.2)) =
4 :=
sorry

end num_ordered_pairs_satisfying_systems_l259_259263


namespace coefficient_x2_expansion_l259_259866

-- Define the problem statement
theorem coefficient_x2_expansion (m : ℝ) 
  (h : binomial 5 3 * m^3 = -10) : 
  m = -1 :=
by
  sorry

end coefficient_x2_expansion_l259_259866


namespace f_2017_is_1_l259_259310

noncomputable def f : ℕ → ℤ := sorry

axiom f_condition1 : ∀ x > 0, f(x+1) = f(x) + f(x+2)
axiom f_condition2 : f(1) = 1
axiom f_condition3 : f(2) = 3

theorem f_2017_is_1 : f(2017) = 1 := sorry

end f_2017_is_1_l259_259310


namespace sum_of_first_2n_terms_l259_259687

theorem sum_of_first_2n_terms
  (a : ℕ → ℕ)
  (h₀ : a 1 = 1)
  (h₁ : ∀ (n : ℕ), a (n + 1) = if n % 2 = 1 then a n + 2 else 3 * a n) :
  (∑ i in Finset.range (2 * n), a (i + 1)) = 4 * 3 ^ n - 4 * n - 4 :=
sorry

end sum_of_first_2n_terms_l259_259687


namespace prove_a_pow_m_plus_2n_log_4_eq_n_div_2m_l259_259304

variable (a m n : ℝ)
variable (h1 : log a 2 = m)
variable (h2 : log a 3 = n)
variable (ha_pos : 0 < a)
variable (ha_ne_one : a ≠ 1)

theorem prove_a_pow_m_plus_2n : a^(m + 2*n) = 18 := by
  sorry

theorem log_4_eq_n_div_2m : log 4 3 = n/(2*m) := by
  sorry

end prove_a_pow_m_plus_2n_log_4_eq_n_div_2m_l259_259304


namespace num_ways_to_queue_ABC_l259_259121

-- Definitions for the problem
def num_people : ℕ := 5
def fixed_order_positions : ℕ := 3

-- Lean statement to prove the problem
theorem num_ways_to_queue_ABC (h : num_people = 5) (h_fop : fixed_order_positions = 3) : 
  (Nat.factorial num_people / Nat.factorial (num_people - fixed_order_positions)) * 1 = 20 := 
by
  sorry

end num_ways_to_queue_ABC_l259_259121


namespace board_election_ways_l259_259628

theorem board_election_ways :
  let total_ways := Nat.choose 20 6 in
  let ways_no_prev_serv := Nat.choose 11 6 in
  total_ways - ways_no_prev_serv = 38298 :=
by
  let total_ways := Nat.choose 20 6
  let ways_no_prev_serv := Nat.choose 11 6
  show total_ways - ways_no_prev_serv = 38298
  sorry

end board_election_ways_l259_259628


namespace smallest_n_with_gn_greater_than_21_l259_259036

def g (n : ℕ) : ℕ :=
  Inf { k : ℕ | k > 0 ∧ factorial k ∣ n }

theorem smallest_n_with_gn_greater_than_21 (r : ℕ) (hr : r ≥ 22) :
  let n := 21 * r in g n > 21 ↔ n = 462 := by
  sorry

end smallest_n_with_gn_greater_than_21_l259_259036


namespace smallest_angle_in_triangle_l259_259510

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l259_259510


namespace smallest_n_with_gn_greater_than_21_l259_259037

def g (n : ℕ) : ℕ :=
  Inf { k : ℕ | k > 0 ∧ factorial k ∣ n }

theorem smallest_n_with_gn_greater_than_21 (r : ℕ) (hr : r ≥ 22) :
  let n := 21 * r in g n > 21 ↔ n = 462 := by
  sorry

end smallest_n_with_gn_greater_than_21_l259_259037


namespace ellipse_parabola_common_point_l259_259761

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔  -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end ellipse_parabola_common_point_l259_259761


namespace range_cosine_range_sine_interval_range_sine_ab_l259_259265

theorem range_cosine (x: ℝ) (h : ∀ x, -1 ≤ cos x ∧ cos x ≤ 1) : 
  ∃ I : Set ℝ, I = Set.Icc (-(1:ℝ)/2) ((7:ℝ)/2) ∧ ∀ x, -2 * cos x + (3/2) ∈ I := 
sorry

theorem range_sine_interval (x: ℝ) (h: x ∈ Set.Icc (-(Real.pi / 6)) (Real.pi / 2)) (hsin: ∀ x, -1/2 ≤ sin x ∧ sin x ≤ 1) : 
  ∃ I : Set ℝ, I = Set.Icc (-(2:ℝ)) (1:ℝ) ∧ ∀ x, 2 * sin x - 1 ∈ I := 
sorry

theorem range_sine_ab (x: ℝ) (a b: ℝ) (h : ∀ x, -1 ≤ sin x ∧ sin x ≤ 1) (ha: a ≠ 0): 
  (a > 0 → ∃ I : Set ℝ, I = Set.Icc (-(a:ℝ) + b) (a + b) ∧ ∀ x, a * sin x + b ∈ I) ∧
  (a < 0 → ∃ I : Set ℝ, I = Set.Icc (a + b) (-(a:ℝ) + b) ∧ ∀ x, a * sin x + b ∈ I) := 
sorry

end range_cosine_range_sine_interval_range_sine_ab_l259_259265


namespace taylor_scores_l259_259523

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l259_259523


namespace largest_three_digit_number_divisible_by_8_l259_259134

-- Define the properties of a number being a three-digit number
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of a number being divisible by 8
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

-- The theorem we want to prove: the largest three-digit number divisible by 8 is 992
theorem largest_three_digit_number_divisible_by_8 : ∃ n, isThreeDigitNumber n ∧ isDivisibleBy8 n ∧ (∀ m, isThreeDigitNumber m ∧ isDivisibleBy8 m → m ≤ 992) :=
  sorry

end largest_three_digit_number_divisible_by_8_l259_259134


namespace bisector_inequality_l259_259836

variable {A B C M : Type} [MetricSpace A]

-- Define points A, B, C, M as variables in a metric space
variable (a b c m : A)

-- Define distances between points
variable (dist_ab : dist a b) (dist_ac : dist a c) (dist_bc : dist b c) (dist_am : dist a m) (dist_bm : dist b m)

-- Conditions from the problem
variable (C1 : m ∈ line a b) -- M lies on the bisector of angle C
variable (C2 : dist_ac > dist_bc) -- AC is greater than CB

-- Statement of the problem
theorem bisector_inequality : |dist_ac - dist_bc| > |dist_am - dist_bm| := 
by
  sorry

end bisector_inequality_l259_259836


namespace jordan_sister_pickled_mangoes_jars_l259_259018

theorem jordan_sister_pickled_mangoes_jars :
  ∀ (total_mangoes ripe_fraction unripe_fraction unripe_kept mangoes_per_jar : ℕ),
    total_mangoes = 120 →
    ripe_fraction = 1 / 4 →
    unripe_fraction = 3 / 4 →
    unripe_kept = 26 →
    mangoes_per_jar = 5 →
    let ripe_mangoes := total_mangoes * ripe_fraction,
        unripe_mangoes := total_mangoes * unripe_fraction,
        unripe_given_to_sister := unripe_mangoes - unripe_kept,
        jars := unripe_given_to_sister / mangoes_per_jar
    in jars = 12 :=
by
  sorry

end jordan_sister_pickled_mangoes_jars_l259_259018


namespace folded_triangle_segment_square_l259_259953

theorem folded_triangle_segment_square 
  (DEF : Triangle) (side_length_DE : DEF.side D E = 15) (side_length_DF : DEF.side D F = 15) (side_length_EF : DEF.side E F = 15) 
  (fold_point_EF : ∃ X : Point, DEF.side E X = 11 ∧ DEF.side F X = 4) :
  let XY_length_sq := (15 - (104 / 19))^2 - ((15 - (104 / 19)) * (15 - (104 / 41))) + (15 - (104 / 41))^2 
  in XY_length_sq = (2174209 / 78281) := 
  by
  sorry

end folded_triangle_segment_square_l259_259953


namespace log_17_not_computable_without_table_l259_259303

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_17_not_computable_without_table :
  (∀ lg2 lg3 : ℝ, lg 2 = lg2 ∧ lg 3 = lg3 →
    ∃ lg_not_computable, lg_not_computable = lg 17 ∧
      ¬ (lg_not_computable = (⟨3, by sorry⟩ : lg 8 = 0.9031 ∧ ⟨2, by sorry⟩ : lg9 = 0.9542)
      ∨ lg_not_computable = some_comp_nat_logarithms)) :=
begin
  sorry
end

end log_17_not_computable_without_table_l259_259303


namespace block_wall_min_blocks_l259_259941

theorem block_wall_min_blocks :
  ∃ n,
    n = 648 ∧
    ∀ (row_height wall_height block1_length block2_length wall_length: ℕ),
    row_height = 1 ∧
    wall_height = 8 ∧
    block1_length = 1 ∧
    block2_length = 3/2 ∧
    wall_length = 120 ∧
    (∀ i : ℕ, i < wall_height → ∃ k m : ℕ, k * block1_length + m * block2_length = wall_length) →
    n = (wall_height * (1 + 2 * 79))
:= by sorry

end block_wall_min_blocks_l259_259941


namespace role_assignment_ways_l259_259184

theorem role_assignment_ways :
  let male_role_options := 4 in
  let female_role_options := 7 in
  let remaining_people := 9 in
  let either_gender_roles := 4 in
  male_role_options * female_role_options * Nat.choose remaining_people either_gender_roles = 3528 :=
by
  -- Male role: 4 options
  let male_role_options := 4
  -- Female role: 7 options
  let female_role_options := 7
  -- Remaining people: 3 men + 6 women = 9 people
  let remaining_people := 9
  -- Either-gender roles to assign: 4
  let either_gender_roles := 4
  -- Number of ways to choose 4 from 9
  have comb := Nat.choose remaining_people either_gender_roles
  -- Multiplying the options
  have total_ways := male_role_options * female_role_options * comb
  -- The final result
  show total_ways = 3528 from sorry

end role_assignment_ways_l259_259184


namespace sum_of_c_l259_259291

-- Define sequences a_n, b_n, and c_n
def a (n : ℕ) := 2 * n + 2
def b (n : ℕ) := 2 ^ (n + 1)
def c (n : ℕ) := a n - b n

-- State the main theorem
theorem sum_of_c (n : ℕ) : 
  ∑ i in Finset.range n, c i = n^2 + 3*n + 4 - 2^(n+2) := 
by 
  sorry

end sum_of_c_l259_259291


namespace tangent_f_a_b_tangent_g_valid_l259_259322

-- Proof that the given tangent line conditions yield correct a and b for the function f(x)
theorem tangent_f_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x : ℝ, x ≠ -1/2 → (f : ℝ → ℝ) = λ x, a * x^3 - x^2 - x + b)
  (h₃ : ∃ (x : ℝ), f' (-1/2) = 3/4) : a = 1 ∧ b = 5/8 :=
by
  sorry

-- Proof that verifies the tangent line validity for the function g(x)
theorem tangent_g_valid (g : ℝ → ℝ) (h₁ : ∀ x : ℝ, g x = (3 * real.sqrt real.exp 1) / 4 * real.exp x)
  (h₂ : ∃ (x : ℝ), g' (-1/2) = 3/4) : (g' (-1/2) = 3/4) ∧ (g (-1/2) = 3/4) :=
by
  sorry

end tangent_f_a_b_tangent_g_valid_l259_259322


namespace surveyed_households_count_l259_259952

theorem surveyed_households_count 
  (neither : ℕ) (only_R : ℕ) (both_B : ℕ) (both : ℕ) (h_main : Ξ)
  (H1 : neither = 80)
  (H2 : only_R = 60)
  (H3 : both = 40)
  (H4 : both_B = 3 * both) : 
  neither + only_R + both_B + both = 300 :=
by
  sorry

end surveyed_households_count_l259_259952


namespace time_per_harvest_is_three_months_l259_259471

variable (area : ℕ) (trees_per_m2 : ℕ) (coconuts_per_tree : ℕ) 
variable (price_per_coconut : ℚ) (total_earning_6_months : ℚ)

theorem time_per_harvest_is_three_months 
  (h1 : area = 20) 
  (h2 : trees_per_m2 = 2) 
  (h3 : coconuts_per_tree = 6) 
  (h4 : price_per_coconut = 0.50) 
  (h5 : total_earning_6_months = 240) :
    (6 / (total_earning_6_months / (area * trees_per_m2 * coconuts_per_tree * price_per_coconut)) = 3) := 
  by 
    sorry

end time_per_harvest_is_three_months_l259_259471


namespace focus_to_asymptote_distance_l259_259730

theorem focus_to_asymptote_distance (x y : ℝ) (h : x^2 - y^2 = 2) : 
  let a := sqrt(2)
  let b := sqrt(2)
  let c := sqrt(a^2 + b^2)
  let foci := (2, 0)
  let asymptote := λ x y, x + y = 0
  distance_from_focus_to_asymptote := sqrt(2) := 
by
  sorry

end focus_to_asymptote_distance_l259_259730


namespace find_general_term_l259_259191

variable (a : ℕ → ℝ) (a1 : a 1 = 1)

def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def isArithmeticSequence (u v w : ℝ) :=
  2 * v = u + w

theorem find_general_term (h1 : a 1 = 1)
  (h2 : (isGeometricSequence a (1 / 2)))
  (h3 : isArithmeticSequence (1 / a 1) (1 / a 3) (1 / a 4 - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

end find_general_term_l259_259191


namespace binomial_ratio_l259_259791

theorem binomial_ratio :
  let a := Nat.choose 10 5 in
  let b := Nat.choose 10 3 * (-2)^3 in
  b / a = - (80 / 21) := by
  sorry

end binomial_ratio_l259_259791


namespace integer_points_on_circle_l259_259997

theorem integer_points_on_circle (n : ℕ) : ∃ r : ℝ, ∃ (points : Finset (ℤ × ℤ)),
  (∀ (p ∈ points), p.1 * p.1 + p.2 * p.2 = r * r) ∧ points.card ≥ n :=
begin
  sorry
end

end integer_points_on_circle_l259_259997


namespace line_equations_satisfy_conditions_l259_259124

-- Definitions and conditions:
def intersects_at_distance (k m b : ℝ) : Prop :=
  |(k^2 + 7*k + 12) - (m*k + b)| = 8

def passes_through_point (m b : ℝ) : Prop :=
  7 = 2*m + b

def line_equation_valid (m b : ℝ) : Prop :=
  b ≠ 0

-- Main theorem:
theorem line_equations_satisfy_conditions :
  (line_equation_valid 1 5 ∧ passes_through_point 1 5 ∧ 
  ∃ k, intersects_at_distance k 1 5) ∨
  (line_equation_valid 5 (-3) ∧ passes_through_point 5 (-3) ∧ 
  ∃ k, intersects_at_distance k 5 (-3)) :=
by
  sorry

end line_equations_satisfy_conditions_l259_259124


namespace francie_has_3_dollars_remaining_l259_259675

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l259_259675


namespace hash_triple_application_l259_259232

def hash (N : ℕ) : ℕ := N^2 - N + 2

theorem hash_triple_application : hash (hash (hash 10)) = 70123304 :=
by
  sorry

end hash_triple_application_l259_259232


namespace range_of_a_l259_259278

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4 * x + 3 else - x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), a ≤ x → x ≤ a + 1 → f (x + a) > f (2 * a - x)) ↔ a < -2 := 
sorry

end range_of_a_l259_259278


namespace range_of_independent_variable_l259_259398

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259398


namespace equal_potatoes_l259_259802

theorem equal_potatoes (total_potatoes : ℕ) (total_people : ℕ) (h_potatoes : total_potatoes = 24) (h_people : total_people = 3) :
  (total_potatoes / total_people) = 8 :=
by {
  sorry
}

end equal_potatoes_l259_259802


namespace sheep_drowned_proof_l259_259579

def animal_problem_statement (S : ℕ) : Prop :=
  let initial_sheep := 20
  let initial_cows := 10
  let initial_dogs := 14
  let total_animals_made_shore := 35
  let sheep_drowned := S
  let cows_drowned := 2 * S
  let dogs_survived := initial_dogs
  let animals_made_shore := initial_sheep + initial_cows + initial_dogs - (sheep_drowned + cows_drowned)
  30 - 3 * S = 35 - 14

theorem sheep_drowned_proof : ∃ S : ℕ, animal_problem_statement S ∧ S = 3 :=
by
  sorry

end sheep_drowned_proof_l259_259579


namespace min_value_of_number_l259_259306

theorem min_value_of_number (a b c d : ℕ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 9) (h6 : 1 ≤ d) : 
  a + b * 10 + c * 100 + d * 1000 = 1119 :=
by
  sorry

end min_value_of_number_l259_259306


namespace symmetric_point_l259_259003

-- Defining the point P and the symmetry property with respect to xOy plane
def point_P := (-3, 2, -1 : ℝ × ℝ × ℝ)

def symmetric_about_xOy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, y, -z)

-- Statement of the problem
theorem symmetric_point : symmetric_about_xOy_plane point_P = (-3, 2, 1) := by
  sorry

end symmetric_point_l259_259003


namespace number_of_yogurts_l259_259248

def slices_per_yogurt : Nat := 8
def slices_per_banana : Nat := 10
def number_of_bananas : Nat := 4

theorem number_of_yogurts (slices_per_yogurt slices_per_banana number_of_bananas : Nat) : 
  slices_per_yogurt = 8 → 
  slices_per_banana = 10 → 
  number_of_bananas = 4 → 
  (number_of_bananas * slices_per_banana) / slices_per_yogurt = 5 :=
by
  intros h1 h2 h3
  sorry

end number_of_yogurts_l259_259248


namespace arrangements_with_AB_together_l259_259105

theorem arrangements_with_AB_together (n : ℕ) (A B: ℕ) (students: Finset ℕ) (h₁ : students.card = 6) (h₂ : A ∈ students) (h₃ : B ∈ students):
  ∃! (count : ℕ), count = 240 :=
by
  sorry

end arrangements_with_AB_together_l259_259105


namespace inversely_proportional_l259_259566

theorem inversely_proportional (X Y K : ℝ) (h : X * Y = K - 1) (hK : K > 1) : 
  (∃ c : ℝ, ∀ x y : ℝ, x * y = c) :=
sorry

end inversely_proportional_l259_259566


namespace clerical_staff_percentage_l259_259453

theorem clerical_staff_percentage (total_employees : ℕ) (initial_clerical_ratio : ℚ) 
  (first_reduction_ratio : ℚ) (second_reduction_ratio : ℚ) (new_non_clerical_hires : ℕ) :
  total_employees = 5500 →
  initial_clerical_ratio = 3 / 7 →
  first_reduction_ratio = 1 / 5 →
  second_reduction_ratio = 2 / 9 →
  new_non_clerical_hires = 100 →
  let initial_clerical_staff := (initial_clerical_ratio * total_employees).toNat,
      first_reduction := (first_reduction_ratio * initial_clerical_staff).toNat,
      after_first_reduction := initial_clerical_staff - first_reduction,
      second_reduction := (second_reduction_ratio * after_first_reduction).toNat,
      remaining_clerical_staff := after_first_reduction - second_reduction,
      total_remaining_employees := total_employees - first_reduction - second_reduction + new_non_clerical_hires in
  (real.to_rat remaining_clerical_staff / real.to_rat total_remaining_employees * 100).round ≈ 31.15 :=
begin
  intros,
  sorry
end

end clerical_staff_percentage_l259_259453


namespace train_to_platform_ratio_l259_259880

-- Define the given conditions as assumptions
def speed_kmh : ℕ := 54 -- speed of the train in km/hr
def train_length_m : ℕ := 450 -- length of the train in meters
def crossing_time_min : ℕ := 1 -- time to cross the platform in minutes

-- Conversion from km/hr to m/min
def speed_mpm : ℕ := (speed_kmh * 1000) / 60

-- Calculate the total distance covered in one minute
def total_distance_m : ℕ := speed_mpm * crossing_time_min

-- Define the length of the platform
def platform_length_m : ℕ := total_distance_m - train_length_m

-- The proof statement to show the ratio of the lengths
theorem train_to_platform_ratio : train_length_m = platform_length_m :=
by 
  -- following from the definition of platform_length_m
  sorry

end train_to_platform_ratio_l259_259880


namespace sum_of_parabola_distances_l259_259271

/--
  For each natural number \(n\), the parabola \(y = (n^2 + n)x^2 - (2n + 1)x + 1\) intersects the x-axis
  at points \(A_n\) and \(B_n\). Let \(|A_nB_n|\) denote the distance between these two points.
  Prove that the sum \(|A_1B_1| + |A_2B_2| + \ldots + |A_{1992}B_{1992}|\) is equal to the sum of
  \(\frac{1}{n(n+1)}\) from \(n=1\) to \(n=1992\).
-/
theorem sum_of_parabola_distances :
  (∑ n in (Finset.range 1992).map (λ x, x + 1), 1 / (n * (n + 1))) = 1 := sorry

end sum_of_parabola_distances_l259_259271


namespace taylor_scores_l259_259524

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l259_259524


namespace mon_inc_interval_cos_alpha_beta_area_triangle_ABC_l259_259281

noncomputable def f (x : ℝ) := Real.sin x + √3 * Real.cos x

theorem mon_inc_interval :
  ∀ k : ℤ, -((5 * Real.pi) / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 6) + 2 * k * Real.pi → 
  (∀ x, (f x) > (f x - ε)) :=
sorry

theorem cos_alpha_beta :
  ∀ alpha beta : ℝ, alpha > beta ∧ α ∈ [-π / 3, 5π / 3) ∧ β ∈ [-π / 3, 5π / 3) ∧ 
  (f alpha = 1 / 2) ∧ (f beta = 1 / 2) → Real.cos(α - β) = -7 / 8 :=
sorry

theorem area_triangle_ABC :
  ∀ (a b c : ℝ) (A B C : ℝ),
  c = √3 ∧ f C = 0 ∧ 
  (Real.sin A + Real.sin B = 2 * √(10) * Real.sin A * Real.sin B) →
  let area := (1 / 2) * a * b * Real.sin C in
  area = (3 * √3) / 20 :=
sorry

end mon_inc_interval_cos_alpha_beta_area_triangle_ABC_l259_259281


namespace arc_length_of_f_l259_259632

def f (x : ℝ) : ℝ := (1 - Real.exp x - Real.exp (-x)) / 2

theorem arc_length_of_f :
  ∫ x in 0..3, Real.sqrt (1 + (0.5 * (-Real.exp x + Real.exp (-x)))^2) = 0.5 * (Real.exp 3 - Real.exp (-3)) :=
by
  sorry

end arc_length_of_f_l259_259632


namespace pizza_combination_l259_259460

theorem pizza_combination : nat.choose 8 3 = 56 :=
by 
  -- Proof goes here. The question only asks for the statement.
  sorry

end pizza_combination_l259_259460


namespace total_men_wages_l259_259169

-- Define our variables and parameters
variable (M W B : ℝ)
variable (W_women : ℝ)

-- Conditions from the problem:
-- 1. 12M = WW (where WW is W_women)
-- 2. WW = 20B
-- 3. 12M + WW + 20B = 450
axiom eq_12M_WW : 12 * M = W_women
axiom eq_WW_20B : W_women = 20 * B
axiom eq_total_earnings : 12 * M + W_women + 20 * B = 450

-- Prove total wages of the men is Rs. 150
theorem total_men_wages : 12 * M = 150 := by
  sorry

end total_men_wages_l259_259169


namespace max_angle_ACD_l259_259808

/-- Given a right-angled triangle ABC with ∠ABC = 90°, and D on AB such that AD = 2DB,
    prove that the maximum possible value of ∠ACD is 30°. -/
theorem max_angle_ACD (A B C D : ℝ × ℝ) 
  (h_triangle : ∠B = 90°)
  (h_D_on_AB : collinear {A, B, D})
  (h_AD_2DB : dist A D = 2 * dist D B) :
  ∠ACD ≤ 30° :=
sorry

end max_angle_ACD_l259_259808


namespace samuel_breaks_2_cups_per_box_l259_259853

theorem samuel_breaks_2_cups_per_box:
  ∀ (total_boxes pans_boxes teacups_end : ℕ),
    total_boxes = 26 →
    pans_boxes = 6 →
    teacups_end = 180 →
    (2 * (total_boxes - pans_boxes) / 2) = 10 →
    5 * 4 = 20 →
    (total_boxes - pans_boxes) / 2 + (total_boxes - pans_boxes) / 2 = (teacups_end + ?m_1) / 20 →
    (?m_1 / ((total_boxes - pans_boxes) / 2 + (total_boxes - pans_boxes) / 2) = 2) := 
by
  intros total_boxes pans_boxes teacups_end h1 h2 h3 h4 h5 h6
  sorry

end samuel_breaks_2_cups_per_box_l259_259853


namespace problem_answer_l259_259732

def line_eq (m : ℝ) (x y : ℝ) := m * x - (m^2 + 1) * y = 4 * m
def circle_eq (x y : ℝ) := x^2 + y^2 - 8 * x + 4 * y + 16 = 0

-- Statements in the problem
def statement1 (m : ℝ) : Prop :=
  let k := m / (m^2 + 1)
  0 ≤ k ∧ k ≤ 1 / 2

def statement2 (m : ℝ) : Prop :=
  let k := m / (m^2 + 1)
  ¬(k = 0 ∨ k = 1 / 2)

def statement3 (m : ℝ) : Prop := 
  let k := m / (m^2 + 1)
  let d := 2 / Real.sqrt(1 + k^2)
  d > 1 / 2 ∧ ¬(d ≤ 2)

def statement4 (m : ℝ) : Prop :=
  let k := m / (m^2 + 1)
  let d := 2 / Real.sqrt(1 + k^2)
  2 * Real.sqrt(4 - (4 / Real.sqrt(5))^2) = 4 * Real.sqrt(5) / 5

-- Main theorem statement
theorem problem_answer :
  ∀ (m : ℝ), m ≥ 0 →
  ∃ (correct_statements : List (ℝ → Prop)),
    ({statement1, statement4} : List (ℝ → Prop)) →
    correct_statements = [statement1, statement4] := 
by sorry

end problem_answer_l259_259732


namespace external_tangent_b_value_l259_259227

theorem external_tangent_b_value:
  ∀ {C1 C2 : ℝ × ℝ} (r1 r2 : ℝ) (m b : ℝ),
  C1 = (3, -2) ∧ r1 = 3 ∧ 
  C2 = (15, 8) ∧ r2 = 8 ∧
  m = (60 / 11) →
  (∃ b, y = m * x + b ∧ b = 720 / 11) :=
by 
  sorry

end external_tangent_b_value_l259_259227


namespace find_a_l259_259307

variable {R : Type} [LinearOrderedField R]

-- Given conditions
def is_odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def f (a : R) (x : R) : R :=
  if x > 0 then 1 + a^x else -f a (-x)

axiom odd_f : is_odd_function (f a)
axiom pos_a : a > 0
axiom neq_one_a : a ≠ 1
axiom f_neg_one : f a (-1) = -3 / 2

-- Goal
theorem find_a (a : R) (h1 : is_odd_function (f a)) (h2 : 0 < a) (h3 : a ≠ 1) (h4 : f a (-1) = -3 / 2) : 
  a = 1 / 2 :=
by
  sorry

end find_a_l259_259307


namespace largest_n_in_binomial_expansion_l259_259360

theorem largest_n_in_binomial_expansion (n : ℕ) :
  (∀ k : ℕ, 2 * nat.choose n k = nat.choose n (k - 1) + nat.choose n (k + 1)) →
  n ≤ 959 :=
by
  sorry

end largest_n_in_binomial_expansion_l259_259360


namespace calculate_expression_l259_259987

theorem calculate_expression :
  √16 - 2 * Real.tan (Real.pi / 4) + abs (-3) + (Real.pi - 2023)^0 = 6 := by
  have h1 : √16 = 4 := by sorry
  have h2 : Real.tan (Real.pi / 4) = 1 := by sorry
  have h3 : abs (-3) = 3 := by sorry
  have h4 : (Real.pi - 2023)^0 = 1 := by sorry
  rw [h1, h2, h3, h4]
  norm_num

end calculate_expression_l259_259987


namespace black_square_area_l259_259199

theorem black_square_area (e : ℝ) (B : ℝ) (h_e : e = 15) (h_B : B = 500) :
  let total_surface_area := 6 * (e * e),
      blue_per_face := B / 6,
      face_area := e * e,
      black_area := face_area - blue_per_face
  in black_area = 141.67 := by 
  sorry

end black_square_area_l259_259199


namespace length_of_second_train_l259_259909

theorem length_of_second_train
  (length_first_train : ℕ)
  (speed_first_train speed_second_train : ℕ)
  (time_clear : ℝ) :
  length_first_train = 140 →
  speed_first_train = 42 →
  speed_second_train = 30 →
  time_clear = 20.99832013438925 →
  let relative_speed := (speed_first_train + speed_second_train) * (1000 / 3600) in
  let total_distance := relative_speed * time_clear in
  let length_second_train := total_distance - length_first_train in
  length_second_train = 280 :=
begin
  sorry
end

end length_of_second_train_l259_259909


namespace volume_of_soil_extracted_l259_259370

-- Define the dimensions of the pond
def length := 20 -- in meters
def width := 12 -- in meters
def height := 5 -- in meters

-- Define the volume calculation
def volume := length * width * height

-- State the theorem that the calculated volume is indeed 1200 cubic meters
theorem volume_of_soil_extracted : volume = 1200 := by
  -- Proof goes here
  sorry

end volume_of_soil_extracted_l259_259370


namespace smallest_repeating_decimal_l259_259611

theorem smallest_repeating_decimal : ∃ (n : ℕ), repeating_decimal 3.1415926 n = 3.141594141594... -- Desired number representing the repeating decimal
:= sorry

end smallest_repeating_decimal_l259_259611


namespace range_of_lambda_l259_259315

def Sn : ℕ → ℝ
| n := let A := 1
           B := -15 / 2 in
       A * n^2 + B * n

def bn (n : ℕ) : ℝ := n * Sn n

theorem range_of_lambda :
  (∃ λ : ℝ, ∀ n : ℕ, n * Sn n ≤ λ → (n = 4 ∨ n = 5 ∨ n = 6)) → 
  λ ∈ Ico (-54 : ℝ) (-81 / 2) :=
sorry

end range_of_lambda_l259_259315


namespace min_value_function_l259_259680

open Real

theorem min_value_function (x y : ℝ) 
  (hx : x > -2 ∧ x < 2) 
  (hy : y > -2 ∧ y < 2) 
  (hxy : x * y = -1) : 
  (∃ u : ℝ, u = (4 / (4 - x^2) + 9 / (9 - y^2)) ∧ u = 12 / 5) :=
sorry

end min_value_function_l259_259680


namespace new_average_mark_of_remaining_students_l259_259489

def new_average (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ) : ℕ :=
  ((total_students * avg_marks) - (excluded_students * excluded_avg_marks)) / (total_students - excluded_students)

theorem new_average_mark_of_remaining_students 
  (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ)
  (h1 : total_students = 33)
  (h2 : excluded_students = 3)
  (h3 : avg_marks = 90)
  (h4 : excluded_avg_marks = 40) : 
  new_average total_students excluded_students avg_marks excluded_avg_marks = 95 :=
by
  sorry

end new_average_mark_of_remaining_students_l259_259489


namespace maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l259_259784

-- Part (a): One blue cube
theorem maximum_amount_one_blue_cube : 
  ∃ (B : ℕ → ℚ) (P : ℕ → ℕ), (B 1 = 2) ∧ (∀ m > 1, B m = 2^m / P m) ∧ (P 1 = 1) ∧ (∀ m > 1, P m = m) ∧ B 100 = 2^100 / 100 :=
by
  sorry

-- Part (b): Exactly n blue cubes
theorem maximum_amount_n_blue_cubes (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 100) : 
  ∃ (B : ℕ × ℕ → ℚ) (P : ℕ × ℕ → ℕ), (B (1, 0) = 2) ∧ (B (1, 1) = 2) ∧ (∀ m > 1, B (m, 0) = 2^m) ∧ (P (1, 0) = 1) ∧ (P (1, 1) = 1) ∧ (∀ m > 1, P (m, 0) = 1) ∧ B (100, n) = 2^100 / Nat.choose 100 n :=
by
  sorry

end maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l259_259784


namespace sum_of_positive_integers_for_quadratic_l259_259094

theorem sum_of_positive_integers_for_quadratic :
  (∑ k in {k : ℕ | (∃ α β : ℤ, α * β = 18 ∧ α + β = k)}.to_finset) = 39 :=
by
  sorry

end sum_of_positive_integers_for_quadratic_l259_259094


namespace product_inequality_l259_259051

theorem product_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  let S : ℝ := ∑ i, x i in
  (∏ i, (1 + x i)) ≤ ∑ k in Finset.range (n + 1), S^k / Nat.factorial k :=
by
  sorry

end product_inequality_l259_259051


namespace problem1_simplification_problem2_solve_fraction_l259_259163

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l259_259163


namespace vector_addition_vector_collinear_l259_259743

open Real

-- Part (1): Prove the coordinates of the vector 3a + 4b
theorem vector_addition :
  let a := (1, -2: ℝ × ℝ)
  let b := (3, 4: ℝ × ℝ)
  3 • a + 4 • b = (15, 10: ℝ × ℝ) :=
by
  sorry

-- Part (2): Prove for k = -3/4, k•a - b is collinear with 3a + 4b
theorem vector_collinear :
  let a := (1, -2: ℝ × ℝ)
  let b := (3, 4: ℝ × ℝ)
  ∃ k : ℝ, k = -3 / 4 ∧ (k • a - b).fst = (15, 10: ℝ × ℝ).fst * ((k • a - b).snd / (15, 10: ℝ × ℝ).snd) :=
by
  sorry

end vector_addition_vector_collinear_l259_259743


namespace beetle_distance_l259_259171

theorem beetle_distance :
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  dist1 + dist2 = 20 :=
by
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  show dist1 + dist2 = 20
  sorry

end beetle_distance_l259_259171


namespace solution_y_l259_259085

theorem solution_y : ∃ y : ℚ, (8 * y^2 + 127 * y + 5) / (4 * y + 41) = 2 * y + 3
  := ∃ y, y = 118 / 33 → (8 * y^2 + 127 * y + 5) / (4 * y + 41) = 2 * y + 3 :=
begin
  use 118 / 33,
  sorry
end

end solution_y_l259_259085


namespace range_of_sqrt_x_plus_3_l259_259385

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259385


namespace quadratic_residue_property_l259_259757

theorem quadratic_residue_property (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ)
  (h : ∃ t : ℤ, ∃ k : ℤ, k * k = p * t + a) : (a ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_property_l259_259757


namespace vasya_guarantee_win_l259_259843

theorem vasya_guarantee_win (x : Fin 10 → ℝ) (cards : Finset (Finset (Fin 10)))
  (h1 : ∀ card ∈ cards, card.card = 5)
  (h2 : ∀ (i j : Fin 10), i ≤ j → x i ≤ x j) :
  ∃ s : Finset (Finset (Fin 10)), s.card = cards.card / 2 ∧
    (∑ card in s, ∏ i in card, x i) > (∑ card in cards \ s, ∏ i in card, x i) :=
by
  sorry

end vasya_guarantee_win_l259_259843


namespace integer_values_satisfying_abs_x_lt_2pi_l259_259343

theorem integer_values_satisfying_abs_x_lt_2pi :
  ∃ (n : ℕ), (n = 13) ∧ (set_of (λ x : ℤ, x ≠ 0 ∧ abs x < 2 * Real.pi)).card = n := by
  sorry

end integer_values_satisfying_abs_x_lt_2pi_l259_259343


namespace book_distribution_l259_259340

def total_books := 180
def senior_class_percentage := 0.4
def remaining_books := total_books * (1 - senior_class_percentage)
def junior_class_books := remaining_books * (4 / (4 + 5))
def middle_class_books := remaining_books * (5 / (4 + 5))

theorem book_distribution :
  junior_class_books = 48 ∧ middle_class_books = 60 := by
  sorry

end book_distribution_l259_259340


namespace rectangle_diagonal_perpendicular_length_l259_259469

theorem rectangle_diagonal_perpendicular_length (PQ QR : ℝ) 
    (PQ_value : PQ = 6) (QR_value : QR = 8) 
    (D P S X Y : Type) 
    (h1 : XY ⊥ PR)
    (h2 : P ∈ DX) (h3 : S ∈ DY) : 
    XY = 7 := 
by sorry

end rectangle_diagonal_perpendicular_length_l259_259469


namespace log_base2_0_3_lt_2_pow_0_3_l259_259638

theorem log_base2_0_3_lt_2_pow_0_3 : Real.log 2 0.3 < 2 ^ 0.3 := by
  sorry

end log_base2_0_3_lt_2_pow_0_3_l259_259638


namespace tripletA_correct_l259_259550

-- Definitions of the triplets
def tripletA : (ℝ × ℝ × ℝ) := (1/2, 3, -5/2)
def tripletB : (ℝ × ℝ × ℝ) := (2, -3, 3)
def tripletC : (ℝ × ℝ × ℝ) := (0.1, 0.4, 1.5)
def tripletD : (ℝ × ℝ × ℝ) := (1.2, -1.2, 1.8)
def tripletE : (ℝ × ℝ × ℝ) := (1, -1/2, 3/2)

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n.prime

-- Predicate to check if a real number is a whole number prime
def is_whole_prime (x : ℝ) : Prop := ∃ (n : ℕ), ↑n = x ∧ is_prime n

-- Function to count the number of prime numbers in a triplet
def count_primes (t : (ℝ × ℝ × ℝ)) : ℕ :=
  let (a, b, c) := t in 
  [a, b, c].countp is_whole_prime

-- Function to calculate the sum of the triplet
def sum_triplet (t : (ℝ × ℝ × ℝ)) : ℝ :=
  let (a, b, c) := t in a + b + c

-- Main theorem
theorem tripletA_correct :
  ∃ (t : (ℝ × ℝ × ℝ)), 
    t = tripletA ∧
    sum_triplet t ≠ 2 ∧
    count_primes t = 1 :=
by
  use tripletA
  split
  . rfl
  . split
    . norm_num
      -- Verify the sum of the tripletA is not equal to 2
    . norm_num
      -- Verify the count of primes in the tripletA are exactly 1
  sorry

end tripletA_correct_l259_259550


namespace count_valid_b_values_l259_259645

-- Definitions of the inequalities and the condition
def inequality1 (x : ℤ) : Prop := 3 * x > 4 * x - 4
def inequality2 (x b: ℤ) : Prop := 4 * x - b > -8

-- The main statement proving that the count of valid b values is 4
theorem count_valid_b_values (x b : ℤ) (h1 : inequality1 x) (h2 : inequality2 x b) :
  ∃ (b_values : Finset ℤ), 
    ((∀ b' ∈ b_values, ∀ x' : ℤ, inequality2 x' b' → x' ≠ 3) ∧ 
     (∀ b' ∈ b_values, 16 ≤ b' ∧ b' < 20) ∧ 
     b_values.card = 4) := by
  sorry

end count_valid_b_values_l259_259645


namespace inequality_proof_l259_259697

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l259_259697


namespace sqrt_add_sqrt_in_Q_l259_259257

noncomputable def pairs_meeting_condition : list (ℤ × ℤ) :=
  [(505, 505), (254, 254), (130, 130), (65, 65), (50, 50), (46, 46), (45, 45)]

theorem sqrt_add_sqrt_in_Q (m n : ℤ) :
  (sqrt (n + sqrt 2016) + sqrt (m - sqrt 2016)) ∈ ℚ ↔ (m, n) ∈ pairs_meeting_condition :=
sorry

end sqrt_add_sqrt_in_Q_l259_259257


namespace arithmetic_series_remainder_l259_259214

noncomputable def arithmetic_series_sum_mod (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d) / 2) % 10

theorem arithmetic_series_remainder :
  let a := 3
  let d := 5
  let n := 21
  arithmetic_series_sum_mod a d n = 3 :=
by
  sorry

end arithmetic_series_remainder_l259_259214


namespace sin_cos_15_degree_l259_259238

theorem sin_cos_15_degree :
  (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_15_degree_l259_259238


namespace car_speed_conversion_l259_259174

theorem car_speed_conversion (V_kmph : ℕ) (h : V_kmph = 36) : (V_kmph * 1000 / 3600) = 10 := by
  sorry

end car_speed_conversion_l259_259174


namespace smallest_n_for_g_gt_21_l259_259032

def g (n : ℕ) : ℕ := sorry

theorem smallest_n_for_g_gt_21 (h : ∃ k, n = 21 * k) (hg : ∀ n, g n > 21 ↔ n = 483) :
  ∃ n, n = 483 :=
by
  use 483
  sorry

end smallest_n_for_g_gt_21_l259_259032


namespace find_m_l259_259739

def A : Set ℤ := {-1, 1}
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

theorem find_m (m : ℤ) (h : B m ⊆ A) : m = 0 ∨ m = 1 ∨ m = -1 := 
sorry

end find_m_l259_259739


namespace base_s_eq_seven_l259_259005

theorem base_s_eq_seven (s : ℕ) :
  let cost := 5 * s ^ 2 + 3 * s + 0
      change := 4 * s ^ 2 + 5 * s + 5
      payment := s ^ 3 + 2 * s ^ 2 in
  cost + change = payment → s = 7 :=
by
  intro h
  sorry

end base_s_eq_seven_l259_259005


namespace points_on_ellipse_with_area_three_l259_259881

open Real

-- Definitions of the line, ellipse and intersection points
def line (x y : ℝ) := x / 4 + y / 3 = 1
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 3)

-- Definition of the area of a triangle using the determinant method
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

-- Definition of points on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Statement that needs to be proved
theorem points_on_ellipse_with_area_three :
  ∃ (P1 P2 : ℝ × ℝ),
    on_ellipse P1 ∧ on_ellipse P2 ∧ 
    triangle_area P1 A B = 3 ∧ triangle_area P2 A B = 3 :=
begin
  sorry
end

end points_on_ellipse_with_area_three_l259_259881


namespace find_number_that_satisfies_congruences_l259_259183

theorem find_number_that_satisfies_congruences :
  ∃ m : ℕ, 
  (m % 13 = 12) ∧ 
  (m % 11 = 10) ∧ 
  (m % 7 = 6) ∧ 
  (m % 5 = 4) ∧ 
  (m % 3 = 2) ∧ 
  m = 15014 :=
by
  sorry

end find_number_that_satisfies_congruences_l259_259183


namespace sqrt_domain_l259_259379

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259379


namespace number_of_fish_bought_each_year_l259_259528

-- Define the conditions
def initial_fish : ℕ := 2
def net_gain_each_year (x : ℕ) : ℕ := x - 1
def years : ℕ := 5
def final_fish : ℕ := 7

-- Define the problem statement as a Lean theorem
theorem number_of_fish_bought_each_year (x : ℕ) : 
  initial_fish + years * net_gain_each_year x = final_fish → x = 2 := 
sorry

end number_of_fish_bought_each_year_l259_259528


namespace taylor_scores_l259_259526

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l259_259526


namespace hyperbola_equation_l259_259713

theorem hyperbola_equation (focal_length : ℝ) (asymptote_perpendicular : ℝ → ℝ) (a b : ℝ) : 
    focal_length = 2 * Real.sqrt 5 → asymptote_perpendicular = -1 / 2 → a > 0 → b > 0 → a = 2 * b → 
    let c := Real.sqrt 5
    (c^2 = a^2 + b^2) →
    (a = 2) ∧ (b = 1) →
    (eqn : (x: ℝ) (y: ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1) → 
    eqn = (λ x y, (x^2 / 4) - y^2 = 1) :=
by
  intros
  sorry

end hyperbola_equation_l259_259713


namespace solve_trig_eq_l259_259146

theorem solve_trig_eq (a b : ℝ) (h : b ≠ 0) : 
  ∃ n k : ℤ, ∀ x : ℝ, a * cos (x / 2)^2 - (a + 2 * b) * sin (x / 2)^2 = a * cos x - b * sin x ↔ (x = 2 * n * π ∨ x = π / 2 * (4 * k + 1)) :=
by
  sorry

end solve_trig_eq_l259_259146


namespace kath_movie_cost_l259_259779

theorem kath_movie_cost :
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  discounted_price * number_of_people = 30 := by
  -- Definitions from conditions
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  -- Derived calculation based on conditions
  have h_discounted_price : discounted_price = 5 := by
    calc
      discounted_price = 8 - 3 : by sorry
      ... = 5 : by sorry
  have h_number_of_people : number_of_people = 6 := by
    calc
      number_of_people = 1 + 2 + 3 : by sorry
      ... = 6 : by sorry
  show 5 * 6 = 30 from sorry

end kath_movie_cost_l259_259779


namespace sqrt_domain_l259_259380

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259380


namespace impossible_to_visit_once_l259_259584

-- Condition: A cube with all vertices and face centers marked, and all face diagonals drawn.
def isMarkedPoint (p : ℕ) : Prop := 
  p < 8 ∨ (p >= 8 ∧ p < 14) -- 8 vertices + 6 face centers = 14 marked points

def isFaceDiagonal (from to : ℕ) : Prop :=
  (from < 8 ∧ to >= 8 ∧ to < 14) ∨ (to < 8 ∧ from >= 8 ∧ from < 14) 

-- Theorem stating the given problem
theorem impossible_to_visit_once :
  ¬ (∃ (path : ℕ → ℕ) (length : ℕ), 
    ∀ i < length, 
      isFaceDiagonal (path i) (path (i + 1)) ∧ 
      ∀ j k, j < length → k < length → j ≠ k → path j ≠ path k ∧ 
      isMarkedPoint (path 0) ∧ isMarkedPoint (path length)) := 
  sorry

end impossible_to_visit_once_l259_259584


namespace value_of_m_if_pure_imaginary_l259_259934

open Complex

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem value_of_m_if_pure_imaginary (m : ℝ) (i : ℂ) (hi : i = Complex.i) :
  is_pure_imaginary ((1 + m * i) * (2 - i)) → m = -2 := by
  rw [hi] at *
  sorry

end value_of_m_if_pure_imaginary_l259_259934


namespace number_of_regions_l259_259480

structure Point2D where
  x : ℤ
  y : ℤ

def A : Point2D := ⟨2, 2⟩
def B : Point2D := ⟨-2, 2⟩
def C : Point2D := ⟨-2, -2⟩
def D : Point2D := ⟨2, -2⟩
def E : Point2D := ⟨1, 0⟩
def F : Point2D := ⟨0, 1⟩
def G : Point2D := ⟨-1, 0⟩
def H : Point2D := ⟨0, -1⟩

theorem number_of_regions :
  let points := [A, B, C, D, E, F, G, H]
  -- number of regions formed:
  -- drawing line segments between every pair of points
  60 = sorry

end number_of_regions_l259_259480


namespace largest_possible_value_l259_259106

theorem largest_possible_value (x y z w : ℕ) (h : {x, y, z, w} = {1, 3, 5, 7}): 
  xy + yz + zw + wx ≤ 64 :=
by
  sorry

end largest_possible_value_l259_259106


namespace angle_bisectors_locus_property_medians_locus_property_altitudes_locus_property_l259_259287

noncomputable def locus_angle_bisectors (A B M : Point) 
  (h_AB_fix : A ≠ B ∧ ∀ M₁ M₂ ∈ Arc A B, ∠AMB₁ + ∠AMB₂ = π ∧ ∠AMB₁ ∈ Arc A B) : Set Point :=
  {P : Point | ∃ (γ : ℝ), ∠PAB = 2 * (π / 2 - γ / 2)}

noncomputable def radius_angle_bisectors (A B: Point) (r : ℝ) : ℝ :=
  2 * r * sin (π / 2)

theorem angle_bisectors_locus_property (A B M : Point) (r : ℝ)
  (h_AB_fix : A ≠ B ∧ ∀ M₁ M₂ ∈ Arc A B, ∠AMB₁ + ∠AMB₂ = π ∧ ∠AMB₁ ∈ Arc A B)
  (locus : Set Point) 
  (h_locus : locus = locus_angle_bisectors A B M h_AB_fix) :
  locus = {P : Point | ∃ γ ∈ Arc A B, dist P (midpoint A B) = radius_angle_bisectors A B r} :=
sorry

noncomputable def locus_medians (A B M N: Point)
  (h_midpoint : N = midpoint A B) : Set Point :=
  {P : Point | ∃ (G : Point), dist P G = 1 / 3 * dist N M}

theorem medians_locus_property (A B M N: Point)
  (h_midpoint : N = midpoint A B)
  (locus : Set Point)
  (h_locus : locus = locus_medians A B M N h_midpoint) :
  locus = {P : Point | ∃ γ ∈ Arc A B, P =  midpoint A B} :=
sorry

noncomputable def locus_altitudes (A B M: Point) : Set Point :=
  {P : Point | ∃ (H : Point), ∠PAH = π - ∠PBA}

noncomputable def radius_altitudes (A B: Point) (r : ℝ) : ℝ := r

theorem altitudes_locus_property (A B M: Point) (r: ℝ)
  (h_AB_fix : A ≠ B ∧ ∀ M₁ M₂ ∈ Arc A B, ∠AMB₁ + ∠AMB₂ = π ∧ ∠AMB₁ ∈ Arc A B)
  (locus : Set Point)
  (h_locus : locus = locus_altitudes A B M) :
  locus = {P : Point | ∃ γ ∈ Arc A B, dist P (midpoint A B) = radius_altitudes A B r} :=
sorry

end angle_bisectors_locus_property_medians_locus_property_altitudes_locus_property_l259_259287


namespace number_properties_l259_259206

-- Define what it means for a digit to be in a specific place
def digit_at_place (n place : ℕ) (d : ℕ) : Prop := 
  (n / 10 ^ place) % 10 = d

-- The given number
def specific_number : ℕ := 670154500

-- Conditions: specific number has specific digit in defined places
theorem number_properties : (digit_at_place specific_number 7 7) ∧ (digit_at_place specific_number 2 5) :=
by
  -- Proof of the theorem
  sorry

end number_properties_l259_259206


namespace translation_of_parabola_l259_259091

theorem translation_of_parabola (x : ℝ) :
  (λ x, 3 * (x - 2) ^ 2 + 1) (x - 2 + 2) = 3 * x ^ 2 + 1 :=
by
  sorry

end translation_of_parabola_l259_259091


namespace proof_problem_l259_259346

noncomputable def log (x : ℝ) : ℝ := real.log x

theorem proof_problem :
  let a := log 8
  let b := log 25
  5^(a/b) + 2^(b/a) = 2 * real.sqrt 2 + 5^(2/3) :=
by
  sorry

end proof_problem_l259_259346


namespace complement_A_union_B_l259_259811

def is_positive_integer_less_than_9 (n : ℕ) : Prop :=
  n > 0 ∧ n < 9

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def U := {n : ℕ | is_positive_integer_less_than_9 n}
noncomputable def A := {n ∈ U | is_odd n}
noncomputable def B := {n ∈ U | is_multiple_of_3 n}

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {2, 4, 8} :=
sorry

end complement_A_union_B_l259_259811


namespace inequality_proof_l259_259050

variable (a : Fin 8 → ℝ)
def x : ℝ := (1 / 8) * (∑ i, a i)
def y : ℝ := (1 / 8) * (∑ i, (a i)^2)

theorem inequality_proof
  (h_sorted : ∀ (i j : Fin 8), (i ≤ j) → a i ≤ a j) :
  a 7 - a 0 ≤ 4 * Real.sqrt (y a - (x a)^2) :=
sorry

end inequality_proof_l259_259050


namespace inequality_proof_l259_259700

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l259_259700


namespace disjoint_sets_l259_259889

def P : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => 4 * x^3 + 3 * x
| (n + 1), x => (4 * x^2 + 2) * P n x - P (n - 1) x

def A (m : ℝ) : Set ℝ := {x | ∃ n : ℕ, P n m = x }

theorem disjoint_sets (m : ℝ) : Disjoint (A m) (A (m + 4)) :=
by
  -- Proof goes here
  sorry

end disjoint_sets_l259_259889


namespace double_burger_cost_l259_259989

theorem double_burger_cost (D : ℝ) : 
  let single_burger_cost := 1.00
  let total_burgers := 50
  let double_burgers := 37
  let total_cost := 68.50
  let single_burgers := total_burgers - double_burgers
  let singles_cost := single_burgers * single_burger_cost
  let doubles_cost := total_cost - singles_cost
  let burger_cost := doubles_cost / double_burgers
  burger_cost = D := 
by 
  sorry

end double_burger_cost_l259_259989


namespace triangle_area_constant_l259_259309

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point :=
classical.some (circumcenter_exists A B C)

def reflection_across (p m : Point) : Point :=
⟨2 * m.x - p.x, 2 * m.y - p.y⟩

def midpoint (A B : Point) : Point :=
⟨(A.x + B.x)/2, (A.y + B.y)/2⟩

def M_points (A B C P : Point) : Point × Point × Point :=
let O := circumcenter A B C;
let P_A := reflection_across P (midpoint B C);
let X := reflection_across P_A (midpoint B C);
let M_A := midpoint X A;
let P_B := reflection_across P (midpoint C A);
let X_B := reflection_across P_B (midpoint C A);
let M_B := midpoint X_B B;
let P_C := reflection_across P (midpoint A B);
let X_C := reflection_across P_C (midpoint A B);
let M_C := midpoint X_C C;
(M_A, M_B, M_C)

theorem triangle_area_constant
  (A B C P : Point) :
  let (M_A, M_B, M_C) := M_points A B C P in
  ∃ k : ℝ, area M_A M_B M_C = k :=
by sorry

end triangle_area_constant_l259_259309


namespace rearrange_decimal_to_rational_l259_259076

theorem rearrange_decimal_to_rational (d : ℕ → ℕ) (h : ∀ n, d n < 10) (h_inf : ∃ n, ∀ m > n, d m = 0) :
  ∃ (d' : ℕ → ℕ), (∀ n, d' n < 10) ∧ d'.nat.digits.repeated_periodic :=
sorry

end rearrange_decimal_to_rational_l259_259076


namespace quadrilateral_angles_arith_prog_l259_259641

theorem quadrilateral_angles_arith_prog {x a b c : ℕ} (d : ℝ):
  (x^2 = 8^2 + 7^2 + 2 * 8 * 7 * Real.sin (3 * d)) →
  x = a + Real.sqrt b + Real.sqrt c →
  x = Real.sqrt 113 →
  a + b + c = 113 :=
by
  sorry

end quadrilateral_angles_arith_prog_l259_259641


namespace cos_value_of_inclined_line_l259_259711

variable (α : ℝ)
variable (l : ℝ) -- representing line as real (though we handle angles here)
variable (h_tan_line : ∃ α, tan α * (-1/2) = -1)

theorem cos_value_of_inclined_line (h_perpendicular : h_tan_line) :
  cos (2015 * Real.pi / 2 + 2 * α) = 4 / 5 := 
sorry

end cos_value_of_inclined_line_l259_259711


namespace find_index_l259_259505

-- Declaration of sequence being arithmetic with first term 1 and common difference 3
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 + (n - 1) * 3

-- The theorem to be proven
theorem find_index (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 672 = 2014) : 672 = 672 :=
by 
  sorry

end find_index_l259_259505


namespace function_domain_l259_259409

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259409


namespace sqrt_225_eq_15_l259_259568

theorem sqrt_225_eq_15 : Real.sqrt 225 = 15 :=
sorry

end sqrt_225_eq_15_l259_259568


namespace a_2012_value_l259_259686

open Nat

variable {a : ℕ → ℕ}

noncomputable def a_test (n : ℕ) (h : n > 0) : ℕ :=
  if h0 : (4 * (n / 2) + 1) = n then 0
  else if (4 * (n / 2) + 3) = n then 1
  else a (n / 2)

theorem a_2012_value :
  (∀ n : ℕ, n > 0 → a (4 * n - 3) = 1) →
  (∀ n : ℕ, n > 0 → a (4 * n - 1) = 0) →
  (∀ n : ℕ, n > 0 → a (2 * n) = a n) →
  a 2012 = 0 := by
  intros h1 h2 h3
  have h1006 : a 2012 = a 1006 := h3 1006 (Nat.succ_pos 1005)
  have h503 : a 1006 = a 503 := h3 503 (Nat.succ_pos 502)
  have h503_value : a 503 = 0 := h2 126 (Nat.succ_pos 125)
  rw [h1006, h503, h503_value]
  sorry

end a_2012_value_l259_259686


namespace compute_N_l259_259639

theorem compute_N : 
  let N := ∑ i in (finset.range 60), (2 * (60 - i) + 1)^2 + (2 * (60 - i))^2 
             - (2 * (60 - i) - 1)^2 - (2 * (60 - i) - 2)^2 
  ∑ i in (finset.range 60 by 3), N = 21840 :=
by
  sorry

end compute_N_l259_259639


namespace trig_identity_l259_259708

variable {α : Real}

theorem trig_identity (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
by
  sorry

end trig_identity_l259_259708


namespace combined_work_rate_one_day_l259_259149

-- Define the work rates of A and B
def work_rate_A (W : ℝ) := W / 4
def work_rate_B (W : ℝ) := W / 2

-- Define the combined work rate of A and B
def combined_work_rate (W : ℝ) := work_rate_A W + work_rate_B W

-- State the theorem
theorem combined_work_rate_one_day (W : ℝ) : combined_work_rate W = (3/4) * W :=
by
  -- This is the statement. The proof will be filled in later
  sorry

end combined_work_rate_one_day_l259_259149


namespace min_value_of_f_A_max_value_of_f_A_range_of_m_range_of_b_div_a_l259_259820

variables {a b x : ℝ}
variable {k : ℕ}

-- Assuming conditions
def f_A (x : ℝ) (a b : ℝ) := (x / a + b / x - 1)^2 - 2 * b / a + 1
def A := set.Ico a b -- Domain A = [a, b)

-- Correct answers from the solution
noncomputable def f_A_min (a b : ℝ) := 2 * (real.sqrt (b / a) - 1)^2
noncomputable def f_A_max (a b : ℝ) := (b / a - 1)^2
noncomputable def f_I_k_min (k : ℕ) := 2 / (k ^ 2 : ℝ)
noncomputable def f_I_k_plus_1_min (k : ℕ) := 2 / ((k + 1) ^ 2 : ℝ)

-- Given conditions
axiom h_a_pos : 0 < a
axiom h_b_pos : 0 < b
axiom h_a_lt_b : a < b

-- Proofs to be made
theorem min_value_of_f_A : f_A (real.sqrt (a * b)) a b = f_A_min a b := sorry
theorem max_value_of_f_A : f_A a a b = f_A_max a b := sorry

theorem range_of_m (m : ℝ) (hk : ∀ k, f_I_k_min k + f_I_k_plus_1_min k < m) : m > 5 / 2 := sorry

theorem range_of_b_div_a : 1 < b / a ∧ b / a < (2 * real.sqrt 2 - 1) ^ 2 := sorry

end min_value_of_f_A_max_value_of_f_A_range_of_m_range_of_b_div_a_l259_259820


namespace f_is_odd_l259_259420

noncomputable def f (x : ℝ) : ℝ := log (x + (1 + x^3)^(1/3))

theorem f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by
  intro x
  sorry

end f_is_odd_l259_259420


namespace inequality_conditions_l259_259819

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + B * C + C * A)) :=
by
  sorry

end inequality_conditions_l259_259819


namespace problem_quad_eq_and_line_quadrants_l259_259736

theorem problem_quad_eq_and_line_quadrants (m : ℝ) :
    (∀ x : ℝ, x^2 + (2 * m + 1) * x + m^2 + 2 = 0 → (2 * m + 1)^2 - 4 * (m^2 + 2) = 0) →
    m = 7 / 4 ∧
    (∀ x : ℝ, (2 * m - 3) * x - 4 * m + 6 = 1 / 2 * x - 1 →
    (x ≠ 0 → ((2 * m - 3) * x - 4 * m + 6 > 0 → (x > 0 → ((2 * m - 3) * x - 4 * m + 6 > 0 → x < 0)))) :=
sorry

end problem_quad_eq_and_line_quadrants_l259_259736


namespace probability_at_least_one_l259_259361

variable (p_A p_B : ℚ) (hA : p_A = 1 / 4) (hB : p_B = 2 / 5)

theorem probability_at_least_one (h : p_A * (1 - p_B) + (1 - p_A) * p_B + p_A * p_B = 11 / 20) : 
  (1 - (1 - p_A) * (1 - p_B) = 11 / 20) :=
by
  rw [hA, hB,←h]
  sorry

end probability_at_least_one_l259_259361


namespace geom_seq_a_n_l259_259684

theorem geom_seq_a_n (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -1) 
  (h_a7 : a 7 = -9) :
  a 5 = -3 :=
sorry

end geom_seq_a_n_l259_259684


namespace frustumViews_l259_259100

-- Define the notion of a frustum
structure Frustum where
  -- You may add necessary geometric properties of a frustum if needed
  
-- Define a function to describe the view of the frustum
def frontView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def sideView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type
def topView (f : Frustum) : Type := sorry -- Placeholder for the actual geometric type

-- Define the properties of the views
def isCongruentIsoscelesTrapezoid (fig : Type) : Prop := sorry -- Define property for congruent isosceles trapezoid
def isTwoConcentricCircles (fig : Type) : Prop := sorry -- Define property for two concentric circles

-- State the theorem based on the given problem
theorem frustumViews (f : Frustum) :
  isCongruentIsoscelesTrapezoid (frontView f) ∧ 
  isCongruentIsoscelesTrapezoid (sideView f) ∧ 
  isTwoConcentricCircles (topView f) := 
sorry

end frustumViews_l259_259100


namespace solve_for_x_l259_259086

theorem solve_for_x (x : ℚ) : (x = 70 / (8 - 3 / 4)) → (x = 280 / 29) :=
by
  intro h
  -- Proof to be provided here
  sorry

end solve_for_x_l259_259086


namespace range_of_a_l259_259714

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : f = (λ x, a * x + Real.sin x)) :
  (∃ x : ℝ, derivative f x = 0) ↔ (-1 < a ∧ a < 1) :=
sorry

end range_of_a_l259_259714


namespace range_of_a_if_exists_x_l259_259749

variable {a x : ℝ}

theorem range_of_a_if_exists_x :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ (a * x^2 - 1 ≥ 0)) → (a > 1) :=
by
  sorry

end range_of_a_if_exists_x_l259_259749


namespace proportion_margin_l259_259357

theorem proportion_margin (S M C : ℝ) (n : ℝ) (hM : M = S / n) (hC : C = (1 - 1 / n) * S) :
  M / C = 1 / (n - 1) :=
by
  sorry

end proportion_margin_l259_259357


namespace a_beats_b_by_18_meters_l259_259362

def distance_A : ℕ := 90
def time_A : ℕ := 20
def distance_B : ℕ := 90
def time_B : ℕ := 25

def speed (distance time : ℕ) : ℚ := distance / time

theorem a_beats_b_by_18_meters :
  let speed_A := speed distance_A time_A,
      speed_B := speed distance_B time_B,
      distance_B_in_time_A := speed_B * time_A,
      distance_beat := distance_A - distance_B_in_time_A
  in distance_beat = 18 := by
  sorry

end a_beats_b_by_18_meters_l259_259362


namespace part_a_part_b_l259_259562

-- Define the context of an angle BAC with a point P inside it
structure TriangleContext :=
  (A B C P : Point)
  (BAC : ∠BAC)
  (P_inside : PointInside ∠BAC P)

-- Part (a)
theorem part_a (context : TriangleContext) :
  ∃ X Y : Point, (LineThrough X Y context.P) ∧ (TriangleIsIsosceles context.A X Y) :=
begin
  sorry,
end

-- Part (b)
theorem part_b (context : TriangleContext) :
  ∃ X Y : Point, (LineThrough X Y context.P) ∧ (IsMidpoint context.P X Y) :=
begin
  sorry,
end

end part_a_part_b_l259_259562


namespace simplify_polynomial_l259_259633

theorem simplify_polynomial (x y : ℝ) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 :=
by
  sorry

end simplify_polynomial_l259_259633


namespace packets_needed_l259_259102

/-- Given 420 seedlings and each packet contains 7 seeds, 
we need 60 packets to contain all the seedlings. -/
theorem packets_needed (seedlings : ℕ) (seeds_per_packet : ℕ) (h_seedlings : seedlings = 420) 
  (h_seeds_per_packet : seeds_per_packet = 7) : (seedlings / seeds_per_packet) = 60 :=
by
  rw [h_seedlings, h_seeds_per_packet]
  norm_num
  exact rfl

end packets_needed_l259_259102


namespace total_friends_l259_259677

theorem total_friends (a b : ℕ) (seokjin_position_front seokjin_position_back : ℕ)
  (front_condition : seokjin_position_front = 8)
  (back_condition : seokjin_position_back = 6) :
  a + b + 1 = 13 :=
by
  have front_friends : a = seokjin_position_front - 1 := by sorry
  have back_friends : b = seokjin_position_back - 1 := by sorry
  rw [front_condition, back_condition] at front_friends back_friends
  rw [front_friends, back_friends]
  exact sorry

end total_friends_l259_259677


namespace emily_can_see_emerson_for_18_minutes_l259_259250

-- Define constants for the speeds of Emily and Emerson
def Emily_speed : ℝ := 15
def Emerson_speed : ℝ := 10

-- Define the distance ahead and behind that Emily can see Emerson
def distance_ahead : ℝ := 3 / 4
def distance_behind : ℝ := 3 / 4

-- Define the total time Emily can see Emerson and prove it to be 18 minutes
theorem emily_can_see_emerson_for_18_minutes :
  (distance_ahead / (Emily_speed - Emerson_speed) + distance_behind / (Emily_speed - Emerson_speed)) * 60 = 18 :=
by
  sorry

end emily_can_see_emerson_for_18_minutes_l259_259250


namespace four_does_not_divide_a2008_l259_259888

def sequence : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 1) := if n = 0 then 1 else sequence (n - 1) * sequence n + 1

theorem four_does_not_divide_a2008 : ¬ (4 ∣ (sequence 2008)) :=
sorry

end four_does_not_divide_a2008_l259_259888


namespace fabric_cut_l259_259341

/-- Given a piece of fabric that is 2/3 meter long,
we can cut a piece measuring 1/2 meter
by folding the original piece into four equal parts and removing one part. -/
theorem fabric_cut :
  ∃ (f : ℚ), f = (2/3 : ℚ) → ∃ (half : ℚ), half = (1/2 : ℚ) ∧ half = f * (3/4 : ℚ) :=
by
  sorry

end fabric_cut_l259_259341


namespace problem_I_problem_II_l259_259936

theorem problem_I :
  let balls := {1, 2, 3, 4}
  let outcomes := {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
  let favorable := {(1, 2), (1, 3)}
  (favorable.size : ℕ) / (outcomes.size : ℕ) = 1 / 3 := by
  sorry

theorem problem_II :
  let balls := {1, 2, 3, 4}
  let outcomes := {(m, n) | m in balls, n in balls}
  let favorable := outcomes.filter (fun (m, n) => n < m + 2)
  (favorable.size : ℕ) / (outcomes.size : ℕ) = 13 / 16 := by
  sorry

end problem_I_problem_II_l259_259936


namespace find_min_value_l259_259745

-- Define a structure to represent vectors in 2D space
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the condition for perpendicular vectors (dot product is zero)
def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

-- Define the problem: given vectors a = (m, 1) and b = (1, n - 2)
-- with conditions m > 0, n > 0, and a ⊥ b, then prove the minimum value of 1/m + 2/n
theorem find_min_value (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0)
  (h₂ : perpendicular ⟨m, 1⟩ ⟨1, n - 2⟩) :
  (1 / m + 2 / n) = (3 + 2 * Real.sqrt 2) / 2 :=
  sorry

end find_min_value_l259_259745


namespace smallest_sin_x_l259_259042

theorem smallest_sin_x (x y z : ℝ) (hx : real.sin x = real.sec y)
  (hy : real.sin y = real.sec z) (hz : real.sin z = real.sec x) : real.sin x = 0 :=
sorry

end smallest_sin_x_l259_259042


namespace sqrt_of_9_l259_259166

theorem sqrt_of_9 : Real.sqrt 9 = 3 :=
by 
  sorry

end sqrt_of_9_l259_259166


namespace ratio_x_to_y_l259_259535

theorem ratio_x_to_y (x y : ℤ) (h : (10*x - 3*y) / (13*x - 2*y) = 3 / 5) : x / y = 9 / 11 := 
by sorry

end ratio_x_to_y_l259_259535


namespace student_A_can_always_ensure_solution_l259_259689

-- Define the concept of replacing asterisks in a sequence of equations
def replace_asterisks (eqns : List (List Option ℝ)) : ℕ → ℕ → ℝ → List (List Option ℝ) :=
  λ row col val,
    eqns.update_nth row (eqns.nth row).getD [].update_nth col (some val)

-- Define the strategy conditions for student A to win
def student_A_wins_strategy : List (List Option ℝ) → Prop :=
  λ initial_eqns,
    ∀ eqns, ∀ turn, -- Assume that for every state of the equations and turn number
      replace_asterisks_eqns_turn initial_eqns turn eqns →
        (turn % 2 = 0 → tactic_for_player_A eqns) ∧ 
        (turn % 2 = 1 → tactic_for_player_B eqns)

theorem student_A_can_always_ensure_solution :
  ∀ initial_eqns : List (List Option ℝ),
    length initial_eqns = 7 →
    (∀ i, length (initial_eqns.nth i).getD [] = i + 1) →
    student_A_wins_strategy initial_eqns :=
by
  intros
  sorry

end student_A_can_always_ensure_solution_l259_259689


namespace extend_line_segment_opposite_direction_l259_259919

-- Define points A and B
variables (A B : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B]

-- Define line segment AB and the ray coming from A passing towards B
def line_segment (A B : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] : set A := { x : A | ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ x = λ • (B - A) + A }
def ray (A B : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] : set A := { x : A | ∃ λ : ℝ, 0 ≤ λ ∧ x = λ • (B - A) + A }

-- The proof problem to show that Ray BA is a valid geometric construction
theorem extend_line_segment_opposite_direction (A B : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] :
  ∃ C : Type, ray B A ⊆ ray C A := sorry

end extend_line_segment_opposite_direction_l259_259919


namespace log_sum_eq_minus_two_l259_259763

theorem log_sum_eq_minus_two (a : ℝ) (ha : a = 2) :
  log a (2 / 7) + log (1 / a) (8 / 7) = -2 := by
sorry

end log_sum_eq_minus_two_l259_259763


namespace radio_selling_price_l259_259491

theorem radio_selling_price (CP LP Loss SP : ℝ) (h1 : CP = 1500) (h2 : LP = 11)
  (h3 : Loss = (LP / 100) * CP) (h4 : SP = CP - Loss) : SP = 1335 := 
  by
  -- hint: Apply the given conditions.
  sorry

end radio_selling_price_l259_259491


namespace knight_can_tour_201x201_l259_259073

/-- Prove that a knight can traverse a (4n+1) x (4n+1) chessboard where n = 50 in such a way that it visits each square exactly once. -/
theorem knight_can_tour_201x201 :
  ∀ (n : ℕ), n = 50 → ∃ tour : List (ℕ × ℕ),
  (∀ x y, (x, y) ∈ tour ↔ (1 ≤ x ∧ x ≤ (4 * n + 1) ∧ 1 ≤ y ∧ y <= (4 * n + 1))) ∧
  ∀ i, i < (4 * n + 1) * (4 * n + 1) - 1 → 
  (nat.abs (tour.nth_le i.succ sorry).fst - (tour.nth_le i sorry).fst, 
   nat.abs (tour.nth_le i.succ sorry).snd - (tour.nth_le i sorry).snd) ∈ {(1, 2), (2, 1)} :=
begin
  intros n hn,
  use sorry, -- Existential witness (knight's tour) is presumed,
  split,
  { -- Proof that all squares are visited
    intros x y,
    simp, sorry -- To be filled with the specifics of the knight's tour
  },
  { -- Proof that every move is a valid knight move
    intros i hi,
    sorry -- To be filled with the specifics of the knight's move verification
  }
end


end knight_can_tour_201x201_l259_259073


namespace probability_of_ant_reaching_C_l259_259616

noncomputable def probability_ending_at_C : ℚ :=
by
  -- Definition of the problem setup
  let total_time : ℕ := 6  -- total time in minutes
  let total_blue_dots : ℕ := 5  -- number of possible blue positions after 6 minutes
  
  -- Assuming uniform probability distribution due to random movement and symmetry
  let probability_C : ℚ := 1 / total_blue_dots
  exact probability_C

theorem probability_of_ant_reaching_C :
  probability_ending_at_C = 1 / 5 :=
by
  apply probability_ending_at_C

end probability_of_ant_reaching_C_l259_259616


namespace count_valid_candidates_l259_259503

open Nat

def is_valid_candidate (n : ℕ) : Prop :=
  ∃ x y z, (show_comp (10 * (10 * (10 * 2 + x)+y)+ z), 
  2 ≠ x ∧ 2 ≠ y ∧ 2 ≠ z ∧ (x = y ∨ y = z ∨ z = x) ∧ 2 = n / 1000 

theorem count_valid_candidates : 
   (card {n : ℕ | is_valid_candidate n}) )
where card means total number:
=
336
:= sorry

end count_valid_candidates_l259_259503


namespace solve_equation_l259_259148

theorem solve_equation (x : ℕ) (h : x > 1) : ( (3 / 5) ^ (2 * log (x + 1) / log 9) * (125 / 27) ^ (log (x - 1) / log 27) = log 27 / log 243 ) ↔ x = 2 := 
sorry

end solve_equation_l259_259148


namespace number_of_divisors_of_2018_or_2019_is_7_l259_259747

theorem number_of_divisors_of_2018_or_2019_is_7 (h1 : Prime 673) (h2 : Prime 1009) : 
  Nat.card {d : Nat | d ∣ 2018 ∨ d ∣ 2019} = 7 := 
  sorry

end number_of_divisors_of_2018_or_2019_is_7_l259_259747


namespace radii_proof_l259_259848

-- Define the main parameters and circles
variables (A B : Point) (O_0 : Point) (r_0 : ℝ)
variables (k_0 : Circle O_0 r_0)

-- Define the initial conditions
axiom AB_diameter_k0 : diameter k_0 A B
axiom O0_center_k0 : center k_0 O_0

-- Define the radii to prove
def r_1 := r_0 / 2
def r_2 := r_0 / 3
def r_3 := r_0 / 6
def r_4 := r_0 / 4
def r_5 := r_0 / 7
def r_6 := r_0 / 8

theorem radii_proof : 
  ∃ (r_1 r_2 r_3 r_4 r_5 r_6 : ℝ), 
  r_1 = r_0 / 2 ∧ 
  r_2 = r_0 / 3 ∧ 
  r_3 = r_0 / 6 ∧ 
  r_4 = r_0 / 4 ∧ 
  r_5 = r_0 / 7 ∧ 
  r_6 = r_0 / 8 :=
by {
  -- Assign values to radii
  let r_1 := r_0 / 2,
  let r_2 := r_0 / 3,
  let r_3 := r_0 / 6,
  let r_4 := r_0 / 4,
  let r_5 := r_0 / 7,
  let r_6 := r_0 / 8,
  -- Prove the radii equalities
  use [r_1, r_2, r_3, r_4, r_5, r_6], 
  -- Provide the equality propositions
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩,
}

end radii_proof_l259_259848


namespace find_k_values_l259_259654

def vector_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.fst ^ 2 + v.snd ^ 2)

def k_values (k : ℝ) : Prop :=
  vector_norm (k * (3, -4) - (5, 8)) = 3 * Real.sqrt 13

theorem find_k_values : {k : ℝ // k_values k} :=
sorry

end find_k_values_l259_259654


namespace cyclic_quadrilateral_inequality_l259_259438

theorem cyclic_quadrilateral_inequality
  {A B C D : ℝ} -- assume A, B, C, D are real numbers representing points on a circle
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D)
  (h_cyclic : ∀ P Q R S, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ P ≠ R ∧ Q ≠ S → (AB = max AB BC CD DA))
  : AB + BD > AC + CD := 
sorry

end cyclic_quadrilateral_inequality_l259_259438


namespace max_l_and_a_l259_259053

def f (a x : ℝ) := a * x ^ 2 + 8 * x + 3

def l (a : ℝ) : ℝ := if a ≠ -8 then (2 / (sqrt (16 + 2 * a) + 4)) else ((sqrt 5 + 1) / 2)

theorem max_l_and_a (a : ℝ) (h_a : a < 0) : 
  (maximizes (λ a, l a) a (-8)) ∧ (l (-8) = (sqrt 5 + 1) / 2) :=
begin
  sorry
end

end max_l_and_a_l259_259053


namespace geometric_sequence_sum_l259_259683

variable {α : Type*} [NormedField α] [CompleteSpace α]

def geometric_sum (a r : α) (n : ℕ) : α :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → α) (a r : α) (hS : ∀ n, S n = geometric_sum a r n) :
  S 2 = 6 → S 4 = 30 → S 6 = 126 :=
by
  sorry

end geometric_sequence_sum_l259_259683


namespace find_S₆_l259_259509

noncomputable def a₁ : ℝ := 1 / 2
noncomputable def S₄ : ℝ := 20

noncomputable def sum_n (n : ℕ) : ℝ → ℝ → ℝ
| 0, _ => 0
| n + 1, d => a₁ + (n * d) + sum_n n (d)

theorem find_S₆ (d : ℝ) (h₁ : S₄ = 4 * a₁ + 6 * d) : sum_n 6 d = 48 :=
by
  have : d = 3 := by linarith [h₁]
  rw this
  have : sum_n 6 3 = 48 := by
    unfold sum_n
    norm_num
  exact this

#eval find_S₆ 3 rfl -- Testing if the theorem can be evaluated and gives the expected result

end find_S₆_l259_259509


namespace smallest_prime_with_even_reverse_l259_259667

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to reverse the digits of a number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

-- The proposition to prove
theorem smallest_prime_with_even_reverse :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (reverse_digits n % 2 = 0) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ is_prime m ∧ (reverse_digits m % 2 = 0)) → n ≤ m :=
begin
  use 23,
  split, norm_num, -- 10 ≤ 23
  split, norm_num, -- 23 < 100
  -- Proof of primality of 23
  sorry,
  -- reverse_digits 23 == 32, and 32 is even
  split, norm_num [reverse_digits], -- (reverse_digits 23) % 2 = 0
  -- Check smallest condition
  intros m,
  intro hm,
  rcases hm with ⟨h1, h2, h3, h4⟩,
  -- proving the chosen number is the smallest
  sorry,
end

end smallest_prime_with_even_reverse_l259_259667


namespace ellipse_properties_l259_259691

theorem ellipse_properties (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : a > b)
  (eccentricity : c = a * (√3 / 2))
  (focus1 : -c) (focus2 : c)
  (points : ∃ A B : ℝ × ℝ, A.1 = -7 * focus2 ∧ B.1 = focus2 
    ∧ dist A (-c, 0) = 7 * dist A (c, 0))
  (slope1 : ∃ M N : ℝ × ℝ, (M.2 - M.1) = focus1 ∧ 
    (dist M (0, 0) * dist N (0, 0)) = 5 * dist M N  * (8 * b^2) / 25
    ∧ (sqrt 3 * b / 2) * (dist M N) = (2 * sqrt 6) / 5 )
  :
  (c / a = √3/2) ∧ (a^2 = 4 ∧ b^2 = 1)
  ∧ equation : (axonometry_ellipse := ∀ x y : ℝ, x^2 / 4 + y^2 = 1) :=
sorry

end ellipse_properties_l259_259691


namespace obtain_any_natural_from_4_l259_259551

/-- Definitions of allowed operations:
  - Append the digit 4.
  - Append the digit 0.
  - Divide by 2, if the number is even.
--/
def append4 (n : ℕ) : ℕ := 10 * n + 4
def append0 (n : ℕ) : ℕ := 10 * n
def divide2 (n : ℕ) : ℕ := n / 2

/-- We'll also define if a number is even --/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Define the set of operations applied on a number --/
inductive operations : ℕ → ℕ → Prop
| initial : operations 4 4
| append4_step (n m : ℕ) : operations n m → operations n (append4 m)
| append0_step (n m : ℕ) : operations n m → operations n (append0 m)
| divide2_step (n m : ℕ) : is_even m → operations n m → operations n (divide2 m)

/-- The main theorem proving that any natural number can be obtained from 4 using the allowed operations --/
theorem obtain_any_natural_from_4 (n : ℕ) : ∃ m, operations 4 m ∧ m = n :=
by sorry

end obtain_any_natural_from_4_l259_259551


namespace printing_machine_completion_time_l259_259957

theorem printing_machine_completion_time :
    ∀ (start_time completion_fraction task_completion_time total_completion_time : ℕ)
    (start completion : time) (one_fourth total_hours completion_time target_completion_time : real) 
    (start_hour completion_hour : ℕ), 
    start = 9 -- Start time in hours from midnight
    ∧ completion = 12.5 -- Equivalent of 12:30 PM in fractional hours
    ∧ one_fourth = 0.25 -- One fourth fraction of the day's work
    ∧ task_completion_time = 12.5 - 9 -- Time to complete one fourth of the work
    ∧ task_completion_time = 3.5 
    ∧ total_completion_time = 4 * task_completion_time -- Total time to complete the work
    ∧ total_completion_time = 14.0 
    ∧ target_completion_time = 9 + total_completion_time -- Time to complete all tasks
    ∧ target_completion_time = 23.0
    → target_completion_time = 23.0 := 
by sorry 

end printing_machine_completion_time_l259_259957


namespace proof_problem_l259_259347

noncomputable def log (x : ℝ) : ℝ := real.log x

theorem proof_problem :
  let a := log 8
  let b := log 25
  5^(a/b) + 2^(b/a) = 2 * real.sqrt 2 + 5^(2/3) :=
by
  sorry

end proof_problem_l259_259347


namespace simplify_expression_solve_fractional_eq_l259_259160

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l259_259160


namespace bike_ride_distance_l259_259807

-- Definitions for conditions from a)
def speed_out := 24 -- miles per hour
def speed_back := 18 -- miles per hour
def total_time := 7 -- hours

-- Problem statement for the proof problem
theorem bike_ride_distance :
  ∃ (D : ℝ), (D / speed_out) + (D / speed_back) = total_time ∧ 2 * D = 144 :=
by {
  sorry
}

end bike_ride_distance_l259_259807


namespace table_price_l259_259194

variable (C T : ℝ)

theorem table_price :
  (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 64) → T = 56 :=
by
  intro h,
  obtain ⟨h1, h2⟩ := h,
  -- This is where the proof would go
  sorry

end table_price_l259_259194


namespace candies_total_on_thursday_to_saturday_l259_259247

theorem candies_total_on_thursday_to_saturday (
  a : ℕ,
  h₁ : a + (a + 1) + (a + 2) = 504
) : (a + 3) + (a + 4) + (a + 5) = 513 := by 
sorry

end candies_total_on_thursday_to_saturday_l259_259247


namespace angle_sum_in_triangle_l259_259795

theorem angle_sum_in_triangle {X Y Z W : Type} [add_comm_group X] [module ℝ X]
  (angle_XYZ : ℝ) (angle_YWZ : ℝ) 
  (H1 : angle_XYZ = 70)
  (H2 : angle_YWZ = 40)
  (angle_bisector : ∀ WZ : ℝ, WZ = angle_YWZ / 2) 
  : ∃ angle_YXZ : ℝ, angle_YXZ = 90 := 
by
  sorry

end angle_sum_in_triangle_l259_259795


namespace math_proof_problem_l259_259694

noncomputable def question1 (k : ℝ) (m : ℕ) : Bool :=
  ∃ k' : ℝ, k = k' ∧ m ≥ 3 ∧ k' ≥ Real.sqrt 3

noncomputable def question2 (k : ℝ) (m : ℕ) (AB CD : ℝ) : Bool :=
  ∃ k' : ℝ, (k = k' ∧ m = 4 ∧ 0 < (|AB| / |CD|) ∧ (|AB| / |CD|) < 4)

theorem math_proof_problem (k : ℝ) (m : ℕ) (AB CD : ℝ) :
  (∃ x : ℝ, x^2 = - (k * x - 3)) ∧ (∃ x : ℝ, x^2 = 4 * (k * x - 3)) →
  question1 k m ∧ question2 k m (Real.sqrt 16 * k^2 - 48) (Real.sqrt k^2 + 12) :=
by
  sorry

end math_proof_problem_l259_259694


namespace updated_mean_corrected_l259_259883

theorem updated_mean_corrected (mean observations decrement : ℕ) 
  (h1 : mean = 350) (h2 : observations = 100) (h3 : decrement = 63) :
  (mean * observations + decrement * observations) / observations = 413 :=
by
  sorry

end updated_mean_corrected_l259_259883


namespace find_line_equation_l259_259098

open Real

noncomputable def line_equation (a b c : ℝ) (x y : ℝ) := a * x + b * y + c = 0

theorem find_line_equation : 
    (∃ P : ℝ × ℝ, line_equation 7 5 (-24) P.1 P.2 ∧ line_equation 1 (-1) 0 P.1 P.2) →
    (∃ l : ℝ × ℝ × ℝ, (∃ k : ℝ, l = (k, -1, 2 - 2*k) ∧ ∃ P : ℝ × ℝ, P = (5, 1) ∧ 
    ∀ Q : ℝ × ℝ, line_equation l.1 l.2 l.3 Q.1 Q.2 →
    sqrt ((Q.1 - 5)^2 + (Q.2 - 1)^2) = sqrt 10)) →
    l = (3, -1, -4) :=
sorry

end find_line_equation_l259_259098


namespace first_space_shuttle_1981_l259_259767

def shuttle_name (name : String) : Prop :=
  name = "Columbia"

theorem first_space_shuttle_1981 : ∃ name : String, name = "Columbia" ∧
  (name = "Houston" ∨ name = "New York" ∨ name = "Columbia" ∨ name = "Apollo") ∧
  (developed_by_US name) ∧
  (manned_first_flight name) ∧
  (year_of_flight name = 1981) :=
by
  sorry

-- Auxiliary Conditions
def developed_by_US (name : String) : Prop :=
  name = "Columbia" -- the only specific provided

def manned_first_flight (name : String) : Prop :=
  name = "Columbia" -- derived from provided

def year_of_flight (name : String) : Nat :=
  if name = "Columbia" then 1981 else 0

end first_space_shuttle_1981_l259_259767


namespace find_equilateral_triangle_side_length_l259_259906

def flag_pole_heights := (11, 13)
def equilateral_triangle_side_length := 8 * Real.sqrt 3

theorem find_equilateral_triangle_side_length
  (h1 h2 : ℕ) (h1_eq : h1 = 11) (h2_eq : h2 = 13) :
  ∃ x : ℝ, (∃ (h_eq : x = 8 * Real.sqrt 3), 
           let height := (Real.sqrt 3 / 2 * x) in 
           height = h2 - h1) :=
begin
  use 8 * Real.sqrt 3,
  split,
  { refl },
  { rw [h1_eq, h2_eq],
    norm_num }
end

end find_equilateral_triangle_side_length_l259_259906


namespace angle_between_hands_at_7_35_l259_259136

def degrees (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60.0) * 360.0
  let hour_angle := (hour % 12) * 30.0 + (minute / 60.0) * 30.0
  abs (hour_angle - minute_angle)

theorem angle_between_hands_at_7_35 :
  degrees 7 35 = 17.5 := sorry

end angle_between_hands_at_7_35_l259_259136


namespace number_of_pupils_l259_259185

theorem number_of_pupils (m1 m2 : ℕ) (avg_increase : ℝ) (h1 : m1 = 83) (h2 : m2 = 63) (h3 : avg_increase = 1 / 2) :
  let n := (m1 - m2) / avg_increase in
  n = 40 :=
by
  let m1 := 83
  let m2 := 63
  let avg_increase := (1 / 2)
  let n := ((m1 - m2) : ℝ) / avg_increase
  have h4 : m1 - m2 = 20 := by norm_num
  have h5 : 20 / (1 / 2) = 40 := by norm_num
  rw [h4, h5]
  exact eq.refl 40 -- Use reflexivity of equality: 40 = 40

end number_of_pupils_l259_259185


namespace single_layer_cake_slices_l259_259629

/-- Prove that the number of single layer cake slices Dusty buys is 7 given the conditions --/
theorem single_layer_cake_slices (x : ℕ) :
  let cost_single_slice := 4
  let cost_double_slice := 7
  let num_double_slices := 5
  let total_money := 100
  let change := 37
  let cost_double := num_double_slices * cost_double_slice
  let total_spent := total_money - change
  4 * x + cost_double = total_spent → x = 7 :=
by 
  intros
  let cost_single_slice := 4
  let cost_double_slice := 7
  let num_double_slices := 5
  let cost_double := num_double_slices * cost_double_slice
  let total_money := 100
  let change := 37
  let total_spent := total_money - change
  trivial -- sorry

end single_layer_cake_slices_l259_259629


namespace f_is_x_plus_1_l259_259046

def f : ℚ → ℚ 

axiom f_one : f 1 = 2

axiom f_property : ∀ (x y : ℚ), f (x * y) = f x * f y - f (x + y) + 1

theorem f_is_x_plus_1 : ∀ (x : ℚ), f x = x + 1 :=
by
  intro x
  sorry

end f_is_x_plus_1_l259_259046


namespace sum_series_l259_259222

theorem sum_series :
  (∑ n in Finset.range 100, 1 / ((2 * (n + 1) - 3) * (2 * (n + 1) + 5))) = 612 / 1640 :=
by
  sorry

end sum_series_l259_259222


namespace prove_smallest_geometric_third_term_value_l259_259962

noncomputable def smallest_value_geometric_third_term : ℝ :=
  let d_1 := -5 + 10 * Real.sqrt 2
  let d_2 := -5 - 10 * Real.sqrt 2
  let g3_1 := 39 + 2 * d_1
  let g3_2 := 39 + 2 * d_2
  min g3_1 g3_2

theorem prove_smallest_geometric_third_term_value :
  smallest_value_geometric_third_term = 29 - 20 * Real.sqrt 2 := by sorry

end prove_smallest_geometric_third_term_value_l259_259962


namespace inequality_of_equilateral_triangle_l259_259800

/-- Definition of an equilateral triangle -/
structure EquilateralTriangle :=
(A B C : Point)
(equilateral : A.distance B = B.distance C ∧ B.distance C = C.distance A)

/-- The main theorem to prove the inequality -/
theorem inequality_of_equilateral_triangle
    (ABC : EquilateralTriangle)
    (A1 : Point)
    (h1 : isInInterior A1 ABC)
    (A2 : Point)
    (h2 : isInInterior A2 (triangle A1 ABC.B ABC.C)) :
    let S1 := area (triangle A1 ABC.B ABC.C)
    let P1 := perimeter (triangle A1 ABC.B ABC.C)
    let S2 := area (triangle A2 ABC.B ABC.C)
    let P2 := perimeter (triangle A2 ABC.B ABC.C)
in
    (S1 / P1^2 > S2 / P2^2) :=
sorry

end inequality_of_equilateral_triangle_l259_259800


namespace collinear_M_P_N_l259_259044

open LinearAlgebra

variables {A B C I D E P M N W : Point}
          {BC AC AB AI DE : Line}

-- Definitions
-- Let's represent our geometrical setup as described:
def incenter (I : Point) (A B C : Point) : Prop := sorry  -- Definition of incenter I of triangle ABC
def incircle_touches (D E : Point) (BC AC : Line) : Prop := sorry -- Definition of points where the incircle touches BC at D and AC at E
def intersection_of_lines (P : Point) (AI DE : Line) : Prop := sorry -- P is the intersection point of lines AI and DE
def midpoints (M N : Point) (BC AB : Line) : Prop := sorry -- M and N are midpoints of BC and AB

-- The theorem to be proved
theorem collinear_M_P_N (h1: incenter I A B C)
                        (h2: incircle_touches D E BC AC)
                        (h3: intersection_of_lines P AI DE)
                        (h4: midpoints M N BC AB) : collinear M N P :=
sorry

end collinear_M_P_N_l259_259044


namespace difference_eq_divisible_by_99_l259_259966

variable (a b c : ℕ)

def original_num := 100 * a + 10 * b + c
def new_num := 100 * c + 10 * b + a

theorem difference_eq : (new_num c b a - original_num a b c) = -99 * a + 99 * c := by
  sorry

theorem divisible_by_99 : ((new_num c b a - original_num a b c) % 99) = 0 := by
  sorry

end difference_eq_divisible_by_99_l259_259966


namespace value_of_x2_minus_y2_l259_259348

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l259_259348


namespace department_store_earnings_l259_259948

theorem department_store_earnings :
  let original_price : ℝ := 1000000
  let discount_rate : ℝ := 0.1
  let prizes := [ (5, 1000), (10, 500), (20, 200), (40, 100), (5000, 10) ]
  let A_earnings := original_price * (1 - discount_rate)
  let total_prizes := prizes.foldl (fun sum (count, amount) => sum + count * amount) 0
  let B_earnings := original_price - total_prizes
  (B_earnings - A_earnings) >= 32000 := by
  sorry

end department_store_earnings_l259_259948


namespace percent_receive_tickets_approx_10_l259_259835

variable (total_motorists : ℝ)
variable (percent_exceed_limit : ℝ := 14.285714285714285 / 100)
variable (percent_not_ticketed : ℝ := 30 / 100)

noncomputable def percent_receive_tickets : ℝ :=
  let percent_ticketed := 1 - percent_not_ticketed
  percent_ticketed * percent_exceed_limit * 100

theorem percent_receive_tickets_approx_10 :
  percent_receive_tickets total_motorists ≈ (10 : ℝ) :=
sorry

end percent_receive_tickets_approx_10_l259_259835


namespace waiter_tips_earned_l259_259565

theorem waiter_tips_earned (total_customers tips_left no_tip_customers tips_per_customer : ℕ) :
  no_tip_customers + tips_left = total_customers ∧ tips_per_customer = 3 ∧ no_tip_customers = 5 ∧ total_customers = 7 → 
  tips_left * tips_per_customer = 6 :=
by
  intro h
  sorry

end waiter_tips_earned_l259_259565


namespace Vasya_wins_l259_259841

theorem Vasya_wins :
  ∃ (assign_values : (Fin 10 → ℝ)) (x : Fin 10 → ℝ) (cards_Vasya cards_Petya : Finset (Fin 5 → ℝ)),
  (∀ i, 0 ≤ x i) ∧
  (∀ i j, i ≤ j → x i ≤ x j) ∧
  (cards_Vasya ∪ cards_Petya = Finset.univ) ∧
  (cards_Vasya ∩ cards_Petya = ∅) ∧
  (∑ c in cards_Vasya, ∏ i, x (c i)) > (∑ c in cards_Petya, ∏ i, x (c i)) := 
sorry

end Vasya_wins_l259_259841


namespace correctTechnologyUsedForVolcanicAshMonitoring_l259_259887

-- Define the choices
inductive Technology
| RemoteSensing : Technology
| GPS : Technology
| GIS : Technology
| DigitalEarth : Technology

-- Define the problem conditions
def primaryTechnologyUsedForVolcanicAshMonitoring := Technology.RemoteSensing

-- The statement to prove
theorem correctTechnologyUsedForVolcanicAshMonitoring : primaryTechnologyUsedForVolcanicAshMonitoring = Technology.RemoteSensing :=
by
  sorry

end correctTechnologyUsedForVolcanicAshMonitoring_l259_259887


namespace brown_eyed_brunettes_l259_259771

theorem brown_eyed_brunettes (total_girls blondes brunettes blue_eyed_blondes brown_eyed_girls : ℕ) 
    (h1 : total_girls = 60) 
    (h2 : blondes + brunettes = total_girls) 
    (h3 : blue_eyed_blondes = 20) 
    (h4 : brunettes = 35) 
    (h5 : brown_eyed_girls = 22) 
    (h6 : blondes = total_girls - brunettes) 
    (h7 : brown_eyed_blondes = blondes - blue_eyed_blondes) :
  brunettes - (brown_eyed_girls - brown_eyed_blondes) = 17 :=
by sorry  -- Proof is not required

end brown_eyed_brunettes_l259_259771


namespace problem_divisibility_l259_259029

theorem problem_divisibility (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by
  sorry

end problem_divisibility_l259_259029


namespace problem1_simplification_problem2_solve_fraction_l259_259161

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l259_259161


namespace smallest_n_with_gn_greater_than_21_l259_259035

def g (n : ℕ) : ℕ :=
  Inf { k : ℕ | k > 0 ∧ factorial k ∣ n }

theorem smallest_n_with_gn_greater_than_21 (r : ℕ) (hr : r ≥ 22) :
  let n := 21 * r in g n > 21 ↔ n = 462 := by
  sorry

end smallest_n_with_gn_greater_than_21_l259_259035


namespace lions_deers_killing_time_l259_259756

theorem lions_deers_killing_time (n : ℕ) 
  (h1 : ∀ x : ℕ, x ≥ 1 → (14 * x / 14 = x)) 
  (h2 : ∀ y : ℕ, y ≥ 1 → (100 * y / 100 = y)) :
  (14 * n / 14 = n) ∧ (100 * n / 100 = n) → 14 :=
by
  intro h
  cases h,
  sorry

end lions_deers_killing_time_l259_259756


namespace ratio_of_area_of_JKLM_to_ABCD_l259_259088

section

variables {s : ℝ} (A B C D J K L M : ℝ → ℝ × ℝ)
variable (h_squareABCD : ∀ (a b c d : ℝ → ℝ × ℝ), 
    A (0, 0) ∧ B (3 * s, 0) ∧ C (3 * s, 3 * s) ∧ D (0, 3 * s))
variable (h_squareJKLM : ∀ (j k l m : ℝ → ℝ × ℝ), 
    J (2 * s, 0) ∧ K (3 * s, s) ∧ L (3 * s, 2 * s + s) ∧ M (2 * s + s, 3 * s))

theorem ratio_of_area_of_JKLM_to_ABCD
    (h_ratio : ∀ (AJ JB : ℝ), AJ = 2 * JB) :
    JKLM.area / ABCD.area = 2 / 9 :=
sorry

end

end ratio_of_area_of_JKLM_to_ABCD_l259_259088


namespace checkerboard_sum_l259_259581

theorem checkerboard_sum :
  let rows := 12 
  let cols := 16 
  let f (i j : ℕ) := 16 * (i - 1) + j
  let g (i j : ℕ) := 12 * (j - 1) + i 
  let points := [(1, 1), (4, 5), (7, 9), (10, 13)]
  let matched_numbers := points.map (λ (i, j), f i j)
  matched_numbers.sum = 364 :=
by {
  sorry
}

end checkerboard_sum_l259_259581


namespace simplify_and_evaluate_expression_l259_259857

theorem simplify_and_evaluate_expression (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) : 
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) :=
by
  sorry

example : (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ (x = 3) ∧ ((x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = 5)) :=
  ⟨3, by norm_num, by norm_num, rfl, by norm_num⟩

end simplify_and_evaluate_expression_l259_259857


namespace num_sets_N_l259_259327

open Set

noncomputable def M : Set ℤ := {-1, 0}

theorem num_sets_N (N : Set ℤ) : M ∪ N = {-1, 0, 1} → 
  (N = {1} ∨ N = {0, 1} ∨ N = {-1, 1} ∨ N = {0, -1, 1}) := 
sorry

end num_sets_N_l259_259327


namespace count_non_integer_angles_l259_259817

theorem count_non_integer_angles : 
  (Set.filter (λ (n : ℕ), (3 ≤ n ∧ n ≤ 15) ∧ n.Prime ∧ ¬ ∀ k : ℕ, 180 * (n - 2) = k * n)
    (Set.Icc 3 15)).card = 3 :=
by 
  sorry

end count_non_integer_angles_l259_259817


namespace quadrilateral_is_trapezoid_l259_259809

theorem quadrilateral_is_trapezoid (A B C D O : Type)
  [has_area_triangles : ∀ (P Q R : Type), triangle_area P Q R]
  (h1 : triangle_area A O B = 1)
  (h2 : triangle_area B O C = 2)
  (h3 : triangle_area C O D = 4) :
  triangle_area A O D = 2 ∧ is_trapezoid A B C D :=
sorry

end quadrilateral_is_trapezoid_l259_259809


namespace parallel_vectors_eq_l259_259339

theorem parallel_vectors_eq (m : ℝ) :
  ((m, 1) = (λ k : ℝ, k • (2, m-1)) → m = 2 ∨ m = -1) ∧
  ((2, m-1) = (λ k : ℝ, k • (m, 1)) → m = 2 ∨ m = -1) :=
begin
  sorry
end

end parallel_vectors_eq_l259_259339


namespace chord_intersect_challenge_l259_259992

theorem chord_intersect_challenge
  (A B C D E F G M : Point)
  (circle : Circle)
  (h1 : on_circle A circle)
  (h2 : on_circle B circle)
  (h3 : on_circle C circle)
  (h4 : on_circle D circle)
  (AB_∩CD_E : intersect (line_through A B) (line_through C D) = E)
  (M_on_BE : on_segment M B E)
  (TE_tangent_DEM : is_tangent (circle_of_triangle D E M) (line_through E F G))
  (TE_intersect_BC_F : intersect (line_through E F G) (line_through B C) = F)
  (TE_intersect_AC_G : intersect (line_through E F G) (line_through A C) = G)
  (AM_over_AB_t : ∃ (t : ℝ), t = length (segment A M) / length (segment A B)) :
  length (segment B G) / length (segment E F) = t / (1 - t) :=
sorry

end chord_intersect_challenge_l259_259992


namespace sum_gcd_lcm_of_4_and_10_l259_259538

theorem sum_gcd_lcm_of_4_and_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 :=
by
  sorry

end sum_gcd_lcm_of_4_and_10_l259_259538


namespace bess_milk_daily_l259_259652

-- Definitions based on conditions from step a)
variable (B : ℕ) -- B is the number of pails Bess gives every day

def BrownieMilk : ℕ := 3 * B
def DaisyMilk : ℕ := B + 1
def TotalDailyMilk : ℕ := B + BrownieMilk B + DaisyMilk B

-- Conditions definition to be used in Lean to ensure the equivalence
axiom weekly_milk_total : 7 * TotalDailyMilk B = 77
axiom daily_milk_eq : TotalDailyMilk B = 11

-- Prove that Bess gives 2 pails of milk everyday
theorem bess_milk_daily : B = 2 :=
by
  sorry

end bess_milk_daily_l259_259652


namespace hypotenuse_altitude_legs_relation_l259_259999

-- Defining the problem given the conditions stated above.
variable (c a b m : ℝ)

-- Stating the conditions of the problem.
def right_triangle_conditions : Prop :=
  a^2 - b^2 = 4 * m^2 ∧ m^2 = (c / 4) * (c - c / 4)

-- The correct answer derived from the problem.
theorem hypotenuse_altitude_legs_relation (hc : right_triangle_conditions c a b m) : 
  (c / 4) = (c * (1 + Real.sqrt 5)) / 4 :=
begin
  -- proof to be completed
  sorry
end

end hypotenuse_altitude_legs_relation_l259_259999


namespace sales_tax_percentage_l259_259583

theorem sales_tax_percentage 
  (total_spent : ℝ)
  (tip_percent : ℝ)
  (food_price : ℝ) 
  (total_with_tip : total_spent = food_price * (1 + tip_percent / 100))
  (sales_tax_percent : ℝ) 
  (total_paid : total_spent = food_price * (1 + sales_tax_percent / 100) * (1 + tip_percent / 100)) :
  sales_tax_percent = 10 :=
by sorry

end sales_tax_percentage_l259_259583


namespace ratio_of_areas_l259_259963

theorem ratio_of_areas {r : ℝ} (h : r > 0) :
  let chord_length := 2 * r,
      square_side := r * (2 - Real.sqrt 3),
      area_square := (square_side)^2,
      area_circle := Real.pi * r^2
  in area_square / area_circle = (7 - 4 * Real.sqrt 3) / Real.pi :=
by
  sorry

end ratio_of_areas_l259_259963


namespace inequality_proof_l259_259702

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l259_259702


namespace total_red_papers_l259_259790

-- Defining the number of red papers in one box and the number of boxes Hoseok has
def red_papers_per_box : ℕ := 2
def number_of_boxes : ℕ := 2

-- Statement to prove
theorem total_red_papers : (red_papers_per_box * number_of_boxes) = 4 := by
  sorry

end total_red_papers_l259_259790


namespace piecewise_function_evaluation_l259_259320

def f (x : ℝ) : ℝ :=
  if x < 1 then x + 1 else -x + 3

theorem piecewise_function_evaluation :
  f (f (5 / 2)) = 3 / 2 :=
by
  sorry

end piecewise_function_evaluation_l259_259320


namespace part_I_part_II_part_III_l259_259726

def f (x a : ℝ) : ℝ := 2 * x ^ 3 - a * x ^ 2 + 1

theorem part_I (m : ℝ) (a : ℝ) (h : a = 6) (ht : ∀ x : ℝ, f x a = -6 * x + m → deriv (f x a) x = -6) :
  m = 3 := sorry

theorem part_II (a : ℝ) (h₁ : ∃ x > 0, f x a = 0) (h₂ : a = 3) :
  ∀ x : ℝ, (x < 0 → deriv (f x a) x > 0) ∧ (x > 1 → deriv (f x a) x > 0) ∧ (0 < x ∧ x < 1 → deriv (f x a) x < 0) := sorry

theorem part_III (a : ℝ) (h₁ : 0 < a) (h₂ : (max (f 0 a) (f 1 a)) + (min (f (-1) a) (f (a/3) a)) = 1) :
  a = 1/2 := sorry

end part_I_part_II_part_III_l259_259726


namespace area_under_g_l259_259026

noncomputable def g : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 6 then x^2 else if 6 ≤ x ∧ x ≤ 10 then 3 * x - 10 else 0

theorem area_under_g : 
  let A1 := ∫ (x : ℝ) in 0..6, g x
  let A2 := ∫ (x : ℝ) in 6..10, g x
  R = A1 + A2 →
  R = 128 := 
by
  sorry

end area_under_g_l259_259026


namespace triangle_inequality_iff_inequality_l259_259463

theorem triangle_inequality_iff_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  sorry

end triangle_inequality_iff_inequality_l259_259463


namespace diameter_increase_l259_259177

theorem diameter_increase (D D' : ℝ) (h : π * (D' / 2) ^ 2 = 2.4336 * π * (D / 2) ^ 2) : D' / D = 1.56 :=
by
  -- Statement only, proof is omitted
  sorry

end diameter_increase_l259_259177


namespace triangle_inequality_equality_condition_l259_259462

variables {A B C a b c : ℝ}

theorem triangle_inequality (A a B b C c : ℝ) :
  A * a + B * b + C * c ≥ 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

theorem equality_condition (A B C a b c : ℝ) :
  (A * a + B * b + C * c = 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b)) ↔ (a = b ∧ b = c ∧ A = B ∧ B = C) :=
sorry

end triangle_inequality_equality_condition_l259_259462


namespace constant_term_coefficient_l259_259283

noncomputable def integral_value : ℝ :=
  ∫ x in 0..2, 2 * x + 1

theorem constant_term_coefficient : integral_value = 6 → 
  (∀ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), ((√x + (2 / x))^integral_value).eval 0 = 60) :=
sorry

end constant_term_coefficient_l259_259283


namespace sqrt_eq_four_implies_x_is_144_l259_259752

theorem sqrt_eq_four_implies_x_is_144 (x : ℝ) (h : sqrt (4 + sqrt x) = 4) : x = 144 := 
by
  sorry

end sqrt_eq_four_implies_x_is_144_l259_259752


namespace inequality_proof_l259_259068

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l259_259068


namespace range_of_sqrt_x_plus_3_l259_259387

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259387


namespace find_multiple_l259_259597

theorem find_multiple (x m : ℝ) (hx : x = 3) (h : x + 17 = m * (1 / x)) : m = 60 := 
by
  sorry

end find_multiple_l259_259597


namespace find_ratio_l259_259412

-- Define the geometric sequence properties and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions stated in the problem
axiom h₁ : a 5 * a 11 = 3
axiom h₂ : a 3 + a 13 = 4

-- The goal is to find the values of a_15 / a_5
theorem find_ratio (h₁ : a 5 * a 11 = 3) (h₂ : a 3 + a 13 = 4) :
  ∃ r : ℝ, r = a 15 / a 5 ∧ (r = 3 ∨ r = 1 / 3) :=
sorry

end find_ratio_l259_259412


namespace polynomial_roots_l259_259655

theorem polynomial_roots:
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end polynomial_roots_l259_259655


namespace mass_percentage_H_l259_259262

theorem mass_percentage_H (mass_percentage_H_is_169 : ∀ compound : Type, mass_percentage_H compound = 1.69) :
  ∀ compound : Type, mass_percentage_H compound = 1.69 :=
by
  intro compound
  apply mass_percentage_H_is_169
  
-- Since the problem requires only the statement and not the solution, the "sorry" is used to skip unnecessary proof details.

end mass_percentage_H_l259_259262


namespace loop_condition_l259_259720

theorem loop_condition (b : ℕ) : (b = 10 ∧ ∀ n, b = 10 + 3 * n ∧ b < 16 → n + 1 = 16) → ∀ (condition : ℕ → Prop), condition b → b = 16 :=
by sorry

end loop_condition_l259_259720


namespace polynomial_degree_l259_259646

noncomputable def polynomial : Polynomial ℝ := (X^6 + 2*X^9 + 3) * (2*X^4 + 5*X^3 + 1) * (X^2 + 3*X + 4)

theorem polynomial_degree : polynomial.degree = 15 :=
sorry

end polynomial_degree_l259_259646


namespace price_of_second_oil_l259_259168

open Real

-- Define conditions
def litres_of_first_oil : ℝ := 10
def price_per_litre_first_oil : ℝ := 50
def litres_of_second_oil : ℝ := 5
def total_volume_of_mixture : ℝ := 15
def rate_of_mixture : ℝ := 55.67
def total_cost_of_mixture : ℝ := total_volume_of_mixture * rate_of_mixture

-- Define total cost of the first oil
def total_cost_first_oil : ℝ := litres_of_first_oil * price_per_litre_first_oil

-- Define total cost of the second oil in terms of unknown price P
def total_cost_second_oil (P : ℝ) : ℝ := litres_of_second_oil * P

-- Theorem to prove price per litre of the second oil
theorem price_of_second_oil : ∃ P : ℝ, total_cost_first_oil + (total_cost_second_oil P) = total_cost_of_mixture ∧ P = 67.01 :=
by
  sorry

end price_of_second_oil_l259_259168


namespace sin_half_angle_product_inequality_l259_259971

theorem sin_half_angle_product_inequality 
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : 0 < A) (h3 : A < 180)
  (h4 : 0 < B) (h5 : B < 180)
  (h6 : 0 < C) (h7 : C < 180)
  : sin (A / 2) * sin (B / 2) * sin (C / 2) < 1 / 4 :=
  sorry

end sin_half_angle_product_inequality_l259_259971


namespace committee_meeting_people_l259_259556

theorem committee_meeting_people (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 11) : 
  A + B = 7 :=
sorry

end committee_meeting_people_l259_259556


namespace value_of_m_l259_259754

theorem value_of_m (m : ℚ) : 
  (m = - -(-(1/3) : ℚ) → m = -1/3) :=
by
  sorry

end value_of_m_l259_259754


namespace polynomial_roots_l259_259662

open Polynomial

theorem polynomial_roots :
  (roots (C 1 * X^4 + C (-3) * X^3 + C 3 * X^2 + C (-1) * X + C (-6))).map (λ x, x.re) = 
    {1 - sqrt 3, 1 + sqrt 3, (1 - sqrt 13) / 2, (1 + sqrt 13) / 2} :=
by sorry

end polynomial_roots_l259_259662


namespace contractor_absence_l259_259180

-- Define the parameters and the proof statement
theorem contractor_absence (x y : ℕ) (h1 : x + y = 30) (h2 : 25 * x - 7.5 * y = 620) : y = 4 :=
by
  -- We provide the outline of the proof here
  sorry

end contractor_absence_l259_259180


namespace shaded_area_percentage_l259_259141

theorem shaded_area_percentage (side_length : ℕ) (shaded_length1 shaded_length2 shaded_length3 : ℕ) :
  side_length = 7 →
  shaded_length1 = 1 →
  shaded_length2 = 2 →
  shaded_length3 = 3 →
  let total_area := side_length^2 in
  let shaded_area1 := shaded_length1^2 in
  let shaded_area2 := (shaded_length1 + shaded_length2) * (shaded_length1 + 4) - shaded_area1 in
  let shaded_area3 := (shaded_length1 + shaded_length2 + shaded_length3) * (1 + 6) - shaded_area2 in
  let total_shaded_area := shaded_area1 + shaded_area2 + shaded_area3 in
  (total_shaded_area / total_area : ℚ) = 42 / 49 :=
by
  intros h1 h2 h3 h4
  sorry

end shaded_area_percentage_l259_259141


namespace range_of_a_l259_259735

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Icc (-2 : ℝ) 1 → x^2 + 2 * x + a ≥ 0) → a ∈ Icc (1 : ℝ) (⊤ : ℝ) := by
  sorry

end range_of_a_l259_259735


namespace no_such_triples_l259_259856

theorem no_such_triples 
  (a b c : ℕ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : ¬ ∃ k, k ∣ a + c ∧ k ∣ b + c ∧ k ∣ a + b) 
  (h₃ : c^2 ∣ a + b) 
  (h₄ : b^2 ∣ a + c) 
  (h₅ : a^2 ∣ b + c) : 
  false :=
sorry

end no_such_triples_l259_259856


namespace exists_point_X_with_condition_l259_259439

-- Define the problem context and conditions
variables {n : ℕ} (A : fin n → ℝ × ℝ) (polygon : set (ℝ × ℝ)) (X : ℝ × ℝ)
  (inside_polygon : X ∈ polygon)

-- Define B_i as the second intersection
def B (i : fin n) : ℝ × ℝ := sorry -- A precise geometric construction for B_i is required

-- Define the geometric condition
def condition (i : fin n) : Prop :=
  dist X (A i) / dist X (B i) ≤ 2

-- Main statement of the theorem
theorem exists_point_X_with_condition :
  ∃ (X : ℝ × ℝ) (hX : X ∈ polygon), ∀ (i : fin n), condition A polygon X B i :=
sorry

end exists_point_X_with_condition_l259_259439


namespace problem_l259_259814

def f (n : ℕ) : ℤ := 3 ^ (2 * n) - 32 * n ^ 2 + 24 * n - 1

theorem problem (n : ℕ) (h : 0 < n) : 512 ∣ f n := sorry

end problem_l259_259814


namespace arc_length_l259_259865

theorem arc_length (circumference : ℝ) (angle : ℝ) (h1 : circumference = 72) (h2 : angle = 45) :
  ∃ length : ℝ, length = 9 :=
by
  sorry

end arc_length_l259_259865


namespace solution_l259_259061

noncomputable def problem_statement : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.mul (Polynomial.X ^ 2 - 4 * Polynomial.X + 4) p = Polynomial.X ^ 4 + 16

theorem solution :
  ∃ (p : Polynomial ℝ), Polynomial.mul (Polynomial.X ^ 2 - 4 * Polynomial.X + 4) p = Polynomial.X ^ 4 + 16 :=
sorry

end solution_l259_259061


namespace madeline_money_l259_259055

variable (M B : ℝ)

theorem madeline_money :
  B = 1/2 * M →
  M + B = 72 →
  M = 48 :=
  by
    intros h1 h2
    sorry

end madeline_money_l259_259055


namespace replace_circles_sums_equal_l259_259371

theorem replace_circles_sums_equal (A B C D : ℕ) (h1 : A ∈ {6, 7, 8, 9})
                                   (h2 : B ∈ {6, 7, 8, 9})
                                   (h3 : C ∈ {6, 7, 8, 9})
                                   (h4 : D ∈ {6, 7, 8, 9})
                                   (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) 
                                   (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D)
                                   (h11 : A + C + 3 + 4 = 5 + D + 2 + 4)
                                   (h12 : A + B + 1 + 8 = 21)
                                   (h13 : C + D + 3 + 4 = 21) :
  (A, B, C, D) = (6, 8, 7, 9) :=
begin
  sorry
end

end replace_circles_sums_equal_l259_259371


namespace sqrt_fourth_root_l259_259631

theorem sqrt_fourth_root (h : Real.sqrt (Real.sqrt (0.00000081)) = 0.1732) : Real.sqrt (Real.sqrt (0.00000081)) = 0.2 :=
by
  sorry

end sqrt_fourth_root_l259_259631


namespace area_of_triangle_QPO_l259_259787

/-- Conditions as defined above:
  1. ABCD is a rectangle.
  2. Line DP bisects AB at M and meets CD extended at P.
  3. Line CQ bisects side CD at N and meets AB extended at Q.
  4. Lines DP and CQ meet at O.
  5. The area of rectangle ABCD is k.
  Prove that the area of triangle QPO is k / 4.
-/ 
theorem area_of_triangle_QPO (A B C D M P Q N O : Point)
  (h1 : is_rectangle A B C D)
  (h2 : midpoint M A B ∧ collinear D P (line_through C D))
  (h3 : midpoint N C D ∧ collinear C Q (line_through A B))
  (h4 : intersect D P C Q O)
  (h5 : area_rectangle A B C D = k) :
  area_triangle Q P O = k / 4 := 
sorry

end area_of_triangle_QPO_l259_259787


namespace shifted_polynomial_sum_l259_259916

theorem shifted_polynomial_sum (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 5) = (a * (x + 5)^2 + b * (x + 5) + c)) →
  a + b + c = 125 :=
by
  sorry

end shifted_polynomial_sum_l259_259916


namespace monotonic_increasing_interval_l259_259502

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) ^ (real.sqrt (x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x, 0 ≤ x → x ≤ 1 → ∀ y, 0 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y :=
by
  intro x hx1 hx2 y hy1 hy2 hxy
  -- proof
  sorry

end monotonic_increasing_interval_l259_259502


namespace chi_square_test_l259_259175

-- Conditions
def n : ℕ := 100
def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25

-- Critical chi-square value for alpha = 0.001
def chi_square_critical : ℝ := 10.828

-- Calculated chi-square value
noncomputable def chi_square_value : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement to prove
theorem chi_square_test : chi_square_value > chi_square_critical :=
by sorry

end chi_square_test_l259_259175


namespace number_of_set_triples_l259_259661

theorem number_of_set_triples :
  let universe := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  ∃ (A B C : Finset ℕ),
    A ⊆ universe ∧ B ⊆ universe ∧ C ⊆ universe ∧
    (A ∩ B).card = 2 ∧ (B ∩ C).card = 2 ∧ (C ∩ A).card = 2 ∧
    A.card = 4 ∧ B.card = 4 ∧ C.card = 4 →
    (universe.card + 6) * (universe.card + 3) * universe.card = 45360 :=
by
  sorry

end number_of_set_triples_l259_259661


namespace area_of_circle_with_diameter_4_l259_259534

theorem area_of_circle_with_diameter_4 :
  let diameter := 4 in
  let radius := diameter / 2 in
  let area := π * radius^2 in
  area = 4 * π :=
by
  sorry

end area_of_circle_with_diameter_4_l259_259534


namespace original_selling_price_l259_259630

theorem original_selling_price (P SP1 SP2 : ℝ) (h1 : SP1 = 1.10 * P)
    (h2 : SP2 = 1.17 * P) (h3 : SP2 - SP1 = 35) : SP1 = 550 :=
by
  sorry

end original_selling_price_l259_259630


namespace b_correct_S_correct_l259_259089

noncomputable def A (n : ℕ) : ℕ := n^2 + 3 * n
noncomputable def B (n : ℕ) : ℕ := n^2 + 5 * n
noncomputable def a (n : ℕ) : ℕ := 2 * (n + 1)
noncomputable def b (n : ℕ) : ℕ := 2 * n + 4
noncomputable def c (n : ℕ) : ℝ := 2 / (A n - 2 * n)
noncomputable def S (n : ℕ) : ℝ := (2 * n) / (n + 1)

theorem b_correct (n : ℕ) : (B n - B (n - 1)) = b n := by sorry

theorem S_correct (n : ℕ) : (∑ k in Finset.range n, c k) = S n := by sorry

end b_correct_S_correct_l259_259089


namespace angle_KOI_eq_angle_MIU_l259_259012

theorem angle_KOI_eq_angle_MIU
  (triangle_KOI : Type)
  [triangle triangle_KOI]
  (point_M : point triangle_KOI)
  (point_S : point triangle_KOI)
  (point_U : point triangle_KOI)
  (K O I : point triangle_KOI)
  (KM_eq_MI : distance K M = distance M I)
  (SI_eq_SO : distance S I = distance S O)
  (MU_parallel_KI : parallel (line_through M U) (line_through K I)) :
  ∠ K O I = ∠ M I U :=
sorry

end angle_KOI_eq_angle_MIU_l259_259012


namespace prism_faces_same_color_l259_259598

structure PrismColoring :=
  (A : Fin 5 → Fin 5 → Bool)
  (B : Fin 5 → Fin 5 → Bool)
  (A_to_B : Fin 5 → Fin 5 → Bool)

def all_triangles_diff_colors (pc : PrismColoring) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
    (pc.A i j = !pc.A i k ∨ pc.A i j = !pc.A j k) ∧
    (pc.B i j = !pc.B i k ∨ pc.B i j = !pc.B j k) ∧
    (pc.A_to_B i j = !pc.A_to_B i k ∨ pc.A_to_B i j = !pc.A_to_B j k)

theorem prism_faces_same_color (pc : PrismColoring) (h : all_triangles_diff_colors pc) :
  (∀ i j : Fin 5, pc.A i j = pc.A 0 1) ∧ (∀ i j : Fin 5, pc.B i j = pc.B 0 1) :=
sorry

end prism_faces_same_color_l259_259598


namespace percentage_increase_in_radius_l259_259590

theorem percentage_increase_in_radius (r R : ℝ) (h1 : ∀ (r : ℝ), r > 0 → (π * R^2 = 2.56 * π * r^2)) :
  (R = 1.6 * r) → 60% = (1.6 - 1) * 100 :=
by sorry

end percentage_increase_in_radius_l259_259590


namespace correct_statements_l259_259721

/-
Problem: Given the following statements, which one is correct?
  A: The product of the range and median of the data 0, 1, 2, 4 is 6.
  B: If the variance of a set of data x1, x2, ..., xn is 5, then the variance of the data 4*x1-1, 4*x2-1, ..., 4*xn-1 is 20.
  C: If the variance of a set of data x1, x2, ..., xn is 0, then the mode of this set of data is unique.
  D: If the average of a set of not completely identical data x1, x2, ..., xn is x0, and a number x0 is added to this set of data to obtain a new set of data x0, x1, x2, ..., xn, then the average of the new set of data is x0.
  
Equivalent proof problem: Prove the correctness of statements A, C, and D given their respective conditions.
-/

theorem correct_statements (data : List ℕ) (x₀ : ℝ) (x : ℕ → ℝ) (n : ℕ) (ndata : List ℝ)
  (average_x : x₀ = (List.sum ndata) / n)
  (newData : List ℝ := x₀ :: ndata)
  (variance_x : Real := (List.sum (ndata.map (λ xi, (xi - x₀) ^ 2))) / n)
  (variance_0 : variance_x = 0) :
    (4 - 0) * ((data.nth 1 + data.nth 2) / 2) = 6 ∧
    true ∧
    List.mode_eq newData (List.mode ndata) ∧
    x₀ = (List.sum newData) / (n + 1) := by
  sorry

#check correct_statements

end correct_statements_l259_259721


namespace find_x_l259_259418

-- Definitions of the conditions in Lean 4
def angle_sum_180 (A B C : ℝ) : Prop := A + B + C = 180
def angle_BAC_eq_90 (A : ℝ) : Prop := A = 90
def angle_BCA_eq_2x (C x : ℝ) : Prop := C = 2 * x
def angle_ABC_eq_3x (B x : ℝ) : Prop := B = 3 * x

-- The theorem we need to prove
theorem find_x (A B C x : ℝ) 
  (h1 : angle_sum_180 A B C) 
  (h2 : angle_BAC_eq_90 A)
  (h3 : angle_BCA_eq_2x C x) 
  (h4 : angle_ABC_eq_3x B x) : x = 18 :=
by 
  sorry

end find_x_l259_259418


namespace possible_number_of_students_l259_259981

noncomputable def number_of_students (total_candies : ℕ) (candies_per_student : ℕ) : set ℕ :=
  {n | ∃ k : ℕ, k * n * candies_per_student = total_candies}

theorem possible_number_of_students :
  number_of_students 120 2 = {5, 6, 10, 12, 15} :=
by
  sorry

end possible_number_of_students_l259_259981


namespace ellipse_equation_chord_length_l259_259295

section ellipse_problem

-- Given Definitions
def minor_axis_length : ℝ := 2
def eccentricity : ℝ := sqrt 3 / 2
def line_AB (x : ℝ) : ℝ := x + 1

-- Define the semi-minor axis
def b : ℝ := minor_axis_length / 2  -- Since 2b = 2

-- Theorem 1: Finding the equation of the ellipse
theorem ellipse_equation : 
  (∃ a : ℝ, ∃ c : ℝ, a > b ∧ 
   c = a * eccentricity ∧ 
   a^2 = 1 + c^2 ∧ 
   ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1 ↔ (x^2 / 4) + y^2 = 1) :=
sorry
  
-- Theorem 2: Finding the length of the chord |AB|
theorem chord_length : 
  ∀ x1 y1 x2 y2 : ℝ, 
  (y1 = line_AB x1) ∧ 
  (y2 = line_AB x2) ∧ 
  ((x1, y1) and (x2, y2) ∈ ({ p : ℝ × ℝ | (p.1^2 / 4) + p.2^2 = 1 })) →
  (dist (x1, y1) (x2, y2) = 8 * sqrt 2 / 5) :=
sorry

end ellipse_problem

end ellipse_equation_chord_length_l259_259295


namespace total_selling_price_is_correct_l259_259967

-- Define the given constants
def meters_of_cloth : ℕ := 85
def profit_per_meter : ℕ := 10
def cost_price_per_meter : ℕ := 95

-- Compute the selling price per meter
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- Calculate the total selling price
def total_selling_price : ℕ := selling_price_per_meter * meters_of_cloth

-- The theorem statement
theorem total_selling_price_is_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_is_correct_l259_259967


namespace number_of_birds_initially_l259_259224

-- Definitions
def initial_monkeys := 6
def percentage_monkeys := 0.6

-- Statement to prove
theorem number_of_birds_initially (B : ℕ) 
  (h : initial_monkeys / (initial_monkeys + B - 2) = percentage_monkeys) : 
  B = 6 :=
sorry

end number_of_birds_initially_l259_259224


namespace range_of_k_l259_259740

noncomputable def A : set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
noncomputable def B (k : ℝ) : set ℝ := {x | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1}

theorem range_of_k (k : ℝ) (h : A ⊇ B k) : -1 ≤ k ∧ k ≤ 1 / 2 :=
by {
  sorry
}

end range_of_k_l259_259740


namespace find_number_l259_259622

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l259_259622


namespace find_angle_BAC_l259_259976

-- Define the types for points and angles
variable (Point : Type) (Angle : Type)
-- Define three points A, B, C and the circumcenter O
variables (A B C O X Y : Point)
-- Define the angles
variables (angleBAC angleAYX angleXYC angleBAC' : Angle)
-- Define the condition that O is the circumcenter of triangle ABC
variable (is_circumcenter : ∀ {A B C O : Point}, Prop)
-- Define the condition that X is on AC and Y is on AB
variables (is_on_AC is_on_AB : ∀ {P1 P2 P3 : Point}, Prop)
-- Define the condition that BX and CY intersect at O
variable (intersects_at : ∀ {P1 P2 P3 : Point}, Prop)
-- Define the equality conditions for the angles
variables (eq_angle1 eq_angle2 : ∀ {α β : Angle}, Prop)

-- Define the proof statement
theorem find_angle_BAC
  (h1 : is_circumcenter A B C O)
  (h2 : is_on_AC A C X)
  (h3 : is_on_AB A B Y)
  (h4 : intersects_at B X O)
  (h5 : intersects_at C Y O)
  (h6 : eq_angle1 angleBAC angleAYX)
  (h7 : eq_angle2 angleAYX angleXYC)
  : eq_angle1 angleBAC' 50 :=
sorry

end find_angle_BAC_l259_259976


namespace math_problem_l259_259542

theorem math_problem :
  ( (1 / 3 * 9) ^ 2 * (1 / 27 * 81) ^ 2 * (1 / 243 * 729) ^ 2) = 729 := by
  sorry

end math_problem_l259_259542


namespace dot_product_PA_PB_eq_neg_one_l259_259045

noncomputable def point_P (x : ℝ) (h : x > 0) :=
  (x, x + 2 / x)

def point_A (x : ℝ) (h : x > 0) :=
  (x + 1 / x, x + 1 / x)

def point_B (x : ℝ) (h : x > 0) :=
  (0, x + 2 / x)

def vec_PA (x : ℝ) (h : x > 0) : ℝ × ℝ :=
  ((x + 1 / x) - x, (x + 1 / x) - (x + 2 / x))

def vec_PB (x : ℝ) (h : x > 0) : ℝ × ℝ :=
  (0 - x, (x + 2 / x) - (x + 2 / x))

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_PA_PB_eq_neg_one (x : ℝ) (h : x > 0) :
  let PA := vec_PA x h,
      PB := vec_PB x h in
  dot_product PA PB = -1 :=
by
  sorry

end dot_product_PA_PB_eq_neg_one_l259_259045


namespace sum_first_five_incredibly_nice_numbers_l259_259985

open Nat

-- Definition of a proper divisor
def proper_divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, d ≠ 1 ∧ d ≠ n) (Finset.range (n + 1))

-- Definition of a sum_square_proper_divisors function
def sum_square_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum (λ d, d * d)

-- Definition of an incredibly nice number
def is_incredibly_nice (n : ℕ) : Prop :=
  n > 1 ∧ n = sum_square_proper_divisors n

-- Theorem stating the sum of the first five incredibly nice numbers is 1834
theorem sum_first_five_incredibly_nice_numbers :
  let incredibly_nice_numbers := Finset.filter is_incredibly_nice (Finset.range 2500)
  Finset.sum (Finset.take 5 incredibly_nice_numbers) id = 1834 :=
by
  sorry

end sum_first_five_incredibly_nice_numbers_l259_259985


namespace probability_roots_real_l259_259186

-- Define the polynomial
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + 3*b*x^3 + (3*b - 5)*x^2 + (-6*b + 4)*x - 3

-- Define the intervals for b
def interval_b1 := Set.Icc (-(15:ℝ)) (20:ℝ)
def interval_b2 := Set.Icc (-(15:ℝ)) (-2/3)
def interval_b3 := Set.Icc (4/3) (20:ℝ)

-- Calculate the lengths of the intervals
def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def length_b1 := length_interval (-(15:ℝ)) (20:ℝ)
noncomputable def length_b2 := length_interval (-(15:ℝ)) (-2/3)
noncomputable def length_b3 := length_interval (4/3) (20:ℝ)
noncomputable def effective_length := length_b2 + length_b3

-- The probability is the ratio of effective lengths
noncomputable def probability := effective_length / length_b1

-- The theorem we want to prove
theorem probability_roots_real : probability = 33/35 :=
  sorry

end probability_roots_real_l259_259186


namespace ways_to_parenthesize_5_distinct_results_from_parenthesizations_l259_259375

-- Definition for the number of ways to parenthesize the expression
def z : ℕ → ℕ
| 2 := 1
| 3 := 2
| 4 := 1 * (z 3) + (z 2) * (z 2) + (z 3) * 1
| 5 := 1 * (z 4) + (z 2) * (z 3) + (z 3) * (z 2) + (z 4) * 1
| _ := 0  -- General case, although not used

-- Theorem 1: Number of ways to parenthesize 3^3^3^3^3
theorem ways_to_parenthesize_5 : z 5 = 14 :=
by sorry

-- Theorem 2: Number of distinct results from parenthesizations of 3^3^3^3^3
theorem distinct_results_from_parenthesizations : Nat :=
by
  -- To represent the 9 unique results, we simply state as a constant
  let distinct_results := 9
  exact distinct_results

end ways_to_parenthesize_5_distinct_results_from_parenthesizations_l259_259375


namespace distance_focus_asymptote_hyperbola_eq_sqrt2_l259_259728

theorem distance_focus_asymptote_hyperbola_eq_sqrt2 :
  ∀ x y, x^2 - y^2 = 2 → distance_from_focus_to_asymptote x y = √2 :=
sorry

end distance_focus_asymptote_hyperbola_eq_sqrt2_l259_259728


namespace inequality_proof_l259_259847

theorem inequality_proof (x : ℝ) (hx : 0 ≤ x) : 
  ( ∑ i in Finset.range (2021), x ^ i ) * (1 + x ^ 2020) ≥ 4040 * x ^ 2020 := 
sorry

end inequality_proof_l259_259847


namespace solve_system_l259_259478

noncomputable def system_of_eqns (x y : ℝ) :=
  (1 / (x^2 + y^2) + x^2 * y^2 = 5 / 4) ∧ (2*x^4 + 2*y^4 + 5 * x^2 * y^2 = 9 / 4)

def solution_1 := [(1 / Real.sqrt 2, 1 / Real.sqrt 2), (1 / Real.sqrt 2, -1 / Real.sqrt 2), 
                   (-1 / Real.sqrt 2, 1 / Real.sqrt 2), (-1 / Real.sqrt 2, -1 / Real.sqrt 2)]

theorem solve_system :
  ∀ (x y : ℝ), system_of_eqns x y → ((x, y) ∈ solution_1 : Prop) :=
by
  sorry

end solve_system_l259_259478


namespace max_value_of_sum_a_l259_259445

noncomputable def max_sum_a : ℝ :=
  ∑ i in finset.range 100, (λ i => a i * a (i+2)).to_fun (i % 100)

variables (a : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_cond : ∀ i < 100, a i + a (i+1) + a (i+2) ≤ 1)
  (h_cyclic1 : a 101 = a 1)
  (h_cyclic2 : a 102 = a 2)

theorem max_value_of_sum_a : max_sum_a = 25 / 2 :=
sorry

end max_value_of_sum_a_l259_259445


namespace incorrect_statements_l259_259548

-- Defining the first function and its zeros
def f1 (x : ℝ) := x^2 - 3 * x

-- Second condition: a + b = 7 given -x^2 + ax + b ≥ 0 on [-2, 3]
def quadratic_inequality (a b : ℝ) := ∀ x : ℝ, -x^2 + a * x + b ≥ 0 → -2 ≤ x ∧ x ≤ 3

-- Function for the third problem and analyzing its domain and monotonicity
def f2 (x : ℝ) := sqrt (x^2 - 4 * x)
def domain_f2 := {x | x ≤ 0 ∨ x ≥ 4}

-- Fourth function and its transformation
def f3 (x : ℝ) := (x + 2) / (x - 1)
def f3_translation (x : ℝ) := f3 x + 1

-- Proving the required statements
theorem incorrect_statements : 
  (f1 0 = 0 ∧ f1 3 = 0) ∧ 
  (∀ (a b : ℝ), quadratic_inequality a b → a + b = 7) ∧ 
  (∀ x : ℝ, x ∈ domain_f2 → ∀ y : ℝ, x ≤ y → y ∈ domain_f2) ∧ 
  (¬ (∀ x : ℝ, f3_translation x = f3 (2 - x + 1))) 
:= 
by sorry

end incorrect_statements_l259_259548


namespace inequality_proof_l259_259703

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l259_259703


namespace not_age_6_l259_259450

theorem not_age_6 (ages : Finset ℕ) (abc : ℕ) :
  (11 ∈ ages) ∧ (7 ∈ ages) ∧ (∀ x ∈ ages, x ≠ 6) ∧ 
  (3 ≤ ages.card) ∧ 
  100 ≤ abc ∧ abc ≤ 999 ∧ 
  (abc % 11 = 0) ∧ (abc % 7 = 0) ∧
  (Digits n).length = 3 ∧ 
  (∀ d ∈ Digits n, d ∈ ages) ∧
  (∃ age : ℕ, age ∈ ages ∧ abc % 10 = age)
  → 6 ∉ ages :=
by sorry

end not_age_6_l259_259450


namespace part_a_part_b_l259_259132
open Set

def fantastic (n : ℕ) : Prop :=
  ∃ a b : ℚ, a > 0 ∧ b > 0 ∧ n = a + 1 / a + b + 1 / b

theorem part_a : ∃ᶠ p in at_top, Prime p ∧ ∀ k, ¬ fantastic (k * p) := 
  sorry

theorem part_b : ∃ᶠ p in at_top, Prime p ∧ ∃ k, fantastic (k * p) :=
  sorry

end part_a_part_b_l259_259132


namespace intersection_point_of_line_and_plane_l259_259260

def line_eq (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 - t, -5 + 4 * t, 1 + 2 * t)

def plane_eq (pt : ℝ × ℝ × ℝ) : Prop :=
  pt.1 - 3 * pt.2 + 7 * pt.3 - 24 = 0

theorem intersection_point_of_line_and_plane :
  ∃ t : ℝ, plane_eq (line_eq t) ∧ line_eq t = (0, -1, 3) :=
sorry

end intersection_point_of_line_and_plane_l259_259260


namespace fraction_work_completed_in_25_days_eq_half_l259_259826

theorem fraction_work_completed_in_25_days_eq_half :
  ∀ (men_total men_new men_rem : ℕ) (length : ℝ) (total_days : ℕ) 
    (work_hours_per_day initial_days remaining_days : ℕ)
    (total_man_hours man_hours_in_initial_days: ℝ),
  men_total = 100 →
  length = 2 →
  total_days = 50 →
  work_hours_per_day = 8 →
  initial_days = 25 →
  men_total * total_days * work_hours_per_day = total_man_hours →
  men_total * initial_days * work_hours_per_day = man_hours_in_initial_days →
  men_new = 60 →
  remaining_days = total_days - initial_days →
  man_hours_in_initial_days / total_man_hours = 1 / 2 :=
by
  intro men_total men_new men_rem length total_days work_hours_per_day initial_days remaining_days total_man_hours man_hours_in_initial_days
  intros h_men_total h_length h_total_days h_whpd h_initial_days h_totalmh h_initialmh h_men_new h_remaining_days
  have h1 : total_man_hours = 100 * 50 * 8 := by rw [h_men_total, h_total_days, h_whpd]; norm_num
  have h2 : man_hours_in_initial_days = 100 * 25 * 8 := by rw [h_men_total, h_initial_days, h_whpd]; norm_num
  have h3 : 100 * 50 * 8 = 40_000 := by norm_num
  have h4 : 100 * 25 * 8 = 20_000 := by norm_num
  rw [h1, h3] at h_totalmh
  rw [h2, h4] at h_initialmh
  norm_num at h_totalmh
  norm_num at h_initialmh
  rw [←h_totalmh] at h_initialmh
  rw h_totalmh
  rw h_initialmh
  norm_num
  sorry

end fraction_work_completed_in_25_days_eq_half_l259_259826


namespace sqrt_domain_l259_259376

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259376


namespace heat_of_reaction_correct_l259_259217

def delta_H_f_NH4Cl : ℝ := -314.43  -- Enthalpy of formation of NH4Cl in kJ/mol
def delta_H_f_H2O : ℝ := -285.83    -- Enthalpy of formation of H2O in kJ/mol
def delta_H_f_HCl : ℝ := -92.31     -- Enthalpy of formation of HCl in kJ/mol
def delta_H_f_NH4OH : ℝ := -80.29   -- Enthalpy of formation of NH4OH in kJ/mol

def delta_H_rxn : ℝ :=
  ((2 * delta_H_f_NH4OH) + (2 * delta_H_f_HCl)) -
  ((2 * delta_H_f_NH4Cl) + (2 * delta_H_f_H2O))

theorem heat_of_reaction_correct :
  delta_H_rxn = 855.32 :=
  by
    -- Calculation and proof steps go here
    sorry

end heat_of_reaction_correct_l259_259217


namespace greatest_five_digit_common_multiple_l259_259546

theorem greatest_five_digit_common_multiple (n : ℕ) :
  (n % 18 = 0) ∧ (10000 ≤ n) ∧ (n ≤ 99999) → n = 99990 :=
by
  sorry

end greatest_five_digit_common_multiple_l259_259546


namespace no_square_from_triangle_pieces_l259_259801

theorem no_square_from_triangle_pieces {A B C : Point} (hAC : AC = 2000) (hBC : BC = 1/1000) :
  ¬ ∃ (pieces : Finset (Set Point)), 
      pieces.card = 1000 ∧ 
      (∀ p ∈ pieces, ∃ q ∈ pieces, p ≠ q ∧ dist p q ≤ sqrt 2) ∧
      (∃ square : Set Point, 
         sqr square ∧ 
         (∀ p ∈ pieces, p ∈ square) 
      ) :=
by
  sorry

end no_square_from_triangle_pieces_l259_259801


namespace car_rental_cost_l259_259472

theorem car_rental_cost
  (S : ℝ)
  (samuel_cost : S + 0.16 * 44.44444444444444)
  (carrey_cost : 20 + 0.25 * 44.44444444444444)
  (equal_cost : samuel_cost = carrey_cost) :
  S = 24 :=
by
  sorry

end car_rental_cost_l259_259472


namespace max_points_on_plane_l259_259065

-- Define the conditions on the points
structure Plane (α : Type*) :=
(points : set α)
(collinear : set α → Prop)
(isosceles : α → α → α → Prop)

-- This will represent a finite plane with points satisfying our conditions
def valid_plane (α : Type*) [Plane α] (n : ℕ) : Prop :=
  ∃ (s : finset α), s.card = n ∧
  (∀ a b c ∈ s, ¬Plane.collinear {a, b, c}) ∧
  (∀ a b c ∈ s, Plane.isosceles a b c)

-- The main theorem based on the problem conditions and solution
theorem max_points_on_plane {α : Type*} [Plane α] : valid_plane α 6 :=
sorry

end max_points_on_plane_l259_259065


namespace penny_difference_l259_259824

variables (p : ℕ)

/-- Liam and Mia have certain numbers of fifty-cent coins. This theorem proves the difference 
    in their total value in pennies. 
-/
theorem penny_difference:
  (3 * p + 2) * 50 - (2 * p + 7) * 50 = 50 * p - 250 :=
by
  sorry

end penny_difference_l259_259824


namespace part1_even_function_part2_two_distinct_zeros_l259_259724

noncomputable def f (x a : ℝ) : ℝ := (4^x + a) / 2^x
noncomputable def g (x a : ℝ) : ℝ := f x a - (a + 1)

theorem part1_even_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) ↔ a = 1 :=
sorry

theorem part2_two_distinct_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (a ∈ Set.Icc (1/2) 1 ∪ Set.Icc 1 2) :=
sorry

end part1_even_function_part2_two_distinct_zeros_l259_259724


namespace number_of_true_propositions_is_2_l259_259001

theorem number_of_true_propositions_is_2
    (P1 : ∀ (l₁ l₂ l₃ : Type), (l₁ ⊥ l₃) → (l₂ ⊥ l₃) → (l₁ ∥ l₂))
    (P2 : ∀ (l₁ l₂ p₁ : Type), (l₁ ⊥ p₁) → (l₂ ⊥ p₁) → (l₁ ∥ l₂))
    (P3 : ∀ (p₁ p₂ l₁ : Type), (p₁ ⊥ l₁) → (p₂ ⊥ l₁) → (p₁ ∥ p₂))
    (P4 : ∀ (p₁ p₂ p₃ : Type), (p₁ ⊥ p₃) → (p₂ ⊥ p₃) → (p₁ ∥ p₂)) :
  P1 = false ∧ P2 = true ∧ P3 = true ∧ P4 = false → 2 = 2 :=
by
  sorry

end number_of_true_propositions_is_2_l259_259001


namespace smallest_divisor_of_2880_that_results_in_perfect_square_l259_259143

theorem smallest_divisor_of_2880_that_results_in_perfect_square : 
  ∃ (n : ℕ), (n ∣ 2880) ∧ (∃ m : ℕ, 2880 / n = m * m) ∧ (∀ k : ℕ, (k ∣ 2880) ∧ (∃ m' : ℕ, 2880 / k = m' * m') → n ≤ k) ∧ n = 10 :=
sorry

end smallest_divisor_of_2880_that_results_in_perfect_square_l259_259143


namespace total_admission_cost_l259_259776

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l259_259776


namespace distribution_schemes_l259_259905

def choose (n k : ℕ) := (nat.choose n k)

theorem distribution_schemes (
    students : ℕ,
    intersections : ℕ,
    students_per_intersection : ℕ
):
    (students = 12) →
    (intersections = 3) →
    (students_per_intersection = 4) →
    choose 12 4 * choose 8 4 * choose 4 4 = choose 12 4 * choose 8 4 * choose 4 4 :=
by
  intros
  sorry

end distribution_schemes_l259_259905


namespace number_of_folds_l259_259644

theorem number_of_folds (n : ℕ) :
  (3 * (8 * 8)) / n = 48 → n = 4 :=
by
  sorry

end number_of_folds_l259_259644


namespace roberto_outfits_l259_259078

theorem roberto_outfits :
  ∀ (trousers shirts jackets shoes : ℕ), 
  trousers = 6 → 
  shirts = 7 → 
  jackets = 4 → 
  shoes = 2 →
  trousers * shirts * jackets * shoes = 336 :=
by intros trousers shirts jackets shoes ht hs hj hs;
   rw [ht, hs, hj, hs];
   exact
   show 6 * 7 * 4 * 2 = 336,
   by norm_num;
   sorry

end roberto_outfits_l259_259078


namespace select_president_and_secretary_l259_259142

-- Define the problem conditions
def students := {A, B, C, D, E, F}

-- Define the function to compute the number of ways to select a president and a secretary
def numberOfWaysToSelect (n : ℕ) : ℕ := n * (n - 1)

-- The theorem statement
theorem select_president_and_secretary :
  numberOfWaysToSelect 6 = 30 :=
by
  -- This part is where the proof would go, currently replaced with sorry
  sorry

end select_president_and_secretary_l259_259142


namespace combined_weight_is_150_l259_259903

-- Definitions based on conditions
def tracy_weight : ℕ := 52
def jake_weight : ℕ := tracy_weight + 8
def weight_range : ℕ := 14
def john_weight : ℕ := tracy_weight - 14

-- Proving the combined weight
theorem combined_weight_is_150 :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

end combined_weight_is_150_l259_259903


namespace number_99_in_column_4_l259_259975

-- Definition of the arrangement rule
def column_of (num : ℕ) : ℕ :=
  ((num % 10) + 4) / 2 % 5 + 1

theorem number_99_in_column_4 : 
  column_of 99 = 4 :=
by
  sorry

end number_99_in_column_4_l259_259975


namespace sampling_method_systematic_l259_259273

theorem sampling_method_systematic :
  ∃ individuals : List ℕ, 
    (∀ n, n ∈ individuals → 1 ≤ n ∧ n ≤ 100) ∧ 
    (∃ sample : List ℕ, sample = [10, 30, 50, 70, 90]) ∧ 
    (∀ n m : ℕ, 
      n ∈ sample → m ∈ sample → (n ≠ m → |n - m| % 20 = 0)) → 
    "Systematic sampling" :=
by
  sorry

end sampling_method_systematic_l259_259273


namespace age_difference_proof_l259_259107

noncomputable def difference_in_ages (R M : ℕ) (x : ℝ) :=
  let ramesh_present_age := 2 * x
  let mahesh_present_age := 5 * x
  let ramesh_age_after_10_years := ramesh_present_age + 10
  let mahesh_age_after_10_years := mahesh_present_age + 10
  let ratio_eq := (ramesh_age_after_10_years / mahesh_age_after_10_years) = 2 / 3
  let x_val := 2.5
  let expected_difference := 7.5
  let age_difference := mahesh_present_age - ramesh_present_age
  age_difference = expected_difference → ratio_eq

theorem age_difference_proof : ∃ R M x, difference_in_ages R M x := by
  use nat.succ 0 -- Dummy value to instantiate R
  use nat.succ 0 -- Dummy value to instantiate M
  use 2.5                                     -- The calculated x value
  sorry

end age_difference_proof_l259_259107


namespace least_number_of_teams_l259_259178

theorem least_number_of_teams
  (total_athletes : ℕ)
  (max_team_size : ℕ)
  (h_total : total_athletes = 30)
  (h_max : max_team_size = 12) :
  ∃ (number_of_teams : ℕ) (team_size : ℕ),
    number_of_teams * team_size = total_athletes ∧
    team_size ≤ max_team_size ∧
    number_of_teams = 3 :=
by
  sorry

end least_number_of_teams_l259_259178


namespace association_members_l259_259782

-- Define the conditions
variables (M W : ℕ)

-- Define the conditions
def proportion_of_homeowners (M W : ℕ) : Prop :=
  0.10 * M + 0.20 * W ≥ 18

-- The main statement to be proved
theorem association_members (M W : ℕ) (h : proportion_of_homeowners M W) :
  M + W = 91 :=
sorry

end association_members_l259_259782


namespace solution_set_of_inequality_l259_259507

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l259_259507


namespace sqrt_domain_l259_259378

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259378


namespace gear_revolutions_difference_l259_259993

noncomputable def gear_revolution_difference (t : ℕ) : ℕ :=
  let p := 10 * t
  let q := 40 * t
  q - p

theorem gear_revolutions_difference (t : ℕ) : gear_revolution_difference t = 30 * t :=
by
  sorry

end gear_revolutions_difference_l259_259993


namespace sum_of_first_n_odd_numbers_eq_576_l259_259115

theorem sum_of_first_n_odd_numbers_eq_576 : ∃ n : ℕ, (∑ i in finset.range n, (2 * i + 1)) = 576 ∧ n = 24 := 
sorry

end sum_of_first_n_odd_numbers_eq_576_l259_259115


namespace range_of_sqrt_x_plus_3_l259_259386

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259386


namespace shaded_area_on_grid_l259_259996

theorem shaded_area_on_grid :
  let grid := 4 × 4
  let center := (1.5, 1.5)
  let triangles := 4
  let area := 0.25
  in  ∀ (grid : nat × nat) (center : ℝ × ℝ) (triangles : nat) (area : ℝ),
      grid = (4, 4) →
      center = (1.5, 1.5) →
      triangles = 4 →
      area = 0.25 →
      ∃ (shape : set (ℝ × ℝ)),
      (is_combination_of_triangles shape center triangles) →
      calculate_area shape = area := sorry

noncomputable def is_combination_of_triangles (shape : set (ℝ × ℝ))
  (center : ℝ × ℝ) (triangles : nat) : Prop :=
  -- Define the property that the shape is a combination of triangles
  sorry

noncomputable def calculate_area (shape : set (ℝ × ℝ)) : ℝ :=
  -- Define the function to calculate the area of the shape
  sorry

end shaded_area_on_grid_l259_259996


namespace KN_perp_AB_MN_perp_AC_l259_259796

variables {A B C M N K : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (BC AC AB : Set (ℝ × ℝ)) 
variables (BK KM MN NC : ℝ) (AN AK : ℝ)
variables [TriangleABC : is_triangle ABC] [angle_A_is_60 : angle_A = 60]
variables [BK_eq_KM : BK = KM] [KM_eq_MN : KM = MN]
variables [MN_eq_NC : MN = NC] [AN_eq_2AK : AN = 2 * AK]

-- Part (a)
theorem KN_perp_AB : KN ⊥ AB := sorry

-- Part (b)
theorem MN_perp_AC : MN ⊥ AC := sorry

end KN_perp_AB_MN_perp_AC_l259_259796


namespace function_domain_l259_259410

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259410


namespace triangle_area_l259_259602

noncomputable def right_triangle (D E F : Type) [metric_space D] [metric_space E] [metric_space F] :=
∃ (angle_D : ℝ) (DE : ℝ), 
  angle_D = real.pi / 3 ∧ 
  DE = 4 ∧ 
  ∃ (DF : ℝ) (EF : ℝ), 
    area_of_triangle DEF = (8 * real.sqrt(3)) / 3

theorem triangle_area :
  ∀ (D E F : Type) [metric_space D] [metric_space E] [metric_space F], 
  right_triangle D E F -> 
  area_of_triangle DEF = (8 * real.sqrt(3)) / 3 := 
by {
  sorry
}

end triangle_area_l259_259602


namespace max_area_of_triangle_PAB_l259_259793

noncomputable def center_of_circle : ℝ × ℝ := (1, -1)

def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * sqrt 2 * cos (θ + π/4)

def line_param_eq (t : ℝ) : ℝ × ℝ := (t, -1 + 2 * sqrt 2 * t)

def max_area_triangle (a b c : ℝ) : ℝ := (1/2) * a * b * sin c

theorem max_area_of_triangle_PAB :
  ∀ P : ℝ × ℝ,
  P ≠ (1, -1) ∧ (∃ (ρ θ : ℝ), P = (ρ * cos θ, ρ * sin θ) ∧ circle_equation ρ θ) ∧
  (∃ t : ℝ, (t, -1 + 2 * sqrt 2 * t) = P) →
  max_area_triangle (2 * sqrt (2 - (2 * sqrt 2 / 3)^2))
                     (sqrt 2 + 2 * sqrt 2 / 3)
                     (π / 2) =
  10 * sqrt 5 / 9 :=
sorry

end max_area_of_triangle_PAB_l259_259793


namespace AI_AJ_eq_AB_AC_l259_259023

-- Let ABC be a triangle, I the incenter, and J the excenter opposite to angle A.
variables {A B C I J : Point}
variables (triangle_ABC : Triangle A B C)
variables (incenter_I : Incenter triangle_ABC I)
variables (excenter_J : ExcenterOppositeA triangle_ABC J)

-- The goal is to show that AI * AJ = AB * AC.
theorem AI_AJ_eq_AB_AC : distance A I * distance A J = distance A B * distance A C := by
  sorry

end AI_AJ_eq_AB_AC_l259_259023


namespace range_of_reciprocal_distances_l259_259002

-- Definitions based on given conditions
def line_parametric (α : ℝ) := 
  λ t : ℝ, (Real.sqrt 3 / 2 + t * Real.cos α, 3 / 2 + t * Real.sin α)

def curve (x y : ℝ) := x^2 + y^2 = 1

def intersects_curve (α : ℝ) (t : ℝ) : Prop := 
  curve (Real.sqrt 3 / 2 + t * Real.cos α) (3 / 2 + t * Real.sin α)

-- The Lean equivalent proof statement
theorem range_of_reciprocal_distances (α : ℝ)
  (h_intersect : ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ intersects_curve α t1 ∧ intersects_curve α t2) :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ - (t1 + t2) / (t1 * t2) ∈ Ioo (Real.sqrt 3 + Real.sin (α + Real.pi / 6)) Real.sqrt 3) :=
sorry

end range_of_reciprocal_distances_l259_259002


namespace select_team_ways_l259_259060

-- Definitions of the conditions and question
def boys := 7
def girls := 10
def boys_needed := 2
def girls_needed := 3
def total_team := 5

-- Theorem statement to prove the number of selecting the team
theorem select_team_ways : (Nat.choose boys boys_needed) * (Nat.choose girls girls_needed) = 2520 := 
by
  -- Place holder for proof
  sorry

end select_team_ways_l259_259060


namespace ratio_AC_BD_l259_259786

variable (A B C D : Point)
variable [Geometry Parallelogram]
variable (drawn_from_obtuse_angle : Altitude B (side DA) (ratio 5 3))
variable (AD_AB_ratio : Ratio AD AB 2)

theorem ratio_AC_BD (h : drawn_from_obtuse_angle ∧ AD_AB_ratio) : Ratio AC BD = 2 := 
sorry

end ratio_AC_BD_l259_259786


namespace rectangle_cos_angle_BAO_l259_259850

noncomputable def cos_angle_BAO :=
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (15, 4)
  let C : ℝ × ℝ := (15, 0)
  let D : ℝ × ℝ := (0, 0)
  let O : ℝ × ℝ := (7.5, 2)
  let AO_length : ℝ := (Real.sqrt ((A.1 - O.1) ^ 2 + (A.2 - O.2) ^ 2)) -- length of AO
  let BO_length : ℝ := (Real.sqrt ((B.1 - O.1) ^ 2 + (B.2 - O.2) ^ 2)) -- length of BO
  AO_length / BO_length = 1

theorem rectangle_cos_angle_BAO (AB BC : ℝ) (hAB : AB = 8) (hBC : BC = 15) :
    cos_angle_BAO = 1 :=
  by
  sorry

end rectangle_cos_angle_BAO_l259_259850


namespace correct_system_of_equations_l259_259947

theorem correct_system_of_equations (x y : ℕ) (h1 : x + y = 145) (h2 : 10 * x + 12 * y = 1580) :
  (x + y = 145) ∧ (10 * x + 12 * y = 1580) :=
by
  sorry

end correct_system_of_equations_l259_259947


namespace probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l259_259123

-- Conditions
def red_ball_probability := 1 / 2
def yellow_ball_probability := 1 / 2
def num_draws := 3

-- Define the events and their probabilities
def prob_three_red : ℚ := red_ball_probability ^ num_draws
def prob_three_same : ℚ := 2 * (red_ball_probability ^ num_draws)
def prob_not_all_same : ℚ := 1 - prob_three_same / 2

-- Lean statements
theorem probability_three_red_balls : prob_three_red = 1 / 8 :=
by
  sorry

theorem probability_three_same_color_balls : prob_three_same = 1 / 4 :=
by
  sorry

theorem probability_not_all_same_color_balls : prob_not_all_same = 3 / 4 :=
by
  sorry

end probability_three_red_balls_probability_three_same_color_balls_probability_not_all_same_color_balls_l259_259123


namespace picky_elephants_l259_259588

theorem picky_elephants (e h : ℕ) (he : (5 * e + 7 * h = 31)) (he' : (8 * e + 4 * h = 28)) :
  (2 = e) :=
begin
    sorry
end

end picky_elephants_l259_259588


namespace polynomial_divisibility_iff_gcd_l259_259254

open Nat Polynomial

-- Defining the polynomial
noncomputable def P (n : ℕ) : Polynomial ℤ :=
  Polynomial.sumRange (n + 1)

noncomputable def Q (m n : ℕ) : Polynomial ℤ :=
  Polynomial.sumRange ((m + 1) * n) (fun i => if i % n = 0 then 1 else 0)

-- Hypothesis gcd condition
theorem polynomial_divisibility_iff_gcd (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
    (Q m n) ∣ (P m) ↔ gcd (m + 1) n = 1 := sorry

end polynomial_divisibility_iff_gcd_l259_259254


namespace regular_polygon_sides_l259_259960

theorem regular_polygon_sides (n : ℕ) (h : (180 * (n - 2) = 135 * n)) : n = 8 := by
  sorry

end regular_polygon_sides_l259_259960


namespace range_of_x_in_sqrt_x_plus_3_l259_259402

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259402


namespace relay_team_order_count_l259_259601

theorem relay_team_order_count :
  ∃ (orders : ℕ), orders = 6 :=
by
  let team_members := 4
  let remaining_members := team_members - 1  -- Excluding Lisa
  let first_lap_choices := remaining_members.choose 3  -- Choices for the first lap
  let third_lap_choices := (remaining_members - 1).choose 2  -- Choices for the third lap
  let fourth_lap_choices := (remaining_members - 2).choose 1  -- The last remaining member choices
  have orders := first_lap_choices * third_lap_choices * fourth_lap_choices
  use orders
  sorry

end relay_team_order_count_l259_259601


namespace different_rectangles_in_fourth_column_l259_259978

theorem different_rectangles_in_fourth_column (grid : fin 7 → fin 7 → ℕ) 
  (h1 : ∀ (x y : fin 7), ∃ (n : ℕ), grid x y = n)
  (h2 : ∀ (rect : set (fin 7 × fin 7)), ∃! n : ℕ, ∀ (i j : fin 7), (i, j) ∈ rect → grid i j = n)
  (h3 : ∀ (y : fin 7), ∃ (rects : fin 7 × (set (fin 7 × fin 1))), rects.1 = y) :
  ∃ ! (R : fin 7 → fin 7 → Prop), 
    (∀ x, (R x 3 → ∃ ! n : ℕ, grid x 3 = n ∧ R x 3 = True) ) ∧ 
    (∑ i in finset.range 7, if (R i 3) then 1 else 0 = 4) := sorry

end different_rectangles_in_fourth_column_l259_259978


namespace product_correct_l259_259164

theorem product_correct : 100 * 19.98 * 1.998 * 1000 = (1998)^2 := 
by
  have h1 : 19.98 * 100 = 1998 := by norm_num
  have h2 : 1.998 * 1000 = 1998 := by norm_num
  calc
    100 * 19.98 * 1.998 * 1000
    = 100 * (19.98 * 100) * (1.998 * 1000) / 100 := by field_simp [h1, h2]
    ... = 1998 * 1998 := by rw [h1, h2]
    ... = 1998^2 := by ring

end product_correct_l259_259164


namespace diagonal_difference_is_eight_l259_259575

def original_matrix : matrix (fin 5) (fin 5) ℕ := 
  ![![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]]

def modified_matrix : matrix (fin 5) (fin 5) ℕ :=
  ![![5, 4, 3, 2, 1], 
    ![6, 7, 8, 9, 10],
    ![15, 14, 13, 12, 11], 
    ![16, 17, 18, 19, 20], 
    ![21, 22, 23, 24, 25]]

def main_diagonal_sum (M : matrix (fin 5) (fin 5) ℕ) :=
  ∑ i, M i i

def anti_diagonal_sum (M : matrix (fin 5) (fin 5) ℕ) :=
  ∑ i, M i (fin.congr n i)

def positive_difference (a b : ℕ) : ℕ :=
  if a > b then a - b else b - a

theorem diagonal_difference_is_eight : 
  positive_difference (main_diagonal_sum modified_matrix) (anti_diagonal_sum modified_matrix) = 8 :=
by
  sorry

end diagonal_difference_is_eight_l259_259575


namespace product_simplification_l259_259474

theorem product_simplification : (∏ n in Finset.range 200, (5 * n + 4) / (5 * n - 1)) = 1009 / 5 := 
by
  sorry

end product_simplification_l259_259474


namespace taylor_scores_l259_259525

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l259_259525


namespace proof_of_subtraction_l259_259531

noncomputable def gun_wowgun : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  λ g n o u w, (10 * g + u)^2 = 100000 * w + 10000 * o + 1000 * w + 100 * g + 10 * u + n

theorem proof_of_subtraction {g n o u w : ℕ} 
  (h: gun_wowgun g n o u w) (h1 : g = 3) (h2 : n = 6) (h3 : o = 1) (h4 : u = 7) (h5 : w = 4) : 
  o - w = 3 := 
by 
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end proof_of_subtraction_l259_259531


namespace find_b_l259_259125

theorem find_b (a b c : ℕ) (h1 : 2 * b = a + c) (h2 : b^2 = c * (a + 1)) (h3 : b^2 = a * (c + 2)) : b = 12 :=
by 
  sorry

end find_b_l259_259125


namespace max_knights_on_20x20_board_l259_259849

theorem max_knights_on_20x20_board :
  ∃ K ≥ 0, ∀ (queenie_strategy : ℕ → option (ℕ × ℕ)),
  (K <= 100) :=
begin
  sorry
end

end max_knights_on_20x20_board_l259_259849


namespace printer_time_ratio_l259_259561

-- Define the conditions as constants or parameters
def time_x : ℝ := 15
def time_y : ℝ := 10
def time_z : ℝ := 20

-- Define the rates as inverses of the times
def rate_x : ℝ := 1 / time_x
def rate_y : ℝ := 1 / time_y
def rate_z : ℝ := 1 / time_z

-- Define the combined rate of printers Y and Z
def combined_rate_yz : ℝ := rate_y + rate_z

-- Define the combined time for printers Y and Z
def time_yz : ℝ := 1 / combined_rate_yz

-- Proof statement that the ratio is equal to 9/4
theorem printer_time_ratio :
  (time_x / time_yz) = (9 / 4) :=
by
  sorry

end printer_time_ratio_l259_259561


namespace range_of_independent_variable_l259_259396

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259396


namespace P_finishes_in_15_minutes_more_l259_259928

variable (P Q : ℝ)

def rate_p := 1 / 4
def rate_q := 1 / 15
def time_together := 3
def total_job := 1

theorem P_finishes_in_15_minutes_more :
  let combined_rate := rate_p + rate_q
  let completed_job_in_3_hours := combined_rate * time_together
  let remaining_job := total_job - completed_job_in_3_hours
  let time_for_P_to_finish := remaining_job / rate_p
  let minutes_needed := time_for_P_to_finish * 60
  minutes_needed = 15 :=
by
  -- Proof steps go here
  sorry

end P_finishes_in_15_minutes_more_l259_259928


namespace remainder_of_A_mod_50_l259_259432

def floor (x : ℝ) : ℤ := int.floor x

def A := ∑ k in finset.range (2016 + 1), floor ((7 : ℝ) ^ k / 8)

theorem remainder_of_A_mod_50 : A % 50 = 42 := 
by
  sorry

end remainder_of_A_mod_50_l259_259432


namespace trapezoid_median_l259_259968

theorem trapezoid_median
  (h : ℝ)
  (area_triangle : ℝ)
  (area_trapezoid : ℝ)
  (bt : ℝ)
  (bt_sum : ℝ)
  (ht_positive : h ≠ 0)
  (triangle_area : area_triangle = (1/2) * bt * h)
  (trapezoid_area : area_trapezoid = area_triangle)
  (trapezoid_bt_sum : bt_sum = 40)
  (triangle_bt : bt = 24)
  : (bt_sum / 2) = 20 :=
by
  sorry

end trapezoid_median_l259_259968


namespace range_of_sqrt_x_plus_3_l259_259384

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259384


namespace complement_intersection_l259_259333

-- Definitions for the sets
def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2, 3}

-- Statement to be proved
theorem complement_intersection (hU : U = {0, 1, 2, 3}) (hA : A = {0, 1}) (hB : B = {1, 2, 3}) :
  ((U \ A) ∩ B) = {2, 3} :=
by
  -- Greek delta: skip proof details
  sorry

end complement_intersection_l259_259333


namespace sum_of_roots_quadratic_eq_l259_259043

variable (h : ℝ)
def quadratic_eq_roots (x : ℝ) : Prop := 6 * x^2 - 5 * h * x - 4 * h = 0

theorem sum_of_roots_quadratic_eq (x1 x2 : ℝ) (h : ℝ) 
  (h_roots : quadratic_eq_roots h x1 ∧ quadratic_eq_roots h x2) 
  (h_distinct : x1 ≠ x2) :
  x1 + x2 = 5 * h / 6 := by
sorry

end sum_of_roots_quadratic_eq_l259_259043


namespace exists_infinitely_many_n_pi_n_divides_n_l259_259564

/-- The prime-counting function π(n) is defined as the number of prime numbers less than or equal to n. -/
def prime_counting_function (n : ℕ) : ℕ := Nat.factorization.count_le n

theorem exists_infinitely_many_n_pi_n_divides_n :
  ∀ k > 1, ∃ n, prime_counting_function n ∣ n :=
by
  sorry

end exists_infinitely_many_n_pi_n_divides_n_l259_259564


namespace polynomial_coefficient_B_l259_259202

theorem polynomial_coefficient_B :
  ∃ (A C D : ℝ), 
    (∀ z : ℂ, z^6 - 10 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 16 = 0 → z ∈ {1, 2} ∧ multiset.card {1, 2} = 6 ∧ multiset.sum {1, 2} = 10) →
    B = -88 :=
sorry

end polynomial_coefficient_B_l259_259202


namespace average_transformed_data_is_3_l259_259292

variable (x1 x2 x3 : ℝ)
variable s2 : ℝ

-- Given conditions
axiom h1 : s2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12)

-- Prove that the average of the data (x1 + 1), (x2 + 1), (x3 + 1) is equal to 3
theorem average_transformed_data_is_3 (s2 : ℝ) (x1 x2 x3 : ℝ) (h1 : s2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12)) :
  ((x1 + 1) + (x2 + 1) + (x3 + 1)) / 3 = 3 := 
by 
  sorry

end average_transformed_data_is_3_l259_259292


namespace sequence_sum_lt_l259_259710

open Nat

/-- Lean 4 Statement for the problem.

Given a sequence of numbers a_1, a_2, ..., a_n where each a_i > 1,
and |a_{k+1} - a_k| < 1 for all 1 ≤ k < n, prove that the sum of
a_1/a_2 + a_2/a_3 + ... + a_{n-1}/a_n + a_n/a_1 is less than 2n - 1.
-/
theorem sequence_sum_lt (n : ℕ) (a : Fin n -> ℝ) (h1 :  ∀ i : Fin n, 1 < a i)
  (h2 : ∀ i : Fin (n - 1), abs (a (Fin.castSucc i) - a i.succ) < 1) :
  ((finset.univ.fin n).sum (λ i, a i / a (i + 1))) + (a (Fin.last n) / a 0) < 2 * n - 1 := 
sorry

end sequence_sum_lt_l259_259710


namespace unique_value_of_a_l259_259715

theorem unique_value_of_a 
  (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x ∈ set.Ioo (4 * -1 - 3) (3 - 2 * (-1)^2) → x ∈ set.Ioo (2 * -1) (3 - (-1)^2))
  (h₂ : even (λ x, f (2 * x - 3))) :
  ∀ a : ℝ, a = -1 :=
sorry

end unique_value_of_a_l259_259715


namespace area_of_inner_square_l259_259479

theorem area_of_inner_square (a b c : ℝ) (h₁ : a = 10) (h₂ : b = 3) : 
    let s := 5 * (Real.sqrt 2 - 2) in 
    s^2 = 50 * (3 - 2 * Real.sqrt 2) :=
by
    -- Place conditions as equalities to ensure they match the problem requirements
    have a_eq : a = 10 := h₁,
    have b_eq : b = 3 := h₂,
    sorry

end area_of_inner_square_l259_259479


namespace cone_generatrix_is_6_l259_259485

noncomputable def cone_generatrix_length (r l : ℝ) (h : ℝ) : Prop :=
  (π * r * l = 2 * π * r^2) ∧
  (1 / 3 * π * r^2 * h = 9 * √3 * π) ∧
  (h = √(l^2 - r^2))

theorem cone_generatrix_is_6 (r l : ℝ) (h : ℝ) (h_cond : cone_generatrix_length r l h) :
  l = 6 :=
by
  sorry

end cone_generatrix_is_6_l259_259485


namespace semicircle_perimeter_approx_l259_259927

-- Define the radius of the semicircle
def radius : ℝ := 3.5

-- Define the value of pi for approximation
def pi_approx : ℝ := 3.14

-- Theorem statement to prove the perimeter of the semicircle with radius 3.5 cm
theorem semicircle_perimeter_approx :
  (π * radius + 2 * radius) ≈ 17.99 :=
by
  -- Proof construction (to be done)
  sorry

end semicircle_perimeter_approx_l259_259927


namespace value_at_neg2_l259_259718

variable (a b : ℝ)

theorem value_at_neg2 (h : 2 * a + b = 3) : a * (-2)^2 - b * (-2) = 6 := by
  calc
    a * (-2)^2 - b * (-2) = 4 * a + 2 * b : by sorry
    ... = 4 * a + 2 * b : by sorry
    ... = 2 * (2 * a + b) : by sorry
    ... = 2 * 3 : by rw [h]
    ... = 6 : by norm_num

end value_at_neg2_l259_259718


namespace train_pass_time_l259_259197

-- Definitions for the given conditions
def length_train : ℝ := 250 -- length of the train in meters
def speed_train_kmh : ℝ := 120 -- speed of the train in km/h
def speed_man_kmh : ℝ := 15 -- speed of the man in km/h
def length_train_tunnel : ℝ := 500 -- length of the train's tunnel in meters
def length_man_tunnel : ℝ := 200 -- length of the man's tunnel in meters

-- Conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

-- Speeds in m/s
def speed_train_ms : ℝ := kmh_to_ms speed_train_kmh
def speed_man_ms : ℝ := kmh_to_ms speed_man_kmh

-- Relative speed since they are moving in opposite directions
def relative_speed : ℝ := speed_train_ms + speed_man_ms

-- Total distance to be covered
def total_distance : ℝ := length_train + length_train_tunnel + length_man_tunnel

-- Calculating the time it takes for the train to pass the man
def time_to_pass : ℝ := total_distance / relative_speed

-- The main theorem stating the result
theorem train_pass_time : 
  time_to_pass = 25.33 :=
by
  -- Placeholder for the proof
  sorry

end train_pass_time_l259_259197


namespace partition_equal_pair_sums_l259_259637

/-- 
Partition the set of integers 1, 2, 3, ..., 16 into two subsets 
of eight numbers each, such that the sums of all possible pairs 
of numbers from the first subset match the sums of all possible 
pairs from the second subset.
-/
theorem partition_equal_pair_sums :
  ∃ (A B : Finset ℤ), A.card = 8 ∧ B.card = 8 ∧ 
    A ∪ B = Finset.range 1 17 ∧
    ∀ (x y : ℤ), x ∈ A → y ∈ A → (x + y) ∈ (A.pair_sums) :=
sorry

end partition_equal_pair_sums_l259_259637


namespace which_calc_is_positive_l259_259144

theorem which_calc_is_positive :
  (-3 + 7 - 5 < 0) ∧
  ((1 - 2) * 3 < 0) ∧
  (-16 / (↑(-3)^2) < 0) ∧
  (-2^4 * (-6) > 0) :=
by
sorry

end which_calc_is_positive_l259_259144


namespace sum_of_three_fractions_is_one_l259_259269

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end sum_of_three_fractions_is_one_l259_259269


namespace no_solutions_abs_eq_3x_plus_6_l259_259264

theorem no_solutions_abs_eq_3x_plus_6 : ¬ ∃ x : ℝ, |x| = 3 * (|x| + 2) :=
by {
  sorry
}

end no_solutions_abs_eq_3x_plus_6_l259_259264


namespace quadratic_has_distinct_real_roots_l259_259113

theorem quadratic_has_distinct_real_roots : 
  ∀ (x : ℝ), x^2 - 3 * x + 1 = 0 → ∀ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = 1 →
  (b^2 - 4 * a * c) > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l259_259113


namespace probability_demand_at_least_21_loaves_daily_profit_given_demand_15_average_profit_over_30_days_l259_259578

-- Given conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 10
def min_loaves : ℕ := 15
def max_loaves : ℕ := 30
def unsold_price : ℝ := 2

def frequency_demand : List (ℕ × ℕ) := [(15, 10), (18, 8), (21, 7), (24, 3), (27, 2)]

-- Problem 1: Probability the daily demand is at least 21 loaves
def probability_at_least (demand: ℕ) : ℚ :=
  let total_days := frequency_demand.sumBy (fun (_, f) => f)
  let at_least_days := frequency_demand.filter (fun (d, f) => d >= 21).sumBy (fun (_, f) => f)
  at_least_days / total_days

-- Problem 2: Profit Calculation
def daily_profit (demand: ℕ) (baked: ℕ) : ℝ :=
  let sold_price := min baked demand * (selling_price - cost_price)
  let unsold_loss := max 0 (baked - demand) * (unsold_price - cost_price)
  sold_price + unsold_loss

def average_daily_profit (baked: ℕ) : ℝ :=
  let daily_profits := frequency_demand.map (fun (demand, freq) => daily_profit demand baked * freq)
  daily_profits.sum / frequency_demand.sumBy (fun (_, f) => f)

theorem probability_demand_at_least_21_loaves : probability_at_least 21 = 2 / 5 :=
  sorry

theorem daily_profit_given_demand_15 : daily_profit 15 21 = 78 :=
  sorry

theorem average_profit_over_30_days : average_daily_profit 21 = 103.6 :=
  sorry

end probability_demand_at_least_21_loaves_daily_profit_given_demand_15_average_profit_over_30_days_l259_259578


namespace tyler_brother_age_difference_l259_259910

-- Definitions of Tyler's age and the sum of their ages:
def tyler_age : ℕ := 7
def sum_of_ages (brother_age : ℕ) : Prop := tyler_age + brother_age = 11

-- Proof problem: Prove that Tyler's brother's age minus Tyler's age equals 4 years.
theorem tyler_brother_age_difference (B : ℕ) (h : sum_of_ages B) : B - tyler_age = 4 :=
by
  sorry

end tyler_brother_age_difference_l259_259910


namespace problem_A_l259_259329

-- Setup the definitions given in the problem
def U := Set.Univ.real

def M : Set ℝ := {x | x ≤ 1}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Problem statement: prove that M ∩ N = N
theorem problem_A : M ∩ N = N :=
  sorry

end problem_A_l259_259329


namespace inequality_solution_l259_259983

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 > (3 * x - 2) / 2 - 1 → x < 2 :=
by
  intro h
  sorry

end inequality_solution_l259_259983


namespace min_distance_shortest_altitude_l259_259443

-- Definitions from conditions
def min_distance_between_opposite_edges (T : Type) [Tetrahedron T] : ℝ := sorry
def shortest_altitude_length (T : Type) [Tetrahedron T] : ℝ := sorry

-- Statement of the proof
theorem min_distance_shortest_altitude {T : Type} [Tetrahedron T] :
  2 * (min_distance_between_opposite_edges T) > (shortest_altitude_length T) := sorry

end min_distance_shortest_altitude_l259_259443


namespace triangle_smallest_angle_l259_259513

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l259_259513


namespace f_value_neg_3_div_2_l259_259709

noncomputable def f : ℝ → ℝ :=
  λ x, if -1 ≤ x ∧ x ≤ 0 then -2 * x * (x + 1) else sorry

theorem f_value_neg_3_div_2 : f (-3 / 2) = -1 / 2 :=
by
  -- condition (1): f(x + 1) is an odd function with a period of 2
  have odd_periodic : ∀ x, f (x + 2) = f x ∧ f (-(x + 1)) = -f (x + 1) := sorry,
  -- condition (2): For -1 ≤ x ≤ 0, f(x) = -2x(x + 1)
  have interval_value : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = -2 * x * (x + 1) := sorry,
  -- prove the main statement using the given conditions and proper substitution
  lift (f (-3 / 2)) using [odd_periodic, interval_value] to -1 / 2 with sorry

end f_value_neg_3_div_2_l259_259709


namespace part_I_part_II_l259_259822

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 3 + a n ^ 2 * (1 - a (n + 1)) + 1

noncomputable def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → b n = (1 - (a n) ^ 2 / (a (n + 1)) ^ 2) / (a n)

theorem part_I (a : ℕ → ℝ) (h_seq : sequence a) : ∀ n : ℕ, n > 0 → a (n + 1) > a n :=
sorry

theorem part_II (a b : ℕ → ℝ) (h_seq : sequence a) (h_b_seq : b_sequence a b) :
  ∀ n : ℕ, n > 0 → 0 < (∑ k in finset.range n, b k.succ) ∧ (∑ k in finset.range n, b k.succ) < 2 :=
sorry

end part_I_part_II_l259_259822


namespace second_day_distance_l259_259006

-- Define the problem conditions
def total_distance (a1 : ℕ): ℕ := 
  let r := (1 / 2 : ℝ)
  in a1 + (a1 * r) + (a1 * r^2) + (a1 * r^3) + (a1 * r^4) + (a1 * r^5)

theorem second_day_distance (a1 : ℕ) : 
  total_distance a1 = 378 → (a1 / 2) = 96 :=
by
  intro h
  sorry

end second_day_distance_l259_259006


namespace calculate_sine_of_theta_l259_259969

-- Define the parameters and the conditions
variable (area side median : ℝ)
variable (θ : ℝ)

-- Condition statements
def triangle_area := area = 24
def side_length := side = 8
def median_length := median = 7.5
def angle_sine := Real.sin θ = 4 / 5

-- The theorem essentially states that these conditions imply the specific value of sine of θ
theorem calculate_sine_of_theta (h1 : triangle_area) (h2 : side_length) (h3 : median_length) :
  1 / 2 * side * median * Real.sin θ = area → angle_sine :=
by
  intros h_eq
  sorry

end calculate_sine_of_theta_l259_259969


namespace total_sum_subsets_l259_259738

def M := {x : ℕ | 1 ≤ x ∧ x ≤ 10}

theorem total_sum_subsets :
  ∑ A in (finset.powerset (finset.range 11) \ {∅}),
    ∑ k in A, (-1 : ℤ) ^ k * (k : ℤ) = 2560 :=
sorry

end total_sum_subsets_l259_259738


namespace pizza_toppings_problem_l259_259939

theorem pizza_toppings_problem
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (pepperoni_mushroom_slices : ℕ)
  (pepperoni_olive_slices : ℕ)
  (mushroom_olive_slices : ℕ)
  (pepperoni_mushroom_olive_slices : ℕ) :
  total_slices = 20 →
  pepperoni_slices = 12 →
  mushroom_slices = 14 →
  olive_slices = 12 →
  pepperoni_mushroom_slices = 8 →
  pepperoni_olive_slices = 8 →
  mushroom_olive_slices = 8 →
  total_slices = pepperoni_slices + mushroom_slices + olive_slices
    - pepperoni_mushroom_slices - pepperoni_olive_slices - mushroom_olive_slices
    + pepperoni_mushroom_olive_slices →
  pepperoni_mushroom_olive_slices = 6 :=
by
  intros
  sorry

end pizza_toppings_problem_l259_259939


namespace consecutive_integers_divisor_l259_259298

theorem consecutive_integers_divisor {m n : ℕ} (hm : m < n) (a : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (a + i) * (a + j) % (m * n) = 0 :=
by
  sorry

end consecutive_integers_divisor_l259_259298


namespace days_before_A_quits_l259_259173

/-- A can complete the project in 20 days -/
def work_rate_A : ℚ := 1 / 20

/-- B can complete the project in 30 days -/
def work_rate_B : ℚ := 1 / 30

/-- Combined work rate of A and B -/
def work_rate_combined : ℚ := work_rate_A + work_rate_B

/-- A and B work together for (18 - x) days and B works alone for x days -/
def project_completed(A_B_work_days B_work_days : ℚ) : Prop :=
  (A_B_work_days * work_rate_combined + B_work_days * work_rate_B = 1)

/-- x is the number of days before completion that A quits,
    and the project is completed in 18 days -/
theorem days_before_A_quits (x : ℚ) :
  let A_B_work_days := 18 - x in
  let B_work_days := x in
  project_completed A_B_work_days B_work_days →
  x = 10 := sorry

end days_before_A_quits_l259_259173


namespace james_total_vegetables_l259_259015

def james_vegetable_count (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

theorem james_total_vegetables 
    (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
    a = 22 → b = 18 → c = 15 → d = 10 → e = 12 →
    james_vegetable_count a b c d e = 77 :=
by
  intros ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end james_total_vegetables_l259_259015


namespace kath_movie_cost_l259_259778

theorem kath_movie_cost :
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  discounted_price * number_of_people = 30 := by
  -- Definitions from conditions
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  -- Derived calculation based on conditions
  have h_discounted_price : discounted_price = 5 := by
    calc
      discounted_price = 8 - 3 : by sorry
      ... = 5 : by sorry
  have h_number_of_people : number_of_people = 6 := by
    calc
      number_of_people = 1 + 2 + 3 : by sorry
      ... = 6 : by sorry
  show 5 * 6 = 30 from sorry

end kath_movie_cost_l259_259778


namespace inequality_proof_l259_259699

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l259_259699


namespace train_crossing_time_l259_259879

-- Definitions based on conditions from the problem
def length_of_train_and_platform := 900 -- in meters
def speed_km_per_hr := 108 -- in km/hr
def distance := 2 * length_of_train_and_platform -- distance to be covered
def speed_m_per_s := (speed_km_per_hr * 1000) / 3600 -- converted speed

-- Theorem stating the time to cross the platform is 60 seconds
theorem train_crossing_time : distance / speed_m_per_s = 60 := by
  sorry

end train_crossing_time_l259_259879


namespace polynomial_exists_and_find_c3_l259_259435

-- Define the conditions for the problem
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def E (m : ℕ) : ℕ := sorry  -- Placeholder for the actual function E(m)

-- Define the statement of the theorem
theorem polynomial_exists_and_find_c3 (m : ℕ) (h1 : m ≥ 6) (h2 : is_even m) : 
  ∃ (p : ℕ → ℕ), (∃ c4 c3 c2 c1 c0 : ℕ, 
    p = λ x, c4 * x^4 + c3 * x^3 + c2 * x^2 + c1 * x + c0) ∧
    p m = E m :=
sorry

end polynomial_exists_and_find_c3_l259_259435


namespace bicyclist_speed_for_remaining_distance_l259_259456

theorem bicyclist_speed_for_remaining_distance
  (total_distance : ℝ)
  (dist_first_part : ℝ)
  (speed_first_part : ℝ)
  (average_speed : ℝ)
  (time_first_part : ℝ := dist_first_part / speed_first_part)
  (remaining_distance : ℝ := total_distance - dist_first_part)
  (total_time : ℝ := total_distance / average_speed)
  (time_remaining : ℝ := total_time - time_first_part)
  : total_distance = 850 ∧ 
    dist_first_part = 400 ∧ 
    speed_first_part = 20 ∧ 
    average_speed = 17 
    → (450 / time_remaining = 15) := 
by 
  -- Use destructuring and arithmetic based on the given conditions
  intros h,
  obtain ⟨h1, h2, h3, h4⟩ := h,
  simp [*, total_distance, dist_first_part, speed_first_part, average_speed] at *,
  sorry

end bicyclist_speed_for_remaining_distance_l259_259456


namespace polygon_sides_l259_259594

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end polygon_sides_l259_259594


namespace square_side_increase_l259_259481

theorem square_side_increase (s : ℝ) :
  let new_side := 1.5 * s
  let new_area := new_side^2
  let original_area := s^2
  let new_perimeter := 4 * new_side
  let original_perimeter := 4 * s
  let new_diagonal := new_side * Real.sqrt 2
  let original_diagonal := s * Real.sqrt 2
  (new_area - original_area) / original_area * 100 = 125 ∧
  (new_perimeter - original_perimeter) / original_perimeter * 100 = 50 ∧
  (new_diagonal - original_diagonal) / original_diagonal * 100 = 50 :=
by
  sorry

end square_side_increase_l259_259481


namespace tangent_line_parallel_x_axis_l259_259765

def f (x : ℝ) : ℝ := x^4 - 4 * x

theorem tangent_line_parallel_x_axis :
  ∃ (m n : ℝ), (n = f m) ∧ (deriv f m = 0) ∧ (m, n) = (1, -3) := by
  sorry

end tangent_line_parallel_x_axis_l259_259765


namespace number_of_subsets_l259_259328

noncomputable def M : set ℤ := {-1, 1}

theorem number_of_subsets (N : set ℤ) (h : N ⊆ M) :
  finset.card (finset.powerset (finset.from_set M)) = 4 :=
by
  sorry

end number_of_subsets_l259_259328


namespace alpha_plus_beta_eq_pi_simplified_expression_l259_259704

noncomputable theory

open Real

variables {α β : ℝ}
-- Conditions
def point_A := (cos α, 0)
def point_B := (0, sin α)
def point_C := (cos β, sin β)

-- Condition that vectors are parallel
def vectors_parallel : Prop := ((-cos α, sin α) = (λx, x * (cos β, sin β)).default) -- Parallel condition \overrightarrow{AB} \parallel \overrightarrow{OC}

-- Additional conditions
def alpha_beta_conditions : Prop := (0 < α) ∧ (α < β) ∧ (β < π)

-- Proof goal 1
theorem alpha_plus_beta_eq_pi (h1 : vectors_parallel) (h2 : alpha_beta_conditions) : α + β = π := 
sorry

-- Proof goal 2
theorem simplified_expression (h1 : vectors_parallel) (h2 : alpha_beta_conditions) : 
  (\frac{(1 + sin α - cos β) * (sin (α/2) - sin (β/2))}{sqrt (2 + 2*cos α)} = -2 * (sin (α / 2) + cos (α / 2)) * cos α) := 
sorry

end alpha_plus_beta_eq_pi_simplified_expression_l259_259704


namespace man_speed_l259_259592

theorem man_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h1 : distance = 12)
  (h2 : time_minutes = 72)
  (h3 : time_hours = time_minutes / 60)
  (h4 : speed = distance / time_hours) : speed = 10 :=
by
  sorry

end man_speed_l259_259592


namespace solve_inequality_l259_259476

theorem solve_inequality (x : ℝ) :
  (x ∈ set.Ioo (neg_infty : ℝ) (-4) ∪ set.Ici 0) ↔ (x / (x + 4) ≥ 0) :=
by
  -- proof omitted
  sorry

end solve_inequality_l259_259476


namespace polynomial_coeff_sum_l259_259356

theorem polynomial_coeff_sum :
  let p1 : Polynomial ℝ := Polynomial.C 4 * Polynomial.X ^ 2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 5
  let p2 : Polynomial ℝ := Polynomial.C 8 - Polynomial.C 3 * Polynomial.X
  let product : Polynomial ℝ := p1 * p2
  let a : ℝ := - (product.coeff 3)
  let b : ℝ := (product.coeff 2)
  let c : ℝ := - (product.coeff 1)
  let d : ℝ := (product.coeff 0)
  8 * a + 4 * b + 2 * c + d = 18 := sorry

end polynomial_coeff_sum_l259_259356


namespace numberOfValidFiveDigitNumbers_l259_259659

namespace MathProof

def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def firstAndLastDigitsEqual (n : ℕ) : Prop := 
  let firstDigit := (n / 10000) % 10
  let lastDigit := n % 10
  firstDigit = lastDigit

def sumOfDigitsDivisibleBy5 (n : ℕ) : Prop := 
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 + d2 + d3 + d4 + d5) % 5 = 0

theorem numberOfValidFiveDigitNumbers :
  ∃ (count : ℕ), count = 200 ∧ 
  count = Nat.card {n : ℕ // isFiveDigitNumber n ∧ 
                                isDivisibleBy5 n ∧ 
                                firstAndLastDigitsEqual n ∧ 
                                sumOfDigitsDivisibleBy5 n} :=
by
  sorry

end MathProof

end numberOfValidFiveDigitNumbers_l259_259659


namespace Bobby_candy_chocolate_sum_l259_259984

/-
  Bobby ate 33 pieces of candy, then ate 4 more, and he also ate 14 pieces of chocolate.
  Prove that the total number of pieces of candy and chocolate he ate altogether is 51.
-/

theorem Bobby_candy_chocolate_sum :
  let initial_candy := 33
  let more_candy := 4
  let chocolate := 14
  let total_candy := initial_candy + more_candy
  total_candy + chocolate = 51 :=
by
  -- The theorem asserts the problem; apologies, the proof is not required here.
  sorry

end Bobby_candy_chocolate_sum_l259_259984


namespace log8_1023_rounded_l259_259532

theorem log8_1023_rounded :
  512 < 1023 ∧ 1023 < 1024 ∧ (∀ x y : ℝ, x < y → real.log x < real.log y) →
  real.log 1023 / real.log 8 ≈ 3 :=
by
  -- proof skipped
  sorry

end log8_1023_rounded_l259_259532


namespace range_of_x_of_sqrt_x_plus_3_l259_259388

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259388


namespace question_1_answer_question_2_answer_l259_259336

structure Vector :=
  (x : ℝ)
  (y : ℝ)

def a : Vector := ⟨1, 0⟩
def b : Vector := ⟨1, 4⟩

def parallel (v1 v2 : Vector) : Prop :=
  v1.y * v2.x = v1.x * v2.y

def dot_product (v1 v2 : Vector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def question_1_condition (k : ℝ) : Prop :=
  parallel ⟨k + 1, 4⟩ ⟨3, 8⟩

theorem question_1_answer : ∀ k : ℝ, question_1_condition k ↔ k = 1 / 2 := 
by sorry

noncomputable def question_2_condition (k : ℝ) : Prop :=
  dot_product ⟨k + 1, 4⟩ ⟨3, 8⟩ > 0 ∧ (parallel ⟨k + 1, 4⟩ ⟨3, 8⟩) = false

theorem question_2_answer : ∀ k : ℝ, question_2_condition k ↔ k > -35 / 3 ∧ k ≠ 1 / 2 := 
by sorry

end question_1_answer_question_2_answer_l259_259336


namespace number_of_possible_values_of_a_l259_259846

theorem number_of_possible_values_of_a 
  (a b c d : ℕ) 
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_sum : a + b + c + d = 2020)
  (h_diff_squared : a^2 - b^2 + c^2 - d^2 = 2020) :
  (finset.range 2021).filter (λ a, ∃ b c d, 
                                 b > 0 ∧ c > 0 ∧ d > 0 ∧ 
                                 a > b ∧ b > c ∧ c > d ∧ 
                                 a + b + c + d = 2020 ∧ 
                                 a^2 - b^2 + c^2 - d^2 = 2020).card = 501 :=
  sorry

end number_of_possible_values_of_a_l259_259846


namespace find_m_l259_259334

def vector (α : Type*) := α × α

def dot_product {α : Type*} [HasMul α] [HasAdd α]
  (v w : vector α) : α :=
(v.1 * w.1) + (v.2 * w.2)

def given_vectors (m : ℝ) : 
  vector ℝ × vector ℝ := 
  ((2, m), (1, -1))

def vector_add {α : Type*} [HasAdd α] 
  (v w : vector α) : vector α :=
(v.1 + w.1, v.2 + w.2)

theorem find_m (m : ℝ) :
  let a := (2, m : ℝ)
  let b := (1, -1 : ℝ)
  dot_product b (vector_add a (vector (1, -1) * 2)) = 0 → m = 6 := 
begin
  intro h,
  sorry
end

end find_m_l259_259334


namespace S_15_is_24_l259_259294

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom h_arith_seq   : ∀ n, a(n + 1) - a n = a 1 - a 0
axiom h_S5          : S 5 = 28
axiom h_S10         : S 10 = 36

-- Prove that S 15 = 24
theorem S_15_is_24 : S 15 = 24 :=
  sorry

end S_15_is_24_l259_259294


namespace total_admission_cost_l259_259775

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l259_259775


namespace no_such_polynomials_exists_polynomials_l259_259571

-- First statement: No polynomials P, Q, R exist such that the given condition holds
theorem no_such_polynomials (P Q R : ℕ → ℕ → ℕ → ℤ) : ¬ ∀ x y z : ℤ,
  (P (x - y + 1) (y - z - 1) (z - 2 * x + 1) = 1 ∧ 
   Q (x - y + 1) (y - z - 1) (z - 2 * x + 1) = 1 ∧ 
   R (x - y + 1) (y - z - 1) (z - 2 * x + 1) = 1) :=
sorry

-- Second statement: Certain polynomials P, Q, R do exist
theorem exists_polynomials : ∃ P Q R : ℕ → ℕ → ℕ → ℤ,
  ∀ (x y z : ℤ), (x - y + 1)^3 * P(x, y, z) + (y - z - 1)^3 * Q(x, y, z) + (z - x + 1)^3 * R(x, y, z) = 1 := 
sorry

end no_such_polynomials_exists_polynomials_l259_259571


namespace polygon_sides_l259_259596

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : exterior_angle = 36 ∧ (∑ i in finset.range n, 360 / n) = 360 → n = 10 := 
by
  intros h
  sorry

end polygon_sides_l259_259596


namespace three_11th_graders_l259_259900

-- Definitions of the girls
inductive Grade
| Eleventh
| Ninth

structure Girl :=
(name : String)
(grade : Grade)
(won : Girl → Prop)

-- Definitions of the girls in the problem
def Veronika : Girl := { name := "Veronika", grade := Grade.Eleventh, won := λ g, g.name = "Rita" ∨ g.name = "Maria" }
def Yulia : Girl := { name := "Yulia", grade := Grade.Eleventh, won := λ g, g.name = "Svetlana" }
def Maria : Girl := { name := "Maria", grade := Grade.Ninth, won := λ g, g.name = "Yulia" }
def Rita : Girl := { name := "Rita", grade := Grade.Ninth, won := λ g, false }
def Svetlana : Girl := { name := "Svetlana", grade := Grade.Ninth, won := λ g, false }

-- Hypothetical third 11th grader (not explicitly named in the problem context but should exist)
noncomputable def Third11thGrader : Girl := { name := "Third11thGrader", grade := Grade.Eleventh, won := λ g, false }

-- Function to list and prove the 11th graders excluding duplicates based on conditions provided
def identify11thGraders : List Girl := 
  [Veronika, Yulia, Third11thGrader]

theorem three_11th_graders (Vs : List Girl) :
  Vs = identify11thGraders → (Veronika ∈ Vs ∧ Yulia ∈ Vs ∧ Third11thGrader ∈ Vs) := by
  intro h
  rw [h]
  simp
  sorry

end three_11th_graders_l259_259900


namespace sum_first_19_natural_numbers_l259_259221

theorem sum_first_19_natural_numbers :
  let n := 19 in nat.sum_range n + 1 = 190 :=
by
  sorry

end sum_first_19_natural_numbers_l259_259221


namespace solve_for_x_l259_259475

theorem solve_for_x : ∃ x : ℝ, 7 + 3.5 * x = 2.1 * x - 30 * 1.5 ∧ x = -37.142857 :=
by
  use -37.142857
  split
  · norm_num
  · rfl

end solve_for_x_l259_259475


namespace count_interesting_numbers_l259_259938

def is_interesting (n : ℕ) : Prop := 
  (nat.digits 10 n).length = 10 ∧ 
  (nat.digits 10 n).nodup ∧ 
  n % 11111 = 0

theorem count_interesting_numbers : 
  (finset.filter is_interesting (finset.Icc 1000000000 9999999999)).card = 3456 :=
sorry

end count_interesting_numbers_l259_259938


namespace intersection_of_PQ_RS_correct_l259_259783

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 1/9
  let s := 2/3
  (3 + 10 * t, -4 - 10 * t, 4 + 5 * t)

theorem intersection_of_PQ_RS_correct :
  let P := (3, -4, 4)
  let Q := (13, -14, 9)
  let R := (-3, 6, -9)
  let S := (1, -2, 7)
  intersection_point P Q R S = (40/9, -76/9, 49/9) :=
by {
  sorry
}

end intersection_of_PQ_RS_correct_l259_259783


namespace no_positive_real_p_exists_l259_259668

noncomputable def f (n : ℕ) : ℕ :=
  ((list.range (n + 1)).filter (λ k, ¬(k.digits 10).any (λ d, d = 9))).length

theorem no_positive_real_p_exists :
  ¬ ∃ p > 0, ∀ n : ℕ, (f n : ℝ) / n ≥ p := by
  sorry

end no_positive_real_p_exists_l259_259668


namespace height_of_stack_correct_l259_259781

namespace PaperStack

-- Define the problem conditions
def sheets_per_package : ℕ := 500
def thickness_per_sheet_mm : ℝ := 0.1
def packages_per_stack : ℕ := 60
def mm_to_m : ℝ := 1000.0

-- Statement: the height of the stack of 60 paper packages
theorem height_of_stack_correct :
  (sheets_per_package * thickness_per_sheet_mm * packages_per_stack) / mm_to_m = 3 :=
sorry

end PaperStack

end height_of_stack_correct_l259_259781


namespace different_rectangles_in_fourth_column_l259_259977

theorem different_rectangles_in_fourth_column (grid : fin 7 → fin 7 → ℕ) 
  (h1 : ∀ (x y : fin 7), ∃ (n : ℕ), grid x y = n)
  (h2 : ∀ (rect : set (fin 7 × fin 7)), ∃! n : ℕ, ∀ (i j : fin 7), (i, j) ∈ rect → grid i j = n)
  (h3 : ∀ (y : fin 7), ∃ (rects : fin 7 × (set (fin 7 × fin 1))), rects.1 = y) :
  ∃ ! (R : fin 7 → fin 7 → Prop), 
    (∀ x, (R x 3 → ∃ ! n : ℕ, grid x 3 = n ∧ R x 3 = True) ) ∧ 
    (∑ i in finset.range 7, if (R i 3) then 1 else 0 = 4) := sorry

end different_rectangles_in_fourth_column_l259_259977


namespace marked_price_correct_l259_259868

noncomputable def marked_price (cost_price : ℝ) (profit_margin : ℝ) (selling_percentage : ℝ) : ℝ :=
  (cost_price * (1 + profit_margin)) / selling_percentage

theorem marked_price_correct :
  marked_price 1360 0.15 0.8 = 1955 :=
by
  sorry

end marked_price_correct_l259_259868


namespace all_equal_l259_259108

theorem all_equal (xs xsp : Fin 2011 → ℝ) (h : ∀ i : Fin 2011, xs i + xs ((i + 1) % 2011) = 2 * xsp i) (perm : ∃ σ : Fin 2011 ≃ Fin 2011, ∀ i, xsp i = xs (σ i)) :
  ∀ i j : Fin 2011, xs i = xs j := 
sorry

end all_equal_l259_259108


namespace polygon_sides_l259_259595

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : exterior_angle = 36 ∧ (∑ i in finset.range n, 360 / n) = 360 → n = 10 := 
by
  intros h
  sorry

end polygon_sides_l259_259595


namespace solution_of_system_l259_259436

variable (n : ℕ) (x : Fin n → ℕ)

def system_of_equations (x : Fin n → ℕ) :=
  (∀ i : Fin n, x i ≥ 0) ∧
  (∑ i in Finset.range n, x i = n + 2) ∧
  (∑ i in Finset.range n, (i + 1) * x i = 2 * n + 2) ∧
  (∑ i in Finset.range n, (i + 1)^2 * x i = n^2 + n + 4) ∧
  (∑ i in Finset.range n, (i + 1)^3 * x i = n^3 + n + 8)

theorem solution_of_system (h₀ : 5 < n) : 
  ∃ x : Fin n → ℕ, 
    system_of_equations n x ∧ 
    (∀ i, x i = if i = 0 then n else if i = 1 then 1 else if i.1 = n-1 then 1 else 0) :=
begin
  sorry,
end

end solution_of_system_l259_259436


namespace a_n_formula_l259_259933

noncomputable def a : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := 1 / (n + 2) * (1 - ∑ i in Finset.range n.succ, a i.succ)

theorem a_n_formula (n : ℕ) (n_pos : 0 < n) : a n = 1 / (n * (n + 1)) :=
by sorry

end a_n_formula_l259_259933


namespace curve_tangents_intersection_l259_259719

theorem curve_tangents_intersection (a : ℝ) :
  (∃ x₀ y₀, y₀ = Real.exp x₀ ∧ y₀ = (x₀ + a)^2 ∧ Real.exp x₀ = 2 * (x₀ + a)) → a = 2 - Real.log 4 :=
by
  sorry

end curve_tangents_intersection_l259_259719


namespace range_of_a_minus_abs_b_l259_259751

theorem range_of_a_minus_abs_b (a b : ℝ) (h₁ : 1 < a ∧ a < 3) (h₂ : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l259_259751


namespace fill_cistern_l259_259154

theorem fill_cistern (p_rate q_rate : ℝ) (total_time first_pipe_time : ℝ) (remaining_fraction : ℝ): 
  p_rate = 1/12 → q_rate = 1/15 → total_time = 2 → remaining_fraction = 7/10 → 
  (remaining_fraction / q_rate) = 10.5 :=
by
  sorry

end fill_cistern_l259_259154


namespace smallest_n_for_g_greater_than_21_l259_259038

noncomputable def g (n : ℕ) : ℕ :=
  Nat.find (λ k, k.factorial % n = 0)

theorem smallest_n_for_g_greater_than_21 {r : ℕ} (hr : r ≥ 23) :
  let n := 21 * r in g n > 21 :=
by
  let n := 21 * r
  have hr23 : r ≥ 23 := hr
  have hdiv : 23 ∣ n := by
    sorry
  have gn : g n ≥ 23 := by
    sorry
  exact Nat.lt_of_le_of_lt 21 gn

end smallest_n_for_g_greater_than_21_l259_259038


namespace player_b_winning_strategy_l259_259241

def tile := (ℕ × ℕ)
def domino_set : set tile := {(6,6), (6,5), (6,4), (6,3), (6,2), (6,1), (6,0),
                             (5,5), (5,4), (5,3), (5,2), (5,1), (5,0),
                             (4,4), (4,3), (4,2), (4,1), (4,0),
                             (3,3), (3,2), (3,1), (3,0),
                             (2,2), (2,1), (2,0),
                             (1,1), (1,0),
                             (0,0)}

def valid_number_set : set ℕ := {0, 1, 2, 3, 4, 5, 6}

def game_state := {numbers : finset ℕ // numbers ⊆ valid_number_set ∧ numbers.card ≤ 7}

def valid_move (s : game_state) (t : tile) : Prop :=
  t.fst ∈ valid_number_set ∧ t.snd ∈ valid_number_set ∧ ¬(t.fst ∈ s.numbers ∧ t.snd ∈ s.numbers)

theorem player_b_winning_strategy :
  ∀ (moves : list tile), 
    (∀ t ∈ moves, t ∈ domino_set) →
    (∀ i, i < moves.length → (if i % 2 = 0 then player "A" else player "B")) →
    (game_state_after_moves moves).numbers = valid_number_set →
    last_move_by_player moves = player "B" :=
sorry

end player_b_winning_strategy_l259_259241


namespace upper_limit_opinion_l259_259624

theorem upper_limit_opinion (w : ℝ) 
  (H1 : 61 < w ∧ w < 72) 
  (H2 : 60 < w ∧ w < 70) 
  (H3 : (61 + w) / 2 = 63) : w = 65 := 
by
  sorry

end upper_limit_opinion_l259_259624


namespace range_of_x_in_sqrt_x_plus_3_l259_259400

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259400


namespace rectangle_breadth_l259_259500

theorem rectangle_breadth (length radius side breadth: ℝ)
  (h1: length = (2/5) * radius)
  (h2: radius = side)
  (h3: side ^ 2 = 1600)
  (h4: length * breadth = 160) :
  breadth = 10 := 
by
  sorry

end rectangle_breadth_l259_259500


namespace problem1_problem2_problem3_problem4_l259_259988

theorem problem1 : 12 - (-1) + (-7) = 6 := by
  sorry

theorem problem2 : -3.5 * (-3 / 4) / (7 / 8) = 3 := by
  sorry

theorem problem3 : (1 / 3 - 1 / 6 - 1 / 12) * (-12) = -1 := by
  sorry

theorem problem4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by
  sorry

end problem1_problem2_problem3_problem4_l259_259988


namespace half_guests_want_two_burgers_l259_259990

theorem half_guests_want_two_burgers 
  (total_guests : ℕ) (half_guests : ℕ)
  (time_per_side : ℕ) (time_per_burger : ℕ)
  (grill_capacity : ℕ) (total_time : ℕ)
  (guests_one_burger : ℕ) (total_burgers : ℕ) : 
  total_guests = 30 →
  time_per_side = 4 →
  time_per_burger = 8 →
  grill_capacity = 5 →
  total_time = 72 →
  guests_one_burger = 15 →
  total_burgers = 45 →
  half_guests * 2 = total_burgers - guests_one_burger →
  half_guests = 15 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end half_guests_want_two_burgers_l259_259990


namespace john_music_hours_per_month_l259_259016

-- Define the conditions
def avg_song_length := 3 -- minutes
def song_price := 0.50 -- dollars
def annual_music_expenditure := 2400 -- dollars

-- Define the question and the correct answer statement
theorem john_music_hours_per_month (Havg : avg_song_length = 3) (Hprice : song_price = 0.50) (Hexpenditure : annual_music_expenditure = 2400) : 
  ((annual_music_expenditure / 12) / song_price) * avg_song_length / 60 = 20 := 
  by 
    rw [Havg, Hprice, Hexpenditure]
    sorry

end john_music_hours_per_month_l259_259016


namespace fraction_of_full_tank_used_l259_259944

def full_tank := 12              -- Full tank in gallons
def initial_speed1 := 50         -- Speed in the first part (miles/hour)
def time1 := 3                   -- Time in the first part (hours)
def consumption_rate1 := 40      -- Consumption rate in the first part (miles/gallon)
def refill := 5                  -- Gallons refilled after the first part
def initial_speed2 := 60         -- Speed in the second part (miles/hour)
def time2 := 4                   -- Time in the second part (hours)
def consumption_rate2 := 30      -- Consumption rate in the second part (miles/gallon)

theorem fraction_of_full_tank_used :
  let distance1 := initial_speed1 * time1 in
  let gasoline_used1 := distance1 / consumption_rate1 in
  let gasoline_after_refill := full_tank - gasoline_used1 + refill in
  let distance2 := initial_speed2 * time2 in
  let gasoline_used2 := distance2 / consumption_rate2 in
  let total_gasoline_used := gasoline_used1 + gasoline_used2 in
  let fraction_used := total_gasoline_used / full_tank in
  fraction_used = 47 / 48 :=
by
  sorry

end fraction_of_full_tank_used_l259_259944


namespace molecular_weight_p_Toluidine_is_correct_l259_259986

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01

def molecular_formula_p_Toluidine : (ℕ × ℕ × ℕ) := (7, 9, 1)

def molecular_weight (f : ℕ × ℕ × ℕ) (wC wH wN : ℝ) : ℝ :=
  (f.1 * wC) + (f.2 * wH) + (f.3 * wN)

theorem molecular_weight_p_Toluidine_is_correct :
  molecular_weight molecular_formula_p_Toluidine atomic_weight_C atomic_weight_H atomic_weight_N = 107.152 :=
by
  sorry

end molecular_weight_p_Toluidine_is_correct_l259_259986


namespace fraction_of_work_completed_in_25_days_l259_259829

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end fraction_of_work_completed_in_25_days_l259_259829


namespace union_of_M_and_N_l259_259326

open Set

def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }  
def N : Set ℝ := { x | (1/2)^x ≤ 4 }

theorem union_of_M_and_N : M ∪ N = univ := 
by 
  sorry

end union_of_M_and_N_l259_259326


namespace right_triangle_AB_CA_BC_l259_259139

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end right_triangle_AB_CA_BC_l259_259139


namespace sum_of_squares_of_distances_in_tetrahedron_l259_259466

theorem sum_of_squares_of_distances_in_tetrahedron
  (A B C D M : EuclideanSpace ℝ (Fin 3))
  (E F G H I J : EuclideanSpace ℝ (Fin 3))
  (G_centroid : EuclideanSpace ℝ (Fin 3))
  (midpoints : (EuclideanSpace ℝ (Fin 3)) → (EuclideanSpace ℝ (Fin 3)) → (EuclideanSpace ℝ (Fin 3)))
  (hE : E = midpoints A B)
  (hF : F = midpoints C D)
  (hG : G = midpoints A C)
  (hH : H = midpoints B D)
  (hI : I = midpoints A D)
  (hJ : J = midpoints B C)
  (h_centroid : G_centroid = (A + B + C + D) / 4) :
  (dist M A) ^ 2 + (dist M B) ^ 2 + (dist M C) ^ 2 + (dist M D) ^ 2 =
    2 * ((dist E F) ^ 2 + (dist G H) ^ 2 + (dist I J) ^ 2) +
    4 * (dist M G_centroid) ^ 2 := by
  sorry

end sum_of_squares_of_distances_in_tetrahedron_l259_259466


namespace find_r_plus_s_l259_259373

noncomputable def quadrilateral_ABCD : Prop :=
  ∃ (r s : ℕ), 
  r > 0 ∧ s > 0 ∧ 
  let AB := r + Real.sqrt s in
  let BC := 7 in
  let CD := 13 in
  let AD := 11 in
  ∃ (A B C D E : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E],
  ∃ (a b c d e : A → ℝ),
  ∃ (angle_A : ℝ) (angle_B : ℝ),
  angle_A = 60 ∧ 
  angle_B = 60 ∧ 
  AB = 8 + Real.sqrt 142

theorem find_r_plus_s : ∃ (r s : ℕ), r + s = 150 :=
by
  have quadrilateral_ABCD
  use 8, 142
  simp
  use 150
  sorry

end find_r_plus_s_l259_259373


namespace probability_product_multiple_of_10_l259_259359

open Finset

def S : Finset ℕ := {2, 3, 5, 6, 9}

theorem probability_product_multiple_of_10 :
  (S.choose 3).filter (λ t => 10 ∣ t.prod id).card = 3 →
  (S.choose 3).card = 10 →
  ((S.choose 3).filter (λ t => 10 ∣ t.prod id).card / (S.choose 3).card : ℚ) = 3 / 10 := by
  intro h_favorable h_total
  rw [div_eq_mul_inv, int.coe_nat_div, h_favorable, h_total]
  norm_num
  sorry

end probability_product_multiple_of_10_l259_259359


namespace tangent_line_at_1_minimum_value_of_f_l259_259725

noncomputable def f (x : ℝ) := x^2 + x - Real.log x

theorem tangent_line_at_1 :
  ∃ m b, (∀ x, (f' : ℝ → ℝ) x = 2 * x + 1 - 1 / x)
    ∧ (∀ x, f' 1 = 2)
    ∧ (∀ x, f 1 = 2)
    ∧ (m = 2)
    ∧ (b = 0)
    ∧ (∀ x, y = m * x + b := sorry)

theorem minimum_value_of_f :
  ∃ x_min, (f' : ℝ → ℝ) x = 0
    ∧ (∀ x, (f' x > 0) ↔ x > 1/2)
    ∧ (∀ x, (f' x < 0) ↔ 0 < x < 1/2)
    ∧ (x_min = 1/2)
    ∧ (f x_min = 3/4 + Real.log 2) := sorry

end tangent_line_at_1_minimum_value_of_f_l259_259725


namespace dot_product_sum_l259_259332

open Real

variables (a b c : ℝ^3)

-- Conditions
theorem dot_product_sum (h : a + b + c = 0) (ha : ‖a‖ = 3) (hb : ‖b‖ = 1) (hc : ‖c‖ = 4) :
  a.dot b + b.dot c + a.dot c = -13 :=
sorry

end dot_product_sum_l259_259332


namespace new_sum_formula_l259_259688

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (s : ℝ)
variable (h_sum : ∑ i, x i = s)

theorem new_sum_formula :
  ∑ i, ((x i + 10) ^ 2 - 30) = 20 * s + 70 * n + ∑ i, (x i)^2 := by
    sorry

end new_sum_formula_l259_259688


namespace rectangle_dimensions_l259_259559

theorem rectangle_dimensions (w l : ℕ) (h : l = w + 5) (hp : 2 * l + 2 * w = 34) : w = 6 ∧ l = 11 := 
by 
  sorry

end rectangle_dimensions_l259_259559


namespace soda_difference_l259_259587

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := 
  by
  sorry

end soda_difference_l259_259587


namespace charlyn_viewable_area_l259_259225

theorem charlyn_viewable_area 
  (length width : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 4)
  (view_distance : ℝ) 
  (h_view_distance : view_distance = 1) 
  : (28 + Real.pi).round = 31 := by
  sorry

end charlyn_viewable_area_l259_259225


namespace product_of_two_numbers_l259_259499

open Nat

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 60) (h_gcd : gcd a b = 5) : a * b = 300 :=
by
  sorry

end product_of_two_numbers_l259_259499


namespace sin_cos_difference_l259_259647

theorem sin_cos_difference (deg65 : ℝ) (deg35 : ℝ) (sin : ℝ → ℝ) (cos : ℝ → ℝ) (sin_diff : ∀ A B, sin (A - B) = sin A * cos B - cos A * sin B) :
  deg65 = 65 → deg35 = 35 → sin 30 = 1 / 2 →
  sin deg65 * cos deg35 - cos deg65 * sin deg35 = 1 / 2 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  have H : sin 30 = 1 / 2 := h3
  rw [sin_diff 65 35, H]
  sorry

end sin_cos_difference_l259_259647


namespace boat_navigation_under_arch_l259_259898

theorem boat_navigation_under_arch (h_arch : ℝ) (w_arch: ℝ) (boat_width: ℝ) (boat_height: ℝ) (boat_above_water: ℝ) :
  (h_arch = 5) → 
  (w_arch = 8) → 
  (boat_width = 4) → 
  (boat_height = 2) → 
  (boat_above_water = 0.75) →
  (h_arch - 2 = 3) :=
by
  intros h_arch_eq w_arch_eq boat_w_eq boat_h_eq boat_above_water_eq
  sorry

end boat_navigation_under_arch_l259_259898


namespace second_part_lent_years_l259_259195

theorem second_part_lent_years (total_sum : ℝ)
  (first_interest_rate : ℝ)
  (second_interest_rate : ℝ)
  (amount_second_part : ℝ)
  (first_time : ℕ) : 
  total_sum = 2665 → first_interest_rate = 0.03 → second_interest_rate = 0.05 →
  amount_second_part = 1332.5 → first_time = 5 → 
  n = 3 :=
by
  -- Assuming we assign the correct variable names and premises
  let x := total_sum - amount_second_part
  let I1 := x * first_interest_rate * first_time 
  let I2 := amount_second_part * second_interest_rate * n
  have h : I1 = I2, sorry
  -- Given x = 1332.5 substituting in I1 = I2 would solve n = 3
  exact sorry

end second_part_lent_years_l259_259195


namespace point_opposite_sides_line_l259_259717

theorem point_opposite_sides_line (a : ℝ) :
  ((1 + a * 1 + 1 > 0) ∧ (0 + a * (-2) + 1 < 0)) ∨ 
  ((1 + a * 1 + 1 < 0) ∧ (0 + a * (-2) + 1 > 0)) ↔ 
  a ∈ set.Ioo (-∞) (-2) ∪ set.Ioo (0.5) (∞) :=
by {
  sorry
}

end point_opposite_sides_line_l259_259717


namespace intersection_A_B_at_3_range_of_a_l259_259028

open Set

-- Definitions from the condition
def A (x : ℝ) : Prop := abs x ≥ 2
def B (x a : ℝ) : Prop := (x - 2 * a) * (x + 3) < 0

-- Part (Ⅰ)
theorem intersection_A_B_at_3 :
  let a := 3
  let A := {x : ℝ | abs x ≥ 2}
  let B := {x : ℝ | (x - 6) * (x + 3) < 0}
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (2 ≤ x ∧ x < 6)} :=
by
  sorry

-- Part (Ⅱ)
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, A x ∨ B x a) → a ≥ 1 :=
by
  sorry

end intersection_A_B_at_3_range_of_a_l259_259028


namespace sqrt_domain_l259_259381

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l259_259381


namespace cookies_per_person_l259_259920

theorem cookies_per_person (total_cookies : ℕ) (people : ℕ) (h1 : total_cookies = 24) (h2 : people = 6) : total_cookies / people = 4 := by
  rw [h1, h2]
  norm_num
  sorry

end cookies_per_person_l259_259920


namespace rectangle_area_increase_l259_259599

theorem rectangle_area_increase (a b : ℝ) :
  let new_length := (1 + 1/4) * a
  let new_width := (1 + 1/5) * b
  let original_area := a * b
  let new_area := new_length * new_width
  let area_increase := new_area - original_area
  (area_increase / original_area) = 1/2 := 
by
  sorry

end rectangle_area_increase_l259_259599


namespace sum_of_cubes_l259_259498

variables (n : ℕ) (x : ℂ)

def a : ℂ := ∑ i in (finset.range n).filter (λ k, k % 3 = 0), nat.choose n i * x ^ i
def b : ℂ := ∑ i in (finset.range n).filter (λ k, k % 3 = 1), nat.choose n i * x ^ i
def c : ℂ := ∑ i in (finset.range n).filter (λ k, k % 3 = 2), nat.choose n i * x ^ i

theorem sum_of_cubes (n : ℕ) (x : ℂ) : a n x ^ 3 + b n x ^ 3 + c n x ^ 3 - 3 * a n x * b n x * c n x = (1 + x^3)^n :=
by 
  sorry

end sum_of_cubes_l259_259498


namespace triangle_ABC_is_right_l259_259137

structure Point (α : Type) :=
  (x : α)
  (y : α)

def dist_sq {α : Type} [Field α] (p1 p2 : Point α) : α :=
  (p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2

def is_right_triangle (α : Type) [Field α] (A B C : Point α) : Prop :=
  let AB_sq := dist_sq A B;
  let BC_sq := dist_sq B C;
  let CA_sq := dist_sq C A in
  (AB_sq = BC_sq + CA_sq ∨ BC_sq = AB_sq + CA_sq ∨ CA_sq = AB_sq + BC_sq)

theorem triangle_ABC_is_right :
  is_right_triangle ℝ ⟨5, -2⟩ ⟨1, 5⟩ ⟨-1, 2⟩ :=
by
  -- We need to show that this triangle is a right triangle by the distances formula
  sorry

end triangle_ABC_is_right_l259_259137


namespace cori_age_l259_259643

theorem cori_age (C A : ℕ) (hA : A = 19) (hEq : C + 5 = (A + 5) / 3) : C = 3 := by
  rw [hA] at hEq
  norm_num at hEq
  linarith

end cori_age_l259_259643


namespace bhanu_house_rent_expenditure_l259_259151

variable (Income house_rent_expenditure petrol_expenditure remaining_income : ℝ)
variable (h1 : petrol_expenditure = (30 / 100) * Income)
variable (h2 : remaining_income = Income - petrol_expenditure)
variable (h3 : house_rent_expenditure = (20 / 100) * remaining_income)
variable (h4 : petrol_expenditure = 300)

theorem bhanu_house_rent_expenditure :
  house_rent_expenditure = 140 :=
by sorry

end bhanu_house_rent_expenditure_l259_259151


namespace shopkeeper_profit_percent_l259_259193

-- Definitions based on conditions
def selling_price (cost_per_kg : ℕ) : ℕ := cost_per_kg
def faulty_grams : ℕ := 900
def true_grams : ℕ := 1000
def actual_cost (cost_per_kg : ℕ) : ℕ := (faulty_grams * cost_per_kg) / true_grams

-- Lean statement of the proof problem
theorem shopkeeper_profit_percent (cost_per_kg : ℕ) (cost_per_kg = 100) :
  let profit := (selling_price cost_per_kg) - (actual_cost cost_per_kg) in
  let profit_percent := (profit * 100) / (actual_cost cost_per_kg) in
  profit_percent = 11.11 :=
sorry

end shopkeeper_profit_percent_l259_259193


namespace tan_theta_pi_div_4_find_phi_l259_259337

variables {θ ϕ : ℝ}

theorem tan_theta_pi_div_4 (h1 : sin θ - 2 * cos θ = 0) (h2 : 0 < θ ∧ θ < π / 2) :
  tan (θ + π / 4) = -3 := 
sorry

theorem find_phi (h1 : sin θ - 2 * cos θ = 0) (h2 : 0 < θ ∧ θ < π / 2) 
  (h3 : 5 * cos (θ - ϕ) = 3 * √5 * cos ϕ) (h4 : 0 < ϕ ∧ ϕ < π / 2) :
  ϕ = π / 4 :=
sorry

end tan_theta_pi_div_4_find_phi_l259_259337


namespace problem_l259_259030

theorem problem (a n : ℕ) (ha : 1 < a) (hn : 1 < n) (h : Nat.prime (a^n + 1)) : (a % 2 = 0) ∧ ∃ k : ℕ, n = 2^k :=
by
  sorry

end problem_l259_259030


namespace range_of_a_l259_259321

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.logBase a x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a x1 - f a x2 > 0) →
  a ∈ set.Ico (1 / 7) (1 / 3) :=
by
  sorry

end range_of_a_l259_259321


namespace determinant_problem_l259_259812

noncomputable def det_matrix (a b c k : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - k^2) - 
  k * (k * (1 + c) - k^2) + 
  k * (k^2 - k * (1 + b))

theorem determinant_problem 
  (a b c p q k : ℝ)
  (h_roots : is_root (λ x : ℝ, x^3 + 2 * p * x + q) a ∧ 
             is_root (λ x : ℝ, x^3 + 2 * p * x + q) b ∧ 
             is_root (λ x : ℝ, x^3 + 2 * p * x + q) c)
  (h_vieta : a + b + c = 0): 
  det_matrix a b c k = -q + 2 * p + 1 + k^2 - k^3 :=
sorry

end determinant_problem_l259_259812


namespace sum_difference_of_consecutive_sets_l259_259907

theorem sum_difference_of_consecutive_sets (a b : ℕ) 
  (h1 : ∃ x, x ∈ {a, a+1, a+2, a+3, a+4, a+5, a+6} ∧ x ∈ {b, b+1, b+2, b+3, b+4, b+5, b+6})
  (h2 : {a, a+1, a+2, a+3, a+4, a+5, a+6} ≠ {b, b+1, b+2, b+3, b+4, b+5, b+6}) :
  (b > a ∧ b = a+1) ∨ (a > b ∧ a = b+1) → 
  abs ((a + (a+1) + (a+2) + (a+3) + (a+4) + (a+5) + (a+6)) - 
      ((b+1) + (b+2) + (b+3) + (b+4) + (b+5) + (b+6) + (b+7))) = 7 := 
by
  sorry

end sum_difference_of_consecutive_sets_l259_259907


namespace values_of_x0_l259_259275

noncomputable def x_seq (x_0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x_0
  | n + 1 => if 3 * (x_seq x_0 n) < 1 then 3 * (x_seq x_0 n)
             else if 3 * (x_seq x_0 n) < 2 then 3 * (x_seq x_0 n) - 1
             else 3 * (x_seq x_0 n) - 2

theorem values_of_x0 (x_0 : ℝ) (h : 0 ≤ x_0 ∧ x_0 < 1) :
  (∃! x_0, x_0 = x_seq x_0 6) → (x_seq x_0 6 = x_0) :=
  sorry

end values_of_x0_l259_259275


namespace distance_from_A_to_l_equation_of_perpendicular_line_l259_259733

structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(a : ℝ)
(b : ℝ)
(c : ℝ)

def distance_to_line (p : Point) (l : Line) : ℝ :=
  |l.a * p.x + l.b * p.y + l.c| / real.sqrt (l.a^2 + l.b^2)

def line_perpendicular_to (p : Point) (l : Line) : Line :=
  let k := -l.a / l.b in
  let perp_slope := l.b / -l.a in
  let y_intercept := p.y - perp_slope * p.x in
  let a := -perp_slope in
  let b := 1 in
  let c := -y_intercept in
  {a, b, c}

def A : Point := {x := 2, y := 1}
def l : Line := {a := 3, b := 4, c := -20}
def l_perp : Line := {a := 4, b := -3, c := -5}

theorem distance_from_A_to_l : distance_to_line A l = 2 :=
by sorry

theorem equation_of_perpendicular_line : line_perpendicular_to A l = l_perp :=
by sorry

end distance_from_A_to_l_equation_of_perpendicular_line_l259_259733


namespace inequality_proof_l259_259069

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l259_259069


namespace triangle_area_find_angle_C_l259_259794

variables {A B C : ℝ} {a b c : ℝ} {area : ℝ}

-- Given conditions
def cos_B_eq : Prop := cos B = 3 / 5
def dot_product_eq : Prop := (a * c * (- (3 / 5))) = -21
def a_eq_7 : Prop := a = 7

-- Prove the area of triangle ABC is 14
theorem triangle_area (h1 : cos_B_eq) (h2 : dot_product_eq) : area = 14 := sorry

-- Prove angle C is π / 4
theorem find_angle_C (h1 : cos_B_eq) (h2 : dot_product_eq) (h3 : a_eq_7) : C = π / 4 := sorry

end triangle_area_find_angle_C_l259_259794


namespace sum_base_5_l259_259266

theorem sum_base_5 (a b c : ℕ) (h1 : a = 213) (h2 : b = 324) (h3 : c = 141) :
  let sum := a + b + c in
  sum = 1333 := by
sorry

end sum_base_5_l259_259266


namespace geo_seq_sum_S5_l259_259861

-- Define the terms and their given values
def a₁ : ℕ := 1

def q : ℕ := 2

-- Define a₄ using the relationship a₄ = a₁ * q^3
def a₄ : ℕ := a₁ * q^3

-- Prove that this sequence condition holds true
theorem geo_seq_sum_S5 : a₁ = 1 → a₄ = 8 → (q = 2) → (S_5 = 31) :=
by
  intros h1 h2 h3
  
  -- condition: a₁ = 1
  have ha1 : a₁ = 1 := h1
  
  -- condition: a₄ = 8
  have ha4 : a₄ = 8 := h2
  
  -- compute q using given condition
  have hq : q = 2 := h3
  
  -- Using formula for sum of first n terms of geometric series
  have S_5 := a₁ * (q^5 - 1) / (q - 1)

  -- Simplifying using known values
  have S_5 := (1 * (2^5 - 1)) / (2 - 1)

  -- Final answer
  have : S_5 = 31 := by
    simp
  
  exact this

end geo_seq_sum_S5_l259_259861


namespace range_of_x_of_sqrt_x_plus_3_l259_259391

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259391


namespace min_distance_statement_l259_259459

noncomputable def min_distance_sum : ℝ :=
  let directrix : ℝ → ℝ × ℝ → ℝ := λ a ⟨x, y⟩ => (x + 1).abs
  let distance_line : (ℝ × ℝ) → ℝ := fun ⟨x, y⟩ => ((x - 2*y + 10).abs / real.sqrt(1^2 + (-2)^2))
  let P : ℝ × ℝ := (2 * 1, y) in -- example point calculation is left for actual full proof development
  directrix (P.1) + distance_line (P)

theorem min_distance_statement : min_distance_sum = (11 * real.sqrt(5) / 5) := 
  sorry

end min_distance_statement_l259_259459


namespace angle_PQR_l259_259810

noncomputable def distance (A B: (ℝ × ℝ × ℝ)) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2)

noncomputable def angle (A B C: (ℝ × ℝ × ℝ)) : ℝ :=
  let AB := distance A B;
  let BC := distance B C;
  let AC := distance A C;
  real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))

theorem angle_PQR : 
  let P : ℝ × ℝ × ℝ := (2, -3, 1);
  let Q : ℝ × ℝ × ℝ := (3, -4, -1);
  let R : ℝ × ℝ × ℝ := (4, -4, 0);
  abs (angle P Q R - 33.557) < 1e-3 :=
by
  sorry

end angle_PQR_l259_259810


namespace sumata_family_miles_driven_l259_259483

def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

theorem sumata_family_miles_driven :
  total_miles_driven 5 50 = 250 :=
by
  sorry

end sumata_family_miles_driven_l259_259483


namespace find_function_satisfaction_l259_259157

theorem find_function_satisfaction :
  ∃ (a b : ℚ) (f : ℚ × ℚ → ℚ), (∀ (x y z : ℚ),
  f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)) ∧ 
  (∀ (x y : ℚ), f (x, y) = a * y^2 + 2 * a * x * y + b * y) := sorry

end find_function_satisfaction_l259_259157


namespace polynomial_coefficient_B_l259_259203

theorem polynomial_coefficient_B :
  ∃ (A C D : ℝ), 
    (∀ z : ℂ, z^6 - 10 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 16 = 0 → z ∈ {1, 2} ∧ multiset.card {1, 2} = 6 ∧ multiset.sum {1, 2} = 10) →
    B = -88 :=
sorry

end polynomial_coefficient_B_l259_259203


namespace third_smallest_four_digit_in_pascals_triangle_l259_259540

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end third_smallest_four_digit_in_pascals_triangle_l259_259540


namespace pizza_slices_l259_259131

theorem pizza_slices (S L : ℕ) (h1 : S + L = 36) (h2 : L = 2 * S) :
  (8 * S + 12 * L) = 384 :=
by
  sorry

end pizza_slices_l259_259131


namespace equation_of_curve_C_sin_alpha_plus_sin_beta_eq_one_l259_259423

-- Defining points and conditions
structure Point := (x : ℝ) (y : ℝ)

def F := Point.mk 1 0
def line := {l : ℝ × ℝ → ℝ // ∀ p, l p = 4 * p.1 + 3 * p.2 + 1}
def circle (F : Point) (r : ℝ) : set Point := { P | (P.x - F.x)^2 + (P.y - F.y)^2 = r^2 }
def parabola := { P : Point | P.y^2 = 4 * P.x ∧ P.x ≠ 0 }

-- Conditions and Questions
axiom C : set Point
axiom tangent_line : line

axiom condition1 : ∀ P ∈ C, P.x > 0
axiom condition2 : ∀ P ∈ C, abs (sqrt ((P.x - 1)^2 + P.y^2) - P.x) = 1
axiom condition3 : ∃ r, circle F r ∧ tangent_line = (4, 3) -- This should be formalized in correct way

-- Define points A, B, P, Q and angles α, β
axiom A : Point
axiom B : Point
axiom P : Point
axiom Q : Point
axiom α β : Real

-- Proving the statements
theorem equation_of_curve_C : C = parabola :=
by
  sorry

theorem sin_alpha_plus_sin_beta_eq_one : sin α + sin β = 1 :=
by
  sorry

end equation_of_curve_C_sin_alpha_plus_sin_beta_eq_one_l259_259423


namespace paddington_more_goats_l259_259838

theorem paddington_more_goats (W P total : ℕ) (hW : W = 140) (hTotal : total = 320) (hTotalGoats : W + P = total) : P - W = 40 :=
by
  sorry

end paddington_more_goats_l259_259838


namespace find_roots_l259_259664

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_l259_259664


namespace find_15th_number_l259_259122

def digits := {1, 3, 6, 8}

def permutations (s : Set ℕ) : Set (List ℕ) := {
  l | l.perm (s.to_list)
}

namespace proofs

theorem find_15th_number :
  ∃ l ∈ permutations digits, 
    (∀ x ∈ permutations digits, (l < x ↔ l = [6, 3, 1, 8])) 
  :=
sorry

end proofs

end find_15th_number_l259_259122


namespace simplify_expr_l259_259083

variable (m : ℝ)

theorem simplify_expr : (1 / (3 * m)) ^ (-3) * (3 * m) ^ 4 = (3 * m) ^ 7 :=
by sorry

end simplify_expr_l259_259083


namespace find_f_5_l259_259873

-- Define the function f satisfying the given conditions
noncomputable def f : ℝ → ℝ :=
sorry

-- Assert the conditions as hypotheses
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x + f y
axiom f_zero : f 0 = 2

-- State the theorem we need to prove
theorem find_f_5 : f 5 = 1 :=
sorry

end find_f_5_l259_259873


namespace uv_divisible_by_3_l259_259074

theorem uv_divisible_by_3
  {u v : ℤ}
  (h : 9 ∣ (u^2 + u * v + v^2)) :
  3 ∣ u ∧ 3 ∣ v :=
sorry

end uv_divisible_by_3_l259_259074


namespace perimeter_of_shaded_region_l259_259374

theorem perimeter_of_shaded_region (O P Q : Point) (r : ℝ) (h1 : dist O P = r) (h2 : dist O Q = r) (h3 : r = 7) (arc_angle : ℝ) (h4 : arc_angle = 270) :
  let circumference := 2 * Real.pi * r in
  let arc_length := (3 / 4) * circumference in
  let perimeter := 2 * r + arc_length in
  perimeter = 14 + (21 * Real.pi) / 2 := 
  sorry

end perimeter_of_shaded_region_l259_259374


namespace partnership_total_annual_gain_l259_259970

noncomputable def total_annual_gain_of_partnership (A_share : ℝ) : ℝ :=
3 * A_share

theorem partnership_total_annual_gain
  (x : ℝ) -- A invests some money at the beginning
  (A_profit : ℝ = 4000) -- A's share of profit is Rs. 4,000
  (B_profit : ℝ = x / 2) -- B invests double amount after 6 months
  (C_profit : ℝ = 3 * x / 4) -- C invests thrice amount after 8 months
  : total_annual_gain_of_partnership A_profit = 12000 := 
by
  sorry

end partnership_total_annual_gain_l259_259970


namespace AD_squared_l259_259429

variable (A B C M D : Type)
variable [MetricSpace A] [HasDistance A] [MetricSpace B] [HasDistance B]
variable [MetricSpace C] [HasDistance C] [MetricSpace M] [HasDistance M]
variable [MetricSpace D] [HasDistance D]

noncomputable def isosceles_triangle (A B C : A) : Prop :=
  dist A B = 3 ∧ dist B C = 3 ∧ dist A C = 4

noncomputable def midpoint (B C M : A) : Prop :=
  dist B M = dist M C ∧ 2 * dist B M = dist B C

noncomputable def equilateral_triangle (A B D : A) : Prop :=
  dist A B = dist B D ∧ dist B D = dist A D ∧ dist A B = dist A D

theorem AD_squared (h_iso : isosceles_triangle A B C)
                   (h_mid : midpoint B C M)
                   (h_eq : equilateral_triangle A B D) :
  dist A D ^ 2 = 31.5625 :=
begin
  sorry
end

end AD_squared_l259_259429


namespace greatest_of_six_consecutive_mixed_numbers_l259_259133

theorem greatest_of_six_consecutive_mixed_numbers (A : ℚ) :
  let B := A + 1
  let C := A + 2
  let D := A + 3
  let E := A + 4
  let F := A + 5
  (A + B + C + D + E + F = 75.5) →
  F = 15 + 1/12 :=
by {
  sorry
}

end greatest_of_six_consecutive_mixed_numbers_l259_259133


namespace mass_percentage_of_Br_in_BaBr2_l259_259533

theorem mass_percentage_of_Br_in_BaBr2 :
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  mass_percentage_Br = 53.80 :=
by
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  sorry

end mass_percentage_of_Br_in_BaBr2_l259_259533


namespace triangle_BC_length_l259_259417

noncomputable def length_BC (A B C : Type) [inner_product_space ℝ C] (AB AC BC : ℝ) (angleA : ℝ) (area : ℝ) :=
  (AB = real.sqrt 2) ∧
  (AC = real.sqrt 6) ∧
  (area = (real.sqrt 3) / 2) ∧
  (angleA < π / 2) ∧
  (BC = real.sqrt 2)

theorem triangle_BC_length (A B C : Type) [inner_product_space ℝ C]
  (AB AC area : ℝ) (angleA : ℝ) (hAB : AB = real.sqrt 2) (hAC : AC = real.sqrt 6)
  (hArea : area = (real.sqrt 3) / 2) (hAngleA : angleA < π / 2) :
  ∃ BC, length_BC A B C AB AC BC angleA area :=
begin
  use real.sqrt 2,
  simp [length_BC, hAB, hAC, hArea, hAngleA],
  sorry,
end

end triangle_BC_length_l259_259417


namespace total_soccer_balls_purchased_l259_259522

theorem total_soccer_balls_purchased : 
  (∃ (x : ℝ), 
    800 / x * 2 = 1560 / (x - 2)) → 
  (800 / x + 1560 / (x - 2) = 30) :=
by
  sorry

end total_soccer_balls_purchased_l259_259522


namespace sum_of_x_coordinates_of_A_l259_259904

theorem sum_of_x_coordinates_of_A :
  ∃ (A : ℝ × ℝ), 
  let B : ℝ × ℝ := (0, 0),
      C : ℝ × ℝ := (447, 0),
      D : ℝ × ℝ := (1360, 760),
      F : ℝ × ℝ := (1378, 778),
      area_ABC : ℝ := 4014,
      area_ADF : ℝ := 14007,
      x_sum : ℝ := 2400 in
  A.1 = x_sum := sorry

end sum_of_x_coordinates_of_A_l259_259904


namespace f_3_2_eq_6_f_nn_eq_fact_l259_259233

noncomputable def f (m n : ℝ) : ℝ := sorry -- define the function f as per given conditions
def A := {p : ℝ × ℝ | true}       -- A is the set of all (m, n) in ℝ 
def B := ℝ

-- Condition 1: f(m, 1) = 1 for all m in ℝ
axiom f_m1 (m : ℝ) : f m 1 = 1

-- Condition 2: If m < n, then f(m, n) = 0
axiom f_mn_0 (m n : ℝ) (h : m < n) : f m n = 0

-- Condition 3: f(m+1, n) = n * (f(m, n) + f(m, n-1))
axiom f_rec (m n : ℝ) : f (m + 1) n = n * (f m n + f m (n - 1))

-- Proof Problem 1: Prove that f(3, 2) = 6
theorem f_3_2_eq_6 : f 3 2 = 6 := by
  sorry

-- Proof Problem 2: Prove that f(n, n) = n!
theorem f_nn_eq_fact (n : ℝ) : f n n = n.factorial := by
  sorry

end f_3_2_eq_6_f_nn_eq_fact_l259_259233


namespace cube_surface_area_percentage_increase_l259_259545

theorem cube_surface_area_percentage_increase (L : ℝ) (hL : L > 0) :
  let L_new := 1.5 * L
      SA_original := 6 * L^2
      SA_new := 6 * (L_new)^2
      percentage_increase := (SA_new - SA_original) / SA_original * 100
  in percentage_increase = 125 :=
by
  let L_new := 1.5 * L
  let SA_original := 6 * L^2
  let SA_new := 6 * (1.5 * L)^2
  let percentage_increase := (SA_new - SA_original) / SA_original * 100
  have h1 : L_new = 1.5 * L := rfl
  have h2 : SA_original = 6 * L^2 := rfl
  have h3 : SA_new = 6 * (1.5 * L)^2 := rfl
  calc
    percentage_increase 
        = (SA_new - SA_original) / SA_original * 100 : rfl
    ... = (6 * (1.5 * L)^2 - 6 * L^2) / (6 * L^2) * 100 : by congr
    ... = (6 * 2.25 * L^2 - 6 * L^2) / (6 * L^2) * 100 : by rw [← mul_assoc, mul_pow]
    ... = (13.5 * L^2 - 6 * L^2) / (6 * L^2) * 100 : by norm_num
    ... = (7.5 * L^2) / (6 * L^2) * 100 : by ring
    ... = 1.25 * 100 : by { field_simp [hL], norm_num }
    ... = 125 : by norm_num

end cube_surface_area_percentage_increase_l259_259545


namespace solve_inequality_l259_259477

theorem solve_inequality (x : ℝ) :
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ (Set.Iio 3 ∪ Set.Ioo 3 5) :=
by
  sorry

end solve_inequality_l259_259477


namespace decimal_to_base5_l259_259231

theorem decimal_to_base5 (n : ℕ) (h : n = 89) : (3 * 5^2 + 2 * 5^1 + 4 * 5^0) = 89 :=
by {
  rw h,
  exact dec_trivial,
}

end decimal_to_base5_l259_259231


namespace gcd_f_10_11_l259_259813

def f (x : ℤ) : ℤ := x^3 - x^2 + 2 * x + 1007

theorem gcd_f_10_11 : Int.gcd (f 10) (f 11) = 1 := by
  -- compute f(10) = 1927 and f(11) = 2239
  have h1 : f 10 = 1927 := by
    calc
      f 10 = 10^3 - 10^2 + 2 * 10 + 1007 : by rfl
      ... = 1000 - 100 + 20 + 1007 : by norm_num
      ... = 1927 : by norm_num
  have h2 : f 11 = 2239 := by
    calc
      f 11 = 11^3 - 11^2 + 2 * 11 + 1007 : by rfl
      ... = 1331 - 121 + 22 + 1007 : by norm_num
      ... = 2239 : by norm_num

  -- compute gcd(1927, 2239) and show it equals 1
  show Int.gcd 1927 2239 = 1 from by
    calc
      Int.gcd 1927 2239 = Int.gcd 1927 (2239 - 1927) : by refi
      ... = Int.gcd 1927 312 : by norm_num
      ... = Int.gcd 312 (1927 % 312) : by exact_mod_cast (by norm_num : 1927 = 6 * 312 + 85)
      ... = Int.gcd 312 83 : by norm_num
      ... = Int.gcd 83 (312 % 83) : by exact_mod_cast (by norm_num : 312 = 3 * 83 + 63)
      ... = Int.gcd 83 63 : by norm_num
      ... = Int.gcd 63 (83 % 63) : by exact_mod_cast (by norm_num : 83 = 1 * 63 + 20)
      ... = Int.gcd 63 20 : by norm_num
      ... = Int.gcd 20 (63 % 20) : by exact_mod_cast (by norm_num : 63 = 3 * 20 + 3)
      ... = Int.gcd 20 3 : by norm_num
      ... = Int.gcd 3 (20 % 3) : by exact_mod_cast (by norm_num : 20 = 6 * 3 + 2)
      ... = Int.gcd 3 2 : by norm_num
      ... = Int.gcd 2 (3 % 2) : by norm_num
      ... = Int.gcd 2 1 : by norm_num
      ... = Int.gcd 1 0 : by norm_num
      ... = 1 : by norm_num

  sorry -- inserting a sorry here to satisfy lean's requirement for a proof.

end gcd_f_10_11_l259_259813


namespace area_of_AKD_is_two_P_l259_259416

-- Definitions based on the conditions identified in Step A
variables (A B C D K : Point)
variables (p AC BD : ℝ) -- p being the area P of trapezoid ABCD
variable (angle_AKD_45 : ∠(A, K, D) = 45)

noncomputable def trapezoid_with_perpendicular_diagonals (AC_perp_CD : AC ⊥ CD) 
  (BD_perp_AB : BD ⊥ AB) 
  (intersect_formed_triangle : line_through A B ∩ line_through C D = K) := true

-- Problem statement to prove
theorem area_of_AKD_is_two_P (h : trapezoid_with_perpendicular_diagonals (A B C D K AC_perp_CD BD_perp_AB intersect_formed_triangle)) : 
  area (triangle A K D) = 2 * p :=
sorry

end area_of_AKD_is_two_P_l259_259416


namespace simplify_expression_solve_fractional_eq_l259_259159

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l259_259159


namespace minimum_tangent_length_l259_259504

theorem minimum_tangent_length 
  (t : ℝ)
  (x := (λ t : ℝ, (√2) / 2 * t))
  (y := (λ t : ℝ, (√2) / 2 * t + 4 * √2))
  (θ : ℝ)
  (ρ := 2 * real.cos (θ + π / 4)) :
  ∃ t θ, ∀ x y, 
    (x = (√2) / 2 * t) → 
    (y = (√2) / 2 * t + 4 * √2) →
    ρ^2 = x^2 + y^2 →
    x = √2 / 2 * t →
    y = √2 / 2 * t + 4 * √2 →
    (√((5)^2 - 1^2) = 2 * √6) :=
by sorry

end minimum_tangent_length_l259_259504


namespace linear_eq_substitution_l259_259764

theorem linear_eq_substitution (x y : ℝ) (h1 : 3 * x - 4 * y = 2) (h2 : x = 2 * y - 1) :
  3 * (2 * y - 1) - 4 * y = 2 :=
by
  sorry

end linear_eq_substitution_l259_259764


namespace emerson_distance_l259_259249

theorem emerson_distance (d1 : ℕ) : 
  (d1 + 15 + 18 = 39) → d1 = 6 := 
by
  intro h
  have h1 : 33 = 39 - d1 := sorry -- Steps to manipulate equation to find d1
  sorry

end emerson_distance_l259_259249


namespace bishop_attacks_king_l259_259940

-- Define the conditions and helpers
def total_ways_to_place : ℕ := 2000 * 2000 * (2000 * 2000 - 1)

def positive_diagonals_ways : ℕ :=
  2 * (2000 * 2001 * 3998 / 6 + 1999 * 2000 * 3997 / 6)

def probability (m n : ℕ) : ℚ := m / n

-- Define the main theorem statement
theorem bishop_attacks_king :
  ∃ m n : ℕ, nat.coprime m n ∧ probability m n = positive_diagonals_ways / total_ways_to_place ∧ m = 1333 ∧ n = 2001000 :=
begin
  sorry
end

end bishop_attacks_king_l259_259940


namespace solution_l259_259858

-- Define the equation
def equation (x : ℝ) := x^2 + 4*x + 3 + (x + 3)*(x + 5) = 0

-- State that x = -3 is a solution to the equation
theorem solution : equation (-3) :=
by
  unfold equation
  simp
  sorry

end solution_l259_259858


namespace union_of_M_and_N_is_correct_l259_259330

def M : Set ℤ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 ≤ n ∧ n ≤ 3 }

theorem union_of_M_and_N_is_correct : M ∪ N = { -2, -1, 0, 1, 2, 3 } := 
by
  sorry

end union_of_M_and_N_is_correct_l259_259330


namespace surface_area_ratio_l259_259087

-- Define the surface area of a sphere given its radius
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

-- Define the radii of spheres A and B
def r_A := 40 -- radius of sphere A
def r_B := 10 -- radius of sphere B

-- Define the surface areas based on the radii
def SA := surface_area r_A
def SB := surface_area r_B

-- State the theorem that the ratio of the surface areas is 16:1
theorem surface_area_ratio : SA / SB = 16 :=
by
  -- This is a placeholder for the proof
  sorry

end surface_area_ratio_l259_259087


namespace bus_departure_l259_259518

theorem bus_departure (current_people : ℕ) (min_people : ℕ) (required_people : ℕ) 
  (h1 : current_people = 9) (h2 : min_people = 16) : required_people = 7 :=
by 
  sorry

end bus_departure_l259_259518


namespace find_initial_men_l259_259833

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end find_initial_men_l259_259833


namespace min_ω_l259_259325

noncomputable def problem :=
  let φ := Real.angle.pi / 6
  let ω := Real.pi / 2
  let f (x : ℝ) := 2 * Real.sin (ω * x + φ)
  let g (x : ℝ) := 2 * Real.sin (ω * x - ω)
  {φ: Real.angle | φ = Real.angle.pi / 6} :=
  {φ: Real.angle | 2 * Real.sin φ = 1} ∧ 
  ∃ x: ℝ, f 0 = 1 ∧ |φ| < π/2
  
theorem min_ω (ω > 0) (φ : Real.angle) (hx : ∃ x: ℝ, f (x + 2) - f x = 4) : ω = Real.pi / 2 :=
_exists x: ℝ, f (x + 2) - f x = 4 ∧ f x:= exists ... sorry

end min_ω_l259_259325


namespace sum_n_for_24_div_odd_l259_259913

theorem sum_n_for_24_div_odd (S : set ℤ) (h : ∀ n ∈ S, 24 % (2 * n - 1) = 0) : S.sum id = 3 :=
sorry

end sum_n_for_24_div_odd_l259_259913


namespace bounded_difference_l259_259563

-- Define constants
def C : ℝ := 1 / (2 * Real.log 10)

-- Define the partial sum function S_n
def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, (Real.log k / k)

-- Define the difference δ_n as given in the problem
def δ_n (n : ℕ) : ℝ := S_n n - C * (Real.log n)^2

-- Theorem statement
theorem bounded_difference :
  ∃ C, ∀ n : ℕ, -0.25 ≤ δ_n n ∧ δ_n n ≤ 0.25 :=
begin
  use (1 / (2 * Real.log 10)), -- This is the C value we identified
  sorry -- Skipping the proof as instructed
end

end bounded_difference_l259_259563


namespace determine_x2_y2_z2_l259_259049

variables (x y z : ℤ)
variables (A B C H M N : Type) -- Using Types here for triangle vertices and points

axiom H_foot_of_altitude : True
axiom M_midpoint : True
axiom N_midpoint : True

axiom HN_squared : ∀ (A B C H M N : Type), 
  16 * (dist H N)^2 = x * (dist B C)^2 + y * (dist C A)^2 + z * (dist A B)^2

theorem determine_x2_y2_z2 : x^2 + y^2 + z^2 = 9 :=
by
  sorry

end determine_x2_y2_z2_l259_259049


namespace ram_task_completion_days_l259_259468

theorem ram_task_completion_days (R : ℕ) (h1 : ∀ k : ℕ, k = R / 2) (h2 : 1 / R + 2 / R = 1 / 12) : R = 36 :=
sorry

end ram_task_completion_days_l259_259468


namespace nhai_initial_men_l259_259831

theorem nhai_initial_men (M : ℕ) (W : ℕ) :
  let totalWork := M * 50 * 8 in
  let partialWork := M * 25 * 8 in
  let remainingWork := (M + 60) * 25 * 10 in
  partialWork = totalWork / 3 →
  remainingWork = (2 * totalWork) / 3 →
  M = 100 :=
by
  intros h1 h2
  have eq1 : totalWork = M * 50 * 8 := rfl
  sorry -- Proof is omitted

end nhai_initial_men_l259_259831


namespace product_of_roots_eq_l259_259821

noncomputable def Q (φ : ℝ) : Polynomial ℂ := 
  Polynomial.of_real (1 : ℝ) + Polynomial.C (Complex.I * (sin (2 * φ))) * (Polynomial.X - Polynomial.C (Complex.of_real (cos φ))) * (Polynomial.X - Polynomial.C (Complex.of_real (cos φ))) 
  + Polynomial.C ((Complex.I * (sin (2 * φ))) ^ 2)

theorem product_of_roots_eq (φ : ℝ) (hφ : 0 < φ ∧ φ < π / 6) : 
  (Q φ).leadingCoeff = 1 ∧ 
  Q φ = (Polynomial.X - (Polynomial.C (Complex.of_real (cos φ)) + Polynomial.C (Complex.I * (sin (2 * φ))))) *
  (Polynomial.X - (Polynomial.C (Complex.of_real (cos φ)) - Polynomial.C (Complex.I * (sin (2 * φ))))) *
  (Polynomial.X - (Polynomial.C (Complex.of_real (cos φ)) + Polynomial.C (Complex.I * (sin (2 * φ))))) *
  (Polynomial.X - (Polynomial.C (Complex.of_real (cos φ)) - Polynomial.C (Complex.I * (sin (2 * φ))))) ->
  Polynomial.eval 0 (Q φ) = 1 + 3 * (sin φ) ^ 2 * (cos φ) ^ 2 :=
by
  sorry

end product_of_roots_eq_l259_259821


namespace intersection_points_abscissa_value_segment_length_range_l259_259437

variables (a b c t : ℝ)
variables (A1 B1 : ℝ)

namespace MathProof

-- (1) Proves that the graphs of these two functions must have two different intersection points
theorem intersection_points (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax ^ 2 + bx + c = ax + b :=
sorry

-- (2) Proves that if t is an odd number, the value of t must be ±1
theorem abscissa_value (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) (odd_t : (t % 2 = 1) ∨ (t % 2 = -1)) : t = 1 ∨ t = -1 :=
sorry

-- (3) Finds the range of the length of segment A1B1
theorem segment_length_range (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) 
: (3/2 : ℝ) < A1B1 ∧ A1B1 < sqrt 3 :=
sorry

end MathProof

end intersection_points_abscissa_value_segment_length_range_l259_259437


namespace tan_alpha_given_cos_l259_259276

theorem tan_alpha_given_cos (α : ℝ) (h1 : cos α = -3/5) (h2 : α ∈ Ioo π (3 * π / 2)) : tan α = 4/3 :=
sorry

end tan_alpha_given_cos_l259_259276


namespace intersection_A_B_intersection_CA_B_intersection_CA_CB_l259_259446

-- Set definitions
def A := {x : ℝ | -5 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | x < -2 ∨ x > 4}
def C_A := {x : ℝ | x < -5 ∨ x > 3}  -- Complement of A
def C_B := {x : ℝ | -2 ≤ x ∧ x ≤ 4}  -- Complement of B

-- Lean statements proving the intersections
theorem intersection_A_B : {x : ℝ | -5 ≤ x ∧ x ≤ 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | -5 ≤ x ∧ x < -2} :=
by sorry

theorem intersection_CA_B : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | x < -5 ∨ x > 4} :=
by sorry

theorem intersection_CA_CB : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 4} = {x : ℝ | 3 < x ∧ x ≤ 4} :=
by sorry

end intersection_A_B_intersection_CA_B_intersection_CA_CB_l259_259446


namespace johns_piano_total_cost_l259_259017

theorem johns_piano_total_cost : 
  let piano_cost := 500
  let original_lessons_cost := 20 * 40
  let discount := (25 / 100) * original_lessons_cost
  let discounted_lessons_cost := original_lessons_cost - discount
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  total_cost = 1275 := 
by
  let piano_cost := 500
  let original_lessons_cost := 800
  let discount := 200
  let discounted_lessons_cost := 600
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  -- Proof skipped
  sorry

end johns_piano_total_cost_l259_259017


namespace zongzi_problem_l259_259901

def zongzi_prices : Prop :=
  ∀ (x y : ℕ), -- x: price of red bean zongzi, y: price of meat zongzi
  10 * x + 12 * y = 136 → -- total cost for the first customer
  y = 2 * x →
  x = 4 ∧ y = 8 -- prices found

def discounted_zongzi_prices : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  20 * a + 30 * b = 270 → -- cost for Xiaohuan's mother
  30 * a + 20 * b = 230 → -- cost for Xiaole's mother
  a = 3 ∧ b = 7 -- discounted prices found

def zongzi_packages (m : ℕ) : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  a = 3 → b = 7 →
  (80 - 4 * m) * (m * a + (40 - m) * b) + (4 * m + 8) * ((40 - m) * a + m * b) = 17280 →
  m ≤ 20 / 2 → -- quantity constraint
  m = 10 -- final m value

-- Statement to prove all together
theorem zongzi_problem :
  zongzi_prices ∧ discounted_zongzi_prices ∧ ∃ (m : ℕ), zongzi_packages m :=
by sorry

end zongzi_problem_l259_259901


namespace ball_hits_ground_at_time_l259_259872

-- Definitions for the conditions
def height (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 60

-- Statement to prove
theorem ball_hits_ground_at_time : ∃ t : ℝ, height t = 0 ∧ t = 3 :=
by
  use 3
  split
  -- Here we'd provide the actual proof steps to show that height 3 = 0.
  sorry

end ball_hits_ground_at_time_l259_259872


namespace four_digit_number_count_l259_259272

theorem four_digit_number_count : 
  {n : ℕ | ∃ (digits : Fin 4 → ℕ), (∀ i, digits i = 1 ∨ digits i = 2) ∧
                (∃ i, digits i = 1) ∧ (∃ i, digits i = 2)}.card = 14 :=
by
  sorry

end four_digit_number_count_l259_259272


namespace find_m_l259_259230

noncomputable def sequence_cubed_sum_eq (y m : ℤ) : Prop :=
  (finset.range (m + 1)).sum (λ k, (y + 3 * k)^3) = -3375

theorem find_m (y : ℤ) (m : ℤ) (h : m > 5) (hc : sequence_cubed_sum_eq y m) : m = 6 :=
  sorry

end find_m_l259_259230


namespace saline_solution_concentration_l259_259758

theorem saline_solution_concentration
    (a b : ℝ)  -- let a and b be the weights in kilograms
    (h₁ : a * (x/100) = (a + b) * 0.20)  -- concentration after adding water
    (h₂ : (a + b) * (1 - 0.20) + b * (1) = [(a + b) + b] * 0.30)  -- concentration after adding salt
    (h₃ : a = 6 * b)  -- derived from simplifying the conditions
  : x = 23 + 1 / 3 := by
  sorry

end saline_solution_concentration_l259_259758


namespace complement_intersection_example_l259_259823

namespace SetTheory

open Set

theorem complement_intersection_example :
  let U := {-3, -2, -1, 0, 1, 2, 3}
  let A := {-3, -2, 2, 3}
  let B := {-3, 0, 1, 2}
  ((U \ A) ∩ B = {0, 1}) :=
by
  simp [U, A, B]
  sorry

end SetTheory

end complement_intersection_example_l259_259823


namespace arithmetic_sequence_a_100_l259_259935

theorem arithmetic_sequence_a_100 :
  (∃ (a : ℕ → ℤ) d, 
      (∑ i in finset.range 9, (a 1 + i * d)) = 27 ∧ 
      (a 1 + 9 * d) = 8) →
  (∃ a, a 100 = 98) :=
begin
  sorry
end

end arithmetic_sequence_a_100_l259_259935


namespace f_xy_le_fxy_l259_259681

variable (a : ℝ) (f : ℝ → ℝ)
variable (h_a_pos : 0 < a)
variable (h_f_cont : ContinuousOn f set.Ioi 0)
variable (h_fa : f a = 1)
variable (h_ineq : ∀ x y, 0 < x → 0 < y → f(x) * f(y) + f(a / x) * f(a / y) ≤ 2 * f(x * y))

theorem f_xy_le_fxy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f(x) * f(y) ≤ f(x * y) :=
by
  sorry

end f_xy_le_fxy_l259_259681


namespace circle_area_A_O_C_l259_259529

noncomputable def area_of_circle_through_A_O_C : Real :=
  let AB := 6
  let AC := 6
  let BC := 4
  let p := (AB + AC + BC) / 2
  let Δ := Real.sqrt (p * (p - AB) * (p - AC) * (p - BC))
  let r := Δ / p
  let R := Real.sqrt ((AB / 2)^2 + r^2)
  let area := Real.pi * R^2
  area

theorem circle_area_A_O_C :
  let AB := 6
  let AC := 6
  let BC := 4
  let p := (AB + AC + BC) / 2
  let Δ := Real.sqrt (p * (p - AB) * (p - AC) * (p - BC))
  let r := Δ / p
  let R := Real.sqrt ((AB / 2)^2 + r^2)
  let area := Real.pi * R^2
  area = 11 * Real.pi := by
  let AB := 6
  let AC := 6
  let BC := 4
  let p := (AB + AC + BC) / 2
  let Δ := Real.sqrt (p * (p - AB) * (p - AC) * (p - BC))
  let r := Δ / p
  let R := Real.sqrt ((AB / 2)^2 + r^2)
  let area := Real.pi * R^2
  have h1 : area = 11 * Real.pi := sorry
  exact h1

end circle_area_A_O_C_l259_259529


namespace value_of_x2_minus_y2_l259_259349

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l259_259349


namespace xiaoli_estimate_l259_259785

variable (x y z : ℝ) (hx : x > y) (hy : y > 0) (hz : 0 < z)

theorem xiaoli_estimate (hx : x > y) (hy : y > 0) (hz : 0 < z) : 
  (x + 2 * z) - (y - 2 * z) > x - y := 
by 
  calc
    (x + 2 * z) - (y - 2 * z) = x - y + 4 * z : by ring
    ... > x - y : by linarith

end xiaoli_estimate_l259_259785


namespace part_time_proof_l259_259364

variable (T U: ℤ) 
variable (prob: ℚ)

def part_time_employees : Prop := 
    let UP : ℤ := (125 * U) / 1000
    let Neither := (prob * T)
    let Uninsured_or_Part_time_or_Both := T - Neither
    let P := Uninsured_or_Part_time_or_Both - (U - UP)
    P = 54

theorem part_time_proof (hT: T = 335) (hU: U = 104) (hprob: prob = 5671641791044776 / 10000000000000000) :
part_time_employees T U prob :=
by
  unfold part_time_employees
  rw [hT, hU, hprob]
  have hUP: (125 * 104) / 1000 = 13 := sorry
  have hP: 335 - (prob * 335 : ℚ) = 145 := sorry
  have final: 145 - (104 - 13) = 54 := sorry
  exact final

end part_time_proof_l259_259364


namespace kids_go_to_camp_l259_259427

theorem kids_go_to_camp (total_kids : ℕ) (kids_stay_home : ℕ) (h1 : total_kids = 898051) (h2 : kids_stay_home = 268627) : total_kids - kids_stay_home = 629424 :=
by
  sorry

end kids_go_to_camp_l259_259427


namespace B_higher_rank_than_A_l259_259520

inductive Person
| knight
| liar
| normal

open Person

variables (A B C : Person)

-- Definitions of ranks where higher rank is represented as a higher value.
def rank : Person → ℕ
| knight := 3
| liar := 1
| normal := 2

-- Statements made by A and B:
def A_statement := rank B > rank C
def B_statement := rank C > rank A

-- Given conditions:
axiom A_normal : A = normal
axiom B_normal : B = normal

-- The proof goal:
theorem B_higher_rank_than_A : rank B > rank A :=
by {
  sorry -- The actual proof would go here.
}

end B_higher_rank_than_A_l259_259520


namespace arithmetic_sequence_ratio_l259_259358

noncomputable def a_n : ℕ → ℝ := sorry -- Define the arithmetic sequence a_n
noncomputable def b_n : ℕ → ℝ := sorry -- Define the arithmetic sequence b_n
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n i -- Sum of first n terms of a_n
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n i -- Sum of first n terms of b_n

theorem arithmetic_sequence_ratio (n : ℕ) (h : n > 0) :
  (S_n n) / (T_n n) = (2 * n + 1) / (n + 2) →
  (a_n 7) / (b_n 7) = 9 / 5 :=
by
  sorry

end arithmetic_sequence_ratio_l259_259358


namespace parallel_and_equidistant_line_l259_259335

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 6 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 3 = 0

-- Define the desired property: a line parallel to line1 and line2, and equidistant from both
theorem parallel_and_equidistant_line :
  ∃ b : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + b = 0) ∧
  (|-6 - b| / Real.sqrt (9 + 4) = |-3/2 - b| / Real.sqrt (9 + 4)) →
  (12 * x + 8 * y - 15 = 0) :=
by
  sorry

end parallel_and_equidistant_line_l259_259335


namespace probability_of_earning_exactly_2400_in_3_spins_l259_259837

-- Definitions corresponding to the given conditions
def spinnerAmounts := {Bankrupt, 1500, 200, 6000, 700}

def probabilityOfLandingOn (x : ℝ) : ℝ := 
  if x ∈ spinnerAmounts then 1 / 5 else 0

def eventEarning2400In3Spins (s1 s2 s3 : ℝ) : Prop :=
  s1 + s2 + s3 = 2400 ∧ s1 ∈ spinnerAmounts ∧ s2 ∈ spinnerAmounts ∧ s3 ∈ spinnerAmounts

-- The statement to be proved
theorem probability_of_earning_exactly_2400_in_3_spins :
  (∑ (s1 s2 s3 : ℝ) in spinnerAmounts, 
    if eventEarning2400In3Spins s1 s2 s3 
    then (probabilityOfLandingOn s1 * probabilityOfLandingOn s2 * probabilityOfLandingOn s3) else 0) = 6 / 125 := 
  sorry

end probability_of_earning_exactly_2400_in_3_spins_l259_259837


namespace min_value_f_min_value_f_achieved_l259_259501

variable {x : ℝ}

def f (x : ℝ) : ℝ := abs (cos x) + abs (cos (2 * x))

theorem min_value_f : ∀ x : ℝ, f x ≥ sqrt 2 / 2 := by sorry

theorem min_value_f_achieved : ∃ x : ℝ, f x = sqrt 2 / 2 := by sorry

end min_value_f_min_value_f_achieved_l259_259501


namespace four_friends_total_fish_l259_259058

-- Define the number of fish each friend has based on the conditions
def micah_fish : ℕ := 7
def kenneth_fish : ℕ := 3 * micah_fish
def matthias_fish : ℕ := kenneth_fish - 15
def total_three_boys_fish : ℕ := micah_fish + kenneth_fish + matthias_fish
def gabrielle_fish : ℕ := 2 * total_three_boys_fish
def total_fish : ℕ := micah_fish + kenneth_fish + matthias_fish + gabrielle_fish

-- The proof goal
theorem four_friends_total_fish : total_fish = 102 :=
by
  -- We assume the proof steps are correct and leave the proof part as sorry
  sorry

end four_friends_total_fish_l259_259058


namespace seashells_remaining_l259_259080

def initial_seashells : ℕ := 35
def given_seashells : ℕ := 18

theorem seashells_remaining : initial_seashells - given_seashells = 17 := by
  sorry

end seashells_remaining_l259_259080


namespace prove_ellipse_results_l259_259690

def ellipse_equation (a b : ℝ) (a_gt_b : a > b) : Prop :=
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def ellipse_incircle_radius (a b : ℝ) (a_gt_b : a > b) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let radius := b / 3
  in radius = ((2 * dist F1 F2 * b) / (2 * (a + dist F1 F2)))

def ellipse_line_intersection (a b : ℝ) (a_gt_b : a > b) (F2 : ℝ × ℝ) (RS : ℝ) : Prop :=
  let l := ∀ x : ℝ, ∃ y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} ∧ p = F2
  in RS = 3

def point_T_exists (a b : ℝ) (a_gt_b : a > b) (TS TR : ℝ) : Prop :=
  ∃ T : ℝ × ℝ, T.2 = 0 ∧ ∃ (l : ℝ → ℝ), 
  (∀ R S : ℝ × ℝ, (R.2, S.2) = l ∧ R.1 = (1 - (S.1) + T.1)) ∧ TS = -TR / 2

theorem prove_ellipse_results :
  ∀ (a b : ℝ) (a_gt_b : a > b)
    (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (RS TS TR : ℝ),
    ellipse_equation a b a_gt_b →
    ellipse_incircle_radius a b a_gt_b F1 F2 P →
    ellipse_line_intersection a b a_gt_b F2 RS →
    point_T_exists a b a_gt_b TS TR →
    ∃ c, c = (4 : ℝ) ∧ P = (4, 0) :=
by sorry

end prove_ellipse_results_l259_259690


namespace part1_part2_part3_l259_259243

open Classical

-- Define the relationship y = kx + b with given points
variables (x y : ℤ) (k b : ℤ)

theorem part1 (h1: y = 9 * k + b) (h2: y = 11 * k + b) : y = -5 * x + 150 := 
sorry

-- Define the profit function and conditions given (8 ≤ x ≤ 15)
noncomputable def profit (x y : ℤ) := (x - 8) * y

variables {profit : ℤ → ℤ}
axiom h3 : ∀ x y ∈ (set.range profit), 8 ≤ x ∧ x ≤ 15

theorem part2 (x : ℤ) (hx : 8 ≤ x ∧ x ≤ 15) (hprofit : profit x (-5 * x + 150) = 425) : x = 13 :=
sorry

-- Define maximum profit calculations
noncomputable def max_profit (x : ℤ) := -5 * (x - 19)^2 + 605

theorem part3 (x : ℤ) (hx : 8 ≤ x ∧ x ≤ 15) (y := max_profit x) : 
  (hx : x = 15 ∧ y = 525 ∧ ∀ x', 8 ≤ x' ∧ x' ≤ 15 → (-5 * (x' - 19)^2 + 605 ≤ 525)) :=
sorry

end part1_part2_part3_l259_259243


namespace limit_f_at_pi_l259_259219

def f (x : ℝ) : ℝ := (sin (x ^ 2 / Real.pi)) / (2 ^ (Real.sqrt (sin x + 1)) - 2)

theorem limit_f_at_pi : filter.tendsto f (nhds_real π) (nhds (2 / Real.log 2)) := by
  sorry

end limit_f_at_pi_l259_259219


namespace determine_a_value_l259_259716

theorem determine_a_value (a : ℝ) :
  (∀ y₁ y₂ : ℝ, ∃ m₁ m₂ : ℝ, (m₁, y₁) = (a, -2) ∧ (m₂, y₂) = (3, -4) ∧ (m₁ = m₂)) → a = 3 :=
by
  sorry

end determine_a_value_l259_259716


namespace sum_of_positive_ks_l259_259096

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end sum_of_positive_ks_l259_259096


namespace additional_machines_l259_259270

theorem additional_machines (r : ℝ) (M : ℝ) : 
  (5 * r * 20 = 1) ∧ (M * r * 10 = 1) → (M - 5 = 95) :=
by
  sorry

end additional_machines_l259_259270


namespace incorrect_statement_b_l259_259297

-- Defining the equation of the circle
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

-- Defining the point not on the circle
def is_not_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 25

-- The proposition to be proved
theorem incorrect_statement_b : ¬ ∀ p : ℝ × ℝ, is_not_on_circle p.1 p.2 → ¬ is_on_circle p.1 p.2 :=
by
  -- Here we should provide the proof, but this is not required based on the instructions.
  sorry

end incorrect_statement_b_l259_259297


namespace derivative_at_2_l259_259869

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_at_2 : deriv f 2 = 15 := by
  sorry

end derivative_at_2_l259_259869


namespace train_B_speed_l259_259128

noncomputable def train_speed_B (V_A : ℕ) (T_A : ℕ) (T_B : ℕ) : ℕ :=
  V_A * T_A / T_B

theorem train_B_speed
  (V_A : ℕ := 60)
  (T_A : ℕ := 9)
  (T_B : ℕ := 4) :
  train_speed_B V_A T_A T_B = 135 := 
by
  sorry

end train_B_speed_l259_259128


namespace figure_D_has_smallest_unshaded_area_l259_259998

def rectangle_area (length : ℝ) (width : ℝ) : ℝ := length * width
def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

def unshaded_area_D : ℝ := 
  rectangle_area 3 4 - circle_area 2

def unshaded_area_E : ℝ := 
  rectangle_area 4 5 - 4 * circle_area 1 / 4

def unshaded_area_F : ℝ := 
  rectangle_area 5 3 - 2 * circle_area 1.5

theorem figure_D_has_smallest_unshaded_area :
  (unshaded_area_D < unshaded_area_E) ∧ (unshaded_area_D < unshaded_area_F) :=
by 
  sorry

end figure_D_has_smallest_unshaded_area_l259_259998


namespace pepperoni_slices_left_l259_259844

theorem pepperoni_slices_left :
  ∀ (total_friends : ℕ) (total_slices : ℕ) (cheese_left : ℕ),
    (total_friends = 4) →
    (total_slices = 16) →
    (cheese_left = 7) →
    (∃ p_slices_left : ℕ, p_slices_left = 4) :=
by
  intros total_friends total_slices cheese_left h_friends h_slices h_cheese
  sorry

end pepperoni_slices_left_l259_259844


namespace range_of_m_l259_259324

theorem range_of_m (m n : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |m * x| - |x - n|) 
  (h_n_pos : 0 < n) (h_n_m : n < 1 + m) 
  (h_integer_sol : ∃ xs : Finset ℤ, xs.card = 3 ∧ ∀ x ∈ xs, f x < 0) : 
  1 < m ∧ m < 3 := 
sorry

end range_of_m_l259_259324


namespace focus_to_asymptote_distance_l259_259729

theorem focus_to_asymptote_distance (x y : ℝ) (h : x^2 - y^2 = 2) : 
  let a := sqrt(2)
  let b := sqrt(2)
  let c := sqrt(a^2 + b^2)
  let foci := (2, 0)
  let asymptote := λ x y, x + y = 0
  distance_from_focus_to_asymptote := sqrt(2) := 
by
  sorry

end focus_to_asymptote_distance_l259_259729


namespace Kath_payment_l259_259774

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l259_259774


namespace cost_of_paving_is_correct_l259_259926

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def rate_per_square_meter : ℝ := 950
def area_of_room : ℝ := length_of_room * width_of_room
def cost_of_paving : ℝ := area_of_room * rate_per_square_meter

theorem cost_of_paving_is_correct : cost_of_paving = 20900 := 
by
  sorry

end cost_of_paving_is_correct_l259_259926


namespace proof_problem1_proof_problem2_l259_259190

def sequence (n : ℕ) : ℕ → ℝ
| 0 => 2
| 1 => 6
| n + 2 => 2 * sequence (n + 1) - sequence n + 2

def proof_part1 : Prop :=
    ∀ n : ℕ, sequence (n + 1) - sequence n = 4 + 2 * (n - 1)

def proof_part2 : Prop :=
  (finset.range 2016).sum (λ n, 1 / sequence (n + 1)) = 2016 / 2017

theorem proof_problem1 : proof_part1 := sorry
theorem proof_problem2 : proof_part2 := sorry

end proof_problem1_proof_problem2_l259_259190


namespace range_of_x_in_sqrt_x_plus_3_l259_259405

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259405


namespace scientific_notation_correct_l259_259415

def distance_moon_km : ℕ := 384000

def scientific_notation (n : ℕ) : ℝ := 3.84 * 10^5

theorem scientific_notation_correct : scientific_notation distance_moon_km = 3.84 * 10^5 := by
  sorry

end scientific_notation_correct_l259_259415


namespace total_books_l259_259521

/-- Define Tim’s and Sam’s number of books. -/
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52

/-- Prove that together they have 96 books. -/
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l259_259521


namespace ratio_of_radii_l259_259768

-- Define the conditions of the problem
variables (r R : ℝ) 

-- The condition that the gray area equals four times the small white circle's area
def condition (r R : ℝ) : Prop := (π * R^2 - π * r^2) = 4 * (π * r^2)

-- The theorem statement: the ratio of r to R
theorem ratio_of_radii (r R : ℝ) (h : condition r R) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l259_259768


namespace existence_of_N_equilateral_triangles_l259_259210

variables (A B C D M N : Type*)
variables [ConvexQuadrilateral A B C D] [InternalPoint A B C D M]
variables (AMB_isosceles : is_isosceles_triangle A M B 120)
variables (CMD_isosceles : is_isosceles_triangle C M D 120)

theorem existence_of_N_equilateral_triangles :
  ∃ N, is_equilateral_triangle B N C ∧ is_equilateral_triangle D N A :=
sorry

end existence_of_N_equilateral_triangles_l259_259210


namespace min_value_expression_l259_259277

theorem min_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ x : ℝ, x = (1 / (a - 1)) + (1 / (2 * b)) ∧ x ≥ (3 / 2 + Real.sqrt 2)) :=
sorry

end min_value_expression_l259_259277


namespace dogs_weight_fraction_l259_259063

-- Define the conditions as constants or definitions
def number_of_people_canoe_can_carry : ℕ := 6
def fraction_people_with_dog : ℝ := 2 / 3
def weight_per_person : ℝ := 140
def total_weight_with_dog : ℝ := 595

-- Calculate the values based on conditions
def number_of_people_with_dog : ℕ :=
  (number_of_people_canoe_can_carry * fraction_people_with_dog).toInt

def total_weight_people : ℝ :=
  number_of_people_with_dog * weight_per_person

def weight_of_dog : ℝ :=
  total_weight_with_dog - total_weight_people

-- Define the fraction of the dog's weight compared to a person's weight
def fraction_dogs_weight : ℝ :=
  weight_of_dog / weight_per_person

theorem dogs_weight_fraction :
  fraction_dogs_weight = 1 / 4 :=
by
  -- The proof will be here
  sorry

end dogs_weight_fraction_l259_259063


namespace triangle_angle_B_l259_259797

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h : a / b = 3 / Real.sqrt 7) (h2 : b / c = Real.sqrt 7 / 2) : B = Real.pi / 3 :=
by
  sorry

end triangle_angle_B_l259_259797


namespace area_inequality_l259_259077

variable {a b c : ℝ} (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a)

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) : ℝ :=
  let p := semiperimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  (2 * (area a b c h))^3 < (a * b * c)^2 := sorry

end area_inequality_l259_259077


namespace sesame_seed_weight_in_scientific_notation_l259_259612

theorem sesame_seed_weight_in_scientific_notation :
  0.00000201 = 2.01 * (10:ℝ) ^ (-6) :=
by
  sorry

end sesame_seed_weight_in_scientific_notation_l259_259612


namespace sum_of_n_an_l259_259430

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Definitions based on conditions
axiom a1_nonzero : a 1 ≠ 0
axiom S_def : ∀ n : ℕ, S (n + 1) = (Finset.range n).sum (λ k, a (k + 1))
axiom a_S_relation : ∀ n : ℕ, 2 * a (n + 1) - a 1 = S 1 * S (n + 1)

-- Theorem to prove
theorem sum_of_n_an : ∀ n : ℕ, T (n + 1) = (n * 2^(n + 1) + 1) := by
 sorry

end sum_of_n_an_l259_259430


namespace angle_AFE_is_130_degrees_l259_259212

-- Given conditions as definitions in Lean
variables (A B C D E F : Type)
variable [∀ x : Type, linear_order x]

def is_square (ABCD : Type) [∀ x : Type, linear_order x] := 
  sorry  -- Definition of a square omitted for brevity

def opposite_half_plane (E : Type) (CD : Type) (A : Type) : Prop := 
  sorry  -- Definition of opposite half plane omitted for brevity

def angle_CDE_100 (CD E : Type) : Prop := 
  sorry  -- Definition that angle CDE is 100 degrees omitted for brevity

def points_on_AD (F : Type) (AD : Type) (DE DF : Type → ℝ) : Prop := 
  DE F = 2 * DF F  

-- Proposition to be proven
theorem angle_AFE_is_130_degrees 
  (sqr : is_square ABCD) 
  (pos_plane : opposite_half_plane E CD A) 
  (angle_cond : angle_CDE_100 CD E) 
  (seg_cond : points_on_AD F AD (λ _, DE) (λ _, DF)) : 
  ∠ A F E = 130 :=
sorry

end angle_AFE_is_130_degrees_l259_259212


namespace number_of_true_propositions_l259_259305

variables {a : Type*} {α β : Type*}

-- Conditions as definitions
def is_line (a : Type*) : Prop := sorry
def is_plane (α : Type*) : Prop := sorry
def perp (a : Type*) (α : Type*) : Prop := sorry
def parallel (α β : Type*) : Prop := sorry

-- Given conditions
variable (line_a : is_line a)
variable (plane_alpha : is_plane α)
variable (plane_beta : is_plane β)
variable (H1 : perp a α)
variable (H2 : perp a β)
variable (H3 : parallel α β)

-- Theorem to prove the problem statement
theorem number_of_true_propositions :
  (H1 ∧ H2 → H3) ∧
  (H1 ∧ H3 → H2) ∧
  (H2 ∧ H3 → H1) :=
by {
  sorry
}

end number_of_true_propositions_l259_259305


namespace prove_lesser_fraction_l259_259116

noncomputable def lesser_fraction (x y : ℚ) : Prop :=
  x + y = 8/9 ∧ x * y = 1/8 ∧ min x y = 7/40

theorem prove_lesser_fraction :
  ∃ x y : ℚ, lesser_fraction x y :=
sorry

end prove_lesser_fraction_l259_259116


namespace arrangement_count_l259_259746

theorem arrangement_count :
  ∑ k in Finset.range 6, (Nat.choose 5 k)^3 = 
  (the number of 15-letter arrangements of 5 A's, 5 B's, and 5 C's 
   with no A's in the first 5 letters, 
   no B's in the next 5 letters, 
   and no C's in the last 5 letters) :=
sorry

end arrangement_count_l259_259746


namespace find_brick_width_l259_259267

def SurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

theorem find_brick_width :
  ∃ width : ℝ, SurfaceArea 10 width 3 = 164 ∧ width = 4 :=
by
  sorry

end find_brick_width_l259_259267


namespace generatrix_length_l259_259487

section cone_problem

variables {r h l : ℝ} (V : ℝ)
-- Given conditions
def sector_area_twice_base_area (r l : ℝ) : Prop := (π * r * l = 2 * π * r^2)

def cone_volume (r h : ℝ) (V : ℝ) : Prop := (1 / 3) * π * r^2 * h = V

-- Theorem to prove
theorem generatrix_length (A_twice : sector_area_twice_base_area r l) (vol : cone_volume r (sqrt(3) * r) 9 * sqrt 3 * π) : l = 6 :=
begin
  sorry
end

end cone_problem

end generatrix_length_l259_259487


namespace number_of_blue_stamps_l259_259056

theorem number_of_blue_stamps (
    red_stamps : ℕ := 20
) (
    yellow_stamps : ℕ := 7
) (
    price_per_red_stamp : ℝ := 1.1
) (
    price_per_blue_stamp : ℝ := 0.8
) (
    total_earnings : ℝ := 100
) (
    price_per_yellow_stamp : ℝ := 2
) : red_stamps = 20 ∧ yellow_stamps = 7 ∧ price_per_red_stamp = 1.1 ∧ price_per_blue_stamp = 0.8 ∧ total_earnings = 100 ∧ price_per_yellow_stamp = 2 → ∃ (blue_stamps : ℕ), blue_stamps = 80 :=
by
  sorry

end number_of_blue_stamps_l259_259056


namespace infinite_series_integer_l259_259020

theorem infinite_series_integer (n : ℕ) (a : ℕ → ℤ)
  (h1 : n > 1) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → n ∣ (a i - i)) :
  ∃ (b : ℕ → ℤ), (∀ k, b k ∈ {a 1, a 2, ..., a n}) ∧ (∑ k, b k / n^k) ∈ ℤ :=
sorry

end infinite_series_integer_l259_259020


namespace center_symmetric_not_axis_symmetric_l259_259207

-- Define the shapes and their properties
structure Shape where
  name : String
  is_center_symmetric : Bool
  is_axis_symmetric : Bool

-- Define specific shapes
def square : Shape := { name := "Square", is_center_symmetric := true, is_axis_symmetric := true }
def equilateral_triangle : Shape := { name := "Equilateral Triangle", is_center_symmetric := false, is_axis_symmetric := true }
def circle : Shape := { name := "Circle", is_center_symmetric := true, is_axis_symmetric := true }
def parallelogram : Shape := { name := "Parallelogram", is_center_symmetric := true, is_axis_symmetric := false }

-- Theorem statement
theorem center_symmetric_not_axis_symmetric (shapes : List Shape) (h1 : ∀ s ∈ shapes, s.name = "Square" → s.is_center_symmetric = true ∧ s.is_axis_symmetric = true)
                                             (h2 : ∀ s ∈ shapes, s.name = "Equilateral Triangle" → s.is_center_symmetric = false)
                                             (h3 : ∀ s ∈ shapes, s.name = "Circle" → s.is_center_symmetric = true ∧ s.is_axis_symmetric = true)
                                             (h4 : ∀ s ∈ shapes, s.name = "Parallelogram" → s.is_center_symmetric = true ∧ s.is_axis_symmetric = false)
                                             : ∃ s ∈ shapes, s.name = "Parallelogram" ∧ s.is_center_symmetric = true ∧ s.is_axis_symmetric = false := 
begin
  sorry
end

-- Define the list of shapes
def shapes := [square, equilateral_triangle, circle, parallelogram]

-- Instantiate the theorem with the given shapes list
example : ∃ s ∈ shapes, s.name = "Parallelogram" ∧ s.is_center_symmetric = true ∧ s.is_axis_symmetric = false :=
  center_symmetric_not_axis_symmetric shapes
    (by intros s hs hname; cases s; split; refl)
    (by intros s hs hname; cases s; refl)
    (by intros s hs hname; cases s; split; refl)
    (by intros s hs hname; cases s; split; refl)

end center_symmetric_not_axis_symmetric_l259_259207


namespace range_y_C_l259_259300

theorem range_y_C (A : ℝ × ℝ) (hA : A = (0, 2))
  (B C : ℝ × ℝ)
  (hB : ∃ y1: ℝ, B = (y1^2 - 4, y1))
  (hC : ∃ y: ℝ, C = (y^2 - 4, y))
  (hPerpendicular : let kAB := (B.2 - A.2) / (B.1 - A.1) in
                    let kBC := (C.2 - B.2) / (C.1 - B.1) in
                    kAB * kBC = -1) :
  C.2 ≤ 0 ∨ C.2 >= 4 :=
by
  sorry

end range_y_C_l259_259300


namespace trig_eq_has_root_l259_259240

open Real

/-- 
    For the equation 1/sin(x) + 1/cos(x) = λ to have a root in the interval (0, π/2),
    we need to prove that λ ≥ 2√2. 
-/
theorem trig_eq_has_root (λ : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ (1 / sin x + 1 / cos x = λ)) ↔ λ ≥ 2 * sqrt 2 :=
by
  sorry

end trig_eq_has_root_l259_259240


namespace diminished_val_by_6_l259_259506

theorem diminished_val_by_6 (n : ℕ) : 
  (16 ∣ (n - 6)) ∧ (18 ∣ (n - 6)) ∧ (21 ∣ (n - 6)) ∧ (28 ∣ (n - 6)) → n = 1014 :=
begin
  sorry
end

end diminished_val_by_6_l259_259506


namespace range_of_x_l259_259101

def y_function (x : ℝ) : ℝ := x

def y_translated (x : ℝ) : ℝ := x + 2

theorem range_of_x {x : ℝ} (h : y_translated x > 0) : x > -2 := 
by {
  sorry
}

end range_of_x_l259_259101


namespace collinear_points_l259_259760

theorem collinear_points (x y : ℝ) (h_collinear : ∃ k : ℝ, (x + 1, y, 3) = (2 * k, 4 * k, 6 * k)) : x - y = -2 := 
by 
  sorry

end collinear_points_l259_259760


namespace additional_books_acquired_l259_259965

def original_stock : ℝ := 40.0
def shelves_used : ℕ := 15
def books_per_shelf : ℝ := 4.0

theorem additional_books_acquired :
  (shelves_used * books_per_shelf) - original_stock = 20.0 :=
by
  sorry

end additional_books_acquired_l259_259965


namespace smallest_value_of_x_l259_259537

theorem smallest_value_of_x : ∃ x, (2 * x^2 + 30 * x - 84 = x * (x + 15)) ∧ (∀ y, (2 * y^2 + 30 * y - 84 = y * (y + 15)) → x ≤ y) ∧ x = -28 := by
  sorry

end smallest_value_of_x_l259_259537


namespace david_money_l259_259557

theorem david_money (S : ℝ) (h_initial : 1500 - S = S - 500) : 1500 - S = 500 :=
by
  sorry

end david_money_l259_259557


namespace polynomial_roots_l259_259663

open Polynomial

theorem polynomial_roots :
  (roots (C 1 * X^4 + C (-3) * X^3 + C 3 * X^2 + C (-1) * X + C (-6))).map (λ x, x.re) = 
    {1 - sqrt 3, 1 + sqrt 3, (1 - sqrt 13) / 2, (1 + sqrt 13) / 2} :=
by sorry

end polynomial_roots_l259_259663


namespace simplify_expression_l259_259914

theorem simplify_expression :
  (sqrt 1 + sqrt (1 + 3) + sqrt (1 + 3 + 5) + sqrt (1 + 3 + 5 + 7) + sqrt (1 + 3 + 5 + 7 + 9) - 3 = 12) :=
by
  sorry

end simplify_expression_l259_259914


namespace remainder_mod7_l259_259670

theorem remainder_mod7 (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := 
by
  sorry

end remainder_mod7_l259_259670


namespace slide_vs_slip_l259_259589

noncomputable def ladder : Type := sorry

def slide_distance (ladder : ladder) : ℝ := sorry
def slip_distance (ladder : ladder) : ℝ := sorry
def is_right_triangle (ladder : ladder) : Prop := sorry

theorem slide_vs_slip (l : ladder) (h : is_right_triangle l) : slip_distance l > slide_distance l :=
sorry

end slide_vs_slip_l259_259589


namespace max_value_g_l259_259497

def g : ℕ → ℤ
| n => if n < 5 then n + 10 else g (n - 3)

theorem max_value_g : ∃ x, (∀ n : ℕ, g n ≤ x) ∧ (∃ y, g y = x) ∧ x = 14 := 
by
  sorry

end max_value_g_l259_259497


namespace polynomial_simplification_l259_259082

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 :=
by
  sorry

end polynomial_simplification_l259_259082


namespace polygon_sides_l259_259593

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end polygon_sides_l259_259593


namespace polynomial_roots_sum_to_ten_and_B_is_neg88_l259_259201

theorem polynomial_roots_sum_to_ten_and_B_is_neg88
  (roots : Fin 6 → ℕ)
  (h_sum : ∑ i, roots i = 10)
  (h_all_pos : ∀ i, 0 < roots i)
  (h_poly : ∑ i in Finset.range 7, (-1)^i * (roots.prod (λ r, r^(6 - i))) = λ A B C D, (0^6 - 10 * 0^5 + A * 0^4 + B * 0^3 + C * 0^2 + D * 0 + 16)) :
  B = -88 := 
sorry

end polynomial_roots_sum_to_ten_and_B_is_neg88_l259_259201


namespace simplify_expression_l259_259223

variable {a b c : ℤ}

theorem simplify_expression (a b c : ℤ) : 3 * a - (4 * a - 6 * b - 3 * c) - 5 * (c - b) = -a + 11 * b - 2 * c :=
by
  sorry

end simplify_expression_l259_259223


namespace inequality_proof_l259_259698

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l259_259698


namespace triangle_ABC_is_right_l259_259138

structure Point (α : Type) :=
  (x : α)
  (y : α)

def dist_sq {α : Type} [Field α] (p1 p2 : Point α) : α :=
  (p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2

def is_right_triangle (α : Type) [Field α] (A B C : Point α) : Prop :=
  let AB_sq := dist_sq A B;
  let BC_sq := dist_sq B C;
  let CA_sq := dist_sq C A in
  (AB_sq = BC_sq + CA_sq ∨ BC_sq = AB_sq + CA_sq ∨ CA_sq = AB_sq + BC_sq)

theorem triangle_ABC_is_right :
  is_right_triangle ℝ ⟨5, -2⟩ ⟨1, 5⟩ ⟨-1, 2⟩ :=
by
  -- We need to show that this triangle is a right triangle by the distances formula
  sorry

end triangle_ABC_is_right_l259_259138


namespace spinner_probability_l259_259604

-- Definitions of the conditions
variables (is_isosceles_triangle : Type) -- The type representing the isosceles triangle
variables (regions : ℕ) -- Total number of regions
variables (shaded_regions : ℕ) -- Number of shaded regions
variables (equal_base_angles : Prop) -- The property that the base angles are equal
variables (altitudes_drawn : Prop) -- The property that altitudes are drawn subdividing the triangle

-- Assumptions based on the given conditions of the problem
axiom isosceles_triangle_def : is_isosceles_triangle
axiom equal_base_angles_def : equal_base_angles
axiom altitudes_drawn_def : altitudes_drawn
axiom regions_def : regions = 7
axiom shaded_regions_def : shaded_regions = 3

-- Proof that the probability of landing in a shaded region is 3/7
theorem spinner_probability : (shaded_regions : ℚ) / regions = 3 / 7 :=
by {
  -- Definitions of our variables
  rw [shaded_regions_def, regions_def],
  norm_num,
}

end spinner_probability_l259_259604


namespace geometric_sequence_17th_term_l259_259099

variable {α : Type*} [Field α]

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

theorem geometric_sequence_17th_term :
  ∀ (a r : α),
    a * r ^ 4 = 9 →  -- Fifth term condition
    a * r ^ 12 = 1152 →  -- Thirteenth term condition
    a * r ^ 16 = 36864 :=  -- Seventeenth term conclusion
by
  intros a r h5 h13
  sorry

end geometric_sequence_17th_term_l259_259099


namespace probability_abs_diff_2_l259_259181

-- Define what it means for an event to have a probability in this context.
noncomputable def probability_of_event (A : set (ℕ × ℕ)) (total : ℕ) : ℚ :=
  (A.to_finset.card : ℚ) / total

-- Define the sample space for rolling a six-sided die twice.
def sample_space : set (ℕ × ℕ) :=
  { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }

-- Define the event where the absolute difference between the numbers is exactly 2.
def event_abs_diff_2 : set (ℕ × ℕ) :=
  { (x, y) | (x - y).nat_abs = 2 }

-- The total number of possible outcomes when rolling a six-sided die twice.
def total_outcomes : ℕ :=
  6 * 6

-- Theorem statement
theorem probability_abs_diff_2 : 
  probability_of_event event_abs_diff_2 total_outcomes = 2 / 9 :=
by sorry

end probability_abs_diff_2_l259_259181


namespace integral_solution_l259_259218

open Real

noncomputable def integral_example : ℝ → ℝ := 
  λ x, x ^ 3 + 6 * x ^ 2 - 10 * x + 52 / (x - 2) / (x + 2) ^ 3

theorem integral_solution (C : ℝ) :
  ∫ integral_example = λ x, (ln (abs(x - 2)) + 11 / (x + 2) ^ 2 + C) :=
sorry

end integral_solution_l259_259218


namespace total_wheels_in_garage_l259_259517

theorem total_wheels_in_garage (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) 
                               (num_cars : ℕ) (wheels_per_car : ℕ) :
  num_bicycles = 9 → wheels_per_bicycle = 2 → 
  num_cars = 16 → wheels_per_car = 4 → 
  (num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car) = 82 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_wheels_in_garage_l259_259517


namespace cannot_be_adjacent_l259_259937

/-- Prove that in a list of 100 consecutive natural numbers sorted by the sum of their digits and by numeric value for identical sums, the numbers 2010 and 2011 cannot be adjacent -/
theorem cannot_be_adjacent (numbers : list ℕ) (h_cons : numbers = list.range' 2000 100) :
  let sorted_numbers := numbers.sort_by (λ n, (n.digits 10).sum) in
  ¬ (2010 ∈ sorted_numbers ∧ 2011 ∈ sorted_numbers ∧
     list.index_of 2010 sorted_numbers + 1 = list.index_of 2011 sorted_numbers) :=
by {
  sorry
}

end cannot_be_adjacent_l259_259937


namespace continuous_function_solution_l259_259256

theorem continuous_function_solution (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x, -1 < x ∧ x < 1 → (1 - x^2) * f ((2 * x) / (1 + x^2)) = (1 + x^2)^2 * f x)
  (h2 : continuous_on f (set.Ioo (-1 : ℝ) 1)) :
  ∀ x, -1 < x ∧ x < 1 → f x = c / (1 - x^2) := 
by
  sorry

end continuous_function_solution_l259_259256


namespace value_of_a_l259_259355

theorem value_of_a (a : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) (⊤ : ℝ), ∃ f : ℝ → ℝ, f(x) = a*x^2 + (a-3)*x + 1 ∧ ∀ x ∈ Icc (-1 : ℝ) (⊤ : ℝ), ∀ y ∈ Icc (-1 : ℝ) (⊤ : ℝ), x ≤ y → f(x) ≥ f(y)) → a = -3 :=
by sorry

end value_of_a_l259_259355


namespace range_of_independent_variable_l259_259395

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259395


namespace one_ton_equals_2000_pounds_l259_259062

-- Define the number of ounces in a pound
def ounces_per_pound : ℤ := 16

-- Define the weight of one packet in ounces
def weight_per_packet_ounces : ℤ := (16 * ounces_per_pound) + 4

-- Define the total number of packets
def total_packets : ℤ := 1840

-- Define the capacity of the gunny bag in tons
def gunny_bag_capacity_tons : ℤ := 13

-- Define a function to convert tons to pounds
def tons_to_pounds (tons : ℤ) (pounds_per_ton : ℤ) : ℤ := tons * pounds_per_ton

-- Axiom to declare the conversion between tons and the total capacity in pounds
axiom one_ton_in_pounds (pounds_per_ton : ℤ) : 
    total_packets * weight_per_packet_ounces = 
    tons_to_pounds gunny_bag_capacity_tons pounds_per_ton * ounces_per_pound

-- Proposition to prove one ton is equal to 2000 pounds
theorem one_ton_equals_2000_pounds : ∃ pounds_per_ton : ℤ, one_ton_in_pounds pounds_per_ton = 2000 * ounces_per_pound :=
by 
  sorry

end one_ton_equals_2000_pounds_l259_259062


namespace isabella_hair_length_l259_259422

theorem isabella_hair_length (h : ℕ) (g : h + 4 = 22) : h = 18 := by
  sorry

end isabella_hair_length_l259_259422


namespace conic_section_propositions_l259_259205

theorem conic_section_propositions :
  (¬ (∀ (x y : ℝ), (x - 1) ^ 2 / 4 + y ^ 2 / 3 = 1 ↔ (x - 1)^2 / 4 + y ^ 2 / 3 = 2) ∧
  ∃ f : ℝ → ℝ, f x = sqrt (2 * x) ∧ (∀ (P Q : ℝ), P = (3, 6) ∧ Q = (3, 6) → dist P Q + dist Q (0, Q.2) = 6) ∧
  ∀ λ > 0, (¬ ∃ (O₁ O₂ : ℝ), dist O₁ O₂ = λ ∧ circle_eq (O₁, O₂)) ∧
  ∃ h : ℝ → ℝ, h x = sqrt ((x - 1) ^ 2 + (2 - 4) ^ 2) ↔ (|2 * x - 4|) ∧ (¬ (hyperbola_eq h)) ∧
  ∃ l : ℝ → ℝ, l (3, 4) = 0 ∧ ∀ (A B C : ℝ), C = (1, 1) →
  A = (-3 + A - 3, -4 + B + (-4)) ∧ ( A ≠ (0, 0)) →
  C midpoint ( A B)) →
 { 2, 5 } := sorry

end conic_section_propositions_l259_259205


namespace find_theta_value_l259_259129

variables {A B C D E F G : Type*}
variables (angle_ACB angle_FEG angle_DCE angle_DEC angle_CDE : ℝ)
variables (right_triangle_sum : ∀ (x y : ℝ), x + y = 90)
variables (straight_line_sum : ∀ (x y z : ℝ), x + y + z = 180)
variables (triangle_sum : ∀ (x y z : ℝ), x + y + z = 180)

noncomputable def theta_value : ℝ :=
    let θ := 4 + 7 in θ

theorem find_theta_value (h1 : angle_ACB = 90 - 10)
                        (h2 : angle_FEG = 90 - 26)
                        (h3 : angle_DCE = 180 - angle_ACB - 14)
                        (h4 : angle_DEC = 180 - angle_FEG - 33)
                        (h5 : angle_CDE = 180 - angle_DCE - angle_DEC)
                        (h6 : angle_CDE = 11) :
    theta_value = 11 :=
begin
  sorry
end

end find_theta_value_l259_259129


namespace function_range_defined_l259_259792

theorem function_range_defined (x : ℝ) : (x ≥ -3 ∧ x ≠ 1) → ∃ y, y = (sqrt (x + 3)) / (x - 1) :=
by {
  sorry
}

end function_range_defined_l259_259792


namespace simplify_log_expression_l259_259084

theorem simplify_log_expression (a b c d x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) :
  log (a ^ 2 / b) + log (b / c ^ 2) + log (c / d) - log (a ^ 2 * y / (d ^ 3 * x)) = log (d ^ 2 * x / y) := 
by
  sorry

end simplify_log_expression_l259_259084


namespace weight_of_each_bag_is_correct_l259_259950

noncomputable def weightOfEachBag
    (days1 : ℕ := 60)
    (consumption1 : ℕ := 2)
    (days2 : ℕ := 305)
    (consumption2 : ℕ := 4)
    (ouncesPerPound : ℕ := 16)
    (numberOfBags : ℕ := 17) : ℝ :=
        let totalOunces := (days1 * consumption1) + (days2 * consumption2)
        let totalPounds := totalOunces / ouncesPerPound
        totalPounds / numberOfBags

theorem weight_of_each_bag_is_correct :
  weightOfEachBag = 4.93 :=
by
  sorry

end weight_of_each_bag_is_correct_l259_259950


namespace part1_part2_l259_259678

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * a * x^2 + 2 * a * x

theorem part1 (x : ℝ) : 
  ∃ a, a = 1 ∧ (f a x) → ∀ x, -1 < x ∧ x < 2 :=
sorry

theorem part2 (a : ℝ) : 
  (∀ x, - x^2 + a * x + 2 * a ≤ 0) ↔ (-8 ≤ a ∧ a ≤ 0) :=
sorry

end part1_part2_l259_259678


namespace perfect_square_divisor_probability_of_15_factorial_l259_259956

theorem perfect_square_divisor_probability_of_15_factorial :
  let n := 15!
  let prime_factors := (2^11) * (3^6) * (5^3) * (7^2) * (11^1) * (13^1)
  let total_divisors := 4032
  let perfect_square_divisors := 56
  let gcd := Nat.gcd 7 504
  let prob_m := 7 / gcd
  let prob_n := 504 / gcd
  gcd = 1 -> prob_m + prob_n = 511 := sorry

end perfect_square_divisor_probability_of_15_factorial_l259_259956


namespace value_of_5_inch_cube_is_977_l259_259586

def volume (side : ℝ) : ℝ := side ^ 3

def value_of_cube (side : ℝ) (value_4in : ℝ) : ℝ :=
  let volume_4in := volume 4
  let volume_side := volume side
  value_4in * (volume_side / volume_4in)

theorem value_of_5_inch_cube_is_977 :
  value_of_cube 5 500 = 977 :=
sorry

end value_of_5_inch_cube_is_977_l259_259586


namespace poolesville_students_after_transfer_l259_259010

/-- The number of students in the Poolesville magnet after the transfers is 170 given the specified conditions. -/
theorem poolesville_students_after_transfer
  (P : ℕ) -- original number of Poolesville students
  (B_before B_after : ℕ) -- number of Blair students before and after the transfer
  (percent_transfer_poolesville : ℝ) -- percentage of Poolesville students transferred to Blair
  (percent_transfer_blair : ℝ) -- percentage of Blair students transferred to Poolesville
  (students_to_blair : ℕ) -- number of Poolesville students transferred to Blair
  (students_to_poolesville : ℕ) -- number of Blair students transferred to Poolesville
  (P_after : ℕ) -- number of students in Poolesville after the transfer
  (h1 : percent_transfer_poolesville = 0.40)
  (h2 : percent_transfer_blair = 0.05)
  (h3 : B_before = 400)
  (h4 : B_after = 480)
  (h5 : students_to_poolesville = 0.05 * 400)
  (h6 : B_after = B_before + students_to_blair - students_to_poolesville)
  (h7 : students_to_blair = 0.40 * P)
  (h8 : students_to_blair + students_to_poolesville = 80 + 20)
  (h9 : P = 250)
  (h10 : students_to_blair = 100)
  (h11 : P_after = P - students_to_blair + students_to_poolesville)
  (h12 : 480 = 400 + 80) 
  (h13 : P_after = 170) 
  : P_after = 170 := 
sorry

end poolesville_students_after_transfer_l259_259010


namespace max_students_pencils_l259_259882

theorem max_students_pencils (P : ℕ) :
  (∃ m : ℕ, P = 91 * m) :=
begin
  sorry
end

end max_students_pencils_l259_259882


namespace find_f1_find_f3_range_of_x_l259_259682

-- Define f as described
axiom f : ℝ → ℝ
axiom f_domain : ∀ (x : ℝ), x > 0 → ∃ (y : ℝ), f y = f x

-- Given conditions
axiom condition1 : ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0
axiom condition2 : ∀ (x y : ℝ), 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom condition3 : f (1 / 3) = 1

-- Prove f(1) = 0
theorem find_f1 : f 1 = 0 := by sorry

-- Prove f(3) = -1
theorem find_f3 : f 3 = -1 := by sorry

-- Given inequality condition
axiom condition4 : ∀ x : ℝ, 0 < x → f x < 2 + f (2 - x)

-- Prove range of x for given inequality
theorem range_of_x : ∀ x, x > 1 / 5 ∧ x < 2 ↔ f x < 2 + f (2 - x) := by sorry

end find_f1_find_f3_range_of_x_l259_259682


namespace Kath_payment_l259_259772

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l259_259772


namespace fourth_column_rectangles_l259_259979

theorem fourth_column_rectangles (grid : array (array ℕ) 7 7) 
  (h_condition : ∀ (i j : ℕ) (h1 : i < 7) (h2 : j < 7), ∃ n : ℕ, grid[i][j] = n ∧ n = (size of the rectangle containing (i, j))) : 
  ∃ (R : set (set (ℕ × ℕ))), 
    (∀ r ∈ R, (∃ a b, r = { (i, j) | a ≤ i < a + x ∧ b ≤ j < b + y } where (x * y = grid[a][b]))) ∧
    ( ∑ r in R, finset.card (r ∩ {(4, j) | j < 7}) = 7) ∧
    (finset.card {r ∈ R | ∃ j < 7, (4, j) ∈ r} = 4) :=
sorry

end fourth_column_rectangles_l259_259979


namespace min_sum_XB_XA_YC_YD_l259_259019

theorem min_sum_XB_XA_YC_YD {ABCD : ConvexQuadrilateral} 
  (O : Point) (X : Point) (Y : Point) 
  (AO : ℝ) (BO : ℝ) (CO : ℝ) (DO : ℝ)
  (h_O_AC : Incident O AC)
  (h_O_BD : Incident O BD)
  (h_XY_collinear : Collinear X O Y)
  (h_X_AB : OnSegment X AB)
  (h_Y_CD : OnSegment Y CD)
  (AO_eq : AO = 3) 
  (BO_eq : BO = 4) 
  (CO_eq : CO = 5) 
  (DO_eq : DO = 6) :
  10 * 2 + 3 + 10 = 23 := 
by 
  sorry

end min_sum_XB_XA_YC_YD_l259_259019


namespace largest_angle_in_right_isosceles_triangle_l259_259126

theorem largest_angle_in_right_isosceles_triangle (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (angle_Z : ℝ) 
  (h1 : angle_X = 45) 
  (h2 : angle_Y = 90)
  (h3 : angle_Y + angle_X + angle_Z = 180) 
  (h4 : angle_X = angle_Z) : angle_Y = 90 := by 
  sorry

end largest_angle_in_right_isosceles_triangle_l259_259126


namespace derivative_even_function_l259_259093

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + Real.exp(1)

theorem derivative_even_function :
  (fun x => (deriv f) x) = (fun x => (deriv f) (-x)) := sorry

end derivative_even_function_l259_259093


namespace distinct_real_roots_arith_progression_l259_259885

theorem distinct_real_roots_arith_progression (a b c d : ℝ) (h₁ : b ≠ 0) (h₂ : d ≠ 0) (h₃ : a = b - d) (h₄ : c = b + d) : 
  (a * x^2 + 2 * real.sqrt 2 * b * x + c = 0) → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (is_root (a * x^2 + 2 * real.sqrt 2 * b * x + c) x1) ∧ (is_root (a * x^2 + 2 * real.sqrt 2 * b * x + c) x2) := 
sorry

end distinct_real_roots_arith_progression_l259_259885


namespace alex_buys_rice_amount_l259_259852

theorem alex_buys_rice_amount
  (r o : ℝ)
  (h1 : r + o = 30)
  (h2 : 1.10 * r + 0.50 * o = 23.50) :
  r = 14.2 :=
begin
  sorry
end

end alex_buys_rice_amount_l259_259852


namespace time_after_1456_minutes_l259_259492

noncomputable def hours_in_minutes := 1456 / 60
noncomputable def minutes_remainder := 1456 % 60

def current_time : Nat := 6 * 60  -- 6:00 a.m. in minutes
def added_time : Nat := current_time + 1456

def six_sixteen_am : Nat := (6 * 60) + 16  -- 6:16 a.m. in minutes the next day

theorem time_after_1456_minutes : added_time % (24 * 60) = six_sixteen_am :=
by
  sorry

end time_after_1456_minutes_l259_259492


namespace smallest_n_for_g_greater_than_21_l259_259040

noncomputable def g (n : ℕ) : ℕ :=
  Nat.find (λ k, k.factorial % n = 0)

theorem smallest_n_for_g_greater_than_21 {r : ℕ} (hr : r ≥ 23) :
  let n := 21 * r in g n > 21 :=
by
  let n := 21 * r
  have hr23 : r ≥ 23 := hr
  have hdiv : 23 ∣ n := by
    sorry
  have gn : g n ≥ 23 := by
    sorry
  exact Nat.lt_of_le_of_lt 21 gn

end smallest_n_for_g_greater_than_21_l259_259040


namespace right_triangle_AB_CA_BC_l259_259140

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end right_triangle_AB_CA_BC_l259_259140


namespace function_domain_l259_259406

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259406


namespace monkey_slip_distance_l259_259182

theorem monkey_slip_distance
  (height : ℕ)
  (climb_per_hour : ℕ)
  (hours : ℕ)
  (s : ℕ)
  (total_hours : ℕ)
  (final_climb : ℕ)
  (reach_top : height = hours * (climb_per_hour - s) + final_climb)
  (total_hours_constraint : total_hours = 17)
  (climb_per_hour_constraint : climb_per_hour = 3)
  (height_constraint : height = 19)
  (final_climb_constraint : final_climb = 3)
  (hours_constraint : hours = 16) :
  s = 2 := sorry

end monkey_slip_distance_l259_259182


namespace verify_segment_lengths_l259_259112

noncomputable def segment_lengths_proof : Prop :=
  let a := 2
  let b := 3
  let alpha := Real.arccos (5 / 16)
  let segment1 := 4 / 3
  let segment2 := 2 / 3
  let segment3 := 2
  let segment4 := 1
  ∀ (s1 s2 s3 s4 : ℝ), 
    (s1 = segment1 ∧ s2 = segment2 ∧ s3 = segment3 ∧ s4 = segment4) ↔
    -- Parallelogram sides and angle constraints
    (s1 + s2 = a ∧ s3 + s4 = b ∧ 
     -- Mutually perpendicular lines divide into equal areas
     (s1 * s3 * Real.sin alpha / 2 = s2 * s4 * Real.sin alpha / 2) )

-- Placeholder for proof
theorem verify_segment_lengths : segment_lengths_proof :=
  sorry

end verify_segment_lengths_l259_259112


namespace parallelogram_XY_squared_l259_259372

/-- In parallelogram ABCD, AB = 24, BC = 13, CD = 24, and DA = 13, with angle A = 60 degrees.
    Points X and Y are the midpoints of AB and CD, respectively.
    The task is to compute XY^2, the square of the length of XY, and show it equals 169.05. -/
theorem parallelogram_XY_squared
  (A B C D X Y : ℝ × ℝ)
  (h_ABCD : parallelogram A B C D)
  (h_AB : dist A B = 24)
  (h_BC : dist B C = 13)
  (h_CD : dist C D = 24)
  (h_DA : dist D A = 13)
  (h_angle_A : ∠ D A B = 60)
  (h_X : X = midpoint A B)
  (h_Y : Y = midpoint C D) :
  dist X Y ^ 2 = 169.05 := 
sorry

end parallelogram_XY_squared_l259_259372


namespace total_profit_correct_l259_259350

noncomputable def total_profit (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : ℝ := Tp

theorem total_profit_correct (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : 
  total_profit Cp Cq Cr Tp h1 h2 hR = 4650 :=
sorry

end total_profit_correct_l259_259350


namespace num_values_multiple_of_seven_l259_259818

-- Define d_1 and d_2 based on the problem statement
def d_1 (a : ℤ) : ℤ := a^3 + 3^a + a * 3^((a + 1) / 3)
def d_2 (a : ℤ) : ℤ := a^3 + 3^a - a * 3^((a + 1) / 3)

-- Define the condition of divisibility by 7
def is_multiple_of_seven (n : ℤ) : Prop := 7 ∣ n

-- Define the range of values
def in_range (a : ℤ) : Prop := 1 ≤ a ∧ a ≤ 300

-- Prove the number of integral values of a for which d_1 * d_2 is a multiple of 7
theorem num_values_multiple_of_seven : 
  (finset.filter (λ a, is_multiple_of_seven (d_1 a * d_2 a)) (finset.range 301)).card = 257 := 
sorry

end num_values_multiple_of_seven_l259_259818


namespace inequality_proof_l259_259696

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l259_259696


namespace sin_15_add_sin_75_l259_259119

theorem sin_15_add_sin_75 : 
  Real.sin (15 * Real.pi / 180) + Real.sin (75 * Real.pi / 180) = Real.sqrt 6 / 2 :=
by
  sorry

end sin_15_add_sin_75_l259_259119


namespace arithmetic_geometric_sequence_l259_259290

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

def is_geometric (a : ℕ → ℤ) (n1 n2 n3 : ℕ) : Prop :=
a n2 ^ 2 = a n1 * a n3

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : d ≠ 0)
  (h2 : is_arithmetic a d)
  (h3 : is_geometric a 0 2 3) :
  (∑ i in Finset.range 4, a i - ∑ i in Finset.range 2, a i) /
  (∑ i in Finset.range 5, a i - ∑ i in Finset.range 3, a i) = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l259_259290


namespace line_general_equation_and_slope_angle_l259_259259

section SlopeAngleOfLine

/- Definitions: conditions of the problem -/
def parametric_line (t : ℝ) : ℝ × ℝ := (-3 + t, 1 + sqrt 3 * t)

/- Theorem: the general equation of the line and its slope angle -/
theorem line_general_equation_and_slope_angle :
  (∃ t : ℝ, parametric_line t = (x, y)) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1) ∧
  (∃ α : ℝ, tan α = sqrt 3 ∧ α = π / 3) := by
  sorry

end SlopeAngleOfLine

end line_general_equation_and_slope_angle_l259_259259


namespace arithmetic_sequence_proof_l259_259316

-- Define the arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) * d) / 2

-- Given conditions
constant a1 d : ℤ
constant a1_value : a1
constant d_value : d
constant h1 : sum_of_first_n_terms a1 d 9 = 27
constant h2 : arithmetic_seq a1 d 10 = 8

-- Target statement
theorem arithmetic_sequence_proof : 
  arithmetic_seq a1 d 100 = 98 :=
sorry

end arithmetic_sequence_proof_l259_259316


namespace range_of_x_in_sqrt_x_plus_3_l259_259403

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259403


namespace coin_value_l259_259448

variables (n d q : ℕ)  -- Number of nickels, dimes, and quarters
variable (total_coins : n + d + q = 30)  -- Total coins condition

-- Original value in cents
def original_value : ℕ := 5 * n + 10 * d + 25 * q

-- Swapped values in cents
def swapped_value : ℕ := 10 * n + 25 * d + 5 * q

-- Condition given about the value difference
variable (value_difference : swapped_value = original_value + 150)

-- Prove the total value of coins is $5.00 (500 cents)
theorem coin_value : original_value = 500 :=
by
  sorry

end coin_value_l259_259448


namespace rahim_average_price_l259_259153

theorem rahim_average_price (books_first_shop books_second_shop : ℕ) 
  (price_first_shop price_second_shop : ℕ) 
  (total_books : books_first_shop + books_second_shop = 115)
  (total_price : price_first_shop + price_second_shop = 1840) : 
  (price_first_shop + price_second_shop) / (books_first_shop + books_second_shop) = 16 := 
by {
  have h_books_first_shop : books_first_shop = 55, sorry,
  have h_books_second_shop : books_second_shop = 60, sorry,
  have h_price_first_shop : price_first_shop = 1500, sorry,
  have h_price_second_shop : price_second_shop = 340, sorry,
  rw [h_books_first_shop, h_books_second_shop, h_price_first_shop, h_price_second_shop],
  norm_num,
}

end rahim_average_price_l259_259153


namespace proof_equation_ellipse_fixed_point_l259_259318

noncomputable def ellipse_equation (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :=
  (eccentricity : ℝ) := real.sqrt (a^2 - b^2) / a = real.sqrt(2) / 2

noncomputable def tangent_line_parabola (b : ℝ) :=
  (tangent_eq: y = x + b, parabola_eq: y^2 = 4 * x)

noncomputable def fixed_point_T (x y : ℝ) := 
  (S : ℝ × ℝ) = (0, -1/3) → 
  (T_coord : (x, y) = (0, 1))

theorem proof_equation_ellipse_fixed_point :
  (∃ a b: ℝ, a > b ∧ b > 0 ∧ (ellipse_equation a b ∧ tangent_line_parabola b ∧ ∃ x y: ℝ, fixed_point_T x y)) := 
begin
  sorry
end

end proof_equation_ellipse_fixed_point_l259_259318


namespace length_of_greater_segment_l259_259871

-- Definitions based on conditions
variable (shorter longer : ℝ)
variable (h1 : longer = shorter + 2)
variable (h2 : (longer^2) - (shorter^2) = 32)

-- Proof goal
theorem length_of_greater_segment : longer = 9 :=
by
  sorry

end length_of_greater_segment_l259_259871


namespace largest_number_greater_than_600_l259_259931

theorem largest_number_greater_than_600 (a : ℕ → ℕ) (h : ∀ i j k l : ℕ, i < j → k < l → i ≠ k → j ≠ l → a i + a j ≠ a k + a l) :
  ∃ i, a i > 600 :=
by
  have h49 : fintype.card (fin 49) = 49 := rfl
  sorry

end largest_number_greater_than_600_l259_259931


namespace total_wheels_in_garage_l259_259516

theorem total_wheels_in_garage (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) 
                               (num_cars : ℕ) (wheels_per_car : ℕ) :
  num_bicycles = 9 → wheels_per_bicycle = 2 → 
  num_cars = 16 → wheels_per_car = 4 → 
  (num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car) = 82 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_wheels_in_garage_l259_259516


namespace correct_propositions_l259_259741

section linear_relationships 

variables {m n : Type} {α β : Type}

def parallel (x y : Type) : Prop := sorry -- Use a proper definition of parallelism in Lean
def perpendicular (x y : Type) : Prop := sorry -- Use a proper definition of perpendicularity in Lean

noncomputable def proposition_1 : Prop := 
  parallel m α ∧ parallel n β ∧ parallel α β → parallel m n

noncomputable def proposition_2 : Prop := 
  parallel m α ∧ perpendicular n β ∧ perpendicular α β → parallel m n

noncomputable def proposition_3 : Prop := 
  perpendicular m α ∧ parallel n β ∧ parallel α β → perpendicular m n

noncomputable def proposition_4 : Prop := 
  perpendicular m α ∧ perpendicular n β ∧ perpendicular α β → perpendicular m n

theorem correct_propositions : 
  (¬proposition_1) ∧ (¬proposition_2) ∧ proposition_3 ∧ proposition_4 :=
by
  sorry

end linear_relationships

end correct_propositions_l259_259741


namespace range_of_independent_variable_l259_259397

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259397


namespace fraction_work_completed_in_25_days_eq_half_l259_259827

theorem fraction_work_completed_in_25_days_eq_half :
  ∀ (men_total men_new men_rem : ℕ) (length : ℝ) (total_days : ℕ) 
    (work_hours_per_day initial_days remaining_days : ℕ)
    (total_man_hours man_hours_in_initial_days: ℝ),
  men_total = 100 →
  length = 2 →
  total_days = 50 →
  work_hours_per_day = 8 →
  initial_days = 25 →
  men_total * total_days * work_hours_per_day = total_man_hours →
  men_total * initial_days * work_hours_per_day = man_hours_in_initial_days →
  men_new = 60 →
  remaining_days = total_days - initial_days →
  man_hours_in_initial_days / total_man_hours = 1 / 2 :=
by
  intro men_total men_new men_rem length total_days work_hours_per_day initial_days remaining_days total_man_hours man_hours_in_initial_days
  intros h_men_total h_length h_total_days h_whpd h_initial_days h_totalmh h_initialmh h_men_new h_remaining_days
  have h1 : total_man_hours = 100 * 50 * 8 := by rw [h_men_total, h_total_days, h_whpd]; norm_num
  have h2 : man_hours_in_initial_days = 100 * 25 * 8 := by rw [h_men_total, h_initial_days, h_whpd]; norm_num
  have h3 : 100 * 50 * 8 = 40_000 := by norm_num
  have h4 : 100 * 25 * 8 = 20_000 := by norm_num
  rw [h1, h3] at h_totalmh
  rw [h2, h4] at h_initialmh
  norm_num at h_totalmh
  norm_num at h_initialmh
  rw [←h_totalmh] at h_initialmh
  rw h_totalmh
  rw h_initialmh
  norm_num
  sorry

end fraction_work_completed_in_25_days_eq_half_l259_259827


namespace max_digit_sum_in_24_hour_clock_l259_259585

theorem max_digit_sum_in_24_hour_clock : ∃ n : ℕ, n = 24 ∧ n = max (∑ d in (00..23).to_finset.bind (λ h, (00..59).to_finset.bind (λ m, [(h % 10) + (h / 10 % 10) + (m % 10) + (m / 10 % 10)]))) 0 :=
by
  sorry

end max_digit_sum_in_24_hour_clock_l259_259585


namespace brianna_wins_probability_l259_259620

theorem brianna_wins_probability :
  let S := 2019 * 1010
  let N_win := {2} ∪ { k : Nat | 8 ≤ k ∧ k % 4 = 0 ∧ k ≤ 2016 }
  let Brianna_prob := (∑ n in N_win, n : ℚ) / S
  let m := 3969
  let n := 16009296875
  (m : ℚ) / n = Brianna_prob ∧ Nat.gcd m n = 1 → m + n = 16009296875 :=
by
  sorry

end brianna_wins_probability_l259_259620


namespace lucas_pay_per_window_l259_259054

-- Conditions
def num_floors : Nat := 3
def windows_per_floor : Nat := 3
def days_to_finish : Nat := 6
def penalty_rate : Nat := 3
def penalty_amount : Nat := 1
def final_payment : Nat := 16

-- Theorem statement
theorem lucas_pay_per_window :
  let total_windows := num_floors * windows_per_floor
  let total_penalty := penalty_amount * (days_to_finish / penalty_rate)
  let original_payment := final_payment + total_penalty
  let payment_per_window := original_payment / total_windows
  payment_per_window = 2 :=
by
  sorry

end lucas_pay_per_window_l259_259054


namespace triangle_not_necessarily_isosceles_l259_259013

theorem triangle_not_necessarily_isosceles
  (A B C O M N : Point)
  (midpoint_M : midpoint A B M)
  (midpoint_N : midpoint A C N)
  (incenter_O : incircle_center A B C O)
  (eq_dist_from_O : dist O M = dist O N)
  (dist_eq_midpoints : ∀ P Q : Point,
      (P ∈ incircle_tangency_points A B C) → 
      (Q ∈ incircle_tangency_points A B C) →
      dist O P = dist O Q) :
  ¬(isosceles_triangle A B C) :=
sorry

end triangle_not_necessarily_isosceles_l259_259013


namespace number_of_divisibles_l259_259748

theorem number_of_divisibles (n : ℕ) :
  (n = 20) ↔
  let S := { x : ℕ | (1 ≤ x ∧ x ≤ 60) ∧ (x % 4 = 0 ∨ x % 6 = 0) } in
  finset.card (finset.filter (λ x, x ∈ S) (finset.range 61)) = n :=
by
  let S := { x : ℕ | (1 ≤ x ∧ x ≤ 60) ∧ (x % 4 = 0 ∨ x % 6 = 0) }
  have h : finset.card (finset.filter (λ x, x ∈ S) (finset.range 61)) = 20 := by sorry
  exact ⟨λ h1, h1.symm ▸ h, λ h2, h2.symm ▸ h.symm⟩

end number_of_divisibles_l259_259748


namespace sqrt_25_24_23_22_plus_1_eq_551_l259_259994

theorem sqrt_25_24_23_22_plus_1_eq_551 :
  let x := 23 in
  (sqrt ((25 * 24 * 23 * 22) + 1) = 551) :=
by
  let x := 23
  sorry

end sqrt_25_24_23_22_plus_1_eq_551_l259_259994


namespace find_a_l259_259308

theorem find_a (a b : ℤ) : (∃ A : ℤ[X], (2 * polynomial.X ^ 4 + polynomial.X ^ 3 - a * polynomial.X ^ 2 + b * polynomial.X + a + b - 1) = (polynomial.X ^ 2 + polynomial.X - 6) * A) → a = 16 :=
by
  intros h
  sorry

end find_a_l259_259308


namespace dot_product_m_n_l259_259433

-- Define i, j, k as standard orthogonal basis vectors
def i : ℝ³ := ⟨1, 0, 0⟩
def j : ℝ³ := ⟨0, 1, 0⟩
def k : ℝ³ := ⟨0, 0, 1⟩

-- Define vectors m and n based on the conditions
def m : ℝ³ := 8 • i + 3 • k
def n : ℝ³ := -1 • i + 5 • j - 4 • k

-- Theorem stating that the dot product of m and n is -20
theorem dot_product_m_n : (m • n) = -20 := 
by
  -- Proof goes here
  sorry

end dot_product_m_n_l259_259433


namespace find_h_neg_one_l259_259229

theorem find_h_neg_one (h : ℝ → ℝ) (H : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := 
by 
  sorry

end find_h_neg_one_l259_259229


namespace percentage_increase_l259_259368

variables (A Q_A Q_B B : ℝ)

-- Conditions
def condition1 : Prop := Q_A / A = 0.70
def condition2 : Prop := B = 1.20 * A
def condition3 : Prop := Q_B / B = 0.875

-- Theorem
theorem percentage_increase (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  ((Q_B - Q_A) / Q_A) * 100 = 50 :=
by sorry

end percentage_increase_l259_259368


namespace evaluate_exponential_operations_l259_259547

theorem evaluate_exponential_operations (a : ℝ) :
  (2 * a^2 - a^2 ≠ 2) ∧
  (a^2 * a^4 = a^6) ∧
  ((a^2)^3 ≠ a^5) ∧
  (a^6 / a^2 ≠ a^3) := by
  sorry

end evaluate_exponential_operations_l259_259547


namespace probability_prime_multiple_assignment_l259_259613

theorem probability_prime_multiple_assignment :
  let numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let primes : Finset ℕ := {2, 3, 5}
  let valid_assignments :=
        { (a, b, c) | a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
          a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
          ∃ p1 ∈ primes, p1 * b = a ∧
          ∃ p2 ∈ primes, p2 * c = b }

  (valid_assignments.card : ℚ) / (numbers.card * (numbers.card - 1) * (numbers.card - 2)) = 1 / 40 :=
by
  sorry

end probability_prime_multiple_assignment_l259_259613


namespace geometric_sequence_xn_range_of_t_sum_inequality_l259_259723

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1) / Real.log 3) / (x + 1)

def P (n : ℕ) (xn : ℝ) := (xn, f xn)

def Q (n : ℕ) (xn : ℝ) := (xn, 0)

def x : ℕ → ℝ
| 0 => 2
| n+1 => 3 * x n + 2

namespace problem1

theorem geometric_sequence_xn : ∀ n, x n + 1 = 3^(n+1) :=
sorry

end problem1

namespace problem2

def y (n : ℕ) : ℝ := (n / 3^n : ℝ)

theorem range_of_t (n : ℕ) (m : ℝ) (hn : 0 < n) (hm : -1 ≤ m ∧ m ≤ 1) :
  ∀ t, 3 * t^2 - 6 * m * t + (1 / 3 : ℝ) > y n → t ∈ (-∞, -2) ∪ (2, ∞) :=
sorry

end problem2

namespace problem3

def S_n (n : ℕ) : ℝ := (4 * n + 1) / 3

theorem sum_inequality (n : ℕ) : (∑ i in Finset.range n, (1 / ((i+1) * S_n (i+1)))) < 3 :=
sorry

end problem3

end geometric_sequence_xn_range_of_t_sum_inequality_l259_259723


namespace find_length_of_bridge_l259_259606

noncomputable def length_of_train : ℝ := 165
noncomputable def speed_of_train_kmph : ℝ := 54
noncomputable def time_to_cross_bridge_seconds : ℝ := 67.66125376636536

noncomputable def speed_of_train_mps : ℝ :=
  speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance_covered : ℝ :=
  speed_of_train_mps * time_to_cross_bridge_seconds

noncomputable def length_of_bridge : ℝ :=
  total_distance_covered - length_of_train

theorem find_length_of_bridge : length_of_bridge = 849.92 := by
  sorry

end find_length_of_bridge_l259_259606


namespace gcd_le_sqrt_sum_l259_259884

theorem gcd_le_sqrt_sum (a b : ℕ) (h : (a + 1 : ℚ) / b + (b + 1 : ℚ) / a ∈ ℤ) : Nat.gcd a b ≤ Nat.sqrt (a + b) := 
sorry

end gcd_le_sqrt_sum_l259_259884


namespace average_is_805_l259_259109

open Real

def earnings : List ℝ := [620, 850, 760, 950, 680, 890, 720, 900, 780, 830, 800, 880]

def total_earnings : ℝ := List.sum earnings

def number_of_days : ℝ := 12

def average_daily_income : ℝ := total_earnings / number_of_days

theorem average_is_805 : average_daily_income = 805 :=
by
  sorry

end average_is_805_l259_259109


namespace range_of_x_of_sqrt_x_plus_3_l259_259392

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259392


namespace height_of_brick_l259_259946

-- Definitions of given conditions
def length_brick : ℝ := 125
def width_brick : ℝ := 11.25
def length_wall : ℝ := 800
def height_wall : ℝ := 600
def width_wall : ℝ := 22.5
def number_bricks : ℝ := 1280

-- Prove that the height of each brick is 6.01 cm
theorem height_of_brick :
  ∃ H : ℝ,
    H = 6.01 ∧
    (number_bricks * (length_brick * width_brick * H) = length_wall * height_wall * width_wall) :=
by
  sorry

end height_of_brick_l259_259946


namespace is_odd_function_l259_259875

def f (x : ℝ) : ℝ := x * Real.cos x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by 
  intro x
  sorry

end is_odd_function_l259_259875


namespace find_number_of_numbers_l259_259490

theorem find_number_of_numbers (S : ℝ) (n : ℝ) (h1 : S - 30 = 16 * n) (h2 : S = 19 * n) : n = 10 :=
by
  sorry

end find_number_of_numbers_l259_259490


namespace smallest_n_satisfying_inequality_l259_259228

noncomputable def sigma_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), Real.logb 10 (1 + 1 / 10 ^ (2 ^ k))

noncomputable def L : ℝ :=
  1 + Real.logb 10 (999 / 1000)

theorem smallest_n_satisfying_inequality :
  ∃ n : ℕ, (σ : sigma_n n) ≥ L ∧
           (∀ m : ℕ, (m < n) → (sigma_n m < L)) :=
by
  use 2
  sorry

end smallest_n_satisfying_inequality_l259_259228


namespace product_even_probability_l259_259274

def numbers := {1, 2, 3, 4, 5}

-- Define what it means for a product of two numbers to be even
def even_prod (a b : ℕ) : Prop := (a * b) % 2 = 0

-- Compute the number of total outcomes
def total_outcomes : ℕ := (numbers.card.choose 2)

-- Compute the number of favorable outcomes (even product)
def favorable_outcomes : ℕ :=
  let odds := {1, 3, 5}
  let evens := {2, 4}
  (odds.card.choose 1) * (evens.card.choose 1) + (evens.card.choose 2)

-- Calculate probability
def probability_even_product : ℚ := favorable_outcomes / total_outcomes

theorem product_even_probability :
  probability_even_product = 7 / 10 :=
sorry

end product_even_probability_l259_259274


namespace nhai_initial_men_l259_259830

theorem nhai_initial_men (M : ℕ) (W : ℕ) :
  let totalWork := M * 50 * 8 in
  let partialWork := M * 25 * 8 in
  let remainingWork := (M + 60) * 25 * 10 in
  partialWork = totalWork / 3 →
  remainingWork = (2 * totalWork) / 3 →
  M = 100 :=
by
  intros h1 h2
  have eq1 : totalWork = M * 50 * 8 := rfl
  sorry -- Proof is omitted

end nhai_initial_men_l259_259830


namespace inf_pow_equals_l259_259494

def inf_pow (x : ℝ) : ℝ := x ^ inf_pow x

theorem inf_pow_equals (x : ℝ) (h : inf_pow x = 4) : x = Real.sqrt 2 :=
by
  sorry

end inf_pow_equals_l259_259494


namespace find_a_with_integer_roots_l259_259319

theorem find_a_with_integer_roots :
  ∀ (x : ℤ), ∀ (a : ℤ),
    (∃ (r1 r2 r3 r4 : ℤ),
      r1 * r2 * r3 * r4 = a^2 - 21 * a + 68 ∧
      r1 + r2 + r3 + r4 = 16 ∧
      r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4 = 81 - 2 * a ∧
      (r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4) = -(16 * a - 142)) →
      a = -4 := 
begin
  sorry
end

end find_a_with_integer_roots_l259_259319


namespace route_time_saving_zero_l259_259059

theorem route_time_saving_zero 
  (distance_X : ℝ) (speed_X : ℝ) 
  (total_distance_Y : ℝ) (construction_distance_Y : ℝ) (construction_speed_Y : ℝ)
  (normal_distance_Y : ℝ) (normal_speed_Y : ℝ)
  (hx1 : distance_X = 7)
  (hx2 : speed_X = 35)
  (hy1 : total_distance_Y = 6)
  (hy2 : construction_distance_Y = 1)
  (hy3 : construction_speed_Y = 10)
  (hy4 : normal_distance_Y = 5)
  (hy5 : normal_speed_Y = 50) :
  (distance_X / speed_X * 60) - 
  ((construction_distance_Y / construction_speed_Y * 60) + 
  (normal_distance_Y / normal_speed_Y * 60)) = 0 := 
sorry

end route_time_saving_zero_l259_259059


namespace least_positive_number_of_24x_plus_16y_is_8_l259_259766

theorem least_positive_number_of_24x_plus_16y_is_8 :
  ∃ (x y : ℤ), 24 * x + 16 * y = 8 :=
by
  sorry

end least_positive_number_of_24x_plus_16y_is_8_l259_259766


namespace max_value_of_seq_l259_259737

noncomputable def a_seq : ℕ → ℝ
| 0       := -1 / 9
| (n + 1) := a_seq n / (8 * a_seq n + 1)

theorem max_value_of_seq : ∃ n : ℕ, ∀ m : ℕ, a_seq m ≤ a_seq n ∧ a_seq n = 1 / 7 :=
by
  sorry

end max_value_of_seq_l259_259737


namespace range_of_a_l259_259280

def f (x : ℝ) : ℝ := |Real.log x|

def g (x : ℝ) (a : ℝ) : ℝ := f x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → g x a = 0) → a ∈ Set.Ioo (Real.log 2 / 2) (1 / Real.exp 1) :=
  sorry

end range_of_a_l259_259280


namespace car_efficiency_l259_259902

-- Define the conditions
def miles_driven_per_round_trip : ℕ := 50
def days_per_week : ℕ := 5
def weeks_duration : ℕ := 4
def gas_cost_per_gallon : ℕ := 2
def total_gas_spent : ℕ := 80

-- Assertion (in miles per gallon)
theorem car_efficiency : 
  let total_gallons := total_gas_spent / gas_cost_per_gallon in
  let miles_per_week := miles_driven_per_round_trip * days_per_week in
  let total_miles := miles_per_week * weeks_duration in
  total_miles / total_gallons = 25 := 
by
  sorry

end car_efficiency_l259_259902


namespace prove_problem_l259_259284

noncomputable def problem (a b c x1 x2 x3 : ℝ) (h1 : a + b + c = 1)
  (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) : Prop :=
  let y1 := a * x1 + b * x2 + c * x3    
  let y2 := a * x2 + b * x3 + c * x1    
  let y3 := a * x3 + b * x1 + c * x2    
  y1 * y2 * y3 ≥ x1 * x2 * x3

theorem prove_problem (a b c x1 x2 x3 : ℝ) (h1 : a + b + c = 1)
  (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) : problem a b c x1 x2 x3 h1 hx1 hx2 hx3 :=
begin
  sorry
end

end prove_problem_l259_259284


namespace partition_subset_sum_l259_259067

theorem partition_subset_sum (m n k : ℕ) 
  (h_sum_eq : (n * (n + 1)) / 2 = m * k) 
  (h_m_ge_n : m ≥ n) : 
  ∃ (subsets : finset (finset ℕ)), 
    (subsets.card = k) ∧ 
    (∀ s ∈ subsets, s.sum id = m) ∧ 
    (finset.univ \subset (finset.bind subsets id)) :=
sorry

end partition_subset_sum_l259_259067


namespace range_of_x_in_sqrt_x_plus_3_l259_259404

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l259_259404


namespace intersection_condition_main_l259_259706

theorem intersection_condition (A : Set ℕ) : (A ∩ {0, 1} = {0}) → (A = {0} → false) :=
begin
  intro h,
  split,
  sorry -- Place-holder for necessary condition proof
end

theorem main :
  (∀ A : Set ℕ, (A ∩ {0, 1} = {0}) → (A = {0} → false)) 
  ∧ 
  (∃ A : Set ℕ, (A ∩ {0, 1} = {0}) ∧ (A ≠ {0})) :=
by
  sorry -- Overall proof to encapsulate both necessary and not sufficient.

end intersection_condition_main_l259_259706


namespace ones_digit_of_p_l259_259672

theorem ones_digit_of_p (p q r s : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) (hs : s.prime)
  (hseq : q = p + 4 ∧ r = q + 4 ∧ s = r + 4) (hp_gt_5 : p > 5) : 
  p % 10 = 9 := 
sorry

end ones_digit_of_p_l259_259672


namespace triangle_similarity_l259_259419

noncomputable def similarity_coefficient : ℝ := (4 / 9) ^ 2021

theorem triangle_similarity :
  ∀ (ABC A_2021C_2021C : Type)
  (AB BC AC : ℝ) (sides : AB = 5 ∧ BC = 6 ∧ AC = 4)
  (angle_bisector : Π (A B C : Type), Type)
  (angle_bisector_A1 : angle_bisector ABC A_2021C_2021C)
  (iterative_bisectors : ∀ (n : ℕ), n ≤ 2021 → angle_bisector (triangle_of (nat_extend n ABC)) (triangle_of (nat_extend (n + 1) ABC))),
  similar_triangles ABC A_2021C_2021C ∧ similarity_ratio ABC A_2021C_2021C = similarity_coefficient := sorry

end triangle_similarity_l259_259419


namespace min_le_S_max_l259_259041

variables {n : ℕ} (hn : n ≥ 2)
variables {x : Fin n → ℝ} (hx : Function.Injective x)
variables {p : Fin n → ℝ} (hp : ∀ i, p i > 0) (hp_sum : ∑ i, p i = 1)

noncomputable def S : ℝ :=
  (∑ i, p i * x i ^ 3 - (∑ i, p i * x i) ^ 3) /
    (3 * (∑ i, p i * x i ^ 2 - (∑ i, p i * x i) ^ 2))

theorem min_le_S_max :
  Finset.min' (Finset.univ.image x) (Finset.univ_nonempty.image hx) ≤ S hp hp_sum x ∧
  S hp hp_sum x ≤ Finset.max' (Finset.univ.image x) (Finset.univ_nonempty.image hx) :=
sorry

end min_le_S_max_l259_259041


namespace smaller_root_of_quadratic_l259_259666

theorem smaller_root_of_quadratic :
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (x₁^2 - 14 * x₁ + 45 = 0) ∧ (x₂^2 - 14 * x₂ + 45 = 0) ∧ (min x₁ x₂ = 5) :=
sorry

end smaller_root_of_quadratic_l259_259666


namespace radical_product_simplified_l259_259216

theorem radical_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (sqrt (50 * x) * sqrt (18 * x) * sqrt (8 * x)) = 60 * x * sqrt x :=
sorry

end radical_product_simplified_l259_259216


namespace shortest_distance_is_1_54_l259_259536

-- Define the circle equations
def circle1_eq (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 6*y - 15 = 0

def circle2_eq (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 + 12*y + 151 = 0

-- Define the centers and radii of the circles after completing the square
def center1 : ℝ × ℝ := (-4, 3)
def radius1 : ℝ := Real.sqrt 40

def center2 : ℝ × ℝ := (8, -6)
def radius2 : ℝ := Real.sqrt 51

-- Calculate Euclidean distance between two points in 2D
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the distance between the centers of the circles
def center_distance : ℝ :=
  euclidean_distance center1 center2

-- Calculate the shortest distance between the circles
def shortest_distance_between_circles : ℝ :=
  center_distance - (radius1 + radius2)

-- Prove that the shortest distance between the two given circles is 1.54
theorem shortest_distance_is_1_54 : shortest_distance_between_circles = 1.54 :=
  sorry

end shortest_distance_is_1_54_l259_259536


namespace find_n_l259_259176

theorem find_n 
  (num_engineers : ℕ) (num_technicians : ℕ) (num_workers : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (systematic_sampling_inclusion_exclusion : ∀ n : ℕ, ∃ k : ℕ, n ∣ total_population ↔ n + 1 ≠ total_population) 
  (stratified_sampling_lcm : ∃ lcm : ℕ, lcm = Nat.lcm (Nat.lcm num_engineers num_technicians) num_workers)
  (total_population_is_36 : total_population = 36)
  (num_engineers_is_6 : num_engineers = 6)
  (num_technicians_is_12 : num_technicians = 12)
  (num_workers_is_18 : num_workers = 18) :
  ∃ n : ℕ, n = 6 :=
by
  sorry

end find_n_l259_259176


namespace area_inside_S_l259_259081

def five_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 5 ∧ z = w - 1 / w

def S : set ℂ := {z : ℂ | five_presentable z}

theorem area_inside_S : 
  let areaS := (24/25) * (26/25) * 25 * Real.pi in
  ∀ z ∈ S, area_inside_S = 624 / 25 * Real.pi :=
by
  sorry

end area_inside_S_l259_259081


namespace no_hikers_in_morning_l259_259170

-- Given Conditions
def morning_rowers : ℕ := 13
def afternoon_rowers : ℕ := 21
def total_rowers : ℕ := 34

-- Statement to be proven
theorem no_hikers_in_morning : (total_rowers - afternoon_rowers = morning_rowers) →
                              (total_rowers - afternoon_rowers = morning_rowers) →
                              0 = 34 - 21 - morning_rowers :=
by
  intros h1 h2
  sorry

end no_hikers_in_morning_l259_259170


namespace find_radius_of_semicircle_l259_259189

-- Definitions for the rectangle and semi-circle
variable (L W : ℝ) -- Length and width of the rectangle
variable (r : ℝ) -- Radius of the semi-circle

-- Conditions given in the problem
def rectangle_perimeter : Prop := 2 * L + 2 * W = 216
def semicircle_diameter_eq_length : Prop := L = 2 * r 
def width_eq_twice_radius : Prop := W = 2 * r

-- Proof statement
theorem find_radius_of_semicircle
  (h_perimeter : rectangle_perimeter L W)
  (h_diameter : semicircle_diameter_eq_length L r)
  (h_width : width_eq_twice_radius W r) :
  r = 27 := by
  sorry

end find_radius_of_semicircle_l259_259189


namespace find_k_eq_neg2_l259_259345

theorem find_k_eq_neg2 (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by sorry

end find_k_eq_neg2_l259_259345


namespace complex_quadrant_l259_259179

theorem complex_quadrant (z : ℂ) (h : (-1 + complex.I) * z = (1 + complex.I) ^ 2) : 
  (1 : ℂ).re > 0 ∧ (-1 : ℂ).im < 0 :=
by
  sorry

end complex_quadrant_l259_259179


namespace two_digit_numbers_units_digit_six_l259_259895

theorem two_digit_numbers_units_digit_six :
  let count := (Finset.filter (λ x : ℕ, (9 * (x / 10)) % 10 = 6) (Finset.Ico 10 100)).card in
  count = 10 :=
by
  sorry

end two_digit_numbers_units_digit_six_l259_259895


namespace probability_transform_in_S_l259_259958

-- Definitions using the conditions identified
def region_S : set ℂ := {z : ℂ | let re := z.re, im := z.im in -2 ≤ re ∧ re ≤ 2 ∧ -2 ≤ im ∧ im ≤ 2}

-- Lean proof statement
theorem probability_transform_in_S :
  ∀ (z : ℂ), z ∈ region_S → (0.5 + 0.5 * complex.I) * z ∈ region_S := 
by 
  sorry

end probability_transform_in_S_l259_259958


namespace range_of_x_of_sqrt_x_plus_3_l259_259393

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259393


namespace problem1_expr_problem2_solution_l259_259932

/-
Problem 1 Proof Statement
-/
theorem problem1_expr : 
  sqrt (2^2 : ℝ) + abs (1 - sqrt 2) - real.cbrt (-8) = 3 + sqrt 2 :=
by
  sorry

/-
Problem 2 Proof Statement
-/
theorem problem2_solution (x y : ℝ) :
  (2 * x + 3 * y = 1) ∧ (3 * x - y = 7) ↔ (x = 2 ∧ y = -1) :=
by
  sorry

end problem1_expr_problem2_solution_l259_259932


namespace points_no_three_collinear_l259_259428

/-- Let p > 2 be a prime number and let L = {0,1,...,p-1}^2. 
    Prove that we can find p points in L with no three of them collinear. --/
theorem points_no_three_collinear (p : ℕ) (hp : p > 2) [fact (nat.prime p)] :
  ∃ (S : finset (fin p × fin p)), S.card = p ∧ 
  ∀ (a b c : fin p × fin p), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
  ¬ collinear {a, b, c} :=
sorry

end points_no_three_collinear_l259_259428


namespace solve_system_eqns_l259_259860

theorem solve_system_eqns :
  ∃ (x y : ℝ), (x + 3 * y = 2) ∧ (4 * x - y = 8) ∧ (x = 2) ∧ (y = 0) :=
by
  use 2, 0
  simp
  split
  · linarith
  · linarith
  · rfl
  · rfl

end solve_system_eqns_l259_259860


namespace sufficient_but_not_necessary_l259_259867

variables {R : Type*} [Field R]

def line1 (m : R) : ℕ × (R × R) := (1, (m, 2*m - 1))
def line2 (m : R) : ℕ × (R × R) := (1, (3, m))

def slopes (m : R) : (R × R) :=
  if (2*m - 1) = 0 then (0, 0) -- Handle vertical line
  else (-(m : R) / (2*m - 1), -3 / m)

def perpendicular_cond (m : R) :=
  let (s1, s2) := slopes m in s1 * s2 = -1

theorem sufficient_but_not_necessary (m : R) :
  m = -1 → 
  perpendicular_cond m ∧
  (∀ m' ≠ -1, ¬ perpendicular_cond m') :=
by
  sorry

end sufficient_but_not_necessary_l259_259867


namespace difference_of_cubes_l259_259457

theorem difference_of_cubes (x y : ℕ) (h1 : x = y + 3) (h2 : x + y = 5) : x^3 - y^3 = 63 :=
by sorry

end difference_of_cubes_l259_259457


namespace regression_equation_l259_259312

theorem regression_equation (slope : ℝ) (mean_x : ℝ) (mean_y : ℝ) (b : ℝ) :
  slope = 1.23 → mean_x = 4 → mean_y = 5 → b = 0.08 → 
  ∀ x : ℝ, (mean_y = slope * mean_x + b) → ∃ y : ℝ, y = slope * x + b :=
begin
  intros h_slope h_mean_x h_mean_y h_b h_equation x,
  use slope * x + b,
  simp,
  exact h_slope,
  exact h_mean_x,
  exact h_mean_y,
  exact h_b,
  exact h_equation,
end

end regression_equation_l259_259312


namespace tetrahedron_face_area_inequality_l259_259473

theorem tetrahedron_face_area_inequality
  (T_ABC T_ABD T_ACD T_BCD : ℝ)
  (h : T_ABC ≥ 0 ∧ T_ABD ≥ 0 ∧ T_ACD ≥ 0 ∧ T_BCD ≥ 0) :
  T_ABC < T_ABD + T_ACD + T_BCD :=
sorry

end tetrahedron_face_area_inequality_l259_259473


namespace ratio_D_E_equal_l259_259825

variable (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ)

def mary_story_conditions (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) : Prop :=
  total_characters = 60 ∧
  initial_A = 1 / 2 * total_characters ∧
  initial_C = 1 / 2 * initial_A ∧
  initial_D + initial_E = total_characters - (initial_A + initial_C)

theorem ratio_D_E_equal (total_characters initial_A initial_C initial_D initial_E : ℕ) :
  mary_story_conditions total_characters initial_A initial_C initial_D initial_E →
  initial_D = initial_E :=
sorry

end ratio_D_E_equal_l259_259825


namespace positive_divisors_d17_l259_259560

theorem positive_divisors_d17 (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (h_order : d 1 = 1 ∧ ∀ i, 1 ≤ i → i ≤ k → d i < d (i + 1)) 
  (h_last : d k = n) (h_pythagorean : d 7 ^ 2 + d 15 ^ 2 = d 16 ^ 2) : 
  d 17 = 28 :=
sorry

end positive_divisors_d17_l259_259560


namespace revenue_fall_correct_l259_259925

variable (old_revenue new_revenue : ℝ)
variable (percent_decrease : ℝ)

def revenue_fall_percent (old_revenue new_revenue : ℝ) : ℝ :=
  ((old_revenue - new_revenue) / old_revenue) * 100

theorem revenue_fall_correct :
  old_revenue = 69.0 ∧ new_revenue = 52.0 →
  revenue_fall_percent old_revenue new_revenue = 24.64 :=
by
  sorry

end revenue_fall_correct_l259_259925


namespace am_gm_inequality_l259_259442

theorem am_gm_inequality 
  (n : ℕ) (a : ℝ) 
  (x : ℕ → ℝ) 
  (h₀ : ∀ i, 0 ≤ x i) 
  (h₁ : ∑ i in finset.range n, x i = a) 
  (h₂ : n ≥ 2) :
  ∑ i in finset.range (n - 1), x i * x (i + 1) ≤ 1 / 4 * a ^ 2 :=
sorry

end am_gm_inequality_l259_259442


namespace find_d_and_q_l259_259712

-- Define the conditions of the problem
variable (n : ℕ) (d q : ℕ)
variable (a_n b_n S_n T_n : ℕ → ℕ)
variable (h1 : n > 0)
variable (h2 : ∀ n, S_n = n * (a_n 1) + (n * (n - 1) * d) / 2)
variable (h3 : ∀ n, T_n = b_n 1 * (1 - q^n) / (1 - q))
variable (h4 : ∀ n, n^2 * (T_n + 1) = 2^n * S_n)

-- State the theorem
theorem find_d_and_q : d = 2 ∧ q = 2 :=
by 
  sorry

end find_d_and_q_l259_259712


namespace smallest_n_for_g_gt_21_l259_259034

def g (n : ℕ) : ℕ := sorry

theorem smallest_n_for_g_gt_21 (h : ∃ k, n = 21 * k) (hg : ∀ n, g n > 21 ↔ n = 483) :
  ∃ n, n = 483 :=
by
  use 483
  sorry

end smallest_n_for_g_gt_21_l259_259034


namespace range_of_sqrt_x_plus_3_l259_259383

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l259_259383


namespace Shekar_marks_in_English_l259_259854

theorem Shekar_marks_in_English:
  let math_marks := 76 in
  let science_marks := 65 in
  let social_studies_marks := 82 in
  let biology_marks := 55 in
  let average_marks := 69 in
  let total_subjects := 5 in
  let marks_english := (average_marks * total_subjects) - (math_marks + science_marks + social_studies_marks + biology_marks) in
  marks_english = 67 :=
by
  let math_marks := 76
  let science_marks := 65
  let social_studies_marks := 82
  let biology_marks := 55
  let average_marks := 69
  let total_subjects := 5
  let marks_english := (average_marks * total_subjects) - (math_marks + science_marks + social_studies_marks + biology_marks)
  have : marks_english = 67 := by
    calc
      marks_english = (69 * 5) - (76 + 65 + 82 + 55) : rfl
      ... = 345 - 278 : by norm_num
      ... = 67 : by norm_num
  exact this

end Shekar_marks_in_English_l259_259854


namespace value_of_f_at_log_94_l259_259313

-- Definitions and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def f (x : ℝ) : ℝ := if x < 0 then 3^x else sorry

-- Main statement to prove
theorem value_of_f_at_log_94 : 
  let f := (λ x : ℝ, if x < 0 then 3^x else sorry) in
  odd_function f →
  f (Real.log 4 / Real.log 9) = -1 / 2 :=
by
  intros f_odd
  -- We need to skip the actual proof
  sorry

end value_of_f_at_log_94_l259_259313


namespace sum_s_t_h_l259_259441

noncomputable def binom : ℕ → ℕ → ℕ := λ n k, nat.choose n k

def S_n (n : ℕ) : ℝ := 
  ∑ k in finset.range (n/3 + 1), binom n (3 * k)

def T_n (n : ℕ) : ℝ := 
  ∑ k in finset.range (n/3 + 1), if 3 * k + 1 ≤ n then binom n (3 * k + 1) else 0

def H_n (n : ℕ) : ℝ := 
  ∑ k in finset.range (n/3 + 1), if 3 * k + 2 ≤ n then binom n (3 * k + 2) else 0

theorem sum_s_t_h (n : ℕ) (hn : 0 < n) :
  S_n n = (1/3) * (2^n + 2 * real.cos (n * real.pi / 3)) ∧
  T_n n = (1/3) * (2^n + 2 * real.cos ((n - 2) * real.pi / 3)) ∧
  H_n n = (1/3) * (2^n - 2 * real.cos ((n - 1) * real.pi / 3)) :=
by
  sorry

end sum_s_t_h_l259_259441


namespace apple_eating_contest_l259_259648

theorem apple_eating_contest :
  let eaten := [3, 5, 8, 6, 7, 4, 2, 5] in
  let adam := 8 in
  let zoe := 2 in
  (adam - zoe = 6) ∧ (List.sum eaten = 40) :=
by
  let eaten := [3, 5, 8, 6, 7, 4, 2, 5]
  let adam := 8
  let zoe := 2
  show (adam - zoe = 6) ∧ (List.sum eaten = 40), from sorry

end apple_eating_contest_l259_259648


namespace quadrilateral_EFGH_perimeter_thm_l259_259912

noncomputable def lengthEF : ℝ := 15
noncomputable def lengthGH : ℝ := 7
noncomputable def lengthFG : ℝ := 14

def is_perpendicular (a b : ℝ) : Prop := a = 0 ∨ b = 0

def quadrilateral_EFGH_perimeter
  (EF_perp_FG : is_perpendicular lengthEF lengthFG)
  (GH_perp_FG : is_perpendicular lengthGH lengthFG)
  (EF : ℝ := lengthEF)
  (GH : ℝ := lengthGH)
  (FG : ℝ := lengthFG) : ℝ :=
  EF + FG + GH + 2 * Real.sqrt (FG ^ 2 + (EF - GH) ^ 2)

theorem quadrilateral_EFGH_perimeter_thm
  (EF_perp_FG : is_perpendicular lengthEF lengthFG)
  (GH_perp_FG : is_perpendicular lengthGH lengthFG) :
  quadrilateral_EFGH_perimeter EF_perp_FG GH_perp_FG = 36 + 2 * Real.sqrt 65 := 
sorry

end quadrilateral_EFGH_perimeter_thm_l259_259912


namespace area_right_triangle_twice_area_AMN_l259_259414

open EuclideanGeometry

variables {A B C D M N : Point} (BC_mid : midpoint B C D)
variables (incenterABD : ∃ I1, Incenter I1 A B D) (incenterACD : ∃ I2, Incenter I2 A C D)
variables (line_I1I2 : ∃ I1 I2 : Point, Collinear ({I1, I2}) ∧ Line I1 I2 ∩ Line A B ≠ ∅ ∧ Line I1 I2 ∩ Line A C ≠ ∅)

/-- Given a right triangle ABC with D as the midpoint of hypotenuse BC, and incenters I1 and I2 of triangles ABD and ACD respectively,
    prove that the area of triangle ABC is twice the area of triangle AMN where line I1I2 intersects AB and AC at points M and N respectively. -/
theorem area_right_triangle_twice_area_AMN :
  is_right_triangle A B C →
  midpoint B C D →
  (∃ I1, Incenter I1 A B D) →
  (∃ I2, Incenter I2 A C D) →
  (∃ I1 I2 : Point, Collinear ({I1, I2}) ∧ Line I1 I2 ∩ Line A B ≠ ∅ ∧ Line I1 I2 ∩ Line A C ≠ ∅) →
  2 * area_triangle A B C = area_triangle A M N := sorry

end area_right_triangle_twice_area_AMN_l259_259414


namespace inequality_a_b_c_l259_259070

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l259_259070


namespace total_admission_cost_l259_259777

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l259_259777


namespace eq_abs_piecewise_l259_259918

theorem eq_abs_piecewise (x : ℝ) : (|x| = if x >= 0 then x else -x) :=
by
  sorry

end eq_abs_piecewise_l259_259918


namespace find_number_l259_259621

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l259_259621


namespace find_AC_of_right_triangle_l259_259788

theorem find_AC_of_right_triangle (A B C : Type)
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (hABC : triangle A B C)
  (right_triangle : ∠ A C B = real.pi / 2)
  (AB : real.norm (A - B) = 9)
  (cos_B : real.cos (∠ B A C) = 2 / 3) :
  real.norm (A - C) = 3 * real.sqrt 5 :=
by sorry

end find_AC_of_right_triangle_l259_259788


namespace count_ways_to_place_dominos_l259_259458

theorem count_ways_to_place_dominos (n : ℕ) :
  let placement_count := (Nat.choose (2 * n) n) ^ 2 in
  ∃ (ways : ℕ), ways = placement_count :=
by
  let placement_count := (Nat.choose (2 * n) n) ^ 2
  exists placement_count
  sorry

end count_ways_to_place_dominos_l259_259458


namespace fourth_column_rectangles_l259_259980

theorem fourth_column_rectangles (grid : array (array ℕ) 7 7) 
  (h_condition : ∀ (i j : ℕ) (h1 : i < 7) (h2 : j < 7), ∃ n : ℕ, grid[i][j] = n ∧ n = (size of the rectangle containing (i, j))) : 
  ∃ (R : set (set (ℕ × ℕ))), 
    (∀ r ∈ R, (∃ a b, r = { (i, j) | a ≤ i < a + x ∧ b ≤ j < b + y } where (x * y = grid[a][b]))) ∧
    ( ∑ r in R, finset.card (r ∩ {(4, j) | j < 7}) = 7) ∧
    (finset.card {r ∈ R | ∃ j < 7, (4, j) ∈ r} = 4) :=
sorry

end fourth_column_rectangles_l259_259980


namespace chessboard_pawn_placement_l259_259352

theorem chessboard_pawn_placement :
  let n := 5 in
  let factorial (n : Nat) : Nat := (List.range n).prod + 1 in
  let ways := factorial n * factorial n in
  ways = 14400 :=
by
  have h_fact_5 : factorial 5 = 120 := by
    sorry
  have h_ways : factorial 5 * factorial 5 = 14400 := by
    rw [h_fact_5],
    norm_num
  exact h_ways

end chessboard_pawn_placement_l259_259352


namespace twenty_four_point_game_l259_259482

theorem twenty_four_point_game : (9 + 7) * 3 / 2 = 24 := by
  sorry -- Proof to be provided

end twenty_four_point_game_l259_259482


namespace find_number_l259_259573

theorem find_number (x : ℝ) (h : (2/3 * x)^3 - 10 = 14) : x ≈ 4.3267 :=
sorry

end find_number_l259_259573


namespace div_37_permutation_l259_259075

-- Let A, B, C be digits of a three-digit number
variables (A B C : ℕ) -- these can take values from 0 to 9
variables (p : ℕ) -- integer multiplier for the divisibility condition

-- The main theorem stated as a Lean 4 problem
theorem div_37_permutation (h : 100 * A + 10 * B + C = 37 * p) : 
  ∃ (M : ℕ), (M = 100 * B + 10 * C + A ∨ M = 100 * C + 10 * A + B ∨ M = 100 * A + 10 * C + B ∨ M = 100 * C + 10 * B + A ∨ M = 100 * B + 10 * A + C) ∧ 37 ∣ M :=
by
  sorry

end div_37_permutation_l259_259075


namespace volleyball_tournament_total_games_l259_259610

theorem volleyball_tournament_total_games (n : ℕ) (games_per_pair : ℕ) (unique_pair_games : ℕ)
  (h1 : n = 10) (h2 : games_per_pair = 4) (h3 : unique_pair_games = n * (n - 1) / 2) :
  4 * unique_pair_games = 180 :=
by
  rw [h1] at h3
  have unique_games_10 : unique_pair_games = 10 * 9 / 2 := by rw [h3, h1]
  rw [unique_games_10]
  norm_num
  sorry

end volleyball_tournament_total_games_l259_259610


namespace function_domain_l259_259408

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259408


namespace parallel_vectors_m_l259_259744

theorem parallel_vectors_m (m : ℝ) :
  let a := (1, 2)
  let b := (m, m + 1)
  a.1 * b.2 = a.2 * b.1 → m = 1 :=
by
  intros a b h
  dsimp at *
  sorry

end parallel_vectors_m_l259_259744


namespace circumcircle_radius_of_sector_l259_259188

theorem circumcircle_radius_of_sector (θ : Real) (r : Real) (cos_val : Real) (R : Real) :
  θ = 30 * Real.pi / 180 ∧ r = 8 ∧ cos_val = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ R = 8 * (Real.sqrt 6 - Real.sqrt 2) →
  R = 8 * (Real.sqrt 6 - Real.sqrt 2) :=
by
  sorry

end circumcircle_radius_of_sector_l259_259188


namespace nora_picked_from_second_tree_l259_259451

theorem nora_picked_from_second_tree (total first third : ℕ) (h_total : total = 260) (h_first : first = 80) (h_third : third = 120) :
  total - first - third = 60 := 
by
  rw [h_total, h_first, h_third]
  exact rfl

end nora_picked_from_second_tree_l259_259451


namespace pool_filling_rate_l259_259991

theorem pool_filling_rate:
  ∃ R: ℝ, R + 20 + 14 - 8 = 34 ∧ R = 8 :=
by
  use 8
  split
  . norm_num
  . rfl

end pool_filling_rate_l259_259991


namespace rectangle_area_l259_259964

theorem rectangle_area
  (a : ℝ) (b: ℝ) (l_rect: ℝ) (w: ℝ)
  (h_square_side : a = 15)
  (h_rect_length : l_rect = 18)
  (h_perimeters_equal : 4 * a = 2 * (l_rect + w)) :
  b = l_rect * w :=
by
  have h_square_perimeter : 4 * 15 = 60 := by norm_num
  have h_rect_perimeter : 2 * (18 + w) = 60 := h_perimeters_equal.subst h_square_perimeter
  have h_solve_width : w = 12 := by linarith [h_rect_perimeter]
  have h_calculate_area : 18 * 12 = 216 := by norm_num
  simp [h_calculate_area] at ⊢ h_solve_width; sorry

end rectangle_area_l259_259964


namespace max_crosses_4x10_no_odd_crosses_5x10_l259_259554

-- Part (a): Maximum number of crosses in a 4x10 grid
theorem max_crosses_4x10 : ∃ (n : ℕ), (∀ row : ℕ, row < 4 → odd (number_of_crosses_in_row row)) ∧
                                      (∀ col : ℕ, col < 10 → odd (number_of_crosses_in_col col)) ∧
                                      n = 30 := by
  sorry

-- Part (b): Impossibility of placing crosses in a 5x10 grid
theorem no_odd_crosses_5x10 : ¬ ∃ (f : ℕ × ℕ → ℕ),
  (∀ row : ℕ, row < 5 → odd (sum (λ col, f (row, col)))) ∧
  (∀ col : ℕ, col < 10 → odd (sum (λ row, f (row, col)))) := by
  sorry

end max_crosses_4x10_no_odd_crosses_5x10_l259_259554


namespace Vasya_wins_l259_259840

theorem Vasya_wins :
  ∃ (assign_values : (Fin 10 → ℝ)) (x : Fin 10 → ℝ) (cards_Vasya cards_Petya : Finset (Fin 5 → ℝ)),
  (∀ i, 0 ≤ x i) ∧
  (∀ i j, i ≤ j → x i ≤ x j) ∧
  (cards_Vasya ∪ cards_Petya = Finset.univ) ∧
  (cards_Vasya ∩ cards_Petya = ∅) ∧
  (∑ c in cards_Vasya, ∏ i, x (c i)) > (∑ c in cards_Petya, ∏ i, x (c i)) := 
sorry

end Vasya_wins_l259_259840


namespace function_domain_l259_259411

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l259_259411


namespace travel_distance_l259_259580

-- Define the average speed of the car
def speed : ℕ := 68

-- Define the duration of the trip in hours
def time : ℕ := 12

-- Define the distance formula for constant speed
def distance (speed time : ℕ) : ℕ := speed * time

-- Proof statement
theorem travel_distance : distance speed time = 756 := by
  -- Provide a placeholder for the proof
  sorry

end travel_distance_l259_259580


namespace find_k_for_smallest_period_l259_259762

theorem find_k_for_smallest_period :
  ∃ k > 0, (∀ x, sin (k * x + π / 5) = sin (k * (x + 2 * π / (3 * k)) + π / 5)) → k = 3 :=
begin
  sorry
end

end find_k_for_smallest_period_l259_259762


namespace solve_keychain_problem_l259_259855

def keychain_problem : Prop :=
  let f_class := 6
  let f_club := f_class / 2
  let thread_total := 108
  let total_friends := f_class + f_club
  let threads_per_keychain := thread_total / total_friends
  threads_per_keychain = 12

theorem solve_keychain_problem : keychain_problem :=
  by sorry

end solve_keychain_problem_l259_259855


namespace parallel_line_and_plane_iff_dot_product_zero_l259_259707

variable {V : Type*} [InnerProductSpace ℝ V]

variable {l : Submodule ℝ V}   -- line l
variable {α : Submodule ℝ V}  -- plane α
variable (a u : V)            -- vectors

-- Definitions based on given conditions
def direction_vector_of_line := l.direction
def normal_vector_of_plane := α.orthogonal

-- Proposition to prove
theorem parallel_line_and_plane_iff_dot_product_zero
    (h₁ : a ∈ direction_vector_of_line) (h₂ : u ∈ normal_vector_of_plane) : 
    a ⬝ u = 0 ↔ l ≤ α.orthogonal := sorry

-- sorry is used since the proof steps are not required

end parallel_line_and_plane_iff_dot_product_zero_l259_259707


namespace sum_of_max_min_values_l259_259323

noncomputable def f (x : ℝ) (α : ℚ) := x^(α : ℝ) + 1

theorem sum_of_max_min_values
  (α : ℚ) (a b : ℝ) (hα : α ∈ ℚ) (ha : 0 < a) (hb : a < b)
  (hmax_f_ab : ∀ x, x ∈ set.Icc a b → f x α ≤ 6)
  (hmin_f_ab : ∀ x, x ∈ set.Icc a b → 3 ≤ f x α) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ set.Icc (-b) (-a) ∧ x₂ ∈ set.Icc (-b) (-a) ∧
    ((f x₁ α + f x₂ α = 9) ∨ (f x₁ α + f x₂ α = -5))) :=
sorry

end sum_of_max_min_values_l259_259323


namespace sum_of_powers_of_4_l259_259995

theorem sum_of_powers_of_4 : 4^0 + 4^1 + 4^2 + 4^3 = 85 :=
by
  sorry

end sum_of_powers_of_4_l259_259995


namespace concyclic_ascertain_l259_259798

-- Definitions for the conditions in the problem
variable {A B C D E F I M N : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited I] [Inhabited M] [Inhabited N]
variable {α β γ : ℝ}
variable {incenter : Triangle A B C → Point} -- incenter of the triangle
variable {perpendicular_bisector : Segment A D → Line} -- perpendicular bisector function

-- Conditions
axiom angle_bisectors_concurrent : ∀ {A B C D E F : Type} (Δ : Triangle A B C) (D E F : Point), 
                                  bisects_angle Δ A B D → 
                                  bisects_angle Δ B C E → 
                                  bisects_angle Δ C A F → 
                                  concurrency D E F

axiom perpendicular_bisector_intersects : ∀ {A B C D M N : Type} (Δ : Triangle A B C) (D : Point) (p_bisector : Line),
                                  midpoint D (A, D) →
                                  intersects p_bisector (B, E) M → 
                                  intersects p_bisector (C, F) N

-- Goal
theorem concyclic_ascertain : ∀ {A B C D E F I M N : Type} (Δ : Triangle A B C) (D E F : Point) (I : Point) (M N : Point),
  angle_bisectors_concurrent Δ A B C D E F I →
  perpendicular_bisector_intersects Δ D (perpendicular_bisector (A, D)) M N →
  (Concyclic A I M N) := sorry

end concyclic_ascertain_l259_259798


namespace find_a_l259_259789

-- Definitions and conditions
def parabola_focus := (1 : ℝ, 0 : ℝ)

def hyperbola (a : ℝ) : Prop :=
  ∃ (c : ℝ), c > 0 ∧ x^2 - y^2 = a^2 ∧ (c, 0) = (1, 0)

-- The theorem to be proved
theorem find_a (a : ℝ) (h : a > 0) (focus_cond : hyperbola a) :
  a = sqrt(2)/2 :=
by
  sorry

end find_a_l259_259789


namespace kath_movie_cost_l259_259780

theorem kath_movie_cost :
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  discounted_price * number_of_people = 30 := by
  -- Definitions from conditions
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  -- Derived calculation based on conditions
  have h_discounted_price : discounted_price = 5 := by
    calc
      discounted_price = 8 - 3 : by sorry
      ... = 5 : by sorry
  have h_number_of_people : number_of_people = 6 := by
    calc
      number_of_people = 1 + 2 + 3 : by sorry
      ... = 6 : by sorry
  show 5 * 6 = 30 from sorry

end kath_movie_cost_l259_259780


namespace fraction_of_work_completed_in_25_days_l259_259828

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end fraction_of_work_completed_in_25_days_l259_259828


namespace correct_option_l259_259917

-- Define options
def optionA : Prop := (-3) - (-5) = -8
def optionB : Prop := (-3) + (-5) = +8
def optionC : Prop := (-3)^3 = -9
def optionD : Prop := -3^2 = -9

-- Theorem statement: Option D is correct and all other options are incorrect.
theorem correct_option : ¬optionA ∧ ¬optionB ∧ ¬optionC ∧ optionD :=
by
  -- Here we don't need to provide the proof. Just the statement is required.
  sorry

end correct_option_l259_259917


namespace jenny_real_estate_investment_l259_259804

def total_investment := 200000
def mutual_funds_investment (m : ℕ) : ℕ := m
def real_estate_investment (m : ℕ) : ℕ := 3 * m

theorem jenny_real_estate_investment (m : ℕ) 
  (htotal : mutual_funds_investment m + real_estate_investment m = total_investment) :
  real_estate_investment m = 150000 :=
begin
  sorry
end

end jenny_real_estate_investment_l259_259804


namespace bus_ride_cost_proof_l259_259607

noncomputable def bus_ride_cost_in_currency_X_before_service_fee : ℝ :=
    let B := 1.85 in B

theorem bus_ride_cost_proof :
  ∀ (B T : ℝ),
    T = B + 6.85 →
    0.85 * T + B + 1.25 = 10.50 →
    B = bus_ride_cost_in_currency_X_before_service_fee :=
begin
  intros B T hT hTotal,
  unfold bus_ride_cost_in_currency_X_before_service_fee at *,
  have hDiscountedTrain := 0.85 * (B + 6.85),
  have hTotalCost := hDiscountedTrain + B + 1.25,
  have hEquation : 0.85 * (B + 6.85) + B + 1.25 = 10.50 := by {
    rw hT at hDiscountedTrain,
    simp [hDiscountedTrain] at hTotalCost,
    exact hTotalCost
  },
  linarith
end

end bus_ride_cost_proof_l259_259607


namespace sin_asymptotic_tan_asymptotic_log_asymptotic_arcsin_asymptotic_arctan_asymptotic_sqrt_asymptotic_l259_259461

variable (α : ℝ)

theorem sin_asymptotic : tendsto (λ α, sin α / α) (nhds 0) (nhds 1) :=
sorry

theorem tan_asymptotic : tendsto (λ α, tan α / α) (nhds 0) (nhds 1) :=
sorry

theorem log_asymptotic : tendsto (λ α, log (1 + α) / α) (nhds 0) (nhds 1) :=
sorry

theorem arcsin_asymptotic : tendsto (λ α, arcsin α / α) (nhds 0) (nhds 1) :=
sorry

theorem arctan_asymptotic : tendsto (λ α, arctan α / α) (nhds 0) (nhds 1) :=
sorry

theorem sqrt_asymptotic : tendsto (λ α, (sqrt (1 + α) - 1) / (α / 2)) (nhds 0) (nhds 1) :=
sorry

end sin_asymptotic_tan_asymptotic_log_asymptotic_arcsin_asymptotic_arctan_asymptotic_sqrt_asymptotic_l259_259461


namespace similar_triangle_perimeter_l259_259908

/-
  Given an isosceles triangle with two equal sides of 18 inches and a base of 12 inches, 
  and a similar triangle with the shortest side of 30 inches, 
  prove that the perimeter of the similar triangle is 120 inches.
-/

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_perimeter
  (a b c : ℕ) (a' b' c' : ℕ) (h1 : is_isosceles a b c)
  (h2 : a = 12) (h3 : b = 18) (h4 : c = 18)
  (h5 : a' = 30) (h6 : a' * 18 = a * b')
  (h7 : a' * 18 = a * c') :
  a' + b' + c' = 120 :=
by {
  sorry
}

end similar_triangle_perimeter_l259_259908


namespace inequality_solution_l259_259892

def solution_set_of_inequality (x : ℝ) : Prop :=
  x * (x - 1) < 0

theorem inequality_solution :
  { x : ℝ | solution_set_of_inequality x } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l259_259892


namespace water_height_in_cylinder_l259_259617

theorem water_height_in_cylinder :
  let r_cone := 10 -- Radius of the cone in cm
  let h_cone := 15 -- Height of the cone in cm
  let r_cylinder := 20 -- Radius of the cylinder in cm
  let volume_cone := (1 / 3) * Real.pi * r_cone^2 * h_cone
  volume_cone = 500 * Real.pi -> 
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  h_cylinder = 1.25 := 
by
  intros r_cone h_cone r_cylinder volume_cone h_volume
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  have : h_cylinder = 1.25 := by
    sorry
  exact this

end water_height_in_cylinder_l259_259617


namespace equilateral_triangle_count_l259_259027

def is_equilateral_with_side {α : Type} [MetricSpace α] (s : Set α) (eq_length : ℝ) :=
  ∀ (a b c : α), a ∈ s → b ∈ s → c ∈ s → 
  dist a b = eq_length ∧ dist b c = eq_length ∧ dist c a = eq_length

def T := { p : (ℤ × ℤ × ℤ) | p.1 ∈ ({0, 1, 2, 3} : Set ℤ) ∧ p.2.1 ∈ ({0, 1, 2, 3} : Set ℤ) ∧ p.2.2 ∈ ({0, 1, 2, 3} : Set ℤ) }

noncomputable def num_equilateral_triangles : ℕ :=
  (∑ p in T, ∑ q in T, ∑ r in T, if is_equilateral_with_side {p, q, r} (sqrt 2) ∨ is_equilateral_with_side {p, q, r} (sqrt 3) then 1 else 0)

theorem equilateral_triangle_count : num_equilateral_triangles = 384 := 
sorry

end equilateral_triangle_count_l259_259027


namespace polynomial_pairs_l259_259253

-- Define polynomials over integers
def poly := polynomial ℤ 

-- Define the conditions for the polynomials P and Q
def satisfies_condition (P Q : poly) : Prop :=
  ∀ n m : ℤ, P.eval (n + Q.eval m) = Q.eval (n + P.eval m)

-- The result we're trying to prove
def valid_polynomials (P Q : poly) : Prop :=
  (P = Q) ∨ (∃ a b : ℤ, P = polynomial.X + polynomial.C a ∧ Q = polynomial.X + polynomial.C b)

-- The theorem statement
theorem polynomial_pairs (P Q : poly) (h : satisfies_condition P Q) : valid_polynomials P Q := 
  sorry

end polynomial_pairs_l259_259253


namespace three_students_with_A_l259_259769

variable (Alan Beth Carlos Diana Ellen : Prop)

theorem three_students_with_A :
  ((Alan → Beth) ∧ (Beth → Carlos) ∧ (Carlos → Diana) ∧ (Diana → Ellen)) →
  ((((Alan ∧ Beth ∧ Carlos) ∧ ¬Diana ∧ ¬Ellen) ∨
   ((Beth ∧ Carlos ∧ Diana) ∧ ¬Alan ∧ ¬Ellen) ∨
   ((Carlos ∧ Diana ∧ Ellen) ∧ ¬Alan ∧ ¬Beth)) ↔
  true) :=
by
  intro h
  apply Iff.intro
  · intro _; trivial
  · intro _; {
    cases h with habc h1,
    cases h1 with hbc h2,
    cases h2 with hcd h3,
    cases h3 with hde _,
    cases habc with ha hb,
    cases ha with h3 h4,
    cases hbc with hb1 hb2,
    cases hcd with hb3 hb4,
    cases hde with hb5 hb6,
    sorry
  }

end three_students_with_A_l259_259769


namespace francie_remaining_money_l259_259674

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end francie_remaining_money_l259_259674


namespace area_PQR_le_t_div_4_l259_259092

theorem area_PQR_le_t_div_4 (A B C P Q R : Type) 
  [triangle : is_triangle A B C]
  (hP : is_angle_bisector_intersect A B P C)
  (hQ : is_angle_bisector_intersect B C Q A)
  (hR : is_angle_bisector_intersect C A R B)
  (area_ABC : ℝ := triangle_area A B C)
  (t : ℝ := area_ABC) :
  (triangle_area P Q R) ≤ t / 4 :=
sorry


end area_PQR_le_t_div_4_l259_259092


namespace greatest_number_of_fruit_baskets_l259_259839

def number_of_oranges : ℕ := 18
def number_of_pears : ℕ := 27
def number_of_bananas : ℕ := 12

theorem greatest_number_of_fruit_baskets :
  Nat.gcd (Nat.gcd number_of_oranges number_of_pears) number_of_bananas = 3 :=
by
  sorry

end greatest_number_of_fruit_baskets_l259_259839


namespace no_grid_rectangle_tiling_with_congruent_grid_pentagons_l259_259014

theorem no_grid_rectangle_tiling_with_congruent_grid_pentagons :
  ¬ (∃ (R : Rectangle) (P : Pentagon), tiling R P) := 
sorry

end no_grid_rectangle_tiling_with_congruent_grid_pentagons_l259_259014


namespace marbles_selection_l259_259424

theorem marbles_selection : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ marbles : Finset ℕ, marbles.card = 15 ∧
  ∃ rgb : Finset ℕ, rgb ⊆ marbles ∧ rgb.card = 3 ∧
  ∃ yellow : ℕ, yellow ∈ marbles ∧ yellow ∉ rgb ∧ 
  ∀ (selection : Finset ℕ), selection.card = 5 →
  (∃ red green blue : ℕ, red ∈ rgb ∧ green ∈ rgb ∧ blue ∈ rgb ∧ 
  (red ∈ selection ∨ green ∈ selection ∨ blue ∈ selection) ∧ yellow ∉ selection) → 
  (selection.card = 5) :=
by
  sorry

end marbles_selection_l259_259424


namespace correct_average_height_l259_259488

theorem correct_average_height 
  (students : ℕ) 
  (incorrect_avg_height : ℝ) 
  (incorrect_height : ℝ) 
  (actual_height : ℝ) 
  (students = 20)
  (incorrect_avg_height = 175) 
  (incorrect_height = 151) 
  (actual_height = 136) : 
  (3485 / 20 = 174.25) :=
by
  sorry

end correct_average_height_l259_259488


namespace arithmetic_sequence_sum_l259_259239

theorem arithmetic_sequence_sum :
  ∃ a b : ℕ, ∀ d : ℕ,
    d = 5 →
    a = 28 →
    b = 33 →
    a + b = 61 :=
by
  sorry

end arithmetic_sequence_sum_l259_259239


namespace volume_of_rectangular_prism_l259_259187

-- Definition of the given conditions
variables (a b c : ℝ)

def condition1 : Prop := a * b = 24
def condition2 : Prop := b * c = 15
def condition3 : Prop := a * c = 10

-- The statement we want to prove
theorem volume_of_rectangular_prism
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c) :
  a * b * c = 60 :=
by sorry

end volume_of_rectangular_prism_l259_259187


namespace action_figure_ratio_l259_259618

variable (initial : ℕ) (sold : ℕ) (remaining : ℕ) (left : ℕ)
variable (h1 : initial = 24)
variable (h2 : sold = initial / 4)
variable (h3 : remaining = initial - sold)
variable (h4 : remaining - left = left)

theorem action_figure_ratio
  (h1 : initial = 24)
  (h2 : sold = initial / 4)
  (h3 : remaining = initial - sold)
  (h4 : remaining - left = left) :
  (remaining - left) * 3 = left :=
by
  sorry

end action_figure_ratio_l259_259618


namespace inequality_proof_l259_259701

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l259_259701


namespace fifth_equation_l259_259452

theorem fifth_equation :
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) := 
by sorry

end fifth_equation_l259_259452


namespace carwash_num_cars_l259_259806

variable (C : ℕ)

theorem carwash_num_cars 
    (h1 : 5 * 7 + 5 * 6 + C * 5 = 100)
    : C = 7 := 
by
    sorry

end carwash_num_cars_l259_259806


namespace geom_seq_conditions_geom_seq_general_formula_geom_seq_sum_2015_l259_259009

variables {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence such that q = 2 and a_2 + a_3 = 12
def geom_seq (a : α) (n : ℕ) : α := a * 2 ^ n

-- Condition: a_2 + a_3 = 12
theorem geom_seq_conditions (a : α) (h : geom_seq a 1 + geom_seq a 2 = 12) :
  geom_seq a 0 = 2 :=
by sorry

-- General formula: a_n = 2^n
theorem geom_seq_general_formula (a : α) (h : geom_seq a 1 + geom_seq a 2 = 12) :
  ∀ n : ℕ, geom_seq a n = 2 ^ n :=
by sorry

-- Sum of the first 2015 terms: S_2015 = 2^{2016} - 2
theorem geom_seq_sum_2015 (a : α) (h : geom_seq a 1 + geom_seq a 2 = 12) :
  let S : α := (finset.range 2015).sum (λ n, geom_seq a n) in
  S = 2 ^ 2016 - 2 :=
by sorry

end geom_seq_conditions_geom_seq_general_formula_geom_seq_sum_2015_l259_259009


namespace problem1_problem2_l259_259705

-- Define points O, A, B are not collinear, and the vector representation OP
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B P : V) (m n : ℝ)
hypothesis h1 : ¬Collinear ℝ ({O, A, B} : Set V)
hypothesis h2 : P = m • A + n • B

-- Problem 1: Prove collinearity of A, P, B given m + n = 1
theorem problem1 (h3 : m + n = 1) : Collinear ℝ ({A, P, B} : Set V) :=
sorry

-- Problem 2: Prove m + n = 1 given collinearity of A, P, B
theorem problem2 (h3 : Collinear ℝ ({A, P, B} : Set V)) : m + n = 1 :=
sorry

end problem1_problem2_l259_259705


namespace travel_time_is_five_hours_l259_259117

-- Define the conditions as constants
def total_distance : ℝ := 200
def fraction_driven_before_lunch : ℝ := 1 / 4
def time_driven_before_lunch : ℝ := 1
def lunch_time : ℝ := 1

-- Define the function to calculate total travel time
def total_travel_time : ℝ :=
  let distance_before_lunch := fraction_driven_before_lunch * total_distance
  let speed := distance_before_lunch / time_driven_before_lunch
  let remaining_distance := total_distance - distance_before_lunch
  let time_to_drive_remaining_distance := remaining_distance / speed
  let total_driving_time := time_driven_before_lunch + time_to_drive_remaining_distance
  total_driving_time + lunch_time

-- Prove that the total travel time equals 5 hours
theorem travel_time_is_five_hours : total_travel_time = 5 := by
  sorry

end travel_time_is_five_hours_l259_259117


namespace jill_clothing_tax_l259_259454

theorem jill_clothing_tax :
  ∀ (total_amount tax_clothing tax_other_items tax_total : ℝ), 
    total_amount > 0 →
    tax_clothing >= 0 →
    tax_clothing <= 100 → 
    tax_other_items = 0.3 * total_amount * 0.1 →
    tax_total = 0.05 * total_amount →
    (tax_clothing / 100) * (0.5 * total_amount) + tax_other_items = tax_total →
    tax_clothing = 4 :=
  
begin
  intros total_amount tax_clothing tax_other_items tax_total h1 h2 h3 h4 h5 h6,
  -- Proof skipped
  sorry,
end

end jill_clothing_tax_l259_259454


namespace ratio_of_pieces_is_one_to_one_l259_259057

-- Melanie starts with 2 slices of bread
def slices_of_bread := 2

-- She ends up with 8 pieces in total after tearing the slices
def total_pieces := 8

-- Assume x is the number of pieces per slice after the first tear
def pieces_per_slice_after_first_tear (x : ℕ) : Prop := 2 * x = total_pieces

-- Assume that the number of pieces after the second tear is the same as the first tear
def pieces_per_slice_after_second_tear := pieces_per_slice_after_first_tear

-- Prove the ratio of pieces after the first tear to pieces after the second tear is 1:1
theorem ratio_of_pieces_is_one_to_one (x : ℕ) :
  pieces_per_slice_after_first_tear x →
  pieces_per_slice_after_second_tear x →
  x = 4 → 
  1 = 1 :=
by
sory

end ratio_of_pieces_is_one_to_one_l259_259057


namespace diagonals_concurrent_200_gon_l259_259111

theorem diagonals_concurrent_200_gon (
  (A : ℕ → ℝ × ℝ)
  (red_sides extensions_of_red : NSidedPolygon) (blue_sides extensions_of_blue : NSidedPolygon) :
  (∀ i ∈ {1, 3, ..., 99}, ∃ P : ℝ × ℝ, ∀ {x y}, (x = A i → y = A (i + 100) → line_through x y intersects P)) →                                  -- condition: diagonals intersection at a point
  (is_convex_polygon 200 A) →                                                          -- condition: A is convex 200-gon
  (are_alternating_colored_sides 200 A red_sides blue_sides) →                          -- condition: sides are alternating colored red and blue
  (is_regular_polygon 100 (sides_to_polygon (red_extensions 200 A red_sides))) →       -- condition: red extensions form a regular 100-gon
  (is_regular_polygon 100 (sides_to_polygon (blue_extensions 200 A blue_sides))) →     -- condition: blue extensions form a regular 100-gon
  ∃ O : ℝ × ℝ, ∀ i ∈ {1, 3, ..., 99}, concurrent (A i) (A (i + 100)) O
:= sorry

end diagonals_concurrent_200_gon_l259_259111


namespace min_area_triangle_l259_259004

noncomputable def point (x y : ℝ) := (x, y)
noncomputable def vector (i j : ℝ) := (i, j)
noncomputable def e := vector 0 1 -- Vector e
noncomputable def A := point (1 / 2) 0 -- Point A

-- B moves on the line x = -1/2
def B (m : ℝ) := point (-1 / 2) m

-- C is the midpoint of AB
def C (B : ℝ → ℝ × ℝ) (m : ℝ) := point 0 (m / 2)

-- Definitions of vectors involved
def BM (B : ℝ → ℝ × ℝ) (M : ℝ × ℝ) (m : ℝ) := vector (M.1 + 1 / 2) (M.2 - m)
def CM (C : ℝ → ℝ × ℝ) (M : ℝ × ℝ) := vector M.1 (M.2 - C.2.2) -- C.2.2 is y-coord

-- Constraints
def BM_dot_e (M : ℝ × ℝ) := M.2
def CM_dot_AB (CM : ℝ → ℝ × ℝ → ℝ × ℝ) (m : ℝ) (M : ℝ × ℝ) := -M.1 + m * (M.2 - m / 2)

-- Trajectory E of M
def trajectory (M : ℝ × ℝ) : Prop := M.2 ^ 2 = 2 * M.1

-- Main theorem
theorem min_area_triangle (P R N : ℝ × ℝ) (E : ℝ → ℝ × ℝ → Prop) : 
  ∀ (x₀ y₀ : ℝ), 
  E (x₀, y₀) → 
  let P := (x₀, y₀), 
      R := (0, b),
      N := (0, c) in
      ( let area := ( (x₀ - 2) + (4 / (x₀ - 2)) + 4 ) / 2 in
      area ≥ 8 ) := 
sorry

end min_area_triangle_l259_259004


namespace range_of_a_l259_259759

theorem range_of_a (a : ℝ) 
  (hP : ∃ α, (3a - 9, a + 2) lies_on_terminal_side_of α) 
  (hcos : ∀ α, lies_on_terminal_side_of α → cos α ≤ 0)
  (hsin : ∀ α, lies_on_terminal_side_of α → sin α > 0) : 
  -2 < a ∧ a ≤ 3 := 
sorry

end range_of_a_l259_259759


namespace curve_C_eqn_line_l_eqn_max_distance_C_to_l_l259_259167

-- Definitions
def parametric_curve (θ : Real) := (√3 * Real.cos θ, Real.sin θ)
def polar_line (θ ρ : Real) := ρ * Real.sin (θ + Real.pi / 4) = 2 * √2

-- General equation of curve C
theorem curve_C_eqn : ∀ θ : Real, let (x, y) := parametric_curve θ in
  x^2 / 3 + y^2 = 1 := 
sorry

-- Cartesian equation of line l
theorem line_l_eqn : ∀ ρ θ : Real, polar_line θ ρ → 
  ∀ x y : Real, x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  x + y = 4 :=
sorry

-- Maximum distance from any point on curve C to line l
theorem max_distance_C_to_l : ∀ θ : Real, 
  let (x, y) := parametric_curve θ in 
  let d := |x + y - 4| / √2 in 
  d ≤ 3 * √2 :=
sorry

end curve_C_eqn_line_l_eqn_max_distance_C_to_l_l259_259167


namespace grazing_months_of_A_l259_259150

-- Definitions of conditions
def oxen_months_A (x : ℕ) := 10 * x
def oxen_months_B := 12 * 5
def oxen_months_C := 15 * 3
def total_rent := 140
def rent_C := 36

-- Assuming a is the number of months a put his oxen for grazing, we need to prove that a = 7
theorem grazing_months_of_A (a : ℕ) :
  (45 * 140 = 36 * (10 * a + 60 + 45)) → a = 7 := 
by
  intro h
  sorry

end grazing_months_of_A_l259_259150


namespace min_sum_of_edges_l259_259288

-- Define the vertices of the cube labeled from 1 to 8
def vertices := (Fin 8)

-- Define the edges based on vertex labels' absolute differences
def edge_len (i j : vertices) : ℕ := abs ((i : ℕ) - (j : ℕ))

-- Define the sum S of all edge values of the cube
def sum_of_edges (vertex_labels : Fin 8 → ℕ) : ℕ :=
  -- List all edges between vertices of a cube (12 edges)
  let edges := [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)];
  -- Sum the absolute differences on all edges
  edges.foldr (λ (e : Fin 8 × Fin 8) (acc : ℕ), acc + edge_len (vertex_labels e.1) (vertex_labels e.2)) 0

-- The theorem stating the minimum sum of these edges
theorem min_sum_of_edges : 
  ∃ vertex_labels : Fin 8 → ℕ, (∀ i, 1 ≤ vertex_labels i ∧ vertex_labels i ≤ 8) ∧ (∀ i j, i ≠ j → vertex_labels i ≠ vertex_labels j) ∧
  sum_of_edges vertex_labels = 28 :=
sorry

end min_sum_of_edges_l259_259288


namespace range_of_independent_variable_l259_259399

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259399


namespace fill_in_blank_correct_l259_259251

theorem fill_in_blank_correct : 
  (let passage_start := "As a great sage, Confucius had a profound understanding of society and life, and he also had his own understanding of fate and spirits."
      passage_end := "Only by achieving 'affection for one's own family' can one 'love others', and only on the basis of 'affection for one's own family' and by strengthening one's cultivation and empathy can one 'not only love one's own relatives but also others' children'."
      sentence1 := "Confucius advocated educating people through the rituals of worship to remember the merits of ancestors, hoping that these rites would transform people's hearts and make 'the morality of the people thick'."
      sentence2 := "The quality of social customs depends on the implementation of filial piety, and the cultivation of love should start with 'filial piety'."
      sentence3 := "Therefore, Confucius said, 'Establish love starting from one's own family'."
      sentence4 := "Confucius admired the rites of the Zhou dynasty, emphasizing 'being cautious at the end and remembering the distant past', valuing funeral and sacrificial rites."
      sentence5 := "Advocating 'burying with rites' and 'worshipping with rites', the focus should be on the order of the living world'."
  in (passage_start ++ "\n" ++ sentence1 ++ "\n" ++ sentence4 ++ "\n" ++ sentence5 ++ "\n" ++ sentence2 ++ "\n" ++ sentence3 ++ "\n" ++ passage_end) = 
     "As a great sage, Confucius had a profound understanding of society and life, and he also had his own understanding of fate and spirits. \
      \nConfucius advocated educating people through the rituals of worship to remember the merits of ancestors, hoping that these rites would transform people's hearts and make 'the morality of the people thick'. \
      \nConfucius admired the rites of the Zhou dynasty, emphasizing 'being cautious at the end and remembering the distant past', valuing funeral and sacrificial rites. \
      \nAdvocating 'burying with rites' and 'worshipping with rites', the focus should be on the order of the living world. \
      \nThe quality of social customs depends on the implementation of filial piety, and the cultivation of love should start with 'filial piety'. \
      \nTherefore, Confucius said, 'Establish love starting from one's own family'. \
      \nOnly by achieving 'affection for one's own family' can one 'love others', and only on the basis of 'affection for one's own family' and by strengthening one's cultivation and empathy can one 'not only love one's own relatives but also others' children'.") := 
  sorry

end fill_in_blank_correct_l259_259251


namespace find_roots_l259_259665

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_l259_259665


namespace find_locus_E_and_directed_line_l259_259226

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

def points_on_circle (a b : ℝ × ℝ) : Prop :=
(a = (-2, 0)) ∧ (b = (2, 0))

def locus_E (x y : ℝ) : Prop := 
x^2 - y^2 = 4

def directional_vector_l (vx vy : ℝ) : Prop := 
(vx = 1) ∧ (vy = 0)

theorem find_locus_E_and_directed_line (A B P : ℝ × ℝ) (x y : ℝ) :
  points_on_circle A B →
  circle_eq (x y) →
  ¬ (P = A) →
  (locus_E (P.1) (P.2)) ∧ 
  ∃ (C D : ℝ × ℝ) (vx vy : ℝ), 
  directional_vector_l vx vy ∧
  (C ≠ A) ∧ (D ≠ A) ∧ (vx = 1 ∧ vy = 0) ∧ 
  (ACy ≠ ADx) :=
by
  sorry

end find_locus_E_and_directed_line_l259_259226


namespace fence_building_cost_l259_259544

theorem fence_building_cost :
  ∀ (area : ℝ) (pricePerFoot : ℝ), area = 144 → pricePerFoot = 58 → 4 * (Real.sqrt area) * pricePerFoot = 2784 := by
  assume (area pricePerFoot : ℝ)
  assume h1 : area = 144
  assume h2 : pricePerFoot = 58
  sorry

end fence_building_cost_l259_259544


namespace trajectory_equation_max_area_triangle_OPQ_l259_259286

noncomputable def circle (x y r : ℝ) := x^2 + y^2 = r^2

theorem trajectory_equation (x y : ℝ) (r : ℝ) (A : r > 0) :
  tangent_to_line (circle x y r) (line (λ x y, x - √(3) * y + 4 = 0)) →
  ∀ (A B : ℝ), ∀ (A_on_M : circle x y r) (AB_perpendicular_x_axis : B = (A, 0)) (v_AB := A - B),
  moving_point_N (x y : ℝ) (satisfies : 2 * v_AB = - (N : ℝ, 2 - y)) →
  curve_eq {z : ℝ × ℝ | (z.1^2) / 4 + z.2^2 = 1} :=
  sorry

theorem max_area_triangle_OPQ (line1 : ℝ → ℝ → Prop) (C : set (ℝ × ℝ)) :
  (perpendicular (line1 (λ x y, x - √(3) * y + 4 = 0))) →
  (intersects line1 C) →
  ∀ (P Q : ℝ × ℝ) (P_on_C : P ∈ C) (Q_on_C : Q ∈ C),
  let l := P, O := (0, 0) in
  max_area (triangle O P Q) = 1 :=
  sorry

end trajectory_equation_max_area_triangle_OPQ_l259_259286


namespace set_intersection_example_l259_259331

theorem set_intersection_example : 
  let A := {1, 2}
  let B := {-1, 1, 4}
  A ∩ B = {1} :=
by
  sorry

end set_intersection_example_l259_259331


namespace max_area_PDE_l259_259862

theorem max_area_PDE {A B C D E P : Point} (hD : D ∈ Segment A B) (hE : E ∈ Segment A C)
  (hP : P = Line B E ∩ Line C D) (hAreaABC : area A B C = 1)
  (hCondition : area_quadrilateral B C E D = 2 * area P B C) :
  ∃ maxArea, maxArea = 10 * sqrt 2 - 14 ∧ area P D E ≤ maxArea :=
begin
  sorry
end

end max_area_PDE_l259_259862


namespace no_winning_strategy_for_either_player_l259_259973

structure Segment :=
(start : ℝ × ℝ)
(end : ℝ × ℝ)

-- Define the game conditions
def valid_segment (s : Segment) (prev_segments : List Segment) : Prop :=
  ∀ (p ∈ prev_segments), (s.start = p.end) ∧
  (¬∃ q ∈ prev_segments, (q ≠ s ∧ ∃ x, x ≠ s.start ∧ (x ∈ segment_points q ∧ x ∈ segment_points s)))

def next_segment_possible (prev_segments : List Segment) : Prop :=
  ∃ s : Segment, valid_segment s prev_segments

theorem no_winning_strategy_for_either_player :
  ∀ (game : List Segment), 
  ∃ (segment : Segment), valid_segment segment game -> next_segment_possible game := 
  sorry

end no_winning_strategy_for_either_player_l259_259973


namespace multiple_of_six_as_four_cubes_integer_as_five_cubes_l259_259156

-- Part (a)
theorem multiple_of_six_as_four_cubes (n : ℤ) : ∃ a b c d : ℤ, 6 * n = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 :=
by
  sorry

-- Part (b)
theorem integer_as_five_cubes (k : ℤ) : ∃ a b c d e : ℤ, k = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 + e ^ 3 :=
by
  have h := multiple_of_six_as_four_cubes
  sorry

end multiple_of_six_as_four_cubes_integer_as_five_cubes_l259_259156


namespace election_winner_votes_difference_l259_259982

theorem election_winner_votes_difference :
  ∃ W S T F, F = 199 ∧ W = S + 53 ∧ W = T + 79 ∧ W + S + T + F = 979 ∧ (W - F = 105) :=
by
  sorry

end election_winner_votes_difference_l259_259982


namespace primes_divisibility_l259_259022

theorem primes_divisibility
  (p1 p2 p3 p4 q1 q2 q3 q4 : ℕ)
  (hp1_lt_p2 : p1 < p2) (hp2_lt_p3 : p2 < p3) (hp3_lt_p4 : p3 < p4)
  (hq1_lt_q2 : q1 < q2) (hq2_lt_q3 : q2 < q3) (hq3_lt_q4 : q3 < q4)
  (hp4_minus_p1 : p4 - p1 = 8) (hq4_minus_q1 : q4 - q1 = 8)
  (hp1_gt_5 : 5 < p1) (hq1_gt_5 : 5 < q1) :
  30 ∣ (p1 - q1) :=
sorry

end primes_divisibility_l259_259022


namespace total_saltwater_animals_l259_259530

variable (numSaltwaterAquariums : Nat)
variable (animalsPerAquarium : Nat)

theorem total_saltwater_animals (h1 : numSaltwaterAquariums = 22) (h2 : animalsPerAquarium = 46) : 
    numSaltwaterAquariums * animalsPerAquarium = 1012 := 
  by
    sorry

end total_saltwater_animals_l259_259530


namespace ellipse_m_value_l259_259493

-- Definition of the problem conditions
def ellipse_equation (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

def foci_on_x_axis (a b : ℝ) : Prop :=
  a^2 >= b^2

def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Main theorem to prove the value of m
theorem ellipse_m_value (m : ℝ) :
  (ellipse_equation x y m) →
  (foci_on_x_axis a b) →
  (eccentricity a (Real.sqrt (a^2 - b^2)) = 1/2) →
  m = 3 :=
by 
  sorry

end ellipse_m_value_l259_259493


namespace find_ordered_pairs_l259_259252

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end find_ordered_pairs_l259_259252


namespace evolution_process_proof_l259_259495

namespace PopulationGrowth

-- Definitions for the population growth models
def Primitive (birth_rate death_rate : ℕ) : Prop :=
  birth_rate > death_rate

def Traditional (birth_rate death_rate : ℕ) : Prop :=
  birth_rate > death_rate ∧ death_rate < (max_possible_death_rate / 2)

def Modern (birth_rate death_rate : ℕ) : Prop :=
  birth_rate < death_rate ∧ birth_rate < (min_possible_birth_rate / 2)

-- Hypotheses reflecting the conditions mentioned in the task.
variable (birth_rate death_rate max_possible_death_rate min_possible_birth_rate : ℕ)
variable (early_prod_forces low_intermediate_prod_forces high_prod_forces : Prop)

axiom primitive_initial : early_prod_forces → Primitive birth_rate death_rate
axiom traditional_evolution : low_intermediate_prod_forces → Traditional birth_rate death_rate
axiom modern_evolution : high_prod_forces → Modern birth_rate death_rate

-- The theorem reflecting the correct answer to the question
theorem evolution_process_proof :
  early_prod_forces → low_intermediate_prod_forces → high_prod_forces →
  Primitive birth_rate death_rate ∧ Traditional birth_rate death_rate ∧ Modern birth_rate death_rate :=
by 
  intros
  sorry

end PopulationGrowth

end evolution_process_proof_l259_259495


namespace no_polyhedron_with_area_ratio_ge_two_l259_259421

theorem no_polyhedron_with_area_ratio_ge_two (n : ℕ) (areas : Fin n → ℝ)
  (h : ∀ (i j : Fin n), i < j → (areas j) / (areas i) ≥ 2) : False := by
  sorry

end no_polyhedron_with_area_ratio_ge_two_l259_259421


namespace infinite_shy_tuples_sum_of_squares_2016_l259_259923

-- Part (a) proof statement: 
-- There exist infinitely many shy tuples satisfying the given conditions
theorem infinite_shy_tuples :
  ∃ᶠ (a1 a2 a3 a4 b1 b2 b3: ℕ),
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ b1 ∧ a1 ≠ b2 ∧ a1 ≠ b3 ∧
     a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ b1 ∧ a2 ≠ b2 ∧ a2 ≠ b3 ∧
     a3 ≠ a4 ∧ a3 ≠ b1 ∧ a3 ≠ b2 ∧ a3 ≠ b3 ∧
     a4 ≠ b1 ∧ a4 ≠ b2 ∧ a4 ≠ b3 ∧
     b1 ≠ b2 ∧ b1 ≠ b3 ∧
     b2 ≠ b3) ∧
    gcd (gcd (gcd (gcd (gcd (gcd a1 a2) a3) a4) b1) b2) b3 = 1 ∧
    a1^2 + a2^2 + a3^2 + a4^2 = b1^2 + b2^2 + b3^2 ∧ 
    ∀ (i j k: ℕ) (h₁ : 1 ≤ i ∧ i < j ∧ j ≤ 4) (h₂ : 1 ≤ k ∧ k ≤ 3),
      ( [a1, a2, a3, a4].nth (i - 1) ).get_or_else 0 ^ 2 + 
      ( [a1, a2, a3, a4].nth (j - 1) ).get_or_else 0 ^ 2 ≠ 
      ( [b1, b2, b3].nth (k - 1) ).get_or_else 0 ^ 2 :=
sorry

-- Part (b) proof statement:
-- There exist distinct natural numbers a, b, c, and d such that their sum of squares equals 2016
theorem sum_of_squares_2016 :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a^2 + b^2 + c^2 + d^2 = 2016 :=
sorry

end infinite_shy_tuples_sum_of_squares_2016_l259_259923


namespace tiling_possible_with_one_type_l259_259048

theorem tiling_possible_with_one_type
  {a b m n : ℕ} (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (H : (∃ (k : ℕ), a = k * n) ∨ (∃ (l : ℕ), b = l * m)) :
  (∃ (i : ℕ), a = i * n) ∨ (∃ (j : ℕ), b = j * m) :=
  sorry

end tiling_possible_with_one_type_l259_259048


namespace final_i_is_11_l259_259651

noncomputable def program_final_i : ℕ :=
  let S_initial := 1
  let i_initial := 3
  let rec loop (S : ℕ) (i : ℕ) : ℕ :=
    if S <= 200 then loop (S * i) (i + 2) else i
  loop S_initial i_initial

theorem final_i_is_11 :
  program_final_i = 11 :=
by
  sorry

end final_i_is_11_l259_259651


namespace pure_gold_to_add_eq_46_67_l259_259208

-- Define the given conditions
variable (initial_alloy_weight : ℝ) (initial_gold_percentage : ℝ) (final_gold_percentage : ℝ)
variable (added_pure_gold : ℝ)

-- State the proof problem
theorem pure_gold_to_add_eq_46_67 :
  initial_alloy_weight = 20 ∧
  initial_gold_percentage = 0.50 ∧
  final_gold_percentage = 0.85 ∧
  (10 + added_pure_gold) / (20 + added_pure_gold) = 0.85 →
  added_pure_gold = 46.67 :=
by
  sorry

end pure_gold_to_add_eq_46_67_l259_259208


namespace proof_distance_between_A_B_l259_259577

noncomputable def distance_between_A_B (vA vB x y d vA' : ℕ) :=
    -- defining the initial conditions
    let condition1 := vA ≠ 0
    let condition2 := vB ≠ 0
    let condition3 := x ≡ 900 -- distance CD
    let condition4 := y ≡ 720 -- distance B is from A after returning
    
    have hCA : vA * (vA / (vA + vB)) = x / (vA - vA * 0.8),
    from by sorry,
    
    have hCB : vB * (vB / (vA + vB)) = y / vB,
    from by sorry,

    have hAB : d,
    from by sorry,  

    -- final distance
    d = 5265

theorem proof_distance_between_A_B :
    ∀ (vA vB x y d: ℕ),
        -- conditions
        vA ≠ 0 → vB ≠ 0 →
        x = 900 →
        y = 720 →
        
        -- prove distance between A and B
        distance_between_A_B vA vB x y d (0.8 * vA) = 5265 :=
sorry

end proof_distance_between_A_B_l259_259577


namespace zero_in_interval_l259_259279

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 4

-- State the proof problem
theorem zero_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 :=
by
  sorry

end zero_in_interval_l259_259279


namespace arrangement_count_l259_259072

-- Define the sets of books
def italian_books : Finset String := { "I1", "I2", "I3" }
def german_books : Finset String := { "G1", "G2", "G3" }
def french_books : Finset String := { "F1", "F2", "F3", "F4", "F5" }

-- Define the arrangement count as a noncomputable definition, because we are going to use factorial which involves an infinite structure
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Prove the required arrangement
theorem arrangement_count : 
  (factorial 3) * ((factorial 3) * (factorial 3) * (factorial 5)) = 25920 := 
by
  -- Provide the solution steps here (omitted for now)
  sorry

end arrangement_count_l259_259072


namespace job_completion_time_l259_259943

theorem job_completion_time :
  ∃ (x : ℝ), (∀ (B : ℝ), B = 20 → (∀ (A_and_B_work_together : ℝ), A_and_B_work_together = 8 → (∀ (fraction_left : ℝ), fraction_left = 0.06666666666666665 → 
  (8 * (1 / x + 1 / B) = 1 - fraction_left) → x = 15))) :=
begin
  sorry
end

end job_completion_time_l259_259943


namespace expected_remaining_bullets_correct_l259_259603

noncomputable def probability_of_hit : ℝ := 0.6
noncomputable def probability_of_miss : ℝ := 1 - probability_of_hit
noncomputable def number_of_bullets : ℕ := 4

def probability_xi_0 : ℝ := probability_of_miss ^ 3
def probability_xi_1 : ℝ := probability_of_hit * (probability_of_miss ^ 2)
def probability_xi_2 : ℝ := probability_of_hit * probability_of_miss
def probability_xi_3 : ℝ := probability_of_hit

noncomputable def expected_remaining_bullets : ℝ :=
  0 * probability_xi_0 +
  1 * probability_xi_1 +
  2 * probability_xi_2 +
  3 * probability_xi_3

theorem expected_remaining_bullets_correct :
  expected_remaining_bullets = 2.376 :=
  by
    sorry

end expected_remaining_bullets_correct_l259_259603


namespace choose_9_4_l259_259425

theorem choose_9_4 : (Nat.choose 9 4) = 126 :=
  by
  sorry

end choose_9_4_l259_259425


namespace parallel_dot_product_magnitude_difference_l259_259338

-- Declare vectors a and b
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)

-- Define dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Proof problem: If vector_a is parallel to vector_b, then the dot product should be -5.
theorem parallel_dot_product : 
  (∀ m, (1 : ℝ) / (-1 : ℝ) = ((-2 : ℝ) / m) → dot_product vector_a (vector_b m) = -5) :=
  sorry

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Proof problem: If m = 1, the magnitude of vector_a - vector_b should be sqrt 13
theorem magnitude_difference :
  (m = 1 → magnitude (1 + 1, -2 - 1) = Real.sqrt 13) := 
  sorry

end parallel_dot_product_magnitude_difference_l259_259338


namespace max_term_value_l259_259447

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- Conditions as definitions
def condition1 : Prop := S_n 2015 > 0
def condition2 : Prop := S_n 2016 < 0

-- Theorem statement
theorem max_term_value (h1 : condition1) (h2 : condition2) : ∃ n : ℕ, n = 1008 ∧ (∀ m : ℕ, m = 1008 ∨ (S_n m / a_n m) ≤ (S_n 1008 / a_n 1008)) :=
by
  sorry

end max_term_value_l259_259447


namespace vityas_miscalculation_l259_259130

/-- Vitya's miscalculated percentages problem -/
theorem vityas_miscalculation :
  ∀ (N : ℕ)
  (acute obtuse nonexistent right depends_geometry : ℕ)
  (H_acute : acute = 5)
  (H_obtuse : obtuse = 5)
  (H_nonexistent : nonexistent = 5)
  (H_right : right = 50)
  (H_total : acute + obtuse + nonexistent + right + depends_geometry = 100),
  depends_geometry = 110 :=
by
  intros
  sorry

end vityas_miscalculation_l259_259130


namespace least_m_n_diff_l259_259021

theorem least_m_n_diff 
  (n m : ℤ)
  (h1 : n ≤ 2007)
  (h2 : 2007 ≤ m)
  (hn : n^n ≡ 4 [MOD 5])
  (hm : m^m ≡ 4 [MOD 5])
  : (m - n) = 7 := 
sorry

end least_m_n_diff_l259_259021


namespace men_joined_l259_259572

noncomputable def number_of_additional_men (M : ℕ) (D D' : ℚ) : ℚ :=
  (M * D - M * D') / D'

theorem men_joined :
  let M := 1000
  let D := 20
  let D' := 16.67
  let X := 200
  number_of_additional_men M D D' ≈ X :=
by
  sorry

end men_joined_l259_259572


namespace triangle_area_MNI_l259_259930

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem triangle_area_MNI :
  let A := (0, 0 : ℝ)
  let B := (10, 0)
  let C := (24, 0)
  let M := midpoint B C
  let N := midpoint A B
  let I := (6.67, 0) -- approximation to the coordinates found in the solution
  dist A B = 10 →
  dist A C = 24 →
  dist B C = 26 →
  (M = (13, 0)) →
  (N = (5, 0)) →
  (I = (6.67, 0)) →
  let area := 1 / 2 * |(M.1 * (N.2 - I.2) + N.1 * (I.2 - M.2) + I.1 * (M.2 - N.2))| in
  area = 30 :=
by
  sorry

end triangle_area_MNI_l259_259930


namespace largest_of_sqrt_cbrt_expressions_l259_259549

noncomputable def sqrt_cube_root_56 : ℝ := Real.sqrt (Real.cbrt 56)
noncomputable def sqrt_cube_root_3584 : ℝ := Real.sqrt (Real.cbrt 3584)
noncomputable def sqrt_cube_root_2744 : ℝ := Real.sqrt (Real.cbrt 2744)
noncomputable def sqrt_cube_root_392 : ℝ := Real.sqrt (Real.cbrt 392)
noncomputable def sqrt_cube_root_448 : ℝ := Real.sqrt (Real.cbrt 448)

theorem largest_of_sqrt_cbrt_expressions :
  max (max (max (max sqrt_cube_root_56 sqrt_cube_root_3584) sqrt_cube_root_2744) sqrt_cube_root_392) sqrt_cube_root_448 = sqrt_cube_root_3584 :=
  by simp [Real.sqrt, Real.cbrt]; sorry

end largest_of_sqrt_cbrt_expressions_l259_259549


namespace trajectory_eq_l259_259299

noncomputable def point_trajectory (A : ℝ × ℝ) (line : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (R : ℝ × ℝ), 
  (line R.1 = R.2) ∧             -- R lies on the line
  (R, A) = (A, P)                -- SA = AP

theorem trajectory_eq (A : ℝ × ℝ) (line : ℝ → ℝ) (P : ℝ × ℝ) :
  A = (1, 0) → 
  (∀ x, line x = 2*x - 4) → 
  point_trajectory (1, 0) (λ x, 2*x - 4) (P) →
  P.2 = 2*P.1 :=
sorry

end trajectory_eq_l259_259299


namespace ellipse_eccentricity_range_l259_259103

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b > 0) 
  (c := Real.sqrt (a^2 - b^2)) 
  (h_range : ∀ (P : ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
    2 * c^2 ≤ (Real.sqrt ((P.1 - c)^2 + P.2^2)) * (Real.sqrt ((P.1 + c)^2 + P.2^2)) ∧ 
    (Real.sqrt ((P.1 - c)^2 + P.2^2)) * (Real.sqrt ((P.1 + c)^2 + P.2^2)) ≤ 3 * c^2) :
  (Real.sqrt 3 / 3) ≤ (Real.sqrt (a^2 - b^2) / a) ∧ 
  (Real.sqrt (a^2 - b^2) / a) ≤ (Real.sqrt 2 / 2) :=
sorry

end ellipse_eccentricity_range_l259_259103


namespace cone_generatrix_is_6_l259_259484

noncomputable def cone_generatrix_length (r l : ℝ) (h : ℝ) : Prop :=
  (π * r * l = 2 * π * r^2) ∧
  (1 / 3 * π * r^2 * h = 9 * √3 * π) ∧
  (h = √(l^2 - r^2))

theorem cone_generatrix_is_6 (r l : ℝ) (h : ℝ) (h_cond : cone_generatrix_length r l h) :
  l = 6 :=
by
  sorry

end cone_generatrix_is_6_l259_259484


namespace determine_liars_and_knights_l259_259365

variable {k b g d : ℕ}

-- Each knight gives one affirmative answer to four questions
def knight_affirmatives (k : ℕ) : ℕ := 4 * k

-- Each liar gives three affirmative answers to four questions
def liar_affirmatives (l : ℕ) : ℕ := 3 * l

-- Total number of affirmative answers is 500
def total_affirmatives (k l : ℕ) : Prop := knight_affirmatives k + liar_affirmatives l = 500

-- If all residents were knights, total affirmative answers would be 200
def total_knights_affirmatives (k : ℕ) : Prop := knight_affirmatives k = 200

-- Additional 300 "yes" answers are due to lies
def extra_affirmatives (l : ℕ) : Prop := liar_affirmatives l = 300

-- Number of liars in district A is 55
def liars_in_A (k : ℕ) : ℕ := 55

-- Number of liars in district B exceeds knights by 35
def liars_in_B (b : ℕ) : ℕ := b + 35

-- Number of liars in district G exceeds knights by 17
def liars_in_G (g : ℕ) : ℕ := g + 17

-- Number of liars in district D is lesser than knights
def liars_in_D (d : ℕ) : ℕ := d - 1

theorem determine_liars_and_knights (k b g d : ℕ) :
  total_affirmatives (k + b + g + d) (liars_in_A k + liars_in_B b + liars_in_G g + liars_in_D d) → 
  total_knights_affirmatives (k + b + g + d) → 
  extra_affirmatives (liars_in_A k + liars_in_B b + liars_in_G g + liars_in_D d) → 
  liars_in_A k = 55 ∧ liars_in_B b = b + 35 ∧ liars_in_G g = g + 17 ∧ liars_in_D d < d := 
begin
  sorry
end

end determine_liars_and_knights_l259_259365


namespace fraction_is_three_eighths_l259_259574

theorem fraction_is_three_eighths (F N : ℝ) 
  (h1 : (4 / 5) * F * N = 24) 
  (h2 : (250 / 100) * N = 199.99999999999997) : 
  F = 3 / 8 :=
by 
  sorry

end fraction_is_three_eighths_l259_259574


namespace exceeding_fraction_l259_259236

def repeatingDecimal : ℚ := 8 / 33
def decimalFraction : ℚ := 6 / 25
def difference : ℚ := repeatingDecimal - decimalFraction

theorem exceeding_fraction :
  difference = 2 / 825 := by
  sorry

end exceeding_fraction_l259_259236


namespace number_of_solid_shapes_is_three_l259_259615

-- Define the geometric shapes and their dimensionality
inductive GeomShape
| square : GeomShape
| cuboid : GeomShape
| circle : GeomShape
| sphere : GeomShape
| cone : GeomShape

def isSolid (shape : GeomShape) : Bool :=
  match shape with
  | GeomShape.square => false
  | GeomShape.cuboid => true
  | GeomShape.circle => false
  | GeomShape.sphere => true
  | GeomShape.cone => true

-- Formal statement of the problem
theorem number_of_solid_shapes_is_three :
  (List.filter isSolid [GeomShape.square, GeomShape.cuboid, GeomShape.circle, GeomShape.sphere, GeomShape.cone]).length = 3 :=
by
  -- proof omitted
  sorry

end number_of_solid_shapes_is_three_l259_259615


namespace combine_octahedrons_tetrahedrons_to_larger_octahedron_l259_259465

theorem combine_octahedrons_tetrahedrons_to_larger_octahedron (edge : ℝ) :
  ∃ (octahedrons : ℕ) (tetrahedrons : ℕ),
    octahedrons = 6 ∧ tetrahedrons = 8 ∧
    (∃ (new_octahedron_edge : ℝ), new_octahedron_edge = 2 * edge) :=
by {
  -- The proof will construct the larger octahedron
  sorry
}

end combine_octahedrons_tetrahedrons_to_larger_octahedron_l259_259465


namespace find_initial_men_l259_259832

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end find_initial_men_l259_259832


namespace price_of_each_sundae_l259_259945

theorem price_of_each_sundae (A B : ℝ) (x y z : ℝ) (hx : 200 * x = 80) (hy : A = y) (hz : y = 0.40)
  (hxy : A - 80 = z) (hyz : 200 * z = B) : y = 0.60 :=
by
  sorry

end price_of_each_sundae_l259_259945


namespace find_digit_A_l259_259891

-- Define the six-digit number for any digit A
def six_digit_number (A : ℕ) : ℕ := 103200 + A * 10 + 4
-- Define the condition that a number is prime
def is_prime (n : ℕ) : Prop := (2 ≤ n) ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → ¬ (m ∣ n)

-- The main theorem stating that A must equal 1 for the number to be prime
theorem find_digit_A (A : ℕ) : A = 1 ↔ is_prime (six_digit_number A) :=
by
  sorry -- Proof to be filled in


end find_digit_A_l259_259891


namespace fraction_simplification_l259_259553

theorem fraction_simplification : 
  ((2 * 7) * (6 * 14)) / ((14 * 6) * (2 * 7)) = 1 :=
by
  sorry

end fraction_simplification_l259_259553


namespace max_distance_origin_l259_259886

noncomputable def max_distance (z : ℂ) (hz : Complex.abs z = 1) : ℝ :=
  let w := 3 * Complex.conj z - (1 - Complex.I) * z
  Complex.abs w

theorem max_distance_origin (z : ℂ) (hz : Complex.abs z = 1) (h_not_collinear : ¬ (3 * Complex.conj z - (1 - Complex.I) * z = z ∨ 3 * Complex.conj z - (1 - Complex.I) * z = (1 - Complex.I) * z ∨ z = (1 - Complex.I) * z)):
  max_distance z hz = 4 :=
sorry

end max_distance_origin_l259_259886


namespace minimum_S_l259_259285

noncomputable def min_value (a : ℕ → ℕ) : ℕ :=
  a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + 
  a 5 * a 6 + a 6 * a 7 + a 7 * a 8 + a 8 * a 9 + 
  a 9 * a 10 + a 10 * a 1

theorem minimum_S :
  ∃ a : ℕ → ℕ, 
  (∀ i j : ℕ, i ≠ j → a i ≠ a j) ∧ (∑ i in finset.range 10, a (i + 1) = 1995) ∧ min_value a = 6044 :=
sorry

end minimum_S_l259_259285


namespace range_of_independent_variable_l259_259394

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l259_259394


namespace trapezoid_area_l259_259608

theorem trapezoid_area (h : ℝ) : 
  let base1 := 3 * h 
  let base2 := 4 * h 
  let average_base := (base1 + base2) / 2 
  let area := average_base * h 
  area = (7 * h^2) / 2 := 
by
  sorry

end trapezoid_area_l259_259608


namespace angles_terminal_side_equiv_l259_259669

theorem angles_terminal_side_equiv (k : ℤ) : (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi + Real.pi) % (2 * Real.pi) ∨ (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi - Real.pi) % (2 * Real.pi) :=
sorry

end angles_terminal_side_equiv_l259_259669


namespace find_values_of_a_and_c_l259_259268

theorem find_values_of_a_and_c
  (a c : ℤ)
  (h: ∀ x, (x^2 - x - 1).is_factor_of (ax^19 + cx^18 + 1)) :
  a = 1597 ∧ c = -2584 := sorry

end find_values_of_a_and_c_l259_259268


namespace polar_to_rect_l259_259642

open Real 

theorem polar_to_rect (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 3 * π / 4) : 
  (r * cos θ, r * sin θ) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) :=
by
  -- Optional step: you can introduce the variables as they have already been proved using the given conditions
  have hr : r = 3 := h_r
  have hθ : θ = 3 * π / 4 := h_θ
  -- Goal changes according to the values of r and θ derived from the conditions
  sorry

end polar_to_rect_l259_259642


namespace line_parallel_slope_l259_259742

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope_l259_259742


namespace partition_exists_l259_259440

noncomputable def partition_nats (c : ℚ) (hc : c ≠ 1) : Prop :=
  ∃ (A B : Finset ℕ), (A ∩ B = ∅) ∧ (A ∪ B = Finset.univ) ∧ ∀ m n ∈ A, m ≠ n → m / n ≠ c ∧ ∀ m n ∈ B, m ≠ n → m / n ≠ c

theorem partition_exists (c : ℚ) (hc : c ≠ 1) : partition_nats c hc :=
  sorry

end partition_exists_l259_259440


namespace range_of_x_of_sqrt_x_plus_3_l259_259390

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259390


namespace intersection_eq_0_l259_259025

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_eq_0 : M ∩ N = {0} := by
  sorry

end intersection_eq_0_l259_259025


namespace sqrt_five_times_sum_l259_259543

theorem sqrt_five_times_sum : sqrt (5 * (4^3 + 4^3 + 4^3 + 4^3)) = 8 * sqrt 5 := 
by sorry

end sqrt_five_times_sum_l259_259543


namespace find_distance_l259_259605

variables (x v : ℝ)
-- Conditions provided in the problem
axiom cond1 : ∀ (x v : ℝ), x / v + 3.5 = 2 + 1 + (x - 2 * v) / (0.8 * v)
axiom cond2 : ∀ (x v : ℝ), x / v + 1.5 = (2 * v + 180) / v + 1 + (x - 2 * v - 180) / (0.8 * v)

-- Define the goal to prove that the distance AB is 270 km
theorem find_distance (x : ℝ) (v : ℝ) (h1 : cond1 x v) (h2 : cond2 x v) : x = 270 :=
sorry

end find_distance_l259_259605


namespace at_least_one_inequality_holds_l259_259467

theorem at_least_one_inequality_holds
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l259_259467


namespace length_of_floor_is_20_meters_l259_259878

-- Define the problem conditions
def length_more_than_breadth_by_200_percent (b : ℝ) : ℝ := b + 2 * b
def cost_to_paint (area : ℝ) (cost_per_sq_m : ℝ) : ℝ := area * cost_per_sq_m

theorem length_of_floor_is_20_meters (b : ℝ) (cost_per_sq_m : ℝ) (total_cost : ℝ) (l : ℝ) 
  (h1 : l = length_more_than_breadth_by_200_percent b)
  (h2 : cost_to_ppaint (3 * b^2) cost_per_sq_m = total_cost)
  (h3 : total_cost = 400) (h4 : cost_per_sq_m = 3) :
  l = 20 :=
by
  sorry

end length_of_floor_is_20_meters_l259_259878


namespace initial_coffee_stock_is_400_l259_259951

theorem initial_coffee_stock_is_400
  (x : ℕ)
  (h1 : 0.20 * x + 60 = 0.28000000000000004 * (x + 100)) :
  x = 400 :=
by
  sorry

end initial_coffee_stock_is_400_l259_259951


namespace direction_vector_arithmetic_sequence_l259_259929

theorem direction_vector_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
    (n : ℕ) 
    (S2_eq_10 : S_n 2 = 10) 
    (S5_eq_55 : S_n 5 = 55)
    (arith_seq_sum : ∀ n, S_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))) / 2): 
    (a_n (n + 2) - a_n n) / (n + 2 - n) = 4 :=
by
  sorry

end direction_vector_arithmetic_sequence_l259_259929


namespace simplify_expression_solve_fractional_eq_l259_259158

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l259_259158


namespace x_cube_plus_y_cube_l259_259755

theorem x_cube_plus_y_cube (x y : ℝ) (h₁ : x + y = 1) (h₂ : x^2 + y^2 = 3) : x^3 + y^3 = 4 :=
sorry

end x_cube_plus_y_cube_l259_259755


namespace cauchy_inequality_special_case_l259_259011

theorem cauchy_inequality_special_case (a b c d : ℝ)
  (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) :
  (ac + bd)^2 ≤ (a^2 + b^2)(c^2 + d^2) ∧ ((ac + bd)^2 = (a^2 + b^2)(c^2 + d^2) ↔ ad = bc) :=
sorry

end cauchy_inequality_special_case_l259_259011


namespace total_wheels_in_garage_l259_259514

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end total_wheels_in_garage_l259_259514


namespace minute_hour_hands_opposite_l259_259627

theorem minute_hour_hands_opposite (x : ℝ) (h1 : 10 * 60 ≤ x) (h2 : x ≤ 11 * 60) : 
  (5.5 * x = 442.5) :=
sorry

end minute_hour_hands_opposite_l259_259627


namespace third_smallest_four_digit_in_pascals_triangle_l259_259541

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end third_smallest_four_digit_in_pascals_triangle_l259_259541


namespace pipe_R_fill_time_l259_259064

theorem pipe_R_fill_time (P_rate Q_rate combined_rate : ℝ) (hP : P_rate = 1 / 2) (hQ : Q_rate = 1 / 4)
  (h_combined : combined_rate = 1 / 1.2) : (∃ R_rate : ℝ, R_rate = 1 / 12) :=
by
  sorry

end pipe_R_fill_time_l259_259064


namespace ken_pencils_kept_l259_259426

-- Define the known quantities and conditions
def initial_pencils : ℕ := 250
def manny_pencils : ℕ := 25
def nilo_pencils : ℕ := manny_pencils * 2
def carlos_pencils : ℕ := nilo_pencils / 2
def tina_pencils : ℕ := carlos_pencils + 10
def rina_pencils : ℕ := tina_pencils - 20

-- Formulate the total pencils given away
def total_given_away : ℕ :=
  manny_pencils + nilo_pencils + carlos_pencils + tina_pencils + rina_pencils

-- Prove the final number of pencils Ken kept.
theorem ken_pencils_kept : initial_pencils - total_given_away = 100 :=
by
  sorry

end ken_pencils_kept_l259_259426


namespace trig_log_exp_identity_l259_259165

theorem trig_log_exp_identity : 
  (Real.sin (330 * Real.pi / 180) + 
   (Real.sqrt 2 - 1)^0 + 
   3^(Real.log 2 / Real.log 3)) = 5 / 2 :=
by
  -- Proof omitted
  sorry

end trig_log_exp_identity_l259_259165


namespace black_pork_zongzi_price_reduction_l259_259242

def price_reduction_15_dollars (initial_profit initial_boxes extra_boxes_per_dollar x : ℕ) : Prop :=
  initial_profit > x ∧ (initial_profit - x) * (initial_boxes + extra_boxes_per_dollar * x) = 2800 -> x = 15

-- Applying the problem conditions explicitly and stating the proposition to prove
theorem black_pork_zongzi_price_reduction:
  price_reduction_15_dollars 50 50 2 15 :=
by
  -- Here we state the question as a proposition based on the identified conditions and correct answer
  sorry

end black_pork_zongzi_price_reduction_l259_259242


namespace mixed_oil_rate_l259_259924

theorem mixed_oil_rate (oil1_vol : ℕ) (oil1_price : ℕ) (oil2_vol : ℕ) (oil2_price : ℕ) :
  oil1_vol = 10 → oil1_price = 40 → oil2_vol = 5 → oil2_price = 66 →
  (oil1_vol * oil1_price + oil2_vol * oil2_price) / (oil1_vol + oil2_vol) = 48.67 := 
by 
  intros h1 h2 h3 h4
  sorry

end mixed_oil_rate_l259_259924


namespace least_six_digit_cong_3_mod_17_l259_259135

theorem least_six_digit_cong_3_mod_17 :
  ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ x % 17 = 3 ∧ x = 100004 :=
by
  sorry

end least_six_digit_cong_3_mod_17_l259_259135


namespace sum_first_n_terms_l259_259293

-- Definitions from the problem conditions
def a₁ : ℕ := 1
def d : ℕ := 4
def a_n (n : ℕ) : ℕ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℕ := (n * (a₁ + a_n n)) / 2

-- Definitions for b_n
def b_n (n k : ℕ) : ℚ := (S_n n) / (n + k)

-- Hypothesis that b_n forms an arithmetic sequence
axiom b_n_arith_seq (k : ℚ) (Hk : k ≠ 0) : ∀ n : ℕ, 2 * b_n 2 k = b_n 1 k + b_n 3 k

-- The sequence {1/(b_n * b_{n+1})}
def seq (n k : ℕ) : ℚ := 1 / (b_n n k * b_n (n + 1) k)

-- Sum of the first n terms of the sequence {1/(b_n * b_{n+1})}
def T_n (n k : ℕ) : ℚ := ∑ i in Finset.range n, seq i k

-- Proof statement
theorem sum_first_n_terms (k : ℚ) (Hk : k ≠ 0) :
  (k = -1/2 → T_n n k = n / (4 * (n + 1))) ∧ (k = 0 → T_n n k = n / (2 * n + 1)) := by
  sorry

end sum_first_n_terms_l259_259293


namespace initial_hair_length_l259_259470

-- Definitions based on the conditions
def hair_cut_off : ℕ := 13
def current_hair_length : ℕ := 1

-- The problem statement to be proved
theorem initial_hair_length : (current_hair_length + hair_cut_off = 14) :=
by
  sorry

end initial_hair_length_l259_259470


namespace pattern_count_4x4_square_l259_259120

theorem pattern_count_4x4_square :
  let total_patterns := (Nat.choose 16 4) * (Nat.choose 12 4) * (Nat.choose 8 8) in
  total_patterns = 900900 :=
by
  sorry

end pattern_count_4x4_square_l259_259120


namespace incorrect_divisor_l259_259770

theorem incorrect_divisor (D x : ℕ) (h1 : D = 24 * x) (h2 : D = 48 * 36) : x = 72 := by
  sorry

end incorrect_divisor_l259_259770


namespace bagel_spending_l259_259671

theorem bagel_spending (B D : ℝ) (h1 : D = 0.5 * B) (h2 : B = D + 15) : B + D = 45 := by
  sorry

end bagel_spending_l259_259671


namespace rose_clothing_tax_l259_259455

theorem rose_clothing_tax {total_spent total_tax tax_other tax_clothing amount_clothing amount_food amount_other clothing_tax_rate : ℝ} 
  (h_total_spent : total_spent = 100)
  (h_amount_clothing : amount_clothing = 0.5 * total_spent)
  (h_amount_food : amount_food = 0.2 * total_spent)
  (h_amount_other : amount_other = 0.3 * total_spent)
  (h_no_tax_food : True)
  (h_tax_other_rate : tax_other = 0.08 * amount_other)
  (h_total_tax_rate : total_tax = 0.044 * total_spent)
  (h_calculate_tax_clothing : tax_clothing = total_tax - tax_other) :
  clothing_tax_rate = (tax_clothing / amount_clothing) * 100 → 
  clothing_tax_rate = 4 := 
by
  sorry

end rose_clothing_tax_l259_259455


namespace geometric_sequence_inequality_b_n_l259_259431

variable {n : ℕ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Assume the sequence a_n and its sum S_n satisfy the given conditions
axiom sum_condition (n : ℕ) (a1 : ℝ) : 2 * S n + a1 = 3 * a n
axiom non_zero_a1 (a1 : ℝ) : a1 ≠ 0
axiom arithmetic_condition (a1 : ℝ) (a2 a3 : ℝ) : 2 * (2 * a2 + 1) = 4 * a1 - 1 + a3

-- Define the sequence b_n
def b_n (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * Real.log 3 (a n) - 1

-- Proof statement for part (1)
theorem geometric_sequence (a1 : ℝ) : 
  (∀ n, 2 * S n + a1 = 3 * a n) →
  a1 ≠ 0 →
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n := 
sorry

-- Proof statement for part (2)
theorem inequality_b_n (a1 a2 a3 : ℝ) : 
  (∀ n, 2 * S n + a1 = 3 * a n) →
  a1 ≠ 0 →
  (2 * (2 * a2 + 1) = 4 * a1 - 1 + a3) →  
  (∀ n, let b := b_n n a in 
         (1 / (b 1 * b 2) + 1 / (b 2 * b 3) + ∀ k < n, 1 / (b k * b (k + 1))) < 1 / 2) := 
sorry

end geometric_sequence_inequality_b_n_l259_259431


namespace fg_neg_sqrt3_l259_259679

-- Define the functions f and g
def f (x : ℝ) := 2 - 2 * x
def g (x : ℝ) := x^2 - 1

-- State the theorem to prove
theorem fg_neg_sqrt3 : f (g (-real.sqrt 3)) = -2 := by
  sorry

end fg_neg_sqrt3_l259_259679


namespace commute_time_abs_diff_l259_259954

theorem commute_time_abs_diff (x y : ℝ) 
  (h1 : (x + y + 10 + 11 + 9)/5 = 10) 
  (h2 : (1/5 : ℝ) * ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) = 2) : 
  |x - y| = 4 :=
by
  sorry

end commute_time_abs_diff_l259_259954


namespace find_natural_number_l259_259658

-- Definitions reflecting the conditions and result
def is_sum_of_two_squares (n : ℕ) := ∃ a b : ℕ, a * a + b * b = n

def has_exactly_one_not_sum_of_two_squares (n : ℕ) :=
  ∃! x : ℤ, ¬is_sum_of_two_squares (x.natAbs % n)

theorem find_natural_number (n : ℕ) (h : n ≥ 2) : 
  has_exactly_one_not_sum_of_two_squares n ↔ n = 4 :=
sorry

end find_natural_number_l259_259658


namespace maximum_abc_827_l259_259031

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end maximum_abc_827_l259_259031


namespace part1_part2_l259_259311

-- Part (1)
theorem part1 (B : ℝ) (b : ℝ) (S : ℝ) (a c : ℝ) (B_eq : B = Real.pi / 3) 
  (b_eq : b = Real.sqrt 7) (S_eq : S = (3 * Real.sqrt 3) / 2) :
  a + c = 5 := 
sorry

-- Part (2)
theorem part2 (C : ℝ) (c : ℝ) (dot_BA_BC AB_AC : ℝ) 
  (C_cond : 2 * Real.cos C * (dot_BA_BC + AB_AC) = c^2) :
  C = Real.pi / 3 := 
sorry

end part1_part2_l259_259311


namespace segments_either_disjoint_or_common_point_l259_259897

theorem segments_either_disjoint_or_common_point (n : ℕ) (segments : List (ℝ × ℝ)) 
  (h_len : segments.length = n^2 + 1) : 
  (∃ (disjoint_segments : List (ℝ × ℝ)), disjoint_segments.length ≥ n + 1 ∧ 
    (∀ (s1 s2 : (ℝ × ℝ)), s1 ∈ disjoint_segments → s2 ∈ disjoint_segments 
    → s1 ≠ s2 → ¬ (s1.1 ≤ s2.2 ∧ s2.1 ≤ s1.2))) 
  ∨ 
  (∃ (common_point_segments : List (ℝ × ℝ)), common_point_segments.length ≥ n + 1 ∧ 
    (∃ (p : ℝ), ∀ (s : (ℝ × ℝ)), s ∈ common_point_segments → s.1 ≤ p ∧ p ≤ s.2)) :=
sorry

end segments_either_disjoint_or_common_point_l259_259897


namespace inequality_a_b_c_l259_259071

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l259_259071


namespace sin_pi_plus_alpha_l259_259302

theorem sin_pi_plus_alpha {α : ℝ} (h1 : cos α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) = -12 / 13 :=
sorry

end sin_pi_plus_alpha_l259_259302


namespace business_profit_l259_259558

theorem business_profit 
  (investmentMary : ℝ) (investmentMike : ℝ) (P : ℝ) 
  (hInvestmentMary : investmentMary = 650)
  (hInvestmentMike : investmentMike = 350)
  (hOneThirdP : (P / 3) / 2)
  (hRemainingP : 2 * P / 3)
  (hMaryShare : (P / 6) + (13 / 20 * (2 * P / 3)))
  (hMikeShare : (P / 6) + (7 / 20 * (2 * P / 3)))
  (hMaryMoreThanMike : hMaryShare - hMikeShare = 600):
  P = 3000 :=
sorry

end business_profit_l259_259558


namespace time_for_son_l259_259922

-- Define the conditions
def man_work_time : ℝ := 6
def combined_work_time : ℝ := 3

-- Define the rate of a man doing work per day
def man_rate : ℝ := 1 / man_work_time

-- Define the rate of a man and son working together
def combined_rate : ℝ := 1 / combined_work_time

-- Prove that the son alone can finish the work in 6 days
theorem time_for_son 
: ∃ time_son : ℝ, time_son = 6 ∧ 1 / time_son = combined_rate - man_rate :=
sorry

end time_for_son_l259_259922


namespace prime_square_condition_no_prime_cube_condition_l259_259255

-- Part (a): Prove p = 3 given 8*p + 1 = n^2 and p is a prime
theorem prime_square_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 2) : 
  p = 3 :=
sorry

-- Part (b): Prove no p exists given 8*p + 1 = n^3 and p is a prime
theorem no_prime_cube_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 3) : 
  False :=
sorry

end prime_square_condition_no_prime_cube_condition_l259_259255


namespace sin_alpha_plus_beta_l259_259753

-- Define the given conditions
def e_i_alpha : ℂ := complex.exp (complex.I * α) = (3/5 + 4/5 * complex.I)
def e_i_beta : ℂ := complex.exp (complex.I * β) = (-12/13 + 5/13 * complex.I)

-- Statement to be proven in Lean 4
theorem sin_alpha_plus_beta :
  (complex.exp (complex.I * (α + β))).im = -33 / 65 
by
  sorry

end sin_alpha_plus_beta_l259_259753


namespace find_angle_y_l259_259007

-- Define the conditions of the problem
def angle_A : ℝ := 50
def angle_B : ℝ := 95
def angle_DCE : ℝ := 90

-- Define the problem statement that needs to be proved
theorem find_angle_y : 
  (∀ (angle_C : ℝ), angle_C = 180 - (angle_A + angle_B) → 
  ∀ (y : ℝ), y = angle_DCE - angle_C → y = 55) :=
by
  intro angle_C h1 y h2
  rw [h1, h2]
  sorry

end find_angle_y_l259_259007


namespace problem1_simplification_problem2_solve_fraction_l259_259162

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l259_259162


namespace largest_sum_digits_24_hour_watch_l259_259949

theorem largest_sum_digits_24_hour_watch : 
  (∃ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
              (h / 10 + h % 10 + m / 10 + m % 10 = 24)) :=
by
  sorry

end largest_sum_digits_24_hour_watch_l259_259949


namespace yellow_balls_in_bag_l259_259369

theorem yellow_balls_in_bag (x : ℕ) (prob : 1 / (1 + x) = 1 / 4) :
  x = 3 :=
sorry

end yellow_balls_in_bag_l259_259369


namespace both_sports_count_l259_259152

theorem both_sports_count (total : ℕ) (tennis : ℕ) (squash : ℕ) (neither : ℕ) :
  total = 38 → tennis = 19 → squash = 21 → neither = 10 →
  (total - neither) = (tennis + squash - (total - neither)) :=
begin
  intros total_eq tennis_eq squash_eq neither_eq,
  rw [total_eq, tennis_eq, squash_eq, neither_eq],
  linarith,
end

end both_sports_count_l259_259152


namespace no_real_roots_of_quadratic_l259_259344

theorem no_real_roots_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -Real.sqrt 5 → c = Real.sqrt 2 →
  (b^2 - 4 * a * c < 0) → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c ha hb hc hD
  rw [ha, hb, hc] at hD
  sorry

end no_real_roots_of_quadratic_l259_259344


namespace plane_equation_through_point_parallel_l259_259656

theorem plane_equation_through_point_parallel (A B C D : ℤ) (hx hy hz : ℤ) (x y z : ℤ)
  (h_point : (A, B, C, D) = (-2, 1, -3, 10))
  (h_coordinates : (hx, hy, hz) = (2, -3, 1))
  (h_plane_parallel : ∀ x y z, -2 * x + y - 3 * z = 7 ↔ A * x + B * y + C * z + D = 0)
  (h_form : A > 0):
  ∃ A' B' C' D', A' * (x : ℤ) + B' * (y : ℤ) + C' * (z : ℤ) + D' = 0 :=
by
  sorry

end plane_equation_through_point_parallel_l259_259656


namespace range_of_x_of_sqrt_x_plus_3_l259_259389

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l259_259389


namespace smaller_square_area_l259_259413

theorem smaller_square_area (A_L : ℝ) (h : A_L = 100) : ∃ A_S : ℝ, A_S = 50 := 
by
  sorry

end smaller_square_area_l259_259413


namespace distinct_orders_scoops_l259_259079

-- Conditions
def total_scoops : ℕ := 4
def chocolate_scoops : ℕ := 2
def vanilla_scoops : ℕ := 1
def strawberry_scoops : ℕ := 1

-- Problem statement
theorem distinct_orders_scoops :
  (Nat.factorial total_scoops) / ((Nat.factorial chocolate_scoops) * (Nat.factorial vanilla_scoops) * (Nat.factorial strawberry_scoops)) = 12 := by
  sorry

end distinct_orders_scoops_l259_259079


namespace no_tiling_possible_2003_board_l259_259634

theorem no_tiling_possible_2003_board :
  ¬∃ (f : Fin 2003 → Fin 2003 → Fin 2003 → ℕ),
  (∀ (x : Fin 2003) (y : Fin 2003),
    f x y 2 = 1 ∨ f x y 2 = 2) ∧ 
  (∀ (x y : Fin 2003) (n : ℕ),
    if n = 2 then f x y 2 + f x (y + 1 % 2003) 2 = 3 else 
    f x y 3 + f (x + 1 % 2003) y 3 + f (x + 2 % 2003) y 3 = 3 ∨ 
    f x y 3 + f (x + 1 % 2003) y 3 + f (x + 2 % 2003) y 3 = 6) ∧ 
  (∑ (x y : Fin 2003), f x y (if x % 2 = 0 then 1 else 2) = 6012009) :=
sorry

end no_tiling_possible_2003_board_l259_259634


namespace find_c_k_l259_259890

theorem find_c_k {d r k : ℕ} (h1 : ∀ n, (1 + (n-1)*d > 0)) (h2 : ∀ n, (r^(n-1) > 0)) 
  (h_arith_seq : ∀ n, ∃ a_n, a_n = 1 + (n-1)*d) 
  (h_geo_seq : ∀ n, ∃ b_n, b_n = r^(n-1)) 
  (h_c_seq : ∀ n, ∃ c_n, c_n = (1 + (n-1)*d) + r^(n-1)) :
  (c_{k-1} = 200 ∧ c_{k+1} = 900) → ∃ c_k, c_k = 928 := 
by 
  sorry

end find_c_k_l259_259890


namespace no_int_pairs_in_ap_l259_259237

theorem no_int_pairs_in_ap : ∀ (a b : ℤ), (6 + b) = 2 * a → (6 + b) * b + 6 = 4 * b → false :=
by
intros a b h1 h2
suffices (b^2 - 2*b + 6 = 0) by
  have : (b^2 - 2*b + 6 < 0) by
    calc
      b^2 - 2*b + 6 = b^2 - 2*b + 6 : by ring
      ... < 0 : by sorry -- Discriminant calculation and negative conclusion here
  exact absurd this (by linarith)
apply sorry -- Derive b^2 - 2*b + 6 = 0 from the conditions h1 and h2

end no_int_pairs_in_ap_l259_259237
