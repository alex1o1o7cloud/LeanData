import Mathlib

namespace continuous_piecewise_function_l850_85026

theorem continuous_piecewise_function (a c : ℝ) (h1 : 2 * a * 2 + 6 = 3 * 2 - 2) (h2 : 4 * (-2) + 2 * c = 3 * (-2) - 2) : 
  a + c = -1/2 := 
sorry

end continuous_piecewise_function_l850_85026


namespace intersection_points_l850_85039

-- Definitions and conditions
def is_ellipse (e : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, e x y ↔ x^2 + 2*y^2 = 2

def is_tangent_or_intersects (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
  ∃ z1 z2 : ℝ, (e z1 (l z1) ∨ e z2 (l z2))

def lines_intersect (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, l1 x = l2 x

theorem intersection_points :
  ∀ (e : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ),
  is_ellipse e →
  is_tangent_or_intersects l1 e →
  is_tangent_or_intersects l2 e →
  lines_intersect l1 l2 →
  ∃ n : ℕ, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros e l1 l2 he hto1 hto2 hl
  sorry

end intersection_points_l850_85039


namespace area_of_intersection_l850_85081

-- Define the region M
def in_region_M (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

-- Define the region N as it changes with t
def in_region_N (t x : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the function f(t) which represents the common area of M and N
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 0.5

-- Prove that f(t) is correct given the above conditions
theorem area_of_intersection (t : ℝ) :
  (∀ x y : ℝ, in_region_M x y → in_region_N t x → y ≤ f t) →
  0 ≤ t ∧ t ≤ 1 →
  f t = -t^2 + t + 0.5 :=
by
  sorry

end area_of_intersection_l850_85081


namespace minimum_value_occurs_at_4_l850_85083

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l850_85083


namespace values_of_n_l850_85033

theorem values_of_n (a b d : ℕ) :
  7 * a + 77 * b + 7777 * d = 6700 →
  ∃ n : ℕ, ∃ (count : ℕ), count = 107 ∧ n = a + 2 * b + 4 * d := 
by
  sorry

end values_of_n_l850_85033


namespace handshake_count_l850_85006

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l850_85006


namespace value_of_fraction_l850_85024

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l850_85024


namespace matchstick_equality_l850_85063

theorem matchstick_equality :
  abs ((22 : ℝ) / 7 - Real.pi) < 0.1 := 
sorry

end matchstick_equality_l850_85063


namespace true_discount_correct_l850_85013

noncomputable def true_discount (banker_gain : ℝ) (average_rate : ℝ) (time_years : ℝ) : ℝ :=
  let r := average_rate
  let t := time_years
  let exp_factor := Real.exp (-r * t)
  let face_value := banker_gain / (1 - exp_factor)
  face_value - (face_value * exp_factor)

theorem true_discount_correct : 
  true_discount 15.8 0.145 5 = 15.8 := 
by
  sorry

end true_discount_correct_l850_85013


namespace red_ball_value_l850_85027

theorem red_ball_value (r b g : ℕ) (blue_points green_points : ℕ)
  (h1 : blue_points = 4)
  (h2 : green_points = 5)
  (h3 : b = g)
  (h4 : r^4 * blue_points^b * green_points^g = 16000)
  (h5 : b = 6) :
  r = 1 :=
by
  sorry

end red_ball_value_l850_85027


namespace euler_characteristic_convex_polyhedron_l850_85003

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end euler_characteristic_convex_polyhedron_l850_85003


namespace lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l850_85078

-- Define a cube with edge length a
structure Cube :=
  (a : ℝ) -- Edge length of the cube

-- Define a pyramid with a given height
structure Pyramid :=
  (h : ℝ) -- Height of the pyramid

-- The main theorem statement for part 4A
theorem lateral_edges_in_same_plane (c : Cube) (p : Pyramid) : p.h = c.a ↔ (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
  O1 = (c.a / 2, c.a / 2, -p.h) ∧
  O2 = (c.a / 2, -p.h, c.a / 2) ∧
  O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

-- The main theorem statement for part 4B
theorem edges_in_planes_for_all_vertices (c : Cube) (p : Pyramid) : p.h = c.a ↔ ∀ (v : ℝ × ℝ × ℝ), -- Iterate over cube vertices
  (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
    O1 = (c.a / 2, c.a / 2, -p.h) ∧
    O2 = (c.a / 2, -p.h, c.a / 2) ∧
    O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

end lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l850_85078


namespace isosceles_right_triangle_ratio_l850_85034

theorem isosceles_right_triangle_ratio {a : ℝ} (h_pos : 0 < a) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 :=
sorry

end isosceles_right_triangle_ratio_l850_85034


namespace Tim_marbles_l850_85059

theorem Tim_marbles (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 :=
by
  sorry

end Tim_marbles_l850_85059


namespace fibonacci_p_arithmetic_periodic_l850_85038

-- Define p-arithmetic system and its properties
def p_arithmetic (p : ℕ) : Prop :=
  ∀ (a : ℤ), a ≠ 0 → a^(p-1) = 1

-- Define extraction of sqrt(5)
def sqrt5_extractable (p : ℕ) : Prop :=
  ∃ (r : ℝ), r^2 = 5

-- Define Fibonacci sequence in p-arithmetic
def fibonacci_p_arithmetic (p : ℕ) (v : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, v (n+2) = v (n+1) + v n

-- Main Theorem
theorem fibonacci_p_arithmetic_periodic (p : ℕ) (v : ℕ → ℤ) :
  p_arithmetic p →
  sqrt5_extractable p →
  fibonacci_p_arithmetic p v →
  (∀ k : ℕ, v (k + p) = v k) :=
by
  intros _ _ _
  sorry

end fibonacci_p_arithmetic_periodic_l850_85038


namespace nabla_eq_37_l850_85069

def nabla (a b : ℤ) : ℤ := a * b + a - b

theorem nabla_eq_37 : nabla (-5) (-7) = 37 := by
  sorry

end nabla_eq_37_l850_85069


namespace john_started_5_days_ago_l850_85092

noncomputable def daily_wage (x : ℕ) : Prop := 250 + 10 * x = 750

theorem john_started_5_days_ago :
  ∃ x : ℕ, daily_wage x ∧ 250 / x = 5 :=
by
  sorry

end john_started_5_days_ago_l850_85092


namespace rectangles_cannot_cover_large_rectangle_l850_85062

theorem rectangles_cannot_cover_large_rectangle (n m : ℕ) (a b c d: ℕ) : 
  n = 14 → m = 9 → a = 2 → b = 3 → c = 3 → d = 2 → 
  (∀ (v_rects : ℕ) (h_rects : ℕ), v_rects = 10 → h_rects = 11 →
    (∀ (rect_area : ℕ), rect_area = n * m →
      (∀ (small_rect_area : ℕ), 
        small_rect_area = (v_rects * (a * b)) + (h_rects * (c * d)) →
        small_rect_area = rect_area → 
        false))) :=
by
  intros n_eq m_eq a_eq b_eq c_eq d_eq
       v_rects h_rects v_rects_eq h_rects_eq
       rect_area rect_area_eq small_rect_area small_rect_area_eq area_sum_eq
  sorry

end rectangles_cannot_cover_large_rectangle_l850_85062


namespace arith_sqrt_abs_neg_nine_l850_85035

theorem arith_sqrt_abs_neg_nine : Real.sqrt (abs (-9)) = 3 := by
  sorry

end arith_sqrt_abs_neg_nine_l850_85035


namespace only_solution_xyz_l850_85032

theorem only_solution_xyz : 
  ∀ (x y z : ℕ), x^3 + 4 * y^3 = 16 * z^3 + 4 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z
  intro h
  sorry

end only_solution_xyz_l850_85032


namespace correct_calculation_l850_85068

variable (a : ℝ)

theorem correct_calculation : (-2 * a) ^ 3 = -8 * a ^ 3 := by
  sorry

end correct_calculation_l850_85068


namespace stratified_sampling_l850_85046

theorem stratified_sampling
  (total_products : ℕ)
  (sample_size : ℕ)
  (workshop_products : ℕ)
  (h1 : total_products = 2048)
  (h2 : sample_size = 128)
  (h3 : workshop_products = 256) :
  (workshop_products / total_products) * sample_size = 16 := 
by
  rw [h1, h2, h3]
  norm_num
  
  sorry

end stratified_sampling_l850_85046


namespace jerry_age_l850_85058

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 2) (h2 : M = 18) : J = 10 := by
  sorry

end jerry_age_l850_85058


namespace solve_system_equations_l850_85028

theorem solve_system_equations (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    ∃ x y z : ℝ,  
      (x * y = (z - a) ^ 2) ∧
      (y * z = (x - b) ^ 2) ∧
      (z * x = (y - c) ^ 2) ∧
      x = ((b ^ 2 - a * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      y = ((c ^ 2 - a * b) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      z = ((a ^ 2 - b * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) :=
sorry

end solve_system_equations_l850_85028


namespace find_intersection_point_l850_85089

/-- Definition of the parabola -/
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 - 4 * y + 7

/-- Condition for intersection at exactly one point -/
def discriminant (m : ℝ) : ℝ := 4 ^ 2 - 4 * 3 * (m - 7)

/-- Main theorem stating the proof problem -/
theorem find_intersection_point (m : ℝ) :
  (discriminant m = 0) → m = 25 / 3 :=
by
  sorry

end find_intersection_point_l850_85089


namespace remainder_mod7_l850_85075

theorem remainder_mod7 (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := 
by
  sorry

end remainder_mod7_l850_85075


namespace geometric_sequence_seventh_term_l850_85070

theorem geometric_sequence_seventh_term :
  let a := 6
  let r := -2
  (a * r^(7 - 1)) = 384 := 
by
  sorry

end geometric_sequence_seventh_term_l850_85070


namespace max_possible_value_e_l850_85073

def b (n : ℕ) : ℕ := (7^n - 1) / 6

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_possible_value_e (n : ℕ) : e n = 1 := by
  sorry

end max_possible_value_e_l850_85073


namespace skateboarder_speed_l850_85042

theorem skateboarder_speed (d t : ℕ) (ft_per_mile hr_to_sec : ℕ)
  (h1 : d = 660) (h2 : t = 30) (h3 : ft_per_mile = 5280) (h4 : hr_to_sec = 3600) :
  ((d / t) / ft_per_mile) * hr_to_sec = 15 :=
by sorry

end skateboarder_speed_l850_85042


namespace find_a_l850_85008

theorem find_a (a : ℝ) : 
  (∃ (r : ℕ), r = 3 ∧ 
  ((-1)^r * (Nat.choose 5 r : ℝ) * a^(5 - r) = -40)) ↔ a = 2 ∨ a = -2 :=
by
    sorry

end find_a_l850_85008


namespace find_side_length_of_square_l850_85099

variable (a : ℝ)

theorem find_side_length_of_square (h1 : a - 3 > 0)
                                   (h2 : 3 * a + 5 * (a - 3) = 57) :
  a = 9 := 
by
  sorry

end find_side_length_of_square_l850_85099


namespace smallest_population_multiple_of_3_l850_85074

theorem smallest_population_multiple_of_3 : 
  ∃ (a : ℕ), ∃ (b c : ℕ), 
  a^2 + 50 = b^2 + 1 ∧ b^2 + 51 = c^2 ∧ 
  (∃ m : ℕ, a * a = 576 ∧ 576 = 3 * m) :=
by
  sorry

end smallest_population_multiple_of_3_l850_85074


namespace pizza_cost_l850_85000

theorem pizza_cost
  (P T : ℕ)
  (hT : T = 1)
  (h_total : 3 * P + 4 * T + 5 = 39) :
  P = 10 :=
by
  sorry

end pizza_cost_l850_85000


namespace tank_capacity_is_correct_l850_85049

-- Definition of the problem conditions
def initial_fraction := 1 / 3
def added_water := 180
def final_fraction := 2 / 3

-- Capacity of the tank
noncomputable def tank_capacity : ℕ := 540

-- Proof statement
theorem tank_capacity_is_correct (x : ℕ) :
  (initial_fraction * x + added_water = final_fraction * x) → x = tank_capacity := 
by
  -- This is where the proof would go
  sorry

end tank_capacity_is_correct_l850_85049


namespace triangle_base_l850_85071

noncomputable def side_length_square (p : ℕ) : ℕ := p / 4

noncomputable def area_square (s : ℕ) : ℕ := s * s

noncomputable def area_triangle (h b : ℕ) : ℕ := (h * b) / 2

theorem triangle_base (p h a b : ℕ) (hp : p = 80) (hh : h = 40) (ha : a = (side_length_square p)^2) (eq_areas : area_square (side_length_square p) = area_triangle h b) : b = 20 :=
by {
  -- Here goes the proof which we are omitting
  sorry
}

end triangle_base_l850_85071


namespace total_jewelry_pieces_l850_85093

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := 2 * initial_earrings / 3
noncomputable def extra_earrings_from_mother : ℕ := bought_earrings / 5

theorem total_jewelry_pieces : initial_necklaces + bought_necklaces + initial_earrings + bought_earrings + extra_earrings_from_mother = 47 :=
by
  have total_necklaces : ℕ := initial_necklaces + bought_necklaces
  have total_earrings : ℕ := initial_earrings + bought_earrings + extra_earrings_from_mother
  have total_jewelry : ℕ := total_necklaces + total_earrings
  exact Eq.refl 47
  
#check total_jewelry_pieces -- Check if the type is correct

end total_jewelry_pieces_l850_85093


namespace solution_set_for_log_inequality_l850_85065

noncomputable def f : ℝ → ℝ := sorry

def isEven (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def isIncreasingOnNonNeg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_positive_at_third : Prop := f (1 / 3) > 0

theorem solution_set_for_log_inequality
  (hf_even : isEven f)
  (hf_increasing : isIncreasingOnNonNeg f)
  (hf_positive : f_positive_at_third) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} := sorry

end solution_set_for_log_inequality_l850_85065


namespace mart_income_percentage_of_juan_l850_85096

theorem mart_income_percentage_of_juan
  (J T M : ℝ)
  (h1 : T = 0.60 * J)
  (h2 : M = 1.60 * T) :
  M = 0.96 * J :=
by 
  sorry

end mart_income_percentage_of_juan_l850_85096


namespace simplify_and_evaluate_l850_85072

theorem simplify_and_evaluate : 
    ∀ (a b : ℤ), a = 1 → b = -1 → 
    ((2 * a^2 * b - 2 * a * b^2 - b^3) / b - (a + b) * (a - b) = 3) := 
by
  intros a b ha hb
  sorry

end simplify_and_evaluate_l850_85072


namespace find_c_find_A_l850_85036

open Real

noncomputable def acute_triangle_sides (A B C a b c : ℝ) : Prop :=
  a = b * cos C + (sqrt 3 / 3) * c * sin B

theorem find_c (A B C a b c : ℝ) (ha : a = 2) (hb : b = sqrt 7) 
  (hab : acute_triangle_sides A B C a b c) : c = 3 := 
sorry

theorem find_A (A B C : ℝ) (h : sqrt 3 * sin (2 * A - π / 6) - 2 * (sin (C - π / 12))^2 = 0)
  (h_range : π / 6 < A ∧ A < π / 2) : A = π / 4 :=
sorry

end find_c_find_A_l850_85036


namespace exists_root_abs_leq_2_abs_c_div_b_l850_85050

theorem exists_root_abs_leq_2_abs_c_div_b (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_real_roots : ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| :=
by
  sorry

end exists_root_abs_leq_2_abs_c_div_b_l850_85050


namespace race_course_length_l850_85057

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : 4 * (d - 69) = d) : d = 92 :=
by
  sorry

end race_course_length_l850_85057


namespace sarah_initial_money_l850_85019

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l850_85019


namespace volume_of_pool_l850_85087

variable (P T V C : ℝ)

/-- 
The volume of the pool is given as P * T divided by percentage C.
The question is to prove that the volume V of the pool equals 90000 cubic feet given:
  P: The hose can remove 60 cubic feet per minute.
  T: It takes 1200 minutes to drain the pool.
  C: The pool was at 80% capacity when draining started.
-/
theorem volume_of_pool (h1 : P = 60) 
                       (h2 : T = 1200) 
                       (h3 : C = 0.80) 
                       (h4 : P * T / C = V) :
  V = 90000 := 
sorry

end volume_of_pool_l850_85087


namespace div_of_abs_values_l850_85018

theorem div_of_abs_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x < y) : x / y = -2 := 
by
  sorry

end div_of_abs_values_l850_85018


namespace picture_area_l850_85094

theorem picture_area (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (3 * x + 4) * (y + 3) - x * y = 54 → x * y = 6 :=
by
  intros h
  sorry

end picture_area_l850_85094


namespace intersection_point_l850_85014

-- Definitions of the lines
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 10
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 20

-- Theorem stating the intersection point
theorem intersection_point : line1 (60 / 23) (50 / 23) ∧ line2 (60 / 23) (50 / 23) :=
by {
  sorry
}

end intersection_point_l850_85014


namespace distance_parallel_lines_distance_point_line_l850_85088

def line1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point : ℝ × ℝ := (0, 2)

noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_parallel_lines : distance_between_lines 2 1 (-1) 1 = (2 * Real.sqrt 5) / 5 := by
  sorry

theorem distance_point_line : distance_point_to_line 2 1 (-1) 0 2 = (Real.sqrt 5) / 5 := by
  sorry

end distance_parallel_lines_distance_point_line_l850_85088


namespace cube_relation_l850_85053

theorem cube_relation (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cube_relation_l850_85053


namespace divisibility_problem_l850_85067

theorem divisibility_problem (n : ℕ) : n-1 ∣ n^n - 7*n + 5*n^2024 + 3*n^2 - 2 := 
by
  sorry

end divisibility_problem_l850_85067


namespace days_required_by_x_l850_85022

theorem days_required_by_x (x y : ℝ) 
  (h1 : (1 / x + 1 / y = 1 / 12)) 
  (h2 : (1 / y = 1 / 24)) : 
  x = 24 := 
by
  sorry

end days_required_by_x_l850_85022


namespace initial_students_count_l850_85002

variable (n T : ℕ)
variables (initial_average remaining_average dropped_score : ℚ)
variables (initial_students remaining_students : ℕ)

theorem initial_students_count :
  initial_average = 62.5 →
  remaining_average = 63 →
  dropped_score = 55 →
  T = initial_average * n →
  T - dropped_score = remaining_average * (n - 1) →
  n = 16 :=
by
  intros h_avg_initial h_avg_remaining h_dropped_score h_total h_total_remaining
  sorry

end initial_students_count_l850_85002


namespace min_possible_value_of_x_l850_85044

theorem min_possible_value_of_x :
  ∀ (x y : ℝ),
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  (∀ y ≤ 100, x ≥ 0) →
  x ≥ 22 :=
by
  intros x y h_avg h_y 
  -- proof steps go here
  sorry

end min_possible_value_of_x_l850_85044


namespace range_of_f_is_pi_div_four_l850_85001

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l850_85001


namespace find_speed_of_B_l850_85060

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l850_85060


namespace equation1_solutions_equation2_solutions_l850_85079

theorem equation1_solutions (x : ℝ) : 3 * x^2 - 6 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

theorem equation2_solutions (x : ℝ) : x^2 + 4 * x - 1 = 0 ↔ (x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end equation1_solutions_equation2_solutions_l850_85079


namespace ratio_of_area_of_smaller_circle_to_larger_rectangle_l850_85047

noncomputable def ratio_areas (w : ℝ) : ℝ :=
  (3.25 * Real.pi * w^2 / 4) / (1.5 * w^2)

theorem ratio_of_area_of_smaller_circle_to_larger_rectangle (w : ℝ) : 
  ratio_areas w = 13 * Real.pi / 24 := 
by 
  sorry

end ratio_of_area_of_smaller_circle_to_larger_rectangle_l850_85047


namespace annual_average_growth_rate_l850_85041

theorem annual_average_growth_rate (x : ℝ) :
  7200 * (1 + x)^2 = 8450 :=
sorry

end annual_average_growth_rate_l850_85041


namespace opposite_of_neg_2023_l850_85043

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l850_85043


namespace evaluate_expression_l850_85056

theorem evaluate_expression :
  (827 * 827) - ((827 - 1) * (827 + 1)) = 1 :=
sorry

end evaluate_expression_l850_85056


namespace part_a_part_b_part_c_l850_85054

-- Part (a)
theorem part_a (m : ℤ) : (m^2 + 10) % (m - 2) = 0 ∧ (m^2 + 10) % (m + 4) = 0 ↔ m = -5 ∨ m = 9 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) : ∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 :=
sorry

-- Part (c)
theorem part_c (n : ℤ) : ∃ N : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m < N :=
sorry

end part_a_part_b_part_c_l850_85054


namespace right_triangle_area_l850_85082

variable (AB AC : ℝ) (angle_A : ℝ)

def is_right_triangle (AB AC : ℝ) (angle_A : ℝ) : Prop :=
  angle_A = 90

def area_of_triangle (AB AC : ℝ) : ℝ :=
  0.5 * AB * AC

theorem right_triangle_area :
  is_right_triangle AB AC angle_A →
  AB = 35 →
  AC = 15 →
  area_of_triangle AB AC = 262.5 :=
by
  intros
  simp [is_right_triangle, area_of_triangle]
  sorry

end right_triangle_area_l850_85082


namespace jane_earnings_in_two_weeks_l850_85030

-- Define the conditions in the lean environment
def number_of_chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def selling_price_per_dozen : ℕ := 2

-- Statement of the proof problem
theorem jane_earnings_in_two_weeks :
  (number_of_chickens * eggs_per_chicken_per_week * 2) / 12 * selling_price_per_dozen = 20 :=
by
  sorry

end jane_earnings_in_two_weeks_l850_85030


namespace find_x_value_l850_85045

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x)
  else Real.log x * Real.log 81

theorem find_x_value (x : ℝ) (h : f x = 1 / 4) : x = 3 :=
sorry

end find_x_value_l850_85045


namespace total_people_in_class_l850_85011

def likes_both (n : ℕ) := n = 5
def likes_only_baseball (n : ℕ) := n = 2
def likes_only_football (n : ℕ) := n = 3
def likes_neither (n : ℕ) := n = 6

theorem total_people_in_class
  (h1 : likes_both n1)
  (h2 : likes_only_baseball n2)
  (h3 : likes_only_football n3)
  (h4 : likes_neither n4) :
  n1 + n2 + n3 + n4 = 16 :=
by 
  sorry

end total_people_in_class_l850_85011


namespace ending_number_divisible_by_six_l850_85084

theorem ending_number_divisible_by_six (first_term : ℕ) (n : ℕ) (common_difference : ℕ) (sequence_length : ℕ) 
  (start : first_term = 12) 
  (diff : common_difference = 6)
  (num_terms : sequence_length = 11) :
  first_term + (sequence_length - 1) * common_difference = 72 := by
  sorry

end ending_number_divisible_by_six_l850_85084


namespace bungee_cord_extension_l850_85098

variables (m g H k h L₀ T_max : ℝ)
  (mass_nonzero : m ≠ 0)
  (gravity_positive : g > 0)
  (H_positive : H > 0)
  (k_positive : k > 0)
  (L₀_nonnegative : L₀ ≥ 0)
  (T_max_eq : T_max = 4 * m * g)
  (L_eq : L₀ + h = H)
  (hooke_eq : T_max = k * h)

theorem bungee_cord_extension :
  h = H / 2 := sorry

end bungee_cord_extension_l850_85098


namespace find_m_from_permutation_l850_85037

theorem find_m_from_permutation (A : Nat → Nat → Nat) (m : Nat) (hA : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5) : m = 7 :=
sorry

end find_m_from_permutation_l850_85037


namespace S_30_value_l850_85021

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ := sorry

axiom S_10 : geometric_sequence_sum 10 = 10
axiom S_20 : geometric_sequence_sum 20 = 30

theorem S_30_value : geometric_sequence_sum 30 = 70 :=
by
  sorry

end S_30_value_l850_85021


namespace solve_equation_l850_85085

theorem solve_equation (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (x^2 + x + 1 = 1 / (x^2 - x + 1)) ↔ x = 1 ∨ x = -1 :=
by sorry

end solve_equation_l850_85085


namespace households_with_both_car_and_bike_l850_85007

theorem households_with_both_car_and_bike 
  (total_households : ℕ) 
  (households_without_either : ℕ) 
  (households_with_car : ℕ) 
  (households_with_bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : households_without_either = 11)
  (H3 : households_with_car = 44)
  (H4 : households_with_bike_only = 35)
  : ∃ B : ℕ, households_with_car - households_with_bike_only = B ∧ B = 9 := 
by
  sorry

end households_with_both_car_and_bike_l850_85007


namespace town_council_original_plan_count_l850_85077

theorem town_council_original_plan_count (planned_trees current_trees : ℕ) (leaves_per_tree total_leaves : ℕ)
  (h1 : leaves_per_tree = 100)
  (h2 : total_leaves = 1400)
  (h3 : current_trees = total_leaves / leaves_per_tree)
  (h4 : current_trees = 2 * planned_trees) : 
  planned_trees = 7 :=
by
  sorry

end town_council_original_plan_count_l850_85077


namespace pages_left_to_read_l850_85052

-- Defining the given conditions
def total_pages : ℕ := 500
def read_first_night : ℕ := (20 * total_pages) / 100
def read_second_night : ℕ := (20 * total_pages) / 100
def read_third_night : ℕ := (30 * total_pages) / 100

-- The total pages read over the three nights
def total_read : ℕ := read_first_night + read_second_night + read_third_night

-- The remaining pages to be read
def remaining_pages : ℕ := total_pages - total_read

theorem pages_left_to_read : remaining_pages = 150 :=
by
  -- Leaving the proof as a placeholder
  sorry

end pages_left_to_read_l850_85052


namespace stapler_problem_l850_85080

noncomputable def staplesLeft (initial_staples : ℕ) (dozens : ℕ) (staples_per_report : ℝ) : ℝ :=
  initial_staples - (dozens * 12) * staples_per_report

theorem stapler_problem : staplesLeft 200 7 0.75 = 137 := 
by
  sorry

end stapler_problem_l850_85080


namespace s_neq_t_if_Q_on_DE_l850_85076

-- Conditions and Definitions
noncomputable def DQ (x : ℝ) := x
noncomputable def QE (x : ℝ) := 10 - x
noncomputable def FQ := 5 * Real.sqrt 3
noncomputable def s (x : ℝ) := (DQ x) ^ 2 + (QE x) ^ 2
noncomputable def t := 2 * FQ ^ 2

-- Lean 4 Statement
theorem s_neq_t_if_Q_on_DE (x : ℝ) : s x ≠ t :=
by
  sorry -- Provided proof step to be filled in

end s_neq_t_if_Q_on_DE_l850_85076


namespace sum_product_of_integers_l850_85009

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l850_85009


namespace possible_teams_count_l850_85012

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

end possible_teams_count_l850_85012


namespace Angelina_speed_grocery_to_gym_l850_85061

-- Define parameters for distances and times
def distance_home_to_grocery : ℕ := 720
def distance_grocery_to_gym : ℕ := 480
def time_difference : ℕ := 40

-- Define speeds
variable (v : ℕ) -- speed in meters per second from home to grocery
def speed_home_to_grocery := v
def speed_grocery_to_gym := 2 * v

-- Define times using given speeds and distances
def time_home_to_grocery := distance_home_to_grocery / speed_home_to_grocery
def time_grocery_to_gym := distance_grocery_to_gym / speed_grocery_to_gym

-- Proof statement for the problem
theorem Angelina_speed_grocery_to_gym
  (v_pos : 0 < v)
  (condition : time_home_to_grocery - time_difference = time_grocery_to_gym) :
  speed_grocery_to_gym = 24 := by
  sorry

end Angelina_speed_grocery_to_gym_l850_85061


namespace find_obtuse_angle_l850_85004

-- Define the conditions
def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180

-- Lean statement assuming the needed conditions
theorem find_obtuse_angle (α : ℝ) (h1 : is_obtuse α) (h2 : 4 * α = 360 + α) : α = 120 :=
by sorry

end find_obtuse_angle_l850_85004


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l850_85016

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l850_85016


namespace checker_on_diagonal_l850_85029

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end checker_on_diagonal_l850_85029


namespace cyclist_speed_l850_85055

/-- 
  Two cyclists A and B start at the same time from Newton to Kingston, a distance of 50 miles. 
  Cyclist A travels 5 mph slower than cyclist B. After reaching Kingston, B immediately turns 
  back and meets A 10 miles from Kingston. --/
theorem cyclist_speed (a b : ℕ) (h1 : b = a + 5) (h2 : 40 / a = 60 / b) : a = 10 :=
by
  sorry

end cyclist_speed_l850_85055


namespace cosine_sum_sine_half_sum_leq_l850_85017

variable {A B C : ℝ}

theorem cosine_sum_sine_half_sum_leq (h : A + B + C = Real.pi) :
  (Real.cos A + Real.cos B + Real.cos C) ≤ (Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2)) :=
sorry

end cosine_sum_sine_half_sum_leq_l850_85017


namespace friend_spent_more_l850_85010

variable (total_spent : ℕ)
variable (friend_spent : ℕ)
variable (you_spent : ℕ)

-- Conditions
axiom total_is_11 : total_spent = 11
axiom friend_is_7 : friend_spent = 7
axiom spending_relation : total_spent = friend_spent + you_spent

-- Question
theorem friend_spent_more : friend_spent - you_spent = 3 :=
by
  sorry -- Here should be the formal proof

end friend_spent_more_l850_85010


namespace find_alpha_l850_85097

theorem find_alpha (α : ℝ) :
    7 * α + 8 * α + 45 = 180 →
    α = 9 :=
by
  sorry

end find_alpha_l850_85097


namespace sum_of_possible_values_l850_85064

theorem sum_of_possible_values :
  ∀ x, (|x - 5| - 4 = 3) → x = 12 ∨ x = -2 → (12 + (-2) = 10) :=
by
  sorry

end sum_of_possible_values_l850_85064


namespace fraction_of_juniors_l850_85095

theorem fraction_of_juniors (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h : 1 / 2 * J = 2 / 3 * S) : J / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l850_85095


namespace purely_imaginary_roots_iff_l850_85066

theorem purely_imaginary_roots_iff (z : ℂ) (k : ℝ) (i : ℂ) (h_i2 : i^2 = -1) :
  (∀ r : ℂ, (20 * r^2 + 6 * i * r - ↑k = 0) → (∃ b : ℝ, r = b * i)) ↔ (k = 9 / 5) :=
sorry

end purely_imaginary_roots_iff_l850_85066


namespace compute_xy_l850_85020

theorem compute_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 :=
by
  sorry

end compute_xy_l850_85020


namespace number_less_than_value_l850_85005

-- Definition for the conditions
def exceeds_condition (x y : ℕ) : Prop := x - 18 = 3 * (y - x)
def specific_value (x : ℕ) : Prop := x = 69

-- Statement of the theorem
theorem number_less_than_value : ∃ y : ℕ, (exceeds_condition 69 y) ∧ (specific_value 69) → y = 86 :=
by
  -- To be proved
  sorry

end number_less_than_value_l850_85005


namespace rick_total_clothes_ironed_l850_85090

def rick_ironing_pieces
  (shirts_per_hour : ℕ)
  (pants_per_hour : ℕ)
  (hours_shirts : ℕ)
  (hours_pants : ℕ) : ℕ :=
  (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants)

theorem rick_total_clothes_ironed :
  rick_ironing_pieces 4 3 3 5 = 27 :=
by
  sorry

end rick_total_clothes_ironed_l850_85090


namespace smallest_n_term_dec_l850_85048

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l850_85048


namespace speed_of_current_l850_85015

-- Definitions of the given conditions
def downstream_time := 6 / 60 -- time in hours to travel 1 km downstream
def upstream_time := 10 / 60 -- time in hours to travel 1 km upstream

-- Definition of speeds
def downstream_speed := 1 / downstream_time -- speed in km/h downstream
def upstream_speed := 1 / upstream_time -- speed in km/h upstream

-- Theorem statement
theorem speed_of_current : 
  (downstream_speed - upstream_speed) / 2 = 2 := 
by 
  -- We skip the proof for now
  sorry

end speed_of_current_l850_85015


namespace min_value_of_a_l850_85025

theorem min_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 + 2*x*y ≤ a*(x^2 + y^2)) → (a ≥ (Real.sqrt 5 + 1) / 2) := 
sorry

end min_value_of_a_l850_85025


namespace toms_total_profit_l850_85091

def total_earnings_mowing : ℕ := 4 * 12 + 3 * 15 + 1 * 20
def total_earnings_side_jobs : ℕ := 2 * 10 + 3 * 8 + 1 * 12
def total_earnings : ℕ := total_earnings_mowing + total_earnings_side_jobs
def total_expenses : ℕ := 17 + 5
def total_profit : ℕ := total_earnings - total_expenses

theorem toms_total_profit : total_profit = 147 := by
  -- Proof omitted
  sorry

end toms_total_profit_l850_85091


namespace fraction_meaningful_condition_l850_85051

theorem fraction_meaningful_condition (x : ℝ) : 3 - x ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_condition_l850_85051


namespace decrement_from_each_observation_l850_85023

theorem decrement_from_each_observation (n : Nat) (mean_original mean_updated decrement : ℝ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 191)
  (h4 : decrement = 9) :
  (mean_original - mean_updated) * (n : ℝ) / n = decrement :=
by
  sorry

end decrement_from_each_observation_l850_85023


namespace sum_X_Y_Z_l850_85040

theorem sum_X_Y_Z (X Y Z : ℕ) (hX : X ∈ Finset.range 10) (hY : Y ∈ Finset.range 10) (hZ : Z = 0)
     (div9 : (1 + 3 + 0 + 7 + 6 + 7 + 4 + X + 2 + 0 + Y + 0 + 0 + 8 + 0) % 9 = 0) 
     (div7 : (307674 * 10 + X * 20 + Y * 10 + 800) % 7 = 0) :
  X + Y + Z = 7 := 
sorry

end sum_X_Y_Z_l850_85040


namespace mary_starting_weight_l850_85086

def initial_weight (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ) : ℕ :=
  final_weight + (lost_3 - gained_4) + (gained_2 - lost_1) + lost_1

theorem mary_starting_weight :
  ∀ (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ),
  final_weight = 81 →
  lost_1 = 12 →
  gained_2 = 2 * lost_1 →
  lost_3 = 3 * lost_1 →
  gained_4 = lost_1 / 2 →
  initial_weight final_weight lost_1 gained_2 lost_3 gained_4 = 99 :=
by
  intros final_weight lost_1 gained_2 lost_3 gained_4 h_final_weight h_lost_1 h_gained_2 h_lost_3 h_gained_4
  rw [h_final_weight, h_lost_1] at *
  rw [h_gained_2, h_lost_3, h_gained_4]
  unfold initial_weight
  sorry

end mary_starting_weight_l850_85086


namespace quad_root_sum_product_l850_85031

theorem quad_root_sum_product (α β : ℝ) (h₁ : α ≠ β) (h₂ : α * α - 5 * α - 2 = 0) (h₃ : β * β - 5 * β - 2 = 0) : 
  α + β + α * β = 3 := 
by
  sorry

end quad_root_sum_product_l850_85031
