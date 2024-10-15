import Mathlib

namespace NUMINAMATH_GPT_peach_cost_l1487_148737

theorem peach_cost 
  (total_fruits : ℕ := 32)
  (total_cost : ℕ := 52)
  (plum_cost : ℕ := 2)
  (num_plums : ℕ := 20)
  (cost_peach : ℕ) :
  (total_cost - (num_plums * plum_cost)) = cost_peach * (total_fruits - num_plums) →
  cost_peach = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_peach_cost_l1487_148737


namespace NUMINAMATH_GPT_geometric_seq_arithmetic_triplet_l1487_148701

-- Definition of being in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Condition that a_5, a_4, and a_6 form an arithmetic sequence
def is_arithmetic_triplet (a : ℕ → ℝ) (n : ℕ) : Prop :=
  2 * a n = a (n+1) + a (n+2)

-- Our specific problem translated into a Lean statement
theorem geometric_seq_arithmetic_triplet {a : ℕ → ℝ} (q : ℝ) :
  is_geometric_sequence a q →
  is_arithmetic_triplet a 4 →
  q = 1 ∨ q = -2 :=
by
  intros h_geo h_arith
  -- Proof here is omitted
  sorry

end NUMINAMATH_GPT_geometric_seq_arithmetic_triplet_l1487_148701


namespace NUMINAMATH_GPT_belt_and_road_scientific_notation_l1487_148766

theorem belt_and_road_scientific_notation : 
  4600000000 = 4.6 * 10^9 := 
by
  sorry

end NUMINAMATH_GPT_belt_and_road_scientific_notation_l1487_148766


namespace NUMINAMATH_GPT_jinho_total_distance_l1487_148710

theorem jinho_total_distance (bus_distance_km : ℝ) (bus_distance_m : ℝ) (walk_distance_m : ℝ) :
  bus_distance_km = 4 → bus_distance_m = 436 → walk_distance_m = 1999 → 
  (2 * (bus_distance_km + bus_distance_m / 1000 + walk_distance_m / 1000)) = 12.87 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_jinho_total_distance_l1487_148710


namespace NUMINAMATH_GPT_number_of_sections_l1487_148741

def total_seats : ℕ := 270
def seats_per_section : ℕ := 30

theorem number_of_sections : total_seats / seats_per_section = 9 := 
by sorry

end NUMINAMATH_GPT_number_of_sections_l1487_148741


namespace NUMINAMATH_GPT_remainder_when_y_squared_divided_by_30_l1487_148700

theorem remainder_when_y_squared_divided_by_30 (y : ℤ) :
  6 * y ≡ 12 [ZMOD 30] → 5 * y ≡ 25 [ZMOD 30] → y ^ 2 ≡ 19 [ZMOD 30] :=
  by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_remainder_when_y_squared_divided_by_30_l1487_148700


namespace NUMINAMATH_GPT_root_quad_eq_sum_l1487_148777

theorem root_quad_eq_sum (a b : ℝ) (h1 : a^2 + a - 2022 = 0) (h2 : b^2 + b - 2022 = 0) (h3 : a + b = -1) : a^2 + 2 * a + b = 2021 :=
by sorry

end NUMINAMATH_GPT_root_quad_eq_sum_l1487_148777


namespace NUMINAMATH_GPT_company_employees_count_l1487_148759

theorem company_employees_count :
  (females : ℕ) ->
  (advanced_degrees : ℕ) ->
  (college_degree_only_males : ℕ) ->
  (advanced_degrees_females : ℕ) ->
  (110 = females) ->
  (90 = advanced_degrees) ->
  (35 = college_degree_only_males) ->
  (55 = advanced_degrees_females) ->
  (females - advanced_degrees_females + college_degree_only_males + advanced_degrees = 180) :=
by
  intros females advanced_degrees college_degree_only_males advanced_degrees_females
  intro h_females h_advanced_degrees h_college_degree_only_males h_advanced_degrees_females
  sorry

end NUMINAMATH_GPT_company_employees_count_l1487_148759


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1487_148767

theorem isosceles_triangle_base_length (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a + a + b = 15 ∨ a + b + b = 15) :
  b = 3 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1487_148767


namespace NUMINAMATH_GPT_natasha_quarters_l1487_148726

theorem natasha_quarters :
  ∃ n : ℕ, (4 < n) ∧ (n < 40) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n = 2) := sorry

end NUMINAMATH_GPT_natasha_quarters_l1487_148726


namespace NUMINAMATH_GPT_union_of_A_and_B_l1487_148785

open Set

variable (A B : Set ℤ)

theorem union_of_A_and_B (hA : A = {0, 1}) (hB : B = {0, -1}) : A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1487_148785


namespace NUMINAMATH_GPT_range_of_m_l1487_148756

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14 * m) ↔ 3 ≤ m ∧ m ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1487_148756


namespace NUMINAMATH_GPT_number_of_divisors_180_l1487_148797

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end NUMINAMATH_GPT_number_of_divisors_180_l1487_148797


namespace NUMINAMATH_GPT_evaluate_expression_l1487_148771

theorem evaluate_expression (x y z : ℕ) (hx : x = 5) (hy : y = 10) (hz : z = 3) : z * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1487_148771


namespace NUMINAMATH_GPT_book_cost_l1487_148772

-- Define the problem parameters
variable (p : ℝ) -- cost of one book in dollars

-- Conditions given in the problem
def seven_copies_cost_less_than_15 (p : ℝ) : Prop := 7 * p < 15
def eleven_copies_cost_more_than_22 (p : ℝ) : Prop := 11 * p > 22

-- The theorem stating the cost is between the given bounds
theorem book_cost (p : ℝ) (h1 : seven_copies_cost_less_than_15 p) (h2 : eleven_copies_cost_more_than_22 p) : 
    2 < p ∧ p < (15 / 7 : ℝ) :=
sorry

end NUMINAMATH_GPT_book_cost_l1487_148772


namespace NUMINAMATH_GPT_books_distribution_l1487_148762

noncomputable def distribution_ways : ℕ :=
  let books := 5
  let people := 4
  let combination := Nat.choose books 2
  let arrangement := Nat.factorial people
  combination * arrangement ^ people

theorem books_distribution : distribution_ways = 240 := by
  sorry

end NUMINAMATH_GPT_books_distribution_l1487_148762


namespace NUMINAMATH_GPT_one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l1487_148708

theorem one_div_add_one_div_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) := 
sorry

theorem one_div_add_one_div_not_upper_bounded (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M := 
sorry

theorem one_div_add_one_div_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (2 ≤ (1 / a + 1 / b) ∧ ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M) := 
sorry

end NUMINAMATH_GPT_one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l1487_148708


namespace NUMINAMATH_GPT_functional_eq_1996_l1487_148783

def f (x : ℝ) : ℝ := sorry

theorem functional_eq_1996 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f y)^2)) :
    ∀ x : ℝ, f (1996 * x) = 1996 * f x := 
sorry

end NUMINAMATH_GPT_functional_eq_1996_l1487_148783


namespace NUMINAMATH_GPT_complement_A_l1487_148730

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < 2 }

theorem complement_A :
  (U \ A) = { x : ℝ | x >= 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_A_l1487_148730


namespace NUMINAMATH_GPT_cube_inequality_l1487_148714

theorem cube_inequality (a b : ℝ) : a > b ↔ a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_inequality_l1487_148714


namespace NUMINAMATH_GPT_find_numbers_l1487_148731

theorem find_numbers 
  (x y z : ℕ) 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 51) 
  (h3 : z = 4 * x - y) : 
  x = 18 ∧ y = 33 ∧ z = 39 :=
by sorry

end NUMINAMATH_GPT_find_numbers_l1487_148731


namespace NUMINAMATH_GPT_total_washer_dryer_cost_l1487_148735

def washer_cost : ℕ := 710
def dryer_cost : ℕ := washer_cost - 220

theorem total_washer_dryer_cost :
  washer_cost + dryer_cost = 1200 :=
  by sorry

end NUMINAMATH_GPT_total_washer_dryer_cost_l1487_148735


namespace NUMINAMATH_GPT_color_guard_team_row_length_l1487_148749

theorem color_guard_team_row_length (n : ℕ) (p d : ℝ)
  (h_n : n = 40)
  (h_p : p = 0.4)
  (h_d : d = 0.5) :
  (n - 1) * d + n * p = 35.5 :=
by
  sorry

end NUMINAMATH_GPT_color_guard_team_row_length_l1487_148749


namespace NUMINAMATH_GPT_cannot_form_complex_pattern_l1487_148757

structure GeometricPieces where
  triangles : Nat
  squares : Nat

def possibleToForm (pieces : GeometricPieces) : Bool :=
  sorry -- Since the formation logic is unknown, it is incomplete.

theorem cannot_form_complex_pattern : 
  let pieces := GeometricPieces.mk 8 7
  ¬ possibleToForm pieces = true := 
sorry

end NUMINAMATH_GPT_cannot_form_complex_pattern_l1487_148757


namespace NUMINAMATH_GPT_truck_cargo_solution_l1487_148780

def truck_cargo_problem (x : ℝ) (n : ℕ) : Prop :=
  (∀ (x : ℝ) (n : ℕ), x = (x / n - 0.5) * (n + 4)) ∧ (55 ≤ x ∧ x ≤ 64)

theorem truck_cargo_solution :
  ∃ y : ℝ, y = 2.5 :=
sorry

end NUMINAMATH_GPT_truck_cargo_solution_l1487_148780


namespace NUMINAMATH_GPT_find_m_l1487_148796

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def circle_eq (x y m : ℝ) : Prop := x * x + y * y + x - 6 * y + m = 0

def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), line_eq x y ∧ line_eq (3 - 2 * y) y ∧ circle_eq x y m ∧ circle_eq (3 - 2 * y) y m) ∧
  (∃ (x1 y1 x2 y2 : ℝ), line_eq x1 y1 ∧ line_eq x2 y2 ∧ perpendicular_vectors x1 y1 x2 y2) → m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l1487_148796


namespace NUMINAMATH_GPT_max_value_l1487_148774

theorem max_value (x y : ℝ) : 
  (x + 3 * y + 4) / (Real.sqrt (x ^ 2 + y ^ 2 + 4)) ≤ Real.sqrt 26 :=
by
  -- Proof should be here
  sorry

end NUMINAMATH_GPT_max_value_l1487_148774


namespace NUMINAMATH_GPT_joan_gave_away_kittens_l1487_148705

-- Definitions based on conditions in the problem
def original_kittens : ℕ := 8
def kittens_left : ℕ := 6

-- Mathematical statement to be proved
theorem joan_gave_away_kittens : original_kittens - kittens_left = 2 :=
by
  sorry

end NUMINAMATH_GPT_joan_gave_away_kittens_l1487_148705


namespace NUMINAMATH_GPT_perpendicular_bisectors_intersect_at_one_point_l1487_148744

-- Define the key geometric concepts
variables {Point : Type*} [MetricSpace Point]

-- Define the given conditions 
variables (A B C M : Point)
variables (h1 : dist M A = dist M B)
variables (h2 : dist M B = dist M C)

-- Define the theorem to be proven
theorem perpendicular_bisectors_intersect_at_one_point :
  dist M A = dist M C :=
by 
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_perpendicular_bisectors_intersect_at_one_point_l1487_148744


namespace NUMINAMATH_GPT_Theorem3_l1487_148702

theorem Theorem3 {f g : ℝ → ℝ} (T1_eq_1 : ∀ x, f (x + 1) = f x)
  (m : ℕ) (h_g_periodic : ∀ x, g (x + 1 / m) = g x) (hm : m > 1) :
  ∃ k : ℕ, k > 0 ∧ (k = 1 ∨ (k ≠ m ∧ ¬(m % k = 0))) ∧ 
    (∀ x, (f x + g x) = (f (x + 1 / k) + g (x + 1 / k))) := 
sorry

end NUMINAMATH_GPT_Theorem3_l1487_148702


namespace NUMINAMATH_GPT_smallest_yellow_marbles_l1487_148727

theorem smallest_yellow_marbles :
  ∃ n : ℕ, (n ≡ 0 [MOD 20]) ∧
           (∃ b : ℕ, b = n / 4) ∧
           (∃ r : ℕ, r = n / 5) ∧
           (∃ g : ℕ, g = 10) ∧
           (∃ y : ℕ, y = n - (b + r + g) ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_smallest_yellow_marbles_l1487_148727


namespace NUMINAMATH_GPT_right_triangle_legs_solutions_l1487_148743

theorem right_triangle_legs_solutions (R r : ℝ) (h_cond : R / r ≥ 1 + Real.sqrt 2) :
  ∃ (a b : ℝ), 
    a = r + R + Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    b = r + R - Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    (2 * R)^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_solutions_l1487_148743


namespace NUMINAMATH_GPT_minimal_degree_g_l1487_148716

theorem minimal_degree_g {f g h : Polynomial ℝ} 
  (h_eq : 2 * f + 5 * g = h)
  (deg_f : f.degree = 6)
  (deg_h : h.degree = 10) : 
  g.degree = 10 :=
sorry

end NUMINAMATH_GPT_minimal_degree_g_l1487_148716


namespace NUMINAMATH_GPT_breadth_of_rectangle_l1487_148776

theorem breadth_of_rectangle (b l : ℝ) (h1 : l * b = 24 * b) (h2 : l - b = 10) : b = 14 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangle_l1487_148776


namespace NUMINAMATH_GPT_initial_markup_percentage_l1487_148779

theorem initial_markup_percentage (C : ℝ) (M : ℝ) :
  (C > 0) →
  (1 + M) * 1.25 * 0.90 = 1.35 →
  M = 0.2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_markup_percentage_l1487_148779


namespace NUMINAMATH_GPT_rectangle_longer_side_l1487_148719

theorem rectangle_longer_side
  (r : ℝ)
  (A_circle : ℝ)
  (A_rectangle : ℝ)
  (shorter_side : ℝ)
  (longer_side : ℝ) :
  r = 5 →
  A_circle = 25 * Real.pi →
  A_rectangle = 3 * A_circle →
  shorter_side = 2 * r →
  longer_side = A_rectangle / shorter_side →
  longer_side = 7.5 * Real.pi :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_longer_side_l1487_148719


namespace NUMINAMATH_GPT_isosceles_triangles_l1487_148798

theorem isosceles_triangles (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_triangle : ∀ n : ℕ, (a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n)) :
  b = c := 
sorry

end NUMINAMATH_GPT_isosceles_triangles_l1487_148798


namespace NUMINAMATH_GPT_prime_p_satisfies_conditions_l1487_148709

theorem prime_p_satisfies_conditions (p : ℕ) (hp : Nat.Prime p) (h1 : Nat.Prime (4 * p^2 + 1)) (h2 : Nat.Prime (6 * p^2 + 1)) : p = 5 :=
sorry

end NUMINAMATH_GPT_prime_p_satisfies_conditions_l1487_148709


namespace NUMINAMATH_GPT_restaurant_cooks_l1487_148794

variable (C W : ℕ)

theorem restaurant_cooks : 
  (C / W = 3 / 10) ∧ (C / (W + 12) = 3 / 14) → C = 9 :=
by sorry

end NUMINAMATH_GPT_restaurant_cooks_l1487_148794


namespace NUMINAMATH_GPT_time_to_travel_downstream_l1487_148790

-- Definitions based on the conditions.
def speed_boat_still_water := 40 -- Speed of the boat in still water (km/hr)
def speed_stream := 5 -- Speed of the stream (km/hr)
def distance_downstream := 45 -- Distance to be traveled downstream (km)

-- The proof statement
theorem time_to_travel_downstream : (distance_downstream / (speed_boat_still_water + speed_stream)) = 1 :=
by
  -- This would be the place to include the proven steps, but it's omitted as per instructions.
  sorry

end NUMINAMATH_GPT_time_to_travel_downstream_l1487_148790


namespace NUMINAMATH_GPT_greatest_int_radius_of_circle_l1487_148713

theorem greatest_int_radius_of_circle (r : ℝ) (A : ℝ) :
  (A < 200 * Real.pi) ∧ (A = Real.pi * r^2) →
  ∃k : ℕ, (k : ℝ) = 14 ∧ ∀n : ℕ, (n : ℝ) = r → n ≤ k := by
  sorry

end NUMINAMATH_GPT_greatest_int_radius_of_circle_l1487_148713


namespace NUMINAMATH_GPT_abs_neg_one_fourth_l1487_148751

theorem abs_neg_one_fourth : |(- (1 / 4))| = (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_one_fourth_l1487_148751


namespace NUMINAMATH_GPT_initial_weight_l1487_148791

theorem initial_weight (lost_weight current_weight : ℕ) (h1 : lost_weight = 35) (h2 : current_weight = 34) :
  lost_weight + current_weight = 69 :=
sorry

end NUMINAMATH_GPT_initial_weight_l1487_148791


namespace NUMINAMATH_GPT_seed_mixture_ryegrass_l1487_148781

theorem seed_mixture_ryegrass (α : ℝ) :
  (0.4667 * 0.4 + 0.5333 * α = 0.32) -> α = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_seed_mixture_ryegrass_l1487_148781


namespace NUMINAMATH_GPT_find_M_l1487_148782

theorem find_M (a b c M : ℚ) 
  (h1 : a + b + c = 100)
  (h2 : a - 10 = M)
  (h3 : b + 10 = M)
  (h4 : 10 * c = M) : 
  M = 1000 / 21 :=
sorry

end NUMINAMATH_GPT_find_M_l1487_148782


namespace NUMINAMATH_GPT_find_max_m_l1487_148720

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * Real.exp (2 * x) - a * x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (x - m) * f x 1 - (1/4) * Real.exp (2 * x) + x^2 + x

theorem find_max_m (h_inc : ∀ x > 0, g x m ≥ g x m) : m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_find_max_m_l1487_148720


namespace NUMINAMATH_GPT_abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l1487_148787

variable (x b : ℝ)

theorem abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1 :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := sorry

end NUMINAMATH_GPT_abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l1487_148787


namespace NUMINAMATH_GPT_star_sum_interior_angles_l1487_148786

theorem star_sum_interior_angles (n : ℕ) (h : n ≥ 6) :
  let S := 180 * n - 360
  S = 180 * (n - 2) :=
by
  let S := 180 * n - 360
  show S = 180 * (n - 2)
  sorry

end NUMINAMATH_GPT_star_sum_interior_angles_l1487_148786


namespace NUMINAMATH_GPT_problem_statement_l1487_148707

noncomputable def p := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

theorem problem_statement :
  (1 / p + 1 / q + 1 / r + 1 / s)^2 = 112 / 3481 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1487_148707


namespace NUMINAMATH_GPT_production_days_l1487_148706

theorem production_days (n : ℕ) (P : ℕ) (h1: P = n * 50) 
    (h2: (P + 110) / (n + 1) = 55) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_production_days_l1487_148706


namespace NUMINAMATH_GPT_ratio_largest_smallest_root_geometric_progression_l1487_148747

theorem ratio_largest_smallest_root_geometric_progression (a b c d : ℤ)
  (h_poly : a * x^3 + b * x^2 + c * x + d = 0) 
  (h_in_geo_prog : ∃ r1 r2 r3 q : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 * q = r2 ∧ r2 * q = r3 ∧ q ≠ 0) : 
  ∃ R : ℝ, R = 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_largest_smallest_root_geometric_progression_l1487_148747


namespace NUMINAMATH_GPT_area_of_octagon_in_square_l1487_148738

theorem area_of_octagon_in_square : 
  let A := (0, 0)
  let B := (6, 0)
  let C := (6, 6)
  let D := (0, 6)
  let E := (3, 0)
  let F := (6, 3)
  let G := (3, 6)
  let H := (0, 3)
  ∃ (octagon_area : ℚ),
    octagon_area = 6 :=
by
  sorry

end NUMINAMATH_GPT_area_of_octagon_in_square_l1487_148738


namespace NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l1487_148704

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l1487_148704


namespace NUMINAMATH_GPT_fraction_subtraction_l1487_148739

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1487_148739


namespace NUMINAMATH_GPT_compare_sqrt_sums_l1487_148718

   noncomputable def a : ℝ := Real.sqrt 8 + Real.sqrt 5
   noncomputable def b : ℝ := Real.sqrt 7 + Real.sqrt 6

   theorem compare_sqrt_sums : a < b :=
   by
     sorry
   
end NUMINAMATH_GPT_compare_sqrt_sums_l1487_148718


namespace NUMINAMATH_GPT_fraction_meaningful_condition_l1487_148732

theorem fraction_meaningful_condition (x : ℝ) : (∃ y, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_condition_l1487_148732


namespace NUMINAMATH_GPT_polynomial_roots_property_l1487_148734

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_property_l1487_148734


namespace NUMINAMATH_GPT_gcd_lcm_product_l1487_148763

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  Nat.gcd a b * Nat.lcm a b = 56700 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1487_148763


namespace NUMINAMATH_GPT_mean_of_counts_is_7_l1487_148795

theorem mean_of_counts_is_7 (counts : List ℕ) (h : counts = [6, 12, 1, 12, 7, 3, 8]) :
  counts.sum / counts.length = 7 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_counts_is_7_l1487_148795


namespace NUMINAMATH_GPT_reduced_price_per_dozen_is_3_l1487_148775

variable (P : ℝ) -- original price of an apple
variable (R : ℝ) -- reduced price of an apple
variable (A : ℝ) -- number of apples originally bought for Rs. 40
variable (cost_per_dozen_reduced : ℝ) -- reduced price per dozen apples

-- Define the conditions
axiom reduction_condition : R = 0.60 * P
axiom apples_bought_condition : 40 = A * P
axiom more_apples_condition : 40 = (A + 64) * R

-- Define the proof problem
theorem reduced_price_per_dozen_is_3 : cost_per_dozen_reduced = 3 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_is_3_l1487_148775


namespace NUMINAMATH_GPT_divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l1487_148788

theorem divisor_probability_of_25_factorial_is_odd_and_multiple_of_5 :
  let prime_factors_25 := 2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1 * 23^1
  let total_divisors := (22+1) * (10+1) * (6+1) * (3+1) * (2+1) * (1+1) * (1+1) * (1+1)
  let odd_and_multiple_of_5_divisors := (6+1) * (3+1) * (2+1) * (1+1) * (1+1)
  (odd_and_multiple_of_5_divisors / total_divisors : ℚ) = 7 / 23 := 
sorry

end NUMINAMATH_GPT_divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l1487_148788


namespace NUMINAMATH_GPT_points_comparison_l1487_148768

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end NUMINAMATH_GPT_points_comparison_l1487_148768


namespace NUMINAMATH_GPT_debbys_sister_candy_l1487_148770

-- Defining the conditions
def debby_candy : ℕ := 32
def eaten_candy : ℕ := 35
def remaining_candy : ℕ := 39

-- The proof problem
theorem debbys_sister_candy : ∃ S : ℕ, debby_candy + S - eaten_candy = remaining_candy → S = 42 :=
by
  sorry  -- The proof goes here

end NUMINAMATH_GPT_debbys_sister_candy_l1487_148770


namespace NUMINAMATH_GPT_alice_min_speed_exceeds_45_l1487_148703

theorem alice_min_speed_exceeds_45 
  (distance : ℕ)
  (bob_speed : ℕ)
  (alice_delay : ℕ)
  (alice_speed : ℕ)
  (bob_time : ℕ)
  (expected_speed : ℕ) 
  (distance_eq : distance = 180)
  (bob_speed_eq : bob_speed = 40)
  (alice_delay_eq : alice_delay = 1/2)
  (bob_time_eq : bob_time = distance / bob_speed)
  (expected_speed_eq : expected_speed = distance / (bob_time - alice_delay)) :
  alice_speed > expected_speed := 
sorry

end NUMINAMATH_GPT_alice_min_speed_exceeds_45_l1487_148703


namespace NUMINAMATH_GPT_product_of_base_9_digits_of_9876_l1487_148748

def base9_digits (n : ℕ) : List ℕ := 
  let rec digits_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else digits_aux (n / 9) ((n % 9) :: acc)
  digits_aux n []

def product (lst : List ℕ) : ℕ := lst.foldl (· * ·) 1

theorem product_of_base_9_digits_of_9876 :
  product (base9_digits 9876) = 192 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_base_9_digits_of_9876_l1487_148748


namespace NUMINAMATH_GPT_find_function_expression_l1487_148740

variable (f : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- conditions
axiom a1 : f 1 = 1
axiom a2 : ∀ (x y : ℝ), f (x + y) = f x + f y + 2 * y * (x + y) + 1

-- proof statement
theorem find_function_expression (x : ℕ) (h : x ≠ 0) : f x = x^2 + 3*x - 3 := sorry

end NUMINAMATH_GPT_find_function_expression_l1487_148740


namespace NUMINAMATH_GPT_average_salary_of_employees_l1487_148750

theorem average_salary_of_employees (A : ℝ)
  (h1 : 24 * A + 11500 = 25 * (A + 400)) :
  A = 1500 := 
by
  sorry

end NUMINAMATH_GPT_average_salary_of_employees_l1487_148750


namespace NUMINAMATH_GPT_find_quotient_l1487_148729

theorem find_quotient (A : ℕ) (h : 41 = (5 * A) + 1) : A = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l1487_148729


namespace NUMINAMATH_GPT_train_length_l1487_148715

theorem train_length (t : ℝ) (v : ℝ) (h1 : t = 13) (h2 : v = 58.15384615384615) : abs (v * t - 756) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1487_148715


namespace NUMINAMATH_GPT_equalize_expenses_l1487_148712

/-- Problem Statement:
Given the amount paid by LeRoy (A), Bernardo (B), and Carlos (C),
prove that the amount LeRoy must adjust to share the costs equally is (B + C - 2A) / 3.
-/
theorem equalize_expenses (A B C : ℝ) : 
  (B+C-2*A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end NUMINAMATH_GPT_equalize_expenses_l1487_148712


namespace NUMINAMATH_GPT_arithmetic_mean_reciprocals_first_four_primes_l1487_148725

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_reciprocals_first_four_primes_l1487_148725


namespace NUMINAMATH_GPT_find_A_l1487_148724

theorem find_A (A B : ℕ) (A_digit : A < 10) (B_digit : B < 10) :
  let fourteenA := 100 * 1 + 10 * 4 + A
  let Bseventy3 := 100 * B + 70 + 3
  fourteenA + Bseventy3 = 418 → A = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1487_148724


namespace NUMINAMATH_GPT_whisky_replacement_l1487_148733

variable (V x : ℝ)

/-- The initial whisky in the jar contains 40% alcohol -/
def initial_volume_of_alcohol (V : ℝ) : ℝ := 0.4 * V

/-- A part (x liters) of this whisky is replaced by another containing 19% alcohol -/
def volume_replaced_whisky (x : ℝ) : ℝ := x
def remaining_whisky (V x : ℝ) : ℝ := V - x

/-- The percentage of alcohol in the jar after replacement is 24% -/
def final_volume_of_alcohol (V x : ℝ) : ℝ := 0.4 * (remaining_whisky V x) + 0.19 * (volume_replaced_whisky x)

/- Prove that the quantity of whisky replaced is 0.16/0.21 times the total volume -/
theorem whisky_replacement :
  final_volume_of_alcohol V x = 0.24 * V → x = (0.16 / 0.21) * V :=
by sorry

end NUMINAMATH_GPT_whisky_replacement_l1487_148733


namespace NUMINAMATH_GPT_parallel_vectors_angle_l1487_148761

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (1 / 2, Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

theorem parallel_vectors_angle (α : ℝ) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (vec_a α).1 = k * (vec_b α).1 ∧ (vec_a α).2 = k * (vec_b α).2) (h_acute : 0 < α ∧ α < π / 2) :
  α = π / 4 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_angle_l1487_148761


namespace NUMINAMATH_GPT_determinant_identity_l1487_148753

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end NUMINAMATH_GPT_determinant_identity_l1487_148753


namespace NUMINAMATH_GPT_percent_of_area_triangle_in_pentagon_l1487_148721

-- Defining a structure for the problem statement
structure PentagonAndTriangle where
  s : ℝ -- side length of the equilateral triangle
  side_square : ℝ -- side of the square
  area_triangle : ℝ
  area_square : ℝ
  area_pentagon : ℝ

noncomputable def calculate_areas (s : ℝ) : PentagonAndTriangle :=
  let height_triangle := s * (Real.sqrt 3) / 2
  let area_triangle := Real.sqrt 3 / 4 * s^2
  let area_square := height_triangle^2
  let area_pentagon := area_square + area_triangle
  { s := s, side_square := height_triangle, area_triangle := area_triangle, area_square := area_square, area_pentagon := area_pentagon }

/--
Prove that the percentage of the pentagon's area that is the area of the equilateral triangle is (3 * (Real.sqrt 3 - 1)) / 6 * 100%.
-/
theorem percent_of_area_triangle_in_pentagon 
  (s : ℝ) 
  (pt : PentagonAndTriangle)
  (h₁ : pt = calculate_areas s)
  : pt.area_triangle / pt.area_pentagon = (3 * (Real.sqrt 3 - 1)) / 6 * 100 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_area_triangle_in_pentagon_l1487_148721


namespace NUMINAMATH_GPT_ellipse_properties_l1487_148778

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b ≥ 0)
  (e : ℝ)
  (hc : e = 4 / 5)
  (directrix : ℝ)
  (hd : directrix = 25 / 4)
  (x y : ℝ)
  (hx : (x - 6)^2 / 25 + (y - 6)^2 / 9 = 1) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_GPT_ellipse_properties_l1487_148778


namespace NUMINAMATH_GPT_connected_distinct_points_with_slope_change_l1487_148773

-- Defining the cost function based on the given conditions
def cost_function (n : ℕ) : ℕ := 
  if n <= 10 then 20 * n else 18 * n

-- The main theorem to prove the nature of the graph as described in the problem
theorem connected_distinct_points_with_slope_change : 
  (∀ n, (1 ≤ n ∧ n ≤ 20) → 
    (∃ k, cost_function n = k ∧ 
    (n <= 10 → cost_function n = 20 * n) ∧ 
    (n > 10 → cost_function n = 18 * n))) ∧
  (∃ n, n = 10 ∧ cost_function n = 200 ∧ cost_function (n + 1) = 198) :=
sorry

end NUMINAMATH_GPT_connected_distinct_points_with_slope_change_l1487_148773


namespace NUMINAMATH_GPT_new_paint_intensity_l1487_148758

theorem new_paint_intensity : 
  let I_original : ℝ := 0.5
  let I_added : ℝ := 0.2
  let replacement_fraction : ℝ := 1 / 3
  let remaining_fraction : ℝ := 2 / 3
  let I_new := remaining_fraction * I_original + replacement_fraction * I_added
  I_new = 0.4 :=
by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_new_paint_intensity_l1487_148758


namespace NUMINAMATH_GPT_cube_painting_l1487_148717

theorem cube_painting (n : ℕ) (h : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) ↔ (n = 8) :=
by
  sorry

end NUMINAMATH_GPT_cube_painting_l1487_148717


namespace NUMINAMATH_GPT_seating_arrangements_family_van_correct_l1487_148752

noncomputable def num_seating_arrangements (parents : Fin 2) (children : Fin 3) : Nat :=
  let perm3_2 := Nat.factorial 3 / Nat.factorial (3 - 2)
  2 * 1 * perm3_2

theorem seating_arrangements_family_van_correct :
  num_seating_arrangements 2 3 = 12 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_family_van_correct_l1487_148752


namespace NUMINAMATH_GPT_primes_sum_product_condition_l1487_148764

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem primes_sum_product_condition (m n p : ℕ) (hm : is_prime m) (hn : is_prime n) (hp : is_prime p)  
  (h : m * n * p = 5 * (m + n + p)) : 
  m^2 + n^2 + p^2 = 78 :=
sorry

end NUMINAMATH_GPT_primes_sum_product_condition_l1487_148764


namespace NUMINAMATH_GPT_neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l1487_148765

def p (x : ℝ) : Prop := (x^2 - x - 2) ≤ 0
def q (x m : ℝ) : Prop := (x^2 - x - m^2 - m) ≤ 0

theorem neg_p_range_of_x (x : ℝ) : ¬ p x → x > 2 ∨ x < -1 :=
by
-- proof steps here
sorry

theorem neg_q_sufficient_not_necessary_for_neg_p (m : ℝ) : 
  (∀ x, ¬ q x m → ¬ p x) ∧ (∃ x, p x → ¬ q x m) → m > 1 ∨ m < -2 :=
by
-- proof steps here
sorry

end NUMINAMATH_GPT_neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l1487_148765


namespace NUMINAMATH_GPT_wire_cut_problem_l1487_148742

variable (x : ℝ)

theorem wire_cut_problem 
  (h₁ : x + (5 / 2) * x = 49) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_problem_l1487_148742


namespace NUMINAMATH_GPT_time_until_meeting_l1487_148792

theorem time_until_meeting (v1 v2 : ℝ) (t2 t1 : ℝ) 
    (h1 : v1 = 6) 
    (h2 : v2 = 4) 
    (h3 : t2 = 10)
    (h4 : v2 * t1 = v1 * (t1 - t2)) : t1 = 30 := 
sorry

end NUMINAMATH_GPT_time_until_meeting_l1487_148792


namespace NUMINAMATH_GPT_parallelogram_and_triangle_area_eq_l1487_148784

noncomputable def parallelogram_area (AB AD : ℝ) : ℝ :=
  AB * AD

noncomputable def right_triangle_area (DG FG : ℝ) : ℝ :=
  (DG * FG) / 2

variables (AB AD DG FG : ℝ)
variables (angleDFG : ℝ)

def parallelogram_ABCD (AB : ℝ) (AD : ℝ) (angleDFG : ℝ) (DG : ℝ) : Prop :=
  parallelogram_area AB AD = 24 ∧ angleDFG = 90 ∧ DG = 6

theorem parallelogram_and_triangle_area_eq (h1 : parallelogram_ABCD AB AD angleDFG DG)
    (h2 : parallelogram_area AB AD = right_triangle_area DG FG) : FG = 8 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_and_triangle_area_eq_l1487_148784


namespace NUMINAMATH_GPT_sum_of_powers_modulo_seven_l1487_148728

theorem sum_of_powers_modulo_seven :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7) = 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_modulo_seven_l1487_148728


namespace NUMINAMATH_GPT_max_similar_triangles_five_points_l1487_148711

-- Let P be a finite set of points on a plane with exactly 5 elements.
def max_similar_triangles(P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 then
    8
  else
    0 -- This is irrelevant for the problem statement, but we need to define it.

-- The main theorem statement
theorem max_similar_triangles_five_points {P : Finset (ℝ × ℝ)} (h : P.card = 5) :
  max_similar_triangles P = 8 :=
sorry

end NUMINAMATH_GPT_max_similar_triangles_five_points_l1487_148711


namespace NUMINAMATH_GPT_max_value_of_4x_plus_3y_l1487_148736

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end NUMINAMATH_GPT_max_value_of_4x_plus_3y_l1487_148736


namespace NUMINAMATH_GPT_positive_multiples_of_4_with_units_digit_4_l1487_148754

theorem positive_multiples_of_4_with_units_digit_4 (n : ℕ) : 
  ∃ n ≤ 15, ∀ m, m = 4 + 10 * (n - 1) → m < 150 ∧ m % 10 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_multiples_of_4_with_units_digit_4_l1487_148754


namespace NUMINAMATH_GPT_sum_of_series_l1487_148755

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1487_148755


namespace NUMINAMATH_GPT_train_time_first_platform_correct_l1487_148723

-- Definitions
variables (L_train L_first_plat L_second_plat : ℕ) (T_second : ℕ) (T_first : ℕ)

-- Given conditions
def length_train := 350
def length_first_platform := 100
def length_second_platform := 250
def time_second_platform := 20
def expected_time_first_platform := 15

-- Derived values
def total_distance_second_platform := length_train + length_second_platform
def speed := total_distance_second_platform / time_second_platform
def total_distance_first_platform := length_train + length_first_platform
def time_first_platform := total_distance_first_platform / speed

-- Proof Statement
theorem train_time_first_platform_correct : 
  time_first_platform = expected_time_first_platform :=
  by
  sorry

end NUMINAMATH_GPT_train_time_first_platform_correct_l1487_148723


namespace NUMINAMATH_GPT_number_of_technicians_l1487_148769

/-- 
In a workshop, the average salary of all the workers is Rs. 8000. 
The average salary of some technicians is Rs. 12000 and the average salary of the rest is Rs. 6000. 
The total number of workers in the workshop is 24.
Prove that there are 8 technicians in the workshop.
-/
theorem number_of_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (num_technicians rest_workers : ℕ) 
  (h_total : total_workers = num_technicians + rest_workers)
  (h_avg_salary : (num_technicians * avg_salary_technicians + rest_workers * avg_salary_rest) = total_workers * avg_salary_all)
  (h1 : total_workers = 24)
  (h2 : avg_salary_all = 8000)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  num_technicians = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_technicians_l1487_148769


namespace NUMINAMATH_GPT_min_handshakes_35_people_l1487_148722

theorem min_handshakes_35_people (n : ℕ) (h1 : n = 35) (h2 : ∀ p : ℕ, p < n → p ≥ 3) : ∃ m : ℕ, m = 51 :=
by
  sorry

end NUMINAMATH_GPT_min_handshakes_35_people_l1487_148722


namespace NUMINAMATH_GPT_inequality_proof_l1487_148799

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1487_148799


namespace NUMINAMATH_GPT_find_k_l1487_148789

-- Define the equation of line m
def line_m (x : ℝ) : ℝ := 2 * x + 8

-- Define the equation of line n with an unknown slope k
def line_n (k : ℝ) (x : ℝ) : ℝ := k * x - 9

-- Define the point of intersection
def intersection_point := (-4, 0)

-- The proof statement
theorem find_k : ∃ k : ℝ, k = -9 / 4 ∧ line_m (-4) = 0 ∧ line_n k (-4) = 0 :=
by
  exists (-9 / 4)
  simp [line_m, line_n, intersection_point]
  sorry

end NUMINAMATH_GPT_find_k_l1487_148789


namespace NUMINAMATH_GPT_cristian_cookie_problem_l1487_148760

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end NUMINAMATH_GPT_cristian_cookie_problem_l1487_148760


namespace NUMINAMATH_GPT_num_floors_each_building_l1487_148793

theorem num_floors_each_building
  (floors_each_building num_apartments_per_floor num_doors_per_apartment total_doors : ℕ)
  (h1 : floors_each_building = F)
  (h2 : num_apartments_per_floor = 6)
  (h3 : num_doors_per_apartment = 7)
  (h4 : total_doors = 1008)
  (eq1 : 2 * floors_each_building * num_apartments_per_floor * num_doors_per_apartment = total_doors) :
  F = 12 :=
sorry

end NUMINAMATH_GPT_num_floors_each_building_l1487_148793


namespace NUMINAMATH_GPT_average_age_combined_rooms_l1487_148745

theorem average_age_combined_rooms :
  (8 * 30 + 5 * 22) / (8 + 5) = 26.9 := by
  sorry

end NUMINAMATH_GPT_average_age_combined_rooms_l1487_148745


namespace NUMINAMATH_GPT_part1_l1487_148746

def f (x : ℝ) := x^2 - 2*x

theorem part1 (x : ℝ) :
  (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ 3 ≤ x ∨ x = 0) :=
sorry

end NUMINAMATH_GPT_part1_l1487_148746
