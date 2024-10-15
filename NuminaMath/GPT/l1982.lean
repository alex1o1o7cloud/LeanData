import Mathlib

namespace NUMINAMATH_GPT_scientific_notation_of_coronavirus_diameter_l1982_198223

theorem scientific_notation_of_coronavirus_diameter:
  0.000000907 = 9.07 * 10^(-7) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_coronavirus_diameter_l1982_198223


namespace NUMINAMATH_GPT_find_m_l1982_198262

-- Define the vectors a and b
def veca (m : ℝ) : ℝ × ℝ := (m, 4)
def vecb (m : ℝ) : ℝ × ℝ := (m + 4, 1)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition that the dot product of a and b is zero
def are_perpendicular (m : ℝ) : Prop :=
  dot_product (veca m) (vecb m) = 0

-- The goal is to prove that if a and b are perpendicular, then m = -2
theorem find_m (m : ℝ) (h : are_perpendicular m) : m = -2 :=
by {
  -- Proof will be filled here
  sorry
}

end NUMINAMATH_GPT_find_m_l1982_198262


namespace NUMINAMATH_GPT_sequence_divisibility_24_l1982_198203

theorem sequence_divisibility_24 :
  ∀ (x : ℕ → ℕ), (x 0 = 2) → (x 1 = 3) →
    (∀ n : ℕ, x (n+2) = 7 * x (n+1) - x n + 280) →
    (∀ n : ℕ, (x n * x (n+1) + x (n+1) * x (n+2) + x (n+2) * x (n+3) + 2018) % 24 = 0) :=
by
  intro x h1 h2 h3
  sorry

end NUMINAMATH_GPT_sequence_divisibility_24_l1982_198203


namespace NUMINAMATH_GPT_range_of_BC_in_triangle_l1982_198207

theorem range_of_BC_in_triangle 
  (A B C : ℝ) 
  (a c BC : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : a * Real.cos C = c * Real.sin A)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : BC = 2 * Real.sin A)
  (h5 : ∃ A1 A2, 0 < A1 ∧ A1 < Real.pi / 2 ∧ Real.pi / 2 < A2 ∧ A2 < Real.pi ∧ Real.sin A = Real.sin A1 ∧ Real.sin A = Real.sin A2)
  : BC ∈ Set.Ioo (Real.sqrt 2) 2 :=
sorry

end NUMINAMATH_GPT_range_of_BC_in_triangle_l1982_198207


namespace NUMINAMATH_GPT_find_n_l1982_198294

theorem find_n (n : ℕ) : (1/5)^35 * (1/4)^18 = 1/(n*(10)^35) → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1982_198294


namespace NUMINAMATH_GPT_composite_sum_of_ab_l1982_198287

theorem composite_sum_of_ab (a b : ℕ) (h : 31 * a = 54 * b) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a + b = k * l :=
sorry

end NUMINAMATH_GPT_composite_sum_of_ab_l1982_198287


namespace NUMINAMATH_GPT_cauchy_schwarz_example_l1982_198246

theorem cauchy_schwarz_example (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end NUMINAMATH_GPT_cauchy_schwarz_example_l1982_198246


namespace NUMINAMATH_GPT_best_representation_is_B_l1982_198212

-- Define the conditions
structure Trip :=
  (home_to_diner : ℝ)
  (diner_stop : ℝ)
  (diner_to_highway : ℝ)
  (highway_to_mall : ℝ)
  (mall_stop : ℝ)
  (highway_return : ℝ)
  (construction_zone : ℝ)
  (return_city_traffic : ℝ)

-- Graph description
inductive Graph
| plateau : Graph
| increasing : Graph → Graph
| decreasing : Graph → Graph

-- Condition that describes the pattern of the graph
def correct_graph (trip : Trip) : Prop :=
  let d1 := trip.home_to_diner
  let d2 := trip.diner_stop
  let d3 := trip.diner_to_highway
  let d4 := trip.highway_to_mall
  let d5 := trip.mall_stop
  let d6 := trip.highway_return
  let d7 := trip.construction_zone
  let d8 := trip.return_city_traffic
  d1 > 0 ∧ d2 = 0 ∧ d3 > 0 ∧ d4 > 0 ∧ d5 = 0 ∧ d6 < 0 ∧ d7 < 0 ∧ d8 < 0

-- Theorem statement
theorem best_representation_is_B (trip : Trip) : correct_graph trip :=
by sorry

end NUMINAMATH_GPT_best_representation_is_B_l1982_198212


namespace NUMINAMATH_GPT_find_n_l1982_198277

theorem find_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28) : n = 27 :=
sorry

end NUMINAMATH_GPT_find_n_l1982_198277


namespace NUMINAMATH_GPT_prism_height_l1982_198284

theorem prism_height (a h : ℝ) 
  (base_side : a = 10) 
  (total_edge_length : 3 * a + 3 * a + 3 * h = 84) : 
  h = 8 :=
by sorry

end NUMINAMATH_GPT_prism_height_l1982_198284


namespace NUMINAMATH_GPT_cost_difference_l1982_198224

-- Define the costs
def cost_chocolate : ℕ := 3
def cost_candy_bar : ℕ := 7

-- Define the difference to be proved
theorem cost_difference :
  cost_candy_bar - cost_chocolate = 4 :=
by
  -- trivial proof steps
  sorry

end NUMINAMATH_GPT_cost_difference_l1982_198224


namespace NUMINAMATH_GPT_factor_polynomial_l1982_198241

theorem factor_polynomial : ∀ y : ℝ, 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) :=
by
  intros y
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1982_198241


namespace NUMINAMATH_GPT_gcd_polynomial_l1982_198251

theorem gcd_polynomial (a : ℕ) (h : 270 ∣ a) : Nat.gcd (5 * a^3 + 3 * a^2 + 5 * a + 45) a = 45 :=
sorry

end NUMINAMATH_GPT_gcd_polynomial_l1982_198251


namespace NUMINAMATH_GPT_shelves_used_l1982_198289

def initial_books : Nat := 87
def sold_books : Nat := 33
def books_per_shelf : Nat := 6

theorem shelves_used :
  (initial_books - sold_books) / books_per_shelf = 9 := by
  sorry

end NUMINAMATH_GPT_shelves_used_l1982_198289


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l1982_198266

variable (d : ℤ)
variable (h : d ≠ 0)

theorem polynomial_coeff_sum : 
  (∃ a b c e : ℤ, (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e ∧ a + b + c + e = 42) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l1982_198266


namespace NUMINAMATH_GPT_shaded_area_ratio_l1982_198220

theorem shaded_area_ratio
  (large_square_area : ℕ := 25)
  (grid_dimension : ℕ := 5)
  (shaded_square_area : ℕ := 2)
  (num_squares : ℕ := 25)
  (ratio : ℚ := 2 / 25) :
  (shaded_square_area : ℚ) / large_square_area = ratio := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_ratio_l1982_198220


namespace NUMINAMATH_GPT_add_fractions_result_l1982_198222

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end NUMINAMATH_GPT_add_fractions_result_l1982_198222


namespace NUMINAMATH_GPT_average_upstream_speed_l1982_198228

/--
There are three boats moving down a river. Boat A moves downstream at a speed of 1 km in 4 minutes 
and upstream at a speed of 1 km in 8 minutes. Boat B moves downstream at a speed of 1 km in 
5 minutes and upstream at a speed of 1 km in 11 minutes. Boat C moves downstream at a speed of 
1 km in 6 minutes and upstream at a speed of 1 km in 10 minutes. Prove that the average speed 
of the boats against the current is 6.32 km/h.
-/
theorem average_upstream_speed :
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  average_speed = 6.32 :=
by
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  sorry

end NUMINAMATH_GPT_average_upstream_speed_l1982_198228


namespace NUMINAMATH_GPT_solution_triples_l1982_198238

noncomputable def find_triples (x y z : ℝ) : Prop :=
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  (1/x) + (1/y) + (1/z) = 1/2008

theorem solution_triples :
  ∃ (x y z : ℝ), find_triples x y z ∧ (x = 2008 ∧ y = 4016 ∧ z = -4016) :=
sorry

end NUMINAMATH_GPT_solution_triples_l1982_198238


namespace NUMINAMATH_GPT_problem_equiv_l1982_198256

theorem problem_equiv :
  ((2001 * 2021 + 100) * (1991 * 2031 + 400)) / (2011^4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_equiv_l1982_198256


namespace NUMINAMATH_GPT_total_songs_performed_l1982_198215

theorem total_songs_performed (lucy_songs : ℕ) (sarah_songs : ℕ) (beth_songs : ℕ) (jane_songs : ℕ) 
  (h1 : lucy_songs = 8)
  (h2 : sarah_songs = 5)
  (h3 : sarah_songs < beth_songs)
  (h4 : sarah_songs < jane_songs)
  (h5 : beth_songs < lucy_songs)
  (h6 : jane_songs < lucy_songs)
  (h7 : beth_songs = 6 ∨ beth_songs = 7)
  (h8 : jane_songs = 6 ∨ jane_songs = 7) :
  (lucy_songs + sarah_songs + beth_songs + jane_songs) / 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_songs_performed_l1982_198215


namespace NUMINAMATH_GPT_eggs_sally_bought_is_correct_l1982_198208

def dozen := 12

def eggs_sally_bought (dozens : Nat) : Nat :=
  dozens * dozen

theorem eggs_sally_bought_is_correct :
  eggs_sally_bought 4 = 48 :=
by
  sorry

end NUMINAMATH_GPT_eggs_sally_bought_is_correct_l1982_198208


namespace NUMINAMATH_GPT_bricks_needed_l1982_198205

theorem bricks_needed 
    (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
    (wall_length_m : ℝ) (wall_height_m : ℝ) (wall_width_cm : ℝ)
    (H1 : brick_length = 25) (H2 : brick_width = 11.25) (H3 : brick_height = 6)
    (H4 : wall_length_m = 7) (H5 : wall_height_m = 6) (H6 : wall_width_cm = 22.5) :
    (wall_length_m * 100 * wall_height_m * 100 * wall_width_cm) / (brick_length * brick_width * brick_height) = 5600 :=
by
    sorry

end NUMINAMATH_GPT_bricks_needed_l1982_198205


namespace NUMINAMATH_GPT_simplify_expression_l1982_198209

theorem simplify_expression (b : ℝ) (h : b ≠ -1) : 
  1 - (1 / (1 - (b / (1 + b)))) = -b :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l1982_198209


namespace NUMINAMATH_GPT_polynomial_division_remainder_zero_l1982_198234

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end NUMINAMATH_GPT_polynomial_division_remainder_zero_l1982_198234


namespace NUMINAMATH_GPT_fraction_one_two_three_sum_l1982_198244

def fraction_one_bedroom : ℝ := 0.12
def fraction_two_bedroom : ℝ := 0.26
def fraction_three_bedroom : ℝ := 0.38
def fraction_four_bedroom : ℝ := 0.24

theorem fraction_one_two_three_sum :
  fraction_one_bedroom + fraction_two_bedroom + fraction_three_bedroom = 0.76 :=
by
  sorry

end NUMINAMATH_GPT_fraction_one_two_three_sum_l1982_198244


namespace NUMINAMATH_GPT_fractions_non_integer_l1982_198232

theorem fractions_non_integer (a b c d : ℤ) : 
  ∃ (a b c d : ℤ), 
    ¬((a-b) % 2 = 0 ∧ 
      (b-c) % 2 = 0 ∧ 
      (c-d) % 2 = 0 ∧ 
      (d-a) % 2 = 0) :=
sorry

end NUMINAMATH_GPT_fractions_non_integer_l1982_198232


namespace NUMINAMATH_GPT_triangle_d_not_right_l1982_198265

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_d_not_right :
  ¬is_right_triangle 7 8 13 :=
by sorry

end NUMINAMATH_GPT_triangle_d_not_right_l1982_198265


namespace NUMINAMATH_GPT_train_speed_km_per_hr_l1982_198242

-- Definitions for the conditions
def length_of_train_meters : ℕ := 250
def time_to_cross_pole_seconds : ℕ := 10

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℚ := m / 1000
def seconds_to_hours (s : ℕ) : ℚ := s / 3600

-- Theorem stating that the speed of the train is 90 km/hr
theorem train_speed_km_per_hr : 
  meters_to_kilometers length_of_train_meters / seconds_to_hours time_to_cross_pole_seconds = 90 := 
by 
  -- We skip the actual proof with sorry
  sorry

end NUMINAMATH_GPT_train_speed_km_per_hr_l1982_198242


namespace NUMINAMATH_GPT_powers_of_i_cyclic_l1982_198253

theorem powers_of_i_cyclic {i : ℂ} (h_i_squared : i^2 = -1) :
  i^(66) + i^(103) = -1 - i :=
by {
  -- Providing the proof steps as sorry.
  -- This is a placeholder for the actual proof.
  sorry
}

end NUMINAMATH_GPT_powers_of_i_cyclic_l1982_198253


namespace NUMINAMATH_GPT_permutations_behind_Alice_l1982_198247

theorem permutations_behind_Alice (n : ℕ) (h : n = 7) : 
  (Nat.factorial n) = 5040 :=
by
  rw [h]
  rw [Nat.factorial]
  sorry

end NUMINAMATH_GPT_permutations_behind_Alice_l1982_198247


namespace NUMINAMATH_GPT_range_of_p_l1982_198227

-- Conditions: p is a prime number and the roots of the quadratic equation are integers 
def p_is_prime (p : ℕ) : Prop := Nat.Prime p

def roots_are_integers (p : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x * y = -204 * p ∧ (x + y) = p

-- Main statement: Prove the range of p
theorem range_of_p (p : ℕ) (hp : p_is_prime p) (hr : roots_are_integers p) : 11 < p ∧ p ≤ 21 :=
  sorry

end NUMINAMATH_GPT_range_of_p_l1982_198227


namespace NUMINAMATH_GPT_x_quad_greater_l1982_198226

theorem x_quad_greater (x : ℝ) : x^4 > x - 1/2 :=
sorry

end NUMINAMATH_GPT_x_quad_greater_l1982_198226


namespace NUMINAMATH_GPT_find_x_l1982_198275

-- Define the given conditions
def constant_ratio (k : ℚ) : Prop :=
  ∀ (x y : ℚ), (3 * x - 4) / (y + 15) = k

def initial_condition (k : ℚ) : Prop :=
  (3 * 5 - 4) / (4 + 15) = k

def new_condition (k : ℚ) (x : ℚ) : Prop :=
  (3 * x - 4) / 30 = k

-- Prove that x = 406/57 given the conditions
theorem find_x (k : ℚ) (x : ℚ) :
  constant_ratio k →
  initial_condition k →
  new_condition k x →
  x = 406 / 57 :=
  sorry

end NUMINAMATH_GPT_find_x_l1982_198275


namespace NUMINAMATH_GPT_water_level_decrease_l1982_198254

theorem water_level_decrease (increase_notation : ℝ) (h : increase_notation = 2) :
  -increase_notation = -2 :=
by
  sorry

end NUMINAMATH_GPT_water_level_decrease_l1982_198254


namespace NUMINAMATH_GPT_value_of_expression_l1982_198216

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1982_198216


namespace NUMINAMATH_GPT_geometric_seq_sum_first_4_terms_l1982_198219

theorem geometric_seq_sum_first_4_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * 2)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 4 = 15 :=
by
  -- The actual proof would go here.
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_first_4_terms_l1982_198219


namespace NUMINAMATH_GPT_find_y_l1982_198237

theorem find_y (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 :=
by sorry

end NUMINAMATH_GPT_find_y_l1982_198237


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l1982_198243

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l1982_198243


namespace NUMINAMATH_GPT_symmetric_intersection_range_l1982_198286

theorem symmetric_intersection_range (k m p : ℝ)
  (intersection_symmetric : ∀ (x y : ℝ), 
    (x = k*y - 1 ∧ (x^2 + y^2 + k*x + m*y + 2*p = 0)) → 
    (y = x)) 
  : p < -3/2 := 
sorry

end NUMINAMATH_GPT_symmetric_intersection_range_l1982_198286


namespace NUMINAMATH_GPT_percentage_less_than_l1982_198295

theorem percentage_less_than (x y z : Real) (h1 : x = 1.20 * y) (h2 : x = 0.84 * z) : 
  ((z - y) / z) * 100 = 30 := 
sorry

end NUMINAMATH_GPT_percentage_less_than_l1982_198295


namespace NUMINAMATH_GPT_angle_between_bisectors_of_trihedral_angle_l1982_198250

noncomputable def angle_between_bisectors_trihedral (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) : ℝ :=
  60

theorem angle_between_bisectors_of_trihedral_angle (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) :
  angle_between_bisectors_trihedral α β γ hα hβ hγ = 60 := 
sorry

end NUMINAMATH_GPT_angle_between_bisectors_of_trihedral_angle_l1982_198250


namespace NUMINAMATH_GPT_translated_graph_symmetric_l1982_198293

noncomputable def f (x : ℝ) : ℝ := sorry

theorem translated_graph_symmetric (f : ℝ → ℝ)
  (h_translate : ∀ x, f (x - 1) = e^x)
  (h_symmetric : ∀ x, f x = f (-x)) :
  ∀ x, f x = e^(-x - 1) :=
by
  sorry

end NUMINAMATH_GPT_translated_graph_symmetric_l1982_198293


namespace NUMINAMATH_GPT_probability_is_one_twelfth_l1982_198283

def probability_red_gt4_green_odd_blue_lt4 : ℚ :=
  let total_outcomes := 6 * 6 * 6
  let successful_outcomes := 2 * 3 * 3
  successful_outcomes / total_outcomes

theorem probability_is_one_twelfth :
  probability_red_gt4_green_odd_blue_lt4 = 1 / 12 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_probability_is_one_twelfth_l1982_198283


namespace NUMINAMATH_GPT_solve_for_x_minus_y_l1982_198201

theorem solve_for_x_minus_y (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 24) : x - y = 4 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_minus_y_l1982_198201


namespace NUMINAMATH_GPT_sum_p_q_l1982_198271

-- Define the cubic polynomial q(x)
def cubic_q (q : ℚ) (x : ℚ) := q * x * (x - 1) * (x + 1)

-- Define the linear polynomial p(x)
def linear_p (p : ℚ) (x : ℚ) := p * x

-- Prove the result for p(x) + q(x)
theorem sum_p_q : 
  (∀ p q : ℚ, linear_p p 4 = 4 → cubic_q q 3 = 3 → (∀ x : ℚ, linear_p p x + cubic_q q x = (1 / 24) * x^3 + (23 / 24) * x)) :=
by
  intros p q hp hq x
  sorry

end NUMINAMATH_GPT_sum_p_q_l1982_198271


namespace NUMINAMATH_GPT_blue_square_area_percentage_l1982_198240

theorem blue_square_area_percentage (k : ℝ) (H1 : 0 < k) 
(Flag_area : ℝ := k^2) -- total area of the flag
(Cross_area : ℝ := 0.49 * Flag_area) -- total area of the cross and blue squares 
(one_blue_square_area : ℝ := Cross_area / 3) -- area of one blue square
(percentage : ℝ := one_blue_square_area / Flag_area * 100) :
percentage = 16.33 :=
by
  sorry

end NUMINAMATH_GPT_blue_square_area_percentage_l1982_198240


namespace NUMINAMATH_GPT_range_m_inequality_l1982_198231

theorem range_m_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 * Real.exp x < m) ↔ m > Real.exp 1 := 
  by
    sorry

end NUMINAMATH_GPT_range_m_inequality_l1982_198231


namespace NUMINAMATH_GPT_find_center_of_circle_l1982_198281

noncomputable def center_of_circle (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 + 4 * y = 16

theorem find_center_of_circle (x y : ℝ) (h : center_of_circle x y) : (x, y) = (4, -2) :=
by 
  sorry

end NUMINAMATH_GPT_find_center_of_circle_l1982_198281


namespace NUMINAMATH_GPT_no_rectangle_from_six_different_squares_l1982_198255

theorem no_rectangle_from_six_different_squares (a1 a2 a3 a4 a5 a6 : ℝ) (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) :
  ¬ (∃ (L W : ℝ), a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = L * W) :=
sorry

end NUMINAMATH_GPT_no_rectangle_from_six_different_squares_l1982_198255


namespace NUMINAMATH_GPT_no_unique_day_in_august_l1982_198204

def july_has_five_tuesdays (N : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30

def july_august_have_30_days (N : ℕ) : Prop :=
  true -- We're asserting this unconditionally since both months have exactly 30 days in the problem

theorem no_unique_day_in_august (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : july_august_have_30_days N) :
  ¬(∃ d : ℕ, ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30 ∧ ∃! wday : ℕ, (d + k * 7 + wday) % 7 = 0) :=
sorry

end NUMINAMATH_GPT_no_unique_day_in_august_l1982_198204


namespace NUMINAMATH_GPT_extreme_points_inequality_l1982_198291

noncomputable def f (a x : ℝ) : ℝ := a * x - (a / x) - 2 * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := (a * x^2 - 2 * x + a) / x^2

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : a > 0) (h2 : 1 < x1) (h3 : x1 < Real.exp 1)
  (h4 : f a x1 = 0) (h5 : f a x2 = 0) (h6 : x1 ≠ x2) : |f a x1 - f a x2| < 1 :=
by
  sorry

end NUMINAMATH_GPT_extreme_points_inequality_l1982_198291


namespace NUMINAMATH_GPT_remaining_movie_time_l1982_198211

def start_time := 200 -- represents 3:20 pm in total minutes from midnight
def end_time := 350 -- represents 5:44 pm in total minutes from midnight
def total_movie_duration := 180 -- 3 hours in minutes

theorem remaining_movie_time : total_movie_duration - (end_time - start_time) = 36 :=
by
  sorry

end NUMINAMATH_GPT_remaining_movie_time_l1982_198211


namespace NUMINAMATH_GPT_cosine_midline_l1982_198230

theorem cosine_midline (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_range : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) : 
  d = 3 := 
by 
  sorry

end NUMINAMATH_GPT_cosine_midline_l1982_198230


namespace NUMINAMATH_GPT_slices_per_large_pizza_l1982_198239

theorem slices_per_large_pizza (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ) 
  (H1 : total_pizzas = 2) (H2 : slices_eaten = 7) (H3 : slices_remaining = 9) : 
  (slices_remaining + slices_eaten) / total_pizzas = 8 := 
by
  sorry

end NUMINAMATH_GPT_slices_per_large_pizza_l1982_198239


namespace NUMINAMATH_GPT_cost_per_pound_beef_is_correct_l1982_198206

variable (budget initial_chicken_cost pounds_beef remaining_budget_after_purchase : ℝ)
variable (spending_on_beef cost_per_pound_beef : ℝ)

axiom h1 : budget = 80
axiom h2 : initial_chicken_cost = 12
axiom h3 : pounds_beef = 5
axiom h4 : remaining_budget_after_purchase = 53
axiom h5 : spending_on_beef = budget - initial_chicken_cost - remaining_budget_after_purchase
axiom h6 : cost_per_pound_beef = spending_on_beef / pounds_beef

theorem cost_per_pound_beef_is_correct : cost_per_pound_beef = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_pound_beef_is_correct_l1982_198206


namespace NUMINAMATH_GPT_find_A_students_l1982_198276

variables (Alan Beth Carlos Diana : Prop)
variable (num_As : ℕ)

def Alan_implies_Beth := Alan → Beth
def Beth_implies_no_Carlos_A := Beth → ¬Carlos
def Carlos_implies_Diana := Carlos → Diana
def Beth_implies_Diana := Beth → Diana

theorem find_A_students 
  (h1 : Alan_implies_Beth Alan Beth)
  (h2 : Beth_implies_no_Carlos_A Beth Carlos)
  (h3 : Carlos_implies_Diana Carlos Diana)
  (h4 : Beth_implies_Diana Beth Diana)
  (h_cond : num_As = 2) :
  (Alan ∧ Beth) ∨ (Beth ∧ Diana) ∨ (Carlos ∧ Diana) :=
by sorry

end NUMINAMATH_GPT_find_A_students_l1982_198276


namespace NUMINAMATH_GPT_triangle_balls_l1982_198245

theorem triangle_balls (n : ℕ) (num_tri_balls : ℕ) (num_sq_balls : ℕ) :
  (∀ n : ℕ, num_tri_balls = n * (n + 1) / 2)
  ∧ (num_sq_balls = num_tri_balls + 424)
  ∧ (∀ s : ℕ, s = n - 8 → s * s = num_sq_balls)
  → num_tri_balls = 820 :=
by sorry

end NUMINAMATH_GPT_triangle_balls_l1982_198245


namespace NUMINAMATH_GPT_avg_goals_l1982_198290

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end NUMINAMATH_GPT_avg_goals_l1982_198290


namespace NUMINAMATH_GPT_power_equality_l1982_198278

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_power_equality_l1982_198278


namespace NUMINAMATH_GPT_total_surface_area_is_correct_l1982_198268

-- Define the problem constants and structure
def num_cubes := 20
def edge_length := 1
def bottom_layer := 9
def middle_layer := 8
def top_layer := 3
def total_painted_area : ℕ := 55

-- Define a function to calculate the exposed surface area
noncomputable def calc_exposed_area (num_bottom : ℕ) (num_middle : ℕ) (num_top : ℕ) (edge : ℕ) : ℕ := 
    let bottom_exposed := num_bottom * (edge * edge)
    let middle_corners_exposed := 4 * 3 * edge
    let middle_edges_exposed := (num_middle - 4) * (2 * edge)
    let top_exposed := num_top * (5 * edge)
    bottom_exposed + middle_corners_exposed + middle_edges_exposed + top_exposed

-- Statement to prove the total painted area
theorem total_surface_area_is_correct : calc_exposed_area bottom_layer middle_layer top_layer edge_length = total_painted_area :=
by
  -- The proof itself is omitted, focus is on the statement.
  sorry

end NUMINAMATH_GPT_total_surface_area_is_correct_l1982_198268


namespace NUMINAMATH_GPT_total_dolls_l1982_198202

def sisters_dolls : ℝ := 8.5

def hannahs_dolls : ℝ := 5.5 * sisters_dolls

theorem total_dolls : hannahs_dolls + sisters_dolls = 55.25 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_dolls_l1982_198202


namespace NUMINAMATH_GPT_certain_event_C_union_D_l1982_198296

variable {Ω : Type} -- Omega, the sample space
variable {P : Set Ω → Prop} -- P as the probability function predicates the events

-- Definitions of the events
variable {A B C D : Set Ω}

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ x, x ∈ A → x ∉ B
def complementary (A C : Set Ω) : Prop := ∀ x, x ∈ C ↔ x ∉ A

-- Given conditions
axiom A_and_B_mutually_exclusive : mutually_exclusive A B
axiom C_is_complementary_to_A : complementary A C
axiom D_is_complementary_to_B : complementary B D

-- Theorem statement
theorem certain_event_C_union_D : ∀ x, x ∈ C ∪ D := by
  sorry

end NUMINAMATH_GPT_certain_event_C_union_D_l1982_198296


namespace NUMINAMATH_GPT_min_knights_l1982_198210

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end NUMINAMATH_GPT_min_knights_l1982_198210


namespace NUMINAMATH_GPT_average_throws_to_lasso_l1982_198285

theorem average_throws_to_lasso (p : ℝ) (h₁ : 1 - (1 - p)^3 = 0.875) : (1 / p) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_throws_to_lasso_l1982_198285


namespace NUMINAMATH_GPT_sum_345_consecutive_sequences_l1982_198297

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end NUMINAMATH_GPT_sum_345_consecutive_sequences_l1982_198297


namespace NUMINAMATH_GPT_esther_walks_975_yards_l1982_198235

def miles_to_feet (miles : ℕ) : ℕ := miles * 5280
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

variable (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) (esther_yards : ℕ)
variable (h_lionel : lionel_miles = 4)
variable (h_niklaus : niklaus_feet = 1287)
variable (h_total : total_feet = 25332)
variable (h_esther : esther_yards = 975)

theorem esther_walks_975_yards :
  let lionel_distance_in_feet := miles_to_feet lionel_miles
  let combined_distance := lionel_distance_in_feet + niklaus_feet
  let esther_distance_in_feet := total_feet - combined_distance
  feet_to_yards esther_distance_in_feet = esther_yards := by {
    sorry
  }

end NUMINAMATH_GPT_esther_walks_975_yards_l1982_198235


namespace NUMINAMATH_GPT_shelves_filled_l1982_198225

theorem shelves_filled (total_teddy_bears teddy_bears_per_shelf : ℕ) (h1 : total_teddy_bears = 98) (h2 : teddy_bears_per_shelf = 7) : 
  total_teddy_bears / teddy_bears_per_shelf = 14 := 
by 
  sorry

end NUMINAMATH_GPT_shelves_filled_l1982_198225


namespace NUMINAMATH_GPT_increasing_function_odd_function_l1982_198258

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem increasing_function (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

theorem odd_function (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) ↔ a = 1 :=
sorry

end NUMINAMATH_GPT_increasing_function_odd_function_l1982_198258


namespace NUMINAMATH_GPT_savings_correct_l1982_198280

-- Define the conditions
def in_store_price : ℝ := 320
def discount_rate : ℝ := 0.05
def monthly_payment : ℝ := 62
def monthly_payments : ℕ := 5
def shipping_handling : ℝ := 10

-- Prove that the savings from buying in-store is 16 dollars.
theorem savings_correct : 
  (monthly_payments * monthly_payment + shipping_handling) - (in_store_price * (1 - discount_rate)) = 16 := 
by
  sorry

end NUMINAMATH_GPT_savings_correct_l1982_198280


namespace NUMINAMATH_GPT_hats_count_l1982_198272

theorem hats_count (T M W : ℕ) (hT : T = 1800)
  (hM : M = (2 * T) / 3) (hW : W = T - M) 
  (hats_men : ℕ) (hats_women : ℕ) (m_hats_condition : hats_men = 15 * M / 100)
  (w_hats_condition : hats_women = 25 * W / 100) :
  hats_men + hats_women = 330 :=
by sorry

end NUMINAMATH_GPT_hats_count_l1982_198272


namespace NUMINAMATH_GPT_pages_left_to_write_l1982_198264

theorem pages_left_to_write : 
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  remaining_pages = 315 :=
by
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  show remaining_pages = 315
  sorry

end NUMINAMATH_GPT_pages_left_to_write_l1982_198264


namespace NUMINAMATH_GPT_parallelogram_height_l1982_198292

theorem parallelogram_height (A b h : ℝ) (hA : A = 288) (hb : b = 18) : h = 16 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_height_l1982_198292


namespace NUMINAMATH_GPT_angles_arithmetic_sequence_sides_l1982_198288

theorem angles_arithmetic_sequence_sides (A B C a b c : ℝ)
  (h_angle_ABC : A + B + C = 180)
  (h_arithmetic_sequence : 2 * B = A + C)
  (h_cos_B : A * A + c * c - b * b = 2 * a * c)
  (angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < 180 ∧ B < 180 ∧ C < 180) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end NUMINAMATH_GPT_angles_arithmetic_sequence_sides_l1982_198288


namespace NUMINAMATH_GPT_percent_pelicans_non_swans_l1982_198218

noncomputable def percent_geese := 0.20
noncomputable def percent_swans := 0.30
noncomputable def percent_herons := 0.10
noncomputable def percent_ducks := 0.25
noncomputable def percent_pelicans := 0.15

theorem percent_pelicans_non_swans :
  (percent_pelicans / (1 - percent_swans)) * 100 = 21.43 := 
by 
  sorry

end NUMINAMATH_GPT_percent_pelicans_non_swans_l1982_198218


namespace NUMINAMATH_GPT_self_employed_tax_amount_l1982_198248

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end NUMINAMATH_GPT_self_employed_tax_amount_l1982_198248


namespace NUMINAMATH_GPT_largest_four_digit_by_two_moves_l1982_198282

def moves (n : Nat) (d1 d2 d3 d4 : Nat) : Prop :=
  ∃ x y : ℕ, d1 = x → d2 = y → n = 1405 → (x ≤ 2 ∧ y ≤ 2)

theorem largest_four_digit_by_two_moves :
  ∃ n : ℕ, moves 1405 1 4 0 5 ∧ n = 7705 :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_by_two_moves_l1982_198282


namespace NUMINAMATH_GPT_min_beans_l1982_198267

theorem min_beans (r b : ℕ) (H1 : r ≥ 3 + 2 * b) (H2 : r ≤ 3 * b) : b ≥ 3 := 
sorry

end NUMINAMATH_GPT_min_beans_l1982_198267


namespace NUMINAMATH_GPT_length_of_AC_l1982_198298

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end NUMINAMATH_GPT_length_of_AC_l1982_198298


namespace NUMINAMATH_GPT_gain_percent_l1982_198229

variables (MP CP SP : ℝ)

-- problem conditions
axiom h1 : CP = 0.64 * MP
axiom h2 : SP = 0.84 * MP

-- To prove: Gain percent is 31.25%
theorem gain_percent (CP MP SP : ℝ) (h1 : CP = 0.64 * MP) (h2 : SP = 0.84 * MP) :
  ((SP - CP) / CP) * 100 = 31.25 :=
by sorry

end NUMINAMATH_GPT_gain_percent_l1982_198229


namespace NUMINAMATH_GPT_problem1_problem2_l1982_198270

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1982_198270


namespace NUMINAMATH_GPT_part1_part2_l1982_198221

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem part1 (a b : ℝ) (h1 : f a b 1 = 8) : a + b = 2 := by
  rw [f] at h1
  sorry

theorem part2 (a b : ℝ) (h1 : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  rw [f] at h1
  sorry

end NUMINAMATH_GPT_part1_part2_l1982_198221


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1982_198261

theorem sufficient_but_not_necessary (a b : ℝ) (hp : a > 1 ∧ b > 1) (hq : a + b > 2 ∧ a * b > 1) : 
  (a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧ ¬(a + b > 2 ∧ a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1982_198261


namespace NUMINAMATH_GPT_average_age_is_26_l1982_198217

noncomputable def devin_age : ℕ := 12
noncomputable def eden_age : ℕ := 2 * devin_age
noncomputable def eden_mom_age : ℕ := 2 * eden_age
noncomputable def eden_grandfather_age : ℕ := (devin_age + eden_age + eden_mom_age) / 2
noncomputable def eden_aunt_age : ℕ := eden_mom_age / devin_age

theorem average_age_is_26 : 
  (devin_age + eden_age + eden_mom_age + eden_grandfather_age + eden_aunt_age) / 5 = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_age_is_26_l1982_198217


namespace NUMINAMATH_GPT_fish_caught_by_dad_l1982_198200

def total_fish_both : ℕ := 23
def fish_caught_morning : ℕ := 8
def fish_thrown_back : ℕ := 3
def fish_caught_afternoon : ℕ := 5
def fish_kept_brendan : ℕ := fish_caught_morning - fish_thrown_back + fish_caught_afternoon

theorem fish_caught_by_dad : total_fish_both - fish_kept_brendan = 13 := by
  sorry

end NUMINAMATH_GPT_fish_caught_by_dad_l1982_198200


namespace NUMINAMATH_GPT_equivalent_expression_l1982_198259

theorem equivalent_expression (m n : ℕ) (P Q : ℕ) (hP : P = 3^m) (hQ : Q = 5^n) :
  15^(m + n) = P * Q :=
by
  sorry

end NUMINAMATH_GPT_equivalent_expression_l1982_198259


namespace NUMINAMATH_GPT_no_solution_inequality_l1982_198213

theorem no_solution_inequality (m : ℝ) : ¬(∃ x : ℝ, 2 * x - 1 > 1 ∧ x < m) → m ≤ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_solution_inequality_l1982_198213


namespace NUMINAMATH_GPT_rectangle_width_is_16_l1982_198263

-- Definitions based on the conditions
def length : ℝ := 24
def ratio := 6 / 5
def perimeter := 80

-- The proposition to prove
theorem rectangle_width_is_16 (W : ℝ) (h1 : length = 24) (h2 : length = ratio * W) (h3 : 2 * length + 2 * W = perimeter) :
  W = 16 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_is_16_l1982_198263


namespace NUMINAMATH_GPT_symmetric_point_origin_l1982_198257

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) (O : Point) : Point :=
  let (x, y, z) := P
  let (ox, oy, oz) := O
  (2 * ox - x, 2 * oy - y, 2 * oz - z)

theorem symmetric_point_origin :
  symmetric_point (1, 3, 5) (0, 0, 0) = (-1, -3, -5) :=
by sorry

end NUMINAMATH_GPT_symmetric_point_origin_l1982_198257


namespace NUMINAMATH_GPT_juan_distance_l1982_198299

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end NUMINAMATH_GPT_juan_distance_l1982_198299


namespace NUMINAMATH_GPT_minimal_hair_loss_l1982_198236

theorem minimal_hair_loss (cards : Fin 100 → ℕ)
    (sum_sage1 : ℕ)
    (communicate_card_numbers : List ℕ)
    (communicate_sum : ℕ) :
    (∀ i : Fin 100, (communicate_card_numbers.contains (cards i))) →
    communicate_sum = sum_sage1 →
    sum_sage1 = List.sum communicate_card_numbers →
    communicate_card_numbers.length = 100 →
    ∃ (minimal_loss : ℕ), minimal_loss = 101 := by
  sorry

end NUMINAMATH_GPT_minimal_hair_loss_l1982_198236


namespace NUMINAMATH_GPT_like_terms_exp_l1982_198273

theorem like_terms_exp (a b : ℝ) (m n x : ℝ)
  (h₁ : 2 * a ^ x * b ^ (n + 1) = -3 * a * b ^ (2 * m))
  (h₂ : x = 1) (h₃ : n + 1 = 2 * m) : 
  (2 * m - n) ^ x = 1 := 
by
  sorry

end NUMINAMATH_GPT_like_terms_exp_l1982_198273


namespace NUMINAMATH_GPT_sequence_formula_l1982_198214

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 5)
  (h3 : ∀ n > 1, a (n + 1) = 2 * a n - a (n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1982_198214


namespace NUMINAMATH_GPT_julies_balls_after_1729_steps_l1982_198260

-- Define the process described
def increment_base_8 (n : ℕ) : List ℕ := 
by
  if n = 0 then
    exact [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 8) (n % 8 :: acc)
    exact loop n []

-- Define the total number of balls after 'steps' steps
def julies_total_balls (steps : ℕ) : ℕ :=
by 
  exact (increment_base_8 steps).sum

theorem julies_balls_after_1729_steps : julies_total_balls 1729 = 7 :=
by
  sorry

end NUMINAMATH_GPT_julies_balls_after_1729_steps_l1982_198260


namespace NUMINAMATH_GPT_water_increase_factor_l1982_198249

theorem water_increase_factor 
  (initial_koolaid : ℝ := 2) 
  (initial_water : ℝ := 16) 
  (evaporated_water : ℝ := 4) 
  (final_koolaid_percentage : ℝ := 4) : 
  (initial_water - evaporated_water) * (final_koolaid_percentage / 100) * initial_koolaid = 4 := 
by
  sorry

end NUMINAMATH_GPT_water_increase_factor_l1982_198249


namespace NUMINAMATH_GPT_alice_arrives_earlier_l1982_198279

/-
Alice and Bob are heading to a park that is 2 miles away from their home. 
They leave home at the same time. 
Alice cycles to the park at a speed of 12 miles per hour, 
while Bob jogs there at a speed of 6 miles per hour. 
Prove that Alice arrives 10 minutes earlier at the park than Bob.
-/

theorem alice_arrives_earlier 
  (d : ℕ) (a_speed : ℕ) (b_speed : ℕ) (arrival_difference_minutes : ℕ) 
  (h1 : d = 2) 
  (h2 : a_speed = 12) 
  (h3 : b_speed = 6) 
  (h4 : arrival_difference_minutes = 10) 
  : (d / a_speed * 60) + arrival_difference_minutes = d / b_speed * 60 :=
by
  sorry

end NUMINAMATH_GPT_alice_arrives_earlier_l1982_198279


namespace NUMINAMATH_GPT_find_k_l1982_198269

variable (m n k : ℝ)

def line (x y : ℝ) : Prop := x = 2 * y + 3
def point1_on_line : Prop := line m n
def point2_on_line : Prop := line (m + 2) (n + k)

theorem find_k (h1 : point1_on_line m n) (h2 : point2_on_line m n k) : k = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1982_198269


namespace NUMINAMATH_GPT_sum_of_integer_n_l1982_198252

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end NUMINAMATH_GPT_sum_of_integer_n_l1982_198252


namespace NUMINAMATH_GPT_odd_perfect_prime_form_n_is_seven_l1982_198233

theorem odd_perfect_prime_form (n p s m : ℕ) (h₁ : n % 2 = 1) (h₂ : ∃ k : ℕ, p = 4 * k + 1) (h₃ : ∃ h : ℕ, s = 4 * h + 1) (h₄ : n = p^s * m^2) (h₅ : ¬ p ∣ m) :
  ∃ k h : ℕ, p = 4 * k + 1 ∧ s = 4 * h + 1 :=
sorry

theorem n_is_seven (n : ℕ) (h₁ : n > 1) (h₂ : ∃ k : ℕ, k * k = n -1) (h₃ : ∃ l : ℕ, l * l = (n * (n + 1)) / 2) :
  n = 7 :=
sorry

end NUMINAMATH_GPT_odd_perfect_prime_form_n_is_seven_l1982_198233


namespace NUMINAMATH_GPT_amount_tom_should_pay_l1982_198274

theorem amount_tom_should_pay (original_price : ℝ) (multiplier : ℝ) 
  (h1 : original_price = 3) (h2 : multiplier = 3) : 
  original_price * multiplier = 9 :=
sorry

end NUMINAMATH_GPT_amount_tom_should_pay_l1982_198274
