import Mathlib

namespace christine_speed_l1021_102192

def distance : ℕ := 20
def time : ℕ := 5

theorem christine_speed :
  (distance / time) = 4 := 
sorry

end christine_speed_l1021_102192


namespace find_integer_n_l1021_102103

theorem find_integer_n : ∃ (n : ℤ), (-90 ≤ n ∧ n ≤ 90) ∧ (Real.sin (n * Real.pi / 180) = Real.cos (456 * Real.pi / 180)) ∧ n = -6 := 
by
  sorry

end find_integer_n_l1021_102103


namespace ratio_of_new_circumference_to_increase_in_area_l1021_102170

theorem ratio_of_new_circumference_to_increase_in_area
  (r k : ℝ) (h_k : 0 < k) :
  (2 * π * (r + k)) / (π * (2 * r * k + k ^ 2)) = 2 * (r + k) / (2 * r * k + k ^ 2) :=
by
  sorry

end ratio_of_new_circumference_to_increase_in_area_l1021_102170


namespace triangle_area_l1021_102105

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l1021_102105


namespace combined_weight_of_contents_l1021_102196

theorem combined_weight_of_contents
    (weight_pencil : ℝ := 28.3)
    (weight_eraser : ℝ := 15.7)
    (weight_paperclip : ℝ := 3.5)
    (weight_stapler : ℝ := 42.2)
    (num_pencils : ℕ := 5)
    (num_erasers : ℕ := 3)
    (num_paperclips : ℕ := 4)
    (num_staplers : ℕ := 2) :
    num_pencils * weight_pencil +
    num_erasers * weight_eraser +
    num_paperclips * weight_paperclip +
    num_staplers * weight_stapler = 287 := 
sorry

end combined_weight_of_contents_l1021_102196


namespace geometric_sequence_third_term_l1021_102188

theorem geometric_sequence_third_term :
  ∃ (a : ℕ) (r : ℝ), a = 5 ∧ a * r^3 = 500 ∧ a * r^2 = 5 * 100^(2/3) :=
by
  sorry

end geometric_sequence_third_term_l1021_102188


namespace equation_of_line_l1021_102121

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the line equation with parameters m and b
def line (m b x : ℝ) : ℝ := m * x + b

-- Define the point of intersection with the parabola on the line x = k
def intersection_point_parabola (k : ℝ) : ℝ := parabola k

-- Define the point of intersection with the line on the line x = k
def intersection_point_line (m b k : ℝ) : ℝ := line m b k

-- Define the vertical distance between the points on x = k
def vertical_distance (k m b : ℝ) : ℝ :=
  abs ((parabola k) - (line m b k))

-- Define the condition that vertical distance is exactly 4 units
def intersection_distance_condition (k m b : ℝ) : Prop :=
  vertical_distance k m b = 4

-- The line passes through point (2, 8)
def passes_through_point (m b : ℝ) : Prop :=
  line m b 2 = 8

-- Non-zero y-intercept condition
def non_zero_intercept (b : ℝ) : Prop := 
  b ≠ 0

-- The final theorem stating the required equation of the line
theorem equation_of_line (m b : ℝ) (h1 : ∃ k, intersection_distance_condition k m b)
  (h2 : passes_through_point m b) (h3 : non_zero_intercept b) : 
  (m = 12 ∧ b = -16) :=
by
  sorry

end equation_of_line_l1021_102121


namespace simplify_120_div_180_l1021_102198

theorem simplify_120_div_180 : (120 : ℚ) / 180 = 2 / 3 :=
by sorry

end simplify_120_div_180_l1021_102198


namespace isosceles_triangle_perimeter_l1021_102101

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : is_isosceles a b c) (h2 : is_triangle a b c) (h3 : a = 4 ∨ a = 9) (h4 : b = 4 ∨ b = 9) :
  perimeter a b c = 22 :=
  sorry

end isosceles_triangle_perimeter_l1021_102101


namespace find_x0_l1021_102150

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x_0 : ℝ) (h : f' x_0 = 2) : x_0 = Real.exp 1 :=
by
  sorry

end find_x0_l1021_102150


namespace hyperbola_eccentricity_l1021_102152

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = 2 →
  a^2 = 2 * b^2 →
  (c : ℝ) = Real.sqrt (a^2 + b^2) →
  Real.sqrt (a^2 + b^2) = Real.sqrt (3 / 2 * a^2) →
  (e : ℝ) = c / a →
  e = Real.sqrt (6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l1021_102152


namespace find_y_l1021_102115

theorem find_y (x y : ℤ) (h1 : x^2 - 5 * x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end find_y_l1021_102115


namespace measurable_masses_l1021_102178

theorem measurable_masses (k : ℤ) (h : -121 ≤ k ∧ k ≤ 121) : 
  ∃ (a b c d e : ℤ), k = a * 1 + b * 3 + c * 9 + d * 27 + e * 81 ∧ 
  (a = -1 ∨ a = 0 ∨ a = 1) ∧
  (b = -1 ∨ b = 0 ∨ b = 1) ∧
  (c = -1 ∨ c = 0 ∨ c = 1) ∧
  (d = -1 ∨ d = 0 ∨ d = 1) ∧
  (e = -1 ∨ e = 0 ∨ e = 1) :=
sorry

end measurable_masses_l1021_102178


namespace complex_fraction_simplification_l1021_102191

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l1021_102191


namespace pete_books_ratio_l1021_102125

theorem pete_books_ratio 
  (M_last : ℝ) (P_last P_this_year M_this_year : ℝ)
  (h1 : P_last = 2 * M_last)
  (h2 : M_this_year = 1.5 * M_last)
  (h3 : P_last + P_this_year = 300)
  (h4 : M_this_year = 75) :
  P_this_year / P_last = 2 :=
by
  sorry

end pete_books_ratio_l1021_102125


namespace arccos_cos_solution_l1021_102160

theorem arccos_cos_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ (Real.pi / 2)) (h₂ : Real.arccos (Real.cos x) = 2 * x) : 
    x = 0 :=
by 
  sorry

end arccos_cos_solution_l1021_102160


namespace march_first_is_sunday_l1021_102159

theorem march_first_is_sunday (days_in_march : ℕ) (num_wednesdays : ℕ) (num_saturdays : ℕ) 
  (h1 : days_in_march = 31) (h2 : num_wednesdays = 4) (h3 : num_saturdays = 4) : 
  ∃ d : ℕ, d = 0 := 
by 
  sorry

end march_first_is_sunday_l1021_102159


namespace vector_opposite_direction_and_magnitude_l1021_102124

theorem vector_opposite_direction_and_magnitude
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ k : ℝ, k < 0 ∧ b = k • a) 
  (hb : ‖b‖ = Real.sqrt 5) :
  b = (1, -2) :=
sorry

end vector_opposite_direction_and_magnitude_l1021_102124


namespace factor_x4_plus_81_l1021_102163

theorem factor_x4_plus_81 (x : ℝ) : (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) = x^4 + 81 := 
by 
   sorry

end factor_x4_plus_81_l1021_102163


namespace pears_sales_l1021_102130

variable (x : ℝ)
variable (morning_sales : ℝ := x)
variable (afternoon_sales : ℝ := 2 * x)
variable (evening_sales : ℝ := 3 * afternoon_sales)
variable (total_sales : ℝ := morning_sales + afternoon_sales + evening_sales)

theorem pears_sales :
  (total_sales = 510) →
  (afternoon_sales = 113.34) :=
by
  sorry

end pears_sales_l1021_102130


namespace negation_of_universal_proposition_l1021_102137

theorem negation_of_universal_proposition (x : ℝ) :
  ¬ (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 → x + 1 / x ≥ 2^m) ↔ ∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (x + 1 / x < 2^m) := by
  sorry

end negation_of_universal_proposition_l1021_102137


namespace number_of_persons_l1021_102109

theorem number_of_persons
    (total_amount : ℕ) 
    (amount_per_person : ℕ) 
    (h1 : total_amount = 42900) 
    (h2 : amount_per_person = 1950) :
    total_amount / amount_per_person = 22 :=
by
  sorry

end number_of_persons_l1021_102109


namespace average_temperature_l1021_102136

theorem average_temperature (T : Fin 5 → ℝ) (h : T = ![52, 67, 55, 59, 48]) :
    (1 / 5) * (T 0 + T 1 + T 2 + T 3 + T 4) = 56.2 := by
  sorry

end average_temperature_l1021_102136


namespace hyperbola_range_k_l1021_102120

theorem hyperbola_range_k (k : ℝ) : (4 + k) * (1 - k) < 0 ↔ k ∈ (Set.Iio (-4) ∪ Set.Ioi 1) := 
by
  sorry

end hyperbola_range_k_l1021_102120


namespace jose_work_time_l1021_102176

-- Define the variables for days taken by Jose and Raju
variables (J R T : ℕ)

-- State the conditions:
-- 1. Raju completes work in 40 days
-- 2. Together, Jose and Raju complete work in 8 days
axiom ra_work : R = 40
axiom together_work : T = 8

-- State the theorem that needs to be proven:
theorem jose_work_time (J R T : ℕ) (h1 : R = 40) (h2 : T = 8) : J = 10 :=
sorry

end jose_work_time_l1021_102176


namespace opposite_of_three_minus_one_l1021_102138

theorem opposite_of_three_minus_one : -(3 - 1) = -2 := 
by
  sorry

end opposite_of_three_minus_one_l1021_102138


namespace find_f1_and_f1_l1021_102145

theorem find_f1_and_f1' (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f 1 + f' 1 = -3 :=
by sorry

end find_f1_and_f1_l1021_102145


namespace copper_price_l1021_102151

theorem copper_price (c : ℕ) (hzinc : ℕ) (zinc_weight : ℕ) (brass_weight : ℕ) (price_brass : ℕ) (used_copper : ℕ) :
  hzinc = 30 →
  zinc_weight = brass_weight - used_copper →
  brass_weight = 70 →
  price_brass = 45 →
  used_copper = 30 →
  (used_copper * c + zinc_weight * hzinc) = brass_weight * price_brass →
  c = 65 :=
by
  sorry

end copper_price_l1021_102151


namespace find_a_l1021_102179

open Set

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (h : (A a ∩ B a) = {2, 5}) : a = 2 :=
sorry

end find_a_l1021_102179


namespace even_function_f_l1021_102113

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2^x - 1 else sorry

theorem even_function_f (h_even : ∀ x : ℝ, f x = f (-x)) : f 1 = -1 / 2 := by
  -- proof development skipped
  sorry

end even_function_f_l1021_102113


namespace x4_plus_y4_l1021_102180
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem x4_plus_y4 :
  (x^2 + (1 / x^2) = 7) →
  (x * y = 1) →
  (x^4 + y^4 = 47) :=
by
  intros h1 h2
  -- The proof will go here.
  sorry

end x4_plus_y4_l1021_102180


namespace smallest_Y_l1021_102108

theorem smallest_Y (S : ℕ) (h1 : (∀ d ∈ S.digits 10, d = 0 ∨ d = 1)) (h2 : 18 ∣ S) : 
  (∃ (Y : ℕ), Y = S / 18 ∧ ∀ (S' : ℕ), (∀ d ∈ S'.digits 10, d = 0 ∨ d = 1) → 18 ∣ S' → S' / 18 ≥ Y) → 
  Y = 6172839500 :=
sorry

end smallest_Y_l1021_102108


namespace max_revenue_l1021_102164

variable (x y : ℝ)

-- Conditions
def ads_time_constraint := x + y ≤ 300
def ads_cost_constraint := 500 * x + 200 * y ≤ 90000
def revenue := 0.3 * x + 0.2 * y

-- Question: Prove that the maximum revenue is 70 million yuan
theorem max_revenue (h_time : ads_time_constraint (x := 100) (y := 200))
                    (h_cost : ads_cost_constraint (x := 100) (y := 200)) :
  revenue (x := 100) (y := 200) = 70 := 
sorry

end max_revenue_l1021_102164


namespace value_of_x2_minus_y2_l1021_102102

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 9 / 17) (h2 : x - y = 1 / 19) : x^2 - y^2 = 9 / 323 :=
by
  -- the proof would go here
  sorry

end value_of_x2_minus_y2_l1021_102102


namespace monotonic_intervals_range_of_m_l1021_102142

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - 2 * x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m - 3

theorem monotonic_intervals :
  ∀ k : ℤ,
    (
      (∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 12 + k * Real.pi → ∃ (d : ℝ), f x = d)
      ∧
      (∀ x, 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 11 * Real.pi / 12 + k * Real.pi → ∃ (i : ℝ), f x = i)
    ) := sorry

theorem range_of_m (m : ℝ) :
  (∀ x1 : ℝ, Real.pi / 12 ≤ x1 ∧ x1 ≤ Real.pi / 2 → ∃ x2 : ℝ, -2 ≤ x2 ∧ x2 ≤ m ∧ f x1 = g x2 m) ↔ -1 ≤ m ∧ m ≤ 3 := sorry

end monotonic_intervals_range_of_m_l1021_102142


namespace experiment_success_probability_l1021_102156

/-- 
There are three boxes, each containing 10 balls. 
- The first box contains 7 balls marked 'A' and 3 balls marked 'B'.
- The second box contains 5 red balls and 5 white balls.
- The third box contains 8 red balls and 2 white balls.

The experiment consists of:
1. Drawing a ball from the first box.
2. If a ball marked 'A' is drawn, drawing from the second box.
3. If a ball marked 'B' is drawn, drawing from the third box.
The experiment is successful if the second ball drawn is red.

Prove that the probability of the experiment being successful is 0.59.
-/
theorem experiment_success_probability (P : ℝ) : 
  P = 0.59 :=
sorry

end experiment_success_probability_l1021_102156


namespace kho_kho_only_l1021_102183

theorem kho_kho_only (K H B total : ℕ) (h1 : K + B = 10) (h2 : B = 5) (h3 : K + H + B = 25) : H = 15 :=
by {
  sorry
}

end kho_kho_only_l1021_102183


namespace probability_solution_l1021_102165

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l1021_102165


namespace pills_needed_for_week_l1021_102110

def pill_mg : ℕ := 50 -- Each pill has 50 mg of Vitamin A.
def recommended_daily_mg : ℕ := 200 -- The recommended daily serving of Vitamin A is 200 mg.
def days_in_week : ℕ := 7 -- There are 7 days in a week.

theorem pills_needed_for_week : (recommended_daily_mg / pill_mg) * days_in_week = 28 := 
by 
  sorry

end pills_needed_for_week_l1021_102110


namespace min_buses_needed_l1021_102189

-- Given definitions from conditions
def students_per_bus : ℕ := 45
def total_students : ℕ := 495

-- The proposition to prove
theorem min_buses_needed : ∃ n : ℕ, 45 * n ≥ 495 ∧ (∀ m : ℕ, 45 * m ≥ 495 → n ≤ m) :=
by
  -- Preliminary calculations that lead to the solution
  let n := total_students / students_per_bus
  have h : total_students % students_per_bus = 0 := by sorry
  
  -- Conclude that the minimum n so that 45 * n ≥ 495 is indeed 11
  exact ⟨n, by sorry, by sorry⟩

end min_buses_needed_l1021_102189


namespace no_minimum_of_f_over_M_l1021_102118

/-- Define the domain M for the function y = log(3 - 4x + x^2) -/
def domain_M (x : ℝ) : Prop := (x > 3 ∨ x < 1)

/-- Define the function f(x) = 2x + 2 - 3 * 4^x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * 4^x

/-- The theorem statement:
    Prove that f(x) does not have a minimum value for x in the domain M -/
theorem no_minimum_of_f_over_M : ¬ ∃ x ∈ {x | domain_M x}, ∀ y ∈ {x | domain_M x}, f x ≤ f y := sorry

end no_minimum_of_f_over_M_l1021_102118


namespace polynomial_factorization_l1021_102149

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2
  = (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) :=
sorry

end polynomial_factorization_l1021_102149


namespace sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l1021_102132

def price_reduction_table : List (ℕ × ℕ) := 
  [(5, 780), (10, 810), (15, 840), (20, 870), (25, 900), (30, 930), (35, 960)]

theorem sales_volume_increase_30_units_every_5_yuan :
  ∀ reduction volume1 volume2, (reduction + 5, volume1) ∈ price_reduction_table →
  (reduction + 10, volume2) ∈ price_reduction_table → volume2 - volume1 = 30 := sorry

theorem initial_sales_volume_750_units :
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (0, 750) ∉ price_reduction_table → 780 - 30 = 750 := sorry

theorem daily_sales_volume_at_540_yuan :
  ∀ P₀ P₁ volume, P₀ = 600 → P₁ = 540 → 
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (15, 840) ∈ price_reduction_table → (20, 870) ∈ price_reduction_table →
  (25, 900) ∈ price_reduction_table → (30, 930) ∈ price_reduction_table →
  (35, 960) ∈ price_reduction_table →
  volume = 750 + (P₀ - P₁) / 5 * 30 → volume = 1110 := sorry

end sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l1021_102132


namespace contradiction_assumption_l1021_102153

-- Define the numbers x, y, z
variables (x y z : ℝ)

-- Define the assumption that all three numbers are non-positive
def all_non_positive (x y z : ℝ) : Prop := x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0

-- State the proposition to prove using the method of contradiction
theorem contradiction_assumption (h : all_non_positive x y z) : ¬ (x > 0 ∨ y > 0 ∨ z > 0) :=
by
  sorry

end contradiction_assumption_l1021_102153


namespace artist_paints_37_sq_meters_l1021_102187

-- Define the structure of the sculpture
def top_layer : ℕ := 1
def middle_layer : ℕ := 5
def bottom_layer : ℕ := 11
def edge_length : ℕ := 1

-- Define the exposed surface areas
def exposed_surface_top_layer := 5 * top_layer
def exposed_surface_middle_layer := 1 * 5 + 4 * 4
def exposed_surface_bottom_layer := bottom_layer

-- Calculate the total exposed surface area
def total_exposed_surface_area := exposed_surface_top_layer + exposed_surface_middle_layer + exposed_surface_bottom_layer

-- The final theorem statement
theorem artist_paints_37_sq_meters (hyp1 : top_layer = 1)
  (hyp2 : middle_layer = 5)
  (hyp3 : bottom_layer = 11)
  (hyp4 : edge_length = 1)
  : total_exposed_surface_area = 37 := 
by
  sorry

end artist_paints_37_sq_meters_l1021_102187


namespace hyperbola_foci_l1021_102123

/-- Define a hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- Definition of the foci of the hyperbola -/
def foci_coords (c : ℝ) : Prop := c = Real.sqrt 29

/-- Proof that the foci of the hyperbola 4y^2 - 25x^2 = 100 are (0, -sqrt(29)) and (0, sqrt(29)) -/
theorem hyperbola_foci (x y : ℝ) (c : ℝ) (hx : hyperbola_eq x y) (hc : foci_coords c) :
  (x = 0 ∧ (y = -c ∨ y = c)) :=
sorry

end hyperbola_foci_l1021_102123


namespace stickers_difference_l1021_102117

theorem stickers_difference (X : ℕ) :
  let Cindy_initial := X
  let Dan_initial := X
  let Cindy_after := Cindy_initial - 15
  let Dan_after := Dan_initial + 18
  Dan_after - Cindy_after = 33 := by
  sorry

end stickers_difference_l1021_102117


namespace height_of_Brixton_l1021_102154

theorem height_of_Brixton
  (I Z B Zr : ℕ)
  (h1 : I = Z + 4)
  (h2 : Z = B - 8)
  (h3 : Zr = B)
  (h4 : (I + Z + B + Zr) / 4 = 61) :
  B = 64 := by
  sorry

end height_of_Brixton_l1021_102154


namespace solution_set_of_f_prime_gt_zero_l1021_102128

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

theorem solution_set_of_f_prime_gt_zero :
  {x : ℝ | 0 < x ∧ 2*x - 2 - (4 / x) > 0} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_of_f_prime_gt_zero_l1021_102128


namespace minimum_value_frac_inverse_l1021_102147

theorem minimum_value_frac_inverse (a b c : ℝ) (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b)) + (1 / c) ≥ 4 / 3 :=
by
  sorry

end minimum_value_frac_inverse_l1021_102147


namespace average_of_11_numbers_l1021_102193

theorem average_of_11_numbers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 58)
  (h2 : (a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 6 = 65)
  (h3 : a₆ = 78) : 
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 11 = 60 := 
by 
  sorry 

end average_of_11_numbers_l1021_102193


namespace reciprocal_of_sum_of_fraction_l1021_102134

theorem reciprocal_of_sum_of_fraction (y : ℚ) (h : y = 6 + 1/6) : 1 / y = 6 / 37 := by
  sorry

end reciprocal_of_sum_of_fraction_l1021_102134


namespace vacation_days_l1021_102162

-- A plane ticket costs $24 for each person
def plane_ticket_cost : ℕ := 24

-- A hotel stay costs $12 for each person per day
def hotel_stay_cost_per_day : ℕ := 12

-- Total vacation cost is $120
def total_vacation_cost : ℕ := 120

-- The number of days they are planning to stay is 3
def number_of_days : ℕ := 3

-- Prove that given the conditions, the number of days (d) they plan to stay satisfies the total vacation cost
theorem vacation_days (d : ℕ) (plane_ticket_cost hotel_stay_cost_per_day total_vacation_cost : ℕ) 
  (h1 : plane_ticket_cost = 24)
  (h2 : hotel_stay_cost_per_day = 12) 
  (h3 : total_vacation_cost = 120) 
  (h4 : 2 * plane_ticket_cost + (2 * hotel_stay_cost_per_day) * d = total_vacation_cost)
  : d = 3 := sorry

end vacation_days_l1021_102162


namespace total_legs_walking_on_ground_l1021_102168

def horses : ℕ := 16
def men : ℕ := 16

def men_walking := men / 2
def men_riding := men / 2

def legs_per_man := 2
def legs_per_horse := 4

def legs_for_men_walking := men_walking * legs_per_man
def legs_for_horses := horses * legs_per_horse

theorem total_legs_walking_on_ground : legs_for_men_walking + legs_for_horses = 80 := 
by
  sorry

end total_legs_walking_on_ground_l1021_102168


namespace volume_of_pyramid_in_cube_l1021_102116

structure Cube :=
(side_length : ℝ)

noncomputable def base_triangle_area (side_length : ℝ) : ℝ :=
(1/2) * side_length * side_length

noncomputable def pyramid_volume (triangle_area : ℝ) (height : ℝ) : ℝ :=
(1/3) * triangle_area * height

theorem volume_of_pyramid_in_cube (c : Cube) (h : c.side_length = 2) : 
  pyramid_volume (base_triangle_area c.side_length) c.side_length = 4/3 :=
by {
  sorry
}

end volume_of_pyramid_in_cube_l1021_102116


namespace alok_age_l1021_102157

theorem alok_age (B A C : ℕ) (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  -- proof would go here
  sorry

end alok_age_l1021_102157


namespace find_a_l1021_102161

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a ^ 2}

theorem find_a (a : ℝ) (h : A ∪ B a = {0, 1, 2, 4}) : a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l1021_102161


namespace product_divisible_by_5_l1021_102141

theorem product_divisible_by_5 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : ∃ k, a * b = 5 * k) : a % 5 = 0 ∨ b % 5 = 0 :=
by
  sorry

end product_divisible_by_5_l1021_102141


namespace path_count_correct_l1021_102172

-- Define the graph-like structure for the octagonal lattice with directional constraints
structure OctagonalLattice :=
  (vertices : Type)
  (edges : vertices → vertices → Prop) -- Directed edges

-- Define a path from A to B respecting the constraints
def path_num_lattice (L : OctagonalLattice) (A B : L.vertices) : ℕ :=
  sorry -- We assume a function counting valid paths exists here

-- Assert the specific conditions for the bug's movement
axiom LatticeStructure : OctagonalLattice
axiom vertex_A : LatticeStructure.vertices
axiom vertex_B : LatticeStructure.vertices

-- Example specific path counting for the problem's lattice
noncomputable def paths_from_A_to_B : ℕ :=
  path_num_lattice LatticeStructure vertex_A vertex_B

theorem path_count_correct : paths_from_A_to_B = 2618 :=
  sorry -- This is where the proof would go

end path_count_correct_l1021_102172


namespace peter_large_glasses_l1021_102119

theorem peter_large_glasses (cost_small cost_large total_money small_glasses change num_large_glasses : ℕ)
    (h1 : cost_small = 3)
    (h2 : cost_large = 5)
    (h3 : total_money = 50)
    (h4 : small_glasses = 8)
    (h5 : change = 1)
    (h6 : total_money - change = 49)
    (h7 : small_glasses * cost_small = 24)
    (h8 : 49 - 24 = 25)
    (h9 : 25 / cost_large = 5) :
  num_large_glasses = 5 :=
by
  sorry

end peter_large_glasses_l1021_102119


namespace min_ring_cuts_l1021_102144

/-- Prove that the minimum number of cuts needed to pay the owner daily with an increasing 
    number of rings for 11 days, given a chain of 11 rings, is 2. -/
theorem min_ring_cuts {days : ℕ} {rings : ℕ} : days = 11 → rings = 11 → (∃ cuts : ℕ, cuts = 2) :=
by intros; sorry

end min_ring_cuts_l1021_102144


namespace parallelogram_area_l1021_102112

theorem parallelogram_area (base height : ℝ) (h_base : base = 14) (h_height : height = 24) :
  base * height = 336 :=
by 
  rw [h_base, h_height]
  sorry

end parallelogram_area_l1021_102112


namespace ratio_b_to_c_l1021_102167

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l1021_102167


namespace find_original_price_each_stocking_l1021_102129

open Real

noncomputable def original_stocking_price (total_stockings total_cost_per_stocking discounted_cost monogramming_cost total_cost : ℝ) : ℝ :=
  let stocking_cost_before_monogramming := total_cost - (total_stockings * monogramming_cost)
  let original_price := stocking_cost_before_monogramming / (total_stockings * discounted_cost)
  original_price

theorem find_original_price_each_stocking :
  original_stocking_price 9 122.22 0.9 5 1035 = 122.22 := by
  sorry

end find_original_price_each_stocking_l1021_102129


namespace angles_in_triangle_l1021_102104

theorem angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = 3 * A) (h3 : 5 * A = 2 * C) :
  B = 54 ∧ C = 90 :=
by
  sorry

end angles_in_triangle_l1021_102104


namespace velocity_at_3_seconds_l1021_102182

variable (t : ℝ)
variable (s : ℝ)

def motion_eq (t : ℝ) : ℝ := 1 + t + t^2

theorem velocity_at_3_seconds : 
  (deriv motion_eq 3) = 7 :=
by
  sorry

end velocity_at_3_seconds_l1021_102182


namespace equality_of_a_b_c_l1021_102197

theorem equality_of_a_b_c
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (eqn : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
by
  sorry

end equality_of_a_b_c_l1021_102197


namespace watermelons_remaining_l1021_102190

theorem watermelons_remaining :
  let initial_watermelons := 10 * 12
  let yesterdays_sale := 0.40 * initial_watermelons
  let remaining_after_yesterday := initial_watermelons - yesterdays_sale
  let todays_sale := (1 / 4) * remaining_after_yesterday
  let remaining_after_today := remaining_after_yesterday - todays_sale
  let tomorrows_sales := 1.5 * todays_sale
  let remaining_after_tomorrow := remaining_after_today - tomorrows_sales
  remaining_after_tomorrow = 27 :=
by
  sorry

end watermelons_remaining_l1021_102190


namespace arctan_sum_pi_over_four_l1021_102186

theorem arctan_sum_pi_over_four (a b c : ℝ) (C : ℝ) (h : Real.sin C = c / (a + b + c)) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
sorry

end arctan_sum_pi_over_four_l1021_102186


namespace initial_quantity_of_milk_in_A_l1021_102173

theorem initial_quantity_of_milk_in_A (A : ℝ) 
  (h1: ∃ C B: ℝ, B = 0.375 * A ∧ C = 0.625 * A) 
  (h2: ∃ M: ℝ, M = 0.375 * A + 154 ∧ M = 0.625 * A - 154) 
  : A = 1232 :=
by
  -- you can use sorry to skip the proof
  sorry

end initial_quantity_of_milk_in_A_l1021_102173


namespace base_number_is_five_l1021_102181

variable (a x y : Real)

theorem base_number_is_five (h1 : xy = 1) (h2 : (a ^ (x + y) ^ 2) / (a ^ (x - y) ^ 2) = 625) : a = 5 := 
sorry

end base_number_is_five_l1021_102181


namespace trig_problems_l1021_102127

variable {A B C : ℝ}
variable {a b c : ℝ}

-- The main theorem statement to prove the magnitude of angle B and find b under given conditions.
theorem trig_problems
  (h₁ : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h₂ : a = Real.sqrt 3)
  (h₃ : c = Real.sqrt 3) :
  Real.cos B = 1 / 2 ∧ b = Real.sqrt 3 := by
sorry

end trig_problems_l1021_102127


namespace find_integers_k_l1021_102135

theorem find_integers_k (k : ℤ) : 
  (k = 15 ∨ k = 30) ↔ 
  (k ≥ 3 ∧ ∃ m n : ℤ, 1 < m ∧ m < k ∧ 1 < n ∧ n < k ∧ 
                       Int.gcd m k = 1 ∧ Int.gcd n k = 1 ∧ 
                       m + n > k ∧ k ∣ (m - 1) * (n - 1)) :=
by
  sorry -- Proof goes here

end find_integers_k_l1021_102135


namespace linear_function_value_l1021_102195

theorem linear_function_value
  (a b c : ℝ)
  (h1 : 3 * a + b = 8)
  (h2 : -2 * a + b = 3)
  (h3 : -3 * a + b = c) :
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 13 :=
by
  sorry

end linear_function_value_l1021_102195


namespace gold_copper_ratio_l1021_102106

theorem gold_copper_ratio (G C : ℕ) (h : 19 * G + 9 * C = 17 * (G + C)) : G = 4 * C :=
by
  sorry

end gold_copper_ratio_l1021_102106


namespace Vihaan_more_nephews_than_Alden_l1021_102199

theorem Vihaan_more_nephews_than_Alden :
  ∃ (a v : ℕ), (a = 100) ∧ (a + v = 260) ∧ (v - a = 60) := by
  sorry

end Vihaan_more_nephews_than_Alden_l1021_102199


namespace sum_of_first_n_terms_sequence_l1021_102169

open Nat

def sequence_term (i : ℕ) : ℚ :=
  if i = 0 then 0 else 1 / (i * (i + 1) / 2 : ℕ)

def sum_of_sequence (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum fun i => sequence_term i

theorem sum_of_first_n_terms_sequence (n : ℕ) : sum_of_sequence n = 2 * n / (n + 1) := by
  sorry

end sum_of_first_n_terms_sequence_l1021_102169


namespace speed_of_first_car_l1021_102143

theorem speed_of_first_car (v : ℝ) 
  (h1 : ∀ v, v > 0 → (first_speed = 1.25 * v))
  (h2 : 720 = (v + 1.25 * v) * 4) : 
  first_speed = 100 := 
by
  sorry

end speed_of_first_car_l1021_102143


namespace prime_exponent_condition_l1021_102139

theorem prime_exponent_condition (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n)
  (h : 2^p + 3^p = a^n) : n = 1 :=
sorry

end prime_exponent_condition_l1021_102139


namespace lcm_smallest_value_l1021_102166

/-- The smallest possible value of lcm(k, l) for positive 5-digit integers k and l such that gcd(k, l) = 5 is 20010000. -/
theorem lcm_smallest_value (k l : ℕ) (h1 : 10000 ≤ k ∧ k < 100000) (h2 : 10000 ≤ l ∧ l < 100000) (h3 : Nat.gcd k l = 5) : Nat.lcm k l = 20010000 :=
sorry

end lcm_smallest_value_l1021_102166


namespace triangle_inequalities_l1021_102100

theorem triangle_inequalities (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) →
  (a = b ∧ a > c) ∨ (a = b ∧ b = c) :=
by
  sorry

end triangle_inequalities_l1021_102100


namespace remainder_when_7n_divided_by_4_l1021_102171

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l1021_102171


namespace unique_solution_mnk_l1021_102175

theorem unique_solution_mnk :
  ∀ (m n k : ℕ), 3^n + 4^m = 5^k → (m, n, k) = (0, 1, 1) :=
by
  intros m n k h
  sorry

end unique_solution_mnk_l1021_102175


namespace greatest_product_l1021_102140

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end greatest_product_l1021_102140


namespace monotonic_function_range_maximum_value_condition_function_conditions_l1021_102177

-- Part (1): Monotonicity condition
theorem monotonic_function_range (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0) ↔ (m ≥ 3) := sorry

-- Part (2): Maximum value condition
theorem maximum_value_condition (m : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4) ↔ (m = -2) := sorry

-- Combined statement (optional if you want to show entire problem in one go)
theorem function_conditions (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0 ∧ 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4)) ↔ (m = -2 ∨ m ≥ 3) := sorry

end monotonic_function_range_maximum_value_condition_function_conditions_l1021_102177


namespace positive_integer_count_l1021_102146

theorem positive_integer_count (n : ℕ) :
  ∃ (count : ℕ), (count = 122) ∧ 
  (∀ (k : ℕ), 27 < k ∧ k < 150 → ((150 * k)^40 > k^80 ∧ k^80 > 3^240)) :=
sorry

end positive_integer_count_l1021_102146


namespace restore_original_price_l1021_102133

-- Defining the original price of the jacket
def original_price (P : ℝ) := P

-- Defining the price after each step of reduction
def price_after_first_reduction (P : ℝ) := P * (1 - 0.25)
def price_after_second_reduction (P : ℝ) := price_after_first_reduction P * (1 - 0.20)
def price_after_third_reduction (P : ℝ) := price_after_second_reduction P * (1 - 0.10)

-- Express the condition to restore the original price
theorem restore_original_price (P : ℝ) (x : ℝ) : 
  original_price P = price_after_third_reduction P * (1 + x) → 
  x = 0.85185185 := 
by
  sorry

end restore_original_price_l1021_102133


namespace points_on_line_l1021_102194

-- Define the points involved
def point1 : ℝ × ℝ := (4, 10)
def point2 : ℝ × ℝ := (-2, -8)
def candidate_points : List (ℝ × ℝ) := [(1, 1), (0, -1), (2, 3), (-1, -5), (3, 7)]
def correct_points : List (ℝ × ℝ) := [(1, 1), (-1, -5), (3, 7)]

-- Define a function to check if a point lies on the line defined by point1 and point2
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (10 - (-8)) / (4 - (-2))
  let b := 10 - m * 4
  p.2 = m * p.1 + b

-- Main theorem statement
theorem points_on_line :
  ∀ p ∈ candidate_points, p ∈ correct_points ↔ lies_on_line p :=
sorry

end points_on_line_l1021_102194


namespace unit_digit_3_pow_2023_l1021_102185

def unit_digit_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0

theorem unit_digit_3_pow_2023 : unit_digit_pattern 2023 = 7 :=
by sorry

end unit_digit_3_pow_2023_l1021_102185


namespace find_a_l1021_102131

theorem find_a (a : ℚ) : (∃ b : ℚ, 4 * (x : ℚ)^2 + 14 * x + a = (2 * x + b)^2) → a = 49 / 4 :=
by
  sorry

end find_a_l1021_102131


namespace area_inside_C_outside_A_B_l1021_102114

/-- Define the radii of circles A, B, and C --/
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

/-- Define the condition of tangency and overlap --/
def circles_tangent_at_one_point (r1 r2 : ℝ) : Prop :=
  r1 = r2 

def circle_C_tangent_to_A_B (rA rB rC : ℝ) : Prop :=
  rA = 1 ∧ rB = 1 ∧ rC = 2 ∧ circles_tangent_at_one_point rA rB

/-- Statement to be proved: The area inside circle C but outside circles A and B is 2π --/
theorem area_inside_C_outside_A_B (h : circle_C_tangent_to_A_B radius_A radius_B radius_C) : 
  π * radius_C^2 - π * (radius_A^2 + radius_B^2) = 2 * π :=
by
  sorry

end area_inside_C_outside_A_B_l1021_102114


namespace fraction_to_decimal_l1021_102155

theorem fraction_to_decimal : (58 / 125 : ℚ) = 0.464 := 
by {
  -- proof omitted
  sorry
}

end fraction_to_decimal_l1021_102155


namespace sector_area_l1021_102107

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 4) :
  1/2 * r^2 * α = π / 2 :=
by
  subst h_r
  subst h_α
  sorry

end sector_area_l1021_102107


namespace total_fish_caught_l1021_102158

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end total_fish_caught_l1021_102158


namespace coat_lifetime_15_l1021_102174

noncomputable def coat_lifetime : ℕ :=
  let cost_coat_expensive := 300
  let cost_coat_cheap := 120
  let years_cheap := 5
  let year_saving := 120
  let duration_comparison := 30
  let yearly_cost_cheaper := cost_coat_cheap / years_cheap
  let yearly_savings := year_saving / duration_comparison
  let cost_savings := yearly_cost_cheaper * duration_comparison - cost_coat_expensive * duration_comparison / (yearly_savings + (cost_coat_expensive / cost_coat_cheap))
  cost_savings

theorem coat_lifetime_15 : coat_lifetime = 15 := by
  sorry

end coat_lifetime_15_l1021_102174


namespace find_multiplier_l1021_102111

theorem find_multiplier (A N : ℕ) (h : A = 32) (eqn : N * (A + 4) - 4 * (A - 4) = A) : N = 4 :=
sorry

end find_multiplier_l1021_102111


namespace movie_friends_l1021_102122

noncomputable def movie_only (M P G MP MG PG MPG : ℕ) : Prop :=
  let total_M := 20
  let total_P := 20
  let total_G := 5
  let total_students := 31
  (MP = 4) ∧ 
  (MG = 2) ∧ 
  (PG = 0) ∧ (MPG = 2) ∧ 
  (M + MP + MG + MPG = total_M) ∧ 
  (P + MP + PG + MPG = total_P) ∧ 
  (G + MG + PG + MPG = total_G) ∧ 
  (M + P + G + MP + MG + PG + MPG = total_students) ∧ 
  (M = 12)

theorem movie_friends (M P G MP MG PG MPG : ℕ) : movie_only M P G MP MG PG MPG := 
by 
  sorry

end movie_friends_l1021_102122


namespace odd_n_cube_minus_n_div_by_24_l1021_102126

theorem odd_n_cube_minus_n_div_by_24 (n : ℤ) (h_odd : n % 2 = 1) : 24 ∣ (n^3 - n) :=
sorry

end odd_n_cube_minus_n_div_by_24_l1021_102126


namespace complete_square_h_l1021_102148

theorem complete_square_h (x h : ℝ) :
  (∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) → h = -3 / 2 :=
by
  sorry

end complete_square_h_l1021_102148


namespace find_xy_l1021_102184

theorem find_xy (x y : ℤ) 
  (h1 : (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3) : 
  x = -35 ∧ y = -35 :=
by 
  sorry

end find_xy_l1021_102184
