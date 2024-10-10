import Mathlib

namespace tangent_line_at_2_and_through_A_l838_83891

/-- The function f(x) = x³ - 4x² + 5x - 4 -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem tangent_line_at_2_and_through_A :
  /- Tangent line at X=2 -/
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 4 = 0 ∧ 
    m = f' 2 ∧ -2 = m*2 + b) ∧ 
  /- Tangent lines through A(2,-2) -/
  (∃ (a : ℝ), 
    (∀ x y, y = -2 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a) ∨
    (∀ x y, x - y - 4 = 0 ↔ f a = f' a * (x - a) + f a ∧ -2 = f' a * (2 - a) + f a)) :=
sorry

end tangent_line_at_2_and_through_A_l838_83891


namespace sum_of_coefficients_P_l838_83870

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

/-- Theorem stating that the sum of coefficients of P is 2019 -/
theorem sum_of_coefficients_P : (P 1) = 2019 := by sorry

end sum_of_coefficients_P_l838_83870


namespace fixed_point_of_power_plus_one_l838_83867

/-- The function f(x) = x^n + 1 has a fixed point at (1, 2) for any positive integer n. -/
theorem fixed_point_of_power_plus_one (n : ℕ+) :
  let f : ℝ → ℝ := fun x ↦ x^(n : ℕ) + 1
  f 1 = 2 := by
  sorry

end fixed_point_of_power_plus_one_l838_83867


namespace inequality_proof_l838_83826

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (b^2 / a + c^2 / b + a^2 / c ≥ 1) := by
sorry


end inequality_proof_l838_83826


namespace extremum_condition_l838_83812

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (fun x => Real.exp x + a * x) x > 0 ∧
   ∀ y : ℝ, (fun x => Real.exp x + a * x) y ≤ (fun x => Real.exp x + a * x) x) →
  a < -1 := by
  sorry

end extremum_condition_l838_83812


namespace irrational_sum_rational_irrational_l838_83852

theorem irrational_sum_rational_irrational (π : ℝ) (h : Irrational π) : Irrational (5 + π) := by
  sorry

end irrational_sum_rational_irrational_l838_83852


namespace complement_M_equals_interval_l838_83855

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x : ℝ | (2 - x) / (x + 3) < 0}

-- Define the complement of M in ℝ
def complement_M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_M_equals_interval : 
  (U \ M) = complement_M :=
sorry

end complement_M_equals_interval_l838_83855


namespace half_red_probability_l838_83809

def num_balls : ℕ := 8
def num_red : ℕ := 4

theorem half_red_probability :
  let p_red : ℚ := 1 / 2
  let p_event : ℚ := (num_balls.choose num_red : ℚ) * p_red ^ num_balls
  p_event = 35 / 128 := by sorry

end half_red_probability_l838_83809


namespace complex_equation_sum_l838_83804

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * i) / i = b + i → a + b = 1 := by
sorry

end complex_equation_sum_l838_83804


namespace parallel_vectors_l838_83806

theorem parallel_vectors (m n : ℝ × ℝ) : 
  m = (2, 8) → n = (-4, t) → m.1 * n.2 = m.2 * n.1 → t = -16 := by
  sorry

end parallel_vectors_l838_83806


namespace product_expansion_sum_l838_83818

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2 * x^2 - 4 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -24 := by
sorry

end product_expansion_sum_l838_83818


namespace cubic_roots_sum_l838_83835

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p / (q*r - 1) + q / (p*r - 1) + r / (p*q - 1) = 17/29 := by
  sorry

end cubic_roots_sum_l838_83835


namespace vector_decomposition_l838_83879

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![-1, 7, 0]
def p : Fin 3 → ℝ := ![0, 3, 1]
def q : Fin 3 → ℝ := ![1, -1, 2]
def r : Fin 3 → ℝ := ![2, -1, 0]

/-- Theorem stating the decomposition of x in terms of p and q -/
theorem vector_decomposition : x = 2 • p - q := by sorry

end vector_decomposition_l838_83879


namespace unit_conversions_l838_83868

-- Define conversion rates
def kgToGrams : ℚ → ℚ := (· * 1000)
def meterToDecimeter : ℚ → ℚ := (· * 10)

-- Theorem statement
theorem unit_conversions :
  (kgToGrams 4 = 4000) ∧
  (meterToDecimeter 3 - 2 = 28) ∧
  (meterToDecimeter 8 = 80) ∧
  ((1600 : ℚ) - 600 = kgToGrams 1) :=
by sorry

end unit_conversions_l838_83868


namespace distribute_negative_five_l838_83861

theorem distribute_negative_five (x y : ℝ) : -5 * (x - y) = -5 * x + 5 * y := by
  sorry

end distribute_negative_five_l838_83861


namespace num_faces_after_transformation_l838_83869

/-- Represents the number of steps in the transformation process -/
def num_steps : ℕ := 5

/-- The initial number of vertices in a cube -/
def initial_vertices : ℕ := 8

/-- The initial number of edges in a cube -/
def initial_edges : ℕ := 12

/-- The factor by which vertices and edges increase in each step -/
def increase_factor : ℕ := 3

/-- Calculates the number of vertices after the transformation -/
def final_vertices : ℕ := initial_vertices * increase_factor ^ num_steps

/-- Calculates the number of edges after the transformation -/
def final_edges : ℕ := initial_edges * increase_factor ^ num_steps

/-- Theorem stating the number of faces after the transformation -/
theorem num_faces_after_transformation : 
  final_vertices - final_edges + 974 = 2 :=
sorry

end num_faces_after_transformation_l838_83869


namespace quadratic_inequality_range_l838_83846

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end quadratic_inequality_range_l838_83846


namespace factorial_10_mod_11_l838_83815

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_11 : factorial 10 % 11 = 10 := by
  sorry

end factorial_10_mod_11_l838_83815


namespace toy_car_cost_l838_83830

theorem toy_car_cost (initial_amount : ℕ) (num_cars : ℕ) (scarf_cost : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 53 →
  num_cars = 2 →
  scarf_cost = 10 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  (initial_amount - remaining_amount - scarf_cost - beanie_cost) / num_cars = 11 :=
by
  sorry

end toy_car_cost_l838_83830


namespace last_k_digits_theorem_l838_83851

theorem last_k_digits_theorem (k : ℕ) (h : k ≥ 2) :
  (∃ n : ℕ+, (10^(10^n.val) : ℤ) ≡ 9^(9^n.val) [ZMOD 10^k]) ↔ k ∈ ({2, 3, 4} : Finset ℕ) :=
sorry

end last_k_digits_theorem_l838_83851


namespace book_arrangement_theorem_l838_83839

theorem book_arrangement_theorem :
  let total_books : ℕ := 8
  let advanced_geometry_copies : ℕ := 5
  let essential_number_theory_copies : ℕ := 3
  total_books = advanced_geometry_copies + essential_number_theory_copies →
  (Nat.choose total_books advanced_geometry_copies) = 56 := by
  sorry

end book_arrangement_theorem_l838_83839


namespace smallest_n_square_and_cube_l838_83837

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < n → 
    (¬∃ (y : ℕ), 5 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 4 * x = z^3)) ∧
  n = 1600 :=
by
  sorry

end smallest_n_square_and_cube_l838_83837


namespace dollar_equality_l838_83814

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem dollar_equality (x y : ℝ) : 
  dollar ((2*x + y)^2) ((x - 2*y)^2) = (3*x^2 + 8*x*y - 3*y^2)^2 := by
  sorry

end dollar_equality_l838_83814


namespace sum_of_odd_decreasing_function_is_negative_l838_83898

-- Define a structure for our function properties
structure OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x y, x < y → f x > f y

-- Main theorem
theorem sum_of_odd_decreasing_function_is_negative
  (f : ℝ → ℝ)
  (h_f : OddDecreasingFunction f)
  (α β γ : ℝ)
  (h_αβ : α + β > 0)
  (h_βγ : β + γ > 0)
  (h_γα : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end sum_of_odd_decreasing_function_is_negative_l838_83898


namespace social_media_time_ratio_l838_83805

/-- Proves that the ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_time_ratio 
  (daily_phone_time : ℝ) 
  (weekly_social_media_time : ℝ) 
  (h1 : daily_phone_time = 6)
  (h2 : weekly_social_media_time = 21) :
  (weekly_social_media_time / 7) / daily_phone_time = 1 / 2 := by
  sorry

end social_media_time_ratio_l838_83805


namespace sum_reciprocals_S_l838_83831

def S : Set ℕ+ := {n : ℕ+ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 2017}

theorem sum_reciprocals_S : ∑' (s : S), (1 : ℝ) / (s : ℝ) = 2017 / 1008 := by
  sorry

end sum_reciprocals_S_l838_83831


namespace smallest_five_digit_square_cube_l838_83882

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 
            (∃ x : ℕ, m = x^2) ∧ 
            (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- smallest such number
  n = 15625                   -- the answer
  := by sorry

end smallest_five_digit_square_cube_l838_83882


namespace dodecagon_ratio_l838_83865

/-- Represents a dodecagon with specific properties -/
structure Dodecagon where
  /-- Total area of the dodecagon -/
  total_area : ℝ
  /-- Area below the bisecting line PQ -/
  area_below_pq : ℝ
  /-- Base of the triangle below PQ -/
  triangle_base : ℝ
  /-- Width of the dodecagon (XQ + QY) -/
  width : ℝ
  /-- Assertion that the dodecagon is made of 12 unit squares -/
  area_is_twelve : total_area = 12
  /-- Assertion that PQ bisects the area -/
  pq_bisects : area_below_pq = total_area / 2
  /-- Assertion about the composition below PQ -/
  below_pq_composition : area_below_pq = 2 + (triangle_base * triangle_base / 12)
  /-- Assertion about the width of the dodecagon -/
  width_is_six : width = 6

/-- Theorem stating that for a dodecagon with given properties, XQ/QY = 2 -/
theorem dodecagon_ratio (d : Dodecagon) : ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = d.width := by
  sorry

end dodecagon_ratio_l838_83865


namespace f_monotonicity_and_range_l838_83853

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 * a - x^2) / Real.exp x

theorem f_monotonicity_and_range (a : ℝ) :
  (a ≤ -1/2 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    ((x < y ∧ y < 1 - Real.sqrt (2 * a + 1)) ∨
     (x > 1 + Real.sqrt (2 * a + 1) ∧ y > x)) →
    f a x < f a y) ∧
  (a > -1/2 → ∀ x y : ℝ,
    (x > 1 - Real.sqrt (2 * a + 1) ∧ y < 1 + Real.sqrt (2 * a + 1) ∧ x < y) →
    f a x > f a y) ∧
  ((∀ x : ℝ, x ≥ 1 → f a x > -1) → a > (1 - Real.exp 1) / 2) :=
by sorry

end f_monotonicity_and_range_l838_83853


namespace sequence_sum_l838_83819

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d →
  b - a = c - b →
  c * c = b * d →
  d - a = 30 →
  a + b + c + d = 129 := by
sorry

end sequence_sum_l838_83819


namespace not_power_of_prime_l838_83876

theorem not_power_of_prime (n : ℕ+) (q : ℕ) (h_prime : Nat.Prime q) :
  ¬∃ k : ℕ, (n : ℝ)^q + ((n - 1 : ℝ) / 2)^2 = (q : ℝ)^k := by
  sorry

end not_power_of_prime_l838_83876


namespace marias_water_bottles_l838_83825

theorem marias_water_bottles (initial bottles_drunk final : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_drunk = 8)
  (h3 : final = 51) :
  final - (initial - bottles_drunk) = 45 := by
  sorry

end marias_water_bottles_l838_83825


namespace radical_conjugate_sum_product_l838_83877

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 8 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 15 →
  x + y = 5 := by sorry

end radical_conjugate_sum_product_l838_83877


namespace fifteen_clockwise_opposite_l838_83847

/-- Represents a circle of equally spaced children -/
structure ChildrenCircle where
  num_children : ℕ
  standard_child : ℕ

/-- The child directly opposite another child in the circle -/
def opposite_child (circle : ChildrenCircle) (child : ℕ) : ℕ :=
  (child + circle.num_children / 2) % circle.num_children

theorem fifteen_clockwise_opposite (circle : ChildrenCircle) :
  opposite_child circle circle.standard_child = (circle.standard_child + 15) % circle.num_children →
  circle.num_children = 30 := by
  sorry

end fifteen_clockwise_opposite_l838_83847


namespace power_difference_equality_l838_83827

theorem power_difference_equality : 4^(2+4+6) - (4^2 + 4^4 + 4^6) = 16772848 := by
  sorry

end power_difference_equality_l838_83827


namespace framed_painting_ratio_l838_83801

theorem framed_painting_ratio :
  ∀ (x : ℝ),
  x > 0 →
  (20 + 2*x) * (30 + 6*x) = 1800 →
  (20 + 2*x) / (30 + 6*x) = 1/2 :=
by
  sorry

end framed_painting_ratio_l838_83801


namespace zero_product_property_l838_83894

theorem zero_product_property {α : Type*} [Semiring α] {a b : α} :
  a * b = 0 → (a = 0 ∨ b = 0) := by sorry

end zero_product_property_l838_83894


namespace regular_star_points_l838_83802

/-- An n-pointed regular star -/
structure RegularStar (n : ℕ) :=
  (edge_length : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (edge_congruent : edge_length > 0)
  (angle_A_congruent : angle_A > 0)
  (angle_B_congruent : angle_B > 0)
  (angle_difference : angle_B = angle_A + 10)
  (exterior_angle_sum : n * (angle_A + angle_B) = 360)

/-- The number of points in a regular star satisfying the given conditions is 36 -/
theorem regular_star_points : ∃ (n : ℕ), n > 0 ∧ ∃ (star : RegularStar n), n = 36 :=
sorry

end regular_star_points_l838_83802


namespace sum_of_valid_divisors_l838_83822

/-- The sum of valid divisors of 360 that satisfy specific conditions --/
theorem sum_of_valid_divisors : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 360 % x = 0 ∧ x ≥ 18 ∧ 360 / x ≥ 12) 
    (Finset.range 361)).sum id = 92 := by sorry

end sum_of_valid_divisors_l838_83822


namespace emily_walks_farther_l838_83860

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_daily_distances : List ℕ := [90, 95, 85, 85, 80]
def emily_daily_distances : List ℕ := [108, 123, 108, 123, 108]

def calculate_total_distance (daily_distances : List ℕ) : ℕ :=
  2 * (daily_distances.sum)

theorem emily_walks_farther :
  calculate_total_distance emily_daily_distances - calculate_total_distance troy_daily_distances = 270 := by
  sorry

end emily_walks_farther_l838_83860


namespace dictionary_cost_l838_83836

theorem dictionary_cost (total_cost dinosaur_cost cookbook_cost : ℕ) 
  (h1 : total_cost = 37)
  (h2 : dinosaur_cost = 19)
  (h3 : cookbook_cost = 7) :
  total_cost - dinosaur_cost - cookbook_cost = 11 := by
  sorry

end dictionary_cost_l838_83836


namespace min_a_value_l838_83807

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

def holds_inequality (a : ℤ) : Prop :=
  ∀ x > 0, f x ≤ ((↑a / 2) - 1) * x^2 + ↑a * x - 1

theorem min_a_value :
  ∃ a : ℤ, holds_inequality a ∧ ∀ b : ℤ, b < a → ¬(holds_inequality b) :=
by sorry

end min_a_value_l838_83807


namespace parabola_point_coordinates_l838_83899

/-- Parabola with vertex at origin and focus on positive x-axis -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_x_axis : focus.2 = 0 ∧ focus.1 > 0

/-- The curve E: x²+y²-6x+4y-3=0 -/
def curve_E (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y - 3 = 0

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : point.2^2 = 2 * p.focus.1 * point.1

theorem parabola_point_coordinates (p : Parabola) 
  (h1 : ∃! x y, curve_E x y ∧ x = -p.focus.1) 
  (A : PointOnParabola p) 
  (h2 : A.point.1 * (A.point.1 - p.focus.1) + A.point.2 * A.point.2 = -4) :
  A.point = (1, 2) ∨ A.point = (1, -2) := by
  sorry

end parabola_point_coordinates_l838_83899


namespace library_fiction_percentage_l838_83844

/-- Proves that given the conditions of the library problem, the percentage of fiction novels in the original collection is approximately 30.66%. -/
theorem library_fiction_percentage 
  (total_volumes : ℕ)
  (transferred_fraction : ℚ)
  (transferred_fiction_fraction : ℚ)
  (remaining_fiction_percentage : ℚ)
  (h_total : total_volumes = 18360)
  (h_transferred : transferred_fraction = 1/3)
  (h_transferred_fiction : transferred_fiction_fraction = 1/5)
  (h_remaining_fiction : remaining_fiction_percentage = 35.99999999999999/100) :
  ∃ (original_fiction_percentage : ℚ), 
    (original_fiction_percentage ≥ 30.65/100) ∧ 
    (original_fiction_percentage ≤ 30.67/100) := by
  sorry

end library_fiction_percentage_l838_83844


namespace sum_coordinates_of_B_l838_83808

/-- Given that M(5,3) is the midpoint of AB and A has coordinates (10,2), 
    prove that the sum of coordinates of point B is 4. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (5, 3) → 
  A = (10, 2) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 4 := by
sorry

end sum_coordinates_of_B_l838_83808


namespace shirt_cost_l838_83863

theorem shirt_cost (total_cost pants_cost tie_cost : ℕ) 
  (h1 : total_cost = 198)
  (h2 : pants_cost = 140)
  (h3 : tie_cost = 15) :
  total_cost - pants_cost - tie_cost = 43 := by
sorry

end shirt_cost_l838_83863


namespace no_integer_solution_l838_83832

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 19 * x^2 - 76 * y^2 = 1976 := by
  sorry

end no_integer_solution_l838_83832


namespace perfect_square_quadratic_l838_83895

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 18 * x + 9 = (r * x + s)^2) → a = 9 := by
  sorry

end perfect_square_quadratic_l838_83895


namespace edwards_purchases_cost_edwards_total_cost_l838_83834

/-- Calculates the total cost of Edward's purchases after applying a discount -/
theorem edwards_purchases_cost (board_game_cost : ℝ) (action_figure_cost : ℝ) 
  (action_figure_count : ℕ) (puzzle_cost : ℝ) (card_deck_cost : ℝ) 
  (discount_percentage : ℝ) : ℝ :=
  let total_action_figures_cost := action_figure_cost * action_figure_count
  let discount_amount := total_action_figures_cost * (discount_percentage / 100)
  let discounted_action_figures_cost := total_action_figures_cost - discount_amount
  board_game_cost + discounted_action_figures_cost + puzzle_cost + card_deck_cost

/-- Proves that Edward's total purchase cost is $36.70 -/
theorem edwards_total_cost : 
  edwards_purchases_cost 2 7 4 6 3.5 10 = 36.7 := by
  sorry

end edwards_purchases_cost_edwards_total_cost_l838_83834


namespace smallest_sum_of_two_distinct_primes_greater_than_500_l838_83841

def is_prime (n : ℕ) : Prop := sorry

def sum_of_two_distinct_primes_greater_than_500 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p > 500 ∧ q > 500 ∧ p ≠ q ∧ n = p + q

theorem smallest_sum_of_two_distinct_primes_greater_than_500 :
  (∀ m : ℕ, sum_of_two_distinct_primes_greater_than_500 m → m ≥ 1012) ∧
  sum_of_two_distinct_primes_greater_than_500 1012 :=
sorry

end smallest_sum_of_two_distinct_primes_greater_than_500_l838_83841


namespace symmetric_points_sum_l838_83885

/-- Given two points P and Q that are symmetric with respect to the origin,
    prove that the sum of their x-coordinates plus the difference of their y-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (m - 1, 5) ∧ Q = (3, 2 - n) ∧ P = (-Q.1, -Q.2)) →
  m + n = 5 := by
sorry

end symmetric_points_sum_l838_83885


namespace men_in_room_l838_83874

theorem men_in_room (x : ℕ) 
  (h1 : 2 * (5 * x - 3) = 24) -- Women doubled and final count is 24
  : 4 * x + 2 = 14 := by
  sorry

end men_in_room_l838_83874


namespace cloth_sale_calculation_l838_83829

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 85

/-- The total selling price in dollars -/
def total_selling_price : ℕ := 8925

/-- The profit per meter of cloth in dollars -/
def profit_per_meter : ℕ := 15

/-- The cost price per meter of cloth in dollars -/
def cost_price_per_meter : ℕ := 90

/-- Theorem stating that the number of meters of cloth sold is correct -/
theorem cloth_sale_calculation :
  meters_of_cloth * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end cloth_sale_calculation_l838_83829


namespace discount_difference_l838_83858

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  bill = 8000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.26 ∧ 
  second_discount = 0.05 → 
  (bill * (1 - first_discount) * (1 - second_discount)) - (bill * (1 - single_discount)) = 24 := by
  sorry

end discount_difference_l838_83858


namespace expression_equals_one_l838_83833

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a - b + c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
  sorry

end expression_equals_one_l838_83833


namespace quadratic_root_problem_l838_83845

theorem quadratic_root_problem (k : ℤ) (b c : ℤ) (h1 : k > 9) 
  (h2 : k^2 - b*k + c = 0) (h3 : b = 2*k + 1) 
  (h4 : (k-7)^2 - b*(k-7) + c = 0) : c = 3*k := by
  sorry

end quadratic_root_problem_l838_83845


namespace polynomial_root_nature_l838_83859

def P (x : ℝ) : ℝ := x^6 - 5*x^5 - 7*x^3 - 2*x + 9

theorem polynomial_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end polynomial_root_nature_l838_83859


namespace computer_game_cost_l838_83813

/-- The cost of a computer game, given the total cost of movie tickets and the total spent on entertainment. -/
theorem computer_game_cost (movie_tickets_cost total_spent : ℕ) : 
  movie_tickets_cost = 36 → total_spent = 102 → total_spent - movie_tickets_cost = 66 := by
  sorry

end computer_game_cost_l838_83813


namespace problem_solution_l838_83821

theorem problem_solution (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
sorry

end problem_solution_l838_83821


namespace chip_rearrangement_l838_83854

/-- Represents a color of a chip -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a position in the rectangle -/
structure Position where
  row : Fin 3
  col : Nat

/-- Represents the state of the rectangle -/
def Rectangle (n : Nat) := Position → Color

/-- Checks if a given rectangle arrangement is valid -/
def isValidArrangement (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ c : Color, ∀ i : Fin 3, ∃ j : Fin n, rect ⟨i, j⟩ = c

/-- Checks if a given rectangle arrangement satisfies the condition -/
def satisfiesCondition (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ j : Fin n, ∀ c : Color, ∃ i : Fin 3, rect ⟨i, j⟩ = c

/-- The main theorem to be proved -/
theorem chip_rearrangement (n : Nat) :
  ∃ (rect : Rectangle n), isValidArrangement n rect ∧ satisfiesCondition n rect := by
  sorry


end chip_rearrangement_l838_83854


namespace sine_symmetry_sum_l838_83817

open Real

theorem sine_symmetry_sum (α β : ℝ) :
  0 ≤ α ∧ α < π ∧
  0 ≤ β ∧ β < π ∧
  α ≠ β ∧
  sin (2 * α + π / 3) = 1 / 2 ∧
  sin (2 * β + π / 3) = 1 / 2 →
  α + β = 7 * π / 6 := by
  sorry

end sine_symmetry_sum_l838_83817


namespace field_of_miracles_l838_83873

/-- The Field of Miracles problem -/
theorem field_of_miracles
  (a b : ℝ)
  (ha : a = 6)
  (hb : b = 2.5)
  (v_malvina : ℝ)
  (hv_malvina : v_malvina = 4)
  (v_buratino : ℝ)
  (hv_buratino : v_buratino = 6)
  (v_artemon : ℝ)
  (hv_artemon : v_artemon = 12) :
  let d := Real.sqrt (a^2 + b^2)
  let t := d / (v_malvina + v_buratino)
  v_artemon * t = 7.8 :=
by sorry

end field_of_miracles_l838_83873


namespace divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l838_83889

theorem divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n (n : ℕ) :
  let m := ⌈(Real.sqrt 3 + 1)^(2*n)⌉
  ∃ k : ℕ, m = 2^(n+1) * k :=
sorry

end divisibility_of_smallest_integer_greater_than_sqrt3_plus_1_power_2n_l838_83889


namespace square_perimeter_relation_l838_83892

theorem square_perimeter_relation (C D : Real) : 
  (C = 16) → -- perimeter of square C is 16 cm
  (D^2 = (C/4)^2 / 3) → -- area of D is one-third the area of C
  (4 * D = 16 * Real.sqrt 3 / 3) -- perimeter of D is 16√3/3 cm
  := by sorry

end square_perimeter_relation_l838_83892


namespace stones_partition_exists_l838_83843

/-- A partition of n into k parts is a list of k positive integers that sum to n. -/
def IsPartition (n k : ℕ) (partition : List ℕ) : Prop :=
  partition.length = k ∧ 
  partition.all (· > 0) ∧
  partition.sum = n

/-- A partition is similar if the maximum value is less than twice the minimum value. -/
def IsSimilarPartition (partition : List ℕ) : Prop :=
  partition.maximum? ≠ none ∧ 
  partition.minimum? ≠ none ∧ 
  (partition.maximum?.get! < 2 * partition.minimum?.get!)

theorem stones_partition_exists : 
  ∃ (partition : List ℕ), IsPartition 660 30 partition ∧ IsSimilarPartition partition := by
  sorry

end stones_partition_exists_l838_83843


namespace twenty_is_forty_percent_l838_83881

theorem twenty_is_forty_percent : ∃ x : ℝ, x = 55 ∧ 20 / (x - 5) = 0.4 := by
  sorry

end twenty_is_forty_percent_l838_83881


namespace pie_chart_most_suitable_l838_83850

/-- Represents a component of milk with its percentage -/
structure MilkComponent where
  name : String
  percentage : Float

/-- Represents a type of graph -/
inductive GraphType
  | PieChart
  | BarGraph
  | LineGraph
  | ScatterPlot

/-- Determines if a list of percentages sums to 100% (allowing for small floating-point errors) -/
def sumsToWhole (components : List MilkComponent) : Bool :=
  let sum := components.map (·.percentage) |>.sum
  sum > 99.99 && sum < 100.01

/-- Determines if a graph type is suitable for representing percentages of a whole -/
def isSuitableForPercentages (graphType : GraphType) : Bool :=
  match graphType with
  | GraphType.PieChart => true
  | _ => false

/-- Theorem: A pie chart is the most suitable graph type for representing milk components -/
theorem pie_chart_most_suitable (components : List MilkComponent) 
  (h_components : components = [
    ⟨"Water", 82⟩, 
    ⟨"Protein", 4.3⟩, 
    ⟨"Fat", 6⟩, 
    ⟨"Lactose", 7⟩, 
    ⟨"Other", 0.7⟩
  ])
  (h_sum : sumsToWhole components) :
  ∀ (graphType : GraphType), 
    isSuitableForPercentages graphType → graphType = GraphType.PieChart :=
by sorry

end pie_chart_most_suitable_l838_83850


namespace sum_of_roots_of_quartic_l838_83866

theorem sum_of_roots_of_quartic (x : ℝ) : 
  (∃ a b c d : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x^2 + a*x + b)*(x^2 + c*x + d)) →
  (∃ r₁ r₂ r₃ r₄ : ℝ, x^4 - 6*x^3 + 8*x - 3 = (x - r₁)*(x - r₂)*(x - r₃)*(x - r₄) ∧ r₁ + r₂ + r₃ + r₄ = 8) :=
by sorry

end sum_of_roots_of_quartic_l838_83866


namespace necessary_not_sufficient_condition_l838_83888

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x > y - 1) ∧
  (∃ x y : ℝ, x > y - 1 ∧ ¬(x > y)) :=
sorry

end necessary_not_sufficient_condition_l838_83888


namespace correct_set_for_60_deg_terminal_side_l838_83886

/-- The set of angles with the same terminal side as a 60° angle -/
def SameTerminalSideAs60Deg : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3}

/-- Theorem stating that SameTerminalSideAs60Deg is the correct set -/
theorem correct_set_for_60_deg_terminal_side :
  SameTerminalSideAs60Deg = {α | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3} := by
  sorry

end correct_set_for_60_deg_terminal_side_l838_83886


namespace f_condition_l838_83823

/-- The function f(x) = x^2 + 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a

/-- The theorem stating the condition for f(f(x)) > x for all x ∈ ℝ -/
theorem f_condition (a : ℝ) : 
  (∀ x : ℝ, f a (f a x) > x) ↔ (1 - Real.sqrt 3 / 2 < a ∧ a < 1 + Real.sqrt 3 / 2) := by
  sorry

end f_condition_l838_83823


namespace contrapositive_equivalence_l838_83838

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end contrapositive_equivalence_l838_83838


namespace power_zero_eq_one_neg_half_power_zero_l838_83803

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem neg_half_power_zero : (-1/2 : ℚ)^0 = 1 := by sorry

end power_zero_eq_one_neg_half_power_zero_l838_83803


namespace circle_intersection_theorem_l838_83893

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the dot product of vectors OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ :=
  P.1 * Q.1 + P.2 * Q.2

theorem circle_intersection_theorem (k : ℝ) :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  (∃ c : ℝ × ℝ, c ∈ circle_C ∧ c.2 = line_y_eq_x c.1) ∧
  (∃ P Q : ℝ × ℝ, P ∈ circle_C ∧ Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧ Q.2 = line_l k Q.1) ∧
  (∃ P Q : ℝ × ℝ, P ∈ circle_C ∧ Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧ Q.2 = line_l k Q.1 ∧
    dot_product_OP_OQ P Q = -2) →
  k = 0 := by sorry

end circle_intersection_theorem_l838_83893


namespace snowflake_puzzle_solution_l838_83856

-- Define the grid as a 3x3 matrix
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the valid numbers
def ValidNumbers : List Nat := [1, 2, 3, 4, 5, 6]

-- Define the function to check if a number is valid in a given position
def isValidPlacement (grid : Grid) (row col : Fin 3) (num : Nat) : Prop :=
  -- Check row
  (∀ j : Fin 3, j ≠ col → grid row j ≠ num) ∧
  -- Check column
  (∀ i : Fin 3, i ≠ row → grid i col ≠ num) ∧
  -- Check diagonal (if applicable)
  (row = col → ∀ i : Fin 3, i ≠ row → grid i i ≠ num) ∧
  (row + col = 2 → ∀ i : Fin 3, grid i (2 - i) ≠ num)

-- Define the partially filled grid (Figure 2)
def initialGrid : Grid := λ i j =>
  if i = 0 ∧ j = 0 then 3
  else if i = 2 ∧ j = 2 then 4
  else 0  -- 0 represents an empty cell

-- Define the positions of A, B, C, D
def posA : Fin 3 × Fin 3 := (0, 1)
def posB : Fin 3 × Fin 3 := (1, 0)
def posC : Fin 3 × Fin 3 := (1, 1)
def posD : Fin 3 × Fin 3 := (1, 2)

-- Theorem statement
theorem snowflake_puzzle_solution :
  ∀ (grid : Grid),
    (∀ i j, grid i j ∈ ValidNumbers ∪ {0}) →
    (∀ i j, initialGrid i j ≠ 0 → grid i j = initialGrid i j) →
    (∀ i j, grid i j ≠ 0 → isValidPlacement grid i j (grid i j)) →
    (grid posA.1 posA.2 = 2 ∧
     grid posB.1 posB.2 = 5 ∧
     grid posC.1 posC.2 = 1 ∧
     grid posD.1 posD.2 = 6) :=
  sorry

end snowflake_puzzle_solution_l838_83856


namespace max_peak_consumption_for_savings_l838_83871

/-- Proves that the maximum average monthly electricity consumption during peak hours
    that allows for at least 10% savings on the original electricity cost is ≤ 118 kWh --/
theorem max_peak_consumption_for_savings (
  original_price : ℝ) (peak_price : ℝ) (off_peak_price : ℝ) (total_consumption : ℝ) 
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : 0 < original_price ∧ 0 < peak_price ∧ 0 < off_peak_price)
  (h6 : total_consumption > 0) :
  let peak_consumption := 
    { x : ℝ | x ≥ 0 ∧ x ≤ total_consumption ∧ 
      (peak_price * x + off_peak_price * (total_consumption - x)) ≤ 
      0.9 * (original_price * total_consumption) }
  ∃ max_peak : ℝ, max_peak ∈ peak_consumption ∧ max_peak ≤ 118 ∧ 
    ∀ y ∈ peak_consumption, y ≤ max_peak := by
  sorry

#check max_peak_consumption_for_savings

end max_peak_consumption_for_savings_l838_83871


namespace interval_bound_l838_83857

theorem interval_bound (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end interval_bound_l838_83857


namespace reciprocal_of_negative_two_thirds_l838_83890

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = -3/2 := by
sorry

end reciprocal_of_negative_two_thirds_l838_83890


namespace exist_distinct_prime_divisors_l838_83875

theorem exist_distinct_prime_divisors (k n : ℕ+) (h : k > n!) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧
                     (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
                     (∀ i : Fin n, (p i) ∣ (k + i.val + 1)) := by
  sorry

end exist_distinct_prime_divisors_l838_83875


namespace y_never_perfect_square_l838_83897

theorem y_never_perfect_square (x : ℕ) : ∃ (n : ℕ), (x^4 + 2*x^3 + 2*x^2 + 2*x + 1) ≠ n^2 := by
  sorry

end y_never_perfect_square_l838_83897


namespace sam_distance_sam_drove_220_miles_l838_83842

/-- Calculates the total distance driven by Sam given Marguerite's speed and Sam's driving conditions. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_initial_time : ℝ) (sam_increased_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let marguerite_speed := marguerite_distance / marguerite_time
  let sam_initial_distance := marguerite_speed * sam_initial_time
  let sam_increased_speed := marguerite_speed * (1 + speed_increase)
  let sam_increased_distance := sam_increased_speed * sam_increased_time
  sam_initial_distance + sam_increased_distance

/-- Proves that Sam drove 220 miles given the problem conditions. -/
theorem sam_drove_220_miles : sam_distance 150 3 2 2 0.2 = 220 := by
  sorry

end sam_distance_sam_drove_220_miles_l838_83842


namespace simplify_expression_l838_83810

theorem simplify_expression (a b : ℝ) :
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := by
  sorry

end simplify_expression_l838_83810


namespace quadratic_inequality_l838_83878

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) →
  min c (a + c + 1) ≤ max (abs (b - a + 1)) (abs (b + a - 1)) ∧
  (min c (a + c + 1) = max (abs (b - a + 1)) (abs (b + a - 1)) ↔
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a ≤ -1 ∧ 2 * a - abs b + c = 0))) :=
by sorry

end quadratic_inequality_l838_83878


namespace least_integer_x_l838_83880

theorem least_integer_x : ∃ x : ℤ, (∀ z : ℤ, |3*z + 5 - 4| ≤ 25 → x ≤ z) ∧ |3*x + 5 - 4| ≤ 25 :=
by
  -- The proof goes here
  sorry

end least_integer_x_l838_83880


namespace valid_plates_count_l838_83872

/-- The number of digits available (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters available (A-Z) -/
def num_letters : ℕ := 26

/-- A license plate is valid if it satisfies the given conditions -/
def is_valid_plate (plate : Fin 4 → Char) : Prop :=
  (plate 0).isDigit ∧
  (plate 1).isAlpha ∧
  (plate 2).isAlpha ∧
  (plate 3).isDigit ∧
  plate 0 = plate 3

/-- The number of valid license plates -/
def num_valid_plates : ℕ := num_digits * num_letters * num_letters

theorem valid_plates_count :
  num_valid_plates = 6760 :=
sorry

end valid_plates_count_l838_83872


namespace product_first_three_terms_l838_83811

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference between consecutive terms
  d : ℝ
  -- The seventh term is 20
  seventh_term : a + 6 * d = 20
  -- The common difference is 2
  common_diff : d = 2

/-- The product of the first three terms of the arithmetic sequence is 960 -/
theorem product_first_three_terms (seq : ArithmeticSequence) :
  seq.a * (seq.a + seq.d) * (seq.a + 2 * seq.d) = 960 := by
  sorry


end product_first_three_terms_l838_83811


namespace system_solution_l838_83840

theorem system_solution (x y : ℝ) 
  (h1 : x * y = -8)
  (h2 : x^2 * y + x * y^2 + 3*x + 3*y = 100) :
  x^2 + y^2 = 416 := by
sorry

end system_solution_l838_83840


namespace possible_values_of_a_l838_83820

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (Q a ⊂ P ∧ Q a ≠ P) ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) :=
by sorry

end possible_values_of_a_l838_83820


namespace sphere_volume_from_surface_area_l838_83862

/-- The volume of a sphere with surface area 8π is equal to (8 * sqrt(2) * π) / 3 -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 8 * π → (4 / 3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3 := by
  sorry

end sphere_volume_from_surface_area_l838_83862


namespace annalise_purchase_l838_83884

/-- Represents the purchase of tissue boxes -/
structure TissuePurchase where
  packs_per_box : ℕ
  tissues_per_pack : ℕ
  tissue_cost_cents : ℕ
  total_spent_dollars : ℕ

/-- Calculates the number of boxes bought given a TissuePurchase -/
def boxes_bought (purchase : TissuePurchase) : ℕ :=
  (purchase.total_spent_dollars * 100) /
  (purchase.packs_per_box * purchase.tissues_per_pack * purchase.tissue_cost_cents)

/-- Theorem stating that Annalise bought 10 boxes -/
theorem annalise_purchase : 
  let purchase := TissuePurchase.mk 20 100 5 1000
  boxes_bought purchase = 10 := by
  sorry

end annalise_purchase_l838_83884


namespace apple_cost_l838_83864

/-- Given that apples cost m yuan per kilogram, prove that the cost of purchasing 3 kilograms of apples is 3m yuan. -/
theorem apple_cost (m : ℝ) : m * 3 = 3 * m := by
  sorry

end apple_cost_l838_83864


namespace decimal_period_11_13_l838_83848

/-- The length of the smallest repeating block in the decimal expansion of a rational number -/
def decimal_period (n d : ℕ) : ℕ :=
  sorry

/-- Theorem: The length of the smallest repeating block in the decimal expansion of 11/13 is 6 -/
theorem decimal_period_11_13 : decimal_period 11 13 = 6 := by
  sorry

end decimal_period_11_13_l838_83848


namespace land_properties_l838_83800

/-- Represents a piece of land with specific measurements -/
structure Land where
  triangle_area : ℝ
  ac_length : ℝ
  cd_length : ℝ
  de_length : ℝ

/-- Calculates the total area of the land -/
def total_area (land : Land) : ℝ := sorry

/-- Calculates the length of CF to divide the land equally -/
def equal_division_length (land : Land) : ℝ := sorry

theorem land_properties (land : Land) 
  (h1 : land.triangle_area = 120)
  (h2 : land.ac_length = 20)
  (h3 : land.cd_length = 10)
  (h4 : land.de_length = 10) :
  total_area land = 270 ∧ equal_division_length land = 1.5 := by sorry

end land_properties_l838_83800


namespace product_remainder_one_l838_83883

theorem product_remainder_one (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
sorry

end product_remainder_one_l838_83883


namespace one_student_reviewed_l838_83887

/-- Represents the students in the problem -/
inductive Student : Type
  | Zhang
  | Li
  | Wang
  | Zhao
  | Liu

/-- The statement made by each student about how many reviewed math -/
def statement (s : Student) : Nat :=
  match s with
  | Student.Zhang => 0
  | Student.Li => 1
  | Student.Wang => 2
  | Student.Zhao => 3
  | Student.Liu => 4

/-- Predicate to determine if a student reviewed math -/
def reviewed : Student → Prop := sorry

/-- The number of students who reviewed math -/
def num_reviewed : Nat := sorry

theorem one_student_reviewed :
  (∃ s : Student, reviewed s) ∧
  (∃ s : Student, ¬reviewed s) ∧
  (∀ s : Student, reviewed s ↔ statement s = num_reviewed) ∧
  (num_reviewed = 1) := by sorry

end one_student_reviewed_l838_83887


namespace test_total_points_l838_83896

/-- Given a test with the following properties:
  * Total number of questions is 30
  * Questions are either worth 5 or 10 points
  * There are 20 questions worth 5 points each
  Prove that the total point value of the test is 200 points -/
theorem test_total_points :
  ∀ (total_questions five_point_questions : ℕ)
    (point_values : Finset ℕ),
  total_questions = 30 →
  five_point_questions = 20 →
  point_values = {5, 10} →
  (total_questions - five_point_questions) * 10 + five_point_questions * 5 = 200 :=
by sorry

end test_total_points_l838_83896


namespace sock_ratio_proof_l838_83828

theorem sock_ratio_proof (green_socks red_socks : ℕ) (price_red : ℚ) :
  green_socks = 6 →
  (6 * (3 * price_red) + red_socks * price_red + 15 : ℚ) * (9/5) = 
    red_socks * (3 * price_red) + 6 * price_red + 15 →
  (green_socks : ℚ) / red_socks = 6 / 23 :=
by
  sorry

end sock_ratio_proof_l838_83828


namespace carl_reach_probability_l838_83824

-- Define the lily pad setup
def num_pads : ℕ := 16
def predator_pads : List ℕ := [4, 7, 12]
def start_pad : ℕ := 0
def goal_pad : ℕ := 14

-- Define Carl's movement probabilities
def hop_prob : ℚ := 1/2
def leap_prob : ℚ := 1/2

-- Define a function to calculate the probability of reaching a specific pad
def reach_prob (pad : ℕ) : ℚ :=
  sorry

-- State the theorem
theorem carl_reach_probability :
  reach_prob goal_pad = 3/512 :=
sorry

end carl_reach_probability_l838_83824


namespace problem_C_most_suitable_for_systematic_sampling_l838_83816

/-- Represents a sampling problem with population size and sample size -/
structure SamplingProblem where
  population_size : ℕ
  sample_size : ℕ

/-- Defines the suitability of a sampling method for a given problem -/
def systematic_sampling_suitability (problem : SamplingProblem) : ℕ :=
  if problem.population_size ≥ 1000 ∧ problem.sample_size ≥ 100 then 3
  else if problem.population_size < 100 ∨ problem.sample_size < 20 then 1
  else 2

/-- The sampling problems given in the question -/
def problem_A : SamplingProblem := ⟨48, 8⟩
def problem_B : SamplingProblem := ⟨210, 21⟩
def problem_C : SamplingProblem := ⟨1200, 100⟩
def problem_D : SamplingProblem := ⟨1200, 10⟩

/-- Theorem stating that problem C is most suitable for systematic sampling -/
theorem problem_C_most_suitable_for_systematic_sampling :
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_A ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_B ∧
  systematic_sampling_suitability problem_C > systematic_sampling_suitability problem_D :=
sorry

end problem_C_most_suitable_for_systematic_sampling_l838_83816


namespace prob_win_match_value_l838_83849

/-- Probability of player A winning a single game -/
def p : ℝ := 0.6

/-- Probability of player A winning the match in a best of 3 games -/
def prob_win_match : ℝ := p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem stating that the probability of player A winning the match is 0.648 -/
theorem prob_win_match_value : prob_win_match = 0.648 := by sorry

end prob_win_match_value_l838_83849
