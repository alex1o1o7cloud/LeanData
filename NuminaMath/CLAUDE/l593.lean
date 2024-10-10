import Mathlib

namespace zero_in_interval_l593_59367

def f (x : ℝ) : ℝ := x^3 + x - 4

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
sorry

end zero_in_interval_l593_59367


namespace quadratic_root_difference_l593_59305

theorem quadratic_root_difference :
  let a : ℝ := 3 + 2 * Real.sqrt 2
  let b : ℝ := 5 + Real.sqrt 2
  let c : ℝ := -4
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = Real.sqrt (177 - 122 * Real.sqrt 2) := by
  sorry

end quadratic_root_difference_l593_59305


namespace power_division_rule_l593_59304

theorem power_division_rule (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end power_division_rule_l593_59304


namespace divisibility_criterion_a_divisibility_criterion_b_l593_59355

-- Part a
theorem divisibility_criterion_a (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) + X^n + 1 = (X^2 + X + 1) * q) ↔ 
  (n % 3 = 1 ∨ n % 3 = 2) :=
sorry

-- Part b
theorem divisibility_criterion_b (n : ℕ) : 
  (∃ q : Polynomial ℚ, X^(2*n) - X^n + 1 = (X^2 - X + 1) * q) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end divisibility_criterion_a_divisibility_criterion_b_l593_59355


namespace minimum_value_of_expression_minimum_value_achieved_l593_59352

theorem minimum_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4 * a^2 + b^2)).sqrt) / (a * b) ≥ Real.sqrt 6 :=
sorry

theorem minimum_value_achieved (a : ℝ) (ha : a > 0) :
  (((a^2 + a^2) * (4 * a^2 + a^2)).sqrt) / (a * a) = Real.sqrt 6 :=
sorry

end minimum_value_of_expression_minimum_value_achieved_l593_59352


namespace min_value_expression_l593_59380

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 4) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 5 ∧
  ((x + 4) / Real.sqrt (x - 1) = 2 * Real.sqrt 5 ↔ x = 6) :=
sorry

end min_value_expression_l593_59380


namespace canada_sqft_per_person_approx_l593_59390

/-- The population of Canada in 2020 -/
def canada_population : ℕ := 38005238

/-- The total area of Canada in square miles -/
def canada_area : ℕ := 3855100

/-- The number of square feet in one square mile -/
def sqft_per_sqmile : ℕ := 5280^2

/-- Theorem stating that the average number of square feet per person in Canada
    is approximately 3,000,000 -/
theorem canada_sqft_per_person_approx :
  let total_sqft := canada_area * sqft_per_sqmile
  let avg_sqft_per_person := total_sqft / canada_population
  ∃ (ε : ℝ), ε > 0 ∧ ε < 200000 ∧ 
    (avg_sqft_per_person : ℝ) ≥ 3000000 - ε ∧ 
    (avg_sqft_per_person : ℝ) ≤ 3000000 + ε :=
sorry

end canada_sqft_per_person_approx_l593_59390


namespace equation_solutions_l593_59332

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2) ∧
  (∀ x : ℝ, (x + 3)^3 = -27 ↔ x = -6) :=
by sorry

end equation_solutions_l593_59332


namespace range_of_a_range_of_g_l593_59369

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 2*a + 12

-- Define the function g
def g (a : ℝ) : ℝ := (a + 1) * (|a - 1| + 2)

-- Theorem 1: Range of a
theorem range_of_a (h : ∀ x : ℝ, f a x ≥ 0) : a ∈ Set.Icc (-3/2) 2 :=
sorry

-- Theorem 2: Range of g(a)
theorem range_of_g : Set.range g = Set.Icc (-9/4) 9 :=
sorry

end range_of_a_range_of_g_l593_59369


namespace polynomial_identity_l593_59310

theorem polynomial_identity (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) :
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
  sorry

end polynomial_identity_l593_59310


namespace circle_through_points_with_center_on_y_axis_l593_59356

/-- The circle passing through points A (-1, 4) and B (3, 2) with its center on the y-axis -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 10

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (-1, 4)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (3, 2)

/-- The center of the circle is on the y-axis -/
def center_on_y_axis (h k : ℝ) : Prop :=
  h = 0

theorem circle_through_points_with_center_on_y_axis :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  ∃ k, center_on_y_axis 0 k ∧
    ∀ x y, circle_equation x y ↔ (x - 0)^2 + (y - k)^2 = (0 - point_A.1)^2 + (k - point_A.2)^2 :=
sorry

end circle_through_points_with_center_on_y_axis_l593_59356


namespace race_distance_l593_59361

theorem race_distance (a : ℝ) (r : ℝ) (S_n : ℝ) (n : ℕ) :
  a = 10 ∧ r = 2 ∧ S_n = 310 ∧ S_n = a * (r^n - 1) / (r - 1) →
  2^n = 32 := by
  sorry

end race_distance_l593_59361


namespace fraction_to_decimal_l593_59338

theorem fraction_to_decimal : 49 / 160 = 0.30625 := by
  sorry

end fraction_to_decimal_l593_59338


namespace millet_exceeds_60_percent_on_day_4_l593_59342

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  millet : Float
  other_seeds : Float

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { millet := state.millet * 0.7 + 0.3,
    other_seeds := state.other_seeds * 0.5 + 0.7 }

/-- Calculates the proportion of millet in the feeder -/
def millet_proportion (state : FeederState) : Float :=
  state.millet / (state.millet + state.other_seeds)

/-- Initial state of the feeder -/
def initial_state : FeederState := { millet := 0.3, other_seeds := 0.7 }

theorem millet_exceeds_60_percent_on_day_4 :
  let day1 := initial_state
  let day2 := next_day_state day1
  let day3 := next_day_state day2
  let day4 := next_day_state day3
  (millet_proportion day1 ≤ 0.6) ∧
  (millet_proportion day2 ≤ 0.6) ∧
  (millet_proportion day3 ≤ 0.6) ∧
  (millet_proportion day4 > 0.6) :=
by sorry

end millet_exceeds_60_percent_on_day_4_l593_59342


namespace work_left_fraction_l593_59375

theorem work_left_fraction (a_days : ℕ) (b_days : ℕ) (work_days : ℕ) :
  a_days = 15 →
  b_days = 20 →
  work_days = 4 →
  1 - (work_days * (1 / a_days + 1 / b_days)) = 8 / 15 :=
by sorry

end work_left_fraction_l593_59375


namespace cubic_root_sum_cubes_l593_59378

theorem cubic_root_sum_cubes (x y z : ℝ) : 
  (x^3 - 5*x - 3 = 0) → 
  (y^3 - 5*y - 3 = 0) → 
  (z^3 - 5*z - 3 = 0) → 
  x^3 * y^3 + x^3 * z^3 + y^3 * z^3 = 99 := by
sorry

end cubic_root_sum_cubes_l593_59378


namespace balloon_difference_l593_59321

theorem balloon_difference (x y : ℚ) 
  (eq1 : x = 2 * y - 3)
  (eq2 : y = x / 4 + 1) : 
  x - y = -5/2 := by sorry

end balloon_difference_l593_59321


namespace sam_nickels_count_l593_59341

/-- Calculates the final number of nickels Sam has -/
def final_nickels (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

theorem sam_nickels_count : final_nickels 29 24 13 = 40 := by
  sorry

end sam_nickels_count_l593_59341


namespace unique_solution_to_equation_l593_59370

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^4 * (y.val : ℝ)^4 - 16 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 15 = 0 :=
by sorry

end unique_solution_to_equation_l593_59370


namespace sam_spent_12_dimes_on_baseball_cards_l593_59307

/-- The number of pennies Sam spent on ice cream -/
def ice_cream_pennies : ℕ := 2

/-- The total amount Sam spent in cents -/
def total_spent_cents : ℕ := 122

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Sam spent on baseball cards -/
def baseball_card_dimes : ℕ := 12

theorem sam_spent_12_dimes_on_baseball_cards :
  (total_spent_cents - ice_cream_pennies * penny_value) / dime_value = baseball_card_dimes := by
  sorry

end sam_spent_12_dimes_on_baseball_cards_l593_59307


namespace parabola_sum_coefficients_l593_59372

-- Define the parabola equation
def parabola_eq (a b c : ℝ) (x y : ℝ) : Prop := x = a * y^2 + b * y + c

-- State the theorem
theorem parabola_sum_coefficients :
  ∀ a b c : ℝ,
  (parabola_eq a b c 6 (-5)) →
  (parabola_eq a b c 2 (-1)) →
  a + b + c = -3.25 := by
sorry

end parabola_sum_coefficients_l593_59372


namespace trevor_dropped_eggs_l593_59331

/-- The number of eggs Trevor collected from each chicken and the number left after dropping some -/
structure EggCollection where
  gertrude : Nat
  blanche : Nat
  nancy : Nat
  martha : Nat
  left : Nat

/-- The total number of eggs collected -/
def total_eggs (e : EggCollection) : Nat :=
  e.gertrude + e.blanche + e.nancy + e.martha

/-- The number of eggs Trevor dropped -/
def dropped_eggs (e : EggCollection) : Nat :=
  total_eggs e - e.left

theorem trevor_dropped_eggs (e : EggCollection) 
  (h1 : e.gertrude = 4)
  (h2 : e.blanche = 3)
  (h3 : e.nancy = 2)
  (h4 : e.martha = 2)
  (h5 : e.left = 9) :
  dropped_eggs e = 2 := by
  sorry

#check trevor_dropped_eggs

end trevor_dropped_eggs_l593_59331


namespace base6_multiplication_l593_59362

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Define the multiplication operation in base 6
def multBase6 (a b : ℕ) : ℕ := 
  base10ToBase6 (base6ToBase10 a * base6ToBase10 b)

-- Theorem statement
theorem base6_multiplication :
  multBase6 132 14 = 1332 := by sorry

end base6_multiplication_l593_59362


namespace quadratic_minimum_l593_59385

theorem quadratic_minimum (x : ℝ) : ∃ (min : ℝ), min = -29 ∧ ∀ y : ℝ, x^2 + 14*x + 20 ≥ min := by
  sorry

end quadratic_minimum_l593_59385


namespace coffee_shop_tables_l593_59300

def base7_to_base10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

theorem coffee_shop_tables (chairs_base7 : Nat) (people_per_table : Nat) : 
  chairs_base7 = 321 ∧ people_per_table = 3 → 
  (base7_to_base10 chairs_base7) / people_per_table = 54 := by
  sorry

end coffee_shop_tables_l593_59300


namespace square_difference_262_258_l593_59325

theorem square_difference_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end square_difference_262_258_l593_59325


namespace range_of_a_l593_59348

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : 1/2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l593_59348


namespace square_side_length_l593_59387

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end square_side_length_l593_59387


namespace no_good_tetrahedron_in_good_parallelepiped_l593_59337

/-- A polyhedron is considered "good" if its volume equals its surface area -/
def isGoodPolyhedron (volume : ℝ) (surfaceArea : ℝ) : Prop :=
  volume = surfaceArea

/-- Properties of a tetrahedron -/
structure Tetrahedron where
  volume : ℝ
  surfaceArea : ℝ
  inscribedSphereRadius : ℝ

/-- Properties of a parallelepiped -/
structure Parallelepiped where
  volume : ℝ
  faceAreas : Fin 3 → ℝ
  heights : Fin 3 → ℝ

/-- Theorem stating the impossibility of fitting a good tetrahedron inside a good parallelepiped -/
theorem no_good_tetrahedron_in_good_parallelepiped :
  ∀ (t : Tetrahedron) (p : Parallelepiped),
    isGoodPolyhedron t.volume t.surfaceArea →
    isGoodPolyhedron p.volume (2 * (p.faceAreas 0 + p.faceAreas 1 + p.faceAreas 2)) →
    t.inscribedSphereRadius = 3 →
    ¬(∃ (h : ℝ), h = p.heights 0 ∧ h > 2 * t.inscribedSphereRadius) :=
by sorry

end no_good_tetrahedron_in_good_parallelepiped_l593_59337


namespace pure_imaginary_condition_l593_59389

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (y : ℝ), m^2 + m - 2 + (m^2 - 1) * Complex.I = y * Complex.I) → m = -2 := by
  sorry

end pure_imaginary_condition_l593_59389


namespace log_5_12_in_terms_of_m_n_l593_59309

theorem log_5_12_in_terms_of_m_n (m n : ℝ) 
  (h1 : Real.log 2 / Real.log 10 = m) 
  (h2 : Real.log 3 / Real.log 10 = n) : 
  Real.log 12 / Real.log 5 = (2*m + n) / (1 - m) := by
  sorry

end log_5_12_in_terms_of_m_n_l593_59309


namespace doughnuts_remaining_l593_59393

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts initially in the box -/
def initial_dozens : ℕ := 2

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := 8

/-- The number of doughnuts left in the box -/
def doughnuts_left : ℕ := initial_dozens * dozen - eaten_doughnuts

theorem doughnuts_remaining : doughnuts_left = 16 := by
  sorry

end doughnuts_remaining_l593_59393


namespace greatest_three_digit_multiple_of_17_l593_59346

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l593_59346


namespace proportional_relationship_l593_59394

theorem proportional_relationship (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ k₁ > 0, x = k₁ * y^3) →
  (∃ k₂ > 0, y = k₂ / z^2) →
  (x = 8 ∧ z = 16) →
  (z = 64 → x = 1/256) :=
by sorry

end proportional_relationship_l593_59394


namespace probability_two_non_defective_10_2_l593_59353

/-- Given a box of pens, calculates the probability of selecting two non-defective pens. -/
def probability_two_non_defective (total_pens : ℕ) (defective_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective : ℚ) / total_pens * (non_defective - 1) / (total_pens - 1)

/-- Theorem stating that the probability of selecting two non-defective pens
    from a box of 10 pens with 2 defective pens is 28/45. -/
theorem probability_two_non_defective_10_2 :
  probability_two_non_defective 10 2 = 28 / 45 := by
  sorry

end probability_two_non_defective_10_2_l593_59353


namespace parabola_equation_l593_59306

/-- A parabola with vertex at the origin and passing through (-4, 4) has the standard equation y² = -4x or x² = 4y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) 
  (vertex_origin : p 0 0)
  (passes_through : p (-4) 4) :
  (∀ x y, p x y ↔ y^2 = -4*x) ∨ (∀ x y, p x y ↔ x^2 = 4*y) :=
sorry

end parabola_equation_l593_59306


namespace rectangle_area_equality_l593_59335

theorem rectangle_area_equality (x y : ℝ) : 
  x * y = (x + 4) * (y - 3) ∧ 
  x * y = (x + 8) * (y - 4) → 
  x + y = 10 := by
sorry

end rectangle_area_equality_l593_59335


namespace unique_number_satisfying_conditions_l593_59363

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def first_digit (n : ℕ) : ℕ := n / 10
def second_digit (n : ℕ) : ℕ := n % 10

def reverse_number (n : ℕ) : ℕ := 10 * (second_digit n) + (first_digit n)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧ 
    (n : ℚ) / ((first_digit n * second_digit n) : ℚ) = 8 / 3 ∧
    n - reverse_number n = 18 ∧
    n = 64 := by
  sorry

end unique_number_satisfying_conditions_l593_59363


namespace stating_production_constraint_equations_l593_59398

/-- Represents the daily production capacity for type A toys -/
def type_A_production : ℕ := 200

/-- Represents the daily production capacity for type B toys -/
def type_B_production : ℕ := 100

/-- Represents the number of type A parts required for one complete toy -/
def type_A_parts_per_toy : ℕ := 1

/-- Represents the number of type B parts required for one complete toy -/
def type_B_parts_per_toy : ℕ := 2

/-- Represents the total number of production days -/
def total_days : ℕ := 30

/-- 
Theorem stating that the given system of equations correctly represents 
the production constraints for maximizing toy assembly within 30 days
-/
theorem production_constraint_equations 
  (x y : ℕ) : 
  (x + y = total_days ∧ 
   type_A_production * type_A_parts_per_toy * x = type_B_production * y) ↔ 
  (x + y = 30 ∧ 400 * x = 100 * y) :=
sorry

end stating_production_constraint_equations_l593_59398


namespace trip_time_difference_l593_59373

def speed : ℝ := 60
def distance1 : ℝ := 360
def distance2 : ℝ := 420

theorem trip_time_difference : 
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end trip_time_difference_l593_59373


namespace convex_polyhedron_three_equal_edges_l593_59318

/-- Represents an edge of a polyhedron --/
structure Edge :=
  (length : ℝ)

/-- Represents a vertex of a polyhedron --/
structure Vertex :=
  (edges : Fin 3 → Edge)

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron :=
  (vertices : Set Vertex)
  (convex : Bool)
  (edge_equality : ∀ v : Vertex, v ∈ vertices → ∃ (i j : Fin 3), i ≠ j ∧ (v.edges i).length = (v.edges j).length)

/-- The main theorem: if a convex polyhedron satisfies the given conditions, it has at least three equal edges --/
theorem convex_polyhedron_three_equal_edges (P : ConvexPolyhedron) : 
  ∃ (e₁ e₂ e₃ : Edge), e₁ ≠ e₂ ∧ e₂ ≠ e₃ ∧ e₁ ≠ e₃ ∧ e₁.length = e₂.length ∧ e₂.length = e₃.length :=
sorry

end convex_polyhedron_three_equal_edges_l593_59318


namespace medical_team_selection_l593_59376

theorem medical_team_selection (nurses : ℕ) (doctors : ℕ) : 
  nurses = 3 → doctors = 6 → 
  (Nat.choose (nurses + doctors) 5 - Nat.choose doctors 5) = 120 := by
  sorry

end medical_team_selection_l593_59376


namespace division_remainder_proof_l593_59374

theorem division_remainder_proof (dividend : ℕ) (divisor : ℚ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 13698 →
  divisor = 153.75280898876406 →
  quotient = 89 →
  dividend = (divisor * quotient).floor + remainder →
  remainder = 14 := by
sorry

end division_remainder_proof_l593_59374


namespace parabola_latus_rectum_p_l593_59366

/-- Given a parabola with equation x^2 = 2py (p > 0) and latus rectum equation y = -3,
    prove that the value of p is 6. -/
theorem parabola_latus_rectum_p (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, x^2 = 2*p*y) → (∃ x : ℝ, x^2 = 2*p*(-3)) → p = 6 := by
  sorry

end parabola_latus_rectum_p_l593_59366


namespace distribution_theorem_l593_59314

/-- Represents the amount of money spent by each person -/
structure Spending where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the amount of money received by Person A and Person B -/
structure Distribution where
  a : ℚ
  b : ℚ

/-- Calculates the distribution of money based on spending ratios -/
def calculateDistribution (s : Spending) (total : ℚ) : Distribution :=
  let ratio_sum := s.a + s.b
  let unit_value := total / ratio_sum
  { a := (s.a - s.b) * unit_value,
    b := (s.b - s.a) * unit_value + total }

/-- The main theorem to prove -/
theorem distribution_theorem (s : Spending) :
  s.b = 12/13 * s.a →
  s.c = 2/3 * s.b →
  let d := calculateDistribution s 9
  d.a = 6 ∧ d.b = 3 := by
  sorry


end distribution_theorem_l593_59314


namespace paper_thickness_l593_59323

/-- Given that 400 sheets of paper are 4 cm thick, prove that 600 sheets of the same paper would be 6 cm thick. -/
theorem paper_thickness (sheets : ℕ) (thickness : ℝ) 
  (h1 : 400 * (thickness / 400) = 4) -- 400 sheets are 4 cm thick
  (h2 : sheets = 600) -- We want to prove for 600 sheets
  : sheets * (thickness / 400) = 6 := by
  sorry

end paper_thickness_l593_59323


namespace discount_percentage_proof_l593_59343

/-- Given a marked price and cost price, where the cost price is 25% of the marked price,
    and a discount percentage such that the selling price after discount is equal to twice
    the cost price, prove that the discount percentage is 50%. -/
theorem discount_percentage_proof (MP CP : ℝ) (D : ℝ) 
    (h1 : CP = 0.25 * MP) 
    (h2 : MP * (1 - D / 100) = 2 * CP) : 
  D = 50 := by
  sorry

#check discount_percentage_proof

end discount_percentage_proof_l593_59343


namespace intersection_A_complement_B_l593_59329

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end intersection_A_complement_B_l593_59329


namespace investment_ratio_l593_59382

theorem investment_ratio (P Q : ℝ) (h : P > 0 ∧ Q > 0) :
  (P * 5) / (Q * 9) = 7 / 9 → P / Q = 7 / 5 := by
  sorry

end investment_ratio_l593_59382


namespace marie_glue_sticks_l593_59319

theorem marie_glue_sticks :
  ∀ (allison_glue allison_paper marie_glue marie_paper : ℕ),
    allison_glue = marie_glue + 8 →
    marie_paper = 6 * allison_paper →
    marie_paper = 30 →
    allison_glue + allison_paper = 28 →
    marie_glue = 15 := by
  sorry

end marie_glue_sticks_l593_59319


namespace basketball_score_l593_59354

/-- Calculates the total points scored in a basketball game given the number of 2-point and 3-point shots made. -/
def totalPoints (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that 7 two-point shots and 3 three-point shots result in a total of 23 points. -/
theorem basketball_score : totalPoints 7 3 = 23 := by
  sorry

end basketball_score_l593_59354


namespace robin_pieces_count_l593_59347

theorem robin_pieces_count (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end robin_pieces_count_l593_59347


namespace triangle_theorem_l593_59311

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : 2 * t.a * Real.sin t.A = (2 * t.b - t.c) * Real.sin t.B + (2 * t.c - t.b) * Real.sin t.C) :
  t.A = Real.pi / 3 ∧ 
  (Real.sin t.B + Real.sin t.C = Real.sqrt 3 → t.A = t.B ∧ t.B = t.C) :=
sorry

end triangle_theorem_l593_59311


namespace simplest_fraction_l593_59334

variable (x : ℝ)

-- Define the fractions
def f1 : ℚ → ℚ := λ x => 4 / (2 * x)
def f2 : ℚ → ℚ := λ x => (x - 1) / (x^2 - 1)
def f3 : ℚ → ℚ := λ x => 1 / (x + 1)
def f4 : ℚ → ℚ := λ x => (1 - x) / (x - 1)

-- Define what it means for a fraction to be simplest
def is_simplest (f : ℚ → ℚ) : Prop :=
  ∀ g : ℚ → ℚ, (∀ x, f x = g x) → f = g

-- Theorem statement
theorem simplest_fraction :
  is_simplest f3 ∧ ¬is_simplest f1 ∧ ¬is_simplest f2 ∧ ¬is_simplest f4 := by
  sorry

end simplest_fraction_l593_59334


namespace fraction_simplification_l593_59396

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1/3 := by
  sorry

end fraction_simplification_l593_59396


namespace complex_quadrant_l593_59320

theorem complex_quadrant (z : ℂ) (h : (1 - I) / (z - 2) = 1 + I) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end complex_quadrant_l593_59320


namespace projection_a_onto_b_l593_59381

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem projection_a_onto_b :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  (dot_product / magnitude_b) = Real.sqrt 5 := by sorry

end projection_a_onto_b_l593_59381


namespace worker_payment_schedule_l593_59368

/-- Represents the worker payment schedule problem -/
theorem worker_payment_schedule 
  (daily_wage : ℕ) 
  (daily_return : ℕ) 
  (days_not_worked : ℕ) : 
  daily_wage = 100 → 
  daily_return = 25 → 
  days_not_worked = 24 → 
  ∃ (days_worked : ℕ), 
    daily_wage * days_worked = daily_return * days_not_worked ∧ 
    days_worked + days_not_worked = 30 := by
  sorry

end worker_payment_schedule_l593_59368


namespace smallest_positive_angle_l593_59301

/-- Given a point P on the terminal side of angle α with coordinates 
    (sin(2π/3), cos(2π/3)), prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (α : Real) : 
  (∃ (P : Real × Real), P.1 = Real.sin (2 * Real.pi / 3) ∧ 
                         P.2 = Real.cos (2 * Real.pi / 3) ∧ 
                         P ∈ {(x, y) | x = Real.sin α ∧ y = Real.cos α}) →
  (∀ β : Real, β > 0 ∧ β < α → β ≥ 11 * Real.pi / 6) ∧ 
  α = 11 * Real.pi / 6 := by
  sorry

end smallest_positive_angle_l593_59301


namespace square_product_extension_l593_59383

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end square_product_extension_l593_59383


namespace digit_150_of_75_over_625_l593_59339

theorem digit_150_of_75_over_625 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a b : ℕ), (75 : ℚ) / 625 = ↑a + (↑b / 100) ∧ 
  (∀ n : ℕ, (75 * 10^(n+2)) % 625 = (75 * 10^(n+150)) % 625) ∧
  d = ((75 * 10^150) / 625) % 10) :=
sorry

end digit_150_of_75_over_625_l593_59339


namespace triangle_area_prime_l593_59303

/-- The area of a triangle formed by the line y = 10x - a and the coordinate axes -/
def triangleArea (a : ℤ) : ℚ := a^2 / 20

/-- Predicate to check if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

theorem triangle_area_prime :
  ∀ a : ℤ,
  (∃ n : ℕ, (triangleArea a).num = n ∧ (triangleArea a).den = 1 ∧ isPrime n) →
  triangleArea a = 5 :=
sorry

end triangle_area_prime_l593_59303


namespace park_area_is_525_l593_59360

/-- Represents a rectangular park with given perimeter and length-width relationship. -/
structure RectangularPark where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_eq : length = 3 * width - 10

/-- Calculates the area of a rectangular park. -/
def parkArea (park : RectangularPark) : ℝ := park.length * park.width

/-- Theorem stating that a rectangular park with perimeter 100 meters and length equal to
    three times the width minus 10 meters has an area of 525 square meters. -/
theorem park_area_is_525 (park : RectangularPark) 
    (h_perimeter : park.perimeter = 100) : parkArea park = 525 := by
  sorry

end park_area_is_525_l593_59360


namespace members_playing_two_sports_l593_59365

theorem members_playing_two_sports
  (total_members : ℕ)
  (badminton_players : ℕ)
  (tennis_players : ℕ)
  (soccer_players : ℕ)
  (no_sport_players : ℕ)
  (badminton_tennis : ℕ)
  (badminton_soccer : ℕ)
  (tennis_soccer : ℕ)
  (h1 : total_members = 60)
  (h2 : badminton_players = 25)
  (h3 : tennis_players = 32)
  (h4 : soccer_players = 14)
  (h5 : no_sport_players = 5)
  (h6 : badminton_tennis = 10)
  (h7 : badminton_soccer = 8)
  (h8 : tennis_soccer = 6)
  (h9 : badminton_tennis + badminton_soccer + tennis_soccer ≤ badminton_players + tennis_players + soccer_players) :
  badminton_tennis + badminton_soccer + tennis_soccer = 24 :=
by sorry

end members_playing_two_sports_l593_59365


namespace log_function_fixed_point_l593_59312

theorem log_function_fixed_point (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by
sorry

end log_function_fixed_point_l593_59312


namespace tile_count_l593_59333

def room_length : ℕ := 18
def room_width : ℕ := 24
def border_width : ℕ := 2
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length + room_width - 2 * border_width) * border_width

def interior_length : ℕ := room_length - 2 * border_width
def interior_width : ℕ := room_width - 2 * border_width

def interior_tiles : ℕ := 
  (interior_length * interior_width) / (large_tile_size * large_tile_size)

def total_tiles : ℕ := border_tiles + interior_tiles

theorem tile_count : total_tiles = 167 := by
  sorry

end tile_count_l593_59333


namespace transform_quadratic_l593_59317

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 + 2

/-- The transformed function -/
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 + 1

/-- Theorem stating that f is the result of transforming g -/
theorem transform_quadratic : 
  ∀ x : ℝ, f x = g (x + 3) - 1 := by sorry

end transform_quadratic_l593_59317


namespace min_difference_triangle_sides_l593_59349

theorem min_difference_triangle_sides (a b c : ℕ) : 
  a < b → b < c → a + b + c = 2509 → 
  (∀ x y z : ℕ, x < y ∧ y < z ∧ x + y + z = 2509 → y - x ≥ b - a) → 
  b - a = 1 :=
sorry

end min_difference_triangle_sides_l593_59349


namespace abc_product_l593_59379

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 152)
  (h2 : b * (c + a) = 162)
  (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end abc_product_l593_59379


namespace rectangles_with_at_least_three_cells_l593_59327

/-- The number of rectangles containing at least three cells in a 6x6 grid -/
def rectanglesWithAtLeastThreeCells : ℕ := 345

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- Total number of rectangles in an n x n grid -/
def totalRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Number of 1x1 rectangles in an n x n grid -/
def oneByOneRectangles (n : ℕ) : ℕ := n * n

/-- Number of 1x2 and 2x1 rectangles in an n x n grid -/
def oneBytwoRectangles (n : ℕ) : ℕ := 2 * n * (n - 1)

theorem rectangles_with_at_least_three_cells :
  rectanglesWithAtLeastThreeCells = 
    totalRectangles gridSize - oneByOneRectangles gridSize - oneBytwoRectangles gridSize :=
by sorry

end rectangles_with_at_least_three_cells_l593_59327


namespace intersection_volume_of_reflected_tetrahedron_l593_59386

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ
  is_regular : Bool

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_through_center (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_of_reflected_tetrahedron (t : RegularTetrahedron) 
  (h1 : t.volume = 1)
  (h2 : t.is_regular = true) :
  tetrahedra_intersection t (reflect_through_center t) = 1/2 := by
  sorry

end intersection_volume_of_reflected_tetrahedron_l593_59386


namespace prob_same_color_l593_59388

/-- Represents the contents of a bag of colored balls -/
structure BagContents where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.red + bag.black

/-- Represents the two bags in the problem -/
def bagA : BagContents := { white := 1, red := 2, black := 3 }
def bagB : BagContents := { white := 2, red := 3, black := 1 }

/-- Calculates the probability of drawing a specific color from a bag -/
def probColor (bag : BagContents) (color : ℕ) : ℚ :=
  color / bag.total

/-- The main theorem: probability of drawing same color from both bags -/
theorem prob_same_color :
  (probColor bagA bagA.white * probColor bagB bagB.white) +
  (probColor bagA bagA.red * probColor bagB bagB.red) +
  (probColor bagA bagA.black * probColor bagB bagB.black) = 11 / 36 := by
  sorry

end prob_same_color_l593_59388


namespace power_function_increasing_interval_l593_59391

/-- Given a power function f(x) = x^a where a is a real number,
    and f(2) = √2, prove that the increasing interval of f is [0, +∞) -/
theorem power_function_increasing_interval
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 0, f x = x ^ a)
  (h2 : f 2 = Real.sqrt 2) :
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end power_function_increasing_interval_l593_59391


namespace petes_nickels_spent_l593_59324

theorem petes_nickels_spent (total_received : ℕ) (total_spent : ℕ) (raymonds_dimes_left : ℕ) 
  (h1 : total_received = 500)
  (h2 : total_spent = 200)
  (h3 : raymonds_dimes_left = 7) :
  (total_spent - (raymonds_dimes_left * 10)) / 5 = 14 := by
  sorry

end petes_nickels_spent_l593_59324


namespace sum_of_divisors_24_l593_59377

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_24 : sum_of_divisors 24 = 60 := by
  sorry

end sum_of_divisors_24_l593_59377


namespace rectangular_solid_surface_area_l593_59302

/-- The surface area of a rectangular solid with edge lengths a, b, and c -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

/-- The edge lengths of the rectangular solid are prime numbers -/
axiom prime_3 : Nat.Prime 3
axiom prime_5 : Nat.Prime 5
axiom prime_17 : Nat.Prime 17

/-- The edge lengths are different -/
axiom different_edges : 3 ≠ 5 ∧ 3 ≠ 17 ∧ 5 ≠ 17

theorem rectangular_solid_surface_area :
  surface_area 3 5 17 = 302 := by sorry

end rectangular_solid_surface_area_l593_59302


namespace hcf_of_ratio_numbers_l593_59350

def ratio_numbers (x : ℕ) : Fin 4 → ℕ
  | 0 => 2 * x
  | 1 => 3 * x
  | 2 => 4 * x
  | 3 => 5 * x

theorem hcf_of_ratio_numbers (x : ℕ) (h1 : Nat.lcm (ratio_numbers x 0) (Nat.lcm (ratio_numbers x 1) (Nat.lcm (ratio_numbers x 2) (ratio_numbers x 3))) = 3600)
  (h2 : Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3) = 4) :
  Nat.gcd (ratio_numbers x 0) (Nat.gcd (ratio_numbers x 1) (Nat.gcd (ratio_numbers x 2) (ratio_numbers x 3))) = 4 :=
by sorry

end hcf_of_ratio_numbers_l593_59350


namespace collinear_vectors_l593_59316

/-- Given vectors a and b, if 2a - b is collinear with b, then n = 9 -/
theorem collinear_vectors (a b : ℝ × ℝ) (n : ℝ) 
  (ha : a = (1, 3))
  (hb : b = (3, n))
  (hcol : ∃ (k : ℝ), 2 • a - b = k • b) :
  n = 9 := by
sorry

end collinear_vectors_l593_59316


namespace quadratic_equation_with_zero_sum_coefficients_l593_59345

theorem quadratic_equation_with_zero_sum_coefficients :
  ∃ (a b c : ℝ), a ≠ 0 ∧ a + b + c = 0 ∧ ∀ x, a * x^2 + b * x + c = 0 := by
  sorry

end quadratic_equation_with_zero_sum_coefficients_l593_59345


namespace magnitude_of_c_for_four_distinct_roots_l593_59397

-- Define the polynomial Q(x)
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 6*x + 18)

-- Theorem statement
theorem magnitude_of_c_for_four_distinct_roots (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧ (∀ x, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 35.25 := by
  sorry

end magnitude_of_c_for_four_distinct_roots_l593_59397


namespace fruit_difference_is_eight_l593_59326

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  redPeaches : ℕ
  yellowPeaches : ℕ
  greenPeaches : ℕ
  blueApples : ℕ
  purpleBananas : ℕ
  orangeKiwis : ℕ

/-- Calculates the difference between peaches and other fruits -/
def peachDifference (basket : FruitBasket) : ℕ :=
  (basket.greenPeaches + basket.yellowPeaches) - (basket.blueApples + basket.purpleBananas)

/-- The theorem to be proved -/
theorem fruit_difference_is_eight :
  ∃ (basket : FruitBasket),
    basket.redPeaches = 2 ∧
    basket.yellowPeaches = 6 ∧
    basket.greenPeaches = 14 ∧
    basket.blueApples = 4 ∧
    basket.purpleBananas = 8 ∧
    basket.orangeKiwis = 12 ∧
    peachDifference basket = 8 := by
  sorry

end fruit_difference_is_eight_l593_59326


namespace salesman_profit_l593_59315

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit : 
  let total_backpacks : ℕ := 48
  let cost_per_case : ℕ := 576
  let first_batch_sold : ℕ := 17
  let first_batch_price : ℕ := 18
  let second_batch_sold : ℕ := 10
  let second_batch_price : ℕ := 25
  let remaining_price : ℕ := 22
  
  let total_sales := 
    first_batch_sold * first_batch_price + 
    second_batch_sold * second_batch_price + 
    (total_backpacks - first_batch_sold - second_batch_sold) * remaining_price
  
  let profit := total_sales - cost_per_case
  
  profit = 442 := by sorry

end salesman_profit_l593_59315


namespace coefficient_x_squared_sum_powers_l593_59384

/-- The sum of the first n natural numbers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem coefficient_x_squared_sum_powers (n : ℕ) (h : n = 10) : 
  sum_triangular n = 165 := by
  sorry

#eval sum_triangular 10

end coefficient_x_squared_sum_powers_l593_59384


namespace complement_A_intersect_B_l593_59364

def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {6, 8} := by sorry

end complement_A_intersect_B_l593_59364


namespace vector_subtraction_l593_59322

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 4) → b = (1, 2) → a - 2 • b = (1, 0) := by sorry

end vector_subtraction_l593_59322


namespace tangent_line_to_circle_l593_59357

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle. -/
def is_tangent_line (a b c : ℝ) (r : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = r

theorem tangent_line_to_circle (b : ℝ) :
  is_tangent_line 2 (-1) b (Real.sqrt 5) ↔ b = 5 ∨ b = -5 := by
  sorry

end tangent_line_to_circle_l593_59357


namespace conic_is_ellipse_l593_59308

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  3 * x^2 + 5 * y^2 - 9 * x + 10 * y + 15 = 0

/-- Definition of an ellipse -/
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y, f x y ↔ A * (x - h)^2 + B * (y - k)^2 = 1

/-- Theorem: The given conic equation represents an ellipse -/
theorem conic_is_ellipse : is_ellipse conic_equation :=
sorry

end conic_is_ellipse_l593_59308


namespace hat_shop_pricing_l593_59351

theorem hat_shop_pricing (x : ℝ) : 
  let increased_price := 1.30 * x
  let final_price := 0.75 * increased_price
  final_price = 0.975 * x := by
sorry

end hat_shop_pricing_l593_59351


namespace power_sum_fifth_l593_59359

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end power_sum_fifth_l593_59359


namespace kim_cousins_count_l593_59344

theorem kim_cousins_count (total_gum : ℕ) (gum_per_cousin : ℕ) (cousin_count : ℕ) : 
  total_gum = 20 → gum_per_cousin = 5 → total_gum = gum_per_cousin * cousin_count → cousin_count = 4 := by
  sorry

end kim_cousins_count_l593_59344


namespace unique_twin_prime_trio_l593_59392

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_twin_prime_trio : 
  ∀ p : ℕ, is_prime p → p > 7 → ¬(is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4)) :=
sorry

end unique_twin_prime_trio_l593_59392


namespace total_cookies_l593_59330

theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) : 
  num_bags = 7 → cookies_per_bag = 2 → num_bags * cookies_per_bag = 14 :=
by
  sorry

end total_cookies_l593_59330


namespace complement_A_intersect_B_l593_59371

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end complement_A_intersect_B_l593_59371


namespace log_equality_implies_golden_ratio_l593_59336

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
sorry

end log_equality_implies_golden_ratio_l593_59336


namespace negative_square_cubed_l593_59328

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l593_59328


namespace zoo_visitors_ratio_l593_59399

theorem zoo_visitors_ratio :
  let friday_visitors : ℕ := 1250
  let saturday_visitors : ℕ := 3750
  (saturday_visitors : ℚ) / (friday_visitors : ℚ) = 3 := by
sorry

end zoo_visitors_ratio_l593_59399


namespace gasoline_price_increase_l593_59395

theorem gasoline_price_increase (initial_price initial_quantity : ℝ) 
  (h_price_increase : ℝ) (h_spending_increase : ℝ) (h_quantity_decrease : ℝ) :
  h_price_increase > 0 →
  h_spending_increase = 0.1 →
  h_quantity_decrease = 0.12 →
  initial_price * initial_quantity * (1 + h_spending_increase) = 
    initial_price * (1 + h_price_increase) * initial_quantity * (1 - h_quantity_decrease) →
  h_price_increase = 0.25 := by
  sorry

end gasoline_price_increase_l593_59395


namespace expression_value_l593_59313

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  (x - y)^2 - x*y = -9 :=
by sorry

end expression_value_l593_59313


namespace system_solution_l593_59358

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x * y = 500 ∧ x^(Real.log y) = 25) ↔
  ((x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100)) := by
sorry

end system_solution_l593_59358


namespace distance_to_place_l593_59340

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  ∃ (distance : ℝ), distance = 2.4 ∧ 
    round_trip_time = distance / (rowing_speed + current_velocity) + 
                      distance / (rowing_speed - current_velocity) :=
by
  sorry

#check distance_to_place

end distance_to_place_l593_59340
