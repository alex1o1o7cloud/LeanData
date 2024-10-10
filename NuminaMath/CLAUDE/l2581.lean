import Mathlib

namespace hyperbola_asymptote_l2581_258118

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
    (|x + c| - |x - c|) / (2 * c) = 1/3) →
  (∃ (k : ℝ), k = 2 * Real.sqrt 2 ∧ 
    ∀ (x y : ℝ), y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptote_l2581_258118


namespace smallest_sum_of_factors_l2581_258128

theorem smallest_sum_of_factors (a b c d : ℕ+) 
  (h : a * b * c * d = Nat.factorial 10) : 
  a + b + c + d ≥ 175 := by
  sorry

end smallest_sum_of_factors_l2581_258128


namespace red_tetrahedron_volume_l2581_258170

/-- The volume of a tetrahedron formed by red vertices in a cube with alternately colored vertices --/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := (1 / 3) * cube_side_length ^ 3 / 2
  let red_tetrahedron_volume := cube_volume - 4 * blue_tetrahedron_volume
  red_tetrahedron_volume = 512 - (4 * 256 / 3) := by
  sorry

#eval 512 - (4 * 256 / 3)  -- To verify the numerical result

end red_tetrahedron_volume_l2581_258170


namespace cyclic_sum_inequality_l2581_258164

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) (h5 : n ≥ 2) :
  a / (b + c)^(1/n : ℝ) + b / (c + a)^(1/n : ℝ) + c / (a + b)^(1/n : ℝ) ≥ 3 / 2^(1/n : ℝ) :=
sorry

end cyclic_sum_inequality_l2581_258164


namespace marian_cookies_l2581_258169

theorem marian_cookies (cookies_per_tray : ℕ) (num_trays : ℕ) (h1 : cookies_per_tray = 12) (h2 : num_trays = 23) :
  cookies_per_tray * num_trays = 276 := by
  sorry

end marian_cookies_l2581_258169


namespace chairs_per_row_l2581_258135

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end chairs_per_row_l2581_258135


namespace smallest_a_is_eight_l2581_258195

/-- A function that represents the expression x^4 + a^2 + x^2 --/
def f (a x : ℤ) : ℤ := x^4 + a^2 + x^2

/-- A predicate that checks if a number is composite --/
def is_composite (n : ℤ) : Prop := ∃ (p q : ℤ), p ≠ 1 ∧ q ≠ 1 ∧ n = p * q

theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬is_composite (f a x)) :=
sorry

end smallest_a_is_eight_l2581_258195


namespace coefficients_of_specific_quadratic_l2581_258158

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the tuple (a, b, c) of its coefficients -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation x^2 - x + 3 = 0 are (1, -1, 3) -/
theorem coefficients_of_specific_quadratic :
  quadratic_coefficients 1 (-1) 3 = (1, -1, 3) := by
  sorry

end coefficients_of_specific_quadratic_l2581_258158


namespace inequality_solution_set_l2581_258182

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 5 / (x + 4) ≤ 1) ↔ (x ≤ -4 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l2581_258182


namespace optimal_price_for_target_profit_l2581_258114

-- Define the problem parameters
def cost : ℝ := 30
def initialPrice : ℝ := 40
def initialSales : ℝ := 600
def priceIncreaseSalesDrop : ℝ := 20
def priceDecreaseSalesIncrease : ℝ := 200
def stock : ℝ := 1210
def targetProfit : ℝ := 8400

-- Define the sales function based on price change
def sales (priceChange : ℝ) : ℝ :=
  initialSales + priceDecreaseSalesIncrease * priceChange

-- Define the profit function
def profit (priceChange : ℝ) : ℝ :=
  (initialPrice - priceChange - cost) * (sales priceChange)

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (priceChange : ℝ), profit priceChange = targetProfit ∧ 
  initialPrice - priceChange = 37 ∧
  sales priceChange ≤ stock :=
sorry

end optimal_price_for_target_profit_l2581_258114


namespace solution_set_l2581_258142

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 →
  x ≥ 18 := by
sorry

end solution_set_l2581_258142


namespace lines_do_not_intersect_l2581_258163

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 3), direction := (5, -8) }

/-- The second line -/
def line2 (k : ℝ) : Line2D :=
  { point := (-1, 4), direction := (2, k) }

/-- Theorem: The lines do not intersect if and only if k = -16/5 -/
theorem lines_do_not_intersect (k : ℝ) : 
  are_parallel line1 (line2 k) ↔ k = -16/5 := by
  sorry

end lines_do_not_intersect_l2581_258163


namespace max_quotient_value_l2581_258179

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
sorry

end max_quotient_value_l2581_258179


namespace factor_implies_c_value_l2581_258101

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (4*x + 14) ∣ (6*x^3 + 19*x^2 + c*x + 70)) → c = 13 :=
by sorry

end factor_implies_c_value_l2581_258101


namespace bottles_bought_l2581_258152

theorem bottles_bought (initial : ℕ) (drunk : ℕ) (final : ℕ) : 
  initial = 14 → drunk = 8 → final = 51 → final - (initial - drunk) = 45 := by
  sorry

end bottles_bought_l2581_258152


namespace train_passengers_with_hats_l2581_258138

theorem train_passengers_with_hats 
  (total_adults : ℕ) 
  (men_percentage : ℚ) 
  (men_with_hats_percentage : ℚ) 
  (women_with_hats_percentage : ℚ) 
  (h1 : total_adults = 3600) 
  (h2 : men_percentage = 40 / 100) 
  (h3 : men_with_hats_percentage = 15 / 100) 
  (h4 : women_with_hats_percentage = 25 / 100) : 
  ℕ := by
  sorry

#check train_passengers_with_hats

end train_passengers_with_hats_l2581_258138


namespace car_travel_time_l2581_258150

theorem car_travel_time (speed_A speed_B : ℝ) (time_A : ℝ) (ratio : ℝ) 
  (h1 : speed_A = 50)
  (h2 : speed_B = 25)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : speed_A > 0)
  (h6 : speed_B > 0)
  (h7 : time_A > 0)
  (h8 : ratio > 0) :
  (speed_A * time_A) / (speed_B * ((speed_A * time_A) / (ratio * speed_B))) = 4 := by
  sorry

#check car_travel_time

end car_travel_time_l2581_258150


namespace total_bugs_equals_63_l2581_258196

/-- The number of bugs eaten by the gecko -/
def gecko_bugs : ℕ := 12

/-- The number of bugs eaten by the lizard -/
def lizard_bugs : ℕ := gecko_bugs / 2

/-- The number of bugs eaten by the frog -/
def frog_bugs : ℕ := lizard_bugs * 3

/-- The number of bugs eaten by the toad -/
def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

/-- The total number of bugs eaten by all animals -/
def total_bugs : ℕ := gecko_bugs + lizard_bugs + frog_bugs + toad_bugs

theorem total_bugs_equals_63 : total_bugs = 63 := by
  sorry

end total_bugs_equals_63_l2581_258196


namespace campaign_fund_distribution_l2581_258123

theorem campaign_fund_distribution (total : ℝ) (family_percent : ℝ) (own_savings : ℝ) :
  total = 10000 →
  family_percent = 0.3 →
  own_savings = 4200 →
  ∃ (friends_contribution : ℝ),
    friends_contribution = total * 0.4 ∧
    total = friends_contribution + (family_percent * (total - friends_contribution)) + own_savings :=
by sorry

end campaign_fund_distribution_l2581_258123


namespace cos_value_given_sin_l2581_258185

theorem cos_value_given_sin (θ : ℝ) (h : Real.sin (θ - π/6) = Real.sqrt 3 / 3) :
  Real.cos (π/3 - 2*θ) = 1/3 := by
  sorry

end cos_value_given_sin_l2581_258185


namespace geese_to_ducks_ratio_l2581_258176

theorem geese_to_ducks_ratio (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) (initial_geese : ℕ) :
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  initial_geese - leaving_geese = initial_ducks + arriving_ducks + 1 →
  (initial_geese : ℚ) / initial_ducks = 8 / 5 := by
sorry

end geese_to_ducks_ratio_l2581_258176


namespace circle_radius_c_value_l2581_258133

theorem circle_radius_c_value (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 2*y + c = 0 ↔ (x+5)^2 + (y+1)^2 = 25) → 
  c = 51 := by
sorry

end circle_radius_c_value_l2581_258133


namespace sufficient_not_necessary_condition_l2581_258159

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end sufficient_not_necessary_condition_l2581_258159


namespace inequality_range_l2581_258162

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) → 
  (m ≥ 4 ∨ m ≤ -1) := by
  sorry

end inequality_range_l2581_258162


namespace grace_reading_time_l2581_258105

/-- Represents Grace's reading speed in pages per hour -/
def reading_speed (pages : ℕ) (hours : ℕ) : ℚ :=
  pages / hours

/-- Calculates the time needed to read a book given the number of pages and reading speed -/
def time_to_read (pages : ℕ) (speed : ℚ) : ℚ :=
  pages / speed

/-- Theorem stating that it takes 25 hours to read a 250-page book given Grace's reading rate -/
theorem grace_reading_time :
  let initial_pages : ℕ := 200
  let initial_hours : ℕ := 20
  let target_pages : ℕ := 250
  let speed := reading_speed initial_pages initial_hours
  time_to_read target_pages speed = 25 := by
  sorry


end grace_reading_time_l2581_258105


namespace thread_length_problem_l2581_258165

theorem thread_length_problem (current_length : ℝ) : 
  current_length + (3/4 * current_length) = 21 → current_length = 12 := by
  sorry

end thread_length_problem_l2581_258165


namespace sum_of_x_and_y_is_three_l2581_258119

theorem sum_of_x_and_y_is_three (x y : ℝ) (h : x^2 + y^2 = 14*x - 8*y - 74) : x + y = 3 := by
  sorry

end sum_of_x_and_y_is_three_l2581_258119


namespace angle_measures_in_special_cyclic_quadrilateral_l2581_258184

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (A B C D : ℝ)
  (cyclic : A + C = 180 ∧ B + D = 180)

-- Define the diagonal property
def DiagonalProperty (q : CyclicQuadrilateral) :=
  ∃ (θ : ℝ), (q.A = 6 * θ ∨ q.C = 6 * θ) ∧ (q.B = 6 * θ ∨ q.D = 6 * θ)

-- Define the set of possible angle measures
def PossibleAngleMeasures : Set ℝ := {45, 135, 225/2, 135/2}

-- Theorem statement
theorem angle_measures_in_special_cyclic_quadrilateral
  (q : CyclicQuadrilateral) (h : DiagonalProperty q) :
  q.A ∈ PossibleAngleMeasures :=
sorry

end angle_measures_in_special_cyclic_quadrilateral_l2581_258184


namespace missing_number_equation_l2581_258130

theorem missing_number_equation (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 ↔ x = 4 := by
  sorry

end missing_number_equation_l2581_258130


namespace employee_count_l2581_258143

theorem employee_count (avg_salary : ℕ) (salary_increase : ℕ) (manager_salary : ℕ)
  (h1 : avg_salary = 1300)
  (h2 : salary_increase = 100)
  (h3 : manager_salary = 3400) :
  ∃ n : ℕ, n * avg_salary + manager_salary = (n + 1) * (avg_salary + salary_increase) ∧ n = 20 :=
by sorry

end employee_count_l2581_258143


namespace sin_plus_cos_equivalence_l2581_258120

theorem sin_plus_cos_equivalence (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end sin_plus_cos_equivalence_l2581_258120


namespace no_prime_solution_l2581_258147

/-- Converts a number from base p to decimal --/
def to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p^i) 0

/-- The equation that p must satisfy --/
def equation (p : Nat) : Prop :=
  to_decimal [9, 0, 0, 1] p + to_decimal [7, 0, 3] p + 
  to_decimal [5, 1, 1] p + to_decimal [6, 2, 1] p + 
  to_decimal [7] p = 
  to_decimal [3, 4, 1] p + to_decimal [4, 7, 2] p + 
  to_decimal [1, 6, 3] p

theorem no_prime_solution : ¬∃ p : Nat, Nat.Prime p ∧ equation p := by
  sorry

end no_prime_solution_l2581_258147


namespace min_sqrt_difference_l2581_258117

theorem min_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m n : ℕ), 
    0 < m ∧ 0 < n ∧ m ≤ n ∧
    (∀ (a b : ℕ), 0 < a → 0 < b → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt m - Real.sqrt n ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    m = (p - 1) / 2 ∧ n = (p + 1) / 2 :=
sorry

end min_sqrt_difference_l2581_258117


namespace four_person_four_office_assignment_l2581_258113

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_person_four_office_assignment :
  number_of_assignments 4 = 24 := by
  sorry

end four_person_four_office_assignment_l2581_258113


namespace ethereum_investment_l2581_258181

theorem ethereum_investment (I : ℝ) : 
  I > 0 →
  (I * 1.25 * 1.5 = 750) →
  I = 400 := by
sorry

end ethereum_investment_l2581_258181


namespace fencing_cost_theorem_l2581_258111

/-- The total cost of fencing a rectangular plot -/
def fencing_cost (length breadth cost_per_metre : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_metre

/-- Theorem: The total cost of fencing a rectangular plot with given dimensions -/
theorem fencing_cost_theorem (length breadth cost_per_metre : ℝ) 
  (h1 : length = 60)
  (h2 : breadth = length - 20)
  (h3 : cost_per_metre = 26.50) :
  fencing_cost length breadth cost_per_metre = 5300 := by
  sorry

#eval fencing_cost 60 40 26.50

end fencing_cost_theorem_l2581_258111


namespace inscribed_triangle_property_l2581_258112

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  xy = 26 ∧ yz = 28 ∧ xz = 27

-- Define the inscribed triangle GHI
def InscribedTriangle (X Y Z G H I : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ,
    0 < t₁ ∧ t₁ < 1 ∧
    0 < t₂ ∧ t₂ < 1 ∧
    0 < t₃ ∧ t₃ < 1 ∧
    G = (t₁ * Y.1 + (1 - t₁) * Z.1, t₁ * Y.2 + (1 - t₁) * Z.2) ∧
    H = (t₂ * X.1 + (1 - t₂) * Z.1, t₂ * X.2 + (1 - t₂) * Z.2) ∧
    I = (t₃ * X.1 + (1 - t₃) * Y.1, t₃ * X.2 + (1 - t₃) * Y.2)

-- Define the equality of arcs
def ArcEqual (X Y Z G H I : ℝ × ℝ) : Prop :=
  let yi := Real.sqrt ((Y.1 - I.1)^2 + (Y.2 - I.2)^2)
  let gz := Real.sqrt ((G.1 - Z.1)^2 + (G.2 - Z.2)^2)
  let xi := Real.sqrt ((X.1 - I.1)^2 + (X.2 - I.2)^2)
  let hz := Real.sqrt ((H.1 - Z.1)^2 + (H.2 - Z.2)^2)
  let xh := Real.sqrt ((X.1 - H.1)^2 + (X.2 - H.2)^2)
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  yi = gz ∧ xi = hz ∧ xh = gy

theorem inscribed_triangle_property
  (X Y Z G H I : ℝ × ℝ)
  (h₁ : Triangle X Y Z)
  (h₂ : InscribedTriangle X Y Z G H I)
  (h₃ : ArcEqual X Y Z G H I) :
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  gy = 27 / 2 := by
  sorry

end inscribed_triangle_property_l2581_258112


namespace worker_count_l2581_258132

/-- Represents the number of workers in the factory -/
def num_workers : ℕ := sorry

/-- The initial average monthly salary of workers and supervisor -/
def initial_average_salary : ℚ := 430

/-- The initial supervisor's monthly salary -/
def initial_supervisor_salary : ℚ := 870

/-- The new average monthly salary after supervisor change -/
def new_average_salary : ℚ := 410

/-- The new supervisor's monthly salary -/
def new_supervisor_salary : ℚ := 690

/-- The total number of people (workers + new supervisor) after the change -/
def total_people : ℕ := 9

theorem worker_count :
  (num_workers + 1) * initial_average_salary - initial_supervisor_salary =
  total_people * new_average_salary - new_supervisor_salary ∧
  num_workers = 8 := by sorry

end worker_count_l2581_258132


namespace polynomial_gp_roots_condition_l2581_258156

/-- A polynomial with coefficients a, b, and c -/
def polynomial (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Condition for three distinct real roots in geometric progression -/
def has_three_distinct_real_roots_in_gp (a b c : ℝ) : Prop :=
  ∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial a b c x = 0 ∧
    polynomial a b c y = 0 ∧
    polynomial a b c z = 0 ∧
    ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r

/-- Theorem stating the conditions on coefficients a, b, and c -/
theorem polynomial_gp_roots_condition (a b c : ℝ) :
  has_three_distinct_real_roots_in_gp a b c ↔ 
    a^3 * c = b^3 ∧ -a^2 < b ∧ b < a^2 / 3 :=
sorry

end polynomial_gp_roots_condition_l2581_258156


namespace workshop_workers_l2581_258122

theorem workshop_workers (total_avg : ℝ) (tech_count : ℕ) (tech_avg : ℝ) (rest_avg : ℝ)
  (h1 : total_avg = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg = 16000)
  (h4 : rest_avg = 6000) :
  ∃ (total_workers : ℕ),
    (total_workers : ℝ) * total_avg = 
      (tech_count : ℝ) * tech_avg + ((total_workers - tech_count) : ℝ) * rest_avg ∧
    total_workers = 35 :=
by sorry

end workshop_workers_l2581_258122


namespace bug_return_probability_l2581_258168

-- Define the tetrahedron structure
structure Tetrahedron where
  vertices : Fin 4 → Point
  edge_length : ℝ
  is_regular : Bool

-- Define the bug's movement
def bug_move (t : Tetrahedron) (current_vertex : Fin 4) : Fin 4 := sorry

-- Define the probability of returning to the starting vertex after n steps
def return_probability (t : Tetrahedron) (n : ℕ) : ℚ := sorry

-- Main theorem
theorem bug_return_probability (t : Tetrahedron) :
  t.is_regular = true →
  t.edge_length = 1 →
  return_probability t 9 = 4920 / 19683 := by sorry

end bug_return_probability_l2581_258168


namespace completing_square_equivalence_l2581_258198

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end completing_square_equivalence_l2581_258198


namespace pure_imaginary_product_l2581_258186

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal 1 - Complex.I) * (Complex.ofReal a + Complex.I) ∈ {z : ℂ | z.re = 0 ∧ z.im ≠ 0} → 
  a = -1 := by
sorry

end pure_imaginary_product_l2581_258186


namespace roi_is_25_percent_l2581_258134

/-- Calculates the return on investment (ROI) percentage for an investor given the dividend rate, face value, and purchase price of shares. -/
def calculate_roi (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the ROI is 25%. -/
theorem roi_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000  -- 12.5%
  let face_value : ℚ := 50
  let purchase_price : ℚ := 25
  calculate_roi dividend_rate face_value purchase_price = 25 := by
  sorry

#eval calculate_roi (125/1000) 50 25  -- This should evaluate to 25

end roi_is_25_percent_l2581_258134


namespace total_chairs_bought_l2581_258125

def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6

theorem total_chairs_bought : living_room_chairs + kitchen_chairs = 9 := by
  sorry

end total_chairs_bought_l2581_258125


namespace equation_solution_l2581_258103

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by sorry

end equation_solution_l2581_258103


namespace simplify_expression_l2581_258136

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  (15 / 8) * Real.sqrt (2 + 10 / 27) / Real.sqrt (25 / (12 * a^3)) = a * Real.sqrt 3 := by
  sorry

end simplify_expression_l2581_258136


namespace picnic_gender_difference_l2581_258173

theorem picnic_gender_difference (total : ℕ) (men : ℕ) (adult_child_diff : ℕ) 
  (h_total : total = 240)
  (h_men : men = 90)
  (h_adult_child : adult_child_diff = 40) : 
  ∃ (women children : ℕ), 
    men + women + children = total ∧ 
    men + women = children + adult_child_diff ∧ 
    men - women = 40 := by
sorry

end picnic_gender_difference_l2581_258173


namespace max_log_value_l2581_258137

theorem max_log_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 4 * a - 2 * b + 25 * c = 0) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → 4 * x - 2 * y + 25 * z = 0 → 
    Real.log x + Real.log z - 2 * Real.log y ≤ Real.log a + Real.log c - 2 * Real.log b) ∧
  Real.log a + Real.log c - 2 * Real.log b = -2 * Real.log 10 := by
  sorry

end max_log_value_l2581_258137


namespace nested_sqrt_eighteen_l2581_258167

theorem nested_sqrt_eighteen (y : ℝ) : y = Real.sqrt (18 + y) → y = (1 + Real.sqrt 73) / 2 := by
  sorry

end nested_sqrt_eighteen_l2581_258167


namespace factorization_proof_l2581_258109

theorem factorization_proof (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_proof_l2581_258109


namespace lcm_gcd_product_equals_product_l2581_258144

theorem lcm_gcd_product_equals_product (a b : ℕ) (ha : a = 12) (hb : b = 18) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end lcm_gcd_product_equals_product_l2581_258144


namespace solution_set_of_inequality_l2581_258110

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = f x

theorem solution_set_of_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_deriv : ∀ x > 0, x * f' x > -2 * f x)
  (g : ℝ → ℝ) (h_g : ∀ x, g x = x^2 * f x) :
  {x : ℝ | g x < g (1 - x)} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1/2)} :=
sorry

end solution_set_of_inequality_l2581_258110


namespace square_root_of_nine_three_is_square_root_of_nine_l2581_258189

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end square_root_of_nine_three_is_square_root_of_nine_l2581_258189


namespace closest_integer_to_sqrt3_plus_1_l2581_258126

theorem closest_integer_to_sqrt3_plus_1 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (Real.sqrt 3 + 1)| ≤ |m - (Real.sqrt 3 + 1)| ∧ n = 3 := by
  sorry

end closest_integer_to_sqrt3_plus_1_l2581_258126


namespace p_necessary_not_sufficient_l2581_258129

-- Define propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := 4 < x ∧ x < 10

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end p_necessary_not_sufficient_l2581_258129


namespace black_chicken_daytime_theorem_l2581_258149

/-- The number of spots in the daytime program -/
def daytime_spots : ℕ := 2

/-- The number of spots in the evening program -/
def evening_spots : ℕ := 3

/-- The total number of spots available -/
def total_spots : ℕ := daytime_spots + evening_spots

/-- The number of black chickens applying -/
def black_chickens : ℕ := 3

/-- The number of white chickens applying -/
def white_chickens : ℕ := 1

/-- The total number of chickens applying -/
def total_chickens : ℕ := black_chickens + white_chickens

/-- The probability of a chicken choosing the daytime program when both are available -/
def daytime_probability : ℚ := 1/2

/-- The probability that at least one black chicken is admitted to the daytime program -/
def black_chicken_daytime_probability : ℚ := 63/64

theorem black_chicken_daytime_theorem :
  (total_spots = daytime_spots + evening_spots) →
  (total_chickens = black_chickens + white_chickens) →
  (total_chickens ≤ total_spots) →
  (daytime_probability = 1/2) →
  black_chicken_daytime_probability = 63/64 := by
  sorry

end black_chicken_daytime_theorem_l2581_258149


namespace modulo_17_residue_l2581_258187

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := by
  sorry

end modulo_17_residue_l2581_258187


namespace fractional_equation_positive_root_l2581_258191

theorem fractional_equation_positive_root (x m : ℝ) : 
  (2 / (x - 2) - (2 * x - m) / (2 - x) = 3) → 
  (x > 0) →
  (m = 6) := by
sorry

end fractional_equation_positive_root_l2581_258191


namespace min_additional_weeks_equals_additional_wins_needed_l2581_258151

/-- Represents the number of dollars Bob has won so far -/
def initial_winnings : ℕ := 200

/-- Represents the number of additional wins needed to afford the puppy -/
def additional_wins_needed : ℕ := 8

/-- Represents the prize money for each win in dollars -/
def prize_money : ℕ := 100

/-- Proves that the minimum number of additional weeks Bob must win first place is equal to the number of additional wins needed -/
theorem min_additional_weeks_equals_additional_wins_needed :
  additional_wins_needed = additional_wins_needed := by sorry

end min_additional_weeks_equals_additional_wins_needed_l2581_258151


namespace partnership_investment_l2581_258102

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a_invest b_invest total_profit a_profit : ℚ) (h1 : a_invest = 6300)
    (h2 : b_invest = 4200) (h3 : total_profit = 14200) (h4 : a_profit = 4260) :
    ∃ c_invest : ℚ, c_invest = 10500 ∧ 
    a_profit / a_invest = total_profit / (a_invest + b_invest + c_invest) :=
  sorry

end partnership_investment_l2581_258102


namespace evaluate_expression_l2581_258192

theorem evaluate_expression : (2023 - 1984)^2 / 144 = 10 := by
  sorry

end evaluate_expression_l2581_258192


namespace sally_quarters_problem_l2581_258157

theorem sally_quarters_problem (initial_quarters : ℕ) 
  (first_purchase : ℕ) (second_purchase : ℕ) :
  initial_quarters = 760 →
  first_purchase = 418 →
  second_purchase = 215 →
  initial_quarters - first_purchase - second_purchase = 127 :=
by sorry

end sally_quarters_problem_l2581_258157


namespace inequality_proof_l2581_258171

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
  1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) ≤ 2 ∧
  (1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
   1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) = 2 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end inequality_proof_l2581_258171


namespace bus_driver_compensation_l2581_258180

-- Define the constants
def regular_rate : ℝ := 12
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℝ := 63.62

-- Define the function to calculate total compensation
def total_compensation : ℝ :=
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings

-- Theorem statement
theorem bus_driver_compensation :
  total_compensation = 976.02 := by sorry

end bus_driver_compensation_l2581_258180


namespace solve_average_problem_l2581_258104

def average_problem (total_average : ℚ) (pair1_average : ℚ) (pair2_average : ℚ) (pair3_average : ℚ) : Prop :=
  ∃ (n : ℕ) (sum : ℚ),
    n > 0 ∧
    sum / n = total_average ∧
    n = 6 ∧
    sum = 2 * pair1_average + 2 * pair2_average + 2 * pair3_average

theorem solve_average_problem :
  average_problem (395/100) (38/10) (385/100) (4200000000000001/1000000000000000) :=
sorry

end solve_average_problem_l2581_258104


namespace painted_cube_theorem_l2581_258194

/-- Represents a cube that has been painted on all sides and then cut into smaller cubes -/
structure PaintedCube where
  edge_length : ℕ
  small_cube_edge : ℕ

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : PaintedCube) (num_faces : ℕ) : ℕ := sorry

theorem painted_cube_theorem (c : PaintedCube) 
  (h1 : c.edge_length = 5) 
  (h2 : c.small_cube_edge = 1) : 
  (count_painted_faces c 3 = 8) ∧ 
  (count_painted_faces c 2 = 36) ∧ 
  (count_painted_faces c 1 = 54) := by sorry

end painted_cube_theorem_l2581_258194


namespace real_roots_quadratic_equation_l2581_258141

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ k ≤ 1 := by
sorry

end real_roots_quadratic_equation_l2581_258141


namespace largest_class_size_l2581_258108

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 140, the number of students in the largest class is 32. -/
theorem largest_class_size (num_classes : Nat) (student_difference : Nat) (total_students : Nat)
    (h1 : num_classes = 5)
    (h2 : student_difference = 2)
    (h3 : total_students = 140) :
    ∃ (x : Nat), x = 32 ∧ 
    (x + (x - student_difference) + (x - 2*student_difference) + 
     (x - 3*student_difference) + (x - 4*student_difference) = total_students) :=
  by sorry

end largest_class_size_l2581_258108


namespace m_range_l2581_258153

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : -2 < m ∧ m < 0 := by
  sorry

end m_range_l2581_258153


namespace white_copy_cost_is_five_cents_l2581_258193

/- Define the problem parameters -/
def total_copies : ℕ := 400
def colored_copies : ℕ := 50
def colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
def total_bill : ℚ := 225 / 10    -- $22.50

/- Define the cost of a white copy -/
def white_copy_cost : ℚ := (total_bill - colored_copies * colored_cost) / (total_copies - colored_copies)

/- Theorem statement -/
theorem white_copy_cost_is_five_cents : white_copy_cost = 5 / 100 := by
  sorry

end white_copy_cost_is_five_cents_l2581_258193


namespace roof_dimension_difference_l2581_258154

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 := by
sorry

end roof_dimension_difference_l2581_258154


namespace lollipop_difference_l2581_258106

theorem lollipop_difference (henry alison diane : ℕ) : 
  henry > alison →
  alison = 60 →
  alison = diane / 2 →
  henry + alison + diane = 45 * 6 →
  henry - alison = 30 := by
sorry

end lollipop_difference_l2581_258106


namespace aartis_work_time_l2581_258160

/-- Given that Aarti completes three times a piece of work in 27 days,
    prove that she can complete one piece of work in 9 days. -/
theorem aartis_work_time :
  ∀ (work_time : ℕ),
  (3 * work_time = 27) →
  (work_time = 9) :=
by sorry

end aartis_work_time_l2581_258160


namespace quadratic_inequality_l2581_258145

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that f(3^x) ≤ f(2^x) for all real x, 
    given that f is monotonically increasing on (-∞, 1] -/
theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f a b c x ≤ f a b c y) →
  ∀ x : ℝ, f a b c (3^x) ≤ f a b c (2^x) := by
  sorry

end quadratic_inequality_l2581_258145


namespace crayon_selection_l2581_258155

theorem crayon_selection (n k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  Nat.choose n k = 3003 := by
  sorry

end crayon_selection_l2581_258155


namespace fraction_not_simplifiable_l2581_258107

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_not_simplifiable_l2581_258107


namespace division_problem_l2581_258175

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 100 →
  divisor = 11 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end division_problem_l2581_258175


namespace integral_equals_antiderivative_l2581_258178

open Real

noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 13*x - 8) / (x*(x-2)^3)

noncomputable def F (x : ℝ) : ℝ := log (abs x) - 1 / (2*(x-2)^2)

theorem integral_equals_antiderivative (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) :
  deriv F x = f x :=
by sorry

end integral_equals_antiderivative_l2581_258178


namespace point_not_on_graph_l2581_258174

-- Define the function
def f (x : ℝ) : ℝ := 1 - 2 * x

-- Theorem statement
theorem point_not_on_graph :
  f (-1) ≠ 0 ∧ 
  f 1 = -1 ∧ 
  f 0 = 1 ∧ 
  f (-1/2) = 2 :=
by sorry

end point_not_on_graph_l2581_258174


namespace correct_years_until_twice_as_old_l2581_258139

/-- Represents the current ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the number of years until Richard is twice as old as Scott -/
def yearsUntilTwiceAsOld (ages : BrothersAges) : ℕ :=
  sorry

theorem correct_years_until_twice_as_old : 
  ∀ (ages : BrothersAges),
    ages.david = 14 →
    ages.richard = ages.david + 6 →
    ages.scott = ages.david - 8 →
    yearsUntilTwiceAsOld ages = 8 :=
  sorry

end correct_years_until_twice_as_old_l2581_258139


namespace sum_of_squares_and_products_l2581_258131

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x^2 + y^2 + z^2 = 75 → 
  x*y + y*z + z*x = 32 → 
  x + y + z = Real.sqrt 139 := by
sorry

end sum_of_squares_and_products_l2581_258131


namespace min_value_expression_l2581_258190

theorem min_value_expression (x : ℝ) : (x^2 + 13) / Real.sqrt (x^2 + 7) ≥ 2 * Real.sqrt 6 := by
  sorry

end min_value_expression_l2581_258190


namespace hallie_earnings_l2581_258177

/-- Calculates the total earnings for a waitress over three days given her hourly wage, hours worked, and tips for each day. -/
def total_earnings (hourly_wage : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) (tips_day1 tips_day2 tips_day3 : ℝ) : ℝ :=
  (hourly_wage * hours_day1 + tips_day1) +
  (hourly_wage * hours_day2 + tips_day2) +
  (hourly_wage * hours_day3 + tips_day3)

/-- Theorem stating that Hallie's total earnings over three days equal $240 given her work schedule and tips. -/
theorem hallie_earnings :
  total_earnings 10 7 5 7 18 12 20 = 240 :=
by
  sorry

end hallie_earnings_l2581_258177


namespace yaras_earnings_l2581_258183

/-- Yara's work and earnings over two weeks -/
theorem yaras_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) 
  (h1 : hours_week1 = 12)
  (h2 : hours_week2 = 18)
  (h3 : extra_earnings = 36)
  (h4 : ∃ (wage : ℚ), wage * (hours_week2 - hours_week1) = extra_earnings) :
  ∃ (total_earnings : ℚ), total_earnings = hours_week1 * (extra_earnings / (hours_week2 - hours_week1)) + 
                           hours_week2 * (extra_earnings / (hours_week2 - hours_week1)) ∧
                           total_earnings = 180 := by
  sorry


end yaras_earnings_l2581_258183


namespace charles_housesitting_rate_l2581_258146

/-- Represents the earnings of Charles from housesitting and dog walking -/
structure Earnings where
  housesitting_rate : ℝ
  dog_walking_rate : ℝ
  housesitting_hours : ℕ
  dogs_walked : ℕ
  total_earnings : ℝ

/-- Theorem stating that given the conditions, Charles earns $15 per hour for housesitting -/
theorem charles_housesitting_rate (e : Earnings) 
  (h1 : e.dog_walking_rate = 22)
  (h2 : e.housesitting_hours = 10)
  (h3 : e.dogs_walked = 3)
  (h4 : e.total_earnings = 216)
  (h5 : e.housesitting_rate * e.housesitting_hours + e.dog_walking_rate * e.dogs_walked = e.total_earnings) :
  e.housesitting_rate = 15 := by
  sorry

end charles_housesitting_rate_l2581_258146


namespace perfect_square_product_iff_factors_l2581_258161

theorem perfect_square_product_iff_factors (x y z : ℕ+) :
  ∃ (n : ℕ), (x * y + 1) * (y * z + 1) * (z * x + 1) = n ^ 2 ↔
  (∃ (a b c : ℕ), (x * y + 1 = a ^ 2) ∧ (y * z + 1 = b ^ 2) ∧ (z * x + 1 = c ^ 2)) :=
by sorry

end perfect_square_product_iff_factors_l2581_258161


namespace intersection_complement_equality_l2581_258172

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_equality :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_complement_equality_l2581_258172


namespace distinct_determinants_count_l2581_258197

-- Define a type for third-order determinants
def ThirdOrderDeterminant := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the number of distinct determinants
def distinctDeterminants (n : ℕ) : ℕ :=
  if n = 9 then Nat.factorial 9 / 36 else 0

theorem distinct_determinants_count :
  distinctDeterminants 9 = 10080 := by
  sorry

#eval distinctDeterminants 9

end distinct_determinants_count_l2581_258197


namespace system_of_equations_l2581_258121

theorem system_of_equations (x y k : ℝ) :
  x - y = k + 2 →
  x + 3 * y = k →
  x + y = 2 →
  k = 1 := by sorry

end system_of_equations_l2581_258121


namespace probability_in_range_l2581_258199

/-- 
Given a random variable ξ with probability distribution:
P(ξ=k) = 1/(2^(k-1)) for k = 2, 3, ..., n
P(ξ=1) = a
Prove that P(2 < ξ ≤ 5) = 7/16
-/
theorem probability_in_range (n : ℕ) (a : ℝ) (ξ : ℕ → ℝ) 
  (h1 : ∀ k ∈ Finset.range (n - 1) \ {0}, ξ (k + 2) = (1 : ℝ) / 2^k)
  (h2 : ξ 1 = a) :
  (ξ 3 + ξ 4 + ξ 5) = 7/16 := by
sorry

end probability_in_range_l2581_258199


namespace proposition_truth_values_l2581_258188

theorem proposition_truth_values :
  let p := ∀ x y : ℝ, x > y → -x < -y
  let q := ∀ x y : ℝ, x > y → x^2 > y^2
  (p ∨ q) ∧ (p ∧ (¬q)) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∨ q) := by
  sorry

end proposition_truth_values_l2581_258188


namespace g_evaluation_l2581_258116

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end g_evaluation_l2581_258116


namespace four_circle_plus_two_l2581_258100

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem four_circle_plus_two : circle_plus 4 2 = 26 := by
  sorry

end four_circle_plus_two_l2581_258100


namespace units_digit_of_large_power_l2581_258124

theorem units_digit_of_large_power (n : ℕ) : n > 0 → (7^(8^5) : ℕ) % 10 = 1 := by
  sorry

end units_digit_of_large_power_l2581_258124


namespace jakes_weight_l2581_258127

theorem jakes_weight (jake kendra : ℝ) 
  (h1 : jake - 8 = 2 * kendra) 
  (h2 : jake + kendra = 293) : 
  jake = 198 := by
sorry

end jakes_weight_l2581_258127


namespace square_diff_sum_l2581_258115

theorem square_diff_sum : 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 200 := by
  sorry

end square_diff_sum_l2581_258115


namespace yellow_balls_count_l2581_258140

/-- Represents a bag containing red and yellow balls -/
structure BallBag where
  redBalls : ℕ
  yellowBalls : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def redProbability (bag : BallBag) : ℚ :=
  bag.redBalls / (bag.redBalls + bag.yellowBalls)

/-- Theorem: Given the conditions, the number of yellow balls is 25 -/
theorem yellow_balls_count (bag : BallBag) 
  (h1 : bag.redBalls = 10)
  (h2 : redProbability bag = 2/7) :
  bag.yellowBalls = 25 := by
  sorry

end yellow_balls_count_l2581_258140


namespace max_abs_sum_on_circle_l2581_258166

theorem max_abs_sum_on_circle : ∀ x y : ℝ, x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 := by
  sorry

end max_abs_sum_on_circle_l2581_258166


namespace g_positive_f_local_min_iff_l2581_258148

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x^3 - (1/2) * x^2

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x^2 - x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f' a x) / x

-- Theorem 1: When a > 0, g(a) > 0
theorem g_positive (a : ℝ) (h : a > 0) : g a a > 0 := by sorry

-- Theorem 2: f(x) has a local minimum if and only if a ∈ (0, +∞)
theorem f_local_min_iff (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) x) ↔ a > 0 := by sorry

end g_positive_f_local_min_iff_l2581_258148
