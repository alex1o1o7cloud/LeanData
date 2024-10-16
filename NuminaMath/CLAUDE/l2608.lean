import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2608_260815

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -3 < x < 2}, prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) :
  (a < 0) ∧ 
  (a + b + c > 0) ∧
  (∀ x, c*x^2 + b*x + a < 0 ↔ -1/3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2608_260815


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l2608_260831

theorem divisibility_of_sum_of_powers (n : ℕ) : 13 ∣ (3^1974 + 2^1974) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l2608_260831


namespace NUMINAMATH_CALUDE_quadratic_roots_l2608_260881

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x^2 + x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2608_260881


namespace NUMINAMATH_CALUDE_set_equivalence_l2608_260873

theorem set_equivalence (U A B : Set α) :
  (A ∩ B = A) ↔ (A ⊆ U ∧ B ⊆ U ∧ (Uᶜ ∩ B)ᶜ ⊆ (Uᶜ ∩ A)ᶜ) := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l2608_260873


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2608_260859

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem sufficient_but_not_necessary
  (a₁ q : ℝ) :
  (∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ↔
  (a₁ > 0 ∧ q > 1 ∨
   ∃ a₁' q', (a₁' ≤ 0 ∨ q' ≤ 1) ∧
   ∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2608_260859


namespace NUMINAMATH_CALUDE_k_range_l2608_260835

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ x, f x ≤ k * x) → 1 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_l2608_260835


namespace NUMINAMATH_CALUDE_max_notebooks_lucy_can_buy_l2608_260857

def lucy_money : ℕ := 2145
def notebook_cost : ℕ := 230

theorem max_notebooks_lucy_can_buy :
  (lucy_money / notebook_cost : ℕ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_lucy_can_buy_l2608_260857


namespace NUMINAMATH_CALUDE_three_competition_participation_l2608_260862

theorem three_competition_participation 
  (total : ℕ) 
  (chinese : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (chinese_math : ℕ) 
  (math_english : ℕ) 
  (chinese_english : ℕ) 
  (none : ℕ) 
  (h1 : total = 100)
  (h2 : chinese = 39)
  (h3 : math = 49)
  (h4 : english = 41)
  (h5 : chinese_math = 14)
  (h6 : math_english = 13)
  (h7 : chinese_english = 9)
  (h8 : none = 1) :
  ∃ (all_three : ℕ), 
    all_three = 6 ∧ 
    total = chinese + math + english - chinese_math - math_english - chinese_english + all_three + none :=
by sorry

end NUMINAMATH_CALUDE_three_competition_participation_l2608_260862


namespace NUMINAMATH_CALUDE_symmetric_about_x_axis_l2608_260812

-- Define the original function
def g (x : ℝ) : ℝ := x^2 - 3*x

-- Define the symmetric function
def f (x : ℝ) : ℝ := -x^2 + 3*x

-- Theorem statement
theorem symmetric_about_x_axis : 
  ∀ x y : ℝ, g x = y ↔ f x = -y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_about_x_axis_l2608_260812


namespace NUMINAMATH_CALUDE_sum_of_f_values_l2608_260887

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem sum_of_f_values : 
  Real.sqrt 3 * (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + 
                 f 1 + f 2 + f 3 + f 4 + f 5 + f 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l2608_260887


namespace NUMINAMATH_CALUDE_total_cookies_l2608_260870

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in one row. -/
def cookies_per_row : ℕ := 6

/-- Theorem: The total number of cookies Lara is baking is 120. -/
theorem total_cookies : 
  num_trays * rows_per_tray * cookies_per_row = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l2608_260870


namespace NUMINAMATH_CALUDE_power_multiplication_l2608_260874

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2608_260874


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l2608_260841

/-- Calculates the total cost of tickets sold at a snooker tournament --/
theorem snooker_tournament_ticket_sales 
  (total_tickets : ℕ) 
  (vip_price general_price : ℚ) 
  (ticket_difference : ℕ) 
  (h1 : total_tickets = 320)
  (h2 : vip_price = 45)
  (h3 : general_price = 20)
  (h4 : ticket_difference = 276) :
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  vip_price * vip_tickets + general_price * general_tickets = 6950 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l2608_260841


namespace NUMINAMATH_CALUDE_sqrt_three_minus_two_times_sqrt_three_plus_two_l2608_260801

theorem sqrt_three_minus_two_times_sqrt_three_plus_two : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_two_times_sqrt_three_plus_two_l2608_260801


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l2608_260884

/-- Given two points on a parabola that are symmetric with respect to a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →  -- A is on the parabola
  y₂ = 2 * x₂^2 →  -- B is on the parabola
  (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₀ = x₀ + m) →  -- midpoint condition for symmetry
  x₁ * x₂ = -1/2 →  -- given condition
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l2608_260884


namespace NUMINAMATH_CALUDE_jills_speed_l2608_260853

/-- Proves that Jill's speed was 8 km/h given the conditions of the problem -/
theorem jills_speed (jack_distance1 jack_distance2 jack_speed1 jack_speed2 : ℝ)
  (h1 : jack_distance1 = 12)
  (h2 : jack_distance2 = 12)
  (h3 : jack_speed1 = 12)
  (h4 : jack_speed2 = 6)
  (jill_distance jill_time : ℝ)
  (h5 : jill_distance = jack_distance1 + jack_distance2)
  (h6 : jill_time = jack_distance1 / jack_speed1 + jack_distance2 / jack_speed2) :
  jill_distance / jill_time = 8 :=
sorry

end NUMINAMATH_CALUDE_jills_speed_l2608_260853


namespace NUMINAMATH_CALUDE_election_votes_l2608_260822

/-- In an election with 3 candidates, where two candidates received 5000 and 15000 votes
    respectively, and the winning candidate got 66.66666666666666% of the total votes,
    the winning candidate (third candidate) received 40000 votes. -/
theorem election_votes :
  let total_votes : ℕ := 60000
  let first_candidate_votes : ℕ := 5000
  let second_candidate_votes : ℕ := 15000
  let winning_percentage : ℚ := 200 / 3
  ∀ third_candidate_votes : ℕ,
    first_candidate_votes + second_candidate_votes + third_candidate_votes = total_votes →
    (third_candidate_votes : ℚ) / total_votes * 100 = winning_percentage →
    third_candidate_votes = 40000 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l2608_260822


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l2608_260824

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  (box.a + box.b + box.c = 40) ∧
  (box.a * box.b + box.b * box.c + box.c * box.a = 432) ∧
  (4 * box.s^2 = box.a^2 + box.b^2 + box.c^2)

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.s = 2 * Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l2608_260824


namespace NUMINAMATH_CALUDE_at_least_one_shot_hit_l2608_260828

theorem at_least_one_shot_hit (p q : Prop) : 
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_at_least_one_shot_hit_l2608_260828


namespace NUMINAMATH_CALUDE_initial_books_eq_sold_plus_left_l2608_260802

/-- The number of books Paul had initially -/
def initial_books : ℕ := 136

/-- The number of books Paul sold -/
def books_sold : ℕ := 109

/-- The number of books Paul was left with after the sale -/
def books_left : ℕ := 27

/-- Theorem stating that the initial number of books is equal to the sum of books sold and books left -/
theorem initial_books_eq_sold_plus_left : initial_books = books_sold + books_left := by
  sorry

end NUMINAMATH_CALUDE_initial_books_eq_sold_plus_left_l2608_260802


namespace NUMINAMATH_CALUDE_plywood_width_l2608_260820

theorem plywood_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 24 →
  length = 4 →
  area = length * width →
  width = 6 := by
sorry

end NUMINAMATH_CALUDE_plywood_width_l2608_260820


namespace NUMINAMATH_CALUDE_correct_calculation_l2608_260805

theorem correct_calculation (m : ℝ) : 6*m + (-2 - 10*m) = -4*m - 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2608_260805


namespace NUMINAMATH_CALUDE_village_population_theorem_l2608_260893

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : total_population % 4 = 0) 
  (h3 : 3 * (total_population / 4) = total_population - (total_population / 4)) :
  total_population / 4 = 200 :=
sorry

end NUMINAMATH_CALUDE_village_population_theorem_l2608_260893


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l2608_260843

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  x^2 - 6*x + 1 = a*(x - h)^2 + k ∧ k = -8 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l2608_260843


namespace NUMINAMATH_CALUDE_binomial_sum_even_n_l2608_260871

theorem binomial_sum_even_n (n : ℕ) (h : Even n) :
  (Finset.sum (Finset.range (n + 1)) (fun k =>
    if k % 2 = 0 then (1 : ℕ) * Nat.choose n k
    else 2 * Nat.choose n k)) = 3 * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_even_n_l2608_260871


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2608_260834

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in slope-intercept form -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.equation x y = (y = 2 * x + 3) →
  symmetric_about_y_axis l₁ l₂ →
  l₂.equation x y = (y = -2 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2608_260834


namespace NUMINAMATH_CALUDE_division_by_self_l2608_260860

theorem division_by_self (a : ℝ) (h : a ≠ 0) : 3 * a / a = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_by_self_l2608_260860


namespace NUMINAMATH_CALUDE_race_distance_l2608_260816

theorem race_distance (speed_A speed_B speed_C : ℝ) : 
  (speed_A / speed_B = 1000 / 900) →
  (speed_B / speed_C = 800 / 700) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5) →
  (∃ D : ℝ, D > 0 ∧ D * (speed_A / speed_C - 1) = 127.5 ∧ D = 600) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l2608_260816


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2608_260832

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3/2) + 3 * (1/6) + q = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2608_260832


namespace NUMINAMATH_CALUDE_shortest_tangent_is_30_l2608_260898

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- The given circles from the problem --/
def problem_circles : TwoCircles :=
  { c1 := λ (x, y) => (x - 12)^2 + y^2 = 25,
    c2 := λ (x, y) => (x + 18)^2 + y^2 = 64 }

/-- The length of the shortest line segment tangent to both circles --/
def shortest_tangent_length (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the shortest tangent length for the given circles is 30 --/
theorem shortest_tangent_is_30 :
  shortest_tangent_length problem_circles = 30 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_is_30_l2608_260898


namespace NUMINAMATH_CALUDE_base8_4532_equals_2394_l2608_260865

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- State the theorem
theorem base8_4532_equals_2394 :
  base8ToBase10 [2, 3, 5, 4] = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base8_4532_equals_2394_l2608_260865


namespace NUMINAMATH_CALUDE_triangle_properties_l2608_260888

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides opposite to A, B, C respectively

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  (abc.B = π/3) ∧ (abc.a = Real.sqrt 3) ∧ (abc.c = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2608_260888


namespace NUMINAMATH_CALUDE_central_region_area_l2608_260872

/-- The area of the central region in a square with intersecting lines --/
theorem central_region_area (s : ℝ) (h : s = 10) : 
  let a := s / 3
  let b := 2 * s / 3
  let central_side := (s - (a + b)) / 2
  central_side ^ 2 = (s / 6) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_central_region_area_l2608_260872


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_9_l2608_260877

/-- The number formed by inserting a digit d between 586 and 17 -/
def number (d : Nat) : Nat := 586000 + d * 1000 + 17

/-- Predicate to check if a number is divisible by 9 -/
def divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem smallest_digit_divisible_by_9 :
  ∃ (d : Nat), d < 10 ∧ divisible_by_9 (number d) ∧
  ∀ (d' : Nat), d' < d → ¬(divisible_by_9 (number d')) :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_9_l2608_260877


namespace NUMINAMATH_CALUDE_problem_statement_l2608_260854

theorem problem_statement : ((18^10 / 18^9)^3 * 16^3) / 8^6 = 91.125 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2608_260854


namespace NUMINAMATH_CALUDE_special_function_form_l2608_260807

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  f_zero_eq_one : f 0 = 1
  f_inequality : ∀ x₁ x₂, f (x₁ + x₂) ≥ f x₁ * f x₂

/-- The main theorem: any function satisfying the given conditions is of the form e^(kx) -/
theorem special_function_form (φ : SpecialFunction) :
  ∃ k : ℝ, ∀ x, φ.f x = Real.exp (k * x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_form_l2608_260807


namespace NUMINAMATH_CALUDE_min_acquaintances_in_village_l2608_260833

/-- Represents a village with residents and their acquaintances. -/
structure Village where
  residents : Finset ℕ
  acquaintances : Finset (ℕ × ℕ)

/-- Checks if a given set of residents can be seated according to the problem's conditions. -/
def canBeSeatedCircularly (v : Village) (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (seating : Fin 6 → ℕ), 
    (∀ i, seating i ∈ group) ∧
    (∀ i, (seating i, seating ((i + 1) % 6)) ∈ v.acquaintances ∧
          (seating i, seating ((i + 5) % 6)) ∈ v.acquaintances)

/-- The main theorem statement. -/
theorem min_acquaintances_in_village (v : Village) :
  v.residents.card = 200 ∧ 
  (∀ group : Finset ℕ, group ⊆ v.residents → canBeSeatedCircularly v group) →
  v.acquaintances.card = 19600 :=
sorry

end NUMINAMATH_CALUDE_min_acquaintances_in_village_l2608_260833


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2608_260863

open Real

/-- A function f : ℝ → ℝ is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem neither_sufficient_nor_necessary
  (a : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1) :
  ¬(IsIncreasing (fun x ↦ a^x) → IsIncreasing (fun x ↦ x^a)) ∧
  ¬(IsIncreasing (fun x ↦ x^a) → IsIncreasing (fun x ↦ a^x)) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2608_260863


namespace NUMINAMATH_CALUDE_expression_value_l2608_260819

theorem expression_value : (3^4 * 5^2 * 7^3 * 11) / (7 * 11^2) = 9025 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2608_260819


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l2608_260804

theorem at_least_one_real_root (p₁ p₂ q₁ q₂ : ℝ) (h : p₁ * p₂ = 2 * (q₁ + q₂)) :
  (∃ x : ℝ, x^2 + p₁*x + q₁ = 0) ∨ (∃ x : ℝ, x^2 + p₂*x + q₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l2608_260804


namespace NUMINAMATH_CALUDE_min_fraction_value_l2608_260856

theorem min_fraction_value (x y : ℝ) (h : Real.sqrt (x - 1) + Real.sqrt (y - 1) = 1) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (z : ℝ), z = x/y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l2608_260856


namespace NUMINAMATH_CALUDE_problem_solution_l2608_260840

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 0)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2608_260840


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2608_260858

-- Define the ellipse E
def ellipse_E (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the vertex condition
def vertex_condition (a b : ℝ) : Prop :=
  ellipse_E 0 1 a b

-- Define the focal length condition
def focal_length_condition (a b : ℝ) : Prop :=
  2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 - b^2)

-- Define the intersection condition
def intersection_condition (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_E x₁ y₁ a b ∧
    ellipse_E x₂ y₂ a b ∧
    y₁ - 1 = k * (x₁ + 2) ∧
    y₂ - 1 = k * (x₂ + 2) ∧
    x₁ ≠ x₂

-- Define the x-intercept distance condition
def x_intercept_distance (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersection_condition k a b ∧
    |x₁ / (1 - y₁) - x₂ / (1 - y₂)| = 2

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) :
  vertex_condition a b ∧ focal_length_condition a b →
  (a = 2 ∧ b = 1) ∧
  (∀ k : ℝ, x_intercept_distance k a b → k = -4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2608_260858


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2608_260806

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2608_260806


namespace NUMINAMATH_CALUDE_prob_at_least_one_from_three_suits_l2608_260892

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Number of cards drawn -/
def numDraws : ℕ := 5

/-- Number of specific suits considered -/
def numSpecificSuits : ℕ := 3

/-- Probability of drawing a card from the specific suits in one draw -/
def probSpecificSuits : ℚ := (cardsPerSuit * numSpecificSuits) / standardDeck

/-- Probability of drawing a card not from the specific suits in one draw -/
def probNotSpecificSuits : ℚ := 1 - probSpecificSuits

/-- 
Theorem: The probability of drawing at least one card from each of three specific suits 
when choosing five cards with replacement from a standard 52-card deck is 1023/1024.
-/
theorem prob_at_least_one_from_three_suits : 
  1 - probNotSpecificSuits ^ numDraws = 1023 / 1024 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_from_three_suits_l2608_260892


namespace NUMINAMATH_CALUDE_polygon_sides_l2608_260885

theorem polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 40) :
  (360 : ℝ) / exterior_angle = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2608_260885


namespace NUMINAMATH_CALUDE_modulo_nine_sum_l2608_260847

theorem modulo_nine_sum (n : ℕ) : n = 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 → 0 ≤ n ∧ n < 9 → n % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_sum_l2608_260847


namespace NUMINAMATH_CALUDE_river_trip_longer_than_lake_trip_l2608_260864

theorem river_trip_longer_than_lake_trip 
  (a b S : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : b < a) 
  (hS : S > 0) : 
  (2 * a * S) / (a^2 - b^2) > (2 * S) / a := by
  sorry

end NUMINAMATH_CALUDE_river_trip_longer_than_lake_trip_l2608_260864


namespace NUMINAMATH_CALUDE_squirrel_count_ratio_l2608_260825

theorem squirrel_count_ratio :
  ∀ (first_count second_count : ℕ),
  first_count = 12 →
  first_count + second_count = 28 →
  second_count > first_count →
  (second_count : ℚ) / first_count = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_ratio_l2608_260825


namespace NUMINAMATH_CALUDE_sandwich_problem_l2608_260869

theorem sandwich_problem (billy_sandwiches katelyn_sandwiches chloe_sandwiches : ℕ) : 
  billy_sandwiches = 49 →
  chloe_sandwiches = katelyn_sandwiches / 4 →
  billy_sandwiches + katelyn_sandwiches + chloe_sandwiches = 169 →
  katelyn_sandwiches > billy_sandwiches →
  katelyn_sandwiches - billy_sandwiches = 47 := by
sorry


end NUMINAMATH_CALUDE_sandwich_problem_l2608_260869


namespace NUMINAMATH_CALUDE_amounts_theorem_l2608_260855

/-- Represents the amounts held by individuals p, q, r, s, and t -/
structure Amounts where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ

/-- The total amount among all individuals is $24,000 -/
def total_amount : ℝ := 24000

/-- The conditions given in the problem -/
def satisfies_conditions (a : Amounts) : Prop :=
  a.p + a.q + a.r + a.s + a.t = total_amount ∧
  a.r = 3/5 * (a.p + a.q) ∧
  a.s = 0.45 * total_amount ∧
  a.t = 1/2 * a.r

/-- The theorem to be proved -/
theorem amounts_theorem (a : Amounts) (h : satisfies_conditions a) : 
  a.r = 4200 ∧ a.s = 10800 ∧ a.t = 2100 ∧ a.p + a.q = 7000 := by
  sorry

end NUMINAMATH_CALUDE_amounts_theorem_l2608_260855


namespace NUMINAMATH_CALUDE_square_difference_identity_l2608_260852

theorem square_difference_identity : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2608_260852


namespace NUMINAMATH_CALUDE_line_intersection_condition_l2608_260899

/-- Given a directed line segment PQ and a line l, prove that l intersects
    the extended line segment PQ if and only if m is within a specific range. -/
theorem line_intersection_condition (m : ℝ) : 
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∃ (t : ℝ), (1 - t) • P + t • Q ∈ l) ↔ -3 < m ∧ m < -2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_condition_l2608_260899


namespace NUMINAMATH_CALUDE_expression1_eval_expression2_eval_l2608_260823

-- Part 1
def expression1 (x : ℝ) : ℝ := -3*x^2 + 5*x - 0.5*x^2 + x - 1

theorem expression1_eval : expression1 2 = -3 := by sorry

-- Part 2
def expression2 (a b : ℝ) : ℝ := (a^2*b + 3*a*b^2) - 3*(a^2*b + a*b^2 - 1)

theorem expression2_eval : expression2 (-2) 2 = -13 := by sorry

end NUMINAMATH_CALUDE_expression1_eval_expression2_eval_l2608_260823


namespace NUMINAMATH_CALUDE_intersection_A_B_l2608_260895

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- Define the interval [2, 3)
def interval_2_3 : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_2_3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2608_260895


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l2608_260826

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 30 ∧ 
  (∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 24 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l2608_260826


namespace NUMINAMATH_CALUDE_range_of_distance_from_origin_l2608_260891

theorem range_of_distance_from_origin : ∀ x y : ℝ,
  x + y = 10 →
  -5 ≤ x - y →
  x - y ≤ 5 →
  5 * Real.sqrt 2 ≤ Real.sqrt (x^2 + y^2) ∧
  Real.sqrt (x^2 + y^2) ≤ (5 * Real.sqrt 10) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_distance_from_origin_l2608_260891


namespace NUMINAMATH_CALUDE_unique_prime_power_condition_l2608_260867

theorem unique_prime_power_condition : ∃! p : ℕ, 
  p ≤ 1000 ∧ 
  Nat.Prime p ∧ 
  ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m ^ n ∧
  p = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_power_condition_l2608_260867


namespace NUMINAMATH_CALUDE_f_at_neg_one_equals_two_l2608_260886

-- Define the function f(x) = -2x
def f (x : ℝ) : ℝ := -2 * x

-- Theorem stating that f(-1) = 2
theorem f_at_neg_one_equals_two : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_neg_one_equals_two_l2608_260886


namespace NUMINAMATH_CALUDE_max_coeff_seventh_term_l2608_260838

theorem max_coeff_seventh_term (n : ℕ) : 
  (∃ k, (Nat.choose n k = Nat.choose n 6) ∧ 
        (∀ j, 0 ≤ j ∧ j ≤ n → Nat.choose n j ≤ Nat.choose n 6)) →
  n ∈ ({11, 12, 13} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_max_coeff_seventh_term_l2608_260838


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2608_260818

-- Define the function
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_satisfies_conditions :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through first quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through second quadrant
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂)  -- Increasing in first quadrant
  := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2608_260818


namespace NUMINAMATH_CALUDE_sin_plus_cos_alpha_l2608_260879

theorem sin_plus_cos_alpha (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10)
  (h2 : Real.cos (2 * α) = 7/25) : 
  Real.sin α + Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_alpha_l2608_260879


namespace NUMINAMATH_CALUDE_ps_length_l2608_260800

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (PQRS : Quadrilateral) : Prop :=
  let (px, py) := PQRS.P
  let (qx, qy) := PQRS.Q
  let (rx, ry) := PQRS.R
  let (sx, sy) := PQRS.S
  -- PQ = 6
  (px - qx)^2 + (py - qy)^2 = 36 ∧
  -- QR = 10
  (qx - rx)^2 + (qy - ry)^2 = 100 ∧
  -- RS = 25
  (rx - sx)^2 + (ry - sy)^2 = 625 ∧
  -- Angle Q is right angle
  (px - qx) * (rx - qx) + (py - qy) * (ry - qy) = 0 ∧
  -- Angle R is right angle
  (qx - rx) * (sx - rx) + (qy - ry) * (sy - ry) = 0

-- Theorem statement
theorem ps_length (PQRS : Quadrilateral) (h : is_valid_quadrilateral PQRS) :
  (PQRS.P.1 - PQRS.S.1)^2 + (PQRS.P.2 - PQRS.S.2)^2 = 461 :=
by sorry

end NUMINAMATH_CALUDE_ps_length_l2608_260800


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_prime_factors_2310_l2608_260811

def sum_of_digits (n : ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_of_sum_of_prime_factors_2310 : 
  sum_of_digits (List.sum (prime_factors 2310)) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_prime_factors_2310_l2608_260811


namespace NUMINAMATH_CALUDE_expression_simplification_l2608_260890

theorem expression_simplification (a : ℝ) (h : a = -2) :
  (1 - a / (a + 1)) / (1 / (1 - a^2)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2608_260890


namespace NUMINAMATH_CALUDE_smallest_n_is_101_l2608_260876

/-- Represents a square in the n × n table -/
structure Square where
  row : Nat
  col : Nat

/-- Represents a rectangle in the table -/
structure Rectangle where
  topLeft : Square
  bottomRight : Square

/-- Represents the n × n table -/
structure Table (n : Nat) where
  blueSquares : Finset Square
  rectangles : Finset Rectangle
  uniquePartition : Prop
  oneBluePerRectangle : Prop

/-- The main theorem -/
theorem smallest_n_is_101 :
  ∀ n : Nat,
  ∃ (t : Table n),
  (t.blueSquares.card = 101) →
  t.uniquePartition →
  t.oneBluePerRectangle →
  n ≥ 101 ∧ ∃ (t' : Table 101), 
    t'.blueSquares.card = 101 ∧
    t'.uniquePartition ∧
    t'.oneBluePerRectangle :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_101_l2608_260876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2608_260821

/-- For an arithmetic sequence {a_n} with a_2 = 3 and a_5 = 12, the common difference d is 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_a2 : a 2 = 3)  -- Given: a_2 = 3
  (h_a5 : a 5 = 12)  -- Given: a_5 = 12
  : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2608_260821


namespace NUMINAMATH_CALUDE_cricket_team_age_theorem_l2608_260830

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ) : Prop :=
  let total_age := team_size * average_age
  let captain_and_keeper_age := captain_age + (captain_age + wicket_keeper_age_diff)
  let remaining_players := team_size - 2
  total_age - captain_and_keeper_age = remaining_players * (average_age - remaining_players_age_diff)
  where
    average_age : ℕ := 23

theorem cricket_team_age_theorem :
  cricket_team_age_problem 11 26 3 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_theorem_l2608_260830


namespace NUMINAMATH_CALUDE_work_completion_larger_group_size_l2608_260875

theorem work_completion (work_days : ℕ) (small_group : ℕ) (large_group_days : ℕ) : ℕ :=
  let total_man_days := work_days * small_group
  total_man_days / large_group_days

theorem larger_group_size : work_completion 25 12 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_larger_group_size_l2608_260875


namespace NUMINAMATH_CALUDE_square_octagon_tessellation_l2608_260803

-- Define the internal angles of regular polygons
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def hexagon_angle : ℝ := 120
def octagon_angle : ℝ := 135

-- Define a predicate for seamless tessellation
def can_tessellate (angle1 angle2 : ℝ) : Prop :=
  ∃ (n m : ℕ), n * angle1 + m * angle2 = 360

-- Theorem statement
theorem square_octagon_tessellation :
  can_tessellate square_angle octagon_angle ∧
  ¬can_tessellate square_angle hexagon_angle ∧
  ¬can_tessellate square_angle pentagon_angle ∧
  ¬can_tessellate hexagon_angle octagon_angle ∧
  ¬can_tessellate pentagon_angle octagon_angle :=
sorry

end NUMINAMATH_CALUDE_square_octagon_tessellation_l2608_260803


namespace NUMINAMATH_CALUDE_trig_identity_l2608_260846

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (65 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2608_260846


namespace NUMINAMATH_CALUDE_rods_per_sheet_is_correct_l2608_260866

/-- Represents the number of metal rods in each metal sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 2

/-- Represents the total number of fence panels -/
def total_panels : ℕ := 10

/-- Represents the number of metal rods in each metal beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the entire fence -/
def total_rods : ℕ := 380

/-- Theorem stating that the number of rods per sheet is correct given the conditions -/
theorem rods_per_sheet_is_correct :
  rods_per_sheet * (sheets_per_panel * total_panels) + 
  rods_per_beam * (beams_per_panel * total_panels) = total_rods :=
by sorry

end NUMINAMATH_CALUDE_rods_per_sheet_is_correct_l2608_260866


namespace NUMINAMATH_CALUDE_similar_triangles_collinearity_l2608_260897

/-- Two triangles are similar if they have the same shape but possibly different size and orientation -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- Two triangles are differently oriented if they cannot be made to coincide by translation and scaling -/
def differently_oriented (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- A point divides a segment in a given ratio -/
def divides_segment_in_ratio (A A' A₁ : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ t : ℝ, A' = (1 - t) • A + t • A₁ ∧ r = t / (1 - t)

/-- Three points are collinear if they lie on a single straight line -/
def collinear (A B C : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_collinearity 
  (A B C A₁ B₁ C₁ A' B' C' : ℝ × ℝ) 
  (ABC : Set (Fin 3 → ℝ × ℝ)) 
  (A₁B₁C₁ : Set (Fin 3 → ℝ × ℝ)) 
  (h_similar : similar_triangles ABC A₁B₁C₁)
  (h_oriented : differently_oriented ABC A₁B₁C₁)
  (h_A' : divides_segment_in_ratio A A' A₁ (dist B C / dist B₁ C₁))
  (h_B' : divides_segment_in_ratio B B' B₁ (dist B C / dist B₁ C₁))
  (h_C' : divides_segment_in_ratio C C' C₁ (dist B C / dist B₁ C₁)) :
  collinear A' B' C' := by sorry

end NUMINAMATH_CALUDE_similar_triangles_collinearity_l2608_260897


namespace NUMINAMATH_CALUDE_factor_polynomial_l2608_260808

theorem factor_polynomial (k : ℤ) : 
  (∀ x : ℝ, (x + k) ∣ (3 * x^2 + 14 * x + 8)) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2608_260808


namespace NUMINAMATH_CALUDE_fred_cantaloupes_l2608_260813

theorem fred_cantaloupes (keith_cantaloupes jason_cantaloupes total_cantaloupes : ℕ)
  (h1 : keith_cantaloupes = 29)
  (h2 : jason_cantaloupes = 20)
  (h3 : total_cantaloupes = 65)
  (h4 : ∃ fred_cantaloupes : ℕ, keith_cantaloupes + jason_cantaloupes + fred_cantaloupes = total_cantaloupes) :
  ∃ fred_cantaloupes : ℕ, fred_cantaloupes = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_fred_cantaloupes_l2608_260813


namespace NUMINAMATH_CALUDE_old_man_coins_l2608_260878

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_old_man_coins_l2608_260878


namespace NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l2608_260868

/-- The function g(x) = x^2 + 2x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + d

/-- The largest value of d such that -5 is in the range of g(x) -/
theorem largest_d_for_negative_five_in_range : 
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g e x = -5) → e ≤ d) ∧ 
  (∃ (x : ℝ), g (-4) x = -5) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l2608_260868


namespace NUMINAMATH_CALUDE_expression_is_integer_not_necessarily_natural_l2608_260809

theorem expression_is_integer_not_necessarily_natural : ∃ (n : ℤ), 
  (((1 + Real.sqrt 1991)^100 - (1 - Real.sqrt 1991)^100) / Real.sqrt 1991 = n) ∧ 
  (n ≠ 0 ∨ n < 0) := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_not_necessarily_natural_l2608_260809


namespace NUMINAMATH_CALUDE_white_balls_added_l2608_260837

theorem white_balls_added (m : ℕ) : 
  (10 + m : ℚ) / (16 + m) = 4/5 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_added_l2608_260837


namespace NUMINAMATH_CALUDE_unique_p_q_solution_l2608_260817

theorem unique_p_q_solution :
  ∀ p q : ℝ,
    p ≠ q →
    p > 1 →
    q > 1 →
    1 / p + 1 / q = 1 →
    p * q = 9 →
    ((p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨
     (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_p_q_solution_l2608_260817


namespace NUMINAMATH_CALUDE_soda_cans_for_euros_l2608_260882

/-- The number of cans of soda that can be purchased for E euros, given that S cans can be purchased for Q quarters and 1 euro is worth 5 quarters. -/
theorem soda_cans_for_euros (S Q E : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : E > 0) :
  (S / Q) * (5 * E) = (5 * S * E) / Q := by
  sorry

#check soda_cans_for_euros

end NUMINAMATH_CALUDE_soda_cans_for_euros_l2608_260882


namespace NUMINAMATH_CALUDE_video_likes_dislikes_ratio_l2608_260844

theorem video_likes_dislikes_ratio :
  ∀ (initial_dislikes : ℕ),
    (initial_dislikes + 1000 = 2600) →
    (initial_dislikes : ℚ) / 3000 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_video_likes_dislikes_ratio_l2608_260844


namespace NUMINAMATH_CALUDE_expression_evaluation_l2608_260836

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1
  3 * (2 * a^2 * b - a * b^2) - 2 * (5 * a^2 * b - 2 * a * b^2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2608_260836


namespace NUMINAMATH_CALUDE_soup_bins_calculation_l2608_260880

def total_bins : ℚ := 75/100
def vegetable_bins : ℚ := 12/100
def pasta_bins : ℚ := 1/2

theorem soup_bins_calculation : 
  total_bins - (vegetable_bins + pasta_bins) = 13/100 := by
  sorry

end NUMINAMATH_CALUDE_soup_bins_calculation_l2608_260880


namespace NUMINAMATH_CALUDE_alcohol_volume_bound_l2608_260845

/-- Represents the volume of pure alcohol in container B after n operations -/
def alcohol_volume (x y z : ℝ) (n : ℕ+) : ℝ :=
  sorry

/-- Theorem stating that the volume of pure alcohol in container B 
    is always less than or equal to xy/(x+y) -/
theorem alcohol_volume_bound (x y z : ℝ) (n : ℕ+) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x < z) (hyz : y < z) :
  alcohol_volume x y z n ≤ (x * y) / (x + y) :=
sorry

end NUMINAMATH_CALUDE_alcohol_volume_bound_l2608_260845


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2608_260814

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0 → a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2608_260814


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l2608_260829

/-- Returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 26 is the smallest positive integer n for which f(n) = 4 -/
theorem smallest_n_with_four_pairs : ∀ m : ℕ, m > 0 → m < 26 → f m ≠ 4 ∧ f 26 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l2608_260829


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2608_260851

theorem ratio_sum_problem (a b : ℕ) : 
  a * 3 = b * 8 →  -- The two numbers are in the ratio 8 to 3
  b = 104 →        -- The bigger number is 104
  a + b = 143      -- The sum of the numbers is 143
  := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2608_260851


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l2608_260883

theorem max_value_3x_4y (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 14*a + 6*b + 6 → 3*a + 4*b ≤ max) ∧ max = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l2608_260883


namespace NUMINAMATH_CALUDE_intersection_dot_product_l2608_260896

/-- Given an ellipse and a hyperbola with common foci, the dot product of vectors from their intersection point to the foci is 21. -/
theorem intersection_dot_product (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  x^2/25 + y^2/16 = 1 →  -- Ellipse equation
  x^2/4 - y^2/5 = 1 →    -- Hyperbola equation
  P = (x, y) →           -- P is on both curves
  (∃ c : ℝ, c > 0 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (5 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (5 - c)^2 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (2 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (c - 2)^2) →  -- Common foci condition
  ((F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) : ℝ) = 21 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l2608_260896


namespace NUMINAMATH_CALUDE_angle_120_in_second_quadrant_l2608_260842

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  /-- The measure of the angle in degrees -/
  measure : ℝ
  /-- The angle's vertex is at the origin -/
  vertex_at_origin : Bool
  /-- The angle's initial side is along the positive x-axis -/
  initial_side_positive_x : Bool

/-- Definition of the second quadrant -/
def is_in_second_quadrant (angle : CartesianAngle) : Prop :=
  angle.measure > 90 ∧ angle.measure < 180

/-- Theorem: An angle of 120° with vertex at origin and initial side along positive x-axis is in the second quadrant -/
theorem angle_120_in_second_quadrant :
  ∀ (angle : CartesianAngle),
    angle.measure = 120 ∧
    angle.vertex_at_origin = true ∧
    angle.initial_side_positive_x = true →
    is_in_second_quadrant angle :=
by sorry

end NUMINAMATH_CALUDE_angle_120_in_second_quadrant_l2608_260842


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2608_260850

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ Bᶜ = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2608_260850


namespace NUMINAMATH_CALUDE_triangle_centroid_existence_and_property_l2608_260894

/-- Given a triangle ABC, there exists a unique point O (the centroid) that lies on all medians and divides each in a 2:1 ratio from the vertex. -/
theorem triangle_centroid_existence_and_property (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃! O : EuclideanSpace ℝ (Fin 2),
    (∃ t : ℝ, O = A + t • (midpoint ℝ B C - A)) ∧
    (∃ u : ℝ, O = B + u • (midpoint ℝ A C - B)) ∧
    (∃ v : ℝ, O = C + v • (midpoint ℝ A B - C)) ∧
    (O = A + (2/3) • (midpoint ℝ B C - A)) ∧
    (O = B + (2/3) • (midpoint ℝ A C - B)) ∧
    (O = C + (2/3) • (midpoint ℝ A B - C)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_existence_and_property_l2608_260894


namespace NUMINAMATH_CALUDE_existence_of_n_l2608_260827

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_cd : c * d = 1) : 
  ∃ n : ℤ, a * b ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_l2608_260827


namespace NUMINAMATH_CALUDE_acid_dilution_l2608_260848

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l2608_260848


namespace NUMINAMATH_CALUDE_percent_relation_l2608_260849

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2608_260849


namespace NUMINAMATH_CALUDE_original_room_width_l2608_260889

/-- Proves that the original width of the room is 18 feet given the problem conditions -/
theorem original_room_width (length : ℝ) (increased_size : ℝ) (total_area : ℝ) : 
  length = 13 →
  increased_size = 2 →
  total_area = 1800 →
  ∃ w : ℝ, 
    (4 * ((length + increased_size) * (w + increased_size)) + 
     2 * ((length + increased_size) * (w + increased_size))) = total_area ∧
    w = 18 := by
  sorry

end NUMINAMATH_CALUDE_original_room_width_l2608_260889


namespace NUMINAMATH_CALUDE_f_4_3_2_1_l2608_260810

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Theorem stating that f(4, 3, 2, 1) = (0, -3, 4, -1) -/
theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by sorry

end NUMINAMATH_CALUDE_f_4_3_2_1_l2608_260810


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l2608_260839

/-- A parabola is defined by its equation in the form y² = -4px, where p is the focal length. -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -4 * p * x

/-- The directrix of a parabola is a vertical line with equation x = p. -/
def directrix (parabola : Parabola) : ℝ → Prop :=
  fun x => x = parabola.p

theorem parabola_directrix_equation :
  ∀ (y : ℝ), ∃ (parabola : Parabola),
    (∀ (x : ℝ), parabola.equation x y ↔ y^2 = -4*x) →
    (∀ (x : ℝ), directrix parabola x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l2608_260839


namespace NUMINAMATH_CALUDE_smallest_power_congruence_l2608_260861

theorem smallest_power_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 100 → (2013 ^ m) % 1000 ≠ 1) ∧ 
  (2013 ^ 100) % 1000 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l2608_260861
