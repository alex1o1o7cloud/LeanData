import Mathlib

namespace mary_books_checked_out_l2593_259369

/-- Calculates the number of books Mary has checked out after a series of transactions. -/
def books_checked_out (initial : ℕ) (first_return : ℕ) (first_checkout : ℕ) (second_return : ℕ) (second_checkout : ℕ) : ℕ :=
  initial - first_return + first_checkout - second_return + second_checkout

/-- Proves that Mary has 12 books checked out after the given transactions. -/
theorem mary_books_checked_out : 
  books_checked_out 5 3 5 2 7 = 12 := by
  sorry

end mary_books_checked_out_l2593_259369


namespace expression_1_equality_expression_2_equality_expression_3_equality_l2593_259330

-- Expression 1
theorem expression_1_equality : (-4)^2 - 6 * (4/3) + 2 * (-1)^3 / (-1/2) = 12 := by sorry

-- Expression 2
theorem expression_2_equality : -1^4 - 1/6 * |2 - (-3)^2| = -13/6 := by sorry

-- Expression 3
theorem expression_3_equality (x y : ℝ) (h : |x+2| + (y-1)^2 = 0) :
  2*(3*x^2*y + x*y^2) - 3*(2*x^2*y - x*y) - 2*x*y^2 + 1 = -5 := by sorry

end expression_1_equality_expression_2_equality_expression_3_equality_l2593_259330


namespace pipe_speed_ratio_l2593_259358

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 28
def rate_B : ℚ := 1 / 14
def rate_C : ℚ := 1 / 7

-- Theorem statement
theorem pipe_speed_ratio :
  -- Given conditions
  (rate_A + rate_B + rate_C = 1 / 4) →  -- All pipes fill the tank in 4 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 28) →                   -- Pipe A alone takes 28 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry


end pipe_speed_ratio_l2593_259358


namespace role_assignment_theorem_l2593_259332

def number_of_ways_to_assign_roles (men : Nat) (women : Nat) (male_roles : Nat) (female_roles : Nat) (either_roles : Nat) : Nat :=
  -- Number of ways to assign male roles
  (men.choose male_roles) * (male_roles.factorial) *
  -- Number of ways to assign female roles
  (women.choose female_roles) * (female_roles.factorial) *
  -- Number of ways to assign either-gender roles
  ((men + women - male_roles - female_roles).choose either_roles) * (either_roles.factorial)

theorem role_assignment_theorem :
  number_of_ways_to_assign_roles 6 7 3 3 2 = 1058400 := by
  sorry

end role_assignment_theorem_l2593_259332


namespace geometric_sequence_ratio_l2593_259384

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →                          -- q ≠ 1 condition
  a 2 = 9 →                        -- a_2 = 9 condition
  a 3 + a 4 = 18 →                 -- a_3 + a_4 = 18 condition
  q = -2 :=                        -- conclusion: q = -2
by
  sorry

end geometric_sequence_ratio_l2593_259384


namespace fractional_equation_solution_range_l2593_259325

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 3 ∧ (2 / (x - 3) + (x + m) / (3 - x) = 2)) →
  m ≤ 8 ∧ m ≠ -1 := by
sorry

end fractional_equation_solution_range_l2593_259325


namespace total_carriages_l2593_259348

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flyingScotsman : ℕ

/-- The conditions given in the problem -/
def problemConditions (t : TownCarriages) : Prop :=
  t.euston = t.norfolk + 20 ∧
  t.norwich = 100 ∧
  t.flyingScotsman = t.norwich + 20 ∧
  t.euston = 130

/-- The theorem to prove -/
theorem total_carriages (t : TownCarriages) 
  (h : problemConditions t) : 
  t.euston + t.norfolk + t.norwich + t.flyingScotsman = 460 :=
by
  sorry

end total_carriages_l2593_259348


namespace pepperoni_coverage_fraction_l2593_259386

-- Define the pizza and pepperoni characteristics
def pizza_diameter : ℝ := 16
def pepperoni_count : ℕ := 32
def pepperoni_across_diameter : ℕ := 8
def pepperoni_overlap_fraction : ℝ := 0.25

-- Theorem statement
theorem pepperoni_coverage_fraction :
  let pepperoni_diameter : ℝ := pizza_diameter / pepperoni_across_diameter
  let pepperoni_radius : ℝ := pepperoni_diameter / 2
  let pepperoni_area : ℝ := π * pepperoni_radius^2
  let effective_pepperoni_area : ℝ := pepperoni_area * (1 - pepperoni_overlap_fraction)
  let total_pepperoni_area : ℝ := pepperoni_count * effective_pepperoni_area
  let pizza_area : ℝ := π * (pizza_diameter / 2)^2
  total_pepperoni_area / pizza_area = 3/8 := by
  sorry

end pepperoni_coverage_fraction_l2593_259386


namespace seats_needed_l2593_259346

theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) 
  (h1 : total_children = 58) 
  (h2 : children_per_seat = 2) : 
  total_children / children_per_seat = 29 := by
sorry

end seats_needed_l2593_259346


namespace smallest_m_divisible_by_31_l2593_259326

theorem smallest_m_divisible_by_31 :
  ∃ (m : ℕ), m = 30 ∧
  (∀ (n : ℕ), n > 0 → 31 ∣ (m + 2^(5*n))) ∧
  (∀ (k : ℕ), k < m → ∃ (n : ℕ), n > 0 ∧ ¬(31 ∣ (k + 2^(5*n)))) :=
by sorry

end smallest_m_divisible_by_31_l2593_259326


namespace total_bones_equals_twelve_l2593_259345

/-- The number of bones carried by each dog in a pack of 5 dogs. -/
def DogBones : Fin 5 → ℕ
  | 0 => 3  -- First dog
  | 1 => DogBones 0 - 1  -- Second dog
  | 2 => 2 * DogBones 1  -- Third dog
  | 3 => 1  -- Fourth dog
  | 4 => 2 * DogBones 3  -- Fifth dog

/-- The theorem states that the sum of bones carried by all 5 dogs equals 12. -/
theorem total_bones_equals_twelve :
  (Finset.sum Finset.univ DogBones) = 12 := by
  sorry


end total_bones_equals_twelve_l2593_259345


namespace single_elimination_tournament_games_l2593_259398

theorem single_elimination_tournament_games (initial_teams : ℕ) (preliminary_games : ℕ) 
  (eliminated_teams : ℕ) (h1 : initial_teams = 24) (h2 : preliminary_games = 4) 
  (h3 : eliminated_teams = 4) :
  preliminary_games + (initial_teams - eliminated_teams - 1) = 23 := by
  sorry

end single_elimination_tournament_games_l2593_259398


namespace lcm_of_4_8_9_10_l2593_259359

theorem lcm_of_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end lcm_of_4_8_9_10_l2593_259359


namespace circle_center_and_radius_l2593_259339

/-- Given a circle C defined by the equation x^2 + y^2 - 2x + 6y = 0,
    prove that its center is (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧
    radius = Real.sqrt 10 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end circle_center_and_radius_l2593_259339


namespace percentage_problem_l2593_259319

theorem percentage_problem :
  ∃ x : ℝ, 0.0425 * x = 2.125 ∧ x = 50 := by
  sorry

end percentage_problem_l2593_259319


namespace functional_equation_solution_l2593_259317

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (f x + y + 1) = x + f y + 1) →
  ((∀ n : ℤ, f n = n) ∨ (∀ n : ℤ, f n = -n - 2)) :=
by sorry

end functional_equation_solution_l2593_259317


namespace tony_ken_ratio_l2593_259396

def total_amount : ℚ := 5250
def ken_amount : ℚ := 1750

theorem tony_ken_ratio :
  let tony_amount := total_amount - ken_amount
  (tony_amount : ℚ) / ken_amount = 2 := by sorry

end tony_ken_ratio_l2593_259396


namespace f_explicit_formula_b_value_l2593_259347

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 5

-- Define the function g
def g (b : ℝ) : ℝ → ℝ := fun x ↦ f x - b * x

-- Theorem for the first part
theorem f_explicit_formula : ∀ x : ℝ, f (x - 2) = x^2 - 4*x + 9 := by sorry

-- Theorem for the second part
theorem b_value : 
  ∃ b : ℝ, b = 1/2 ∧ 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, g b x ≤ 11/2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, g b x = 11/2) := by sorry

end f_explicit_formula_b_value_l2593_259347


namespace f_of_3_eq_19_l2593_259312

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem: f(3) = 19 -/
theorem f_of_3_eq_19 : f 3 = 19 := by
  sorry

end f_of_3_eq_19_l2593_259312


namespace inequality_and_equality_l2593_259378

theorem inequality_and_equality (x : ℝ) (h : x > 0) :
  Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) ≥ 1 ∧
  (Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) = 1 ↔ x = 1) :=
by sorry

end inequality_and_equality_l2593_259378


namespace system_solutions_l2593_259320

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x / (x - a) + y / (y - b) = 2 ∧ a * x + b * y = 2 * a * b

-- Theorem statement
theorem system_solutions (a b : ℝ) :
  (∀ x y : ℝ, system x y a b → x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b)) ∨
  (a = b ∧ ∀ x y : ℝ, system x y a b → x + y = 2 * a) ∨
  (a = -b ∧ ¬∃ x y : ℝ, system x y a b) :=
by sorry

end system_solutions_l2593_259320


namespace unique_solution_cube_difference_l2593_259373

theorem unique_solution_cube_difference (n m : ℤ) : 
  (n + 2)^4 - n^4 = m^3 ↔ n = -1 ∧ m = 0 := by
  sorry

end unique_solution_cube_difference_l2593_259373


namespace f_at_2_l2593_259367

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8) 
  (h2 : f (-2) = 10) : f 2 = -26 := by
  sorry

end f_at_2_l2593_259367


namespace inequality_proof_l2593_259380

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^2 * (c + a)) + 1 / (c^2 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l2593_259380


namespace triplets_shirts_l2593_259349

/-- The number of shirts Hazel, Razel, and Gazel have in total -/
def total_shirts (hazel razel gazel : ℕ) : ℕ := hazel + razel + gazel

/-- Theorem stating the total number of shirts given the conditions -/
theorem triplets_shirts : 
  ∀ (hazel razel gazel : ℕ),
  hazel = 6 →
  razel = 2 * hazel →
  gazel = razel / 2 - 1 →
  total_shirts hazel razel gazel = 23 := by
sorry

end triplets_shirts_l2593_259349


namespace fraction_unchanged_l2593_259329

theorem fraction_unchanged (x y : ℝ) (h : y ≠ 0) :
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) := by sorry

end fraction_unchanged_l2593_259329


namespace area_between_concentric_circles_l2593_259379

/-- The area of the region between two concentric circles, where the diameter of the larger circle
    is twice the diameter of the smaller circle, and the smaller circle has a diameter of 4 units,
    is equal to 12π square units. -/
theorem area_between_concentric_circles (π : ℝ) : 
  let d_small : ℝ := 4
  let r_small : ℝ := d_small / 2
  let r_large : ℝ := 2 * r_small
  let area_small : ℝ := π * r_small^2
  let area_large : ℝ := π * r_large^2
  area_large - area_small = 12 * π :=
by sorry

end area_between_concentric_circles_l2593_259379


namespace tangent_squares_roots_l2593_259311

theorem tangent_squares_roots : ∃ (a b c : ℝ),
  a + b + c = 33 ∧
  a * b * c = 33 ∧
  a * b + b * c + c * a = 27 ∧
  ∀ (x : ℝ), x^3 - 33*x^2 + 27*x - 33 = 0 ↔ (x = a ∨ x = b ∨ x = c) := by
  sorry

end tangent_squares_roots_l2593_259311


namespace helen_cookies_baked_this_morning_l2593_259304

theorem helen_cookies_baked_this_morning (total_cookies : ℕ) (yesterday_cookies : ℕ) 
  (h1 : total_cookies = 574)
  (h2 : yesterday_cookies = 435) :
  total_cookies - yesterday_cookies = 139 := by
sorry

end helen_cookies_baked_this_morning_l2593_259304


namespace equation_implication_l2593_259323

theorem equation_implication (x y : ℝ) 
  (h1 : x^2 - 3*x*y + 2*y^2 + x - y = 0)
  (h2 : x^2 - 2*x*y + y^2 - 5*x + 2*y = 0) :
  x*y - 12*x + 15*y = 0 := by
sorry

end equation_implication_l2593_259323


namespace fibonacci_property_l2593_259310

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_property :
  let a := fibonacci
  (a 0 * a 2 + a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7) -
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 + a 6^2) = 0 := by
  sorry

end fibonacci_property_l2593_259310


namespace homework_problem_distribution_l2593_259381

theorem homework_problem_distribution (total : ℕ) 
  (multiple_choice free_response true_false : ℕ) : 
  total = 45 → 
  multiple_choice = 2 * free_response → 
  free_response = true_false + 7 → 
  total = multiple_choice + free_response + true_false → 
  true_false = 6 := by
  sorry

end homework_problem_distribution_l2593_259381


namespace travis_cereal_consumption_l2593_259333

/-- Represents the number of boxes of cereal Travis eats per week -/
def boxes_per_week : ℕ := sorry

/-- The cost of one box of cereal in dollars -/
def cost_per_box : ℚ := 3

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Travis spends on cereal in a year in dollars -/
def total_spent : ℚ := 312

theorem travis_cereal_consumption :
  boxes_per_week = 2 ∧
  cost_per_box * boxes_per_week * weeks_in_year = total_spent :=
by sorry

end travis_cereal_consumption_l2593_259333


namespace pyramid_display_rows_l2593_259314

/-- Represents the number of cans in a pyramid display. -/
def pyramid_display (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that a pyramid display with 210 cans has 14 rows. -/
theorem pyramid_display_rows :
  ∃ (n : ℕ), pyramid_display n = 210 ∧ n = 14 := by
  sorry

end pyramid_display_rows_l2593_259314


namespace max_quad_area_l2593_259315

/-- The ellipse defined by x²/8 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the ellipse -/
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- Segment AB passes through the center of the ellipse -/
def segment_through_center (a b : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral F₁AF₂B -/
def quad_area (a b : ℝ × ℝ) : ℝ := sorry

theorem max_quad_area :
  ∀ (a b : ℝ × ℝ),
    point_on_ellipse a →
    point_on_ellipse b →
    segment_through_center a b →
    quad_area a b ≤ 8 :=
sorry

end max_quad_area_l2593_259315


namespace households_with_only_bike_l2593_259360

theorem households_with_only_bike 
  (total : Nat) 
  (neither : Nat) 
  (both : Nat) 
  (with_car : Nat) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
sorry

end households_with_only_bike_l2593_259360


namespace q_investment_time_l2593_259399

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.time_p) / (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the problem conditions, Q's investment time is 20 months -/
theorem q_investment_time (data : PartnershipData)
  (h1 : data.investment_ratio_p = 7)
  (h2 : data.investment_ratio_q = 5)
  (h3 : data.profit_ratio_p = 7)
  (h4 : data.profit_ratio_q = 10)
  (h5 : data.time_p = 10) :
  calculate_time_q data = 20 := by
  sorry

end q_investment_time_l2593_259399


namespace binomial_coefficient_7_4_l2593_259357

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_coefficient_7_4_l2593_259357


namespace sharons_harvest_l2593_259365

theorem sharons_harvest (greg_harvest : ℝ) (difference : ℝ) (sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : greg_harvest = sharon_harvest + difference)
  (h3 : difference = 0.3) :
  sharon_harvest = 0.1 := by
sorry

end sharons_harvest_l2593_259365


namespace sum_of_roots_arithmetic_sequence_l2593_259383

theorem sum_of_roots_arithmetic_sequence (a b c d : ℝ) : 
  0 < c ∧ 0 < b ∧ 0 < a ∧ 
  a > b ∧ b > c ∧ 
  b = a - d ∧ c = a - 2*d ∧ 
  0 < d ∧
  (b^2 - 4*a*c > 0) →
  -(b / a) = -1/3 := by
sorry

end sum_of_roots_arithmetic_sequence_l2593_259383


namespace steven_owes_jeremy_l2593_259356

theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) (amount_owed : ℚ) : 
  rate = 9/4 → rooms = 8/5 → amount_owed = rate * rooms → amount_owed = 18/5 := by
  sorry

end steven_owes_jeremy_l2593_259356


namespace osmanthus_price_is_300_l2593_259306

/-- The unit price of osmanthus trees given the following conditions:
  - Total amount raised is 7000 yuan
  - Total number of trees is 30
  - Cost of osmanthus trees is 3000 yuan
  - Unit price of osmanthus trees is 50% higher than cherry trees
-/
def osmanthus_price : ℝ :=
  let total_amount : ℝ := 7000
  let total_trees : ℝ := 30
  let osmanthus_cost : ℝ := 3000
  let price_ratio : ℝ := 1.5
  300

theorem osmanthus_price_is_300 :
  osmanthus_price = 300 := by sorry

end osmanthus_price_is_300_l2593_259306


namespace triangle_abc_properties_l2593_259374

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) 
  (h5 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0) 
  (h6 : a = Real.sqrt 13) (h7 : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) : 
  A = π/3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end triangle_abc_properties_l2593_259374


namespace pyramid_on_cylinder_radius_l2593_259301

/-- A regular square pyramid with all edges equal to 1 -/
structure RegularSquarePyramid where
  edge_length : ℝ
  edge_equal : edge_length = 1

/-- An infinite right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Predicate to check if all vertices of the pyramid lie on the lateral surface of the cylinder -/
def vertices_on_cylinder (p : RegularSquarePyramid) (c : RightCircularCylinder) : Prop :=
  sorry

/-- The main theorem stating the possible values of the cylinder's radius -/
theorem pyramid_on_cylinder_radius (p : RegularSquarePyramid) (c : RightCircularCylinder) :
  vertices_on_cylinder p c → (c.radius = 3 / (4 * Real.sqrt 2) ∨ c.radius = 1 / Real.sqrt 3) :=
sorry

end pyramid_on_cylinder_radius_l2593_259301


namespace triangle_identity_l2593_259362

/-- The triangle operation on pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x, y) = (1, 0) := by sorry

end triangle_identity_l2593_259362


namespace smallest_n_for_Q_less_than_threshold_l2593_259370

def Q (n : ℕ) : ℚ := (2^(n-1) : ℚ) / (n.factorial * (2*n + 1))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 10 → Q k ≥ 1/5000 ∧ Q 10 < 1/5000 := by sorry

end smallest_n_for_Q_less_than_threshold_l2593_259370


namespace infinite_non_fractional_numbers_l2593_259300

/-- A number is p-good if it cannot be expressed as p^x * (p^(yz) - 1) / (p^y - 1) for any nonnegative integers x, y, z -/
def IsPGood (n : ℕ) (p : ℕ) : Prop :=
  ∀ x y z : ℕ, n ≠ p^x * (p^(y*z) - 1) / (p^y - 1)

/-- The set of numbers that cannot be expressed as (p^a - p^b) / (p^c - p^d) for any prime p and integers a, b, c, d -/
def NonFractionalSet : Set ℕ :=
  {n : ℕ | ∀ p : ℕ, Prime p → IsPGood n p}

theorem infinite_non_fractional_numbers : Set.Infinite NonFractionalSet := by
  sorry

end infinite_non_fractional_numbers_l2593_259300


namespace bus_count_l2593_259328

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end bus_count_l2593_259328


namespace problem1_simplification_l2593_259327

theorem problem1_simplification (x y : ℝ) : 
  y * (4 * x - 3 * y) + (x - 2 * y)^2 = x^2 + y^2 := by sorry

end problem1_simplification_l2593_259327


namespace fraction_subtraction_l2593_259318

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end fraction_subtraction_l2593_259318


namespace geometric_sequence_sum_l2593_259390

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by sorry

end geometric_sequence_sum_l2593_259390


namespace king_can_equalize_l2593_259308

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → ℤ

/-- Represents a sequence of king's moves --/
def KingPath := List Square

/-- Checks if a move between two squares is valid for a king --/
def isValidKingMove (s1 s2 : Square) : Prop :=
  (abs (s1.row - s2.row) ≤ 1) ∧ (abs (s1.col - s2.col) ≤ 1)

/-- Applies a sequence of king's moves to a chessboard --/
def applyMoves (board : Chessboard) (path : KingPath) : Chessboard :=
  sorry

/-- The main theorem --/
theorem king_can_equalize (initial : Chessboard) :
  ∃ (path : KingPath), ∀ (s1 s2 : Square), (applyMoves initial path s1) = (applyMoves initial path s2) :=
sorry

end king_can_equalize_l2593_259308


namespace abc_equals_314_l2593_259351

/-- Represents a base-5 number with two digits -/
def BaseFiveNumber (tens : Nat) (ones : Nat) : Nat :=
  5 * tens + ones

/-- Proposition: Given the conditions, ABC = 314 -/
theorem abc_equals_314 
  (A B C : Nat) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A < 5 ∧ B < 5 ∧ C < 5)
  (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h4 : BaseFiveNumber A B + C = BaseFiveNumber C 0)
  (h5 : BaseFiveNumber A B + BaseFiveNumber B A = BaseFiveNumber C C) :
  100 * A + 10 * B + C = 314 :=
sorry

end abc_equals_314_l2593_259351


namespace specific_plate_probability_l2593_259307

/-- Represents the set of vowels used in license plates -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- Represents the set of non-vowel letters used in license plates -/
def NonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'}

/-- Represents the set of even digits used in license plates -/
def EvenDigits : Finset Char := {'0', '2', '4', '6', '8'}

/-- Represents a license plate -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char
  h1 : first ∈ Vowels
  h2 : second ∈ Vowels
  h3 : first ≠ second
  h4 : third ∈ NonVowels
  h5 : fourth ∈ NonVowels
  h6 : third ≠ fourth
  h7 : fifth ∈ EvenDigits

/-- The probability of a specific license plate occurring -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (Vowels.card * (Vowels.card - 1) * NonVowels.card * (NonVowels.card - 1) * EvenDigits.card)

theorem specific_plate_probability :
  ∃ (plate : LicensePlate), licensePlateProbability plate = 1 / 50600 :=
sorry

end specific_plate_probability_l2593_259307


namespace parabola_c_value_l2593_259387

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- Checks if (h, k) is the vertex of the parabola -/
def Parabola.hasVertex (p : Parabola) (h k : ℝ) : Prop :=
  h = p.a * k^2 + p.b * k + p.c ∧
  ∀ y, p.a * y^2 + p.b * y + p.c ≤ h

/-- States that the parabola opens downwards -/
def Parabola.opensDownwards (p : Parabola) : Prop :=
  p.a < 0

theorem parabola_c_value
  (p : Parabola)
  (vertex : p.hasVertex 5 3)
  (point : p.contains 7 6)
  (down : p.opensDownwards) :
  p.c = 7 := by
  sorry

end parabola_c_value_l2593_259387


namespace boat_speed_in_still_water_l2593_259336

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water (along_stream speed_against_stream : ℝ) 
  (h1 : along_stream = 15)
  (h2 : speed_against_stream = 5) :
  (along_stream + speed_against_stream) / 2 = 10 := by
  sorry

end boat_speed_in_still_water_l2593_259336


namespace quadratic_properties_l2593_259334

/-- The quadratic function f(x) = 2x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 3

theorem quadratic_properties :
  (∀ x y : ℝ, y = f x → y > f (-1) → x ≠ -1) ∧ 
  (f (-1) = -5) ∧
  (∀ x : ℝ, -2 ≤ x → x ≤ 1 → -5 ≤ f x ∧ f x ≤ 3) ∧
  (∀ x y : ℝ, y = 2 * (x - 1)^2 - 4 ↔ y = f (x - 2) + 1) :=
by sorry


end quadratic_properties_l2593_259334


namespace inequality_implies_identity_or_negation_l2593_259335

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)

/-- The main theorem stating that a function satisfying the inequality
    must be either the identity function or its negation -/
theorem inequality_implies_identity_or_negation (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end inequality_implies_identity_or_negation_l2593_259335


namespace tan_five_pi_fourth_l2593_259375

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_fourth_l2593_259375


namespace social_media_weekly_time_l2593_259388

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 8 hours daily phone usage with half on social media, weekly social media time is 28 hours -/
theorem social_media_weekly_time : 
  weekly_social_media_time 8 0.5 = 28 := by
  sorry


end social_media_weekly_time_l2593_259388


namespace binary_of_89_l2593_259353

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Theorem: The binary representation of 89 is [true, false, true, true, false, false, true] -/
theorem binary_of_89 :
  toBinary 89 = [true, false, true, true, false, false, true] := by
  sorry

#eval toBinary 89

end binary_of_89_l2593_259353


namespace counterexample_exists_l2593_259343

theorem counterexample_exists : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^3 + b^3 < 2 * a * b^2 := by
  sorry

end counterexample_exists_l2593_259343


namespace fiftieth_ring_squares_l2593_259305

/-- Represents the number of unit squares in the nth ring of the described square array. -/
def ring_squares (n : ℕ) : ℕ := 32 * n - 16

/-- The theorem states that the 50th ring contains 1584 unit squares. -/
theorem fiftieth_ring_squares : ring_squares 50 = 1584 := by
  sorry

end fiftieth_ring_squares_l2593_259305


namespace parallelogram_area_l2593_259394

/-- The area of a parallelogram with given properties -/
theorem parallelogram_area (s2 : ℝ) (a : ℝ) (h_s2_pos : s2 > 0) (h_a_pos : a > 0) (h_a_lt_180 : a < 180) :
  let s1 := 2 * s2
  let θ := a * π / 180
  2 * s2^2 * Real.sin θ = s1 * s2 * Real.sin θ :=
by sorry

end parallelogram_area_l2593_259394


namespace sum_of_three_squares_l2593_259364

theorem sum_of_three_squares (s t : ℚ) 
  (h1 : 3 * t + 2 * s = 27)
  (h2 : 2 * t + 3 * s = 25) :
  3 * s = 63 / 5 := by
sorry

end sum_of_three_squares_l2593_259364


namespace only_B_on_line_l2593_259313

-- Define the points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (-2, 1)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (2, -9)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem only_B_on_line :
  line_equation B.1 B.2 ∧
  ¬line_equation A.1 A.2 ∧
  ¬line_equation C.1 C.2 ∧
  ¬line_equation D.1 D.2 := by
  sorry

end only_B_on_line_l2593_259313


namespace range_of_m_max_min_distance_exists_line_l_l2593_259366

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - m = 0

-- Define point A
def point_A (m : ℝ) : ℝ × ℝ := (m, -2)

-- Define the condition for a point to be inside the circle
def inside_circle (x y m : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 < 5 + m

-- Theorem 1
theorem range_of_m (m : ℝ) : 
  (∃ x y, circle_C x y m ∧ inside_circle m (-2) m) → -1 < m ∧ m < 4 :=
sorry

-- Theorem 2
theorem max_min_distance (x y : ℝ) :
  circle_C x y 4 → 4 ≤ (x - 4)^2 + (y - 2)^2 ∧ (x - 4)^2 + (y - 2)^2 ≤ 64 :=
sorry

-- Define the line l
def line_l (k b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Theorem 3
theorem exists_line_l :
  ∃ b, (b = -4 ∨ b = 1) ∧
    ∃ x₁ y₁ x₂ y₂, 
      circle_C x₁ y₁ 4 ∧ circle_C x₂ y₂ 4 ∧
      line_l 1 b x₁ y₁ ∧ line_l 1 b x₂ y₂ ∧
      (x₁ + x₂ = 0) ∧ (y₁ + y₂ = 0) :=
sorry

end range_of_m_max_min_distance_exists_line_l_l2593_259366


namespace solution_set_inequality_l2593_259391

theorem solution_set_inequality (x : ℝ) :
  (2*x - 3) * (x + 1) < 0 ↔ -1 < x ∧ x < 3/2 := by sorry

end solution_set_inequality_l2593_259391


namespace max_four_digit_prime_product_l2593_259321

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem max_four_digit_prime_product :
  ∃ (m x y z : Nat),
    isPrime x ∧ isPrime y ∧ isPrime z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isPrime (10 * x + y) ∧
    isPrime (10 * z + x) ∧
    m = x * y * (10 * x + y) ∧
    m ≥ 1000 ∧ m < 10000 ∧
    (∀ (m' x' y' z' : Nat),
      isPrime x' ∧ isPrime y' ∧ isPrime z' ∧
      x' < 10 ∧ y' < 10 ∧ z' < 10 ∧
      x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      isPrime (10 * x' + y') ∧
      isPrime (10 * z' + x') ∧
      m' = x' * y' * (10 * x' + y') ∧
      m' ≥ 1000 ∧ m' < 10000 →
      m' ≤ m) ∧
    m = 1533 :=
by sorry

end max_four_digit_prime_product_l2593_259321


namespace savings_calculation_l2593_259338

/-- Calculates the total savings for a year given monthly expenses and average monthly income -/
def yearly_savings (expense1 expense2 expense3 : ℕ) (months1 months2 months3 : ℕ) (avg_income : ℕ) : ℕ :=
  let total_expense := expense1 * months1 + expense2 * months2 + expense3 * months3
  let total_income := avg_income * 12
  total_income - total_expense

/-- Proves that the yearly savings is 5200 given the specific expenses and income -/
theorem savings_calculation : yearly_savings 1700 1550 1800 3 4 5 2125 = 5200 := by
  sorry

#eval yearly_savings 1700 1550 1800 3 4 5 2125

end savings_calculation_l2593_259338


namespace triangle_count_difference_l2593_259385

/-- The number of distinct, incongruent, integer-sided triangles with perimeter n -/
def t (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem triangle_count_difference (n : ℕ) (h : n ≥ 3) :
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋) ∨
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋ + 1) :=
sorry

end triangle_count_difference_l2593_259385


namespace probability_at_least_one_head_three_coins_l2593_259342

theorem probability_at_least_one_head_three_coins :
  let p_head : ℝ := 1 / 2
  let p_tail : ℝ := 1 - p_head
  let p_three_tails : ℝ := p_tail ^ 3
  let p_at_least_one_head : ℝ := 1 - p_three_tails
  p_at_least_one_head = 7 / 8 := by
sorry

end probability_at_least_one_head_three_coins_l2593_259342


namespace weightlifting_time_l2593_259368

def practice_duration : ℕ := 120  -- 2 hours in minutes

theorem weightlifting_time (shooting_time running_time weightlifting_time : ℕ) :
  shooting_time = practice_duration / 2 →
  running_time + weightlifting_time = practice_duration - shooting_time →
  running_time = 2 * weightlifting_time →
  weightlifting_time = 20 := by
  sorry

end weightlifting_time_l2593_259368


namespace watermelon_banana_weights_l2593_259377

theorem watermelon_banana_weights :
  ∀ (watermelon_weight banana_weight : ℕ),
    2 * watermelon_weight + banana_weight = 8100 →
    2 * watermelon_weight + 3 * banana_weight = 8300 →
    watermelon_weight = 4000 ∧ banana_weight = 100 := by
  sorry

end watermelon_banana_weights_l2593_259377


namespace tan_alpha_value_l2593_259392

theorem tan_alpha_value (α : Real) (h : Real.tan (π/4 - α) = 1/5) : Real.tan α = 2/3 := by
  sorry

end tan_alpha_value_l2593_259392


namespace least_n_with_gcd_conditions_l2593_259302

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 30 (n + 80) = 15 ∧ 
  Nat.gcd (n + 30) 100 = 50 ∧
  (∀ m : ℕ, m > 1000 → 
    (Nat.gcd 30 (m + 80) = 15 ∧ Nat.gcd (m + 30) 100 = 50) → 
    m ≥ n) ∧
  n = 1270 :=
sorry

end least_n_with_gcd_conditions_l2593_259302


namespace sphere_radius_with_inscribed_box_l2593_259397

theorem sphere_radius_with_inscribed_box (x y z r : ℝ) : 
  x > 0 → y > 0 → z > 0 → r > 0 →
  2 * (x * y + y * z + x * z) = 384 →
  4 * (x + y + z) = 112 →
  (2 * r) ^ 2 = x ^ 2 + y ^ 2 + z ^ 2 →
  r = 10 :=
by sorry

end sphere_radius_with_inscribed_box_l2593_259397


namespace odd_divisors_iff_perfect_square_l2593_259340

/-- A number is a perfect square if and only if it has an odd number of divisors -/
theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  Odd (Nat.card {d : ℕ | d ∣ n}) ↔ ∃ k : ℕ, n = k^2 := by
  sorry


end odd_divisors_iff_perfect_square_l2593_259340


namespace convention_handshakes_l2593_259382

/-- The number of handshakes in a convention with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes in the specific convention scenario is 160 --/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end convention_handshakes_l2593_259382


namespace min_omega_value_l2593_259341

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  (∀ t > 0, f (t + T) = f t) →
  (∀ t > T, ∃ s ∈ Set.Ioo 0 T, f t = f s) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' > 0, (∀ x, Real.cos (ω' * x + φ) = f x) → ω ≤ ω' :=
by sorry

end min_omega_value_l2593_259341


namespace fifth_term_value_l2593_259393

theorem fifth_term_value (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n, S n = 2 * n^2 + 3 * n - 1)
  (h2 : a 5 = S 5 - S 4) : 
  a 5 = 21 := by
  sorry

end fifth_term_value_l2593_259393


namespace dan_buys_five_dozens_l2593_259324

/-- The number of golf balls in one dozen -/
def balls_per_dozen : ℕ := 12

/-- The total number of golf balls purchased -/
def total_balls : ℕ := 132

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- Theorem stating that Dan buys 5 dozens of golf balls -/
theorem dan_buys_five_dozens :
  (total_balls - gus_dozens * balls_per_dozen - chris_balls) / balls_per_dozen = 5 :=
sorry

end dan_buys_five_dozens_l2593_259324


namespace sports_club_members_l2593_259350

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 48)
  (hT : T = 46)
  (hBoth : Both = 21)
  (hNeither : Neither = 7) :
  (B + T) - Both + Neither = 80 := by
  sorry

end sports_club_members_l2593_259350


namespace james_total_toys_l2593_259316

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by
  sorry

end james_total_toys_l2593_259316


namespace unique_losses_l2593_259389

/-- Represents a participant in the badminton tournament -/
structure Participant where
  id : Fin 16
  gamesWon : Nat
  gamesLost : Nat

/-- The set of all participants in the tournament -/
def Tournament := Fin 16 → Participant

theorem unique_losses (t : Tournament) : 
  (∀ i j : Fin 16, i ≠ j → (t i).gamesWon ≠ (t j).gamesWon) →
  (∀ i : Fin 16, (t i).gamesWon + (t i).gamesLost = 15) →
  (∀ i : Fin 16, (t i).gamesWon < 16) →
  (∀ i j : Fin 16, i ≠ j → (t i).gamesLost ≠ (t j).gamesLost) :=
by sorry

end unique_losses_l2593_259389


namespace planes_not_parallel_l2593_259372

/-- Represents a 3D vector --/
structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space --/
structure Plane where
  normal : Vec3

/-- Check if two planes are parallel --/
def are_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.normal = Vec3.mk (k * p2.normal.x) (k * p2.normal.y) (k * p2.normal.z)

theorem planes_not_parallel : ¬ (are_parallel 
  (Plane.mk (Vec3.mk 0 1 3)) 
  (Plane.mk (Vec3.mk 1 0 3))) := by
  sorry

#check planes_not_parallel

end planes_not_parallel_l2593_259372


namespace scooter_selling_price_l2593_259395

/-- Calculates the selling price of a scooter given initial costs and gain percent -/
theorem scooter_selling_price
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (gain_percent : ℝ)
  (h1 : purchase_price = 4700)
  (h2 : repair_cost = 600)
  (h3 : gain_percent = 9.433962264150944)
  : ∃ (selling_price : ℝ), selling_price = 5800 := by
  sorry

end scooter_selling_price_l2593_259395


namespace reeya_average_score_l2593_259361

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score :
  let total := reeya_scores.sum
  let count := reeya_scores.length
  total / count = 76.6 := by sorry

end reeya_average_score_l2593_259361


namespace root_product_theorem_l2593_259344

theorem root_product_theorem (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  (r = 16/3) := by
sorry

end root_product_theorem_l2593_259344


namespace rectangle_circle_area_ratio_l2593_259331

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end rectangle_circle_area_ratio_l2593_259331


namespace simplify_and_evaluate_l2593_259371

theorem simplify_and_evaluate (a : ℚ) : 
  let b : ℚ := -1/3
  (a + b)^2 - a * (2*b + a) = 1/9 := by
  sorry

end simplify_and_evaluate_l2593_259371


namespace reciprocal_sum_geometric_progression_l2593_259303

theorem reciprocal_sum_geometric_progression 
  (q : ℝ) (n : ℕ) (S : ℝ) (h1 : q ≠ 1) :
  let a := 3
  let r := q^2
  let original_sum := a * (1 - r^(2*n)) / (1 - r)
  let reciprocal_sum := (1/a) * (1 - (1/r)^(2*n)) / (1 - 1/r)
  S = original_sum →
  reciprocal_sum = 1/S :=
by sorry

end reciprocal_sum_geometric_progression_l2593_259303


namespace minutes_after_midnight_theorem_l2593_259376

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2021, month := 1, day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 1453

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 0, minute := 13 }

theorem minutes_after_midnight_theorem :
  addMinutes startTime minutesToAdd = expectedResult :=
sorry

end minutes_after_midnight_theorem_l2593_259376


namespace function_difference_bound_l2593_259322

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ) 
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x y : Set.Icc 0 1), x ≠ y → |f x - f y| < |x.val - y.val|) :
  ∀ (x y : Set.Icc 0 1), |f x - f y| < (1/2 : ℝ) := by
  sorry

end function_difference_bound_l2593_259322


namespace inequality_solution_l2593_259309

theorem inequality_solution (p x y : ℝ) : 
  p = x + y → x^2 + 4*y^2 + 8*y + 4 ≤ 4*x → -3 - Real.sqrt 5 ≤ p ∧ p ≤ -3 + Real.sqrt 5 :=
by sorry

end inequality_solution_l2593_259309


namespace exercise_book_distribution_l2593_259337

theorem exercise_book_distribution (total_books : ℕ) (num_classes : ℕ) 
  (h1 : total_books = 338) (h2 : num_classes = 3) :
  ∃ (books_per_class : ℕ) (books_left : ℕ),
    books_per_class = 112 ∧ 
    books_left = 2 ∧
    total_books = books_per_class * num_classes + books_left :=
by sorry

end exercise_book_distribution_l2593_259337


namespace arithmetic_sequence_minimum_value_l2593_259355

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 16) / (a_n + 3) is 4,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem arithmetic_sequence_minimum_value (d : ℝ) (n : ℕ) :
  d ≠ 0 →
  let a : ℕ → ℝ := λ k => 1 + (k - 1) * d
  let S : ℕ → ℝ := λ k => k * (a 1 + a k) / 2
  (a 3)^2 = (a 1) * (a 13) →
  (∀ k : ℕ, (2 * S k + 16) / (a k + 3) ≥ 4) ∧
  (∃ k : ℕ, (2 * S k + 16) / (a k + 3) = 4) :=
by sorry

end arithmetic_sequence_minimum_value_l2593_259355


namespace parabola_triangle_area_l2593_259352

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Point of intersection between a line and a parabola -/
def intersection (p : Parabola) (l : Line) : ℝ × ℝ := sorry

/-- Foot of the perpendicular from a point to a line -/
def perpendicularFoot (point : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_triangle_area 
  (p : Parabola) 
  (l : Line) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : p.directrix = fun x y => x = -1)
  (h4 : l.point = (1, 0))
  (h5 : l.slope = Real.sqrt 3)
  (h6 : (intersection p l).2 > 0) :
  let A := intersection p l
  let K := perpendicularFoot A p.directrix
  triangleArea A K p.focus = 4 * Real.sqrt 3 := by sorry

end parabola_triangle_area_l2593_259352


namespace inequality_preservation_l2593_259363

theorem inequality_preservation (x y : ℝ) (h : x > y) : x + 5 > y + 5 := by
  sorry

end inequality_preservation_l2593_259363


namespace box_volume_increase_l2593_259354

/-- Given a rectangular box with dimensions l, w, h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a specific new volume -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5184)
  (hs : 2 * (l * w + w * h + h * l) = 1944)
  (he : 4 * (l + w + h) = 216) :
  (l + 2) * (w + 2) * (h + 2) = 7352 := by
  sorry

end box_volume_increase_l2593_259354
