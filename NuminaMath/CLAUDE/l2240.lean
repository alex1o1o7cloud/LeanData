import Mathlib

namespace NUMINAMATH_CALUDE_base5_sum_equality_l2240_224020

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base-5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_equality :
  addBase5 (toBase5 122) (toBase5 78) = toBase5 200 :=
sorry

end NUMINAMATH_CALUDE_base5_sum_equality_l2240_224020


namespace NUMINAMATH_CALUDE_election_votes_proof_l2240_224060

theorem election_votes_proof (V : ℕ) (W L : ℕ) : 
  (W > L) →  -- Winner has more votes than loser
  (W - L = (V : ℚ) * (1 / 5)) →  -- Winner's margin is 20% of total votes
  ((L + 1000) - (W - 1000) = (V : ℚ) * (1 / 5)) →  -- Loser would win by 20% if 1000 votes change
  V = 5000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_proof_l2240_224060


namespace NUMINAMATH_CALUDE_proportionality_analysis_l2240_224006

/-- Represents a relationship between x and y -/
inductive Relationship
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the relationship between x and y given an equation -/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x + y = 0 -/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- Equation B: 3xy = 10 -/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- Equation C: x = 5y -/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- Equation D: x^2 + 3x + y = 10 -/
def equationD (x y : ℝ) : Prop := x^2 + 3*x + y = 10

/-- Equation E: x/y = √3 -/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem proportionality_analysis :
  (determineRelationship equationA = Relationship.DirectlyProportional) ∧
  (determineRelationship equationB = Relationship.InverselyProportional) ∧
  (determineRelationship equationC = Relationship.DirectlyProportional) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.DirectlyProportional) :=
by
  sorry

end NUMINAMATH_CALUDE_proportionality_analysis_l2240_224006


namespace NUMINAMATH_CALUDE_sphere_cube_paint_equivalence_l2240_224041

theorem sphere_cube_paint_equivalence (M : ℝ) : 
  let cube_side : ℝ := 3
  let cube_surface_area : ℝ := 6 * cube_side^2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_volume : ℝ := (M * Real.sqrt 3) / Real.sqrt Real.pi
  (∃ (r : ℝ), 
    sphere_surface_area = 4 * Real.pi * r^2 ∧ 
    sphere_volume = (4 / 3) * Real.pi * r^3) →
  M = 36 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_paint_equivalence_l2240_224041


namespace NUMINAMATH_CALUDE_shelves_needed_l2240_224023

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 46)
  (h2 : books_taken = 10)
  (h3 : books_per_shelf = 4) :
  (total_books - books_taken) / books_per_shelf = 9 :=
by sorry

end NUMINAMATH_CALUDE_shelves_needed_l2240_224023


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2240_224068

theorem quadratic_inequality_solution (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, (a * x^2 + b * x + 2 < 0) ↔ (x < -1/2 ∨ x > 1/3)) →
  (a - b) / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2240_224068


namespace NUMINAMATH_CALUDE_prob_red_ball_specific_bag_l2240_224055

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem stating the probability of drawing a red ball from a specific bag -/
theorem prob_red_ball_specific_bag :
  let bag : ColoredBalls := { total := 9, red := 6, green := 3 }
  prob_red_ball bag = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_prob_red_ball_specific_bag_l2240_224055


namespace NUMINAMATH_CALUDE_x_value_l2240_224054

theorem x_value : ∃ x : ℝ, 
  ((x * (9^2)) / ((8^2) * (3^5)) = 0.16666666666666666) ∧ 
  (x = 5.333333333333333) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2240_224054


namespace NUMINAMATH_CALUDE_jerry_age_l2240_224056

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 18 → 
  mickey_age = 2 * jerry_age - 2 → 
  jerry_age = 10 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l2240_224056


namespace NUMINAMATH_CALUDE_simplify_expression_l2240_224095

theorem simplify_expression (a b c : ℝ) (h : a * b ≠ c^2) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - c^2) = a / b + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2240_224095


namespace NUMINAMATH_CALUDE_square_field_area_l2240_224004

/-- Given a square field with barbed wire drawn around it, if the total cost of the wire
    at a specific rate per meter is a certain amount, then we can determine the area of the field. -/
theorem square_field_area (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  wire_cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  ∃ (side_length : ℝ), 
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l2240_224004


namespace NUMINAMATH_CALUDE_equation_solution_l2240_224080

theorem equation_solution (x y : ℝ) 
  (hx0 : x ≠ 0) (hx3 : x ≠ 3) (hy0 : y ≠ 0) (hy4 : y ≠ 4)
  (h_eq : 3 / x + 2 / y = 5 / 6) :
  x = 18 * y / (5 * y - 12) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2240_224080


namespace NUMINAMATH_CALUDE_prove_x_value_l2240_224030

theorem prove_x_value (x a b c d : ℕ) 
  (h1 : x = a + 7)
  (h2 : a = b + 12)
  (h3 : b = c + 15)
  (h4 : c = d + 25)
  (h5 : d = 95) : x = 154 := by
  sorry

end NUMINAMATH_CALUDE_prove_x_value_l2240_224030


namespace NUMINAMATH_CALUDE_power_equality_l2240_224046

theorem power_equality (K : ℕ) : 32^2 * 4^4 = 2^K → K = 18 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 4 = 2^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_equality_l2240_224046


namespace NUMINAMATH_CALUDE_f_of_two_equals_five_l2240_224069

/-- Given a function f(x) = x^2 + 2x - 3, prove that f(2) = 5 -/
theorem f_of_two_equals_five (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x - 3) : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_five_l2240_224069


namespace NUMINAMATH_CALUDE_power_negative_multiply_l2240_224031

theorem power_negative_multiply (m : ℝ) : (-m)^2 * m^5 = m^7 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_multiply_l2240_224031


namespace NUMINAMATH_CALUDE_existence_implies_lower_bound_l2240_224077

theorem existence_implies_lower_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a ≤ a*x - 3) → a ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_lower_bound_l2240_224077


namespace NUMINAMATH_CALUDE_rectangle_area_l2240_224047

/-- The area of a rectangle with length 1.2 meters and width 0.5 meters is 0.6 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.2
  let width : ℝ := 0.5
  length * width = 0.6 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2240_224047


namespace NUMINAMATH_CALUDE_inverse_sum_equals_golden_ratio_minus_one_l2240_224022

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2 - x else x^3 - 2*x^2 + x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≤ 0 then (1 + Real.sqrt 5) / 2
  else if y = 1 then 1
  else -2

-- Theorem statement
theorem inverse_sum_equals_golden_ratio_minus_one :
  f_inv (-1) + f_inv 1 + f_inv 4 = (Real.sqrt 5 - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_golden_ratio_minus_one_l2240_224022


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2240_224067

/-- The slope of a chord in an ellipse with given midpoint -/
theorem ellipse_chord_slope (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) →  -- ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / 16 + y₁^2 / 9 = 1 ∧  -- endpoint 1 on ellipse
    x₂^2 / 16 + y₂^2 / 9 = 1 ∧  -- endpoint 2 on ellipse
    (x₁ + x₂) / 2 = -1 ∧        -- x-coordinate of midpoint
    (y₁ + y₂) / 2 = 2 ∧         -- y-coordinate of midpoint
    (y₂ - y₁) / (x₂ - x₁) = 9 / 32) -- slope of chord
  := by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2240_224067


namespace NUMINAMATH_CALUDE_kates_remaining_money_is_7_80_l2240_224001

/-- Calculates the amount of money Kate has left after her savings and expenses --/
def kates_remaining_money (march_savings april_savings may_savings june_savings : ℚ)
  (keyboard_cost mouse_cost headset_cost video_game_cost : ℚ)
  (book_cost : ℚ)
  (euro_to_dollar pound_to_dollar : ℚ) : ℚ :=
  let total_savings := march_savings + april_savings + may_savings + june_savings + 2 * april_savings
  let euro_expenses := (keyboard_cost + mouse_cost + headset_cost + video_game_cost) * euro_to_dollar
  let pound_expenses := book_cost * pound_to_dollar
  total_savings - euro_expenses - pound_expenses

/-- Theorem stating that Kate has $7.80 left after her savings and expenses --/
theorem kates_remaining_money_is_7_80 :
  kates_remaining_money 27 13 28 35 42 4 16 25 12 1.2 1.4 = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_kates_remaining_money_is_7_80_l2240_224001


namespace NUMINAMATH_CALUDE_max_colors_is_six_l2240_224086

/-- A cube is a structure with edges and a coloring function. -/
structure Cube where
  edges : Finset (Fin 12)
  coloring : Fin 12 → Nat

/-- Two edges are adjacent if they share a common vertex. -/
def adjacent (e1 e2 : Fin 12) : Prop := sorry

/-- A valid coloring satisfies the problem conditions. -/
def valid_coloring (c : Cube) : Prop :=
  ∀ (color1 color2 : Nat), color1 ≠ color2 →
    ∃ (e1 e2 : Fin 12), adjacent e1 e2 ∧ c.coloring e1 = color1 ∧ c.coloring e2 = color2

/-- The maximum number of colors that can be used. -/
def max_colors (c : Cube) : Nat :=
  Finset.card (Finset.image c.coloring c.edges)

/-- The main theorem: The maximum number of colors is 6. -/
theorem max_colors_is_six (c : Cube) (h : valid_coloring c) : max_colors c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_colors_is_six_l2240_224086


namespace NUMINAMATH_CALUDE_homework_problem_count_l2240_224052

/-- Calculates the total number of homework problems given the number of pages and problems per page -/
def total_problems (math_pages reading_pages problems_per_page : ℕ) : ℕ :=
  (math_pages + reading_pages) * problems_per_page

/-- Proves that given 6 pages of math homework, 4 pages of reading homework, and 3 problems per page, the total number of problems is 30 -/
theorem homework_problem_count : total_problems 6 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l2240_224052


namespace NUMINAMATH_CALUDE_committee_size_l2240_224050

/-- Given a committee of n members where any 3 individuals can be sent for a social survey,
    if the probability of female student B being chosen given that male student A is chosen is 0.4,
    then n = 6. -/
theorem committee_size (n : ℕ) : 
  (n ≥ 3) →  -- Ensure committee size is at least 3
  (((n - 2 : ℚ) / ((n - 1) * (n - 2) / 2)) = 0.4) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_l2240_224050


namespace NUMINAMATH_CALUDE_boat_race_spacing_l2240_224099

theorem boat_race_spacing (river_width : ℝ) (num_boats : ℕ) (boat_width : ℝ)
  (hw : river_width = 42)
  (hn : num_boats = 8)
  (hb : boat_width = 3) :
  (river_width - num_boats * boat_width) / (num_boats + 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boat_race_spacing_l2240_224099


namespace NUMINAMATH_CALUDE_parabola_c_value_l2240_224084

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola, its vertex, and a point it passes through, prove that c = 9/2 -/
theorem parabola_c_value (p : Parabola) (h1 : p.a * (-1)^2 + p.b * (-1) + p.c = 5)
    (h2 : p.a * 1^2 + p.b * 1 + p.c = 3) : p.c = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_c_value_l2240_224084


namespace NUMINAMATH_CALUDE_solve_rational_equation_l2240_224096

theorem solve_rational_equation (x : ℚ) :
  (x^2 - 10*x + 9) / (x - 1) + (2*x^2 + 17*x - 15) / (2*x - 3) = -5 →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_rational_equation_l2240_224096


namespace NUMINAMATH_CALUDE_store_discount_percentage_l2240_224092

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let final_price := 1.32 * C
  let discount_percentage := (new_year_price - final_price) / new_year_price * 100
  discount_percentage = 12 := by sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l2240_224092


namespace NUMINAMATH_CALUDE_proportion_equality_l2240_224083

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2240_224083


namespace NUMINAMATH_CALUDE_aluminum_ball_radius_l2240_224065

theorem aluminum_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) (large_radius : ℝ) :
  small_radius = 0.5 →
  num_small_balls = 12 →
  (4 / 3) * π * large_radius^3 = num_small_balls * ((4 / 3) * π * small_radius^3) →
  large_radius = (3 / 2)^(1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_ball_radius_l2240_224065


namespace NUMINAMATH_CALUDE_problem_solution_l2240_224090

theorem problem_solution (x y : ℝ) : 
  let A := 2 * x^2 - x + y - 3 * x * y
  let B := x^2 - 2 * x - y + x * y
  (A - 2 * B = 3 * x + 3 * y - 5 * x * y) ∧ 
  (x + y = 4 → x * y = -1/5 → A - 2 * B = 13) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2240_224090


namespace NUMINAMATH_CALUDE_gym_towels_theorem_l2240_224018

def gym_problem (first_hour_guests : ℕ) : Prop :=
  let second_hour_guests := first_hour_guests + (first_hour_guests * 20 / 100)
  let third_hour_guests := second_hour_guests + (second_hour_guests * 25 / 100)
  let fourth_hour_guests := third_hour_guests + (third_hour_guests * 33 / 100)
  let fifth_hour_guests := fourth_hour_guests - (fourth_hour_guests * 15 / 100)
  let sixth_hour_guests := fifth_hour_guests
  let seventh_hour_guests := sixth_hour_guests - (sixth_hour_guests * 30 / 100)
  let eighth_hour_guests := seventh_hour_guests - (seventh_hour_guests * 50 / 100)
  let total_guests := first_hour_guests + second_hour_guests + third_hour_guests + 
                      fourth_hour_guests + fifth_hour_guests + sixth_hour_guests + 
                      seventh_hour_guests + eighth_hour_guests
  let total_towels := total_guests * 2
  total_towels = 868

theorem gym_towels_theorem : 
  gym_problem 40 := by
  sorry

#check gym_towels_theorem

end NUMINAMATH_CALUDE_gym_towels_theorem_l2240_224018


namespace NUMINAMATH_CALUDE_equation_solution_l2240_224010

theorem equation_solution : 
  let x : ℚ := 30
  40 * x + (12 + 8) * 3 / 5 = 1212 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2240_224010


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2240_224033

theorem min_value_of_expression (x y z w : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) 
  (hw : -2 < w ∧ w < 2) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w/2)) + 
   1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w/2))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0/2)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0/2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2240_224033


namespace NUMINAMATH_CALUDE_onesDigit_73_pow_355_l2240_224093

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem onesDigit_73_pow_355 : onesDigit (73^355) = 7 := by
  sorry

end NUMINAMATH_CALUDE_onesDigit_73_pow_355_l2240_224093


namespace NUMINAMATH_CALUDE_all_points_collinear_l2240_224002

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- Main theorem -/
theorem all_points_collinear (M : Set Point) (h_finite : Set.Finite M)
  (h_line : ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → p ≠ q → 
    (∃ l : Line, p.onLine l ∧ q.onLine l) → (∃ s : Point, s ∈ M ∧ s ≠ p ∧ s ≠ q ∧ s.onLine l)) :
  ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → collinear p q r :=
sorry

end NUMINAMATH_CALUDE_all_points_collinear_l2240_224002


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l2240_224081

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l2240_224081


namespace NUMINAMATH_CALUDE_problem_solution_l2240_224027

theorem problem_solution (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) :
  c = (n * a) / (n - 2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2240_224027


namespace NUMINAMATH_CALUDE_check_error_proof_l2240_224043

theorem check_error_proof (x y : ℕ) : 
  x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 →  -- x and y are two-digit numbers
  y = 3 * x - 6 →                        -- y = 3x - 6
  100 * y + x - (100 * x + y) = 2112 →   -- difference is $21.12 (2112 cents)
  x = 14 ∧ y = 36 :=                     -- conclusion: x = 14 and y = 36
by sorry

end NUMINAMATH_CALUDE_check_error_proof_l2240_224043


namespace NUMINAMATH_CALUDE_equation_coefficients_l2240_224049

/-- Given a quadratic equation of the form ax^2 + bx + c = 0,
    this function returns a triple (a, b, c) of the coefficients -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem equation_coefficients :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 1
  quadratic_coefficients f = (-1, 3, -1) := by sorry

end NUMINAMATH_CALUDE_equation_coefficients_l2240_224049


namespace NUMINAMATH_CALUDE_rachel_chocolate_sales_l2240_224061

/-- The amount of money Rachel made by selling chocolate bars -/
def rachel_money_made (total_bars : ℕ) (price_per_bar : ℚ) (unsold_bars : ℕ) : ℚ :=
  (total_bars - unsold_bars : ℚ) * price_per_bar

/-- Theorem stating that Rachel made $58.50 from selling chocolate bars -/
theorem rachel_chocolate_sales : rachel_money_made 25 3.25 7 = 58.50 := by
  sorry

end NUMINAMATH_CALUDE_rachel_chocolate_sales_l2240_224061


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l2240_224036

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 4|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2) 2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≤ m - m^2) → m ∈ Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l2240_224036


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2240_224082

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2240_224082


namespace NUMINAMATH_CALUDE_max_cashback_selection_l2240_224075

structure Category where
  name : String
  cashback : Float
  expenses : Float

def calculate_cashback (c : Category) : Float :=
  c.cashback * c.expenses / 100

def total_cashback (categories : List Category) : Float :=
  categories.map calculate_cashback |> List.sum

theorem max_cashback_selection (categories : List Category) :
  let transport := { name := "Transport", cashback := 5, expenses := 2000 }
  let groceries := { name := "Groceries", cashback := 3, expenses := 5000 }
  let clothing := { name := "Clothing", cashback := 4, expenses := 3000 }
  let entertainment := { name := "Entertainment", cashback := 5, expenses := 3000 }
  let sports := { name := "Sports", cashback := 6, expenses := 1500 }
  let all_categories := [transport, groceries, clothing, entertainment, sports]
  let best_selection := [groceries, entertainment, clothing]
  categories = all_categories →
  (∀ selection : List Category,
    selection.length ≤ 3 →
    selection ⊆ categories →
    total_cashback selection ≤ total_cashback best_selection) :=
by sorry

end NUMINAMATH_CALUDE_max_cashback_selection_l2240_224075


namespace NUMINAMATH_CALUDE_book_cost_solution_l2240_224057

def book_cost_problem (p : ℝ) : Prop :=
  7 * p < 15 ∧ 11 * p > 22

theorem book_cost_solution :
  ∃ p : ℝ, book_cost_problem p ∧ p = 2.10 := by
sorry

end NUMINAMATH_CALUDE_book_cost_solution_l2240_224057


namespace NUMINAMATH_CALUDE_largest_root_of_g_l2240_224070

def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

theorem largest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (7/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l2240_224070


namespace NUMINAMATH_CALUDE_range_of_a_l2240_224058

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x > 0, x + 4 / x ≥ a) 
  (h2 : ∃ x : ℝ, x^2 + 2*x + a = 0) : 
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2240_224058


namespace NUMINAMATH_CALUDE_parabola_vertex_after_transformation_l2240_224019

/-- The vertex of a parabola after transformation -/
theorem parabola_vertex_after_transformation :
  let f (x : ℝ) := (x - 2)^2 - 2*(x - 2) + 6
  ∃! (h : ℝ × ℝ), (h.1 = 3 ∧ h.2 = 5 ∧ ∀ x, f x ≥ f h.1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_after_transformation_l2240_224019


namespace NUMINAMATH_CALUDE_cobbler_hourly_rate_l2240_224064

theorem cobbler_hourly_rate 
  (mold_cost : ℝ) 
  (work_hours : ℝ) 
  (discount_rate : ℝ) 
  (total_payment : ℝ) 
  (h1 : mold_cost = 250)
  (h2 : work_hours = 8)
  (h3 : discount_rate = 0.8)
  (h4 : total_payment = 730) :
  ∃ hourly_rate : ℝ, 
    hourly_rate = 75 ∧ 
    total_payment = mold_cost + discount_rate * work_hours * hourly_rate :=
by
  sorry

end NUMINAMATH_CALUDE_cobbler_hourly_rate_l2240_224064


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2240_224032

theorem quadratic_expression_value : 
  let x : ℤ := -2
  (x^2 + 6*x - 7) = -15 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2240_224032


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2240_224037

theorem ellipse_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m > 0 ∧ -(m + 1) > 0) ∧ 
   (2 + m ≠ -(m + 1))) ↔ 
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2240_224037


namespace NUMINAMATH_CALUDE_xoxox_probability_l2240_224038

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

theorem xoxox_probability :
  (x_tiles : ℚ) / total_tiles *
  (o_tiles : ℚ) / (total_tiles - 1) *
  ((x_tiles - 1) : ℚ) / (total_tiles - 2) *
  ((o_tiles - 1) : ℚ) / (total_tiles - 3) *
  ((x_tiles - 2) : ℚ) / (total_tiles - 4) = 1 / 10 :=
sorry

end NUMINAMATH_CALUDE_xoxox_probability_l2240_224038


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2240_224079

theorem nested_fraction_evaluation : 
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2240_224079


namespace NUMINAMATH_CALUDE_arrangement_counts_l2240_224073

/-- Counts the number of valid arrangements of crosses and zeros -/
def countArrangements (n : ℕ) (zeros : ℕ) : ℕ :=
  sorry

theorem arrangement_counts :
  (countArrangements 29 14 = 15) ∧ (countArrangements 28 14 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_counts_l2240_224073


namespace NUMINAMATH_CALUDE_subtracted_amount_l2240_224009

theorem subtracted_amount (n : ℚ) (x : ℚ) : 
  n = 25 / 3 → 3 * n + 15 = 6 * n - x → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2240_224009


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l2240_224034

def regular_decagon : ℕ := 10

def total_triangles : ℕ := regular_decagon.choose 3

def favorable_outcomes : ℕ := regular_decagon * (regular_decagon - 4)

def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l2240_224034


namespace NUMINAMATH_CALUDE_equation_solutions_l2240_224017

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6) / ((x - 1) * (x + 6)) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2240_224017


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l2240_224040

theorem gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 150 → m ≤ n) → 
  n = 146 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l2240_224040


namespace NUMINAMATH_CALUDE_sunny_candles_proof_l2240_224007

/-- Calculates the total number of candles used by Sunny --/
def total_candles (initial_cakes : ℕ) (given_away : ℕ) (candles_per_cake : ℕ) : ℕ :=
  (initial_cakes - given_away) * candles_per_cake

/-- Proves that Sunny will use 36 candles in total --/
theorem sunny_candles_proof :
  total_candles 8 2 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sunny_candles_proof_l2240_224007


namespace NUMINAMATH_CALUDE_coffee_milk_problem_l2240_224091

/-- Represents the liquid mixture in a cup -/
structure Mixture where
  coffee : ℚ
  milk : ℚ

/-- The process of mixing and transferring liquids -/
def mix_and_transfer (coffee_cup milk_cup : Mixture) : Mixture :=
  let transferred_coffee := coffee_cup.coffee / 3
  let mixed_cup := Mixture.mk (milk_cup.coffee + transferred_coffee) milk_cup.milk
  let total_mixed := mixed_cup.coffee + mixed_cup.milk
  let transferred_back := total_mixed / 2
  let coffee_ratio := mixed_cup.coffee / total_mixed
  let milk_ratio := mixed_cup.milk / total_mixed
  Mixture.mk 
    (coffee_cup.coffee - transferred_coffee + transferred_back * coffee_ratio)
    (transferred_back * milk_ratio)

theorem coffee_milk_problem :
  let initial_coffee_cup := Mixture.mk 6 0
  let initial_milk_cup := Mixture.mk 0 3
  let final_coffee_cup := mix_and_transfer initial_coffee_cup initial_milk_cup
  final_coffee_cup.milk / (final_coffee_cup.coffee + final_coffee_cup.milk) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_coffee_milk_problem_l2240_224091


namespace NUMINAMATH_CALUDE_borrowing_lending_period_l2240_224076

theorem borrowing_lending_period (principal : ℝ) (borrowing_rate : ℝ) (lending_rate : ℝ) (gain_per_year : ℝ) :
  principal = 9000 ∧ 
  borrowing_rate = 0.04 ∧ 
  lending_rate = 0.06 ∧ 
  gain_per_year = 180 → 
  (gain_per_year / (principal * (lending_rate - borrowing_rate))) = 1 := by
sorry

end NUMINAMATH_CALUDE_borrowing_lending_period_l2240_224076


namespace NUMINAMATH_CALUDE_triangle_inequality_from_condition_l2240_224021

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (h : 5 * (a^2 + b^2 + c^2) < 6 * (a*b + b*c + c*a)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_condition_l2240_224021


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l2240_224059

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * 4 = 6 * p.2) (Finset.product (Finset.range 25) (Finset.range 25))).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l2240_224059


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2240_224039

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2240_224039


namespace NUMINAMATH_CALUDE_sequence_constant_l2240_224012

/-- A sequence of positive real numbers satisfying a specific inequality is constant. -/
theorem sequence_constant (a : ℤ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_ineq : ∀ n, a n ≥ (a (n + 2) + a (n + 1) + a (n - 1) + a (n - 2)) / 4) :
  ∀ m n, a m = a n :=
by sorry

end NUMINAMATH_CALUDE_sequence_constant_l2240_224012


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2240_224051

theorem quadratic_coefficient (c : ℝ) : 
  ((-9 : ℝ)^2 + c * (-9) - 36 = 0) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2240_224051


namespace NUMINAMATH_CALUDE_scientific_notation_of_899000_l2240_224097

/-- Theorem: 899,000 expressed in scientific notation is 8.99 × 10^5 -/
theorem scientific_notation_of_899000 :
  899000 = 8.99 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_899000_l2240_224097


namespace NUMINAMATH_CALUDE_lunch_cost_is_24_l2240_224003

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℕ := sorry

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- John's bill is equal to the cost of the Taco Grande Plate -/
def johns_bill : ℕ := taco_grande_cost

/-- Mike's bill is equal to the cost of the Taco Grande Plate plus the additional items -/
def mikes_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- Mike's bill is twice as large as John's bill -/
axiom mikes_bill_twice_johns : mikes_bill = 2 * johns_bill

/-- The combined total cost of Mike and John's lunch -/
def total_cost : ℕ := johns_bill + mikes_bill

theorem lunch_cost_is_24 : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_lunch_cost_is_24_l2240_224003


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_l2240_224078

theorem not_necessary_nor_sufficient : ∃ (x y : ℝ), 
  ((x / y > 1 ∧ x ≤ y) ∨ (x / y ≤ 1 ∧ x > y)) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_l2240_224078


namespace NUMINAMATH_CALUDE_investment_income_is_575_l2240_224008

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Represents the total annual income from two investments with simple interest. -/
def totalAnnualIncome (investment1 : ℝ) (rate1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) : ℝ :=
  simpleInterest investment1 rate1 1 + simpleInterest investment2 rate2 1

/-- Theorem stating that the total annual income from the given investments is $575. -/
theorem investment_income_is_575 :
  totalAnnualIncome 3000 0.085 5000 0.064 = 575 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_575_l2240_224008


namespace NUMINAMATH_CALUDE_expression_factorization_l2240_224098

theorem expression_factorization (x : ℚ) :
  (x^2 - 3*x + 2) - (x^2 - x + 6) + (x - 1)*(x - 2) + x^2 + 2 = (2*x - 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2240_224098


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l2240_224066

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ x = -1) → m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l2240_224066


namespace NUMINAMATH_CALUDE_polynomial_roots_l2240_224087

/-- The polynomial x^3 - 3x^2 - x + 3 -/
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- The roots of the polynomial -/
def roots : Set ℝ := {1, -1, 3}

theorem polynomial_roots : 
  (∀ x ∈ roots, p x = 0) ∧ 
  (∀ x : ℝ, p x = 0 → x ∈ roots) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2240_224087


namespace NUMINAMATH_CALUDE_gardener_flower_expenses_l2240_224024

/-- The total expenses for flowers ordered by a gardener -/
theorem gardener_flower_expenses :
  let tulips : ℕ := 250
  let carnations : ℕ := 375
  let roses : ℕ := 320
  let price_per_flower : ℕ := 2
  let total_flowers : ℕ := tulips + carnations + roses
  let total_expenses : ℕ := total_flowers * price_per_flower
  total_expenses = 1890 := by sorry

end NUMINAMATH_CALUDE_gardener_flower_expenses_l2240_224024


namespace NUMINAMATH_CALUDE_impossible_table_l2240_224089

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the table -/
def Table := Cell → Int

/-- Two cells are adjacent if they share a side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- The table satisfies the adjacency condition -/
def satisfies_adjacency (t : Table) : Prop :=
  ∀ c1 c2 : Cell, adjacent c1 c2 → |t c1 - t c2| ≤ 18

/-- The table contains different integers -/
def all_different (t : Table) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 → t c1 ≠ t c2

/-- The main theorem -/
theorem impossible_table : ¬∃ t : Table, satisfies_adjacency t ∧ all_different t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_l2240_224089


namespace NUMINAMATH_CALUDE_homework_problem_l2240_224026

theorem homework_problem (a b c d : ℤ) : 
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) →
  (-a - b = -a * b) →
  (c * d = -182 * (1 / (-c - d))) →
  ((a = -2 ∧ b = -2) ∧ ((c = -1 ∧ d = -13) ∨ (c = -13 ∧ d = -1))) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l2240_224026


namespace NUMINAMATH_CALUDE_age_sum_in_two_years_l2240_224029

theorem age_sum_in_two_years :
  let fem_current_age : ℕ := 11
  let matt_current_age : ℕ := 4 * fem_current_age
  let jake_current_age : ℕ := matt_current_age + 5
  let fem_future_age : ℕ := fem_current_age + 2
  let matt_future_age : ℕ := matt_current_age + 2
  let jake_future_age : ℕ := jake_current_age + 2
  fem_future_age + matt_future_age + jake_future_age = 110
  := by sorry

end NUMINAMATH_CALUDE_age_sum_in_two_years_l2240_224029


namespace NUMINAMATH_CALUDE_sixth_term_value_l2240_224035

/-- A sequence of positive integers where each term after the first is 1/4 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) : ℚ) = (1 / 4) * ((a n : ℚ) + (a (n + 2) : ℚ))

theorem sixth_term_value (a : ℕ → ℕ+) (h : SpecialSequence a) (h1 : a 1 = 3) (h5 : a 5 = 43) :
  a 6 = 129 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2240_224035


namespace NUMINAMATH_CALUDE_lice_check_time_proof_l2240_224005

theorem lice_check_time_proof (kindergarteners first_graders second_graders third_graders : ℕ)
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19)
  (h3 : second_graders = 20) (h4 : third_graders = 25) (h5 : time_per_check = 2) :
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 :=
by sorry

end NUMINAMATH_CALUDE_lice_check_time_proof_l2240_224005


namespace NUMINAMATH_CALUDE_franks_change_is_four_l2240_224053

/-- Calculates the change Frank has after buying peanuts -/
def franks_change (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  let total_money := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let total_peanuts := daily_consumption * days
  let peanut_cost := peanut_cost_per_pound * total_peanuts
  total_money - peanut_cost

theorem franks_change_is_four :
  franks_change 7 4 2 1 3 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_change_is_four_l2240_224053


namespace NUMINAMATH_CALUDE_marble_remainder_l2240_224015

theorem marble_remainder (r p : ℕ) 
  (h_ringo : r % 6 = 4) 
  (h_paul : p % 6 = 3) : 
  (r + p) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l2240_224015


namespace NUMINAMATH_CALUDE_progress_primary_grade3_students_l2240_224072

/-- The number of students in Grade 3 of Progress Primary School -/
def total_students (num_classes : ℕ) (special_class_size : ℕ) (regular_class_size : ℕ) : ℕ :=
  special_class_size + (num_classes - 1) * regular_class_size

/-- Theorem stating the total number of students in Grade 3 of Progress Primary School -/
theorem progress_primary_grade3_students :
  total_students 10 48 50 = 48 + 9 * 50 := by
  sorry

end NUMINAMATH_CALUDE_progress_primary_grade3_students_l2240_224072


namespace NUMINAMATH_CALUDE_train_route_length_l2240_224014

/-- Given two trains traveling towards each other on a route, where Train X takes 4 hours
    to complete the trip, Train Y takes 3 hours to complete the trip, and Train X has
    traveled 60 km when they meet, prove that the total length of the route is 140 km. -/
theorem train_route_length (x_time y_time x_distance : ℝ) 
    (hx : x_time = 4)
    (hy : y_time = 3)
    (hd : x_distance = 60) : 
  x_distance * (1 / x_time + 1 / y_time) = 140 := by
  sorry

#check train_route_length

end NUMINAMATH_CALUDE_train_route_length_l2240_224014


namespace NUMINAMATH_CALUDE_car_speed_problem_l2240_224011

/-- Given a car traveling for two hours, where the speed in the second hour is 70 km/h
    and the average speed over two hours is 84 km/h, prove that the speed in the first hour
    must be 98 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) 
  (h1 : speed_second_hour = 70)
  (h2 : average_speed = 84) :
  ∃ (speed_first_hour : ℝ),
    speed_first_hour = 98 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2240_224011


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2240_224085

theorem cubic_root_sum (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) :
  (∃ x : ℝ, 27 * x^3 - 11 * x^2 - 11 * x - 3 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / r) →
  p + q + r = 782 := by
sorry


end NUMINAMATH_CALUDE_cubic_root_sum_l2240_224085


namespace NUMINAMATH_CALUDE_odometer_square_sum_l2240_224042

theorem odometer_square_sum : ∃ (a b c : ℕ),
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * b + 10 * c + a) ∧ (100 * b + 10 * c + a < 1000) ∧
  ((100 * b + 10 * c + a) - (100 * a + 10 * b + c)) % 60 = 0 ∧
  a^2 + b^2 + c^2 = 77 := by
sorry

end NUMINAMATH_CALUDE_odometer_square_sum_l2240_224042


namespace NUMINAMATH_CALUDE_value_of_a_l2240_224044

theorem value_of_a (S T : Set ℕ) (a : ℕ) : 
  S = {1, 2} → T = {a} → S ∪ T = S → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2240_224044


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2240_224071

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2240_224071


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2240_224016

theorem complex_expression_equality : 
  (Real.sqrt 3 - 1)^2 + (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 2 + Real.sqrt 3) + 
  (Real.sqrt 2 + 1) / (Real.sqrt 2 - 1) - 3 * Real.sqrt (1/2) = 
  8 - 2 * Real.sqrt 3 + Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2240_224016


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2240_224088

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 5 / y) : x + y = 24 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2240_224088


namespace NUMINAMATH_CALUDE_system_solution_l2240_224074

theorem system_solution :
  let x : ℚ := 57 / 31
  let y : ℚ := 195 / 62
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2240_224074


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l2240_224045

/-- Given an ellipse and a line, this theorem states the range of the y-intercept of the line for which
    there exist two distinct points on the ellipse symmetric about the line. -/
theorem ellipse_symmetric_points_range (x y : ℝ) (m : ℝ) : 
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- Point 1 on ellipse
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- Point 2 on ellipse
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧        -- Points are distinct
    (y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m)  -- Points symmetric about y = 4x + m
  ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l2240_224045


namespace NUMINAMATH_CALUDE_chord_length_is_two_l2240_224062

/-- The chord length intercepted by y = 1 - x on x² + y² + 2y - 2 = 0 is 2 -/
theorem chord_length_is_two (x y : ℝ) : 
  (x^2 + y^2 + 2*y - 2 = 0) → 
  (y = 1 - x) → 
  ∃ (a b : ℝ), (a^2 + b^2 = 1) ∧ 
               ((x - a)^2 + (y - b)^2 = 2^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_two_l2240_224062


namespace NUMINAMATH_CALUDE_add_ten_to_number_l2240_224000

theorem add_ten_to_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_add_ten_to_number_l2240_224000


namespace NUMINAMATH_CALUDE_bob_pays_48_percent_l2240_224028

-- Define the suggested retail price
variable (P : ℝ)

-- Define the marked price as 80% of the suggested retail price
def markedPrice (P : ℝ) : ℝ := 0.8 * P

-- Define Bob's purchase price as 60% of the marked price
def bobPrice (P : ℝ) : ℝ := 0.6 * markedPrice P

-- Theorem statement
theorem bob_pays_48_percent (P : ℝ) (h : P > 0) : 
  bobPrice P / P = 0.48 := by
sorry

end NUMINAMATH_CALUDE_bob_pays_48_percent_l2240_224028


namespace NUMINAMATH_CALUDE_rosette_area_l2240_224025

/-- The area of a rosette formed by four semicircles on the sides of a square -/
theorem rosette_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let rosette_area := (a^2 * (Real.pi - 2)) / 2
  rosette_area = (square_side^2 * (Real.pi - 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rosette_area_l2240_224025


namespace NUMINAMATH_CALUDE_square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l2240_224063

theorem square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven :
  ∀ a : ℝ, a = Real.sqrt 11 - 1 → a^2 + 2*a + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_root_eleven_minus_one_squared_plus_two_times_plus_one_equals_eleven_l2240_224063


namespace NUMINAMATH_CALUDE_line_parallel_perp_plane_implies_perp_line_l2240_224094

/-- In three-dimensional space -/
structure Space :=
  (points : Type*)
  (vectors : Type*)

/-- A line in space -/
structure Line (S : Space) :=
  (point : S.points)
  (direction : S.vectors)

/-- A plane in space -/
structure Plane (S : Space) :=
  (point : S.points)
  (normal : S.vectors)

/-- Parallel relation between lines -/
def parallel (S : Space) (a b : Line S) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perp_line_plane (S : Space) (l : Line S) (α : Plane S) : Prop := sorry

/-- Perpendicular relation between lines -/
def perp_line_line (S : Space) (l1 l2 : Line S) : Prop := sorry

/-- The main theorem -/
theorem line_parallel_perp_plane_implies_perp_line 
  (S : Space) (a b l : Line S) (α : Plane S) :
  parallel S a b → perp_line_plane S l α → perp_line_line S l b := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perp_plane_implies_perp_line_l2240_224094


namespace NUMINAMATH_CALUDE_expression_value_l2240_224013

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 1)    -- absolute value of m is 1
  : m + (2024 * (a + b)) / 2023 - (c * d)^2 = 0 ∨ 
    m + (2024 * (a + b)) / 2023 - (c * d)^2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2240_224013


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2240_224048

/-- A function f: ℝ → ℝ is monotonic on an interval [a, b] if it is either
    monotonically increasing or monotonically decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = 2x^2 - kx + 1 is monotonic on [1, 3]
    if and only if k ≤ 4 or k ≥ 12. -/
theorem quadratic_monotonicity (k : ℝ) :
  IsMonotonic (fun x => 2 * x^2 - k * x + 1) 1 3 ↔ k ≤ 4 ∨ k ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2240_224048
