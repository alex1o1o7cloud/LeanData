import Mathlib

namespace NUMINAMATH_CALUDE_investment_interest_difference_l3133_313318

theorem investment_interest_difference 
  (total_investment : ℝ)
  (investment_x : ℝ)
  (rate_x : ℝ)
  (rate_y : ℝ)
  (h1 : total_investment = 100000)
  (h2 : investment_x = 42000)
  (h3 : rate_x = 0.23)
  (h4 : rate_y = 0.17) :
  let investment_y := total_investment - investment_x
  let interest_x := investment_x * rate_x
  let interest_y := investment_y * rate_y
  interest_y - interest_x = 200 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_difference_l3133_313318


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3133_313310

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2*x + 7) / (x^2 - 2*x - 63)
  let g (x : ℝ) := 25 / (16 * (x - 9)) + 7 / (16 * (x + 7))
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3133_313310


namespace NUMINAMATH_CALUDE_lock_combinations_count_l3133_313349

/-- The number of digits on the lock -/
def n : ℕ := 4

/-- The number of possible digits (0 to 9) -/
def k : ℕ := 10

/-- The number of ways to select n digits from k possibilities in non-decreasing order -/
def lockCombinations : ℕ := (n + k - 1).choose (k - 1)

theorem lock_combinations_count : lockCombinations = 715 := by
  sorry

end NUMINAMATH_CALUDE_lock_combinations_count_l3133_313349


namespace NUMINAMATH_CALUDE_csc_240_degrees_l3133_313330

theorem csc_240_degrees : Real.cos (240 * π / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_csc_240_degrees_l3133_313330


namespace NUMINAMATH_CALUDE_sum_of_roots_is_nine_halves_l3133_313341

-- Define the polynomials
def p (x : ℝ) : ℝ := 2 * x^3 + x^2 - 8 * x + 20
def q (x : ℝ) : ℝ := 5 * x^3 - 25 * x^2 + 19

-- Define the equation
def equation (x : ℝ) : Prop := p x = 0 ∨ q x = 0

-- Theorem statement
theorem sum_of_roots_is_nine_halves :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, equation x) ∧
    (∀ x, equation x → x ∈ roots) ∧
    (Finset.sum roots id = 9/2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_nine_halves_l3133_313341


namespace NUMINAMATH_CALUDE_other_triangle_rectangle_area_ratio_l3133_313372

/-- Represents a right triangle with a point on its hypotenuse -/
structure RightTriangleWithPoint where
  /-- Length of the side of the rectangle along the hypotenuse -/
  side_along_hypotenuse : ℝ
  /-- Length of the side of the rectangle perpendicular to the hypotenuse -/
  side_perpendicular : ℝ
  /-- Ratio of the area of one small right triangle to the area of the rectangle -/
  area_ratio : ℝ
  /-- Condition: The side along the hypotenuse has length 1 -/
  hypotenuse_side_length : side_along_hypotenuse = 1
  /-- Condition: The area of one small right triangle is n times the area of the rectangle -/
  area_ratio_condition : area_ratio > 0

/-- Theorem: The ratio of the area of the other small right triangle to the area of the rectangle -/
theorem other_triangle_rectangle_area_ratio 
  (t : RightTriangleWithPoint) : 
  ∃ (ratio : ℝ), ratio = t.side_perpendicular / t.area_ratio := by
  sorry

end NUMINAMATH_CALUDE_other_triangle_rectangle_area_ratio_l3133_313372


namespace NUMINAMATH_CALUDE_polyhedron_sum_l3133_313365

/-- A convex polyhedron with triangular and hexagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  hexagons : ℕ
  vertices : ℕ
  edges : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  faces_sum : faces = triangles + hexagons
  faces_20 : faces = 20
  edges_formula : edges = (3 * triangles + 6 * hexagons) / 2
  euler_formula : vertices - edges + faces = 2

/-- The theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.vertices = 227 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l3133_313365


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3133_313381

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, -3 < a → a ≤ 0 → a ≠ -1 → a ≠ 0 → a ≠ 1 →
  let original_expr := (a - (2*a - 1) / a) / (1/a - a)
  let simplified_expr := (1 - a) / (1 + a)
  original_expr = simplified_expr ∧
  (a = -2 → simplified_expr = -3) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3133_313381


namespace NUMINAMATH_CALUDE_heels_cost_calculation_solve_shopping_problem_l3133_313368

def shopping_problem (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) : Prop :=
  ∃ (heels_cost : ℕ),
    initial_amount = jumper_cost + tshirt_cost + heels_cost + remaining_amount

theorem heels_cost_calculation (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) 
  (h : shopping_problem initial_amount jumper_cost tshirt_cost remaining_amount) :
  ∃ (heels_cost : ℕ), heels_cost = initial_amount - jumper_cost - tshirt_cost - remaining_amount :=
by
  sorry

#check @heels_cost_calculation

theorem solve_shopping_problem :
  shopping_problem 26 9 4 8 ∧ 
  (∃ (heels_cost : ℕ), heels_cost = 26 - 9 - 4 - 8 ∧ heels_cost = 5) :=
by
  sorry

#check @solve_shopping_problem

end NUMINAMATH_CALUDE_heels_cost_calculation_solve_shopping_problem_l3133_313368


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_minus_b_l3133_313348

theorem quadratic_roots_imply_a_minus_b (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_minus_b_l3133_313348


namespace NUMINAMATH_CALUDE_tree_spacing_l3133_313374

theorem tree_spacing (total_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
  (h1 : total_length = 157)
  (h2 : num_trees = 13)
  (h3 : tree_space = 1) :
  (total_length - num_trees * tree_space) / (num_trees - 1) = 12 :=
sorry

end NUMINAMATH_CALUDE_tree_spacing_l3133_313374


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l3133_313379

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 3 * x * (x - 3) = 4 ↔ 3 * x^2 - 9 * x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l3133_313379


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_l3133_313304

/-- Two complex numbers are symmetric about the angle bisector of the first and third quadrants if their real and imaginary parts are interchanged. -/
def symmetric_about_bisector (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetric_complex : ∀ z₁ z₂ : ℂ,
  symmetric_about_bisector z₁ z₂ → z₁ = 1 + 2*I → z₁ * z₂ = 5*I :=
by sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_l3133_313304


namespace NUMINAMATH_CALUDE_boys_on_playground_l3133_313315

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end NUMINAMATH_CALUDE_boys_on_playground_l3133_313315


namespace NUMINAMATH_CALUDE_magic_square_x_free_l3133_313333

/-- Represents a 3x3 magic square with given entries -/
structure MagicSquare where
  x : ℝ
  sum : ℝ
  top_middle : ℝ
  top_right : ℝ
  middle_left : ℝ
  is_magic : sum = x + top_middle + top_right
           ∧ sum = x + middle_left + (sum - x - middle_left)
           ∧ sum = top_right + (sum - top_right - (sum - x - middle_left))

/-- Theorem stating that x can be any real number in the given magic square -/
theorem magic_square_x_free (m : MagicSquare) (h : m.top_middle = 35 ∧ m.top_right = 58 ∧ m.middle_left = 8 ∧ m.sum = 85) :
  ∀ y : ℝ, ∃ m' : MagicSquare, m'.x = y ∧ m'.top_middle = m.top_middle ∧ m'.top_right = m.top_right ∧ m'.middle_left = m.middle_left ∧ m'.sum = m.sum :=
sorry

end NUMINAMATH_CALUDE_magic_square_x_free_l3133_313333


namespace NUMINAMATH_CALUDE_daniel_has_five_dogs_l3133_313339

/-- The number of legs for a healthy horse -/
def horse_legs : ℕ := 4

/-- The number of legs for a healthy cat -/
def cat_legs : ℕ := 4

/-- The number of legs for a healthy turtle -/
def turtle_legs : ℕ := 4

/-- The number of legs for a healthy goat -/
def goat_legs : ℕ := 4

/-- The number of legs for a healthy dog -/
def dog_legs : ℕ := 4

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The total number of legs of all animals Daniel has -/
def total_legs : ℕ := 72

theorem daniel_has_five_dogs :
  ∃ (num_dogs : ℕ), 
    num_dogs * dog_legs + 
    num_horses * horse_legs + 
    num_cats * cat_legs + 
    num_turtles * turtle_legs + 
    num_goats * goat_legs = total_legs ∧ 
    num_dogs = 5 := by
  sorry

end NUMINAMATH_CALUDE_daniel_has_five_dogs_l3133_313339


namespace NUMINAMATH_CALUDE_real_part_of_i_over_one_plus_i_l3133_313375

theorem real_part_of_i_over_one_plus_i : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  Complex.re z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_over_one_plus_i_l3133_313375


namespace NUMINAMATH_CALUDE_train_length_l3133_313336

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 3 → ∃ (length_m : ℝ), abs (length_m - 50.01) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3133_313336


namespace NUMINAMATH_CALUDE_fifteen_non_congruent_triangles_l3133_313369

-- Define the points
variable (A B C M N P : ℝ × ℝ)

-- Define the isosceles triangle
def is_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

-- Define M as midpoint of AB
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define N on AC with 1:2 ratio
def divides_in_ratio_one_two (N A C : ℝ × ℝ) : Prop :=
  dist A N = (1/3) * dist A C

-- Define P on BC with 1:3 ratio
def divides_in_ratio_one_three (P B C : ℝ × ℝ) : Prop :=
  dist B P = (1/4) * dist B C

-- Define a function to count non-congruent triangles
def count_non_congruent_triangles (A B C M N P : ℝ × ℝ) : ℕ := sorry

-- State the theorem
theorem fifteen_non_congruent_triangles
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_midpoint M A B)
  (h3 : divides_in_ratio_one_two N A C)
  (h4 : divides_in_ratio_one_three P B C) :
  count_non_congruent_triangles A B C M N P = 15 := by sorry

end NUMINAMATH_CALUDE_fifteen_non_congruent_triangles_l3133_313369


namespace NUMINAMATH_CALUDE_floor_plus_self_equation_l3133_313350

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 10.3 ↔ r = 5.3 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_equation_l3133_313350


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_radius_l3133_313397

theorem triangle_circumscribed_circle_radius 
  (α : Real) (a b : Real) (R : Real) : 
  α = π / 3 →  -- 60° in radians
  a = 6 → 
  b = 2 → 
  R = (2 * Real.sqrt 21) / 3 → 
  2 * R = (Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))) / (Real.sin α) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_radius_l3133_313397


namespace NUMINAMATH_CALUDE_range_a_theorem_l3133_313346

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := a < 1 ∧ a ≠ 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l3133_313346


namespace NUMINAMATH_CALUDE_football_field_fertilizer_l3133_313382

/-- Given a football field and fertilizer distribution, calculate the total fertilizer used. -/
theorem football_field_fertilizer 
  (field_area : ℝ) 
  (partial_area : ℝ) 
  (partial_fertilizer : ℝ) 
  (h1 : field_area = 8400)
  (h2 : partial_area = 3500)
  (h3 : partial_fertilizer = 500)
  (h4 : partial_area > 0)
  (h5 : field_area > 0) :
  (field_area * partial_fertilizer) / partial_area = 1200 :=
by sorry

end NUMINAMATH_CALUDE_football_field_fertilizer_l3133_313382


namespace NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l3133_313378

def tamika_set : Finset ℕ := {10, 11, 12}
def carlos_set : Finset ℕ := {4, 6, 7}

def tamika_sums : Finset ℕ := {21, 22, 23}
def carlos_sums : Finset ℕ := {10, 11, 13}

def favorable_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

def total_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

theorem probability_tamika_greater_carlos :
  (favorable_outcomes : ℚ) / total_outcomes = 1 := by sorry

end NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l3133_313378


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt5_plus_1_l3133_313338

theorem consecutive_integers_around_sqrt5_plus_1 (x y : ℤ) :
  (y = x + 1) →  -- x and y are consecutive integers
  (x < Real.sqrt 5 + 1) →  -- x < √5 + 1
  (Real.sqrt 5 + 1 < y) →  -- √5 + 1 < y
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt5_plus_1_l3133_313338


namespace NUMINAMATH_CALUDE_cell_count_after_twelve_days_l3133_313387

/-- Represents the cell growth and death process over 12 days -/
def cell_growth (initial_cells : ℕ) (split_interval : ℕ) (total_days : ℕ) (death_day : ℕ) (cells_died : ℕ) : ℕ :=
  let cycles := total_days / split_interval
  let final_count := initial_cells * 2^cycles
  if death_day ≤ total_days then final_count - cells_died else final_count

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_twelve_days :
  cell_growth 5 3 12 9 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_twelve_days_l3133_313387


namespace NUMINAMATH_CALUDE_f_n_formula_l3133_313340

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f x
  | m + 1 => deriv (f_n m) x

theorem f_n_formula (n : ℕ) (x : ℝ) :
  f_n (n + 1) x = ((-1)^(n + 1) * (x - (n + 1))) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_f_n_formula_l3133_313340


namespace NUMINAMATH_CALUDE_claire_age_l3133_313319

/-- Given the ages of Claire, Leo, and Mia, prove Claire's age --/
theorem claire_age (mia leo claire : ℕ) 
  (h1 : claire = leo - 5) 
  (h2 : leo = mia + 4) 
  (h3 : mia = 20) : 
  claire = 19 := by
sorry

end NUMINAMATH_CALUDE_claire_age_l3133_313319


namespace NUMINAMATH_CALUDE_grid_division_theorem_l3133_313306

/-- A grid division is valid if it satisfies the given conditions -/
def is_valid_division (n : ℕ) : Prop :=
  ∃ (m : ℕ), n^2 = 4 + 5*m ∧ 
  ∃ (square_pos : ℕ × ℕ), square_pos.1 < n ∧ square_pos.2 < n-1 ∧
  (square_pos.1 = 0 ∨ square_pos.1 = n-2 ∨ square_pos.2 = 0 ∨ square_pos.2 = n-2)

/-- The main theorem stating the condition for valid grid division -/
theorem grid_division_theorem (n : ℕ) : 
  is_valid_division n ↔ n % 5 = 2 :=
sorry

end NUMINAMATH_CALUDE_grid_division_theorem_l3133_313306


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3133_313395

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧
  ¬(∀ a b, a < b → (a - b) * a^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3133_313395


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_neg_95_l3133_313332

theorem largest_multiple_of_12_less_than_neg_95 : 
  ∀ n : ℤ, n * 12 < -95 → n * 12 ≤ -96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_less_than_neg_95_l3133_313332


namespace NUMINAMATH_CALUDE_gasoline_price_change_l3133_313389

/-- Represents the price change of gasoline over two months -/
theorem gasoline_price_change 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) 
  (h1 : initial_price = 7.5)
  (h2 : final_price = 8.4)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  : initial_price * (1 + x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_gasoline_price_change_l3133_313389


namespace NUMINAMATH_CALUDE_battle_treaty_day_l3133_313323

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculates the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Calculates the date after adding a number of days to a given date -/
def addDays (date : Date) (days : Nat) : Date :=
  sorry

/-- The statement of the theorem -/
theorem battle_treaty_day :
  let battleStart : Date := ⟨1800, 3, 3⟩
  let battleStartDay : DayOfWeek := DayOfWeek.Monday
  let treatyDate : Date := addDays battleStart 1000
  dayOfWeek treatyDate = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_battle_treaty_day_l3133_313323


namespace NUMINAMATH_CALUDE_stating_no_room_for_other_animals_l3133_313380

/-- Represents the composition of animals in a circus --/
structure CircusAnimals where
  total : ℕ
  lions : ℕ
  tigers : ℕ
  h_lions : lions = (total - lions) / 5
  h_tigers : tigers = total - tigers + 5

/-- 
Theorem stating that in a circus where the number of lions is 1/5 of the number of non-lions, 
and the number of tigers is 5 more than the number of non-tigers, 
there is no room for any other animals.
-/
theorem no_room_for_other_animals (c : CircusAnimals) : 
  c.lions + c.tigers = c.total :=
sorry

end NUMINAMATH_CALUDE_stating_no_room_for_other_animals_l3133_313380


namespace NUMINAMATH_CALUDE_election_win_percentage_l3133_313390

/-- The minimum percentage of votes needed to win an election --/
def min_win_percentage (total_votes : ℕ) (geoff_percentage : ℚ) (additional_votes_needed : ℕ) : ℚ :=
  ((geoff_percentage * total_votes + additional_votes_needed) / total_votes) * 100

/-- Theorem stating the minimum percentage of votes needed to win the election --/
theorem election_win_percentage :
  let total_votes : ℕ := 6000
  let geoff_percentage : ℚ := 1/200
  let additional_votes_needed : ℕ := 3000
  min_win_percentage total_votes geoff_percentage additional_votes_needed = 101/2 := by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3133_313390


namespace NUMINAMATH_CALUDE_sas_congruence_l3133_313337

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- SAS Congruence Theorem
theorem sas_congruence (t1 t2 : Triangle) 
  (h1 : t1.a = t2.a)  -- First side equal
  (h2 : t1.b = t2.b)  -- Second side equal
  (h3 : t1.α = t2.α)  -- Included angle equal
  : congruent t1 t2 :=
by
  sorry


end NUMINAMATH_CALUDE_sas_congruence_l3133_313337


namespace NUMINAMATH_CALUDE_percentage_difference_in_earnings_l3133_313357

def mike_hourly_rate : ℝ := 12
def phil_hourly_rate : ℝ := 6

theorem percentage_difference_in_earnings : 
  (mike_hourly_rate - phil_hourly_rate) / mike_hourly_rate * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_in_earnings_l3133_313357


namespace NUMINAMATH_CALUDE_racket_price_l3133_313399

theorem racket_price (total_spent sneakers_cost outfit_cost : ℕ) 
  (h1 : total_spent = 750)
  (h2 : sneakers_cost = 200)
  (h3 : outfit_cost = 250) :
  total_spent - sneakers_cost - outfit_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_racket_price_l3133_313399


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_l3133_313300

theorem simplify_and_evaluate_fraction (a : ℝ) (h : a = 5) :
  (a^2 - 4) / a^2 / (1 - 2/a) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_fraction_l3133_313300


namespace NUMINAMATH_CALUDE_sandbag_weight_sandbag_problem_l3133_313371

theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : ℝ :=
  let sand_weight := capacity * fill_percentage
  let extra_weight := sand_weight * weight_increase
  sand_weight + extra_weight

theorem sandbag_problem :
  sandbag_weight 250 0.8 0.4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_sandbag_problem_l3133_313371


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l3133_313314

def p (x : ℝ) : ℝ := 2 * abs x - 1

def q (x : ℝ) : ℝ := -abs x - 1

def xValues : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function : 
  (xValues.map (fun x => q (p x))).sum = -42 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l3133_313314


namespace NUMINAMATH_CALUDE_spherical_coordinate_reflection_l3133_313329

/-- Given a point with rectangular coordinates (3, -4, 12) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, -θ, φ) has rectangular coordinates (3, 4, 12) -/
theorem spherical_coordinate_reflection (ρ θ φ : ℝ) :
  (3 = ρ * Real.sin φ * Real.cos θ) →
  (-4 = ρ * Real.sin φ * Real.sin θ) →
  (12 = ρ * Real.cos φ) →
  (3 = ρ * Real.sin φ * Real.cos (-θ)) ∧
  (4 = ρ * Real.sin φ * Real.sin (-θ)) ∧
  (12 = ρ * Real.cos φ) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_reflection_l3133_313329


namespace NUMINAMATH_CALUDE_min_value_theorem_l3133_313308

/-- Given positive real numbers a and b, this theorem proves that the minimum value of
    (a + 2/b)(a + 2/b - 1010) + (b + 2/a)(b + 2/a - 1010) + 101010 is -404040. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 2/y) * (x + 2/y - 1010) + (y + 2/x) * (y + 2/x - 1010) + 101010 <
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010) →
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010 ≥ -404040 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3133_313308


namespace NUMINAMATH_CALUDE_expression_evaluation_l3133_313356

theorem expression_evaluation : 1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 = 190 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3133_313356


namespace NUMINAMATH_CALUDE_pascal_triangle_43_numbers_l3133_313312

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The second number in a row of Pascal's triangle -/
def pascal_second_number (n : ℕ) : ℕ := n

theorem pascal_triangle_43_numbers :
  ∃ n : ℕ, pascal_row_length n = 43 ∧ pascal_second_number n = 42 :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_43_numbers_l3133_313312


namespace NUMINAMATH_CALUDE_lattice_triangle_circumcircle_diameter_bound_l3133_313331

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The side lengths of a LatticeTriangle -/
def side_lengths (t : LatticeTriangle) : Fin 3 → ℝ := sorry

/-- The diameter of the circumcircle of a LatticeTriangle -/
def circumcircle_diameter (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The diameter of the circumcircle of a triangle with lattice point vertices
    does not exceed the product of its side lengths -/
theorem lattice_triangle_circumcircle_diameter_bound (t : LatticeTriangle) :
  circumcircle_diameter t ≤ (side_lengths t 0) * (side_lengths t 1) * (side_lengths t 2) := by
  sorry

end NUMINAMATH_CALUDE_lattice_triangle_circumcircle_diameter_bound_l3133_313331


namespace NUMINAMATH_CALUDE_Q_neither_sufficient_nor_necessary_for_P_l3133_313325

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def P (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∀ x, (a₁ * x^2 + b₁ * x + c₁ > 0) ↔ (a₂ * x^2 + b₂ * x + c₂ > 0)

/-- Proposition Q: The ratios of corresponding coefficients are equal -/
def Q (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem Q_neither_sufficient_nor_necessary_for_P :
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, Q a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬P a₁ b₁ c₁ a₂ b₂ c₂) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, P a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬Q a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_Q_neither_sufficient_nor_necessary_for_P_l3133_313325


namespace NUMINAMATH_CALUDE_light_year_scientific_notation_l3133_313366

theorem light_year_scientific_notation :
  9500000000000 = 9.5 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_light_year_scientific_notation_l3133_313366


namespace NUMINAMATH_CALUDE_alien_minerals_count_l3133_313328

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of minerals collected by the alien --/
def alienMinerals : ℕ := base7ToBase10 3 2 1

theorem alien_minerals_count :
  alienMinerals = 162 := by sorry

end NUMINAMATH_CALUDE_alien_minerals_count_l3133_313328


namespace NUMINAMATH_CALUDE_football_practice_hours_l3133_313309

/-- Calculates the total practice hours for a football team in a week with one missed day -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 6 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 36 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l3133_313309


namespace NUMINAMATH_CALUDE_not_always_perfect_square_exists_l3133_313321

/-- Given an n-digit number x, prove that there doesn't always exist a non-negative integer y ≤ 9
    and an integer z such that 10^(n+1) * z + 10x + y is a perfect square. -/
theorem not_always_perfect_square_exists (n : ℕ) : 
  ∃ x : ℕ, (10^n ≤ x ∧ x < 10^(n+1)) →
    ¬∃ (y z : ℤ), 0 ≤ y ∧ y ≤ 9 ∧ ∃ (k : ℤ), 10^(n+1) * z + 10 * x + y = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_always_perfect_square_exists_l3133_313321


namespace NUMINAMATH_CALUDE_pants_price_problem_l3133_313359

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) :
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  pants_price = 34.00 := by
sorry

end NUMINAMATH_CALUDE_pants_price_problem_l3133_313359


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3133_313316

theorem fraction_equivalence : (15 : ℚ) / 25 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3133_313316


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3133_313376

theorem trigonometric_identity (α : ℝ) :
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * (Real.cos (2 * α + Real.pi))^2 - 1) = 2 * Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3133_313376


namespace NUMINAMATH_CALUDE_stating_max_girls_in_class_l3133_313320

/-- Represents the number of students in the class -/
def total_students : ℕ := 25

/-- Represents the maximum number of girls in the class -/
def max_girls : ℕ := 13

/-- 
Theorem stating that given a class of 25 students where no two girls 
have the same number of boy friends, the maximum number of girls is 13.
-/
theorem max_girls_in_class :
  ∀ (girls boys : ℕ),
  girls + boys = total_students →
  (∀ (g₁ g₂ : ℕ), g₁ < girls → g₂ < girls → g₁ ≠ g₂ → 
    ∃ (b₁ b₂ : ℕ), b₁ ≤ boys ∧ b₂ ≤ boys ∧ b₁ ≠ b₂) →
  girls ≤ max_girls :=
by sorry

end NUMINAMATH_CALUDE_stating_max_girls_in_class_l3133_313320


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l3133_313367

theorem jungkook_balls_count (num_boxes : ℕ) (balls_per_box : ℕ) : 
  num_boxes = 2 → balls_per_box = 3 → num_boxes * balls_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l3133_313367


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3133_313343

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3133_313343


namespace NUMINAMATH_CALUDE_cylinder_height_l3133_313327

/-- The height of a cylinder given its base perimeter and side surface diagonal --/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (h : base_perimeter = 6 ∧ diagonal = 10) :
  ∃ (height : ℝ), height = 8 ∧ height ^ 2 + base_perimeter ^ 2 = diagonal ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l3133_313327


namespace NUMINAMATH_CALUDE_mean_temperature_and_humidity_l3133_313305

def temperatures : List Int := [-6, -2, -2, -3, 2, 4, 3]
def humidities : List Int := [70, 65, 65, 72, 80, 75, 77]

theorem mean_temperature_and_humidity :
  (temperatures.sum : ℚ) / temperatures.length = -4/7 ∧
  (humidities.sum : ℚ) / humidities.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_and_humidity_l3133_313305


namespace NUMINAMATH_CALUDE_y_not_between_l3133_313354

theorem y_not_between (a b x y : ℝ) (ha : a > 0) (hb : b > 0) 
  (hy : y = (a * Real.sin x + b) / (a * Real.sin x - b)) :
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_y_not_between_l3133_313354


namespace NUMINAMATH_CALUDE_ceo_dividends_calculation_l3133_313358

/-- Calculates the CEO's dividends based on company financial data -/
theorem ceo_dividends_calculation (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ) 
  (monthly_loan_payment : ℝ) (months_in_year : ℕ) (total_shares : ℕ) (ceo_ownership : ℝ) 
  (h1 : revenue = 2500000)
  (h2 : expenses = 1576250)
  (h3 : tax_rate = 0.2)
  (h4 : monthly_loan_payment = 25000)
  (h5 : months_in_year = 12)
  (h6 : total_shares = 1600)
  (h7 : ceo_ownership = 0.35) :
  ∃ (ceo_dividends : ℝ),
    ceo_dividends = 153440 ∧
    ceo_dividends = 
      ((revenue - expenses - (revenue - expenses) * tax_rate - 
        (monthly_loan_payment * months_in_year)) / total_shares) * 
      ceo_ownership * total_shares :=
by
  sorry

end NUMINAMATH_CALUDE_ceo_dividends_calculation_l3133_313358


namespace NUMINAMATH_CALUDE_solution_volume_l3133_313353

/-- Proves that the total volume of a solution is 10 liters, given that it contains 2.5 liters of pure acid and has a 25% concentration. -/
theorem solution_volume (acid_volume : ℝ) (concentration : ℝ) :
  acid_volume = 2.5 →
  concentration = 0.25 →
  acid_volume / concentration = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l3133_313353


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3133_313383

theorem solve_linear_equation (x y a : ℚ) : 
  x = 2 → y = a → 2 * x - 3 * y = 5 → a = -1/3 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3133_313383


namespace NUMINAMATH_CALUDE_count_1973_in_I_1000000_l3133_313344

-- Define the sequence type
def Sequence := List Nat

-- Define the initial sequence
def I₀ : Sequence := [1, 1]

-- Define the rule for generating the next sequence
def nextSequence (I : Sequence) : Sequence :=
  sorry

-- Define the n-th sequence
def Iₙ (n : Nat) : Sequence :=
  sorry

-- Define the count of a number in a sequence
def count (m : Nat) (I : Sequence) : Nat :=
  sorry

-- Euler's totient function
def φ (n : Nat) : Nat :=
  sorry

-- The main theorem
theorem count_1973_in_I_1000000 :
  count 1973 (Iₙ 1000000) = φ 1973 :=
sorry

end NUMINAMATH_CALUDE_count_1973_in_I_1000000_l3133_313344


namespace NUMINAMATH_CALUDE_product_modulo_600_l3133_313303

theorem product_modulo_600 : (2537 * 1985) % 600 = 145 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_600_l3133_313303


namespace NUMINAMATH_CALUDE_inequality_proof_l3133_313326

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / (x * y * z) ^ (1/3) ≤ x/y + y/z + z/x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3133_313326


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3133_313398

theorem stratified_sample_size
  (ratio_A ratio_B ratio_C : ℕ)
  (sample_A : ℕ)
  (h_ratio : ratio_A = 3 ∧ ratio_B = 4 ∧ ratio_C = 7)
  (h_sample_A : sample_A = 15) :
  ∃ n : ℕ, n = sample_A * (ratio_A + ratio_B + ratio_C) / ratio_A ∧ n = 70 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3133_313398


namespace NUMINAMATH_CALUDE_condition_relationship_l3133_313324

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧ 
  (∃ a b, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l3133_313324


namespace NUMINAMATH_CALUDE_economizable_disjoint_from_non_economizable_l3133_313351

-- Define the type for expenses
inductive Expense
  | LoanPayment
  | TaxPayment
  | QualificationCourse
  | HomeInternet
  | TravelExpense
  | VideoCamera
  | DomainPayment
  | CoffeeShopVisit

-- Define the property of being economizable
def is_economizable (e : Expense) : Prop :=
  match e with
  | Expense.HomeInternet => True
  | Expense.TravelExpense => True
  | Expense.VideoCamera => True
  | Expense.DomainPayment => True
  | Expense.CoffeeShopVisit => True
  | _ => False

-- Define the sets of economizable and non-economizable expenses
def economizable_expenses : Set Expense :=
  {e | is_economizable e}

def non_economizable_expenses : Set Expense :=
  {e | ¬is_economizable e}

-- Theorem statement
theorem economizable_disjoint_from_non_economizable :
  economizable_expenses ∩ non_economizable_expenses = ∅ :=
by sorry

end NUMINAMATH_CALUDE_economizable_disjoint_from_non_economizable_l3133_313351


namespace NUMINAMATH_CALUDE_gcd_18_30_l3133_313335

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3133_313335


namespace NUMINAMATH_CALUDE_cos_240_deg_l3133_313317

/-- Cosine of 240 degrees is equal to -1/2 -/
theorem cos_240_deg : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_deg_l3133_313317


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3133_313342

theorem x_intercepts_count : ∃! (s : Finset ℝ), 
  (∀ x ∈ s, (x - 3) * (x^2 + 4*x + 3) = 0) ∧ 
  s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3133_313342


namespace NUMINAMATH_CALUDE_misha_second_round_score_l3133_313373

/-- Represents the points scored in each round of dart throwing -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of Misha's dart game -/
def valid_dart_game (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = (3 * scores.second) / 2 ∧
  scores.first ≥ 24 ∧
  scores.third ≤ 72

/-- Theorem stating that Misha must have scored 48 points in the second round -/
theorem misha_second_round_score (scores : DartScores) 
  (h : valid_dart_game scores) : scores.second = 48 := by
  sorry

end NUMINAMATH_CALUDE_misha_second_round_score_l3133_313373


namespace NUMINAMATH_CALUDE_both_glasses_and_hair_tied_l3133_313370

def students : Finset ℕ := Finset.range 30

def glasses : Finset ℕ := {1, 3, 7, 10, 23, 27}

def hairTied : Finset ℕ := {1, 9, 11, 20, 23}

theorem both_glasses_and_hair_tied :
  (glasses ∩ hairTied).card = 2 := by sorry

end NUMINAMATH_CALUDE_both_glasses_and_hair_tied_l3133_313370


namespace NUMINAMATH_CALUDE_mr_zhang_birthday_l3133_313347

-- Define the possible dates
inductive Date
| feb5 | feb7 | feb9
| may5 | may8
| aug4 | aug7
| sep4 | sep6 | sep9

def Date.month : Date → Nat
| .feb5 | .feb7 | .feb9 => 2
| .may5 | .may8 => 5
| .aug4 | .aug7 => 8
| .sep4 | .sep6 | .sep9 => 9

def Date.day : Date → Nat
| .feb5 => 5
| .feb7 => 7
| .feb9 => 9
| .may5 => 5
| .may8 => 8
| .aug4 => 4
| .aug7 => 7
| .sep4 => 4
| .sep6 => 6
| .sep9 => 9

-- Define the statements made by A and B
def A_statement1 (d : Date) : Prop := 
  ∃ d' : Date, d.month = d'.month ∧ d ≠ d'

def B_statement (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' → d.day ≠ d'.day

def A_statement2 (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' ∧ B_statement d' → d = d'

-- Theorem to prove
theorem mr_zhang_birthday : 
  ∃! d : Date, A_statement1 d ∧ B_statement d ∧ A_statement2 d ∧ d = Date.aug4 := by
  sorry

end NUMINAMATH_CALUDE_mr_zhang_birthday_l3133_313347


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3133_313362

theorem smaller_number_proof (x y : ℝ) : 
  y = 2 * x - 3 → 
  x + y = 51 → 
  min x y = 18 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3133_313362


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l3133_313394

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 150) (h2 : books_per_shelf = 15) : 
  num_shelves * books_per_shelf = 2250 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l3133_313394


namespace NUMINAMATH_CALUDE_square_area_sum_l3133_313393

theorem square_area_sum (a b : ℕ) (h : a^2 + b^2 = 400) : a + b = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_area_sum_l3133_313393


namespace NUMINAMATH_CALUDE_cos_2017_deg_l3133_313352

theorem cos_2017_deg : Real.cos (2017 * π / 180) = -Real.cos (37 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_2017_deg_l3133_313352


namespace NUMINAMATH_CALUDE_min_disks_required_l3133_313345

def total_files : ℕ := 40
def disk_capacity : ℚ := 2
def files_1mb : ℕ := 4
def files_0_9mb : ℕ := 16
def file_size_1mb : ℚ := 1
def file_size_0_9mb : ℚ := 9/10
def file_size_0_5mb : ℚ := 1/2

theorem min_disks_required :
  let remaining_files := total_files - files_1mb - files_0_9mb
  let total_size := files_1mb * file_size_1mb + 
                    files_0_9mb * file_size_0_9mb + 
                    remaining_files * file_size_0_5mb
  let min_disks := Int.ceil (total_size / disk_capacity)
  min_disks = 16 := by sorry

end NUMINAMATH_CALUDE_min_disks_required_l3133_313345


namespace NUMINAMATH_CALUDE_sixth_candy_to_pete_l3133_313307

/-- Represents the recipients of candy wrappers -/
inductive Recipient : Type
  | Pete : Recipient
  | Vasey : Recipient

/-- Represents the sequence of candy wrapper distributions -/
def CandySequence : Fin 6 → Recipient
  | ⟨0, _⟩ => Recipient.Pete
  | ⟨1, _⟩ => Recipient.Pete
  | ⟨2, _⟩ => Recipient.Pete
  | ⟨3, _⟩ => Recipient.Vasey
  | ⟨4, _⟩ => Recipient.Vasey
  | ⟨5, _⟩ => Recipient.Pete

theorem sixth_candy_to_pete :
  CandySequence ⟨5, by norm_num⟩ = Recipient.Pete := by sorry

end NUMINAMATH_CALUDE_sixth_candy_to_pete_l3133_313307


namespace NUMINAMATH_CALUDE_fair_coin_expectation_l3133_313396

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The expected value of heads for a single toss of a fair coin -/
def expectedValueSingleToss (p : ℝ) (h : fairCoin p) : ℝ := p

/-- The number of tosses -/
def numTosses : ℕ := 5

/-- The mathematical expectation of heads for multiple tosses of a fair coin -/
def expectedValueMultipleTosses (p : ℝ) (h : fairCoin p) : ℝ :=
  (expectedValueSingleToss p h) * numTosses

theorem fair_coin_expectation (p : ℝ) (h : fairCoin p) :
  expectedValueMultipleTosses p h = 5/2 := by sorry

end NUMINAMATH_CALUDE_fair_coin_expectation_l3133_313396


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l3133_313391

theorem michelle_sandwiches (total : ℕ) (given : ℕ) (kept : ℕ) (remaining : ℕ) : 
  total = 20 → 
  given = 4 → 
  kept = 2 * given → 
  remaining = total - given - kept → 
  remaining = 8 := by
sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l3133_313391


namespace NUMINAMATH_CALUDE_school_ratio_change_l3133_313384

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student population -/
structure School where
  boarders : ℕ
  day_students : ℕ

def initial_ratio : Ratio := { boarders := 5, day_students := 12 }

def initial_school : School := { boarders := 150, day_students := 360 }

def new_boarders : ℕ := 30

def final_school : School := { 
  boarders := initial_school.boarders + new_boarders,
  day_students := initial_school.day_students
}

def final_ratio : Ratio := { 
  boarders := 1,
  day_students := 2
}

theorem school_ratio_change :
  (initial_ratio.boarders * initial_school.day_students = initial_ratio.day_students * initial_school.boarders) ∧
  (final_school.boarders = initial_school.boarders + new_boarders) ∧
  (final_school.day_students = initial_school.day_students) →
  (final_ratio.boarders * final_school.day_students = final_ratio.day_students * final_school.boarders) :=
by sorry

end NUMINAMATH_CALUDE_school_ratio_change_l3133_313384


namespace NUMINAMATH_CALUDE_sequence_formula_l3133_313301

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 1 - n * a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 1 - n * a n) : 
  ∀ n : ℕ+, a n = 1 / (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3133_313301


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l3133_313388

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₁ d : ℤ) : ℕ → ℤ :=
  fun n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic progression -/
def SumArithmeticProgression (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_progression_first_term
  (a₁ d : ℤ)
  (h_increasing : d > 0)
  (h_condition1 : (ArithmeticProgression a₁ d 9) * (ArithmeticProgression a₁ d 17) >
    (SumArithmeticProgression a₁ d 14) + 12)
  (h_condition2 : (ArithmeticProgression a₁ d 11) * (ArithmeticProgression a₁ d 15) <
    (SumArithmeticProgression a₁ d 14) + 47) :
  a₁ ∈ ({-9, -8, -7, -6, -4, -3, -2, -1} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l3133_313388


namespace NUMINAMATH_CALUDE_least_integer_square_75_more_than_double_l3133_313322

theorem least_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_75_more_than_double_l3133_313322


namespace NUMINAMATH_CALUDE_ratio_problem_l3133_313364

theorem ratio_problem (x y : ℤ) : 
  (y = 4 * x) →  -- The two integers are in the ratio of 1 to 4
  (x + 12 = y) →  -- Adding 12 to the smaller number makes the ratio 1 to 1
  y = 16 :=  -- The larger integer is 16
by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3133_313364


namespace NUMINAMATH_CALUDE_bird_families_left_l3133_313377

theorem bird_families_left (total : ℕ) (to_africa : ℕ) (to_asia : ℕ) 
  (h1 : total = 85) 
  (h2 : to_africa = 23) 
  (h3 : to_asia = 37) : 
  total - (to_africa + to_asia) = 25 := by
sorry

end NUMINAMATH_CALUDE_bird_families_left_l3133_313377


namespace NUMINAMATH_CALUDE_condition_equivalence_l3133_313334

theorem condition_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l3133_313334


namespace NUMINAMATH_CALUDE_eldoria_population_2070_l3133_313302

/-- The population growth function for Eldoria -/
def eldoria_population (initial_population : ℕ) (years_since_2000 : ℕ) : ℕ :=
  initial_population * (2 ^ (years_since_2000 / 15))

/-- Theorem: The population of Eldoria in 2070 is 8000 -/
theorem eldoria_population_2070 : 
  eldoria_population 500 70 = 8000 := by
  sorry

#eval eldoria_population 500 70

end NUMINAMATH_CALUDE_eldoria_population_2070_l3133_313302


namespace NUMINAMATH_CALUDE_equation_solutions_l3133_313363

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 10 ∧ x₂ = 3 - Real.sqrt 10 ∧
    x₁^2 - 6*x₁ = 1 ∧ x₂^2 - 6*x₂ = 1) ∧
  (∃ x₃ x₄ : ℝ, x₃ = 2/3 ∧ x₄ = -4 ∧
    (x₃ - 3)^2 = (2*x₃ + 1)^2 ∧ (x₄ - 3)^2 = (2*x₄ + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3133_313363


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l3133_313311

theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l3133_313311


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3133_313360

/-- Given a line in a rectangular coordinate system passing through points (-2, y), (10, 3),
    and having an x-intercept of 4, prove that the y-coordinate of the point with x-coordinate -2 is -3. -/
theorem line_y_coordinate (y : ℝ) : 
  ∃ (m b : ℝ), 
    (∀ x, y = m * x + b) ∧  -- Line equation
    (y = m * (-2) + b) ∧    -- Line passes through (-2, y)
    (3 = m * 10 + b) ∧      -- Line passes through (10, 3)
    (0 = m * 4 + b) →       -- Line has x-intercept at 4
  y = -3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3133_313360


namespace NUMINAMATH_CALUDE_number_multiples_l3133_313392

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F

def satisfies_conditions (n : ℕ) : Prop :=
  is_valid_number n ∧
  ∃ (A B C D E F : ℕ),
    4 * n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F ∧
    13 * n = F * 100000 + A * 10000 + B * 1000 + C * 100 + D * 10 + E ∧
    22 * n = C * 100000 + D * 10000 + E * 1000 + F * 100 + A * 10 + B

theorem number_multiples (n : ℕ) (h : satisfies_conditions n) :
  (∃ k : ℕ, n * k = 984126) ∧
  (∀ k : ℕ, n * k ≠ 269841) ∧
  (∀ k : ℕ, n * k ≠ 841269) :=
sorry

end NUMINAMATH_CALUDE_number_multiples_l3133_313392


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l3133_313361

/-- Time to remove wallpaper from one wall in hours -/
def time_per_wall : ℕ := 2

/-- Number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Calculates the total time to remove remaining wallpaper -/
def total_time : ℕ :=
  time_per_wall * (dining_room_walls - completed_walls) +
  time_per_wall * living_room_walls

theorem wallpaper_removal_time : total_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_removal_time_l3133_313361


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3133_313385

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (a : ℝ) 
  (hx : x ∈ Set.Icc (-π/4) (π/4)) 
  (hy : y ∈ Set.Icc (-π/4) (π/4)) 
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l3133_313385


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3133_313313

theorem absolute_value_inequality (a b c : ℝ) : 
  |a - c| < |b| → |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3133_313313


namespace NUMINAMATH_CALUDE_inequality_proof_l3133_313386

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3133_313386


namespace NUMINAMATH_CALUDE_angle_AMB_largest_l3133_313355

/-- Given a right angle XOY with OA = a and OB = b (a < b) on side OY, 
    and a point M on OX such that OM = x, 
    prove that the angle AMB is largest when x = √(ab) -/
theorem angle_AMB_largest (a b x : ℝ) (h_ab : 0 < a ∧ a < b) :
  let φ := Real.arctan ((b - a) * x / (x^2 + a * b))
  ∀ y : ℝ, y > 0 → φ ≤ Real.arctan ((b - a) * y / (y^2 + a * b)) →
  x = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_angle_AMB_largest_l3133_313355
