import Mathlib

namespace NUMINAMATH_CALUDE_eva_marks_ratio_l262_26280

/-- Represents the marks Eva scored in a subject for a semester -/
structure Marks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for both semesters -/
structure YearlyMarks where
  first_semester : Marks
  second_semester : Marks

def total_marks (ym : YearlyMarks) : ℕ :=
  ym.first_semester.maths + ym.first_semester.arts + ym.first_semester.science +
  ym.second_semester.maths + ym.second_semester.arts + ym.second_semester.science

theorem eva_marks_ratio :
  ∀ (ym : YearlyMarks),
    ym.second_semester.maths = 80 →
    ym.second_semester.arts = 90 →
    ym.second_semester.science = 90 →
    ym.first_semester.maths = ym.second_semester.maths + 10 →
    ym.first_semester.arts = ym.second_semester.arts - 15 →
    ym.first_semester.science < ym.second_semester.science →
    total_marks ym = 485 →
    ∃ (x : ℕ), 
      ym.second_semester.science - ym.first_semester.science = x ∧
      x = 30 ∧
      (x : ℚ) / ym.second_semester.science = 1 / 3 :=
by sorry


end NUMINAMATH_CALUDE_eva_marks_ratio_l262_26280


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l262_26287

/-- The number of ways to distribute n identical balls into k distinct boxes with at least one ball in each box -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 4 ways to distribute 5 identical balls into 4 distinct boxes with at least one ball in each box -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l262_26287


namespace NUMINAMATH_CALUDE_white_balls_count_l262_26240

theorem white_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 21 →
  ∃ (white : ℕ), white ≤ total ∧ 
    (white : ℚ) / total * (white - 1) / (total - 1) = prob ∧ 
    white = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l262_26240


namespace NUMINAMATH_CALUDE_sum_positive_not_sufficient_nor_necessary_for_product_positive_l262_26297

theorem sum_positive_not_sufficient_nor_necessary_for_product_positive :
  ∃ (a b : ℝ), (a + b > 0 ∧ a * b ≤ 0) ∧ ∃ (c d : ℝ), (c + d ≤ 0 ∧ c * d > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_not_sufficient_nor_necessary_for_product_positive_l262_26297


namespace NUMINAMATH_CALUDE_paint_needed_for_one_door_l262_26291

theorem paint_needed_for_one_door 
  (total_doors : ℕ) 
  (pint_cost : ℚ) 
  (gallon_cost : ℚ) 
  (pints_per_gallon : ℕ) 
  (savings : ℚ) 
  (h1 : total_doors = 8)
  (h2 : pint_cost = 8)
  (h3 : gallon_cost = 55)
  (h4 : pints_per_gallon = 8)
  (h5 : savings = 9)
  (h6 : total_doors * pint_cost - gallon_cost = savings) :
  (1 : ℚ) = pints_per_gallon / total_doors := by
sorry

end NUMINAMATH_CALUDE_paint_needed_for_one_door_l262_26291


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l262_26224

theorem polynomial_divisibility (x : ℝ) : 
  let P : ℝ → ℝ := λ x => (x + 1)^6 - x^6 - 2*x - 1
  ∃ Q : ℝ → ℝ, P x = (x * (x + 1) * (2*x + 1)) * Q x := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l262_26224


namespace NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l262_26251

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem factorial_800_trailing_zeros :
  trailingZeros 800 = 199 := by
  sorry

end NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l262_26251


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l262_26261

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 0 < x}

theorem intersection_of_intervals : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l262_26261


namespace NUMINAMATH_CALUDE_gcd_91_72_l262_26283

theorem gcd_91_72 : Nat.gcd 91 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_72_l262_26283


namespace NUMINAMATH_CALUDE_tournament_teams_l262_26259

theorem tournament_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_tournament_teams_l262_26259


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l262_26209

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 20| = |2*x - 44| :=
by
  -- The unique solution is x = 22
  use 22
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l262_26209


namespace NUMINAMATH_CALUDE_price_reduction_equation_l262_26226

theorem price_reduction_equation (initial_price final_price : ℝ) 
  (h1 : initial_price = 188) 
  (h2 : final_price = 108) 
  (x : ℝ) -- x represents the percentage of each reduction
  (h3 : x ≥ 0 ∧ x < 1) -- ensure x is a valid percentage
  (h4 : final_price = initial_price * (1 - x)^2) -- two equal reductions
  : initial_price * (1 - x)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l262_26226


namespace NUMINAMATH_CALUDE_intersection_and_center_l262_26258

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (4, 0)

-- Define the lines
def line_from_A (x : ℝ) : ℝ := x
def line_from_B (x : ℝ) : ℝ := 4 - x

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 2)

theorem intersection_and_center :
  (∀ x : ℝ, line_from_A x = line_from_B x → x = intersection_point.1) ∧
  (line_from_A intersection_point.1 = intersection_point.2) ∧
  (intersection_point.1 = (C.1 - A.1) / 2) ∧
  (intersection_point.2 = (C.2 - A.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_center_l262_26258


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l262_26222

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 5 ∧ 
  (∀ (p' q' : ℕ+), (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 5 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l262_26222


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l262_26268

theorem fraction_to_decimal : 49 / 160 = 0.30625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l262_26268


namespace NUMINAMATH_CALUDE_no_good_tetrahedron_in_good_parallelepiped_l262_26267

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

end NUMINAMATH_CALUDE_no_good_tetrahedron_in_good_parallelepiped_l262_26267


namespace NUMINAMATH_CALUDE_calculate_expression_l262_26253

theorem calculate_expression : 3.75 - 1.267 + 0.48 = 2.963 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l262_26253


namespace NUMINAMATH_CALUDE_function_properties_l262_26215

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem proves properties about its maximum value and zero points. -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ x - a * Real.exp x + b
  -- The maximum value of f occurs at ln(1/a) and equals ln(1/a) - 1 + b
  ∃ (x_max : ℝ), x_max = Real.log (1/a) ∧
    ∀ x, f x ≤ f x_max ∧ f x_max = Real.log (1/a) - 1 + b ∧
  -- If f has two distinct zero points, their sum is less than -2*ln(a)
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l262_26215


namespace NUMINAMATH_CALUDE_mars_network_connected_min_tunnels_for_connectivity_l262_26262

/-- A graph representing the Mars settlement network -/
structure MarsNetwork where
  settlements : Nat
  tunnels : Nat

/-- The property that a MarsNetwork is connected -/
def is_connected (network : MarsNetwork) : Prop :=
  network.tunnels ≥ network.settlements - 1

/-- The Mars settlement network with 2004 settlements -/
def mars_network : MarsNetwork :=
  { settlements := 2004, tunnels := 2003 }

/-- Theorem stating that the Mars network with 2003 tunnels is connected -/
theorem mars_network_connected :
  is_connected mars_network :=
sorry

/-- Theorem stating that 2003 is the minimum number of tunnels required for connectivity -/
theorem min_tunnels_for_connectivity (network : MarsNetwork) :
  network.settlements = 2004 →
  is_connected network →
  network.tunnels ≥ 2003 :=
sorry

end NUMINAMATH_CALUDE_mars_network_connected_min_tunnels_for_connectivity_l262_26262


namespace NUMINAMATH_CALUDE_arman_age_problem_l262_26228

/-- Given that Arman is six times older than his sister, his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_problem (sister_age_4_years_ago : ℕ) (arman_age sister_age : ℕ) :
  sister_age_4_years_ago = 2 →
  sister_age = sister_age_4_years_ago + 4 →
  arman_age = 6 * sister_age →
  40 - arman_age = 4 := by
sorry

end NUMINAMATH_CALUDE_arman_age_problem_l262_26228


namespace NUMINAMATH_CALUDE_train_speed_conversion_l262_26233

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 30.002399999999998

/-- Theorem stating that the train's speed in km/h is equal to 108.00863999999999 -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 108.00863999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l262_26233


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l262_26214

theorem expression_simplification_and_evaluation :
  let x : ℚ := -1
  let y : ℚ := -1/2
  let original_expression := 4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y)
  let simplified_expression := 3*x*y - y^2
  original_expression = simplified_expression ∧ simplified_expression = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l262_26214


namespace NUMINAMATH_CALUDE_stephens_number_l262_26220

theorem stephens_number : ∃! n : ℕ, 
  9000 ≤ n ∧ n ≤ 15000 ∧ 
  n % 216 = 0 ∧ 
  n % 55 = 0 ∧ 
  n = 11880 := by sorry

end NUMINAMATH_CALUDE_stephens_number_l262_26220


namespace NUMINAMATH_CALUDE_min_minutes_for_plan_c_l262_26200

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes perMinute : ℕ) : ℕ := flatFee * 100 + minutes * perMinute

/-- Checks if Plan C is cheaper than both Plan A and Plan B for a given number of minutes -/
def IsPlanCCheaper (minutes : ℕ) : Prop :=
  PlanCost 15 minutes 10 < PlanCost 0 minutes 15 ∧ 
  PlanCost 15 minutes 10 < PlanCost 25 minutes 8

theorem min_minutes_for_plan_c : ∀ m : ℕ, m ≥ 301 → IsPlanCCheaper m ∧ ∀ n : ℕ, n < 301 → ¬IsPlanCCheaper n := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_plan_c_l262_26200


namespace NUMINAMATH_CALUDE_randys_trip_length_l262_26281

theorem randys_trip_length :
  ∀ (total_length : ℚ),
    (1 / 3 : ℚ) * total_length + 20 + (1 / 5 : ℚ) * total_length = total_length →
    total_length = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_randys_trip_length_l262_26281


namespace NUMINAMATH_CALUDE_fifth_bush_berries_l262_26278

def berry_sequence : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | 2 => 7
  | 3 => 12
  | n + 4 => berry_sequence (n + 3) + (berry_sequence (n + 3) - berry_sequence (n + 2) + 2)

theorem fifth_bush_berries : berry_sequence 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bush_berries_l262_26278


namespace NUMINAMATH_CALUDE_oliver_shelf_capacity_l262_26292

/-- The number of books Oliver can fit on a shelf -/
def books_per_shelf (total_books librarian_books shelves : ℕ) : ℕ :=
  (total_books - librarian_books) / shelves

/-- Theorem: Oliver can fit 4 books on a shelf -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelf_capacity_l262_26292


namespace NUMINAMATH_CALUDE_cubic_inequality_l262_26273

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 44*x - 16 > 0 ↔ 
  (x > 4 ∧ x < 4 + 2*Real.sqrt 3) ∨ x > 4 + 2*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l262_26273


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l262_26246

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1055 ∧ m = 23) :
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l262_26246


namespace NUMINAMATH_CALUDE_pencil_and_pen_choices_l262_26274

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size : ℕ) (set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 3 and one from a set of 5 results in 15 possibilities -/
theorem pencil_and_pen_choices :
  choose_one_from_each 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_and_pen_choices_l262_26274


namespace NUMINAMATH_CALUDE_rainbow_pencils_count_l262_26206

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def num_people : ℕ := 6

/-- The total number of pencils -/
def total_pencils : ℕ := rainbow_colors * num_people

theorem rainbow_pencils_count : total_pencils = 42 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_pencils_count_l262_26206


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l262_26244

/-- 
Given two lines in the form of linear equations:
  5y + 2x - 7 = 0 and 4y + bx - 8 = 0
If these lines are perpendicular, then b = -10.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 5 * y + 2 * x - 7 = 0 ∧ 4 * y + b * x - 8 = 0) →
  ((-2/5) * (-b/4) = -1) →
  b = -10 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l262_26244


namespace NUMINAMATH_CALUDE_f_properties_l262_26201

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem f_properties :
  ∃ (p : ℝ),
    (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
    (p = 2 * Real.pi) ∧
    (∀ (x : ℝ), f x ≤ 2) ∧
    (∃ (x : ℝ), f x = 2) ∧
    (∀ (k : ℤ),
      ∀ (x : ℝ),
        (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
        (∀ (y : ℝ), x < y → f (-y) < f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l262_26201


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l262_26263

/-- The total cost of typing a manuscript with given revision requirements -/
def manuscript_typing_cost (initial_cost : ℕ) (revision_cost : ℕ) (total_pages : ℕ) 
  (once_revised : ℕ) (twice_revised : ℕ) : ℕ :=
  (initial_cost * total_pages) + 
  (revision_cost * once_revised) + 
  (2 * revision_cost * twice_revised)

/-- Theorem stating the total cost of typing the manuscript -/
theorem manuscript_cost_calculation : 
  manuscript_typing_cost 6 4 100 35 15 = 860 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l262_26263


namespace NUMINAMATH_CALUDE_line_equation_theorem_l262_26210

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if the given equation represents the line -/
def isEquationOfLine (a b c : ℝ) (l : Line) : Prop :=
  a ≠ 0 ∧ 
  l.slope = -a / b ∧
  l.yIntercept = -c / b

/-- The main theorem: the equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_theorem : 
  let l : Line := { slope := 3, yIntercept := 4 }
  isEquationOfLine 3 (-1) 4 l := by
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l262_26210


namespace NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l262_26286

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α := by
  sorry

end NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l262_26286


namespace NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l262_26275

def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

theorem intersection_empty_iff_b_in_range :
  ∀ b : ℝ, (∀ m : ℝ, M ∩ N m b = ∅) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_b_in_range_l262_26275


namespace NUMINAMATH_CALUDE_nail_polish_theorem_l262_26265

def nail_polish_problem (kim heidi karen : ℕ) : Prop :=
  kim = 25 ∧
  heidi = kim + 8 ∧
  karen = kim - 6 ∧
  heidi + karen = 52

theorem nail_polish_theorem :
  ∃ (kim heidi karen : ℕ), nail_polish_problem kim heidi karen := by
  sorry

end NUMINAMATH_CALUDE_nail_polish_theorem_l262_26265


namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l262_26249

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l262_26249


namespace NUMINAMATH_CALUDE_sum_f_neg1_0_1_l262_26212

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom f_add (x y : ℝ) : f x + f y = f (x + y)

-- State the theorem to be proved
theorem sum_f_neg1_0_1 : f (-1) + f 0 + f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_neg1_0_1_l262_26212


namespace NUMINAMATH_CALUDE_cheese_cookies_per_box_l262_26279

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- The price of a pack of cheese cookies in dollars -/
def price_per_pack : ℕ := 1

/-- The cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- The number of packs of cheese cookies in each box -/
def packs_per_box : ℕ := 10

theorem cheese_cookies_per_box :
  packs_per_box = 10 := by sorry

end NUMINAMATH_CALUDE_cheese_cookies_per_box_l262_26279


namespace NUMINAMATH_CALUDE_f_constant_iff_max_value_expression_exists_max_value_expression_l262_26254

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem f_constant_iff (x : ℝ) : (∀ y ∈ Set.Icc (-3) 1, f y = f x) ↔ x ∈ Set.Icc (-3) 1 := by sorry

-- Part 2
theorem max_value_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) :
  Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z ≤ 3 := by sorry

theorem exists_max_value_expression :
  ∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ Real.sqrt 2 * x + Real.sqrt 2 * y + Real.sqrt 5 * z = 3 := by sorry

end NUMINAMATH_CALUDE_f_constant_iff_max_value_expression_exists_max_value_expression_l262_26254


namespace NUMINAMATH_CALUDE_sequence_inequality_l262_26252

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n, n ≥ 2 ∧ n ≤ 100 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l262_26252


namespace NUMINAMATH_CALUDE_domino_coverage_iff_even_uncoverable_boards_l262_26207

/-- Represents a checkerboard -/
structure Checkerboard where
  squares : ℕ

/-- Predicate to determine if a checkerboard can be fully covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even board.squares

theorem domino_coverage_iff_even (board : Checkerboard) :
  can_be_covered board ↔ Even board.squares :=
sorry

/-- 6x4 rectangular board -/
def board_6x4 : Checkerboard :=
  ⟨6 * 4⟩

/-- 5x5 square board -/
def board_5x5 : Checkerboard :=
  ⟨5 * 5⟩

/-- L-shaped board (5x5 with 2x2 removed) -/
def board_L : Checkerboard :=
  ⟨5 * 5 - 2 * 2⟩

/-- 3x7 rectangular board -/
def board_3x7 : Checkerboard :=
  ⟨3 * 7⟩

/-- Plus-shaped board (3x3 with 1x3 extension) -/
def board_plus : Checkerboard :=
  ⟨3 * 3 + 1 * 3⟩

theorem uncoverable_boards :
  ¬can_be_covered board_5x5 ∧
  ¬can_be_covered board_L ∧
  ¬can_be_covered board_3x7 :=
sorry

end NUMINAMATH_CALUDE_domino_coverage_iff_even_uncoverable_boards_l262_26207


namespace NUMINAMATH_CALUDE_batsman_average_increase_l262_26250

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalRuns : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalRuns + newScore) / (prevInnings + 1)
  let prevAverage := prevTotalRuns / prevInnings
  newAverage - prevAverage

/-- Theorem: The batsman's average increased by 3 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 12 →
    b.average = 47 →
    averageIncrease 11 (11 * (b.totalRuns / 11)) 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l262_26250


namespace NUMINAMATH_CALUDE_canada_sqft_per_person_approx_l262_26282

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

end NUMINAMATH_CALUDE_canada_sqft_per_person_approx_l262_26282


namespace NUMINAMATH_CALUDE_parabola_transformation_l262_26276

/-- The original parabola function -/
def f (x : ℝ) : ℝ := -(x + 3) * (x - 2)

/-- The transformed parabola function -/
def g (x : ℝ) : ℝ := -(x - 3) * (x + 2)

/-- The transformation function -/
def T (x : ℝ) : ℝ := x + 1

theorem parabola_transformation :
  ∀ x : ℝ, f x = g (T x) :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l262_26276


namespace NUMINAMATH_CALUDE_student_tickets_sold_l262_26227

theorem student_tickets_sold (total_tickets : ℕ) (student_price non_student_price total_money : ℚ)
  (h1 : total_tickets = 193)
  (h2 : student_price = 1/2)
  (h3 : non_student_price = 3/2)
  (h4 : total_money = 206.5)
  (h5 : ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_tickets * student_price + non_student_tickets * non_student_price = total_money) :
  ∃ (student_tickets : ℕ), student_tickets = 83 :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l262_26227


namespace NUMINAMATH_CALUDE_carl_stamps_l262_26243

/-- Given that Kevin has 57 stamps and Carl has 32 more stamps than Kevin, 
    prove that Carl has 89 stamps. -/
theorem carl_stamps (kevin_stamps : ℕ) (carl_extra_stamps : ℕ) : 
  kevin_stamps = 57 → 
  carl_extra_stamps = 32 → 
  kevin_stamps + carl_extra_stamps = 89 := by
sorry

end NUMINAMATH_CALUDE_carl_stamps_l262_26243


namespace NUMINAMATH_CALUDE_matrix_condition_l262_26260

variable (a b c d : ℂ)

def N : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_condition (h1 : N a b c d ^ 2 = 1) (h2 : a * b * c * d = 1) :
  a^4 + b^4 + c^4 + d^4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_matrix_condition_l262_26260


namespace NUMINAMATH_CALUDE_stadium_height_l262_26236

/-- The height of a rectangular stadium given its length, width, and the length of the longest pole that can fit diagonally. -/
theorem stadium_height (length width diagonal : ℝ) (h1 : length = 24) (h2 : width = 18) (h3 : diagonal = 34) :
  Real.sqrt (diagonal^2 - length^2 - width^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_stadium_height_l262_26236


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l262_26294

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

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l262_26294


namespace NUMINAMATH_CALUDE_percentage_problem_l262_26295

theorem percentage_problem (x : ℝ) (h : 24 = (75 / 100) * x) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l262_26295


namespace NUMINAMATH_CALUDE_reservoir_percentage_before_storm_l262_26256

-- Define the reservoir capacity in billion gallons
def reservoir_capacity : ℝ := 550

-- Define the original contents in billion gallons
def original_contents : ℝ := 220

-- Define the amount of water added by the storm in billion gallons
def storm_water : ℝ := 110

-- Define the percentage full after the storm
def post_storm_percentage : ℝ := 0.60

-- Theorem to prove
theorem reservoir_percentage_before_storm :
  (original_contents / reservoir_capacity) * 100 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_reservoir_percentage_before_storm_l262_26256


namespace NUMINAMATH_CALUDE_friday_temperature_l262_26299

theorem friday_temperature
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 46)
  (h3 : temp_mon = 43)
  : temp_fri = 35 := by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l262_26299


namespace NUMINAMATH_CALUDE_greatest_NPM_value_l262_26239

theorem greatest_NPM_value : ∀ M N P : ℕ,
  (M ≥ 1 ∧ M ≤ 9) →  -- M is a one-digit integer
  (N ≥ 1 ∧ N ≤ 9) →  -- N is a one-digit integer
  (P ≥ 0 ∧ P ≤ 9) →  -- P is a one-digit integer
  (10 * M + M) * M = 100 * N + 10 * P + M →  -- MM * M = NPM
  100 * N + 10 * P + M ≤ 396 :=
by sorry

end NUMINAMATH_CALUDE_greatest_NPM_value_l262_26239


namespace NUMINAMATH_CALUDE_circle_equation_l262_26270

/-- Prove that a circle with given properties has the equation (x+5)^2 + y^2 = 5 -/
theorem circle_equation (a : ℝ) (h1 : a < 0) :
  let O' : ℝ × ℝ := (a, 0)
  let r : ℝ := Real.sqrt 5
  let line : ℝ × ℝ → Prop := λ p => p.1 + 2 * p.2 = 0
  (∀ p, line p → (p.1 - O'.1)^2 + (p.2 - O'.2)^2 = r^2) →
  (∀ x y, (x + 5)^2 + y^2 = 5 ↔ (x - a)^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l262_26270


namespace NUMINAMATH_CALUDE_range_of_m_l262_26234

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  ((∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) → m ≥ 9) ∧
  ((∀ x, ¬q x m → ¬p x) ∧ (∃ x, ¬p x ∧ q x m) → 0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l262_26234


namespace NUMINAMATH_CALUDE_integral_split_l262_26218

-- Define f as a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem integral_split (h : ∫ x in (1:ℝ)..(3:ℝ), f x = 56) :
  ∫ x in (1:ℝ)..(2:ℝ), f x + ∫ x in (2:ℝ)..(3:ℝ), f x = 56 := by
  sorry

end NUMINAMATH_CALUDE_integral_split_l262_26218


namespace NUMINAMATH_CALUDE_largest_a_for_integer_solution_l262_26213

theorem largest_a_for_integer_solution : 
  ∃ (a : ℝ), ∀ (b : ℝ), 
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b*x + 3*y = 1 → x - 4*y = 1 → b ≤ a) ∧
    a = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_for_integer_solution_l262_26213


namespace NUMINAMATH_CALUDE_system_two_solutions_l262_26289

/-- The system of equations has exactly two solutions iff a = 4 or a = 100 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x y : ℝ), |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ∧
  (∃! (x' y' : ℝ), (x', y') ≠ (x, y) ∧ |x' - 6 - y'| + |x' - 6 + y'| = 12 ∧ (|x'| - 6)^2 + (|y'| - 8)^2 = a) ↔
  (a = 4 ∨ a = 100) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l262_26289


namespace NUMINAMATH_CALUDE_matrix_power_in_M_l262_26232

/-- The set M of 2x2 complex matrices where ab = cd -/
def M : Set (Matrix (Fin 2) (Fin 2) ℂ) :=
  {A | A 0 0 * A 0 1 = A 1 0 * A 1 1}

/-- Theorem statement -/
theorem matrix_power_in_M
  (A : Matrix (Fin 2) (Fin 2) ℂ)
  (k : ℕ)
  (hk : k ≥ 1)
  (hA : A ∈ M)
  (hAk : A ^ k ∈ M)
  (hAk1 : A ^ (k + 1) ∈ M)
  (hAk2 : A ^ (k + 2) ∈ M) :
  ∀ n : ℕ, n ≥ 1 → A ^ n ∈ M :=
sorry

end NUMINAMATH_CALUDE_matrix_power_in_M_l262_26232


namespace NUMINAMATH_CALUDE_choose_five_from_ten_l262_26225

theorem choose_five_from_ten : Nat.choose 10 5 = 252 := by sorry

end NUMINAMATH_CALUDE_choose_five_from_ten_l262_26225


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_equals_binomial_l262_26269

theorem largest_n_binomial_sum_equals_binomial (n : ℕ) : 
  (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) → n ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_equals_binomial_l262_26269


namespace NUMINAMATH_CALUDE_remainder_451951_div_5_l262_26255

theorem remainder_451951_div_5 : 451951 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_451951_div_5_l262_26255


namespace NUMINAMATH_CALUDE_three_balls_per_can_l262_26290

/-- Represents a tennis tournament with a given structure and ball usage. -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  total_balls : Nat

/-- Calculates the number of tennis balls per can in a given tournament. -/
def balls_per_can (t : TennisTournament) : Nat :=
  let total_games := t.games_per_round.sum
  let total_cans := total_games * t.cans_per_game
  t.total_balls / total_cans

/-- Theorem stating that for the given tournament structure, there are 3 balls per can. -/
theorem three_balls_per_can :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    total_balls := 225
  }
  balls_per_can t = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_balls_per_can_l262_26290


namespace NUMINAMATH_CALUDE_f_properties_l262_26219

noncomputable section

def f (x : ℝ) := (Real.log x) / x

theorem f_properties :
  ∀ x > 0,
  (∃ y, f x = y ∧ x - y - 1 = 0) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (f (Real.exp 1) = (Real.exp 1)⁻¹) ∧
  (∀ x, x > 0 → f x ≤ (Real.exp 1)⁻¹) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l262_26219


namespace NUMINAMATH_CALUDE_range_of_p_l262_26247

-- Define the function p(x)
def p (x : ℝ) : ℝ := (x^3 + 3)^2

-- Define the domain of p(x)
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- Define the range of p(x)
def range : Set ℝ := {y : ℝ | ∃ x ∈ domain, p x = y}

-- Theorem statement
theorem range_of_p : range = {y : ℝ | y ≥ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_p_l262_26247


namespace NUMINAMATH_CALUDE_factor_expression_l262_26288

theorem factor_expression (x : ℝ) : 60 * x^5 - 180 * x^9 = 60 * x^5 * (1 - 3 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l262_26288


namespace NUMINAMATH_CALUDE_solve_for_y_l262_26284

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l262_26284


namespace NUMINAMATH_CALUDE_m_range_l262_26221

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : p m ∧ q m → m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l262_26221


namespace NUMINAMATH_CALUDE_clock_adjustment_l262_26248

/-- Represents the number of minutes lost per day by the clock -/
def minutes_lost_per_day : ℕ := 3

/-- Represents the number of days between March 15 1 P.M. and March 22 9 A.M. -/
def days_elapsed : ℕ := 7

/-- Represents the total number of minutes lost by the clock -/
def total_minutes_lost : ℕ := minutes_lost_per_day * days_elapsed

theorem clock_adjustment :
  total_minutes_lost = 21 := by sorry

end NUMINAMATH_CALUDE_clock_adjustment_l262_26248


namespace NUMINAMATH_CALUDE_board_sum_possible_l262_26237

theorem board_sum_possible : ∃ (a b : ℕ), 
  a ≤ 10 ∧ b ≤ 11 ∧ 
  (10 - a : ℝ) * 1.11 + (11 - b : ℝ) * 1.01 = 20.19 := by
sorry

end NUMINAMATH_CALUDE_board_sum_possible_l262_26237


namespace NUMINAMATH_CALUDE_olympic_medals_l262_26238

theorem olympic_medals (total gold silver bronze : ℕ) : 
  total = 89 → 
  gold + silver = 4 * bronze - 6 → 
  gold + silver + bronze = total → 
  bronze = 19 := by
sorry

end NUMINAMATH_CALUDE_olympic_medals_l262_26238


namespace NUMINAMATH_CALUDE_child_ticket_cost_l262_26230

/-- Proves that the cost of a child ticket is 25 cents given the specified conditions. -/
theorem child_ticket_cost
  (adult_price : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 60)
  (h2 : total_attendees = 280)
  (h3 : total_revenue = 14000)  -- in cents
  (h4 : num_children = 80) :
  ∃ (child_price : ℕ),
    child_price * num_children + adult_price * (total_attendees - num_children) = total_revenue ∧
    child_price = 25 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l262_26230


namespace NUMINAMATH_CALUDE_smallest_divisible_ones_l262_26277

/-- A number composed of n ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- A number composed of n threes -/
def threes (n : ℕ) : ℕ := 3 * ones n

theorem smallest_divisible_ones (n : ℕ) : 
  (∀ k < n, ¬ (threes 100 ∣ ones k)) ∧ (threes 100 ∣ ones n) → n = 300 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_ones_l262_26277


namespace NUMINAMATH_CALUDE_composite_sum_of_power_l262_26204

theorem composite_sum_of_power (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_power_l262_26204


namespace NUMINAMATH_CALUDE_matchsticks_left_l262_26271

/-- Calculates the number of matchsticks left in a box after Elvis and Ralph make their squares -/
theorem matchsticks_left (initial_count : ℕ) (elvis_square_size elvis_squares : ℕ) (ralph_square_size ralph_squares : ℕ) : 
  initial_count = 50 → 
  elvis_square_size = 4 → 
  ralph_square_size = 8 → 
  elvis_squares = 5 → 
  ralph_squares = 3 → 
  initial_count - (elvis_square_size * elvis_squares + ralph_square_size * ralph_squares) = 6 := by
sorry

end NUMINAMATH_CALUDE_matchsticks_left_l262_26271


namespace NUMINAMATH_CALUDE_find_x_value_l262_26229

theorem find_x_value (x : ℝ) :
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.9365079365079367) →
  x = 1.21 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l262_26229


namespace NUMINAMATH_CALUDE_task_selection_count_l262_26203

def num_males : ℕ := 3
def num_females : ℕ := 3
def total_students : ℕ := num_males + num_females
def num_selected : ℕ := 4

def num_single_person_tasks : ℕ := 2
def num_two_person_tasks : ℕ := 1

def selection_methods : ℕ := 144

theorem task_selection_count :
  (num_males = 3) →
  (num_females = 3) →
  (total_students = num_males + num_females) →
  (num_selected = 4) →
  (num_single_person_tasks = 2) →
  (num_two_person_tasks = 1) →
  selection_methods = 144 := by
  sorry

end NUMINAMATH_CALUDE_task_selection_count_l262_26203


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l262_26205

/-- Given a cuboid with edges x, 5, and 6, and volume 120, prove x = 4 -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 120 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l262_26205


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l262_26223

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.cos t.B = t.b * Real.sin t.A)
  (h2 : (Real.sqrt 3 / 4) * t.b^2 = (1/2) * t.a * t.c * Real.sin t.B) :
  t.a / t.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l262_26223


namespace NUMINAMATH_CALUDE_binary_1101001101_equals_base4_311310_l262_26272

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_311310 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 1, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101001101_equals_base4_311310_l262_26272


namespace NUMINAMATH_CALUDE_counterexample_exists_l262_26264

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) ∧ 
  (∃ k : ℕ, n = 3 * k) ∧ 
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n - 2 = x * y) ∧ 
  n - 2 ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l262_26264


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l262_26211

/-- Yan's problem statement -/
theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
  w > 0 →  -- walking speed is positive
  x > 0 →  -- distance from Yan to home is positive
  y > 0 →  -- distance from Yan to stadium is positive
  x + y > 0 →  -- Yan is between home and stadium
  y / w = (x / w + (x + y) / (9 * w)) →  -- both choices take the same time
  x / y = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l262_26211


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_fourth_l262_26298

theorem units_digit_of_six_to_fourth (n : ℕ) : n = 6^4 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_fourth_l262_26298


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l262_26257

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The midpoints of the legs
  m : ℝ × ℝ
  n : ℝ × ℝ
  -- Conditions
  right_angle : a^2 + b^2 = (a + b)^2 / 2
  m_midpoint : m = (a/2, 0)
  n_midpoint : n = (0, b/2)
  xn_length : a^2 + (b/2)^2 = 22^2
  ym_length : b^2 + (a/2)^2 = 31^2

/-- The theorem to be proved -/
theorem right_triangle_hypotenuse (t : RightTriangle) : 
  Real.sqrt (t.a^2 + t.b^2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l262_26257


namespace NUMINAMATH_CALUDE_probability_of_not_hearing_favorite_song_l262_26241

def num_songs : ℕ := 12
def shortest_song : ℕ := 45  -- in seconds
def song_increment : ℕ := 15 -- in seconds
def favorite_song_length : ℕ := 240 -- 4 minutes in seconds
def time_window : ℕ := 300 -- 5 minutes in seconds

def song_length (n : ℕ) : ℕ :=
  shortest_song + n * song_increment

def songs_shorter_than_favorite : Finset ℕ :=
  Finset.filter (fun i => song_length i < favorite_song_length) (Finset.range num_songs)

def probability_not_hearing_favorite : ℚ :=
  1 - (3 : ℚ) / (num_songs * (num_songs - 1))

theorem probability_of_not_hearing_favorite_song :
  probability_not_hearing_favorite = 43 / 44 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_not_hearing_favorite_song_l262_26241


namespace NUMINAMATH_CALUDE_sock_drawer_problem_l262_26217

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose a pair of socks of the same color -/
def sameColorPairs (white brown blue red : ℕ) : ℕ :=
  choose white 2 + choose brown 2 + choose blue 2 + choose red 2

theorem sock_drawer_problem :
  sameColorPairs 5 5 3 2 = 24 := by sorry

end NUMINAMATH_CALUDE_sock_drawer_problem_l262_26217


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l262_26235

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a^2 + c^2 = b^2 + Real.sqrt 2 * a * c →
  c = Real.sqrt 3 + 1 →
  Real.sin A = 1/2 →
  (B = π/4 ∧ 1/2 * a * b * Real.sin C = (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l262_26235


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_a_l262_26208

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem for the range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, on_circle x y → -Real.sqrt 5 + 1 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 5 + 1 := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  (∀ x y : ℝ, on_circle x y → ∀ a : ℝ, x + y + a ≥ 0) →
  ∀ a : ℝ, a ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_a_l262_26208


namespace NUMINAMATH_CALUDE_base6_multiplication_l262_26293

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

end NUMINAMATH_CALUDE_base6_multiplication_l262_26293


namespace NUMINAMATH_CALUDE_quadratic_inequality_proof_l262_26231

theorem quadratic_inequality_proof (a : ℝ) 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  (0 < a ∧ a < 1) ∧ 
  (∀ x : ℝ, (a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1) ↔ x > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_proof_l262_26231


namespace NUMINAMATH_CALUDE_linear_function_inequality_solution_l262_26296

/-- Given a linear function y = kx + b, prove that under certain conditions, 
    the solution set of an inequality is x < 1 -/
theorem linear_function_inequality_solution 
  (k b n : ℝ) 
  (h_k : k ≠ 0)
  (h_n : n > 2)
  (h_y_neg1 : k * (-1) + b = n)
  (h_y_1 : k * 1 + b = 2) :
  {x : ℝ | (k - 2) * x + b > 0} = {x : ℝ | x < 1} := by
sorry

end NUMINAMATH_CALUDE_linear_function_inequality_solution_l262_26296


namespace NUMINAMATH_CALUDE_basketball_conference_games_l262_26266

/-- The number of divisions in the basketball conference -/
def num_divisions : ℕ := 3

/-- The number of teams in each division -/
def teams_per_division : ℕ := 4

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams from other divisions -/
def inter_division_games : ℕ := 2

/-- The total number of scheduled games in the basketball conference -/
def total_games : ℕ := 150

theorem basketball_conference_games :
  (num_divisions * (teams_per_division.choose 2) * intra_division_games) +
  (num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games / 2) = total_games :=
by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l262_26266


namespace NUMINAMATH_CALUDE_tan_value_second_quadrant_l262_26216

/-- Given that α is an angle in the second quadrant and sin(π - α) = 3/5, prove that tan(α) = -3/4 -/
theorem tan_value_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin (π - α) = 3/5) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_second_quadrant_l262_26216


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l262_26285

/-- Conversion of rectangular coordinates (8, 2√6) to polar coordinates (r, θ) -/
theorem rect_to_polar_conversion :
  ∃ (r θ : ℝ), 
    r = 2 * Real.sqrt 22 ∧ 
    Real.tan θ = Real.sqrt 6 / 4 ∧ 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l262_26285


namespace NUMINAMATH_CALUDE_divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l262_26242

/-- Represents the number 2013 repeated n times -/
def repeat_2013 (n : ℕ) : ℕ :=
  2013 * (10 ^ (4 * n) - 1) / 9

/-- The sum of digits of 2013 -/
def sum_of_digits_2013 : ℕ := 2 + 0 + 1 + 3

theorem divisible_by_9_when_repeated_thrice :
  ∃ k : ℕ, repeat_2013 3 = 9 * k :=
sorry

/-- The resulting number when 2013 is repeated 3 times is divisible by 9 -/
theorem repeat_2013_thrice_divisible_by_9 :
  9 ∣ repeat_2013 3 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_9_when_repeated_thrice_repeat_2013_thrice_divisible_by_9_l262_26242


namespace NUMINAMATH_CALUDE_sqrt_real_range_l262_26245

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y^2 = 1 - 3*x) ↔ x ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l262_26245


namespace NUMINAMATH_CALUDE_inequality_proof_l262_26202

theorem inequality_proof (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l262_26202
