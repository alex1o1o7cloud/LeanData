import Mathlib

namespace NUMINAMATH_CALUDE_teresa_social_studies_score_l3292_329276

/-- Teresa's exam scores -/
structure ExamScores where
  science : ℕ
  music : ℕ
  physics : ℕ
  social_studies : ℕ
  total : ℕ

/-- Theorem: Given Teresa's exam scores satisfying certain conditions, her social studies score is 85 -/
theorem teresa_social_studies_score (scores : ExamScores) 
  (h1 : scores.science = 70)
  (h2 : scores.music = 80)
  (h3 : scores.physics = scores.music / 2)
  (h4 : scores.total = 275)
  (h5 : scores.total = scores.science + scores.music + scores.physics + scores.social_studies) :
  scores.social_studies = 85 := by
    sorry

#check teresa_social_studies_score

end NUMINAMATH_CALUDE_teresa_social_studies_score_l3292_329276


namespace NUMINAMATH_CALUDE_dwarf_milk_problem_l3292_329254

/-- Represents the amount of milk in each cup after a dwarf pours -/
def milk_distribution (initial_amount : ℚ) (k : Fin 7) : ℚ :=
  initial_amount * k / 6

/-- The total amount of milk after all distributions -/
def total_milk (initial_amount : ℚ) : ℚ :=
  (Finset.sum Finset.univ (milk_distribution initial_amount)) + initial_amount

theorem dwarf_milk_problem (initial_amount : ℚ) :
  (∀ (k : Fin 7), milk_distribution initial_amount k ≤ initial_amount) →
  total_milk initial_amount = 3 →
  initial_amount = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_dwarf_milk_problem_l3292_329254


namespace NUMINAMATH_CALUDE_quarter_point_quadrilateral_area_is_3_plus_2root2_l3292_329274

/-- Regular octagon with apothem 2 -/
structure RegularOctagon :=
  (apothem : ℝ)
  (is_regular : apothem = 2)

/-- Quarter point on a side of the octagon -/
def quarter_point (O : RegularOctagon) (i : Fin 8) : ℝ × ℝ := sorry

/-- The area of the quadrilateral formed by connecting quarter points -/
def quarter_point_quadrilateral_area (O : RegularOctagon) : ℝ :=
  let Q1 := quarter_point O 0
  let Q3 := quarter_point O 2
  let Q5 := quarter_point O 4
  let Q7 := quarter_point O 6
  sorry -- Area calculation

/-- Theorem: The area of the quadrilateral formed by connecting
    the quarter points of every other side of a regular octagon
    with apothem 2 is 3 + 2√2 -/
theorem quarter_point_quadrilateral_area_is_3_plus_2root2 (O : RegularOctagon) :
  quarter_point_quadrilateral_area O = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_point_quadrilateral_area_is_3_plus_2root2_l3292_329274


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3292_329248

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^5 + 4 = (X - 3)^2 * q + r ∧ 
  r = 331 * X - 746 ∧
  r.degree < 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3292_329248


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3292_329268

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_receives_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_receives_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define the property of a valid distribution
def valid_distribution (d : Distribution) : Prop :=
  ∀ (c : Card), ∃! (p : Person), d p = c

theorem events_mutually_exclusive_but_not_opposite :
  (∀ (d : Distribution), valid_distribution d →
    ¬(A_receives_red d ∧ B_receives_red d)) ∧
  (∃ (d : Distribution), valid_distribution d ∧
    ¬A_receives_red d ∧ ¬B_receives_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3292_329268


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l3292_329270

theorem smallest_integers_difference : ℕ → Prop := fun n =>
  (∃ m : ℕ, m > 1 ∧ 
    (∀ k : ℕ, 2 ≤ k → k ≤ 13 → m % k = 1) ∧
    (∀ j : ℕ, j > 1 → 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → j % k = 1) → 
      j ≥ m) ∧
    (∃ p : ℕ, p > m ∧ 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → p % k = 1) ∧
      (∀ q : ℕ, q > m → 
        (∀ k : ℕ, 2 ≤ k → k ≤ 13 → q % k = 1) → 
        q ≥ p) ∧
      p - m = n)) →
  n = 360360

theorem smallest_integers_difference_exists : 
  ∃ n : ℕ, smallest_integers_difference n := by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l3292_329270


namespace NUMINAMATH_CALUDE_morning_sodas_count_l3292_329218

theorem morning_sodas_count (afternoon_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : afternoon_sodas = 19)
  (h2 : total_sodas = 96) :
  total_sodas - afternoon_sodas = 77 := by
  sorry

end NUMINAMATH_CALUDE_morning_sodas_count_l3292_329218


namespace NUMINAMATH_CALUDE_square_difference_divided_by_three_l3292_329267

theorem square_difference_divided_by_three : (121^2 - 112^2) / 3 = 699 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_three_l3292_329267


namespace NUMINAMATH_CALUDE_not_divisible_by_4_or_6_count_l3292_329212

theorem not_divisible_by_4_or_6_count : ℕ :=
  let range := Finset.range 1000
  let not_div_4_or_6 := {n : ℕ | n ∈ range ∧ n % 4 ≠ 0 ∧ n % 6 ≠ 0}
  667

#check not_divisible_by_4_or_6_count

end NUMINAMATH_CALUDE_not_divisible_by_4_or_6_count_l3292_329212


namespace NUMINAMATH_CALUDE_bmw_sales_l3292_329260

/-- Proves that the number of BMWs sold is 135 given the specified conditions -/
theorem bmw_sales (total : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ)
  (h_total : total = 300)
  (h_audi : audi_percent = 12 / 100)
  (h_toyota : toyota_percent = 25 / 100)
  (h_acura : acura_percent = 18 / 100)
  (h_sum : audi_percent + toyota_percent + acura_percent < 1) :
  ↑total * (1 - (audi_percent + toyota_percent + acura_percent)) = 135 := by
  sorry


end NUMINAMATH_CALUDE_bmw_sales_l3292_329260


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3292_329230

/-- A curve defined by y = -x³ + 2 -/
def curve (x : ℝ) : ℝ := -x^3 + 2

/-- A line defined by y = -6x + b -/
def line (b : ℝ) (x : ℝ) : ℝ := -6*x + b

/-- The derivative of the curve -/
def curve_derivative (x : ℝ) : ℝ := -3*x^2

theorem tangent_line_b_value :
  ∀ b : ℝ,
  (∃ x : ℝ, curve x = line b x ∧ curve_derivative x = -6) →
  (b = 2 + 4 * Real.sqrt 2 ∨ b = 2 - 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l3292_329230


namespace NUMINAMATH_CALUDE_direct_proportion_points_l3292_329244

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_points : 
  ∀ (f : ℝ → ℝ), 
  (∃ k : ℝ, ∀ x, f x = k * x) → -- f is a direct proportion function
  f (-1) = 2 →                  -- f passes through (-1, 2)
  f 1 = -2                      -- f passes through (1, -2)
:= by sorry

end NUMINAMATH_CALUDE_direct_proportion_points_l3292_329244


namespace NUMINAMATH_CALUDE_rock_climbing_participants_number_of_rock_climbing_participants_l3292_329204

/- Define the total number of students in the school -/
def total_students : ℕ := 800

/- Define the percentage of students who went on the camping trip -/
def camping_percentage : ℚ := 25 / 100

/- Define the percentage of camping students who took more than $100 -/
def more_than_100_percentage : ℚ := 15 / 100

/- Define the percentage of camping students who took exactly $100 -/
def exactly_100_percentage : ℚ := 30 / 100

/- Define the percentage of camping students who took between $50 and $100 -/
def between_50_and_100_percentage : ℚ := 40 / 100

/- Define the percentage of students with more than $100 who participated in rock climbing -/
def rock_climbing_participation_percentage : ℚ := 50 / 100

/- Theorem stating the number of students who participated in rock climbing -/
theorem rock_climbing_participants : ℕ := by
  sorry

/- Main theorem to prove -/
theorem number_of_rock_climbing_participants : rock_climbing_participants = 15 := by
  sorry

end NUMINAMATH_CALUDE_rock_climbing_participants_number_of_rock_climbing_participants_l3292_329204


namespace NUMINAMATH_CALUDE_cube_preserves_order_l3292_329271

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l3292_329271


namespace NUMINAMATH_CALUDE_middle_group_frequency_l3292_329275

theorem middle_group_frequency 
  (n : ℕ) 
  (total_area : ℝ) 
  (middle_area : ℝ) 
  (sample_size : ℕ) 
  (h1 : n > 0) 
  (h2 : middle_area = (1 / 5) * (total_area - middle_area)) 
  (h3 : sample_size = 300) : 
  (middle_area / total_area) * sample_size = 50 := by
sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l3292_329275


namespace NUMINAMATH_CALUDE_prove_nested_max_min_l3292_329237

/-- Given distinct real numbers p, q, r, s, t satisfying p < q < r < s < t,
    prove that M(M(p, m(q, s)), m(r, m(p, t))) = q -/
theorem prove_nested_max_min (p q r s t : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) 
  (h_order : p < q ∧ q < r ∧ r < s ∧ s < t) : 
  max (max p (min q s)) (min r (min p t)) = q := by
  sorry

end NUMINAMATH_CALUDE_prove_nested_max_min_l3292_329237


namespace NUMINAMATH_CALUDE_interest_frequency_proof_l3292_329291

/-- The nominal interest rate per annum -/
def nominal_rate : ℝ := 0.10

/-- The effective annual rate -/
def effective_annual_rate : ℝ := 0.1025

/-- The frequency of interest payment (number of compounding periods per year) -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency results in the correct effective annual rate -/
theorem interest_frequency_proof :
  (1 + nominal_rate / frequency) ^ frequency - 1 = effective_annual_rate :=
by sorry

end NUMINAMATH_CALUDE_interest_frequency_proof_l3292_329291


namespace NUMINAMATH_CALUDE_complex_multiplication_l3292_329277

theorem complex_multiplication : (Complex.I : ℂ) * (1 - Complex.I) = 1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3292_329277


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l3292_329246

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 2

-- State the theorem
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 0, StrictMonoOn f (Set.Ioo (-2 : ℝ) 0) := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l3292_329246


namespace NUMINAMATH_CALUDE_julia_monday_playmates_l3292_329206

/-- The number of kids Julia played with on different days -/
structure JuliaPlaymates where
  wednesday : ℕ
  monday : ℕ

/-- Given conditions about Julia's playmates -/
def julia_conditions (j : JuliaPlaymates) : Prop :=
  j.wednesday = 4 ∧ j.monday = j.wednesday + 2

/-- Theorem: Julia played with 6 kids on Monday -/
theorem julia_monday_playmates (j : JuliaPlaymates) (h : julia_conditions j) : j.monday = 6 := by
  sorry

end NUMINAMATH_CALUDE_julia_monday_playmates_l3292_329206


namespace NUMINAMATH_CALUDE_harry_green_weights_l3292_329283

/-- Represents the weight configuration of Harry's custom creation at the gym -/
structure WeightConfiguration where
  blue_weight : ℕ        -- Weight of each blue weight in pounds
  green_weight : ℕ       -- Weight of each green weight in pounds
  bar_weight : ℕ         -- Weight of the bar in pounds
  num_blue : ℕ           -- Number of blue weights used
  total_weight : ℕ       -- Total weight of the custom creation in pounds

/-- Calculates the number of green weights in Harry's custom creation -/
def num_green_weights (config : WeightConfiguration) : ℕ :=
  (config.total_weight - config.bar_weight - config.num_blue * config.blue_weight) / config.green_weight

/-- Theorem stating that Harry put 5 green weights on the bar -/
theorem harry_green_weights :
  let config : WeightConfiguration := {
    blue_weight := 2,
    green_weight := 3,
    bar_weight := 2,
    num_blue := 4,
    total_weight := 25
  }
  num_green_weights config = 5 := by
  sorry

end NUMINAMATH_CALUDE_harry_green_weights_l3292_329283


namespace NUMINAMATH_CALUDE_shifted_quadratic_equation_solutions_l3292_329210

/-- Given an equation a(x+m)²+b=0 with solutions x₁=-2 and x₂=1, 
    prove that a(x+m+2)²+b=0 has solutions x₁=-4 and x₂=-1 -/
theorem shifted_quadratic_equation_solutions 
  (a m b : ℝ) 
  (ha : a ≠ 0) 
  (h1 : a * ((-2 : ℝ) + m)^2 + b = 0) 
  (h2 : a * ((1 : ℝ) + m)^2 + b = 0) :
  a * ((-4 : ℝ) + m + 2)^2 + b = 0 ∧ a * ((-1 : ℝ) + m + 2)^2 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_equation_solutions_l3292_329210


namespace NUMINAMATH_CALUDE_transform_F_coordinates_l3292_329217

/-- Reflects a point over the x-axis -/
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (3, -1)

theorem transform_F_coordinates :
  (rotate_90_ccw (reflect_over_x F)) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_transform_F_coordinates_l3292_329217


namespace NUMINAMATH_CALUDE_smallest_X_l3292_329228

/-- A function that checks if a natural number consists only of 0s and 1s --/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting of only 0s and 1s that is divisible by 18 --/
def T : ℕ := 111111111000

/-- X is defined as T divided by 18 --/
def X : ℕ := T / 18

/-- Main theorem: X is the smallest positive integer satisfying the given conditions --/
theorem smallest_X : 
  (onlyZerosAndOnes T) ∧ 
  (X * 18 = T) ∧ 
  (∀ Y : ℕ, (∃ S : ℕ, onlyZerosAndOnes S ∧ Y * 18 = S) → X ≤ Y) ∧
  X = 6172839500 := by sorry

end NUMINAMATH_CALUDE_smallest_X_l3292_329228


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3292_329297

theorem trigonometric_inequality (a b A G : ℝ) : 
  a = Real.sin (π / 3) →
  b = Real.cos (π / 3) →
  A = (a + b) / 2 →
  G = Real.sqrt (a * b) →
  b < G ∧ G < A ∧ A < a :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3292_329297


namespace NUMINAMATH_CALUDE_monica_savings_l3292_329236

theorem monica_savings (weekly_saving : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) 
  (h1 : weekly_saving = 15)
  (h2 : weeks_per_cycle = 60)
  (h3 : num_cycles = 5) :
  weekly_saving * weeks_per_cycle * num_cycles = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l3292_329236


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3292_329240

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(x) = 1 - x²/2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = 1 - x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3292_329240


namespace NUMINAMATH_CALUDE_margin_formula_in_terms_of_selling_price_l3292_329235

/-- Prove that the margin formula can be expressed in terms of selling price -/
theorem margin_formula_in_terms_of_selling_price 
  (n : ℝ) (C S M : ℝ) 
  (h1 : M = (C + S) / n) 
  (h2 : M = S - C) : 
  M = 2 * S / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_margin_formula_in_terms_of_selling_price_l3292_329235


namespace NUMINAMATH_CALUDE_expression_evaluation_l3292_329296

theorem expression_evaluation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3292_329296


namespace NUMINAMATH_CALUDE_count_triangles_in_polygon_l3292_329209

/-- The number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed by diagonals
    and whose vertices are vertices of the polygon -/
def triangles_in_polygon (n : ℕ) : ℕ :=
  n * (n - 4) * (n - 5) / 6

/-- Theorem stating the number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed
    by diagonals and whose vertices are vertices of the polygon -/
theorem count_triangles_in_polygon (n : ℕ) (h : n ≥ 6) :
  triangles_in_polygon n = n * (n - 4) * (n - 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_in_polygon_l3292_329209


namespace NUMINAMATH_CALUDE_new_person_weight_l3292_329266

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 70)
  (h3 : avg_increase = 2.5) :
  let new_weight := replaced_weight + n * avg_increase
  new_weight = 90 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3292_329266


namespace NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3292_329202

/-- Proves that Gaeun is taller than Nana by 0.5 centimeters -/
theorem gaeun_taller_than_nana :
  let nana_height_m : ℝ := 1.618
  let gaeun_height_cm : ℝ := 162.3
  let m_to_cm : ℝ := 100
  gaeun_height_cm - (nana_height_m * m_to_cm) = 0.5 := by sorry

end NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3292_329202


namespace NUMINAMATH_CALUDE_dave_book_spending_l3292_329223

/-- The total amount Dave spent on books -/
def total_spent (animal_books animal_price space_books space_price train_books train_price history_books history_price science_books science_price : ℕ) : ℕ :=
  animal_books * animal_price + space_books * space_price + train_books * train_price + history_books * history_price + science_books * science_price

/-- Theorem stating the total amount Dave spent on books -/
theorem dave_book_spending :
  total_spent 8 10 6 12 9 8 4 15 5 18 = 374 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_spending_l3292_329223


namespace NUMINAMATH_CALUDE_vasya_greater_than_petya_l3292_329286

def vasya_calculation : ℕ := 4 * (27 * 9)
def petya_calculation : ℕ := 55 * 3

theorem vasya_greater_than_petya : vasya_calculation > petya_calculation := by
  sorry

end NUMINAMATH_CALUDE_vasya_greater_than_petya_l3292_329286


namespace NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l3292_329269

/-- The shortest distance from a point on the line y=x-1 to the circle x^2+y^2+4x-2y+4=0 is 2√2 - 1 -/
theorem shortest_distance_line_to_circle : ∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x y : ℝ),
    (y = x - 1) →
    (x^2 + y^2 + 4*x - 2*y + 4 = 0) →
    d ≤ Real.sqrt ((x - 0)^2 + (y - 0)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l3292_329269


namespace NUMINAMATH_CALUDE_income_ratio_l3292_329201

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio (c_monthly : ℕ) (a_annual : ℕ) : 
  c_monthly = 15000 →
  a_annual = 504000 →
  (a_annual / 12 : ℚ) / ((1 + 12/100) * c_monthly) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_l3292_329201


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3292_329290

theorem arithmetic_calculation : 4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3292_329290


namespace NUMINAMATH_CALUDE_choose_officers_count_l3292_329284

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in the club -/
def choose_officers (club : Club) : ℕ :=
  club.boys * club.girls * (club.boys - 1) +
  club.girls * club.boys * (club.girls - 1)

/-- The main theorem stating the number of ways to choose officers -/
theorem choose_officers_count (club : Club)
  (h1 : club.total_members = 25)
  (h2 : club.boys = 12)
  (h3 : club.girls = 13)
  (h4 : club.total_members = club.boys + club.girls) :
  choose_officers club = 3588 := by
  sorry

#eval choose_officers ⟨25, 12, 13⟩

end NUMINAMATH_CALUDE_choose_officers_count_l3292_329284


namespace NUMINAMATH_CALUDE_area_of_triangle_BCD_l3292_329238

/-- Given a triangle ABC with area 36 and base 6, and a triangle BCD sharing the same height as ABC
    with base 34, prove that the area of triangle BCD is 204. -/
theorem area_of_triangle_BCD (area_ABC : ℝ) (base_AC : ℝ) (base_CD : ℝ) (height : ℝ) :
  area_ABC = 36 →
  base_AC = 6 →
  base_CD = 34 →
  area_ABC = (1/2) * base_AC * height →
  (1/2) * base_CD * height = 204 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BCD_l3292_329238


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3292_329282

/-- Represents the number of people in the group -/
def num_people : Nat := 5

/-- Represents the number of seats in the car -/
def num_seats : Nat := 5

/-- Represents the number of people who can drive (Mr. and Mrs. Lopez) -/
def num_drivers : Nat := 2

/-- Calculates the number of seating arrangements -/
def seating_arrangements : Nat :=
  num_drivers * (num_people - 1) * Nat.factorial (num_seats - 2)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_count :
  seating_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3292_329282


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l3292_329289

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℕ, x ≤ 15 ↔ 9 * x - 8 < 130 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l3292_329289


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3292_329234

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 42)
  (sum3 : c + a = 58) :
  a + b + c = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3292_329234


namespace NUMINAMATH_CALUDE_min_prime_angle_in_linear_pair_l3292_329250

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_prime_angle_in_linear_pair (a b : ℕ) :
  a + b = 180 →
  is_prime a →
  is_prime b →
  a > b →
  b ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_prime_angle_in_linear_pair_l3292_329250


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l3292_329232

/-- A power function that passes through the point (3, 9) -/
def f (x : ℝ) : ℝ := x^2

/-- The point (3, 9) lies on the graph of f -/
axiom point_on_graph : f 3 = 9

/-- Theorem: The interval of monotonic increase for f is [0, +∞) -/
theorem monotonic_increase_interval :
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l3292_329232


namespace NUMINAMATH_CALUDE_art_exhibit_revenue_l3292_329288

/-- Calculates the total revenue from ticket sales for an art exhibit --/
theorem art_exhibit_revenue :
  let start_time : Nat := 9 * 60  -- 9:00 AM in minutes
  let end_time : Nat := 16 * 60 + 55  -- 4:55 PM in minutes
  let interval : Nat := 5  -- 5 minutes
  let group_size : Nat := 30
  let regular_price : Nat := 10
  let student_price : Nat := 6
  let regular_to_student_ratio : Nat := 3

  let total_intervals : Nat := (end_time - start_time) / interval + 1
  let total_tickets : Nat := total_intervals * group_size
  let student_tickets : Nat := total_tickets / (regular_to_student_ratio + 1)
  let regular_tickets : Nat := total_tickets - student_tickets

  let total_revenue : Nat := student_tickets * student_price + regular_tickets * regular_price

  total_revenue = 25652 := by sorry

end NUMINAMATH_CALUDE_art_exhibit_revenue_l3292_329288


namespace NUMINAMATH_CALUDE_frank_problems_per_type_l3292_329243

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

theorem frank_problems_per_type :
  frank_problems / problem_types = 30 := by sorry

end NUMINAMATH_CALUDE_frank_problems_per_type_l3292_329243


namespace NUMINAMATH_CALUDE_officers_selection_count_l3292_329247

/-- Represents the number of ways to choose officers from a club. -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  2 * (boys * (boys - 1) * (boys - 2))

/-- Theorem stating the number of ways to choose officers under given conditions. -/
theorem officers_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 2640 := by
  sorry

#eval choose_officers 24 12 12

end NUMINAMATH_CALUDE_officers_selection_count_l3292_329247


namespace NUMINAMATH_CALUDE_ashley_exam_result_l3292_329292

/-- The percentage of marks Ashley secured in the exam -/
def ashley_percentage (marks_secured : ℕ) (maximum_marks : ℕ) : ℚ :=
  (marks_secured : ℚ) / (maximum_marks : ℚ) * 100

/-- Theorem stating that Ashley secured 83% in the exam -/
theorem ashley_exam_result : ashley_percentage 332 400 = 83 := by
  sorry

end NUMINAMATH_CALUDE_ashley_exam_result_l3292_329292


namespace NUMINAMATH_CALUDE_modular_congruence_l3292_329224

theorem modular_congruence (n : ℕ) : 
  0 ≤ n ∧ n < 31 ∧ (3 * n) % 31 = 1 → 
  (((2^n) ^ 3) - 2) % 31 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_l3292_329224


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3292_329215

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_pass_point = 120)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 220 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l3292_329215


namespace NUMINAMATH_CALUDE_diamonds_in_F_10_l3292_329261

/-- Number of diamonds in figure F_n -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 1 + 3 * (n * (n - 1) / 2)

/-- Theorem stating that F_10 contains 136 diamonds -/
theorem diamonds_in_F_10 : diamonds 10 = 136 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_in_F_10_l3292_329261


namespace NUMINAMATH_CALUDE_batsman_average_l3292_329298

/-- 
Given a batsman who has played 16 innings, prove that if he scores 87 runs 
in the 17th inning and this increases his average by 4 runs, 
then his new average after the 17th inning is 23 runs.
-/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 87) / 17 = prev_average + 4 → 
  prev_average + 4 = 23 := by sorry

end NUMINAMATH_CALUDE_batsman_average_l3292_329298


namespace NUMINAMATH_CALUDE_specific_regular_polygon_l3292_329293

/-- Properties of a regular polygon -/
structure RegularPolygon where
  perimeter : ℝ
  side_length : ℝ
  sides : ℕ
  interior_angle : ℝ

/-- The theorem about the specific regular polygon -/
theorem specific_regular_polygon :
  ∃ (p : RegularPolygon),
    p.perimeter = 180 ∧
    p.side_length = 15 ∧
    p.sides = 12 ∧
    p.interior_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_specific_regular_polygon_l3292_329293


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3292_329249

theorem expand_and_simplify (a : ℝ) : 3*a*(2*a^2 - 4*a) - 2*a^2*(3*a + 4) = -20*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3292_329249


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3292_329263

def museum_interval : ℕ := 18
def library_interval : ℕ := 24
def town_hall_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_occurrence :
  ∃ (h : ℕ), h * minutes_in_hour = lcm museum_interval (lcm library_interval town_hall_interval) ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3292_329263


namespace NUMINAMATH_CALUDE_bananas_left_l3292_329214

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas Denise removes from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem stating the number of bananas left in the jar after Denise removes some -/
theorem bananas_left : original_bananas - removed_bananas = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l3292_329214


namespace NUMINAMATH_CALUDE_jacob_walking_distance_l3292_329252

/-- Calculates the distance traveled given a constant rate and time --/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: Jacob walks 8 miles in 2 hours at a rate of 4 miles per hour --/
theorem jacob_walking_distance :
  let rate : ℝ := 4
  let time : ℝ := 2
  distance rate time = 8 := by
  sorry

end NUMINAMATH_CALUDE_jacob_walking_distance_l3292_329252


namespace NUMINAMATH_CALUDE_dracula_is_alive_l3292_329259

-- Define the possible states of a person
inductive PersonState
| Sane
| MadVampire
| Other

-- Define the Transylvanian's statement
def transylvanianStatement (personState : PersonState) (draculaAlive : Prop) : Prop :=
  (personState = PersonState.Sane ∨ personState = PersonState.MadVampire) → draculaAlive

-- Theorem to prove
theorem dracula_is_alive : ∃ (personState : PersonState), transylvanianStatement personState (∃ dracula, dracula = "alive") :=
sorry

end NUMINAMATH_CALUDE_dracula_is_alive_l3292_329259


namespace NUMINAMATH_CALUDE_trig_identities_l3292_329200

/-- Given an angle θ with vertex at the origin, initial side on positive x-axis,
    and terminal side on y = 1/2x (x ≤ 0), prove trigonometric identities. -/
theorem trig_identities (θ α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = (1/2) * x ∧ 
   Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧ 
   Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  (Real.cos (π/2 + θ) = Real.sqrt 5 / 5) ∧
  (Real.cos (α + π/4) = Real.sin θ → 
   (Real.sin (2*α + π/4) = 7 * Real.sqrt 2 / 10 ∨ 
    Real.sin (2*α + π/4) = - Real.sqrt 2 / 10)) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3292_329200


namespace NUMINAMATH_CALUDE_add_three_preserves_inequality_l3292_329265

theorem add_three_preserves_inequality (a b : ℝ) : a > b → a + 3 > b + 3 := by
  sorry

end NUMINAMATH_CALUDE_add_three_preserves_inequality_l3292_329265


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3292_329299

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i + 1

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : z_condition z) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3292_329299


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3292_329287

theorem arithmetic_expression_equality : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3292_329287


namespace NUMINAMATH_CALUDE_problem_solution_l3292_329241

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 110) : x = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3292_329241


namespace NUMINAMATH_CALUDE_fifth_root_unity_product_l3292_329208

/-- Given a complex number z that is a fifth root of unity, 
    prove that the product (1 - z)(1 - z^2)(1 - z^3)(1 - z^4) equals 5 -/
theorem fifth_root_unity_product (z : ℂ) 
  (h : z = Complex.exp (2 * Real.pi * I / 5)) : 
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_product_l3292_329208


namespace NUMINAMATH_CALUDE_inequality_proof_l3292_329211

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  Real.sqrt (1 + a) + Real.sqrt (1 + b) ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3292_329211


namespace NUMINAMATH_CALUDE_cube_root_function_l3292_329251

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l3292_329251


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3292_329205

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) :
  x^2 + y^2 = 33 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3292_329205


namespace NUMINAMATH_CALUDE_student_group_arrangements_l3292_329229

/-- The number of ways to divide n students into k equal groups -/
def divide_students (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different topics -/
def assign_topics (k : ℕ) : ℕ := sorry

theorem student_group_arrangements :
  let n : ℕ := 6  -- number of students
  let k : ℕ := 3  -- number of groups
  divide_students n k * assign_topics k = 540 :=
by sorry

end NUMINAMATH_CALUDE_student_group_arrangements_l3292_329229


namespace NUMINAMATH_CALUDE_max_value_sine_function_l3292_329258

theorem max_value_sine_function (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π/3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l3292_329258


namespace NUMINAMATH_CALUDE_cards_not_in_box_l3292_329220

theorem cards_not_in_box (total_cards : ℕ) (cards_per_box : ℕ) (boxes_given : ℕ) (boxes_kept : ℕ) : 
  total_cards = 75 →
  cards_per_box = 10 →
  boxes_given = 2 →
  boxes_kept = 5 →
  total_cards - (cards_per_box * (boxes_given + boxes_kept)) = 5 := by
sorry

end NUMINAMATH_CALUDE_cards_not_in_box_l3292_329220


namespace NUMINAMATH_CALUDE_probability_same_tribe_l3292_329222

def total_participants : ℕ := 18
def tribe_size : ℕ := 9
def num_quitters : ℕ := 3

theorem probability_same_tribe :
  (Nat.choose tribe_size num_quitters * 2 : ℚ) / Nat.choose total_participants num_quitters = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_tribe_l3292_329222


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3292_329239

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos : q > 0) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_eq : a 3 * a 9 = 2 * (a 5)^2) : 
  q = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3292_329239


namespace NUMINAMATH_CALUDE_vincent_book_purchase_l3292_329262

/-- The number of books about outer space Vincent bought -/
def books_outer_space : ℕ := 1

/-- The number of books about animals Vincent bought -/
def books_animals : ℕ := 10

/-- The number of books about trains Vincent bought -/
def books_trains : ℕ := 3

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_book_purchase :
  books_outer_space = 1 ∧
  books_animals = 10 ∧
  books_trains = 3 ∧
  cost_per_book = 16 ∧
  total_spent = 224 →
  books_outer_space = 1 :=
by sorry

end NUMINAMATH_CALUDE_vincent_book_purchase_l3292_329262


namespace NUMINAMATH_CALUDE_exactly_two_females_adjacent_l3292_329285

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of arrangements where exactly two female students are standing next to each other -/
def arrangements_two_females_adjacent : ℕ := 3600

/-- Theorem stating that the number of arrangements where exactly two female students
    are standing next to each other is 3600 -/
theorem exactly_two_females_adjacent :
  (num_male = 4 ∧ num_female = 3) →
  arrangements_two_females_adjacent = 3600 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_females_adjacent_l3292_329285


namespace NUMINAMATH_CALUDE_parallel_line_distance_l3292_329221

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1 : ℝ
  /-- Length of the second chord -/
  chord2 : ℝ
  /-- Length of the third chord -/
  chord3 : ℝ
  /-- Assertion that the first and third chords are equal -/
  chord1_eq_chord3 : chord1 = chord3
  /-- Assertion that the first chord has length 42 -/
  chord1_length : chord1 = 42
  /-- Assertion that the second chord has length 40 -/
  chord2_length : chord2 = 40

/-- Theorem stating that the distance between adjacent parallel lines is √(92/11) -/
theorem parallel_line_distance (c : CircleWithParallelLines) : 
  c.line_distance = Real.sqrt (92 / 11) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_distance_l3292_329221


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3292_329207

theorem complex_power_modulus : Complex.abs ((4 + 2*Complex.I)^5) = 160 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3292_329207


namespace NUMINAMATH_CALUDE_vector_calculation_l3292_329253

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_calculation : 
  (-2 • a + 4 • b) = ![-6, -8] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l3292_329253


namespace NUMINAMATH_CALUDE_smallest_three_digit_sum_product_l3292_329257

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_sum_product (a b c : ℕ) : ℕ := a + b + c + a*b + b*c + a*c + a*b*c

theorem smallest_three_digit_sum_product :
  ∃ (n : ℕ) (a b c : ℕ),
    is_three_digit n ∧
    n = 100*a + 10*b + c ∧
    n = digits_sum_product a b c ∧
    (∀ (m : ℕ) (x y z : ℕ),
      is_three_digit m ∧
      m = 100*x + 10*y + z ∧
      m = digits_sum_product x y z →
      n ≤ m) ∧
    n = 199 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_sum_product_l3292_329257


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3292_329256

/-- The volume of ice cream in a cone and sphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3292_329256


namespace NUMINAMATH_CALUDE_min_sum_with_gcd_and_divisibility_l3292_329213

theorem min_sum_with_gcd_and_divisibility (a b : ℕ+) :
  (Nat.gcd a.val b.val = 2015) →
  ((a.val + b.val) ∣ ((a.val - b.val)^2016 + b.val^2016)) →
  (∀ c d : ℕ+, (Nat.gcd c.val d.val = 2015) → 
    ((c.val + d.val) ∣ ((c.val - d.val)^2016 + d.val^2016)) → 
    (a.val + b.val ≤ c.val + d.val)) →
  a.val + b.val = 10075 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_gcd_and_divisibility_l3292_329213


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l3292_329242

/-- Given complex numbers p, q, and r that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 360, then the square of the hypotenuse of the
    triangle is 540. -/
theorem right_triangle_hypotenuse_squared 
  (p q r : ℂ) 
  (h_zeros : ∃ (s t u : ℂ), p^3 + s*p^2 + t*p + u = 0 ∧ 
                             q^3 + s*q^2 + t*q + u = 0 ∧ 
                             r^3 + s*r^2 + t*r + u = 0)
  (h_right_triangle : ∃ (k : ℝ), (Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
                                 (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
                                 (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2)
  (h_sum_squares : Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 360) :
  ∃ (k : ℝ), k^2 = 540 ∧ 
    ((Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
     (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
     (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l3292_329242


namespace NUMINAMATH_CALUDE_sarah_reading_capacity_l3292_329226

/-- The number of complete books Sarah can read given her reading speed and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) : ℕ :=
  (pages_per_hour * hours_available) / pages_per_book

/-- Theorem: Sarah can read 2 books in 8 hours -/
theorem sarah_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_reading_capacity_l3292_329226


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3292_329280

theorem complex_modulus_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 5 / (2 - i) * i
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3292_329280


namespace NUMINAMATH_CALUDE_count_integers_between_square_roots_l3292_329295

theorem count_integers_between_square_roots : 
  (Finset.range 25 \ Finset.range 10).card = 15 := by sorry

end NUMINAMATH_CALUDE_count_integers_between_square_roots_l3292_329295


namespace NUMINAMATH_CALUDE_coin_problem_l3292_329255

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins of each type -/
def totalValueInRupees (coins20 : ℕ) (coins25 : ℕ) : ℚ :=
  (coins20 * 20 + coins25 * 25) / 100

theorem coin_problem :
  let totalCoins : ℕ := 344
  let coins20 : ℕ := 300
  let coins25 : ℕ := totalCoins - coins20
  totalValueInRupees coins20 coins25 = 71 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3292_329255


namespace NUMINAMATH_CALUDE_expand_polynomial_l3292_329216

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 4) * 3*x^3 = 15*x^5 + 9*x^4 - 12*x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3292_329216


namespace NUMINAMATH_CALUDE_pentagram_star_angle_pentagram_star_angle_proof_l3292_329281

/-- The angle at each point of a regular pentagram formed by extending the sides of a regular pentagon inscribed in a circle is 216°. -/
theorem pentagram_star_angle : ℝ :=
  let regular_pentagon_external_angle : ℝ := 360 / 5
  let star_point_angle : ℝ := 360 - 2 * regular_pentagon_external_angle
  216

/-- Proof of the pentagram star angle theorem. -/
theorem pentagram_star_angle_proof : pentagram_star_angle = 216 := by
  sorry

end NUMINAMATH_CALUDE_pentagram_star_angle_pentagram_star_angle_proof_l3292_329281


namespace NUMINAMATH_CALUDE_inequality_proof_l3292_329231

theorem inequality_proof (a b : ℝ) (h : 4 * b + a = 1) : a^2 + 4 * b^2 ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3292_329231


namespace NUMINAMATH_CALUDE_least_distinct_values_l3292_329294

/-- Given a list of positive integers with the specified properties,
    the least number of distinct values is 218. -/
theorem least_distinct_values (list : List ℕ+) : 
  (list.length = 3042) →
  (∃! m, list.count m = 15 ∧ ∀ n, list.count n ≤ list.count m) →
  (list.toFinset.card ≥ 218 ∧ ∀ k, k < 218 → ¬(list.toFinset.card = k)) := by
  sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3292_329294


namespace NUMINAMATH_CALUDE_circle_equation_l3292_329219

/-- The equation of a circle with center (1,1) passing through the origin (0,0) -/
theorem circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 1)^2 + (y - 1)^2 = r^2 ∧ 0^2 + 0^2 = r^2) → 
  (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3292_329219


namespace NUMINAMATH_CALUDE_square_side_ratio_sum_l3292_329233

theorem square_side_ratio_sum (area_ratio : ℚ) : 
  area_ratio = 128 / 50 →
  ∃ (p q r : ℕ), 
    (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) ∧
    p + q + r = 14 :=
by sorry

end NUMINAMATH_CALUDE_square_side_ratio_sum_l3292_329233


namespace NUMINAMATH_CALUDE_all_positive_integers_in_A_l3292_329278

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the properties of set A
def HasPropertyA (A : Set ℕ) : Prop :=
  A ⊆ PositiveIntegers ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (∀ m : ℕ, m ∈ A → ∀ d : ℕ, d > 0 ∧ m % d = 0 → d ∈ A) ∧
  (∀ b c : ℕ, b ∈ A → c ∈ A → 1 < b → b < c → (1 + b * c) ∈ A)

-- Theorem statement
theorem all_positive_integers_in_A (A : Set ℕ) (h : HasPropertyA A) :
  A = PositiveIntegers := by
  sorry

end NUMINAMATH_CALUDE_all_positive_integers_in_A_l3292_329278


namespace NUMINAMATH_CALUDE_proposition_b_l3292_329272

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_l3292_329272


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3292_329245

theorem multiply_mixed_number : 8 * (12 + 2/5) = 99 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3292_329245


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l3292_329264

theorem power_of_three_mod_seven : 3^1503 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l3292_329264


namespace NUMINAMATH_CALUDE_parallelepiped_edge_lengths_l3292_329273

/-- Given a rectangular parallelepiped with mass M and density ρ, and thermal power ratios of 1:2:8
    when connected to different pairs of faces, this theorem states the edge lengths of the parallelepiped. -/
theorem parallelepiped_edge_lengths (M ρ : ℝ) (hM : M > 0) (hρ : ρ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a < b ∧ b < c ∧
    a * b * c = M / ρ ∧
    b^2 / a^2 = 2 ∧
    c^2 / b^2 = 4 ∧
    a = (M / (4 * ρ))^(1/3) ∧
    b = Real.sqrt 2 * (M / (4 * ρ))^(1/3) ∧
    c = 2 * Real.sqrt 2 * (M / (4 * ρ))^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_edge_lengths_l3292_329273


namespace NUMINAMATH_CALUDE_january_rainfall_l3292_329227

theorem january_rainfall (first_week : ℝ) (second_week : ℝ) :
  second_week = 1.5 * first_week →
  second_week = 21 →
  first_week + second_week = 35 := by
sorry

end NUMINAMATH_CALUDE_january_rainfall_l3292_329227


namespace NUMINAMATH_CALUDE_set_relationship_l3292_329279

def M : Set ℝ := {x : ℝ | ∃ m : ℤ, x = m + 1/6}
def S : Set ℝ := {x : ℝ | ∃ s : ℤ, x = 1/2 * s - 1/3}
def P : Set ℝ := {x : ℝ | ∃ p : ℤ, x = 1/2 * p + 1/6}

theorem set_relationship : M ⊆ S ∧ S = P := by sorry

end NUMINAMATH_CALUDE_set_relationship_l3292_329279


namespace NUMINAMATH_CALUDE_car_speed_problem_l3292_329225

theorem car_speed_problem (x : ℝ) : 
  x > 0 →  -- Assuming speed is positive
  (x + 70) / 2 = 84 → 
  x = 98 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3292_329225


namespace NUMINAMATH_CALUDE_line_inclination_45_implies_a_equals_1_l3292_329203

/-- If the line ax + (2a - 3)y = 0 has an angle of inclination of 45°, then a = 1 -/
theorem line_inclination_45_implies_a_equals_1 (a : ℝ) : 
  (∃ x y : ℝ, a * x + (2 * a - 3) * y = 0 ∧ 
   Real.arctan ((3 - 2 * a) / a) = π / 4) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_implies_a_equals_1_l3292_329203
