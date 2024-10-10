import Mathlib

namespace min_bound_sqrt_two_l3877_387701

theorem min_bound_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (y + 1/x) (1/y)) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (min (y + 1/x) (1/y)) = Real.sqrt 2 := by
  sorry

end min_bound_sqrt_two_l3877_387701


namespace opposite_roots_quadratic_l3877_387768

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 1)*x + k + 1 = 0 ∧ 
               y^2 + (k^2 - 1)*y + k + 1 = 0 ∧ 
               x = -y ∧ x ≠ y) → 
  k = -1 := by
sorry

end opposite_roots_quadratic_l3877_387768


namespace scientific_notation_361000000_l3877_387780

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_361000000 :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by sorry

end scientific_notation_361000000_l3877_387780


namespace polynomial_factorization_l3877_387700

theorem polynomial_factorization (x : ℝ) :
  x^6 - 3*x^4 + 3*x^2 - 1 = (x-1)^3*(x+1)^3 := by
  sorry

end polynomial_factorization_l3877_387700


namespace students_in_grade_l3877_387744

theorem students_in_grade (n : ℕ) (misha : ℕ) : 
  (misha = n - 59) ∧ (misha = 60) → n = 119 :=
by sorry

end students_in_grade_l3877_387744


namespace prob_at_least_one_red_l3877_387714

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one red ball when drawing 2 balls from a bag containing 2 red and 2 white balls -/
theorem prob_at_least_one_red : 
  (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / Nat.choose total_balls drawn_balls = 5 / 6 := by
  sorry

end prob_at_least_one_red_l3877_387714


namespace bus_passengers_after_four_stops_l3877_387722

/-- Represents the change in passengers at a bus stop -/
structure StopChange where
  boarding : Int
  alighting : Int

/-- Calculates the final number of passengers on a bus after a series of stops -/
def finalPassengers (initial : Int) (changes : List StopChange) : Int :=
  changes.foldl (fun acc stop => acc + stop.boarding - stop.alighting) initial

/-- Theorem stating the final number of passengers after 4 stops -/
theorem bus_passengers_after_four_stops :
  let initial := 22
  let changes := [
    { boarding := 3, alighting := 6 },
    { boarding := 8, alighting := 5 },
    { boarding := 2, alighting := 4 },
    { boarding := 1, alighting := 8 }
  ]
  finalPassengers initial changes = 13 := by
  sorry

end bus_passengers_after_four_stops_l3877_387722


namespace distance_covered_l3877_387753

/-- Proves that the distance covered is 100 km given the conditions of the problem -/
theorem distance_covered (usual_speed usual_time increased_speed : ℝ) : 
  usual_speed = 20 →
  increased_speed = 25 →
  usual_speed * usual_time = increased_speed * (usual_time - 1) →
  usual_speed * usual_time = 100 := by
  sorry

#check distance_covered

end distance_covered_l3877_387753


namespace runners_meet_time_l3877_387747

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 600

/-- The speeds of the four runners in meters per second -/
def runner_speeds : List ℝ := [5.0, 5.5, 6.0, 6.5]

/-- The time in seconds for the runners to meet again -/
def meeting_time : ℝ := 1200

/-- Theorem stating that the given meeting time is the minimum time for the runners to meet again -/
theorem runners_meet_time : 
  meeting_time = (track_length / (runner_speeds[1] - runner_speeds[0])) ∧
  meeting_time = (track_length / (runner_speeds[2] - runner_speeds[1])) ∧
  meeting_time = (track_length / (runner_speeds[3] - runner_speeds[2])) ∧
  (∀ t : ℝ, t > 0 → t < meeting_time → 
    ∃ i j : Fin 4, i ≠ j ∧ 
    (runner_speeds[i] * t) % track_length ≠ (runner_speeds[j] * t) % track_length) :=
by sorry

end runners_meet_time_l3877_387747


namespace cube_root_simplification_l3877_387775

theorem cube_root_simplification (ω : ℂ) :
  ω ≠ 1 →
  ω^3 = 1 →
  (1 - 2*ω + 3*ω^2)^3 + (2 + 3*ω - 4*ω^2)^3 = -83 := by sorry

end cube_root_simplification_l3877_387775


namespace sum_product_ratio_l3877_387748

theorem sum_product_ratio (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : x + y + z = 3) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end sum_product_ratio_l3877_387748


namespace larger_integer_is_48_l3877_387732

theorem larger_integer_is_48 (x y : ℤ) : 
  y = 4 * x →                           -- Two integers are in the ratio of 1 to 4
  (x + 12) * 2 = y →                    -- If 12 is added to the smaller number, the ratio becomes 1 to 2
  y = 48 :=                             -- The larger integer is 48
by
  sorry


end larger_integer_is_48_l3877_387732


namespace cashier_bills_l3877_387710

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 126) 
  (h2 : total_value = 840) : ∃ (five_dollar_bills ten_dollar_bills : ℕ), 
  five_dollar_bills + ten_dollar_bills = total_bills ∧ 
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧ 
  five_dollar_bills = 84 := by
sorry

end cashier_bills_l3877_387710


namespace sum_of_even_positive_integers_less_than_100_l3877_387739

theorem sum_of_even_positive_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n > 0) (Finset.range 100)).sum id = 2450 := by
  sorry

end sum_of_even_positive_integers_less_than_100_l3877_387739


namespace colorings_count_l3877_387765

/-- The number of ways to color the edges of an m × n rectangle with three colors,
    such that each unit square has two sides of one color and two sides of another color. -/
def colorings (m n : ℕ) : ℕ :=
  18 * 2^(m*n - 1) * 3^(m + n - 2)

/-- Theorem stating that the number of valid colorings for an m × n rectangle
    with three colors is equal to 18 × 2^(mn-1) × 3^(m+n-2). -/
theorem colorings_count (m n : ℕ) :
  colorings m n = 18 * 2^(m*n - 1) * 3^(m + n - 2) :=
by sorry

end colorings_count_l3877_387765


namespace quadratic_roots_order_l3877_387705

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The statement of the problem -/
theorem quadratic_roots_order (b c x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, f b c x - x = 0 ↔ x = x₁ ∨ x = x₂) →
  x₂ - x₁ > 2 →
  (∀ x, f b c (f b c x) = x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  x₃ > x₄ →
  x₄ < x₁ ∧ x₁ < x₃ ∧ x₃ < x₂ := by
sorry

end quadratic_roots_order_l3877_387705


namespace jessica_almonds_l3877_387776

theorem jessica_almonds : ∃ (j : ℕ), 
  (∃ (l : ℕ), j = l + 8 ∧ l = j / 3) → j = 12 :=
by
  sorry

end jessica_almonds_l3877_387776


namespace shopkeeper_profit_l3877_387792

/-- Calculates the profit percentage when selling a number of articles at the cost price of a different number of articles. -/
def profit_percentage (articles_sold : ℕ) (articles_cost_price : ℕ) : ℚ :=
  ((articles_cost_price : ℚ) - (articles_sold : ℚ)) / (articles_sold : ℚ) * 100

/-- Theorem stating that when a shopkeeper sells 50 articles at the cost price of 60 articles, the profit percentage is 20%. -/
theorem shopkeeper_profit : profit_percentage 50 60 = 20 := by
  sorry

end shopkeeper_profit_l3877_387792


namespace differential_equation_holds_l3877_387730

open Real

noncomputable def y (x : ℝ) : ℝ := 1 / sqrt (sin x + x)

theorem differential_equation_holds (x : ℝ) (h : sin x + x > 0) :
  2 * sin x * (deriv y x) + y x * cos x = (y x)^3 * (x * cos x - sin x) := by
  sorry

end differential_equation_holds_l3877_387730


namespace crazy_silly_school_series_l3877_387708

theorem crazy_silly_school_series (total_movies : ℕ) (books_read : ℕ) (movies_watched : ℕ) (movies_to_watch : ℕ) :
  total_movies = 17 →
  books_read = 19 →
  movies_watched + movies_to_watch = total_movies →
  (∃ (different_books : ℕ), different_books = books_read) :=
by sorry

end crazy_silly_school_series_l3877_387708


namespace complex_equation_solution_l3877_387719

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  (1 + i) * z = Complex.abs (1 + Real.sqrt 3 * i) →
  z = 1 - i :=
by sorry

end complex_equation_solution_l3877_387719


namespace ellipse_intersection_l3877_387794

/-- Definition of the ellipse with given properties -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (0, 3)
  let F₂ : ℝ × ℝ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7

theorem ellipse_intersection :
  ellipse (0, 0) → 
  (∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0)) → 
  ellipse (56/11, 0) :=
sorry

end ellipse_intersection_l3877_387794


namespace sqrt_equation_solution_l3877_387760

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l3877_387760


namespace mary_weight_loss_l3877_387793

/-- Given Mary's weight changes, prove her initial weight loss --/
theorem mary_weight_loss (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) : 
  ∃ x : ℝ, x = 10.5 ∧ initial_weight - x + 2*x - 3*x + 3 = final_weight :=
by sorry

end mary_weight_loss_l3877_387793


namespace min_sum_distances_l3877_387738

/-- The minimum value of PA + PB for a point P on the parabola y² = 4x -/
theorem min_sum_distances (y : ℝ) : 
  let x := y^2 / 4
  let PA := x
  let PB := |x - y + 4| / Real.sqrt 2
  (∀ y', (y'^2 / 4 - y' / Real.sqrt 2 + 2 * Real.sqrt 2) ≥ 
         (y^2 / 4 - y / Real.sqrt 2 + 2 * Real.sqrt 2)) →
  PA + PB = 5 * Real.sqrt 2 / 2 - 1 := by
sorry

end min_sum_distances_l3877_387738


namespace expression_equivalence_l3877_387779

theorem expression_equivalence (a b : ℝ) 
  (h1 : (a + b) - (a - b) ≠ 0) 
  (h2 : a + b + a - b ≠ 0) : 
  let P := a + b
  let Q := a - b
  ((P + Q)^2 / (P - Q)^2) - ((P - Q)^2 / (P + Q)^2) = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) := by
sorry

end expression_equivalence_l3877_387779


namespace largest_divisor_when_square_divisible_by_50_l3877_387718

theorem largest_divisor_when_square_divisible_by_50 (n : ℕ) (h1 : n > 0) (h2 : 50 ∣ n^2) :
  ∃ (d : ℕ), d ∣ n ∧ d = 10 ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
sorry

end largest_divisor_when_square_divisible_by_50_l3877_387718


namespace aisha_has_largest_answer_l3877_387721

def starting_number : ℕ := 15

def maria_calculation (n : ℕ) : ℕ := ((n - 2) * 3) + 5

def liam_calculation (n : ℕ) : ℕ := (n * 3 - 2) + 5

def aisha_calculation (n : ℕ) : ℕ := ((n - 2) + 5) * 3

theorem aisha_has_largest_answer :
  aisha_calculation starting_number > maria_calculation starting_number ∧
  aisha_calculation starting_number > liam_calculation starting_number :=
sorry

end aisha_has_largest_answer_l3877_387721


namespace movie_theater_attendance_l3877_387791

theorem movie_theater_attendance 
  (total_seats : ℕ) 
  (empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end movie_theater_attendance_l3877_387791


namespace complex_product_theorem_l3877_387772

theorem complex_product_theorem (a : ℝ) (z₁ z₂ : ℂ) : 
  z₁ = a - 2*I ∧ z₂ = -1 + a*I ∧ (∃ b : ℝ, z₁ + z₂ = b*I) → z₁ * z₂ = 1 + 3*I :=
by sorry

end complex_product_theorem_l3877_387772


namespace min_n_for_expansion_terms_min_n_value_l3877_387746

theorem min_n_for_expansion_terms (n : ℕ) : (n + 1) ^ 2 ≥ 2021 ↔ n ≥ 44 := by sorry

theorem min_n_value : ∃ (n : ℕ), n > 0 ∧ (n + 1) ^ 2 ≥ 2021 ∧ ∀ (m : ℕ), m > 0 → (m + 1) ^ 2 ≥ 2021 → m ≥ n := by
  use 44
  sorry

end min_n_for_expansion_terms_min_n_value_l3877_387746


namespace f_properties_l3877_387757

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

theorem f_properties :
  let f : ℝ × ℝ → ℝ × ℝ := λ p ↦ (p.1 + p.2, p.1 * p.2)
  (f (1, -2) = (-1, -2)) ∧
  (f (2, -1) = (1, -2)) ∧
  (f (-1, 2) = (1, -2)) ∧
  (∀ a b : ℝ, f (a, b) = (1, -2) → (a = 2 ∧ b = -1) ∨ (a = -1 ∧ b = 2)) :=
by sorry

end f_properties_l3877_387757


namespace certain_number_proof_l3877_387786

theorem certain_number_proof (X : ℝ) : 0.8 * X - 0.35 * 300 = 31 → X = 170 := by
  sorry

end certain_number_proof_l3877_387786


namespace steven_shirt_count_l3877_387724

def brian_shirts : ℕ := 3

def andrew_shirts : ℕ := 6 * brian_shirts

def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end steven_shirt_count_l3877_387724


namespace solution_satisfies_system_l3877_387761

/-- The system of linear equations -/
def system (x : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (x₁, x₂, x₃, x₄) := x
  x₁ + 2*x₂ + 3*x₃ + x₄ = 1 ∧
  3*x₁ + 13*x₂ + 13*x₃ + 5*x₄ = 3 ∧
  3*x₁ + 7*x₂ + 7*x₃ + 2*x₄ = 12 ∧
  x₁ + 5*x₂ + 3*x₃ + x₄ = 7 ∧
  4*x₁ + 5*x₂ + 6*x₃ + x₄ = 19

/-- The general solution to the system -/
def solution (α : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (4 - α, 2, α, -7 - 2*α)

/-- Theorem stating that the general solution satisfies the system for any α -/
theorem solution_satisfies_system :
  ∀ α : ℝ, system (solution α) :=
by sorry

end solution_satisfies_system_l3877_387761


namespace parabola_properties_l3877_387796

/-- A parabola passing through (-1, 0) and (m, 0) opening downwards -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_pass_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

/-- The properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧ 
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    (p.a * x₁^2 + p.b * x₁ + p.c = y₁) → 
    (p.a * x₂^2 + p.b * x₂ + p.c = y₂) → 
    x₁ < x₂ → 
    x₁ + x₂ > 1 → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ 
    p.a * x₂^2 + p.b * x₂ + p.c = 1) := by
  sorry

end parabola_properties_l3877_387796


namespace pizza_topping_options_l3877_387771

/-- Represents the number of topping options for each category --/
structure ToppingOptions where
  cheese : Nat
  meat : Nat
  vegetable : Nat

/-- Represents the restriction between pepperoni and peppers --/
def hasPepperoniPepperRestriction : Bool := true

/-- Calculates the total number of topping combinations --/
def totalCombinations (options : ToppingOptions) (restriction : Bool) : Nat :=
  if restriction then
    options.cheese * (options.meat - 1) * options.vegetable +
    options.cheese * 1 * (options.vegetable - 1)
  else
    options.cheese * options.meat * options.vegetable

/-- The main theorem to prove --/
theorem pizza_topping_options :
  ∃ (options : ToppingOptions),
    options.cheese = 3 ∧
    options.vegetable = 5 ∧
    hasPepperoniPepperRestriction = true ∧
    totalCombinations options hasPepperoniPepperRestriction = 57 ∧
    options.meat = 4 := by
  sorry


end pizza_topping_options_l3877_387771


namespace ball_selection_probability_l3877_387751

theorem ball_selection_probability (n : ℕ) (h : n = 48) : 
  (Nat.choose (n - 6) 7) / (Nat.choose n 7) = 
  (6 * Nat.choose (n - 7) 6) / (Nat.choose n 7) := by
  sorry

end ball_selection_probability_l3877_387751


namespace pentagonal_prism_coloring_l3877_387717

structure PentagonalPrism where
  vertices : Fin 10 → Point
  color : Fin 45 → Color

inductive Color
  | Red
  | Blue

def isEdge (i j : Fin 10) : Bool :=
  (i < j ∧ (i.val + 1 = j.val ∨ (i.val = 4 ∧ j.val = 0) ∨ (i.val = 9 ∧ j.val = 5))) ∨
  (j < i ∧ (j.val + 1 = i.val ∨ (j.val = 4 ∧ i.val = 0) ∨ (j.val = 9 ∧ i.val = 5)))

def isTopFaceEdge (i j : Fin 10) : Bool :=
  i < 5 ∧ j < 5 ∧ isEdge i j

def isBottomFaceEdge (i j : Fin 10) : Bool :=
  i ≥ 5 ∧ j ≥ 5 ∧ isEdge i j

def getEdgeColor (p : PentagonalPrism) (i j : Fin 10) : Color :=
  if i < j then p.color ⟨i.val * 9 + j.val - (i.val * (i.val + 1) / 2), sorry⟩
  else p.color ⟨j.val * 9 + i.val - (j.val * (j.val + 1) / 2), sorry⟩

def noMonochromaticTriangle (p : PentagonalPrism) : Prop :=
  ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(getEdgeColor p i j = getEdgeColor p j k ∧ getEdgeColor p j k = getEdgeColor p i k)

theorem pentagonal_prism_coloring (p : PentagonalPrism) 
  (h : noMonochromaticTriangle p) :
  (∀ i j : Fin 10, isTopFaceEdge i j → getEdgeColor p i j = getEdgeColor p 0 1) ∧
  (∀ i j : Fin 10, isBottomFaceEdge i j → getEdgeColor p i j = getEdgeColor p 5 6) :=
sorry

end pentagonal_prism_coloring_l3877_387717


namespace simplify_square_roots_l3877_387789

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end simplify_square_roots_l3877_387789


namespace least_five_digit_congruent_to_6_mod_17_l3877_387725

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (n : ℕ), (n ≥ 10000 ∧ n < 100000) ∧ 
             (n % 17 = 6) ∧ 
             (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m) ∧
             n = 10002 :=
by sorry

end least_five_digit_congruent_to_6_mod_17_l3877_387725


namespace negation_equivalence_l3877_387734

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end negation_equivalence_l3877_387734


namespace half_vector_MN_l3877_387785

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN is (1/2, -4) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (-2, 3)) (h2 : ON = (-1, -5)) :
  (1 / 2 : ℝ) • (ON - OM) = (1/2, -4) := by
  sorry

end half_vector_MN_l3877_387785


namespace inequality_proof_l3877_387749

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (b + c) / a) + (Real.sqrt (c + a) / b) + (Real.sqrt (a + b) / c) ≥
  (4 * (a + b + c)) / Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end inequality_proof_l3877_387749


namespace point_on_y_axis_l3877_387797

/-- 
If a point P with coordinates (a-1, a²-9) lies on the y-axis, 
then its coordinates are (0, -8).
-/
theorem point_on_y_axis (a : ℝ) : 
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end point_on_y_axis_l3877_387797


namespace track_width_l3877_387715

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 10 * Real.pi) : 
  r₁ - r₂ = 5 := by
  sorry

end track_width_l3877_387715


namespace intersection_A_complement_B_l3877_387709

open Set

theorem intersection_A_complement_B (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {2, 4}) (hB : B = {4, 5}) : A ∩ (U \ B) = {2} := by
  sorry

end intersection_A_complement_B_l3877_387709


namespace fraction_equality_l3877_387712

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x - 2 * y) / (2 * x + 5 * y) = 3) : 
  (2 * x - 5 * y) / (x + 2 * y) = 13 / 5 := by
  sorry

end fraction_equality_l3877_387712


namespace quadratic_root_condition_l3877_387790

theorem quadratic_root_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + k*x₁ + 4*k^2 - 3 = 0 ∧ 
   x₂^2 + k*x₂ + 4*k^2 - 3 = 0 ∧
   x₁ + x₂ = x₁ * x₂) → 
  k = 3/4 := by
sorry

end quadratic_root_condition_l3877_387790


namespace power_24_in_terms_of_a_and_t_l3877_387727

theorem power_24_in_terms_of_a_and_t (x a t : ℝ) 
  (h1 : 2^x = a) (h2 : 3^x = t) : 24^x = a^3 * t := by
  sorry

end power_24_in_terms_of_a_and_t_l3877_387727


namespace f_strictly_increasing_l3877_387702

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (x₁ - x₂) * (f x₁ - f x₂) > 0 :=
by sorry

end f_strictly_increasing_l3877_387702


namespace sandy_earnings_l3877_387728

/-- Calculates the total earnings for a given hourly rate and hours worked over three days -/
def total_earnings (hourly_rate : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) : ℝ :=
  hourly_rate * (hours_day1 + hours_day2 + hours_day3)

/-- Sandy's earnings problem -/
theorem sandy_earnings : 
  let hourly_rate : ℝ := 15
  let hours_friday : ℝ := 10
  let hours_saturday : ℝ := 6
  let hours_sunday : ℝ := 14
  total_earnings hourly_rate hours_friday hours_saturday hours_sunday = 450 := by
  sorry

end sandy_earnings_l3877_387728


namespace tangent_parallel_points_l3877_387703

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 2 ↔ (x = 1 ∧ y = 3) ∨ (x = -1 ∧ y = 3) := by
  sorry

end tangent_parallel_points_l3877_387703


namespace smallest_integer_greater_than_neg_seventeen_thirds_l3877_387764

theorem smallest_integer_greater_than_neg_seventeen_thirds :
  Int.ceil (-17 / 3 : ℚ) = -5 := by sorry

end smallest_integer_greater_than_neg_seventeen_thirds_l3877_387764


namespace sum_of_quadratic_solutions_l3877_387778

theorem sum_of_quadratic_solutions (x : ℝ) : 
  (x^2 + 6*x - 22 = 4*x - 18) → 
  (∃ a b : ℝ, (a + b = -2) ∧ (x = a ∨ x = b)) :=
by sorry

end sum_of_quadratic_solutions_l3877_387778


namespace collinear_vectors_product_l3877_387706

/-- Given two non-collinear vectors i and j in a vector space V over ℝ,
    if AB = i + m*j, AD = n*i + j, m ≠ 1, and points A, B, and D are collinear,
    then mn = 1 -/
theorem collinear_vectors_product (V : Type*) [AddCommGroup V] [Module ℝ V]
  (i j : V) (m n : ℝ) (A B D : V) :
  LinearIndependent ℝ ![i, j] →
  B - A = i + m • j →
  D - A = n • i + j →
  m ≠ 1 →
  ∃ (k : ℝ), B - A = k • (D - A) →
  m * n = 1 := by
  sorry

end collinear_vectors_product_l3877_387706


namespace total_bird_families_l3877_387777

/-- The number of bird families that migrated to Africa -/
def africa : ℕ := 42

/-- The number of bird families that migrated to Asia -/
def asia : ℕ := 31

/-- The difference between the number of families that migrated to Africa and Asia -/
def difference : ℕ := 11

/-- Theorem: The total number of bird families before migration is 73 -/
theorem total_bird_families : africa + asia = 73 ∧ africa = asia + difference := by
  sorry

end total_bird_families_l3877_387777


namespace max_product_of_focal_distances_l3877_387766

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The maximum value of |PF1| * |PF2| is 25 for any point P on the ellipse -/
theorem max_product_of_focal_distances :
  ∀ P : ℝ × ℝ, is_on_ellipse P.1 P.2 →
  ∃ M : ℝ, M = 25 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
  (distance P F1) * (distance P F2) ≤ M :=
sorry

end max_product_of_focal_distances_l3877_387766


namespace bike_shop_profit_l3877_387737

/-- Jim's bike shop problem -/
theorem bike_shop_profit (tire_repair_price : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
  (complex_repair_price : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
  (fixed_expenses : ℕ) (total_profit : ℕ) :
  tire_repair_price = 20 →
  tire_repair_cost = 5 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  (tire_repairs * (tire_repair_price - tire_repair_cost) +
   complex_repairs * (complex_repair_price - complex_repair_cost) -
   fixed_expenses + 2000) = total_profit :=
by sorry

end bike_shop_profit_l3877_387737


namespace squirrel_acorns_l3877_387798

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 7 →
  acorns_collected = 875 →
  acorns_needed_per_squirrel = 170 →
  (acorns_needed_per_squirrel * num_squirrels - acorns_collected) / num_squirrels = 45 :=
by
  sorry

end squirrel_acorns_l3877_387798


namespace ferry_speed_proof_l3877_387741

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 1

/-- The travel time of ferry P in hours -/
def time_P : ℝ := 3

/-- The travel time of ferry Q in hours -/
def time_Q : ℝ := time_P + 5

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := speed_Q * time_Q

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 1 ∧
  time_P = 3 ∧
  time_Q = time_P + 5 ∧
  distance_Q = 3 * distance_P :=
by sorry

end ferry_speed_proof_l3877_387741


namespace decreased_equilateral_angle_l3877_387774

/-- The measure of an angle in an equilateral triangle -/
def equilateral_angle : ℝ := 60

/-- The amount by which angle E is decreased -/
def angle_decrease : ℝ := 15

/-- Theorem: In an equilateral triangle where one angle is decreased by 15 degrees, 
    the measure of the decreased angle is 45 degrees -/
theorem decreased_equilateral_angle :
  equilateral_angle - angle_decrease = 45 := by sorry

end decreased_equilateral_angle_l3877_387774


namespace square_sum_inequality_l3877_387743

theorem square_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  a^2 + b^2 ≥ 8 := by
  sorry

end square_sum_inequality_l3877_387743


namespace positive_number_square_sum_l3877_387720

theorem positive_number_square_sum (x : ℝ) : 
  0 < x → x < 15 → x^2 + x = 210 → x = 14 := by sorry

end positive_number_square_sum_l3877_387720


namespace root_magnitude_theorem_l3877_387713

theorem root_magnitude_theorem (p : ℝ) (r₁ r₂ : ℝ) :
  (r₁ ≠ r₂) →
  (r₁^2 + p*r₁ + 12 = 0) →
  (r₂^2 + p*r₂ + 12 = 0) →
  (abs r₁ > 4 ∨ abs r₂ > 4) :=
by sorry

end root_magnitude_theorem_l3877_387713


namespace line_parabola_intersection_range_l3877_387782

/-- The range of m for which a line and a parabola have exactly one common point -/
theorem line_parabola_intersection_range (m : ℝ) : 
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ 
   (2 * x - 2 * m = x^2 + m * x - 1)) ↔ 
  (-3/5 < m ∧ m < 5) :=
by sorry

end line_parabola_intersection_range_l3877_387782


namespace smallest_composite_with_large_factors_l3877_387740

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_with_large_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_with_large_factors_l3877_387740


namespace crayon_selection_count_l3877_387767

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl must select -/
def crayons_to_select : ℕ := 5

/-- The number of non-red crayons to select -/
def non_red_to_select : ℕ := crayons_to_select - 1

/-- The number of non-red crayons available -/
def available_non_red : ℕ := total_crayons - 1

theorem crayon_selection_count :
  (Nat.choose available_non_red non_red_to_select) = 1001 := by
  sorry

end crayon_selection_count_l3877_387767


namespace sixth_television_is_three_l3877_387795

def selected_televisions : List Nat := [20, 26, 24, 19, 23, 3]

theorem sixth_television_is_three : 
  selected_televisions.length = 6 ∧ selected_televisions.getLast? = some 3 :=
sorry

end sixth_television_is_three_l3877_387795


namespace prime_counting_inequality_characterize_equality_cases_l3877_387783

-- Define π(x) as the prime counting function
def prime_counting (x : ℕ) : ℕ := sorry

-- Define φ(x) as Euler's totient function
def euler_totient (x : ℕ) : ℕ := sorry

theorem prime_counting_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  prime_counting m - prime_counting n ≤ ((m - 1) * euler_totient n) / n :=
sorry

def equality_cases : List (ℕ × ℕ) :=
  [(1, 1), (2, 1), (3, 1), (3, 2), (5, 2), (7, 2)]

theorem characterize_equality_cases (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (prime_counting m - prime_counting n = ((m - 1) * euler_totient n) / n) ↔
  (m, n) ∈ equality_cases :=
sorry

end prime_counting_inequality_characterize_equality_cases_l3877_387783


namespace josiah_cookies_per_day_l3877_387745

/-- Proves that Josiah purchased 2 cookies each day in March given the conditions --/
theorem josiah_cookies_per_day :
  let total_spent : ℕ := 992
  let cookie_price : ℕ := 16
  let days_in_march : ℕ := 31
  (total_spent / cookie_price) / days_in_march = 2 := by
  sorry

end josiah_cookies_per_day_l3877_387745


namespace seeds_in_first_plot_l3877_387787

/-- The number of seeds planted in the first plot -/
def seeds_plot1 : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_plot2 : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_plot1 : ℚ := 30 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_plot2 : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 32 / 100

/-- Theorem stating that the number of seeds planted in the first plot is 300 -/
theorem seeds_in_first_plot : 
  (germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 : ℚ) = 
  total_germination_rate * (seeds_plot1 + seeds_plot2) ∧ 
  seeds_plot1 = 300 := by sorry

end seeds_in_first_plot_l3877_387787


namespace multiply_divide_sqrt_l3877_387742

theorem multiply_divide_sqrt (x : ℝ) (y : ℝ) (h1 : x = 0.42857142857142855) (h2 : x ≠ 0) :
  Real.sqrt ((x * y) / 7) = x → y = 3 := by
  sorry

end multiply_divide_sqrt_l3877_387742


namespace unique_symmetry_center_l3877_387750

/-- A point is symmetric to another point with respect to a center -/
def isSymmetric (A B O : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 2 * O.1 ∧ A.2 + B.2 = 2 * O.2

/-- A point is a symmetry center of a set of points -/
def isSymmetryCenter (O : ℝ × ℝ) (H : Set (ℝ × ℝ)) : Prop :=
  ∀ A ∈ H, ∃ B ∈ H, isSymmetric A B O

theorem unique_symmetry_center (H : Set (ℝ × ℝ)) (hfin : Set.Finite H) :
  ∀ O O' : ℝ × ℝ, isSymmetryCenter O H → isSymmetryCenter O' H → O = O' := by
  sorry

#check unique_symmetry_center

end unique_symmetry_center_l3877_387750


namespace maximum_marks_proof_l3877_387731

/-- Given a student needs 50% to pass, got 200 marks, and failed by 20 marks, prove the maximum marks are 440. -/
theorem maximum_marks_proof (passing_percentage : Real) (student_marks : Nat) (failing_margin : Nat) :
  passing_percentage = 0.5 →
  student_marks = 200 →
  failing_margin = 20 →
  ∃ (max_marks : Nat), max_marks = 440 ∧ 
    passing_percentage * max_marks = student_marks + failing_margin :=
by sorry

end maximum_marks_proof_l3877_387731


namespace chocolate_ratio_problem_l3877_387752

/-- The number of dark chocolate bars sold given the ratio and white chocolate bars sold -/
def dark_chocolate_bars (white_ratio : ℕ) (dark_ratio : ℕ) (white_bars : ℕ) : ℕ :=
  (dark_ratio * white_bars) / white_ratio

/-- Theorem: Given a ratio of 4:3 for white to dark chocolate and 20 white chocolate bars sold,
    the number of dark chocolate bars sold is 15 -/
theorem chocolate_ratio_problem :
  dark_chocolate_bars 4 3 20 = 15 := by
  sorry

end chocolate_ratio_problem_l3877_387752


namespace power_of_two_preserves_order_l3877_387769

theorem power_of_two_preserves_order (a b : ℝ) : a > b → (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end power_of_two_preserves_order_l3877_387769


namespace inverse_composition_problem_l3877_387756

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the theorem
theorem inverse_composition_problem
  (h : ∀ x, f_inv (g x) = 3 * x + 5)
  (h_inv_f : ∀ x, f_inv (f x) = x)
  (h_inv_g : ∀ x, g_inv (g x) = x)
  (h_f_inv : ∀ x, f (f_inv x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x) :
  g_inv (f (-8)) = -13/3 :=
sorry

end inverse_composition_problem_l3877_387756


namespace random_events_identification_l3877_387711

-- Define the types of events
inductive EventType
  | Random
  | Impossible
  | Certain

-- Define the events
structure Event :=
  (description : String)
  (eventType : EventType)

-- Define the function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  e.eventType = EventType.Random

-- Define the events
def coinEvent : Event :=
  { description := "Picking a 10 cent coin from a pocket with 50 cent, 10 cent, and 1 yuan coins"
  , eventType := EventType.Random }

def waterEvent : Event :=
  { description := "Water boiling at 90°C under standard atmospheric pressure"
  , eventType := EventType.Impossible }

def shooterEvent : Event :=
  { description := "A shooter hitting the 10-ring in one shot"
  , eventType := EventType.Random }

def diceEvent : Event :=
  { description := "Rolling two dice and the sum not exceeding 12"
  , eventType := EventType.Certain }

-- Theorem to prove
theorem random_events_identification :
  (isRandomEvent coinEvent ∧ isRandomEvent shooterEvent) ∧
  (¬isRandomEvent waterEvent ∧ ¬isRandomEvent diceEvent) :=
sorry

end random_events_identification_l3877_387711


namespace intersection_sum_l3877_387726

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 1 = (y - 2)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 20 :=
by
  sorry


end intersection_sum_l3877_387726


namespace A_is_integer_l3877_387704

def n : ℤ := 8795685

def A : ℚ :=
  (((n + 4) * (n + 3) * (n + 2) * (n + 1)) - ((n - 1) * (n - 2) * (n - 3) * (n - 4))) /
  ((n + 3)^2 + (n + 1)^2 + (n - 1)^2 + (n - 3)^2)

theorem A_is_integer : ∃ (k : ℤ), A = k := by
  sorry

end A_is_integer_l3877_387704


namespace no_intersection_and_constraint_l3877_387707

theorem no_intersection_and_constraint (a b : ℝ) : 
  ¬(∃ (x : ℤ), a * (x : ℝ) + b = 3 * (x : ℝ)^2 + 15 ∧ a^2 + b^2 ≤ 144) :=
sorry

end no_intersection_and_constraint_l3877_387707


namespace range_of_g_l3877_387799

open Set
open Function

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  range g = {y : ℝ | y < -27 ∨ y > -27} :=
sorry

end range_of_g_l3877_387799


namespace proof_by_contradiction_elements_l3877_387762

/-- Elements used as conditions in a proof by contradiction -/
inductive ProofByContradictionElement
  | NegatedConclusion
  | OriginalConditions
  | AxiomsTheoremsDefinitions
  | OriginalConclusion

/-- The set of elements that should be used in a proof by contradiction -/
def ValidProofByContradictionElements : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegatedConclusion,
   ProofByContradictionElement.OriginalConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating which elements should be used in a proof by contradiction -/
theorem proof_by_contradiction_elements :
  ValidProofByContradictionElements =
    {ProofByContradictionElement.NegatedConclusion,
     ProofByContradictionElement.OriginalConditions,
     ProofByContradictionElement.AxiomsTheoremsDefinitions} :=
by sorry

end proof_by_contradiction_elements_l3877_387762


namespace smallest_in_odd_set_l3877_387758

/-- A set of consecutive odd integers -/
def ConsecutiveOddIntegers : Set ℤ := sorry

/-- The median of a set of integers -/
def median (s : Set ℤ) : ℚ := sorry

/-- The greatest integer in a set -/
def greatest (s : Set ℤ) : ℤ := sorry

/-- The smallest integer in a set -/
def smallest (s : Set ℤ) : ℤ := sorry

theorem smallest_in_odd_set (s : Set ℤ) :
  s = ConsecutiveOddIntegers ∧
  median s = 152.5 ∧
  greatest s = 161 →
  smallest s = 138 := by sorry

end smallest_in_odd_set_l3877_387758


namespace nonzero_real_solution_l3877_387716

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 := by
  sorry

end nonzero_real_solution_l3877_387716


namespace probability_empty_bottle_day14_expected_pills_taken_l3877_387754

/-- Represents the pill-taking scenario with two bottles --/
structure PillScenario where
  totalDays : ℕ
  pillsPerBottle : ℕ
  bottles : ℕ

/-- Calculates the probability of finding an empty bottle on a specific day --/
def probabilityEmptyBottle (scenario : PillScenario) (day : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of pills taken when discovering an empty bottle --/
def expectedPillsTaken (scenario : PillScenario) : ℚ :=
  sorry

/-- Theorem stating the probability of finding an empty bottle on the 14th day --/
theorem probability_empty_bottle_day14 (scenario : PillScenario) :
  scenario.totalDays = 14 ∧ scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  probabilityEmptyBottle scenario 14 = 143 / 4096 :=
sorry

/-- Theorem stating the expected number of pills taken when discovering an empty bottle --/
theorem expected_pills_taken (scenario : PillScenario) (ε : ℚ) :
  scenario.pillsPerBottle = 10 ∧ scenario.bottles = 2 →
  ∃ n : ℕ, abs (expectedPillsTaken scenario - 173 / 10) < ε ∧ n > 0 :=
sorry

end probability_empty_bottle_day14_expected_pills_taken_l3877_387754


namespace a_plus_b_eighth_power_l3877_387759

theorem a_plus_b_eighth_power (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^8 + b^8 = 47 := by
  sorry

end a_plus_b_eighth_power_l3877_387759


namespace age_calculation_l3877_387781

-- Define the current ages and time intervals
def luke_current_age : ℕ := 20
def years_to_future : ℕ := 8
def years_to_luke_future : ℕ := 4

-- Define the relationships between ages
def mr_bernard_future_age : ℕ := 3 * luke_current_age
def luke_future_age : ℕ := luke_current_age + years_to_future
def sarah_future_age : ℕ := 2 * (luke_current_age + years_to_luke_future)

-- Calculate the average future age
def average_future_age : ℚ := (mr_bernard_future_age + luke_future_age + sarah_future_age) / 3

-- Define the final result
def result : ℚ := average_future_age - 10

-- Theorem to prove
theorem age_calculation :
  result = 35 + 1/3 :=
sorry

end age_calculation_l3877_387781


namespace continued_fraction_value_l3877_387773

theorem continued_fraction_value : ∃ x : ℝ, 
  x = 3 + 4 / (1 + 4 / (3 + 4 / ((1/2) + x))) ∧ 
  x = (43 + Real.sqrt 4049) / 22 := by
  sorry

end continued_fraction_value_l3877_387773


namespace number_of_valid_arrangements_l3877_387755

-- Define the triangular arrangement
structure TriangularArrangement :=
  (cells : Fin 9 → Nat)

-- Define the condition for valid placement
def ValidPlacement (arr : TriangularArrangement) : Prop :=
  -- Each number from 1 to 9 is used exactly once
  (∀ n : Fin 9, ∃! i : Fin 9, arr.cells i = n.val + 1) ∧
  -- The sum in each four-cell triangle is 23
  (arr.cells 0 + arr.cells 1 + arr.cells 3 + arr.cells 4 = 23) ∧
  (arr.cells 1 + arr.cells 2 + arr.cells 4 + arr.cells 5 = 23) ∧
  (arr.cells 3 + arr.cells 4 + arr.cells 6 + arr.cells 7 = 23) ∧
  -- Specific placements as indicated by arrows
  (arr.cells 3 = 7 ∨ arr.cells 6 = 7) ∧
  (arr.cells 1 = 2 ∨ arr.cells 2 = 2 ∨ arr.cells 4 = 2 ∨ arr.cells 5 = 2)

-- The theorem to be proved
theorem number_of_valid_arrangements :
  ∃! (arrangements : Finset TriangularArrangement),
    (∀ arr ∈ arrangements, ValidPlacement arr) ∧
    arrangements.card = 4 := by
  sorry

end number_of_valid_arrangements_l3877_387755


namespace regression_line_equation_l3877_387788

/-- Given a regression line with slope 1.2 passing through the point (4, 5),
    prove that its equation is ŷ = 1.2x + 0.2 -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center_x : ℝ) 
  (center_y : ℝ) 
  (h1 : slope = 1.2) 
  (h2 : center_x = 4) 
  (h3 : center_y = 5) : 
  ∃ (a : ℝ), ∀ (x y : ℝ), y = slope * x + a ↔ (x = center_x ∧ y = center_y) ∨ y = 1.2 * x + 0.2 :=
sorry

end regression_line_equation_l3877_387788


namespace inequality_direction_change_l3877_387770

theorem inequality_direction_change (a b x : ℝ) (h : x < 0) :
  (a < b) ↔ (a * x > b * x) :=
by sorry

end inequality_direction_change_l3877_387770


namespace value_of_a_l3877_387735

theorem value_of_a (x y z a : ℝ) 
  (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) 
  (h2 : a > 0)
  (h3 : ∀ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a → x' + y' + z' ≤ 1) 
  (h4 : ∃ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a ∧ x' + y' + z' = 1) : 
  a = 1 := by
sorry

end value_of_a_l3877_387735


namespace floor_sqrt_50_l3877_387784

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by sorry

end floor_sqrt_50_l3877_387784


namespace least_positive_integer_for_multiple_of_five_l3877_387733

theorem least_positive_integer_for_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by
  sorry

end least_positive_integer_for_multiple_of_five_l3877_387733


namespace round_trip_average_speed_l3877_387729

/-- The average speed of a round trip given different speeds for each direction -/
theorem round_trip_average_speed (speed_to_school speed_from_school : ℝ) :
  speed_to_school > 0 →
  speed_from_school > 0 →
  let average_speed := 2 / (1 / speed_to_school + 1 / speed_from_school)
  average_speed = 4.8 ↔ speed_to_school = 6 ∧ speed_from_school = 4 :=
by sorry

end round_trip_average_speed_l3877_387729


namespace hyperbola_k_range_l3877_387723

/-- Represents a hyperbola with the given equation and foci on the y-axis -/
structure Hyperbola (k : ℝ) :=
  (equation : ∀ x y : ℝ, x^2 / (k - 3) + y^2 / (k + 3) = 1)
  (foci_on_y_axis : True)  -- We can't directly represent this condition, so we use a placeholder

/-- The range of k for a hyperbola with the given properties -/
def k_range (h : Hyperbola k) : Set ℝ :=
  {k | -3 < k ∧ k < 3}

/-- Theorem stating that for any hyperbola satisfying the given conditions, k is in the range (-3, 3) -/
theorem hyperbola_k_range (k : ℝ) (h : Hyperbola k) : k ∈ k_range h := by
  sorry

end hyperbola_k_range_l3877_387723


namespace number_of_students_in_class_l3877_387736

/-- Proves that the number of students in a class is 23 given certain grade conditions --/
theorem number_of_students_in_class 
  (recorded_biology : ℝ) 
  (recorded_chemistry : ℝ)
  (actual_biology : ℝ) 
  (actual_chemistry : ℝ)
  (subject_weight : ℝ)
  (class_average_increase : ℝ)
  (initial_class_average : ℝ)
  (h1 : recorded_biology = 83)
  (h2 : recorded_chemistry = 85)
  (h3 : actual_biology = 70)
  (h4 : actual_chemistry = 75)
  (h5 : subject_weight = 0.5)
  (h6 : class_average_increase = 0.5)
  (h7 : initial_class_average = 80) :
  ∃ n : ℕ, n = 23 ∧ n * class_average_increase = (recorded_biology * subject_weight + recorded_chemistry * subject_weight) - (actual_biology * subject_weight + actual_chemistry * subject_weight) := by
  sorry

end number_of_students_in_class_l3877_387736


namespace integer_valued_poly_implies_24P_integer_coeffs_l3877_387763

/-- A polynomial of degree 4 that takes integer values for integer inputs -/
def IntegerValuedPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  (∀ n : ℤ, ∃ m : ℤ, P n = m)

/-- The coefficients of 24P(x) are integers -/
def Coefficients24PAreIntegers (P : ℝ → ℝ) : Prop :=
  ∃ a' b' c' d' e' : ℤ, ∀ x, 24 * P x = a'*x^4 + b'*x^3 + c'*x^2 + d'*x + e'

theorem integer_valued_poly_implies_24P_integer_coeffs
  (P : ℝ → ℝ) (h : IntegerValuedPolynomial P) :
  Coefficients24PAreIntegers P :=
sorry

end integer_valued_poly_implies_24P_integer_coeffs_l3877_387763
