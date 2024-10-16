import Mathlib

namespace NUMINAMATH_CALUDE_carnival_tickets_l3070_307082

theorem carnival_tickets (num_games : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ) :
  num_games = 5 →
  found_tickets = 5 →
  ticket_value = 3 →
  total_value = 30 →
  ∃ (tickets_per_game : ℕ),
    (tickets_per_game * num_games + found_tickets) * ticket_value = total_value ∧
    tickets_per_game = 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l3070_307082


namespace NUMINAMATH_CALUDE_solution_proof_l3070_307044

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem solution_proof : ∃ x : ℝ, g x = 20 ∧ x = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_proof_l3070_307044


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l3070_307026

theorem rectangular_field_with_pond (w l : ℝ) : 
  l = 2 * w →                 -- length is double the width
  36 = (1/8) * (l * w) →      -- pond area (6^2) is 1/8 of field area
  l = 24 := by               -- length of the field is 24 meters
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l3070_307026


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3070_307006

theorem fixed_point_on_line (a b c : ℝ) (h : a + b - c = 0) (h2 : ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  a * (-1) + b * (-1) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3070_307006


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3070_307052

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (average_books : ℚ) (h1 : total_students = 25) (h2 : zero_books = 3) (h3 : one_book = 10) 
  (h4 : two_books = 4) (h5 : average_books = 5/2) : ℕ :=
  let total_books := (total_students : ℚ) * average_books
  let accounted_students := zero_books + one_book + two_books
  let remaining_students := total_students - accounted_students
  let accounted_books := one_book * 1 + two_books * 2
  let remaining_books := total_books - accounted_books
  let min_books_per_remaining := 3
  24

end NUMINAMATH_CALUDE_max_books_borrowed_l3070_307052


namespace NUMINAMATH_CALUDE_max_sum_of_arithmetic_progression_l3070_307021

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem max_sum_of_arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :
  arithmetic_progression a d →
  (∀ n, a n > 0) →
  a 3 = 13 →
  (∀ n, a (n + 1) > a n) →
  (a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) ≤ 365) ∧
  (∃ a d, arithmetic_progression a d ∧
          (∀ n, a n > 0) ∧
          a 3 = 13 ∧
          (∀ n, a (n + 1) > a n) ∧
          a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) = 365) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_arithmetic_progression_l3070_307021


namespace NUMINAMATH_CALUDE_percentage_calculation_l3070_307087

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 300) : 1.2 * x = 1800 := by
  sorry


end NUMINAMATH_CALUDE_percentage_calculation_l3070_307087


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3070_307004

theorem arithmetic_geometric_mean_inequality
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3070_307004


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3070_307001

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of 5 coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def totalOutcomes : Nat := 32

/-- Predicate for successful outcomes (penny, dime, and half-dollar are heads) -/
def isSuccessfulOutcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads ∧ cs.halfDollar = CoinOutcome.Heads

/-- The number of successful outcomes -/
def successfulOutcomes : Nat := 4

/-- The probability of getting heads on penny, dime, and half-dollar -/
def probability : Rat := 1 / 8

theorem coin_flip_probability :
  (successfulOutcomes : Rat) / totalOutcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3070_307001


namespace NUMINAMATH_CALUDE_max_S_value_l3070_307016

theorem max_S_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_S_value_l3070_307016


namespace NUMINAMATH_CALUDE_f_monotone_increasing_range_l3070_307031

/-- The function f(x) defined on the interval [0,1] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a - 1) * x + 3

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a - 1)

/-- Theorem stating the range of a for which f(x) is monotonically increasing on [0,1] -/
theorem f_monotone_increasing_range :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, f_derivative a x ≥ 0} = Set.Iic (1/2) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_range_l3070_307031


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3070_307097

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3070_307097


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l3070_307027

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l3070_307027


namespace NUMINAMATH_CALUDE_kwik_e_tax_revenue_l3070_307055

/-- Calculates the total revenue for Kwik-e-Tax Center given the prices and number of returns sold --/
def total_revenue (federal_price state_price quarterly_price : ℕ) 
                  (federal_sold state_sold quarterly_sold : ℕ) : ℕ :=
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold

/-- Theorem stating that the total revenue for the given scenario is $4400 --/
theorem kwik_e_tax_revenue : 
  total_revenue 50 30 80 60 20 10 = 4400 := by
sorry

end NUMINAMATH_CALUDE_kwik_e_tax_revenue_l3070_307055


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3070_307030

theorem quadratic_form_equivalence (y : ℝ) : y^2 - 8*y = (y - 4)^2 - 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3070_307030


namespace NUMINAMATH_CALUDE_car_speed_increase_l3070_307051

/-- Calculates the final speed of a car after modifications -/
def final_speed (original_speed : ℝ) (supercharge_percentage : ℝ) (weight_cut_increase : ℝ) : ℝ :=
  original_speed * (1 + supercharge_percentage) + weight_cut_increase

/-- Theorem stating that the final speed is 205 mph given the specified conditions -/
theorem car_speed_increase :
  final_speed 150 0.3 10 = 205 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_increase_l3070_307051


namespace NUMINAMATH_CALUDE_max_inspector_sum_l3070_307073

/-- Represents the configuration of towers in the city of Flat -/
structure TowerConfiguration where
  one_floor : ℕ  -- Number of 1-floor towers
  two_floor : ℕ  -- Number of 2-floor towers

/-- Calculates the total height of all towers -/
def total_height (config : TowerConfiguration) : ℕ :=
  config.one_floor + 2 * config.two_floor

/-- Calculates the inspector's sum for a given configuration -/
def inspector_sum (config : TowerConfiguration) : ℕ :=
  config.one_floor * config.two_floor

/-- Theorem stating that the maximum inspector's sum is 112 -/
theorem max_inspector_sum :
  ∃ (config : TowerConfiguration),
    total_height config = 30 ∧
    inspector_sum config = 112 ∧
    ∀ (other : TowerConfiguration),
      total_height other = 30 →
      inspector_sum other ≤ 112 := by
  sorry

end NUMINAMATH_CALUDE_max_inspector_sum_l3070_307073


namespace NUMINAMATH_CALUDE_square_area_is_40_l3070_307088

/-- A parabola defined by y = x^2 + 4x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 1

/-- The y-coordinate of the line that coincides with one side of the square -/
def line_y : ℝ := 7

/-- The theorem stating that the area of the square is 40 -/
theorem square_area_is_40 :
  ∃ (x1 x2 : ℝ),
    parabola x1 = line_y ∧
    parabola x2 = line_y ∧
    x1 ≠ x2 ∧
    (x2 - x1)^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_square_area_is_40_l3070_307088


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3070_307071

/-- Given a point M with polar coordinates (2, 2π/3), its Cartesian coordinates are (-1, √3) -/
theorem polar_to_cartesian :
  let ρ : ℝ := 2
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3070_307071


namespace NUMINAMATH_CALUDE_f_of_3_equals_11_f_equiv_l3070_307058

-- Define the function f
def f (t : ℝ) : ℝ := t^2 + 2

-- State the theorem
theorem f_of_3_equals_11 : f 3 = 11 := by
  sorry

-- Define the original function property
axiom f_property (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2

-- Prove the equivalence of the two function definitions
theorem f_equiv (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_11_f_equiv_l3070_307058


namespace NUMINAMATH_CALUDE_tournament_ceremony_theorem_l3070_307007

def tournament_ceremony_length (initial_players : Nat) (initial_ceremony_length : Nat) (ceremony_increase : Nat) : Nat :=
  let rounds := Nat.log2 initial_players
  let ceremony_lengths := List.range rounds |>.map (λ i => initial_ceremony_length + i * ceremony_increase)
  let winners_per_round := List.range rounds |>.map (λ i => initial_players / (2^(i+1)))
  List.sum (List.zipWith (·*·) ceremony_lengths winners_per_round)

theorem tournament_ceremony_theorem :
  tournament_ceremony_length 16 10 10 = 260 := by
  sorry

end NUMINAMATH_CALUDE_tournament_ceremony_theorem_l3070_307007


namespace NUMINAMATH_CALUDE_equation_solution_l3070_307060

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (1/3 * (4 * x₁^2 - 2) = (x₁^2 - 60*x₁ - 15) * (x₁^2 + 30*x₁ + 3)) ∧
    (1/3 * (4 * x₂^2 - 2) = (x₂^2 - 60*x₂ - 15) * (x₂^2 + 30*x₂ + 3)) ∧
    x₁ = 30 + Real.sqrt 917 ∧
    x₂ = -15 + Real.sqrt 8016 / 6 ∧
    ∀ (y : ℝ), 
      y > 0 ∧ (1/3 * (4 * y^2 - 2) = (y^2 - 60*y - 15) * (y^2 + 30*y + 3)) →
      (y = x₁ ∨ y = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3070_307060


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l3070_307070

theorem triangle_ratio_theorem (a b c : ℝ) (A B C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  (3 * a + b) / (3 * Real.sin A + Real.sin B) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l3070_307070


namespace NUMINAMATH_CALUDE_largest_y_value_l3070_307002

theorem largest_y_value : 
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y ∧ y = 3/25) := by
  sorry

end NUMINAMATH_CALUDE_largest_y_value_l3070_307002


namespace NUMINAMATH_CALUDE_square_equals_self_implies_zero_or_one_l3070_307056

theorem square_equals_self_implies_zero_or_one (a : ℝ) : a^2 = a → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_self_implies_zero_or_one_l3070_307056


namespace NUMINAMATH_CALUDE_mary_stickers_remaining_l3070_307050

theorem mary_stickers_remaining (initial_stickers : ℕ) 
                                 (front_page_stickers : ℕ) 
                                 (other_pages : ℕ) 
                                 (stickers_per_other_page : ℕ) 
                                 (h1 : initial_stickers = 89)
                                 (h2 : front_page_stickers = 3)
                                 (h3 : other_pages = 6)
                                 (h4 : stickers_per_other_page = 7) : 
  initial_stickers - (front_page_stickers + other_pages * stickers_per_other_page) = 44 := by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_remaining_l3070_307050


namespace NUMINAMATH_CALUDE_range_of_k_l3070_307083

def system_of_inequalities (x k : ℝ) : Prop :=
  x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+7)*x + 7*k < 0

def integer_solutions (k : ℝ) : Prop :=
  ∀ x : ℤ, system_of_inequalities (x : ℝ) k ↔ x = -3 ∨ x = -2

theorem range_of_k :
  ∀ k : ℝ, integer_solutions k → k ∈ Set.Ici (-3) ∩ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3070_307083


namespace NUMINAMATH_CALUDE_sam_seashells_l3070_307038

/-- The number of seashells Sam has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sam has 17 seashells after giving away 18 from his initial 35 -/
theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l3070_307038


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_l3070_307041

def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, -1, -2)

theorem magnitude_a_minus_2b : 
  ‖(a.1 - 2 * b.1, a.2 - 2 * b.2, a.2.2 - 2 * b.2.2)‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_l3070_307041


namespace NUMINAMATH_CALUDE_spelling_bee_participants_l3070_307066

/-- In a competition, given a participant's ranking from best and worst, determine the total number of participants. -/
theorem spelling_bee_participants (n : ℕ) 
  (h_best : n = 75)  -- Priya is the 75th best
  (h_worst : n = 75) -- Priya is the 75th worst
  : (2 * n - 1 : ℕ) = 149 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_participants_l3070_307066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3070_307003

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3070_307003


namespace NUMINAMATH_CALUDE_gcd_of_117_and_182_l3070_307034

theorem gcd_of_117_and_182 :
  Nat.gcd 117 182 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_117_and_182_l3070_307034


namespace NUMINAMATH_CALUDE_bicycle_cost_l3070_307009

def hourly_rate : ℕ := 5
def monday_hours : ℕ := 2
def wednesday_hours : ℕ := 1
def friday_hours : ℕ := 3
def weeks_to_work : ℕ := 6

def weekly_hours : ℕ := monday_hours + wednesday_hours + friday_hours

def weekly_earnings : ℕ := weekly_hours * hourly_rate

theorem bicycle_cost : weekly_earnings * weeks_to_work = 180 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_l3070_307009


namespace NUMINAMATH_CALUDE_triangle_area_l3070_307077

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(5, 11) is 27 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (5, 11)
  let area := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 27 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3070_307077


namespace NUMINAMATH_CALUDE_fraction_equality_l3070_307090

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3070_307090


namespace NUMINAMATH_CALUDE_farm_animals_l3070_307047

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8)
  (h2 : total_legs = 24) :
  ∃ (ducks dogs : ℕ),
    ducks + dogs = total_animals ∧
    2 * ducks + 4 * dogs = total_legs ∧
    ducks = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3070_307047


namespace NUMINAMATH_CALUDE_practicing_to_writing_ratio_l3070_307059

/-- Represents the time spent on different activities for a speech --/
structure SpeechTime where
  outlining : ℕ
  writing : ℕ
  practicing : ℕ
  total : ℕ

/-- Defines the conditions of Javier's speech preparation --/
def javierSpeechTime : SpeechTime where
  outlining := 30
  writing := 30 + 28
  practicing := 117 - (30 + 58)
  total := 117

/-- Theorem stating the ratio of practicing to writing time --/
theorem practicing_to_writing_ratio :
  (javierSpeechTime.practicing : ℚ) / javierSpeechTime.writing = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_practicing_to_writing_ratio_l3070_307059


namespace NUMINAMATH_CALUDE_sine_function_value_l3070_307089

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0,
    if the distance between adjacent maximum and minimum points is 2√2,
    then f(1) = √3/2 -/
theorem sine_function_value (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (∃ A B : ℝ × ℝ, 
    (A.2 = f A.1 ∧ B.2 = f B.1) ∧ 
    (∀ x ∈ Set.Icc A.1 B.1, f x ≤ A.2 ∧ f x ≥ B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2) →
  f 1 = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_function_value_l3070_307089


namespace NUMINAMATH_CALUDE_negation_of_false_l3070_307081

theorem negation_of_false (p q : Prop) : p ∧ ¬q → ¬q := by
  sorry

end NUMINAMATH_CALUDE_negation_of_false_l3070_307081


namespace NUMINAMATH_CALUDE_range_of_a_l3070_307072

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3070_307072


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3070_307036

-- Define the colors
inductive Color
| Red
| Blue

-- Define the coloring function type
def ColoringFunction := ℕ → Color

-- Define an infinite arithmetic progression
def IsArithmeticProgression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a property that a coloring function contains both colors in any arithmetic progression
def ContainsBothColors (f : ColoringFunction) : Prop :=
  ∀ (a : ℕ → ℕ) (d : ℕ), 
    IsArithmeticProgression a d → 
    (∃ n : ℕ, f (a n) = Color.Red) ∧ (∃ m : ℕ, f (a m) = Color.Blue)

-- The main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ContainsBothColors f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3070_307036


namespace NUMINAMATH_CALUDE_prob_diff_absolute_l3070_307023

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / total_marbles.choose 2

/-- Theorem: The absolute difference between the probability of drawing two marbles
    of the same color and the probability of drawing two marbles of different colors
    is 789/19990 -/
theorem prob_diff_absolute : |prob_same_color - prob_diff_color| = 789 / 19990 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_absolute_l3070_307023


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l3070_307014

def base7_to_base10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

def pirate_loot : Nat :=
  base7_to_base10 4516 + base7_to_base10 3216 + base7_to_base10 654 + base7_to_base10 301

theorem pirate_loot_sum :
  pirate_loot = 3251 := by sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l3070_307014


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3070_307099

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3070_307099


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3070_307080

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i : i^2024 + i^2025 + i^2026 + i^2027 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3070_307080


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3070_307024

/-- Man's rowing problem with wind resistance -/
theorem mans_rowing_speed (upstream_speed downstream_speed wind_effect : ℝ) 
  (h1 : upstream_speed = 25)
  (h2 : downstream_speed = 45)
  (h3 : wind_effect = 2) :
  let still_water_speed := (upstream_speed + downstream_speed) / 2
  let adjusted_upstream_speed := upstream_speed - wind_effect
  let adjusted_downstream_speed := downstream_speed + wind_effect
  let adjusted_still_water_speed := (adjusted_upstream_speed + adjusted_downstream_speed) / 2
  adjusted_still_water_speed = still_water_speed :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3070_307024


namespace NUMINAMATH_CALUDE_paving_cost_proof_l3070_307053

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Proof that the cost of paving the given floor is Rs. 28,875 -/
theorem paving_cost_proof :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_proof_l3070_307053


namespace NUMINAMATH_CALUDE_age_ratio_l3070_307068

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.betty = 5

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) 
  (h : satisfiesConditions ages) : 
  ages.albert / ages.mary = 2 := by
sorry


end NUMINAMATH_CALUDE_age_ratio_l3070_307068


namespace NUMINAMATH_CALUDE_jen_addition_problem_l3070_307010

/-- Rounds a natural number to the nearest hundred. -/
def roundToNearestHundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

/-- The problem statement -/
theorem jen_addition_problem :
  roundToNearestHundred (178 + 269) = 400 := by
  sorry

end NUMINAMATH_CALUDE_jen_addition_problem_l3070_307010


namespace NUMINAMATH_CALUDE_unique_solution_for_F_l3070_307033

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c

/-- Theorem stating that -5/19 is the unique solution for a in F(a, 2, 3) = F(a, 3, 8) -/
theorem unique_solution_for_F :
  ∃! a : ℝ, F a 2 3 = F a 3 8 ∧ a = -5/19 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_F_l3070_307033


namespace NUMINAMATH_CALUDE_constant_function_proof_l3070_307091

def IsFunctionalRelation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * x)

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : Continuous f) 
  (h2 : IsFunctionalRelation f) : 
  ∀ x : ℝ, f x = f 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3070_307091


namespace NUMINAMATH_CALUDE_sam_fish_count_l3070_307084

theorem sam_fish_count (harry joe sam : ℕ) 
  (harry_joe : harry = 4 * joe)
  (joe_sam : joe = 8 * sam)
  (harry_count : harry = 224) : 
  sam = 7 := by
sorry

end NUMINAMATH_CALUDE_sam_fish_count_l3070_307084


namespace NUMINAMATH_CALUDE_percentage_product_theorem_l3070_307025

theorem percentage_product_theorem :
  let p1 : ℝ := 40
  let p2 : ℝ := 35
  let p3 : ℝ := 60
  let p4 : ℝ := 70
  let result : ℝ := p1 * p2 * p3 * p4 / 1000000 * 100
  result = 5.88 := by
sorry

end NUMINAMATH_CALUDE_percentage_product_theorem_l3070_307025


namespace NUMINAMATH_CALUDE_coffee_cost_theorem_l3070_307039

/-- The cost of coffee A per kilogram -/
def coffee_A_cost : ℝ := 10

/-- The cost of coffee B per kilogram -/
def coffee_B_cost : ℝ := 12

/-- The selling price of the mixture per kilogram -/
def mixture_price : ℝ := 11

/-- The total weight of the mixture in kilograms -/
def total_mixture : ℝ := 480

/-- The weight of coffee A used in the mixture in kilograms -/
def coffee_A_weight : ℝ := 240

/-- The weight of coffee B used in the mixture in kilograms -/
def coffee_B_weight : ℝ := 240

theorem coffee_cost_theorem :
  coffee_A_weight * coffee_A_cost + coffee_B_weight * coffee_B_cost = total_mixture * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_cost_theorem_l3070_307039


namespace NUMINAMATH_CALUDE_inequality_solution_l3070_307017

theorem inequality_solution (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 7) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 3) ∨ (7/2 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3070_307017


namespace NUMINAMATH_CALUDE_adams_final_score_l3070_307074

/-- Calculates the final score in a trivia game -/
def final_score (correct_first_half correct_second_half points_per_question : ℕ) : ℕ :=
  (correct_first_half + correct_second_half) * points_per_question

/-- Theorem: Adam's final score in the trivia game is 50 points -/
theorem adams_final_score : 
  final_score 5 5 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_adams_final_score_l3070_307074


namespace NUMINAMATH_CALUDE_sphere_intersection_radius_l3070_307086

/-- Given a sphere that intersects the xy-plane in a circle centered at (3, 5, 0) with radius 2
    and intersects the yz-plane in a circle centered at (0, 5, -8),
    prove that the radius of the circle in the yz-plane is √59. -/
theorem sphere_intersection_radius (center : ℝ × ℝ × ℝ) (R : ℝ) :
  center = (3, 5, -8) →
  R^2 = 68 →
  let r := Real.sqrt (R^2 - 3^2)
  r^2 = 59 := by
  sorry


end NUMINAMATH_CALUDE_sphere_intersection_radius_l3070_307086


namespace NUMINAMATH_CALUDE_gustran_nails_cost_l3070_307061

structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

def gustran_salon : Salon := {
  name := "Gustran Salon"
  haircut := 45
  facial := 22
  nails := 0  -- Unknown, to be proved
}

def barbaras_shop : Salon := {
  name := "Barbara's Shop"
  haircut := 30
  facial := 28
  nails := 40
}

def fancy_salon : Salon := {
  name := "The Fancy Salon"
  haircut := 34
  facial := 30
  nails := 20
}

def total_cost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

theorem gustran_nails_cost :
  ∃ (x : ℕ), 
    gustran_salon.nails = x ∧ 
    total_cost gustran_salon = 84 ∧
    total_cost barbaras_shop ≥ 84 ∧
    total_cost fancy_salon = 84 ∧
    x = 17 := by sorry

end NUMINAMATH_CALUDE_gustran_nails_cost_l3070_307061


namespace NUMINAMATH_CALUDE_prob_end_multiple_3_l3070_307037

/-- The number of cards --/
def num_cards : ℕ := 15

/-- The probability of moving left on the spinner --/
def prob_left : ℚ := 1/4

/-- The probability of moving right on the spinner --/
def prob_right : ℚ := 3/4

/-- The probability of starting at a multiple of 3 --/
def prob_start_multiple_3 : ℚ := 1/3

/-- The probability of starting one more than a multiple of 3 --/
def prob_start_one_more : ℚ := 4/15

/-- The probability of starting one less than a multiple of 3 --/
def prob_start_one_less : ℚ := 1/3

/-- The probability of ending at a multiple of 3 after two spins --/
theorem prob_end_multiple_3 : 
  prob_start_multiple_3 * prob_left * prob_left +
  prob_start_one_more * prob_right * prob_right +
  prob_start_one_less * prob_left * prob_left = 7/30 := by sorry

end NUMINAMATH_CALUDE_prob_end_multiple_3_l3070_307037


namespace NUMINAMATH_CALUDE_number_puzzle_l3070_307032

theorem number_puzzle (y : ℝ) (h : y ≠ 0) : y = (1 / y) * (-y) + 5 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3070_307032


namespace NUMINAMATH_CALUDE_science_class_ends_at_350pm_l3070_307063

-- Define the start time and class durations
def school_start_time : Nat := 12 * 60  -- 12:00 pm in minutes
def maths_duration : Nat := 45
def history_duration : Nat := 75  -- 1 hour and 15 minutes
def geography_duration : Nat := 30
def science_duration : Nat := 50
def break_duration : Nat := 10

-- Define a function to calculate the end time of Science class
def science_class_end_time : Nat :=
  school_start_time +
  maths_duration + break_duration +
  history_duration + break_duration +
  geography_duration + break_duration +
  science_duration

-- Convert minutes to hours and minutes
def minutes_to_time (minutes : Nat) : String :=
  let hours := minutes / 60
  let mins := minutes % 60
  s!"{hours}:{mins}"

-- Theorem to prove
theorem science_class_ends_at_350pm :
  minutes_to_time science_class_end_time = "3:50" :=
by sorry

end NUMINAMATH_CALUDE_science_class_ends_at_350pm_l3070_307063


namespace NUMINAMATH_CALUDE_first_class_product_rate_l3070_307043

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    calculate the overall rate of first-class products. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h_pass_rate : pass_rate = 0.95)
  (h_first_class_rate_among_qualified : first_class_rate_among_qualified = 0.2) :
  pass_rate * first_class_rate_among_qualified = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_first_class_product_rate_l3070_307043


namespace NUMINAMATH_CALUDE_jerry_lawsuit_amount_correct_l3070_307029

/-- Calculates the amount Jerry gets from his lawsuit --/
def jerryLawsuitAmount (annualSalary : ℕ) (years : ℕ) (medicalBills : ℕ) (punitiveMultiplier : ℕ) (awardedPercentage : ℚ) : ℚ :=
  let totalSalary := annualSalary * years
  let directDamages := totalSalary + medicalBills
  let punitiveDamages := directDamages * punitiveMultiplier
  let totalAsked := directDamages + punitiveDamages
  totalAsked * awardedPercentage

theorem jerry_lawsuit_amount_correct :
  jerryLawsuitAmount 50000 30 200000 3 (4/5) = 5440000 := by
  sorry

#eval jerryLawsuitAmount 50000 30 200000 3 (4/5)

end NUMINAMATH_CALUDE_jerry_lawsuit_amount_correct_l3070_307029


namespace NUMINAMATH_CALUDE_trapezoid_base_lengths_l3070_307045

theorem trapezoid_base_lengths (h : ℝ) (leg1 leg2 larger_base : ℝ) :
  h = 12 ∧ leg1 = 20 ∧ leg2 = 15 ∧ larger_base = 42 →
  ∃ (smaller_base : ℝ), (smaller_base = 17 ∨ smaller_base = 35) ∧
  (∃ (x y : ℝ), x^2 + h^2 = leg1^2 ∧ y^2 + h^2 = leg2^2 ∧
  (larger_base = x + y + smaller_base ∨ larger_base = x - y + smaller_base)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_base_lengths_l3070_307045


namespace NUMINAMATH_CALUDE_paint_area_is_129_l3070_307079

/-- The area of the wall to be painted, given the dimensions of the wall, window, and door. -/
def areaToBePainted (wallHeight wallLength windowHeight windowLength doorHeight doorLength : ℕ) : ℕ :=
  wallHeight * wallLength - (windowHeight * windowLength + doorHeight * doorLength)

/-- Theorem stating that the area to be painted is 129 square feet. -/
theorem paint_area_is_129 :
  areaToBePainted 10 15 3 5 2 3 = 129 := by
  sorry

#eval areaToBePainted 10 15 3 5 2 3

end NUMINAMATH_CALUDE_paint_area_is_129_l3070_307079


namespace NUMINAMATH_CALUDE_rice_yield_80kg_l3070_307064

/-- Linear regression equation for rice yield prediction -/
def rice_yield_prediction (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem: The predicted rice yield for 80 kg of fertilizer is 650 kg -/
theorem rice_yield_80kg : rice_yield_prediction 80 = 650 := by
  sorry

end NUMINAMATH_CALUDE_rice_yield_80kg_l3070_307064


namespace NUMINAMATH_CALUDE_equation_solution_l3070_307000

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (5*x^2 + 70*x + 2) / (3*x + 28) - (4*x + 2)
  let sol1 : ℝ := (-48 + 28*Real.sqrt 22) / 14
  let sol2 : ℝ := (-48 - 28*Real.sqrt 22) / 14
  f sol1 = 0 ∧ f sol2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3070_307000


namespace NUMINAMATH_CALUDE_max_sum_is_fifty_l3070_307093

/-- A hexagonal prism with an added pyramid -/
structure HexagonalPrismWithPyramid where
  /-- Number of faces when pyramid is added to hexagonal face -/
  faces_hex : ℕ
  /-- Number of vertices when pyramid is added to hexagonal face -/
  vertices_hex : ℕ
  /-- Number of edges when pyramid is added to hexagonal face -/
  edges_hex : ℕ
  /-- Number of faces when pyramid is added to rectangular face -/
  faces_rect : ℕ
  /-- Number of vertices when pyramid is added to rectangular face -/
  vertices_rect : ℕ
  /-- Number of edges when pyramid is added to rectangular face -/
  edges_rect : ℕ

/-- The maximum sum of exterior faces, vertices, and edges -/
def max_sum (shape : HexagonalPrismWithPyramid) : ℕ :=
  max (shape.faces_hex + shape.vertices_hex + shape.edges_hex)
      (shape.faces_rect + shape.vertices_rect + shape.edges_rect)

/-- Theorem: The maximum sum of exterior faces, vertices, and edges is 50 -/
theorem max_sum_is_fifty (shape : HexagonalPrismWithPyramid) 
  (h1 : shape.faces_hex = 13)
  (h2 : shape.vertices_hex = 13)
  (h3 : shape.edges_hex = 24)
  (h4 : shape.faces_rect = 11)
  (h5 : shape.vertices_rect = 13)
  (h6 : shape.edges_rect = 22) :
  max_sum shape = 50 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_is_fifty_l3070_307093


namespace NUMINAMATH_CALUDE_problem_statement_l3070_307005

/-- Given two positive real numbers x and y satisfying the equations
     1/x + 1/y = 5 and x² + y² = 14, prove that x²y + xy² = 3.2 -/
theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : 1/x + 1/y = 5) (h2 : x^2 + y^2 = 14) :
  x^2 * y + x * y^2 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3070_307005


namespace NUMINAMATH_CALUDE_forest_growth_l3070_307015

/-- The number of trees in a forest follows a specific growth pattern --/
theorem forest_growth (trees : ℕ → ℕ) (k : ℚ) : 
  (∀ n, trees (n + 2) - trees n = k * trees (n + 1)) →
  trees 1993 = 50 →
  trees 1994 = 75 →
  trees 1996 = 140 →
  trees 1995 = 99 := by
sorry

end NUMINAMATH_CALUDE_forest_growth_l3070_307015


namespace NUMINAMATH_CALUDE_double_counted_integer_l3070_307095

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem double_counted_integer (n : ℕ) (x : ℕ) :
  sum_of_first_n n + x = 5053 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_integer_l3070_307095


namespace NUMINAMATH_CALUDE_paint_calculation_l3070_307042

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 : ℚ) * total_paint + (1/2 : ℚ) * ((3/4 : ℚ) * total_paint) = 225 → 
  total_paint = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l3070_307042


namespace NUMINAMATH_CALUDE_two_true_propositions_l3070_307040

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a predicate for a right angle
def is_right_angle (angle : Real) : Prop := angle = 90

-- Define a predicate for a right triangle
def is_right_triangle (t : Triangle) : Prop := ∃ angle, is_right_angle angle

-- Define the original proposition
def original_prop (t : Triangle) : Prop :=
  is_right_angle t.C → is_right_triangle t

-- Define the converse proposition
def converse_prop (t : Triangle) : Prop :=
  is_right_triangle t → is_right_angle t.C

-- Define the inverse proposition
def inverse_prop (t : Triangle) : Prop :=
  ¬(is_right_angle t.C) → ¬(is_right_triangle t)

-- Define the contrapositive proposition
def contrapositive_prop (t : Triangle) : Prop :=
  ¬(is_right_triangle t) → ¬(is_right_angle t.C)

-- Theorem stating that exactly two of these propositions are true
theorem two_true_propositions :
  ∃ (t : Triangle),
    (original_prop t ∧ contrapositive_prop t) ∧
    ¬(converse_prop t ∨ inverse_prop t) :=
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3070_307040


namespace NUMINAMATH_CALUDE_cubic_polynomial_unique_solution_l3070_307069

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a + b * x + c * x^2 + d * x^3

theorem cubic_polynomial_unique_solution 
  (P : ℝ → ℝ) 
  (h_cubic : cubic_polynomial P) 
  (h_neg_one : P (-1) = 2)
  (h_zero : P 0 = 3)
  (h_one : P 1 = 1)
  (h_two : P 2 = 15) :
  ∀ x, P x = 3 + x - 2 * x^2 - x^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_unique_solution_l3070_307069


namespace NUMINAMATH_CALUDE_eggs_in_boxes_l3070_307011

theorem eggs_in_boxes (eggs_per_box : ℕ) (num_boxes : ℕ) :
  eggs_per_box = 15 → num_boxes = 7 → eggs_per_box * num_boxes = 105 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_boxes_l3070_307011


namespace NUMINAMATH_CALUDE_flowers_left_in_peters_garden_l3070_307075

/-- The number of flowers in Amanda's garden -/
def amanda_flowers : ℕ := 20

/-- The number of flowers in Peter's garden before giving away -/
def peter_flowers : ℕ := 3 * amanda_flowers

/-- The number of flowers Peter gave away -/
def flowers_given_away : ℕ := 15

/-- Theorem: The number of flowers left in Peter's garden is 45 -/
theorem flowers_left_in_peters_garden :
  peter_flowers - flowers_given_away = 45 := by
  sorry

end NUMINAMATH_CALUDE_flowers_left_in_peters_garden_l3070_307075


namespace NUMINAMATH_CALUDE_simplify_expression_l3070_307062

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3070_307062


namespace NUMINAMATH_CALUDE_bulb_switch_problem_l3070_307028

theorem bulb_switch_problem :
  let n : Nat := 11
  let target_state : Fin n → Bool := fun i => i.val + 1 == n
  let valid_state (state : Fin n → Bool) :=
    ∃ (k : Nat), k < 2^n ∧ state = fun i => (k.digits 2).get? i.val == some 1
  { count : Nat // ∀ state, valid_state state ∧ state = target_state → count = 2^(n-1) } :=
by
  sorry

#check bulb_switch_problem

end NUMINAMATH_CALUDE_bulb_switch_problem_l3070_307028


namespace NUMINAMATH_CALUDE_pairball_play_time_l3070_307076

theorem pairball_play_time (total_duration : ℕ) (num_children : ℕ) (children_per_game : ℕ) :
  total_duration = 120 →
  num_children = 6 →
  children_per_game = 2 →
  (total_duration * children_per_game) / num_children = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_pairball_play_time_l3070_307076


namespace NUMINAMATH_CALUDE_chess_swimming_enrollment_percentage_l3070_307078

theorem chess_swimming_enrollment_percentage 
  (total_students : ℕ) 
  (chess_percentage : ℚ) 
  (swimming_students : ℕ) 
  (h1 : total_students = 2000)
  (h2 : chess_percentage = 1/10)
  (h3 : swimming_students = 100) :
  (swimming_students : ℚ) / ((chess_percentage * total_students) : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chess_swimming_enrollment_percentage_l3070_307078


namespace NUMINAMATH_CALUDE_total_trip_cost_l3070_307018

def rental_cost : ℝ := 150
def gas_price : ℝ := 3.50
def gas_purchased : ℝ := 8
def mileage_cost : ℝ := 0.50
def distance_driven : ℝ := 320

theorem total_trip_cost : 
  rental_cost + gas_price * gas_purchased + mileage_cost * distance_driven = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_trip_cost_l3070_307018


namespace NUMINAMATH_CALUDE_division_result_l3070_307085

theorem division_result : (0.18 : ℚ) / (0.003 : ℚ) = 60 := by sorry

end NUMINAMATH_CALUDE_division_result_l3070_307085


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3070_307013

theorem solve_exponential_equation :
  ∃ x : ℝ, (125 : ℝ) = 5 * (25 : ℝ)^(x - 2) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3070_307013


namespace NUMINAMATH_CALUDE_multiple_sum_properties_l3070_307092

theorem multiple_sum_properties (x y : ℤ) 
  (hx : ∃ (m : ℤ), x = 6 * m) 
  (hy : ∃ (n : ℤ), y = 12 * n) : 
  (∃ (k : ℤ), x + y = 2 * k) ∧ (∃ (l : ℤ), x + y = 6 * l) := by
  sorry

end NUMINAMATH_CALUDE_multiple_sum_properties_l3070_307092


namespace NUMINAMATH_CALUDE_total_animals_is_100_l3070_307096

/-- Given the number of rabbits, calculates the total number of chickens, ducks, and rabbits. -/
def total_animals (num_rabbits : ℕ) : ℕ :=
  let num_ducks := num_rabbits + 12
  let num_chickens := 5 * num_ducks
  num_chickens + num_ducks + num_rabbits

/-- Theorem stating that given 4 rabbits, the total number of animals is 100. -/
theorem total_animals_is_100 : total_animals 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_is_100_l3070_307096


namespace NUMINAMATH_CALUDE_vegetable_baskets_weight_l3070_307098

def num_baskets : ℕ := 5
def standard_weight : ℕ := 50
def excess_deficiency : List ℤ := [3, -6, -4, 2, -1]

theorem vegetable_baskets_weight :
  (List.sum excess_deficiency = -6) ∧
  (num_baskets * standard_weight + List.sum excess_deficiency = 244) := by
sorry

end NUMINAMATH_CALUDE_vegetable_baskets_weight_l3070_307098


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3070_307008

theorem shaded_area_calculation (S T : ℝ) : 
  (16 / S = 4) → 
  (S / T = 4) → 
  (S^2 + 16 * T^2 = 32) := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3070_307008


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3070_307020

def f (x : ℝ) : ℝ := x^2 - x

theorem at_least_one_non_negative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  f m ≥ 0 ∨ f n ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3070_307020


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3070_307046

theorem roots_of_quadratic (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 12) :
  x^2 - 10*x - 11 = 0 ∧ y^2 - 10*y - 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3070_307046


namespace NUMINAMATH_CALUDE_probability_two_red_value_l3070_307012

/-- A deck of cards with 5 suits -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (h_total : total_cards = 65)
  (h_red : red_cards = 39)

/-- The probability of drawing two red cards from the deck -/
def probability_two_red (d : Deck) : ℚ :=
  (d.red_cards.choose 2) / (d.total_cards.choose 2)

/-- Theorem stating the probability of drawing two red cards -/
theorem probability_two_red_value (d : Deck) : 
  probability_two_red d = 741 / 2080 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_value_l3070_307012


namespace NUMINAMATH_CALUDE_cube_sum_geq_sqrt_product_square_sum_l3070_307054

theorem cube_sum_geq_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_sqrt_product_square_sum_l3070_307054


namespace NUMINAMATH_CALUDE_free_throw_contest_l3070_307057

theorem free_throw_contest (x : ℕ) : 
  x + 3*x + 6*x = 80 → x = 8 := by sorry

end NUMINAMATH_CALUDE_free_throw_contest_l3070_307057


namespace NUMINAMATH_CALUDE_part_to_third_ratio_l3070_307065

theorem part_to_third_ratio (N P : ℝ) (h1 : (1/4) * (1/3) * P = 14) (h2 : 0.40 * N = 168) :
  P / ((1/3) * N) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_third_ratio_l3070_307065


namespace NUMINAMATH_CALUDE_cookies_calculation_l3070_307019

/-- The number of people Brenda's mother made cookies for -/
def num_people : ℕ := 14

/-- The number of cookies each person had -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 420 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l3070_307019


namespace NUMINAMATH_CALUDE_total_points_l3070_307094

def game1_mike : ℕ := 5
def game1_john : ℕ := game1_mike + 2

def game2_mike : ℕ := 7
def game2_john : ℕ := game2_mike - 3

def game3_mike : ℕ := 10
def game3_john : ℕ := game3_mike / 2

def game4_mike : ℕ := 12
def game4_john : ℕ := game4_mike * 2

def game5_mike : ℕ := 6
def game5_john : ℕ := game5_mike

def game6_john : ℕ := 8
def game6_mike : ℕ := game6_john + 4

def mike_total : ℕ := game1_mike + game2_mike + game3_mike + game4_mike + game5_mike + game6_mike
def john_total : ℕ := game1_john + game2_john + game3_john + game4_john + game5_john + game6_john

theorem total_points : mike_total + john_total = 106 := by
  sorry

end NUMINAMATH_CALUDE_total_points_l3070_307094


namespace NUMINAMATH_CALUDE_prob_both_selected_l3070_307067

theorem prob_both_selected (prob_x prob_y prob_both : ℚ) : 
  prob_x = 1/5 → prob_y = 2/7 → prob_both = prob_x * prob_y → prob_both = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l3070_307067


namespace NUMINAMATH_CALUDE_legos_set_cost_legos_set_cost_is_30_l3070_307035

/-- The cost of the Legos set given the total earnings and the price of little cars -/
theorem legos_set_cost (total_earnings : ℕ) (little_car_price : ℕ) (num_cars : ℕ) : ℕ :=
  total_earnings - (num_cars * little_car_price)

/-- Proof that the Legos set costs $30 -/
theorem legos_set_cost_is_30 :
  legos_set_cost 45 5 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_legos_set_cost_legos_set_cost_is_30_l3070_307035


namespace NUMINAMATH_CALUDE_average_difference_due_to_input_error_l3070_307022

theorem average_difference_due_to_input_error :
  ∀ (data_points : ℕ) (incorrect_value : ℝ) (correct_value : ℝ),
    data_points = 30 →
    incorrect_value = 105 →
    correct_value = 15 →
    (incorrect_value - correct_value) / data_points = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_difference_due_to_input_error_l3070_307022


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3070_307049

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3070_307049


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l3070_307048

theorem pizza_slices_per_person 
  (total_slices : Nat) 
  (people : Nat) 
  (slices_left : Nat) 
  (h1 : total_slices = 16) 
  (h2 : people = 6) 
  (h3 : slices_left = 4) 
  (h4 : people > 0) : 
  (total_slices - slices_left) / people = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l3070_307048
