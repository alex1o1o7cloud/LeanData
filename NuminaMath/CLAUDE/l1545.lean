import Mathlib

namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1545_154505

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l1545_154505


namespace NUMINAMATH_CALUDE_f_shifted_passes_through_point_one_zero_l1545_154585

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define the shifted function f(x-1)
def f_shifted (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

-- Theorem statement
theorem f_shifted_passes_through_point_one_zero (a : ℝ) :
  f_shifted a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_passes_through_point_one_zero_l1545_154585


namespace NUMINAMATH_CALUDE_evaluate_expression_l1545_154515

theorem evaluate_expression : (20 ^ 40) / (80 ^ 10) = 5 ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1545_154515


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l1545_154574

theorem sum_reciprocal_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l1545_154574


namespace NUMINAMATH_CALUDE_license_plate_count_l1545_154588

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of vowels in the English alphabet (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits

theorem license_plate_count : total_plates = 24000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1545_154588


namespace NUMINAMATH_CALUDE_vector_collinearity_l1545_154580

theorem vector_collinearity (m : ℝ) : 
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1545_154580


namespace NUMINAMATH_CALUDE_zigzag_angle_in_rectangle_l1545_154562

theorem zigzag_angle_in_rectangle (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 = 10)
  (h2 : angle2 = 14)
  (h3 : angle3 = 26)
  (h4 : angle4 = 33) :
  ∃ θ : ℝ, θ = 11 ∧ 
  (90 - angle1) + (90 - angle3) + θ = 180 ∧
  (180 - (90 - angle1) - angle2) + (180 - (90 - angle3) - angle4) + θ = 180 :=
by sorry

end NUMINAMATH_CALUDE_zigzag_angle_in_rectangle_l1545_154562


namespace NUMINAMATH_CALUDE_function_eventually_constant_l1545_154533

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem function_eventually_constant
  (f : ℕ+ → ℕ+)
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f := by
sorry

end NUMINAMATH_CALUDE_function_eventually_constant_l1545_154533


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l1545_154535

/-- 
Given an arithmetic progression with three consecutive terms a, a+d, a+2d,
this theorem states the conditions for d such that the squares of these terms
form a geometric progression.
-/
theorem arithmetic_to_geometric_progression (a d : ℝ) :
  (∃ r : ℝ, (a + d)^2 = a^2 * r ∧ (a + 2*d)^2 = (a + d)^2 * r) ↔ 
  (d = 0 ∨ d = a*(-2 + Real.sqrt 2) ∨ d = a*(-2 - Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l1545_154535


namespace NUMINAMATH_CALUDE_sine_monotonicity_implies_omega_range_l1545_154519

open Real

theorem sine_monotonicity_implies_omega_range 
  (f : ℝ → ℝ) (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π/2) π, 
    ∀ y ∈ Set.Ioo (π/2) π, 
    x < y → f x < f y) →
  (∀ x, f x = 2 * sin (ω * x + π/6)) →
  0 < ω ∧ ω ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sine_monotonicity_implies_omega_range_l1545_154519


namespace NUMINAMATH_CALUDE_irrationality_of_lambda_l1545_154546

theorem irrationality_of_lambda (n : ℕ) : 
  Irrational (Real.sqrt (3 * n^2 + 2 * n + 2)) := by sorry

end NUMINAMATH_CALUDE_irrationality_of_lambda_l1545_154546


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l1545_154534

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

/-- The number of cookies put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 188 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l1545_154534


namespace NUMINAMATH_CALUDE_acute_angles_are_first_quadrant_l1545_154500

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define first quadrant angle
def is_first_quadrant_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem: All acute angles are first quadrant angles
theorem acute_angles_are_first_quadrant :
  ∀ θ : ℝ, is_acute_angle θ → is_first_quadrant_angle θ :=
by
  sorry


end NUMINAMATH_CALUDE_acute_angles_are_first_quadrant_l1545_154500


namespace NUMINAMATH_CALUDE_fred_red_marbles_l1545_154564

/-- Fred's marble collection --/
structure MarbleCollection where
  total : ℕ
  darkBlue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Theorem: Fred has 60 red marbles --/
theorem fred_red_marbles (m : MarbleCollection) : m.red = 60 :=
  by
  have h1 : m.total = 120 := by sorry
  have h2 : m.darkBlue = m.total / 4 := by sorry
  have h3 : m.red = 2 * m.darkBlue := by sorry
  have h4 : m.green = 10 := by sorry
  have h5 : m.yellow = 5 := by sorry
  
  -- Proof
  sorry


end NUMINAMATH_CALUDE_fred_red_marbles_l1545_154564


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1545_154518

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1545_154518


namespace NUMINAMATH_CALUDE_valid_routes_count_l1545_154572

-- Define the cities
inductive City : Type
| P | Q | R | S | T | U

-- Define the roads
inductive Road : Type
| PQ | PS | PT | PU | QR | QS | RS | RT | SU | UT

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid
def isValidRoute (r : Route) : Bool :=
  -- Implementation details omitted
  sorry

-- Function to count valid routes
def countValidRoutes : Nat :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem valid_routes_count :
  countValidRoutes = 15 :=
sorry

end NUMINAMATH_CALUDE_valid_routes_count_l1545_154572


namespace NUMINAMATH_CALUDE_dave_had_18_tickets_l1545_154541

/-- Calculates the number of tickets Dave had left after playing games and receiving tickets from a friend -/
def daves_tickets : ℕ :=
  let first_set := 14 - 2
  let second_set := 8 - 5
  let third_set := (first_set * 3) - 15
  let total_after_games := first_set + second_set + third_set
  let after_buying_toys := total_after_games - 25
  after_buying_toys + 7

/-- Theorem stating that Dave had 18 tickets left -/
theorem dave_had_18_tickets : daves_tickets = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_had_18_tickets_l1545_154541


namespace NUMINAMATH_CALUDE_debby_vacation_pictures_l1545_154573

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := 12

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The total number of pictures Debby took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Debby still has from her vacation -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem debby_vacation_pictures : remaining_pictures = 22 := by
  sorry

end NUMINAMATH_CALUDE_debby_vacation_pictures_l1545_154573


namespace NUMINAMATH_CALUDE_triangle_area_l1545_154576

/-- The area of a triangle composed of two right-angled triangles -/
theorem triangle_area (base1 height1 base2 height2 : ℝ) 
  (h1 : base1 = 1) (h2 : height1 = 1) 
  (h3 : base2 = 2) (h4 : height2 = 1) : 
  (1/2 * base1 * height1) + (1/2 * base2 * height2) = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1545_154576


namespace NUMINAMATH_CALUDE_a7_value_in_arithmetic_sequence_l1545_154557

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a7_value_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_a7_value_in_arithmetic_sequence_l1545_154557


namespace NUMINAMATH_CALUDE_rv_parking_probability_l1545_154579

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars that have already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required for the RV -/
def required_adjacent_spaces : ℕ := 3

/-- The probability of being able to park the RV -/
def parking_probability : ℚ := 232 / 323

theorem rv_parking_probability :
  let empty_spaces := total_spaces - parked_cars
  let total_arrangements := Nat.choose total_spaces parked_cars
  let valid_arrangements := total_arrangements - Nat.choose (empty_spaces + parked_cars - required_adjacent_spaces + 1) empty_spaces
  (valid_arrangements : ℚ) / total_arrangements = parking_probability := by
  sorry

end NUMINAMATH_CALUDE_rv_parking_probability_l1545_154579


namespace NUMINAMATH_CALUDE_smallest_integer_2011m_55555n_l1545_154538

theorem smallest_integer_2011m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 2011*m + 55555*n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 2011*m + 55555*n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_2011m_55555n_l1545_154538


namespace NUMINAMATH_CALUDE_brick_surface_area_l1545_154544

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1545_154544


namespace NUMINAMATH_CALUDE_expression_evaluation_l1545_154581

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := 1
  2 * (x - 2*y)^2 - (2*y + x) * (-2*y + x) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1545_154581


namespace NUMINAMATH_CALUDE_two_removable_cells_exist_l1545_154525

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Bool

-- Define a cell position
structure CellPosition where
  row : Fin 4
  col : Fin 4

-- Define a function to remove a cell from the grid
def removeCell (g : Grid) (pos : CellPosition) : Grid :=
  fun r c => if r = pos.row ∧ c = pos.col then false else g r c

-- Define congruence between two parts of the grid
def isCongruent (part1 part2 : Set CellPosition) : Prop := sorry

-- Define a function to check if a grid can be divided into three congruent parts
def canDivideIntoThreeCongruentParts (g : Grid) : Prop := sorry

-- Theorem statement
theorem two_removable_cells_exist :
  ∃ (pos1 pos2 : CellPosition),
    pos1 ≠ pos2 ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos1) ∧
    canDivideIntoThreeCongruentParts (removeCell (fun _ _ => true) pos2) := by
  sorry


end NUMINAMATH_CALUDE_two_removable_cells_exist_l1545_154525


namespace NUMINAMATH_CALUDE_inequality_proof_l1545_154509

theorem inequality_proof (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) : 
  (Real.sqrt x + Real.sqrt y) * (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y)) > 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1545_154509


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1545_154565

theorem simplest_quadratic_radical : 
  let options : List ℝ := [Real.sqrt (1/2), Real.sqrt 8, Real.sqrt 15, Real.sqrt 20]
  ∀ x ∈ options, x ≠ Real.sqrt 15 → 
    ∃ y z : ℕ, (y > 1 ∧ z > 1 ∧ x = Real.sqrt y * z) ∨ 
              (y > 1 ∧ z > 1 ∧ x = (Real.sqrt y) / z) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1545_154565


namespace NUMINAMATH_CALUDE_min_value_theorem_l1545_154516

theorem min_value_theorem (a b c : ℝ) 
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) : 
  ∃ m : ℝ, m = a + 2*b - 3*c ∧ ∀ a' b' c' : ℝ, 
    (∀ x y : ℝ, x + 2*y - 3 ≤ a'*x + b'*y + c' ∧ a'*x + b'*y + c' ≤ x + 2*y + 3) →
    m ≤ a' + 2*b' - 3*c' :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1545_154516


namespace NUMINAMATH_CALUDE_pen_cost_l1545_154595

/-- The cost of a pen given Elizabeth's budget and purchasing constraints -/
theorem pen_cost (total_budget : ℝ) (pencil_cost : ℝ) (pencil_count : ℕ) (pen_count : ℕ) :
  total_budget = 20 →
  pencil_cost = 1.6 →
  pencil_count = 5 →
  pen_count = 6 →
  (pencil_count * pencil_cost + pen_count * ((total_budget - pencil_count * pencil_cost) / pen_count) = total_budget) →
  (total_budget - pencil_count * pencil_cost) / pen_count = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l1545_154595


namespace NUMINAMATH_CALUDE_mixture_ratio_l1545_154528

def mixture (initial_water : ℝ) : Prop :=
  let initial_alcohol : ℝ := 10
  let added_water : ℝ := 10
  let new_ratio_alcohol : ℝ := 2
  let new_ratio_water : ℝ := 7
  (initial_alcohol / (initial_water + added_water) = new_ratio_alcohol / new_ratio_water) ∧
  (initial_alcohol / initial_water = 2 / 5)

theorem mixture_ratio : ∃ (initial_water : ℝ), mixture initial_water :=
  sorry

end NUMINAMATH_CALUDE_mixture_ratio_l1545_154528


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1545_154532

/-- A number is divisible by 8 if and only if its last three digits form a number divisible by 8 -/
axiom divisible_by_8 (n : ℕ) : n % 8 = 0 ↔ (n % 1000) % 8 = 0

/-- The largest five-digit number -/
def largest_five_digit : ℕ := 99999

/-- The largest five-digit number divisible by 8 -/
def largest_five_digit_div_8 : ℕ := 99992

theorem largest_five_digit_divisible_by_8 :
  largest_five_digit_div_8 % 8 = 0 ∧
  ∀ n : ℕ, n > largest_five_digit_div_8 → n ≤ largest_five_digit → n % 8 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1545_154532


namespace NUMINAMATH_CALUDE_expression_simplification_l1545_154554

theorem expression_simplification (x : ℝ) : 
  (((x+1)^3*(x^2-x+1)^3)/(x^3+1)^3)^3 * (((x-1)^3*(x^2+x+1)^3)/(x^3-1)^3)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1545_154554


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1545_154590

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2) 
  (h_x_pos : x > 0) 
  (h_cos : Real.cos (θ / 3) = Real.sqrt ((x + 2) / (3 * x))) : 
  Real.tan θ = 
    (Real.sqrt (1 - ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) ^ 2)) / 
    ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1545_154590


namespace NUMINAMATH_CALUDE_remainder_equality_l1545_154514

theorem remainder_equality (Q Q' E S S' s s' : ℕ) 
  (hQ : Q > Q') 
  (hS : S = Q % E) 
  (hS' : S' = Q' % E) 
  (hs : s = (Q^2 * Q') % E) 
  (hs' : s' = (S^2 * S') % E) : 
  s = s' := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l1545_154514


namespace NUMINAMATH_CALUDE_larger_number_problem_l1545_154597

theorem larger_number_problem (x y : ℝ) 
  (h1 : 5 * y = 7 * x) 
  (h2 : y - x = 10) : 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1545_154597


namespace NUMINAMATH_CALUDE_hannah_son_cutting_rate_l1545_154540

/-- The number of strands Hannah's son can cut per minute -/
def sonCuttingRate (totalStrands : ℕ) (hannahRate : ℕ) (totalTime : ℕ) : ℕ :=
  (totalStrands - hannahRate * totalTime) / totalTime

theorem hannah_son_cutting_rate :
  sonCuttingRate 22 8 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_son_cutting_rate_l1545_154540


namespace NUMINAMATH_CALUDE_austin_weeks_to_buy_bicycle_l1545_154531

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Theorem: Austin needs 6 weeks to buy the bicycle -/
theorem austin_weeks_to_buy_bicycle :
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_austin_weeks_to_buy_bicycle_l1545_154531


namespace NUMINAMATH_CALUDE_jacks_age_problem_l1545_154566

theorem jacks_age_problem (jack_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jack_age_2010 = 12 →
  mother_age_multiplier = 3 →
  ∃ (years_after_2010 : ℕ), 
    (jack_age_2010 + years_after_2010) * 2 = (jack_age_2010 * mother_age_multiplier + years_after_2010) ∧
    years_after_2010 = 12 :=
by sorry

end NUMINAMATH_CALUDE_jacks_age_problem_l1545_154566


namespace NUMINAMATH_CALUDE_percentage_increase_l1545_154536

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 800 → final = 1680 → 
  ((final - initial) / initial) * 100 = 110 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l1545_154536


namespace NUMINAMATH_CALUDE_sum_of_integers_l1545_154559

theorem sum_of_integers (w x y z : ℤ) 
  (eq1 : w - x + y = 7)
  (eq2 : x - y + z = 8)
  (eq3 : y - z + w = 4)
  (eq4 : z - w + x = 3) :
  w + x + y + z = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1545_154559


namespace NUMINAMATH_CALUDE_grid_product_theorem_l1545_154587

theorem grid_product_theorem : ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 120 ∧
   d * e * f = 120 ∧
   g * h * i = 120 ∧
   a * d * g = 120 ∧
   b * e * h = 120 ∧
   c * f * i = 120) ∧
  (∀ (p : ℕ), (∃ (x y z u v w : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧ x ≠ w ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧ y ≠ w ∧
    z ≠ u ∧ z ≠ v ∧ z ≠ w ∧
    u ≠ v ∧ u ≠ w ∧
    v ≠ w ∧
    x * y * z = p ∧ u * v * w = p) → p ≥ 120) :=
by sorry

end NUMINAMATH_CALUDE_grid_product_theorem_l1545_154587


namespace NUMINAMATH_CALUDE_second_number_value_l1545_154583

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a → 
  b = 60 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1545_154583


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l1545_154527

/-- If a and b are opposite numbers, then a + b + 3 = 3 -/
theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → a + b + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l1545_154527


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1545_154504

/-- Represents the points on the circle --/
inductive Point
  | one
  | two
  | three
  | four
  | five
  | six

/-- Determines if a point is odd-numbered --/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one | Point.three | Point.five => true
  | _ => false

/-- Calculates the next point after a jump --/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.one
  | Point.six => Point.one

/-- Calculates the point after n jumps --/
def jumpN (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN Point.six 2023 = Point.one := by
  sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1545_154504


namespace NUMINAMATH_CALUDE_distance_from_point_on_number_line_l1545_154545

theorem distance_from_point_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_distance_from_point_on_number_line_l1545_154545


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1545_154501

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ 
  (k > 7 - 4 * Real.sqrt 2 ∧ k < 7 + 4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1545_154501


namespace NUMINAMATH_CALUDE_number_puzzle_l1545_154558

theorem number_puzzle : ∃! x : ℤ, x - 2 + 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1545_154558


namespace NUMINAMATH_CALUDE_problem_1_l1545_154513

theorem problem_1 : Real.sqrt 3 ^ 2 + |-(Real.sqrt 3 / 3)| - (π - Real.sqrt 2) ^ 0 - Real.tan (π / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1545_154513


namespace NUMINAMATH_CALUDE_gcd_of_256_450_720_l1545_154594

theorem gcd_of_256_450_720 : Nat.gcd 256 (Nat.gcd 450 720) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_450_720_l1545_154594


namespace NUMINAMATH_CALUDE_binomial_coefficient_property_l1545_154584

theorem binomial_coefficient_property :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_property_l1545_154584


namespace NUMINAMATH_CALUDE_disk_rotation_on_clock_face_l1545_154550

theorem disk_rotation_on_clock_face (clock_radius disk_radius : ℝ) 
  (h1 : clock_radius = 30)
  (h2 : disk_radius = 15)
  (h3 : disk_radius = clock_radius / 2) :
  let initial_position := 0 -- 12 o'clock
  let final_position := π -- 6 o'clock (π radians)
  ∃ (θ : ℝ), 
    θ * disk_radius = final_position * clock_radius ∧ 
    θ % (2 * π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_disk_rotation_on_clock_face_l1545_154550


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1545_154568

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ 
  (c^2 - 6*c + 8 = 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  a + b + c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1545_154568


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1545_154547

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original circle -/
def original_circle : Circle :=
  { center := (-2, -1), radius := 2 }

/-- The symmetric circle with respect to x-axis -/
def symmetric_circle : Circle :=
  { center := reflect_x original_circle.center, radius := original_circle.radius }

/-- Equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem symmetric_circle_equation :
  ∀ x y : ℝ, circle_equation symmetric_circle x y ↔ (x + 2)^2 + (y - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1545_154547


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l1545_154524

/-- The cost of a first-class postage stamp in pence -/
def first_class_cost : ℕ := 85

/-- The cost of a second-class postage stamp in pence -/
def second_class_cost : ℕ := 66

/-- The number of pence in a pound -/
def pence_per_pound : ℕ := 100

/-- The proposition that (r, s) is a valid solution to the stamp problem -/
def is_valid_solution (r s : ℕ) : Prop :=
  r ≥ 1 ∧ s ≥ 1 ∧ ∃ t : ℕ, t > 0 ∧ first_class_cost * r + second_class_cost * s = pence_per_pound * t

/-- The proposition that (r, s) is the optimal solution to the stamp problem -/
def is_optimal_solution (r s : ℕ) : Prop :=
  is_valid_solution r s ∧ ∀ r' s' : ℕ, is_valid_solution r' s' → r + s ≤ r' + s'

/-- The theorem stating the optimal solution to the stamp problem -/
theorem stamp_problem_solution :
  is_optimal_solution 2 5 ∧ 2 + 5 = 7 := by sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l1545_154524


namespace NUMINAMATH_CALUDE_sqrt3_minus_1_power_equation_solution_is_16_l1545_154526

theorem sqrt3_minus_1_power_equation : ∃ (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 :=
by sorry

theorem solution_is_16 : ∃! (N : ℕ), (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 ∧ N = 16 :=
by sorry

end NUMINAMATH_CALUDE_sqrt3_minus_1_power_equation_solution_is_16_l1545_154526


namespace NUMINAMATH_CALUDE_total_books_eq_sum_l1545_154520

/-- The number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def total_movies : ℕ := 10

/-- The number of books you have read -/
def books_read : ℕ := 12

/-- The number of movies you have watched -/
def movies_watched : ℕ := 56

/-- The number of books you still have to read -/
def books_to_read : ℕ := 10

/-- Theorem: The total number of books in the series is equal to the sum of books read and books yet to read -/
theorem total_books_eq_sum : total_books = books_read + books_to_read := by sorry

end NUMINAMATH_CALUDE_total_books_eq_sum_l1545_154520


namespace NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l1545_154569

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l1545_154569


namespace NUMINAMATH_CALUDE_island_puzzle_l1545_154543

-- Define the types of people
inductive PersonType
| Truthful
| Liar

-- Define the genders
inductive Gender
| Boy
| Girl

-- Define a person
structure Person where
  type : PersonType
  gender : Gender

-- Define the statements made by A and B
def statement_A (a b : Person) : Prop :=
  a.type = PersonType.Truthful → b.type = PersonType.Liar

def statement_B (a b : Person) : Prop :=
  b.gender = Gender.Boy → a.gender = Gender.Girl

-- Theorem to prove
theorem island_puzzle :
  ∃ (a b : Person),
    (statement_A a b ↔ a.type = PersonType.Truthful) ∧
    (statement_B a b ↔ b.type = PersonType.Liar) ∧
    a.type = PersonType.Truthful ∧
    a.gender = Gender.Boy ∧
    b.type = PersonType.Liar ∧
    b.gender = Gender.Boy :=
  sorry

end NUMINAMATH_CALUDE_island_puzzle_l1545_154543


namespace NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l1545_154539

theorem rectangle_cannot_fit_in_square :
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 →
  rect_length * rect_width = 90 →
  rect_length / rect_width = 5 / 3 →
  square_side * square_side = 100 →
  rect_length > square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cannot_fit_in_square_l1545_154539


namespace NUMINAMATH_CALUDE_perpendicular_lines_plane_theorem_l1545_154529

/-- Represents a plane in 3D space -/
structure Plane :=
  (α : Type*)

/-- Represents a line in 3D space -/
structure Line :=
  (l : Type*)

/-- Indicates that a line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line) : Prop :=
  sorry

/-- Indicates that a line is in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop :=
  sorry

/-- Indicates that a line is outside a plane -/
def line_outside_plane (l : Line) (α : Plane) : Prop :=
  sorry

theorem perpendicular_lines_plane_theorem 
  (α : Plane) (a b l : Line) 
  (h1 : a ≠ b)
  (h2 : line_in_plane a α)
  (h3 : line_in_plane b α)
  (h4 : line_outside_plane l α) :
  (∀ (α : Plane) (l : Line), perpendicular_to_plane l α → 
    perpendicular_to_line l a ∧ perpendicular_to_line l b) ∧
  (∃ (α : Plane) (a b l : Line), 
    perpendicular_to_line l a ∧ perpendicular_to_line l b ∧
    ¬perpendicular_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_plane_theorem_l1545_154529


namespace NUMINAMATH_CALUDE_blue_or_green_probability_l1545_154567

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  all_faces : sides = red + yellow + blue + green

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

theorem blue_or_green_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.red = 5 ∧ d.yellow = 3 ∧ d.blue = 1 ∧ d.green = 1) :
    probability (d.blue + d.green) d.sides = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_blue_or_green_probability_l1545_154567


namespace NUMINAMATH_CALUDE_soybean_price_l1545_154571

/-- Proves that the price of soybean is 20.5 given the conditions of the mixture problem -/
theorem soybean_price (peas_price : ℝ) (mixture_price : ℝ) (ratio : ℝ) :
  peas_price = 16 →
  ratio = 2 →
  mixture_price = 19 →
  (peas_price + ratio * (20.5 : ℝ)) / (1 + ratio) = mixture_price := by
sorry

end NUMINAMATH_CALUDE_soybean_price_l1545_154571


namespace NUMINAMATH_CALUDE_theater_seat_count_l1545_154549

/-- Represents the number of seats in a theater with a specific seating arrangement. -/
def theaterSeats (firstRowSeats : ℕ) (lastRowSeats : ℕ) : ℕ :=
  let additionalRows := (lastRowSeats - firstRowSeats) / 2
  let totalRows := additionalRows + 1
  let sumAdditionalSeats := additionalRows * (2 + (lastRowSeats - firstRowSeats)) / 2
  firstRowSeats * totalRows + sumAdditionalSeats

/-- Theorem stating that a theater with the given seating arrangement has 3434 seats. -/
theorem theater_seat_count :
  theaterSeats 12 128 = 3434 :=
by sorry

end NUMINAMATH_CALUDE_theater_seat_count_l1545_154549


namespace NUMINAMATH_CALUDE_yuko_wins_l1545_154517

theorem yuko_wins (yuri_total yuko_known x y : ℕ) : 
  yuri_total = 17 → yuko_known = 6 → yuko_known + x + y > yuri_total → x + y > 11 := by
  sorry

end NUMINAMATH_CALUDE_yuko_wins_l1545_154517


namespace NUMINAMATH_CALUDE_series_numerator_divisibility_l1545_154537

theorem series_numerator_divisibility (n : ℕ+) (h : Nat.Prime (3 * n + 1)) :
  ∃ k : ℤ, 2 * n - 1 = k * (3 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_series_numerator_divisibility_l1545_154537


namespace NUMINAMATH_CALUDE_factorization_proof_l1545_154502

theorem factorization_proof (x : ℝ) : 5*x*(x-5) + 7*(x-5) = (x-5)*(5*x+7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1545_154502


namespace NUMINAMATH_CALUDE_expression_factorization_l1545_154508

theorem expression_factorization (a b c : ℝ) :
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l1545_154508


namespace NUMINAMATH_CALUDE_fraction_sum_l1545_154523

theorem fraction_sum (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1545_154523


namespace NUMINAMATH_CALUDE_hotel_arrangement_l1545_154563

/-- The number of ways to distribute n distinct objects into k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition n distinct objects into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem hotel_arrangement :
  distribute 5 3 = 150 :=
by
  -- Define distribute in terms of stirling2 and factorial
  have h1 : ∀ n k, distribute n k = stirling2 n k * Nat.factorial k
  sorry
  
  -- Use the specific values for our problem
  have h2 : stirling2 5 3 = 25
  sorry
  
  -- Apply the definitions and properties
  rw [h1]
  simp [h2]
  -- The proof is completed by computation
  sorry

end NUMINAMATH_CALUDE_hotel_arrangement_l1545_154563


namespace NUMINAMATH_CALUDE_ellipse_and_slopes_l1545_154551

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the foci F1 and F2 -/
def foci (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  F1.1 < 0 ∧ F2.1 > 0 ∧ F1.2 = 0 ∧ F2.2 = 0 ∧ F1.1^2 = F2.1^2 ∧ F2.1^2 = a^2 - b^2

/-- Definition of the circles intersecting on C -/
def circles_intersect_on_C (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, ellipse_C P.1 P.2 a b ∧ 
    (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 9 ∧
    (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 1

/-- Definition of point A -/
def point_A (a b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of angle F1AF2 -/
def angle_F1AF2 (F1 F2 : ℝ × ℝ) (a b : ℝ) : Prop :=
  let A := point_A a b
  Real.cos (2 * Real.pi / 3) = 
    ((F1.1 - A.1) * (F2.1 - A.1) + (F1.2 - A.2) * (F2.2 - A.2)) /
    (Real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2) * Real.sqrt ((F2.1 - A.1)^2 + (F2.2 - A.2)^2))

/-- Definition of line l -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y + 1 = k * (x - 2)

/-- Definition of points M and N -/
def points_M_N (a b k : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, ellipse_C M.1 M.2 a b ∧ ellipse_C N.1 N.2 a b ∧
    line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧ M ≠ N

/-- Main theorem -/
theorem ellipse_and_slopes (a b : ℝ) (F1 F2 : ℝ × ℝ) (k : ℝ) :
  ellipse_C 0 b a b →
  foci F1 F2 a b →
  circles_intersect_on_C F1 F2 a b →
  angle_F1AF2 F1 F2 a b →
  points_M_N a b k →
  (a = 2 ∧ b = 1) ∧
  (∃ k1 k2 : ℝ, k1 + k2 = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_slopes_l1545_154551


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1545_154598

theorem sum_of_specific_numbers : 
  22000000 + 22000 + 2200 + 22 = 22024222 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1545_154598


namespace NUMINAMATH_CALUDE_steps_to_rockefeller_center_l1545_154591

theorem steps_to_rockefeller_center 
  (total_steps : ℕ) 
  (steps_to_times_square : ℕ) 
  (h1 : total_steps = 582) 
  (h2 : steps_to_times_square = 228) : 
  total_steps - steps_to_times_square = 354 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_rockefeller_center_l1545_154591


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_l1545_154560

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (∁A) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_l1545_154560


namespace NUMINAMATH_CALUDE_machine_problem_solution_l1545_154506

-- Define the equation
def machine_equation (y : ℝ) : Prop :=
  1 / (y + 4) + 1 / (y + 3) + 1 / (4 * y) = 1 / y

-- Theorem statement
theorem machine_problem_solution :
  ∃ y : ℝ, y > 0 ∧ machine_equation y ∧ y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_problem_solution_l1545_154506


namespace NUMINAMATH_CALUDE_apartments_per_floor_l1545_154599

theorem apartments_per_floor 
  (stories : ℕ) 
  (people_per_apartment : ℕ) 
  (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : people_per_apartment = 2)
  (h3 : total_people = 200) :
  (total_people / (stories * people_per_apartment) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_apartments_per_floor_l1545_154599


namespace NUMINAMATH_CALUDE_daria_credit_card_debt_l1545_154575

def couch_price : ℝ := 800
def couch_discount : ℝ := 0.10
def table_price : ℝ := 120
def table_discount : ℝ := 0.05
def lamp_price : ℝ := 50
def rug_price : ℝ := 250
def rug_discount : ℝ := 0.20
def bookshelf_price : ℝ := 180
def bookshelf_discount : ℝ := 0.15
def artwork_price : ℝ := 100
def artwork_discount : ℝ := 0.25
def savings : ℝ := 500

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price couch_price couch_discount +
  discounted_price table_price table_discount +
  lamp_price +
  discounted_price rug_price rug_discount +
  discounted_price bookshelf_price bookshelf_discount +
  discounted_price artwork_price artwork_discount

theorem daria_credit_card_debt :
  total_cost - savings = 812 := by sorry

end NUMINAMATH_CALUDE_daria_credit_card_debt_l1545_154575


namespace NUMINAMATH_CALUDE_min_value_a_l1545_154530

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) → 
  a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1545_154530


namespace NUMINAMATH_CALUDE_gcd_problem_l1545_154512

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 2927 * k) :
  Int.gcd (3 * a^2 + 61 * a + 143) (a + 19) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1545_154512


namespace NUMINAMATH_CALUDE_prob_two_defective_shipment_l1545_154577

/-- The probability of selecting two defective smartphones from a shipment -/
def prob_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of selecting two defective smartphones from a 
    shipment of 250 smartphones, of which 76 are defective, is equal to 
    (76/250) * (75/249) -/
theorem prob_two_defective_shipment : 
  prob_two_defective 250 76 = 76 / 250 * 75 / 249 := by
  sorry

#eval prob_two_defective 250 76

end NUMINAMATH_CALUDE_prob_two_defective_shipment_l1545_154577


namespace NUMINAMATH_CALUDE_reflection_yoz_plane_l1545_154542

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point P
def P : Point3D := ⟨3, 1, 5⟩

-- Define the function for reflection across the yOz plane
def reflectYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

-- Theorem statement
theorem reflection_yoz_plane :
  reflectYOZ P = ⟨-3, 1, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflection_yoz_plane_l1545_154542


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l1545_154570

theorem sum_reciprocals_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∃ S : Set ℝ, S = { x | x ≥ 9 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧ x = 1/a' + 1/b' + 1/c' } :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l1545_154570


namespace NUMINAMATH_CALUDE_fibonacci_sequence_contains_one_l1545_154511

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence x_n
def x (k m : ℕ) : ℕ → ℚ
  | 0 => (fib k : ℚ) / (fib m : ℚ)
  | (n + 1) =>
      let xn := x k m n
      if xn = 1 then 1 else (2 * xn - 1) / (1 - xn)

-- Main theorem
theorem fibonacci_sequence_contains_one (k m : ℕ) (hk : k > 0) (hm : m > k) :
  (∃ n, x k m n = 1) ↔ ∃ t : ℕ, k = 2 * t + 1 ∧ m = 2 * t + 2 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_sequence_contains_one_l1545_154511


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l1545_154596

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℤ × ℕ)), 
    (∀ (x : ℤ) (y : ℕ), (x, y) ∈ pairs ↔ 
      (∃ (m : ℕ), y = m^2 ∧ y = (x - 90)^2 - 4907)) ∧ 
    pairs.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l1545_154596


namespace NUMINAMATH_CALUDE_discount_amount_l1545_154578

theorem discount_amount (t_shirt_price backpack_price cap_price total_before_discount total_after_discount : ℕ) : 
  t_shirt_price = 30 →
  backpack_price = 10 →
  cap_price = 5 →
  total_before_discount = t_shirt_price + backpack_price + cap_price →
  total_after_discount = 43 →
  total_before_discount - total_after_discount = 2 := by
sorry

end NUMINAMATH_CALUDE_discount_amount_l1545_154578


namespace NUMINAMATH_CALUDE_town_street_lights_l1545_154548

/-- Calculates the total number of street lights in a town -/
def total_street_lights (num_neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : ℕ :=
  num_neighborhoods * roads_per_neighborhood * lights_per_side * 2

theorem town_street_lights :
  total_street_lights 10 4 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_street_lights_l1545_154548


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1545_154586

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1)
  (h_relation : a 6 = a 5 + 2 * a 4) :
  ∃ m n : ℕ, 1 / m + 4 / n = 3 / 2 ∧
    ∀ k l : ℕ, k > 0 ∧ l > 0 → 1 / k + 4 / l ≥ 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1545_154586


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1545_154561

theorem sqrt_equation_solution : ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1545_154561


namespace NUMINAMATH_CALUDE_first_employee_wage_is_12_l1545_154582

/-- The hourly wage of the first employee -/
def first_employee_wage : ℝ := sorry

/-- The hourly wage of the second employee -/
def second_employee_wage : ℝ := 22

/-- The hourly subsidy for hiring the second employee -/
def hourly_subsidy : ℝ := 6

/-- The number of hours worked per week -/
def hours_per_week : ℝ := 40

/-- The weekly savings by hiring the first employee -/
def weekly_savings : ℝ := 160

theorem first_employee_wage_is_12 :
  first_employee_wage = 12 :=
by
  have h1 : hours_per_week * (second_employee_wage - hourly_subsidy) - 
            hours_per_week * first_employee_wage = weekly_savings := by sorry
  sorry

end NUMINAMATH_CALUDE_first_employee_wage_is_12_l1545_154582


namespace NUMINAMATH_CALUDE_f_condition_iff_a_range_l1545_154522

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then 2 - a * x
  else 1/3 * x^3 - 3/2 * a * x^2 + (2 * a^2 + 2) * x - 11/6

theorem f_condition_iff_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ - f a x₂ < 2 * x₁ - 2 * x₂) ↔ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_f_condition_iff_a_range_l1545_154522


namespace NUMINAMATH_CALUDE_skee_ball_tickets_count_l1545_154552

/-- The number of tickets Tom won playing 'skee ball' -/
def skee_ball_tickets : ℕ := sorry

/-- The number of tickets Tom won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- The number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

theorem skee_ball_tickets_count : skee_ball_tickets = 25 := by
  sorry

end NUMINAMATH_CALUDE_skee_ball_tickets_count_l1545_154552


namespace NUMINAMATH_CALUDE_max_value_expression_l1545_154521

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 →
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) ≤ x) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (|4*a - 10*b| + |2*(a - b*Real.sqrt 3) - 5*(a*Real.sqrt 3 + b)|) / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 87) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1545_154521


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1545_154556

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.000000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.3 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1545_154556


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1545_154592

/-- 
Given a rectangular plot where:
- The area is 23 times its breadth
- The difference between the length and breadth is 10 metres

This theorem proves that the breadth of the plot is 13 metres.
-/
theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 23 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1545_154592


namespace NUMINAMATH_CALUDE_fraction_denominator_l1545_154553

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / x + (3 * y) / 10 = 0.7 * y) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1545_154553


namespace NUMINAMATH_CALUDE_similar_pentagons_longest_side_l1545_154507

/-- A structure representing a pentagon with its longest and shortest sides -/
structure Pentagon where
  longest : ℝ
  shortest : ℝ
  longest_ge_shortest : longest ≥ shortest

/-- Two pentagons are similar if the ratio of their corresponding sides is constant -/
def similar_pentagons (p1 p2 : Pentagon) : Prop :=
  p1.longest / p2.longest = p1.shortest / p2.shortest

theorem similar_pentagons_longest_side 
  (p1 p2 : Pentagon)
  (h_similar : similar_pentagons p1 p2)
  (h_p1_longest : p1.longest = 20)
  (h_p1_shortest : p1.shortest = 4)
  (h_p2_shortest : p2.shortest = 3) :
  p2.longest = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_pentagons_longest_side_l1545_154507


namespace NUMINAMATH_CALUDE_palace_windows_and_doors_l1545_154503

structure Palace where
  rooms : ℕ
  grid_size : ℕ
  outer_walls : ℕ
  internal_partitions : ℕ

def window_count (p : Palace) : ℕ :=
  4 * p.grid_size

def door_count (p : Palace) : ℕ :=
  p.internal_partitions * p.grid_size

theorem palace_windows_and_doors (p : Palace)
  (h1 : p.rooms = 100)
  (h2 : p.grid_size = 10)
  (h3 : p.outer_walls = 4)
  (h4 : p.internal_partitions = 18) :
  window_count p = 40 ∧ door_count p = 180 := by
  sorry

end NUMINAMATH_CALUDE_palace_windows_and_doors_l1545_154503


namespace NUMINAMATH_CALUDE_friends_games_total_l1545_154510

/-- The total number of games Katie's friends have -/
def total_friends_games (new_friends_games old_friends_games : ℕ) : ℕ :=
  new_friends_games + old_friends_games

/-- Theorem: Katie's friends have 141 games in total -/
theorem friends_games_total :
  total_friends_games 88 53 = 141 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_total_l1545_154510


namespace NUMINAMATH_CALUDE_day300_is_saturday_l1545_154589

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in the year 2004 -/
structure Date2004 where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Function to advance a date by a given number of days -/
def advanceDate (d : Date2004) (days : Nat) : Date2004 :=
  sorry

/-- The 50th day of 2004 is a Monday -/
def day50 : Date2004 :=
  { dayNumber := 50, dayOfWeek := DayOfWeek.Monday }

theorem day300_is_saturday :
  (advanceDate day50 250).dayOfWeek = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_day300_is_saturday_l1545_154589


namespace NUMINAMATH_CALUDE_min_words_for_spanish_exam_l1545_154555

/-- Represents the Spanish vocabulary exam scenario -/
structure SpanishExam where
  total_words : ℕ
  min_score_percent : ℕ

/-- Calculates the minimum number of words needed to achieve the desired score -/
def min_words_needed (exam : SpanishExam) : ℕ :=
  (exam.min_score_percent * exam.total_words + 99) / 100

/-- Theorem stating the minimum number of words needed for the given exam conditions -/
theorem min_words_for_spanish_exam :
  let exam : SpanishExam := { total_words := 500, min_score_percent := 85 }
  min_words_needed exam = 425 := by
  sorry

#eval min_words_needed { total_words := 500, min_score_percent := 85 }

end NUMINAMATH_CALUDE_min_words_for_spanish_exam_l1545_154555


namespace NUMINAMATH_CALUDE_mink_babies_problem_l1545_154593

/-- Represents the problem of determining the number of babies each mink had --/
theorem mink_babies_problem (initial_minks : ℕ) (coats_made : ℕ) (skins_per_coat : ℕ) 
  (h1 : initial_minks = 30)
  (h2 : coats_made = 7)
  (h3 : skins_per_coat = 15) :
  ∃ babies_per_mink : ℕ, 
    (initial_minks + initial_minks * babies_per_mink) / 2 = coats_made * skins_per_coat ∧ 
    babies_per_mink = 6 := by
  sorry

end NUMINAMATH_CALUDE_mink_babies_problem_l1545_154593
