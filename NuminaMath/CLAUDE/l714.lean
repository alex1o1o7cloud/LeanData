import Mathlib

namespace complex_equation_real_part_l714_71469

theorem complex_equation_real_part (z : ℂ) (a b : ℝ) (h1 : z = a + b * Complex.I) 
  (h2 : b > 0) (h3 : z * (z + 2 * Complex.I) * (z - 2 * Complex.I) * (z + 5 * Complex.I) = 8000) :
  a^3 - 4*a = 8000 := by
  sorry

end complex_equation_real_part_l714_71469


namespace frac_5_23_150th_digit_l714_71442

/-- The decimal expansion of 5/23 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The period of the decimal expansion of 5/23 -/
def period : ℕ := 23

theorem frac_5_23_150th_digit : 
  decimal_expansion ((150 - 1) % period + 1) = 1 := by sorry

end frac_5_23_150th_digit_l714_71442


namespace root_product_theorem_l714_71440

theorem root_product_theorem (c d m r s : ℝ) : 
  (c^2 - m*c + 3 = 0) →
  (d^2 - m*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end root_product_theorem_l714_71440


namespace point_not_outside_implies_on_or_inside_l714_71401

-- Define a circle in a 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the possible relationships between a point and a circle
inductive PointCircleRelation
  | Inside
  | On
  | Outside

-- Function to determine the relation between a point and a circle
def pointCircleRelation (p : ℝ × ℝ) (c : Circle) : PointCircleRelation :=
  sorry

-- Theorem statement
theorem point_not_outside_implies_on_or_inside
  (p : ℝ × ℝ) (c : Circle) :
  pointCircleRelation p c ≠ PointCircleRelation.Outside →
  (pointCircleRelation p c = PointCircleRelation.On ∨
   pointCircleRelation p c = PointCircleRelation.Inside) :=
by sorry

end point_not_outside_implies_on_or_inside_l714_71401


namespace additional_cupcakes_count_l714_71434

-- Define the initial number of cupcakes
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes after making additional ones
def total_cupcakes : ℕ := 49

-- Theorem to prove
theorem additional_cupcakes_count :
  total_cupcakes - (initial_cupcakes - sold_cupcakes) = 28 :=
by sorry

end additional_cupcakes_count_l714_71434


namespace point_movement_l714_71460

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Move a point up by a given number of units -/
def moveUp (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

/-- Move a point left by a given number of units -/
def moveLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

theorem point_movement :
  let A : Point2D := { x := 1, y := -2 }
  let B : Point2D := moveLeft (moveUp A 3) 2
  B.x = -1 ∧ B.y = 1 := by sorry

end point_movement_l714_71460


namespace probability_of_specific_arrangement_l714_71443

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

theorem probability_of_specific_arrangement :
  let total_arrangements := Nat.choose total_tiles x_tiles
  let specific_arrangement := 1
  (specific_arrangement : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end probability_of_specific_arrangement_l714_71443


namespace total_coughs_after_20_minutes_l714_71449

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of coughs per minute for Robert -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The duration in minutes -/
def duration : ℕ := 20

/-- The total number of coughs after the given duration -/
def total_coughs : ℕ := (georgia_coughs_per_minute + robert_coughs_per_minute) * duration

theorem total_coughs_after_20_minutes :
  total_coughs = 300 := by sorry

end total_coughs_after_20_minutes_l714_71449


namespace midpoint_vector_relation_l714_71461

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C M E : V)

/-- Given a triangle ABC with M as the midpoint of BC and E on AC such that EC = 2AE,
    prove that EM = (1/6)AC - (1/2)AB -/
theorem midpoint_vector_relation 
  (h_midpoint : M = (1/2 : ℝ) • (B + C))
  (h_on_side : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C)
  (h_ec_ae : C - E = (2 : ℝ) • (E - A)) :
  E - M = (1/6 : ℝ) • (C - A) - (1/2 : ℝ) • (B - A) := by sorry

end midpoint_vector_relation_l714_71461


namespace production_order_machines_correct_initial_machines_l714_71416

/-- The number of machines initially used to complete a production order -/
def initial_machines : ℕ := 3

/-- The time (in hours) to complete the order with the initial number of machines -/
def initial_time : ℕ := 44

/-- The time (in hours) to complete the order with one additional machine -/
def reduced_time : ℕ := 33

/-- The production rate of a single machine (assumed to be constant) -/
def machine_rate : ℚ := 1 / initial_machines / initial_time

theorem production_order_machines :
  (initial_machines * machine_rate * initial_time : ℚ) =
  ((initial_machines + 1) * machine_rate * reduced_time : ℚ) :=
sorry

theorem correct_initial_machines :
  initial_machines = 3 :=
sorry

end production_order_machines_correct_initial_machines_l714_71416


namespace sign_determination_l714_71485

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end sign_determination_l714_71485


namespace radio_price_proof_l714_71451

theorem radio_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 5) : 
  ∃ (original_price : ℝ), 
    original_price = 490 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end radio_price_proof_l714_71451


namespace rhombus_diagonal_l714_71444

/-- Proves that in a rhombus with one diagonal of 40 m and an area of 600 m², 
    the length of the other diagonal is 30 m. -/
theorem rhombus_diagonal (d₁ d₂ : ℝ) (area : ℝ) : 
  d₁ = 40 → area = 600 → area = (d₁ * d₂) / 2 → d₂ = 30 := by
  sorry

end rhombus_diagonal_l714_71444


namespace tangent_perpendicular_and_inequality_l714_71427

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ k : ℝ, (deriv (f a) 1 = k) ∧ (k * (-1) = -1)) →
  (∃ m : ℝ, ∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x ∈ Set.Icc 1 (exp 1), f a x ≤ m * x) → m ≥ 4 / (3 * exp 1 + 1)) :=
by sorry

end tangent_perpendicular_and_inequality_l714_71427


namespace no_multiple_of_five_2c4_l714_71490

/-- A positive three-digit number in the form 2C4, where C is a single digit. -/
def number (c : ℕ) : ℕ := 200 + 10 * c + 4

/-- Predicate to check if a number is a multiple of 5. -/
def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

/-- Theorem stating that there are no digits C such that 2C4 is a multiple of 5. -/
theorem no_multiple_of_five_2c4 :
  ¬ ∃ c : ℕ, c < 10 ∧ is_multiple_of_five (number c) := by
  sorry


end no_multiple_of_five_2c4_l714_71490


namespace second_bakery_sacks_per_week_l714_71448

/-- Proves that the second bakery needs 4 sacks per week given the conditions of Antoine's strawberry supply -/
theorem second_bakery_sacks_per_week 
  (total_sacks : ℕ) 
  (num_weeks : ℕ) 
  (first_bakery_sacks_per_week : ℕ) 
  (third_bakery_sacks_per_week : ℕ) 
  (h1 : total_sacks = 72) 
  (h2 : num_weeks = 4) 
  (h3 : first_bakery_sacks_per_week = 2) 
  (h4 : third_bakery_sacks_per_week = 12) : 
  (total_sacks - (first_bakery_sacks_per_week * num_weeks) - (third_bakery_sacks_per_week * num_weeks)) / num_weeks = 4 := by
sorry

end second_bakery_sacks_per_week_l714_71448


namespace power_zero_plus_two_l714_71402

theorem power_zero_plus_two : (-2010)^0 + 2 = 3 := by
  sorry

end power_zero_plus_two_l714_71402


namespace parallel_line_slope_l714_71464

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) : 
  (3 : ℝ) * a - (6 : ℝ) * b = (12 : ℝ) → 
  ∃ (m : ℝ), m = (1 : ℝ) / (2 : ℝ) ∧ 
  ∀ (x y : ℝ), (y = m * x + c) → 
  (∃ (k : ℝ), (3 : ℝ) * x - (6 : ℝ) * y = k) :=
by sorry

end parallel_line_slope_l714_71464


namespace probability_unqualified_example_l714_71438

/-- Represents the probability of selecting at least one unqualified can -/
def probability_unqualified (total_cans : ℕ) (qualified_cans : ℕ) (unqualified_cans : ℕ) (selected_cans : ℕ) : ℚ :=
  1 - (Nat.choose qualified_cans selected_cans : ℚ) / (Nat.choose total_cans selected_cans : ℚ)

/-- Theorem stating that the probability of selecting at least one unqualified can
    when randomly choosing 2 cans from a box containing 3 qualified cans and 2 unqualified cans
    is equal to 0.7 -/
theorem probability_unqualified_example : probability_unqualified 5 3 2 2 = 7/10 := by
  sorry

end probability_unqualified_example_l714_71438


namespace inequality_not_always_true_l714_71435

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c :=
by sorry

end inequality_not_always_true_l714_71435


namespace derivative_at_one_l714_71475

def f (x : ℝ) : ℝ := (x - 2)^2

theorem derivative_at_one :
  deriv f 1 = -2 := by
  sorry

end derivative_at_one_l714_71475


namespace zoe_coloring_books_l714_71463

/-- Given two coloring books with the same number of pictures and the number of pictures left to color,
    calculate the number of pictures colored. -/
def pictures_colored (pictures_per_book : ℕ) (books : ℕ) (pictures_left : ℕ) : ℕ :=
  pictures_per_book * books - pictures_left

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color,
    the number of pictures colored is 20. -/
theorem zoe_coloring_books : pictures_colored 44 2 68 = 20 := by
  sorry

end zoe_coloring_books_l714_71463


namespace first_two_nonzero_digits_of_one_over_137_l714_71456

theorem first_two_nonzero_digits_of_one_over_137 :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (1 : ℚ) / 137 = (a * 10 + b : ℕ) / 1000 + r ∧ 0 ≤ r ∧ r < 1 / 100 ∧ a = 7 ∧ b = 6 := by
  sorry

end first_two_nonzero_digits_of_one_over_137_l714_71456


namespace total_spent_by_pete_and_raymond_l714_71404

def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def initial_amount : ℕ := 250

def pete_nickels : ℕ := 4
def pete_dimes : ℕ := 3
def pete_quarters : ℕ := 2

def raymond_dimes_left : ℕ := 7
def raymond_quarters_left : ℕ := 4
def raymond_nickels_left : ℕ := 5

theorem total_spent_by_pete_and_raymond : 
  (initial_amount - (raymond_dimes_left * dime_value + raymond_quarters_left * quarter_value + raymond_nickels_left * nickel_value)) +
  (pete_nickels * nickel_value + pete_dimes * dime_value + pete_quarters * quarter_value) = 155 := by
  sorry

end total_spent_by_pete_and_raymond_l714_71404


namespace tight_sequence_x_range_l714_71468

/-- Definition of a tight sequence -/
def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

/-- Theorem about the range of x in a specific tight sequence -/
theorem tight_sequence_x_range (a : ℕ → ℝ) (x : ℝ) 
  (h_tight : is_tight_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3/2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) :
  2 ≤ x ∧ x ≤ 3 := by
sorry

end tight_sequence_x_range_l714_71468


namespace total_sample_volume_l714_71419

/-- The total sample volume -/
def M : ℝ := 50

/-- The frequency of the first group -/
def freq1 : ℝ := 10

/-- The frequency of the second group -/
def freq2 : ℝ := 0.35

/-- The frequency of the third group -/
def freq3 : ℝ := 0.45

/-- Theorem stating that M is the correct total sample volume given the frequencies -/
theorem total_sample_volume : M = freq1 + freq2 * M + freq3 * M := by
  sorry

end total_sample_volume_l714_71419


namespace space_diagonals_of_specific_polyhedron_l714_71417

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 30 ∧
    Q.pentagonal_faces = 10 ∧
    space_diagonals Q = 315 :=
  sorry

end space_diagonals_of_specific_polyhedron_l714_71417


namespace percent_decrease_l714_71400

theorem percent_decrease (X Y : ℝ) (h : Y = 1.2 * X) :
  X = Y * (1 - 1/6) :=
by sorry

end percent_decrease_l714_71400


namespace remainder_theorem_polynomial_remainder_l714_71466

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∃ q : ℝ → ℝ, f = λ x => (x - a) * q x + f a) :=
sorry

theorem polynomial_remainder (f : ℝ → ℝ) (h : f = λ x => x^8 + 3) :
  ∃ q : ℝ → ℝ, f = λ x => (x + 1) * q x + 4 := by
  sorry

end remainder_theorem_polynomial_remainder_l714_71466


namespace tiles_needed_to_cover_floor_l714_71425

-- Define the dimensions of the floor and tiles
def floor_length : ℚ := 10
def floor_width : ℚ := 14
def tile_length : ℚ := 1/2  -- 6 inches in feet
def tile_width : ℚ := 2/3   -- 8 inches in feet

-- Theorem statement
theorem tiles_needed_to_cover_floor :
  (floor_length * floor_width) / (tile_length * tile_width) = 420 := by
  sorry

end tiles_needed_to_cover_floor_l714_71425


namespace third_smallest_prime_squared_cubed_l714_71424

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- State the theorem
theorem third_smallest_prime_squared_cubed :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by sorry

end third_smallest_prime_squared_cubed_l714_71424


namespace scientific_notation_425000_l714_71459

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_425000 :
  toScientificNotation 425000 = ScientificNotation.mk 4.25 5 sorry := by
  sorry

end scientific_notation_425000_l714_71459


namespace digit_150_of_one_thirteenth_l714_71418

def decimal_representation (n : ℕ) : ℚ → List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem digit_150_of_one_thirteenth :
  let rep := decimal_representation 13 (1/13)
  nth_digit 150 rep = 3 := by sorry

end digit_150_of_one_thirteenth_l714_71418


namespace symmetric_point_x_axis_l714_71445

def point_symmetric_to_x_axis (x y : ℝ) : ℝ × ℝ := (x, -y)

theorem symmetric_point_x_axis :
  let M : ℝ × ℝ := (1, 2)
  point_symmetric_to_x_axis M.1 M.2 = (1, -2) := by sorry

end symmetric_point_x_axis_l714_71445


namespace addilynns_broken_eggs_l714_71491

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of eggs Addilynn bought -/
def dozens_bought : ℕ := 6

/-- The number of eggs left on the shelf -/
def eggs_left : ℕ := 21

/-- The number of eggs Addilynn accidentally broke -/
def eggs_broken : ℕ := dozens_bought * dozen / 2 - eggs_left

theorem addilynns_broken_eggs :
  eggs_broken = 15 :=
sorry

end addilynns_broken_eggs_l714_71491


namespace negation_of_existence_cubic_equation_negation_l714_71483

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end negation_of_existence_cubic_equation_negation_l714_71483


namespace hyperbola_equation_l714_71405

/-- Given a hyperbola with eccentricity 5/4 and semi-major axis length 4,
    prove that its equation is x²/16 - y²/9 = 1 --/
theorem hyperbola_equation (x y : ℝ) :
  let a : ℝ := 4
  let e : ℝ := 5/4
  let c : ℝ := e * a
  let b : ℝ := Real.sqrt (c^2 - a^2)
  (x^2 / a^2) - (y^2 / b^2) = 1 → x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end hyperbola_equation_l714_71405


namespace cyclic_inequality_l714_71447

theorem cyclic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  18 * (1 / ((3 - a) * (4 - a)) + 1 / ((3 - b) * (4 - b)) + 1 / ((3 - c) * (4 - c))) + 
  2 * (a * b + b * c + c * a) ≥ 15 := by
sorry


end cyclic_inequality_l714_71447


namespace f_properties_l714_71410

/-- The function f(x) = x³ - 5x² + 3x -/
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 10*x + 3

theorem f_properties :
  (f' 3 = 0) ∧ 
  (f 1 = -1) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≥ -9) ∧
  (f 3 = -9) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≤ -4) ∧
  (f 4 = -4) := by
  sorry

end f_properties_l714_71410


namespace inscribed_circle_radius_l714_71498

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let side₁ := R * Real.sqrt 3
  let height := side₁ * (Real.sqrt 3 / 2)
  let side₂ := 2 * height / Real.sqrt 3
  let r := side₂ * Real.sqrt 3 / 6
  r = 3 / 4 := by sorry

end inscribed_circle_radius_l714_71498


namespace quadratic_solution_existence_l714_71476

theorem quadratic_solution_existence (a b c : ℝ) (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (h1 : f 3.11 < 0)
  (h2 : f 3.12 > 0) :
  ∃ x : ℝ, f x = 0 ∧ 3.11 < x ∧ x < 3.12 := by
  sorry

end quadratic_solution_existence_l714_71476


namespace arithmetic_sequence_properties_l714_71489

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 5 = 3 ∧ 
  a 6 = -2

/-- The first term of the sequence -/
def FirstTerm (a : ℕ → ℤ) : ℤ := a 1

/-- The common difference of the sequence -/
def CommonDifference (a : ℕ → ℤ) : ℤ := a 2 - a 1

/-- The general term formula of the sequence -/
def GeneralTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := 28 - 5 * n

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : ArithmeticSequence a) : 
  FirstTerm a = 23 ∧ 
  CommonDifference a = -5 ∧ 
  (∀ n : ℕ, a n = GeneralTerm a n) := by
  sorry

end arithmetic_sequence_properties_l714_71489


namespace shelby_today_stars_l714_71420

-- Define the variables
def yesterday_stars : ℕ := 4
def total_stars : ℕ := 7

-- State the theorem
theorem shelby_today_stars : 
  total_stars - yesterday_stars = 3 := by
  sorry

end shelby_today_stars_l714_71420


namespace coins_missing_fraction_l714_71471

theorem coins_missing_fraction (initial_coins : ℚ) : 
  initial_coins > 0 →
  let lost_coins := (1 / 3 : ℚ) * initial_coins
  let found_coins := (2 / 3 : ℚ) * lost_coins
  let remaining_coins := initial_coins - lost_coins + found_coins
  (initial_coins - remaining_coins) / initial_coins = (1 / 9 : ℚ) :=
by sorry

end coins_missing_fraction_l714_71471


namespace power_of_three_product_l714_71453

theorem power_of_three_product (x : ℕ) : 3^12 * 3^18 = x^6 → x = 243 := by
  sorry

end power_of_three_product_l714_71453


namespace nina_raisins_l714_71497

/-- The number of raisins Nina and Max received satisfies the given conditions -/
def raisin_distribution (nina max : ℕ) : Prop :=
  nina = max + 8 ∧ 
  max = nina / 3 ∧ 
  nina + max = 16

theorem nina_raisins :
  ∀ nina max : ℕ, raisin_distribution nina max → nina = 12 := by
  sorry

end nina_raisins_l714_71497


namespace pyramid_layers_l714_71487

/-- Calculates the number of exterior golfballs for a given layer -/
def exteriorGolfballs (layer : ℕ) : ℕ :=
  if layer ≤ 2 then layer * layer else 4 * (layer - 1)

/-- Calculates the total number of exterior golfballs up to a given layer -/
def totalExteriorGolfballs (n : ℕ) : ℕ :=
  (List.range n).map exteriorGolfballs |>.sum

/-- Theorem stating that a pyramid with 145 exterior golfballs has 9 layers -/
theorem pyramid_layers (n : ℕ) : totalExteriorGolfballs n = 145 ↔ n = 9 := by
  sorry

#eval totalExteriorGolfballs 9  -- Should output 145

end pyramid_layers_l714_71487


namespace inequality_proof_l714_71470

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 := by
  sorry

end inequality_proof_l714_71470


namespace sin_cos_sum_equals_quarter_l714_71409

theorem sin_cos_sum_equals_quarter : 
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by sorry

end sin_cos_sum_equals_quarter_l714_71409


namespace total_students_l714_71429

theorem total_students (middle_school : ℕ) (elementary_school : ℕ) : 
  middle_school = 50 →
  elementary_school = 4 * middle_school - 3 →
  middle_school + elementary_school = 247 := by
sorry

end total_students_l714_71429


namespace smallest_integer_satisfying_inequality_l714_71496

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 14 → x ≥ 8 ∧ 8 < 3*8 - 14 :=
by sorry

end smallest_integer_satisfying_inequality_l714_71496


namespace at_most_one_perfect_square_l714_71465

def sequence_a : ℕ → ℤ
  | 0 => 0  -- arbitrary initial value
  | n + 1 => (sequence_a n)^3 + 103

theorem at_most_one_perfect_square :
  ∃ (n : ℕ), (∃ (k : ℤ), sequence_a n = k^2) →
    ∀ (m : ℕ), m ≠ n → ¬∃ (l : ℤ), sequence_a m = l^2 :=
by sorry

end at_most_one_perfect_square_l714_71465


namespace red_notes_per_row_l714_71472

theorem red_notes_per_row (total_rows : ℕ) (total_notes : ℕ) (extra_blue : ℕ) :
  total_rows = 5 →
  total_notes = 100 →
  extra_blue = 10 →
  ∃ (red_per_row : ℕ),
    red_per_row = 6 ∧
    total_notes = total_rows * red_per_row + 2 * (total_rows * red_per_row) + extra_blue :=
by sorry

end red_notes_per_row_l714_71472


namespace smallest_quadratic_coefficient_l714_71482

theorem smallest_quadratic_coefficient (a b c : ℤ) :
  (∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
   a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  |a| ≥ 5 :=
sorry

end smallest_quadratic_coefficient_l714_71482


namespace functional_equation_solution_l714_71413

-- Define the functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y * f (x - y) = x^2 * f y - y^2 * f x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry


end functional_equation_solution_l714_71413


namespace parabola_equation_with_sqrt3_distance_l714_71422

/-- Represents a parabola opening upwards -/
structure UprightParabola where
  /-- The distance from the focus to the directrix -/
  focus_directrix_distance : ℝ
  /-- Condition that the parabola opens upwards -/
  opens_upward : focus_directrix_distance > 0

/-- The standard equation of an upright parabola -/
def standard_equation (p : UprightParabola) : Prop :=
  ∀ x y : ℝ, x^2 = 2 * p.focus_directrix_distance * y

/-- Theorem stating the standard equation of a parabola with focus-directrix distance √3 -/
theorem parabola_equation_with_sqrt3_distance :
  ∀ (p : UprightParabola),
    p.focus_directrix_distance = Real.sqrt 3 →
    standard_equation p
    := by sorry

end parabola_equation_with_sqrt3_distance_l714_71422


namespace roots_eccentricity_l714_71436

theorem roots_eccentricity (x₁ x₂ : ℝ) : 
  x₁ * x₂ = 1 → x₁ + x₂ = 79 → (x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1) := by
  sorry

end roots_eccentricity_l714_71436


namespace vertices_form_parabola_l714_71481

/-- The set of vertices of a family of parabolas forms another parabola -/
theorem vertices_form_parabola (a c : ℝ) (h_a : a = 2) (h_c : c = 6) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, f t = (-(t / (2 * a)), a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + c)) ∧
    (∃ (g : ℝ → ℝ), ∀ x, (x, g x) ∈ Set.range f ↔ g x = -a * x^2 + c) :=
by sorry

end vertices_form_parabola_l714_71481


namespace abs_neg_2023_eq_2023_l714_71499

theorem abs_neg_2023_eq_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l714_71499


namespace even_sum_implies_one_even_l714_71457

theorem even_sum_implies_one_even (a b c : ℕ) :
  Even (a + b + c) →
  ¬((Odd a ∧ Odd b ∧ Odd c) ∨ 
    (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)) :=
by sorry

end even_sum_implies_one_even_l714_71457


namespace clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l714_71454

/-- The time in minutes when the hour and minute hands of a clock coincide after midnight -/
def coincidence_time : ℚ :=
  720 / 11

theorem clock_hands_coincidence :
  let minute_speed : ℚ := 360 / 60  -- degrees per minute
  let hour_speed : ℚ := 360 / 720   -- degrees per minute
  ∀ t : ℚ,
    t > 0 →
    t < coincidence_time →
    minute_speed * t ≠ hour_speed * t + 360 * (t / 720).floor →
    minute_speed * coincidence_time = hour_speed * coincidence_time + 360 :=
by sorry

theorem coincidence_time_in_hours_and_minutes :
  (coincidence_time / 60).floor = 1 ∧
  (coincidence_time % 60 : ℚ) = 65 / 11 :=
by sorry

end clock_hands_coincidence_coincidence_time_in_hours_and_minutes_l714_71454


namespace monster_family_eyes_l714_71415

/-- Represents the number of eyes for each family member -/
structure MonsterEyes where
  mom : Nat
  dad : Nat
  child : Nat
  num_children : Nat

/-- Calculates the total number of eyes in the monster family -/
def total_eyes (m : MonsterEyes) : Nat :=
  m.mom + m.dad + m.child * m.num_children

/-- Theorem stating that the total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ m : MonsterEyes, m.mom = 1 ∧ m.dad = 3 ∧ m.child = 4 ∧ m.num_children = 3 ∧ total_eyes m = 16 := by
  sorry

end monster_family_eyes_l714_71415


namespace unique_negative_zero_implies_a_gt_two_l714_71467

/-- The function f(x) = ax³ - 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The unique zero point of f(x) -/
noncomputable def x₀ (a : ℝ) : ℝ := sorry

theorem unique_negative_zero_implies_a_gt_two (a : ℝ) :
  (∃! x, f a x = 0) ∧ (x₀ a < 0) → a > 2 :=
by sorry

end unique_negative_zero_implies_a_gt_two_l714_71467


namespace smallest_x_for_cube_l714_71477

theorem smallest_x_for_cube (x : ℕ+) (N : ℤ) : 
  (∀ y : ℕ+, y < x → ¬∃ M : ℤ, 1890 * y = M^3) ∧ 
  1890 * x = N^3 ↔ 
  x = 4900 := by
sorry

end smallest_x_for_cube_l714_71477


namespace normal_carwash_cost_l714_71452

/-- The normal cost of a single carwash, given a discounted package deal -/
theorem normal_carwash_cost (package_size : ℕ) (discount_rate : ℚ) (package_price : ℚ) : 
  package_size = 20 →
  discount_rate = 3/5 →
  package_price = 180 →
  package_price = discount_rate * (package_size * (15 : ℚ)) := by
sorry

end normal_carwash_cost_l714_71452


namespace average_marks_combined_classes_l714_71426

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 50 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = 4500 →
  (n1 + n2 : ℚ) = 80 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / (n1 + n2 : ℚ) = 56.25 := by
  sorry

end average_marks_combined_classes_l714_71426


namespace probability_blue_then_yellow_l714_71462

def blue_marbles : ℕ := 3
def yellow_marbles : ℕ := 4
def pink_marbles : ℕ := 9

def total_marbles : ℕ := blue_marbles + yellow_marbles + pink_marbles

theorem probability_blue_then_yellow :
  (blue_marbles : ℚ) / total_marbles * yellow_marbles / (total_marbles - 1) = 1 / 20 := by
  sorry

end probability_blue_then_yellow_l714_71462


namespace f_negative_2016_l714_71406

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

theorem f_negative_2016 (a : ℝ) :
  f a 2016 = 5 → f a (-2016) = -7 := by
  sorry

end f_negative_2016_l714_71406


namespace cube_diff_even_iff_sum_even_l714_71455

theorem cube_diff_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end cube_diff_even_iff_sum_even_l714_71455


namespace cos_difference_l714_71432

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end cos_difference_l714_71432


namespace inequality_proof_l714_71439

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) : 
  2 * Real.sqrt (x + Real.sqrt y) + 2 * Real.sqrt (y + Real.sqrt z) + 2 * Real.sqrt (z + Real.sqrt x) 
  ≤ Real.sqrt (8 + x - y) + Real.sqrt (8 + y - z) + Real.sqrt (8 + z - x) := by
sorry

end inequality_proof_l714_71439


namespace cube_difference_l714_71414

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by sorry

end cube_difference_l714_71414


namespace product_purchase_l714_71437

theorem product_purchase (misunderstood_total : ℕ) (actual_total : ℕ) 
  (h1 : misunderstood_total = 189)
  (h2 : actual_total = 147) :
  ∃ (price : ℕ) (quantity : ℕ),
    price * quantity = actual_total ∧
    (price + 6) * quantity = misunderstood_total ∧
    price = 21 ∧
    quantity = 7 := by
  sorry

end product_purchase_l714_71437


namespace right_handed_players_count_l714_71474

/-- Calculates the total number of right-handed players in a cricket team. -/
theorem right_handed_players_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : 
  throwers + ((total_players - throwers) * 2 / 3) = 59 := by
  sorry

#check right_handed_players_count

end right_handed_players_count_l714_71474


namespace sentence_reappears_l714_71492

-- Define the type for documents
def Document := List String

-- Define William's word assignment function
def wordAssignment : Char → String := sorry

-- Generate the nth document
def generateDocument : ℕ → Document
  | 0 => [wordAssignment 'A']
  | n + 1 => sorry -- Replace each letter in the previous document with its assigned word

-- The 40th document starts with this sentence
def startingSentence : List String :=
  ["Till", "whatsoever", "star", "that", "guides", "my", "moving"]

-- Main theorem
theorem sentence_reappears (d : Document) (h : d = generateDocument 40) :
  ∃ (i j : ℕ), i < j ∧ j < d.length ∧
  (List.take 7 (List.drop i d) = startingSentence) ∧
  (List.take 7 (List.drop j d) = startingSentence) :=
sorry

end sentence_reappears_l714_71492


namespace rational_expression_proof_l714_71403

theorem rational_expression_proof (a b c : ℚ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ∃ (q : ℚ), q = |1 / (a - b) + 1 / (b - c) + 1 / (c - a)| := by
  sorry

end rational_expression_proof_l714_71403


namespace same_sign_as_B_l714_71495

/-- A line in 2D space defined by the equation Ax + By + C = 0 -/
structure Line2D where
  A : ℝ
  B : ℝ
  C : ℝ
  A_nonzero : A ≠ 0
  B_nonzero : B ≠ 0

/-- Determines if a point (x, y) is above a given line -/
def IsAboveLine (l : Line2D) (x y : ℝ) : Prop :=
  l.A * x + l.B * y + l.C > 0

/-- The main theorem stating that for a point above the line, 
    Ax + By + C has the same sign as B -/
theorem same_sign_as_B (l : Line2D) (x y : ℝ) 
    (h : IsAboveLine l x y) : 
    (l.A * x + l.B * y + l.C) * l.B > 0 := by
  sorry

end same_sign_as_B_l714_71495


namespace pizza_three_toppings_l714_71450

/-- Represents a pizza with 24 slices and three toppings -/
structure Pizza :=
  (pepperoni : Finset Nat)
  (mushrooms : Finset Nat)
  (olives : Finset Nat)
  (h1 : pepperoni ∪ mushrooms ∪ olives = Finset.range 24)
  (h2 : pepperoni.card = 15)
  (h3 : mushrooms.card = 14)
  (h4 : olives.card = 12)
  (h5 : (pepperoni ∩ mushrooms).card = 6)
  (h6 : (mushrooms ∩ olives).card = 5)
  (h7 : (pepperoni ∩ olives).card = 4)

theorem pizza_three_toppings (p : Pizza) : (p.pepperoni ∩ p.mushrooms ∩ p.olives).card = 0 := by
  sorry

end pizza_three_toppings_l714_71450


namespace complement_of_intersection_l714_71478

open Set

theorem complement_of_intersection (A B : Set ℝ) : 
  A = {x : ℝ | x ≤ 1} → 
  B = {x : ℝ | x < 2} → 
  (A ∩ B)ᶜ = {x : ℝ | x > 1} := by
sorry

end complement_of_intersection_l714_71478


namespace johns_journey_distance_l714_71493

theorem johns_journey_distance :
  let total_distance : ℚ := 360 / 7
  let highway_distance : ℚ := total_distance / 4
  let city_distance : ℚ := 30
  let country_distance : ℚ := total_distance / 6
  highway_distance + city_distance + country_distance = total_distance := by
  sorry

end johns_journey_distance_l714_71493


namespace vampire_blood_requirement_l714_71421

-- Define the constants
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Define the theorem
theorem vampire_blood_requirement :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := by
  sorry

end vampire_blood_requirement_l714_71421


namespace johns_price_calculation_l714_71431

/-- The price per sheet charged by John's Photo World -/
def johns_price_per_sheet : ℚ := 2.75

/-- The sitting fee charged by John's Photo World -/
def johns_sitting_fee : ℚ := 125

/-- The price per sheet charged by Sam's Picture Emporium -/
def sams_price_per_sheet : ℚ := 1.50

/-- The sitting fee charged by Sam's Picture Emporium -/
def sams_sitting_fee : ℚ := 140

/-- The number of sheets for which both companies charge the same amount -/
def num_sheets : ℕ := 12

theorem johns_price_calculation :
  johns_price_per_sheet * num_sheets + johns_sitting_fee =
  sams_price_per_sheet * num_sheets + sams_sitting_fee :=
by sorry

end johns_price_calculation_l714_71431


namespace right_triangular_pyramid_volume_l714_71412

/-- A right triangular pyramid with base edge length 2 and pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  side_edges_perpendicular : Prop

/-- The volume of a right triangular pyramid -/
def volume (p : RightTriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of a right triangular pyramid with base edge length 2 
    and pairwise perpendicular side edges is √2/3 -/
theorem right_triangular_pyramid_volume :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge_length = 2 ∧ p.side_edges_perpendicular →
    volume p = Real.sqrt 2 / 3 := by sorry

end right_triangular_pyramid_volume_l714_71412


namespace square_sum_product_l714_71473

theorem square_sum_product (x : ℝ) : 
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
sorry

end square_sum_product_l714_71473


namespace truck_weight_l714_71458

/-- Given a truck and trailer with specified weight relationship, prove the truck's weight -/
theorem truck_weight (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end truck_weight_l714_71458


namespace gnuff_tutoring_cost_l714_71441

/-- Calculates the total amount paid for a tutoring session -/
def tutoring_cost (flat_rate : ℕ) (per_minute_rate : ℕ) (minutes : ℕ) : ℕ :=
  flat_rate + per_minute_rate * minutes

/-- Theorem: The total amount paid for Gnuff's tutoring session is $146 -/
theorem gnuff_tutoring_cost :
  tutoring_cost 20 7 18 = 146 := by
  sorry

end gnuff_tutoring_cost_l714_71441


namespace a_upper_bound_l714_71430

theorem a_upper_bound (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x * y = 2 → 
    2 - x ≥ a / (4 - y)) → 
  a ≤ 0 := by
sorry

end a_upper_bound_l714_71430


namespace max_rectangle_area_max_area_condition_l714_71411

/-- The maximum area of a rectangle with perimeter 40 meters (excluding one side) is 200 square meters. -/
theorem max_rectangle_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : x * y ≤ 200 := by
  sorry

/-- The maximum area is achieved when the length is twice the width. -/
theorem max_area_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : 
  x * y = 200 ↔ x = 2*y := by
  sorry

end max_rectangle_area_max_area_condition_l714_71411


namespace cubic_equation_natural_roots_l714_71433

theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ (x y : ℕ) (z : ℝ), 
    x ≠ y ∧ 
    (5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p) ∧
    (5 * y^3 - 5*(p+1)*y^2 + (71*p-1)*y + 1 = 66*p) ∧
    (5 * z^3 - 5*(p+1)*z^2 + (71*p-1)*z + 1 = 66*p)) ↔ 
  p = 76 := by
sorry

end cubic_equation_natural_roots_l714_71433


namespace biking_distance_difference_l714_71494

/-- The distance biked by Daniela in four hours -/
def daniela_distance : ℤ := 75

/-- The distance biked by Carlos in four hours -/
def carlos_distance : ℤ := 60

/-- The distance biked by Emilio in four hours -/
def emilio_distance : ℤ := 45

/-- Theorem stating the difference between Daniela's distance and the sum of Carlos' and Emilio's distances -/
theorem biking_distance_difference : 
  daniela_distance - (carlos_distance + emilio_distance) = -30 := by
  sorry

end biking_distance_difference_l714_71494


namespace prob_only_one_selected_l714_71479

/-- The probability of only one person being selected given individual and joint selection probabilities -/
theorem prob_only_one_selected
  (pH : ℚ) (pW : ℚ) (pHW : ℚ)
  (hpH : pH = 2 / 5)
  (hpW : pW = 3 / 7)
  (hpHW : pHW = 1 / 3) :
  pH * (1 - pW) + (1 - pH) * pW = 17 / 35 := by
sorry


end prob_only_one_selected_l714_71479


namespace only_one_valid_assignment_l714_71446

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Assign (lhs : String) (rhs : String)

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Assign lhs rhs => 
    (lhs.all Char.isAlpha) && (rhs ≠ "")

/-- The list of given statements --/
def givenStatements : List AssignmentStatement := [
  AssignmentStatement.Assign "2" "A",
  AssignmentStatement.Assign "x+y" "2",
  AssignmentStatement.Assign "A-B" "-2",
  AssignmentStatement.Assign "A" "A*A"
]

/-- Theorem: Only one of the given statements is a valid assignment --/
theorem only_one_valid_assignment :
  (givenStatements.filter isValidAssignment).length = 1 :=
sorry

end only_one_valid_assignment_l714_71446


namespace parallelepiped_length_l714_71488

theorem parallelepiped_length : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n - 2 > 0) ∧ 
  (n - 4 > 0) ∧
  ((n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3) ∧
  (n = 18) := by
sorry

end parallelepiped_length_l714_71488


namespace sequence_equality_l714_71407

/-- Sequence definition -/
def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

/-- Main theorem -/
theorem sequence_equality (x : ℝ) :
  (a x 2)^2 = (a x 1) * (a x 3) →
  ∀ n ≥ 3, (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end sequence_equality_l714_71407


namespace problem_statement_l714_71423

-- Define the set X
def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

-- Define the set S
def S : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 ∈ X ∧ t.2.1 ∈ X ∧ t.2.2 ∈ X ∧
    ((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∨
     (t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∨
     (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.1 < t.2.2 ∧ t.2.2 < t.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1))}

-- Theorem statement
theorem problem_statement (x y z w : ℕ) 
  (h1 : (x, y, z) ∈ S) (h2 : (z, w, x) ∈ S) :
  (y, z, w) ∈ S ∧ (x, y, w) ∈ S := by
  sorry

end problem_statement_l714_71423


namespace system_solution_l714_71484

theorem system_solution (x y z : ℝ) :
  (x^3 = z/y - 2*y/z ∧ y^3 = x/z - 2*z/x ∧ z^3 = y/x - 2*x/y) →
  ((x = 1 ∧ y = 1 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end system_solution_l714_71484


namespace least_subtraction_for_divisibility_l714_71408

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 15 ∧ (899830 - x) % 16 = 0 ∧ ∀ y : ℕ, y < x → (899830 - y) % 16 ≠ 0 :=
by
  use 6
  sorry

end least_subtraction_for_divisibility_l714_71408


namespace range_of_a_l714_71480

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) ↔ -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l714_71480


namespace probability_theorem_l714_71486

/-- The probability of drawing one white ball and one black ball from an urn -/
def probability_one_white_one_black (a b : ℕ) : ℚ :=
  (2 * a * b : ℚ) / ((a + b) * (a + b - 1))

/-- Theorem stating the probability of drawing one white and one black ball -/
theorem probability_theorem (a b : ℕ) (h : a + b > 1) :
  probability_one_white_one_black a b =
    (2 * a * b : ℚ) / ((a + b) * (a + b - 1)) := by
  sorry

end probability_theorem_l714_71486


namespace solution_satisfies_inequalities_l714_71428

theorem solution_satisfies_inequalities :
  ∀ (x y z : ℝ),
  x = 3 ∧ y = 4 ∧ z = 5 →
  x < y ∧ y < z ∧ z < 6 ∧
  1 / (y - x) + 1 / (z - y) ≤ 2 ∧
  1 / (6 - z) + 2 ≤ x :=
by sorry

end solution_satisfies_inequalities_l714_71428
