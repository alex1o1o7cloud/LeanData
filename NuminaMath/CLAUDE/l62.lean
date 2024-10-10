import Mathlib

namespace solve_exponential_equation_l62_6219

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 2) = 27 ^ y ∧ y = 1 := by
  sorry

end solve_exponential_equation_l62_6219


namespace salt_calculation_l62_6208

/-- Calculates the amount of salt obtained from seawater evaporation -/
def salt_from_seawater (volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  volume * salt_percentage * 1000

/-- Proves that 2 liters of seawater with 20% salt content yields 400 ml of salt -/
theorem salt_calculation :
  salt_from_seawater 2 0.20 = 400 := by
  sorry

end salt_calculation_l62_6208


namespace factorial_ratio_l62_6283

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by
  sorry

end factorial_ratio_l62_6283


namespace huron_michigan_fishes_l62_6241

def total_fishes : ℕ := 97
def ontario_erie_fishes : ℕ := 23
def superior_fishes : ℕ := 44

theorem huron_michigan_fishes :
  total_fishes - (ontario_erie_fishes + superior_fishes) = 30 := by
  sorry

end huron_michigan_fishes_l62_6241


namespace extreme_points_imply_a_negative_l62_6204

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A function has two extreme points if its derivative has two distinct real roots -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem extreme_points_imply_a_negative (a : ℝ) :
  has_two_extreme_points a → a < 0 := by sorry

end extreme_points_imply_a_negative_l62_6204


namespace sum_first_seven_primes_l62_6236

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_seven_primes : 
  (first_seven_primes.sum = 58) ∧ (∀ p ∈ first_seven_primes, Nat.Prime p) ∧ (first_seven_primes.length = 7) := by
  sorry

end sum_first_seven_primes_l62_6236


namespace doctor_appointment_distance_l62_6266

/-- Represents the distances Tony needs to drive for his errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the total distance for all errands -/
def totalDistance (d : ErrandDistances) : ℕ :=
  d.groceries + d.haircut + d.doctor

theorem doctor_appointment_distance :
  ∀ (d : ErrandDistances),
    d.groceries = 10 →
    d.haircut = 15 →
    totalDistance d / 2 = 15 →
    d.doctor = 5 := by
  sorry

end doctor_appointment_distance_l62_6266


namespace square_equation_solution_l62_6243

theorem square_equation_solution : ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end square_equation_solution_l62_6243


namespace rectangle_polygon_perimeter_l62_6291

theorem rectangle_polygon_perimeter : 
  let n : ℕ := 20
  let rectangle_dimensions : ℕ → ℕ × ℕ := λ i => (i, i + 1)
  let perimeter : ℕ := 2 * (List.range (n + 1)).sum
  perimeter = 462 := by sorry

end rectangle_polygon_perimeter_l62_6291


namespace min_adventurers_l62_6222

/-- Represents a group of adventurers with their gem possessions -/
structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

/-- The conditions for the adventurer group -/
def validGroup (g : AdventurerGroup) : Prop :=
  (g.rubies.card = 4) ∧
  (g.emeralds.card = 10) ∧
  (g.sapphires.card = 6) ∧
  (g.diamonds.card = 14) ∧
  (∀ a ∈ g.rubies, (a ∈ g.emeralds ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.diamonds)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.sapphires))

/-- The theorem stating the minimum number of adventurers -/
theorem min_adventurers (g : AdventurerGroup) (h : validGroup g) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 18 := by
  sorry


end min_adventurers_l62_6222


namespace sqrt_49_times_sqrt_25_l62_6272

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 := by
  sorry

end sqrt_49_times_sqrt_25_l62_6272


namespace fourth_part_diminished_l62_6240

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 160) (h2 : (x / 5) + 4 = (x / 4) - y) : y = 4 := by
  sorry

end fourth_part_diminished_l62_6240


namespace exactly_one_line_with_two_rational_points_l62_6263

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A rational point is a point with rational coordinates -/
def RationalPoint (p : Point) : Prop :=
  ∃ (qx qy : ℚ), p.x = qx ∧ p.y = qy

/-- A line passes through a point if the point satisfies the line equation -/
def LinePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- A line contains at least two rational points -/
def LineContainsTwoRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ RationalPoint p1 ∧ RationalPoint p2 ∧
    LinePassesThrough l p1 ∧ LinePassesThrough l p2

/-- The main theorem -/
theorem exactly_one_line_with_two_rational_points
  (a : ℝ) (h_irrational : ¬ ∃ (q : ℚ), a = q) :
  ∃! (l : Line), LinePassesThrough l (Point.mk a 0) ∧ LineContainsTwoRationalPoints l :=
sorry

end exactly_one_line_with_two_rational_points_l62_6263


namespace smallest_number_l62_6275

theorem smallest_number (S : Set ℤ) (h : S = {-2, 0, -3, 1}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 :=
by sorry

end smallest_number_l62_6275


namespace complex_equation_solution_l62_6221

def i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 + a * i) / (1 + Real.sqrt 2 * i) = -Real.sqrt 2 * i →
  a = -Real.sqrt 2 := by
sorry

end complex_equation_solution_l62_6221


namespace farmer_tomatoes_l62_6271

theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (difference : ℕ) 
  (h1 : picked = 9)
  (h2 : difference = 8)
  (h3 : initial - picked = difference) :
  initial = 17 := by
  sorry

end farmer_tomatoes_l62_6271


namespace divisible_by_nine_l62_6229

theorem divisible_by_nine (A : Nat) : A < 10 → (7000 + 100 * A + 46) % 9 = 0 ↔ A = 1 := by
  sorry

end divisible_by_nine_l62_6229


namespace light_source_height_l62_6261

/-- The length of the cube's edge in centimeters -/
def cube_edge : ℝ := 2

/-- The area of the shadow cast by the cube, excluding the area beneath the cube, in square centimeters -/
def shadow_area : ℝ := 98

/-- The height of the light source above a top vertex of the cube in centimeters -/
def y : ℝ := sorry

/-- The theorem stating that the greatest integer not exceeding 1000y is 500 -/
theorem light_source_height : ⌊1000 * y⌋ = 500 := by sorry

end light_source_height_l62_6261


namespace product_of_five_integers_l62_6228

theorem product_of_five_integers (E F G H I : ℕ) 
  (sum_condition : E + F + G + H + I = 110)
  (equality_condition : (E : ℚ) / 2 = (F : ℚ) / 3 ∧ 
                        (F : ℚ) / 3 = G * 4 ∧ 
                        G * 4 = H * 2 ∧ 
                        H * 2 = I - 5) : 
  (E : ℚ) * F * G * H * I = 623400000 / 371293 := by
sorry

end product_of_five_integers_l62_6228


namespace fraction_division_equivalence_l62_6276

theorem fraction_division_equivalence : 5 / (8 / 13) = 65 / 8 := by
  sorry

end fraction_division_equivalence_l62_6276


namespace collinear_points_sum_l62_6268

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) : 
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by sorry

end collinear_points_sum_l62_6268


namespace quadratic_roots_sum_l62_6289

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 3 = 0 → 
  q^2 - 5*q + 3 = 0 → 
  p^2 + q^2 + p + q = 24 := by
  sorry

end quadratic_roots_sum_l62_6289


namespace inequality_proof_l62_6200

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) : 
  (x + y + z) / 3 ≥ (2*x*y*z)^(1/3) := by
  sorry

end inequality_proof_l62_6200


namespace alcohol_mixture_percentage_l62_6213

theorem alcohol_mixture_percentage :
  ∀ x : ℝ,
  (x / 100) * 8 + (12 / 100) * 2 = (22.4 / 100) * 10 →
  x = 25 := by
sorry

end alcohol_mixture_percentage_l62_6213


namespace ratio_of_numbers_l62_6254

theorem ratio_of_numbers (sum : ℚ) (bigger : ℚ) (h1 : sum = 143) (h2 : bigger = 104) :
  (sum - bigger) / bigger = 39 / 104 := by
  sorry

end ratio_of_numbers_l62_6254


namespace vector_equality_l62_6234

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AD : ℝ × ℝ := D - A
def BC : ℝ × ℝ := C - B
def BD : ℝ × ℝ := D - B
def DC : ℝ × ℝ := C - D

-- State the theorem
theorem vector_equality (h : BD = 3 • DC) : AD = (1/4) • AB + (3/4) • AC := by
  sorry

end vector_equality_l62_6234


namespace mbmt_equation_solution_l62_6292

theorem mbmt_equation_solution :
  ∃ (T H E M B : ℕ),
    T ≠ H ∧ T ≠ E ∧ T ≠ M ∧ T ≠ B ∧
    H ≠ E ∧ H ≠ M ∧ H ≠ B ∧
    E ≠ M ∧ E ≠ B ∧
    M ≠ B ∧
    T < 10 ∧ H < 10 ∧ E < 10 ∧ M < 10 ∧ B < 10 ∧
    B = 4 ∧ E = 2 ∧ T = 6 ∧
    (100 * T + 10 * H + E) + (1000 * M + 100 * B + 10 * M + T) = 2018 := by
  sorry

end mbmt_equation_solution_l62_6292


namespace no_real_solution_for_sqrt_equation_l62_6223

theorem no_real_solution_for_sqrt_equation :
  ¬∃ t : ℝ, Real.sqrt (49 - t^2) + 7 = 0 := by
sorry

end no_real_solution_for_sqrt_equation_l62_6223


namespace ninety_degrees_to_radians_l62_6280

theorem ninety_degrees_to_radians :
  (90 : ℝ) * π / 180 = π / 2 := by sorry

end ninety_degrees_to_radians_l62_6280


namespace z_as_percentage_of_x_l62_6207

theorem z_as_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.90 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.5 * x := by
sorry

end z_as_percentage_of_x_l62_6207


namespace largest_three_digit_square_base_7_l62_6231

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Predicate to check if a number has exactly 3 digits in base 7 -/
def has_three_digits_base_7 (n : ℕ) : Prop :=
  7^2 ≤ n ∧ n < 7^3

theorem largest_three_digit_square_base_7 :
  M = 18 ∧
  has_three_digits_base_7 (M^2) ∧
  ∀ n : ℕ, n > M → ¬has_three_digits_base_7 (n^2) :=
by sorry

end largest_three_digit_square_base_7_l62_6231


namespace lines_parallel_iff_l62_6202

/-- Two lines in R² defined by parametric equations -/
structure ParallelLines where
  k : ℝ
  line1 : ℝ → ℝ × ℝ := λ t => (1 + 5*t, 3 - 3*t)
  line2 : ℝ → ℝ × ℝ := λ s => (4 - 2*s, 1 + k*s)

/-- The lines are parallel (do not intersect) if and only if k = 6/5 -/
theorem lines_parallel_iff (pl : ParallelLines) : 
  (∀ t s, pl.line1 t ≠ pl.line2 s) ↔ pl.k = 6/5 := by
  sorry

end lines_parallel_iff_l62_6202


namespace acute_angle_trig_equation_l62_6295

theorem acute_angle_trig_equation (x : Real) (h1 : 0 < x) (h2 : x < π / 2) 
  (h3 : Real.sin x ^ 3 + Real.cos x ^ 3 = Real.sqrt 2 / 2) : x = π / 4 := by
  sorry

end acute_angle_trig_equation_l62_6295


namespace lemonade_stand_profit_l62_6288

/-- Calculate the profit from a lemonade stand --/
theorem lemonade_stand_profit
  (lemon_cost sugar_cost cup_cost : ℕ)
  (price_per_cup cups_sold : ℕ)
  (h1 : lemon_cost = 10)
  (h2 : sugar_cost = 5)
  (h3 : cup_cost = 3)
  (h4 : price_per_cup = 4)
  (h5 : cups_sold = 21) :
  (price_per_cup * cups_sold) - (lemon_cost + sugar_cost + cup_cost) = 66 := by
  sorry

end lemonade_stand_profit_l62_6288


namespace lunch_to_novel_ratio_l62_6279

theorem lunch_to_novel_ratio (initial_amount : ℕ) (novel_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 50)
  (h2 : novel_cost = 7)
  (h3 : remaining_amount = 29) :
  (initial_amount - novel_cost - remaining_amount) / novel_cost = 2 := by
  sorry

end lunch_to_novel_ratio_l62_6279


namespace card_selection_ways_l62_6293

theorem card_selection_ways (left_cards right_cards : ℕ) 
  (h1 : left_cards = 15) 
  (h2 : right_cards = 20) : 
  left_cards + right_cards = 35 := by
  sorry

end card_selection_ways_l62_6293


namespace rational_function_value_l62_6233

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x * (a * x + b)
def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

-- State the theorem
theorem rational_function_value
  (a b : ℝ)
  (h1 : p a b 1 / q 1 = -1)
  (h2 : a + b = 1/4) :
  p a b (-1) / q (-1) = (a - b) / 4 := by
sorry

end rational_function_value_l62_6233


namespace division_problem_l62_6225

theorem division_problem (d : ℕ) (h : 23 = d * 4 + 3) : d = 5 := by
  sorry

end division_problem_l62_6225


namespace largest_eight_digit_with_even_digits_l62_6203

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
by sorry

end largest_eight_digit_with_even_digits_l62_6203


namespace smallest_square_containing_circle_l62_6296

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end smallest_square_containing_circle_l62_6296


namespace writer_tea_and_hours_l62_6277

structure WriterData where
  sunday_hours : ℝ
  sunday_tea : ℝ
  wednesday_hours : ℝ
  thursday_tea : ℝ

def inverse_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

theorem writer_tea_and_hours (data : WriterData) :
  inverse_proportional data.sunday_hours data.sunday_tea (data.sunday_hours * data.sunday_tea) →
  inverse_proportional data.wednesday_hours (data.sunday_hours * data.sunday_tea / data.wednesday_hours) (data.sunday_hours * data.sunday_tea) ∧
  inverse_proportional (data.sunday_hours * data.sunday_tea / data.thursday_tea) data.thursday_tea (data.sunday_hours * data.sunday_tea) :=
by
  sorry

#check writer_tea_and_hours

end writer_tea_and_hours_l62_6277


namespace complex_number_location_l62_6285

theorem complex_number_location (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (m - 1) (3*m - 2)) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_location_l62_6285


namespace max_collisions_l62_6246

/-- Represents an ant walking on a line -/
structure Ant where
  position : ℝ
  speed : ℝ
  direction : Bool -- true for right, false for left

/-- The state of the system at any given time -/
structure AntSystem where
  n : ℕ
  ants : Fin n → Ant

/-- Predicate to check if the total number of collisions is finite -/
def HasFiniteCollisions (system : AntSystem) : Prop := sorry

/-- The number of collisions that have occurred in the system -/
def NumberOfCollisions (system : AntSystem) : ℕ := sorry

/-- Theorem stating the maximum number of collisions possible -/
theorem max_collisions (n : ℕ) (h : n > 0) :
  ∃ (system : AntSystem),
    system.n = n ∧
    HasFiniteCollisions system ∧
    ∀ (other_system : AntSystem),
      other_system.n = n →
      HasFiniteCollisions other_system →
      NumberOfCollisions other_system ≤ NumberOfCollisions system ∧
      NumberOfCollisions system = n * (n - 1) / 2 := by
  sorry

end max_collisions_l62_6246


namespace root_sum_equals_square_sum_l62_6252

theorem root_sum_equals_square_sum (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*a*(x₁-1) - 1 = 0 ∧ 
                x₂^2 - 2*a*(x₂-1) - 1 = 0 ∧ 
                x₁ + x₂ = x₁^2 + x₂^2) ↔ 
  (a = 1 ∨ a = 1/2) := by
sorry

end root_sum_equals_square_sum_l62_6252


namespace ellipse_minor_axis_length_l62_6216

/-- For an ellipse with given eccentricity and focal length, prove the length of its minor axis. -/
theorem ellipse_minor_axis_length 
  (e : ℝ) -- eccentricity
  (f : ℝ) -- focal length
  (h_e : e = 1/2)
  (h_f : f = 2) :
  ∃ (minor_axis : ℝ), minor_axis = 2 * Real.sqrt 3 :=
sorry

end ellipse_minor_axis_length_l62_6216


namespace sophia_saves_two_dimes_l62_6242

/-- Represents the number of pennies in a dime -/
def pennies_per_dime : ℕ := 10

/-- Represents the number of days Sophia saves -/
def saving_days : ℕ := 20

/-- Represents the number of pennies Sophia saves per day -/
def pennies_per_day : ℕ := 1

/-- Calculates the total number of pennies saved -/
def total_pennies : ℕ := saving_days * pennies_per_day

/-- Theorem: Sophia saves 2 dimes in total -/
theorem sophia_saves_two_dimes : 
  total_pennies / pennies_per_dime = 2 := by sorry

end sophia_saves_two_dimes_l62_6242


namespace equation_solution_l62_6238

theorem equation_solution : ∃ x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 10 := by
  sorry

end equation_solution_l62_6238


namespace simplify_expression_l62_6256

theorem simplify_expression : (5 + 7 + 3) / 3 - 2 / 3 - 1 = 10 / 3 := by
  sorry

end simplify_expression_l62_6256


namespace place_value_ratio_l62_6201

def number : ℚ := 56439.2071

theorem place_value_ratio : 
  (10000 : ℚ) * (number - number.floor) * 10 = (number.floor % 100000 - number.floor % 10000) / 10 := by
  sorry

end place_value_ratio_l62_6201


namespace union_determines_a_l62_6274

theorem union_determines_a (A B : Set ℝ) (a : ℝ) : 
  A = {1, 2} → 
  B = {a, a^2 + 1} → 
  A ∪ B = {0, 1, 2} → 
  a = 0 := by sorry

end union_determines_a_l62_6274


namespace identity_unique_solution_l62_6286

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The identity function is the unique solution to the functional equation -/
theorem identity_unique_solution :
  ∃! f : ℝ → ℝ, SatisfiesEquation f ∧ ∀ x : ℝ, f x = x :=
sorry

end identity_unique_solution_l62_6286


namespace sum_of_smallest_solutions_l62_6210

noncomputable def smallest_solutions (x : ℝ) : Prop :=
  x > 2017 ∧ 
  (Real.cos (9*x))^5 + (Real.cos x)^5 = 
    32 * (Real.cos (5*x))^5 * (Real.cos (4*x))^5 + 
    5 * (Real.cos (9*x))^2 * (Real.cos x)^2 * (Real.cos (9*x) + Real.cos x)

theorem sum_of_smallest_solutions :
  ∃ (x₁ x₂ : ℝ), 
    smallest_solutions x₁ ∧ 
    smallest_solutions x₂ ∧ 
    x₁ < x₂ ∧
    (∀ (y : ℝ), smallest_solutions y → y ≥ x₂ ∨ y = x₁) ∧
    x₁ + x₂ = 4064 := by
  sorry

end sum_of_smallest_solutions_l62_6210


namespace largest_number_l62_6206

-- Define the function to convert a number from any base to decimal (base 10)
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def binary : List Nat := [1, 1, 1, 1, 1]
def ternary : List Nat := [1, 2, 2, 1]
def quaternary : List Nat := [2, 1, 3]
def octal : List Nat := [6, 5]

-- State the theorem
theorem largest_number :
  to_decimal quaternary 4 = 54 ∧
  to_decimal quaternary 4 > to_decimal binary 2 ∧
  to_decimal quaternary 4 > to_decimal ternary 3 ∧
  to_decimal quaternary 4 > to_decimal octal 8 :=
sorry

end largest_number_l62_6206


namespace greatest_x_with_lcm_l62_6215

theorem greatest_x_with_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ, y = 180 ∧ Nat.lcm y (Nat.lcm 12 18) = 180 :=
by sorry

end greatest_x_with_lcm_l62_6215


namespace distinct_angles_in_twelve_sided_polygon_l62_6282

/-- A circle with an inscribed regular pentagon and heptagon -/
structure InscribedPolygons where
  circle : Set ℝ × ℝ  -- Representing a circle in 2D plane
  pentagon : Set (ℝ × ℝ)  -- Vertices of the pentagon
  heptagon : Set (ℝ × ℝ)  -- Vertices of the heptagon

/-- The resulting 12-sided polygon -/
def twelveSidedPolygon (ip : InscribedPolygons) : Set (ℝ × ℝ) :=
  ip.pentagon ∪ ip.heptagon

/-- Predicate to check if two polygons have no common vertices -/
def noCommonVertices (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  p1 ∩ p2 = ∅

/-- Predicate to check if two polygons have no common axes of symmetry -/
def noCommonAxesOfSymmetry (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Function to count distinct angle values in a polygon -/
def countDistinctAngles (p : Set (ℝ × ℝ)) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem distinct_angles_in_twelve_sided_polygon
  (ip : InscribedPolygons)
  (h1 : noCommonVertices ip.pentagon ip.heptagon)
  (h2 : noCommonAxesOfSymmetry ip.pentagon ip.heptagon)
  : countDistinctAngles (twelveSidedPolygon ip) = 6 :=
sorry

end distinct_angles_in_twelve_sided_polygon_l62_6282


namespace unique_n_for_consecutive_prime_products_l62_6224

def x (n : ℕ) : ℕ := 2 * n + 49

def is_product_of_two_distinct_primes_with_same_difference (m : ℕ) : Prop :=
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ ∃ (d : ℕ), m = p * q ∧ q - p = d

theorem unique_n_for_consecutive_prime_products : 
  ∃! (n : ℕ), n > 0 ∧ 
    is_product_of_two_distinct_primes_with_same_difference (x n) ∧
    is_product_of_two_distinct_primes_with_same_difference (x (n + 1)) ∧
    n = 7 :=
sorry

end unique_n_for_consecutive_prime_products_l62_6224


namespace quadrilateral_reconstruction_l62_6281

/-- A quadrilateral in 2D space -/
structure Quadrilateral (V : Type*) [AddCommGroup V] :=
  (P Q R S : V)

/-- Extended points of a quadrilateral -/
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] extends Quadrilateral V :=
  (P' Q' R' S' : V)

/-- Condition that P, Q, R, S are midpoints of PP', QQ', RR', SS' respectively -/
def is_midpoint_quadrilateral {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) : Prop :=
  quad.P = (1/2 : ℚ) • (quad.P + ext_quad.P') ∧
  quad.Q = (1/2 : ℚ) • (quad.Q + ext_quad.Q') ∧
  quad.R = (1/2 : ℚ) • (quad.R + ext_quad.R') ∧
  quad.S = (1/2 : ℚ) • (quad.S + ext_quad.S')

/-- Main theorem -/
theorem quadrilateral_reconstruction {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) 
  (h : is_midpoint_quadrilateral quad ext_quad) :
  quad.P = (1/15 : ℚ) • ext_quad.P' + (2/15 : ℚ) • ext_quad.Q' + 
           (4/15 : ℚ) • ext_quad.R' + (8/15 : ℚ) • ext_quad.S' := by
  sorry

end quadrilateral_reconstruction_l62_6281


namespace square_greater_than_self_for_x_greater_than_one_l62_6294

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end square_greater_than_self_for_x_greater_than_one_l62_6294


namespace maple_trees_after_planting_l62_6265

theorem maple_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) : 
  initial_trees = 2 → new_trees = 9 → initial_trees + new_trees = 11 := by
  sorry

end maple_trees_after_planting_l62_6265


namespace pentagonal_pyramid_base_areas_l62_6287

theorem pentagonal_pyramid_base_areas (total_surface_area lateral_surface_area : ℝ) :
  total_surface_area = 30 →
  lateral_surface_area = 25 →
  total_surface_area - lateral_surface_area = 5 := by
  sorry

end pentagonal_pyramid_base_areas_l62_6287


namespace right_triangle_hypotenuse_l62_6284

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 3 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 90 →
  hypotenuse^2 = short_leg^2 + long_leg^2 →
  hypotenuse = Real.sqrt 593 := by
sorry

end right_triangle_hypotenuse_l62_6284


namespace volume_of_one_gram_l62_6227

/-- Given a substance where 1 cubic meter has a mass of 100 kg, 
    prove that 1 gram of this substance has a volume of 10 cubic centimeters. -/
theorem volume_of_one_gram (substance_mass : ℝ) (substance_volume : ℝ) 
  (h1 : substance_mass = 100) 
  (h2 : substance_volume = 1) 
  (h3 : (1 : ℝ) = 1000 * (1 / 1000)) -- 1 kg = 1000 g
  (h4 : (1 : ℝ) = 1000000 * (1 / 1000000)) -- 1 m³ = 1,000,000 cm³
  : (1 / 1000) / (substance_mass / substance_volume) = 10 * (1 / 1000000) := by
  sorry

end volume_of_one_gram_l62_6227


namespace store_gross_profit_l62_6230

theorem store_gross_profit (purchase_price : ℝ) (initial_markup_percent : ℝ) (price_decrease_percent : ℝ) : 
  purchase_price = 210 →
  initial_markup_percent = 25 →
  price_decrease_percent = 20 →
  let original_selling_price := purchase_price / (1 - initial_markup_percent / 100)
  let discounted_price := original_selling_price * (1 - price_decrease_percent / 100)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 14 := by
sorry

end store_gross_profit_l62_6230


namespace sum_of_recorded_products_25_coins_l62_6258

/-- Represents the process of dividing coins into groups and recording products. -/
def divide_coins (n : ℕ) : ℕ := 
  (n * (n - 1)) / 2

/-- The theorem stating that the sum of recorded products when dividing 25 coins is 300. -/
theorem sum_of_recorded_products_25_coins : 
  divide_coins 25 = 300 := by sorry

end sum_of_recorded_products_25_coins_l62_6258


namespace amoeba_count_after_week_l62_6257

/-- The number of amoebas in the puddle on a given day -/
def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day = 1 then 1
  else 2 * amoeba_count (day - 1)

/-- The theorem stating that after 7 days, there are 64 amoebas in the puddle -/
theorem amoeba_count_after_week : amoeba_count 7 = 64 := by
  sorry

#eval amoeba_count 7  -- This should output 64

end amoeba_count_after_week_l62_6257


namespace solve_for_a_l62_6298

theorem solve_for_a : ∀ (x a : ℝ), (3 * x - 5 = x + a) ∧ (x = 2) → a = -1 := by
  sorry

end solve_for_a_l62_6298


namespace village_population_l62_6250

theorem village_population (P : ℝ) : 
  P > 0 → 
  (P * 0.9 * 0.8 = 3240) → 
  P = 4500 := by
sorry

end village_population_l62_6250


namespace train_length_calculation_l62_6211

/-- Given a train that crosses a platform and a signal pole, calculate its length. -/
theorem train_length_calculation
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : platform_length = 200)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 30)
  : ∃ (train_length : ℝ),
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time ∧
    train_length = 400 := by
  sorry

end train_length_calculation_l62_6211


namespace min_shortest_side_is_12_l62_6217

/-- Represents a triangle with integer side lengths and given altitudes -/
structure Triangle where
  -- Side lengths
  AB : ℕ
  BC : ℕ
  CA : ℕ
  -- Altitude lengths
  AD : ℕ
  BE : ℕ
  CF : ℕ
  -- Conditions
  altitude_AD : AD = 3
  altitude_BE : BE = 4
  altitude_CF : CF = 5
  -- Area equality conditions
  area_eq_1 : BC * AD = CA * BE
  area_eq_2 : CA * BE = AB * CF

/-- The minimum possible length of the shortest side of the triangle -/
def min_shortest_side (t : Triangle) : ℕ := min t.AB (min t.BC t.CA)

/-- Theorem stating the minimum possible length of the shortest side is 12 -/
theorem min_shortest_side_is_12 (t : Triangle) : min_shortest_side t = 12 := by
  sorry

#check min_shortest_side_is_12

end min_shortest_side_is_12_l62_6217


namespace sin_390_l62_6255

-- Define the period of the sine function
def sine_period : ℝ := 360

-- Define the periodicity property of sine
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the known value of sin 30°
axiom sin_30 : Real.sin 30 = 1 / 2

-- Theorem to prove
theorem sin_390 : Real.sin 390 = 1 / 2 := by
  sorry

end sin_390_l62_6255


namespace largest_even_digit_multiple_of_9_proof_l62_6253

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 8820

theorem largest_even_digit_multiple_of_9_proof :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 10000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 10000 ∧ m % 9 = 0)) :=
by
  sorry

end largest_even_digit_multiple_of_9_proof_l62_6253


namespace arithmetic_sequence_sum_l62_6297

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 5 + a 10 = 12 → 3 * a 7 + a 9 = 24 := by
  sorry

end arithmetic_sequence_sum_l62_6297


namespace complex_point_on_line_l62_6247

theorem complex_point_on_line (a : ℝ) : 
  (∃ (z : ℂ), z = (a - 1 : ℝ) + 3*I ∧ z.im = z.re + 2) → a = 2 :=
by sorry

end complex_point_on_line_l62_6247


namespace brandy_trail_mix_peanuts_l62_6237

/-- Represents the composition of trail mix -/
structure TrailMix where
  peanuts : ℝ
  chocolate_chips : ℝ
  raisins : ℝ

/-- The total weight of the trail mix -/
def total_weight (mix : TrailMix) : ℝ :=
  mix.peanuts + mix.chocolate_chips + mix.raisins

theorem brandy_trail_mix_peanuts :
  ∀ (mix : TrailMix),
    mix.chocolate_chips = 0.17 →
    mix.raisins = 0.08 →
    total_weight mix = 0.42 →
    mix.peanuts = 0.17 := by
  sorry

end brandy_trail_mix_peanuts_l62_6237


namespace sum_gcf_lcm_18_30_45_l62_6214

theorem sum_gcf_lcm_18_30_45 : 
  let A := Nat.gcd 18 (Nat.gcd 30 45)
  let B := Nat.lcm 18 (Nat.lcm 30 45)
  A + B = 93 := by
sorry

end sum_gcf_lcm_18_30_45_l62_6214


namespace fibonacci_divisibility_and_gcd_l62_6299

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_divisibility_and_gcd (m n : ℕ) :
  (m ∣ n → fib m ∣ fib n) ∧ (Nat.gcd (fib m) (fib n) = fib (Nat.gcd m n)) := by
  sorry

end fibonacci_divisibility_and_gcd_l62_6299


namespace cube_sum_from_conditions_l62_6259

theorem cube_sum_from_conditions (x y : ℝ) 
  (sum_condition : x + y = 5)
  (sum_squares_condition : x^2 + y^2 = 20) :
  x^3 + y^3 = 87.5 := by
  sorry

end cube_sum_from_conditions_l62_6259


namespace geometric_sequence_minimum_value_l62_6245

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_mean : Real.sqrt (a 4 * a 14) = 2 * Real.sqrt 2) :
  (2 * a 7 + a 11) ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 8 ∧ x * y = 8 :=
by sorry

end geometric_sequence_minimum_value_l62_6245


namespace increasing_quadratic_l62_6220

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

-- State the theorem
theorem increasing_quadratic :
  ∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > x₁ → f x₂ > f x₁ :=
by sorry

end increasing_quadratic_l62_6220


namespace divisors_of_sum_for_K_6_l62_6273

def number_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n.succ)).card

theorem divisors_of_sum_for_K_6 :
  let K : ℕ := 6
  let L : ℕ := number_of_divisors K
  number_of_divisors (K + 2 * L) = 4 := by sorry

end divisors_of_sum_for_K_6_l62_6273


namespace fourth_root_plus_cube_root_equation_solutions_l62_6267

theorem fourth_root_plus_cube_root_equation_solutions :
  ∀ x : ℝ, (((3 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/3) : ℝ) = 1) ↔ (x = 2 ∨ x = 3) := by
  sorry

end fourth_root_plus_cube_root_equation_solutions_l62_6267


namespace gift_wrapping_combinations_l62_6232

/-- The number of different types of wrapping paper -/
def wrapping_paper_types : ℕ := 10

/-- The number of different colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of different types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of different styles of gift tags -/
def gift_tag_styles : ℕ := 2

/-- The total number of different combinations for gift wrapping -/
def total_combinations : ℕ := wrapping_paper_types * ribbon_colors * gift_card_types * gift_tag_styles

theorem gift_wrapping_combinations : total_combinations = 400 := by
  sorry

end gift_wrapping_combinations_l62_6232


namespace expression_equals_one_l62_6262

theorem expression_equals_one : (-1)^2 - |(-3)| + (-5) / (-5/3) = 1 := by
  sorry

end expression_equals_one_l62_6262


namespace ball_hit_ground_time_l62_6212

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (y t : ℝ) : 
  y = -9.8 * t^2 + 5.6 * t + 10 →
  y = 0 →
  t = 131 / 98 := by sorry

end ball_hit_ground_time_l62_6212


namespace cannot_empty_both_piles_l62_6264

/-- Represents the state of the two piles of coins -/
structure CoinPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)

/-- Represents the allowed operations on the piles -/
inductive Operation
  | transferAndAdd : Operation
  | removeFour : Operation

/-- Applies an operation to the current state of the piles -/
def applyOperation (state : CoinPiles) (op : Operation) : CoinPiles :=
  match op with
  | Operation.transferAndAdd => 
      if state.pile1 > 0 then 
        CoinPiles.mk (state.pile1 - 1) (state.pile2 + 3)
      else 
        CoinPiles.mk (state.pile1 + 3) (state.pile2 - 1)
  | Operation.removeFour => 
      if state.pile1 ≥ 4 then 
        CoinPiles.mk (state.pile1 - 4) state.pile2
      else 
        CoinPiles.mk state.pile1 (state.pile2 - 4)

/-- The initial state of the piles -/
def initialState : CoinPiles := CoinPiles.mk 1 0

/-- Theorem stating that it's impossible to empty both piles -/
theorem cannot_empty_both_piles :
  ¬∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    finalState.pile1 = 0 ∧ finalState.pile2 = 0 :=
sorry

end cannot_empty_both_piles_l62_6264


namespace trivia_team_size_l62_6218

/-- The original number of members in a trivia team -/
def original_members (absent : ℕ) (points_per_member : ℕ) (total_points : ℕ) : ℕ :=
  (total_points / points_per_member) + absent

theorem trivia_team_size :
  original_members 3 2 12 = 9 := by
  sorry

end trivia_team_size_l62_6218


namespace solution_set_x_squared_minus_one_l62_6244

theorem solution_set_x_squared_minus_one (x : ℝ) : x^2 - 1 ≥ 0 ↔ x ≥ 1 ∨ x ≤ -1 := by
  sorry

end solution_set_x_squared_minus_one_l62_6244


namespace find_A_l62_6290

theorem find_A (A B : ℕ) : 
  A ≤ 9 →
  B ≤ 9 →
  100 ≤ A * 100 + 78 →
  A * 100 + 78 < 1000 →
  100 ≤ 200 + B →
  200 + B < 1000 →
  A * 100 + 78 - (200 + B) = 364 →
  A = 5 := by
sorry

end find_A_l62_6290


namespace fifteen_initial_points_theorem_l62_6235

/-- Represents the number of points after k densifications -/
def points_after_densification (initial_points : ℕ) (densifications : ℕ) : ℕ :=
  initial_points * 2^densifications - (2^densifications - 1)

/-- Theorem stating that 15 initial points results in 113 points after 3 densifications -/
theorem fifteen_initial_points_theorem :
  ∃ (n : ℕ), n > 0 ∧ points_after_densification n 3 = 113 → n = 15 :=
sorry

end fifteen_initial_points_theorem_l62_6235


namespace triangle_max_area_l62_6260

theorem triangle_max_area (a b c : ℝ) (h : 2 * a^2 + b^2 + c^2 = 4) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2)
  S ≤ Real.sqrt 5 / 5 := by
sorry

end triangle_max_area_l62_6260


namespace quadratic_equation_roots_l62_6249

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 3*x - m*x + m - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 3*x₁ - x₁*x₂ + 3*x₂ = 12 →
  x₁ = 0 ∧ x₂ = 4 :=
by sorry

end quadratic_equation_roots_l62_6249


namespace f_2008_l62_6248

-- Define a real-valued function with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the condition f(9) = 18
axiom f_9 : f 9 = 18

-- Define the inverse relationship for f(x+1)
axiom inverse_shift : Function.LeftInverse (fun x => f⁻¹ (x + 1)) (fun x => f (x + 1))

-- State the theorem
theorem f_2008 : f 2008 = -1981 := by sorry

end f_2008_l62_6248


namespace subsets_exist_l62_6278

/-- A type representing a set of subsets of positive integers -/
def SubsetCollection := Finset (Set ℕ+)

/-- A function that constructs the required subsets -/
def constructSubsets (n : ℕ) : SubsetCollection :=
  sorry

/-- Predicate to check if subsets are pairwise nonintersecting -/
def pairwiseNonintersecting (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if all subsets are nonempty -/
def allNonempty (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if each positive integer can be uniquely expressed
    as a sum of at most n integers from different subsets -/
def uniqueRepresentation (s : SubsetCollection) (n : ℕ) : Prop :=
  sorry

/-- The main theorem stating the existence of the required subsets -/
theorem subsets_exist (n : ℕ) (h : n ≥ 2) :
  ∃ s : SubsetCollection,
    s.card = n ∧
    pairwiseNonintersecting s ∧
    allNonempty s ∧
    uniqueRepresentation s n :=
  sorry

end subsets_exist_l62_6278


namespace max_value_x_sqrt_1_minus_x_squared_l62_6269

theorem max_value_x_sqrt_1_minus_x_squared :
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) := by
  sorry

end max_value_x_sqrt_1_minus_x_squared_l62_6269


namespace bacterial_growth_l62_6270

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total duration of the culturing process in minutes -/
def total_time : ℕ := 3 * 60

/-- The number of divisions that occur during the culturing process -/
def num_divisions : ℕ := total_time / division_interval

/-- The final number of bacteria after the culturing process -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacterial_growth :
  final_bacteria_count = 512 :=
sorry

end bacterial_growth_l62_6270


namespace max_tickets_jane_can_buy_l62_6226

theorem max_tickets_jane_can_buy (ticket_cost : ℕ) (service_charge : ℕ) (budget : ℕ) :
  ticket_cost = 15 →
  service_charge = 10 →
  budget = 120 →
  ∃ (n : ℕ), n = 7 ∧ 
    n * ticket_cost + service_charge ≤ budget ∧
    ∀ (m : ℕ), m * ticket_cost + service_charge ≤ budget → m ≤ n :=
by sorry

end max_tickets_jane_can_buy_l62_6226


namespace beaver_count_l62_6209

theorem beaver_count (initial_beavers : Float) (additional_beavers : Float) : 
  initial_beavers = 2.0 → additional_beavers = 1.0 → initial_beavers + additional_beavers = 3.0 := by
  sorry

end beaver_count_l62_6209


namespace multiple_of_six_last_digit_l62_6205

theorem multiple_of_six_last_digit (n : ℕ) : 
  n ≥ 85670 ∧ n < 85680 ∧ n % 6 = 0 → n = 85676 := by sorry

end multiple_of_six_last_digit_l62_6205


namespace square_ratio_side_length_sum_l62_6251

theorem square_ratio_side_length_sum (s1 s2 : ℝ) (h : s1^2 / s2^2 = 32 / 63) :
  ∃ (a b c : ℕ), (s1 / s2 = a * Real.sqrt b / c) ∧ (a + b + c = 39) := by
  sorry

end square_ratio_side_length_sum_l62_6251


namespace g_of_3_equals_6_l62_6239

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

theorem g_of_3_equals_6 : g 3 = 6 := by
  sorry

end g_of_3_equals_6_l62_6239
