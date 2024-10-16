import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_squared_l2823_282346

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_squared : (2 * i / (1 + i)) ^ 2 = 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l2823_282346


namespace NUMINAMATH_CALUDE_min_employees_for_agency_l2823_282304

/-- Represents the number of employees needed for different pollution monitoring tasks -/
structure EmployeeRequirements where
  water : ℕ
  air : ℕ
  both : ℕ
  soil : ℕ

/-- Calculates the minimum number of employees needed given the requirements -/
def minEmployees (req : EmployeeRequirements) : ℕ :=
  req.water + req.air - req.both

/-- Theorem stating that given the specific requirements, 160 employees are needed -/
theorem min_employees_for_agency (req : EmployeeRequirements) 
  (h_water : req.water = 120)
  (h_air : req.air = 105)
  (h_both : req.both = 65)
  (h_soil : req.soil = 40)
  : minEmployees req = 160 := by
  sorry

#eval minEmployees { water := 120, air := 105, both := 65, soil := 40 }

end NUMINAMATH_CALUDE_min_employees_for_agency_l2823_282304


namespace NUMINAMATH_CALUDE_interest_rate_equivalence_l2823_282312

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_equivalence : ∃ (rate : ℝ),
  simple_interest 100 0.05 8 = simple_interest 200 rate 2 ∧ rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equivalence_l2823_282312


namespace NUMINAMATH_CALUDE_root_value_l2823_282397

-- Define the polynomials and their roots
def f (x : ℝ) := x^3 + 5*x^2 + 2*x - 8
def g (x p q r : ℝ) := x^3 + p*x^2 + q*x + r

-- Define the roots
variable (a b c : ℝ)

-- State the conditions
axiom root_f : f a = 0 ∧ f b = 0 ∧ f c = 0
axiom root_g : ∃ p q r, g (2*a + b) p q r = 0 ∧ g (2*b + c) p q r = 0 ∧ g (2*c + a) p q r = 0

-- State the theorem to be proved
theorem root_value : ∃ p q, g (2*a + b) p q 18 = 0 ∧ g (2*b + c) p q 18 = 0 ∧ g (2*c + a) p q 18 = 0 :=
sorry

end NUMINAMATH_CALUDE_root_value_l2823_282397


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2823_282317

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 5 / 4)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 4)
  (hsum : w + x + y + z = 60) :
  w / y = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2823_282317


namespace NUMINAMATH_CALUDE_arithmetic_mean_sequence_l2823_282381

theorem arithmetic_mean_sequence (a b c d e f g : ℝ) 
  (hb : b = (a + c) / 2)
  (hc : c = (b + d) / 2)
  (hd : d = (c + e) / 2)
  (he : e = (d + f) / 2)
  (hf : f = (e + g) / 2) :
  d = (a + g) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sequence_l2823_282381


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2823_282330

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The equation (x - 3)² + (y + 1)² = 25 represents the circle
    with center (3, -1) passing through the point (7, -4) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : Point := ⟨3, -1⟩
  let point : Point := ⟨7, -4⟩
  (x - center.x)^2 + (y - center.y)^2 = squaredDistance center point := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2823_282330


namespace NUMINAMATH_CALUDE_f_of_3_equals_13_l2823_282333

theorem f_of_3_equals_13 (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = 2 * x + 5) : f 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_13_l2823_282333


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l2823_282307

/-- Proves that the time taken to remove wallpaper from the first wall is 2 hours -/
theorem wallpaper_removal_time (total_walls : ℕ) (walls_removed : ℕ) (remaining_time : ℕ) :
  total_walls = 8 →
  walls_removed = 1 →
  remaining_time = 14 →
  remaining_time / (total_walls - walls_removed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_removal_time_l2823_282307


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l2823_282315

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Probability mass function for a binomial random variable -/
def pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p ^ k * (1 - X.p) ^ (X.n - k)

theorem binomial_probability_theorem (X : BinomialRV) 
  (h_exp : expected_value X = 2)
  (h_var : variance X = 4/3) :
  pmf X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l2823_282315


namespace NUMINAMATH_CALUDE_power_product_equals_78125_l2823_282385

theorem power_product_equals_78125 (a : ℕ) (h : a = 5) : a^3 * a^4 = 78125 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_78125_l2823_282385


namespace NUMINAMATH_CALUDE_radical_simplification_l2823_282331

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (75 * x) * Real.sqrt (2 * x) * Real.sqrt (14 * x) = 10 * x * Real.sqrt (21 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2823_282331


namespace NUMINAMATH_CALUDE_common_difference_is_half_l2823_282383

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 1 + a 6 + a 11 = 6
  fourth_term : a 4 = 1

/-- The common difference of an arithmetic sequence is 1/2 given the conditions -/
theorem common_difference_is_half (seq : ArithmeticSequence) : 
  ∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_half_l2823_282383


namespace NUMINAMATH_CALUDE_pentagon_perimeter_even_l2823_282389

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a pentagon as a list of 5 points
def Pentagon : Type := List IntPoint

-- Function to calculate the distance between two points
def distance (p1 p2 : IntPoint) : Int :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

-- Function to check if a pentagon has integer side lengths
def hasIntegerSideLengths (p : Pentagon) : Prop :=
  match p with
  | [a, b, c, d, e] => 
    ∃ (l1 l2 l3 l4 l5 : Int),
      distance a b = l1 ^ 2 ∧
      distance b c = l2 ^ 2 ∧
      distance c d = l3 ^ 2 ∧
      distance d e = l4 ^ 2 ∧
      distance e a = l5 ^ 2
  | _ => False

-- Function to calculate the perimeter of a pentagon
def perimeter (p : Pentagon) : Int :=
  match p with
  | [a, b, c, d, e] => 
    Int.sqrt (distance a b) +
    Int.sqrt (distance b c) +
    Int.sqrt (distance c d) +
    Int.sqrt (distance d e) +
    Int.sqrt (distance e a)
  | _ => 0

-- Theorem statement
theorem pentagon_perimeter_even (p : Pentagon) 
  (h1 : p.length = 5)
  (h2 : hasIntegerSideLengths p) :
  Even (perimeter p) := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_even_l2823_282389


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2823_282398

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -2 →
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x - 1)) = 1 / (x + 2) ∧
  (1 - 1 / (-1 - 1)) / (((-1)^2 - 4) / (-1 - 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2823_282398


namespace NUMINAMATH_CALUDE_lee_cookies_l2823_282344

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any number of cups of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (18 / 2) * cups

/-- Theorem stating that Lee can make 45 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l2823_282344


namespace NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_line_l2823_282396

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) 
  (h_point : point.x = 2 ∧ point.y = 1) 
  (h_given_line : given_line.a = 2 ∧ given_line.b = -1 ∧ given_line.c = 2) :
  ∃ (result_line : Line), 
    result_line.a = 2 ∧ 
    result_line.b = -1 ∧ 
    result_line.c = -3 ∧
    parallel result_line given_line ∧
    on_line point result_line :=
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_parallel_to_line_l2823_282396


namespace NUMINAMATH_CALUDE_quadratic_solution_l2823_282302

theorem quadratic_solution (b : ℤ) : 
  ((-5 : ℤ)^2 + b * (-5) - 35 = 0) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2823_282302


namespace NUMINAMATH_CALUDE_percent_decrease_l2823_282338

theorem percent_decrease (original_price sale_price : ℝ) (h1 : original_price = 100) (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l2823_282338


namespace NUMINAMATH_CALUDE_yearly_reading_pages_l2823_282365

/-- The number of pages read in a year, given the number of novels read per month,
    pages per novel, and months in a year. -/
def pages_read_in_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_in_year : ℕ) : ℕ :=
  novels_per_month * pages_per_novel * months_in_year

/-- Theorem stating that reading 4 novels of 200 pages each month for 12 months
    results in reading 9600 pages in a year. -/
theorem yearly_reading_pages :
  pages_read_in_year 4 200 12 = 9600 := by
  sorry

end NUMINAMATH_CALUDE_yearly_reading_pages_l2823_282365


namespace NUMINAMATH_CALUDE_root_equation_result_l2823_282341

/-- Given two constants c and d, if the equation ((x+c)(x+d)(x-15))/((x-4)^2) = 0 has exactly 3 distinct roots,
    and the equation ((x+2c)(x-4)(x-9))/((x+d)(x-15)) = 0 has exactly 1 distinct root,
    then 100c + d = -391 -/
theorem root_equation_result (c d : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    ∀ x, (x + c) * (x + d) * (x - 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)
  (h2 : ∃! (r : ℝ), ∀ x, (x + 2*c) * (x - 4) * (x - 9) = 0 ↔ x = r) :
  100 * c + d = -391 :=
sorry

end NUMINAMATH_CALUDE_root_equation_result_l2823_282341


namespace NUMINAMATH_CALUDE_odd_function_property_l2823_282316

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (hf : IsOdd f) :
  g f 1 = 1 → g f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2823_282316


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2823_282313

theorem complex_multiplication_result : 
  (2 + 2 * Complex.I) * (1 - 2 * Complex.I) = 6 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2823_282313


namespace NUMINAMATH_CALUDE_range_of_a_l2823_282373

open Set

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 - 3 * a * x₀ + 9 < 0) ↔ 
  a ∈ (Iio (-2 * Real.sqrt 2) ∪ Ioi (2 * Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2823_282373


namespace NUMINAMATH_CALUDE_no_single_non_divisible_l2823_282311

/-- Represents a 5x5 table of non-zero digits -/
def Table := Fin 5 → Fin 5 → Fin 9

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0

/-- Sums the digits in a row or column -/
def sumDigits (digits : Fin 5 → Fin 9) : ℕ :=
  (Finset.univ.sum fun i => (digits i).val) + 5

/-- Theorem stating the impossibility of having exactly one number not divisible by 3 -/
theorem no_single_non_divisible (t : Table) : 
  ¬ (∃! n : Fin 10, ¬ isDivisibleBy3 (sumDigits (fun i => 
    if n.val < 5 then t i n.val else t n.val (i - 5)))) := by
  sorry

end NUMINAMATH_CALUDE_no_single_non_divisible_l2823_282311


namespace NUMINAMATH_CALUDE_complex_cube_l2823_282380

theorem complex_cube (i : ℂ) : i^2 = -1 → (2 - 3*i)^3 = -46 - 9*i := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_l2823_282380


namespace NUMINAMATH_CALUDE_firewood_collection_l2823_282308

theorem firewood_collection (total kimberley houston : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : houston = 12) :
  ∃ ela : ℕ, total = kimberley + houston + ela ∧ ela = 13 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l2823_282308


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2823_282347

theorem ping_pong_rackets_sold (total_amount : ℝ) (average_price : ℝ) (h1 : total_amount = 490) (h2 : average_price = 9.8) :
  total_amount / average_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2823_282347


namespace NUMINAMATH_CALUDE_scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2823_282355

-- Problem 1
theorem scientific_notation_pm25 : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.5 ∧ n = -6 :=
sorry

-- Problem 2
theorem express_y_in_terms_of_x (x y : ℝ) :
  2 * x - 5 * y = 5 → y = 0.4 * x - 1 :=
sorry

-- Problem 3
theorem power_evaluation (x y : ℝ) :
  x + 2 * y - 4 = 0 → (2 : ℝ) ^ (2 * y) * (2 : ℝ) ^ (x - 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_pm25_express_y_in_terms_of_x_power_evaluation_l2823_282355


namespace NUMINAMATH_CALUDE_arccos_cos_ten_l2823_282324

theorem arccos_cos_ten :
  Real.arccos (Real.cos 10) = 10 - 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_l2823_282324


namespace NUMINAMATH_CALUDE_min_votes_to_win_l2823_282309

-- Define the voting structure
def total_voters : ℕ := 135
def num_districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Define winning conditions
def win_precinct (votes : ℕ) : Prop := votes > voters_per_precinct / 2
def win_district (precincts_won : ℕ) : Prop := precincts_won > precincts_per_district / 2
def win_final (districts_won : ℕ) : Prop := districts_won > num_districts / 2

-- Theorem statement
theorem min_votes_to_win (min_votes : ℕ) : 
  (∃ (districts_won precincts_won votes_per_precinct : ℕ),
    win_final districts_won ∧
    win_district precincts_won ∧
    win_precinct votes_per_precinct ∧
    min_votes = districts_won * precincts_won * votes_per_precinct) →
  min_votes = 30 := by sorry

end NUMINAMATH_CALUDE_min_votes_to_win_l2823_282309


namespace NUMINAMATH_CALUDE_davids_original_portion_l2823_282384

/-- Given a total initial amount of $1500 shared among David, Elisa, and Frank,
    where the total final amount is $2700, Elisa and Frank both triple their initial investments,
    and David loses $200, prove that David's original portion is $800. -/
theorem davids_original_portion (d e f : ℝ) : 
  d + e + f = 1500 →
  d - 200 + 3 * e + 3 * f = 2700 →
  d = 800 := by
sorry

end NUMINAMATH_CALUDE_davids_original_portion_l2823_282384


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2823_282376

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2823_282376


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_and_48_l2823_282374

theorem smallest_divisible_by_15_and_48 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 48 ∣ n → n ≥ 240 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_and_48_l2823_282374


namespace NUMINAMATH_CALUDE_travel_distance_proof_l2823_282322

def speed_limit : ℝ := 60
def speed_above_limit : ℝ := 15
def travel_time : ℝ := 2

theorem travel_distance_proof :
  let actual_speed := speed_limit + speed_above_limit
  actual_speed * travel_time = 150 := by sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l2823_282322


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2823_282351

theorem pure_imaginary_fraction (b : ℝ) : 
  (∃ (y : ℝ), (b + Complex.I) / (2 + Complex.I) = Complex.I * y) → b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2823_282351


namespace NUMINAMATH_CALUDE_contrapositive_example_l2823_282300

theorem contrapositive_example : 
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l2823_282300


namespace NUMINAMATH_CALUDE_alphabet_composition_l2823_282345

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  only_line : ℕ
  only_dot : ℕ
  both : ℕ
  all_accounted : total = only_line + only_dot + both

/-- Theorem: In an alphabet of 40 letters, if 24 contain only a straight line
    and 5 contain only a dot, then 11 must contain both -/
theorem alphabet_composition (a : Alphabet)
  (h1 : a.total = 40)
  (h2 : a.only_line = 24)
  (h3 : a.only_dot = 5) :
  a.both = 11 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_composition_l2823_282345


namespace NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l2823_282382

theorem power_of_seven_mod_twelve : 7^203 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l2823_282382


namespace NUMINAMATH_CALUDE_divisibility_property_l2823_282328

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2823_282328


namespace NUMINAMATH_CALUDE_cristinas_pace_cristinas_pace_is_five_l2823_282372

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let total_distance := head_start + nickys_pace * catch_up_time
  total_distance / catch_up_time

/-- Prove that Cristina's pace is 5 meters per second -/
theorem cristinas_pace_is_five :
  cristinas_pace 48 3 24 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_pace_cristinas_pace_is_five_l2823_282372


namespace NUMINAMATH_CALUDE_parabola_equation_holds_l2823_282386

/-- A parabola with vertex at (2, 9) intersecting the x-axis to form a segment of length 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  intersection_length : ℝ
  vertex_condition : vertex = (2, 9)
  length_condition : intersection_length = 6

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  y = -(x - 2)^2 + 9

/-- Theorem stating that the given parabola satisfies the equation -/
theorem parabola_equation_holds (p : Parabola) :
  ∀ x y : ℝ, parabola_equation p x y ↔ 
    ∃ a : ℝ, y = a * (x - p.vertex.1)^2 + p.vertex.2 ∧
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      a * (x₁ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      a * (x₂ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      |x₁ - x₂| = p.intersection_length :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_holds_l2823_282386


namespace NUMINAMATH_CALUDE_polar_curve_arc_length_l2823_282320

noncomputable def arcLength (ρ : Real → Real) (a b : Real) : Real :=
  ∫ x in a..b, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem polar_curve_arc_length :
  let ρ : Real → Real := λ φ ↦ 8 * Real.cos φ
  arcLength ρ 0 (Real.pi / 4) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_arc_length_l2823_282320


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2823_282354

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem multiplication_puzzle : ∃! (a b : ℕ), 
  (1000 ≤ a * b) ∧ (a * b < 10000) ∧  -- 4-digit product
  (digit_sum a = digit_sum b) ∧       -- same digit sum
  (a * b % 10 = a % 10) ∧             -- ones digit condition
  ((a * b / 10) % 10 = 2) ∧           -- tens digit condition
  a = 2231 ∧ b = 26 := by
sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2823_282354


namespace NUMINAMATH_CALUDE_defective_units_percentage_l2823_282352

/-- The percentage of defective units that are shipped for sale -/
def shipped_defective_percent : ℝ := 5

/-- The percentage of total units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.5

/-- The percentage of defective units produced -/
def defective_percent : ℝ := 10

theorem defective_units_percentage :
  shipped_defective_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l2823_282352


namespace NUMINAMATH_CALUDE_bandi_has_winning_strategy_l2823_282357

/-- Represents a player in the game -/
inductive Player
| Andi
| Bandi

/-- Represents a digit in the binary number -/
inductive Digit
| Zero
| One

/-- Represents a strategy for a player -/
def Strategy := List Digit → Digit

/-- The game state -/
structure GameState :=
(sequence : List Digit)
(turn : Player)
(moves_left : Nat)

/-- The result of the game -/
inductive GameResult
| AndiWin
| BandiWin

/-- Converts a list of digits to a natural number -/
def binary_to_nat (digits : List Digit) : Nat :=
  sorry

/-- Checks if a number is the sum of two squares -/
def is_sum_of_squares (n : Nat) : Prop :=
  sorry

/-- Plays the game given strategies for both players -/
def play_game (andi_strategy : Strategy) (bandi_strategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that Bandi has a winning strategy -/
theorem bandi_has_winning_strategy :
  ∃ (bandi_strategy : Strategy),
    ∀ (andi_strategy : Strategy),
      play_game andi_strategy bandi_strategy = GameResult.BandiWin :=
sorry

end NUMINAMATH_CALUDE_bandi_has_winning_strategy_l2823_282357


namespace NUMINAMATH_CALUDE_no_obtuse_equilateral_triangle_l2823_282332

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∧ t.angle2 = t.angle3

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse equilateral triangle cannot exist
theorem no_obtuse_equilateral_triangle :
  ¬ ∃ (t : Triangle), t.isEquilateral ∧ t.isObtuse ∧ t.angle1 + t.angle2 + t.angle3 = 180 :=
by sorry

end NUMINAMATH_CALUDE_no_obtuse_equilateral_triangle_l2823_282332


namespace NUMINAMATH_CALUDE_prime_divides_product_l2823_282370

theorem prime_divides_product (p a b : ℕ) : 
  Prime p → (p ∣ (a * b)) → (p ∣ a) ∨ (p ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_product_l2823_282370


namespace NUMINAMATH_CALUDE_fish_weight_l2823_282358

/-- Given a barrel of fish with the following properties:
  1. The total weight of the barrel and fish is 54 kg.
  2. After removing half of the fish, the total weight is 29 kg.
  This theorem proves that the initial weight of the fish alone is 50 kg. -/
theorem fish_weight (barrel_weight : ℝ) (fish_weight : ℝ) 
  (h1 : barrel_weight + fish_weight = 54)
  (h2 : barrel_weight + fish_weight / 2 = 29) :
  fish_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_fish_weight_l2823_282358


namespace NUMINAMATH_CALUDE_sams_new_books_l2823_282388

theorem sams_new_books : 
  ∀ (adventure_books mystery_books used_books : ℕ),
    adventure_books = 24 →
    mystery_books = 37 →
    used_books = 18 →
    adventure_books + mystery_books - used_books = 43 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_books_l2823_282388


namespace NUMINAMATH_CALUDE_total_students_l2823_282301

theorem total_students (A B C : ℕ) : 
  B = A - 8 →
  C = 5 * B →
  B = 25 →
  A + B + C = 183 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2823_282301


namespace NUMINAMATH_CALUDE_leftmost_digit_89_base_5_l2823_282393

def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem leftmost_digit_89_base_5 :
  (to_base_5 89).reverse.head? = some 3 := by sorry

end NUMINAMATH_CALUDE_leftmost_digit_89_base_5_l2823_282393


namespace NUMINAMATH_CALUDE_total_barks_after_duration_l2823_282392

/-- Represents the number of barks per minute for a single dog -/
def barks_per_minute : ℕ := 30

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the duration in minutes -/
def duration : ℕ := 10

/-- Theorem stating that the total number of barks after the given duration is 600 -/
theorem total_barks_after_duration :
  num_dogs * barks_per_minute * duration = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_barks_after_duration_l2823_282392


namespace NUMINAMATH_CALUDE_tangent_circle_problem_l2823_282306

theorem tangent_circle_problem (center_to_intersection : ℚ) (radius : ℚ) (center_to_line : ℚ) (x : ℚ) :
  center_to_intersection = 3/8 →
  radius = 3/16 →
  center_to_line = 1/2 →
  x = center_to_intersection + radius - center_to_line →
  x = 1/16 := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_problem_l2823_282306


namespace NUMINAMATH_CALUDE_compatible_pairs_theorem_l2823_282318

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def is_compatible (a b : ℕ) : Prop :=
  (a = sum_of_digits b ∧ b = product_of_digits a) ∨
  (b = sum_of_digits a ∧ a = product_of_digits b)

def compatible_pairs_within (n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ is_compatible p.1 p.2}

def compatible_pairs_within_one_greater (n m : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ (p.1 > m ∨ p.2 > m) ∧ is_compatible p.1 p.2}

theorem compatible_pairs_theorem :
  compatible_pairs_within 100 = {(9, 11), (12, 36)} ∧
  compatible_pairs_within_one_greater 1000 99 = {(135, 19), (144, 19)} := by sorry

end NUMINAMATH_CALUDE_compatible_pairs_theorem_l2823_282318


namespace NUMINAMATH_CALUDE_juan_milk_needed_l2823_282326

/-- The number of cookies that can be baked with one half-gallon of milk -/
def cookies_per_half_gallon : ℕ := 48

/-- The number of cookies Juan wants to bake -/
def cookies_to_bake : ℕ := 40

/-- The amount of milk needed for baking, in half-gallons -/
def milk_needed : ℕ := 1

theorem juan_milk_needed :
  cookies_to_bake ≤ cookies_per_half_gallon → milk_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_juan_milk_needed_l2823_282326


namespace NUMINAMATH_CALUDE_total_amount_spent_l2823_282395

/-- Calculates the total amount spent on pencils, cucumbers, and notebooks given specific conditions --/
theorem total_amount_spent 
  (initial_cost : ℝ)
  (pencil_discount : ℝ)
  (notebook_discount : ℝ)
  (pencil_tax : ℝ)
  (cucumber_tax : ℝ)
  (cucumber_count : ℕ)
  (notebook_count : ℕ)
  (h1 : initial_cost = 20)
  (h2 : pencil_discount = 0.2)
  (h3 : notebook_discount = 0.3)
  (h4 : pencil_tax = 0.05)
  (h5 : cucumber_tax = 0.1)
  (h6 : cucumber_count = 100)
  (h7 : notebook_count = 25) :
  (cucumber_count / 2 : ℝ) * (initial_cost * (1 - pencil_discount) * (1 + pencil_tax)) +
  (cucumber_count : ℝ) * (initial_cost * (1 + cucumber_tax)) +
  (notebook_count : ℝ) * (initial_cost * (1 - notebook_discount)) = 3390 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_spent_l2823_282395


namespace NUMINAMATH_CALUDE_inequality_proof_l2823_282363

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2823_282363


namespace NUMINAMATH_CALUDE_vector_properties_l2823_282377

/-- Given vectors a and b, prove the sine of their angle and the value of m for perpendicularity. -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (3, -4)) (h2 : b = (1, 2)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  ∃ (m : ℝ),
    Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
    (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2) = 0 ∧
    m = 0 :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2823_282377


namespace NUMINAMATH_CALUDE_least_stamps_stamps_23_robert_stamps_l2823_282323

theorem least_stamps (n : ℕ) : n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 → n ≥ 23 := by
  sorry

theorem stamps_23 : 23 % 7 = 2 ∧ 23 % 4 = 3 := by
  sorry

theorem robert_stamps : ∃ n : ℕ, n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 ∧ 
  ∀ m : ℕ, (m > 0 ∧ m % 7 = 2 ∧ m % 4 = 3) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_stamps_stamps_23_robert_stamps_l2823_282323


namespace NUMINAMATH_CALUDE_lindas_furniture_fraction_l2823_282349

theorem lindas_furniture_fraction (savings : ℚ) (tv_cost : ℚ) 
  (h1 : savings = 920)
  (h2 : tv_cost = 230) :
  (savings - tv_cost) / savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_fraction_l2823_282349


namespace NUMINAMATH_CALUDE_negation_equivalence_l2823_282343

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (|x| + |x - 1| < 2)) ↔ (∀ x : ℝ, |x| + |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2823_282343


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2823_282367

theorem complex_expression_simplification :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2/3)) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2823_282367


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2823_282342

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2823_282342


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2823_282359

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : perpendicular n β)
  (h3 : perpendicular n α) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2823_282359


namespace NUMINAMATH_CALUDE_area_ratio_l2823_282319

-- Define the side lengths of the squares
def side_length_A (x : ℝ) : ℝ := x
def side_length_B (x : ℝ) : ℝ := 3 * x
def side_length_C (x : ℝ) : ℝ := 2 * x

-- Define the areas of the squares
def area_A (x : ℝ) : ℝ := (side_length_A x) ^ 2
def area_B (x : ℝ) : ℝ := (side_length_B x) ^ 2
def area_C (x : ℝ) : ℝ := (side_length_C x) ^ 2

-- Theorem stating the ratio of areas
theorem area_ratio (x : ℝ) (h : x > 0) : 
  area_A x / (area_B x + area_C x) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_l2823_282319


namespace NUMINAMATH_CALUDE_consecutive_majors_probability_l2823_282339

/-- Represents the number of people around the table -/
def total_people : ℕ := 12

/-- Represents the number of math majors -/
def math_majors : ℕ := 5

/-- Represents the number of physics majors -/
def physics_majors : ℕ := 4

/-- Represents the number of biology majors -/
def biology_majors : ℕ := 3

/-- Represents the probability of the desired seating arrangement -/
def seating_probability : ℚ := 1 / 5775

theorem consecutive_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := 
    Nat.factorial (math_majors - 1) * Nat.factorial physics_majors * Nat.factorial biology_majors
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability := by
  sorry

end NUMINAMATH_CALUDE_consecutive_majors_probability_l2823_282339


namespace NUMINAMATH_CALUDE_gumball_pigeonhole_min_gumballs_for_five_same_color_l2823_282360

theorem gumball_pigeonhole : ∀ (draw : ℕ),
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) →
  draw ≥ 17 :=
by
  sorry

theorem min_gumballs_for_five_same_color :
  ∃ (draw : ℕ), draw = 17 ∧
  (∀ (smaller : ℕ), smaller < draw →
    ¬∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) ∧
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) :=
by
  sorry

end NUMINAMATH_CALUDE_gumball_pigeonhole_min_gumballs_for_five_same_color_l2823_282360


namespace NUMINAMATH_CALUDE_mole_fractions_C4H8O2_l2823_282368

/-- Represents a chemical compound with counts of carbon, hydrogen, and oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the total number of atoms in a compound -/
def totalAtoms (c : Compound) : ℕ := c.carbon + c.hydrogen + c.oxygen

/-- Calculates the mole fraction of an element in a compound -/
def moleFraction (elementCount : ℕ) (c : Compound) : ℚ :=
  elementCount / (totalAtoms c)

/-- The compound C4H8O2 -/
def C4H8O2 : Compound := ⟨4, 8, 2⟩

theorem mole_fractions_C4H8O2 :
  moleFraction C4H8O2.carbon C4H8O2 = 2/7 ∧
  moleFraction C4H8O2.hydrogen C4H8O2 = 4/7 ∧
  moleFraction C4H8O2.oxygen C4H8O2 = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_mole_fractions_C4H8O2_l2823_282368


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2823_282399

theorem binomial_coefficient_congruence 
  (p : Nat) 
  (hp : p.Prime ∧ p > 3 ∧ Odd p) 
  (a b : Nat) 
  (hab : a > b ∧ b > 1) : 
  Nat.choose (a * p) (a * p) ≡ Nat.choose a b [MOD p^3] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2823_282399


namespace NUMINAMATH_CALUDE_min_value_theorem_l2823_282337

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ 
    t' > 0 ∧ u' > 0 ∧ v' > 0 ∧ w' > 0 ∧
    p' * q' * r' * s' = 16 ∧ t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2823_282337


namespace NUMINAMATH_CALUDE_normalized_coordinates_sum_of_squares_is_one_l2823_282310

/-- The sum of squares of normalized coordinates is 1 -/
theorem normalized_coordinates_sum_of_squares_is_one
  (a b : ℝ) -- Coordinates of point Q
  (d : ℝ) -- Distance from origin to Q
  (h_d : d = Real.sqrt (a^2 + b^2)) -- Definition of distance
  (u : ℝ) (h_u : u = b / d) -- Definition of u
  (v : ℝ) (h_v : v = a / d) -- Definition of v
  : u^2 + v^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_normalized_coordinates_sum_of_squares_is_one_l2823_282310


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_system_l2823_282371

theorem unique_solution_absolute_value_system :
  ∃! (x y : ℝ), 
    (abs (x + y) + abs (1 - x) = 6) ∧
    (abs (x + y + 1) + abs (1 - y) = 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_system_l2823_282371


namespace NUMINAMATH_CALUDE_garage_bikes_l2823_282353

/-- The number of bikes that can be assembled given a certain number of wheels -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

/-- Theorem: Given 20 bike wheels and 2 wheels per bike, 10 bikes can be assembled -/
theorem garage_bikes : bikes_assembled 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_garage_bikes_l2823_282353


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2823_282348

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2823_282348


namespace NUMINAMATH_CALUDE_am_gm_inequality_l2823_282366

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : a < b) (hbc : b < c) :
  ((a + c) / 2) - Real.sqrt (a * c) < (c - a)^2 / (8 * a) := by
sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l2823_282366


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2823_282375

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2823_282375


namespace NUMINAMATH_CALUDE_a_squared_greater_than_b_squared_l2823_282369

theorem a_squared_greater_than_b_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a = Real.log (1 + b) - Real.log (1 - b)) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_greater_than_b_squared_l2823_282369


namespace NUMINAMATH_CALUDE_greatest_b_proof_l2823_282394

/-- The greatest integer b for which x^2 + bx + 17 ≠ 0 for all real x -/
def greatest_b : ℤ := 8

theorem greatest_b_proof :
  (∀ x : ℝ, x^2 + (greatest_b : ℝ) * x + 17 ≠ 0) ∧
  (∀ b : ℤ, b > greatest_b → ∃ x : ℝ, x^2 + (b : ℝ) * x + 17 = 0) :=
by sorry

#check greatest_b_proof

end NUMINAMATH_CALUDE_greatest_b_proof_l2823_282394


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2823_282340

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def decimal_division (a b : Nat) : Nat :=
  a / b

theorem base_conversion_and_arithmetic :
  let n1 := base_to_decimal [3, 6, 4, 1] 7
  let n2 := base_to_decimal [1, 2, 1] 5
  let n3 := base_to_decimal [4, 5, 7, 1] 6
  let n4 := base_to_decimal [6, 5, 4, 3] 7
  decimal_division n1 n2 - n3 * 2 + n4 = 278 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l2823_282340


namespace NUMINAMATH_CALUDE_correct_divisor_problem_l2823_282321

theorem correct_divisor_problem (dividend : ℕ) (incorrect_divisor : ℕ) (incorrect_answer : ℕ) (correct_answer : ℕ) :
  dividend = incorrect_divisor * incorrect_answer →
  dividend / correct_answer = 36 →
  incorrect_divisor = 48 →
  incorrect_answer = 24 →
  correct_answer = 32 →
  36 = dividend / correct_answer :=
by sorry

end NUMINAMATH_CALUDE_correct_divisor_problem_l2823_282321


namespace NUMINAMATH_CALUDE_overall_percentage_favor_l2823_282305

-- Define the given percentages
def starting_favor_percent : ℝ := 0.40
def experienced_favor_percent : ℝ := 0.70

-- Define the number of surveyed entrepreneurs
def num_starting : ℕ := 300
def num_experienced : ℕ := 500

-- Define the total number surveyed
def total_surveyed : ℕ := num_starting + num_experienced

-- Define the number in favor for each group
def num_starting_favor : ℝ := starting_favor_percent * num_starting
def num_experienced_favor : ℝ := experienced_favor_percent * num_experienced

-- Define the total number in favor
def total_favor : ℝ := num_starting_favor + num_experienced_favor

-- Theorem to prove
theorem overall_percentage_favor :
  (total_favor / total_surveyed) * 100 = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_favor_l2823_282305


namespace NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2823_282379

/-- Two circles C₁ and C₂ in a 2D plane -/
structure TwoCircles where
  /-- Center of circle C₁ -/
  center1 : ℝ × ℝ
  /-- Radius of circle C₁ -/
  radius1 : ℝ
  /-- Center of circle C₂ -/
  center2 : ℝ × ℝ
  /-- Radius of circle C₂ -/
  radius2 : ℝ

/-- The length of the common chord between two intersecting circles -/
def commonChordLength (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem: The length of the common chord between the given intersecting circles is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  let circles : TwoCircles := {
    center1 := (2, 1),
    radius1 := Real.sqrt 10,
    center2 := (-6, -3),
    radius2 := Real.sqrt 50
  }
  commonChordLength circles = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l2823_282379


namespace NUMINAMATH_CALUDE_triangle_QCA_area_l2823_282327

/-- The area of triangle QCA given the coordinates of Q, A, and C, and that QA is perpendicular to QC -/
theorem triangle_QCA_area (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  -- QA is perpendicular to QC (implicit in the coordinate system)
  (45 - 3 * p) / 2 = (1 / 2) * 3 * (15 - p) := by
  sorry

end NUMINAMATH_CALUDE_triangle_QCA_area_l2823_282327


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2823_282362

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≥ 0) ↔ -1 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2823_282362


namespace NUMINAMATH_CALUDE_unique_triple_l2823_282335

theorem unique_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 1 ∧ (x, y, z) = (2, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l2823_282335


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2823_282361

theorem perfect_square_sum (a b : ℕ) : 
  ∃ (n : ℕ), 3^a + 4^b = n^2 ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2823_282361


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2823_282364

-- Define the direction vectors
def v1 (a : ℝ) : Fin 3 → ℝ := ![2*a, 3, 2]
def v2 : Fin 3 → ℝ := ![2, 3, 2]

-- Define the condition for parallel lines
def are_parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, v1 a i = k * v2 i

-- Theorem statement
theorem parallel_lines_imply_a_equals_one :
  are_parallel 1 → ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2823_282364


namespace NUMINAMATH_CALUDE_solve_turtle_problem_l2823_282378

def turtle_problem (kristen_turtles : ℕ) (kris_ratio : ℚ) (trey_multiplier : ℕ) : Prop :=
  let kris_turtles : ℚ := kris_ratio * kristen_turtles
  let trey_turtles : ℚ := trey_multiplier * kris_turtles
  (trey_turtles - kristen_turtles : ℚ) = 9

theorem solve_turtle_problem :
  turtle_problem 12 (1/4) 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_turtle_problem_l2823_282378


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l2823_282390

/-- Prove that for a parabola y = ax^2 (a > 0) where the distance from focus to directrix is 1/4,
    and two points A(x₁, y₁) and B(x₂, y₂) on the parabola are symmetric about y = x + m,
    and x₁x₂ = -1/2, then m = 3/2 -/
theorem parabola_symmetric_points (a : ℝ) (x₁ y₁ x₂ y₂ m : ℝ) : 
  a > 0 →
  (1 / (4 * a) = 1 / 4) →
  y₁ = a * x₁^2 →
  y₂ = a * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2 * m →
  x₁ * x₂ = -1 / 2 →
  m = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l2823_282390


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2823_282391

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty parts -/
def partitions_into_at_most (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 7 ways to partition 6 indistinguishable objects into 3 or fewer non-empty parts -/
theorem six_balls_three_boxes : partitions_into_at_most 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2823_282391


namespace NUMINAMATH_CALUDE_prob_all_heads_or_five_plus_tails_is_one_eighth_l2823_282356

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The probability of getting heads on a single fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails on a single fair coin flip -/
def p_tails : ℚ := 1/2

/-- The probability of getting all heads or at least five tails when flipping six fair coins -/
def prob_all_heads_or_five_plus_tails : ℚ := 1/8

/-- Theorem stating that the probability of getting all heads or at least five tails 
    when flipping six fair coins is 1/8 -/
theorem prob_all_heads_or_five_plus_tails_is_one_eighth :
  prob_all_heads_or_five_plus_tails = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_heads_or_five_plus_tails_is_one_eighth_l2823_282356


namespace NUMINAMATH_CALUDE_ellipse_parabola_tangent_lines_l2823_282334

/-- Given an ellipse and a parabola with specific properties, prove the equation of the parabola and its tangent lines. -/
theorem ellipse_parabola_tangent_lines :
  ∀ (b : ℝ) (p : ℝ),
  0 < b → b < 2 → p > 0 →
  (∀ (x y : ℝ), x^2 / 4 + y^2 / b^2 = 1 → (x^2 + y^2) / 4 = 3 / 4) →
  (∀ (x y : ℝ), x^2 = 2 * p * y) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / 4 + y₀^2 / b^2 = 1 ∧ x₀^2 = 2 * p * y₀ ∧ (x₀ = 0 ∨ y₀ = 1 ∨ y₀ = -1)) →
  (∀ (x y : ℝ), x^2 = 4 * y) ∧
  (∀ (x y : ℝ), (y = 0 ∨ x + y + 1 = 0) → 
    (x + 1)^2 = 4 * y ∧ (x + 1 = -1 → y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_tangent_lines_l2823_282334


namespace NUMINAMATH_CALUDE_cauchy_schwarz_on_unit_circle_l2823_282336

theorem cauchy_schwarz_on_unit_circle (a b x y : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : x^2 + y^2 = 1) : a*x + b*y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_on_unit_circle_l2823_282336


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_at_14_l2823_282387

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Divisibility condition for the polynomial -/
def DivisibilityCondition (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ p : ℝ → ℝ, ∀ x, (q x)^3 - x = p x * (x - 2) * (x + 2) * (x - 9)

theorem quadratic_polynomial_value_at_14 
  (q : QuadraticPolynomial ℝ) 
  (h : DivisibilityCondition q) : 
  q 14 = -82 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_at_14_l2823_282387


namespace NUMINAMATH_CALUDE_polar_equation_is_parabola_l2823_282325

theorem polar_equation_is_parabola :
  ∀ (r θ x y : ℝ),
  (r = 2 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ (a b : ℝ), x^2 = a * y + b :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_parabola_l2823_282325


namespace NUMINAMATH_CALUDE_coloring_satisfies_conditions_l2823_282350

-- Define the color type
inductive Color
| White
| Red
| Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define the coloring function
def color (p : LatticePoint) : Color :=
  match p.x, p.y with
  | x, y => if x % 2 = 0 then Color.Red
            else if y % 2 = 0 then Color.Black
            else Color.White

-- Define a line parallel to x-axis
def Line (y : Int) := { p : LatticePoint | p.y = y }

-- Define a parallelogram
def isParallelogram (a b c d : LatticePoint) : Prop :=
  d.x = a.x + c.x - b.x ∧ d.y = a.y + c.y - b.y

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, ∃ (S : Set Int), Infinite S ∧ ∀ y ∈ S, ∃ x : Int, color ⟨x, y⟩ = c) ∧
  (∀ a b c : LatticePoint, 
    color a = Color.White → color b = Color.Red → color c = Color.Black →
    ∃ d : LatticePoint, color d = Color.Red ∧ isParallelogram a b c d) :=
sorry


end NUMINAMATH_CALUDE_coloring_satisfies_conditions_l2823_282350


namespace NUMINAMATH_CALUDE_power_comparison_a_l2823_282314

theorem power_comparison_a : 3^200 > 2^300 := by sorry

end NUMINAMATH_CALUDE_power_comparison_a_l2823_282314


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2823_282329

-- Define the number of jelly beans for each color
def red_beans : ℕ := 7
def green_beans : ℕ := 9
def yellow_beans : ℕ := 8
def blue_beans : ℕ := 10
def orange_beans : ℕ := 5

-- Define the total number of jelly beans
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + orange_beans

-- Define the number of blue or orange jelly beans
def blue_or_orange_beans : ℕ := blue_beans + orange_beans

-- Theorem statement
theorem jelly_bean_probability : 
  (blue_or_orange_beans : ℚ) / (total_beans : ℚ) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2823_282329


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2823_282303

theorem complex_fraction_equality : ∀ (z₁ z₂ : ℂ), 
  z₁ = -1 + 3*I ∧ z₂ = 1 + I → (z₁ + z₂) / (z₁ - z₂) = 1 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2823_282303
