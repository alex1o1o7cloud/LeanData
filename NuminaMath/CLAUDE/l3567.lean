import Mathlib

namespace NUMINAMATH_CALUDE_age_difference_l3567_356732

theorem age_difference (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 196 →
  monica_age - patrick_age = 64 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3567_356732


namespace NUMINAMATH_CALUDE_odot_calculation_l3567_356729

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end NUMINAMATH_CALUDE_odot_calculation_l3567_356729


namespace NUMINAMATH_CALUDE_right_angle_equation_l3567_356724

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- Definition of a right-angled triangle --/
def IsRightAngled (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

/-- The equation condition --/
def SatisfiesEquation (t : Triangle) : Prop :=
  t.a - t.b = t.c * (Real.cos t.B - Real.cos t.A)

/-- The theorem to be proved --/
theorem right_angle_equation (t : Triangle) :
  IsRightAngled t → SatisfiesEquation t ∧ 
  ¬(SatisfiesEquation t → IsRightAngled t) :=
sorry

end NUMINAMATH_CALUDE_right_angle_equation_l3567_356724


namespace NUMINAMATH_CALUDE_return_trip_speed_l3567_356791

/-- Proves that the average speed on the return trip from Syracuse to Albany is 50 miles per hour -/
theorem return_trip_speed
  (distance : ℝ)
  (speed_to_syracuse : ℝ)
  (total_time : ℝ)
  (h1 : distance = 120)
  (h2 : speed_to_syracuse = 40)
  (h3 : total_time = 5.4)
  : (distance / (total_time - distance / speed_to_syracuse)) = 50 :=
by sorry

end NUMINAMATH_CALUDE_return_trip_speed_l3567_356791


namespace NUMINAMATH_CALUDE_binomial_1350_2_l3567_356713

theorem binomial_1350_2 : Nat.choose 1350 2 = 910575 := by sorry

end NUMINAMATH_CALUDE_binomial_1350_2_l3567_356713


namespace NUMINAMATH_CALUDE_annual_pension_l3567_356761

/-- The annual pension problem -/
theorem annual_pension (c p q : ℝ) (x : ℝ) (k : ℝ) :
  (k * Real.sqrt (x + c) = k * Real.sqrt x + 3 * p) →
  (k * Real.sqrt (x + 2 * c) = k * Real.sqrt x + 4 * q) →
  (k * Real.sqrt x = (16 * q^2 - 18 * p^2) / (12 * p - 8 * q)) :=
by sorry

end NUMINAMATH_CALUDE_annual_pension_l3567_356761


namespace NUMINAMATH_CALUDE_ratio_bounds_l3567_356737

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_bounds_l3567_356737


namespace NUMINAMATH_CALUDE_profit_percentage_l3567_356778

/-- If selling an article at 2/3 of a certain price results in a 15% loss,
    then selling at the full certain price results in a 27.5% profit. -/
theorem profit_percentage (certain_price : ℝ) (cost_price : ℝ) :
  certain_price > 0 →
  cost_price > 0 →
  (2 / 3 : ℝ) * certain_price = 0.85 * cost_price →
  (certain_price - cost_price) / cost_price = 0.275 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_l3567_356778


namespace NUMINAMATH_CALUDE_subtract_fractions_l3567_356730

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l3567_356730


namespace NUMINAMATH_CALUDE_f_increasing_f_sum_positive_l3567_356706

-- Define the function f(x) = x + x^3
def f (x : ℝ) : ℝ := x + x^3

-- Theorem 1: f is an increasing function
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

-- Theorem 2: For any a, b ∈ ℝ where a + b > 0, f(a) + f(b) > 0
theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_sum_positive_l3567_356706


namespace NUMINAMATH_CALUDE_paintings_per_room_l3567_356726

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
sorry

end NUMINAMATH_CALUDE_paintings_per_room_l3567_356726


namespace NUMINAMATH_CALUDE_part_one_part_two_l3567_356741

/-- The function f(x) = ax^2 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

/-- Part 1: Given the solution set of f(x) > 0, prove a and b -/
theorem part_one (a b : ℝ) : 
  (∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)) → a = 1 ∧ b = 2 :=
sorry

/-- Part 2: Given f(x) > 0 for all x, prove the range of a -/
theorem part_two (a : ℝ) :
  (∀ x, f a x > 0) → 0 ≤ a ∧ a < 8/9 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3567_356741


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l3567_356777

theorem movie_ticket_cost (x : ℝ) : 
  (2 * x + 3 * (x - 2) = 39) →
  x = 9 := by sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l3567_356777


namespace NUMINAMATH_CALUDE_max_abs_z_l3567_356779

/-- Given a complex number z satisfying |z - 8| + |z + 6i| = 10, 
    the maximum value of |z| is 8. -/
theorem max_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6*I) = 10) : 
  ∃ (w : ℂ), Complex.abs (w - 8) + Complex.abs (w + 6*I) = 10 ∧ 
             ∀ (u : ℂ), Complex.abs (u - 8) + Complex.abs (u + 6*I) = 10 → 
             Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 8 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l3567_356779


namespace NUMINAMATH_CALUDE_absent_fraction_proof_l3567_356712

/-- Proves that if work increases by 1/6 when a fraction of members are absent,
    then the fraction of absent members is 1/7 -/
theorem absent_fraction_proof (p : ℕ) (p_pos : p > 0) :
  let increase_factor : ℚ := 1 / 6
  let absent_fraction : ℚ := 1 / 7
  (1 : ℚ) + increase_factor = 1 / (1 - absent_fraction) :=
by sorry

end NUMINAMATH_CALUDE_absent_fraction_proof_l3567_356712


namespace NUMINAMATH_CALUDE_rational_function_equality_l3567_356731

theorem rational_function_equality (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 120*x + 3480) / (x^2 + 54*x - 2835)) →
  α + β = 123 := by
sorry

end NUMINAMATH_CALUDE_rational_function_equality_l3567_356731


namespace NUMINAMATH_CALUDE_max_value_of_function_l3567_356705

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (y : ℝ), y = x^2 * (1 - 2*x) ∧ y ≤ 1/27 ∧ ∃ (x0 : ℝ), 0 < x0 ∧ x0 < 1/2 ∧ x0^2 * (1 - 2*x0) = 1/27 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3567_356705


namespace NUMINAMATH_CALUDE_distance_home_to_school_l3567_356752

/-- The distance between home and school satisfies the given conditions -/
theorem distance_home_to_school :
  ∃ (d : ℝ), d > 0 ∧
  ∃ (t : ℝ), t > 0 ∧
  (5 * (t + 7/60) = d) ∧
  (10 * (t - 8/60) = d) ∧
  d = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l3567_356752


namespace NUMINAMATH_CALUDE_burrito_cheese_amount_l3567_356784

/-- The amount of cheese (in ounces) required for a burrito -/
def cheese_per_burrito : ℝ := 4

/-- The amount of cheese (in ounces) required for a taco -/
def cheese_per_taco : ℝ := 9

/-- The total amount of cheese (in ounces) required for 7 burritos and 1 taco -/
def total_cheese : ℝ := 37

/-- Theorem stating that the amount of cheese required for a burrito is 4 ounces -/
theorem burrito_cheese_amount :
  cheese_per_burrito = 4 ∧
  cheese_per_taco = 9 ∧
  7 * cheese_per_burrito + cheese_per_taco = total_cheese :=
by sorry

end NUMINAMATH_CALUDE_burrito_cheese_amount_l3567_356784


namespace NUMINAMATH_CALUDE_diagonals_difference_octagon_heptagon_l3567_356793

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem diagonals_difference_octagon_heptagon :
  num_diagonals octagon_sides - num_diagonals heptagon_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_difference_octagon_heptagon_l3567_356793


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3567_356707

theorem infinite_geometric_series_first_term
  (r : ℚ)
  (S : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 20)
  (h3 : S = a / (1 - r)) :
  a = 15 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3567_356707


namespace NUMINAMATH_CALUDE_triangle_angle_expression_range_l3567_356750

theorem triangle_angle_expression_range (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -25/16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 
  3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_expression_range_l3567_356750


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3567_356764

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ x ∈ solutions, x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3567_356764


namespace NUMINAMATH_CALUDE_cookie_cost_l3567_356758

theorem cookie_cost (cheeseburger_cost milkshake_cost coke_cost fries_cost tax : ℚ)
  (toby_initial toby_change : ℚ) (cookie_count : ℕ) :
  cheeseburger_cost = 365/100 ∧ 
  milkshake_cost = 2 ∧ 
  coke_cost = 1 ∧ 
  fries_cost = 4 ∧ 
  tax = 1/5 ∧
  toby_initial = 15 ∧ 
  toby_change = 7 ∧
  cookie_count = 3 →
  let total_before_cookies := 2 * cheeseburger_cost + milkshake_cost + coke_cost + fries_cost + tax
  let total_spent := 2 * (toby_initial - toby_change)
  let cookie_total_cost := total_spent - total_before_cookies
  cookie_total_cost / cookie_count = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cookie_cost_l3567_356758


namespace NUMINAMATH_CALUDE_class_books_total_l3567_356790

/-- The total number of books a class received from the library -/
def total_books (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the class received 77 books in total -/
theorem class_books_total : total_books 54 23 = 77 := by
  sorry

end NUMINAMATH_CALUDE_class_books_total_l3567_356790


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3567_356742

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x + y ≤ 1 → x ≤ 1/2 ∨ y ≤ 1/2) ∧
  (∃ x y, (x ≤ 1/2 ∨ y ≤ 1/2) ∧ x + y > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3567_356742


namespace NUMINAMATH_CALUDE_binary_subtraction_equiv_l3567_356736

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

theorem binary_subtraction_equiv :
  let a := [true, true, false, true, true]  -- 11011 in binary
  let b := [true, false, true]              -- 101 in binary
  let result := [true, true, false, true]   -- 1011 in binary (11 in decimal)
  binary_to_decimal a - binary_to_decimal b = binary_to_decimal result :=
by
  sorry

#eval binary_to_decimal [true, true, false, true, true]  -- Should output 27
#eval binary_to_decimal [true, false, true]              -- Should output 5
#eval binary_to_decimal [true, true, false, true]        -- Should output 11

end NUMINAMATH_CALUDE_binary_subtraction_equiv_l3567_356736


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l3567_356760

theorem triangle_angle_difference (A B C : ℝ) : 
  A = 24 →
  B = 5 * A →
  A + B + C = 180 →
  C - A = 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l3567_356760


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3567_356740

theorem unique_solution_for_equation :
  ∃! (m n : ℝ), 21 * (m^2 + n) + 21 * Real.sqrt n = 21 * (-m^3 + n^2) + 21 * m^2 * n ∧ m = -1 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3567_356740


namespace NUMINAMATH_CALUDE_divide_flour_possible_l3567_356714

/-- Represents the result of a weighing operation -/
inductive Weighing
| MeasuredFlour (amount : ℕ)
| CombinedFlour (amount1 amount2 : ℕ)

/-- Represents a weighing operation using the balance scale -/
def weigh (yeast ginger flour : ℕ) : Weighing :=
  sorry

/-- Represents the process of dividing flour using two weighings -/
def divideFlour (totalFlour yeast ginger : ℕ) : Option (ℕ × ℕ) :=
  sorry

/-- Theorem stating that it's possible to divide 500g of flour into 400g and 100g parts
    using only two weighings with 5g of yeast and 30g of ginger -/
theorem divide_flour_possible :
  ∃ (w1 w2 : Weighing),
    let result := divideFlour 500 5 30
    result = some (400, 100) ∧
    (∃ (f1 : ℕ), w1 = Weighing.MeasuredFlour f1) ∧
    (∃ (f2 : ℕ), w2 = Weighing.MeasuredFlour f2) ∧
    f1 + f2 = 100 :=
  sorry

end NUMINAMATH_CALUDE_divide_flour_possible_l3567_356714


namespace NUMINAMATH_CALUDE_cubic_roots_expression_l3567_356718

theorem cubic_roots_expression (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p^3 + q^3 + r^3 - 3*p*q*r = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_expression_l3567_356718


namespace NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3567_356749

/-- Represents a periodic decimal fraction -/
structure PeriodicDecimal where
  nonRepeatingPart : List Nat
  repeatingPart : List Nat
  nonEmpty : repeatingPart.length > 0

/-- Converts a PeriodicDecimal to a real number -/
noncomputable def toReal (x : PeriodicDecimal) : Real := sorry

/-- Theorem: Every periodic decimal fraction is a rational number -/
theorem periodic_decimal_is_rational (x : PeriodicDecimal) :
  ∃ (p q : Int), q ≠ 0 ∧ toReal x = p / q := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_is_rational_l3567_356749


namespace NUMINAMATH_CALUDE_insurance_calculation_l3567_356754

/-- Insurance calculation parameters --/
structure InsuranceParams where
  baseRate : Float
  noTransitionCoeff : Float
  noMedCertCoeff : Float
  assessedValue : Float
  cadasterValue : Float

/-- Calculate adjusted tariff --/
def calcAdjustedTariff (params : InsuranceParams) : Float :=
  params.baseRate * params.noTransitionCoeff * params.noMedCertCoeff

/-- Determine insurance amount --/
def determineInsuranceAmount (params : InsuranceParams) : Float :=
  max params.assessedValue params.cadasterValue

/-- Calculate insurance premium --/
def calcInsurancePremium (amount : Float) (tariff : Float) : Float :=
  amount * tariff

/-- Main theorem --/
theorem insurance_calculation (params : InsuranceParams) 
  (h1 : params.baseRate = 0.002)
  (h2 : params.noTransitionCoeff = 0.8)
  (h3 : params.noMedCertCoeff = 1.3)
  (h4 : params.assessedValue = 14500000)
  (h5 : params.cadasterValue = 15000000) :
  let adjustedTariff := calcAdjustedTariff params
  let insuranceAmount := determineInsuranceAmount params
  let insurancePremium := calcInsurancePremium insuranceAmount adjustedTariff
  adjustedTariff = 0.00208 ∧ 
  insuranceAmount = 15000000 ∧ 
  insurancePremium = 31200 := by
  sorry

end NUMINAMATH_CALUDE_insurance_calculation_l3567_356754


namespace NUMINAMATH_CALUDE_river_speed_theorem_l3567_356796

/-- Represents the equation for a ship traveling upstream and downstream -/
def river_equation (s v d1 d2 : ℝ) : Prop :=
  d1 / (s + v) = d2 / (s - v)

/-- Theorem stating that the river equation holds for the given conditions -/
theorem river_speed_theorem (s v d1 d2 : ℝ) 
  (h_s : s > 0)
  (h_v : 0 < v ∧ v < s)
  (h_d1 : d1 > 0)
  (h_d2 : d2 > 0)
  (h_s_still : s = 30)
  (h_d1 : d1 = 144)
  (h_d2 : d2 = 96) :
  river_equation s v d1 d2 :=
sorry

end NUMINAMATH_CALUDE_river_speed_theorem_l3567_356796


namespace NUMINAMATH_CALUDE_magicians_number_l3567_356769

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≤ 9
  h2 : b ≤ 9
  h3 : c ≤ 9
  h4 : 0 < a

/-- Calculates the value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Calculates the sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  (value n) + 
  (100 * n.a + 10 * n.c + n.b) +
  (100 * n.b + 10 * n.c + n.a) +
  (100 * n.b + 10 * n.a + n.c) +
  (100 * n.c + 10 * n.a + n.b) +
  (100 * n.c + 10 * n.b + n.a)

/-- The main theorem to prove -/
theorem magicians_number (n : ThreeDigitNumber) 
  (h : sumOfPermutations n = 4332) : value n = 118 := by
  sorry

end NUMINAMATH_CALUDE_magicians_number_l3567_356769


namespace NUMINAMATH_CALUDE_diamond_operation_l3567_356775

def diamond (a b : ℤ) : ℤ := 12 * a - 10 * b

theorem diamond_operation : diamond (diamond (diamond (diamond 20 22) 22) 22) 22 = 20 := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_l3567_356775


namespace NUMINAMATH_CALUDE_monotone_decreasing_cubic_l3567_356785

/-- A function f is monotonically decreasing on an open interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem monotone_decreasing_cubic (a : ℝ) :
  MonotonicallyDecreasing (fun x => x^3 - a*x^2 + 1) 0 2 → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_cubic_l3567_356785


namespace NUMINAMATH_CALUDE_barbara_candies_l3567_356773

theorem barbara_candies (original_boxes : Nat) (original_candies_per_box : Nat)
                         (new_boxes : Nat) (new_candies_per_box : Nat) :
  original_boxes = 9 →
  original_candies_per_box = 25 →
  new_boxes = 18 →
  new_candies_per_box = 35 →
  original_boxes * original_candies_per_box + new_boxes * new_candies_per_box = 855 :=
by sorry

end NUMINAMATH_CALUDE_barbara_candies_l3567_356773


namespace NUMINAMATH_CALUDE_divisors_not_div_by_3_eq_6_l3567_356710

/-- The number of positive divisors of 180 that are not divisible by 3 -/
def divisors_not_div_by_3 : ℕ :=
  (Finset.filter (fun d => d ∣ 180 ∧ ¬(3 ∣ d)) (Finset.range 181)).card

/-- Theorem stating that the number of positive divisors of 180 not divisible by 3 is 6 -/
theorem divisors_not_div_by_3_eq_6 : divisors_not_div_by_3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisors_not_div_by_3_eq_6_l3567_356710


namespace NUMINAMATH_CALUDE_circle_passes_800_squares_l3567_356766

/-- A circle on a unit square grid -/
structure GridCircle where
  radius : ℕ
  -- The circle does not touch any grid lines or pass through any lattice points
  no_grid_touch : True

/-- The number of squares a circle passes through on a unit square grid -/
def squares_passed (c : GridCircle) : ℕ :=
  4 * (2 * c.radius)

/-- Theorem: A circle with radius 100 passes through 800 squares -/
theorem circle_passes_800_squares (c : GridCircle) (h : c.radius = 100) :
  squares_passed c = 800 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_800_squares_l3567_356766


namespace NUMINAMATH_CALUDE_total_earnings_value_l3567_356753

def friday_earnings : ℚ := 147
def saturday_earnings : ℚ := 2 * friday_earnings + 7
def sunday_earnings : ℚ := friday_earnings + 78
def monday_earnings : ℚ := 0.75 * friday_earnings
def tuesday_earnings : ℚ := 1.25 * monday_earnings
def wednesday_earnings : ℚ := 0.8 * tuesday_earnings

def total_earnings : ℚ := friday_earnings + saturday_earnings + sunday_earnings + 
                          monday_earnings + tuesday_earnings + wednesday_earnings

theorem total_earnings_value : total_earnings = 1031.3125 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_value_l3567_356753


namespace NUMINAMATH_CALUDE_three_people_on_third_stop_l3567_356789

/-- Represents the number of people on a bus and its changes at stops -/
structure BusRide where
  initial : ℕ
  first_off : ℕ
  second_off : ℕ
  second_on : ℕ
  third_off : ℕ
  final : ℕ

/-- Calculates the number of people who got on at the third stop -/
def people_on_third_stop (ride : BusRide) : ℕ :=
  ride.final - (ride.initial - ride.first_off - ride.second_off + ride.second_on - ride.third_off)

/-- Theorem stating that 3 people got on at the third stop -/
theorem three_people_on_third_stop (ride : BusRide) 
  (h_initial : ride.initial = 50)
  (h_first_off : ride.first_off = 15)
  (h_second_off : ride.second_off = 8)
  (h_second_on : ride.second_on = 2)
  (h_third_off : ride.third_off = 4)
  (h_final : ride.final = 28) :
  people_on_third_stop ride = 3 := by
  sorry

#eval people_on_third_stop { initial := 50, first_off := 15, second_off := 8, second_on := 2, third_off := 4, final := 28 }

end NUMINAMATH_CALUDE_three_people_on_third_stop_l3567_356789


namespace NUMINAMATH_CALUDE_yogurt_refund_calculation_l3567_356765

theorem yogurt_refund_calculation (total_packs : ℕ) (expired_percentage : ℚ) (price_per_pack : ℚ) : 
  total_packs = 80 →
  expired_percentage = 40 / 100 →
  price_per_pack = 12 →
  (total_packs : ℚ) * expired_percentage * price_per_pack = 384 := by
sorry

end NUMINAMATH_CALUDE_yogurt_refund_calculation_l3567_356765


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3567_356795

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  a > 0 →
  x₂ + x₁ = 15 →
  a = 15/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3567_356795


namespace NUMINAMATH_CALUDE_separation_sister_chromatids_not_in_first_division_l3567_356748

-- Define the events
inductive MeioticEvent
| PairingHomologousChromosomes
| CrossingOver
| SeparationSisterChromatids
| SeparationHomologousChromosomes

-- Define the property of occurring during the first meiotic division
def occursInFirstMeioticDivision : MeioticEvent → Prop :=
  fun event =>
    match event with
    | MeioticEvent.PairingHomologousChromosomes => True
    | MeioticEvent.CrossingOver => True
    | MeioticEvent.SeparationSisterChromatids => False
    | MeioticEvent.SeparationHomologousChromosomes => True

-- Theorem stating that separation of sister chromatids is the only event
-- that does not occur during the first meiotic division
theorem separation_sister_chromatids_not_in_first_division :
  ∀ (e : MeioticEvent),
    ¬occursInFirstMeioticDivision e ↔ e = MeioticEvent.SeparationSisterChromatids :=
by sorry

end NUMINAMATH_CALUDE_separation_sister_chromatids_not_in_first_division_l3567_356748


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3567_356786

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 8*x^8 + x^4 = 1365 := by
sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3567_356786


namespace NUMINAMATH_CALUDE_new_students_average_age_l3567_356792

theorem new_students_average_age
  (original_avg : ℝ)
  (new_students : ℕ)
  (avg_decrease : ℝ)
  (original_strength : ℕ)
  (h1 : original_avg = 40)
  (h2 : new_students = 17)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 17) :
  let new_avg := original_avg - avg_decrease
  let total_students := original_strength + new_students
  let original_total_age := original_strength * original_avg
  let new_total_age := total_students * new_avg
  let new_students_total_age := new_total_age - original_total_age
  new_students_total_age / new_students = 32 :=
sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3567_356792


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3567_356768

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of technicians
def num_technicians : ℕ := 7

-- Define the average salary of technicians
def avg_salary_technicians : ℕ := 12000

-- Define the average salary of other workers
def avg_salary_others : ℕ := 8000

-- Theorem statement
theorem workshop_average_salary :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = 10000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3567_356768


namespace NUMINAMATH_CALUDE_lauras_weekly_driving_distance_l3567_356700

/-- Laura's weekly driving distance calculation -/
theorem lauras_weekly_driving_distance :
  -- Definitions based on given conditions
  let round_trip_to_school : ℕ := 20
  let supermarket_distance_from_school : ℕ := 10
  let days_to_school_per_week : ℕ := 7
  let supermarket_trips_per_week : ℕ := 2

  -- Derived calculations
  let round_trip_to_supermarket : ℕ := round_trip_to_school + 2 * supermarket_distance_from_school
  let weekly_school_distance : ℕ := days_to_school_per_week * round_trip_to_school
  let weekly_supermarket_distance : ℕ := supermarket_trips_per_week * round_trip_to_supermarket

  -- Theorem statement
  weekly_school_distance + weekly_supermarket_distance = 220 := by
  sorry

end NUMINAMATH_CALUDE_lauras_weekly_driving_distance_l3567_356700


namespace NUMINAMATH_CALUDE_ascending_concept_chain_l3567_356744

-- Define the concept hierarchy
def IsNatural (n : ℕ) : Prop := True
def IsInteger (n : ℤ) : Prop := True
def IsRational (q : ℚ) : Prop := True
def IsReal (r : ℝ) : Prop := True
def IsNumber (x : ℝ) : Prop := True

-- Define the chain of ascending concepts
def ConceptChain : Prop :=
  ∃ (n : ℕ) (z : ℤ) (q : ℚ) (r : ℝ),
    n = 3 ∧
    IsNatural n ∧
    (↑n : ℤ) = z ∧
    IsInteger z ∧
    (↑z : ℚ) = q ∧
    IsRational q ∧
    (↑q : ℝ) = r ∧
    IsReal r ∧
    IsNumber r

-- Theorem statement
theorem ascending_concept_chain : ConceptChain :=
  sorry

end NUMINAMATH_CALUDE_ascending_concept_chain_l3567_356744


namespace NUMINAMATH_CALUDE_fraction_sign_change_l3567_356720

theorem fraction_sign_change (a b : ℝ) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sign_change_l3567_356720


namespace NUMINAMATH_CALUDE_volleyball_team_physics_count_l3567_356708

theorem volleyball_team_physics_count 
  (total_players : ℕ) 
  (math_players : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_players = 15)
  (h2 : math_players = 10)
  (h3 : both_subjects = 4)
  (h4 : both_subjects ≤ math_players)
  (h5 : math_players ≤ total_players) :
  total_players - (math_players - both_subjects) = 9 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_physics_count_l3567_356708


namespace NUMINAMATH_CALUDE_battery_current_l3567_356763

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48/R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  R = 12 → I = 48 / R → I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_l3567_356763


namespace NUMINAMATH_CALUDE_max_t_value_l3567_356733

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  r ≤ 17 →
  t ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l3567_356733


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3567_356782

theorem arithmetic_calculation : 
  3889 + 12.952 - 47.95000000000027 = 3854.0019999999997 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3567_356782


namespace NUMINAMATH_CALUDE_language_course_enrollment_l3567_356725

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 150 →
  french = 58 →
  german = 40 →
  spanish = 35 →
  french_german = 20 →
  french_spanish = 15 →
  german_spanish = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 62 :=
by sorry

end NUMINAMATH_CALUDE_language_course_enrollment_l3567_356725


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3567_356738

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (eq : 26 * x^2 + 23 * x * y - 3 * y^2 - 19 = 0) 
  (progression : ∃ (a d : ℤ), x = a + 5 * d ∧ y = a + 10 * d ∧ d < 0) :
  ∃ (a : ℤ), x = a + 5 * (-3) ∧ y = a + 10 * (-3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l3567_356738


namespace NUMINAMATH_CALUDE_least_common_multiple_5_to_10_l3567_356788

theorem least_common_multiple_5_to_10 : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 5 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_5_to_10_l3567_356788


namespace NUMINAMATH_CALUDE_apple_purchase_remainder_l3567_356755

theorem apple_purchase_remainder (mark_money carolyn_money apple_cost : ℚ) : 
  mark_money = 2/3 →
  carolyn_money = 1/5 →
  apple_cost = 1/2 →
  mark_money + carolyn_money - apple_cost = 11/30 := by
sorry

end NUMINAMATH_CALUDE_apple_purchase_remainder_l3567_356755


namespace NUMINAMATH_CALUDE_eight_reader_permutations_l3567_356702

theorem eight_reader_permutations : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_reader_permutations_l3567_356702


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3567_356780

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let S := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = S := by sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3567_356780


namespace NUMINAMATH_CALUDE_total_sample_variance_stratified_sampling_l3567_356762

/-- Calculates the total sample variance for stratified sampling of student heights -/
theorem total_sample_variance_stratified_sampling 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (male_mean : ℝ) 
  (female_mean : ℝ) 
  (male_variance : ℝ) 
  (female_variance : ℝ) 
  (h_male_count : male_count = 100)
  (h_female_count : female_count = 60)
  (h_male_mean : male_mean = 172)
  (h_female_mean : female_mean = 164)
  (h_male_variance : male_variance = 18)
  (h_female_variance : female_variance = 30) :
  let total_count := male_count + female_count
  let combined_mean := (male_count * male_mean + female_count * female_mean) / total_count
  let total_variance := 
    (male_count : ℝ) / total_count * (male_variance + (male_mean - combined_mean)^2) +
    (female_count : ℝ) / total_count * (female_variance + (female_mean - combined_mean)^2)
  total_variance = 37.5 := by
sorry


end NUMINAMATH_CALUDE_total_sample_variance_stratified_sampling_l3567_356762


namespace NUMINAMATH_CALUDE_range_of_m_when_p_implies_q_l3567_356703

/-- Represents an ellipse equation with parameter m -/
def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) + y^2 / (1-m) = 1 ∧ 0 < m ∧ m < 1/3

/-- Represents a hyperbola equation with parameter m -/
def is_hyperbola_with_eccentricity_between_1_and_2 (m : ℝ) : Prop :=
  ∃ x y e : ℝ, x^2 / 5 - y^2 / m = 1 ∧ 1 < e ∧ e < 2 ∧ m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m_when_p_implies_q :
  (∀ m : ℝ, is_ellipse_with_foci_on_y_axis m → is_hyperbola_with_eccentricity_between_1_and_2 m) →
  ∃ m : ℝ, 1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_implies_q_l3567_356703


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3567_356734

/-- A three-digit number is represented by its digits a, b, and c. -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The product of the digits of a three-digit number. -/
def digit_product (a b c : ℕ) : ℕ := a * b * c

/-- Predicate for a valid three-digit number. -/
def is_valid_three_digit (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- The main theorem: 175 is the only three-digit number that is 5 times the product of its digits. -/
theorem unique_three_digit_number :
  ∀ a b c : ℕ,
    is_valid_three_digit a b c →
    (three_digit_number a b c = 5 * digit_product a b c) →
    (a = 1 ∧ b = 7 ∧ c = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3567_356734


namespace NUMINAMATH_CALUDE_units_digit_of_41_cubed_plus_23_cubed_l3567_356722

theorem units_digit_of_41_cubed_plus_23_cubed : ∃ n : ℕ, 41^3 + 23^3 ≡ 8 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_41_cubed_plus_23_cubed_l3567_356722


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l3567_356723

theorem power_of_product_with_negative (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l3567_356723


namespace NUMINAMATH_CALUDE_percentage_difference_l3567_356704

theorem percentage_difference (A B C x : ℝ) : 
  C > B → B > A → A > 0 → C = A + 2*B → A = B * (1 - x/100) → 
  x = 100 * ((B - A) / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3567_356704


namespace NUMINAMATH_CALUDE_vector_collinearity_l3567_356728

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 5)
def c : ℝ → ℝ × ℝ := λ x => (x, 1)

-- Define collinearity
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_collinearity (x : ℝ) :
  collinear (2 * a - b) (c x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l3567_356728


namespace NUMINAMATH_CALUDE_smallest_marble_count_thirty_is_smallest_l3567_356774

theorem smallest_marble_count : ℕ → Prop :=
  fun n => n > 0 ∧ 
    (∃ w g r b : ℕ, 
      w + g + r + b = n ∧ 
      w = n / 6 ∧ 
      g = n / 5 ∧ 
      r + b = 19 * n / 30) →
  n ≥ 30

theorem thirty_is_smallest : smallest_marble_count 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_thirty_is_smallest_l3567_356774


namespace NUMINAMATH_CALUDE_x_cubed_coefficient_l3567_356751

theorem x_cubed_coefficient (p q : Polynomial ℤ) : 
  p = 3 * X^3 + 2 * X^2 + 5 * X + 6 →
  q = 4 * X^3 + 7 * X^2 + 9 * X + 8 →
  (p * q).coeff 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_coefficient_l3567_356751


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_system_2_l3567_356721

-- Equation (1)
theorem solve_equation_1 : 
  ∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 ↔ x = -1/4 := by sorry

-- System of equations (2)
theorem solve_system_2 : 
  ∃ x y : ℚ, (3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19) ↔ (x = 5 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_system_2_l3567_356721


namespace NUMINAMATH_CALUDE_segment_distance_inequality_l3567_356781

-- Define the plane
variable {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] [CompleteSpace α]

-- Define points A, B, C, D, and P
variable (A B C D P : α)

-- Define the conditions
variable (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = (1 - t) • A + t • D)
variable (h2 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ C = (1 - s) • A + s • D)
variable (h3 : ‖A - B‖ = ‖C - D‖)

-- State the theorem
theorem segment_distance_inequality :
  ‖P - A‖ + ‖P - D‖ ≥ ‖P - B‖ + ‖P - C‖ :=
sorry

end NUMINAMATH_CALUDE_segment_distance_inequality_l3567_356781


namespace NUMINAMATH_CALUDE_largest_510_triple_l3567_356743

/-- Converts a base-10 number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 5) :: aux (m / 5)
  aux n

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 5-10 triple -/
def is510Triple (n : ℕ) : Prop :=
  fromDigits (toBase5 n) = 3 * n

theorem largest_510_triple :
  (∀ m : ℕ, m > 115 → ¬ is510Triple m) ∧ is510Triple 115 :=
sorry

end NUMINAMATH_CALUDE_largest_510_triple_l3567_356743


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3567_356727

theorem polynomial_division_remainder :
  ∃ Q : Polynomial ℝ, (X : Polynomial ℝ)^5 - 3 * X^3 + 4 * X + 5 = 
  (X - 3)^2 * Q + (261 * X - 643) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3567_356727


namespace NUMINAMATH_CALUDE_quadratic_radical_simplification_l3567_356717

theorem quadratic_radical_simplification (a m n : ℕ+) :
  (a : ℝ) + 2 * Real.sqrt 21 = (Real.sqrt (m : ℝ) + Real.sqrt (n : ℝ))^2 →
  a = 10 ∨ a = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_simplification_l3567_356717


namespace NUMINAMATH_CALUDE_problem_solution_l3567_356715

theorem problem_solution (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -18)
  (h2 : z*y/(x+y) + z*x/(y+z) + y*x/(z+x) = 20) :
  y/(x+y) + z/(y+z) + x/(z+x) = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3567_356715


namespace NUMINAMATH_CALUDE_max_rice_plates_l3567_356798

def chapati_count : ℕ := 16
def chapati_cost : ℕ := 6
def mixed_veg_count : ℕ := 7
def mixed_veg_cost : ℕ := 70
def ice_cream_count : ℕ := 6
def rice_cost : ℕ := 45
def total_paid : ℕ := 985

theorem max_rice_plates (rice_count : ℕ) : 
  rice_count * rice_cost + 
  chapati_count * chapati_cost + 
  mixed_veg_count * mixed_veg_cost ≤ total_paid →
  rice_count ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_rice_plates_l3567_356798


namespace NUMINAMATH_CALUDE_calculation_result_l3567_356709

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the calculation in base 10 --/
def result : Rat :=
  (toBase10 [3, 1, 0, 2] 5 : Rat) / (toBase10 [1, 1] 3) -
  (toBase10 [4, 2, 1, 3] 6 : Rat) +
  (toBase10 [1, 2, 3, 4] 7 : Rat)

theorem calculation_result : result = 898.5 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l3567_356709


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l3567_356787

theorem number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 5865863355 ∧ x = 586650 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l3567_356787


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l3567_356794

theorem lcm_gcf_ratio : (Nat.lcm 144 756) / (Nat.gcd 144 756) = 84 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l3567_356794


namespace NUMINAMATH_CALUDE_max_a_for_integer_solutions_l3567_356716

theorem max_a_for_integer_solutions : 
  (∃ (a : ℕ+), ∀ (x : ℤ), x^2 + a*x = -30 → 
    (∀ (b : ℕ+), (∀ (y : ℤ), y^2 + b*y = -30 → b ≤ a))) ∧
  (∃ (x : ℤ), x^2 + 31*x = -30) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_integer_solutions_l3567_356716


namespace NUMINAMATH_CALUDE_instrument_players_l3567_356772

theorem instrument_players (total_people : ℕ) 
  (at_least_one_ratio : ℚ) (exactly_one_prob : ℝ) 
  (h1 : total_people = 800)
  (h2 : at_least_one_ratio = 1/5)
  (h3 : exactly_one_prob = 0.12) : 
  ℕ := by
  sorry

#check instrument_players

end NUMINAMATH_CALUDE_instrument_players_l3567_356772


namespace NUMINAMATH_CALUDE_square_window_side_length_l3567_356770

/-- Given three rectangles with perimeters 8, 10, and 12 that form a square window,
    prove that the side length of the square window is 4. -/
theorem square_window_side_length 
  (a b c : ℝ) 
  (h1 : 2*b + 2*c = 8)   -- perimeter of bottom-left rectangle
  (h2 : 2*(a - b) + 2*a = 10) -- perimeter of top rectangle
  (h3 : 2*b + 2*(a - c) = 12) -- perimeter of right rectangle
  : a = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_window_side_length_l3567_356770


namespace NUMINAMATH_CALUDE_complex_distance_l3567_356711

theorem complex_distance (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 2)
  (h₂ : Complex.abs z₁ = Real.sqrt 3)
  (h₃ : Complex.abs z₂ = Real.sqrt 2) :
  Complex.abs (z₁ - z₂) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_l3567_356711


namespace NUMINAMATH_CALUDE_math_club_attendance_l3567_356701

theorem math_club_attendance (total_sessions : Nat) (students_per_session : Nat)
  (three_session_attendees : Nat) (two_session_attendees : Nat) (one_session_attendees : Nat)
  (h1 : total_sessions = 4)
  (h2 : students_per_session = 20)
  (h3 : three_session_attendees = 9)
  (h4 : two_session_attendees = 5)
  (h5 : one_session_attendees = 3) :
  ∃ (all_session_attendees : Nat),
    all_session_attendees * total_sessions +
    three_session_attendees * 3 +
    two_session_attendees * 2 +
    one_session_attendees * 1 =
    total_sessions * students_per_session ∧
    all_session_attendees = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_club_attendance_l3567_356701


namespace NUMINAMATH_CALUDE_least_frood_number_l3567_356783

def droppingScore (n : ℕ) : ℕ := n * (n + 1) / 2
def eatingScore (n : ℕ) : ℕ := n^2

theorem least_frood_number : 
  (∀ k < 21, droppingScore k ≤ eatingScore k) ∧ 
  (droppingScore 21 > eatingScore 21) := by sorry

end NUMINAMATH_CALUDE_least_frood_number_l3567_356783


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3567_356757

/-- A trapezoid with the given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  angle_between_diagonals : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given properties has a perimeter of 22 -/
theorem trapezoid_perimeter (t : Trapezoid) 
  (h1 : t.base1 = 3)
  (h2 : t.base2 = 5)
  (h3 : t.diagonal = 8)
  (h4 : t.angle_between_diagonals = 60 * π / 180) :
  perimeter t = 22 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3567_356757


namespace NUMINAMATH_CALUDE_convention_handshakes_count_l3567_356799

/-- Represents the number of handshakes at a convention with twins and triplets -/
def convention_handshakes (twin_sets triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2) / 2
  let triplet_handshakes := triplets * (triplets - 3) / 2
  let cross_handshakes := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

/-- The number of handshakes at the convention is 900 -/
theorem convention_handshakes_count : convention_handshakes 12 8 = 900 := by
  sorry

#eval convention_handshakes 12 8

end NUMINAMATH_CALUDE_convention_handshakes_count_l3567_356799


namespace NUMINAMATH_CALUDE_trig_identity_l3567_356759

theorem trig_identity (α : Real) (h : Real.sin (π / 4 + α) = 1 / 2) :
  Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α) * Real.cos (7 * π / 4 - α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3567_356759


namespace NUMINAMATH_CALUDE_ma_xiaotiao_rank_l3567_356747

theorem ma_xiaotiao_rank (total_participants : ℕ) (ma_rank : ℕ) : 
  total_participants = 34 →
  ma_rank > 0 →
  ma_rank ≤ total_participants →
  total_participants - ma_rank = 2 * (ma_rank - 1) →
  ma_rank = 12 := by
  sorry

end NUMINAMATH_CALUDE_ma_xiaotiao_rank_l3567_356747


namespace NUMINAMATH_CALUDE_shirt_fabric_sum_l3567_356776

theorem shirt_fabric_sum (a : ℝ) (r : ℝ) (h1 : a = 2011) (h2 : r = 4/5) (h3 : r < 1) :
  a / (1 - r) = 10055 := by
  sorry

end NUMINAMATH_CALUDE_shirt_fabric_sum_l3567_356776


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3567_356746

/-- For the equation (m-1)x^2 + mx - 1 = 0 to be a quadratic equation in x,
    m must not equal 1. -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, (m - 1) * x^2 + m * x - 1 = 0 → (m - 1) ≠ 0) ↔ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3567_356746


namespace NUMINAMATH_CALUDE_mean_of_car_counts_l3567_356745

theorem mean_of_car_counts : 
  let counts : List ℝ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℝ) = 20.8 := by
sorry

end NUMINAMATH_CALUDE_mean_of_car_counts_l3567_356745


namespace NUMINAMATH_CALUDE_house_savings_l3567_356735

theorem house_savings (total_savings : ℕ) (years : ℕ) (people : ℕ) : 
  total_savings = 108000 → 
  years = 3 → 
  people = 2 → 
  (total_savings / (years * 12)) / people = 1500 := by
sorry

end NUMINAMATH_CALUDE_house_savings_l3567_356735


namespace NUMINAMATH_CALUDE_min_value_theorem_l3567_356739

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 108/x^4 ≥ 36 ∧ 
  (x^2 + 12*x + 108/x^4 = 36 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3567_356739


namespace NUMINAMATH_CALUDE_solution_set_intersection_condition_l3567_356797

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem for part (1)
theorem solution_set (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part (2)
theorem intersection_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m y = g x) ↔ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_intersection_condition_l3567_356797


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l3567_356767

/-- The path length of a vertex of an equilateral triangle rotating around a square --/
theorem triangle_rotation_path_length 
  (square_side : ℝ) 
  (triangle_side : ℝ) 
  (h_square : square_side = 6) 
  (h_triangle : triangle_side = 3) : 
  let path_length := 4 * 3 * (2 * π * triangle_side / 3)
  path_length = 24 * π := by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l3567_356767


namespace NUMINAMATH_CALUDE_survey_result_l3567_356756

/-- Represents the number of questionnaires collected from each unit -/
structure QuestionnaireData where
  total : ℕ
  sample : ℕ
  sample_b : ℕ

/-- Proves that given the conditions from the survey, the number of questionnaires drawn from unit D is 60 -/
theorem survey_result (data : QuestionnaireData) 
  (h_total : data.total = 1000)
  (h_sample : data.sample = 150)
  (h_sample_b : data.sample_b = 30)
  (h_arithmetic : ∃ (a d : ℚ), ∀ i : Fin 4, a + i * d = (data.total : ℚ) / 4)
  (h_prop_arithmetic : ∃ (b e : ℚ), ∀ i : Fin 4, b + i * e = (data.sample : ℚ) / 4 ∧ b + 1 * e = data.sample_b) :
  ∃ (b e : ℚ), b + 3 * e = 60 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l3567_356756


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3567_356719

theorem power_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3567_356719


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3567_356771

/-- A function f: ℝ₊ → ℝ₊ satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

/-- The main theorem stating the solution to the functional equation. -/
theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : ℝ → ℝ, FunctionalEquation f α) ↔ (α = 1 ∧ ∃ f : ℝ → ℝ, FunctionalEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3567_356771
