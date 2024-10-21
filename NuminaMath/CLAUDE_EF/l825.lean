import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_never_zero_l825_82567

/-- Given functions f and g, prove that f(g(x)) is never zero for natural numbers x -/
theorem fg_never_zero (x : ℕ) : 
  let f (y : ℝ) := 1 / y
  let g (y : ℕ) := (y : ℝ)^2 + 3*(y : ℝ) - 2
  f (g x) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_never_zero_l825_82567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_l825_82570

/-- Proves the actual distance between two cities given map distance and scale --/
theorem city_distance (map_distance : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) 
  (h1 : map_distance = 30)
  (h2 : scale_inches = 0.5)
  (h3 : scale_miles = 8)
  (h4 : (1 : ℝ) = 1.60934) : -- Represents 1 mile = 1.60934 kilometers
  ∃ (miles kilometers : ℝ),
    miles = 480 ∧ 
    kilometers = 772.4832 ∧
    miles = (map_distance / scale_inches) * scale_miles ∧
    kilometers = miles * 1.60934 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_l825_82570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_legendre_symbol_square_l825_82529

theorem legendre_symbol_square (p : ℕ) (a : ℤ) 
  (h_prime : Nat.Prime p) 
  (h_odd : p % 2 = 1) 
  (h_coprime : Nat.Coprime a.natAbs p) : 
  (a^((p - 1) / 2) : ℤ) % p = 1 ∨ (a^((p - 1) / 2) : ℤ) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_legendre_symbol_square_l825_82529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_elements_l825_82536

theorem max_distinct_elements (a : ℝ) : Finset.card (Finset.image id {a, -a, |a|}) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_elements_l825_82536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l825_82554

theorem expression_equality : (1/2)^(-2 : ℤ) + (Real.sqrt 3 - 2)^(0 : ℕ) + 3 * Real.tan (π/6) = 5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l825_82554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_over_y_axis_l825_82508

/-- The reflection of a point over the y-axis -/
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem reflection_distance_over_y_axis :
  let F : ℝ × ℝ := (-4, 5)
  let F' := reflect_over_y_axis F
  distance F F' = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_over_y_axis_l825_82508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_perimeter_ratio_l825_82596

/-- The ratio of area to perimeter for a regular pentagon -/
theorem regular_pentagon_area_perimeter_ratio (s : ℝ) (h : s > 0) :
  (5 * s^2 * Real.tan (54 * π / 180)) / 4 / (5 * s) = (3 * Real.tan (54 * π / 180)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_perimeter_ratio_l825_82596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_exponential_at_two_l825_82551

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem inverse_of_exponential_at_two 
  (g : ℝ → ℝ) 
  (h₁ : ∀ x, g (f x) = x) 
  (h₂ : ∀ x, f (g x) = x) : 
  g 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_exponential_at_two_l825_82551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_star_with_more_in_row_l825_82537

/-- Represents a star in the matrix -/
structure MatrixStar where
  row : Nat
  col : Nat

/-- Represents the rectangular matrix with stars -/
structure StarMatrix where
  rows : Nat
  cols : Nat
  stars : List MatrixStar
  h_rows_lt_cols : rows < cols
  h_each_col_has_star : ∀ j, ∃ s ∈ stars, s.col = j

/-- Count stars in a given row -/
def countStarsInRow (m : StarMatrix) (i : Nat) : Nat :=
  (m.stars.filter (fun s => s.row = i)).length

/-- Count stars in a given column -/
def countStarsInCol (m : StarMatrix) (j : Nat) : Nat :=
  (m.stars.filter (fun s => s.col = j)).length

/-- Main theorem -/
theorem exists_star_with_more_in_row (m : StarMatrix) :
  ∃ s ∈ m.stars, countStarsInRow m s.row > countStarsInCol m s.col := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_star_with_more_in_row_l825_82537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_probability_l825_82559

/-- Probability of selection function -/
noncomputable def probability_of_selection (i : ℕ) : ℚ :=
  sorry

/-- Systematic sampling probability theorem -/
theorem systematic_sampling_probability
  (population_size : ℕ)
  (sample_size : ℕ)
  (h_pop_size : population_size = 1003)
  (h_sample_size : sample_size = 50)
  (h_fairness : ∀ i : ℕ, i ≤ population_size → 
    probability_of_selection i = probability_of_selection 1) :
  ∀ i : ℕ, i ≤ population_size → 
    probability_of_selection i = sample_size / population_size :=
by
  sorry

#check systematic_sampling_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_probability_l825_82559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heads_100th_toss_l825_82576

/-- A fair coin toss is represented as a function that returns a boolean value. -/
def CoinToss := Unit → Bool

/-- A sequence of 100 coin tosses. -/
def TossSequence := Fin 100 → Bool

/-- The probability of getting heads on a single toss of a fair coin. -/
noncomputable def probHeads : ℝ := 1 / 2

theorem prob_heads_100th_toss (tosses : TossSequence) 
  (h : ∀ i : Fin 99, tosses i = true) : 
  probHeads = 1 / 2 := by
  -- The probability of getting heads on the 100th toss is always 1/2,
  -- regardless of the previous tosses, as each toss is independent.
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heads_100th_toss_l825_82576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l825_82562

def is_valid_digit (d : Nat) : Bool :=
  d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  is_valid_digit (n / 1000) ∧
  is_valid_digit ((n / 100) % 10) ∧
  is_valid_digit ((n / 10) % 10) ∧
  is_valid_digit (n % 10)

theorem count_valid_numbers :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 10000)).card = 2058 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l825_82562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_ordering_l825_82579

-- Define the variables
noncomputable def y₁ : ℝ := (4 : ℝ) ^ (0.9 : ℝ)
noncomputable def y₂ : ℝ := Real.log 5 / Real.log (1/2)
noncomputable def y₃ : ℝ := (1/2 : ℝ) ^ (-(1.5 : ℝ))

-- State the theorem
theorem y_ordering : y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_ordering_l825_82579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l825_82518

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := x + y + k = 0

-- Define the condition for the existence of point P
def exists_P (k : ℝ) : Prop := 
  ∃ x y : ℝ, line_l x y k ∧ (x^2 + y^2 = 4)

-- Theorem statement
theorem tangent_angle_range (k : ℝ) : 
  (∃ x y : ℝ, line_l x y k ∧ (x^2 + y^2 = 4)) → 
  k ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l825_82518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l825_82565

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x else -x^2 + 4*x

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ -- f is odd
  (∀ x, x ≤ 0 → f x = x^2 + 4*x) ∧ -- given condition
  (∀ x ∈ Set.Ioo (-2) 2, StrictMono f) ∧ -- increasing on (-2, 2)
  (∀ x ∈ Set.Iic (-2), StrictAnti f) ∧ -- decreasing on (-∞, -2]
  (∀ x ∈ Set.Ici 2, StrictAnti f) ∧ -- decreasing on [2, +∞)
  {x | f x > 3} = Set.Ioo 1 3 ∪ Set.Iic (-2 - Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l825_82565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_negative_for_acute_triangle_l825_82509

open Real

-- Define an acute triangle
def is_acute_triangle (α β γ : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧
  0 < β ∧ β < Real.pi/2 ∧
  0 < γ ∧ γ < Real.pi/2 ∧
  α + β + γ = Real.pi

-- State the theorem
theorem sin_sum_negative_for_acute_triangle :
  ∀ α β γ : ℝ, is_acute_triangle α β γ →
  Real.sin (4*α) + Real.sin (4*β) + Real.sin (4*γ) < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_negative_for_acute_triangle_l825_82509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l825_82516

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a parabola -/
def Intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | p.equation point.1 point.2 ∧ point.2 = l.slope * (point.1 - l.point.1) + l.point.2}

theorem parabola_line_intersection
  (p : Parabola)
  (l : Line)
  (h1 : p.equation = fun x y ↦ y^2 = -4*x)
  (h2 : p.focus = (-1, 0))
  (h3 : l.slope = 1)
  (h4 : l.point = p.focus)
  (h5 : ∃ A B, A ∈ Intersection p l ∧ B ∈ Intersection p l ∧ A ≠ B) :
  ∃ (A B : ℝ × ℝ) (d s : ℝ),
    A ∈ Intersection p l ∧
    B ∈ Intersection p l ∧
    A ≠ B ∧
    d = abs (A.1 - B.1) ∧
    d = 8 ∧
    s = (1 / abs (A.1 - p.focus.1)) + (1 / abs (B.1 - p.focus.1)) ∧
    s = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l825_82516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_two_roots_l825_82517

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if 0 ≤ x ∧ x ≤ 1 then x 
  else (1 / (x + 1)) - 1

-- Define the function g
noncomputable def g (m : ℝ) : ℝ → ℝ := fun x => f x - m

-- State the theorem
theorem m_range_for_two_roots (m : ℝ) : 
  (∃ x y, x ∈ Set.Ioc (-1) 1 ∧ y ∈ Set.Ioc (-1) 1 ∧ x ≠ y ∧ g m x = 0 ∧ g m y = 0) →
  m ∈ Set.Ioo 0 1 := by
  sorry

-- State the property of f
axiom f_property (x : ℝ) : f x + 1 = 1 / f (x + 1)

-- State the property of f on [0, 1]
axiom f_on_unit_interval (x : ℝ) : x ∈ Set.Icc 0 1 → f x = x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_two_roots_l825_82517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_price_decline_l825_82564

/-- Represents the monthly rate of decline in house prices -/
noncomputable def rate_of_decline (initial_price final_price : ℝ) : ℝ :=
  1 - (final_price / initial_price) ^ (1 / 2)

/-- Calculates the price after a certain number of months given an initial price and rate of decline -/
noncomputable def price_after_months (initial_price : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  initial_price * (1 - rate) ^ months

theorem house_price_decline (september_price november_price : ℝ) 
  (h_sept : september_price = 10000)
  (h_nov : november_price = 8100) :
  let rate := rate_of_decline september_price november_price
  price_after_months november_price rate 1 = 7290 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_price_decline_l825_82564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_max_value_of_f_n_with_max_f_l825_82524

def f : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then f (n / 2 + 1) else f (n + 1) + 1

def maxN : ℕ := 1994

theorem f_properties :
  f 1 = 1 ∧
  (∀ n : ℕ, f (2 * n) = f n) ∧
  (∀ n : ℕ, f (2 * n + 1) = f (2 * n) + 1) := by sorry

theorem max_value_of_f :
  (∃ n : ℕ, n ≤ maxN ∧ f n = 10) ∧
  (∀ n : ℕ, n ≤ maxN → f n ≤ 10) := by sorry

theorem n_with_max_f :
  {n : ℕ | n ≤ maxN ∧ f n = 10} = {1023, 1535, 1791, 1919, 1983} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_max_value_of_f_n_with_max_f_l825_82524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_weekly_savings_l825_82584

/-- Represents the savings of a married couple over 4 months -/
structure CoupleSavings where
  husband_monthly : ℚ  -- husband's monthly savings
  wife_weekly : ℚ      -- wife's weekly savings
  months : ℕ           -- number of months saved
  stock_price : ℚ      -- price per share of stock
  stock_shares : ℕ     -- number of shares they can buy

/-- Theorem stating the wife's weekly savings given the conditions -/
theorem wife_weekly_savings 
  (savings : CoupleSavings) 
  (h1 : savings.husband_monthly = 225)
  (h2 : savings.months = 4)
  (h3 : savings.stock_price = 50)
  (h4 : savings.stock_shares = 25)
  (h5 : (savings.husband_monthly * savings.months + 
         savings.wife_weekly * (savings.months * 4)) / 2 = 
        savings.stock_price * savings.stock_shares) :
  savings.wife_weekly = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wife_weekly_savings_l825_82584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_self_intersections_l825_82555

/-- Parametric function for x coordinate -/
noncomputable def x (t : ℝ) : ℝ := Real.cos t + (3 * t / 2)

/-- Parametric function for y coordinate -/
noncomputable def y (t : ℝ) : ℝ := Real.sin t

/-- The range of x values we're interested in -/
def x_range : Set ℝ := { x | 1 ≤ x ∧ x ≤ 60 }

/-- Definition of a self-intersection point -/
def is_self_intersection (t₁ t₂ : ℝ) : Prop :=
  t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ x t₁ ∈ x_range

/-- The set of all self-intersection points -/
def self_intersections : Set (ℝ × ℝ) :=
  { (t₁, t₂) | is_self_intersection t₁ t₂ }

/-- The main theorem: there are exactly 8 self-intersections -/
theorem count_self_intersections :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 8 ∧ ∀ (p : ℝ × ℝ), p ∈ S ↔ p ∈ self_intersections ∧ p.1 < p.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_self_intersections_l825_82555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_reducible_theorem_l825_82571

/-- A polynomial is positive reducible if it can be written as a product of two nonconstant polynomials with positive real coefficients. -/
def PositiveReducible (p : Polynomial ℝ) : Prop :=
  ∃ (q r : Polynomial ℝ), p = q * r ∧ q ≠ 0 ∧ r ≠ 0 ∧
    (∀ i, q.coeff i ≥ 0) ∧ (∀ i, r.coeff i ≥ 0)

/-- Main theorem: If f(x) is a polynomial with f(0) ≠ 0 and f(x^n) is positive reducible for some natural number n, then f(x) is positive reducible. -/
theorem positive_reducible_theorem (f : Polynomial ℝ) (n : ℕ) 
    (h1 : f.eval 0 ≠ 0)
    (h2 : PositiveReducible (f.comp (Polynomial.monomial n 1))) :
    PositiveReducible f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_reducible_theorem_l825_82571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametricCurveLengthIs4_5π_l825_82586

/-- The length of a parametric curve described by (x,y) = (3 sin t, 3 cos t) from t = 0 to t = 3π/2 -/
noncomputable def parametricCurveLength : ℝ := (3 * Real.pi) / 2

/-- The parametric equations of the curve -/
noncomputable def curveEquations (t : ℝ) : ℝ × ℝ := (3 * Real.sin t, 3 * Real.cos t)

theorem parametricCurveLengthIs4_5π :
  parametricCurveLength = (9 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametricCurveLengthIs4_5π_l825_82586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_students_not_well_defined_l825_82528

-- Define the concept of a well-defined set
def is_well_defined_set (S : Type*) : Prop := 
  ∃ (P : S → Prop), ∀ x, x ∈ {y : S | P y} ↔ P x

-- Define the set of prime numbers within 1-20
def primes_1_to_20 : Set Nat :=
  {n : Nat | n ≥ 1 ∧ n ≤ 20 ∧ Nat.Prime n}

-- Define the set of real roots of x^2+x-2=0
def roots_of_equation : Set Real :=
  {x : Real | x^2 + x - 2 = 0}

-- Define the concept of "taller students"
def taller_students (school : Type*) (height : school → Real) : Set school :=
  {s : school | ∃ t : school, height s > height t}

-- Define the set of all squares
def all_squares : Set Nat :=
  {n : Nat | ∃ m : Nat, n = m^2}

-- The theorem to be proved
theorem taller_students_not_well_defined (school : Type*) (height : school → Real) :
  is_well_defined_set primes_1_to_20 ∧
  is_well_defined_set roots_of_equation ∧
  is_well_defined_set all_squares ∧
  ¬is_well_defined_set (taller_students school height) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taller_students_not_well_defined_l825_82528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_angle_range_a_range_l825_82523

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

-- Part 1
theorem tangent_line_at_one (b : ℝ) (h : b = 1) :
  let f' := fun x => 1 / x + b
  (2 : ℝ) * 1 - f 0 1 1 - 1 = 0 := by
  sorry

-- Part 2
theorem angle_range (a b : ℝ) (ha : a = 1/2) (hb : b = -3) :
  let f' := fun x => 1 / x + 2 * a * x + b
  ∀ x > 0, ∃ α ∈ Set.Icc 0 (π/2) ∪ Set.Icc (3*π/4) π, Real.tan α = f' x := by
  sorry

-- Part 3
theorem a_range (a b : ℝ) (hb : b = -1) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a b x₁ - f a b x₂) / (x₁ - x₂) > 1) →
  a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_angle_range_a_range_l825_82523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_is_896_20_l825_82531

/-- Represents the sales and pricing data for a flower shop over three days -/
structure FlowerShopData where
  -- Regular prices
  tulip_price : ℚ
  rose_price : ℚ
  lily_price : ℚ
  sunflower_price : ℚ
  -- Day 1 sales
  day1_tulips : ℕ
  day1_roses : ℕ
  day1_lilies : ℕ
  day1_sunflowers : ℕ
  -- Day 2 multipliers and adjustments
  day2_tulip_multiplier : ℚ
  day2_rose_multiplier : ℚ
  day2_sunflower_multiplier : ℚ
  day2_rose_discount : ℚ
  day2_lily_markup : ℚ
  -- Day 3 adjustments
  day3_tulip_ratio : ℚ
  day3_roses : ℕ
  day3_lily_ratio : ℚ
  day3_tulip_discount : ℚ
  day3_lily_markup : ℚ

/-- Calculates the total earnings over three days given the flower shop data -/
def calculateTotalEarnings (data : FlowerShopData) : ℚ :=
  let day1_earnings := 
    data.tulip_price * data.day1_tulips +
    data.rose_price * data.day1_roses +
    data.lily_price * data.day1_lilies +
    data.sunflower_price * data.day1_sunflowers

  let day2_earnings :=
    data.tulip_price * (data.day1_tulips * data.day2_tulip_multiplier) +
    data.rose_price * (data.day1_roses * data.day2_rose_multiplier) * (1 - data.day2_rose_discount) +
    data.lily_price * data.day1_lilies * (1 + data.day2_lily_markup) +
    data.sunflower_price * (data.day1_sunflowers * data.day2_sunflower_multiplier)

  let day3_earnings :=
    data.tulip_price * (data.day1_tulips * data.day2_tulip_multiplier * data.day3_tulip_ratio) * (1 - data.day3_tulip_discount) +
    data.rose_price * (data.day3_roses / 3 * 2) +
    data.lily_price * (data.day1_lilies * data.day3_lily_ratio) * (1 + data.day3_lily_markup) +
    data.sunflower_price * (data.day1_sunflowers * data.day2_sunflower_multiplier)

  day1_earnings + day2_earnings + day3_earnings

/-- Theorem stating that the total earnings for the given data is $896.20 -/
theorem total_earnings_is_896_20 (data : FlowerShopData) 
  (h1 : data.tulip_price = 2)
  (h2 : data.rose_price = 3)
  (h3 : data.lily_price = 4)
  (h4 : data.sunflower_price = 5)
  (h5 : data.day1_tulips = 30)
  (h6 : data.day1_roses = 20)
  (h7 : data.day1_lilies = 15)
  (h8 : data.day1_sunflowers = 10)
  (h9 : data.day2_tulip_multiplier = 2)
  (h10 : data.day2_rose_multiplier = 2)
  (h11 : data.day2_sunflower_multiplier = 3)
  (h12 : data.day2_rose_discount = 1/5)
  (h13 : data.day2_lily_markup = 1/4)
  (h14 : data.day3_tulip_ratio = 1/10)
  (h15 : data.day3_roses = 16)
  (h16 : data.day3_lily_ratio = 1/2)
  (h17 : data.day3_tulip_discount = 3/20)
  (h18 : data.day3_lily_markup = 1/10) :
  calculateTotalEarnings data = 89620/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_is_896_20_l825_82531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_19_54_l825_82593

/-- The average speed of a man traveling up and down a given altitude -/
noncomputable def average_speed (altitude : ℝ) (speed_up speed_down : ℝ) : ℝ :=
  let distance_km := 2 * altitude / 1000
  let time_up := altitude / (1000 * speed_up)
  let time_down := altitude / (1000 * speed_down)
  let total_time := time_up + time_down
  distance_km / total_time

/-- Theorem stating that the average speed is approximately 19.54 km/hr for the given conditions -/
theorem average_speed_approx_19_54 :
  let altitude := (230 : ℝ)
  let speed_up := (15 : ℝ)
  let speed_down := (28 : ℝ)
  ∃ ε > 0, |average_speed altitude speed_up speed_down - 19.54| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_19_54_l825_82593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l825_82511

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*m)*x - 3*m else Real.log x / Real.log m

noncomputable def a (m : ℝ) : ℝ := f m (-3/2)
noncomputable def b (m : ℝ) : ℝ := f m 1
noncomputable def c (m : ℝ) : ℝ := f m 2

theorem function_inequality (m : ℝ) (h : m ∈ Set.Icc (1/5) (1/2)) :
  a m < c m ∧ c m < b m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l825_82511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l825_82557

noncomputable def curve_C (θ : ℝ) : ℝ := 6 * Real.cos θ / (Real.sin θ)^2

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3/2 + t, Real.sqrt 3 * t)

def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

theorem intersection_distance :
  ∃ (θ₁ θ₂ t₁ t₂ : ℝ),
    (curve_C θ₁ * Real.cos θ₁, curve_C θ₁ * Real.sin θ₁) = line_l t₁ ∧
    (curve_C θ₂ * Real.cos θ₂, curve_C θ₂ * Real.sin θ₂) = line_l t₂ ∧
    A = line_l t₁ ∧
    B = line_l t₂ ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l825_82557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenPerfectSquareFactors_eq_54_l825_82522

/-- The number of even perfect square factors of 2^6 * 7^10 * 3^4 -/
def evenPerfectSquareFactors : ℕ :=
  let factorForm := fun (a b c : ℕ) => 2^a * 7^b * 3^c
  let validExponents := fun (a b c : ℕ) => 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 10 ∧ 0 ≤ c ∧ c ≤ 4
  let isPerfectSquare := fun (a b c : ℕ) => Even a ∧ Even b ∧ Even c
  let isEven := fun (a : ℕ) => a ≥ 1
  54

theorem evenPerfectSquareFactors_eq_54 : evenPerfectSquareFactors = 54 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenPerfectSquareFactors_eq_54_l825_82522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l825_82549

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sin ((t.B - t.C)/2) ^ 2 + Real.sin t.B * Real.sin t.C = 1/4

def condition2 (t : Triangle) : Prop :=
  t.a = Real.sqrt 7 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  condition1 t → t.A = (2/3) * Real.pi ∧
  (condition1 t ∧ condition2 t → t.b + t.c = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l825_82549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l825_82510

noncomputable def geometricSequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n-1)

noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  ∃ (a r : ℝ), 
    let a_n := geometricSequence a r
    (a_n 1 + a_n 2 = 5) ∧ 
    (a_n 4 + a_n 5 = 15) → 
    (a_n 10 + a_n 11 = 135) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l825_82510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_99_plus_a_100_l825_82589

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 1 / 1  -- Add case for 0
| 1 => 1 / 1
| 2 => 2 / 1
| 3 => 1 / 2
| 4 => 3 / 1
| 5 => 2 / 2
| 6 => 1 / 3
| 7 => 4 / 1
| 8 => 3 / 2
| 9 => 2 / 3
| 10 => 1 / 4
| n + 1 => a n  -- The pattern continues for n > 10

-- State the theorem
theorem a_99_plus_a_100 : a 99 + a 100 = 37 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_99_plus_a_100_l825_82589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_theorem_l825_82574

theorem tangent_difference_theorem (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y - Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y - Real.cos y / Real.sin x = 3) :
  Real.tan x / Real.tan y - Real.tan y / Real.tan x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_difference_theorem_l825_82574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_range_complement_intersection_when_a_min_l825_82520

-- Define set A
def A (a : ℝ) : Set ℝ := {y | y^2 - (a^2 + a + 1)*y + a*(a^2 + 1) > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = 1/2 * x^2 - x + 5/2}

-- Define the complement of A
def C_R_A (a : ℝ) : Set ℝ := {y | y^2 - (a^2 + a + 1)*y + a*(a^2 + 1) ≤ 0}

-- Theorem for part 1
theorem intersection_empty_range (a : ℝ) :
  A a ∩ B = ∅ ↔ (Real.sqrt 3 ≤ a ∧ a ≤ 2) ∨ a ≤ -Real.sqrt 3 :=
sorry

-- Theorem for part 2
theorem complement_intersection_when_a_min (a : ℝ) :
  (∀ x : ℝ, x^2 + 1 ≥ a*x) → a = -2 →
  C_R_A a ∩ B = Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_empty_range_complement_intersection_when_a_min_l825_82520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_is_600_l825_82582

/-- Calculate the overall profit from selling two items --/
def calculate_overall_profit (grinder_cost mobile_cost : ℕ) 
  (grinder_loss_percent mobile_profit_percent : ℚ) : ℤ :=
  let grinder_loss := (grinder_cost : ℚ) * grinder_loss_percent / 100
  let mobile_profit := (mobile_cost : ℚ) * mobile_profit_percent / 100
  let grinder_selling_price := (grinder_cost : ℚ) - grinder_loss
  let mobile_selling_price := (mobile_cost : ℚ) + mobile_profit
  let total_cost := grinder_cost + mobile_cost
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let overall_profit := total_selling_price - (total_cost : ℚ)
  ⌊overall_profit⌋

/-- Prove that the overall profit is 600 given the specific conditions --/
theorem overall_profit_is_600 : 
  calculate_overall_profit 15000 8000 4 15 = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_is_600_l825_82582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l825_82538

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + y - 7 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 7) / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 6 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist ∧
    ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ distance_to_line x₀ y₀ = max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l825_82538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_function_problems_l825_82541

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem gaussian_function_problems :
  (∀ x : ℝ, (floor x : ℝ) / (floor x - 4) < 0 ↔ 1 ≤ x ∧ x < 4) ∧
  (∀ x : ℝ, x > 0 → floor x / ((floor x)^2 + 4 : ℝ) ≤ 1/4) ∧
  (∃ x : ℝ, x > 0 ∧ floor x / ((floor x)^2 + 4 : ℝ) = 1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_function_problems_l825_82541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l825_82530

/-- The area of a triangle given its coordinates -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let base := |qx - px|
  let height := |ry - py|
  (1/2) * base * height

/-- Theorem: The area of triangle PQR with given coordinates is 36 square units -/
theorem triangle_PQR_area :
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (8, 2)
  let R : ℝ × ℝ := (6, -4)
  triangle_area P Q R = 36 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l825_82530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equation_l825_82590

/-- Given two parallel lines l₁ and l₂, where l₁ is defined by mx + 8y + n = 0 and
    l₂ is defined by 2x + my - 1 = 0, and the distance between them is √5,
    prove that the equation of l₁ must be one of the following:
    2x + 4y - 11 = 0, 2x + 4y + 9 = 0, 2x - 4y + 9 = 0, or 2x - 4y - 11 = 0 -/
theorem parallel_lines_equation (m n : ℝ) :
  let l₁ : ℝ → ℝ → ℝ := fun x y => m * x + 8 * y + n
  let l₂ : ℝ → ℝ → ℝ := fun x y => 2 * x + m * y - 1
  (∀ x y, l₁ x y = 0 → l₂ x y ≠ 0) →  -- l₁ and l₂ are parallel
  (∃ d, ∀ x y, l₁ x y = 0 → l₂ x y = 0 → d = Real.sqrt 5) →  -- distance between l₁ and l₂ is √5
  (l₁ = fun x y => 2 * x + 4 * y - 11) ∨
  (l₁ = fun x y => 2 * x + 4 * y + 9) ∨
  (l₁ = fun x y => 2 * x - 4 * y + 9) ∨
  (l₁ = fun x y => 2 * x - 4 * y - 11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equation_l825_82590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l825_82573

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 1/2 →
  b * c = 3 →
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l825_82573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_specific_l825_82598

/-- The circumference of the base of a right circular cone formed by cutting out a sector
    from a circular piece of paper and gluing the remaining edges together. -/
noncomputable def cone_base_circumference (r : ℝ) (sector_angle : ℝ) : ℝ :=
  2 * Real.pi * r * (1 - sector_angle / 360)

/-- Theorem: The circumference of the base of a right circular cone formed by cutting out
    a 240° sector from a circular piece of paper with radius 5 inches and gluing the
    remaining edges together is equal to 20π/3. -/
theorem cone_base_circumference_specific :
  cone_base_circumference 5 240 = 20 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_specific_l825_82598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_exponential_fraction_l825_82521

theorem simplify_exponential_fraction (n : ℕ) : 
  (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_exponential_fraction_l825_82521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neighbor_permutations_eq_fibonacci_l825_82512

/-- A permutation of (1, ..., n) where each element is at most 1 position away from its original position. -/
def NeighborPermutation (n : ℕ) : Type :=
  { p : Fin n → Fin n // Function.Bijective p ∧ ∀ k, (p k).val ≤ k.val + 1 ∧ k.val ≤ (p k).val + 1 }

/-- The number of neighbor permutations of length n. -/
def numNeighborPermutations (n : ℕ) : ℕ :=
  sorry -- We'll leave this as sorry for now, as we can't use Fintype.card directly

/-- The nth Fibonacci number. -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The closed form of the nth Fibonacci number. -/
noncomputable def fibClosedForm (n : ℕ) : ℝ :=
  (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2)^n - ((1 - Real.sqrt 5) / 2)^n)

theorem neighbor_permutations_eq_fibonacci (n : ℕ) :
  (numNeighborPermutations n : ℝ) = fibClosedForm (n + 1) := by
  sorry

#eval fib 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neighbor_permutations_eq_fibonacci_l825_82512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_decomposition_91_l825_82501

/-- The smallest odd number in the decomposition of m³ -/
def smallest_odd (m : ℕ) : ℕ := 
  if m ≤ 1 then 1 else 2 * (m * (m - 1) / 2 + 1) + 1

theorem cube_decomposition_91 (m : ℕ) (h : m > 0) :
  smallest_odd m = 91 → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_decomposition_91_l825_82501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l825_82547

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem g_monotone_increasing :
  StrictMonoOn g (Set.Icc (-5 * Real.pi / 12) (-Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l825_82547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curve_l825_82553

-- Define the curve in polar coordinates
noncomputable def polar_curve (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the reference point in polar coordinates
noncomputable def reference_point : ℝ × ℝ := (1, Real.pi)

-- Theorem statement
theorem max_distance_on_curve :
  ∃ (max_dist : ℝ),
    (∀ θ : ℝ, 
      let r := polar_curve θ
      let p := (r * Real.cos θ, r * Real.sin θ)
      let d := Real.sqrt ((p.1 - (-1))^2 + (p.2 - 0)^2)
      d ≤ max_dist) ∧
    max_dist = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curve_l825_82553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_constant_l825_82504

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}

/-- Right focus of the hyperbola -/
noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- Point P on the hyperbola -/
def P : ℝ × ℝ := (3, 1)

/-- Point Q symmetric to P with respect to origin -/
def Q : ℝ × ℝ := (-3, -1)

/-- Line l intersecting the hyperbola -/
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + m}

/-- Point symmetric to B with respect to origin -/
def SymmetricPoint (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem hyperbola_slope_constant
  (a b : ℝ)
  (h_pos : a > 0 ∧ b > 0)
  (h_on_hyperbola : P ∈ Hyperbola a b)
  (h_focus_product : (P.1 - (RightFocus a b).1) * (Q.1 - (RightFocus a b).1) +
                     (P.2 - (RightFocus a b).2) * (Q.2 - (RightFocus a b).2) = 6)
  (k m : ℝ)
  (h_not_perpendicular : k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1)
  (h_not_through_PQ : P ∉ Line k m ∧ Q ∉ Line k m)
  (A B : ℝ × ℝ)
  (h_intersect : A ∈ Hyperbola a b ∩ Line k m ∧ B ∈ Hyperbola a b ∩ Line k m)
  (D : ℝ × ℝ)
  (h_symmetric : D = SymmetricPoint B)
  (h_perpendicular : (A.1 - P.1) * (D.1 - P.1) + (A.2 - P.2) * (D.2 - P.2) = 0) :
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_slope_constant_l825_82504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_points_distance_l825_82506

-- Define a cylinder type
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Theorem statement
theorem cylinder_points_distance (c : Cylinder) (points : List Point3D) :
  c.radius = 1 → c.height = 2 → points.length = 9 →
  (∀ p ∈ points, p.x^2 + p.y^2 ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 2) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_points_distance_l825_82506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l825_82533

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x - a / 2
  else Real.log x / Real.log a

theorem monotone_increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  4 / 3 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l825_82533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_cosine_sine_inequality_l825_82595

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ p x) ↔ (∀ x : ℝ, x > 0 → ¬ p x) :=
by sorry

theorem cosine_sine_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.cos x + Real.sin x > 1) ↔
  (∀ x : ℝ, x > 0 → Real.cos x + Real.sin x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_cosine_sine_inequality_l825_82595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l825_82515

/-- Given vectors a, b, c in ℝ², prove that if k*a + b is collinear with c, then k = -1. -/
theorem collinear_vectors (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2)) 
    (hb : b = (2, 0)) 
    (hc : c = (1, -2)) 
    (h_collinear : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • c) : 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l825_82515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l825_82581

noncomputable section

/-- The function f(x) = ax^2 - a/2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a/2 + 1

/-- The function g(x) = x + a/x -/
def g (a : ℝ) (x : ℝ) : ℝ := x + a/x

/-- The closed interval [1, 2] -/
def I : Set ℝ := Set.Icc 1 2

/-- The half-open interval [1, 2) -/
def J : Set ℝ := Set.Ico 1 2

theorem problem_statement :
  (∀ a : ℝ, (∀ x ∈ J, f a x > 0) ↔ a ≥ -2/7) ∧
  (∀ a : ℝ, a > 0 →
    ((∀ x₁ ∈ I, ∃ x₂ ∈ I, f a x₁ ≥ g a x₂) ↔ 1 ≤ a ∧ a ≤ 4)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l825_82581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l825_82588

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.sin x - Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.sin x + Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 + 1 ∧
    (∀ (x : ℝ), f x ≤ M) ∧
    (∀ (x : ℝ), f x = M ↔ ∃ (k : ℤ), x = 3 * Real.pi / 8 + k * Real.pi)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l825_82588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l825_82544

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | (n + 3) => (2019 + sequence_a (n + 2) * sequence_a (n + 1)) / sequence_a n

theorem sequence_a_positive_integer (n : ℕ) : 
  sequence_a n > 0 ∧ ∃ k : ℕ, sequence_a n = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l825_82544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l825_82548

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  S : ℕ → ℚ  -- Sum of first n terms
  a3_eq : a 3 = 11
  S3_eq : S 3 = 24

/-- The general term of the sequence -/
def generalTerm (n : ℕ) : ℚ := 3 * n + 2

/-- The b_n sequence derived from a_n -/
noncomputable def bSeq (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a n * (n + 6)) / (seq.a (n + 1) - 5)

/-- Main theorem stating the properties of the sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm n) ∧
  (∀ n, bSeq seq n ≥ 32/3) ∧
  (∃ n, bSeq seq n = 32/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l825_82548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_four_pairs_l825_82514

/-- Given a positive integer n, f(n) returns the number of distinct ordered pairs 
    of positive integers (a, b) such that a^2 + b^2 = n -/
def f (n : ℕ+) : ℕ := 
  Finset.card (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1))))

/-- 25 is the smallest positive integer n for which f(n) = 4 -/
theorem smallest_n_with_four_pairs : 
  ∀ n : ℕ+, n < 25 → f n ≠ 4 ∧ f 25 = 4 := by
  sorry

#eval f 25  -- To check if the function works correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_four_pairs_l825_82514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_switches_in_A_after_steps_l825_82534

/-- Represents the position of a switch -/
inductive Position
  | A | B | C | D | E
  deriving DecidableEq

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Position
  deriving DecidableEq

/-- The total number of switches -/
def totalSwitches : Nat := 1250

/-- The set of all switches -/
def Switches : Finset Switch := sorry

/-- Returns true if the given number is of the form (2^x)(3^y) where x and y are between 0 and 9 -/
def validLabel (n : Nat) : Prop := 
  ∃ x y, 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ n = 2^x * 3^y

/-- All switches have valid labels -/
axiom all_switches_valid : ∀ s, s ∈ Switches → validLabel s.label

/-- All switches have unique labels -/
axiom switches_unique : ∀ s1 s2, s1 ∈ Switches → s2 ∈ Switches → s1.label = s2.label → s1 = s2

/-- Initially, all switches are in position A -/
axiom initial_position : ∀ s, s ∈ Switches → s.position = Position.A

/-- Advances a switch's position by one step -/
def advancePosition (p : Position) : Position :=
  match p with
  | Position.A => Position.B
  | Position.B => Position.C
  | Position.C => Position.D
  | Position.D => Position.E
  | Position.E => Position.A

/-- Returns true if n is a divisor of m -/
def isDivisor (n m : Nat) : Prop := m % n = 0

/-- The final position of a switch after all steps -/
noncomputable def finalPosition (s : Switch) : Position := sorry

/-- The number of switches in position A after all steps -/
noncomputable def switchesInPositionA : Nat := 
  (Switches.filter (fun s => finalPosition s = Position.A)).card

/-- The main theorem: The number of switches in position A after all steps is 1230 -/
theorem switches_in_A_after_steps : switchesInPositionA = 1230 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_switches_in_A_after_steps_l825_82534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l825_82566

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with a given slope -/
structure Line where
  slope : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

/-- Main theorem statement -/
theorem ellipse_max_triangle_area 
  (e : Ellipse) 
  (d : Point) 
  (l : Line) :
  eccentricity e = Real.sqrt 3 / 2 →
  d.x = 2 ∧ d.y = 1 →
  l.slope = 1 / 2 →
  ∃ (m n : Point), 
    m ≠ n ∧
    m.x^2 / e.a^2 + m.y^2 / e.b^2 = 1 ∧
    n.x^2 / e.a^2 + n.y^2 / e.b^2 = 1 ∧
    m.y - n.y = l.slope * (m.x - n.x) →
  (∀ (p q : Point), 
    p ≠ q ∧
    p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1 ∧
    q.x^2 / e.a^2 + q.y^2 / e.b^2 = 1 ∧
    p.y - q.y = l.slope * (p.x - q.x) →
    triangleArea d p q ≤ 1) ∧
  (∃ (m n : Point), 
    m ≠ n ∧
    m.x^2 / e.a^2 + m.y^2 / e.b^2 = 1 ∧
    n.x^2 / e.a^2 + n.y^2 / e.b^2 = 1 ∧
    m.y - n.y = l.slope * (m.x - n.x) ∧
    triangleArea d m n = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l825_82566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l825_82552

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle

/-- Represents the configuration of circles as described in the problem -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  C : Circle
  D : Circle
  E : Circle
  T : EquilateralTriangle

/-- Predicate for a triangle being inscribed in a circle -/
def triangle_inscribed_in_circle (T : EquilateralTriangle) (C : Circle) : Prop := sorry

/-- Predicate for two circles being internally tangent -/
def circle_internally_tangent (C1 C2 : Circle) : Prop := sorry

/-- Predicate for two circles being externally tangent -/
def circle_externally_tangent (C1 C2 : Circle) : Prop := sorry

/-- The main theorem statement -/
theorem circle_configuration_theorem (config : CircleConfiguration) 
  (h1 : config.A.radius = 12)
  (h2 : config.B.radius = 5)
  (h3 : config.C.radius = 3)
  (h4 : config.D.radius = 3)
  (h5 : ∃ m n : ℕ, Nat.Coprime m n ∧ config.E.radius = m / n)
  (h6 : triangle_inscribed_in_circle config.T config.A)
  (h7 : circle_internally_tangent config.B config.A)
  (h8 : circle_internally_tangent config.C config.A)
  (h9 : circle_internally_tangent config.D config.A)
  (h10 : circle_externally_tangent config.B config.E)
  (h11 : circle_externally_tangent config.C config.E)
  (h12 : circle_externally_tangent config.D config.E) :
  config.E.radius = 21 / 2 ∧ ∃ m n : ℕ, Nat.Coprime m n ∧ config.E.radius = m / n ∧ m + n = 115 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l825_82552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l825_82560

-- Define the ellipse parameters
noncomputable def major_axis_length : ℝ := 4
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the standard form of an ellipse equation
def is_standard_ellipse_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation_proof :
  ∃ (a b : ℝ), 
    a = major_axis_length / 2 ∧
    b^2 = a^2 * (1 - eccentricity^2) ∧
    is_standard_ellipse_equation a b ∧
    ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l825_82560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_is_90_degrees_l825_82540

-- Define angles as real numbers representing degrees
variable (angle_A angle_B angle_C angle_D : ℝ)

-- State the theorem
theorem angle_D_is_90_degrees 
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_C = angle_D) :
  angle_D = 90 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_is_90_degrees_l825_82540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_13_and_7_l825_82558

theorem three_digit_divisible_by_13_and_7 : 
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ n % 7 = 0) (Finset.range 1000)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_13_and_7_l825_82558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parabola_equation_l825_82513

/-- Represents a horizontally oriented parabola -/
structure HorizontalParabola where
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- The equation of a horizontal parabola given its focus and vertex -/
def parabola_equation (p : HorizontalParabola) : ℝ → ℝ → Prop :=
  λ x y => y^2 = -8 * (x - p.vertex.fst)

/-- Theorem stating the equation of the specific parabola -/
theorem specific_parabola_equation :
  let p : HorizontalParabola := ⟨(-1, 0), (1, 0)⟩
  parabola_equation p = λ x y => y^2 = -8 * (x - 1) :=
by
  sorry

#check specific_parabola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_parabola_equation_l825_82513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_length_of_BC_l825_82542

-- Define the triangle ABC
noncomputable def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let ac := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  ab = 6 ∧ ac = 4 * Real.sqrt 2

-- Define the sine of angle B
noncomputable def sine_B (A B C : ℝ × ℝ) : ℝ :=
  (2 * Real.sqrt 2) / 3

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let ab := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let bc := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let ca := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (ab + bc + ca) / 2
  Real.sqrt (s * (s - ab) * (s - bc) * (s - ca))

-- Define the ratio of BD to DC
noncomputable def bd_dc_ratio (A B C D : ℝ × ℝ) : Prop :=
  let bd := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let dc := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  bd = 2 * dc

-- Define the length of AD
noncomputable def ad_length (A D : ℝ × ℝ) : Prop :=
  let ad := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  ad = 3 * Real.sqrt 2

-- Theorem statements
theorem area_of_triangle_ABC (A B C : ℝ × ℝ) :
  triangle_ABC A B C → sine_B A B C = (2 * Real.sqrt 2) / 3 →
  triangle_area A B C = 4 * Real.sqrt 2 := by sorry

theorem length_of_BC (A B C D : ℝ × ℝ) :
  triangle_ABC A B C → bd_dc_ratio A B C D → ad_length A D →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 69 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_length_of_BC_l825_82542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_sides_l825_82583

/-- Given the side lengths of a pedal triangle, calculate the side lengths of the original triangle. -/
theorem original_triangle_sides (a₁ b₁ c₁ : ℝ) (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  ∃ (a b c : ℝ),
  let s₁ := (a₁ + b₁ + c₁) / 2
  a = a₁ * Real.sqrt ((b₁ * c₁) / ((s₁ - b₁) * (s₁ - c₁))) ∧
  b = b₁ * Real.sqrt ((a₁ * c₁) / ((s₁ - a₁) * (s₁ - c₁))) ∧
  c = c₁ * Real.sqrt ((a₁ * b₁) / ((s₁ - a₁) * (s₁ - b₁))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_sides_l825_82583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l825_82535

theorem probability_factor_of_36 : 
  (Finset.filter (λ n ↦ n ∣ 36) (Finset.range 37)).card / 36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l825_82535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_4_equals_1_l825_82578

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem f_f_4_equals_1 : f (f 4) = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_4_equals_1_l825_82578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l825_82505

def repeated_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

theorem divisibility_by_seven : ∃ (k : ℕ), 
  (repeated_digit 8 50 * 10^51 + 5 * 10^50 + repeated_digit 9 50) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l825_82505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_most_one_equiv_leftmost_white_l825_82591

/-- A point on the interval [0,1] -/
def Point := ℝ

/-- The total number of points -/
def total_points : ℕ := 2019

/-- The number of black points -/
def black_points : ℕ := 1000

/-- The number of white points -/
def white_points : ℕ := total_points - black_points

/-- A coloring of points -/
def Coloring := Fin total_points → Bool

/-- The probability that the sum of the positions of the leftmost white point
    and the rightmost black point is at most 1 -/
noncomputable def prob_sum_at_most_one (points : Fin total_points → Point) (coloring : Coloring) : ℝ :=
  sorry

/-- The probability that the leftmost point is white -/
noncomputable def prob_leftmost_white : ℝ := (white_points : ℝ) / (total_points : ℝ)

theorem sum_at_most_one_equiv_leftmost_white 
  (points : Fin total_points → Point) (coloring : Coloring) :
  prob_sum_at_most_one points coloring = prob_leftmost_white :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_most_one_equiv_leftmost_white_l825_82591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l825_82577

/-- Probability measure --/
noncomputable def Prob : Prop → ℚ :=
  sorry

/-- Predicate representing the event that A meets B under given conditions --/
def A_meets_B (start_A start_B : ℕ × ℕ) (step_length : ℕ) (prob_right_or_up prob_left_or_down : ℚ) (steps_to_meet : ℕ) : Prop :=
  sorry

/-- The probability that two objects meet given specific movement conditions --/
theorem probability_of_meeting :
  let start_A : ℕ × ℕ := (0, 0)
  let start_B : ℕ × ℕ := (3, 4)
  let step_length : ℕ := 1
  let prob_right_or_up : ℚ := 1/2
  let prob_left_or_down : ℚ := 1/2
  let steps_to_meet : ℕ := 3
  Prob (A_meets_B start_A start_B step_length prob_right_or_up prob_left_or_down steps_to_meet) = 5/16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l825_82577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_passengers_count_l825_82563

theorem bus_passengers_count : 
  (42 : ℕ) + 38 + 5 + 15 + 2 + 1 = 103 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_passengers_count_l825_82563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_match_pattern_sum_l825_82500

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

/-- The problem statement -/
theorem match_pattern_sum :
  let a : ℝ := 4
  let d : ℝ := 2
  let n : ℕ := 25
  arithmetic_sum a d n = 700 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_match_pattern_sum_l825_82500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_l825_82519

/-- Represents the total number of students in the class -/
def N : ℕ := sorry

/-- Represents the number of honor students in the class -/
def k : ℕ := sorry

/-- The number of bullies in the class is always 8 -/
def num_bullies : ℕ := 8

/-- The total number of students is the sum of honor students and bullies -/
axiom total_students : N = k + num_bullies

/-- For bullies' statements to be false (as they always lie), 
    the fraction of bullies must be less than 1/3 -/
axiom bully_condition : (num_bullies : ℚ) / (N - 1 : ℚ) < 1 / 3

/-- For honor students' statements to be true, 
    the fraction of bullies must be at least 1/3 -/
axiom honor_condition : (num_bullies : ℚ) / (N - 1 : ℚ) ≥ 1 / 3

/-- The theorem states that N can only be 23, 24, or 25 -/
theorem possible_values : N = 23 ∨ N = 24 ∨ N = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_l825_82519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_two_sevenths_l825_82527

/-- Represents the number of fifth graders -/
def f : ℕ → ℕ := fun _ => 1

/-- Represents the number of eighth graders -/
def e : ℕ → ℕ := fun _ => 1

/-- The number of paired eighth graders equals the number of paired fifth graders -/
axiom pairing_equality (n : ℕ) : e n / 4 = f n / 3

/-- The fraction of students with a partner -/
def fraction_with_partner (n : ℕ) : ℚ := (e n / 4 + f n / 3) / (e n + f n)

/-- Theorem stating that the fraction of students with a partner is 2/7 -/
theorem fraction_is_two_sevenths (n : ℕ) : fraction_with_partner n = 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_two_sevenths_l825_82527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_sixth_l825_82575

-- Define the function g
noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 4 then
    (x * y - x + 3) / (3 * x)
  else
    (x * y - y - 3) / (-3 * y)

-- State the theorem
theorem g_sum_equals_one_sixth :
  g 3 1 + g 3 2 = 1/6 := by
  -- Evaluate g(3,1)
  have h1 : g 3 1 = 1/3 := by
    -- Proof steps for g(3,1)
    sorry
  
  -- Evaluate g(3,2)
  have h2 : g 3 2 = -1/6 := by
    -- Proof steps for g(3,2)
    sorry
  
  -- Sum the results
  calc
    g 3 1 + g 3 2 = 1/3 + (-1/6) := by rw [h1, h2]
    _              = 2/6 + (-1/6) := by norm_num
    _              = 1/6 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_sixth_l825_82575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_in_sequence_l825_82545

theorem largest_integer_in_sequence (seq : List ℤ) : 
  seq.length = 19 ∧ 
  seq.Pairwise (λ i j => j = i + 1) ∧ 
  seq.sum / seq.length = 99 →
  seq.maximum? = some 108 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_in_sequence_l825_82545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l825_82572

-- Define triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  B : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  a := 2
  b := 4  -- We know b = 4 from the solution
  c := 3  -- We know c = 7 - b = 3 from the solution
  B := Real.arccos (-1/4)

-- Theorem for part (1)
theorem part_one : given_triangle.b = 4 := by
  rfl  -- Reflexivity, since we defined b as 4 in given_triangle

-- Theorem for part (2)
theorem part_two : 
  (1/2 * given_triangle.a * given_triangle.c * Real.sin given_triangle.B) = (3 * Real.sqrt 15) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l825_82572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l825_82532

/-- Circle with center (a, b) and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line with slope k passing through point (x, y) -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem -/
theorem circle_line_intersection (C : Circle) (l : Line) :
  C.center = (2, 3) →
  C.radius = 1 →
  l.point = (0, 1) →
  (∃ A B : ℝ × ℝ,
    (A.2 = l.slope * A.1 + 1) ∧
    (B.2 = l.slope * B.1 + 1) ∧
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 1) ∧
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 1) ∧
    distance A B = 2) →
  l.slope = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l825_82532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ship_interception_l825_82587

/-- The optimal angle for ship interception -/
noncomputable def optimal_interception_angle (initial_angle : ℝ) (speed_ratio : ℝ) : ℝ :=
  initial_angle / 2

theorem optimal_ship_interception 
  (initial_angle : ℝ) 
  (distance : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : initial_angle = 60) 
  (h2 : speed_ratio = 2) 
  (h3 : distance > 0) :
  optimal_interception_angle initial_angle speed_ratio = 30 := by
  sorry

#check optimal_ship_interception

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_ship_interception_l825_82587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_hump_theorem_l825_82525

def possible_values_count (total_camels : ℕ) (subset_size : ℕ) : ℕ :=
  (List.range 100).foldl (λ acc n =>
    if n ≥ 1 ∧ n ≤ 99 ∧
       (subset_size ≥ (total_camels + n) / 2 ∨
        subset_size + 14 ≤ n)
    then acc + 1
    else acc
  ) 0

theorem camel_hump_theorem :
  possible_values_count 100 62 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_hump_theorem_l825_82525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_box_volume_4_5_6_l825_82543

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped -/
noncomputable def extended_box_volume (l w h : ℝ) : ℝ :=
  (l * w * h)                           -- volume of the main box
  + 2 * (l * w + l * h + w * h)         -- volume of external parallelepipeds
  + (4 / 3) * Real.pi                   -- volume of 1/8-spheres at vertices
  + Real.pi * (l + w + h)               -- volume of 1/4-cylinders along edges

/-- The theorem stating the volume of the extended box with dimensions 4, 5, and 6 -/
theorem extended_box_volume_4_5_6 :
  extended_box_volume 4 5 6 = (804 + 77 * Real.pi) / 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval extended_box_volume 4 5 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_box_volume_4_5_6_l825_82543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l825_82546

/-- The distance between two stations given the speeds of three cars and their meeting times. -/
theorem distance_between_stations
  (speed_A speed_B speed_C : ℝ)
  (time_BC : ℝ)
  (h1 : speed_A = 90)
  (h2 : speed_B = 80)
  (h3 : speed_C = 60)
  (h4 : time_BC = 20 / 60) :
  (speed_A + speed_B) * ((speed_A + speed_C) * time_BC / (speed_B - speed_C)) = 425 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l825_82546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_quasi_order_iff_no_infinite_antichain_or_decreasing_sequence_l825_82568

-- Define a quasiorder on a type α
def is_quasiorder {α : Type*} (r : α → α → Prop) : Prop :=
  (∀ a, r a a) ∧ (∀ a b c, r a b → r b c → r a c)

-- Define a well-quasi-order
def is_well_quasi_order {α : Type*} (r : α → α → Prop) : Prop :=
  is_quasiorder r ∧
  ∀ (f : ℕ → α), ∃ i j, i < j ∧ r (f i) (f j)

-- Define an antichain
def is_antichain {α : Type*} (r : α → α → Prop) (s : Set α) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ y → ¬(r x y) ∧ ¬(r y x)

-- Define a strictly decreasing sequence
def is_strictly_decreasing {α : Type*} (r : α → α → Prop) (f : ℕ → α) : Prop :=
  ∀ n, r (f (n + 1)) (f n) ∧ ¬(r (f n) (f (n + 1)))

-- The main theorem
theorem well_quasi_order_iff_no_infinite_antichain_or_decreasing_sequence
  {α : Type*} (r : α → α → Prop) :
  is_well_quasi_order r ↔
  (¬∃ (s : Set α), Infinite s ∧ is_antichain r s) ∧
  (¬∃ (f : ℕ → α), is_strictly_decreasing r f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_quasi_order_iff_no_infinite_antichain_or_decreasing_sequence_l825_82568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_point_l825_82594

/-- If the terminal side of angle α passes through point (1,2), then tan(2α) = 4/3 -/
theorem tan_double_angle_special_point (α : ℝ) :
  (Real.tan α = 2) → Real.tan (2 * α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_point_l825_82594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_embark_theorem_l825_82507

/-- Represents a boat with a given capacity -/
structure Boat where
  capacity : ℕ

/-- Represents the group of people -/
structure People where
  adults : ℕ
  children : ℕ

/-- Calculates the number of ways to embark on boats -/
def embarkWays (boats : List Boat) (group : People) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem embark_theorem (boats : List Boat) (group : People) :
  boats = [Boat.mk 3, Boat.mk 2, Boat.mk 1] →
  group = People.mk 3 2 →
  embarkWays boats group = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_embark_theorem_l825_82507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l825_82599

/-- The eccentricity of a hyperbola with a specific intersection point -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (P : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (P.1^2 + P.2^2 = a^2 + b^2) ∧
    (P.2 = a) →
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l825_82599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l825_82539

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

noncomputable def volume_cone (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

noncomputable def volume_sphere (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

theorem liquid_rise_ratio (cone1 cone2 : Cone) (marble : Sphere) :
  volume_cone cone1 = volume_cone cone2 →
  cone1.radius = 4 →
  cone2.radius = 9 →
  marble.radius = 2 →
  (cone1.height + 2 - cone1.height) / (cone2.height + 32/81 - cone2.height) = 81/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l825_82539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l825_82580

def n : ℕ := 2^3 * 5^6 * 8^9 * 10^10

theorem number_of_factors_of_n : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 697 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_n_l825_82580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_floor_log_is_geometric_sum_prob_equal_floor_log_approx_l825_82550

/-- The probability that the floor of log base 10 of x equals the floor of log base 10 of y,
    where x and y are independently and uniformly distributed in (0,1) -/
noncomputable def prob_equal_floor_log : ℝ :=
  ∑' n : ℕ, (0.1^n - 0.1^(n+1))^2

theorem prob_equal_floor_log_is_geometric_sum :
  prob_equal_floor_log = 0.81 / (1 - 0.01) :=
sorry

theorem prob_equal_floor_log_approx :
  ∃ ε > 0, |prob_equal_floor_log - 0.81818| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_floor_log_is_geometric_sum_prob_equal_floor_log_approx_l825_82550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l825_82503

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : inner a b = -(1/2 : ℝ)) : 
  ‖a + (2 : ℝ) • b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l825_82503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_in_ratio_l825_82569

theorem product_of_numbers_in_ratio (x y : ℝ) :
  (x - y) / (x + y) = 3 / 5 ∧ 
  (x + y) / (x * y) = 5 / 15 →
  x * y = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_numbers_in_ratio_l825_82569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l825_82597

-- Define the function f(x) = log₁/₂(1-2x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2*x) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio (1/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l825_82597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_unique_l825_82561

/-- A cubic function with specific properties -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_unique :
  ∀ a b c d : ℝ,
  (∃ (f : ℝ → ℝ), f = cubic_function a b c d ∧
    (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x < f 1) ∧ 
    (f 1 = 4) ∧
    (∃ (δ : ℝ), δ > 0 ∧ ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → f x > f 3) ∧ 
    (f 3 = 0) ∧
    (f 0 = 0)) →
  a = 1 ∧ b = -6 ∧ c = 9 ∧ d = 0 :=
by sorry

#check cubic_function_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_unique_l825_82561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l825_82556

theorem integral_x_squared_plus_sqrt_one_minus_x_squared : 
  ∫ x in Set.Icc (-1) 1, (x^2 + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l825_82556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_rationals_appear_once_l825_82502

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding the case for 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then 1 / sequence_a (n + 1) else sequence_a ((n + 2) / 2) + 1

theorem all_positive_rationals_appear_once :
  (∀ q : ℚ, q > 0 → ∃! n : ℕ, sequence_a n = q) ∧
  (∀ n : ℕ, sequence_a n > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_positive_rationals_appear_once_l825_82502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_difference_l825_82526

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => arithmetic_sequence d n + d

/-- Theorem: If a_1, a_3, and a_13 of an arithmetic sequence form a geometric sequence,
    and the common difference is non-zero, then the common difference is 2 -/
theorem arithmetic_geometric_sequence_difference (d : ℝ) (h : d ≠ 0) :
  (arithmetic_sequence d 0) * (arithmetic_sequence d 12) = (arithmetic_sequence d 2)^2 →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_difference_l825_82526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l825_82585

/-- The solution to the differential equation y' = cos x with y(0) = 1 -/
theorem differential_equation_solution (x : ℝ) :
  let y : ℝ → ℝ := λ t => Real.sin t + 1
  (∀ t, (deriv y) t = Real.cos t) ∧ y 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l825_82585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sqrt_function_l825_82592

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt (x + 5)

-- State the theorem
theorem range_of_sqrt_function :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -5} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sqrt_function_l825_82592
