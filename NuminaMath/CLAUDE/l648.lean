import Mathlib

namespace NUMINAMATH_CALUDE_extremum_implies_deriv_root_exists_deriv_root_without_extremum_l648_64843

-- Define a differentiable function on the real line
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define what it means for f'(x) = 0 to have a real root
def deriv_has_root (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, deriv f x = 0

-- Statement 1: If f has an extremum, then f'(x) = 0 has a real root
theorem extremum_implies_deriv_root :
  has_extremum f → deriv_has_root f :=
sorry

-- Statement 2: There exists a function f such that f'(x) = 0 has a real root,
-- but f does not have an extremum
theorem exists_deriv_root_without_extremum :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv_has_root f ∧ ¬has_extremum f :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_deriv_root_exists_deriv_root_without_extremum_l648_64843


namespace NUMINAMATH_CALUDE_choose_four_from_thirty_l648_64827

theorem choose_four_from_thirty : Nat.choose 30 4 = 27405 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_thirty_l648_64827


namespace NUMINAMATH_CALUDE_equal_selection_probability_l648_64879

/-- Represents the selection process for a student survey -/
structure StudentSurvey where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ
  remaining_students : ℕ

/-- The probability of a student being selected in the survey -/
def selection_probability (survey : StudentSurvey) : ℚ :=
  (survey.remaining_students : ℚ) / (survey.total_students : ℚ) *
  (survey.selected_students : ℚ) / (survey.remaining_students : ℚ)

/-- The specific survey described in the problem -/
def school_survey : StudentSurvey :=
  { total_students := 2012
  , selected_students := 50
  , eliminated_students := 12
  , remaining_students := 2000 }

/-- Theorem stating that the selection probability is equal for all students -/
theorem equal_selection_probability :
  ∀ (s1 s2 : StudentSurvey),
    s1 = school_survey → s2 = school_survey →
    selection_probability s1 = selection_probability s2 :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l648_64879


namespace NUMINAMATH_CALUDE_carmina_coins_count_l648_64880

theorem carmina_coins_count :
  ∀ (n d : ℕ),
  (5 * n + 10 * d = 360) →
  (10 * n + 5 * d = 540) →
  n + d = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_carmina_coins_count_l648_64880


namespace NUMINAMATH_CALUDE_points_on_angle_bisector_l648_64890

/-- Given two points A and B, proves that if they lie on the angle bisector of the first and third quadrants, their coordinates satisfy specific conditions. -/
theorem points_on_angle_bisector 
  (a b : ℝ) 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (a - 1, 2)) 
  (h2 : B = (-3, b + 1)) 
  (h3 : (a - 1) = 2 ∧ (b + 1) = -3) : 
  a = 3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_points_on_angle_bisector_l648_64890


namespace NUMINAMATH_CALUDE_inequality_constraint_l648_64847

theorem inequality_constraint (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  |a| + |b| ≥ 2 / Real.sqrt 3 →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_inequality_constraint_l648_64847


namespace NUMINAMATH_CALUDE_total_inflation_time_is_900_l648_64821

/-- The time in minutes it takes to inflate one soccer ball -/
def inflation_time : ℕ := 20

/-- The number of soccer balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermias_additional_balls : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + ermias_additional_balls

/-- The total time taken by Alexia and Ermias to inflate all soccer balls -/
def total_inflation_time : ℕ := inflation_time * (alexia_balls + ermias_balls)

theorem total_inflation_time_is_900 : total_inflation_time = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_inflation_time_is_900_l648_64821


namespace NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_l648_64895

/-- A sequence is a jump sequence if (a_i - a_i+2)(a_i+2 - a_i+1) > 0 for any three consecutive terms -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if a_(n+1) = q * a_n for all n -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio {a : ℕ → ℝ} {q : ℝ} 
  (h_geometric : is_geometric_sequence a q)
  (h_jump : is_jump_sequence a) :
  -1 < q ∧ q < 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_l648_64895


namespace NUMINAMATH_CALUDE_unique_base_thirteen_l648_64846

/-- Converts a digit character to its numeric value -/
def char_to_digit (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if c = 'A' then 10
  else if c = 'B' then 11
  else if c = 'C' then 12
  else 0

/-- Converts a string representation of a number in base a to its decimal value -/
def to_decimal (s : String) (a : ℕ) : ℕ :=
  s.foldr (fun c acc => char_to_digit c + a * acc) 0

/-- Checks if the equation 375_a + 592_a = 9C7_a is satisfied for a given base a -/
def equation_satisfied (a : ℕ) : Prop :=
  to_decimal "375" a + to_decimal "592" a = to_decimal "9C7" a

theorem unique_base_thirteen :
  ∃! a : ℕ, a > 12 ∧ equation_satisfied a ∧ char_to_digit 'C' = 12 :=
sorry

end NUMINAMATH_CALUDE_unique_base_thirteen_l648_64846


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l648_64819

/-- The purchase price of a jacket given selling price and profit conditions -/
theorem jacket_purchase_price (S P : ℝ) (h1 : S = P + 0.25 * S) 
  (h2 : ∃ D : ℝ, D = 0.8 * S ∧ D - P = 4) : P = 60 := by
  sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l648_64819


namespace NUMINAMATH_CALUDE_valid_book_pairs_18_4_l648_64809

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different pairs of books that can be chosen from a collection of books,
    given the total number of books and the number of books in a series,
    with the restriction that two books from the series cannot be chosen together. -/
def validBookPairs (totalBooks seriesBooks : ℕ) : ℕ :=
  choose totalBooks 2 - choose seriesBooks 2

theorem valid_book_pairs_18_4 :
  validBookPairs 18 4 = 147 := by sorry

end NUMINAMATH_CALUDE_valid_book_pairs_18_4_l648_64809


namespace NUMINAMATH_CALUDE_min_value_of_sum_l648_64849

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ m : ℝ, m = 5 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → x + 1/x + y + 1/y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l648_64849


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_length_l648_64842

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_line : ℝ

/-- The property that the line joining the midpoints of the diagonals is half the difference of the bases -/
def midpoint_line_property (t : Trapezoid) : Prop :=
  t.midpoint_line = (t.long_base - t.short_base) / 2

/-- Theorem: In a trapezoid where the line joining the midpoints of the diagonals has length 4
    and the longer base is 100, the shorter base has length 92 -/
theorem trapezoid_shorter_base_length :
  ∀ t : Trapezoid,
    t.long_base = 100 →
    t.midpoint_line = 4 →
    midpoint_line_property t →
    t.short_base = 92 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_length_l648_64842


namespace NUMINAMATH_CALUDE_binomial_2057_1_l648_64858

theorem binomial_2057_1 : Nat.choose 2057 1 = 2057 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2057_1_l648_64858


namespace NUMINAMATH_CALUDE_logarithm_comparison_l648_64882

theorem logarithm_comparison : ∃ (a b c : ℝ),
  a = Real.log 2 / Real.log 3 ∧
  b = Real.log 2 / Real.log 5 ∧
  c = Real.log 3 / Real.log 2 ∧
  c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_logarithm_comparison_l648_64882


namespace NUMINAMATH_CALUDE_no_primes_between_factorial_plus_n_and_factorial_plus_2n_l648_64876

theorem no_primes_between_factorial_plus_n_and_factorial_plus_2n (n : ℕ) (hn : n > 1) :
  ∀ p, Nat.Prime p → ¬(n! + n < p ∧ p < n! + 2*n) :=
sorry

end NUMINAMATH_CALUDE_no_primes_between_factorial_plus_n_and_factorial_plus_2n_l648_64876


namespace NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l648_64868

def decimal_representation (n d : ℕ) : List ℕ := sorry

def count_digit (l : List ℕ) (digit : ℕ) : ℕ := sorry

def probability_of_digit (n d digit : ℕ) : ℚ :=
  let rep := decimal_representation n d
  (count_digit rep digit : ℚ) / (rep.length : ℚ)

theorem probability_of_nine_in_three_elevenths :
  probability_of_digit 3 11 9 = 0 := by sorry

end NUMINAMATH_CALUDE_probability_of_nine_in_three_elevenths_l648_64868


namespace NUMINAMATH_CALUDE_shirt_markup_l648_64873

theorem shirt_markup (P : ℝ) (h : 2 * P - 1.8 * P = 5) : 1.8 * P = 45 := by
  sorry

end NUMINAMATH_CALUDE_shirt_markup_l648_64873


namespace NUMINAMATH_CALUDE_composition_ratio_l648_64864

def f (x : ℝ) : ℝ := 3 * x + 5

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio :
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = 380 / 653 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l648_64864


namespace NUMINAMATH_CALUDE_gp_sum_ratio_l648_64857

/-- For a geometric progression with common ratio 3, the ratio of the sum
    of the first 6 terms to the sum of the first 3 terms is 28. -/
theorem gp_sum_ratio (a : ℝ) : 
  let r := 3
  let S₃ := a * (1 - r^3) / (1 - r)
  let S₆ := a * (1 - r^6) / (1 - r)
  S₆ / S₃ = 28 := by
sorry


end NUMINAMATH_CALUDE_gp_sum_ratio_l648_64857


namespace NUMINAMATH_CALUDE_extreme_points_and_inequality_l648_64861

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - x

theorem extreme_points_and_inequality (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    (∀ (y : ℝ), |y - x₁| < δ → f a y ≤ f a x₁ + ε) ∧
    (∀ (y : ℝ), |y - x₂| < δ → f a y ≥ f a x₂ - ε)) ∧
  f a x₂ < 1 + (Real.sin x₂ - x₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_and_inequality_l648_64861


namespace NUMINAMATH_CALUDE_exponential_inequality_l648_64894

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1.5 : ℝ) ^ a > (1.5 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l648_64894


namespace NUMINAMATH_CALUDE_mitch_macarons_count_l648_64805

/-- The number of macarons Mitch made -/
def mitch_macarons : ℕ := 20

/-- The number of macarons Joshua made -/
def joshua_macarons : ℕ := mitch_macarons + 6

/-- The number of macarons Miles made -/
def miles_macarons : ℕ := 2 * joshua_macarons

/-- The number of macarons Renz made -/
def renz_macarons : ℕ := (3 * miles_macarons) / 4 - 1

/-- The total number of macarons given to kids -/
def total_macarons : ℕ := 68 * 2

theorem mitch_macarons_count : 
  mitch_macarons + joshua_macarons + miles_macarons + renz_macarons = total_macarons :=
by sorry

end NUMINAMATH_CALUDE_mitch_macarons_count_l648_64805


namespace NUMINAMATH_CALUDE_betty_doug_age_sum_l648_64810

/-- The sum of Betty's and Doug's ages given the conditions of the problem -/
theorem betty_doug_age_sum : ∀ (betty_age : ℕ) (doug_age : ℕ),
  doug_age = 40 →
  2 * betty_age * 20 = 2000 →
  betty_age + doug_age = 90 := by
  sorry

end NUMINAMATH_CALUDE_betty_doug_age_sum_l648_64810


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l648_64820

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l648_64820


namespace NUMINAMATH_CALUDE_divisor_condition_l648_64823

theorem divisor_condition (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k :=
sorry

end NUMINAMATH_CALUDE_divisor_condition_l648_64823


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_450_l648_64806

theorem largest_multiple_of_15_less_than_450 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 450 → n ≤ 435 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_450_l648_64806


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l648_64811

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 6
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 2 * Real.sqrt 22 ∧ θ = Real.arctan (Real.sqrt 6 / 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l648_64811


namespace NUMINAMATH_CALUDE_clown_count_l648_64808

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 5

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 140 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l648_64808


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l648_64884

theorem crazy_silly_school_series (total_books : ℕ) (books_read : ℕ) (movies_watched : ℕ) :
  total_books = 11 →
  books_read = 7 →
  movies_watched = 21 →
  movies_watched = books_read + 14 →
  ∃ (total_movies : ℕ), total_movies = 7 ∧ total_movies = movies_watched - 14 :=
by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l648_64884


namespace NUMINAMATH_CALUDE_songbook_cost_is_seven_l648_64807

/-- The cost of Jason's music purchases -/
structure MusicPurchase where
  flute : ℝ
  stand : ℝ
  total : ℝ

/-- The cost of the song book given Jason's other music purchases -/
def songbook_cost (p : MusicPurchase) : ℝ :=
  p.total - (p.flute + p.stand)

/-- Theorem: The cost of the song book is $7.00 -/
theorem songbook_cost_is_seven (p : MusicPurchase)
  (h1 : p.flute = 142.46)
  (h2 : p.stand = 8.89)
  (h3 : p.total = 158.35) :
  songbook_cost p = 7.00 := by
  sorry

#eval songbook_cost { flute := 142.46, stand := 8.89, total := 158.35 }

end NUMINAMATH_CALUDE_songbook_cost_is_seven_l648_64807


namespace NUMINAMATH_CALUDE_distance_to_larger_section_specific_case_l648_64893

/-- Represents a right triangular pyramid with two parallel cross sections -/
structure RightTriangularPyramid where
  /-- Area of the smaller cross section -/
  area_small : ℝ
  /-- Area of the larger cross section -/
  area_large : ℝ
  /-- Distance between the two cross sections -/
  cross_section_distance : ℝ

/-- Calculates the distance from the apex to the larger cross section -/
def distance_to_larger_section (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the distance to the larger cross section for specific conditions -/
theorem distance_to_larger_section_specific_case :
  let p : RightTriangularPyramid := {
    area_small := 150 * Real.sqrt 3,
    area_large := 300 * Real.sqrt 3,
    cross_section_distance := 10
  }
  distance_to_larger_section p = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_specific_case_l648_64893


namespace NUMINAMATH_CALUDE_hyperbola_sum_a_h_l648_64837

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote equations
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  -- Point the hyperbola passes through
  point : ℝ × ℝ
  -- Standard form parameters
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  -- Conditions
  asymptote1_eq : ∀ x, asymptote1 x = 3 * x + 2
  asymptote2_eq : ∀ x, asymptote2 x = -3 * x + 8
  point_on_hyperbola : point = (1, 6)
  standard_form : ∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1
  positive_params : a > 0 ∧ b > 0

/-- Theorem: For the given hyperbola, a + h = 2 -/
theorem hyperbola_sum_a_h (hyp : Hyperbola) : hyp.a + hyp.h = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_a_h_l648_64837


namespace NUMINAMATH_CALUDE_child_ticket_price_l648_64828

theorem child_ticket_price (total_revenue : ℕ) (adult_price : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  total_revenue = 104 →
  adult_price = 6 →
  total_tickets = 21 →
  child_tickets = 11 →
  ∃ (child_price : ℕ), child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧ child_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_l648_64828


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l648_64872

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                    -- length is double the width
  l * w = 8 * 49 →               -- area of field is 8 times area of pond (7^2 = 49)
  l = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l648_64872


namespace NUMINAMATH_CALUDE_min_value_of_f_l648_64835

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else x + 6/x - 7

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 * Real.sqrt 6 - 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l648_64835


namespace NUMINAMATH_CALUDE_james_toy_cars_l648_64841

/-- Proves that James buys 20 toy cars given the problem conditions -/
theorem james_toy_cars : 
  ∀ (cars soldiers : ℕ),
  soldiers = 2 * cars →
  cars + soldiers = 60 →
  cars = 20 := by
sorry

end NUMINAMATH_CALUDE_james_toy_cars_l648_64841


namespace NUMINAMATH_CALUDE_sum_of_fractions_l648_64886

theorem sum_of_fractions : 
  (2 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 6 / 100000 = 0.02586 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l648_64886


namespace NUMINAMATH_CALUDE_multiply_and_add_l648_64822

theorem multiply_and_add : 42 * 52 + 48 * 42 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l648_64822


namespace NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l648_64867

theorem sphere_volume_to_surface_area :
  ∀ (r : ℝ), 
    (4 / 3 * π * r^3 = 32 * π / 3) →
    (4 * π * r^2 = 16 * π) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_to_surface_area_l648_64867


namespace NUMINAMATH_CALUDE_washers_remaining_l648_64832

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining (pipe_length : ℕ) (feet_per_bolt : ℕ) (washers_per_bolt : ℕ) (initial_washers : ℕ) : 
  pipe_length = 40 ∧ 
  feet_per_bolt = 5 ∧ 
  washers_per_bolt = 2 ∧ 
  initial_washers = 20 → 
  initial_washers - (pipe_length / feet_per_bolt * washers_per_bolt) = 4 := by
sorry

end NUMINAMATH_CALUDE_washers_remaining_l648_64832


namespace NUMINAMATH_CALUDE_length_PF_is_16_over_3_l648_64801

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 0)

-- Define the line through the focus
def line_through_focus (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the intersection points A and B (we don't calculate them explicitly)
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line_through_focus A.1 A.2 ∧ line_through_focus B.1 B.2

-- Define point P on x-axis
def point_P (P : ℝ × ℝ) : Prop := P.2 = 0

-- Main theorem
theorem length_PF_is_16_over_3 
  (A B P : ℝ × ℝ) 
  (h_intersect : intersection_points A B)
  (h_P : point_P P)
  (h_perpendicular : sorry) -- Additional hypothesis for P being on the perpendicular bisector
  : ‖P - focus‖ = 16/3 :=
sorry

end NUMINAMATH_CALUDE_length_PF_is_16_over_3_l648_64801


namespace NUMINAMATH_CALUDE_parallelepiped_height_l648_64898

/-- The surface area of a rectangular parallelepiped -/
def surface_area (l w h : ℝ) : ℝ := 2*l*w + 2*l*h + 2*w*h

/-- Theorem: The height of a rectangular parallelepiped with given dimensions -/
theorem parallelepiped_height (w l : ℝ) (h : ℝ) :
  w = 7 → l = 8 → surface_area l w h = 442 → h = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_height_l648_64898


namespace NUMINAMATH_CALUDE_benny_baseball_gear_expense_l648_64818

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Theorem stating that Benny spent 34 dollars on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 67 33 = 34 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_expense_l648_64818


namespace NUMINAMATH_CALUDE_snake_count_l648_64829

theorem snake_count (breeding_balls : Nat) (snake_pairs : Nat) (total_snakes : Nat) :
  breeding_balls = 3 →
  snake_pairs = 6 →
  total_snakes = 36 →
  ∃ snakes_per_ball : Nat, snakes_per_ball * breeding_balls + snake_pairs * 2 = total_snakes ∧ snakes_per_ball = 8 := by
  sorry

end NUMINAMATH_CALUDE_snake_count_l648_64829


namespace NUMINAMATH_CALUDE_tan_identity_l648_64803

theorem tan_identity (α : ℝ) (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l648_64803


namespace NUMINAMATH_CALUDE_complex_equation_solution_l648_64885

theorem complex_equation_solution : 
  ∃ (a b c d e : ℤ),
    (2 * (2 : ℝ)^(2/3) + (2 : ℝ)^(1/3) * a + 2 * b + (2 : ℝ)^(2/3) * c + (2 : ℝ)^(1/3) * d + e = 0) ∧
    (25 * (Complex.I * Real.sqrt 5) + 25 * a - 5 * (Complex.I * Real.sqrt 5) * b - 5 * c + (Complex.I * Real.sqrt 5) * d + e = 0) ∧
    (abs (a + b + c + d + e) = 7) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l648_64885


namespace NUMINAMATH_CALUDE_sixtieth_element_is_2064_l648_64896

/-- The set of sums of powers of 2 with natural number exponents where the first exponent is less than the second -/
def PowerSumSet : Set ℕ :=
  {n | ∃ (x y : ℕ), x < y ∧ n = 2^x + 2^y}

/-- The 60th element in the ascending order of PowerSumSet -/
def sixtieth_element : ℕ := sorry

/-- Theorem stating that the 60th element of PowerSumSet is 2064 -/
theorem sixtieth_element_is_2064 : sixtieth_element = 2064 := by sorry

end NUMINAMATH_CALUDE_sixtieth_element_is_2064_l648_64896


namespace NUMINAMATH_CALUDE_total_balls_l648_64852

theorem total_balls (S V B : ℕ) : 
  S = 68 ∧ 
  S = V - 12 ∧ 
  S = B + 23 → 
  S + V + B = 193 := by
sorry

end NUMINAMATH_CALUDE_total_balls_l648_64852


namespace NUMINAMATH_CALUDE_class_8_3_final_score_l648_64836

/-- The final score of a choir competition is calculated based on three categories:
    singing quality, spirit, and coordination. Each category has a specific weight
    in the final score calculation. -/
def final_score (singing_quality : ℝ) (spirit : ℝ) (coordination : ℝ)
                (singing_weight : ℝ) (spirit_weight : ℝ) (coordination_weight : ℝ) : ℝ :=
  singing_quality * singing_weight + spirit * spirit_weight + coordination * coordination_weight

/-- Theorem stating that the final score of Class 8-3 in the choir competition is 81.8 points -/
theorem class_8_3_final_score :
  final_score 92 80 70 0.4 0.3 0.3 = 81.8 := by
  sorry

end NUMINAMATH_CALUDE_class_8_3_final_score_l648_64836


namespace NUMINAMATH_CALUDE_f_of_3_equals_10_l648_64854

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_of_3_equals_10 : f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_10_l648_64854


namespace NUMINAMATH_CALUDE_sum_m_n_equals_five_l648_64838

theorem sum_m_n_equals_five (m n : ℚ) (h : (m - 3) * Real.sqrt 5 + 2 - n = 0) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_five_l648_64838


namespace NUMINAMATH_CALUDE_at_most_two_special_numbers_l648_64839

/-- A positive integer n is special if it can be expressed as 2^a * 3^b for some nonnegative integers a and b. -/
def is_special (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 3^b

/-- For any positive integer k, there are at most two special numbers in the range (k^2, k^2 + 2k + 1). -/
theorem at_most_two_special_numbers (k : ℕ+) :
  ∃ n₁ n₂ : ℕ+, ∀ n : ℕ+,
    k^2 < n ∧ n < k^2 + 2*k + 1 ∧ is_special n →
    n = n₁ ∨ n = n₂ :=
  sorry

end NUMINAMATH_CALUDE_at_most_two_special_numbers_l648_64839


namespace NUMINAMATH_CALUDE_prob_at_least_one_X_correct_l648_64889

/-- Represents the probability of selecting at least one person who used model X
    when randomly selecting 2 people from a group of 5, where 3 used model X and 2 used model Y. -/
def prob_at_least_one_X : ℚ := 9 / 10

/-- The total number of people in the experience group -/
def total_people : ℕ := 5

/-- The number of people who used model X bicycles -/
def model_X_users : ℕ := 3

/-- The number of people who used model Y bicycles -/
def model_Y_users : ℕ := 2

/-- The number of ways to select 2 people from the group -/
def total_selections : ℕ := total_people.choose 2

/-- The number of ways to select 2 people who both used model Y -/
def both_Y_selections : ℕ := model_Y_users.choose 2

theorem prob_at_least_one_X_correct :
  prob_at_least_one_X = 1 - (both_Y_selections : ℚ) / total_selections :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_X_correct_l648_64889


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l648_64830

theorem same_solution_implies_a_equals_seven (a : ℝ) : 
  (∃ x : ℝ, 6 * (x + 8) = 18 * x ∧ 6 * x - 2 * (a - x) = 2 * a + x) → 
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l648_64830


namespace NUMINAMATH_CALUDE_simplify_expression_l648_64825

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a / b + b / a + 1 / (a * b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l648_64825


namespace NUMINAMATH_CALUDE_robins_count_l648_64875

theorem robins_count (total : ℕ) (robins penguins pigeons : ℕ) : 
  robins = 2 * total / 3 →
  penguins = total / 8 →
  pigeons = 5 →
  total = robins + penguins + pigeons →
  robins = 16 := by
sorry

end NUMINAMATH_CALUDE_robins_count_l648_64875


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l648_64887

/-- Represents a tetrahedron with four heights -/
structure Tetrahedron where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The property that the sum of areas of any three faces is greater than the area of the fourth face -/
def validTetrahedron (t : Tetrahedron) : Prop :=
  ∀ (v : ℝ), v > 0 →
    (3 * v / t.h1 < 3 * v / t.h2 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h2 < 3 * v / t.h1 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h3 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h4) ∧
    (3 * v / t.h4 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h3)

/-- Theorem stating that no tetrahedron exists with heights 1, 2, 3, and 6 -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h1 = 1 ∧ t.h2 = 2 ∧ t.h3 = 3 ∧ t.h4 = 6 ∧ validTetrahedron t :=
sorry

end NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l648_64887


namespace NUMINAMATH_CALUDE_polynomial_roots_l648_64826

def P (x : ℂ) : ℂ := x^5 - 5*x^4 + 11*x^3 - 13*x^2 + 9*x - 3

theorem polynomial_roots :
  let roots : List ℂ := [1, (3 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2,
                         (3 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2]
  ∀ x : ℂ, (P x = 0) ↔ (x ∈ roots) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l648_64826


namespace NUMINAMATH_CALUDE_min_buses_needed_l648_64831

/-- The number of students to be transported -/
def total_students : ℕ := 540

/-- The maximum number of students each bus can hold -/
def bus_capacity : ℕ := 45

/-- The minimum number of buses needed is the ceiling of the quotient of total students divided by bus capacity -/
theorem min_buses_needed : 
  (total_students + bus_capacity - 1) / bus_capacity = 12 := by sorry

end NUMINAMATH_CALUDE_min_buses_needed_l648_64831


namespace NUMINAMATH_CALUDE_late_attendees_fraction_l648_64815

theorem late_attendees_fraction 
  (total : ℕ) 
  (total_pos : total > 0)
  (male_fraction : Rat)
  (male_on_time_fraction : Rat)
  (female_on_time_fraction : Rat)
  (h_male : male_fraction = 2 / 3)
  (h_male_on_time : male_on_time_fraction = 3 / 4)
  (h_female_on_time : female_on_time_fraction = 5 / 6) :
  (1 : Rat) - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction) = 2 / 9 := by
  sorry

#check late_attendees_fraction

end NUMINAMATH_CALUDE_late_attendees_fraction_l648_64815


namespace NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l648_64804

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- 1. Line passing through first, second, and fourth quadrants
theorem line_through_quadrants (l : Line) :
  (∃ x y, x > 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept) →
  l.slope < 0 ∧ l.intercept > 0 :=
sorry

-- 2. Line passing through a fixed point
theorem line_through_fixed_point (k : ℝ) :
  ∃ x y, k * x - y - 2 * k + 3 = 0 ∧ x = 2 ∧ y = 3 :=
sorry

-- 3. Point-slope form equation
theorem point_slope_form (p : Point) (m : ℝ) :
  p.x = 2 ∧ p.y = -1 ∧ m = -Real.sqrt 3 →
  ∀ x y, y + 1 = -Real.sqrt 3 * (x - 2) ↔ y - p.y = m * (x - p.x) :=
sorry

-- 4. Slope-intercept form equation
theorem slope_intercept_form (l : Line) :
  l.slope = -2 ∧ l.intercept = 3 →
  ∀ x y, y = l.slope * x + l.intercept ↔ y = -2 * x + 3 :=
sorry

end NUMINAMATH_CALUDE_line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l648_64804


namespace NUMINAMATH_CALUDE_distance_to_place_l648_64840

/-- The distance to the place -/
def distance : ℝ := 48

/-- The rowing speed in still water (km/h) -/
def rowing_speed : ℝ := 10

/-- The current velocity (km/h) -/
def current_velocity : ℝ := 2

/-- The wind speed (km/h) -/
def wind_speed : ℝ := 4

/-- The total time for the round trip (hours) -/
def total_time : ℝ := 15

/-- The effective speed towards the place (km/h) -/
def speed_to_place : ℝ := rowing_speed - wind_speed - current_velocity

/-- The effective speed returning from the place (km/h) -/
def speed_from_place : ℝ := rowing_speed + wind_speed + current_velocity

theorem distance_to_place : 
  distance = (total_time * speed_to_place * speed_from_place) / (speed_to_place + speed_from_place) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l648_64840


namespace NUMINAMATH_CALUDE_matching_socks_probability_theorem_l648_64863

/-- The number of different pairs of socks -/
def num_pairs : ℕ := 5

/-- The number of days socks are selected -/
def num_days : ℕ := 5

/-- The probability of wearing matching socks on both the third and fifth day -/
def matching_socks_probability : ℚ := 1 / 63

/-- Theorem stating the probability of wearing matching socks on both the third and fifth day -/
theorem matching_socks_probability_theorem :
  let total_socks := 2 * num_pairs
  let favorable_outcomes := num_pairs * (num_pairs - 1) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  let total_outcomes := (Nat.choose total_socks 2) * (Nat.choose (total_socks - 2) 2) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  (favorable_outcomes : ℚ) / total_outcomes = matching_socks_probability :=
by sorry

#check matching_socks_probability_theorem

end NUMINAMATH_CALUDE_matching_socks_probability_theorem_l648_64863


namespace NUMINAMATH_CALUDE_square_ratio_sum_l648_64853

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l648_64853


namespace NUMINAMATH_CALUDE_factorization_problem_1_l648_64800

theorem factorization_problem_1 (x y : ℝ) : x * y - x + y - 1 = (x + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l648_64800


namespace NUMINAMATH_CALUDE_mary_farm_animals_l648_64848

-- Define the initial state and transactions
def initial_lambs : ℕ := 18
def initial_alpacas : ℕ := 5
def lamb_babies : ℕ := 7 * 4
def traded_lambs : ℕ := 8
def traded_alpacas : ℕ := 2
def gained_goats : ℕ := 3
def gained_chickens : ℕ := 10
def alpacas_from_chickens : ℕ := 2
def additional_lambs : ℕ := 20
def additional_alpacas : ℕ := 6

-- Define the theorem
theorem mary_farm_animals :
  let lambs := initial_lambs + lamb_babies - traded_lambs + additional_lambs
  let alpacas := initial_alpacas - traded_alpacas + alpacas_from_chickens + additional_alpacas
  let goats := gained_goats
  let chickens := gained_chickens / 2
  (lambs = 58 ∧ alpacas = 11 ∧ goats = 3 ∧ chickens = 5) := by
  sorry

end NUMINAMATH_CALUDE_mary_farm_animals_l648_64848


namespace NUMINAMATH_CALUDE_costs_equal_at_60_l648_64860

/-- Represents the pricing and discount options for appliances -/
structure AppliancePricing where
  washing_machine_price : ℕ
  cooker_price : ℕ
  option1_free_cookers : ℕ
  option2_discount : ℚ

/-- Calculates the cost for Option 1 -/
def option1_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℕ :=
  p.washing_machine_price * washing_machines + p.cooker_price * (cookers - p.option1_free_cookers)

/-- Calculates the cost for Option 2 -/
def option2_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℚ :=
  (p.washing_machine_price * washing_machines + p.cooker_price * cookers : ℚ) * p.option2_discount

/-- Theorem: Costs of Option 1 and Option 2 are equal when x = 60 -/
theorem costs_equal_at_60 (p : AppliancePricing) 
    (h1 : p.washing_machine_price = 800)
    (h2 : p.cooker_price = 200)
    (h3 : p.option1_free_cookers = 10)
    (h4 : p.option2_discount = 9/10) :
    (option1_cost p 10 60 : ℚ) = option2_cost p 10 60 := by
  sorry

end NUMINAMATH_CALUDE_costs_equal_at_60_l648_64860


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l648_64813

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.ρ p.θ

/-- Checks if a line is parallel to the polar axis -/
def parallelToPolarAxis (l : PolarLine) : Prop :=
  ∀ ρ θ, l.equation ρ θ ↔ ∃ k, ρ * Real.sin θ = k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) 
  (h_p : p.ρ = 2 ∧ p.θ = Real.pi / 6) :
  ∃ l : PolarLine, 
    pointOnLine p l ∧ 
    parallelToPolarAxis l ∧
    (∀ ρ θ, l.equation ρ θ ↔ ρ * Real.sin θ = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l648_64813


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l648_64877

theorem complex_sum_to_polar : 
  15 * Complex.exp (Complex.I * Real.pi / 7) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 7) = 
  (30 * Real.cos (3 * Real.pi / 14) * Real.cos (Real.pi / 14)) * Complex.exp (Complex.I * 3 * Real.pi / 7) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l648_64877


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l648_64855

theorem quadratic_inequality_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l648_64855


namespace NUMINAMATH_CALUDE_molecular_weight_constant_l648_64865

-- Define the molecular weight of Aluminum carbonate
def aluminum_carbonate_mw : ℝ := 233.99

-- Define temperature and pressure
def temperature : ℝ := 298
def pressure : ℝ := 1

-- Define compressibility and thermal expansion coefficients
-- (We don't use these in the theorem, but they're mentioned in the problem)
def compressibility : ℝ := sorry
def thermal_expansion : ℝ := sorry

-- Theorem stating that the molecular weight remains constant
theorem molecular_weight_constant (T P : ℝ) :
  aluminum_carbonate_mw = 233.99 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_constant_l648_64865


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_some_but_not_all_l648_64897

/-- A function that checks if a number is divisible by some but not all integers from 1 to 10 -/
def isDivisibleBySomeButNotAll (m : ℕ) : Prop :=
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k = 0) ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ m % k ≠ 0)

/-- The main theorem stating that 3 is the least positive integer satisfying the condition -/
theorem least_positive_integer_divisible_by_some_but_not_all :
  (∀ n : ℕ, 0 < n ∧ n < 3 → ¬isDivisibleBySomeButNotAll (n^2 - n)) ∧
  isDivisibleBySomeButNotAll (3^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_some_but_not_all_l648_64897


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l648_64824

theorem tan_half_product_squared (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l648_64824


namespace NUMINAMATH_CALUDE_correct_mean_after_error_correction_l648_64883

theorem correct_mean_after_error_correction (n : ℕ) (incorrect_mean correct_value incorrect_value : ℝ) :
  n = 30 →
  incorrect_mean = 250 →
  correct_value = 165 →
  incorrect_value = 135 →
  (n : ℝ) * incorrect_mean + (correct_value - incorrect_value) = n * 251 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_after_error_correction_l648_64883


namespace NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l648_64871

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 5 →
  white_balls = 4 →
  (1 - (red_balls / total_balls * (red_balls - 1) / (total_balls - 1))) = 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_white_ball_l648_64871


namespace NUMINAMATH_CALUDE_solve_equation_l648_64859

theorem solve_equation :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ y = 250 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l648_64859


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l648_64899

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola xy = 1 -/
def hyperbola (p : Point) : Prop := p.x * p.y = 1

/-- Four points lie on the same circle -/
def on_same_circle (p1 p2 p3 p4 : Point) : Prop := 
  ∃ (h k s : ℝ), 
    (p1.x - h)^2 + (p1.y - k)^2 = s^2 ∧
    (p2.x - h)^2 + (p2.y - k)^2 = s^2 ∧
    (p3.x - h)^2 + (p3.y - k)^2 = s^2 ∧
    (p4.x - h)^2 + (p4.y - k)^2 = s^2

theorem fourth_intersection_point : 
  let p1 : Point := ⟨3, 1/3⟩
  let p2 : Point := ⟨-4, -1/4⟩
  let p3 : Point := ⟨1/6, 6⟩
  let p4 : Point := ⟨-1/2, -2⟩
  hyperbola p1 ∧ hyperbola p2 ∧ hyperbola p3 ∧ hyperbola p4 ∧
  on_same_circle p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l648_64899


namespace NUMINAMATH_CALUDE_sum_of_transformed_roots_equals_one_l648_64834

theorem sum_of_transformed_roots_equals_one : 
  ∀ α β γ : ℂ, 
  (α^3 = α + 1) → (β^3 = β + 1) → (γ^3 = γ + 1) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_transformed_roots_equals_one_l648_64834


namespace NUMINAMATH_CALUDE_better_fit_larger_R_squared_l648_64870

-- Define the correlation index R²
def correlation_index (R : ℝ) : Prop := 0 ≤ R ∧ R ≤ 1

-- Define the concept of model fit
def model_fit (fit : ℝ) : Prop := 0 ≤ fit

-- Theorem stating that a larger R² indicates a better model fit
theorem better_fit_larger_R_squared 
  (R1 R2 fit1 fit2 : ℝ) 
  (h1 : correlation_index R1) 
  (h2 : correlation_index R2) 
  (h3 : model_fit fit1) 
  (h4 : model_fit fit2) 
  (h5 : R1 < R2) : 
  fit1 < fit2 := by
sorry


end NUMINAMATH_CALUDE_better_fit_larger_R_squared_l648_64870


namespace NUMINAMATH_CALUDE_exp_iff_gt_l648_64802

-- Define the exponential function as monotonically increasing on ℝ
axiom exp_monotone : ∀ (x y : ℝ), x < y → Real.exp x < Real.exp y

theorem exp_iff_gt (a b : ℝ) : a > b ↔ Real.exp a > Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_iff_gt_l648_64802


namespace NUMINAMATH_CALUDE_max_value_of_f_l648_64812

def f (x : ℝ) : ℝ := x^2 + 4*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l648_64812


namespace NUMINAMATH_CALUDE_bits_of_88888_base16_l648_64869

/-- The number of bits required to represent 88888₁₆ in base-2 is 20. -/
theorem bits_of_88888_base16 : ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, 2^m > 88888 * 16^4 + 88888 * 16^3 + 88888 * 16^2 + 88888 * 16 + 88888 → m ≥ n) ∧
  2^n > 88888 * 16^4 + 88888 * 16^3 + 88888 * 16^2 + 88888 * 16 + 88888 :=
by sorry

end NUMINAMATH_CALUDE_bits_of_88888_base16_l648_64869


namespace NUMINAMATH_CALUDE_completing_square_addition_l648_64814

theorem completing_square_addition (x : ℝ) : 
  (∃ k : ℝ, (x^2 - 4*x + k)^(1/2) = x - 2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_addition_l648_64814


namespace NUMINAMATH_CALUDE_unique_solution_l648_64850

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_solution (a x y p : ℕ) : Prop :=
  is_single_digit a ∧ is_single_digit x ∧ is_single_digit y ∧ is_single_digit p ∧
  a ≠ x ∧ a ≠ y ∧ a ≠ p ∧ x ≠ y ∧ x ≠ p ∧ y ≠ p ∧
  10 * a + x + 10 * y + x = 100 * y + 10 * p + a

theorem unique_solution :
  ∀ a x y p : ℕ, is_solution a x y p → a = 8 ∧ x = 9 ∧ y = 1 ∧ p = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l648_64850


namespace NUMINAMATH_CALUDE_unknown_number_value_l648_64866

theorem unknown_number_value (y : ℝ) : (12 : ℝ)^3 * y^4 / 432 = 5184 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l648_64866


namespace NUMINAMATH_CALUDE_line_does_not_intersect_circle_l648_64833

/-- Proves that a line does not intersect a circle given the radius and distance from center to line -/
theorem line_does_not_intersect_circle (r d : ℝ) (hr : r = 10) (hd : d = 13) :
  d > r → ¬ (∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2 ∧ d = |p.1|) :=
by sorry

end NUMINAMATH_CALUDE_line_does_not_intersect_circle_l648_64833


namespace NUMINAMATH_CALUDE_competition_problem_l648_64816

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  onlyA : ℕ
  onlyB : ℕ
  onlyC : ℕ
  AB : ℕ
  AC : ℕ
  BC : ℕ
  ABC : ℕ

/-- The theorem statement -/
theorem competition_problem (s : ProblemSolvers) : s.onlyB = 6 :=
  by
  have total : s.onlyA + s.onlyB + s.onlyC + s.AB + s.AC + s.BC + s.ABC = 25 := by sorry
  have solved_A : s.onlyA = s.AB + s.AC + s.ABC + 1 := by sorry
  have not_A_BC : s.onlyB + s.BC = 2 * (s.onlyC + s.BC) := by sorry
  have only_one_not_A : s.onlyB + s.onlyC = s.onlyA := by sorry
  sorry

end NUMINAMATH_CALUDE_competition_problem_l648_64816


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l648_64856

/-- The total surface area of a cuboid with dimensions in the ratio 6:5:4 and actual dimensions 90 cm, 75 cm, and 60 cm is 33300 cm². -/
theorem cuboid_surface_area : 
  let length : ℝ := 90
  let breadth : ℝ := 75
  let height : ℝ := 60
  let ratio_length : ℝ := 6
  let ratio_breadth : ℝ := 5
  let ratio_height : ℝ := 4
  -- Ensure the dimensions are in the correct ratio
  length / ratio_length = breadth / ratio_breadth ∧ 
  breadth / ratio_breadth = height / ratio_height →
  -- Calculate the total surface area
  2 * (length * breadth + breadth * height + height * length) = 33300 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l648_64856


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l648_64844

/-- Proves that given a 6-liter solution with an unknown initial alcohol percentage,
    adding 1.8 liters of pure alcohol to create a 50% alcohol solution
    implies that the initial alcohol percentage was 35%. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.8)
  (h3 : final_percentage = 50)
  (h4 : final_percentage / 100 * (initial_volume + added_alcohol) = 
        (initial_volume * x / 100) + added_alcohol) :
  x = 35 :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_percentage_l648_64844


namespace NUMINAMATH_CALUDE_total_apples_is_45_l648_64874

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

/-- Theorem stating that the total number of apples is 45.0 -/
theorem total_apples_is_45 : total_apples = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_45_l648_64874


namespace NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l648_64891

/-- Represents the craft selling scenario -/
structure CraftSelling where
  cost_price : ℕ
  base_price : ℕ
  base_volume : ℕ
  price_volume_ratio : ℕ
  max_price : ℕ

/-- Calculates the daily sales volume based on the selling price -/
def daily_volume (cs : CraftSelling) (price : ℕ) : ℤ :=
  cs.base_volume - cs.price_volume_ratio * (price - cs.base_price)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (cs : CraftSelling) (price : ℕ) : ℤ :=
  (price - cs.cost_price) * daily_volume cs price

/-- The craft selling scenario for the given problem -/
def craft_scenario : CraftSelling := {
  cost_price := 30
  base_price := 40
  base_volume := 80
  price_volume_ratio := 2
  max_price := 55
}

/-- Theorem for the daily sales profit at 45 yuan -/
theorem profit_at_45 : daily_profit craft_scenario 45 = 1050 := by sorry

/-- Theorem for the selling price that achieves 1200 yuan daily profit -/
theorem price_for_1200_profit :
  ∃ (price : ℕ), price ≤ craft_scenario.max_price ∧ daily_profit craft_scenario price = 1200 ∧
  ∀ (p : ℕ), p ≤ craft_scenario.max_price → daily_profit craft_scenario p = 1200 → p = price := by sorry

end NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l648_64891


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l648_64878

theorem largest_x_floor_div : 
  ∀ x : ℝ, (↑⌊x⌋ / x = 7 / 8) → x ≤ 48 / 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l648_64878


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l648_64851

theorem smallest_m_for_integral_solutions : 
  (∀ m : ℕ, m > 0 ∧ m < 160 → ¬∃ x : ℤ, 10 * x^2 - m * x + 630 = 0) ∧ 
  (∃ x : ℤ, 10 * x^2 - 160 * x + 630 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l648_64851


namespace NUMINAMATH_CALUDE_line_slope_l648_64892

/-- Given a line l with equation y = (1/2)x + 1, its slope is 1/2 -/
theorem line_slope (l : Set (ℝ × ℝ)) (h : l = {(x, y) | y = (1/2) * x + 1}) :
  (∃ m : ℝ, ∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ 
  (∀ m : ℝ, (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) → m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l648_64892


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l648_64881

/-- The total surface area of a cylinder with height 8 and radius 5 is 130π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 5
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 130 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l648_64881


namespace NUMINAMATH_CALUDE_triangle_area_is_12_l648_64845

/-- The area of a triangular region bounded by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
def xIntercept : ℝ := 4

/-- The y-intercept of the line -/
def yIntercept : ℝ := 6

theorem triangle_area_is_12 :
  triangleArea = (1 / 2) * xIntercept * yIntercept :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_12_l648_64845


namespace NUMINAMATH_CALUDE_exponent_multiplication_l648_64862

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l648_64862


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_line_l648_64888

/-- Given a function f(x) = a*sin(x) + b*cos(x) where x is real,
    if x₀ is an axis of symmetry for f(x) and tan(x₀) = 2,
    then the point (a,b) lies on the line x - 2y = 0. -/
theorem symmetry_axis_implies_line (a b x₀ : ℝ) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x, f (x₀ + x) = f (x₀ - x)) →  -- x₀ is an axis of symmetry
  Real.tan x₀ = 2 →
  a - 2 * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_line_l648_64888


namespace NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l648_64817

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l648_64817
