import Mathlib

namespace NUMINAMATH_CALUDE_f_f_zero_equals_3pi_squared_minus_4_l3718_371898

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

-- Theorem statement
theorem f_f_zero_equals_3pi_squared_minus_4 :
  f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_equals_3pi_squared_minus_4_l3718_371898


namespace NUMINAMATH_CALUDE_minimum_garden_width_l3718_371897

theorem minimum_garden_width (w : ℝ) (l : ℝ) :
  w > 0 →
  l = w + 10 →
  w * l ≥ 120 →
  w ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_garden_width_l3718_371897


namespace NUMINAMATH_CALUDE_vision_survey_is_sampling_l3718_371811

/-- Represents a survey method -/
inductive SurveyMethod
| Sampling
| Census
| Other

/-- Represents a school with a given population of eighth-grade students -/
structure School where
  population : ℕ

/-- Represents a vision survey conducted in a school -/
structure VisionSurvey where
  school : School
  sample_size : ℕ
  selection_method : String

/-- Determines the survey method based on the vision survey parameters -/
def determine_survey_method (survey : VisionSurvey) : SurveyMethod :=
  if survey.sample_size < survey.school.population ∧ survey.selection_method = "Random" then
    SurveyMethod.Sampling
  else if survey.sample_size = survey.school.population then
    SurveyMethod.Census
  else
    SurveyMethod.Other

/-- Theorem stating that the given vision survey uses a sampling survey method -/
theorem vision_survey_is_sampling (school : School) (survey : VisionSurvey) :
  school.population = 400 →
  survey.school = school →
  survey.sample_size = 80 →
  survey.selection_method = "Random" →
  determine_survey_method survey = SurveyMethod.Sampling :=
by
  sorry

#check vision_survey_is_sampling

end NUMINAMATH_CALUDE_vision_survey_is_sampling_l3718_371811


namespace NUMINAMATH_CALUDE_unique_triple_l3718_371867

theorem unique_triple : ∃! (x y z : ℕ+), 
  x ≤ y ∧ y ≤ z ∧ 
  x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧
  x = 2 ∧ y = 251 ∧ z = 252 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l3718_371867


namespace NUMINAMATH_CALUDE_league_games_l3718_371831

theorem league_games (n : ℕ) (m : ℕ) (h1 : n = 20) (h2 : m = 4) :
  (n * (n - 1) / 2) * m = 760 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l3718_371831


namespace NUMINAMATH_CALUDE_inverse_f_at_2_l3718_371851

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_2 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≥ 0, f_inv (f x) = x) ∧
    (∀ y ≥ -1, f (f_inv y) = y) ∧
    f_inv 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_2_l3718_371851


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3718_371833

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3718_371833


namespace NUMINAMATH_CALUDE_distinct_words_count_l3718_371856

def first_digit (n : ℕ) : ℕ := 
  -- Definition of the function that returns the first digit of 2^n
  sorry

def word_sequence (start : ℕ) : List ℕ := 
  -- Definition of a list of 13 consecutive terms starting from 'start'
  (List.range 13).map (λ i => first_digit (start + i))

def distinct_words : Finset (List ℕ) :=
  -- Set of all distinct words in the sequence
  sorry

theorem distinct_words_count : Finset.card distinct_words = 57 := by
  sorry

end NUMINAMATH_CALUDE_distinct_words_count_l3718_371856


namespace NUMINAMATH_CALUDE_balloons_given_l3718_371889

theorem balloons_given (initial : ℕ) (current : ℕ) (given : ℕ) :
  initial = 709 →
  current = 488 →
  given = initial - current →
  given = 221 := by
sorry

end NUMINAMATH_CALUDE_balloons_given_l3718_371889


namespace NUMINAMATH_CALUDE_percent_increase_in_sales_l3718_371846

def sales_last_year : ℝ := 320
def sales_this_year : ℝ := 480

theorem percent_increase_in_sales :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_in_sales_l3718_371846


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l3718_371801

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem -/
theorem two_digit_reverse_sum_square :
  (∃ (S : Finset ℕ), S.card = 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n)) ∧
  ¬∃ (S : Finset ℕ), S.card > 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n) := by
  sorry


end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l3718_371801


namespace NUMINAMATH_CALUDE_smallest_fixed_point_of_R_l3718_371812

/-- The transformation R that reflects a line first on l₁: y = √3x and then on l₂: y = -√3x -/
def R (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The n-th iteration of R -/
def R_iter (n : ℕ) (l : ℝ → ℝ) : ℝ → ℝ :=
  match n with
  | 0 => l
  | n + 1 => R (R_iter n l)

/-- Any line can be represented as y = kx for some k -/
def line (k : ℝ) : ℝ → ℝ := λ x => k * x

theorem smallest_fixed_point_of_R :
  ∀ k : ℝ, ∃ m : ℕ, m > 0 ∧ R_iter m (line k) = line k ∧
  ∀ n : ℕ, 0 < n → n < m → R_iter n (line k) ≠ line k :=
by sorry

end NUMINAMATH_CALUDE_smallest_fixed_point_of_R_l3718_371812


namespace NUMINAMATH_CALUDE_birthday_45_days_later_l3718_371849

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

/-- Function to add days to a given day of the week -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (start, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 0) => DayOfWeek.Monday
  | (DayOfWeek.Monday, 1) => DayOfWeek.Tuesday
  | (DayOfWeek.Monday, 2) => DayOfWeek.Wednesday
  | (DayOfWeek.Monday, 3) => DayOfWeek.Thursday
  | (DayOfWeek.Monday, 4) => DayOfWeek.Friday
  | (DayOfWeek.Monday, 5) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 6) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 0) => DayOfWeek.Tuesday
  | (DayOfWeek.Tuesday, 1) => DayOfWeek.Wednesday
  | (DayOfWeek.Tuesday, 2) => DayOfWeek.Thursday
  | (DayOfWeek.Tuesday, 3) => DayOfWeek.Friday
  | (DayOfWeek.Tuesday, 4) => DayOfWeek.Saturday
  | (DayOfWeek.Tuesday, 5) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 6) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 0) => DayOfWeek.Wednesday
  | (DayOfWeek.Wednesday, 1) => DayOfWeek.Thursday
  | (DayOfWeek.Wednesday, 2) => DayOfWeek.Friday
  | (DayOfWeek.Wednesday, 3) => DayOfWeek.Saturday
  | (DayOfWeek.Wednesday, 4) => DayOfWeek.Sunday
  | (DayOfWeek.Wednesday, 5) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 6) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 0) => DayOfWeek.Thursday
  | (DayOfWeek.Thursday, 1) => DayOfWeek.Friday
  | (DayOfWeek.Thursday, 2) => DayOfWeek.Saturday
  | (DayOfWeek.Thursday, 3) => DayOfWeek.Sunday
  | (DayOfWeek.Thursday, 4) => DayOfWeek.Monday
  | (DayOfWeek.Thursday, 5) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 6) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 0) => DayOfWeek.Friday
  | (DayOfWeek.Friday, 1) => DayOfWeek.Saturday
  | (DayOfWeek.Friday, 2) => DayOfWeek.Sunday
  | (DayOfWeek.Friday, 3) => DayOfWeek.Monday
  | (DayOfWeek.Friday, 4) => DayOfWeek.Tuesday
  | (DayOfWeek.Friday, 5) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 6) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 0) => DayOfWeek.Saturday
  | (DayOfWeek.Saturday, 1) => DayOfWeek.Sunday
  | (DayOfWeek.Saturday, 2) => DayOfWeek.Monday
  | (DayOfWeek.Saturday, 3) => DayOfWeek.Tuesday
  | (DayOfWeek.Saturday, 4) => DayOfWeek.Wednesday
  | (DayOfWeek.Saturday, 5) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 6) => DayOfWeek.Friday
  | _ => DayOfWeek.Sunday  -- This case should never happen

theorem birthday_45_days_later (birthday : DayOfWeek) :
  birthday = DayOfWeek.Tuesday → addDays birthday 45 = DayOfWeek.Friday :=
by sorry

end NUMINAMATH_CALUDE_birthday_45_days_later_l3718_371849


namespace NUMINAMATH_CALUDE_G_equals_negative_three_F_l3718_371892

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((5 * x - x^3) / (1 - 5 * x^2))

theorem G_equals_negative_three_F (x : ℝ) : G x = -3 * F x :=
by sorry

end NUMINAMATH_CALUDE_G_equals_negative_three_F_l3718_371892


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l3718_371872

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value from S.S. AOPS in base 5 -/
def aops_value : List ℕ := [4, 2, 1, 3]

/-- The value from S.S. BOPS in base 7 -/
def bops_value : List ℕ := [2, 1, 0, 1]

/-- The value from S.S. COPS in base 8 -/
def cops_value : List ℕ := [3, 2, 1]

/-- The theorem to be proved -/
theorem pirate_loot_sum :
  to_base_10 aops_value 5 + to_base_10 bops_value 7 + to_base_10 cops_value 8 = 849 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l3718_371872


namespace NUMINAMATH_CALUDE_point_A_outside_circle_l3718_371806

/-- The position of point A on the number line after t seconds -/
def position_A (t : ℝ) : ℝ := 2 * t

/-- The center of circle B on the number line -/
def center_B : ℝ := 16

/-- The radius of circle B -/
def radius_B : ℝ := 4

/-- Predicate for point A being outside circle B -/
def is_outside_circle (t : ℝ) : Prop :=
  position_A t < center_B - radius_B ∨ position_A t > center_B + radius_B

theorem point_A_outside_circle (t : ℝ) :
  is_outside_circle t ↔ t < 6 ∨ t > 10 := by sorry

end NUMINAMATH_CALUDE_point_A_outside_circle_l3718_371806


namespace NUMINAMATH_CALUDE_final_book_count_l3718_371845

/-- Represents the number of books in the library system -/
structure LibraryState where
  books : ℕ

/-- Represents a transaction that changes the number of books -/
inductive Transaction
  | TakeOut (n : ℕ)
  | Return (n : ℕ)
  | Withdraw (n : ℕ)

/-- Applies a transaction to the library state -/
def applyTransaction (state : LibraryState) (t : Transaction) : LibraryState :=
  match t with
  | Transaction.TakeOut n => ⟨state.books - n⟩
  | Transaction.Return n => ⟨state.books + n⟩
  | Transaction.Withdraw n => ⟨state.books - n⟩

/-- Applies a list of transactions to the library state -/
def applyTransactions (state : LibraryState) (ts : List Transaction) : LibraryState :=
  ts.foldl applyTransaction state

/-- The initial state of the library -/
def initialState : LibraryState := ⟨250⟩

/-- The transactions that occur over the three weeks -/
def transactions : List Transaction := [
  Transaction.TakeOut 120,  -- Week 1 Tuesday
  Transaction.Return 35,    -- Week 1 Wednesday
  Transaction.Withdraw 15,  -- Week 1 Thursday
  Transaction.TakeOut 42,   -- Week 1 Friday
  Transaction.Return 72,    -- Week 2 Monday (60% of 120)
  Transaction.Return 34,    -- Week 2 Tuesday (80% of 42, rounded)
  Transaction.Withdraw 75,  -- Week 2 Wednesday
  Transaction.TakeOut 40,   -- Week 2 Thursday
  Transaction.Return 20,    -- Week 3 Monday (50% of 40)
  Transaction.TakeOut 20,   -- Week 3 Tuesday
  Transaction.Return 46,    -- Week 3 Wednesday (95% of 48, rounded)
  Transaction.Withdraw 10,  -- Week 3 Thursday
  Transaction.TakeOut 55    -- Week 3 Friday
]

/-- The theorem stating that after applying all transactions, the library has 80 books -/
theorem final_book_count :
  (applyTransactions initialState transactions).books = 80 := by
  sorry

end NUMINAMATH_CALUDE_final_book_count_l3718_371845


namespace NUMINAMATH_CALUDE_rhombus_construction_exists_l3718_371886

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define a rhombus
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

-- Define the property of sides being parallel to diagonals
def parallel_to_diagonals (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

-- Define the property of vertices lying on the sides of the quadrilateral
def vertices_on_sides (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

theorem rhombus_construction_exists (q : ConvexQuadrilateral) :
  ∃ (r : Rhombus), vertices_on_sides r q ∧ parallel_to_diagonals r q :=
sorry

end NUMINAMATH_CALUDE_rhombus_construction_exists_l3718_371886


namespace NUMINAMATH_CALUDE_mary_credit_limit_l3718_371874

/-- The credit limit at Mary's grocery store -/
def credit_limit : ℕ := sorry

/-- The amount Mary paid on Tuesday -/
def tuesday_payment : ℕ := 15

/-- The amount Mary paid on Thursday -/
def thursday_payment : ℕ := 23

/-- The amount Mary still needs to pay -/
def remaining_payment : ℕ := 62

theorem mary_credit_limit : 
  credit_limit = tuesday_payment + thursday_payment + remaining_payment := by sorry

end NUMINAMATH_CALUDE_mary_credit_limit_l3718_371874


namespace NUMINAMATH_CALUDE_workshop_theorem_l3718_371880

def workshop_problem (total_members : ℕ) (avg_age_all : ℝ) 
                     (num_girls : ℕ) (num_boys : ℕ) (num_adults : ℕ) 
                     (avg_age_girls : ℝ) (avg_age_boys : ℝ) : Prop :=
  let total_age := total_members * avg_age_all
  let girls_age := num_girls * avg_age_girls
  let boys_age := num_boys * avg_age_boys
  let adults_age := total_age - girls_age - boys_age
  (adults_age / num_adults) = 26.2

theorem workshop_theorem : 
  workshop_problem 50 20 22 18 10 18 19 := by
  sorry

end NUMINAMATH_CALUDE_workshop_theorem_l3718_371880


namespace NUMINAMATH_CALUDE_polynomial_composition_problem_l3718_371853

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem polynomial_composition_problem (a : ℝ) (m n : ℕ) 
  (h1 : a > 0)
  (h2 : P (P (P a)) = 99)
  (h3 : a^2 = m + Real.sqrt n)
  (h4 : n > 0)
  (h5 : ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ n)) :
  m + n = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_problem_l3718_371853


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l3718_371860

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the properties of the triangle
def triangle_properties (t : IsoscelesTriangle) (area1 area2 : ℝ) : Prop :=
  area1 = 6 * 6 / 11 ∧ 
  area2 = 5 * 5 / 11 ∧ 
  area1 + area2 = 1 / 2 * t.base * (t.leg ^ 2 - (t.base / 2) ^ 2).sqrt

-- Theorem statement
theorem isosceles_triangle_sides 
  (t : IsoscelesTriangle) 
  (area1 area2 : ℝ) 
  (h : triangle_properties t area1 area2) : 
  t.base = 6 ∧ t.leg = 5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_sides_l3718_371860


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3718_371879

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3718_371879


namespace NUMINAMATH_CALUDE_parabola_intersection_dot_product_l3718_371871

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the intersection of the line and the parabola
def intersection (k : ℝ) (p : PointOnParabola) : Prop :=
  line_through_focus k p.x p.y

theorem parabola_intersection_dot_product :
  ∀ (k : ℝ) (A B : PointOnParabola),
    intersection k A →
    intersection k B →
    A ≠ B →
    A.x * B.x + A.y * B.y = -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_dot_product_l3718_371871


namespace NUMINAMATH_CALUDE_conjunction_false_l3718_371810

theorem conjunction_false (p q : Prop) (hp : p) (hq : ¬q) : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_conjunction_false_l3718_371810


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3718_371896

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + 3*x^2 + 6) * (2*x^3 + x^2 + 10) = x^7 + 2*x^6 + 3*x^5 + 10*x^4 + 6*x^3 + 30*x^2 + 60 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l3718_371896


namespace NUMINAMATH_CALUDE_total_wage_calculation_l3718_371827

-- Define the basic parameters
def basic_rate : ℝ := 20
def regular_hours : ℕ := 40
def total_hours : ℕ := 48
def overtime_rate_increase : ℝ := 0.25

-- Define the calculation functions
def regular_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours
def overtime_rate (rate : ℝ) (increase : ℝ) : ℝ := rate * (1 + increase)
def overtime_hours (total : ℕ) (regular : ℕ) : ℕ := total - regular
def overtime_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours

-- Theorem statement
theorem total_wage_calculation :
  let reg_pay := regular_pay basic_rate regular_hours
  let ot_rate := overtime_rate basic_rate overtime_rate_increase
  let ot_hours := overtime_hours total_hours regular_hours
  let ot_pay := overtime_pay ot_rate ot_hours
  reg_pay + ot_pay = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_wage_calculation_l3718_371827


namespace NUMINAMATH_CALUDE_local_minimum_condition_l3718_371841

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- Define the derivative of f(x)
def f_prime (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*b

-- Theorem statement
theorem local_minimum_condition (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ IsLocalMin (f b) x) →
  (f_prime b 0 < 0 ∧ f_prime b 1 > 0) :=
sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l3718_371841


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3718_371873

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (2*a - b)^2 - (2*a + b)*(b - 2*a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3718_371873


namespace NUMINAMATH_CALUDE_initial_rope_length_correct_l3718_371836

/-- The initial length of rope before decorating trees -/
def initial_rope_length : ℝ := 8.9

/-- The length of string used to decorate one tree -/
def string_per_tree : ℝ := 0.84

/-- The number of trees decorated -/
def num_trees : ℕ := 10

/-- The length of rope remaining after decorating trees -/
def remaining_rope : ℝ := 0.5

/-- Theorem stating that the initial rope length is correct -/
theorem initial_rope_length_correct :
  initial_rope_length = string_per_tree * num_trees + remaining_rope :=
by sorry

end NUMINAMATH_CALUDE_initial_rope_length_correct_l3718_371836


namespace NUMINAMATH_CALUDE_power_sum_difference_l3718_371848

theorem power_sum_difference : 2^(1+2+3+4) - (2^1 + 2^2 + 2^3 + 2^4) = 994 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3718_371848


namespace NUMINAMATH_CALUDE_sally_and_jolly_money_l3718_371821

theorem sally_and_jolly_money (total : ℕ) (jolly_plus_20 : ℕ) :
  total = 150 →
  jolly_plus_20 = 70 →
  ∃ (sally : ℕ) (jolly : ℕ),
    sally + jolly = total ∧
    jolly + 20 = jolly_plus_20 ∧
    sally = 100 ∧
    jolly = 50 :=
by sorry

end NUMINAMATH_CALUDE_sally_and_jolly_money_l3718_371821


namespace NUMINAMATH_CALUDE_hcf_problem_l3718_371808

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2560) (h2 : Nat.lcm a b = 128) :
  Nat.gcd a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3718_371808


namespace NUMINAMATH_CALUDE_fifteen_power_division_l3718_371800

theorem fifteen_power_division : (15 : ℕ) ^ 11 / (15 : ℕ) ^ 8 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_power_division_l3718_371800


namespace NUMINAMATH_CALUDE_bakery_order_cost_is_54_l3718_371858

/-- Calculates the final cost of a bakery order with a possible discount --/
def bakery_order_cost (quiche_price croissant_price biscuit_price : ℚ) 
  (quiche_quantity croissant_quantity biscuit_quantity : ℕ)
  (discount_rate : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_before_discount := 
    quiche_price * quiche_quantity + 
    croissant_price * croissant_quantity + 
    biscuit_price * biscuit_quantity
  let discount := 
    if total_before_discount > discount_threshold 
    then total_before_discount * discount_rate 
    else 0
  total_before_discount - discount

/-- Theorem stating that the bakery order cost is $54.00 given the specified conditions --/
theorem bakery_order_cost_is_54 : 
  bakery_order_cost 15 3 2 2 6 6 (1/10) 50 = 54 := by
  sorry

end NUMINAMATH_CALUDE_bakery_order_cost_is_54_l3718_371858


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l3718_371854

/-- Given a quadrilateral EFGH with the following properties:
  - m∠F = m∠G = 135°
  - EF = 4
  - FG = 6
  - GH = 8
  Prove that the area of EFGH is 18√2 -/
theorem area_of_quadrilateral (E F G H : ℝ × ℝ) : 
  let angle (A B C : ℝ × ℝ) := Real.arccos ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2)) / 
    (((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2)^(1/2))
  angle F E G = 135 * π / 180 ∧
  angle G F H = 135 * π / 180 ∧
  ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) = 4 ∧
  ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) = 6 ∧
  ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) = 8 →
  let area := 
    1/2 * ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * Real.sin (angle F E G) +
    1/2 * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) * Real.sin (angle G F H)
  area = 18 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l3718_371854


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3718_371888

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 5 ∧ b = 12 ∧ c = 13)

/-- Square inscribed in a right triangle with one vertex at the right angle -/
def square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x ≤ min t.a t.b

/-- Square inscribed in a right triangle with one side on the hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y ≤ t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : square_at_right_angle t1 x)
  (h2 : square_on_hypotenuse t2 y) :
  x / y = 1800 / 2863 := by
  sorry

#check inscribed_squares_ratio

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3718_371888


namespace NUMINAMATH_CALUDE_distinct_collections_l3718_371894

/-- Represents the count of each letter in MATHEMATICSH -/
def letter_count : Finset (Char × ℕ) :=
  {('A', 3), ('E', 1), ('I', 1), ('T', 2), ('M', 2), ('H', 2), ('C', 1), ('S', 1)}

/-- The set of vowels in MATHEMATICSH -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- The set of consonants in MATHEMATICSH -/
def consonants : Finset Char := {'T', 'M', 'H', 'C', 'S'}

/-- The number of distinct vowel combinations -/
def vowel_combinations : ℕ := 5

/-- The number of distinct consonant combinations -/
def consonant_combinations : ℕ := 48

/-- Theorem stating the number of distinct possible collections -/
theorem distinct_collections :
  vowel_combinations * consonant_combinations = 240 :=
by sorry

end NUMINAMATH_CALUDE_distinct_collections_l3718_371894


namespace NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l3718_371877

-- Definition of opposite numbers
def are_opposite (a b : ℝ) : Prop := (abs a = abs b) ∧ (a = -b)

-- Theorem to prove
theorem three_and_negative_three_are_opposite : are_opposite 3 (-3) := by
  sorry

end NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l3718_371877


namespace NUMINAMATH_CALUDE_rectangle_new_perimeter_l3718_371869

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased so that the new area is 1 (1/3) times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_new_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  2 * (width + new_area / width) = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_new_perimeter_l3718_371869


namespace NUMINAMATH_CALUDE_permit_increase_l3718_371844

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letters in old permits -/
def old_permit_letters : ℕ := 2

/-- The number of digits in old permits -/
def old_permit_digits : ℕ := 3

/-- The number of letters in new permits -/
def new_permit_letters : ℕ := 4

/-- The number of digits in new permits -/
def new_permit_digits : ℕ := 4

/-- The ratio of new permits to old permits -/
def permit_ratio : ℕ := 67600

theorem permit_increase :
  (alphabet_size ^ new_permit_letters * digit_count ^ new_permit_digits) /
  (alphabet_size ^ old_permit_letters * digit_count ^ old_permit_digits) = permit_ratio :=
sorry

end NUMINAMATH_CALUDE_permit_increase_l3718_371844


namespace NUMINAMATH_CALUDE_sum_of_coefficients_for_specific_polynomial_l3718_371890

/-- A polynomial with real coefficients -/
def RealPolynomial (p q r s : ℝ) : ℂ → ℂ :=
  fun x => x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: Sum of coefficients for a specific polynomial -/
theorem sum_of_coefficients_for_specific_polynomial
  (p q r s : ℝ) :
  (RealPolynomial p q r s (3*I) = 0) →
  (RealPolynomial p q r s (3+I) = 0) →
  p + q + r + s = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_for_specific_polynomial_l3718_371890


namespace NUMINAMATH_CALUDE_base_ten_proof_l3718_371834

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 162_b + 235_b = 407_b holds for a given base b --/
def equation_holds (b : ℕ) : Prop :=
  to_decimal 162 b + to_decimal 235 b = to_decimal 407 b

theorem base_ten_proof :
  ∃! b : ℕ, b > 1 ∧ equation_holds b ∧ b = 10 :=
sorry

end NUMINAMATH_CALUDE_base_ten_proof_l3718_371834


namespace NUMINAMATH_CALUDE_sale_month2_l3718_371884

def average_sale : ℕ := 5600
def num_months : ℕ := 6
def sale_month1 : ℕ := 5400
def sale_month3 : ℕ := 6300
def sale_month4 : ℕ := 7200
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200

theorem sale_month2 : 
  average_sale * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) = 9000 :=
by sorry

end NUMINAMATH_CALUDE_sale_month2_l3718_371884


namespace NUMINAMATH_CALUDE_dustin_reading_speed_l3718_371824

/-- The number of pages Sam can read in an hour -/
def sam_pages_per_hour : ℕ := 24

/-- The additional pages Dustin reads compared to Sam in 40 minutes -/
def dustin_additional_pages : ℕ := 34

/-- The time period in minutes for which the comparison is made -/
def comparison_time : ℕ := 40

/-- The number of pages Dustin can read in an hour -/
def dustin_pages_per_hour : ℕ := 75

theorem dustin_reading_speed :
  dustin_pages_per_hour = 75 :=
by sorry

end NUMINAMATH_CALUDE_dustin_reading_speed_l3718_371824


namespace NUMINAMATH_CALUDE_store_visitor_count_l3718_371814

/-- The number of people who entered the store in the first hour -/
def first_hour_entry : ℕ := 94

/-- The number of people who left the store in the first hour -/
def first_hour_exit : ℕ := 27

/-- The number of people who left the store in the second hour -/
def second_hour_exit : ℕ := 9

/-- The number of people remaining in the store after 2 hours -/
def remaining_after_two_hours : ℕ := 76

/-- The number of people who entered the store in the second hour -/
def second_hour_entry : ℕ := 18

theorem store_visitor_count :
  (first_hour_entry - first_hour_exit) + second_hour_entry - second_hour_exit = remaining_after_two_hours :=
by sorry

end NUMINAMATH_CALUDE_store_visitor_count_l3718_371814


namespace NUMINAMATH_CALUDE_m_neg_one_necessary_not_sufficient_l3718_371829

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop := ∃ (k₁ k₂ : ℝ), k₁ * k₂ = -1 ∧
  (∀ (x y : ℝ), l₁ m x y → m * x + (2 * m - 1) * y = k₁) ∧
  (∀ (x y : ℝ), l₂ m x y → 3 * x + m * y = k₂)

-- State the theorem
theorem m_neg_one_necessary_not_sufficient :
  (∀ m : ℝ, m = -1 → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_m_neg_one_necessary_not_sufficient_l3718_371829


namespace NUMINAMATH_CALUDE_divisible_by_37_l3718_371802

theorem divisible_by_37 (n d : ℕ) (h : d ≤ 9) : 
  ∃ k : ℕ, (d * (10^(3*n) - 1) / 9) = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_37_l3718_371802


namespace NUMINAMATH_CALUDE_sean_cricket_theorem_l3718_371883

def sean_cricket_problem (total_days : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : Prop :=
  let sean_total_minutes := total_minutes - indira_minutes
  sean_total_minutes / total_days = 50

theorem sean_cricket_theorem :
  sean_cricket_problem 14 1512 812 := by
  sorry

end NUMINAMATH_CALUDE_sean_cricket_theorem_l3718_371883


namespace NUMINAMATH_CALUDE_set_operations_l3718_371863

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Define the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 5}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 6}) ∧
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x ≤ 6}) ∧
  ((A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3718_371863


namespace NUMINAMATH_CALUDE_cosine_period_l3718_371857

theorem cosine_period (ω : ℝ) (h1 : ω > 0) : 
  (∃ y : ℝ → ℝ, y = λ x => Real.cos (ω * x - π / 6)) →
  (π / 5 = 2 * π / ω) →
  ω = 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_period_l3718_371857


namespace NUMINAMATH_CALUDE_solution_range_l3718_371817

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 2 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 3 * Real.sqrt (x - 1)) = 2 → 
  2 ≤ x ∧ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_range_l3718_371817


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3718_371822

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 1}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 4)}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 1 4 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3718_371822


namespace NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l3718_371878

/-- The area of a triangle formed by the diagonal and one side of a rectangle with length 40 units and width 24 units is 480 square units. -/
theorem rectangle_diagonal_triangle_area : 
  let rectangle_length : ℝ := 40
  let rectangle_width : ℝ := 24
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let triangle_area : ℝ := rectangle_area / 2
  triangle_area = 480 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l3718_371878


namespace NUMINAMATH_CALUDE_emily_walks_farther_l3718_371887

/-- The distance Emily walks farther than Troy over five days -/
def distance_difference (troy_distance emily_distance : ℕ) : ℕ :=
  ((emily_distance - troy_distance) * 2) * 5

/-- Theorem stating the difference in distance walked by Emily and Troy over five days -/
theorem emily_walks_farther :
  distance_difference 75 98 = 230 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l3718_371887


namespace NUMINAMATH_CALUDE_triangle_area_angle_l3718_371825

theorem triangle_area_angle (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2) →
  S = (1/2) * a * b * Real.sin C →
  C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l3718_371825


namespace NUMINAMATH_CALUDE_equation_solutions_l3718_371862

theorem equation_solutions :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3718_371862


namespace NUMINAMATH_CALUDE_complement_of_union_l3718_371828

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3718_371828


namespace NUMINAMATH_CALUDE_unique_solution_abc_l3718_371865

theorem unique_solution_abc : 
  ∀ a b c : ℕ+, 
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + a * c) + 18) → 
    (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l3718_371865


namespace NUMINAMATH_CALUDE_road_signs_total_l3718_371815

/-- The total number of road signs at six intersections -/
def total_road_signs (first second third fourth fifth sixth : ℕ) : ℕ :=
  first + second + third + fourth + fifth + sixth

/-- Theorem stating the total number of road signs at six intersections -/
theorem road_signs_total : ∃ (first second third fourth fifth sixth : ℕ),
  (first = 50) ∧
  (second = first + first / 5) ∧
  (third = 2 * second - 10) ∧
  (fourth = ((first + second) + 1) / 2) ∧
  (fifth = third - second) ∧
  (sixth = first + fourth - 15) ∧
  (total_road_signs first second third fourth fifth sixth = 415) :=
by sorry

end NUMINAMATH_CALUDE_road_signs_total_l3718_371815


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l3718_371852

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ

/-- Represents the state of the runners on the track -/
structure RunnerState where
  track : Track
  runner1 : Runner
  runner2 : Runner
  meetingPoints : Finset ℝ  -- Set of points where runners meet

/-- The theorem statement -/
theorem runners_speed_ratio 
  (state : RunnerState) 
  (h1 : state.runner1.direction ≠ state.runner2.direction)  -- Runners move in opposite directions
  (h2 : state.runner1.speed ≠ 0 ∧ state.runner2.speed ≠ 0)  -- Both runners have non-zero speed
  (h3 : state.meetingPoints.card = 3)  -- There are exactly three meeting points
  (h4 : ∀ p ∈ state.meetingPoints, p < state.track.circumference)  -- Meeting points are on the track
  : state.runner2.speed / state.runner1.speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l3718_371852


namespace NUMINAMATH_CALUDE_equation_to_general_form_l3718_371864

theorem equation_to_general_form :
  ∀ x : ℝ, 5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_to_general_form_l3718_371864


namespace NUMINAMATH_CALUDE_altitudes_sum_eq_nine_inradius_implies_equilateral_l3718_371847

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a vertex to the opposite side of a triangle -/
def altitude (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- The radius of the inscribed circle of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the sum of the altitudes of a triangle is equal to nine times 
the radius of its inscribed circle, then the triangle is equilateral 
-/
theorem altitudes_sum_eq_nine_inradius_implies_equilateral (t : Triangle) :
  altitude t t.A + altitude t t.B + altitude t t.C = 9 * inradius t →
  is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_altitudes_sum_eq_nine_inradius_implies_equilateral_l3718_371847


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3718_371866

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3718_371866


namespace NUMINAMATH_CALUDE_projection_vector_l3718_371823

/-- Given a vector b and the dot product of vectors a and b, 
    prove that the projection of a onto b is as calculated. -/
theorem projection_vector (a b : ℝ × ℝ) (h : a • b = 10) 
    (hb : b = (3, 4)) : 
  (a • b / (b • b)) • b = (6/5, 8/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l3718_371823


namespace NUMINAMATH_CALUDE_main_theorem_l3718_371895

-- Define the propositions p and q
def p (m a : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

-- Define the condition for m to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

-- Main theorem
theorem main_theorem (m a : ℝ) 
  (h1 : m^2 - 4*a*m + 3*a^2 ≤ 0)
  (h2 : is_hyperbola m) :
  (a = -1 ∧ (p m a ∨ q m) → -3 ≤ m ∧ m ≤ -1) ∧
  (∀ m, p m a → ¬q m) ∧ (∃ m, p m a ∧ q m) →
  (-1/3 ≤ a ∧ a < 0) ∨ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l3718_371895


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3718_371819

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l3718_371819


namespace NUMINAMATH_CALUDE_light_distance_250_years_l3718_371804

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 250

/-- The theorem stating the distance light travels in 250 years -/
theorem light_distance_250_years : 
  light_year_distance * years = 1.4675 * (10 : ℝ) ^ 15 := by
  sorry

end NUMINAMATH_CALUDE_light_distance_250_years_l3718_371804


namespace NUMINAMATH_CALUDE_new_cube_edge_theorem_l3718_371816

/-- The edge length of a cube formed by melting five cubes -/
def new_cube_edge (a b c d e : ℝ) : ℝ :=
  (a^3 + b^3 + c^3 + d^3 + e^3) ^ (1/3)

/-- Theorem stating that the edge of the new cube is the cube root of the sum of volumes -/
theorem new_cube_edge_theorem :
  new_cube_edge 6 8 10 12 14 = (6^3 + 8^3 + 10^3 + 12^3 + 14^3) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_new_cube_edge_theorem_l3718_371816


namespace NUMINAMATH_CALUDE_inverse_proportion_l3718_371838

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x - y = 10, then y = 50 when x = 4 -/
theorem inverse_proportion (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 4 → y = 50 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3718_371838


namespace NUMINAMATH_CALUDE_inequality_range_l3718_371893

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x - 1| + |x + 1| > a) → a < (3/2) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3718_371893


namespace NUMINAMATH_CALUDE_race_outcomes_l3718_371839

theorem race_outcomes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l3718_371839


namespace NUMINAMATH_CALUDE_dehydrated_men_fraction_l3718_371868

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (3 : ℚ) / 4 * total_men)
  (h4 : (2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men) = total_men - finished_men - ((1 : ℚ) / 4 * total_men)) :
  (total_men - finished_men - (1 : ℚ) / 4 * total_men) / ((2 : ℚ) / 3 * ((3 : ℚ) / 4 * total_men)) = (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_dehydrated_men_fraction_l3718_371868


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3718_371809

/-- A line defined by the equation (m-1)x-y+2m+1=0 for any real number m -/
def line (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point (-2, 3) -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line passes through the fixed point for any real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3718_371809


namespace NUMINAMATH_CALUDE_total_profit_calculation_l3718_371820

-- Define the investments and c's profit share
def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def c_profit_share : ℕ := 3000

-- Theorem statement
theorem total_profit_calculation :
  let total_investment := investment_a + investment_b + investment_c
  let profit_ratio_c := investment_c / total_investment
  let total_profit := c_profit_share / profit_ratio_c
  total_profit = 5000 := by sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l3718_371820


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3718_371861

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) →
  A + B = 46 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3718_371861


namespace NUMINAMATH_CALUDE_smallest_perimeter_cross_section_area_is_sqrt_6_l3718_371805

/-- Represents a quadrilateral pyramid with a square base -/
structure QuadPyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  pyramid : QuadPyramid
  point_on_base : ℝ  -- Distance from A to the point on AB

/-- The area of the cross-section with the smallest perimeter -/
def smallest_perimeter_cross_section_area (p : QuadPyramid) : ℝ := sorry

/-- The theorem stating that the area of the smallest perimeter cross-section is √6 -/
theorem smallest_perimeter_cross_section_area_is_sqrt_6 
  (p : QuadPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 2) : 
  smallest_perimeter_cross_section_area p = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_cross_section_area_is_sqrt_6_l3718_371805


namespace NUMINAMATH_CALUDE_smallest_number_l3718_371803

theorem smallest_number (a b c d : ℤ) (ha : a = -1) (hb : b = -2) (hc : c = 1) (hd : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3718_371803


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3718_371870

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 4) = (x^2 + 2) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3718_371870


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3718_371840

/-- Simple interest calculation -/
theorem simple_interest_calculation (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 4016.25 →
  rate = 1 / 100 →
  time = 3 →
  ∃ principal : ℚ, principal = 133875 ∧ interest = principal * rate * time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3718_371840


namespace NUMINAMATH_CALUDE_smallest_divisor_of_repeated_three_digit_number_l3718_371891

theorem smallest_divisor_of_repeated_three_digit_number : ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  let abc := 100 * a + 10 * b + c
  let abcabcabc := 1000000 * abc + 1000 * abc + abc
  (101 ∣ abcabcabc) ∧ ∀ d : ℕ, 1 < d → d < 101 → ¬(d ∣ abcabcabc) :=
by sorry

#check smallest_divisor_of_repeated_three_digit_number

end NUMINAMATH_CALUDE_smallest_divisor_of_repeated_three_digit_number_l3718_371891


namespace NUMINAMATH_CALUDE_log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l3718_371842

-- Define the base for logarithms
variable (a : ℝ)

-- Define the conditions for the base
variable (ha : a > 0 ∧ a ≠ 1)

-- Theorem 1: With only integer exponents, logarithms exist only for powers of the base
theorem log_exists_iff_power_of_base (b : ℝ) :
  (∃ n : ℤ, b = a^n) ↔ ∃ x : ℝ, a^x = b :=
sorry

-- Theorem 2: With only positive exponents, logarithms don't exist for numbers between 0 and 1
theorem no_log_for_numbers_between_zero_and_one (x : ℝ) (hx : 0 < x ∧ x < 1) :
  ¬∃ y : ℝ, y > 0 ∧ a^y = x :=
sorry

end NUMINAMATH_CALUDE_log_exists_iff_power_of_base_no_log_for_numbers_between_zero_and_one_l3718_371842


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l3718_371830

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

-- Define symmetry about the y-axis
def symmetricAboutYAxis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ, symmetricAboutYAxis (translate f 1) exp → f = λ x => exp (-x - 1) := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_and_translation_l3718_371830


namespace NUMINAMATH_CALUDE_walking_problem_solution_l3718_371835

/-- Two people walking on a line between points A and B -/
def WalkingProblem (distance_AB : ℝ) : Prop :=
  ∃ (first_meeting second_meeting : ℝ),
    first_meeting = 5 ∧
    second_meeting = distance_AB - 4 ∧
    2 * distance_AB = first_meeting + second_meeting

theorem walking_problem_solution :
  WalkingProblem 11 := by sorry

end NUMINAMATH_CALUDE_walking_problem_solution_l3718_371835


namespace NUMINAMATH_CALUDE_measure_45_minutes_l3718_371832

/-- Represents a string that can be burned --/
structure BurnableString where
  burnTime : ℝ
  nonUniformRate : Bool

/-- Represents a lighter --/
structure Lighter

/-- Represents the state of burning strings --/
inductive BurningState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Function to measure time using burnable strings and a lighter --/
def measureTime (strings : List BurnableString) (lighter : Lighter) : ℝ :=
  sorry

/-- Theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes :
  ∃ (strings : List BurnableString) (lighter : Lighter),
    strings.length = 2 ∧
    (∀ s ∈ strings, s.burnTime = 1 ∧ s.nonUniformRate = true) ∧
    measureTime strings lighter = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_measure_45_minutes_l3718_371832


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3718_371813

/-- Given a rectangle and a triangle where one side of the rectangle is the base of the triangle
    and one vertex of the triangle is on the opposite side of the rectangle,
    the ratio of the area of the rectangle to the area of the triangle is 2:1 -/
theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ) (rectangle_area triangle_area : ℝ),
    L > 0 → W > 0 →
    rectangle_area = L * W →
    triangle_area = (1 / 2) * L * W →
    rectangle_area / triangle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3718_371813


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3718_371881

theorem quadratic_root_difference (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ x - y = Real.sqrt 77) →
  k ≤ Real.sqrt 109 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3718_371881


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l3718_371899

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_equals_two (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l3718_371899


namespace NUMINAMATH_CALUDE_quadratic_trinomials_sum_l3718_371875

/-- 
Given two quadratic trinomials that differ by exchanging the free term and the second coefficient,
if their sum has a unique root, then the sum evaluated at x = 1 is either 2 or 18.
-/
theorem quadratic_trinomials_sum (p q : ℝ) : 
  (∃! x, 2 * x^2 + (p + q) * x + (p + q) = 0) →
  (2 * 1^2 + (p + q) * 1 + (p + q) = 2 ∨ 2 * 1^2 + (p + q) * 1 + (p + q) = 18) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomials_sum_l3718_371875


namespace NUMINAMATH_CALUDE_point_coordinates_given_distance_and_xaxis_l3718_371885

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

/-- A point is on the x-axis if its y-coordinate is 0 -/
def isOnXAxis (p : Point) : Prop := p.y = 0

theorem point_coordinates_given_distance_and_xaxis (p : Point) 
  (h1 : isOnXAxis p) 
  (h2 : distanceFromYAxis p = 3) : 
  p = ⟨3, 0⟩ ∨ p = ⟨-3, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_given_distance_and_xaxis_l3718_371885


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l3718_371882

theorem angle_complement_supplement (x : Real) : 
  (90 - x = (1/3) * (180 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l3718_371882


namespace NUMINAMATH_CALUDE_sin_angle_DAE_sin_angle_DAE_value_l3718_371859

/-- An equilateral triangle with side length 9 -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Points D and E on side BC -/
structure PointsOnBC where
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Main theorem: sin ∠DAE in the given configuration -/
theorem sin_angle_DAE (triangle : EquilateralTriangle) (points : PointsOnBC) : ℝ :=
  sorry

/-- The value of sin ∠DAE is √3/2 -/
theorem sin_angle_DAE_value (triangle : EquilateralTriangle) (points : PointsOnBC) :
    sin_angle_DAE triangle points = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_angle_DAE_sin_angle_DAE_value_l3718_371859


namespace NUMINAMATH_CALUDE_tire_sale_price_l3718_371850

/-- Calculates the sale price of a tire given the number of tires, total savings, and original price. -/
def sale_price (num_tires : ℕ) (total_savings : ℚ) (original_price : ℚ) : ℚ :=
  original_price - (total_savings / num_tires)

/-- Theorem stating that the sale price of each tire is $75 given the problem conditions. -/
theorem tire_sale_price :
  let num_tires : ℕ := 4
  let total_savings : ℚ := 36
  let original_price : ℚ := 84
  sale_price num_tires total_savings original_price = 75 := by
sorry

end NUMINAMATH_CALUDE_tire_sale_price_l3718_371850


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3718_371826

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h_prop : InverselyProportional x₁ y₁)
  (h_init : x₁ = 40 ∧ y₁ = 8)
  (h_final : y₂ = 10) :
  x₂ = 32 ∧ InverselyProportional x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3718_371826


namespace NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l3718_371843

theorem triangle_rectangle_equal_area (s h : ℝ) (s_pos : 0 < s) :
  (1 / 2) * s * h = 2 * s^2 → h = 4 * s :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_equal_area_l3718_371843


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3718_371855

-- Define the function f(x) = ax³ + bx²
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  (a = -6 ∧ b = 9) ∧
  (∀ x : ℝ, f (-6) 9 x ≥ f (-6) 9 0) ∧
  f (-6) 9 0 = 0 := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_cubic_function_properties_l3718_371855


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3718_371807

def num_maple_trees : ℕ := 4
def num_oak_trees : ℕ := 5
def num_birch_trees : ℕ := 6

def total_trees : ℕ := num_maple_trees + num_oak_trees + num_birch_trees

def favorable_arrangements : ℕ := (Nat.choose 10 6) * (Nat.choose 9 4)
def total_arrangements : ℕ := Nat.factorial total_trees

theorem birch_tree_arrangement_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3718_371807


namespace NUMINAMATH_CALUDE_probability_product_216_l3718_371837

/-- A standard die has 6 faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a standard die. -/
def DieProbability (event : Finset ℕ) : ℚ :=
  event.card / StandardDie.card

/-- The product of three numbers obtained from rolling three standard dice. -/
def ThreeDiceProduct (a b c : ℕ) : ℕ := a * b * c

/-- The event of rolling three sixes on three standard dice. -/
def ThreeSixes : Finset (ℕ × ℕ × ℕ) :=
  {(6, 6, 6)}

theorem probability_product_216 :
  DieProbability (ThreeSixes.image (fun (a, b, c) => ThreeDiceProduct a b c)) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_product_216_l3718_371837


namespace NUMINAMATH_CALUDE_store_ordered_15_boxes_of_pencils_l3718_371818

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 80

/-- The cost of each pencil in dollars -/
def pencil_cost : ℕ := 4

/-- The cost of each pen in dollars -/
def pen_cost : ℕ := 5

/-- The additional number of pens ordered beyond twice the number of pencils -/
def additional_pens : ℕ := 300

/-- The total cost of the stationery order in dollars -/
def total_cost : ℕ := 18300

/-- Proves that the store ordered 15 boxes of pencils given the conditions -/
theorem store_ordered_15_boxes_of_pencils :
  ∃ (x : ℕ),
    x * pencils_per_box * pencil_cost +
    (2 * x * pencils_per_box + additional_pens) * pen_cost = total_cost ∧
    x = 15 := by
  sorry

end NUMINAMATH_CALUDE_store_ordered_15_boxes_of_pencils_l3718_371818


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3718_371876

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  (Complex.exp (Complex.I * θ) = (4 : ℝ) / 5 + (3 : ℝ) / 5 * Complex.I) →
  (Complex.exp (Complex.I * φ) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I) →
  Real.sin (θ + φ) = (33 : ℝ) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l3718_371876
