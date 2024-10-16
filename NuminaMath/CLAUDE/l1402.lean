import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygon_with_12_degree_exterior_angles_has_30_sides_l1402_140200

/-- A regular polygon with exterior angles measuring 12 degrees has 30 sides. -/
theorem regular_polygon_with_12_degree_exterior_angles_has_30_sides :
  ∀ n : ℕ, 
  n > 0 →
  (360 : ℝ) / n = 12 →
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_12_degree_exterior_angles_has_30_sides_l1402_140200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1402_140214

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_eq : a 2^2 + 2*(a 2)*(a 6) + a 6^2 - 4 = 0) :
  a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1402_140214


namespace NUMINAMATH_CALUDE_winnie_the_pooh_escalator_steps_l1402_140274

theorem winnie_the_pooh_escalator_steps :
  ∀ (u v L : ℝ),
    u > 0 →
    v > 0 →
    L > 0 →
    (L * u) / (u + v) = 55 →
    (L * u) / (u - v) = 1155 →
    L = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_escalator_steps_l1402_140274


namespace NUMINAMATH_CALUDE_tag_sum_is_1000_l1402_140203

/-- The sum of the numbers tagged on four cards W, X, Y, Z -/
def total_tag_sum (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating that the sum of the tagged numbers is 1000 -/
theorem tag_sum_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = w + x →
  z = 400 →
  total_tag_sum w x y z = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tag_sum_is_1000_l1402_140203


namespace NUMINAMATH_CALUDE_total_pages_left_to_read_l1402_140213

/-- Calculates the total number of pages left to read from three books -/
def pagesLeftToRead (book1Total book1Read book2Total book2Read book3Total book3Read : ℕ) : ℕ :=
  (book1Total - book1Read) + (book2Total - book2Read) + (book3Total - book3Read)

/-- Theorem: The total number of pages left to read from three books is 1442 -/
theorem total_pages_left_to_read : 
  pagesLeftToRead 563 147 849 389 700 134 = 1442 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_left_to_read_l1402_140213


namespace NUMINAMATH_CALUDE_remainder_theorem_l1402_140202

theorem remainder_theorem : (9 * 10^20 + 1^20) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1402_140202


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l1402_140281

theorem equation_implies_fraction_value (a x y : ℝ) :
  x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (Real.log (x - a) - Real.log (a - y)) →
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l1402_140281


namespace NUMINAMATH_CALUDE_sum_of_seventh_terms_l1402_140236

/-- First sequence defined by a_n = n^2 + n - 1 -/
def sequence_a (n : ℕ) : ℕ := n^2 + n - 1

/-- Second sequence defined by b_n = n(n+1)/2 -/
def sequence_b (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 7th terms of both sequences is 83 -/
theorem sum_of_seventh_terms :
  sequence_a 7 + sequence_b 7 = 83 := by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_terms_l1402_140236


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1402_140261

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1402_140261


namespace NUMINAMATH_CALUDE_sine_sum_equality_l1402_140217

theorem sine_sum_equality : 
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) + 
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l1402_140217


namespace NUMINAMATH_CALUDE_ellipse_properties_l1402_140297

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define eccentricity
def e : ℝ := 0.8

-- Define dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem ellipse_properties :
  ∃ (a b : ℝ),
    -- Standard equation of the ellipse
    (a = 5 ∧ b = 3) ∧
    -- Existence of point P
    ∃ (P : ℝ × ℝ),
      P ∈ Ellipse a b ∧
      dot_product (vector_to F₁ P) (vector_to F₂ P) = 0 ∧
      -- Coordinates of point P
      ((P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1402_140297


namespace NUMINAMATH_CALUDE_S_is_valid_set_l1402_140284

-- Define the set of non-negative integers not exceeding 10
def S : Set ℕ := {n : ℕ | n ≤ 10}

-- Theorem stating that S is a valid set
theorem S_is_valid_set :
  -- S has definite elements
  (∀ n : ℕ, n ∈ S ↔ n ≤ 10) ∧
  -- S has disordered elements (always true for sets)
  True ∧
  -- S has distinct elements (follows from the definition of ℕ)
  (∀ a b : ℕ, a ∈ S → b ∈ S → a = b → a = b) :=
sorry

end NUMINAMATH_CALUDE_S_is_valid_set_l1402_140284


namespace NUMINAMATH_CALUDE_equation_solution_l1402_140234

theorem equation_solution : 
  let x : ℝ := 405 / 8
  (2 * x - 60) / 3 = (2 * x - 5) / 7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1402_140234


namespace NUMINAMATH_CALUDE_festival_attendance_l1402_140237

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900) : ℕ :=
by
  let girls : ℕ := sorry
  let boys : ℕ := sorry
  have h3 : girls + boys = total_students := sorry
  have h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = festival_attendees := sorry
  have h5 : (3 * girls / 4 : ℕ) = 900 := sorry
  exact 900

#check festival_attendance

end NUMINAMATH_CALUDE_festival_attendance_l1402_140237


namespace NUMINAMATH_CALUDE_cost_price_proof_l1402_140276

/-- The cost price of an article satisfying the given profit and loss conditions. -/
def cost_price : ℝ := 49

/-- The selling price that results in a profit. -/
def profit_price : ℝ := 56

/-- The selling price that results in a loss. -/
def loss_price : ℝ := 42

theorem cost_price_proof :
  (profit_price - cost_price = cost_price - loss_price) →
  cost_price = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_proof_l1402_140276


namespace NUMINAMATH_CALUDE_max_value_ratio_l1402_140290

theorem max_value_ratio (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3) :
  (a + c) / (b + d) ≤ (7 + 2 * Real.sqrt 6) / 5 ∧ 
  ∃ a b c d, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0 ∧
    a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3 ∧
    (a + c) / (b + d) = (7 + 2 * Real.sqrt 6) / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_ratio_l1402_140290


namespace NUMINAMATH_CALUDE_probability_of_being_leader_l1402_140263

theorem probability_of_being_leader (total_people : ℕ) (num_groups : ℕ) 
  (h1 : total_people = 12) 
  (h2 : num_groups = 2) 
  (h3 : total_people % num_groups = 0) : 
  (1 : ℚ) / (total_people / num_groups) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_of_being_leader_l1402_140263


namespace NUMINAMATH_CALUDE_only_14_satisfies_l1402_140273

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end NUMINAMATH_CALUDE_only_14_satisfies_l1402_140273


namespace NUMINAMATH_CALUDE_problem_statement_l1402_140233

theorem problem_statement (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * (Real.log x / Real.log y) + 2 * (Real.log y / Real.log x) = 8) 
  (h4 : x * y = 256) : (x + y) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1402_140233


namespace NUMINAMATH_CALUDE_tangent_implies_one_point_one_point_not_always_tangent_l1402_140246

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a parabola in 2D space
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line being tangent to a parabola
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the property of a line and a parabola having exactly one common point
def has_one_common_point (l : Line) (p : Parabola) : Prop := sorry

-- Theorem stating the relationship between tangency and having one common point
theorem tangent_implies_one_point (l : Line) (p : Parabola) :
  is_tangent l p → has_one_common_point l p :=
sorry

-- Theorem stating that having one common point doesn't always imply tangency
theorem one_point_not_always_tangent :
  ∃ l : Line, ∃ p : Parabola, has_one_common_point l p ∧ ¬is_tangent l p :=
sorry

end NUMINAMATH_CALUDE_tangent_implies_one_point_one_point_not_always_tangent_l1402_140246


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_zero_l1402_140278

/-- The system of equations has no solution if and only if n = 0 -/
theorem no_solution_iff_n_eq_zero (n : ℝ) :
  (∃ (x y z : ℝ), 2*n*x + y = 2 ∧ 3*n*y + z = 3 ∧ x + 2*n*z = 2) ↔ n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_zero_l1402_140278


namespace NUMINAMATH_CALUDE_adults_attending_concert_concert_attendance_proof_l1402_140272

/-- The number of adults attending a music festival concert, given ticket prices and total revenue --/
theorem adults_attending_concert (adult_price : ℕ) (child_price : ℕ) (num_children : ℕ) (total_revenue : ℕ) : ℕ :=
  let adults : ℕ := (total_revenue - num_children * child_price) / adult_price
  adults

/-- Proof that 183 adults attended the concert given the specific conditions --/
theorem concert_attendance_proof :
  adults_attending_concert 26 13 28 5122 = 183 := by
  sorry

end NUMINAMATH_CALUDE_adults_attending_concert_concert_attendance_proof_l1402_140272


namespace NUMINAMATH_CALUDE_expression_evaluation_l1402_140299

theorem expression_evaluation : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1402_140299


namespace NUMINAMATH_CALUDE_household_size_proof_l1402_140238

/-- The number of slices of bread consumed by each member daily. -/
def daily_consumption : ℕ := 5

/-- The number of slices in a loaf of bread. -/
def slices_per_loaf : ℕ := 12

/-- The number of loaves that last for 3 days. -/
def loaves_for_three_days : ℕ := 5

/-- The number of days the loaves last. -/
def days : ℕ := 3

/-- The number of members in the household. -/
def household_members : ℕ := 4

theorem household_size_proof :
  household_members * daily_consumption * days = loaves_for_three_days * slices_per_loaf :=
by sorry

end NUMINAMATH_CALUDE_household_size_proof_l1402_140238


namespace NUMINAMATH_CALUDE_orthic_triangle_right_angled_iff_45_or_135_angle_l1402_140275

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the orthic triangle of a given triangle
def orthicTriangle (t : Triangle) : Triangle := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

-- Define the condition for an angle to be 45° or 135°
def has45or135Angle (t : Triangle) : Prop := sorry

-- Theorem statement
theorem orthic_triangle_right_angled_iff_45_or_135_angle (t : Triangle) :
  has45or135Angle t ↔ isRightAngled (orthicTriangle t) := by sorry

end NUMINAMATH_CALUDE_orthic_triangle_right_angled_iff_45_or_135_angle_l1402_140275


namespace NUMINAMATH_CALUDE_persimmons_in_box_l1402_140207

/-- Given a box containing apples and persimmons, prove the number of persimmons. -/
theorem persimmons_in_box (apples : ℕ) (persimmons : ℕ) : apples = 3 → persimmons = 2 → persimmons = 2 := by
  sorry

end NUMINAMATH_CALUDE_persimmons_in_box_l1402_140207


namespace NUMINAMATH_CALUDE_factory_works_ten_hours_per_day_l1402_140247

/-- Represents a chocolate factory with its production parameters -/
structure ChocolateFactory where
  production_rate : ℕ  -- candies per hour
  order_size : ℕ       -- total candies to produce
  days_to_complete : ℕ -- number of days to complete the order

/-- Calculates the number of hours the factory works each day -/
def hours_per_day (factory : ChocolateFactory) : ℚ :=
  (factory.order_size / factory.production_rate : ℚ) / factory.days_to_complete

/-- Theorem stating that for the given parameters, the factory works 10 hours per day -/
theorem factory_works_ten_hours_per_day :
  let factory := ChocolateFactory.mk 50 4000 8
  hours_per_day factory = 10 := by
  sorry

end NUMINAMATH_CALUDE_factory_works_ten_hours_per_day_l1402_140247


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1402_140257

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) (h : f' a 1 = 3) :
  ∃ (m b : ℝ), m * 1 - b = f a 1 ∧ 
                ∀ x, m * x - b = 3 * x - 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1402_140257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1402_140211

theorem arithmetic_sequence_count (a l d : ℤ) (h1 : a = -58) (h2 : l = 78) (h3 : d = 7) :
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1) * d ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1402_140211


namespace NUMINAMATH_CALUDE_exactly_one_multiple_of_five_l1402_140295

theorem exactly_one_multiple_of_five (a b : ℤ) (h : 24 * a^2 + 1 = b^2) :
  (a % 5 = 0 ∧ b % 5 ≠ 0) ∨ (a % 5 ≠ 0 ∧ b % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_multiple_of_five_l1402_140295


namespace NUMINAMATH_CALUDE_complex_number_value_l1402_140285

theorem complex_number_value (z : ℂ) (h : (z - 1) * Complex.I = Complex.abs (Complex.I + 1)) :
  z = 1 - Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l1402_140285


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l1402_140241

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l1402_140241


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l1402_140209

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k := by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l1402_140209


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_and_evaluation_l1402_140265

theorem algebraic_expression_simplification_and_evaluation :
  let x : ℝ := 4 * Real.sin (45 * π / 180) - 2
  let original_expression := (1 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) - x / (x + 2)
  let simplified_expression := -1 / (x + 2)
  original_expression = simplified_expression ∧ simplified_expression = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_and_evaluation_l1402_140265


namespace NUMINAMATH_CALUDE_high_scam_probability_l1402_140253

/-- Represents an email message -/
structure Email :=
  (claims_prize : Bool)
  (asks_for_phone : Bool)
  (requests_payment : Bool)
  (payment_amount : ℕ)

/-- Represents the probability of an email being a scam -/
def scam_probability (e : Email) : ℝ := sorry

/-- Theorem: Given an email with specific characteristics, the probability of it being a scam is high -/
theorem high_scam_probability (e : Email) 
  (h1 : e.claims_prize = true)
  (h2 : e.asks_for_phone = true)
  (h3 : e.requests_payment = true)
  (h4 : e.payment_amount = 150) :
  scam_probability e > 0.9 := by sorry

end NUMINAMATH_CALUDE_high_scam_probability_l1402_140253


namespace NUMINAMATH_CALUDE_dice_line_probability_l1402_140216

-- Define the dice outcomes
def DiceOutcome : Type := Fin 6

-- Define the probability space
def Ω : Type := DiceOutcome × DiceOutcome

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event that (x, y) lies on the line 2x - y = 1
def E : Set Ω :=
  {ω : Ω | 2 * (ω.1.val + 1) - (ω.2.val + 1) = 1}

-- Theorem statement
theorem dice_line_probability :
  P E = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_line_probability_l1402_140216


namespace NUMINAMATH_CALUDE_exists_prime_not_dividing_euclid_l1402_140267

/-- Definition of Euclid numbers -/
def euclid : ℕ → ℕ
  | 0 => 3
  | n + 1 => euclid n * euclid (n - 1) + 1

/-- Theorem: There exists a prime that does not divide any Euclid number -/
theorem exists_prime_not_dividing_euclid : ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬(p ∣ euclid n) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_not_dividing_euclid_l1402_140267


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l1402_140227

/-- Calculates the profit percentage for a retailer selling a machine --/
theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_rate = 0.1)
  : (((retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price) * 100 = 20) := by
  sorry

#check retailer_profit_percentage

end NUMINAMATH_CALUDE_retailer_profit_percentage_l1402_140227


namespace NUMINAMATH_CALUDE_max_value_of_function_l1402_140294

open Real

theorem max_value_of_function (x : ℝ) : 
  ∃ (M : ℝ), M = 2 - sqrt 3 ∧ ∀ y : ℝ, sin (2 * y) - 2 * sqrt 3 * sin y ^ 2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1402_140294


namespace NUMINAMATH_CALUDE_vowel_initial_probability_l1402_140212

/-- The probability of selecting a student with vowel initials -/
theorem vowel_initial_probability 
  (total_students : ℕ) 
  (vowels : List Char) 
  (students_per_vowel : ℕ) : 
  total_students = 34 → 
  vowels = ['A', 'E', 'I', 'O', 'U', 'Y'] → 
  students_per_vowel = 2 → 
  (students_per_vowel * vowels.length : ℚ) / total_students = 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_vowel_initial_probability_l1402_140212


namespace NUMINAMATH_CALUDE_geometric_sequence_example_l1402_140210

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_example :
  is_geometric_sequence 3 (-3 * Real.sqrt 3) 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_example_l1402_140210


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_28_l1402_140269

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first small triangle -/
  area1 : ℝ
  /-- Area of the second small triangle -/
  area2 : ℝ
  /-- Area of the third small triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 4, 8, and 8,
    then the area of the quadrilateral is 28 -/
theorem quadrilateral_area_is_28 (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 8) : 
  t.areaQuad = 28 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_is_28_l1402_140269


namespace NUMINAMATH_CALUDE_square_sum_fifteen_l1402_140254

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fifteen_l1402_140254


namespace NUMINAMATH_CALUDE_smallest_multiple_with_factors_l1402_140252

theorem smallest_multiple_with_factors : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), (1452 * m : ℕ) % 2^4 = 0 ∧ 
                (1452 * m : ℕ) % 3^3 = 0 ∧ 
                (1452 * m : ℕ) % 13^3 = 0 → n ≤ m) ∧
  (1452 * n : ℕ) % 2^4 = 0 ∧ 
  (1452 * n : ℕ) % 3^3 = 0 ∧ 
  (1452 * n : ℕ) % 13^3 = 0 ∧
  n = 79092 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_factors_l1402_140252


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l1402_140226

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℝ := 18

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℝ := 15

/-- Represents the number of kayaks rented -/
def num_kayaks : ℕ := 10

/-- Represents the number of canoes rented -/
def num_canoes : ℕ := 15

/-- Represents the total revenue for one day -/
def total_revenue : ℝ := 405

theorem kayak_rental_cost :
  (kayak_cost * num_kayaks + canoe_cost * num_canoes = total_revenue) ∧
  (num_canoes = num_kayaks + 5) ∧
  (3 * num_kayaks = 2 * num_canoes) :=
by sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l1402_140226


namespace NUMINAMATH_CALUDE_defective_product_probability_l1402_140293

theorem defective_product_probability
  (p_first : ℝ)
  (p_second : ℝ)
  (h1 : p_first = 0.65)
  (h2 : p_second = 0.3)
  (h3 : p_first + p_second + p_defective = 1)
  (h4 : p_first ≥ 0 ∧ p_second ≥ 0 ∧ p_defective ≥ 0)
  : p_defective = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_defective_product_probability_l1402_140293


namespace NUMINAMATH_CALUDE_quadrilateral_formation_count_l1402_140286

theorem quadrilateral_formation_count :
  let rod_lengths : Finset ℕ := Finset.range 25
  let chosen_rods : Finset ℕ := {4, 9, 12}
  let remaining_rods := rod_lengths \ chosen_rods
  (remaining_rods.filter (fun d => 
    d + 4 + 9 > 12 ∧ d + 4 + 12 > 9 ∧ d + 9 + 12 > 4 ∧ 4 + 9 + 12 > d
  )).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_formation_count_l1402_140286


namespace NUMINAMATH_CALUDE_science_score_calculation_l1402_140270

def average_score : ℝ := 95
def chinese_score : ℝ := 90
def math_score : ℝ := 98

theorem science_score_calculation :
  ∃ (science_score : ℝ),
    (chinese_score + math_score + science_score) / 3 = average_score ∧
    science_score = 97 := by sorry

end NUMINAMATH_CALUDE_science_score_calculation_l1402_140270


namespace NUMINAMATH_CALUDE_fraction_chain_l1402_140205

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 3)
  (h3 : c / d = 5) :
  d / a = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_chain_l1402_140205


namespace NUMINAMATH_CALUDE_triangle_inequality_l1402_140249

theorem triangle_inequality (m : ℝ) : m > 0 → (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) → m = 5 := by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l1402_140249


namespace NUMINAMATH_CALUDE_bearded_male_percentage_is_40_percent_l1402_140201

/-- Represents the data for Scrabble champions over a period of years -/
structure ScrabbleChampionData where
  total_years : ℕ
  women_percentage : ℚ
  champions_per_year : ℕ
  bearded_men : ℕ

/-- Calculates the percentage of male Scrabble champions with beards -/
def bearded_male_percentage (data : ScrabbleChampionData) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the percentage of male Scrabble champions with beards is 40% -/
theorem bearded_male_percentage_is_40_percent 
  (data : ScrabbleChampionData)
  (h1 : data.total_years = 25)
  (h2 : data.women_percentage = 60 / 100)
  (h3 : data.champions_per_year = 1)
  (h4 : data.bearded_men = 4) :
  bearded_male_percentage data = 40 / 100 :=
sorry

end NUMINAMATH_CALUDE_bearded_male_percentage_is_40_percent_l1402_140201


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l1402_140243

theorem probability_two_red_shoes :
  let total_shoes : ℕ := 9
  let red_shoes : ℕ := 5
  let green_shoes : ℕ := 4
  let draw_count : ℕ := 2
  
  total_shoes = red_shoes + green_shoes →
  (Nat.choose red_shoes draw_count : ℚ) / (Nat.choose total_shoes draw_count : ℚ) = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l1402_140243


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1402_140296

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : RealFunction) (h_even : IsEven g) :
  IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1402_140296


namespace NUMINAMATH_CALUDE_sum_of_max_marks_is_1300_l1402_140239

/-- Given the conditions for three tests (Math, Science, and English) in an examination,
    this theorem proves that the sum of maximum marks for all three tests is 1300. -/
theorem sum_of_max_marks_is_1300 
  (math_pass_percent : ℝ) 
  (math_marks_obtained : ℕ) 
  (math_marks_failed_by : ℕ)
  (science_pass_percent : ℝ) 
  (science_marks_obtained : ℕ) 
  (science_marks_failed_by : ℕ)
  (english_pass_percent : ℝ) 
  (english_marks_obtained : ℕ) 
  (english_marks_failed_by : ℕ)
  (h_math_percent : math_pass_percent = 0.3)
  (h_science_percent : science_pass_percent = 0.5)
  (h_english_percent : english_pass_percent = 0.4)
  (h_math_marks : math_marks_obtained = 80 ∧ math_marks_failed_by = 100)
  (h_science_marks : science_marks_obtained = 120 ∧ science_marks_failed_by = 80)
  (h_english_marks : english_marks_obtained = 60 ∧ english_marks_failed_by = 60) :
  ↑((math_marks_obtained + math_marks_failed_by) / math_pass_percent +
    (science_marks_obtained + science_marks_failed_by) / science_pass_percent +
    (english_marks_obtained + english_marks_failed_by) / english_pass_percent) = 1300 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_max_marks_is_1300_l1402_140239


namespace NUMINAMATH_CALUDE_counterexample_exists_l1402_140292

theorem counterexample_exists (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b : ℝ, c * b^2 ≥ a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1402_140292


namespace NUMINAMATH_CALUDE_goods_selection_counts_l1402_140289

def total_goods : ℕ := 35
def counterfeit_goods : ℕ := 15
def genuine_goods : ℕ := total_goods - counterfeit_goods
def selected_goods : ℕ := 3

theorem goods_selection_counts :
  (Nat.choose (total_goods - 1) (selected_goods - 1) = 561) ∧
  (Nat.choose (total_goods - 1) selected_goods = 5984) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose genuine_goods 1 = 2100) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose genuine_goods 1 + Nat.choose counterfeit_goods 3 = 2555) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose genuine_goods 1 + 
   Nat.choose counterfeit_goods 1 * Nat.choose genuine_goods 2 + 
   Nat.choose genuine_goods 3 = 6090) := by
  sorry

end NUMINAMATH_CALUDE_goods_selection_counts_l1402_140289


namespace NUMINAMATH_CALUDE_binomial_coefficient_15_4_l1402_140221

theorem binomial_coefficient_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_15_4_l1402_140221


namespace NUMINAMATH_CALUDE_range_of_a_l1402_140230

open Set

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = ∅ ↔ 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1402_140230


namespace NUMINAMATH_CALUDE_min_boxes_for_muffins_l1402_140235

/-- Represents the number of muffins that can be packed in each box type -/
structure BoxCapacity where
  large : Nat
  medium : Nat
  small : Nat

/-- Represents the number of boxes used for each type -/
structure BoxCount where
  large : Nat
  medium : Nat
  small : Nat

def total_muffins : Nat := 250

def box_capacity : BoxCapacity := ⟨12, 8, 4⟩

def box_count : BoxCount := ⟨20, 1, 1⟩

/-- Calculates the total number of muffins that can be packed in the given boxes -/
def muffins_packed (capacity : BoxCapacity) (count : BoxCount) : Nat :=
  capacity.large * count.large + capacity.medium * count.medium + capacity.small * count.small

/-- Calculates the total number of boxes used -/
def total_boxes (count : BoxCount) : Nat :=
  count.large + count.medium + count.small

theorem min_boxes_for_muffins :
  muffins_packed box_capacity box_count = total_muffins ∧
  total_boxes box_count = 22 ∧
  ∀ (other_count : BoxCount),
    muffins_packed box_capacity other_count ≥ total_muffins →
    total_boxes other_count ≥ total_boxes box_count :=
by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_muffins_l1402_140235


namespace NUMINAMATH_CALUDE_lee_cookies_proportion_l1402_140223

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 40 cookies with 5 cups of flour, 
    assuming a proportional relationship between flour and cookies. -/
theorem lee_cookies_proportion (flour_cups : ℚ) (cookies : ℕ) 
  (h1 : flour_cups > 0)
  (h2 : cookies > 0)
  (h3 : flour_cups / 3 = cookies / 24) :
  5 * cookies / flour_cups = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_proportion_l1402_140223


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1402_140248

theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ (y : ℝ), (y^2 + 6*y + 9 = y^2 + 25) → y = x) ∧
    x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l1402_140248


namespace NUMINAMATH_CALUDE_largest_difference_even_digits_l1402_140259

/-- A function that checks if a natural number has all even digits -/
def allEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → Even d

/-- A function that checks if a natural number has at least one odd digit -/
def hasOddDigit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ Odd d

/-- The theorem stating the largest possible difference between two 6-digit numbers
    with all even digits, where any number between them has at least one odd digit -/
theorem largest_difference_even_digits :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    allEvenDigits a ∧
    allEvenDigits b ∧
    (∀ n, a < n ∧ n < b → hasOddDigit n) ∧
    b - a = 111112 ∧
    (∀ a' b', (100000 ≤ a' ∧ a' < 1000000) →
              (100000 ≤ b' ∧ b' < 1000000) →
              allEvenDigits a' →
              allEvenDigits b' →
              (∀ n, a' < n ∧ n < b' → hasOddDigit n) →
              b' - a' ≤ 111112) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_even_digits_l1402_140259


namespace NUMINAMATH_CALUDE_josh_remaining_money_l1402_140283

/-- Calculates the remaining money after spending two amounts -/
def remaining_money (initial : ℚ) (spent1 : ℚ) (spent2 : ℚ) : ℚ :=
  initial - (spent1 + spent2)

/-- Theorem: Given Josh's initial $9 and his spending of $1.75 and $1.25, he has $6 left -/
theorem josh_remaining_money :
  remaining_money 9 (175/100) (125/100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l1402_140283


namespace NUMINAMATH_CALUDE_recycle_388_cans_l1402_140206

/-- Recursively calculate the number of new cans produced from recycling -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 6 then 0
  else
    let new_cans := (2 * initial_cans) / 6
    new_cans + recycle_cans new_cans

/-- The total number of new cans produced from 388 initial cans -/
def total_new_cans : ℕ := recycle_cans 388

/-- Theorem stating that 193 new cans are produced from 388 initial cans -/
theorem recycle_388_cans : total_new_cans = 193 := by
  sorry

end NUMINAMATH_CALUDE_recycle_388_cans_l1402_140206


namespace NUMINAMATH_CALUDE_arithmetic_proof_l1402_140222

theorem arithmetic_proof : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l1402_140222


namespace NUMINAMATH_CALUDE_car_speed_theorem_l1402_140219

def car_speed_problem (first_hour_speed average_speed : ℝ) : Prop :=
  let total_time : ℝ := 2
  let second_hour_speed : ℝ := 2 * average_speed - first_hour_speed
  second_hour_speed = 50

theorem car_speed_theorem :
  car_speed_problem 90 70 := by sorry

end NUMINAMATH_CALUDE_car_speed_theorem_l1402_140219


namespace NUMINAMATH_CALUDE_proposition_equivalence_l1402_140204

theorem proposition_equivalence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l1402_140204


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1402_140231

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1402_140231


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1402_140245

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = x}
def N : Set ℝ := {x : ℝ | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1402_140245


namespace NUMINAMATH_CALUDE_difference_of_squares_l1402_140220

theorem difference_of_squares (a : ℝ) : a^2 - 81 = (a+9)*(a-9) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1402_140220


namespace NUMINAMATH_CALUDE_inverse_function_difference_l1402_140262

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property that f and f_inv are inverse functions
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the property that f(x+2) and f^(-1)(x-1) are inverse functions
def special_inverse_property (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv (x - 1) + 2) = x ∧ f_inv (f (x + 2) - 1) = x

-- Theorem statement
theorem inverse_function_difference
  (h1 : is_inverse f f_inv)
  (h2 : special_inverse_property f f_inv) :
  f_inv 2004 - f_inv 1 = 4006 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_difference_l1402_140262


namespace NUMINAMATH_CALUDE_students_taking_both_languages_l1402_140260

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ)
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : neither = 6)
  : (french + german) - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_languages_l1402_140260


namespace NUMINAMATH_CALUDE_fraction_equality_l1402_140250

theorem fraction_equality : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5000 = 750.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1402_140250


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1402_140225

theorem complex_sum_equality (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1402_140225


namespace NUMINAMATH_CALUDE_kims_weekly_production_l1402_140251

/-- Represents Kim's daily sweater production for a week --/
structure WeeklyKnitting where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of sweaters knit in a week --/
def totalSweaters (week : WeeklyKnitting) : ℕ :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Theorem stating that Kim's total sweater production for the given week is 34 --/
theorem kims_weekly_production :
  ∃ (week : WeeklyKnitting),
    week.monday = 8 ∧
    week.tuesday = week.monday + 2 ∧
    week.wednesday = week.tuesday - 4 ∧
    week.thursday = week.tuesday - 4 ∧
    week.friday = week.monday / 2 ∧
    totalSweaters week = 34 := by
  sorry


end NUMINAMATH_CALUDE_kims_weekly_production_l1402_140251


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l1402_140224

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
by sorry

theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l1402_140224


namespace NUMINAMATH_CALUDE_sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l1402_140282

/-- The angle α with vertex at the origin and initial side on positive x-axis -/
structure Angle (α : ℝ) : Prop where
  vertex_origin : True
  initial_side_positive_x : True

/-- Point P on the terminal side of angle α -/
structure TerminalPoint (α : ℝ) (x y : ℝ) : Prop where
  on_terminal_side : True

/-- The terminal side of angle α lies on the line y = mx -/
structure TerminalLine (α : ℝ) (m : ℝ) : Prop where
  on_line : True

theorem sin_cos_product (α : ℝ) (h : Angle α) (p : TerminalPoint α (-1) 2) :
  Real.sin α * Real.cos α = -2/5 := by sorry

theorem tan_plus_sec_second_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : 0 < α ∧ α < π) :
  Real.tan α + 3 / Real.cos α = -3 - 3 * Real.sqrt 10 := by sorry

theorem tan_plus_sec_fourth_quadrant (α : ℝ) (h : Angle α) (l : TerminalLine α (-3)) 
  (q : -π/2 < α ∧ α < 0) :
  Real.tan α + 3 / Real.cos α = -3 + 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_sin_cos_product_tan_plus_sec_second_quadrant_tan_plus_sec_fourth_quadrant_l1402_140282


namespace NUMINAMATH_CALUDE_geometric_sequences_exist_and_unique_l1402_140271

/-- Three geometric sequences satisfying the given conditions -/
def geometric_sequences (a q : ℝ) : Fin 3 → ℕ → ℝ
| ⟨0, _⟩ => λ n => a * (q - 2) ^ n
| ⟨1, _⟩ => λ n => 2 * a * (q - 1) ^ n
| ⟨2, _⟩ => λ n => 4 * a * q ^ n

/-- The theorem stating the existence and uniqueness of the geometric sequences -/
theorem geometric_sequences_exist_and_unique :
  ∃ (a q : ℝ),
    (∀ i : Fin 3, geometric_sequences a q i 0 = a * (2 ^ i.val)) ∧
    (geometric_sequences a q 1 1 - geometric_sequences a q 0 1 =
     geometric_sequences a q 2 1 - geometric_sequences a q 1 1) ∧
    (geometric_sequences a q 0 1 + geometric_sequences a q 1 1 + geometric_sequences a q 2 1 = 24) ∧
    (geometric_sequences a q 0 0 + geometric_sequences a q 1 0 + geometric_sequences a q 2 0 = 84) ∧
    ((a = 1 ∧ q = 4) ∨ (a = 192 / 31 ∧ q = 9 / 8)) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequences_exist_and_unique_l1402_140271


namespace NUMINAMATH_CALUDE_jackson_percentage_difference_l1402_140240

/-- Represents the count of birds seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  blueJays : ℕ
  goldfinches : ℕ
  starlings : ℕ

/-- Calculates the total number of birds seen by a person -/
def totalBirds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.blueJays + count.goldfinches + count.starlings

/-- Calculates the percentage difference from the average -/
def percentageDifference (individual : ℕ) (average : ℚ) : ℚ :=
  ((individual : ℚ) - average) / average * 100

theorem jackson_percentage_difference :
  let gabrielle := BirdCount.mk 7 5 4 3 6
  let chase := BirdCount.mk 4 3 4 2 1
  let maria := BirdCount.mk 5 3 2 4 7
  let jackson := BirdCount.mk 6 2 3 5 2
  let total := totalBirds gabrielle + totalBirds chase + totalBirds maria + totalBirds jackson
  let average : ℚ := (total : ℚ) / 4
  abs (percentageDifference (totalBirds jackson) average - (-7.69)) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jackson_percentage_difference_l1402_140240


namespace NUMINAMATH_CALUDE_circle_symmetry_l1402_140264

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric point
def symmetric_point (x y x' y' : ℝ) : Prop := x' = -y ∧ y' = -x

-- Theorem statement
theorem circle_symmetry (x y : ℝ) :
  (∃ (x' y' : ℝ), symmetric_point x y x' y' ∧ 
   symmetry_line x y ∧ 
   original_circle x' y') →
  x^2 + (y + 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1402_140264


namespace NUMINAMATH_CALUDE_days_worked_l1402_140255

/-- Given a person works 8 hours each day and a total of 32 hours, prove that the number of days worked is 4. -/
theorem days_worked (hours_per_day : ℕ) (total_hours : ℕ) (h1 : hours_per_day = 8) (h2 : total_hours = 32) :
  total_hours / hours_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_l1402_140255


namespace NUMINAMATH_CALUDE_complex_sum_direction_l1402_140208

theorem complex_sum_direction (r : ℝ) (h : r > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (21 * Real.pi * Complex.I / 60) +
  Complex.exp (31 * Real.pi * Complex.I / 60) +
  Complex.exp (41 * Real.pi * Complex.I / 60) +
  Complex.exp (51 * Real.pi * Complex.I / 60) =
  r * Complex.exp (31 * Real.pi * Complex.I / 60) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_direction_l1402_140208


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1402_140215

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1 / x < 1 / 2) ∧
  (∃ x, 1 / x < 1 / 2 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1402_140215


namespace NUMINAMATH_CALUDE_total_turnips_count_l1402_140258

/-- The number of turnips grown by Melanie -/
def melanie_turnips : ℕ := 1395

/-- The number of turnips grown by Benny -/
def benny_turnips : ℕ := 11380

/-- The number of turnips grown by Jack -/
def jack_turnips : ℕ := 15825

/-- The number of turnips grown by Lynn -/
def lynn_turnips : ℕ := 23500

/-- The total number of turnips grown by all four people -/
def total_turnips : ℕ := melanie_turnips + benny_turnips + jack_turnips + lynn_turnips

theorem total_turnips_count : total_turnips = 52100 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_count_l1402_140258


namespace NUMINAMATH_CALUDE_literature_club_students_l1402_140287

theorem literature_club_students (total : ℕ) (english : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 120)
  (h_english : english = 72)
  (h_french : french = 52)
  (h_both : both = 12) :
  total - (english + french - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_literature_club_students_l1402_140287


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1402_140277

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1402_140277


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1402_140268

/-- Represents a systematic sample of size 4 from 52 employees -/
structure SystematicSample where
  size : Nat
  total : Nat
  elements : Fin 4 → Nat
  is_valid : size = 4 ∧ total = 52 ∧ ∀ i, elements i ≤ total

/-- Checks if a given sample is arithmetic -/
def is_arithmetic_sample (s : SystematicSample) : Prop :=
  ∃ d, ∀ i j, s.elements i - s.elements j = (i.val - j.val : ℤ) * d

/-- The main theorem -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.elements 0 = 6)
  (h2 : s.elements 2 = 32)
  (h3 : s.elements 3 = 45)
  (h4 : is_arithmetic_sample s) :
  s.elements 1 = 19 := by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l1402_140268


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1402_140256

theorem solution_set_inequality (x : ℝ) : (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1402_140256


namespace NUMINAMATH_CALUDE_tank_filling_time_l1402_140229

/-- Represents the time (in minutes) it takes for pipe A to fill the tank alone -/
def A : ℝ := 24

/-- Represents the time (in minutes) it takes for pipe B to fill the tank alone -/
def B : ℝ := 32

/-- Represents the time (in minutes) both pipes are open before pipe B is closed -/
def t_both : ℝ := 8

/-- Represents the total time (in minutes) to fill the tank using both pipes as described -/
def t_total : ℝ := 18

theorem tank_filling_time : 
  (t_both * (1 / A + 1 / B)) + ((t_total - t_both) * (1 / A)) = 1 ∧ 
  A = 24 := by
  sorry

#check tank_filling_time

end NUMINAMATH_CALUDE_tank_filling_time_l1402_140229


namespace NUMINAMATH_CALUDE_particular_number_exists_l1402_140298

theorem particular_number_exists : ∃ x : ℝ, 4 * 25 * x = 812 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_exists_l1402_140298


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_second_quadrant_condition_l1402_140242

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 6) (m^2 - 11*m + 24)

-- Theorem for part 1
theorem pure_imaginary_condition (m : ℝ) :
  z m = Complex.I * Complex.im (z m) ↔ m = -2 :=
sorry

-- Theorem for part 2
theorem second_quadrant_condition (m : ℝ) :
  Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 ↔ -2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_second_quadrant_condition_l1402_140242


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l1402_140244

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, m > 6 → Nat.choose 10 4 + Nat.choose 10 5 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l1402_140244


namespace NUMINAMATH_CALUDE_tshirt_count_l1402_140266

/-- The price of a pant in rupees -/
def pant_price : ℝ := sorry

/-- The price of a t-shirt in rupees -/
def tshirt_price : ℝ := sorry

/-- The total cost of 3 pants and 6 t-shirts in rupees -/
def total_cost_1 : ℝ := 750

/-- The total cost of 1 pant and 12 t-shirts in rupees -/
def total_cost_2 : ℝ := 750

/-- The amount to be spent on t-shirts in rupees -/
def tshirt_budget : ℝ := 400

theorem tshirt_count : 
  3 * pant_price + 6 * tshirt_price = total_cost_1 →
  pant_price + 12 * tshirt_price = total_cost_2 →
  (tshirt_budget / tshirt_price : ℝ) = 8 := by
sorry

end NUMINAMATH_CALUDE_tshirt_count_l1402_140266


namespace NUMINAMATH_CALUDE_largest_divisor_of_15_less_than_15_l1402_140291

theorem largest_divisor_of_15_less_than_15 :
  ∃ n : ℕ, n ∣ 15 ∧ n ≠ 15 ∧ ∀ m : ℕ, m ∣ 15 ∧ m ≠ 15 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_15_less_than_15_l1402_140291


namespace NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_l1402_140228

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 7

/-- The number of successful rolls (odd numbers) we want -/
def num_success : ℕ := 5

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls :
  (Nat.choose num_rolls num_success : ℚ) * prob_odd ^ num_success * (1 - prob_odd) ^ (num_rolls - num_success) = 21/128 :=
sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_l1402_140228


namespace NUMINAMATH_CALUDE_christines_speed_l1402_140279

theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 20 ∧ time = 5 ∧ speed = distance / time → speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_christines_speed_l1402_140279


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1402_140218

/-- Given two rectangles A and B, prove the ratio of their sides -/
theorem rectangle_side_ratio 
  (a b c d : ℝ) 
  (h1 : a * b / (c * d) = 0.16) 
  (h2 : a / c = 2 / 5) : 
  b / d = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1402_140218


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1402_140280

/-- Given a rectangle with length 40 cm, if reducing the length by 5 cm and
    increasing the width by 5 cm results in an area increase of 75 cm²,
    then the ratio of the original length to the original width is 2:1. -/
theorem rectangle_ratio (w : ℝ) : 
  (40 - 5) * (w + 5) = 40 * w + 75 → 40 / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1402_140280


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1402_140232

/-- Given a group of 10 persons, prove that if replacing one person with a new person
    weighing 110 kg increases the average weight by 4 kg, then the weight of the
    replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (initial_avg : ℝ)
  (h1 : initial_avg > 0)
  (h2 : (10 * (initial_avg + 4) - 10 * initial_avg) = (110 - 70)) :
  70 = 110 - (10 * (initial_avg + 4) - 10 * initial_avg) :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1402_140232


namespace NUMINAMATH_CALUDE_point_upper_right_of_line_l1402_140288

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is to the upper right of a line -/
def isUpperRight (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

theorem point_upper_right_of_line (a : ℝ) :
  let p : Point := ⟨-1, a⟩
  let l : Line := ⟨1, 1, -3⟩
  isUpperRight p l → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_point_upper_right_of_line_l1402_140288
