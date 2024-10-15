import Mathlib

namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1005_100536

/-- The set of even digits -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- A function to check if a natural number contains all even digits -/
def containsAllEvenDigits (n : Nat) : Prop :=
  ∀ d ∈ evenDigits, ∃ k : Nat, n / (10 ^ k) % 10 = d

/-- A function to check if a natural number is an eight-digit number -/
def isEightDigitNumber (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

/-- The theorem stating that 99986420 is the largest eight-digit number containing all even digits -/
theorem largest_eight_digit_with_even_digits :
  (∀ n : Nat, isEightDigitNumber n → containsAllEvenDigits n → n ≤ 99986420) ∧
  isEightDigitNumber 99986420 ∧
  containsAllEvenDigits 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l1005_100536


namespace NUMINAMATH_CALUDE_parabola_focus_l1005_100595

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola x = (1/4)y^2 is at (1, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_equation x y → 
  (∃ (p : ℝ × ℝ), p = focus ∧ 
   (x - p.1)^2 + (y - p.2)^2 = (x + p.1)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1005_100595


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1005_100515

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x, 2 * f x + f (1 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 + (2/3) * x - 1/3) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1005_100515


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1005_100593

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 20000) → x = 55172 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1005_100593


namespace NUMINAMATH_CALUDE_min_value_theorem_l1005_100518

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (3 / x + 4 / y) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1005_100518


namespace NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l1005_100538

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, a * b = 19 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 362 / 361 := by
  sorry

end NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l1005_100538


namespace NUMINAMATH_CALUDE_pen_ratio_l1005_100580

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  dorothy : ℕ
  julia : ℕ
  robert : ℕ

/-- The cost of one pen in cents -/
def pen_cost : ℕ := 150

/-- The total amount spent by the three friends in cents -/
def total_spent : ℕ := 3300

/-- Conditions of the pen purchase -/
def pen_purchase_conditions (p : PenPurchase) : Prop :=
  p.julia = 3 * p.robert ∧
  p.robert = 4 ∧
  p.dorothy + p.julia + p.robert = total_spent / pen_cost

theorem pen_ratio (p : PenPurchase) 
  (h : pen_purchase_conditions p) : 
  p.dorothy * 2 = p.julia := by
  sorry

end NUMINAMATH_CALUDE_pen_ratio_l1005_100580


namespace NUMINAMATH_CALUDE_number_triangle_problem_l1005_100556

theorem number_triangle_problem (x y : ℕ+) (h : x * y = 2022) : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), (m * m ∣ x) ∧ (m * m ∣ y) → m ≤ n) ∧
  (∀ (n : ℕ+), (n * n ∣ x) ∧ (n * n ∣ y) → n = 1) :=
sorry

end NUMINAMATH_CALUDE_number_triangle_problem_l1005_100556


namespace NUMINAMATH_CALUDE_min_value_sum_l1005_100503

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1005_100503


namespace NUMINAMATH_CALUDE_claire_age_in_two_years_l1005_100554

/-- Given that Jessica is 24 years old and 6 years older than Claire, 
    prove that Claire will be 20 years old in two years. -/
theorem claire_age_in_two_years 
  (jessica_age : ℕ) 
  (claire_age : ℕ) 
  (h1 : jessica_age = 24)
  (h2 : jessica_age = claire_age + 6) : 
  claire_age + 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_claire_age_in_two_years_l1005_100554


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1005_100574

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (final_price : ℝ) :
  original_price = 9795.3216374269 →
  first_discount = 20 →
  third_discount = 5 →
  final_price = 6700 →
  ∃ (second_discount : ℝ), 
    (original_price * (1 - first_discount / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) = final_price) ∧
    (abs (second_discount - 10) < 0.0000000001) := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1005_100574


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1005_100509

theorem unique_solution_power_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (18 * x)^9 :=
by
  use 2/9
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1005_100509


namespace NUMINAMATH_CALUDE_min_value_inequality_l1005_100507

theorem min_value_inequality (x : ℝ) (h : x ≥ 4) : x + 4 / (x - 1) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1005_100507


namespace NUMINAMATH_CALUDE_inequality_proof_l1005_100552

theorem inequality_proof (a b c : ℝ) (n : ℕ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hn : n > 0) :
  a^n + b^n + c^n ≥ a*b^(n-1) + b*c^(n-1) + c*a^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1005_100552


namespace NUMINAMATH_CALUDE_sequence_ratio_implies_half_l1005_100553

/-- Represents a positive rational number less than 1 -/
structure PositiveRationalLessThanOne where
  val : ℚ
  pos : 0 < val
  lt_one : val < 1

/-- Given conditions for the sequences and their relationship -/
structure SequenceConditions where
  d : ℚ
  d_nonzero : d ≠ 0
  q : PositiveRationalLessThanOne
  a : ℕ → ℚ
  b : ℕ → ℚ
  a_def : ∀ n, a n = d * n
  b_def : ∀ n, b n = d^2 * q.val^(n-1)
  sum_ratio_integer : ∃ k : ℕ+, (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) = k

/-- The main theorem stating that under the given conditions, q must equal 1/2 -/
theorem sequence_ratio_implies_half (cond : SequenceConditions) : cond.q.val = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_implies_half_l1005_100553


namespace NUMINAMATH_CALUDE_equation_solution_l1005_100582

theorem equation_solution :
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1005_100582


namespace NUMINAMATH_CALUDE_dave_lost_tickets_l1005_100598

/-- Prove that Dave lost 2 tickets at the arcade -/
theorem dave_lost_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) 
  (h1 : initial_tickets = 14)
  (h2 : spent_tickets = 10)
  (h3 : remaining_tickets = 2) :
  initial_tickets - (spent_tickets + remaining_tickets) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_lost_tickets_l1005_100598


namespace NUMINAMATH_CALUDE_overlapping_rectangle_area_l1005_100561

theorem overlapping_rectangle_area (Y : ℝ) (X : ℝ) (h1 : Y > 0) (h2 : X > 0) 
  (h3 : X = (1/8) * (2*Y - X)) : X = (2/9) * Y := by
  sorry

end NUMINAMATH_CALUDE_overlapping_rectangle_area_l1005_100561


namespace NUMINAMATH_CALUDE_profit_after_reduction_profit_for_target_l1005_100500

/-- Represents the daily sales and profit calculations for a product. -/
structure ProductSales where
  basePrice : ℝ
  baseSales : ℝ
  profitPerItem : ℝ
  salesIncreasePerYuan : ℝ

/-- Calculates the daily profit given a price reduction. -/
def dailyProfit (p : ProductSales) (priceReduction : ℝ) : ℝ :=
  (p.profitPerItem - priceReduction) * (p.baseSales + p.salesIncreasePerYuan * priceReduction)

/-- Theorem stating that a 3 yuan price reduction results in 1692 yuan daily profit. -/
theorem profit_after_reduction (p : ProductSales) 
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 3 = 1692 := by sorry

/-- Theorem stating that a 25 yuan price reduction results in 2000 yuan daily profit. -/
theorem profit_for_target (p : ProductSales)
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 25 = 2000 := by sorry

end NUMINAMATH_CALUDE_profit_after_reduction_profit_for_target_l1005_100500


namespace NUMINAMATH_CALUDE_no_convincing_statement_when_guilty_l1005_100558

/-- Represents a statement made in court -/
def Statement : Type := String

/-- Represents the state of being guilty or innocent -/
inductive GuiltState
| Guilty
| Innocent

/-- Represents a jury's belief about guilt -/
inductive JuryBelief
| BelievesGuilty
| BelievesInnocent

/-- A function that models how a rational jury processes a statement -/
def rationalJuryProcess : Statement → GuiltState → JuryBelief := sorry

/-- The theorem stating that it's impossible to convince a rational jury of innocence when guilty -/
theorem no_convincing_statement_when_guilty :
  ∀ (s : Statement), rationalJuryProcess s GuiltState.Guilty ≠ JuryBelief.BelievesInnocent := by
  sorry

end NUMINAMATH_CALUDE_no_convincing_statement_when_guilty_l1005_100558


namespace NUMINAMATH_CALUDE_officer_selection_l1005_100589

theorem officer_selection (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  (n.choose m) * m.factorial = 60 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_l1005_100589


namespace NUMINAMATH_CALUDE_work_completion_time_l1005_100542

/-- The time it takes for A to complete the entire work -/
def a_complete_time : ℝ := 21

/-- The time it takes for B to complete the entire work -/
def b_complete_time : ℝ := 15

/-- The number of days B worked before leaving -/
def b_worked_days : ℝ := 10

/-- The time it takes for A to complete the remaining work after B leaves -/
def a_remaining_time : ℝ := 7

theorem work_completion_time :
  a_complete_time = 21 →
  b_complete_time = 15 →
  b_worked_days = 10 →
  a_remaining_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1005_100542


namespace NUMINAMATH_CALUDE_geom_seq_sum_property_l1005_100555

/-- Represents a geometric sequence and its properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)
  geom_seq : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence with S_4 = 1 and S_12 = 13,
prove that a_13 + a_14 + a_15 + a_16 = 27
-/
theorem geom_seq_sum_property (g : GeometricSequence) 
  (h1 : g.S 4 = 1) (h2 : g.S 12 = 13) :
  g.a 13 + g.a 14 + g.a 15 + g.a 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geom_seq_sum_property_l1005_100555


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1005_100571

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1005_100571


namespace NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l1005_100568

/-- 
Given odd integers a and b, the number of odd terms 
in the expansion of (a+b)^8 is equal to 2.
-/
theorem odd_terms_in_binomial_expansion (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  (Finset.filter (fun i => Odd (Nat.choose 8 i * a^(8-i) * b^i)) 
    (Finset.range 9)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l1005_100568


namespace NUMINAMATH_CALUDE_notebook_payment_l1005_100564

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The cost of the notebook in cents -/
def notebook_cost : ℕ := 130

/-- The number of nickels needed to pay for the notebook -/
def nickels_needed : ℕ := notebook_cost / nickel_value

theorem notebook_payment :
  nickels_needed = 26 := by sorry

end NUMINAMATH_CALUDE_notebook_payment_l1005_100564


namespace NUMINAMATH_CALUDE_locus_of_Q_l1005_100523

/-- The locus of point Q given an ellipse with specific properties -/
theorem locus_of_Q (a b : ℝ) (P : ℝ × ℝ) (E : ℝ × ℝ) (Q : ℝ × ℝ) :
  a > b → b > 0 →
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1 →
  P ≠ (-2, 0) → P ≠ (2, 0) →
  a = 2 →
  (1 : ℝ) / 2 = Real.sqrt (1 - b^2 / a^2) →
  E.1 - (-4) = (3 / 5) * (P.1 - (-4)) →
  E.2 = (3 / 5) * P.2 →
  (Q.2 + 2) / (Q.1 + 2) = P.2 / (P.1 + 2) →
  (Q.2 - 0) / (Q.1 - 2) = E.2 / (E.1 - 2) →
  Q.2 ≠ 0 →
  (Q.1 + 1)^2 + (4 * Q.2^2) / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_locus_of_Q_l1005_100523


namespace NUMINAMATH_CALUDE_star_comm_star_assoc_star_disprove_l1005_100575

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem 1: Commutativity of *
theorem star_comm (a b : ℝ) : star a b = star b a := by sorry

-- Theorem 2: Associativity of *
theorem star_assoc (a b c : ℝ) : star (star a b) c = star a (star b c) := by sorry

-- Theorem 3: Disprove the given property
theorem star_disprove : ¬(∀ (a b : ℝ), star (a + 1) b = star a b + star 1 b) := by sorry

end NUMINAMATH_CALUDE_star_comm_star_assoc_star_disprove_l1005_100575


namespace NUMINAMATH_CALUDE_kickball_players_l1005_100534

theorem kickball_players (wednesday : ℕ) (thursday : ℕ) (difference : ℕ) : 
  wednesday = 37 →
  difference = 9 →
  thursday = wednesday - difference →
  wednesday + thursday = 65 := by
sorry

end NUMINAMATH_CALUDE_kickball_players_l1005_100534


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1005_100524

theorem lcm_hcf_problem (n : ℕ) 
  (h1 : Nat.lcm 12 n = 60) 
  (h2 : Nat.gcd 12 n = 3) : 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1005_100524


namespace NUMINAMATH_CALUDE_cliff_rock_collection_l1005_100530

theorem cliff_rock_collection (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  igneous / 3 = 30 →
  igneous + sedimentary = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_cliff_rock_collection_l1005_100530


namespace NUMINAMATH_CALUDE_vector_collinearity_l1005_100540

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity (m : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 2*m - 3)
  collinear a b → m = -3 := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1005_100540


namespace NUMINAMATH_CALUDE_video_votes_l1005_100577

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) : 
  likes + dislikes = total_votes →
  likes = (3 * total_votes) / 4 →
  dislikes = total_votes / 4 →
  likes - dislikes = 50 →
  total_votes = 100 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l1005_100577


namespace NUMINAMATH_CALUDE_smallest_d_for_factorization_l1005_100559

theorem smallest_d_for_factorization : 
  (∃ (p q : ℤ), x^2 + 107*x + 2050 = (x + p) * (x + q)) ∧ 
  (∀ (d : ℕ), d < 107 → ¬∃ (p q : ℤ), x^2 + d*x + 2050 = (x + p) * (x + q)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_factorization_l1005_100559


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l1005_100537

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 45 →
  editors > 36 →
  writers + editors - x + 2 * x = total →
  x ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l1005_100537


namespace NUMINAMATH_CALUDE_lowest_salary_grade_l1005_100511

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on the salary grade -/
def hourlyWage (s : SalaryGrade) : ℝ :=
  7.50 + 0.25 * (s.val - 1)

/-- States that the difference in hourly wage between the highest and lowest salary grade is $1.25 -/
axiom wage_difference (s_min s_max : SalaryGrade) :
  s_min.val = 1 ∧ s_max.val = 5 →
  hourlyWage s_max - hourlyWage s_min = 1.25

theorem lowest_salary_grade :
  ∃ (s_min : SalaryGrade), s_min.val = 1 ∧
  ∀ (s : SalaryGrade), s_min.val ≤ s.val :=
by sorry

end NUMINAMATH_CALUDE_lowest_salary_grade_l1005_100511


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1005_100529

/-- The volume of a cylinder formed by rotating a rectangle about its longer side. -/
theorem cylinder_volume_from_rectangle (width length : ℝ) (h_width : width = 8) (h_length : length = 20) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 320 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1005_100529


namespace NUMINAMATH_CALUDE_problem_statement_l1005_100560

theorem problem_statement (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : a^2 + b^2 = 4*a*b) : 
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1005_100560


namespace NUMINAMATH_CALUDE_ninth_power_sum_l1005_100525

/-- Given two real numbers m and n satisfying specific conditions, prove that m⁹ + n⁹ = 76 -/
theorem ninth_power_sum (m n : ℝ) 
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) : 
  m^9 + n^9 = 76 := by
  sorry

#check ninth_power_sum

end NUMINAMATH_CALUDE_ninth_power_sum_l1005_100525


namespace NUMINAMATH_CALUDE_dogs_added_on_monday_l1005_100594

theorem dogs_added_on_monday
  (initial_dogs : ℕ)
  (sunday_dogs : ℕ)
  (total_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : sunday_dogs = 5)
  (h3 : total_dogs = 10)
  : total_dogs - (initial_dogs + sunday_dogs) = 3 :=
by sorry

end NUMINAMATH_CALUDE_dogs_added_on_monday_l1005_100594


namespace NUMINAMATH_CALUDE_alices_number_l1005_100526

def possible_numbers : List ℕ := [1080, 1440, 1800, 2160, 2520, 2880]

theorem alices_number (n : ℕ) :
  (40 ∣ n) →
  (72 ∣ n) →
  1000 < n →
  n < 3000 →
  n ∈ possible_numbers := by
sorry

end NUMINAMATH_CALUDE_alices_number_l1005_100526


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1005_100527

theorem grade_assignment_count : 
  (number_of_grades : ℕ) → 
  (number_of_students : ℕ) → 
  number_of_grades = 4 → 
  number_of_students = 12 → 
  number_of_grades ^ number_of_students = 16777216 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1005_100527


namespace NUMINAMATH_CALUDE_combined_mpg_l1005_100539

/-- Combined rate of miles per gallon for two cars -/
theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℚ) :
  ray_mpg = 50 →
  tom_mpg = 25 →
  ray_miles = 100 →
  tom_miles = 200 →
  (ray_miles + tom_miles) / (ray_miles / ray_mpg + tom_miles / tom_mpg) = 30 := by
  sorry


end NUMINAMATH_CALUDE_combined_mpg_l1005_100539


namespace NUMINAMATH_CALUDE_trajectory_and_circle_properties_l1005_100549

-- Define the vectors a and b
def a (m x y : ℝ) : ℝ × ℝ := (m * x, y + 1)
def b (x y : ℝ) : ℝ × ℝ := (x, y - 1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicularity condition
def perpendicular (m x y : ℝ) : Prop := dot_product (a m x y) (b x y) = 0

-- Define the equation of trajectory E
def trajectory_equation (m x y : ℝ) : Prop := m * x^2 + y^2 = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4/5

-- Define a tangent line to the circle
def tangent_line (k t x y : ℝ) : Prop := y = k * x + t

-- Define the perpendicularity condition for OA and OB
def OA_perp_OB (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem trajectory_and_circle_properties (m : ℝ) :
  (∀ x y : ℝ, perpendicular m x y → trajectory_equation m x y) ∧
  (m = 1/4 →
    ∃ k t x1 y1 x2 y2 : ℝ,
      tangent_line k t x1 y1 ∧
      tangent_line k t x2 y2 ∧
      trajectory_equation m x1 y1 ∧
      trajectory_equation m x2 y2 ∧
      circle_equation x1 y1 ∧
      circle_equation x2 y2 ∧
      OA_perp_OB x1 y1 x2 y2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_circle_properties_l1005_100549


namespace NUMINAMATH_CALUDE_min_sum_exponents_520_l1005_100519

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Expresses a natural number as a sum of distinct powers of 2 -/
def expressAsPowersOf2 (n : ℕ) : List ℕ := sorry

theorem min_sum_exponents_520 :
  let powers := expressAsPowersOf2 520
  powers.length ≥ 2 ∧ sumOfExponents 520 = 12 :=
sorry

end NUMINAMATH_CALUDE_min_sum_exponents_520_l1005_100519


namespace NUMINAMATH_CALUDE_compare_expressions_l1005_100514

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > (Real.sqrt (a*b) + 1/Real.sqrt (a*b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > ((a+b)/2 + 2/(a+b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) < ((a+b)/2 + 2/(a+b))^2 :=
by sorry

end NUMINAMATH_CALUDE_compare_expressions_l1005_100514


namespace NUMINAMATH_CALUDE_correct_celsius_to_fahrenheit_conversion_l1005_100531

/-- Conversion function from Celsius to Fahrenheit -/
def celsiusToFahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating the correct conversion from Celsius to Fahrenheit -/
theorem correct_celsius_to_fahrenheit_conversion (c : ℝ) : 
  celsiusToFahrenheit c = 1.8 * c + 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_celsius_to_fahrenheit_conversion_l1005_100531


namespace NUMINAMATH_CALUDE_investment_solution_l1005_100570

def investment_problem (x : ℝ) : Prop :=
  let total_investment : ℝ := 1500
  let rate1 : ℝ := 1.04  -- 4% annual compound interest
  let rate2 : ℝ := 1.06  -- 6% annual compound interest
  let total_after_year : ℝ := 1590
  (x * rate1 + (total_investment - x) * rate2 = total_after_year) ∧
  (0 ≤ x) ∧ (x ≤ total_investment)

theorem investment_solution :
  ∃! x : ℝ, investment_problem x ∧ x = 0 :=
sorry

end NUMINAMATH_CALUDE_investment_solution_l1005_100570


namespace NUMINAMATH_CALUDE_circle_radius_determines_c_l1005_100532

/-- The equation of a circle with center (h, k) and radius r can be written as
    (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r c : ℝ) : Prop :=
  ∀ x y, x^2 + 6*x + y^2 - 4*y + c = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2

theorem circle_radius_determines_c : 
  ∀ c : ℝ, (CircleEquation (-3) 2 4 c) → c = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_determines_c_l1005_100532


namespace NUMINAMATH_CALUDE_min_value_abc_l1005_100541

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/1152 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/1152 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l1005_100541


namespace NUMINAMATH_CALUDE_shirts_per_minute_l1005_100502

/-- A machine that makes shirts -/
structure ShirtMachine where
  yesterday_production : ℕ
  today_production : ℕ
  total_working_time : ℕ

/-- Theorem: The machine can make 8 shirts per minute -/
theorem shirts_per_minute (m : ShirtMachine)
  (h1 : m.yesterday_production = 13)
  (h2 : m.today_production = 3)
  (h3 : m.total_working_time = 2) :
  (m.yesterday_production + m.today_production) / m.total_working_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_minute_l1005_100502


namespace NUMINAMATH_CALUDE_twenty_sheets_joined_length_l1005_100566

/-- The length of joined papers given the number of sheets, length per sheet, and overlap length -/
def joinedPapersLength (numSheets : ℕ) (sheetLength : ℝ) (overlapLength : ℝ) : ℝ :=
  numSheets * sheetLength - (numSheets - 1) * overlapLength

/-- Theorem stating that 20 sheets of 10 cm paper with 0.5 cm overlap results in 190.5 cm total length -/
theorem twenty_sheets_joined_length :
  joinedPapersLength 20 10 0.5 = 190.5 := by
  sorry

#eval joinedPapersLength 20 10 0.5

end NUMINAMATH_CALUDE_twenty_sheets_joined_length_l1005_100566


namespace NUMINAMATH_CALUDE_platform_length_l1005_100544

/-- Given a train of length 450 m, running at 108 kmph, crosses a platform in 25 seconds,
    prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 450 ∧ 
  train_speed_kmph = 108 ∧ 
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time - train_length) = 300 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l1005_100544


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1005_100572

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1005_100572


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1005_100585

open Real

theorem min_value_trigonometric_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  (4 * cos θ + 3 / sin θ + 2 * sqrt 2 * tan θ) ≥ 6 * sqrt 3 * (2 ^ (1/6)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1005_100585


namespace NUMINAMATH_CALUDE_bob_spending_theorem_l1005_100508

def monday_spending (initial_amount : ℚ) : ℚ := initial_amount / 2

def tuesday_spending (monday_remainder : ℚ) : ℚ := monday_remainder / 5

def wednesday_spending (tuesday_remainder : ℚ) : ℚ := tuesday_remainder * 3 / 8

def final_amount (initial_amount : ℚ) : ℚ :=
  let monday_remainder := initial_amount - monday_spending initial_amount
  let tuesday_remainder := monday_remainder - tuesday_spending monday_remainder
  tuesday_remainder - wednesday_spending tuesday_remainder

theorem bob_spending_theorem :
  final_amount 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bob_spending_theorem_l1005_100508


namespace NUMINAMATH_CALUDE_rug_dimension_l1005_100516

theorem rug_dimension (floor_area : ℝ) (rug_width : ℝ) (coverage_fraction : ℝ) :
  floor_area = 64 ∧ 
  rug_width = 2 ∧ 
  coverage_fraction = 0.21875 →
  ∃ (rug_length : ℝ), 
    rug_length * rug_width = floor_area * coverage_fraction ∧
    rug_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_rug_dimension_l1005_100516


namespace NUMINAMATH_CALUDE_olympic_medals_l1005_100586

/-- Olympic Medals Theorem -/
theorem olympic_medals (china_total russia_total us_total : ℕ)
  (china_gold china_silver china_bronze : ℕ)
  (russia_gold russia_silver russia_bronze : ℕ)
  (us_gold us_silver us_bronze : ℕ)
  (h1 : china_total = 100)
  (h2 : russia_total = 72)
  (h3 : us_total = 110)
  (h4 : china_silver + china_bronze = russia_silver + russia_bronze)
  (h5 : russia_gold + 28 = china_gold)
  (h6 : us_gold = russia_gold + 13)
  (h7 : us_gold = us_bronze)
  (h8 : us_silver = us_gold + 2)
  (h9 : china_bronze = china_silver + 7)
  (h10 : china_total = china_gold + china_silver + china_bronze)
  (h11 : russia_total = russia_gold + russia_silver + russia_bronze)
  (h12 : us_total = us_gold + us_silver + us_bronze) :
  china_gold = 51 ∧ us_silver = 38 ∧ russia_bronze = 28 := by
  sorry


end NUMINAMATH_CALUDE_olympic_medals_l1005_100586


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_when_black_exceeds_white_l1005_100501

/-- Represents a face of a polyhedron -/
structure Face where
  area : ℝ
  color : Bool  -- True for black, False for white

/-- Represents a convex polyhedron -/
structure Polyhedron where
  faces : List Face
  is_convex : Bool
  no_adjacent_black : Bool

/-- Checks if a sphere can be inscribed in the polyhedron -/
def can_inscribe_sphere (p : Polyhedron) : Prop :=
  sorry

/-- Calculates the total area of faces of a given color -/
def total_area (p : Polyhedron) (color : Bool) : ℝ :=
  sorry

/-- Main theorem -/
theorem no_inscribed_sphere_when_black_exceeds_white (p : Polyhedron) :
  p.is_convex ∧ 
  p.no_adjacent_black ∧ 
  (total_area p true > total_area p false) →
  ¬(can_inscribe_sphere p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_when_black_exceeds_white_l1005_100501


namespace NUMINAMATH_CALUDE_second_quadrant_point_coordinates_l1005_100546

/-- A point in the second quadrant of a coordinate plane. -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The theorem stating that a point in the second quadrant with given distances to the axes has specific coordinates. -/
theorem second_quadrant_point_coordinates (P : SecondQuadrantPoint) 
  (dist_x_axis : |P.y| = 4)
  (dist_y_axis : |P.x| = 5) :
  P.x = -5 ∧ P.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_point_coordinates_l1005_100546


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1005_100588

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 4) 
  (h2 : x ≠ 2) 
  (h3 : x ≠ 5) 
  (h4 : x ≠ 3) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  (x - 3)^2 / ((x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1005_100588


namespace NUMINAMATH_CALUDE_viktor_tally_theorem_l1005_100578

/-- Represents Viktor's tally system -/
structure TallySystem where
  x_value : ℕ  -- number of rallies represented by an X
  o_value : ℕ  -- number of rallies represented by an O

/-- Represents the final tally -/
structure FinalTally where
  o_count : ℕ  -- number of O's in the tally
  x_count : ℕ  -- number of X's in the tally

/-- Calculates the range of possible rallies given a tally system and final tally -/
def rally_range (system : TallySystem) (tally : FinalTally) : 
  {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1} :=
sorry

theorem viktor_tally_theorem (system : TallySystem) (tally : FinalTally) :
  system.x_value = 10 ∧ 
  system.o_value = 100 ∧ 
  tally.o_count = 3 ∧ 
  tally.x_count = 7 →
  ∃ (range : {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1}),
    range = rally_range system tally ∧
    range.val = 370 ∧
    (∃ max_rallies : ℕ, range.property.choose = 379) :=
sorry

end NUMINAMATH_CALUDE_viktor_tally_theorem_l1005_100578


namespace NUMINAMATH_CALUDE_triangle_sides_from_heights_l1005_100592

/-- Given a triangle with heights d, e, and f corresponding to sides a, b, and c respectively,
    this theorem states the relationship between the sides and heights. -/
theorem triangle_sides_from_heights (d e f : ℝ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  ∃ (a b c : ℝ),
    let A := ((1/d + 1/e + 1/f) * (-1/d + 1/e + 1/f) * (1/d - 1/e + 1/f) * (1/d + 1/e - 1/f))
    a = 2 / (d * Real.sqrt A) ∧
    b = 2 / (e * Real.sqrt A) ∧
    c = 2 / (f * Real.sqrt A) :=
sorry

end NUMINAMATH_CALUDE_triangle_sides_from_heights_l1005_100592


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1005_100510

/-- A geometric sequence with first term 2 and fifth term 8 has its third term equal to 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))  -- a is a geometric sequence
  (h_a1 : a 1 = 2)  -- first term is 2
  (h_a5 : a 5 = 8)  -- fifth term is 8
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1005_100510


namespace NUMINAMATH_CALUDE_exam_mean_score_l1005_100576

/-- Given a distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean of the
    distribution is 76. -/
theorem exam_mean_score (μ σ : ℝ)
    (below_mean : μ - 2 * σ = 60)
    (above_mean : μ + 3 * σ = 100) :
    μ = 76 := by
  sorry


end NUMINAMATH_CALUDE_exam_mean_score_l1005_100576


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1005_100565

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1005_100565


namespace NUMINAMATH_CALUDE_intersection_point_mod17_l1005_100513

theorem intersection_point_mod17 :
  ∃ x : ℕ, x < 17 ∧
  (∀ y : ℕ, (y ≡ 7 * x + 3 [MOD 17]) ↔ (y ≡ 13 * x + 4 [MOD 17])) ∧
  x = 14 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_mod17_l1005_100513


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l1005_100545

theorem min_reciprocal_sum_squares (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x^2 + 1 / y^2) ≥ 2 / 25 :=
by sorry

theorem min_reciprocal_sum_squares_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 20 ∧ 1 / x^2 + 1 / y^2 < 2 / 25 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l1005_100545


namespace NUMINAMATH_CALUDE_range_of_a_l1005_100505

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 5*x + 4 ≤ 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1005_100505


namespace NUMINAMATH_CALUDE_chi_square_greater_than_critical_l1005_100551

/-- Represents the contingency table data --/
structure ContingencyTable where
  total_sample : ℕ
  disease_probability : ℚ
  blue_collar_with_disease : ℕ
  white_collar_without_disease : ℕ

/-- Calculates the chi-square value for the given contingency table --/
def calculate_chi_square (table : ContingencyTable) : ℚ :=
  let white_collar_with_disease := table.total_sample * table.disease_probability - table.blue_collar_with_disease
  let blue_collar_without_disease := table.total_sample * (1 - table.disease_probability) - table.white_collar_without_disease
  let n := table.total_sample
  let a := white_collar_with_disease
  let b := table.white_collar_without_disease
  let c := table.blue_collar_with_disease
  let d := blue_collar_without_disease
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for α = 0.005 --/
def critical_value : ℚ := 7879 / 1000

/-- Theorem stating that the calculated chi-square value is greater than the critical value --/
theorem chi_square_greater_than_critical (table : ContingencyTable) 
  (h1 : table.total_sample = 50)
  (h2 : table.disease_probability = 3/5)
  (h3 : table.blue_collar_with_disease = 10)
  (h4 : table.white_collar_without_disease = 5) :
  calculate_chi_square table > critical_value :=
sorry

end NUMINAMATH_CALUDE_chi_square_greater_than_critical_l1005_100551


namespace NUMINAMATH_CALUDE_beam_cost_calculation_l1005_100581

/-- Represents the dimensions of a beam -/
structure BeamDimensions where
  thickness : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a beam given its dimensions -/
def beamVolume (d : BeamDimensions) : ℕ :=
  d.thickness * d.width * d.length

/-- Calculates the total volume of multiple beams with the same dimensions -/
def totalVolume (count : ℕ) (d : BeamDimensions) : ℕ :=
  count * beamVolume d

/-- Theorem: Given the cost of 30 beams with dimensions 12x16x14,
    the cost of 14 beams with dimensions 8x12x10 is 16 2/3 coins -/
theorem beam_cost_calculation (cost_30_beams : ℚ) :
  let d1 : BeamDimensions := ⟨12, 16, 14⟩
  let d2 : BeamDimensions := ⟨8, 12, 10⟩
  cost_30_beams = 100 →
  (14 : ℚ) * cost_30_beams * (totalVolume 14 d2 : ℚ) / (totalVolume 30 d1 : ℚ) = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_beam_cost_calculation_l1005_100581


namespace NUMINAMATH_CALUDE_problem_l1005_100550

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l1005_100550


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1005_100547

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1005_100547


namespace NUMINAMATH_CALUDE_square_land_side_length_l1005_100557

theorem square_land_side_length (area : ℝ) (side : ℝ) :
  area = 400 →
  side * side = area →
  side = 20 := by
sorry

end NUMINAMATH_CALUDE_square_land_side_length_l1005_100557


namespace NUMINAMATH_CALUDE_probability_log_base_2_equal_1_l1005_100587

def dice_face := Fin 6

def is_valid_roll (x y : dice_face) : Prop :=
  (y.val : ℝ) = 2 * (x.val : ℝ)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem probability_log_base_2_equal_1 :
  (favorable_outcomes : ℝ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_log_base_2_equal_1_l1005_100587


namespace NUMINAMATH_CALUDE_infinitely_many_m_for_composite_sum_l1005_100563

theorem infinitely_many_m_for_composite_sum : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (m : ℕ+), m ∈ S → 
      ∀ (n : ℕ+), ∃ (a b : ℕ+), a * b = n^4 + m ∧ a ≠ 1 ∧ b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_m_for_composite_sum_l1005_100563


namespace NUMINAMATH_CALUDE_root_equation_a_value_l1005_100535

theorem root_equation_a_value 
  (x₁ x₂ x₃ a b : ℚ) : 
  x₁ = -3 - 5 * Real.sqrt 3 → 
  x₂ = -3 + 5 * Real.sqrt 3 → 
  x₃ = 15 / 11 → 
  x₁ * x₂ * x₃ = -90 → 
  x₁^3 + a*x₁^2 + b*x₁ + 90 = 0 → 
  a = -15 / 11 := by
sorry

end NUMINAMATH_CALUDE_root_equation_a_value_l1005_100535


namespace NUMINAMATH_CALUDE_betty_and_sister_book_ratio_l1005_100506

theorem betty_and_sister_book_ratio : 
  ∀ (betty_books sister_books : ℕ),
    betty_books = 20 →
    betty_books + sister_books = 45 →
    (sister_books : ℚ) / betty_books = 5 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_and_sister_book_ratio_l1005_100506


namespace NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l1005_100573

def vacation_days_earned (days_worked : ℕ) : ℕ :=
  days_worked / 10

def days_count_for_vacation (total_days_worked public_holidays sick_leave : ℕ) : ℕ :=
  total_days_worked - public_holidays - sick_leave

theorem andrew_remaining_vacation_days 
  (total_days_worked : ℕ) 
  (public_holidays : ℕ) 
  (sick_leave : ℕ) 
  (march_vacation : ℕ) 
  (h1 : total_days_worked = 290)
  (h2 : public_holidays = 10)
  (h3 : sick_leave = 5)
  (h4 : march_vacation = 5) :
  vacation_days_earned (days_count_for_vacation total_days_worked public_holidays sick_leave) - 
  (march_vacation + 2 * march_vacation) = 12 :=
by
  sorry

#eval vacation_days_earned (days_count_for_vacation 290 10 5) - (5 + 2 * 5)

end NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l1005_100573


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_alpha_l1005_100548

theorem cos_pi_third_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 3 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_alpha_l1005_100548


namespace NUMINAMATH_CALUDE_one_real_solution_l1005_100567

/-- The number of distinct real solutions to the equation (x-5)(x^2 + 5x + 10) = 0 -/
def num_solutions : ℕ := 1

/-- The equation (x-5)(x^2 + 5x + 10) = 0 has exactly one real solution -/
theorem one_real_solution : num_solutions = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_real_solution_l1005_100567


namespace NUMINAMATH_CALUDE_copy_machines_output_l1005_100562

/-- The rate of the first copy machine in copies per minute -/
def rate1 : ℕ := 30

/-- The rate of the second copy machine in copies per minute -/
def rate2 : ℕ := 55

/-- The time period in minutes -/
def time : ℕ := 30

/-- The total number of copies made by both machines in the given time period -/
def total_copies : ℕ := rate1 * time + rate2 * time

theorem copy_machines_output : total_copies = 2550 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_output_l1005_100562


namespace NUMINAMATH_CALUDE_system_solution_l1005_100521

theorem system_solution : 
  ∃ (x y : ℝ), (x^4 - y^4 = 3 * Real.sqrt (abs y) - 3 * Real.sqrt (abs x)) ∧ 
                (x^2 - 2*x*y = 27) ↔ 
  ((x = 3 ∧ y = -3) ∨ (x = -3 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1005_100521


namespace NUMINAMATH_CALUDE_square_land_area_l1005_100522

/-- The area of a square land plot with side length 20 units is 400 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 20) : side_length ^ 2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l1005_100522


namespace NUMINAMATH_CALUDE_min_product_xy_l1005_100599

theorem min_product_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end NUMINAMATH_CALUDE_min_product_xy_l1005_100599


namespace NUMINAMATH_CALUDE_range_of_a_l1005_100520

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ |x - a| + x - 4 ≤ 0) → a ∈ Set.Icc (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1005_100520


namespace NUMINAMATH_CALUDE_square_root_equation_l1005_100533

theorem square_root_equation : Real.sqrt 1936 / 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1005_100533


namespace NUMINAMATH_CALUDE_cos_negative_52_thirds_pi_l1005_100504

theorem cos_negative_52_thirds_pi : 
  Real.cos (-52 / 3 * Real.pi) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_52_thirds_pi_l1005_100504


namespace NUMINAMATH_CALUDE_cola_cost_l1005_100590

/-- Proves that the cost of each cola bottle is $2 given the conditions of Wilson's purchase. -/
theorem cola_cost (hamburger_price : ℚ) (num_hamburgers : ℕ) (num_cola : ℕ) (discount : ℚ) (total_paid : ℚ) :
  hamburger_price = 5 →
  num_hamburgers = 2 →
  num_cola = 3 →
  discount = 4 →
  total_paid = 12 →
  (total_paid + discount - num_hamburgers * hamburger_price) / num_cola = 2 := by
  sorry

end NUMINAMATH_CALUDE_cola_cost_l1005_100590


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1005_100591

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (is_pure_imaginary (a + b * I) → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary (a + b * I)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l1005_100591


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l1005_100596

theorem arithmetic_square_root_of_16 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x ^ 2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l1005_100596


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_recipes_l1005_100517

/-- Given a recipe that requires a certain amount of an ingredient and a total amount of that ingredient needed, calculate the number of recipes that can be made. -/
def recipes_to_make (cups_per_recipe : ℚ) (total_cups_needed : ℚ) : ℚ :=
  total_cups_needed / cups_per_recipe

/-- Prove that 23 recipes can be made given the conditions of the chocolate chip cookie problem. -/
theorem chocolate_chip_cookie_recipes : 
  recipes_to_make 2 46 = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_recipes_l1005_100517


namespace NUMINAMATH_CALUDE_winter_sales_calculation_l1005_100569

/-- Represents the sales of pizzas in millions for each season -/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- Calculates the total annual sales given the seasonal sales -/
def totalAnnualSales (sales : SeasonalSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- Theorem: Given the conditions, the number of pizzas sold in winter is 6.6 million -/
theorem winter_sales_calculation (sales : SeasonalSales)
    (h1 : sales.fall = 0.2 * totalAnnualSales sales)
    (h2 : sales.winter = 1.1 * sales.summer)
    (h3 : sales.spring = 5)
    (h4 : sales.summer = 6) :
    sales.winter = 6.6 := by
  sorry

#check winter_sales_calculation

end NUMINAMATH_CALUDE_winter_sales_calculation_l1005_100569


namespace NUMINAMATH_CALUDE_min_value_expression_l1005_100528

theorem min_value_expression (x y z : ℝ) (hx : x ≥ 3) (hy : y ≥ 3) (hz : z ≥ 3) :
  let A := ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x*y + y*z + z*x)
  A ≥ 1 ∧ (A = 1 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1005_100528


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l1005_100584

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions
def satisfies_conditions (A B : Point) : Prop :=
  A.y = parabola A.x ∧ 
  B.y = parabola B.x ∧ 
  A.x < 0 ∧ 
  B.x > 0 ∧
  A.y > B.y

-- Define the theorem
theorem parabola_points_theorem (A B : Point) (h : satisfies_conditions A B) :
  (A.x = -4 ∧ B.x = 2) ∨ (A.x = 4 ∧ B.x = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l1005_100584


namespace NUMINAMATH_CALUDE_books_in_boxes_l1005_100583

/-- The number of ways to place n different objects into k different boxes -/
def arrangements (n k : ℕ) : ℕ := k^n

/-- There are 6 different books -/
def num_books : ℕ := 6

/-- There are 5 different boxes -/
def num_boxes : ℕ := 5

/-- Theorem: The number of ways to place 6 different books into 5 different boxes is 5^6 -/
theorem books_in_boxes : arrangements num_books num_boxes = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_books_in_boxes_l1005_100583


namespace NUMINAMATH_CALUDE_age_difference_l1005_100512

/-- Proves that A is 10 years older than B given the conditions in the problem -/
theorem age_difference (A B : ℕ) : 
  B = 70 →  -- B's present age is 70 years
  A + 20 = 2 * (B - 20) →  -- In 20 years, A will be twice as old as B was 20 years ago
  A - B = 10  -- A is 10 years older than B
  := by sorry

end NUMINAMATH_CALUDE_age_difference_l1005_100512


namespace NUMINAMATH_CALUDE_y_divisibility_l1005_100543

theorem y_divisibility : ∃ k : ℕ, 
  (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) = 4 * k ∧ 
  (∃ m : ℕ, (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 8 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 16 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 32 * m) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l1005_100543


namespace NUMINAMATH_CALUDE_cow_count_l1005_100579

theorem cow_count (total_legs : ℕ) (legs_per_cow : ℕ) (h1 : total_legs = 460) (h2 : legs_per_cow = 4) : 
  total_legs / legs_per_cow = 115 := by
sorry

end NUMINAMATH_CALUDE_cow_count_l1005_100579


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1005_100597

theorem polynomial_division_theorem (x : ℝ) : 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1) + 9 = x^6 + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1005_100597
