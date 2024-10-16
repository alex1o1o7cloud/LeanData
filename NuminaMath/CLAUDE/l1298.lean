import Mathlib

namespace NUMINAMATH_CALUDE_bookshop_revenue_l1298_129890

-- Define book types and their prices
structure BookType where
  name : String
  price : Nat

-- Define a day's transactions
structure DayTransactions where
  novels_sold : Nat
  comics_sold : Nat
  biographies_sold : Nat
  novels_returned : Nat
  comics_returned : Nat
  biographies_returned : Nat
  discount : Nat  -- Discount percentage (0 for no discount)

def calculate_revenue (novel : BookType) (comic : BookType) (biography : BookType) 
                      (monday : DayTransactions) (tuesday : DayTransactions) 
                      (wednesday : DayTransactions) (thursday : DayTransactions) 
                      (friday : DayTransactions) : Nat :=
  sorry  -- Proof to be implemented

theorem bookshop_revenue : 
  let novel : BookType := { name := "Novel", price := 10 }
  let comic : BookType := { name := "Comic", price := 5 }
  let biography : BookType := { name := "Biography", price := 15 }
  
  let monday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 25,
    novels_returned := 1, comics_returned := 5, biographies_returned := 0,
    discount := 0
  }
  
  let tuesday : DayTransactions := {
    novels_sold := 20, comics_sold := 10, biographies_sold := 20,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 20
  }
  
  let wednesday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 14,
    novels_returned := 5, comics_returned := 0, biographies_returned := 3,
    discount := 0
  }
  
  let thursday : DayTransactions := {
    novels_sold := 40, comics_sold := 25, biographies_sold := 13,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 10
  }
  
  let friday : DayTransactions := {
    novels_sold := 55, comics_sold := 40, biographies_sold := 40,
    novels_returned := 2, comics_returned := 5, biographies_returned := 3,
    discount := 0
  }
  
  calculate_revenue novel comic biography monday tuesday wednesday thursday friday = 3603 :=
by sorry


end NUMINAMATH_CALUDE_bookshop_revenue_l1298_129890


namespace NUMINAMATH_CALUDE_expression_evaluation_l1298_129873

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℤ := -3
  3 * (x^2 - 2*x^2*y) - 3*x^2 + 2*y - 2*(x^2*y + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1298_129873


namespace NUMINAMATH_CALUDE_share_distribution_l1298_129862

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 392 →
  a = (1 / 2) * b →
  b = (1 / 2) * c →
  total = a + b + c →
  c = 224 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l1298_129862


namespace NUMINAMATH_CALUDE_no_prime_satisfies_congruence_l1298_129829

theorem no_prime_satisfies_congruence : 
  ¬ ∃ (p : Nat) (r s : Int), 
    Nat.Prime p ∧ 
    (∀ (x : Int), (x^3 - x + 2) ≡ ((x - r)^2 * (x - s)) [ZMOD p]) ∧
    (∀ (r' s' : Int), 
      (∀ (x : Int), (x^3 - x + 2) ≡ ((x - r')^2 * (x - s')) [ZMOD p]) → 
      r' = r ∧ s' = s) :=
sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_congruence_l1298_129829


namespace NUMINAMATH_CALUDE_segment_point_difference_l1298_129879

/-- Given a line segment PQ with endpoints P(6,-2) and Q(-3,10), and a point R(a,b) on PQ such that
    the distance from P to R is one-third the distance from P to Q, prove that b-a = -1. -/
theorem segment_point_difference (a b : ℝ) : 
  let p : ℝ × ℝ := (6, -2)
  let q : ℝ × ℝ := (-3, 10)
  let r : ℝ × ℝ := (a, b)
  (r.1 - p.1) / (q.1 - p.1) = (r.2 - p.2) / (q.2 - p.2) ∧  -- R is on PQ
  (r.1 - p.1)^2 + (r.2 - p.2)^2 = (1/9) * ((q.1 - p.1)^2 + (q.2 - p.2)^2) -- PR = (1/3)PQ
  →
  b - a = -1 := by
sorry

end NUMINAMATH_CALUDE_segment_point_difference_l1298_129879


namespace NUMINAMATH_CALUDE_carlotta_performance_length_l1298_129869

/-- Represents the length of Carlotta's final stage performance in minutes -/
def performance_length : ℝ := 6

/-- For every minute of singing, Carlotta spends 3 minutes practicing -/
def practice_ratio : ℝ := 3

/-- For every minute of singing, Carlotta spends 5 minutes throwing tantrums -/
def tantrum_ratio : ℝ := 5

/-- The total combined time of singing, practicing, and throwing tantrums in minutes -/
def total_time : ℝ := 54

theorem carlotta_performance_length :
  performance_length * (1 + practice_ratio + tantrum_ratio) = total_time :=
sorry

end NUMINAMATH_CALUDE_carlotta_performance_length_l1298_129869


namespace NUMINAMATH_CALUDE_lanas_nickels_l1298_129834

theorem lanas_nickels (num_stacks : ℕ) (nickels_per_stack : ℕ) 
  (h1 : num_stacks = 9) (h2 : nickels_per_stack = 8) : 
  num_stacks * nickels_per_stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_lanas_nickels_l1298_129834


namespace NUMINAMATH_CALUDE_non_officers_count_l1298_129826

/-- Proves that the number of non-officers is 525 given the salary information --/
theorem non_officers_count (total_avg : ℝ) (officer_avg : ℝ) (non_officer_avg : ℝ) (officer_count : ℕ) :
  total_avg = 120 →
  officer_avg = 470 →
  non_officer_avg = 110 →
  officer_count = 15 →
  ∃ (non_officer_count : ℕ),
    (officer_count * officer_avg + non_officer_count * non_officer_avg) / (officer_count + non_officer_count) = total_avg ∧
    non_officer_count = 525 :=
by sorry

end NUMINAMATH_CALUDE_non_officers_count_l1298_129826


namespace NUMINAMATH_CALUDE_students_allowance_l1298_129840

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3 / 5 * allowance + 1 / 3 * (2 / 5 * allowance) + 60 / 100 = allowance) →
  allowance = 225 / 100 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l1298_129840


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1298_129851

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1298_129851


namespace NUMINAMATH_CALUDE_bench_cost_is_150_l1298_129882

/-- The cost of a bench and garden table, where the table costs twice as much as the bench. -/
def BenchAndTableCost (bench_cost : ℝ) : ℝ := bench_cost + 2 * bench_cost

/-- Theorem stating that the bench costs 150 dollars given the conditions. -/
theorem bench_cost_is_150 :
  ∃ (bench_cost : ℝ), BenchAndTableCost bench_cost = 450 ∧ bench_cost = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_bench_cost_is_150_l1298_129882


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1298_129872

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1298_129872


namespace NUMINAMATH_CALUDE_sum_even_correct_sum_even_range_correct_l1298_129835

/-- Sum of first n even numbers -/
def sum_even (n : ℕ) : ℕ := n^2 + n

/-- Sum of even numbers from a to b -/
def sum_even_range (a b : ℕ) : ℕ := sum_even (b/2) - sum_even ((a-2)/2)

theorem sum_even_correct (n : ℕ) : 
  2 * (n * (n + 1) / 2) = sum_even n := by sorry

theorem sum_even_range_correct : 
  sum_even_range 102 200 = 7550 := by sorry

end NUMINAMATH_CALUDE_sum_even_correct_sum_even_range_correct_l1298_129835


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1298_129846

/-- Proves that 25x^2 + 40x + 16 is a perfect square binomial -/
theorem perfect_square_binomial : 
  ∃ (p q : ℝ), ∀ x : ℝ, 25*x^2 + 40*x + 16 = (p*x + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1298_129846


namespace NUMINAMATH_CALUDE_figure_to_square_partition_l1298_129860

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a planar figure on a grid --/
def PlanarFigure := Set GridPoint

/-- Represents a transformation that can be applied to a set of points --/
structure Transformation where
  rotate : ℤ → GridPoint → GridPoint
  translate : ℤ → ℤ → GridPoint → GridPoint

/-- Checks if a set of points forms a square --/
def is_square (s : Set GridPoint) : Prop := sorry

/-- The main theorem --/
theorem figure_to_square_partition 
  (F : PlanarFigure) 
  (G : Set GridPoint) -- The grid
  (T : Transformation) -- Available transformations
  : 
  ∃ (S1 S2 S3 : Set GridPoint),
    (S1 ∪ S2 ∪ S3 = F) ∧ 
    (S1 ∩ S2 ≠ ∅) ∧ 
    (S2 ∩ S3 ≠ ∅) ∧ 
    (S3 ∩ S1 ≠ ∅) ∧
    ∃ (S : Set GridPoint), 
      is_square S ∧ 
      ∃ (f1 f2 f3 : Set GridPoint → Set GridPoint),
        (∀ p ∈ S1, ∃ q, q = T.rotate r1 (T.translate dx1 dy1 p) ∧ f1 {p} = {q}) ∧
        (∀ p ∈ S2, ∃ q, q = T.rotate r2 (T.translate dx2 dy2 p) ∧ f2 {p} = {q}) ∧
        (∀ p ∈ S3, ∃ q, q = T.rotate r3 (T.translate dx3 dy3 p) ∧ f3 {p} = {q}) ∧
        (f1 S1 ∪ f2 S2 ∪ f3 S3 = S)
  := by sorry

end NUMINAMATH_CALUDE_figure_to_square_partition_l1298_129860


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l1298_129857

theorem unique_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 15)
  (h2 : y + 1 / x = 7 / 20)
  (h3 : x * y = 2) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l1298_129857


namespace NUMINAMATH_CALUDE_pipe_problem_l1298_129810

theorem pipe_problem (fill_rate_A fill_rate_B empty_rate_C : ℝ) 
  (h_A : fill_rate_A = 1 / 20)
  (h_B : fill_rate_B = 1 / 30)
  (h_C : empty_rate_C > 0)
  (h_fill : 2 * fill_rate_A + 2 * fill_rate_B - 2 * empty_rate_C = 1) :
  empty_rate_C = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pipe_problem_l1298_129810


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1298_129885

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ 
  (a * x^2 + 2 * x + 1 = 0) ∧ 
  (a * y^2 + 2 * y + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1298_129885


namespace NUMINAMATH_CALUDE_combine_like_terms_l1298_129855

theorem combine_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1298_129855


namespace NUMINAMATH_CALUDE_sons_age_l1298_129842

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1298_129842


namespace NUMINAMATH_CALUDE_debby_water_bottles_l1298_129836

theorem debby_water_bottles (initial_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 264)
  (h2 : bottles_per_day = 15)
  (h3 : remaining_bottles = 99) :
  (initial_bottles - remaining_bottles) / bottles_per_day = 11 :=
by sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l1298_129836


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1298_129845

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Given linear function passes through a point -/
def passesThroughPoint (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

/-- The main theorem to be proved -/
theorem linear_function_not_in_third_quadrant :
  ∀ (p : Point), isInThirdQuadrant p → ¬passesThroughPoint ⟨-5, 2023⟩ p := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1298_129845


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l1298_129899

/-- The perimeter of a regular hexagon inscribed in a circle -/
theorem hexagon_perimeter (r : ℝ) (h : r = 10) : 
  6 * (2 * r * Real.sin (π / 6)) = 60 := by
  sorry

#check hexagon_perimeter

end NUMINAMATH_CALUDE_hexagon_perimeter_l1298_129899


namespace NUMINAMATH_CALUDE_odd_power_function_l1298_129800

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m

theorem odd_power_function (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f m x = -f m (-x)) →
  f m (m + 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_odd_power_function_l1298_129800


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l1298_129884

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : initial = 440)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  initial - lost - remaining = 111 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l1298_129884


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1298_129817

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1298_129817


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1298_129806

theorem magnitude_of_z (z : ℂ) (h : (z + 1) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1298_129806


namespace NUMINAMATH_CALUDE_seashell_collection_l1298_129875

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection -/
theorem seashell_collection (henry paul total : ℕ) (h1 : henry = 11) (h2 : paul = 24) (h3 : total = 59) :
  let leo := total - henry - paul
  let leo_remaining := leo - (leo / 4)
  henry + paul + leo_remaining = 53 := by sorry

end NUMINAMATH_CALUDE_seashell_collection_l1298_129875


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1298_129893

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h4 : x ≠ 4) (h6 : x ≠ 6) :
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) =
  (-4/5) / (x - 1) + (-1/2) / (x - 4) + (23/10) / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1298_129893


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1298_129868

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  2 * x + 1 / (x - 1) ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 1) :
  2 * x + 1 / (x - 1) = 2 + 2 * Real.sqrt 2 ↔ x = 1 + Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l1298_129868


namespace NUMINAMATH_CALUDE_dandelions_to_grandmother_value_l1298_129812

/-- The number of dandelion puffs Caleb gave to his grandmother -/
def dandelions_to_grandmother (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_dog : ℕ) 
  (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_dog + num_friends * to_each_friend)

theorem dandelions_to_grandmother_value : 
  dandelions_to_grandmother 40 3 3 2 3 9 = 5 := by sorry

end NUMINAMATH_CALUDE_dandelions_to_grandmother_value_l1298_129812


namespace NUMINAMATH_CALUDE_rachel_and_sarah_return_trip_money_l1298_129844

theorem rachel_and_sarah_return_trip_money :
  let initial_amount : ℚ := 50
  let gasoline_cost : ℚ := 8
  let lunch_cost : ℚ := 15.65
  let gift_cost_per_person : ℚ := 5
  let grandma_gift_per_person : ℚ := 10
  let num_people : ℕ := 2

  let total_spent : ℚ := gasoline_cost + lunch_cost + (gift_cost_per_person * num_people)
  let total_received_from_grandma : ℚ := grandma_gift_per_person * num_people
  let remaining_amount : ℚ := initial_amount - total_spent + total_received_from_grandma

  remaining_amount = 36.35 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_and_sarah_return_trip_money_l1298_129844


namespace NUMINAMATH_CALUDE_compound_interest_rate_calculation_l1298_129871

/-- Compound interest rate calculation -/
theorem compound_interest_rate_calculation
  (P : ℝ) (A : ℝ) (t : ℝ) (n : ℝ)
  (h_P : P = 12000)
  (h_A : A = 15200)
  (h_t : t = 7)
  (h_n : n = 1)
  : ∃ r : ℝ, (A = P * (1 + r / n) ^ (n * t)) ∧ (abs (r - 0.0332) < 0.0001) :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_calculation_l1298_129871


namespace NUMINAMATH_CALUDE_felix_weight_ratio_l1298_129895

/-- The weight ratio of Felix's brother to Felix -/
theorem felix_weight_ratio :
  let felix_lift_ratio : ℝ := 1.5
  let brother_lift_ratio : ℝ := 3
  let felix_lift_weight : ℝ := 150
  let brother_lift_weight : ℝ := 600
  let felix_weight := felix_lift_weight / felix_lift_ratio
  let brother_weight := brother_lift_weight / brother_lift_ratio
  brother_weight / felix_weight = 2 := by
sorry


end NUMINAMATH_CALUDE_felix_weight_ratio_l1298_129895


namespace NUMINAMATH_CALUDE_student_travel_fraction_l1298_129830

theorem student_travel_fraction (total_distance : ℝ) (bus_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 →
  bus_fraction = 2/3 →
  car_distance = 12 →
  (total_distance - (bus_fraction * total_distance + car_distance)) / total_distance = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_student_travel_fraction_l1298_129830


namespace NUMINAMATH_CALUDE_gorilla_exhibit_visitors_l1298_129891

def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_percentage : ℚ := 4/5

theorem gorilla_exhibit_visitors :
  (visitors_per_hour * open_hours : ℚ) * gorilla_exhibit_percentage = 320 := by
  sorry

end NUMINAMATH_CALUDE_gorilla_exhibit_visitors_l1298_129891


namespace NUMINAMATH_CALUDE_matthew_owns_26_cheap_shares_l1298_129802

/-- Calculates the number of shares of the less valuable stock Matthew owns --/
def calculate_less_valuable_shares (total_assets : ℕ) (expensive_share_price : ℕ) (expensive_shares : ℕ) : ℕ :=
  let cheap_share_price := expensive_share_price / 2
  let expensive_stock_value := expensive_share_price * expensive_shares
  let cheap_stock_value := total_assets - expensive_stock_value
  cheap_stock_value / cheap_share_price

/-- Proves that Matthew owns 26 shares of the less valuable stock --/
theorem matthew_owns_26_cheap_shares :
  calculate_less_valuable_shares 2106 78 14 = 26 := by
  sorry

end NUMINAMATH_CALUDE_matthew_owns_26_cheap_shares_l1298_129802


namespace NUMINAMATH_CALUDE_cone_slant_height_l1298_129823

/-- Represents the properties of a cone --/
structure Cone where
  baseRadius : ℝ
  sectorAngle : ℝ
  slantHeight : ℝ

/-- Theorem: For a cone with base radius 6 cm and sector angle 240°, the slant height is 9 cm --/
theorem cone_slant_height (c : Cone) 
  (h1 : c.baseRadius = 6)
  (h2 : c.sectorAngle = 240) : 
  c.slantHeight = 9 := by
  sorry

#check cone_slant_height

end NUMINAMATH_CALUDE_cone_slant_height_l1298_129823


namespace NUMINAMATH_CALUDE_inequality_direction_change_l1298_129843

theorem inequality_direction_change : ∃ (a b c : ℝ), a < b ∧ c * a > c * b :=
sorry

end NUMINAMATH_CALUDE_inequality_direction_change_l1298_129843


namespace NUMINAMATH_CALUDE_field_trip_minibusses_l1298_129853

theorem field_trip_minibusses (num_vans : ℕ) (students_per_van : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) : 
  num_vans = 6 →
  students_per_van = 10 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / students_per_minibus = 4 := by
sorry

end NUMINAMATH_CALUDE_field_trip_minibusses_l1298_129853


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1298_129819

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem smallest_multiplier_for_perfect_square :
  ∃ n : ℕ, n > 0 ∧ is_perfect_square (n * y) ∧
  ∀ m : ℕ, 0 < m ∧ m < n → ¬is_perfect_square (m * y) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1298_129819


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1298_129888

theorem trigonometric_identity (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1298_129888


namespace NUMINAMATH_CALUDE_mouse_seeds_count_l1298_129870

/-- Represents the number of seeds per burrow for the mouse -/
def mouse_seeds_per_burrow : ℕ := 4

/-- Represents the number of seeds per burrow for the rabbit -/
def rabbit_seeds_per_burrow : ℕ := 7

/-- Represents the difference in number of burrows between mouse and rabbit -/
def burrow_difference : ℕ := 3

theorem mouse_seeds_count (mouse_burrows rabbit_burrows : ℕ) 
  (h1 : mouse_burrows = rabbit_burrows + burrow_difference)
  (h2 : mouse_seeds_per_burrow * mouse_burrows = rabbit_seeds_per_burrow * rabbit_burrows) :
  mouse_seeds_per_burrow * mouse_burrows = 28 := by
  sorry

end NUMINAMATH_CALUDE_mouse_seeds_count_l1298_129870


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1298_129852

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + (3 - 2*I)*z = 2 + 5*I*z ∧ z = -(9/58) - (21/58)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1298_129852


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1298_129858

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a = 3 → b = 4 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = Real.sqrt 7 ∨ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1298_129858


namespace NUMINAMATH_CALUDE_seat_difference_is_three_l1298_129833

/-- Represents a bus with seats on left and right sides, and a back seat. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  seatCapacity : Nat
  totalCapacity : Nat

/-- The number of fewer seats on the right side compared to the left side. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats between left and right sides. -/
theorem seat_difference_is_three :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.seatCapacity = 3 ∧
    bus.backSeatCapacity = 10 ∧
    bus.totalCapacity = 91 ∧
    seatDifference bus = 3 := by
  sorry

#check seat_difference_is_three

end NUMINAMATH_CALUDE_seat_difference_is_three_l1298_129833


namespace NUMINAMATH_CALUDE_factorize_expression_1_l1298_129801

theorem factorize_expression_1 (x y : ℝ) :
  x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_expression_1_l1298_129801


namespace NUMINAMATH_CALUDE_abs_value_condition_l1298_129859

theorem abs_value_condition (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (abs a > 1 ∧ abs b < 1) ∨ (abs a < 1 ∧ abs b > 1) := by
  sorry

end NUMINAMATH_CALUDE_abs_value_condition_l1298_129859


namespace NUMINAMATH_CALUDE_pizza_slices_l1298_129863

/-- The number of slices in a whole pizza -/
def total_slices : ℕ := sorry

/-- The number of slices each person ate -/
def slices_per_person : ℚ := 3/2

/-- The number of people who ate pizza -/
def num_people : ℕ := 2

/-- The number of slices left -/
def slices_left : ℕ := 5

/-- Theorem: The original number of slices in the pizza is 8 -/
theorem pizza_slices : total_slices = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1298_129863


namespace NUMINAMATH_CALUDE_divisors_of_2160_l1298_129848

def n : ℕ := 2160

-- Define the prime factorization of n
axiom n_factorization : n = 2^4 * 3^3 * 5

-- Define the number of positive divisors
def num_divisors (m : ℕ) : ℕ := sorry

-- Define the sum of positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

theorem divisors_of_2160 :
  (num_divisors n = 40) ∧ (sum_divisors n = 7440) := by sorry

end NUMINAMATH_CALUDE_divisors_of_2160_l1298_129848


namespace NUMINAMATH_CALUDE_steve_nickels_l1298_129815

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total value of coins in cents -/
def total_value : ℕ := 70

/-- Proves that Steve is holding 2 nickels -/
theorem steve_nickels :
  ∃ (n : ℕ), 
    (n * nickel_value + (n + 4) * dime_value = total_value) ∧
    (n = 2) := by
  sorry

end NUMINAMATH_CALUDE_steve_nickels_l1298_129815


namespace NUMINAMATH_CALUDE_fraction_repetend_correct_l1298_129822

/-- The repetend of the decimal representation of 7/19 -/
def repetend : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The fraction we're considering -/
def fraction : Rat := 7 / 19

theorem fraction_repetend_correct :
  ∃ (k : Nat), fraction = (k : Rat) / 10^repetend.length + 
    (List.sum (List.zipWith (λ (d i : Nat) => d * 10^(repetend.length - 1 - i)) repetend (List.range repetend.length)) : Rat) / 
    (10^repetend.length - 1) / 19 :=
sorry

end NUMINAMATH_CALUDE_fraction_repetend_correct_l1298_129822


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1298_129878

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the condition for f to have extreme values on ℝ -/
theorem extreme_value_condition (a : ℝ) : 
  (∃ x : ℝ, (f' a x = 0 ∧ ∀ y : ℝ, f' a y = 0 → y = x) → False) ↔ (a > 6 ∨ a < -3) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1298_129878


namespace NUMINAMATH_CALUDE_product_difference_bound_l1298_129804

theorem product_difference_bound (x y a b m ε : ℝ) 
  (ε_pos : ε > 0) (m_pos : m > 0) 
  (h1 : |x - a| < ε / (2 * m))
  (h2 : |y - b| < ε / (2 * |a|))
  (h3 : 0 < y) (h4 : y < m) : 
  |x * y - a * b| < ε := by
sorry

end NUMINAMATH_CALUDE_product_difference_bound_l1298_129804


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l1298_129808

-- Define fuel consumption rates
def highway_consumption_60 : ℝ := 3
def highway_consumption_70 : ℝ := 3.5
def city_consumption_30 : ℝ := 5
def city_consumption_15 : ℝ := 4.5

-- Define driving durations and speeds
def day1_highway_60 : ℝ := 2
def day1_highway_70 : ℝ := 1
def day1_city_30 : ℝ := 4

def day2_highway_70 : ℝ := 3
def day2_city_15 : ℝ := 3
def day2_city_30 : ℝ := 1

def day3_highway_60 : ℝ := 1.5
def day3_city_30 : ℝ := 3
def day3_city_15 : ℝ := 1

-- Theorem statement
theorem total_fuel_consumption :
  let day1 := day1_highway_60 * 60 * highway_consumption_60 +
              day1_highway_70 * 70 * highway_consumption_70 +
              day1_city_30 * 30 * city_consumption_30
  let day2 := day2_highway_70 * 70 * highway_consumption_70 +
              day2_city_15 * 15 * city_consumption_15 +
              day2_city_30 * 30 * city_consumption_30
  let day3 := day3_highway_60 * 60 * highway_consumption_60 +
              day3_city_30 * 30 * city_consumption_30 +
              day3_city_15 * 15 * city_consumption_15
  day1 + day2 + day3 = 3080 := by
  sorry

end NUMINAMATH_CALUDE_total_fuel_consumption_l1298_129808


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l1298_129866

-- Define the two lines
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (-2 + r, 5 - 3*k*r, k*r)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + 2*t, -2*t)

-- Define coplanarity
def coplanar (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t s : ℝ),
    a * (l1 t).1 + b * (l1 t).2.1 + c * (l1 t).2.2 + d =
    a * (l2 s).1 + b * (l2 s).2.1 + c * (l2 s).2.2 + d

-- Theorem statement
theorem lines_coplanar_iff_k_eq_neg_one_or_neg_one_third :
  ∀ k : ℝ, coplanar (line1 · k) line2 ↔ k = -1 ∨ k = -1/3 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l1298_129866


namespace NUMINAMATH_CALUDE_angle_B_measure_max_area_l1298_129816

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given condition a^2 + c^2 = b^2 - ac -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 - t.a * t.c

theorem angle_B_measure (t : Triangle) (h : satisfiesCondition t) :
  t.B = 2 * π / 3 := by sorry

theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.b = 2 * Real.sqrt 3) :
  (t.a * t.c * Real.sin t.B) / 2 ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_max_area_l1298_129816


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1298_129864

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃x < 1, P x) ↔ (∀x < 1, ¬P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬∃x < 1, x^2 + 2*x + 1 ≤ 0) ↔ (∀x < 1, x^2 + 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1298_129864


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l1298_129889

theorem tripled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l1298_129889


namespace NUMINAMATH_CALUDE_sugar_in_recipe_l1298_129811

/-- Given a cake recipe with specific flour requirements and a relation between
    sugar and remaining flour, this theorem proves the amount of sugar needed. -/
theorem sugar_in_recipe (total_flour remaining_flour sugar : ℕ) : 
  total_flour = 14 →
  remaining_flour = total_flour - 4 →
  remaining_flour = sugar + 1 →
  sugar = 9 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l1298_129811


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l1298_129841

/-- Checks if a triple of natural numbers forms a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem not_pythagorean_triple : 
  ¬(isPythagoreanTriple 15 8 19) ∧ 
  (isPythagoreanTriple 6 8 10) ∧ 
  (isPythagoreanTriple 5 12 13) ∧ 
  (isPythagoreanTriple 3 5 4) := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l1298_129841


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1298_129856

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1298_129856


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1298_129831

/-- Represents the colors of the pegs -/
inductive Color
| Red
| Blue
| Green

/-- Represents a position on the triangular board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular board -/
def Board := List Position

/-- Defines a valid triangular board with 3 rows -/
def validBoard : Board :=
  [(Position.mk 1 1),
   (Position.mk 2 1), (Position.mk 2 2),
   (Position.mk 3 1), (Position.mk 3 2), (Position.mk 3 3)]

/-- Represents a peg placement on the board -/
structure Placement :=
  (pos : Position)
  (color : Color)

/-- Checks if a list of placements is valid according to the color restriction rule -/
def isValidPlacement (placements : List Placement) : Bool :=
  sorry

/-- Counts the number of valid arrangements of pegs on the board -/
def countValidArrangements (board : Board) (redPegs bluePegs greenPegs : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  countValidArrangements validBoard 4 3 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1298_129831


namespace NUMINAMATH_CALUDE_digit_sum_l1298_129883

theorem digit_sum (w x y z : ℕ) : 
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 →
  y + w = 11 →
  x + y + 1 = 10 →
  w + z + 1 = 11 →
  w + x + y + z = 20 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_l1298_129883


namespace NUMINAMATH_CALUDE_multiply_power_result_l1298_129892

theorem multiply_power_result : 112 * (5^4) = 70000 := by
  sorry

end NUMINAMATH_CALUDE_multiply_power_result_l1298_129892


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l1298_129850

theorem smaller_integer_problem (a b : ℕ+) : 
  (a : ℕ) + 8 = (b : ℕ) → a * b = 80 → (a : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l1298_129850


namespace NUMINAMATH_CALUDE_sets_properties_l1298_129854

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
def B : Set ℝ := {x | x - 5 < 0}

-- State the theorem
theorem sets_properties :
  (A ∩ B = Icc 4 5) ∧
  (A ∪ B = univ) ∧
  (Aᶜ = Ioo (-1) 4) := by sorry

end NUMINAMATH_CALUDE_sets_properties_l1298_129854


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1298_129803

theorem divisibility_by_five (n : ℕ) : (76 * n^5 + 115 * n^4 + 19 * n) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1298_129803


namespace NUMINAMATH_CALUDE_son_age_proof_l1298_129880

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = 36 →
  4 * son_age = father_age →
  father_age - son_age = 27 →
  son_age = 9 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l1298_129880


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1298_129861

theorem smallest_x_absolute_value_equation : 
  ∀ x : ℝ, |4*x + 12| = 40 → x ≥ -13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1298_129861


namespace NUMINAMATH_CALUDE_andrews_cookies_l1298_129867

/-- Proves that Andrew purchased 3 cookies each day in May --/
theorem andrews_cookies (total_spent : ℕ) (cookie_cost : ℕ) (days_in_may : ℕ) 
  (h1 : total_spent = 1395)
  (h2 : cookie_cost = 15)
  (h3 : days_in_may = 31) :
  total_spent / (cookie_cost * days_in_may) = 3 :=
by
  sorry

#check andrews_cookies

end NUMINAMATH_CALUDE_andrews_cookies_l1298_129867


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1298_129813

-- Define an isosceles triangle with side lengths 3 and 6
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 6 ∧ b = 3 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1298_129813


namespace NUMINAMATH_CALUDE_S_lower_bound_l1298_129828

/-- The least positive integer S(n) such that S(n) ≡ n (mod 2), S(n) ≥ n, 
    and there are no positive integers k, x₁, x₂, ..., xₖ such that 
    n = x₁ + x₂ + ... + xₖ and S(n) = x₁² + x₂² + ... + xₖ² -/
noncomputable def S (n : ℕ) : ℕ := sorry

/-- S(n) grows at least as fast as c * n^(3/2) for some constant c > 0 
    and for all sufficiently large n -/
theorem S_lower_bound :
  ∃ (c : ℝ) (n₀ : ℕ), c > 0 ∧ ∀ n ≥ n₀, (S n : ℝ) ≥ c * n^(3/2) := by sorry

end NUMINAMATH_CALUDE_S_lower_bound_l1298_129828


namespace NUMINAMATH_CALUDE_last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l1298_129809

theorem last_digits_of_powers (n m : ℕ) : 
  n^(n^n) ≡ n^(n^(n^n)) [MOD 10^m] :=
sorry

theorem last_two_digits_of_nine_powers : 
  9^(9^9) ≡ 9^(9^(9^9)) [MOD 100] ∧ 
  9^(9^9) ≡ 99 [MOD 100] :=
sorry

theorem last_six_digits_of_seven_powers : 
  7^(7^(7^7)) ≡ 7^(7^(7^(7^7))) [MOD 1000000] ∧ 
  7^(7^(7^7)) ≡ 999999 [MOD 1000000] :=
sorry

end NUMINAMATH_CALUDE_last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l1298_129809


namespace NUMINAMATH_CALUDE_deposit_percentage_l1298_129805

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 140 →
  remaining = 1260 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l1298_129805


namespace NUMINAMATH_CALUDE_victorias_initial_money_l1298_129838

/-- Theorem: Victoria's Initial Money --/
theorem victorias_initial_money (rice_price wheat_price soda_price : ℕ)
  (rice_quantity wheat_quantity : ℕ) (remaining_balance : ℕ) :
  rice_price = 20 →
  wheat_price = 25 →
  soda_price = 150 →
  rice_quantity = 2 →
  wheat_quantity = 3 →
  remaining_balance = 235 →
  rice_quantity * rice_price + wheat_quantity * wheat_price + soda_price + remaining_balance = 500 :=
by
  sorry

#check victorias_initial_money

end NUMINAMATH_CALUDE_victorias_initial_money_l1298_129838


namespace NUMINAMATH_CALUDE_battery_collection_theorem_l1298_129814

/-- Represents the number of batteries collected by students. -/
structure BatteryCollection where
  jiajia : ℕ
  qiqi : ℕ

/-- Represents the state of battery collection before and after the exchange. -/
structure BatteryExchange where
  initial : BatteryCollection
  final : BatteryCollection

/-- Theorem about battery collection and exchange between Jiajia and Qiqi. -/
theorem battery_collection_theorem (m : ℕ) :
  ∃ (exchange : BatteryExchange),
    -- Initial conditions
    exchange.initial.jiajia = m ∧
    exchange.initial.qiqi = 2 * m - 2 ∧
    -- Condition that Qiqi would have twice as many if she collected two more
    exchange.initial.qiqi + 2 = 2 * exchange.initial.jiajia ∧
    -- Final conditions after Qiqi gives two batteries to Jiajia
    exchange.final.jiajia = exchange.initial.jiajia + 2 ∧
    exchange.final.qiqi = exchange.initial.qiqi - 2 ∧
    -- Prove that Qiqi has m - 6 more batteries than Jiajia after the exchange
    exchange.final.qiqi - exchange.final.jiajia = m - 6 :=
by
  sorry

end NUMINAMATH_CALUDE_battery_collection_theorem_l1298_129814


namespace NUMINAMATH_CALUDE_cara_in_middle_groups_l1298_129825

theorem cara_in_middle_groups (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_in_middle_groups_l1298_129825


namespace NUMINAMATH_CALUDE_different_sum_of_digits_l1298_129881

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Statement: For any natural number N, the sum of digits of N(N-1) is not equal to the sum of digits of (N+1)² -/
theorem different_sum_of_digits (N : ℕ) : 
  sum_of_digits (N * (N - 1)) ≠ sum_of_digits ((N + 1) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_different_sum_of_digits_l1298_129881


namespace NUMINAMATH_CALUDE_carter_school_earnings_l1298_129820

/-- Represents the number of students from each school --/
def students_adams : ℕ := 8
def students_bentley : ℕ := 6
def students_carter : ℕ := 7

/-- Represents the number of days worked by students from each school --/
def days_adams : ℕ := 4
def days_bentley : ℕ := 6
def days_carter : ℕ := 10

/-- Total amount paid for all students' work --/
def total_paid : ℚ := 1020

/-- Theorem stating that the earnings for Carter school students is approximately $517.39 --/
theorem carter_school_earnings : 
  let total_student_days := students_adams * days_adams + students_bentley * days_bentley + students_carter * days_carter
  let daily_wage := total_paid / total_student_days
  let carter_earnings := daily_wage * (students_carter * days_carter)
  ∃ ε > 0, |carter_earnings - 517.39| < ε :=
sorry

end NUMINAMATH_CALUDE_carter_school_earnings_l1298_129820


namespace NUMINAMATH_CALUDE_food_distribution_l1298_129847

/-- Represents the number of days the initial food supply lasts -/
def initial_days : ℕ := 22

/-- Represents the number of days that pass before additional men join -/
def days_passed : ℕ := 2

/-- Represents the number of additional men who join -/
def additional_men : ℕ := 2280

/-- Represents the number of days the food lasts after additional men join -/
def remaining_days : ℕ := 5

/-- Represents the initial number of men -/
def initial_men : ℕ := 760

theorem food_distribution (M : ℕ) :
  M * (initial_days - days_passed) = (M + additional_men) * remaining_days →
  M = initial_men := by sorry

end NUMINAMATH_CALUDE_food_distribution_l1298_129847


namespace NUMINAMATH_CALUDE_bag_probability_l1298_129897

theorem bag_probability (red_balls : ℕ) (total_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 10 → 
  prob_red = 1 / 3 → 
  red_balls + (total_balls - red_balls) = total_balls →
  (red_balls : ℚ) / total_balls = prob_red →
  total_balls - red_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_bag_probability_l1298_129897


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1298_129849

/-- Given that z = (1-mi)/(2+i) is a pure imaginary number, prove that m = 2 --/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 - m * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1298_129849


namespace NUMINAMATH_CALUDE_abc_inequality_l1298_129894

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a + b + c) ≤ a^3 * b + b^3 * c + c^3 * a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1298_129894


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l1298_129824

/-- Given a quadratic function f(x) = 3x^2 - 2x + 5, when shifted 7 units right
    and 3 units up, the resulting function g(x) = ax^2 + bx + c
    satisfies a + b + c = 128 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 5) →
  (∀ x, g x = f (x - 7) + 3) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l1298_129824


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l1298_129887

theorem sally_pokemon_cards 
  (initial_cards : ℕ) 
  (dan_cards : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 27) 
  (h2 : dan_cards = 41) 
  (h3 : total_cards = 88) : 
  total_cards - (initial_cards + dan_cards) = 20 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l1298_129887


namespace NUMINAMATH_CALUDE_jim_travels_two_miles_l1298_129876

def john_distance : ℝ := 15

def jill_distance (john_dist : ℝ) : ℝ := john_dist - 5

def jim_distance (jill_dist : ℝ) : ℝ := 0.20 * jill_dist

theorem jim_travels_two_miles :
  jim_distance (jill_distance john_distance) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_travels_two_miles_l1298_129876


namespace NUMINAMATH_CALUDE_max_good_sequences_75_each_l1298_129839

/-- Represents a string of beads -/
structure BeadString :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Defines a "good" sequence of beads -/
def is_good_sequence (seq : List Char) : Bool :=
  seq.length = 5 ∧ 
  seq.count 'G' = 3 ∧ 
  seq.count 'R' = 1 ∧ 
  seq.count 'B' = 1

/-- Calculates the maximum number of "good" sequences in a bead string -/
def max_good_sequences (s : BeadString) : ℕ :=
  min (s.green * 5 / 3) (min s.red s.blue)

/-- Theorem stating the maximum number of "good" sequences for the given bead string -/
theorem max_good_sequences_75_each (s : BeadString) 
  (h1 : s.blue = 75) (h2 : s.red = 75) (h3 : s.green = 75) : 
  max_good_sequences s = 123 := by
  sorry

end NUMINAMATH_CALUDE_max_good_sequences_75_each_l1298_129839


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l1298_129821

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_third_of_one_fourth_of_one_fifth_of_sixty :
  (1 : ℚ) / 3 * (1 : ℚ) / 4 * (1 : ℚ) / 5 * 60 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_one_third_of_one_fourth_of_one_fifth_of_sixty_l1298_129821


namespace NUMINAMATH_CALUDE_increasing_function_range_increasing_function_and_hyperbola_range_l1298_129818

/-- The function f(x) = x² + (a-1)x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x

/-- The property that f is increasing on (1, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → x < y → f a x < f a y

/-- The equation x² - ay² = 1 represents a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x^2 - a*y^2 = 1

theorem increasing_function_range (a : ℝ) :
  is_increasing_on_interval a → a > -1 :=
sorry

theorem increasing_function_and_hyperbola_range (a : ℝ) :
  is_increasing_on_interval a → is_hyperbola a → a > 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_range_increasing_function_and_hyperbola_range_l1298_129818


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1298_129865

theorem complex_number_quadrant : 
  let z : ℂ := Complex.mk (Real.sin 1) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1298_129865


namespace NUMINAMATH_CALUDE_fifth_term_constant_binomial_l1298_129807

theorem fifth_term_constant_binomial (n : ℕ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * (-2)^4 * k = (Nat.choose n 4) * (-2)^4) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_constant_binomial_l1298_129807


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1298_129827

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1298_129827


namespace NUMINAMATH_CALUDE_sin_cos_product_l1298_129896

theorem sin_cos_product (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 2) : 
  Real.sin α * Real.cos α = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_product_l1298_129896


namespace NUMINAMATH_CALUDE_fractional_equation_range_l1298_129877

theorem fractional_equation_range (x a : ℝ) : 
  (2 * x - a) / (x + 1) = 1 → x > 0 → a > -1 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_range_l1298_129877


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_l1298_129898

/-- The total cost of a fruit purchase given its quantity and price per kg -/
def fruit_cost (quantity : ℕ) (price_per_kg : ℕ) : ℕ :=
  quantity * price_per_kg

/-- The total amount paid for apples and mangoes -/
def total_paid (apple_quantity : ℕ) (apple_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  fruit_cost apple_quantity apple_price + fruit_cost mango_quantity mango_price

/-- Theorem: Given the quantities and prices of apples and mangoes, the total amount paid is 1145 -/
theorem tom_fruit_purchase : total_paid 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_l1298_129898


namespace NUMINAMATH_CALUDE_tile_border_ratio_l1298_129886

theorem tile_border_ratio (p b : ℝ) (h_positive : p > 0 ∧ b > 0) : 
  (225 * p^2) / ((15 * p + 30 * b)^2) = 49/100 → b/p = 4/7 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l1298_129886


namespace NUMINAMATH_CALUDE_bran_leftover_amount_l1298_129837

/-- Represents Bran's financial situation for a semester --/
structure BranFinances where
  tuitionFee : ℝ
  additionalExpenses : ℝ
  hourlyWage : ℝ
  weeklyHours : ℝ
  scholarshipPercentage : ℝ
  semesterMonths : ℕ

/-- Calculates the amount left after paying expenses --/
def calculateLeftoverAmount (finances : BranFinances) : ℝ :=
  let scholarshipAmount := finances.tuitionFee * finances.scholarshipPercentage
  let tuitionAfterScholarship := finances.tuitionFee - scholarshipAmount
  let totalExpenses := tuitionAfterScholarship + finances.additionalExpenses
  let weeklyEarnings := finances.hourlyWage * finances.weeklyHours
  let totalEarnings := weeklyEarnings * (finances.semesterMonths * 4 : ℝ)
  totalEarnings - totalExpenses

/-- Theorem stating that Bran will have $1,481 left after expenses --/
theorem bran_leftover_amount :
  let finances : BranFinances := {
    tuitionFee := 2500,
    additionalExpenses := 600,
    hourlyWage := 18,
    weeklyHours := 12,
    scholarshipPercentage := 0.45,
    semesterMonths := 4
  }
  calculateLeftoverAmount finances = 1481 := by
  sorry

end NUMINAMATH_CALUDE_bran_leftover_amount_l1298_129837


namespace NUMINAMATH_CALUDE_recreation_spending_ratio_l1298_129874

theorem recreation_spending_ratio : 
  ∀ (last_week_wages : ℝ),
  last_week_wages > 0 →
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_ratio_l1298_129874


namespace NUMINAMATH_CALUDE_operation_result_l1298_129832

theorem operation_result (c : ℚ) : 
  2 * ((3 * c + 6 - 5 * c) / 3) = -4/3 * c + 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1298_129832
