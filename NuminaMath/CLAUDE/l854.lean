import Mathlib

namespace NUMINAMATH_CALUDE_difference_girls_boys_l854_85428

/-- The number of male students in village A -/
def male_A : ℕ := 204

/-- The number of female students in village A -/
def female_A : ℕ := 468

/-- The number of male students in village B -/
def male_B : ℕ := 334

/-- The number of female students in village B -/
def female_B : ℕ := 516

/-- The number of male students in village C -/
def male_C : ℕ := 427

/-- The number of female students in village C -/
def female_C : ℕ := 458

/-- The number of male students in village D -/
def male_D : ℕ := 549

/-- The number of female students in village D -/
def female_D : ℕ := 239

/-- The total number of male students in all villages -/
def total_males : ℕ := male_A + male_B + male_C + male_D

/-- The total number of female students in all villages -/
def total_females : ℕ := female_A + female_B + female_C + female_D

/-- Theorem: The difference between the total number of girls and boys in the town is 167 -/
theorem difference_girls_boys : total_females - total_males = 167 := by
  sorry

end NUMINAMATH_CALUDE_difference_girls_boys_l854_85428


namespace NUMINAMATH_CALUDE_larger_root_of_quadratic_l854_85459

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 40 = 0 → x ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_root_of_quadratic_l854_85459


namespace NUMINAMATH_CALUDE_base_seven_sum_of_digits_product_l854_85478

def to_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def from_decimal (n : ℕ) (base : ℕ) : ℕ := sorry

def add_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base + to_decimal b base) base

def mult_base (a b base : ℕ) : ℕ := 
  from_decimal (to_decimal a base * to_decimal b base) base

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem base_seven_sum_of_digits_product : 
  let base := 7
  let a := 35
  let b := add_base 12 16 base
  let product := mult_base a b base
  sum_of_digits product = 7 := by sorry

end NUMINAMATH_CALUDE_base_seven_sum_of_digits_product_l854_85478


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_million_l854_85487

theorem smallest_n_exceeding_million (n : ℕ) : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k < n, (12 : ℝ) ^ ((k * (k + 1) : ℝ) / (2 * 13)) ≤ 1000000) ∧
  (12 : ℝ) ^ ((n * (n + 1) : ℝ) / (2 * 13)) > 1000000 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_million_l854_85487


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l854_85448

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (num_subjects : ℕ)
  (h1 : english = 81)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 76)
  (h6 : num_subjects = 5)
  : ∃ chemistry : ℕ,
    chemistry = average * num_subjects - (english + mathematics + physics + biology) :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l854_85448


namespace NUMINAMATH_CALUDE_integer_root_values_l854_85424

/-- The polynomial for which we're finding integer roots -/
def P (a : ℤ) (x : ℤ) : ℤ := x^3 + 2*x^2 + a*x + 10

/-- The set of possible values for a -/
def A : Set ℤ := {-1210, -185, -26, -13, -11, -10, 65, 790}

/-- Theorem stating that A contains exactly the values of a for which P has an integer root -/
theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, P a x = 0) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l854_85424


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l854_85447

open Set

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l854_85447


namespace NUMINAMATH_CALUDE_sequence_increasing_b_range_l854_85471

theorem sequence_increasing_b_range (b : ℝ) :
  (∀ n : ℕ, n^2 + b*n < (n+1)^2 + b*(n+1)) →
  b > -3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_increasing_b_range_l854_85471


namespace NUMINAMATH_CALUDE_tim_ten_dollar_bills_l854_85419

/-- Given Tim's bill composition and payment requirements, prove he has 6 ten-dollar bills -/
theorem tim_ten_dollar_bills :
  ∀ (x : ℕ),
  (10 * x + 11 * 5 + 17 * 1 = 128) →
  (x + 11 + 17 ≥ 16) →
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_tim_ten_dollar_bills_l854_85419


namespace NUMINAMATH_CALUDE_point_in_region_range_l854_85423

/-- Given a point P(a, 2) within the region represented by 2x + y < 4,
    the range of values for a is (-∞, 1) -/
theorem point_in_region_range (a : ℝ) : 
  (2 * a + 2 < 4) → (∀ x, x < 1 → x ≤ a) ∧ (∀ ε > 0, ∃ y, y ≤ a ∧ y > 1 - ε) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_range_l854_85423


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l854_85425

theorem average_of_c_and_d (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → 
  e = c + 6 → 
  (c + d) / 2 = 47.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l854_85425


namespace NUMINAMATH_CALUDE_goose_survival_fraction_l854_85472

theorem goose_survival_fraction :
  ∀ (total_eggs : ℕ)
    (hatched_fraction : ℚ)
    (first_month_survival_fraction : ℚ)
    (first_year_survivors : ℕ),
  total_eggs = 500 →
  hatched_fraction = 2 / 3 →
  first_month_survival_fraction = 3 / 4 →
  first_year_survivors = 100 →
  (total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction > (first_year_survivors : ℚ) →
  (((total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction - first_year_survivors) /
   ((total_eggs : ℚ) * hatched_fraction * first_month_survival_fraction)) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_goose_survival_fraction_l854_85472


namespace NUMINAMATH_CALUDE_base16_to_base4_C2A_l854_85475

/-- Represents a digit in base 16 --/
inductive Base16Digit
| C | Two | A

/-- Represents a number in base 16 --/
def Base16Number := List Base16Digit

/-- Represents a digit in base 4 --/
inductive Base4Digit
| Zero | One | Two | Three

/-- Represents a number in base 4 --/
def Base4Number := List Base4Digit

/-- Converts a Base16Number to a Base4Number --/
def convertBase16ToBase4 (n : Base16Number) : Base4Number := sorry

/-- The main theorem --/
theorem base16_to_base4_C2A :
  convertBase16ToBase4 [Base16Digit.C, Base16Digit.Two, Base16Digit.A] =
  [Base4Digit.Three, Base4Digit.Zero, Base4Digit.Zero,
   Base4Digit.Two, Base4Digit.Two, Base4Digit.Two] :=
by sorry

end NUMINAMATH_CALUDE_base16_to_base4_C2A_l854_85475


namespace NUMINAMATH_CALUDE_square_difference_204_202_l854_85414

theorem square_difference_204_202 : 204^2 - 202^2 = 812 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_204_202_l854_85414


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l854_85461

/-- Given a man's rowing speed against the stream and his rate in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (rate_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : rate_still_water = 8) :
  rate_still_water + (rate_still_water - speed_against_stream) = 12 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l854_85461


namespace NUMINAMATH_CALUDE_tshirt_production_rate_l854_85443

theorem tshirt_production_rate (rate1 : ℝ) (total : ℕ) (rate2 : ℝ) : 
  rate1 = 12 → total = 15 → rate2 = (120 : ℝ) / ((total : ℝ) - 60 / rate1) → rate2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_production_rate_l854_85443


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l854_85467

theorem floor_ceiling_sum : ⌊(3.999 : ℝ)⌋ + ⌈(4.001 : ℝ)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l854_85467


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l854_85497

/-- A seating arrangement with rows of 7 or 9 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_7 : ℕ

/-- The total number of people seated -/
def total_seated (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 7 * s.rows_of_7

/-- The seating arrangement is valid if it seats exactly 112 people -/
def is_valid (s : SeatingArrangement) : Prop :=
  total_seated s = 112

theorem seating_arrangement_solution :
  ∃ (s : SeatingArrangement), is_valid s ∧ s.rows_of_9 = 7 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l854_85497


namespace NUMINAMATH_CALUDE_tangent_asymptote_implies_m_value_l854_85488

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

-- Define the hyperbola
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 - x^2/m^2 = 1

-- Define the asymptote of the hyperbola
def asymptote_equation (x y m : ℝ) : Prop :=
  y = x/m ∨ y = -x/m

-- Main theorem
theorem tangent_asymptote_implies_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ x y : ℝ, circle_equation x y ∧ 
    asymptote_equation x y m ∧
    hyperbola_equation x y m) →
  m = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_asymptote_implies_m_value_l854_85488


namespace NUMINAMATH_CALUDE_first_valid_year_is_2028_l854_85493

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2020 ∧ sum_of_digits year = 10

theorem first_valid_year_is_2028 :
  ∀ year : ℕ, year < 2028 → ¬(is_valid_year year) ∧ is_valid_year 2028 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2028_l854_85493


namespace NUMINAMATH_CALUDE_square_sum_equals_negative_45_l854_85416

theorem square_sum_equals_negative_45 (x y : ℝ) 
  (h1 : x - 3 * y = 3) 
  (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = -45 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_negative_45_l854_85416


namespace NUMINAMATH_CALUDE_larger_number_proof_l854_85480

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 830) (h3 : x = 22 * y + 2) : x = 794 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l854_85480


namespace NUMINAMATH_CALUDE_gina_coin_value_l854_85401

/-- Calculates the total value of a pile of coins given the total number of coins and the number of dimes. -/
def total_coin_value (total_coins : ℕ) (num_dimes : ℕ) : ℚ :=
  let num_nickels : ℕ := total_coins - num_dimes
  let dime_value : ℚ := 10 / 100
  let nickel_value : ℚ := 5 / 100
  (num_dimes : ℚ) * dime_value + (num_nickels : ℚ) * nickel_value

/-- Proves that given 50 total coins with 14 dimes, the total value is $3.20. -/
theorem gina_coin_value : total_coin_value 50 14 = 32 / 10 := by
  sorry

end NUMINAMATH_CALUDE_gina_coin_value_l854_85401


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l854_85421

theorem x_equation_implies_polynomial_value :
  ∀ x : ℝ, x + 1/x = 2 → x^9 - 5*x^5 + x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l854_85421


namespace NUMINAMATH_CALUDE_repeating_decimal_limit_l854_85485

/-- Define the sequence of partial sums for 0.9999... -/
def partialSum (n : ℕ) : ℚ := 1 - (1 / 10 ^ n)

/-- Theorem: The limit of the sequence of partial sums for 0.9999... is 1 -/
theorem repeating_decimal_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |partialSum n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_limit_l854_85485


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l854_85458

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l854_85458


namespace NUMINAMATH_CALUDE_seating_arrangements_l854_85436

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem seating_arrangements (total_people : ℕ) (restricted_people : ℕ) 
  (h1 : total_people = 10) 
  (h2 : restricted_people = 3) : 
  factorial total_people - 
  (factorial (total_people - restricted_people + 1) * factorial restricted_people + 
   restricted_people * choose (total_people - restricted_people + 1) 1 * 
   factorial (total_people - restricted_people) - 
   restricted_people * (factorial (total_people - restricted_people + 1) * 
   factorial restricted_people)) = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l854_85436


namespace NUMINAMATH_CALUDE_range_f_and_a_condition_l854_85427

/-- The function f(x) = 3|x-1| + |3x+1| -/
def f (x : ℝ) : ℝ := 3 * abs (x - 1) + abs (3 * x + 1)

/-- The function g(x) = |x+2| + |x-a| -/
def g (a : ℝ) (x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

/-- The set A, which is the range of f -/
def A : Set ℝ := Set.range f

/-- The set B, which is the range of g for a given a -/
def B (a : ℝ) : Set ℝ := Set.range (g a)

theorem range_f_and_a_condition (a : ℝ) :
  (A = Set.Ici 4) ∧ (A ∪ B a = B a) → a ∈ Set.Icc (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_f_and_a_condition_l854_85427


namespace NUMINAMATH_CALUDE_inequality_condition_l854_85452

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 4 → (1 + x) * Real.log x + x ≤ x * a) ↔ 
  a ≥ (5 * Real.log 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l854_85452


namespace NUMINAMATH_CALUDE_slot_machine_game_l854_85462

-- Define the slot machine game
def SlotMachine :=
  -- A slot machine that outputs positive integers k with probability 2^(-k)
  Unit

-- Define the winning condition for Ann
def AnnWins (n m : ℕ) : Prop :=
  -- Ann wins if she receives at least n tokens before Drew receives m tokens
  True

-- Define the winning condition for Drew
def DrewWins (n m : ℕ) : Prop :=
  -- Drew wins if he receives m tokens before Ann receives n tokens
  True

-- Define the equal probability of winning
def EqualProbability (n m : ℕ) : Prop :=
  -- The probability of Ann winning equals the probability of Drew winning
  True

-- Theorem statement
theorem slot_machine_game (m : ℕ) (h : m = 2^2018) :
  ∃ n : ℕ, EqualProbability n m ∧ n % 2018 = 2 :=
sorry

end NUMINAMATH_CALUDE_slot_machine_game_l854_85462


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l854_85404

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 9) :
  w / y = 8 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l854_85404


namespace NUMINAMATH_CALUDE_derivative_at_a_l854_85464

theorem derivative_at_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, DifferentiableAt ℝ f x) →
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (a + 2*Δx) - f a) / (3*Δx)) - 1| < ε) →
  deriv f a = 3/2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_a_l854_85464


namespace NUMINAMATH_CALUDE_least_total_cost_equal_quantity_l854_85413

def strawberry_pack_size : ℕ := 6
def strawberry_pack_price : ℕ := 2
def blueberry_pack_size : ℕ := 5
def blueberry_pack_price : ℕ := 3
def cherry_pack_size : ℕ := 8
def cherry_pack_price : ℕ := 4

def least_common_multiple (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def total_cost (lcm : ℕ) : ℕ :=
  (lcm / strawberry_pack_size) * strawberry_pack_price +
  (lcm / blueberry_pack_size) * blueberry_pack_price +
  (lcm / cherry_pack_size) * cherry_pack_price

theorem least_total_cost_equal_quantity :
  total_cost (least_common_multiple strawberry_pack_size blueberry_pack_size cherry_pack_size) = 172 := by
  sorry

end NUMINAMATH_CALUDE_least_total_cost_equal_quantity_l854_85413


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l854_85439

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x, x^3 - 3*x^2 + 4*x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 - 12*x^2 + 49*x - 67 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l854_85439


namespace NUMINAMATH_CALUDE_square_field_area_l854_85441

/-- The area of a square field given the time and speed of a horse running around it -/
theorem square_field_area (time : ℝ) (speed : ℝ) : 
  time = 8 → speed = 12 → (time * speed / 4) ^ 2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l854_85441


namespace NUMINAMATH_CALUDE_second_class_size_l854_85498

theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℚ) 
  (second_class_avg : ℚ) (total_avg : ℚ) :
  first_class_size = 30 →
  first_class_avg = 40 →
  second_class_avg = 80 →
  total_avg = 65 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_class_size_l854_85498


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l854_85402

theorem pasta_preference_ratio (total_students : ℕ) 
  (spaghetti ravioli fettuccine penne : ℕ) : 
  total_students = 800 →
  spaghetti = 300 →
  ravioli = 200 →
  fettuccine = 150 →
  penne = 150 →
  (fettuccine : ℚ) / penne = 1 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l854_85402


namespace NUMINAMATH_CALUDE_max_distinct_values_exists_four_valued_function_l854_85482

/-- A function that assigns a number to each vector in space -/
def VectorFunction (n : ℕ) := (Fin n → ℝ) → ℝ

/-- The property of the vector function as described in the problem -/
def HasMaxProperty (n : ℕ) (f : VectorFunction n) : Prop :=
  ∀ (u v : Fin n → ℝ) (α β : ℝ), 
    f (fun i => α * u i + β * v i) ≤ max (f u) (f v)

/-- The theorem stating that a function with the given property can take at most 4 distinct values -/
theorem max_distinct_values (n : ℕ) (f : VectorFunction n) 
    (h : HasMaxProperty n f) : 
    ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S ≤ 4 := by
  sorry

/-- The theorem stating that there exists a function taking exactly 4 distinct values -/
theorem exists_four_valued_function : 
    ∃ (f : VectorFunction 3), HasMaxProperty 3 f ∧ 
      ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_values_exists_four_valued_function_l854_85482


namespace NUMINAMATH_CALUDE_fifteen_ways_to_assign_teachers_l854_85496

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n_teachers : ℕ) (n_classes : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (Nat.choose n_classes classes_per_teacher * 
   Nat.choose (n_classes - classes_per_teacher) classes_per_teacher * 
   Nat.choose (n_classes - 2 * classes_per_teacher) classes_per_teacher) / 
  Nat.factorial n_teachers

/-- Theorem stating that there are 15 ways to assign 3 teachers to 6 classes -/
theorem fifteen_ways_to_assign_teachers : 
  assign_teachers 3 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_ways_to_assign_teachers_l854_85496


namespace NUMINAMATH_CALUDE_trick_or_treat_duration_l854_85415

/-- The number of hours Tim and his children were out trick or treating -/
def trick_or_treat_hours (num_children : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) (total_treats : ℕ) : ℕ :=
  total_treats / (num_children * houses_per_hour * treats_per_child_per_house)

/-- Theorem stating that Tim and his children were out for 4 hours -/
theorem trick_or_treat_duration :
  trick_or_treat_hours 3 5 3 180 = 4 := by
  sorry

#eval trick_or_treat_hours 3 5 3 180

end NUMINAMATH_CALUDE_trick_or_treat_duration_l854_85415


namespace NUMINAMATH_CALUDE_number_puzzle_l854_85473

theorem number_puzzle (x : ℤ) (h : x + 30 = 55) : x - 23 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l854_85473


namespace NUMINAMATH_CALUDE_dave_tshirts_l854_85451

/-- The number of white T-shirt packs Dave bought -/
def white_packs : ℕ := 3

/-- The number of T-shirts in each white pack -/
def white_per_pack : ℕ := 6

/-- The number of blue T-shirt packs Dave bought -/
def blue_packs : ℕ := 2

/-- The number of T-shirts in each blue pack -/
def blue_per_pack : ℕ := 4

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem dave_tshirts : total_tshirts = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_tshirts_l854_85451


namespace NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l854_85483

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l854_85483


namespace NUMINAMATH_CALUDE_stratified_sample_size_l854_85466

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- 
Theorem: In a stratified sampling scenario with a total population of 750,
where one stratum has 250 members and 5 are sampled from this stratum,
the total sample size is 15.
-/
theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 750) 
  (h2 : s.stratum_size = 250) 
  (h3 : s.stratum_sample = 5) 
  (h4 : s.stratum_sample / s.stratum_size = s.total_sample / s.total_population) : 
  s.total_sample = 15 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l854_85466


namespace NUMINAMATH_CALUDE_inequality_solution_l854_85429

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 + (1 - a) * x - a < 0}

theorem inequality_solution (a : ℝ) :
  solution_set a = 
    if a > -1 then
      {x | -1 < x ∧ x < a}
    else if a < -1 then
      {x | a < x ∧ x < -1}
    else
      ∅ :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l854_85429


namespace NUMINAMATH_CALUDE_rational_sum_l854_85490

theorem rational_sum (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_l854_85490


namespace NUMINAMATH_CALUDE_sandwich_interval_is_40_minutes_l854_85410

/-- Represents the Sandwich Shop's operations -/
structure SandwichShop where
  hours_per_day : ℕ
  peppers_per_day : ℕ
  peppers_per_sandwich : ℕ

/-- Calculates the interval between sandwiches in minutes -/
def sandwich_interval (shop : SandwichShop) : ℕ :=
  let sandwiches_per_day := shop.peppers_per_day / shop.peppers_per_sandwich
  let minutes_per_day := shop.hours_per_day * 60
  minutes_per_day / sandwiches_per_day

/-- The theorem stating the interval between sandwiches is 40 minutes -/
theorem sandwich_interval_is_40_minutes :
  ∀ (shop : SandwichShop),
    shop.hours_per_day = 8 →
    shop.peppers_per_day = 48 →
    shop.peppers_per_sandwich = 4 →
    sandwich_interval shop = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_sandwich_interval_is_40_minutes_l854_85410


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_l854_85417

theorem consecutive_even_numbers (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  y = x + 2 →             -- y is the next consecutive even number
  z = y + 2 →             -- z is the next consecutive even number after y
  x + y + z = x + 18 →    -- sum is 18 greater than the smallest
  z = 10 :=               -- the largest number is 10
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_l854_85417


namespace NUMINAMATH_CALUDE_train_distance_difference_l854_85479

/-- Proves that the difference in distance traveled by two trains meeting each other is 100 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 50) 
  (h2 : v2 = 60)
  (h3 : total_distance = 1100) : 
  (v2 * (total_distance / (v1 + v2))) - (v1 * (total_distance / (v1 + v2))) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_difference_l854_85479


namespace NUMINAMATH_CALUDE_senior_mean_score_l854_85418

theorem senior_mean_score (total_students : ℕ) (overall_mean : ℚ) 
  (senior_count : ℕ) (non_senior_count : ℕ) (senior_mean : ℚ) (non_senior_mean : ℚ) :
  total_students = 120 →
  overall_mean = 110 →
  non_senior_count = 2 * senior_count →
  senior_mean = (3/2) * non_senior_mean →
  senior_count + non_senior_count = total_students →
  (senior_count * senior_mean + non_senior_count * non_senior_mean) / total_students = overall_mean →
  senior_mean = 141.43 := by
sorry

#eval (141.43 : ℚ)

end NUMINAMATH_CALUDE_senior_mean_score_l854_85418


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_42_l854_85422

theorem sum_of_fractions_equals_42 
  (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_42_l854_85422


namespace NUMINAMATH_CALUDE_absolute_value_sum_equals_four_l854_85449

theorem absolute_value_sum_equals_four (x : ℝ) :
  (abs (x - 1) + abs (x - 5) = 4) ↔ (1 ≤ x ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_equals_four_l854_85449


namespace NUMINAMATH_CALUDE_total_memory_space_l854_85477

def morning_songs : ℕ := 10
def afternoon_songs : ℕ := 15
def night_songs : ℕ := 3
def song_size : ℕ := 5

theorem total_memory_space : 
  (morning_songs + afternoon_songs + night_songs) * song_size = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_memory_space_l854_85477


namespace NUMINAMATH_CALUDE_inequality_solution_l854_85492

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 24 * x - 63) / ((3 * x - 4) * (x + 5)) < 4 ↔ 
  x < -5 ∨ x > 4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l854_85492


namespace NUMINAMATH_CALUDE_problem_statement_l854_85435

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l854_85435


namespace NUMINAMATH_CALUDE_crease_length_eq_sqrt_six_over_four_l854_85460

/-- An isosceles right triangle with hypotenuse 1 -/
structure IsoscelesRightTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 1 -/
  hypotenuse_eq_one : hypotenuse = 1

/-- The crease formed by folding one vertex to the other on the hypotenuse -/
def crease_length (t : IsoscelesRightTriangle) : ℝ :=
  sorry  -- Definition of crease length calculation

theorem crease_length_eq_sqrt_six_over_four (t : IsoscelesRightTriangle) :
  crease_length t = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_crease_length_eq_sqrt_six_over_four_l854_85460


namespace NUMINAMATH_CALUDE_problem_statement_l854_85405

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2007 = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l854_85405


namespace NUMINAMATH_CALUDE_least_number_satisfying_conditions_l854_85431

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_11 n ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions : 
  satisfies_conditions 3782 ∧ 
  ∀ m : ℕ, m < 3782 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_number_satisfying_conditions_l854_85431


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l854_85440

/-- The average birth rate in the city (people per 2 seconds) -/
def average_birth_rate : ℝ := sorry

/-- The death rate in the city (people per 2 seconds) -/
def death_rate : ℝ := 2

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 172800

/-- The number of 2-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation :
  average_birth_rate = 6 :=
sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l854_85440


namespace NUMINAMATH_CALUDE_random_selection_properties_l854_85433

/-- Represents the number of female students chosen -/
inductive FemaleCount : Type
  | zero : FemaleCount
  | one : FemaleCount
  | two : FemaleCount

/-- The probability distribution of choosing female students -/
def prob_dist : FemaleCount → ℚ
  | FemaleCount.zero => 1/5
  | FemaleCount.one => 3/5
  | FemaleCount.two => 1/5

/-- The expected value of the number of female students chosen -/
def expected_value : ℚ := 1

/-- The probability of choosing at most one female student -/
def prob_at_most_one : ℚ := 4/5

/-- Theorem stating the properties of the random selection -/
theorem random_selection_properties 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (chosen_count : ℕ) 
  (h1 : male_count = 4) 
  (h2 : female_count = 2) 
  (h3 : chosen_count = 3) :
  (∀ x : FemaleCount, prob_dist x = prob_dist x) ∧
  expected_value = 1 ∧
  prob_at_most_one = 4/5 := by
  sorry

#check random_selection_properties

end NUMINAMATH_CALUDE_random_selection_properties_l854_85433


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l854_85463

/-- The definition of an ellipse in 2D space -/
def is_ellipse (S : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  ∀ M ∈ S, dist M F₁ + dist M F₂ = c

/-- The set of points M satisfying the given condition -/
def trajectory (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M | dist M F₁ + dist M F₂ = 8}

theorem trajectory_is_ellipse (F₁ F₂ : ℝ × ℝ) (h : dist F₁ F₂ = 6) :
  is_ellipse (trajectory F₁ F₂) F₁ F₂ 8 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l854_85463


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l854_85411

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l854_85411


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l854_85408

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) :
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l854_85408


namespace NUMINAMATH_CALUDE_total_pets_l854_85403

theorem total_pets (dogs : ℕ) (fish : ℕ) (cats : ℕ)
  (h1 : dogs = 43)
  (h2 : fish = 72)
  (h3 : cats = 34) :
  dogs + fish + cats = 149 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l854_85403


namespace NUMINAMATH_CALUDE_madeline_water_intake_l854_85432

structure WaterBottle where
  capacity : ℕ

structure Activity where
  name : String
  goal : ℕ
  bottle : WaterBottle
  refills : ℕ

def total_intake (activities : List Activity) : ℕ :=
  activities.foldl (λ acc activity => acc + activity.bottle.capacity * (activity.refills + 1)) 0

def madeline_water_plan : List Activity :=
  [{ name := "Morning yoga", goal := 15, bottle := { capacity := 8 }, refills := 1 },
   { name := "Work", goal := 35, bottle := { capacity := 12 }, refills := 2 },
   { name := "Afternoon jog", goal := 20, bottle := { capacity := 16 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 8 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 16 }, refills := 1 }]

theorem madeline_water_intake :
  total_intake madeline_water_plan = 132 := by
  sorry

end NUMINAMATH_CALUDE_madeline_water_intake_l854_85432


namespace NUMINAMATH_CALUDE_expression_value_l854_85489

theorem expression_value (x y z : ℝ) (hx : x = 2) (hy : y = -3) (hz : z = 1) :
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l854_85489


namespace NUMINAMATH_CALUDE_function_properties_l854_85406

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = -f (2 + x)) 
  (h2 : ∀ x, f (x + 2) = -f x) : 
  (f 0 = 0) ∧ 
  (∀ x, f (x + 4) = f x) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (2 + x) = -f (2 - x)) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l854_85406


namespace NUMINAMATH_CALUDE_soda_cans_purchased_l854_85434

/-- The number of cans of soda that can be purchased for a given amount of money -/
theorem soda_cans_purchased (S Q D : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : D ≥ 0) :
  let cans_per_quarter := S / Q
  let quarters_per_dollar := 4
  let cans_per_dollar := cans_per_quarter * quarters_per_dollar
  cans_per_dollar * D = 4 * D * S / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_purchased_l854_85434


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l854_85444

/-- For a geometric sequence with common ratio q, the condition a_5 * a_6 < a_4^2 is necessary but not sufficient for 0 < q < 1 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence definition
  (a 5 * a 6 < a 4 ^ 2) →       -- given condition
  (∃ q', 0 < q' ∧ q' < 1 ∧ ¬(a 5 * a 6 < a 4 ^ 2 → 0 < q' ∧ q' < 1)) ∧
  (0 < q ∧ q < 1 → a 5 * a 6 < a 4 ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l854_85444


namespace NUMINAMATH_CALUDE_lb_medium_is_control_l854_85499

/-- Represents an experiment setup -/
structure ExperimentSetup where
  name : String
  experimental_medium : String
  control_medium : String

/-- Represents the purpose of a medium in an experiment -/
inductive MediumPurpose
  | Experimental
  | Control

/-- The purpose of preparing LB full-nutrient medium in the given experiment -/
def lb_medium_purpose (setup : ExperimentSetup) : MediumPurpose := sorry

/-- The main theorem stating the purpose of LB full-nutrient medium -/
theorem lb_medium_is_control
  (setup : ExperimentSetup)
  (h1 : setup.name = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (h2 : setup.experimental_medium = "Urea as only nitrogen source")
  (h3 : setup.control_medium = "LB full-nutrient")
  : lb_medium_purpose setup = MediumPurpose.Control := sorry

end NUMINAMATH_CALUDE_lb_medium_is_control_l854_85499


namespace NUMINAMATH_CALUDE_rectangle_to_square_trapezoid_l854_85465

theorem rectangle_to_square_trapezoid (width height area_square : ℝ) (y : ℝ) : 
  width = 16 →
  height = 9 →
  area_square = width * height →
  y = (Real.sqrt area_square) / 2 →
  y = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_trapezoid_l854_85465


namespace NUMINAMATH_CALUDE_trig_identity_l854_85456

theorem trig_identity : 
  Real.sin (130 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (40 * π / 180) * Real.sin (10 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l854_85456


namespace NUMINAMATH_CALUDE_c_rent_share_l854_85400

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℕ) (a_oxen b_oxen c_oxen : ℕ) (a_months b_months c_months : ℕ) : ℕ :=
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  let c_ox_months := c_oxen * c_months
  (total_rent * c_ox_months) / total_ox_months

/-- Theorem stating that c's share of the rent is 45 Rs -/
theorem c_rent_share :
  RentShare 175 10 12 15 7 5 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l854_85400


namespace NUMINAMATH_CALUDE_kim_distance_difference_l854_85420

/-- Represents a person's drive with time and speed -/
structure Drive where
  time : ℝ
  speed : ℝ

/-- Calculates the distance traveled given a Drive -/
def distance (d : Drive) : ℝ := d.time * d.speed

theorem kim_distance_difference (ian : Drive) 
  (h1 : distance { time := ian.time + 2, speed := ian.speed + 5 } = distance ian + 100)
  (h2 : 2 * ian.speed + 5 * ian.time = 90) :
  distance { time := ian.time + 4, speed := ian.speed + 15 } = distance ian + 240 := by
  sorry

end NUMINAMATH_CALUDE_kim_distance_difference_l854_85420


namespace NUMINAMATH_CALUDE_converse_square_right_angles_false_l854_85430

-- Define a quadrilateral
structure Quadrilateral :=
  (is_right_angled : Bool)
  (is_square : Bool)

-- Define the property that all angles are right angles
def all_angles_right (q : Quadrilateral) : Prop :=
  q.is_right_angled = true

-- Define the property of being a square
def is_square (q : Quadrilateral) : Prop :=
  q.is_square = true

-- Theorem: The converse of "All four angles of a square are right angles" is false
theorem converse_square_right_angles_false :
  ¬ (∀ q : Quadrilateral, all_angles_right q → is_square q) :=
by sorry

end NUMINAMATH_CALUDE_converse_square_right_angles_false_l854_85430


namespace NUMINAMATH_CALUDE_store_sale_revenue_l854_85445

/-- Calculates the amount left after a store's inventory sale --/
theorem store_sale_revenue (total_items : ℕ) (category_a_items : ℕ) (category_b_items : ℕ) (category_c_items : ℕ)
  (price_a : ℝ) (price_b : ℝ) (price_c : ℝ)
  (discount_a : ℝ) (discount_b : ℝ) (discount_c : ℝ)
  (sales_percent_a : ℝ) (sales_percent_b : ℝ) (sales_percent_c : ℝ)
  (return_rate : ℝ) (advertising_cost : ℝ) (creditors_amount : ℝ) :
  total_items = category_a_items + category_b_items + category_c_items →
  category_a_items = 1000 →
  category_b_items = 700 →
  category_c_items = 300 →
  price_a = 50 →
  price_b = 75 →
  price_c = 100 →
  discount_a = 0.8 →
  discount_b = 0.7 →
  discount_c = 0.6 →
  sales_percent_a = 0.85 →
  sales_percent_b = 0.75 →
  sales_percent_c = 0.9 →
  return_rate = 0.03 →
  advertising_cost = 2000 →
  creditors_amount = 15000 →
  ∃ (revenue : ℝ), revenue = 13172.50 ∧ 
    revenue = (category_a_items * sales_percent_a * price_a * (1 - discount_a) * (1 - return_rate) +
               category_b_items * sales_percent_b * price_b * (1 - discount_b) * (1 - return_rate) +
               category_c_items * sales_percent_c * price_c * (1 - discount_c) * (1 - return_rate)) -
              advertising_cost - creditors_amount :=
by
  sorry


end NUMINAMATH_CALUDE_store_sale_revenue_l854_85445


namespace NUMINAMATH_CALUDE_opposite_of_five_l854_85481

theorem opposite_of_five : -(5 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_five_l854_85481


namespace NUMINAMATH_CALUDE_ellipse_m_range_l854_85450

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_equation m → m ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l854_85450


namespace NUMINAMATH_CALUDE_rectangle_division_l854_85455

theorem rectangle_division (P : ℝ) (p₁ p₂ : ℝ) 
  (h_P : P = 76) 
  (h_p₁ : p₁ = 40) 
  (h_p₂ : p₂ = 52) : 
  ∃ (l w : ℝ), l = 30 ∧ w = 8 ∧ 2 * (l + w) = P ∧ 
  ∃ (a : ℝ), p₁ + p₂ = P + 2 * a ∧ l = a + w :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_l854_85455


namespace NUMINAMATH_CALUDE_hard_round_points_is_five_l854_85409

/-- A math contest with three rounds -/
structure MathContest where
  easy_correct : ℕ
  easy_points : ℕ
  avg_correct : ℕ
  avg_points : ℕ
  hard_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest -/
def kim_contest : MathContest := {
  easy_correct := 6
  easy_points := 2
  avg_correct := 2
  avg_points := 3
  hard_correct := 4
  total_points := 38
}

/-- Calculate the points per correct answer in the hard round -/
def hard_round_points (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_correct * contest.easy_points + contest.avg_correct * contest.avg_points)) / contest.hard_correct

/-- Theorem: The points per correct answer in the hard round is 5 -/
theorem hard_round_points_is_five : hard_round_points kim_contest = 5 := by
  sorry


end NUMINAMATH_CALUDE_hard_round_points_is_five_l854_85409


namespace NUMINAMATH_CALUDE_log_product_equals_one_l854_85407

theorem log_product_equals_one :
  Real.log 5 / Real.log 2 * Real.log 2 / Real.log 3 * Real.log 3 / Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l854_85407


namespace NUMINAMATH_CALUDE_zero_after_double_one_l854_85470

/-- Represents a binary sequence -/
def BinarySequence := List Bool

/-- Counts the occurrences of a given subsequence in a binary sequence -/
def count_subsequence (seq : BinarySequence) (subseq : BinarySequence) : Nat :=
  sorry

/-- The main theorem -/
theorem zero_after_double_one (seq : BinarySequence) : 
  (count_subsequence seq [false, true] = 16) →
  (count_subsequence seq [true, false] = 15) →
  (count_subsequence seq [false, true, false] = 8) →
  (count_subsequence seq [true, true, false] = 7) :=
sorry

end NUMINAMATH_CALUDE_zero_after_double_one_l854_85470


namespace NUMINAMATH_CALUDE_inverse_j_minus_j_inv_l854_85446

-- Define the complex number i
def i : ℂ := Complex.I

-- Define j in terms of i
def j : ℂ := i + 1

-- Theorem statement
theorem inverse_j_minus_j_inv :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_j_minus_j_inv_l854_85446


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l854_85495

theorem min_value_quadratic (x y : ℝ) : 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 ≥ -10 := by
  sorry

theorem min_value_quadratic_achieved : 
  ∃ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x + 2 * y - 5 = -10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l854_85495


namespace NUMINAMATH_CALUDE_unique_function_property_l854_85442

theorem unique_function_property (f : ℤ → ℤ) 
  (h1 : f 0 = 1)
  (h2 : ∀ (n : ℕ), f (f n) = n)
  (h3 : ∀ (n : ℕ), f (f (n + 2) + 2) = n) :
  ∀ (n : ℤ), f n = 1 - n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l854_85442


namespace NUMINAMATH_CALUDE_town_population_growth_l854_85491

/-- Given an initial population and a final population after a certain number of years,
    calculate the average percent increase of population per year. -/
def average_percent_increase (initial_population final_population : ℕ) (years : ℕ) : ℚ :=
  ((final_population - initial_population : ℚ) / initial_population / years) * 100

/-- Theorem: The average percent increase of population per year for a town
    that grew from 175000 to 297500 in 10 years is 7%. -/
theorem town_population_growth : average_percent_increase 175000 297500 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l854_85491


namespace NUMINAMATH_CALUDE_math_city_intersections_l854_85457

/-- Represents a city with a given number of streets -/
structure City where
  numStreets : ℕ
  noParallel : Bool
  noTripleIntersections : Bool

/-- Calculates the number of intersections in a city -/
def numIntersections (c : City) : ℕ :=
  (c.numStreets.pred * c.numStreets.pred) / 2

theorem math_city_intersections (c : City) 
  (h1 : c.numStreets = 10)
  (h2 : c.noParallel = true)
  (h3 : c.noTripleIntersections = true) :
  numIntersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l854_85457


namespace NUMINAMATH_CALUDE_arccos_cos_2x_solution_set_l854_85454

theorem arccos_cos_2x_solution_set :
  ∀ x : ℝ, (Real.arccos (Real.cos (2 * x)) = x) ↔ 
    (∃ k : ℤ, x = 2 * k * π ∨ x = 2 * π / 3 + 2 * k * π ∨ x = -(2 * π / 3) + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_2x_solution_set_l854_85454


namespace NUMINAMATH_CALUDE_jogger_train_distance_l854_85469

/-- Calculates the distance between a jogger and a train engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600)  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600)  -- 45 km/hr in m/s
  (h3 : train_length = 120)              -- 120 meters
  (h4 : passing_time = 31)               -- 31 seconds
  : ∃ (distance : ℝ), distance = 190 ∧ distance = (train_speed - jogger_speed) * passing_time - train_length :=
by
  sorry


end NUMINAMATH_CALUDE_jogger_train_distance_l854_85469


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l854_85438

/-- Given a geometric series with first term a and common ratio r,
    if the sum of the series is 20 and the sum of terms involving odd powers of r is 8,
    then r = 1/4 -/
theorem geometric_series_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 20)
  (h2 : a * r / (1 - r^2) = 8) :
  r = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l854_85438


namespace NUMINAMATH_CALUDE_abc_inequality_l854_85412

theorem abc_inequality (a b c : ℝ) (h : a^2*b*c + a*b^2*c + a*b*c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l854_85412


namespace NUMINAMATH_CALUDE_range_of_a_l854_85453

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l854_85453


namespace NUMINAMATH_CALUDE_cubic_quadratic_equation_solution_l854_85486

theorem cubic_quadratic_equation_solution :
  ∃! (y : ℝ), y ≠ 0 ∧ (8 * y)^3 = (16 * y)^2 ∧ y = 1/2 := by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_equation_solution_l854_85486


namespace NUMINAMATH_CALUDE_sock_matching_probability_l854_85474

def total_socks : ℕ := 8
def black_socks : ℕ := 6
def white_socks : ℕ := 2

def total_combinations : ℕ := total_socks.choose 2
def matching_combinations : ℕ := black_socks.choose 2 + 1

theorem sock_matching_probability :
  (matching_combinations : ℚ) / total_combinations = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_sock_matching_probability_l854_85474


namespace NUMINAMATH_CALUDE_complement_of_M_union_N_in_U_l854_85468

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_of_M_union_N_in_U :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_union_N_in_U_l854_85468


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l854_85437

-- Define the number of houses on the block
def num_houses : ℕ := 6

-- Define the total number of junk mail pieces
def total_junk_mail : ℕ := 24

-- Define the function to calculate junk mail per house
def junk_mail_per_house (houses : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / houses

-- Theorem statement
theorem junk_mail_distribution :
  junk_mail_per_house num_houses total_junk_mail = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l854_85437


namespace NUMINAMATH_CALUDE_non_square_sequence_2003_l854_85484

/-- The sequence of positive integers with perfect squares removed -/
def non_square_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the non-square sequence -/
def term_2003 : ℕ := non_square_sequence 2003

theorem non_square_sequence_2003 : term_2003 = 2048 := by sorry

end NUMINAMATH_CALUDE_non_square_sequence_2003_l854_85484


namespace NUMINAMATH_CALUDE_total_oil_leak_l854_85494

def initial_leak_A : ℕ := 6522
def initial_leak_B : ℕ := 3894
def initial_leak_C : ℕ := 1421

def leak_rate_A : ℕ := 257
def leak_rate_B : ℕ := 182
def leak_rate_C : ℕ := 97

def repair_time_A : ℕ := 20
def repair_time_B : ℕ := 15
def repair_time_C : ℕ := 12

theorem total_oil_leak :
  initial_leak_A + initial_leak_B + initial_leak_C +
  leak_rate_A * repair_time_A + leak_rate_B * repair_time_B + leak_rate_C * repair_time_C = 20871 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_leak_l854_85494


namespace NUMINAMATH_CALUDE_skateboard_cost_l854_85476

theorem skateboard_cost (toy_cars_cost toy_trucks_cost total_toys_cost : ℚ)
  (h1 : toy_cars_cost = 14.88)
  (h2 : toy_trucks_cost = 5.86)
  (h3 : total_toys_cost = 25.62) :
  total_toys_cost - (toy_cars_cost + toy_trucks_cost) = 4.88 := by
sorry

end NUMINAMATH_CALUDE_skateboard_cost_l854_85476


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l854_85426

def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
def B (a b : ℝ) : Set ℝ := {x | |x - b| < a}

theorem intersection_nonempty_implies_b_range :
  (∀ b : ℝ, (A ∩ B 1 b).Nonempty) →
  ∀ b : ℝ, -2 < b ∧ b < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l854_85426
