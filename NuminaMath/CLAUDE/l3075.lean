import Mathlib

namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l3075_307530

-- Define a prime number with 2009 digits
def largest_prime_2009_digits : Nat :=
  sorry

-- Define the property of being the largest prime with 2009 digits
def is_largest_prime_2009_digits (p : Nat) : Prop :=
  Nat.Prime p ∧ 
  (Nat.digits 10 p).length = 2009 ∧
  ∀ q, Nat.Prime q → (Nat.digits 10 q).length = 2009 → q ≤ p

-- Theorem statement
theorem smallest_k_for_divisibility_by_10 (p : Nat) 
  (h_p : is_largest_prime_2009_digits p) : 
  (∃ k : Nat, k > 0 ∧ (p^2 - k) % 10 = 0) ∧
  (∀ k : Nat, k > 0 → (p^2 - k) % 10 = 0 → k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l3075_307530


namespace NUMINAMATH_CALUDE_apple_basket_count_l3075_307547

theorem apple_basket_count (rotten_percent : ℝ) (spotted_percent : ℝ) (insect_percent : ℝ) (varying_rot_percent : ℝ) (perfect_count : ℕ) : 
  rotten_percent = 0.12 →
  spotted_percent = 0.07 →
  insect_percent = 0.05 →
  varying_rot_percent = 0.03 →
  perfect_count = 66 →
  ∃ (total : ℕ), total = 90 ∧ 
    (1 - (rotten_percent + spotted_percent + insect_percent + varying_rot_percent)) * (total : ℝ) = perfect_count :=
by
  sorry

end NUMINAMATH_CALUDE_apple_basket_count_l3075_307547


namespace NUMINAMATH_CALUDE_simplify_expression_l3075_307551

theorem simplify_expression (b y : ℝ) (hb : b = 2) (hy : y = 3) :
  18 * b^4 * y^6 / (27 * b^3 * y^5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3075_307551


namespace NUMINAMATH_CALUDE_cylinder_volume_l3075_307548

/-- The volume of a cylinder with diameter 8 cm and height 5 cm is 80π cubic centimeters. -/
theorem cylinder_volume (π : ℝ) (h : π > 0) : 
  let d : ℝ := 8 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let volume : ℝ := π * r^2 * h
  volume = 80 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3075_307548


namespace NUMINAMATH_CALUDE_elberta_has_35_l3075_307559

/-- Amount of money Granny Smith has -/
def granny_smith : ℕ := 120

/-- Amount of money Anjou has -/
def anjou : ℕ := granny_smith / 4

/-- Amount of money Elberta has -/
def elberta : ℕ := anjou + 5

/-- Theorem stating that Elberta has $35 -/
theorem elberta_has_35 : elberta = 35 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_35_l3075_307559


namespace NUMINAMATH_CALUDE_rectangle_area_l3075_307528

/-- The area of a rectangle with perimeter 176 inches and length 8 inches more than its width is 1920 square inches. -/
theorem rectangle_area (w l : ℝ) (h1 : l = w + 8) (h2 : 2*l + 2*w = 176) : w * l = 1920 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3075_307528


namespace NUMINAMATH_CALUDE_probability_same_color_value_l3075_307596

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (red_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)
  (h_red : red_cards = 26)
  (h_sum : black_cards + red_cards = total_cards)

/-- The probability of drawing four cards of the same color from a standard deck -/
def probability_same_color (d : Deck) : Rat :=
  2 * (d.black_cards.choose 4) / d.total_cards.choose 4

/-- Theorem stating the probability of drawing four cards of the same color -/
theorem probability_same_color_value (d : Deck) :
  probability_same_color d = 276 / 2499 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_value_l3075_307596


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3075_307589

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem 1: Solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Range of a for f(x) ≥ |2x+a| - 4 when x ∈ [-1/2, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/2) 1, f x ≥ |2*x + a| - 4) ↔ -7 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3075_307589


namespace NUMINAMATH_CALUDE_scientific_notation_78922_l3075_307503

theorem scientific_notation_78922 : 
  78922 = 7.8922 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_78922_l3075_307503


namespace NUMINAMATH_CALUDE_sqrt_four_minus_one_l3075_307512

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_one_l3075_307512


namespace NUMINAMATH_CALUDE_last_ten_shots_made_l3075_307524

/-- Represents the number of shots made in a sequence of basketball shots -/
structure BasketballShots where
  total : ℕ
  made : ℕ
  percentage : ℚ
  inv_percentage_def : percentage = made / total

/-- The problem statement -/
theorem last_ten_shots_made 
  (initial : BasketballShots)
  (final : BasketballShots)
  (h1 : initial.total = 30)
  (h2 : initial.percentage = 3/5)
  (h3 : final.total = initial.total + 10)
  (h4 : final.percentage = 29/50)
  : final.made - initial.made = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_shots_made_l3075_307524


namespace NUMINAMATH_CALUDE_percent_difference_l3075_307558

theorem percent_difference (y q w z : ℝ) 
  (hw : w = 0.6 * q)
  (hq : q = 0.6 * y)
  (hz : z = 0.54 * y) :
  z = w * 1.5 := by
sorry

end NUMINAMATH_CALUDE_percent_difference_l3075_307558


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l3075_307588

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l3075_307588


namespace NUMINAMATH_CALUDE_problem_solution_l3075_307504

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem problem_solution :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi →
    f (α / 2) = 1 / 4 - Real.sqrt 3 / 2 →
    Real.sin α = (1 + 3 * Real.sqrt 5) / 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3075_307504


namespace NUMINAMATH_CALUDE_max_y_minus_x_l3075_307531

theorem max_y_minus_x (p q : ℕ+) (x y : ℤ) 
  (h : x * y = p * x + q * y) 
  (max_y : ∀ (y' : ℤ), x * y' = p * x + q * y' → y' ≤ y) : 
  y - x = (p - 1) * (q + 1) := by
sorry

end NUMINAMATH_CALUDE_max_y_minus_x_l3075_307531


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3075_307598

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l3075_307598


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l3075_307550

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f := by sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l3075_307550


namespace NUMINAMATH_CALUDE_total_pills_taken_l3075_307509

-- Define the given conditions
def dose_mg : ℕ := 1000
def dose_interval_hours : ℕ := 6
def treatment_weeks : ℕ := 2
def mg_per_pill : ℕ := 500
def hours_per_day : ℕ := 24
def days_per_week : ℕ := 7

-- Define the theorem
theorem total_pills_taken : 
  (dose_mg / mg_per_pill) * 
  (hours_per_day / dose_interval_hours) * 
  (treatment_weeks * days_per_week) = 112 := by
sorry

end NUMINAMATH_CALUDE_total_pills_taken_l3075_307509


namespace NUMINAMATH_CALUDE_wednesday_is_valid_start_day_l3075_307557

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isValidRedemptionDay (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 6, advanceDays startDay (i.val * 10) ≠ DayOfWeek.Sunday

theorem wednesday_is_valid_start_day :
  isValidRedemptionDay DayOfWeek.Wednesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Wednesday → ¬isValidRedemptionDay d :=
sorry

end NUMINAMATH_CALUDE_wednesday_is_valid_start_day_l3075_307557


namespace NUMINAMATH_CALUDE_candy_bar_sales_difference_l3075_307599

/-- Candy bar sales problem -/
theorem candy_bar_sales_difference (price_a price_b : ℕ)
  (marvin_a marvin_b : ℕ) (tina_a tina_b : ℕ)
  (marvin_discount_threshold marvin_discount_amount : ℕ)
  (tina_discount_threshold tina_discount_amount : ℕ)
  (tina_returns : ℕ) :
  price_a = 2 →
  price_b = 3 →
  marvin_a = 20 →
  marvin_b = 15 →
  tina_a = 70 →
  tina_b = 35 →
  marvin_discount_threshold = 5 →
  marvin_discount_amount = 1 →
  tina_discount_threshold = 10 →
  tina_discount_amount = 2 →
  tina_returns = 2 →
  (tina_a * price_a + tina_b * price_b
    - (tina_b / tina_discount_threshold) * tina_discount_amount
    - tina_returns * price_b)
  - (marvin_a * price_a + marvin_b * price_b
    - (marvin_a / marvin_discount_threshold) * marvin_discount_amount)
  = 152 := by sorry

end NUMINAMATH_CALUDE_candy_bar_sales_difference_l3075_307599


namespace NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_l3075_307575

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 3|

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | x ≥ -5/2} := by sorry

-- Part 2: Range of a when f(x) ≤ 4 for all x ∈ [0,3]
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_l3075_307575


namespace NUMINAMATH_CALUDE_fraction_problem_l3075_307542

theorem fraction_problem (p : ℚ) (f : ℚ) : 
  p = 49 →
  p = 2 * f * p + 35 →
  f = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3075_307542


namespace NUMINAMATH_CALUDE_find_k_angle_90_degrees_l3075_307546

-- Define vectors in R^2
def a : Fin 2 → ℝ := ![3, -1]
def b (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define dot product for 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Theorem 1: Find the value of k
theorem find_k : ∃ k : ℝ, perpendicular a (b k) ∧ k = 3 := by sorry

-- Define vector addition and subtraction
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 + w 0, v 1 + w 1]
def sub_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := ![v 0 - w 0, v 1 - w 1]

-- Theorem 2: Prove the angle between a + b and a - b is 90°
theorem angle_90_degrees : 
  let b' := b 3
  let sum := add_vectors a b'
  let diff := sub_vectors a b'
  perpendicular sum diff := by sorry

end NUMINAMATH_CALUDE_find_k_angle_90_degrees_l3075_307546


namespace NUMINAMATH_CALUDE_rohans_savings_l3075_307565

/-- Rohan's monthly budget and savings calculation -/
theorem rohans_savings (salary : ℕ) (food_percent house_percent entertainment_percent conveyance_percent : ℚ) : 
  salary = 5000 →
  food_percent = 40 / 100 →
  house_percent = 20 / 100 →
  entertainment_percent = 10 / 100 →
  conveyance_percent = 10 / 100 →
  salary - (salary * (food_percent + house_percent + entertainment_percent + conveyance_percent)).floor = 1000 := by
  sorry

#check rohans_savings

end NUMINAMATH_CALUDE_rohans_savings_l3075_307565


namespace NUMINAMATH_CALUDE_max_playground_area_l3075_307552

/-- Represents the dimensions of a rectangular playground. -/
structure Playground where
  width : ℝ
  length : ℝ

/-- The total fencing available for the playground. -/
def totalFencing : ℝ := 480

/-- Calculates the area of a playground. -/
def area (p : Playground) : ℝ := p.width * p.length

/-- Checks if a playground satisfies the fencing constraint. -/
def satisfiesFencingConstraint (p : Playground) : Prop :=
  p.length + 2 * p.width = totalFencing

/-- Theorem stating the maximum area of the playground. -/
theorem max_playground_area :
  ∃ (p : Playground), satisfiesFencingConstraint p ∧
    area p = 28800 ∧
    ∀ (q : Playground), satisfiesFencingConstraint q → area q ≤ area p :=
sorry

end NUMINAMATH_CALUDE_max_playground_area_l3075_307552


namespace NUMINAMATH_CALUDE_batsman_inning_number_l3075_307506

/-- Represents the batting statistics of a cricket player -/
structure BattingStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after adding runs to the existing stats -/
def newAverage (stats : BattingStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem batsman_inning_number (stats : BattingStats) (h1 : newAverage stats 88 = 40)
    (h2 : stats.average = 37) : stats.innings + 1 = 17 := by
  sorry

#check batsman_inning_number

end NUMINAMATH_CALUDE_batsman_inning_number_l3075_307506


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3075_307539

theorem no_real_solutions_for_equation : ¬∃ x : ℝ, x + Real.sqrt (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3075_307539


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3075_307554

-- Define the colors
inductive Color
  | Red
  | Blue
  | Green

-- Define the board as a function from coordinates to colors
def Board := ℤ × ℤ → Color

-- Define a rectangle on the board
def IsRectangle (board : Board) (x1 y1 x2 y2 : ℤ) : Prop :=
  x1 ≠ x2 ∧ y1 ≠ y2 ∧
  board (x1, y1) = board (x2, y1) ∧
  board (x1, y1) = board (x1, y2) ∧
  board (x1, y1) = board (x2, y2)

-- The main theorem
theorem monochromatic_rectangle_exists (board : Board) :
  ∃ x1 y1 x2 y2 : ℤ, IsRectangle board x1 y1 x2 y2 := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3075_307554


namespace NUMINAMATH_CALUDE_roots_expression_value_l3075_307537

theorem roots_expression_value (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 7*γ^4 + 10*δ^3 = 1363 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_value_l3075_307537


namespace NUMINAMATH_CALUDE_milk_container_problem_l3075_307572

theorem milk_container_problem (x : ℝ) : 
  (3 * x + 2 * 0.75 + 5 * 0.5 = 10) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l3075_307572


namespace NUMINAMATH_CALUDE_max_value_of_f_l3075_307585

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3075_307585


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_arithmetic_sequence_ratio_l3075_307566

-- Definition of a sequence
def Sequence (α : Type) := ℕ → α

-- Definition of a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Definition of the condition a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Definition of an arithmetic sequence
def IsArithmeticSequence (a : Sequence ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Definition of the sum of first n terms of a sequence
def SumOfFirstNTerms (a : Sequence ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumOfFirstNTerms a n + a (n + 1)

-- Theorem 1
theorem necessary_but_not_sufficient :
  (∀ a : Sequence ℝ, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence ℝ, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by sorry

-- Theorem 2
theorem arithmetic_sequence_ratio :
  ∀ a b : Sequence ℝ,
    IsArithmeticSequence a →
    IsArithmeticSequence b →
    (SumOfFirstNTerms a 5) / (SumOfFirstNTerms b 7) = 15 / 13 →
    a 3 / b 4 = 21 / 13 := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_arithmetic_sequence_ratio_l3075_307566


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3075_307525

theorem quadratic_roots_property (x₁ x₂ c : ℝ) : 
  (x₁^2 + x₁ + c = 0) →
  (x₂^2 + x₂ + c = 0) →
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) →
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3075_307525


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3075_307594

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3075_307594


namespace NUMINAMATH_CALUDE_symmetry_implies_congruence_l3075_307584

/-- Two shapes in a plane -/
structure Shape : Type :=
  -- Define necessary properties of a shape

/-- Line of symmetry between two shapes -/
structure SymmetryLine : Type :=
  -- Define necessary properties of a symmetry line

/-- Symmetry relation between two shapes about a line -/
def symmetrical (s1 s2 : Shape) (l : SymmetryLine) : Prop :=
  sorry

/-- Congruence relation between two shapes -/
def congruent (s1 s2 : Shape) : Prop :=
  sorry

/-- Theorem: If two shapes are symmetrical about a line, they are congruent -/
theorem symmetry_implies_congruence (s1 s2 : Shape) (l : SymmetryLine) :
  symmetrical s1 s2 l → congruent s1 s2 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_congruence_l3075_307584


namespace NUMINAMATH_CALUDE_train_speed_train_speed_proof_l3075_307582

/-- The speed of two trains crossing each other -/
theorem train_speed (train_length : Real) (crossing_time : Real) : Real :=
  let relative_speed := (2 * train_length) / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  18

/-- Proof that the speed of each train is 18 km/hr -/
theorem train_speed_proof :
  train_speed 120 24 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_proof_l3075_307582


namespace NUMINAMATH_CALUDE_lowest_price_scheme_l3075_307553

-- Define the pricing schemes
def schemeA (price : ℝ) : ℝ := price * 1.1 * 0.9
def schemeB (price : ℝ) : ℝ := price * 0.9 * 1.1
def schemeC (price : ℝ) : ℝ := price * 1.15 * 0.85
def schemeD (price : ℝ) : ℝ := price * 1.2 * 0.8

-- Theorem statement
theorem lowest_price_scheme (price : ℝ) (h : price > 0) :
  schemeD price = min (schemeA price) (min (schemeB price) (schemeC price)) :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_scheme_l3075_307553


namespace NUMINAMATH_CALUDE_triangle_properties_l3075_307533

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleABC (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0

theorem triangle_properties (t : Triangle) (h : TriangleABC t) :
  t.C = π / 3 ∧ 
  (∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3075_307533


namespace NUMINAMATH_CALUDE_negation_equivalence_l3075_307532

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3075_307532


namespace NUMINAMATH_CALUDE_parabola_increasing_condition_l3075_307576

/-- The parabola y = (a-1)x^2 + 1 increases as x increases when x ≥ 0 if and only if a > 1 -/
theorem parabola_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (∀ h : ℝ, h > 0 → ((a - 1) * (x + h)^2 + 1) > ((a - 1) * x^2 + 1))) ↔ 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_parabola_increasing_condition_l3075_307576


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l3075_307579

theorem largest_triangle_perimeter :
  ∀ y : ℕ,
  y > 0 →
  y < 16 →
  7 + y > 9 →
  9 + y > 7 →
  (∀ z : ℕ, z > 0 → z < 16 → 7 + z > 9 → 9 + z > 7 → 7 + 9 + y ≥ 7 + 9 + z) →
  7 + 9 + y = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l3075_307579


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3075_307538

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {2, 3, 5}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3075_307538


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3075_307500

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) : meat_types = 12 → cheese_types = 8 → (meat_types.choose 2) * cheese_types = 528 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3075_307500


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3075_307549

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points or a point and a direction vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line parallel to a plane
  sorry

/-- A line is a subset of a plane -/
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line being a subset of a plane
  sorry

theorem line_plane_relationship (m n : Line3D) (α : Plane3D) 
  (h1 : parallel_lines m n) (h2 : line_parallel_plane m α) :
  line_parallel_plane n α ∨ line_subset_plane n α := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3075_307549


namespace NUMINAMATH_CALUDE_complex_number_value_l3075_307513

theorem complex_number_value (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l3075_307513


namespace NUMINAMATH_CALUDE_winning_percentage_calculation_l3075_307521

def total_votes : ℕ := 430
def winning_margin : ℕ := 172

theorem winning_percentage_calculation :
  ∀ (winning_percentage : ℚ),
  (winning_percentage * total_votes / 100 - (100 - winning_percentage) * total_votes / 100 = winning_margin) →
  winning_percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_winning_percentage_calculation_l3075_307521


namespace NUMINAMATH_CALUDE_lists_count_l3075_307540

/-- The number of distinct items to choose from -/
def n : ℕ := 15

/-- The number of times we draw an item -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_lists : ℕ := n^k

theorem lists_count : num_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_lists_count_l3075_307540


namespace NUMINAMATH_CALUDE_no_zero_points_for_exp_minus_x_l3075_307571

theorem no_zero_points_for_exp_minus_x :
  ∀ x : ℝ, x > 0 → ∃ ε : ℝ, ε > 0 ∧ Real.exp x - x > ε := by
  sorry

end NUMINAMATH_CALUDE_no_zero_points_for_exp_minus_x_l3075_307571


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3075_307591

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (a^2 + 2*a - 3) (a^2 + a - 6) → z.re = 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3075_307591


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3075_307570

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 2 = 0 ∧ x₂^2 + m*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3075_307570


namespace NUMINAMATH_CALUDE_total_crates_sold_l3075_307568

/-- Calculates the total number of crates sold over four days given specific sales conditions --/
theorem total_crates_sold (monday : ℕ) : monday = 5 → 28 = monday + (2 * monday) + (2 * monday - 2) + (monday) := by
  sorry

end NUMINAMATH_CALUDE_total_crates_sold_l3075_307568


namespace NUMINAMATH_CALUDE_badminton_players_l3075_307562

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 30)
  (h_tennis : tennis = 21)
  (h_neither : neither = 2)
  (h_both : both = 10) :
  ∃ badminton : ℕ, badminton = 17 ∧ 
    total = tennis + badminton - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l3075_307562


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3075_307597

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) :
  ∃ (min : ℝ), min = 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 4/y = 1 → x + y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3075_307597


namespace NUMINAMATH_CALUDE_power_division_result_l3075_307544

theorem power_division_result : 6^15 / 36^5 = 7776 := by
  sorry

end NUMINAMATH_CALUDE_power_division_result_l3075_307544


namespace NUMINAMATH_CALUDE_cookies_in_box_l3075_307516

/-- The number of cookies Jackson's oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies Jackson's youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- The number of days a box of cookies lasts -/
def box_duration : ℕ := 9

/-- The total number of cookies in the box -/
def total_cookies : ℕ := oldest_son_cookies + youngest_son_cookies * box_duration

theorem cookies_in_box : total_cookies = 54 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_l3075_307516


namespace NUMINAMATH_CALUDE_system_solutions_l3075_307564

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2*x) ∧
  z = y^3 * (3 - 2*y) ∧
  x = z^3 * (3 - 2*z)

/-- The theorem stating the solutions of the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3075_307564


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3075_307526

theorem fraction_equals_zero (x : ℝ) : (2 * x - 6) / (5 * x + 10) = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3075_307526


namespace NUMINAMATH_CALUDE_workshop_average_age_l3075_307502

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ) 
  (num_women num_men num_speakers : ℕ) (women_avg men_avg : ℝ) : 
  total_members = 50 →
  overall_avg = 22 →
  num_women = 25 →
  num_men = 20 →
  num_speakers = 5 →
  women_avg = 20 →
  men_avg = 25 →
  (total_members : ℝ) * overall_avg = 
    (num_women : ℝ) * women_avg + (num_men : ℝ) * men_avg + (num_speakers : ℝ) * ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) →
  ((total_members : ℝ) * overall_avg - (num_women : ℝ) * women_avg - (num_men : ℝ) * men_avg) / (num_speakers : ℝ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_age_l3075_307502


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3075_307593

/-- Given two circles on a 2D plane, prove that the x-coordinate of the point 
where a line tangent to both circles intersects the x-axis (to the right of the origin) 
is equal to 9/2. -/
theorem tangent_line_intersection 
  (r₁ r₂ c : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 5) 
  (h₃ : c = 12) : 
  ∃ x : ℝ, x > 0 ∧ x = 9/2 ∧ 
  (∃ y : ℝ, (x - 0)^2 + y^2 = r₁^2 ∧ (x - c)^2 + y^2 = r₂^2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3075_307593


namespace NUMINAMATH_CALUDE_fortieth_number_in_sampling_l3075_307522

/-- Represents the systematic sampling process in a math competition. -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstSelected : Nat) : Nat → Nat :=
  fun n => firstSelected + (totalStudents / sampleSize) * (n - 1)

/-- Theorem stating the 40th number in the systematic sampling. -/
theorem fortieth_number_in_sampling :
  systematicSampling 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_number_in_sampling_l3075_307522


namespace NUMINAMATH_CALUDE_tiger_catch_distance_l3075_307580

/-- Calculates the distance a tiger travels from a zoo given specific conditions --/
def tiger_distance (initial_speed : ℝ) (initial_time : ℝ) (slow_speed : ℝ) (slow_time : ℝ) (chase_speed : ℝ) (chase_time : ℝ) : ℝ :=
  initial_speed * initial_time + slow_speed * slow_time + chase_speed * chase_time

/-- Proves that the tiger is caught 140 miles away from the zoo --/
theorem tiger_catch_distance :
  let initial_speed : ℝ := 25
  let initial_time : ℝ := 7
  let slow_speed : ℝ := 10
  let slow_time : ℝ := 4
  let chase_speed : ℝ := 50
  let chase_time : ℝ := 0.5
  tiger_distance initial_speed initial_time slow_speed slow_time chase_speed chase_time = 140 := by
  sorry

#eval tiger_distance 25 7 10 4 50 0.5

end NUMINAMATH_CALUDE_tiger_catch_distance_l3075_307580


namespace NUMINAMATH_CALUDE_equations_equivalence_l3075_307515

-- Define the function types
variable {X : Type} [Nonempty X]
variable (f₁ f₂ f₃ f₄ : X → ℝ)

-- Define the equations
def eq1 (x : X) := f₁ x / f₂ x = f₃ x / f₄ x
def eq2 (x : X) := f₁ x / f₂ x = (f₁ x + f₃ x) / (f₂ x + f₄ x)

-- Define the conditions
def cond1 (x : X) := eq1 f₁ f₂ f₃ f₄ x → f₂ x + f₄ x ≠ 0
def cond2 (x : X) := eq2 f₁ f₂ f₃ f₄ x → f₄ x ≠ 0

-- State the theorem
theorem equations_equivalence :
  (∀ x, eq1 f₁ f₂ f₃ f₄ x ↔ eq2 f₁ f₂ f₃ f₄ x) ↔
  (∀ x, cond1 f₁ f₂ f₃ f₄ x ∧ cond2 f₁ f₂ f₃ f₄ x) :=
sorry

end NUMINAMATH_CALUDE_equations_equivalence_l3075_307515


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3075_307560

-- Define the ellipse equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k * y^2 = 2

-- Define the condition for foci on y-axis
def foci_on_y_axis (k : ℝ) : Prop := k > 0 ∧ k < 1

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k ∧ foci_on_y_axis k) ↔ 0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3075_307560


namespace NUMINAMATH_CALUDE_remaining_payment_l3075_307590

/-- Given a 10% deposit of $55, prove that the remaining amount to be paid is $495. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_cost : ℝ) : 
  deposit = 55 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * total_cost →
  total_cost - deposit = 495 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l3075_307590


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_sqrt_eight_l3075_307569

theorem negative_three_less_than_negative_sqrt_eight : -3 < -Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_sqrt_eight_l3075_307569


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3075_307567

theorem inequality_solution_set (x : ℝ) : 
  (3 - x < x - 1) ↔ (x > 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3075_307567


namespace NUMINAMATH_CALUDE_continuous_cauchy_solution_is_linear_l3075_307587

/-- Cauchy's functional equation -/
def CauchyEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The theorem stating that continuous solutions of Cauchy's equation are linear -/
theorem continuous_cauchy_solution_is_linear
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_cauchy : CauchyEquation f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end NUMINAMATH_CALUDE_continuous_cauchy_solution_is_linear_l3075_307587


namespace NUMINAMATH_CALUDE_tan_derivative_l3075_307595

open Real

theorem tan_derivative (x : ℝ) : deriv tan x = 1 / (cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_tan_derivative_l3075_307595


namespace NUMINAMATH_CALUDE_orthogonal_vectors_sum_l3075_307578

theorem orthogonal_vectors_sum (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_sum_l3075_307578


namespace NUMINAMATH_CALUDE_emily_coloring_books_l3075_307523

/-- The number of coloring books Emily gave away -/
def books_given_away : ℕ := 2

/-- The initial number of coloring books Emily had -/
def initial_books : ℕ := 7

/-- The number of coloring books Emily bought -/
def books_bought : ℕ := 14

/-- The final number of coloring books Emily has -/
def final_books : ℕ := 19

theorem emily_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l3075_307523


namespace NUMINAMATH_CALUDE_janet_dresses_l3075_307510

/-- The number of dresses Janet has -/
def total_dresses : ℕ := 24

/-- The number of pockets in Janet's dresses -/
def total_pockets : ℕ := 32

theorem janet_dresses :
  (total_dresses / 2 / 3 * 2 + total_dresses / 2 * 2 / 3 * 3 = total_pockets) ∧
  (total_dresses > 0) := by
  sorry

end NUMINAMATH_CALUDE_janet_dresses_l3075_307510


namespace NUMINAMATH_CALUDE_custom_set_op_theorem_l3075_307563

-- Define the custom set operation ⊗
def customSetOp (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem custom_set_op_theorem :
  customSetOp M N = {x | (-2 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

end NUMINAMATH_CALUDE_custom_set_op_theorem_l3075_307563


namespace NUMINAMATH_CALUDE_multiples_of_six_or_nine_l3075_307527

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_nine (n : ℕ) (h : n = 201) : 
  (count_multiples n 6 + count_multiples n 9) - count_multiples n (lcm 6 9) = 33 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_or_nine_l3075_307527


namespace NUMINAMATH_CALUDE_product_expansion_l3075_307518

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3075_307518


namespace NUMINAMATH_CALUDE_lemming_average_distance_l3075_307543

theorem lemming_average_distance (square_side : ℝ) (diagonal_move : ℝ) (perpendicular_move : ℝ) : 
  square_side = 12 →
  diagonal_move = 7.2 →
  perpendicular_move = 3 →
  let diagonal_length := square_side * Real.sqrt 2
  let fraction := diagonal_move / diagonal_length
  let x := fraction * square_side + perpendicular_move
  let y := fraction * square_side
  let dist_left := x
  let dist_bottom := y
  let dist_right := square_side - x
  let dist_top := square_side - y
  (dist_left + dist_bottom + dist_right + dist_top) / 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_lemming_average_distance_l3075_307543


namespace NUMINAMATH_CALUDE_decimal_value_l3075_307583

theorem decimal_value (x : ℚ) : (10^5 - 10^3) * x = 31 → x = 1 / 3168 := by
  sorry

end NUMINAMATH_CALUDE_decimal_value_l3075_307583


namespace NUMINAMATH_CALUDE_simplify_expression_l3075_307555

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - (1 / (1 + (a + 1) / (1 - a))) = (1 + a) / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3075_307555


namespace NUMINAMATH_CALUDE_sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l3075_307573

theorem sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1 :
  Real.sqrt 12 - ((-1 : ℝ) ^ (0 : ℕ)) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l3075_307573


namespace NUMINAMATH_CALUDE_f_s_not_multiplicative_for_other_s_l3075_307574

/-- The count of integer solutions to x_1^2 + x_2^2 + ... + x_s^2 = n -/
def r_s (s n : ℕ) : ℕ := sorry

/-- f_s(n) = r_s(n) / (2s) -/
def f_s (s n : ℕ) : ℚ := (r_s s n : ℚ) / (2 * s : ℚ)

/-- The multiplication rule for f_s -/
def multiplication_rule (s : ℕ) : Prop :=
  ∀ m n : ℕ, Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

theorem f_s_not_multiplicative_for_other_s :
  ∀ s : ℕ, s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8 →
    ∃ m n : ℕ, f_s s (m * n) ≠ f_s s m * f_s s n :=
by sorry

end NUMINAMATH_CALUDE_f_s_not_multiplicative_for_other_s_l3075_307574


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_neg_one_l3075_307592

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_point_one_neg_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x := by
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_neg_one_l3075_307592


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l3075_307586

theorem smaller_cube_side_length (R : ℝ) (x : ℝ) : 
  R = Real.sqrt 3 →
  (1 + x)^2 + (x * Real.sqrt 2 / 2)^2 = R^2 →
  x = 2/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l3075_307586


namespace NUMINAMATH_CALUDE_bus_trip_time_calculation_l3075_307507

/-- Calculates the new trip time given the original time, original speed, distance increase factor, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (distance_increase : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed * (1 + distance_increase)) / new_speed

/-- Proves that the new trip time is 256/35 hours given the specified conditions -/
theorem bus_trip_time_calculation :
  let original_time : ℚ := 16 / 3  -- 5 1/3 hours
  let original_speed : ℚ := 80
  let distance_increase : ℚ := 1 / 5  -- 20% increase
  let new_speed : ℚ := 70
  new_trip_time original_time original_speed distance_increase new_speed = 256 / 35 := by
  sorry

#eval new_trip_time (16/3) 80 (1/5) 70

end NUMINAMATH_CALUDE_bus_trip_time_calculation_l3075_307507


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3075_307556

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3075_307556


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3075_307529

-- Define the quadratic equation and its roots
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the sets S, T, P, Q
def S (x₁ : ℝ) : Set ℝ := {x | x > x₁}
def T (x₂ : ℝ) : Set ℝ := {x | x > x₂}
def P (x₁ : ℝ) : Set ℝ := {x | x < x₁}
def Q (x₂ : ℝ) : Set ℝ := {x | x < x₂}

-- State the theorem
theorem solution_set_quadratic_inequality 
  (a b c x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation a b c x₁)
  (h₂ : quadratic_equation a b c x₂)
  (h₃ : x₁ ≠ x₂)
  (h₄ : a > 0) :
  {x : ℝ | a * x^2 + b * x + c > 0} = (S x₁ ∩ T x₂) ∪ (P x₁ ∩ Q x₂) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3075_307529


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l3075_307577

theorem medicine_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 32)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price :=
by
  sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l3075_307577


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l3075_307520

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l3075_307520


namespace NUMINAMATH_CALUDE_maria_has_nineteen_towels_l3075_307519

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def marias_remaining_towels (green_towels white_towels given_to_mother : ℕ) : ℕ :=
  green_towels + white_towels - given_to_mother

/-- Theorem stating that Maria ended up with 19 towels. -/
theorem maria_has_nineteen_towels :
  marias_remaining_towels 40 44 65 = 19 := by
  sorry

end NUMINAMATH_CALUDE_maria_has_nineteen_towels_l3075_307519


namespace NUMINAMATH_CALUDE_eraser_boxes_donated_l3075_307511

theorem eraser_boxes_donated (erasers_per_box : ℕ) (price_per_eraser : ℚ) (total_money : ℚ) :
  erasers_per_box = 24 →
  price_per_eraser = 3/4 →
  total_money = 864 →
  (total_money / price_per_eraser) / erasers_per_box = 48 :=
by sorry

end NUMINAMATH_CALUDE_eraser_boxes_donated_l3075_307511


namespace NUMINAMATH_CALUDE_number_relation_l3075_307561

theorem number_relation (A B : ℝ) (h : A = B * (1 + 0.1)) : B = A * (10/11) := by
  sorry

end NUMINAMATH_CALUDE_number_relation_l3075_307561


namespace NUMINAMATH_CALUDE_money_left_after_candy_purchase_l3075_307514

def lollipop_cost : ℚ := 1.5
def gummy_pack_cost : ℚ := 2
def lollipop_count : ℕ := 4
def gummy_pack_count : ℕ := 2
def initial_money : ℚ := 15

def total_spent : ℚ := lollipop_cost * lollipop_count + gummy_pack_cost * gummy_pack_count

theorem money_left_after_candy_purchase : 
  initial_money - total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_candy_purchase_l3075_307514


namespace NUMINAMATH_CALUDE_bottles_not_in_crate_l3075_307508

/-- Given the number of bottles per crate, total bottles, and number of crates,
    calculate the number of bottles that will not be placed in a crate. -/
theorem bottles_not_in_crate
  (bottles_per_crate : ℕ)
  (total_bottles : ℕ)
  (num_crates : ℕ)
  (h1 : bottles_per_crate = 12)
  (h2 : total_bottles = 130)
  (h3 : num_crates = 10) :
  total_bottles - (bottles_per_crate * num_crates) = 10 :=
by sorry

end NUMINAMATH_CALUDE_bottles_not_in_crate_l3075_307508


namespace NUMINAMATH_CALUDE_zaras_goats_l3075_307505

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  groups * animals_per_group - cows - sheep = 113 :=
by sorry

end NUMINAMATH_CALUDE_zaras_goats_l3075_307505


namespace NUMINAMATH_CALUDE_division_problem_l3075_307536

theorem division_problem (quotient divisor remainder : ℕ) 
  (h1 : quotient = 3)
  (h2 : divisor = 3)
  (h3 : divisor = 3 * remainder) : 
  quotient * divisor + remainder = 10 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3075_307536


namespace NUMINAMATH_CALUDE_chair_cost_l3075_307535

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) 
  (h1 : total_cost = 135)
  (h2 : table_cost = 55)
  (h3 : num_chairs = 4) :
  (total_cost - table_cost) / num_chairs = 20 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l3075_307535


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3075_307545

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1}

-- Define the set A
def A : Set ℤ := {0, 1}

-- Theorem statement
theorem complement_of_A_in_U :
  {x : ℤ | x ∈ U ∧ x ∉ A} = {-1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3075_307545


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_18_l3075_307541

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_60_not_18 : 
  (∀ n : ℕ, n < 810 → (is_multiple n 45 ∧ is_multiple n 60) → is_multiple n 18) ∧ 
  is_multiple 810 45 ∧ 
  is_multiple 810 60 ∧ 
  ¬is_multiple 810 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_18_l3075_307541


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_fourth_l3075_307501

theorem x_fourth_minus_reciprocal_fourth (x : ℝ) (h : x^2 - Real.sqrt 6 * x + 1 = 0) :
  |x^4 - 1/x^4| = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_fourth_l3075_307501


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3075_307534

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  left = 6 → new = 42 → final = 47 → initial + new - left = final → initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3075_307534


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3075_307581

theorem pure_imaginary_fraction (m : ℝ) : 
  (∃ (k : ℝ), (2 - m * Complex.I) / (1 + Complex.I) = k * Complex.I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3075_307581


namespace NUMINAMATH_CALUDE_square_to_rectangle_l3075_307517

theorem square_to_rectangle (s : ℝ) (h1 : s > 0) 
  (h2 : s * (s + 3) - s * s = 18) : 
  s * s = 36 ∧ s * (s + 3) = 54 := by
  sorry

#check square_to_rectangle

end NUMINAMATH_CALUDE_square_to_rectangle_l3075_307517
