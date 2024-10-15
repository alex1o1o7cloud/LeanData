import Mathlib

namespace NUMINAMATH_CALUDE_min_max_values_on_interval_monotone_increasing_condition_l3594_359459

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

-- Part I
theorem min_max_values_on_interval (a : ℝ) (h : a = 2) :
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 1) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 6) := by
  sorry

-- Part II
theorem monotone_increasing_condition :
  (∀ a : ℝ, a ∈ Set.Icc (-2) 0 ↔ Monotone (f a)) := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_on_interval_monotone_increasing_condition_l3594_359459


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3594_359401

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 3 4 = {x | x^2 + a*x + b < 0}) : 
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/4) ∪ Set.Ici (1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3594_359401


namespace NUMINAMATH_CALUDE_paper_sheet_length_l3594_359425

/-- The length of the second sheet of paper satisfies the area equation -/
theorem paper_sheet_length : ∃ L : ℝ, 2 * (11 * 17) = 2 * (8.5 * L) + 100 := by
  sorry

end NUMINAMATH_CALUDE_paper_sheet_length_l3594_359425


namespace NUMINAMATH_CALUDE_thirteen_to_six_mod_eight_l3594_359434

theorem thirteen_to_six_mod_eight (m : ℕ) : 
  13^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_to_six_mod_eight_l3594_359434


namespace NUMINAMATH_CALUDE_element_in_M_l3594_359474

def M : Set (ℕ × ℕ) := {(2, 3)}

theorem element_in_M : (2, 3) ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_M_l3594_359474


namespace NUMINAMATH_CALUDE_batsman_average_problem_l3594_359453

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem statement for the batsman's average problem -/
theorem batsman_average_problem (stats : BatsmanStats) 
  (h1 : stats.innings = 11)
  (h2 : newAverage stats 55 = stats.average + 1) :
  newAverage stats 55 = 44 := by
  sorry

#check batsman_average_problem

end NUMINAMATH_CALUDE_batsman_average_problem_l3594_359453


namespace NUMINAMATH_CALUDE_supplementary_angle_of_39_23_l3594_359439

-- Define the angle type with degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the supplementary angle function
def supplementaryAngle (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem supplementary_angle_of_39_23 :
  let a : Angle := ⟨39, 23⟩
  supplementaryAngle a = ⟨140, 37⟩ := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_39_23_l3594_359439


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3594_359430

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem statement -/
theorem parabola_midpoint_to_directrix 
  (para : Parabola) 
  (A B M : Point) 
  (h_line : (B.y - A.y) / (B.x - A.x) = 1) -- Slope of line AB is 1
  (h_on_parabola : A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x) -- A and B are on the parabola
  (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) -- M is midpoint of AB
  (h_m_y : M.y = 2) -- y-coordinate of M is 2
  : M.x - (-para.p) = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l3594_359430


namespace NUMINAMATH_CALUDE_linear_coefficient_is_zero_l3594_359436

/-- The coefficient of the linear term in the standard form of (2 - x)(3x + 4) = 2x - 1 is 0 -/
theorem linear_coefficient_is_zero : 
  let f : ℝ → ℝ := λ x => (2 - x) * (3 * x + 4) - (2 * x - 1)
  ∃ a c : ℝ, ∀ x, f x = -3 * x^2 + 0 * x + c :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_is_zero_l3594_359436


namespace NUMINAMATH_CALUDE_constant_sequence_conditions_l3594_359494

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal -/
def is_constant (a : Sequence) : Prop :=
  ∀ n m : ℕ, a n = a m

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def is_geometric (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is arithmetic if the difference of consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_sequence_conditions (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (a : Sequence) :
  (is_geometric a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_arithmetic a ∧ is_geometric (fun n ↦ k * a n + b)) ∨
  (is_geometric a ∧ is_arithmetic (fun n ↦ k * a n + b))
  → is_constant a := by
  sorry

end NUMINAMATH_CALUDE_constant_sequence_conditions_l3594_359494


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l3594_359437

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: When x ∈ (0, +∞), f'(x) > 0
  (∀ x : ℝ, x > 0 → deriv f x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, deriv f (-x) = -(deriv f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_f_satisfies_properties_l3594_359437


namespace NUMINAMATH_CALUDE_parabola_shift_right_parabola_shift_result_l3594_359489

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right (x : ℝ) :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  (x^2 + 6*x) = ((x-4)^2 + 6*(x-4)) :=
by sorry

theorem parabola_shift_result :
  let original := Parabola.mk 1 6 0
  let shifted := shift_parabola original 4
  shifted.a * x^2 + shifted.b * x + shifted.c = (x - 1)^2 - 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_right_parabola_shift_result_l3594_359489


namespace NUMINAMATH_CALUDE_smallest_factor_smallest_factor_exists_l3594_359415

theorem smallest_factor (n : ℕ) : n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 → n ≥ 468 := by
  sorry

theorem smallest_factor_exists : ∃ n : ℕ, n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 ∧ n = 468 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_smallest_factor_exists_l3594_359415


namespace NUMINAMATH_CALUDE_problem_solution_l3594_359458

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x - 11
def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem problem_solution :
  ∀ a k : ℝ,
  (f' a (-1) = 0 → a = -2) ∧
  (∃ x y : ℝ, f a x = k * x + 9 ∧ f' a x = k ∧ g x = k * x + 9 ∧ (3 * 2 * x + 6) = k → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3594_359458


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l3594_359454

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x| + |x + 2|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 7} = {x : ℝ | -3 < x ∧ x < 4} := by sorry

-- Part II
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 1 2, |f a x| ≤ |x + 4|} = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l3594_359454


namespace NUMINAMATH_CALUDE_jam_eaten_for_lunch_l3594_359452

theorem jam_eaten_for_lunch (x : ℚ) : 
  (1 - x) * (1 - 1/7) = 4/7 → x = 1/21 := by
  sorry

end NUMINAMATH_CALUDE_jam_eaten_for_lunch_l3594_359452


namespace NUMINAMATH_CALUDE_vacation_cost_division_l3594_359499

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) : 
  total_cost = 375 ∧ 
  initial_people = 3 ∧ 
  cost_reduction = 50 ∧ 
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l3594_359499


namespace NUMINAMATH_CALUDE_house_rent_calculation_l3594_359472

/-- Given a person's expenditure pattern and petrol cost, calculate house rent -/
theorem house_rent_calculation (income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_cost : ℝ) : 
  petrol_percentage = 0.3 →
  rent_percentage = 0.2 →
  petrol_cost = 300 →
  petrol_cost = petrol_percentage * income →
  rent_percentage * (income - petrol_cost) = 140 :=
by sorry

end NUMINAMATH_CALUDE_house_rent_calculation_l3594_359472


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3594_359483

-- Define the universal set U
def U : Set ℝ := {x | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x | 2^x > Real.sqrt 2}

-- Statement to prove
theorem complement_of_A_in_U :
  Set.compl A ∩ U = Set.Icc (-Real.sqrt 3) (1/2) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3594_359483


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3594_359422

theorem purely_imaginary_complex_fraction (a b : ℝ) (h : b ≠ 0) :
  (∃ (k : ℝ), (Complex.I : ℂ) * k = (a + Complex.I * b) / (4 + Complex.I * 3)) →
  a / b = -3/4 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3594_359422


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3594_359421

def total_cars : ℕ := 300
def ford_percentage : ℚ := 20 / 100
def nissan_percentage : ℚ := 25 / 100
def volkswagen_percentage : ℚ := 10 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (ford_percentage + nissan_percentage + volkswagen_percentage)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3594_359421


namespace NUMINAMATH_CALUDE_sum_of_24_numbers_l3594_359440

theorem sum_of_24_numbers (numbers : List ℤ) : 
  numbers.length = 24 → numbers.sum = 576 → 
  (∀ n ∈ numbers, Even n) ∨ 
  (∃ (evens odds : List ℤ), 
    numbers = evens ++ odds ∧ 
    (∀ n ∈ evens, Even n) ∧ 
    (∀ n ∈ odds, Odd n) ∧ 
    Even (odds.length)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_24_numbers_l3594_359440


namespace NUMINAMATH_CALUDE_sams_french_bulldogs_count_l3594_359490

/-- The number of French Bulldogs Sam has -/
def sams_french_bulldogs : ℕ := 4

/-- The number of German Shepherds Sam has -/
def sams_german_shepherds : ℕ := 3

/-- The total number of dogs Peter wants -/
def peters_total_dogs : ℕ := 17

theorem sams_french_bulldogs_count :
  sams_french_bulldogs = 4 :=
by
  have h1 : peters_total_dogs = 3 * sams_german_shepherds + 2 * sams_french_bulldogs :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_sams_french_bulldogs_count_l3594_359490


namespace NUMINAMATH_CALUDE_equation_solution_l3594_359447

theorem equation_solution (x y : ℝ) :
  (2 * x) / (1 + x^2) = (1 + y^2) / (2 * y) →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3594_359447


namespace NUMINAMATH_CALUDE_circle_center_l3594_359431

/-- The center of a circle given by the equation x^2 - 8x + y^2 + 4y = -3 -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 + 4*y = -3) → (∃ r : ℝ, (x - 4)^2 + (y + 2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3594_359431


namespace NUMINAMATH_CALUDE_optimal_cylinder_ratio_l3594_359464

/-- Theorem: Optimal ratio of height to radius for a cylinder with minimal surface area -/
theorem optimal_cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  V = π * r^2 * h ∧ V = 1000 → 
  (∀ h' r', h' > 0 → r' > 0 → V = π * r'^2 * h' → 
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') →
  h / r = 1 := by
sorry

end NUMINAMATH_CALUDE_optimal_cylinder_ratio_l3594_359464


namespace NUMINAMATH_CALUDE_set_intersection_example_l3594_359470

theorem set_intersection_example :
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {1, 2, 3, 4}
  A ∩ B = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3594_359470


namespace NUMINAMATH_CALUDE_square_difference_pattern_l3594_359442

theorem square_difference_pattern (n : ℕ) :
  (2*n + 2)^2 - (2*n)^2 = 4*(2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l3594_359442


namespace NUMINAMATH_CALUDE_root_sum_equality_l3594_359416

-- Define the polynomial f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x

-- Define the theorem
theorem root_sum_equality 
  (a b c : ℝ) 
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) : 
  (f a b c x₁ = 1) ∧ (f a b c x₂ = 1) ∧ (f a b c x₃ = 1) ∧ (f a b c x₄ = 1) →
  (f a b c y₁ = 2) ∧ (f a b c y₂ = 2) ∧ (f a b c y₃ = 2) ∧ (f a b c y₄ = 2) →
  (x₁ + x₂ = x₃ + x₄) →
  (y₁ + y₂ = y₃ + y₄) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_equality_l3594_359416


namespace NUMINAMATH_CALUDE_transportation_theorem_l3594_359407

/-- Represents a type of transportation with its quantity and number of wheels -/
structure Transportation where
  name : String
  quantity : Nat
  wheels : Nat

/-- Calculates the total number of wheels for a given transportation -/
def totalWheels (t : Transportation) : Nat :=
  t.quantity * t.wheels

/-- Calculates the total number of wheels for a list of transportations -/
def sumWheels (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + totalWheels t) 0

/-- Calculates the total quantity of all transportations -/
def totalQuantity (ts : List Transportation) : Nat :=
  ts.foldl (fun acc t => acc + t.quantity) 0

/-- Calculates the quantity of bicycles and tricycles -/
def bikeAndTricycleCount (ts : List Transportation) : Nat :=
  ts.filter (fun t => t.name = "bicycle" || t.name = "tricycle")
    |>.foldl (fun acc t => acc + t.quantity) 0

theorem transportation_theorem (observations : List Transportation) 
  (h1 : observations = [
    ⟨"car", 15, 4⟩, 
    ⟨"bicycle", 3, 2⟩, 
    ⟨"pickup truck", 8, 4⟩, 
    ⟨"tricycle", 1, 3⟩, 
    ⟨"motorcycle", 4, 2⟩, 
    ⟨"skateboard", 2, 4⟩, 
    ⟨"unicycle", 1, 1⟩
  ]) : 
  sumWheels observations = 118 ∧ 
  (bikeAndTricycleCount observations : Rat) / (totalQuantity observations : Rat) = 4/34 := by
  sorry

end NUMINAMATH_CALUDE_transportation_theorem_l3594_359407


namespace NUMINAMATH_CALUDE_card_probability_l3594_359484

def standard_deck : ℕ := 52
def num_jacks : ℕ := 4
def num_queens : ℕ := 4

theorem card_probability : 
  let p_two_queens := (num_queens / standard_deck) * ((num_queens - 1) / (standard_deck - 1))
  let p_one_jack := 2 * (num_jacks / standard_deck) * ((standard_deck - num_jacks) / (standard_deck - 1))
  let p_two_jacks := (num_jacks / standard_deck) * ((num_jacks - 1) / (standard_deck - 1))
  p_two_queens + p_one_jack + p_two_jacks = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_l3594_359484


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l3594_359492

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ all_odd_digits (n + reverse_digits n)

theorem smallest_satisfying_number :
  satisfies_condition 209 ∧
  ∀ m : ℕ, satisfies_condition m → m ≥ 209 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l3594_359492


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_with_second_digit_6_l3594_359408

/-- A function that returns true if all digits in a number are different --/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the digit at a specific position in a number --/
def digitAt (n : ℕ) (pos : ℕ) : ℕ := sorry

theorem smallest_four_digit_number_with_second_digit_6 :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n < 10000) →  -- four-digit number
  (digitAt n 2 = 6) →       -- second digit is 6
  allDigitsDifferent n →    -- all digits are different
  1602 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_with_second_digit_6_l3594_359408


namespace NUMINAMATH_CALUDE_max_intersections_proof_l3594_359451

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := Nat.choose num_x_points 2 * Nat.choose num_y_points 2

theorem max_intersections_proof :
  max_intersections = 990 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_proof_l3594_359451


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3594_359405

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ (117 + 36*Real.sqrt 3) / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3594_359405


namespace NUMINAMATH_CALUDE_sum_of_distinct_remainders_div_13_l3594_359450

def remainders : List Nat :=
  (List.range 10).map (fun n => (n + 1)^2 % 13)

def distinct_remainders : List Nat :=
  remainders.eraseDups

theorem sum_of_distinct_remainders_div_13 :
  (distinct_remainders.sum) / 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_remainders_div_13_l3594_359450


namespace NUMINAMATH_CALUDE_union_equals_A_l3594_359426

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | Real.log x ≤ a}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_union_equals_A_l3594_359426


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l3594_359417

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 20

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percent : ℝ := 50

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket_percent = exceed_limit_percent * (100 - no_ticket_percent) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l3594_359417


namespace NUMINAMATH_CALUDE_egg_marble_distribution_unique_l3594_359403

/-- Represents the distribution of eggs and marbles among three groups. -/
structure EggMarbleDistribution where
  eggs_a : ℕ
  eggs_b : ℕ
  eggs_c : ℕ
  marbles_a : ℕ
  marbles_b : ℕ
  marbles_c : ℕ

/-- Checks if the given distribution satisfies all conditions. -/
def is_valid_distribution (d : EggMarbleDistribution) : Prop :=
  d.eggs_a + d.eggs_b + d.eggs_c = 15 ∧
  d.marbles_a + d.marbles_b + d.marbles_c = 4 ∧
  d.eggs_a ≠ d.eggs_b ∧ d.eggs_b ≠ d.eggs_c ∧ d.eggs_a ≠ d.eggs_c ∧
  d.eggs_b = d.marbles_b - d.marbles_a ∧
  d.eggs_c = d.marbles_c - d.marbles_b

theorem egg_marble_distribution_unique :
  ∃! d : EggMarbleDistribution, is_valid_distribution d ∧
    d.eggs_a = 12 ∧ d.eggs_b = 1 ∧ d.eggs_c = 2 :=
sorry

end NUMINAMATH_CALUDE_egg_marble_distribution_unique_l3594_359403


namespace NUMINAMATH_CALUDE_max_popsicles_l3594_359481

def lucy_budget : ℚ := 15
def popsicle_cost : ℚ := 2.4

theorem max_popsicles : 
  ∀ n : ℕ, (n : ℚ) * popsicle_cost ≤ lucy_budget → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_l3594_359481


namespace NUMINAMATH_CALUDE_quadratic_sum_l3594_359435

/-- A quadratic function with specified properties -/
structure QuadraticFunction where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = d * vertex_x^2 + e * vertex_x + f
  point_condition : point_y = d * point_x^2 + e * point_x + f
  is_vertex : ∀ x : ℝ, d * x^2 + e * x + f ≥ vertex_y

/-- Theorem: For a quadratic function with given properties, d + e + 2f = 19 -/
theorem quadratic_sum (g : QuadraticFunction) 
  (h1 : g.vertex_x = -2) 
  (h2 : g.vertex_y = 3) 
  (h3 : g.point_x = 0) 
  (h4 : g.point_y = 7) : 
  g.d + g.e + 2 * g.f = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3594_359435


namespace NUMINAMATH_CALUDE_haley_cider_pints_l3594_359491

/-- Represents the number of pints of cider Haley can make --/
def cider_pints (golden_apples_per_pint : ℕ) (pink_apples_per_pint : ℕ) 
  (apples_per_hour : ℕ) (farmhands : ℕ) (hours_worked : ℕ) 
  (golden_to_pink_ratio : ℚ) : ℕ :=
  let total_apples := apples_per_hour * farmhands * hours_worked
  let apples_per_pint := golden_apples_per_pint + pink_apples_per_pint
  total_apples / apples_per_pint

/-- Theorem stating the number of pints of cider Haley can make --/
theorem haley_cider_pints : 
  cider_pints 20 40 240 6 5 (1/3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_haley_cider_pints_l3594_359491


namespace NUMINAMATH_CALUDE_kids_staying_home_l3594_359432

def total_kids : ℕ := 898051
def kids_at_camp : ℕ := 629424

theorem kids_staying_home : total_kids - kids_at_camp = 268627 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l3594_359432


namespace NUMINAMATH_CALUDE_card_combinations_l3594_359479

theorem card_combinations : Nat.choose 40 7 = 1860480 := by sorry

end NUMINAMATH_CALUDE_card_combinations_l3594_359479


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l3594_359402

theorem polynomial_value_relation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) :
  6 * x^2 + 9 * y + 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l3594_359402


namespace NUMINAMATH_CALUDE_janet_movie_cost_l3594_359411

/-- The cost per minute to film Janet's previous movie -/
def previous_cost_per_minute : ℝ := 5

/-- The length of Janet's previous movie in minutes -/
def previous_movie_length : ℝ := 120

/-- The length of Janet's new movie in minutes -/
def new_movie_length : ℝ := previous_movie_length * 1.6

/-- The cost per minute to film Janet's new movie -/
def new_cost_per_minute : ℝ := 2 * previous_cost_per_minute

/-- The total cost to film Janet's new movie -/
def new_movie_total_cost : ℝ := 1920

theorem janet_movie_cost : 
  new_movie_length * new_cost_per_minute = new_movie_total_cost :=
by sorry

end NUMINAMATH_CALUDE_janet_movie_cost_l3594_359411


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3594_359418

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3594_359418


namespace NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l3594_359467

/-- The maximum number of non-intersecting diagonals in a convex n-gon --/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem stating that the maximum number of non-intersecting diagonals in a convex n-gon is n-3 --/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l3594_359467


namespace NUMINAMATH_CALUDE_count_non_negative_l3594_359498

def number_set : List ℚ := [-15, 16/3, -23/100, 0, 76/10, 2, -3/5, 314/100]

theorem count_non_negative : (number_set.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_non_negative_l3594_359498


namespace NUMINAMATH_CALUDE_stuffed_animals_count_l3594_359461

/-- The total number of stuffed animals for three girls -/
def total_stuffed_animals (mckenna kenley tenly : ℕ) : ℕ :=
  mckenna + kenley + tenly

/-- Theorem stating the total number of stuffed animals for the three girls -/
theorem stuffed_animals_count :
  ∃ (kenley tenly : ℕ),
    let mckenna := 34
    kenley = 2 * mckenna ∧
    tenly = kenley + 5 ∧
    total_stuffed_animals mckenna kenley tenly = 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_count_l3594_359461


namespace NUMINAMATH_CALUDE_two_out_of_five_permutation_l3594_359475

theorem two_out_of_five_permutation : 
  (Finset.range 5).card * (Finset.range 4).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_five_permutation_l3594_359475


namespace NUMINAMATH_CALUDE_meeting_point_coordinates_l3594_359413

/-- The point that divides a line segment in a given ratio -/
def dividing_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℚ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Proof that the point dividing the line segment from (2, 5) to (10, 1) 
    in the ratio 1:3 starting from (2, 5) has coordinates (4, 4) -/
theorem meeting_point_coordinates : 
  dividing_point 2 5 10 1 1 3 = (4, 4) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_coordinates_l3594_359413


namespace NUMINAMATH_CALUDE_vector_magnitude_l3594_359409

theorem vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  (‖a‖ = 2) → (‖b‖ = 3) → (‖a + b‖ = 3) → ‖a + 2 • b‖ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3594_359409


namespace NUMINAMATH_CALUDE_div_chain_equals_fraction_l3594_359446

theorem div_chain_equals_fraction : (132 / 6) / 3 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_div_chain_equals_fraction_l3594_359446


namespace NUMINAMATH_CALUDE_strawberry_jam_money_l3594_359441

/-- Calculates the total money made from selling strawberry jam given the number of strawberries picked by Betty, Matthew, and Natalie, and the jam-making and selling conditions. -/
theorem strawberry_jam_money (betty_strawberries : ℕ) (matthew_extra : ℕ) (jam_strawberries : ℕ) (jar_price : ℕ) : 
  betty_strawberries = 25 →
  matthew_extra = 30 →
  jam_strawberries = 12 →
  jar_price = 6 →
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 3
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars := total_strawberries / jam_strawberries
  let total_money := jars * jar_price
  total_money = 48 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jam_money_l3594_359441


namespace NUMINAMATH_CALUDE_max_product_constraint_l3594_359460

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3594_359460


namespace NUMINAMATH_CALUDE_polynomial_divisibility_sum_l3594_359477

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def f (C D E : ℝ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

/-- The polynomial x^2 + x + 1 -/
def g (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility_sum (C D E : ℝ) :
  (∀ x, g x = 0 → f C D E x = 0) → C + D + E = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_sum_l3594_359477


namespace NUMINAMATH_CALUDE_new_unsigned_books_l3594_359456

def adventure_books : ℕ := 13
def mystery_books : ℕ := 17
def scifi_books : ℕ := 25
def nonfiction_books : ℕ := 10
def used_books : ℕ := 42
def signed_books : ℕ := 10

theorem new_unsigned_books : 
  adventure_books + mystery_books + scifi_books + nonfiction_books - used_books - signed_books = 13 := by
  sorry

end NUMINAMATH_CALUDE_new_unsigned_books_l3594_359456


namespace NUMINAMATH_CALUDE_marble_count_exceeds_200_l3594_359469

def marbles (n : ℕ) : ℕ := 3 * 2^n

theorem marble_count_exceeds_200 :
  (∃ k : ℕ, marbles k > 200) ∧ 
  (∀ j : ℕ, j < 8 → marbles j ≤ 200) ∧
  (marbles 8 > 200) := by
sorry

end NUMINAMATH_CALUDE_marble_count_exceeds_200_l3594_359469


namespace NUMINAMATH_CALUDE_incorrect_statement_C_l3594_359420

theorem incorrect_statement_C :
  (∀ a b c : ℚ, a / 4 = c / 5 → (a - 4) / 4 = (c - 5) / 5) ∧
  (∀ a b : ℚ, (a - b) / b = 1 / 7 → a / b = 8 / 7) ∧
  (∃ a b : ℚ, a / b = 2 / 5 ∧ (a ≠ 2 ∨ b ≠ 5)) ∧
  (∀ a b c d : ℚ, a / b = c / d ∧ a / b = 2 / 3 ∧ b - d ≠ 0 → (a - c) / (b - d) = 2 / 3) :=
by sorry


end NUMINAMATH_CALUDE_incorrect_statement_C_l3594_359420


namespace NUMINAMATH_CALUDE_real_roots_quadratic_complex_coeff_l3594_359429

theorem real_roots_quadratic_complex_coeff (i : ℂ) (m : ℝ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_complex_coeff_l3594_359429


namespace NUMINAMATH_CALUDE_kola_solution_water_added_l3594_359463

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  volume : ℝ
  water_percent : ℝ
  kola_percent : ℝ
  sugar_percent : ℝ

def initial_solution : KolaSolution :=
  { volume := 440
  , water_percent := 88
  , kola_percent := 8
  , sugar_percent := 100 - 88 - 8 }

def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percent : ℝ := 4.521739130434784

/-- The amount of water added to the solution -/
def water_added : ℝ := 10

theorem kola_solution_water_added :
  let initial_sugar := initial_solution.volume * initial_solution.sugar_percent / 100
  let total_sugar := initial_sugar + added_sugar
  let final_volume := total_sugar / (final_sugar_percent / 100)
  water_added = final_volume - initial_solution.volume - added_sugar - added_kola :=
by sorry

end NUMINAMATH_CALUDE_kola_solution_water_added_l3594_359463


namespace NUMINAMATH_CALUDE_differential_equation_solution_l3594_359433

/-- Given a differential equation y = x * y' + a / (2 * y'), where a is a constant,
    prove that the solutions are:
    1. y = C * x + a / (2 * C), where C is a constant
    2. y^2 = 2 * a * x
-/
theorem differential_equation_solution (a : ℝ) (x y : ℝ → ℝ) (y' : ℝ → ℝ) :
  (∀ t, y t = t * y' t + a / (2 * y' t)) →
  (∃ C : ℝ, ∀ t, y t = C * t + a / (2 * C)) ∨
  (∀ t, (y t)^2 = 2 * a * t) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l3594_359433


namespace NUMINAMATH_CALUDE_limit_one_minus_cos_over_exp_squared_l3594_359495

theorem limit_one_minus_cos_over_exp_squared :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((1 - Real.cos x) / (Real.exp (3 * x) - 1)^2) - (1/18)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_one_minus_cos_over_exp_squared_l3594_359495


namespace NUMINAMATH_CALUDE_polygon_sides_l3594_359493

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 (for a valid polygon)
  ((n - 2) * 180 = 3 * 360) →         -- sum of interior angles = 3 * sum of exterior angles
  n = 8                               -- the polygon has 8 sides
:= by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3594_359493


namespace NUMINAMATH_CALUDE_third_term_geometric_series_l3594_359488

/-- Theorem: Third term of a specific geometric series -/
theorem third_term_geometric_series
  (q : ℝ) 
  (h₁ : |q| < 1)
  (h₂ : 2 / (1 - q) = 8 / 5)
  (h₃ : 2 * q = -1 / 2) :
  2 * q^2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_third_term_geometric_series_l3594_359488


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3594_359404

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3594_359404


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3594_359428

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x/2 + a ≥ 2 ∧ 2*x - b < 3)) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3594_359428


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3594_359486

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3594_359486


namespace NUMINAMATH_CALUDE_speed_difference_calculation_l3594_359419

/-- Calculates the difference in average speed between no traffic and heavy traffic conditions --/
theorem speed_difference_calculation (distance : ℝ) (heavy_traffic_time : ℝ) (no_traffic_time : ℝ)
  (construction_zones : ℕ) (construction_delay : ℝ) (heavy_traffic_rest_stops : ℕ)
  (no_traffic_rest_stops : ℕ) (rest_stop_duration : ℝ) :
  distance = 200 →
  heavy_traffic_time = 5 →
  no_traffic_time = 4 →
  construction_zones = 2 →
  construction_delay = 0.25 →
  heavy_traffic_rest_stops = 3 →
  no_traffic_rest_stops = 2 →
  rest_stop_duration = 1/6 →
  let heavy_traffic_driving_time := heavy_traffic_time - (construction_zones * construction_delay + heavy_traffic_rest_stops * rest_stop_duration)
  let no_traffic_driving_time := no_traffic_time - (no_traffic_rest_stops * rest_stop_duration)
  let heavy_traffic_speed := distance / heavy_traffic_driving_time
  let no_traffic_speed := distance / no_traffic_driving_time
  ∃ ε > 0, |no_traffic_speed - heavy_traffic_speed - 4.5| < ε :=
by sorry

end NUMINAMATH_CALUDE_speed_difference_calculation_l3594_359419


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l3594_359449

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosC : ℝ

/-- Theorem about the side length and area of a specific triangle -/
theorem triangle_side_and_area (t : Triangle) 
  (h1 : t.a = 1)
  (h2 : t.b = 2)
  (h3 : t.cosC = 1/4) :
  t.c = 2 ∧ (1/2 * t.a * t.b * Real.sqrt (1 - t.cosC^2)) = Real.sqrt 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l3594_359449


namespace NUMINAMATH_CALUDE_max_pawns_2019_l3594_359473

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

/-- Represents the placement of pieces on the chessboard -/
structure Placement (n : ℕ) where
  board : Chessboard n
  pawns : ℕ
  rooks : ℕ
  no_rooks_see_each_other : Bool

/-- The maximum number of pawns that can be placed -/
def max_pawns (n : ℕ) : ℕ := (n / 2) ^ 2

/-- Theorem stating the maximum number of pawns for a 2019 x 2019 chessboard -/
theorem max_pawns_2019 :
  ∃ (p : Placement 2019),
    p.pawns = max_pawns 2019 ∧
    p.rooks = p.pawns + 2019 ∧
    p.no_rooks_see_each_other = true ∧
    ∀ (q : Placement 2019),
      q.no_rooks_see_each_other = true →
      q.rooks = q.pawns + 2019 →
      q.pawns ≤ p.pawns :=
by sorry

end NUMINAMATH_CALUDE_max_pawns_2019_l3594_359473


namespace NUMINAMATH_CALUDE_ted_stick_count_l3594_359444

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- The scenario of Bill and Ted throwing objects into the river -/
def river_throw_scenario (ted : ThrowCount) (bill : ThrowCount) : Prop :=
  bill.sticks = ted.sticks + 6 ∧
  ted.rocks = 2 * bill.rocks ∧
  bill.sticks + bill.rocks = 21

theorem ted_stick_count (ted : ThrowCount) (bill : ThrowCount) 
  (h : river_throw_scenario ted bill) : ted.sticks = 15 := by
  sorry

#check ted_stick_count

end NUMINAMATH_CALUDE_ted_stick_count_l3594_359444


namespace NUMINAMATH_CALUDE_lemonade_glasses_per_gallon_l3594_359406

theorem lemonade_glasses_per_gallon 
  (total_gallons : ℕ) 
  (cost_per_gallon : ℚ) 
  (price_per_glass : ℚ) 
  (glasses_drunk : ℕ) 
  (glasses_unsold : ℕ) 
  (net_profit : ℚ) :
  total_gallons = 2 ∧ 
  cost_per_gallon = 7/2 ∧ 
  price_per_glass = 1 ∧ 
  glasses_drunk = 5 ∧ 
  glasses_unsold = 6 ∧ 
  net_profit = 14 →
  ∃ (glasses_per_gallon : ℕ),
    glasses_per_gallon = 16 ∧
    (total_gallons * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass
    = total_gallons * cost_per_gallon + net_profit :=
by sorry

end NUMINAMATH_CALUDE_lemonade_glasses_per_gallon_l3594_359406


namespace NUMINAMATH_CALUDE_inequality_proof_l3594_359497

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hu₁ : x₁ * y₁ - z₁^2 > 0) (hu₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3594_359497


namespace NUMINAMATH_CALUDE_ravens_age_l3594_359478

theorem ravens_age (phoebe_age : ℕ) (raven_age : ℕ) : 
  phoebe_age = 10 →
  raven_age + 5 = 4 * (phoebe_age + 5) →
  raven_age = 55 := by
sorry

end NUMINAMATH_CALUDE_ravens_age_l3594_359478


namespace NUMINAMATH_CALUDE_ten_steps_climb_l3594_359485

def climb_stairs (n : ℕ) : ℕ :=
  if n ≤ 1 then 1
  else if n = 2 then 2
  else climb_stairs (n - 1) + climb_stairs (n - 2)

theorem ten_steps_climb : climb_stairs 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_steps_climb_l3594_359485


namespace NUMINAMATH_CALUDE_average_salary_proof_l3594_359471

theorem average_salary_proof (n : ℕ) (total_salary : ℕ → ℕ) : 
  (∃ (m : ℕ), m > 0 ∧ total_salary m / m = 8000) →
  total_salary 4 / 4 = 8450 →
  total_salary 1 = 6500 →
  total_salary 5 = 4700 →
  (total_salary 5 + (total_salary 4 - total_salary 1)) / 4 = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l3594_359471


namespace NUMINAMATH_CALUDE_no_solution_quadratic_with_constraint_l3594_359480

theorem no_solution_quadratic_with_constraint : 
  ¬ ∃ (x : ℝ), x^2 - 4*x + 4 = 0 ∧ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_with_constraint_l3594_359480


namespace NUMINAMATH_CALUDE_max_triangle_area_l3594_359445

/-- Given a triangle ABC with side lengths a, b, c and internal angles A, B, C,
    this theorem states that the maximum area of the triangle is √2 when
    a = √2, b² - c² = 6, and angle A is at its maximum. -/
theorem max_triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b^2 - c^2 = 6 →
  (∀ (a' b' c' : ℝ) (A' B' C' : ℝ),
    a' = Real.sqrt 2 →
    b'^2 - c'^2 = 6 →
    A' ≤ A) →
  (1/2 * b * c * Real.sin A) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3594_359445


namespace NUMINAMATH_CALUDE_red_balls_count_l3594_359443

theorem red_balls_count (total : ℕ) (white : ℕ) (red : ℕ) (prob_white : ℚ) : 
  white = 5 →
  total = white + red →
  prob_white = 1/4 →
  (white : ℚ) / total = prob_white →
  red = 15 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l3594_359443


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l3594_359487

/-- Sum of positive factors of a natural number -/
def sumOfFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- Theorem: The sum of all positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sumOfFactors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l3594_359487


namespace NUMINAMATH_CALUDE_smallest_consecutive_number_l3594_359414

theorem smallest_consecutive_number (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 4574880 → a = 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_number_l3594_359414


namespace NUMINAMATH_CALUDE_sum_areas_externally_tangent_circles_l3594_359476

/-- Given a 5-12-13 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 113π. -/
theorem sum_areas_externally_tangent_circles (r s t : ℝ) : 
  r + s = 5 →
  s + t = 12 →
  r + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_areas_externally_tangent_circles_l3594_359476


namespace NUMINAMATH_CALUDE_stating_min_swaps_equals_num_pairs_l3594_359424

/-- Represents the number of volumes in the encyclopedia --/
def n : ℕ := 30

/-- Represents a swap operation between adjacent volumes --/
def Swap : Type := Unit

/-- 
Represents the minimum number of swap operations required to guarantee 
correct ordering of n volumes from any initial arrangement 
--/
def minSwaps (n : ℕ) : ℕ := sorry

/-- Calculates the number of possible pairs from n elements --/
def numPairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- 
Theorem stating that the minimum number of swaps required is equal to 
the number of possible pairs of volumes
--/
theorem min_swaps_equals_num_pairs : 
  minSwaps n = numPairs n := by sorry

end NUMINAMATH_CALUDE_stating_min_swaps_equals_num_pairs_l3594_359424


namespace NUMINAMATH_CALUDE_leo_money_after_settling_debts_l3594_359457

/-- The total amount of money Leo and Ryan have together -/
def total_amount : ℚ := 48

/-- The fraction of the total amount that Ryan owns -/
def ryan_fraction : ℚ := 2/3

/-- The amount Ryan owes Leo -/
def ryan_owes_leo : ℚ := 10

/-- The amount Leo owes Ryan -/
def leo_owes_ryan : ℚ := 7

/-- Leo's final amount after settling debts -/
def leo_final_amount : ℚ := 19

theorem leo_money_after_settling_debts :
  let ryan_initial := ryan_fraction * total_amount
  let leo_initial := total_amount - ryan_initial
  let net_debt := ryan_owes_leo - leo_owes_ryan
  leo_initial + net_debt = leo_final_amount := by
sorry

end NUMINAMATH_CALUDE_leo_money_after_settling_debts_l3594_359457


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3594_359482

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3594_359482


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l3594_359496

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if f(-1) = 0, f(1) = 0, and f(0) = 2, then b = -2 -/
theorem cubic_function_coefficient (a b c d : ℝ) : 
  let f := λ x : ℝ => a * x^3 + b * x^2 + c * x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l3594_359496


namespace NUMINAMATH_CALUDE_correct_expression_l3594_359400

theorem correct_expression : 
  (-((-8 : ℝ) ^ (1/3 : ℝ)) = 2) ∧ 
  (Real.sqrt 9 ≠ 3 ∧ Real.sqrt 9 ≠ -3) ∧
  (-Real.sqrt 16 ≠ 4) ∧
  (Real.sqrt ((-2)^2) ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_l3594_359400


namespace NUMINAMATH_CALUDE_find_A_in_rounding_l3594_359448

theorem find_A_in_rounding : ∃ A : ℕ, 
  (A < 10) ∧ 
  (6000 + A * 100 + 35 ≥ 6100) ∧ 
  (6000 + (A + 1) * 100 + 35 > 6100) → 
  A = 1 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_rounding_l3594_359448


namespace NUMINAMATH_CALUDE_triangle_side_length_l3594_359423

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = π/3 → C = π/4 → 
  c = 10 * (Real.sqrt 3 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3594_359423


namespace NUMINAMATH_CALUDE_sticker_difference_l3594_359462

/-- Represents the distribution of stickers in boxes following an arithmetic sequence -/
structure StickerDistribution where
  total : ℕ
  boxes : ℕ
  first : ℕ
  difference : ℕ

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the difference between highest and lowest sticker quantities -/
theorem sticker_difference (dist : StickerDistribution)
  (h1 : dist.total = 250)
  (h2 : dist.boxes = 5)
  (h3 : dist.first = 30)
  (h4 : arithmeticSum dist.first dist.difference dist.boxes = dist.total) :
  dist.first + (dist.boxes - 1) * dist.difference - dist.first = 40 := by
  sorry

#check sticker_difference

end NUMINAMATH_CALUDE_sticker_difference_l3594_359462


namespace NUMINAMATH_CALUDE_parking_lot_increase_l3594_359438

def initial_cars : ℕ := 24
def final_cars : ℕ := 48

def percentage_increase (initial final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem parking_lot_increase :
  percentage_increase initial_cars final_cars = 100 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_increase_l3594_359438


namespace NUMINAMATH_CALUDE_min_guests_banquet_l3594_359412

def total_food : ℝ := 337
def max_food_per_guest : ℝ := 2

theorem min_guests_banquet :
  ∃ (min_guests : ℕ), 
    (min_guests : ℝ) * max_food_per_guest ≥ total_food ∧
    ∀ (n : ℕ), (n : ℝ) * max_food_per_guest ≥ total_food → n ≥ min_guests ∧
    min_guests = 169 :=
by sorry

end NUMINAMATH_CALUDE_min_guests_banquet_l3594_359412


namespace NUMINAMATH_CALUDE_expression_evaluation_l3594_359468

theorem expression_evaluation : (900^2 : ℝ) / (306^2 - 294^2) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3594_359468


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l3594_359410

-- Define an isosceles right triangle with rational hypotenuse
structure IsoscelesRightTriangle where
  hypotenuse : ℚ
  hypotenuse_positive : hypotenuse > 0

-- Define the area of the triangle
def area (t : IsoscelesRightTriangle) : ℚ :=
  t.hypotenuse ^ 2 / 4

-- Define the perimeter of the triangle
noncomputable def perimeter (t : IsoscelesRightTriangle) : ℝ :=
  t.hypotenuse * (2 + Real.sqrt 2)

-- Theorem statement
theorem isosceles_right_triangle_area_and_perimeter (t : IsoscelesRightTriangle) :
  (∃ q : ℚ, area t = q) ∧ (∀ q : ℚ, perimeter t ≠ q) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_and_perimeter_l3594_359410


namespace NUMINAMATH_CALUDE_B_power_98_l3594_359465

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_98 : B^98 = ![![-1, 0, 0],
                              ![0, -1, 0],
                              ![0, 0, 1]] := by sorry

end NUMINAMATH_CALUDE_B_power_98_l3594_359465


namespace NUMINAMATH_CALUDE_ab_greater_than_b_squared_l3594_359427

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_b_squared_l3594_359427


namespace NUMINAMATH_CALUDE_find_different_coins_possible_l3594_359466

/-- Represents the result of a weighing --/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a set of coins --/
structure CoinSet where
  total : Nat
  heavy : Nat
  light : Nat
  h_equal_light : heavy = light
  h_total : total = heavy + light

/-- Represents a weighing operation --/
def weigh (left right : CoinSet) : WeighResult :=
  sorry

/-- Represents the process of finding two coins of different weights --/
def findDifferentCoins (coins : CoinSet) (maxWeighings : Nat) : Bool :=
  sorry

/-- The main theorem to be proved --/
theorem find_different_coins_possible :
  ∃ (strategy : CoinSet → Nat → Bool),
    let initialCoins : CoinSet := {
      total := 128,
      heavy := 64,
      light := 64,
      h_equal_light := rfl,
      h_total := rfl
    }
    strategy initialCoins 7 = true :=
  sorry

end NUMINAMATH_CALUDE_find_different_coins_possible_l3594_359466


namespace NUMINAMATH_CALUDE_crayons_difference_proof_l3594_359455

-- Define the given conditions
def initial_crayons : ℕ := 4 * 8
def crayons_to_mae : ℕ := 5
def crayons_left : ℕ := 15

-- Define the theorem
theorem crayons_difference_proof : 
  (initial_crayons - crayons_to_mae - crayons_left) - crayons_to_mae = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_proof_l3594_359455
