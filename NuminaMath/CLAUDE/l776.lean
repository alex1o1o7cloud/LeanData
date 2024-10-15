import Mathlib

namespace NUMINAMATH_CALUDE_license_plate_count_l776_77690

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 20

/-- The number of vowels (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_consonants * num_digits * num_vowels * (num_consonants - 1)

/-- Theorem stating the number of possible license plates -/
theorem license_plate_count : num_license_plates = 22800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l776_77690


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_l776_77632

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis 
  (D : ℝ × ℝ) -- Point D in the plane
  (h : D = (3, -2)) -- D has coordinates (3, -2)
  : ‖D - (D.1, -D.2)‖ = 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_l776_77632


namespace NUMINAMATH_CALUDE_connor_test_scores_l776_77669

theorem connor_test_scores (test1 test2 test3 test4 : ℕ) : 
  test1 = 82 →
  test2 = 75 →
  test1 ≤ 100 ∧ test2 ≤ 100 ∧ test3 ≤ 100 ∧ test4 ≤ 100 →
  (test1 + test2 + test3 + test4) / 4 = 85 →
  (test3 = 83 ∧ test4 = 100) ∨ (test3 = 100 ∧ test4 = 83) :=
by sorry

end NUMINAMATH_CALUDE_connor_test_scores_l776_77669


namespace NUMINAMATH_CALUDE_log_equation_solution_l776_77685

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log x 81 = 4/2 → x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l776_77685


namespace NUMINAMATH_CALUDE_dave_book_spending_l776_77645

/-- The total amount Dave spent on books -/
def total_spent (animal_books animal_price space_books space_price train_books train_price history_books history_price science_books science_price : ℕ) : ℕ :=
  animal_books * animal_price + space_books * space_price + train_books * train_price + history_books * history_price + science_books * science_price

/-- Theorem stating the total amount Dave spent on books -/
theorem dave_book_spending :
  total_spent 8 10 6 12 9 8 4 15 5 18 = 374 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_spending_l776_77645


namespace NUMINAMATH_CALUDE_doug_money_l776_77633

/-- Represents the amount of money each person has -/
structure Money where
  josh : ℚ
  doug : ℚ
  brad : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.josh + m.doug + m.brad = 68 ∧
  m.josh = 2 * m.brad ∧
  m.josh = 3/4 * m.doug

/-- The theorem to prove -/
theorem doug_money (m : Money) (h : problem_conditions m) : m.doug = 32 := by
  sorry

end NUMINAMATH_CALUDE_doug_money_l776_77633


namespace NUMINAMATH_CALUDE_parabola_ratio_l776_77601

/-- A parabola passing through points (-1, 1) and (3, 1) has a/b = -2 --/
theorem parabola_ratio (a b c : ℝ) : 
  (a * (-1)^2 + b * (-1) + c = 1) → 
  (a * 3^2 + b * 3 + c = 1) → 
  a / b = -2 := by
sorry

end NUMINAMATH_CALUDE_parabola_ratio_l776_77601


namespace NUMINAMATH_CALUDE_bird_cages_count_l776_77617

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 48

/-- Theorem stating that the number of bird cages is correct given the conditions -/
theorem bird_cages_count :
  (parrots_per_cage + parakeets_per_cage) * num_cages = total_birds := by
  sorry

end NUMINAMATH_CALUDE_bird_cages_count_l776_77617


namespace NUMINAMATH_CALUDE_collinearity_iff_sum_one_l776_77630

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define points
variable (O A B P : V)

-- Define real numbers m and n
variable (m n : ℝ)

-- Define the condition that O, A, B are not collinear
def not_collinear (O A B : V) : Prop := 
  ∀ (t : ℝ), (B - O) ≠ t • (A - O)

-- Define the vector equation
def vector_equation (O A B P : V) (m n : ℝ) : Prop :=
  (P - O) = m • (A - O) + n • (B - O)

-- Define collinearity of points A, P, B
def collinear (A P B : V) : Prop :=
  ∃ (t : ℝ), (P - A) = t • (B - A)

-- State the theorem
theorem collinearity_iff_sum_one
  (h₁ : not_collinear O A B)
  (h₂ : vector_equation O A B P m n) :
  collinear A P B ↔ m + n = 1 := by sorry

end NUMINAMATH_CALUDE_collinearity_iff_sum_one_l776_77630


namespace NUMINAMATH_CALUDE_equation_solutions_l776_77687

open Real

-- Define the tangent function
noncomputable def tg (x : ℝ) : ℝ := tan x

-- Define the equation
def equation (x : ℝ) : Prop := tg x + tg (2*x) + tg (3*x) + tg (4*x) = 0

-- Define the set of solutions
def solution_set : Set ℝ := {0, π/7.2, π/5, π/3.186, π/2.5, -π/7.2, -π/5, -π/3.186, -π/2.5}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l776_77687


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l776_77622

theorem magical_red_knights_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (magical : ℕ) 
  (magical_red : ℕ) 
  (magical_blue : ℕ) 
  (h1 : red = (3 * total) / 8)
  (h2 : blue = total - red)
  (h3 : magical = total / 8)
  (h4 : magical_red * blue = 3 * magical_blue * red)
  (h5 : magical = magical_red + magical_blue) :
  magical_red * 14 = red * 3 := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l776_77622


namespace NUMINAMATH_CALUDE_car_average_speed_l776_77671

/-- Calculate the average speed of a car given its uphill and downhill speeds and distances --/
theorem car_average_speed (uphill_speed downhill_speed uphill_distance downhill_distance : ℝ) :
  uphill_speed = 30 →
  downhill_speed = 40 →
  uphill_distance = 100 →
  downhill_distance = 50 →
  let total_distance := uphill_distance + downhill_distance
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  let total_time := uphill_time + downhill_time
  let average_speed := total_distance / total_time
  average_speed = 1800 / 55 := by
  sorry

#eval (1800 : ℚ) / 55

end NUMINAMATH_CALUDE_car_average_speed_l776_77671


namespace NUMINAMATH_CALUDE_siwoo_cranes_per_hour_l776_77674

/-- The number of cranes Siwoo folds in 30 minutes -/
def cranes_per_30_min : ℕ := 180

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of cranes Siwoo folds in 1 hour -/
def cranes_per_hour : ℕ := cranes_per_30_min * (minutes_per_hour / 30)

/-- Theorem stating that Siwoo folds 360 cranes in 1 hour -/
theorem siwoo_cranes_per_hour :
  cranes_per_hour = 360 := by
  sorry

end NUMINAMATH_CALUDE_siwoo_cranes_per_hour_l776_77674


namespace NUMINAMATH_CALUDE_investment_percentage_l776_77602

/-- The investment problem with Vishal, Trishul, and Raghu -/
theorem investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →                  -- Vishal invested 10% more than Trishul
  raghu = 2200 →                            -- Raghu invested Rs. 2200
  vishal + trishul + raghu = 6358 →         -- Total sum of investments
  trishul < raghu →                         -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=   -- Percentage Trishul invested less than Raghu
by sorry

end NUMINAMATH_CALUDE_investment_percentage_l776_77602


namespace NUMINAMATH_CALUDE_concrete_mixture_cement_percentage_l776_77624

/-- Proves that given two types of concrete mixed in equal amounts to create a total mixture with a specific cement percentage, if one type has a known cement percentage, then the other type's cement percentage can be determined. -/
theorem concrete_mixture_cement_percentage 
  (total_weight : ℝ) 
  (final_cement_percentage : ℝ) 
  (weight_each_type : ℝ) 
  (cement_percentage_type1 : ℝ) :
  total_weight = 4500 →
  final_cement_percentage = 10.8 →
  weight_each_type = 1125 →
  cement_percentage_type1 = 10.8 →
  ∃ (cement_percentage_type2 : ℝ),
    cement_percentage_type2 = 32.4 ∧
    weight_each_type * cement_percentage_type1 / 100 + 
    weight_each_type * cement_percentage_type2 / 100 = 
    total_weight * final_cement_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_concrete_mixture_cement_percentage_l776_77624


namespace NUMINAMATH_CALUDE_solution_set_eq_four_points_l776_77609

/-- The set of solutions to the system of equations:
    a^4 - b^4 = c
    b^4 - c^4 = a
    c^4 - a^4 = b
    where a, b, c are real numbers. -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {abc | let (a, b, c) := abc
         a^4 - b^4 = c ∧
         b^4 - c^4 = a ∧
         c^4 - a^4 = b}

/-- The theorem stating that the solution set is equal to the given set of four points. -/
theorem solution_set_eq_four_points :
  SolutionSet = {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_four_points_l776_77609


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l776_77644

/-- Given a positive real number S representing the area of a rectangle,
    prove that the square with side length √S has the smallest perimeter
    among all rectangles with area S, and this minimum perimeter is 4√S. -/
theorem min_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x * y = S ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → a * b = S → 2*(x + y) ≤ 2*(a + b)) ∧
    x = Real.sqrt S ∧ y = Real.sqrt S ∧
    2*(x + y) = 4 * Real.sqrt S :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l776_77644


namespace NUMINAMATH_CALUDE_chess_group_age_sum_l776_77661

/-- Given 4 children and 2 coaches with specific age relationships, 
    prove that if the sum of squares of their ages is 2796, 
    then the sum of their ages is 94. -/
theorem chess_group_age_sum 
  (a : ℕ) -- age of the youngest child
  (b : ℕ) -- age of the younger coach
  (h1 : a^2 + (a+2)^2 + (a+4)^2 + (a+6)^2 + b^2 + (b+2)^2 = 2796) :
  a + (a+2) + (a+4) + (a+6) + b + (b+2) = 94 := by
sorry

end NUMINAMATH_CALUDE_chess_group_age_sum_l776_77661


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l776_77658

theorem gcd_of_powers_of_two : Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l776_77658


namespace NUMINAMATH_CALUDE_probability_theorem_l776_77653

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  al ≠ bill ∧ bill ≠ cal ∧ al ≠ cal

def satisfies_conditions (al bill cal : ℕ) : Prop :=
  is_multiple al bill ∧ is_multiple bill cal ∧ is_even (al + bill + cal)

def total_assignments : ℕ := 12 * 11 * 10

theorem probability_theorem :
  (∃ valid_count : ℕ,
    (∀ al bill cal : ℕ, valid_assignment al bill cal → satisfies_conditions al bill cal →
      valid_count > 0) ∧
    (valid_count : ℚ) / total_assignments = 2 / 110) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l776_77653


namespace NUMINAMATH_CALUDE_range_of_a_l776_77684

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 regarding x is an empty set. -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: Given the quadratic function f(x) = x^2 - mx + 2 satisfies f(3/2 + x) = f(3/2 - x),
    and its maximum value is 2 when x ∈ [0,a]. -/
def proposition_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m (3/2 + x) = f m (3/2 - x)) ∧
       (∀ x ∈ Set.Icc 0 a, f m x ≤ 2) ∧
       (∃ x ∈ Set.Icc 0 a, f m x = 2)

/-- The range of a given the logical conditions on p and q -/
theorem range_of_a :
  ∀ a : ℝ, (¬(proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a)) ↔
            a ∈ Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l776_77684


namespace NUMINAMATH_CALUDE_nell_gave_136_cards_to_jeff_l776_77682

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (original_cards : ℕ) (cards_left : ℕ) : ℕ :=
  original_cards - cards_left

/-- Proof that Nell gave 136 cards to Jeff -/
theorem nell_gave_136_cards_to_jeff :
  cards_given_to_jeff 242 106 = 136 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_136_cards_to_jeff_l776_77682


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_sum_of_A_and_B_proof_l776_77696

theorem sum_of_A_and_B : ℕ → ℕ → Prop :=
  fun A B =>
    (A < 10 ∧ B < 10) →  -- A and B are single digit numbers
    (A = 2 + 4) →        -- A is 4 greater than 2
    (B - 3 = 1) →        -- 3 less than B is 1
    A + B = 10           -- The sum of A and B is 10

-- Proof
theorem sum_of_A_and_B_proof : sum_of_A_and_B 6 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_sum_of_A_and_B_proof_l776_77696


namespace NUMINAMATH_CALUDE_hours_worked_per_day_l776_77681

theorem hours_worked_per_day 
  (total_hours : ℕ) 
  (weeks_worked : ℕ) 
  (h1 : total_hours = 140) 
  (h2 : weeks_worked = 4) :
  (total_hours : ℚ) / (weeks_worked * 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hours_worked_per_day_l776_77681


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l776_77662

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(m-3)*x + 16 = (x - a)^2) → 
  (m = 7 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l776_77662


namespace NUMINAMATH_CALUDE_sqrt_and_pi_comparisons_l776_77680

theorem sqrt_and_pi_comparisons : 
  (Real.sqrt 2 < Real.sqrt 3) ∧ (3.14 < Real.pi) := by sorry

end NUMINAMATH_CALUDE_sqrt_and_pi_comparisons_l776_77680


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l776_77628

theorem complex_expression_simplification :
  let c : ℂ := 3 + 2*I
  let d : ℂ := -2 - I
  3*c + 4*d = 1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l776_77628


namespace NUMINAMATH_CALUDE_coin_value_equality_l776_77677

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The theorem stating the equality of coin values -/
theorem coin_value_equality (n : ℕ) : 
  15 * coin_value "quarter" + 20 * coin_value "dime" = 
  10 * coin_value "quarter" + n * coin_value "dime" + 5 * coin_value "nickel" → 
  n = 30 := by
  sorry

#check coin_value_equality

end NUMINAMATH_CALUDE_coin_value_equality_l776_77677


namespace NUMINAMATH_CALUDE_sin_cos_difference_l776_77643

theorem sin_cos_difference (x y : Real) : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l776_77643


namespace NUMINAMATH_CALUDE_teaspoon_knife_ratio_l776_77614

/-- Proves that the ratio of initial teaspoons to initial knives is 2:1 --/
theorem teaspoon_knife_ratio : 
  ∀ (initial_teaspoons : ℕ),
  let initial_knives : ℕ := 24
  let additional_knives : ℕ := initial_knives / 3
  let additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3
  let total_cutlery : ℕ := 112
  (initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery) →
  (initial_teaspoons : ℚ) / initial_knives = 2 := by
  sorry

end NUMINAMATH_CALUDE_teaspoon_knife_ratio_l776_77614


namespace NUMINAMATH_CALUDE_direction_vector_value_l776_77625

/-- A line with direction vector (a, 4) passing through points (-2, 3) and (3, 5) has a = 10 -/
theorem direction_vector_value (a : ℝ) : 
  let v : ℝ × ℝ := (a, 4)
  let p₁ : ℝ × ℝ := (-2, 3)
  let p₂ : ℝ × ℝ := (3, 5)
  (∃ (t : ℝ), p₂ = p₁ + t • v) → a = 10 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_value_l776_77625


namespace NUMINAMATH_CALUDE_matematika_arrangements_l776_77664

/-- The number of distinct letters in "MATEMATIKA" excluding "A" -/
def n : ℕ := 7

/-- The number of repeated letters (M and T) -/
def r : ℕ := 2

/-- The number of "A"s in "MATEMATIKA" -/
def a : ℕ := 3

/-- The number of positions to place "A"s -/
def p : ℕ := n + 1

theorem matematika_arrangements : 
  (n.factorial / (r.factorial * r.factorial)) * Nat.choose p a = 70560 := by
  sorry

end NUMINAMATH_CALUDE_matematika_arrangements_l776_77664


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l776_77660

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a^3 + b^3 = 81) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l776_77660


namespace NUMINAMATH_CALUDE_dwarf_milk_problem_l776_77648

/-- Represents the amount of milk in each cup after a dwarf pours -/
def milk_distribution (initial_amount : ℚ) (k : Fin 7) : ℚ :=
  initial_amount * k / 6

/-- The total amount of milk after all distributions -/
def total_milk (initial_amount : ℚ) : ℚ :=
  (Finset.sum Finset.univ (milk_distribution initial_amount)) + initial_amount

theorem dwarf_milk_problem (initial_amount : ℚ) :
  (∀ (k : Fin 7), milk_distribution initial_amount k ≤ initial_amount) →
  total_milk initial_amount = 3 →
  initial_amount = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_dwarf_milk_problem_l776_77648


namespace NUMINAMATH_CALUDE_root_of_two_quadratics_l776_77663

theorem root_of_two_quadratics (a b c d : ℂ) (k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^2 + b * k + c = 0)
  (hk2 : b * k^2 + c * k + d = 0) :
  k = 1 ∨ k = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k = (-1 - Complex.I * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_root_of_two_quadratics_l776_77663


namespace NUMINAMATH_CALUDE_abs_neg_two_fifths_l776_77619

theorem abs_neg_two_fifths : |(-2 : ℚ) / 5| = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_fifths_l776_77619


namespace NUMINAMATH_CALUDE_complex_difference_on_unit_circle_l776_77699

theorem complex_difference_on_unit_circle (z₁ z₂ : ℂ) : 
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≤ Complex.abs (z₁ + 1 + Complex.I)) →
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≥ Complex.abs (z₂ + 1 + Complex.I)) →
  Complex.abs z₁ = 1 →
  Complex.abs z₂ = 1 →
  z₁ - z₂ = Complex.mk (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_difference_on_unit_circle_l776_77699


namespace NUMINAMATH_CALUDE_office_supplies_cost_l776_77616

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_quantity : ℕ := 24  -- two dozen
def folder_quantity : ℕ := 20

def total_cost : ℝ := pencil_cost * pencil_quantity + folder_cost * folder_quantity

theorem office_supplies_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_office_supplies_cost_l776_77616


namespace NUMINAMATH_CALUDE_negative_cube_root_of_negative_eight_equals_two_l776_77649

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem negative_cube_root_of_negative_eight_equals_two :
  -cubeRoot (-8) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_cube_root_of_negative_eight_equals_two_l776_77649


namespace NUMINAMATH_CALUDE_symmetry_implies_periodicity_l776_77659

def is_symmetrical_about (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x = 2 * b - f (2 * a - x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity 
  (f : ℝ → ℝ) (a b c d : ℝ) (h1 : a ≠ c) 
  (h2 : is_symmetrical_about f a b) 
  (h3 : is_symmetrical_about f c d) : 
  is_periodic f (2 * |a - c|) :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_periodicity_l776_77659


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l776_77670

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 + 5*x + 6 < 0
def inequality2 (x : ℝ) : Prop := -x^2 + 9*x - 20 < 0
def inequality3 (x : ℝ) : Prop := x^2 + x - 56 < 0
def inequality4 (x : ℝ) : Prop := 9*x^2 + 4 < 12*x

-- State the theorems
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ -3 < x ∧ x < -2 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x < 4 ∨ x > 5 := by sorry

theorem inequality3_solution : 
  ∀ x : ℝ, inequality3 x ↔ -8 < x ∧ x < 7 := by sorry

theorem inequality4_no_solution : 
  ¬∃ x : ℝ, inequality4 x := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l776_77670


namespace NUMINAMATH_CALUDE_percentage_calculation_l776_77641

theorem percentage_calculation (x y : ℝ) (P : ℝ) 
  (h1 : x / y = 4)
  (h2 : 0.8 * x = P / 100 * y) :
  P = 320 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l776_77641


namespace NUMINAMATH_CALUDE_ellipse_other_x_intercept_l776_77629

/-- Definition of an ellipse with given foci and one x-intercept -/
def Ellipse (f1 f2 x1 : ℝ × ℝ) : Prop :=
  let d1 (x y : ℝ) := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 (x y : ℝ) := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  ∀ x y : ℝ, d1 x y + d2 x y = d1 x1.1 x1.2 + d2 x1.1 x1.2

/-- The main theorem -/
theorem ellipse_other_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  let x1 : ℝ × ℝ := (1, 0)
  let x2 : ℝ × ℝ := ((13 - 14 * Real.sqrt 10) / (2 * Real.sqrt 10 + 14), 0)
  Ellipse f1 f2 x1 → x2.1 ≠ x1.1 → Ellipse f1 f2 x2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_other_x_intercept_l776_77629


namespace NUMINAMATH_CALUDE_central_square_area_central_square_area_proof_l776_77608

/-- Given a square with side length 6 composed of smaller squares with side length 2,
    the area of the central square formed by removing one small square from each corner is 20. -/
theorem central_square_area : ℕ → ℕ → ℕ → Prop :=
  fun large_side small_side central_area =>
    large_side = 6 ∧ 
    small_side = 2 ∧ 
    large_side % small_side = 0 ∧
    (large_side / small_side) ^ 2 - 4 = 5 ∧ 
    central_area = 5 * small_side ^ 2 ∧
    central_area = 20

/-- Proof of the theorem -/
theorem central_square_area_proof : central_square_area 6 2 20 := by
  sorry

end NUMINAMATH_CALUDE_central_square_area_central_square_area_proof_l776_77608


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l776_77651

/-- Given a hyperbola and a circle, prove that the x-coordinate of their intersection point in the first quadrant is (√3 + 1) / 2 -/
theorem hyperbola_circle_intersection (b c : ℝ) (P : ℝ × ℝ) : 
  let (x, y) := P
  (x^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (x^2 + y^2 = c^2) →      -- Circle equation
  (x > 0 ∧ y > 0) →        -- P is in the first quadrant
  ((x - c)^2 + y^2 = (c + 2)^2) →  -- |PF1| = c + 2
  (x = (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l776_77651


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l776_77646

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem inequality_solution (x : ℝ) : 
  f (x - 1) + f (1 - x) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

-- Theorem for part II
theorem inequality_proof (x a : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l776_77646


namespace NUMINAMATH_CALUDE_total_time_outside_class_l776_77620

def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

theorem total_time_outside_class :
  first_recess + second_recess + lunch + third_recess = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_outside_class_l776_77620


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_product_l776_77621

theorem unique_two_digit_sum_product : ∃! (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  10 * a + b = a + 2 * b + a * b :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_product_l776_77621


namespace NUMINAMATH_CALUDE_v_3003_equals_3_l776_77688

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for inputs not in the table

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_3003_equals_3 : v 3003 = 3 := by
  sorry

end NUMINAMATH_CALUDE_v_3003_equals_3_l776_77688


namespace NUMINAMATH_CALUDE_hall_paving_l776_77607

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width) / (stone_length * stone_width)

/-- Theorem: 1800 stones are required to pave a 36m x 15m hall with 6dm x 5dm stones -/
theorem hall_paving :
  stones_required 36 15 0.6 0.5 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_hall_paving_l776_77607


namespace NUMINAMATH_CALUDE_tangent_line_parallel_and_inequality_l776_77665

noncomputable def f (x : ℝ) := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) := f x + a / x - 1

theorem tangent_line_parallel_and_inequality (a : ℝ) :
  (∃ (m : ℝ), m = (1 / 2 : ℝ) - a / 4 ∧ m = -(1 / 2 : ℝ)) ∧
  (∀ (m n : ℝ), m > n → n > 0 → (m - n) / (m + n) < (Real.log m - Real.log n) / 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_and_inequality_l776_77665


namespace NUMINAMATH_CALUDE_abs_difference_given_product_and_sum_l776_77676

theorem abs_difference_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 6) 
  (h2 : a + b = 7) : 
  |a - b| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_given_product_and_sum_l776_77676


namespace NUMINAMATH_CALUDE_milk_butterfat_calculation_l776_77634

/-- Represents the butterfat percentage as a real number between 0 and 100 -/
def ButterfatPercentage := { x : ℝ // 0 ≤ x ∧ x ≤ 100 }

/-- Calculates the initial butterfat percentage of milk given the conditions -/
def initial_butterfat_percentage (
  initial_volume : ℝ) 
  (cream_volume : ℝ) 
  (cream_butterfat : ButterfatPercentage) 
  (final_butterfat : ButterfatPercentage) : ButterfatPercentage :=
  sorry

theorem milk_butterfat_calculation :
  let initial_volume : ℝ := 1000
  let cream_volume : ℝ := 50
  let cream_butterfat : ButterfatPercentage := ⟨23, by norm_num⟩
  let final_butterfat : ButterfatPercentage := ⟨3, by norm_num⟩
  let result := initial_butterfat_percentage initial_volume cream_volume cream_butterfat final_butterfat
  result.val = 4 := by sorry

end NUMINAMATH_CALUDE_milk_butterfat_calculation_l776_77634


namespace NUMINAMATH_CALUDE_existence_of_reduction_sequence_l776_77668

/-- The game operation: either multiply by 2 or remove the unit digit -/
inductive GameOperation
| multiply_by_two
| remove_unit_digit

/-- Apply a single game operation to a natural number -/
def apply_operation (n : ℕ) (op : GameOperation) : ℕ :=
  match op with
  | GameOperation.multiply_by_two => 2 * n
  | GameOperation.remove_unit_digit => n / 10

/-- Predicate to check if a sequence of operations reduces a number to 1 -/
def reduces_to_one (start : ℕ) (ops : List GameOperation) : Prop :=
  start ≠ 0 ∧ List.foldl apply_operation start ops = 1

/-- Theorem: For any non-zero natural number, there exists a sequence of operations that reduces it to 1 -/
theorem existence_of_reduction_sequence (n : ℕ) : 
  n ≠ 0 → ∃ (ops : List GameOperation), reduces_to_one n ops := by
  sorry


end NUMINAMATH_CALUDE_existence_of_reduction_sequence_l776_77668


namespace NUMINAMATH_CALUDE_total_movies_is_74_l776_77647

/-- Represents the number of movies watched by each person -/
structure MovieCounts where
  dalton : ℕ
  hunter : ℕ
  alex : ℕ
  bella : ℕ
  chris : ℕ

/-- Represents the number of movies watched by different groups -/
structure SharedMovies where
  all_five : ℕ
  dalton_hunter_alex : ℕ
  bella_chris : ℕ
  dalton_bella : ℕ
  alex_chris : ℕ

/-- Calculates the total number of different movies watched -/
def total_different_movies (individual : MovieCounts) (shared : SharedMovies) : ℕ :=
  individual.dalton + individual.hunter + individual.alex + individual.bella + individual.chris -
  (4 * shared.all_five + 2 * shared.dalton_hunter_alex + shared.bella_chris + shared.dalton_bella + shared.alex_chris)

/-- Theorem stating that the total number of different movies watched is 74 -/
theorem total_movies_is_74 (individual : MovieCounts) (shared : SharedMovies)
    (h1 : individual = ⟨20, 26, 35, 29, 16⟩)
    (h2 : shared = ⟨5, 4, 3, 2, 4⟩) :
    total_different_movies individual shared = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_is_74_l776_77647


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l776_77673

theorem quadratic_two_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) ↔ k < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l776_77673


namespace NUMINAMATH_CALUDE_car_y_win_probability_l776_77655

/-- The probability of car Y winning a race given specific conditions -/
theorem car_y_win_probability (total_cars : ℕ) (prob_x prob_z prob_xyz : ℝ) : 
  total_cars = 15 →
  prob_x = 1/4 →
  prob_z = 1/12 →
  prob_xyz = 0.4583333333333333 →
  ∃ (prob_y : ℝ), prob_y = 1/8 ∧ prob_x + prob_y + prob_z = prob_xyz :=
by sorry

end NUMINAMATH_CALUDE_car_y_win_probability_l776_77655


namespace NUMINAMATH_CALUDE_final_grasshoppers_count_l776_77642

/-- Represents the state of the cage --/
structure CageState where
  crickets : ℕ
  grasshoppers : ℕ

/-- Represents a magician's trick --/
inductive Trick
  | Red
  | Green

/-- Applies a single trick to the cage state --/
def applyTrick (state : CageState) (trick : Trick) : CageState :=
  match trick with
  | Trick.Red => CageState.mk (state.crickets + 1) (state.grasshoppers - 4)
  | Trick.Green => CageState.mk (state.crickets - 5) (state.grasshoppers + 2)

/-- Applies a sequence of tricks to the cage state --/
def applyTricks (state : CageState) (tricks : List Trick) : CageState :=
  tricks.foldl applyTrick state

theorem final_grasshoppers_count (tricks : List Trick) :
  tricks.length = 18 →
  (applyTricks (CageState.mk 30 30) tricks).crickets = 0 →
  (applyTricks (CageState.mk 30 30) tricks).grasshoppers = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_grasshoppers_count_l776_77642


namespace NUMINAMATH_CALUDE_correlation_theorem_l776_77638

-- Define the relation between x and y
def relation (x y : ℝ) : Prop := y = -0.1 * x + 1

-- Define positive correlation
def positively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x < a y ∧ b x < b y

-- Define negative correlation
def negatively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x > a y ∧ b x < b y

-- The main theorem
theorem correlation_theorem (x y z : ℝ → ℝ) 
  (h1 : ∀ t, relation (x t) (y t))
  (h2 : positively_correlated y z) :
  negatively_correlated x y ∧ negatively_correlated x z := by
  sorry

end NUMINAMATH_CALUDE_correlation_theorem_l776_77638


namespace NUMINAMATH_CALUDE_condition_relationship_l776_77611

theorem condition_relationship (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ (x y : ℝ), x + y > 4 ∧ (x ≤ 2 ∨ y ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l776_77611


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l776_77603

/-- A function that represents the cubic polynomial f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: ∀x ∈ ℝ, x²-4x+3m > 0 -/
def condition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0

/-- The condition q: f(x) is strictly increasing on (-∞,+∞) -/
def condition_q (m : ℝ) : Prop := StrictMono (f m)

/-- Theorem stating that p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m → condition_q m) ∧
  (∃ m : ℝ, condition_q m ∧ ¬condition_p m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l776_77603


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l776_77683

/-- The equation of a circle given the endpoints of its diameter -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-3, -1) →
  B = (5, 5) →
  ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ∃ t : ℝ, (x, y) = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l776_77683


namespace NUMINAMATH_CALUDE_unique_integer_l776_77618

theorem unique_integer (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_unique_integer_l776_77618


namespace NUMINAMATH_CALUDE_circle_area_equality_l776_77656

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 34) :
  ∃ r : ℝ, π * r^2 = π * (r₂^2 - r₁^2) ∧ r = 2 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l776_77656


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l776_77666

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y - x*y = 0 ∧ x + 2*y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l776_77666


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l776_77604

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l776_77604


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l776_77615

theorem quadratic_root_zero (a : ℝ) : 
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧ 
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l776_77615


namespace NUMINAMATH_CALUDE_power_inequality_l776_77612

theorem power_inequality (x : ℝ) (α : ℝ) (h1 : x > -1) :
  (0 < α ∧ α < 1 → (1 + x)^α ≤ 1 + α * x) ∧
  ((α < 0 ∨ α > 1) → (1 + x)^α ≥ 1 + α * x) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l776_77612


namespace NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l776_77606

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in a 30-day month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: No starting day results in equal Tuesdays and Fridays in a 30-day month -/
theorem no_equal_tuesdays_fridays :
  ∀ (startDay : DayOfWeek),
    countDayOccurrences startDay DayOfWeek.Tuesday ≠ 
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l776_77606


namespace NUMINAMATH_CALUDE_parabola_decreasing_for_positive_x_l776_77626

theorem parabola_decreasing_for_positive_x (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) :
  -x₂^2 + 3 < -x₁^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_for_positive_x_l776_77626


namespace NUMINAMATH_CALUDE_A_minus_2B_general_A_minus_2B_specific_l776_77667

-- Define the algebraic expressions A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - 5 * x * y - 2 * y^2
def B (x y : ℝ) : ℝ := x^2 - 3 * y

-- Theorem for part 1
theorem A_minus_2B_general (x y : ℝ) : 
  A x y - 2 * B x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by sorry

-- Theorem for part 2
theorem A_minus_2B_specific : 
  A 2 (-1) - 2 * B 2 (-1) = 6 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_general_A_minus_2B_specific_l776_77667


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l776_77694

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 14*m + 40 ≤ 0 → n ≤ m) ∧ n^2 - 14*n + 40 ≤ 0 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l776_77694


namespace NUMINAMATH_CALUDE_toothpick_100th_stage_l776_77605

/-- Arithmetic sequence with first term 4 and common difference 4 -/
def toothpick_sequence (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- The 100th term of the toothpick sequence is 400 -/
theorem toothpick_100th_stage : toothpick_sequence 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_100th_stage_l776_77605


namespace NUMINAMATH_CALUDE_sumata_vacation_miles_l776_77672

/-- The total miles driven on a vacation -/
def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

/-- Proof that the Sumata family drove 1250 miles on their vacation -/
theorem sumata_vacation_miles : 
  total_miles_driven 5.0 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_sumata_vacation_miles_l776_77672


namespace NUMINAMATH_CALUDE_inequality_system_solution_l776_77637

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l776_77637


namespace NUMINAMATH_CALUDE_equation_solution_set_l776_77678

-- Define the equation
def equation (x : ℝ) : Prop := Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)

-- Define the solution set
def solution_set : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}

-- Theorem statement
theorem equation_solution_set : {x : ℝ | equation x} = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l776_77678


namespace NUMINAMATH_CALUDE_odd_binomials_count_l776_77623

/-- The number of 1's in the binary representation of a natural number -/
def numOnes (n : ℕ) : ℕ := sorry

/-- The number of odd binomial coefficients in the n-th row of Pascal's triangle -/
def numOddBinomials (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd binomial coefficients in the n-th row of Pascal's triangle
    is equal to 2^k, where k is the number of 1's in the binary representation of n -/
theorem odd_binomials_count (n : ℕ) : numOddBinomials n = 2^(numOnes n) := by sorry

end NUMINAMATH_CALUDE_odd_binomials_count_l776_77623


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l776_77613

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 4 + 3 + 2
  let geometry_books : ℕ := 4
  let number_theory_books : ℕ := 3
  let algebra_books : ℕ := 2
  Nat.choose total_books geometry_books * 
  Nat.choose (total_books - geometry_books) number_theory_books * 
  Nat.choose (total_books - geometry_books - number_theory_books) algebra_books

theorem book_arrangement_proof : book_arrangement_count = 1260 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l776_77613


namespace NUMINAMATH_CALUDE_old_computer_wattage_is_1500_l776_77640

/-- The wattage of John's old computer --/
def old_computer_wattage : ℝ := 1500

/-- The price increase of electricity --/
def electricity_price_increase : ℝ := 0.25

/-- The wattage increase of the new computer compared to the old one --/
def new_computer_wattage_increase : ℝ := 0.5

/-- The old price of electricity in dollars per kilowatt-hour --/
def old_electricity_price : ℝ := 0.12

/-- The cost to run the old computer for 50 hours in dollars --/
def old_computer_cost_50_hours : ℝ := 9

/-- The number of hours the old computer runs --/
def run_hours : ℝ := 50

/-- Theorem stating that the old computer's wattage is 1500 watts --/
theorem old_computer_wattage_is_1500 :
  old_computer_wattage = 
    (old_computer_cost_50_hours / run_hours) / old_electricity_price * 1000 :=
by sorry

end NUMINAMATH_CALUDE_old_computer_wattage_is_1500_l776_77640


namespace NUMINAMATH_CALUDE_unique_solution_natural_system_l776_77600

theorem unique_solution_natural_system :
  ∃! (a b c d : ℕ), a * b = c + d ∧ c * d = a + b :=
by
  -- The unique solution is (2, 2, 2, 2)
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_natural_system_l776_77600


namespace NUMINAMATH_CALUDE_xyz_product_one_l776_77686

theorem xyz_product_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 1/x = 3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_one_l776_77686


namespace NUMINAMATH_CALUDE_seryozha_healthy_eating_days_l776_77698

/-- Represents the daily cookie consumption pattern -/
structure DailyCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Represents the total cookie consumption over a period -/
structure TotalCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Calculates the total cookies consumed over a period given the initial and final daily consumption -/
def calculateTotalCookies (initial final : DailyCookies) (days : ℕ) : TotalCookies :=
  { chocolate := (initial.chocolate + final.chocolate) * days / 2,
    sugarFree := (initial.sugarFree + final.sugarFree) * days / 2 }

/-- Theorem stating the number of days in Seryozha's healthy eating regimen -/
theorem seryozha_healthy_eating_days : 
  ∃ (initial : DailyCookies) (days : ℕ),
    let final : DailyCookies := ⟨initial.chocolate - (days - 1), initial.sugarFree + (days - 1)⟩
    let total : TotalCookies := calculateTotalCookies initial final days
    total.chocolate = 264 ∧ total.sugarFree = 187 ∧ days = 11 := by
  sorry


end NUMINAMATH_CALUDE_seryozha_healthy_eating_days_l776_77698


namespace NUMINAMATH_CALUDE_square_area_is_400_l776_77692

/-- A square cut into five rectangles of equal area -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one of the rectangles -/
  rect_width : ℝ
  /-- The number of rectangles the square is cut into -/
  num_rectangles : ℕ
  /-- The rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle -/
  given_width : ℝ
  /-- Condition: The number of rectangles is 5 -/
  h1 : num_rectangles = 5
  /-- Condition: The given width is 5 -/
  h2 : given_width = 5
  /-- Condition: The area of each rectangle is the total area divided by the number of rectangles -/
  h3 : equal_area = side^2 / num_rectangles
  /-- Condition: One of the rectangles has the given width -/
  h4 : rect_width = given_width

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_400_l776_77692


namespace NUMINAMATH_CALUDE_subtraction_result_l776_77679

theorem subtraction_result : 3.05 - 5.678 = -2.628 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l776_77679


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l776_77652

theorem circle_radius_is_two (r : ℝ) : r > 0 →
  3 * (2 * Real.pi * r) = 3 * (Real.pi * r^2) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l776_77652


namespace NUMINAMATH_CALUDE_modular_congruence_l776_77695

theorem modular_congruence (n : ℕ) : 
  0 ≤ n ∧ n < 31 ∧ (3 * n) % 31 = 1 → 
  (((2^n) ^ 3) - 2) % 31 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_l776_77695


namespace NUMINAMATH_CALUDE_opposite_silver_is_black_l776_77689

-- Define the colors
inductive Color
  | Yellow
  | Orange
  | Blue
  | Black
  | Silver
  | Pink

-- Define a face of the cube
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define a view of the cube
structure CubeView where
  top : Face
  front : Face
  right : Face

-- Define the theorem
theorem opposite_silver_is_black (c : Cube) 
  (view1 view2 view3 : CubeView)
  (h1 : c.top.color = Color.Black ∧ 
        c.right.color = Color.Blue)
  (h2 : view1.top.color = Color.Black ∧ 
        view1.front.color = Color.Pink ∧ 
        view1.right.color = Color.Blue)
  (h3 : view2.top.color = Color.Black ∧ 
        view2.front.color = Color.Orange ∧ 
        view2.right.color = Color.Blue)
  (h4 : view3.top.color = Color.Black ∧ 
        view3.front.color = Color.Yellow ∧ 
        view3.right.color = Color.Blue)
  (h5 : c.bottom.color = Color.Silver) :
  c.top.color = Color.Black :=
sorry

end NUMINAMATH_CALUDE_opposite_silver_is_black_l776_77689


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l776_77610

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l776_77610


namespace NUMINAMATH_CALUDE_expression_evaluation_l776_77639

theorem expression_evaluation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l776_77639


namespace NUMINAMATH_CALUDE_complement_union_M_N_l776_77627

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l776_77627


namespace NUMINAMATH_CALUDE_find_starting_number_l776_77657

theorem find_starting_number :
  ∀ n : ℤ,
  (300 : ℝ) = (n + 200 : ℝ) / 2 + 150 →
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_find_starting_number_l776_77657


namespace NUMINAMATH_CALUDE_average_first_16_even_numbers_l776_77635

theorem average_first_16_even_numbers : 
  let first_16_even : List ℕ := List.range 16 |>.map (fun n => 2 * (n + 1))
  (first_16_even.sum / first_16_even.length : ℚ) = 17 := by
sorry

end NUMINAMATH_CALUDE_average_first_16_even_numbers_l776_77635


namespace NUMINAMATH_CALUDE_equation_solution_l776_77631

theorem equation_solution (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l776_77631


namespace NUMINAMATH_CALUDE_second_digit_max_l776_77675

def original_number : ℚ := 0.123456789

-- Function to change a digit to 9 and swap with the next digit
def change_and_swap (n : ℚ) (pos : ℕ) : ℚ := sorry

-- Function to get the maximum value after change and swap operations
def max_after_change_and_swap (n : ℚ) : ℚ := sorry

-- Theorem stating that changing the second digit gives the maximum value
theorem second_digit_max :
  change_and_swap original_number 2 = max_after_change_and_swap original_number :=
sorry

end NUMINAMATH_CALUDE_second_digit_max_l776_77675


namespace NUMINAMATH_CALUDE_graph_of_2x_plus_5_is_straight_line_l776_77697

-- Define what it means for a function to be linear
def is_linear_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define what it means for a graph to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop := 
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- Define our specific function
def f : ℝ → ℝ := λ x => 2 * x + 5

-- State the theorem
theorem graph_of_2x_plus_5_is_straight_line :
  (∀ g : ℝ → ℝ, is_linear_function g → is_straight_line g) →
  is_linear_function f →
  is_straight_line f := by
  sorry

end NUMINAMATH_CALUDE_graph_of_2x_plus_5_is_straight_line_l776_77697


namespace NUMINAMATH_CALUDE_tournament_properties_l776_77691

structure Tournament :=
  (teams : ℕ)
  (scores : List ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)

def round_robin (t : Tournament) : Prop :=
  t.teams = 10 ∧ t.scores.length = 10 ∧ t.win_points = 3 ∧ t.draw_points = 1 ∧ t.loss_points = 0

theorem tournament_properties (t : Tournament) (h : round_robin t) :
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 1)).length = 2 * k) ∧
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 0)).length = 2 * k) ∧
  ¬(∃ a b c : ℕ, a < b ∧ b < c ∧ c < t.scores.length ∧ t.scores[a]! = 0 ∧ t.scores[b]! = 0 ∧ t.scores[c]! = 0) ∧
  (∃ scores : List ℕ, scores.length = 10 ∧ scores.sum < 135 ∧ round_robin ⟨10, scores, 3, 1, 0⟩) ∧
  (∃ m : ℕ, m ≥ 15 ∧ m ∈ t.scores) :=
by sorry

end NUMINAMATH_CALUDE_tournament_properties_l776_77691


namespace NUMINAMATH_CALUDE_farmers_children_l776_77650

/-- Represents the problem of determining the number of farmer's children --/
theorem farmers_children (apples_per_child : ℕ) (apples_eaten : ℕ) (apples_sold : ℕ) (apples_left : ℕ) : 
  apples_per_child = 15 → 
  apples_eaten = 8 → 
  apples_sold = 7 → 
  apples_left = 60 → 
  (apples_left + apples_eaten + apples_sold) / apples_per_child = 5 := by
  sorry

#check farmers_children

end NUMINAMATH_CALUDE_farmers_children_l776_77650


namespace NUMINAMATH_CALUDE_angle_B_is_45_degrees_l776_77693

theorem angle_B_is_45_degrees (A B : ℝ) 
  (h : 90 - (A + B) = 180 - (A - B)) : B = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_45_degrees_l776_77693


namespace NUMINAMATH_CALUDE_locus_equation_rectangle_perimeter_bound_l776_77654

-- Define the locus W
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 1/2)^2)}

-- Define a rectangle with three vertices on W
structure RectangleOnW where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  h_a_on_w : a ∈ W
  h_b_on_w : b ∈ W
  h_c_on_w : c ∈ W
  h_is_rectangle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∧
                   (a.1 - d.1) * (c.1 - d.1) + (a.2 - d.2) * (c.2 - d.2) = 0

-- Theorem statements
theorem locus_equation (p : ℝ × ℝ) :
  p ∈ W ↔ p.2 = p.1^2 + 1/4 := by sorry

theorem rectangle_perimeter_bound (rect : RectangleOnW) :
  let perimeter := 2 * (Real.sqrt ((rect.a.1 - rect.b.1)^2 + (rect.a.2 - rect.b.2)^2) +
                        Real.sqrt ((rect.b.1 - rect.c.1)^2 + (rect.b.2 - rect.c.2)^2))
  perimeter > 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_locus_equation_rectangle_perimeter_bound_l776_77654


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_6_or_28_l776_77636

theorem fraction_integer_iff_p_6_or_28 (p : ℕ+) :
  (∃ (n : ℕ+), (4 * p + 28 : ℚ) / (3 * p - 7) = n) ↔ p = 6 ∨ p = 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_6_or_28_l776_77636
