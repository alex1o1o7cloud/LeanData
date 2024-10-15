import Mathlib

namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_condition_l1160_116053

theorem abs_diff_eq_diff_abs_condition (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_condition_l1160_116053


namespace NUMINAMATH_CALUDE_towel_set_price_l1160_116095

/-- The price of towel sets for guest and master bathrooms -/
theorem towel_set_price (guest_sets master_sets : ℕ) (master_price : ℝ) 
  (discount : ℝ) (total_spent : ℝ) (h1 : guest_sets = 2) 
  (h2 : master_sets = 4) (h3 : master_price = 50) 
  (h4 : discount = 0.2) (h5 : total_spent = 224) : 
  ∃ (guest_price : ℝ), guest_price = 40 ∧ 
  (1 - discount) * (guest_sets * guest_price + master_sets * master_price) = total_spent :=
by
  sorry

#check towel_set_price

end NUMINAMATH_CALUDE_towel_set_price_l1160_116095


namespace NUMINAMATH_CALUDE_pizza_combinations_l1160_116062

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem pizza_combinations :
  Nat.choose number_of_toppings toppings_per_pizza = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l1160_116062


namespace NUMINAMATH_CALUDE_initial_average_problem_l1160_116061

theorem initial_average_problem (initial_count : Nat) (new_value : ℝ) (average_decrease : ℝ) :
  initial_count = 6 →
  new_value = 7 →
  average_decrease = 1 →
  ∃ initial_average : ℝ,
    initial_average * initial_count + new_value = 
    (initial_average - average_decrease) * (initial_count + 1) ∧
    initial_average = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_problem_l1160_116061


namespace NUMINAMATH_CALUDE_function_equality_l1160_116042

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x : ℝ, f x ≤ x
axiom cond2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y

-- State the theorem
theorem function_equality : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l1160_116042


namespace NUMINAMATH_CALUDE_triangle_count_is_36_l1160_116074

/-- A hexagon with diagonals and midpoint segments -/
structure HexagonWithDiagonalsAndMidpoints :=
  (vertices : Fin 6 → Point)
  (diagonals : List (Point × Point))
  (midpoint_segments : List (Point × Point))

/-- Count of triangles in the hexagon figure -/
def count_triangles (h : HexagonWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the count of triangles is 36 -/
theorem triangle_count_is_36 (h : HexagonWithDiagonalsAndMidpoints) : 
  count_triangles h = 36 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_36_l1160_116074


namespace NUMINAMATH_CALUDE_solve_system_l1160_116049

theorem solve_system (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1160_116049


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1160_116044

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  x^4 - 16 = (x-2)*(x+2)*(x^2+4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -9*x^2*y + 12*x*y^2 - 4*y^3 = -y*(3*x-2*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1160_116044


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l1160_116057

theorem least_five_digit_congruent_to_7_mod_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 → n ≥ 10015 :=
by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l1160_116057


namespace NUMINAMATH_CALUDE_intersection_S_complement_T_l1160_116000

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 4, 5, 7}

-- Theorem statement
theorem intersection_S_complement_T : S ∩ (U \ T) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_S_complement_T_l1160_116000


namespace NUMINAMATH_CALUDE_chicks_increase_l1160_116024

theorem chicks_increase (first_day : ℕ) (second_day : ℕ) : first_day = 23 → second_day = 12 → first_day + second_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_chicks_increase_l1160_116024


namespace NUMINAMATH_CALUDE_xy_equals_one_l1160_116008

theorem xy_equals_one (x y : ℝ) (h : x + y = 1/x + 1/y ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l1160_116008


namespace NUMINAMATH_CALUDE_sequence_not_periodic_l1160_116090

theorem sequence_not_periodic (x : ℝ) (h1 : x > 1) (h2 : ¬ ∃ n : ℤ, x = n) : 
  ¬ ∃ p : ℕ, ∀ n : ℕ, (⌊x^(n+1)⌋ - x * ⌊x^n⌋) = (⌊x^(n+1+p)⌋ - x * ⌊x^(n+p)⌋) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_periodic_l1160_116090


namespace NUMINAMATH_CALUDE_fish_eaten_l1160_116070

theorem fish_eaten (initial_fish : ℕ) (temp_added : ℕ) (exchanged : ℕ) (final_fish : ℕ)
  (h1 : initial_fish = 14)
  (h2 : temp_added = 2)
  (h3 : exchanged = 3)
  (h4 : final_fish = 11) :
  initial_fish - (final_fish - exchanged) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fish_eaten_l1160_116070


namespace NUMINAMATH_CALUDE_base7_calculation_l1160_116020

/-- Represents a number in base 7 --/
def Base7 : Type := Nat

/-- Converts a base 7 number to its decimal representation --/
def toDecimal (n : Base7) : Nat := sorry

/-- Converts a decimal number to its base 7 representation --/
def toBase7 (n : Nat) : Base7 := sorry

/-- Adds two base 7 numbers --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def subBase7 (a b : Base7) : Base7 := sorry

theorem base7_calculation : 
  let a := toBase7 2000
  let b := toBase7 1256
  let c := toBase7 345
  let d := toBase7 1042
  subBase7 (addBase7 (subBase7 a b) c) d = toBase7 0 := by sorry

end NUMINAMATH_CALUDE_base7_calculation_l1160_116020


namespace NUMINAMATH_CALUDE_min_participants_is_eleven_l1160_116082

/-- Represents the number of participants in each grade --/
structure Participants where
  fifth : Nat
  sixth : Nat
  seventh : Nat

/-- Checks if the given number of participants satisfies all conditions --/
def satisfiesConditions (n : Nat) (p : Participants) : Prop :=
  p.fifth + p.sixth + p.seventh = n ∧
  (25 * n < 100 * p.fifth) ∧ (100 * p.fifth < 35 * n) ∧
  (30 * n < 100 * p.sixth) ∧ (100 * p.sixth < 40 * n) ∧
  (35 * n < 100 * p.seventh) ∧ (100 * p.seventh < 45 * n)

/-- States that 11 is the minimum number of participants satisfying all conditions --/
theorem min_participants_is_eleven :
  ∃ (p : Participants), satisfiesConditions 11 p ∧
  ∀ (m : Nat) (q : Participants), m < 11 → ¬satisfiesConditions m q :=
by sorry

end NUMINAMATH_CALUDE_min_participants_is_eleven_l1160_116082


namespace NUMINAMATH_CALUDE_polynomial_properties_l1160_116073

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  a * x^2 + b * x + c

/-- The polynomial we want to prove about -/
def our_polynomial (x : ℂ) : ℂ :=
  QuadraticPolynomial 2 (-12) 20 x

theorem polynomial_properties :
  (our_polynomial (3 + Complex.I) = 0) ∧
  (∀ x : ℂ, our_polynomial x = 2 * x^2 + (-12) * x + 20) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1160_116073


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l1160_116045

theorem power_mod_thirteen : 7^1234 ≡ 4 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l1160_116045


namespace NUMINAMATH_CALUDE_cupcake_business_loan_payment_l1160_116063

/-- Calculates the monthly payment for a loan given the total loan amount, down payment, and loan term in years. -/
def calculate_monthly_payment (total_loan : ℕ) (down_payment : ℕ) (years : ℕ) : ℕ :=
  let amount_to_finance := total_loan - down_payment
  let months := years * 12
  amount_to_finance / months

/-- Proves that for a loan of $46,000 with a $10,000 down payment to be paid over 5 years, the monthly payment is $600. -/
theorem cupcake_business_loan_payment :
  calculate_monthly_payment 46000 10000 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_business_loan_payment_l1160_116063


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l1160_116001

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ x : ℂ, x^6 - 12*x^5 + A*x^4 + (-162)*x^3 + C*x^2 + D*x + 36 = 
      (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) * (x - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l1160_116001


namespace NUMINAMATH_CALUDE_simplify_square_root_l1160_116021

theorem simplify_square_root (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_l1160_116021


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_2d_l1160_116041

theorem cauchy_schwarz_inequality_2d (a₁ a₂ b₁ b₂ : ℝ) :
  a₁ * b₁ + a₂ * b₂ ≤ Real.sqrt (a₁^2 + a₂^2) * Real.sqrt (b₁^2 + b₂^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_2d_l1160_116041


namespace NUMINAMATH_CALUDE_optimal_price_for_max_revenue_l1160_116051

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating the optimal price for maximum revenue --/
theorem optimal_price_for_max_revenue :
  ∃ (p : ℝ), 0 < p ∧ p ≤ 37.5 ∧
  ∀ (q : ℝ), 0 < q → q ≤ 37.5 → R p ≥ R q ∧
  p = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_max_revenue_l1160_116051


namespace NUMINAMATH_CALUDE_probability_one_red_one_white_l1160_116047

/-- The probability of drawing 1 red ball and 1 white ball when drawing two balls with replacement 
    from a bag containing 2 red balls and 3 white balls is equal to 2/5. -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 2)
  (h_white : white_balls = 3) :
  (red_balls / total_balls) * (white_balls / total_balls) * 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_one_white_l1160_116047


namespace NUMINAMATH_CALUDE_intersection_x_difference_l1160_116033

/-- The difference between the x-coordinates of the intersection points of two parabolas -/
theorem intersection_x_difference (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 3*x^2 - 6*x + 5) 
  (h₂ : ∀ x, g x = -2*x^2 - 4*x + 6) : 
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 6 / 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_difference_l1160_116033


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l1160_116013

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def smallest_two_digit_multiple_of_5 (a : ℕ) : Prop :=
  is_two_digit a ∧ 5 ∣ a ∧ ∀ m : ℕ, is_two_digit m → 5 ∣ m → a ≤ m

def smallest_three_digit_multiple_of_7 (b : ℕ) : Prop :=
  is_three_digit b ∧ 7 ∣ b ∧ ∀ m : ℕ, is_three_digit m → 7 ∣ m → b ≤ m

theorem sum_of_smallest_multiples (a b : ℕ) :
  smallest_two_digit_multiple_of_5 a →
  smallest_three_digit_multiple_of_7 b →
  a + b = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l1160_116013


namespace NUMINAMATH_CALUDE_corvette_trip_speed_l1160_116014

theorem corvette_trip_speed (total_distance : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 640)
  (h2 : average_speed = 40) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_time_ratio := 3
  let first_half_speed := 
    (2 * total_distance * average_speed) / (total_distance + 2 * first_half_distance)
  have h3 : first_half_speed = 80 := by sorry
  first_half_speed

#check corvette_trip_speed

end NUMINAMATH_CALUDE_corvette_trip_speed_l1160_116014


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1160_116075

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/6 -/
theorem infinite_series_sum : 
  ∑' n : ℕ+, (3 * n - 2 : ℚ) / (n * (n + 1) * (n + 3)) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1160_116075


namespace NUMINAMATH_CALUDE_vasya_has_more_placements_l1160_116078

/-- Represents a chessboard --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a king placement on a board --/
def KingPlacement (b : Board) := Fin b.rows → Fin b.cols

/-- Predicate to check if a king placement is valid (no kings attack each other) --/
def IsValidPlacement (b : Board) (p : KingPlacement b) : Prop := sorry

/-- Number of valid king placements on a board --/
def NumValidPlacements (b : Board) (n : ℕ) : ℕ := sorry

/-- Petya's board --/
def PetyaBoard : Board := ⟨100, 50⟩

/-- Vasya's board (only white cells of a 100 × 100 checkerboard) --/
def VasyaBoard : Board := ⟨100, 50⟩

theorem vasya_has_more_placements :
  NumValidPlacements VasyaBoard 500 > NumValidPlacements PetyaBoard 500 := by
  sorry

end NUMINAMATH_CALUDE_vasya_has_more_placements_l1160_116078


namespace NUMINAMATH_CALUDE_sucrose_solution_volume_l1160_116068

/-- Given a sucrose solution where 60 cubic centimeters contain 6 grams of sucrose,
    prove that 100 cubic centimeters contain 10 grams of sucrose. -/
theorem sucrose_solution_volume (solution_volume : ℝ) (sucrose_mass : ℝ) :
  (60 : ℝ) / solution_volume = 6 / sucrose_mass →
  (100 : ℝ) / solution_volume = 10 / sucrose_mass :=
by
  sorry

end NUMINAMATH_CALUDE_sucrose_solution_volume_l1160_116068


namespace NUMINAMATH_CALUDE_equation_solution_l1160_116085

theorem equation_solution (k x m n : ℝ) :
  (∃ x, ∀ k, 2 * k * x + 2 * m = 6 - 2 * x + n * k) →
  4 * m + 2 * n = 12 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1160_116085


namespace NUMINAMATH_CALUDE_sum_of_roots_l1160_116069

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0)
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1160_116069


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1160_116087

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : (R^2) / (r^2) = 4) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1160_116087


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1160_116022

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1160_116022


namespace NUMINAMATH_CALUDE_sloth_shoe_pairs_needed_l1160_116052

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for the sloth -/
def shoes_per_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets the sloth already owns -/
def owned_sets : ℕ := 1

/-- Represents the total number of complete sets the sloth needs -/
def total_sets_needed : ℕ := 5

/-- Theorem stating the number of pairs of shoes the sloth needs to buy -/
theorem sloth_shoe_pairs_needed :
  (total_sets_needed - owned_sets) * shoes_per_set / shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_CALUDE_sloth_shoe_pairs_needed_l1160_116052


namespace NUMINAMATH_CALUDE_barney_towel_count_l1160_116043

/-- The number of towels Barney owns -/
def num_towels : ℕ := 18

/-- The number of towels Barney uses per day -/
def towels_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Barney can use clean towels before running out -/
def days_before_running_out : ℕ := 9

/-- Theorem stating that Barney owns 18 towels -/
theorem barney_towel_count : 
  num_towels = towels_per_day * days_before_running_out :=
by sorry

end NUMINAMATH_CALUDE_barney_towel_count_l1160_116043


namespace NUMINAMATH_CALUDE_f_seven_equals_negative_two_l1160_116018

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_seven_equals_negative_two :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 4) = f x) →  -- f has period 4
  (∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) →  -- f(x) = 2x^2 for x in (0,2)
  f 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_negative_two_l1160_116018


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1160_116096

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 55053 →
  divisor = 456 →
  quotient = 120 →
  dividend = divisor * quotient + remainder →
  remainder = 333 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1160_116096


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1160_116012

-- Define the sample space
def SampleSpace := Finset (Fin 4 × Fin 4)

-- Define the event of selecting exactly one girl
def exactlyOneGirl (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∨ p.2 > 1)).card = 1

-- Define the event of selecting exactly two girls
def exactlyTwoGirls (s : SampleSpace) : Prop :=
  (s.card = 2) ∧ (s.filter (λ p => p.1 > 1 ∧ p.2 > 1)).card = 2

-- State the theorem
theorem mutually_exclusive_but_not_complementary :
  (∀ s : SampleSpace, ¬(exactlyOneGirl s ∧ exactlyTwoGirls s)) ∧
  (∃ s : SampleSpace, ¬(exactlyOneGirl s ∨ exactlyTwoGirls s)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1160_116012


namespace NUMINAMATH_CALUDE_meals_given_away_l1160_116039

theorem meals_given_away (initial_meals : ℕ) (additional_meals : ℕ) (meals_left : ℕ) : 
  initial_meals = 113 → additional_meals = 50 → meals_left = 78 → 
  initial_meals + additional_meals - meals_left = 85 := by
  sorry

end NUMINAMATH_CALUDE_meals_given_away_l1160_116039


namespace NUMINAMATH_CALUDE_class_overall_score_l1160_116028

/-- Calculates the overall score for a class based on four aspects --/
def calculate_overall_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the overall score for the given class is 84 --/
theorem class_overall_score :
  calculate_overall_score 85 90 80 75 = 84 := by
  sorry

#eval calculate_overall_score 85 90 80 75

end NUMINAMATH_CALUDE_class_overall_score_l1160_116028


namespace NUMINAMATH_CALUDE_rollo_guinea_pigs_food_l1160_116048

/-- The amount of food needed to feed all guinea pigs -/
def total_food (first_pig_food second_pig_food third_pig_food : ℕ) : ℕ :=
  first_pig_food + second_pig_food + third_pig_food

/-- Theorem stating the total amount of food needed for Rollo's guinea pigs -/
theorem rollo_guinea_pigs_food :
  ∃ (first_pig_food second_pig_food third_pig_food : ℕ),
    first_pig_food = 2 ∧
    second_pig_food = 2 * first_pig_food ∧
    third_pig_food = second_pig_food + 3 ∧
    total_food first_pig_food second_pig_food third_pig_food = 13 :=
by
  sorry

#check rollo_guinea_pigs_food

end NUMINAMATH_CALUDE_rollo_guinea_pigs_food_l1160_116048


namespace NUMINAMATH_CALUDE_original_bales_count_l1160_116058

/-- The number of bales Jason stacked today -/
def bales_stacked : ℕ := 23

/-- The total number of bales in the barn after Jason stacked -/
def total_bales : ℕ := 96

/-- The original number of bales in the barn -/
def original_bales : ℕ := total_bales - bales_stacked

theorem original_bales_count : original_bales = 73 := by
  sorry

end NUMINAMATH_CALUDE_original_bales_count_l1160_116058


namespace NUMINAMATH_CALUDE_movie_admission_price_l1160_116054

theorem movie_admission_price (regular_price : ℝ) : 
  (∀ discounted_price : ℝ, 
    discounted_price = regular_price - 3 →
    6 * discounted_price = 30) →
  regular_price = 8 := by
sorry

end NUMINAMATH_CALUDE_movie_admission_price_l1160_116054


namespace NUMINAMATH_CALUDE_complex_equality_implies_real_value_l1160_116071

theorem complex_equality_implies_real_value (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_real_value_l1160_116071


namespace NUMINAMATH_CALUDE_equation_solutions_l1160_116031

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 2)} ∪ {(k, 2, 3*k) | k : ℕ} ∪ {(2, 3, 18)} ∪ {(1, 2*k, 3*k) | k : ℕ} ∪ {(2, 2, 6)}

theorem equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x > 0 ∧ y > 0 ∧ z > 0 ∧ (1 : ℚ) / x + 2 / y - 3 / z = 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1160_116031


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l1160_116060

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l1160_116060


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l1160_116080

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_has_20_sides_l1160_116080


namespace NUMINAMATH_CALUDE_spring_properties_l1160_116088

-- Define the spring's properties
def initial_length : ℝ := 20
def rate_of_change : ℝ := 0.5

-- Define the relationship between weight and length
def spring_length (weight : ℝ) : ℝ := initial_length + rate_of_change * weight

-- Theorem stating the properties of the spring
theorem spring_properties :
  (∀ w : ℝ, w ≥ 0 → spring_length w ≥ initial_length) ∧
  (∀ w1 w2 : ℝ, w1 < w2 → spring_length w1 < spring_length w2) ∧
  (∀ w : ℝ, (spring_length (w + 1) - spring_length w) = rate_of_change) :=
by sorry

end NUMINAMATH_CALUDE_spring_properties_l1160_116088


namespace NUMINAMATH_CALUDE_walters_age_l1160_116027

theorem walters_age (walter_age_1994 : ℝ) (grandmother_age_1994 : ℝ) : 
  walter_age_1994 = grandmother_age_1994 / 3 →
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = 3750 →
  walter_age_1994 + 6 = 65.5 := by
sorry

end NUMINAMATH_CALUDE_walters_age_l1160_116027


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l1160_116002

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children_per_family : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children_per_family) / (total_families - childless_families) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l1160_116002


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_120_l1160_116072

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def sum_of_coefficients : ℕ :=
  (Finset.range 8).sum (fun i => binomial_coefficient (i + 2) 2)

theorem sum_of_coefficients_eq_120 : sum_of_coefficients = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_120_l1160_116072


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l1160_116040

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * (-x) + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l1160_116040


namespace NUMINAMATH_CALUDE_pens_multiple_of_ten_l1160_116098

/-- Given that 920 pencils and some pens can be distributed equally among 10 students,
    prove that the number of pens must be a multiple of 10. -/
theorem pens_multiple_of_ten (num_pens : ℕ) (h : ∃ (pens_per_student : ℕ), num_pens = 10 * pens_per_student) :
  ∃ k : ℕ, num_pens = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_pens_multiple_of_ten_l1160_116098


namespace NUMINAMATH_CALUDE_malcolm_lights_problem_l1160_116009

theorem malcolm_lights_problem (initial_white : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : initial_white = 59)
  (h2 : red = 12)
  (h3 : green = 6) :
  initial_white - (red + 3 * red + green) = 5 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_lights_problem_l1160_116009


namespace NUMINAMATH_CALUDE_susan_homework_start_time_l1160_116006

def volleyball_practice_start : Nat := 16 * 60 -- 4:00 p.m. in minutes since midnight

def homework_duration : Nat := 96 -- in minutes

def time_between_homework_and_practice : Nat := 25 -- in minutes

def homework_start_time : Nat := volleyball_practice_start - homework_duration - time_between_homework_and_practice

theorem susan_homework_start_time :
  homework_start_time = 13 * 60 + 59 -- 1:59 p.m. in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_susan_homework_start_time_l1160_116006


namespace NUMINAMATH_CALUDE_count_solutions_eq_two_l1160_116026

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem count_solutions_eq_two :
  (∃ (A : Finset ℕ), (∀ n ∈ A, n + S n + S (S n) = 2050) ∧ A.card = 2 ∧
   ∀ n : ℕ, n + S n + S (S n) = 2050 → n ∈ A) := by sorry

end NUMINAMATH_CALUDE_count_solutions_eq_two_l1160_116026


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1160_116046

theorem largest_angle_in_pentagon (P Q R S T : ℝ) : 
  P = 70 → 
  Q = 100 → 
  R = S → 
  T = 3 * R - 25 → 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 212 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1160_116046


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1160_116093

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_sum : a 2 + a 4 = 3) 
  (h_product : a 3 * a 5 = 2) : 
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1160_116093


namespace NUMINAMATH_CALUDE_students_not_taking_languages_l1160_116003

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28)
  (h2 : french = 5)
  (h3 : spanish = 10)
  (h4 : both = 4) :
  total - (french + spanish - both) = 17 := by
  sorry

#check students_not_taking_languages

end NUMINAMATH_CALUDE_students_not_taking_languages_l1160_116003


namespace NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l1160_116023

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits inside a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width ∧ c.height ≤ d.height) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.width) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.length)

/-- The main theorem stating that the maximum radius of a cylinder that fits in the given crate is 1.5 feet -/
theorem max_cylinder_radius_in_crate :
  let d := CrateDimensions.mk 3 8 12
  ∀ c : Cylinder, cylinderFitsInCrate c d → c.radius ≤ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l1160_116023


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l1160_116065

/-- The total cost to replace a floor given the room dimensions, removal cost, and new flooring cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l1160_116065


namespace NUMINAMATH_CALUDE_mike_total_cards_l1160_116016

/-- The total number of baseball cards Mike has after his birthday -/
def total_cards (initial_cards birthday_cards : ℕ) : ℕ :=
  initial_cards + birthday_cards

/-- Theorem stating that Mike has 82 cards in total -/
theorem mike_total_cards : 
  total_cards 64 18 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_cards_l1160_116016


namespace NUMINAMATH_CALUDE_prove_train_car_capacity_l1160_116025

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The additional passengers a train can carry compared to 2 airplanes -/
def additional_passengers : ℕ := 228

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

theorem prove_train_car_capacity : 
  train_car_capacity * train_cars = 2 * airplane_capacity + additional_passengers :=
sorry

end NUMINAMATH_CALUDE_prove_train_car_capacity_l1160_116025


namespace NUMINAMATH_CALUDE_paint_usage_l1160_116055

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (total_used : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 4)
  (h3 : total_used = 225) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_after_first_week := total_paint - first_week_usage
  let second_week_usage := total_used - first_week_usage
  second_week_usage / remaining_after_first_week = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l1160_116055


namespace NUMINAMATH_CALUDE_min_framing_for_picture_l1160_116029

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- The minimum number of linear feet of framing needed for the given picture specifications -/
theorem min_framing_for_picture : min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_picture_l1160_116029


namespace NUMINAMATH_CALUDE_sum_equals_three_halves_l1160_116059

theorem sum_equals_three_halves : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13 + 1 / 15
  let removed_terms := 1 / 13 + 1 / 15
  original_sum - removed_terms = 3 / 2 →
  (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_three_halves_l1160_116059


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1160_116083

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series : ℕ → ℝ := fun n =>
    1 / ((n - 1) * a - (n - 3) * b) / (n * a - (2 * n - 3) * b)
  ∑' n, series n = 1 / ((a - b) * b) :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1160_116083


namespace NUMINAMATH_CALUDE_missing_panels_l1160_116036

/-- Calculates the number of missing solar panels in Faith's neighborhood. -/
theorem missing_panels (total_homes : Nat) (panels_per_home : Nat) (homes_with_panels : Nat) :
  total_homes = 20 →
  panels_per_home = 10 →
  homes_with_panels = 15 →
  total_homes * panels_per_home - homes_with_panels * panels_per_home = 50 :=
by
  sorry

#check missing_panels

end NUMINAMATH_CALUDE_missing_panels_l1160_116036


namespace NUMINAMATH_CALUDE_average_weight_problem_l1160_116066

theorem average_weight_problem (d e f : ℝ) 
  (h1 : (d + e) / 2 = 35)
  (h2 : (e + f) / 2 = 41)
  (h3 : e = 26) :
  (d + e + f) / 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1160_116066


namespace NUMINAMATH_CALUDE_hazel_walk_l1160_116010

/-- The distance Hazel walked in the first hour -/
def first_hour_distance : ℝ := 2

/-- The distance Hazel walked in the second hour -/
def second_hour_distance (x : ℝ) : ℝ := 2 * x

/-- The total distance Hazel walked in 2 hours -/
def total_distance : ℝ := 6

theorem hazel_walk :
  first_hour_distance + second_hour_distance first_hour_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_hazel_walk_l1160_116010


namespace NUMINAMATH_CALUDE_odd_functions_identification_l1160_116004

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given functions
def F1 (f : RealFunction) : RealFunction := fun x ↦ -|f x|
def F2 (f : RealFunction) : RealFunction := fun x ↦ x * f (x^2)
def F3 (f : RealFunction) : RealFunction := fun x ↦ -f (-x)
def F4 (f : RealFunction) : RealFunction := fun x ↦ f x - f (-x)

-- State the theorem
theorem odd_functions_identification (f : RealFunction) :
  ¬IsOdd (F1 f) ∧ IsOdd (F2 f) ∧ IsOdd (F4 f) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_identification_l1160_116004


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1160_116079

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
sorry

/-- The area of a ring formed by two concentric circles with radii 10 and 6 is 64π -/
theorem area_of_specific_ring : 
  (π * 10^2 - π * 6^2 : ℝ) = 64 * π :=
sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1160_116079


namespace NUMINAMATH_CALUDE_machine_net_worth_l1160_116067

/-- Calculate the total net worth of a machine after 2 years given depreciation and maintenance costs -/
theorem machine_net_worth 
  (initial_value : ℝ)
  (depreciation_rate : ℝ)
  (initial_maintenance_cost : ℝ)
  (maintenance_increase_rate : ℝ)
  (h1 : initial_value = 40000)
  (h2 : depreciation_rate = 0.05)
  (h3 : initial_maintenance_cost = 2000)
  (h4 : maintenance_increase_rate = 0.03) :
  let value_after_year_1 := initial_value * (1 - depreciation_rate)
  let value_after_year_2 := value_after_year_1 * (1 - depreciation_rate)
  let maintenance_cost_year_1 := initial_maintenance_cost
  let maintenance_cost_year_2 := initial_maintenance_cost * (1 + maintenance_increase_rate)
  let total_maintenance_cost := maintenance_cost_year_1 + maintenance_cost_year_2
  let net_worth := value_after_year_2 - total_maintenance_cost
  net_worth = 32040 := by
  sorry


end NUMINAMATH_CALUDE_machine_net_worth_l1160_116067


namespace NUMINAMATH_CALUDE_intersection_sum_l1160_116076

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) → 
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) → 
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1160_116076


namespace NUMINAMATH_CALUDE_power_division_equals_one_l1160_116019

theorem power_division_equals_one (a : ℝ) (h : a ≠ 0) : a^5 / a^5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_one_l1160_116019


namespace NUMINAMATH_CALUDE_purple_valley_skirts_l1160_116015

def azure_skirts : ℕ := 60

def seafoam_skirts (azure : ℕ) : ℕ := (2 * azure) / 3

def purple_skirts (seafoam : ℕ) : ℕ := seafoam / 4

theorem purple_valley_skirts : 
  purple_skirts (seafoam_skirts azure_skirts) = 10 := by
  sorry

end NUMINAMATH_CALUDE_purple_valley_skirts_l1160_116015


namespace NUMINAMATH_CALUDE_sin_cos_sum_greater_than_one_l1160_116056

theorem sin_cos_sum_greater_than_one (α : Real) (h : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin α + Real.cos α > 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_greater_than_one_l1160_116056


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1160_116077

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt (8 / 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C' * (x + y + z)) → C' ≤ C) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1160_116077


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1160_116094

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I) / z = 4 + 2 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1160_116094


namespace NUMINAMATH_CALUDE_chips_note_taking_schedule_l1160_116007

/-- Chip's note-taking schedule --/
theorem chips_note_taking_schedule 
  (pages_per_class : ℕ) 
  (num_classes : ℕ) 
  (sheets_per_pack : ℕ) 
  (num_weeks : ℕ) 
  (packs_used : ℕ) 
  (h1 : pages_per_class = 2)
  (h2 : num_classes = 5)
  (h3 : sheets_per_pack = 100)
  (h4 : num_weeks = 6)
  (h5 : packs_used = 3) :
  (packs_used * sheets_per_pack) / (pages_per_class * num_classes * num_weeks) = 5 := by
  sorry

#check chips_note_taking_schedule

end NUMINAMATH_CALUDE_chips_note_taking_schedule_l1160_116007


namespace NUMINAMATH_CALUDE_polygon_arrangement_exists_l1160_116084

/-- A polygon constructed from squares and equilateral triangles -/
structure PolygonArrangement where
  squares : ℕ
  triangles : ℕ
  side_length : ℝ
  perimeter : ℝ

/-- The existence of a polygon arrangement with the given properties -/
theorem polygon_arrangement_exists : ∃ (p : PolygonArrangement), 
  p.squares = 9 ∧ 
  p.triangles = 19 ∧ 
  p.side_length = 1 ∧ 
  p.perimeter = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_arrangement_exists_l1160_116084


namespace NUMINAMATH_CALUDE_derivatives_at_zero_l1160_116011

open Function Real

/-- A function f satisfying the given conditions -/
def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → f (1 / n) = n^2 / (n^2 + 1)

/-- The theorem statement -/
theorem derivatives_at_zero
  (f : ℝ → ℝ)
  (h_smooth : ContDiff ℝ ⊤ f)
  (h_cond : f_condition f) :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv^[2] f 0 = -2 ∧
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by sorry

end NUMINAMATH_CALUDE_derivatives_at_zero_l1160_116011


namespace NUMINAMATH_CALUDE_binomial_15_12_l1160_116005

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l1160_116005


namespace NUMINAMATH_CALUDE_peaches_picked_l1160_116092

def initial_peaches : ℕ := 34
def current_peaches : ℕ := 86

theorem peaches_picked (initial : ℕ) (current : ℕ) :
  current ≥ initial → current - initial = current - initial :=
by sorry

end NUMINAMATH_CALUDE_peaches_picked_l1160_116092


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l1160_116064

theorem modulus_of_complex_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l1160_116064


namespace NUMINAMATH_CALUDE_digit_58_is_4_l1160_116091

/-- The repeating part of the decimal representation of 1/17 -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating part -/
def repeat_length : Nat := decimal_rep_1_17.length

/-- The 58th digit after the decimal point in the decimal representation of 1/17 -/
def digit_58 : Nat :=
  decimal_rep_1_17[(58 - 1) % repeat_length]

theorem digit_58_is_4 : digit_58 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_58_is_4_l1160_116091


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l1160_116035

theorem greatest_multiple_under_1000 : ∃ (n : ℕ), n = 990 ∧ 
  (∀ m : ℕ, m < 1000 → m % 5 = 0 → m % 6 = 0 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l1160_116035


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1160_116034

theorem concentric_circles_radii_difference
  (s L : ℝ)
  (h_positive : s > 0)
  (h_ratio : L^2 / s^2 = 4) :
  L - s = s :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1160_116034


namespace NUMINAMATH_CALUDE_complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l1160_116099

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 4}

-- Part I
theorem complement_A_inter_B_when_m_is_one :
  (A ∩ B 1)ᶜ = {x | x < 3 ∨ x ≥ 6} := by sorry

-- Part II
theorem one_in_A_union_B_iff_m_in_range (m : ℝ) :
  (1 ∈ A ∪ B m) ↔ (-3/2 < m ∧ m < 0) := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l1160_116099


namespace NUMINAMATH_CALUDE_equation_solutions_l1160_116081

theorem equation_solutions :
  (∃ x : ℝ, 4.8 - 3 * x = 1.8 ∧ x = 1) ∧
  (∃ x : ℝ, (1/8) / (1/5) = x / 24 ∧ x = 15) ∧
  (∃ x : ℝ, 7.5 * x + 6.5 * x = 2.8 ∧ x = 0.2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1160_116081


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1160_116086

/-- The number of ways to arrange 6 students with specific constraints -/
def arrangement_count : ℕ := 144

/-- Two specific students (A and B) must be adjacent -/
def adjacent_pair : ℕ := 2

/-- Number of students excluding A, B, and C -/
def other_students : ℕ := 3

/-- Number of valid positions for student C -/
def valid_positions_for_c : ℕ := 3

theorem student_arrangement_count :
  arrangement_count = 
    (Nat.factorial other_students) * 
    (Nat.factorial (other_students + 1) / Nat.factorial (other_students - 1)) * 
    adjacent_pair := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1160_116086


namespace NUMINAMATH_CALUDE_cos_negative_third_quadrants_l1160_116032

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the possible quadrants for a given cosine value
def possibleQuadrants (cosθ : ℝ) : Set Quadrant :=
  if cosθ > 0 then {Quadrant.First, Quadrant.Fourth}
  else if cosθ < 0 then {Quadrant.Second, Quadrant.Third}
  else {Quadrant.First, Quadrant.Second, Quadrant.Third, Quadrant.Fourth}

-- Theorem statement
theorem cos_negative_third_quadrants :
  let cosθ : ℝ := -1/3
  possibleQuadrants cosθ = {Quadrant.Second, Quadrant.Third} :=
by sorry


end NUMINAMATH_CALUDE_cos_negative_third_quadrants_l1160_116032


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1160_116037

/-- A line passing through (-2, 3) with slope 2 has the equation 2x - y - 7 = 0 -/
theorem line_equation_through_point_with_slope :
  let point : ℝ × ℝ := (-2, 3)
  let slope : ℝ := 2
  let line_equation (x y : ℝ) := 2 * x - y - 7 = 0
  (∀ x y, line_equation x y ↔ y - point.2 = slope * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1160_116037


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l1160_116017

theorem circle_diameter_ratio (R S : ℝ) (harea : π * R^2 = 0.04 * π * S^2) :
  2 * R = 0.4 * (2 * S) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l1160_116017


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1160_116089

/-- Determinant of a 2x2 matrix --/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence --/
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : isGeometric a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1160_116089


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l1160_116097

theorem fraction_cube_equality : (81000 ^ 3) / (27000 ^ 3) = 27 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l1160_116097


namespace NUMINAMATH_CALUDE_sum_reciprocal_l1160_116030

theorem sum_reciprocal (x : ℝ) (w : ℝ) (h1 : x ≠ 0) (h2 : w = x^2 + (1/x)^2) (h3 : w = 23) :
  x + (1/x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_l1160_116030


namespace NUMINAMATH_CALUDE_train_length_l1160_116038

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 179.99999999999997) (h2 : time = 3) :
  speed * time = 540 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1160_116038


namespace NUMINAMATH_CALUDE_cubic_root_product_l1160_116050

theorem cubic_root_product (u v w : ℝ) : 
  (u^3 - 15*u^2 + 13*u - 6 = 0) →
  (v^3 - 15*v^2 + 13*v - 6 = 0) →
  (w^3 - 15*w^2 + 13*w - 6 = 0) →
  (1 + u) * (1 + v) * (1 + w) = 35 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l1160_116050
