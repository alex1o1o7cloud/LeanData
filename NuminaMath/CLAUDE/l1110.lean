import Mathlib

namespace lily_bushes_theorem_l1110_111078

theorem lily_bushes_theorem (bushes : Fin 19 → ℕ) : 
  ∃ i : Fin 19, Even ((bushes i) + (bushes ((i + 1) % 19))) := by
  sorry

end lily_bushes_theorem_l1110_111078


namespace rational_equation_solution_l1110_111085

theorem rational_equation_solution :
  ∃ y : ℝ, y ≠ 3 ∧ y ≠ (1/3) ∧
  (y^2 - 7*y + 12)/(y - 3) + (3*y^2 + 5*y - 8)/(3*y - 1) = -8 ∧
  y = -6 := by
sorry

end rational_equation_solution_l1110_111085


namespace remaining_students_l1110_111055

def number_of_groups : ℕ := 3
def students_per_group : ℕ := 8
def students_who_left : ℕ := 2

theorem remaining_students :
  (number_of_groups * students_per_group) - students_who_left = 22 := by
  sorry

end remaining_students_l1110_111055


namespace average_speed_calculation_l1110_111032

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 40 := by
  sorry

end average_speed_calculation_l1110_111032


namespace average_problem_l1110_111008

theorem average_problem (y : ℝ) : (15 + 26 + y) / 3 = 23 → y = 28 := by
  sorry

end average_problem_l1110_111008


namespace selected_is_sample_size_l1110_111054

/-- Represents a statistical study -/
structure StatisticalStudy where
  population_size : ℕ
  selected_size : ℕ
  selected_size_le_population : selected_size ≤ population_size

/-- Definition of sample size -/
def sample_size (study : StatisticalStudy) : ℕ := study.selected_size

theorem selected_is_sample_size (study : StatisticalStudy) 
  (h1 : study.population_size = 3000) 
  (h2 : study.selected_size = 100) : 
  sample_size study = study.selected_size :=
by
  sorry

#check selected_is_sample_size

end selected_is_sample_size_l1110_111054


namespace monica_milk_amount_l1110_111069

-- Define the initial amount of milk Don has
def dons_milk : ℚ := 3/4

-- Define the fraction of milk Don gives to Rachel
def fraction_to_rachel : ℚ := 1/2

-- Define the fraction of Rachel's milk that Monica drinks
def fraction_monica_drinks : ℚ := 1/3

-- Theorem statement
theorem monica_milk_amount :
  fraction_monica_drinks * (fraction_to_rachel * dons_milk) = 1/8 := by
  sorry

end monica_milk_amount_l1110_111069


namespace fraction_to_decimal_l1110_111076

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l1110_111076


namespace optimal_investment_l1110_111088

/-- Represents an investment project with profit and loss rates -/
structure Project where
  maxProfitRate : Rat
  maxLossRate : Rat

/-- Represents an investment allocation -/
structure Investment where
  projectA : Rat
  projectB : Rat

def totalInvestment (i : Investment) : Rat :=
  i.projectA + i.projectB

def possibleLoss (p : Project) (i : Rat) : Rat :=
  i * p.maxLossRate

def possibleProfit (p : Project) (i : Rat) : Rat :=
  i * p.maxProfitRate

theorem optimal_investment
  (projectA : Project)
  (projectB : Project)
  (maxInvestment : Rat)
  (maxLoss : Rat)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 1/2)
  (h3 : projectA.maxLossRate = 3/10)
  (h4 : projectB.maxLossRate = 1/10)
  (h5 : maxInvestment = 100000)
  (h6 : maxLoss = 18000) :
  ∃ (i : Investment),
    totalInvestment i ≤ maxInvestment ∧
    possibleLoss projectA i.projectA + possibleLoss projectB i.projectB ≤ maxLoss ∧
    ∀ (j : Investment),
      totalInvestment j ≤ maxInvestment →
      possibleLoss projectA j.projectA + possibleLoss projectB j.projectB ≤ maxLoss →
      possibleProfit projectA i.projectA + possibleProfit projectB i.projectB ≥
      possibleProfit projectA j.projectA + possibleProfit projectB j.projectB ∧
    i.projectA = 40000 ∧
    i.projectB = 60000 :=
  sorry

#check optimal_investment

end optimal_investment_l1110_111088


namespace equal_paper_distribution_l1110_111006

theorem equal_paper_distribution (total_sheets : ℕ) (num_friends : ℕ) (sheets_per_friend : ℕ) :
  total_sheets = 15 →
  num_friends = 3 →
  total_sheets = num_friends * sheets_per_friend →
  sheets_per_friend = 5 := by
  sorry

end equal_paper_distribution_l1110_111006


namespace chord_length_squared_l1110_111073

/-- Given three circles with radii 4, 7, and 9, where the circles with radii 4 and 7 
    are externally tangent to each other and internally tangent to the circle with radius 9, 
    the square of the length of the chord of the circle with radius 9 that is a common 
    external tangent to the other two circles is equal to 224. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : r₃ = 9) 
  (h_ext_tangent : r₃ = r₁ + r₂) 
  (h_int_tangent₁ : r₃ - r₁ = r₂) (h_int_tangent₂ : r₃ - r₂ = r₁) : 
  ∃ (chord_length : ℝ), chord_length^2 = 224 := by
  sorry

end chord_length_squared_l1110_111073


namespace logarithm_sum_equals_two_l1110_111007

theorem logarithm_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end logarithm_sum_equals_two_l1110_111007


namespace probability_same_color_top_three_l1110_111090

def total_cards : ℕ := 52
def cards_per_color : ℕ := 26

theorem probability_same_color_top_three (total : ℕ) (per_color : ℕ) 
  (h1 : total = 52) 
  (h2 : per_color = 26) 
  (h3 : total = 2 * per_color) :
  (2 * (per_color.choose 3)) / (total.choose 3) = 12 / 51 := by
  sorry

end probability_same_color_top_three_l1110_111090


namespace fair_attendance_difference_l1110_111001

theorem fair_attendance_difference : 
  ∀ (last_year : ℕ) (this_year : ℕ) (next_year : ℕ),
    this_year = 600 →
    next_year = 2 * this_year →
    last_year + this_year + next_year = 2800 →
    last_year < next_year →
    next_year - last_year = 200 := by
  sorry

end fair_attendance_difference_l1110_111001


namespace greatest_multiple_of_four_l1110_111036

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 4 ∣ x ∧ x^3 < 1728 → x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ 4 ∣ y ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end greatest_multiple_of_four_l1110_111036


namespace smallest_base_for_124_l1110_111014

theorem smallest_base_for_124 (b : ℕ) : b ≥ 5 ↔ b ^ 2 ≤ 124 ∧ 124 < b ^ 3 :=
sorry

end smallest_base_for_124_l1110_111014


namespace local_maximum_at_one_l1110_111062

/-- The function y = (x+1)/(x^2+3) has a local maximum at x = 1 -/
theorem local_maximum_at_one :
  ∃ δ > 0, ∀ x : ℝ, x ≠ 1 → |x - 1| < δ →
    (x + 1) / (x^2 + 3) ≤ (1 + 1) / (1^2 + 3) := by
  sorry

end local_maximum_at_one_l1110_111062


namespace range_of_trigonometric_function_l1110_111074

theorem range_of_trigonometric_function :
  ∀ x : ℝ, -1 ≤ Real.sin x * Real.cos x + Real.sin x + Real.cos x ∧ 
           Real.sin x * Real.cos x + Real.sin x + Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end range_of_trigonometric_function_l1110_111074


namespace polynomial_value_symmetry_l1110_111015

theorem polynomial_value_symmetry (a b c : ℝ) :
  ((-3)^5 * a + (-3)^3 * b + (-3) * c - 5 = 7) →
  (3^5 * a + 3^3 * b + 3 * c - 5 = -17) := by
  sorry

end polynomial_value_symmetry_l1110_111015


namespace average_of_sample_l1110_111009

def sample_average (x : Fin 10 → ℝ) (a b : ℝ) : Prop :=
  (x 0 + x 1 + x 2) / 3 = a ∧
  (x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 7 = b

theorem average_of_sample (x : Fin 10 → ℝ) (a b : ℝ) 
  (h : sample_average x a b) : 
  (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 10 = (3 * a + 7 * b) / 10 := by
  sorry

end average_of_sample_l1110_111009


namespace solution_to_logarithmic_equation_l1110_111056

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, lg (3 * x + 4) = 1 ∧ x = 2 :=
by
  sorry

end solution_to_logarithmic_equation_l1110_111056


namespace sallys_quarters_l1110_111026

/-- Given that Sally had 760 quarters initially and spent 418 quarters,
    prove that she now has 342 quarters. -/
theorem sallys_quarters (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
    (h1 : initial = 760)
    (h2 : spent = 418)
    (h3 : remaining = initial - spent) :
  remaining = 342 := by
  sorry

end sallys_quarters_l1110_111026


namespace crescent_moon_area_l1110_111020

/-- The area of a crescent moon formed by two circles -/
theorem crescent_moon_area :
  let large_circle_radius : ℝ := 4
  let small_circle_radius : ℝ := 2
  let large_quarter_circle_area : ℝ := π * large_circle_radius^2 / 4
  let small_half_circle_area : ℝ := π * small_circle_radius^2 / 2
  large_quarter_circle_area - small_half_circle_area = 2 * π := by
sorry

end crescent_moon_area_l1110_111020


namespace quadratic_always_nonnegative_l1110_111050

theorem quadratic_always_nonnegative (x y : ℝ) : x^2 + x*y + y^2 ≥ 0 := by
  sorry

end quadratic_always_nonnegative_l1110_111050


namespace initial_children_on_bus_l1110_111059

theorem initial_children_on_bus (children_off : ℕ) (children_on : ℕ) (final_children : ℕ) :
  children_off = 10 →
  children_on = 5 →
  final_children = 16 →
  final_children + (children_off - children_on) = 21 :=
by sorry

end initial_children_on_bus_l1110_111059


namespace min_q_value_l1110_111086

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_right_triangle (p q : ℕ) : Prop := p + q = 90

theorem min_q_value (p q : ℕ) (h1 : is_right_triangle p q) (h2 : is_prime p) (h3 : p > q) :
  q ≥ 7 := by sorry

end min_q_value_l1110_111086


namespace function_nature_l1110_111035

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_nature (f : ℝ → ℝ) 
  (h1 : f 0 ≠ 0)
  (h2 : ∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 2 * f ((x₁ + x₂) / 2) * f ((x₁ - x₂) / 2)) :
  is_even f ∧ ¬ is_odd f := by
sorry

end function_nature_l1110_111035


namespace expression_equality_l1110_111025

theorem expression_equality (y : ℝ) (Q : ℝ) (h : 5 * (3 * y - 7 * Real.pi) = Q) :
  10 * (6 * y - 14 * Real.pi) = 4 * Q := by
  sorry

end expression_equality_l1110_111025


namespace cone_volume_l1110_111098

/-- The volume of a cone with slant height 15 cm and height 13 cm is (728/3)π cubic centimeters. -/
theorem cone_volume (π : ℝ) (slant_height height : ℝ) 
  (h1 : slant_height = 15)
  (h2 : height = 13) :
  (1/3 : ℝ) * π * (slant_height^2 - height^2) * height = (728/3) * π := by
  sorry

end cone_volume_l1110_111098


namespace swimmer_speed_l1110_111066

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water (km/h)
  stream : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.man + s.stream else s.man - s.stream

/-- Theorem stating that given the conditions, the man's speed in still water is 15.5 km/h. -/
theorem swimmer_speed (s : SwimmerSpeeds) 
  (h1 : effectiveSpeed s true * 2 = 36)  -- Downstream condition
  (h2 : effectiveSpeed s false * 2 = 26) -- Upstream condition
  : s.man = 15.5 := by
  sorry

#check swimmer_speed

end swimmer_speed_l1110_111066


namespace housewife_spending_l1110_111043

theorem housewife_spending (initial_amount : ℚ) (spent_fraction : ℚ) :
  initial_amount = 150 →
  spent_fraction = 2/3 →
  initial_amount * (1 - spent_fraction) = 50 :=
by
  sorry

end housewife_spending_l1110_111043


namespace chess_competition_games_l1110_111040

theorem chess_competition_games (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)
  (h2 : M * (M - 1) / 2 = 190) :
  W * M = 200 := by
  sorry

end chess_competition_games_l1110_111040


namespace inequality_proof_l1110_111023

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3/4 := by
  sorry

end inequality_proof_l1110_111023


namespace unique_coin_combination_l1110_111021

/-- Represents a coin with a value in kopecks -/
structure Coin where
  value : ℕ

/-- Represents a wallet containing two coins -/
structure Wallet where
  coin1 : Coin
  coin2 : Coin

/-- The total value of coins in a wallet -/
def walletValue (w : Wallet) : ℕ := w.coin1.value + w.coin2.value

/-- Predicate to check if a coin is not a five-kopeck coin -/
def isNotFiveKopecks (c : Coin) : Prop := c.value ≠ 5

/-- Theorem stating the only possible combination of coins -/
theorem unique_coin_combination (w : Wallet) 
  (h1 : walletValue w = 15)
  (h2 : isNotFiveKopecks w.coin1 ∨ isNotFiveKopecks w.coin2) :
  (w.coin1.value = 5 ∧ w.coin2.value = 10) ∨ (w.coin1.value = 10 ∧ w.coin2.value = 5) :=
sorry

end unique_coin_combination_l1110_111021


namespace cos_difference_special_case_l1110_111071

theorem cos_difference_special_case (x₁ x₂ : Real) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 2 * Real.pi)
  (h4 : Real.sin x₁ = 1/3) (h5 : Real.sin x₂ = 1/3) : 
  Real.cos (x₁ - x₂) = -7/9 := by
  sorry

end cos_difference_special_case_l1110_111071


namespace second_polygon_sides_l1110_111081

/-- Given two regular polygons with the same perimeter, where the first polygon has 45 sides
    and a side length three times as long as the second, prove that the second polygon has 135 sides. -/
theorem second_polygon_sides (s : ℝ) (sides_second : ℕ) : 
  s > 0 →  -- Assume positive side length
  45 * (3 * s) = sides_second * s →  -- Same perimeter condition
  sides_second = 135 := by
sorry

end second_polygon_sides_l1110_111081


namespace division_problem_l1110_111030

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1565 → divisor = 24 → remainder = 5 → quotient = 65 →
  dividend = divisor * quotient + remainder :=
by sorry

end division_problem_l1110_111030


namespace acorns_given_calculation_l1110_111046

/-- The number of acorns Megan gave to her sister -/
def acorns_given : ℕ := sorry

/-- The initial number of acorns Megan had -/
def initial_acorns : ℕ := 16

/-- The number of acorns Megan has left -/
def acorns_left : ℕ := 9

/-- Theorem stating that the number of acorns given is the difference between
    the initial number and the number left -/
theorem acorns_given_calculation : acorns_given = initial_acorns - acorns_left := by
  sorry

end acorns_given_calculation_l1110_111046


namespace quadratic_inverse_sum_l1110_111060

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The inverse of a quadratic function -/
def InverseQuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ c * x^2 + b * x + a

theorem quadratic_inverse_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c (InverseQuadraticFunction a b c x) = x) →
  a + c = -1 := by
  sorry

end quadratic_inverse_sum_l1110_111060


namespace even_integers_in_pascal_triangle_l1110_111016

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Counts the number of even integers in the first n rows of Pascal's Triangle -/
def countEvenIntegers (pt : PascalTriangle n) : ℕ := sorry

theorem even_integers_in_pascal_triangle :
  ∀ (pt10 : PascalTriangle 10) (pt15 : PascalTriangle 15),
    countEvenIntegers pt10 = 22 →
    countEvenIntegers pt15 = 53 := by sorry

end even_integers_in_pascal_triangle_l1110_111016


namespace composite_product_quotient_l1110_111049

/-- The first ten positive composite integers -/
def first_ten_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

/-- The product of the first five positive composite integers -/
def product_first_five : ℕ := (first_ten_composites.take 5).prod

/-- The product of the next five positive composite integers -/
def product_next_five : ℕ := (first_ten_composites.drop 5).prod

/-- Theorem stating that the quotient of the product of the first five positive composite integers
    divided by the product of the next five composite integers equals 1/42 -/
theorem composite_product_quotient :
  (product_first_five : ℚ) / (product_next_five : ℚ) = 1 / 42 := by
  sorry

end composite_product_quotient_l1110_111049


namespace fifth_term_value_l1110_111011

/-- Given a sequence {aₙ} where Sₙ denotes the sum of its first n terms and Sₙ = n² + 1,
    prove that a₅ = 9. -/
theorem fifth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end fifth_term_value_l1110_111011


namespace count_special_numbers_proof_l1110_111094

/-- The number of five-digit numbers with two pairs of adjacent equal digits,
    where digits from different pairs are different, and the remaining digit
    is different from all other digits. -/
def count_special_numbers : ℕ := 1944

/-- The set of valid configurations for the special five-digit numbers. -/
inductive Configuration : Type
  | AABBC : Configuration
  | AACBB : Configuration
  | CAABB : Configuration

/-- The number of possible choices for the first digit of the number. -/
def first_digit_choices : ℕ := 9

/-- The number of possible choices for the second digit of the number. -/
def second_digit_choices : ℕ := 9

/-- The number of possible choices for the third digit of the number. -/
def third_digit_choices : ℕ := 8

/-- The number of valid configurations. -/
def num_configurations : ℕ := 3

theorem count_special_numbers_proof :
  count_special_numbers =
    num_configurations * first_digit_choices * second_digit_choices * third_digit_choices :=
by sorry

end count_special_numbers_proof_l1110_111094


namespace berry_collection_theorem_l1110_111089

def berry_collection (total_berries : ℕ) (sergey_speed_ratio : ℕ) : Prop :=
  let sergey_picked := (2 * total_berries) / 3
  let dima_picked := total_berries / 3
  let sergey_collected := sergey_picked / 2
  let dima_collected := (2 * dima_picked) / 3
  sergey_collected - dima_collected = 100

theorem berry_collection_theorem :
  berry_collection 900 2 := by
  sorry

end berry_collection_theorem_l1110_111089


namespace cornbread_pieces_l1110_111034

def pan_length : ℕ := 24
def pan_width : ℕ := 20
def piece_size : ℕ := 3

theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_size * piece_size) = 53 :=
by sorry

end cornbread_pieces_l1110_111034


namespace polar_to_rectangular_equation_l1110_111038

/-- Given a curve C with polar coordinate equation ρ sin (θ - π/4) = √2,
    where the origin is at the pole and the polar axis lies on the x-axis
    in a rectangular coordinate system, prove that the rectangular
    coordinate equation of C is x - y + 2 = 0. -/
theorem polar_to_rectangular_equation :
  ∀ (ρ θ x y : ℝ),
  (ρ * Real.sin (θ - π/4) = Real.sqrt 2) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - y + 2 = 0) :=
by sorry

end polar_to_rectangular_equation_l1110_111038


namespace preimage_of_3_1_l1110_111068

/-- The transformation f that maps (x, y) to (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that the preimage of (3, 1) under f is (1, 1) -/
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by sorry

end preimage_of_3_1_l1110_111068


namespace complex_difference_magnitude_l1110_111037

def i : ℂ := Complex.I

theorem complex_difference_magnitude : Complex.abs ((1 + i)^13 - (1 - i)^13) = 128 := by
  sorry

end complex_difference_magnitude_l1110_111037


namespace smallest_land_fraction_for_120_members_l1110_111080

/-- Represents a noble family with land inheritance rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (has_original_plot : Bool)

/-- The smallest fraction of land a family member can receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^39)

/-- Theorem stating the smallest possible land fraction for a family of 120 members -/
theorem smallest_land_fraction_for_120_members 
  (family : NobleFamily) 
  (h1 : family.total_members = 120) 
  (h2 : family.has_original_plot = true) : 
  smallest_land_fraction family = 1 / (2 * 3^39) := by
  sorry

end smallest_land_fraction_for_120_members_l1110_111080


namespace division_evaluation_l1110_111019

theorem division_evaluation : 250 / (5 + 12 * 3^2) = 250 / 113 := by sorry

end division_evaluation_l1110_111019


namespace parabola_fixed_point_l1110_111033

/-- The parabola passes through the point (3, 36) for all real t -/
theorem parabola_fixed_point :
  ∀ t : ℝ, 36 = 4 * (3 : ℝ)^2 + t * 3 - t^2 - 3 * t := by
  sorry

end parabola_fixed_point_l1110_111033


namespace simultaneous_equations_solution_l1110_111095

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 2 * y = 12) ∧ 
    (9 * y - 6 * x = -18) ∧ 
    (x = 24/5) ∧ 
    (y = 6/5) := by
  sorry

end simultaneous_equations_solution_l1110_111095


namespace area_difference_is_quarter_l1110_111003

/-- Represents a regular octagon with side length 1 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- Represents the cutting operation on the octagon -/
def cut (o : RegularOctagon) : ℝ × ℝ := sorry

/-- The difference in area between the larger and smaller parts after cutting -/
def area_difference (o : RegularOctagon) : ℝ :=
  let (larger, smaller) := cut o
  larger - smaller

/-- Theorem stating that the area difference is 1/4 -/
theorem area_difference_is_quarter (o : RegularOctagon) :
  area_difference o = 1/4 := by sorry

end area_difference_is_quarter_l1110_111003


namespace common_solution_y_values_l1110_111039

theorem common_solution_y_values : 
  ∃ y₁ y₂ : ℝ, 
    (∀ x y : ℝ, x^2 + y^2 - 9 = 0 ∧ x^2 - 4*y + 8 = 0 → y = y₁ ∨ y = y₂) ∧
    y₁ = -2 + Real.sqrt 21 ∧
    y₂ = -2 - Real.sqrt 21 :=
by sorry

end common_solution_y_values_l1110_111039


namespace zoo_animals_ratio_l1110_111096

theorem zoo_animals_ratio (snakes monkeys lions pandas dogs : ℕ) : 
  snakes = 15 →
  monkeys = 2 * snakes →
  lions = monkeys - 5 →
  pandas = lions + 8 →
  snakes + monkeys + lions + pandas + dogs = 114 →
  dogs * 3 = pandas := by
sorry

end zoo_animals_ratio_l1110_111096


namespace logan_tower_height_l1110_111051

/-- The height of the city's water tower in meters -/
def city_tower_height : ℝ := 60

/-- The volume of water the city's water tower can hold in liters -/
def city_tower_volume : ℝ := 150000

/-- The volume of water Logan's miniature water tower can hold in liters -/
def miniature_tower_volume : ℝ := 0.15

/-- The height of Logan's miniature water tower in meters -/
def miniature_tower_height : ℝ := 0.6

/-- Theorem stating that the height of Logan's miniature tower should be 0.6 meters -/
theorem logan_tower_height : miniature_tower_height = 0.6 := by
  sorry

end logan_tower_height_l1110_111051


namespace arithmetic_geometric_sequence_a6_l1110_111029

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_geometric_sequence_a6 (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 6 = 2 := by
sorry

end arithmetic_geometric_sequence_a6_l1110_111029


namespace count_arrangements_l1110_111093

/-- The number of arrangements of 5 students (2 male and 3 female) in a line formation,
    where one specific male student does not stand at either end and only two of the
    three female students stand next to each other. -/
def num_arrangements : ℕ := 48

/-- Proves that the number of different possible arrangements is 48. -/
theorem count_arrangements :
  let total_students : ℕ := 5
  let male_students : ℕ := 2
  let female_students : ℕ := 3
  let specific_male_not_at_ends : Bool := true
  let two_females_adjacent : Bool := true
  num_arrangements = 48 := by sorry

end count_arrangements_l1110_111093


namespace complex_equation_solution_l1110_111087

theorem complex_equation_solution (i : ℂ) (z : ℂ) :
  i * i = -1 →
  (1 + i) * z = 1 + 3 * i →
  z = 2 + i := by
sorry

end complex_equation_solution_l1110_111087


namespace min_sum_squares_min_sum_squares_zero_l1110_111028

theorem min_sum_squares (x y s : ℝ) (h : x + y + s = 0) : 
  ∀ a b c : ℝ, a + b + c = 0 → x^2 + y^2 + s^2 ≤ a^2 + b^2 + c^2 :=
by
  sorry

theorem min_sum_squares_zero (x y s : ℝ) (h : x + y + s = 0) : 
  ∃ a b c : ℝ, a + b + c = 0 ∧ a^2 + b^2 + c^2 = 0 :=
by
  sorry

end min_sum_squares_min_sum_squares_zero_l1110_111028


namespace ben_money_after_seven_days_l1110_111075

/-- Ben's daily allowance -/
def daily_allowance : ℕ := 50

/-- Ben's daily spending -/
def daily_spending : ℕ := 15

/-- Number of days -/
def num_days : ℕ := 7

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_allowance - daily_spending

/-- Ben's total savings before mom's contribution -/
def initial_savings : ℕ := daily_savings * num_days

/-- Ben's savings after mom's contribution -/
def savings_after_mom : ℕ := 2 * initial_savings

/-- Dad's contribution -/
def dad_contribution : ℕ := 10

/-- Ben's final amount -/
def ben_final_amount : ℕ := savings_after_mom + dad_contribution

theorem ben_money_after_seven_days : ben_final_amount = 500 := by
  sorry

end ben_money_after_seven_days_l1110_111075


namespace geometric_sequence_fourth_term_l1110_111031

/-- Given a geometric sequence {a_n}, if a_2 and a_6 are roots of x^2 - 34x + 81 = 0, then a_4 = 9 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) :
  a 4 = 9 := by sorry

end geometric_sequence_fourth_term_l1110_111031


namespace grid_sum_invariant_l1110_111058

/-- Represents a 5x5 grid where each cell contains a natural number -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Represents a sequence of 25 moves to fill the grid -/
def MoveSequence := Fin 25 → Fin 5 × Fin 5

/-- Checks if two cells are adjacent in the grid -/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Generates a grid based on a move sequence -/
def generateGrid (moves : MoveSequence) : Grid :=
  sorry

/-- Calculates the sum of all numbers in a grid -/
def gridSum (g : Grid) : ℕ :=
  sorry

/-- The main theorem: the sum of all numbers in the grid is always 40 -/
theorem grid_sum_invariant (moves : MoveSequence) :
  gridSum (generateGrid moves) = 40 :=
  sorry

end grid_sum_invariant_l1110_111058


namespace necessary_but_not_sufficient_l1110_111018

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ C → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ C) :=
by sorry

end necessary_but_not_sufficient_l1110_111018


namespace ceiling_floor_difference_l1110_111077

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l1110_111077


namespace karens_cookies_l1110_111083

/-- Theorem: Karen's Cookies --/
theorem karens_cookies (
  kept_for_self : ℕ)
  (given_to_grandparents : ℕ)
  (class_size : ℕ)
  (cookies_per_person : ℕ)
  (h1 : kept_for_self = 10)
  (h2 : given_to_grandparents = 8)
  (h3 : class_size = 16)
  (h4 : cookies_per_person = 2)
  : kept_for_self + given_to_grandparents + class_size * cookies_per_person = 50 := by
  sorry

end karens_cookies_l1110_111083


namespace sharon_salary_increase_l1110_111027

theorem sharon_salary_increase (S : ℝ) (h1 : S + 0.20 * S = 600) (h2 : S + x * S = 575) : x = 0.15 := by
  sorry

end sharon_salary_increase_l1110_111027


namespace lcm_36_100_l1110_111041

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end lcm_36_100_l1110_111041


namespace line_inclination_angle_l1110_111013

/-- The angle of inclination of the line x - √3y + 6 = 0 is 30°. -/
theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 6 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end line_inclination_angle_l1110_111013


namespace keaton_apple_earnings_l1110_111067

/-- Represents Keaton's farm earnings -/
structure FarmEarnings where
  orangeHarvestFrequency : ℕ  -- Number of orange harvests per year
  orangeHarvestValue : ℕ      -- Value of each orange harvest in dollars
  totalAnnualEarnings : ℕ     -- Total annual earnings in dollars

/-- Calculates the annual earnings from apple harvest -/
def appleEarnings (f : FarmEarnings) : ℕ :=
  f.totalAnnualEarnings - (f.orangeHarvestFrequency * f.orangeHarvestValue)

/-- Theorem: Keaton's annual earnings from apple harvest is $120 -/
theorem keaton_apple_earnings :
  ∃ (f : FarmEarnings),
    f.orangeHarvestFrequency = 6 ∧
    f.orangeHarvestValue = 50 ∧
    f.totalAnnualEarnings = 420 ∧
    appleEarnings f = 120 := by
  sorry

end keaton_apple_earnings_l1110_111067


namespace asymptotic_function_part1_non_asymptotic_function_part2_l1110_111091

/-- Definition of asymptotic function -/
def is_asymptotic_function (f g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x ≥ 0, f x = g x + h x) ∧
  (Monotone (fun x ↦ -h x)) ∧
  (∀ x ≥ 0, 0 < h x ∧ h x ≤ p)

/-- Part I: Asymptotic function for f(x) = (x^2 + 2x + 3) / (x + 1) -/
theorem asymptotic_function_part1 :
  is_asymptotic_function (fun x ↦ (x^2 + 2*x + 3) / (x + 1)) (fun x ↦ x + 1) 2 :=
sorry

/-- Part II: Non-asymptotic function for f(x) = √(x^2 + 1) -/
theorem non_asymptotic_function_part2 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ¬ is_asymptotic_function (fun x ↦ Real.sqrt (x^2 + 1)) (fun x ↦ a * x) p :=
sorry

end asymptotic_function_part1_non_asymptotic_function_part2_l1110_111091


namespace x_value_theorem_l1110_111097

theorem x_value_theorem (x y : ℝ) :
  (x / (x + 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) →
  x = (2*y^2 + 6*y - 4) / 3 := by
sorry

end x_value_theorem_l1110_111097


namespace race_completion_time_l1110_111064

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 40 meters
  960 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

/-- The theorem stating A's completion time -/
theorem race_completion_time (a b : Runner) (h : Race a b) : a.time = 250 := by
  sorry

end race_completion_time_l1110_111064


namespace segment_ratio_l1110_111000

/-- Given four points A, B, C, D on a line segment, 
    if AB : BC = 1 : 2 and BC : CD = 8 : 5, 
    then AB : BD = 4 : 13 -/
theorem segment_ratio (A B C D : ℝ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (ratio1 : (B - A) / (C - B) = 1 / 2)
  (ratio2 : (C - B) / (D - C) = 8 / 5) :
  (B - A) / (D - B) = 4 / 13 := by
  sorry

end segment_ratio_l1110_111000


namespace star_sum_larger_than_emilio_sum_l1110_111057

def star_numbers : List ℕ := List.range 50

def emilio_numbers : List ℕ :=
  star_numbers.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 ∨ tens = 3 then
      (if tens = 2 then 5 else 5) * 10 + ones
    else if ones = 2 ∨ ones = 3 then
      tens * 10 + 5
    else
      n
  )

theorem star_sum_larger_than_emilio_sum :
  (star_numbers.sum - emilio_numbers.sum) = 550 := by
  sorry

end star_sum_larger_than_emilio_sum_l1110_111057


namespace square_field_area_l1110_111010

/-- Represents the properties of a square field with barbed wire fencing --/
structure SquareField where
  side : ℝ
  wireRate : ℝ
  gateWidth : ℝ
  gateCount : ℕ
  totalCost : ℝ

/-- Calculates the area of the square field --/
def fieldArea (field : SquareField) : ℝ :=
  field.side * field.side

/-- Calculates the length of barbed wire needed --/
def wireLength (field : SquareField) : ℝ :=
  4 * field.side - field.gateWidth * field.gateCount

/-- Theorem stating the area of the square field given the conditions --/
theorem square_field_area (field : SquareField)
  (h1 : field.wireRate = 1)
  (h2 : field.gateWidth = 1)
  (h3 : field.gateCount = 2)
  (h4 : field.totalCost = 666)
  (h5 : wireLength field * field.wireRate = field.totalCost) :
  fieldArea field = 27889 := by
  sorry

#eval 167 * 167  -- To verify the result

end square_field_area_l1110_111010


namespace f_period_f_definition_f_negative_one_l1110_111063

def f (x : ℝ) : ℝ := sorry

theorem f_period (x : ℝ) : f (x + 2) = f x := sorry

theorem f_definition (x : ℝ) (h : x ∈ Set.Icc 1 3) : f x = x - 2 := sorry

theorem f_negative_one : f (-1) = -1 := by sorry

end f_period_f_definition_f_negative_one_l1110_111063


namespace biology_class_percentage_l1110_111065

theorem biology_class_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 880 →
  not_enrolled = 572 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end biology_class_percentage_l1110_111065


namespace downstream_speed_l1110_111024

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 40 →
  still_water_speed + (still_water_speed - upstream_speed) = 60 :=
by sorry

end downstream_speed_l1110_111024


namespace function_composition_constant_l1110_111017

theorem function_composition_constant (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 5 * x + b
  let g : ℝ → ℝ := λ x ↦ b * x + 3
  (∀ x, f (g x) = 15 * x + 18) :=
by
  sorry

end function_composition_constant_l1110_111017


namespace game_value_conversion_l1110_111053

/-- Calculates the final value of sold games in USD after multiple currency conversions and fees --/
theorem game_value_conversion (initial_value : ℝ) (usd_to_eur_rate : ℝ) (eur_to_usd_fee : ℝ)
  (value_increase : ℝ) (eur_to_jpy_rate : ℝ) (eur_to_jpy_fee : ℝ) (sell_percentage : ℝ)
  (japan_tax_rate : ℝ) (jpy_to_usd_rate : ℝ) (jpy_to_usd_fee : ℝ) :
  initial_value = 200 →
  usd_to_eur_rate = 0.85 →
  eur_to_usd_fee = 0.03 →
  value_increase = 3 →
  eur_to_jpy_rate = 130 →
  eur_to_jpy_fee = 0.02 →
  sell_percentage = 0.4 →
  japan_tax_rate = 0.1 →
  jpy_to_usd_rate = 0.0085 →
  jpy_to_usd_fee = 0.01 →
  ∃ final_value : ℝ, abs (final_value - 190.93) < 0.01 ∧
  final_value = initial_value * usd_to_eur_rate * (1 - eur_to_usd_fee) * value_increase *
                eur_to_jpy_rate * (1 - eur_to_jpy_fee) * sell_percentage *
                (1 - japan_tax_rate) * jpy_to_usd_rate * (1 - jpy_to_usd_fee) := by
  sorry

end game_value_conversion_l1110_111053


namespace systematic_sampling_interval_for_60_bottles_6_samples_l1110_111047

/-- Systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- The problem statement -/
theorem systematic_sampling_interval_for_60_bottles_6_samples :
  systematicSamplingInterval 60 6 = 10 := by
  sorry

end systematic_sampling_interval_for_60_bottles_6_samples_l1110_111047


namespace geometric_sequence_problem_l1110_111092

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ (s : ℝ), 81 * s = b ∧ b * s = 8/27) → 
  b = 2 * Real.sqrt 6 := by
sorry

end geometric_sequence_problem_l1110_111092


namespace john_pennies_l1110_111082

/-- Given that Kate has 223 pennies and John has 165 more pennies than Kate,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_extra : ℕ) 
    (h1 : kate_pennies = 223)
    (h2 : john_extra = 165) :
    kate_pennies + john_extra = 388 := by
  sorry

end john_pennies_l1110_111082


namespace cube_volume_after_cylinder_removal_l1110_111072

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_volume_after_cylinder_removal (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  let cube_volume := cube_side ^ 3
  let cylinder_volume := π * cylinder_radius ^ 2 * cube_side
  cube_volume - cylinder_volume = 216 - 54 * π := by
  sorry

end cube_volume_after_cylinder_removal_l1110_111072


namespace max_prob_with_C_second_l1110_111099

/-- Represents the probability of winning against a player -/
structure WinProbability (α : Type) where
  prob : α → ℝ
  pos : ∀ x, prob x > 0

variable {α : Type}

/-- The players A, B, and C -/
inductive Player : Type where
  | A : Player
  | B : Player
  | C : Player

/-- The probabilities of winning against each player -/
def win_prob (p : WinProbability Player) : Prop :=
  p.prob Player.A < p.prob Player.B ∧ p.prob Player.B < p.prob Player.C

/-- The probability of winning two consecutive games when player x is in the second game -/
def prob_two_consec_wins (p : WinProbability Player) (x : Player) : ℝ :=
  2 * (p.prob Player.A * p.prob x + p.prob Player.B * p.prob x + p.prob Player.C * p.prob x
     - 2 * p.prob Player.A * p.prob Player.B * p.prob Player.C)

/-- The theorem stating that the probability is maximized when C is in the second game -/
theorem max_prob_with_C_second (p : WinProbability Player) (h : win_prob p) :
    ∀ x : Player, prob_two_consec_wins p Player.C ≥ prob_two_consec_wins p x :=
  sorry

end max_prob_with_C_second_l1110_111099


namespace unique_a_value_l1110_111004

theorem unique_a_value (a : ℕ) : 
  (∃ k : ℕ, 88 * a = 2 * k + 1) →  -- 88a is odd
  (∃ m : ℕ, 88 * a = 3 * m) →      -- 88a is a multiple of 3
  a = 5 := by
sorry

end unique_a_value_l1110_111004


namespace equation_solutions_l1110_111084

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  (13*x - x^2) / (x + 1) * (x + (13 - x) / (x + 1)) = 42

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2) :=
by sorry

end equation_solutions_l1110_111084


namespace consecutive_integers_average_l1110_111005

theorem consecutive_integers_average (c d : ℝ) : 
  (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6 = d →
  ((d-1) + d + (d+1) + (d+2) + (d+3) + (d+4)) / 6 = c + 4 := by
sorry

end consecutive_integers_average_l1110_111005


namespace triangle_inequality_l1110_111044

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 * (t.b + t.c - t.a) + t.b^2 * (t.c + t.a - t.b) + t.c^2 * (t.a + t.b - t.c) ≤ 3 * t.a * t.b * t.c := by
  sorry

end triangle_inequality_l1110_111044


namespace solve_for_c_l1110_111042

theorem solve_for_c (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (b_eq : b = (a + c) / 5) :
  c = 35 := by
  sorry

end solve_for_c_l1110_111042


namespace triple_coverage_theorem_l1110_111022

/-- Represents a rectangular rug with given dimensions -/
structure Rug where
  width : ℝ
  length : ℝ

/-- Represents the arrangement of rugs in the auditorium -/
structure AuditoriumArrangement where
  auditorium_size : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs simultaneously -/
def triple_coverage_area (arrangement : AuditoriumArrangement) : ℝ :=
  sorry

/-- The specific arrangement in the problem -/
def problem_arrangement : AuditoriumArrangement :=
  { auditorium_size := 10
  , rug1 := { width := 6, length := 8 }
  , rug2 := { width := 6, length := 6 }
  , rug3 := { width := 5, length := 7 }
  }

theorem triple_coverage_theorem :
  triple_coverage_area problem_arrangement = 6 := by sorry

end triple_coverage_theorem_l1110_111022


namespace arithmetic_sequence_count_l1110_111045

theorem arithmetic_sequence_count :
  let first : ℤ := 162
  let last : ℤ := 42
  let diff : ℤ := -3
  let count := (last - first) / diff + 1
  count = 41 := by sorry

end arithmetic_sequence_count_l1110_111045


namespace saline_mixture_proof_l1110_111061

def initial_volume : ℝ := 50
def initial_concentration : ℝ := 0.4
def added_concentration : ℝ := 0.1
def final_concentration : ℝ := 0.25
def added_volume : ℝ := 50

theorem saline_mixture_proof :
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end saline_mixture_proof_l1110_111061


namespace matrix_product_result_l1110_111052

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (λ acc i => acc * !![1, 2*(i+1); 0, 1])
    (!![1, 0; 0, 1])

theorem matrix_product_result :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end matrix_product_result_l1110_111052


namespace digit_sum_19_or_20_l1110_111079

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def equation_holds (a b c d : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_digit x ∧ is_digit y ∧ is_digit z ∧
  (a * 100 + 50 + b) + (400 + c * 10 + d) = x * 100 + y * 10 + z

theorem digit_sum_19_or_20 (a b c d : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  are_different a b c d ∧
  equation_holds a b c d →
  a + b + c + d = 19 ∨ a + b + c + d = 20 := by
sorry

end digit_sum_19_or_20_l1110_111079


namespace age_ratio_problem_l1110_111070

theorem age_ratio_problem (alma_age melina_age alma_score : ℕ) : 
  alma_age + melina_age = 2 * alma_score →
  melina_age = 60 →
  alma_score = 40 →
  melina_age / alma_age = 3 := by
sorry

end age_ratio_problem_l1110_111070


namespace negation_of_universal_proposition_l1110_111012

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end negation_of_universal_proposition_l1110_111012


namespace liam_picked_40_oranges_l1110_111048

/-- The number of oranges Liam picked -/
def liam_oranges : ℕ := sorry

/-- The price of 2 of Liam's oranges in cents -/
def liam_price : ℕ := 250

/-- The number of oranges Claire picked -/
def claire_oranges : ℕ := 30

/-- The price of each of Claire's oranges in cents -/
def claire_price : ℕ := 120

/-- The total amount saved in cents -/
def total_saved : ℕ := 8600

theorem liam_picked_40_oranges :
  liam_oranges = 40 ∧
  liam_price = 250 ∧
  claire_oranges = 30 ∧
  claire_price = 120 ∧
  total_saved = 8600 ∧
  (liam_oranges * liam_price / 2 + claire_oranges * claire_price = total_saved) :=
by sorry

end liam_picked_40_oranges_l1110_111048


namespace total_retail_price_calculation_l1110_111002

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let retail_before_discount := wholesale_price * (1 + profit_margin)
  retail_before_discount * (1 - discount)

theorem total_retail_price_calculation (P Q R : ℝ) 
  (h1 : P = 90) (h2 : Q = 120) (h3 : R = 150) : 
  calculate_retail_price P 0.2 0.1 + 
  calculate_retail_price Q 0.25 0.15 + 
  calculate_retail_price R 0.3 0.2 = 380.7 := by
  sorry

#eval calculate_retail_price 90 0.2 0.1 + 
      calculate_retail_price 120 0.25 0.15 + 
      calculate_retail_price 150 0.3 0.2

end total_retail_price_calculation_l1110_111002
