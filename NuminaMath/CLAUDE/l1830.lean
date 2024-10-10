import Mathlib

namespace rectangles_in_5x5_grid_l1830_183026

/-- The number of dots on each side of the square array -/
def n : ℕ := 5

/-- The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in an n×n square array of dots -/
def num_rectangles (n : ℕ) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid :
  num_rectangles n = 100 := by
  sorry

end rectangles_in_5x5_grid_l1830_183026


namespace find_g_of_x_l1830_183089

theorem find_g_of_x (x : ℝ) : 
  let g := fun (x : ℝ) ↦ -4*x^4 + 5*x^3 - 2*x^2 + 7*x + 2
  4*x^4 + 2*x^2 - 7*x + g x = 5*x^3 - 4*x + 2 := by sorry

end find_g_of_x_l1830_183089


namespace quadratic_coefficient_l1830_183068

/-- A quadratic function g(x) = px^2 + qx + r -/
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

/-- Theorem: If g(-2) = 0, g(3) = 0, and g(1) = 5, then q = 5/6 -/
theorem quadratic_coefficient (p q r : ℝ) :
  g p q r (-2) = 0 → g p q r 3 = 0 → g p q r 1 = 5 → q = 5/6 := by
  sorry

end quadratic_coefficient_l1830_183068


namespace cube_to_sphere_surface_area_ratio_l1830_183059

theorem cube_to_sphere_surface_area_ratio :
  ∀ (a R : ℝ), a > 0 → R > 0 →
  (a^3 = (4/3) * π * R^3) →
  ((6 * a^2) / (4 * π * R^2) = 3 * (6/π)) :=
by sorry

end cube_to_sphere_surface_area_ratio_l1830_183059


namespace fireflies_remaining_joined_fireflies_l1830_183045

/-- The number of fireflies remaining after some join and some leave --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Proof that 9 fireflies remain given the initial conditions --/
theorem fireflies_remaining : remaining_fireflies 3 8 2 = 9 := by
  sorry

/-- The number of fireflies that joined is 4 less than a dozen --/
theorem joined_fireflies : (12 : ℕ) - 4 = 8 := by
  sorry

end fireflies_remaining_joined_fireflies_l1830_183045


namespace line_intersection_y_axis_l1830_183042

/-- The line passing through points (4, 2) and (6, 14) intersects the y-axis at (0, -22) -/
theorem line_intersection_y_axis :
  let p1 : ℝ × ℝ := (4, 2)
  let p2 : ℝ × ℝ := (6, 14)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, -22) :=
by sorry

end line_intersection_y_axis_l1830_183042


namespace polynomial_transformation_l1830_183054

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by sorry

end polynomial_transformation_l1830_183054


namespace arthur_walked_seven_miles_l1830_183009

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_east blocks_north blocks_west : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north + blocks_west : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 7 miles -/
theorem arthur_walked_seven_miles :
  arthur_distance 8 15 5 (1/4) = 7 := by
  sorry

end arthur_walked_seven_miles_l1830_183009


namespace kate_candy_count_l1830_183023

/-- Given a distribution of candy among four children (Kate, Robert, Bill, and Mary),
    prove that Kate gets 4 pieces of candy. -/
theorem kate_candy_count (kate robert bill mary : ℕ)
  (total : kate + robert + bill + mary = 20)
  (robert_kate : robert = kate + 2)
  (bill_mary : bill = mary - 6)
  (mary_robert : mary = robert + 2)
  (kate_bill : kate = bill + 2) :
  kate = 4 := by
  sorry

end kate_candy_count_l1830_183023


namespace classroom_gpa_l1830_183086

theorem classroom_gpa (total_students : ℕ) (gpa1 gpa2 gpa3 : ℚ) 
  (h1 : total_students = 60)
  (h2 : gpa1 = 54)
  (h3 : gpa2 = 48)
  (h4 : gpa3 = 45)
  (h5 : (total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
        (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3 = 
        total_students * 48.75) : 
  (((total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
    (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3) / total_students) = 48.75 :=
by sorry

end classroom_gpa_l1830_183086


namespace five_Z_three_equals_nineteen_l1830_183084

def Z (x y : ℝ) : ℝ := x^2 - x*y + y^2

theorem five_Z_three_equals_nineteen : Z 5 3 = 19 := by
  sorry

end five_Z_three_equals_nineteen_l1830_183084


namespace identity_function_is_unique_solution_l1830_183024

theorem identity_function_is_unique_solution
  (f : ℕ → ℕ)
  (h : ∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) :
  ∀ n : ℕ, f n = n :=
by sorry

end identity_function_is_unique_solution_l1830_183024


namespace train_platform_ratio_l1830_183000

/-- Given a train passing a pole and a platform, prove the ratio of platform length to train length -/
theorem train_platform_ratio (l t v : ℝ) (h1 : l > 0) (h2 : t > 0) (h3 : v > 0) :
  let pole_time := t
  let platform_time := 3.5 * t
  let train_length := l
  let platform_length := v * platform_time - train_length
  platform_length / train_length = 2.5 := by sorry

end train_platform_ratio_l1830_183000


namespace max_value_theorem_l1830_183074

theorem max_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 * (b + c - a) = b^2 * (a + c - b))
  (h2 : b^2 * (a + c - b) = c^2 * (b + a - c)) :
  ∀ x : ℝ, (2*b + 3*c) / a ≤ 5 :=
by sorry

end max_value_theorem_l1830_183074


namespace geometric_series_relation_l1830_183038

/-- Given two infinite geometric series with specific properties, prove that n = 6 -/
theorem geometric_series_relation (n : ℝ) : 
  let a₁ : ℝ := 15  -- First term of both series
  let b₁ : ℝ := 6   -- Second term of first series
  let b₂ : ℝ := 6 + n  -- Second term of second series
  let r₁ : ℝ := b₁ / a₁  -- Common ratio of first series
  let r₂ : ℝ := b₂ / a₁  -- Common ratio of second series
  let S₁ : ℝ := a₁ / (1 - r₁)  -- Sum of first series
  let S₂ : ℝ := a₁ / (1 - r₂)  -- Sum of second series
  S₂ = 3 * S₁ →  -- Condition: sum of second series is three times the sum of first series
  n = 6 := by
sorry

end geometric_series_relation_l1830_183038


namespace probability_specific_individual_in_sample_l1830_183072

/-- The probability of selecting a specific individual in a simple random sample -/
theorem probability_specific_individual_in_sample 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10)
  (h2 : sample_size = 3)
  (h3 : sample_size < population_size) :
  (sample_size : ℚ) / population_size = 3 / 10 :=
by sorry

end probability_specific_individual_in_sample_l1830_183072


namespace sequence_product_l1830_183032

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that b₂(a₂-a₁) = -8 --/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- arithmetic sequence condition
  (∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d) →  -- arithmetic sequence definition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- geometric sequence condition
  (∃ q : ℝ, b₁ = -9 * q ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q ∧ -1 = b₃ * q) →  -- geometric sequence definition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end sequence_product_l1830_183032


namespace total_collected_is_4336_5_l1830_183044

/-- Represents the total amount collected by Mark during the week in US dollars -/
def total_collected : ℝ :=
  let households_per_day : ℕ := 60
  let days : ℕ := 7
  let total_households : ℕ := households_per_day * days
  let usd_20_percent : ℝ := 0.25
  let eur_15_percent : ℝ := 0.15
  let gbp_10_percent : ℝ := 0.10
  let both_percent : ℝ := 0.05
  let no_donation_percent : ℝ := 0.30
  let usd_20_amount : ℝ := 20
  let eur_15_amount : ℝ := 15
  let gbp_10_amount : ℝ := 10
  let eur_to_usd : ℝ := 1.1
  let gbp_to_usd : ℝ := 1.3

  let usd_20_donation := (usd_20_percent * total_households) * usd_20_amount
  let eur_15_donation := (eur_15_percent * total_households) * eur_15_amount * eur_to_usd
  let gbp_10_donation := (gbp_10_percent * total_households) * gbp_10_amount * gbp_to_usd
  let both_donation := (both_percent * total_households) * (usd_20_amount + eur_15_amount * eur_to_usd)

  usd_20_donation + eur_15_donation + gbp_10_donation + both_donation

theorem total_collected_is_4336_5 :
  total_collected = 4336.5 := by
  sorry

end total_collected_is_4336_5_l1830_183044


namespace greatest_number_of_baskets_l1830_183028

theorem greatest_number_of_baskets (oranges pears bananas : ℕ) 
  (h_oranges : oranges = 18) 
  (h_pears : pears = 27) 
  (h_bananas : bananas = 12) : 
  (Nat.gcd oranges (Nat.gcd pears bananas)) = 3 := by
  sorry

end greatest_number_of_baskets_l1830_183028


namespace tim_grew_44_cantaloupes_l1830_183070

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end tim_grew_44_cantaloupes_l1830_183070


namespace right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l1830_183056

-- Define angles in degrees
def RightAngle : ℝ := 90
def FullCircle : ℝ := 360

-- Define angle types
def IsAcuteAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < RightAngle
def IsObtuseAngle (θ : ℝ) : Prop := RightAngle < θ ∧ θ < 180

theorem right_plus_acute_is_obtuse (θ : ℝ) (h : IsAcuteAngle θ) :
  IsObtuseAngle (RightAngle + θ) := by sorry

theorem quarter_circle_is_right_angle :
  FullCircle / 4 = RightAngle := by sorry

end right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l1830_183056


namespace x_value_l1830_183088

theorem x_value : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ∧ x = 80 / 3 := by
  sorry

end x_value_l1830_183088


namespace book_price_theorem_l1830_183050

theorem book_price_theorem (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.75 * marked_price
  alice_paid / suggested_retail_price = 0.45 := by
  sorry

end book_price_theorem_l1830_183050


namespace special_sequence_2023_l1830_183080

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ m n : ℕ, m > 0 → n > 0 → a (m + n) = a m + a n

/-- The 2023rd term of the special sequence equals 6069 -/
theorem special_sequence_2023 (a : ℕ → ℕ) (h : special_sequence a) : a 2023 = 6069 := by
  sorry

end special_sequence_2023_l1830_183080


namespace sum_of_digits_7_pow_2023_l1830_183004

theorem sum_of_digits_7_pow_2023 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 7^2023 ≡ 10 * a + b [ZMOD 100] ∧ a + b = 16 := by
  sorry

end sum_of_digits_7_pow_2023_l1830_183004


namespace terrys_trip_distance_l1830_183010

/-- Proves that given the conditions of Terry's trip, the total distance driven is 780 miles. -/
theorem terrys_trip_distance :
  ∀ (scenic_road_mpg freeway_mpg : ℝ),
  freeway_mpg = scenic_road_mpg + 6.5 →
  (9 * scenic_road_mpg + 17 * freeway_mpg) / (9 + 17) = 30 →
  9 * scenic_road_mpg + 17 * freeway_mpg = 780 :=
by
  sorry

#check terrys_trip_distance

end terrys_trip_distance_l1830_183010


namespace decimal_93_to_binary_binary_to_decimal_93_l1830_183079

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

theorem binary_to_decimal_93 :
  fromBinary [true, false, true, true, true, false, true] = 93 := by
  sorry

end decimal_93_to_binary_binary_to_decimal_93_l1830_183079


namespace unique_valid_n_l1830_183048

def is_valid (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 210 ∧ 
  (∀ k ∈ Finset.range 2013, (n + (k + 1).factorial) % 210 = 0)

theorem unique_valid_n : ∃! n : ℕ, is_valid n := by
  sorry

end unique_valid_n_l1830_183048


namespace combinations_equal_sixty_l1830_183039

/-- The number of paint colors available -/
def num_colors : ℕ := 5

/-- The number of painting methods available -/
def num_methods : ℕ := 4

/-- The number of pattern options available -/
def num_patterns : ℕ := 3

/-- The total number of unique combinations of color, method, and pattern -/
def total_combinations : ℕ := num_colors * num_methods * num_patterns

/-- Theorem stating that the total number of combinations is 60 -/
theorem combinations_equal_sixty : total_combinations = 60 := by
  sorry

end combinations_equal_sixty_l1830_183039


namespace simple_interest_difference_l1830_183091

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal : ℕ) (rate : ℚ) (time : ℕ) :
  principal = 2500 →
  rate = 4 / 100 →
  time = 5 →
  principal - (principal * rate * time) = 2000 := by
  sorry

end simple_interest_difference_l1830_183091


namespace polynomial_divisibility_l1830_183014

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
  sorry

end polynomial_divisibility_l1830_183014


namespace lcm_factor_proof_l1830_183020

theorem lcm_factor_proof (A B X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 23 →
  A = 414 →
  ∃ (Y : ℕ), Nat.lcm A B = 23 * 13 * X ∧ Nat.lcm A B = 23 * 13 * Y →
  X = 18 := by sorry

end lcm_factor_proof_l1830_183020


namespace probability_of_winning_pair_l1830_183098

def total_cards : ℕ := 12
def cards_per_color : ℕ := 4
def num_colors : ℕ := 3
def num_numbers : ℕ := 4

def winning_pairs : ℕ := 
  (num_colors * (cards_per_color.choose 2)) + (num_numbers * (num_colors.choose 2))

def total_pairs : ℕ := total_cards.choose 2

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / total_pairs = 5 / 11 := by sorry

end probability_of_winning_pair_l1830_183098


namespace unique_factorial_sum_l1830_183002

/-- factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the hundreds digit of a natural number -/
def hundreds_digit (n : ℕ) : ℕ := 
  (n / 100) % 10

/-- Function to get the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := 
  (n / 10) % 10

/-- Function to get the units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := 
  n % 10

/-- Theorem stating that 145 is the only three-digit number with 1 as its hundreds digit 
    that is equal to the sum of the factorials of its digits -/
theorem unique_factorial_sum : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hundreds_digit n = 1 → 
  (n = factorial (hundreds_digit n) + factorial (tens_digit n) + factorial (units_digit n) ↔ n = 145) := by
  sorry

end unique_factorial_sum_l1830_183002


namespace probability_between_C_and_E_l1830_183057

/-- Given points A, B, C, D, and E on a line segment AB, where AB = 4AD = 8BC = 2DE,
    the probability of a randomly selected point on AB being between C and E is 7/8. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < B ∧ A ≤ C ∧ C < D ∧ D < E ∧ E ≤ B ∧
  B - A = 4 * (D - A) ∧
  B - A = 8 * (C - B) ∧
  B - A = 2 * (E - D) →
  (E - C) / (B - A) = 7 / 8 := by
sorry

end probability_between_C_and_E_l1830_183057


namespace blithe_lost_toys_l1830_183025

theorem blithe_lost_toys (initial_toys : ℕ) (found_toys : ℕ) (final_toys : ℕ) 
  (h1 : initial_toys = 40)
  (h2 : found_toys = 9)
  (h3 : final_toys = 43)
  : initial_toys - (final_toys - found_toys) = 9 := by
  sorry

end blithe_lost_toys_l1830_183025


namespace stratified_sample_size_l1830_183082

/-- Represents the total number of employees -/
def total_employees : ℕ := 750

/-- Represents the number of young employees -/
def young_employees : ℕ := 350

/-- Represents the number of middle-aged employees -/
def middle_aged_employees : ℕ := 250

/-- Represents the number of elderly employees -/
def elderly_employees : ℕ := 150

/-- Represents the number of young employees in the sample -/
def young_in_sample : ℕ := 7

/-- Theorem stating that the sample size is 15 given the conditions -/
theorem stratified_sample_size :
  ∃ (sample_size : ℕ),
    sample_size * young_employees = young_in_sample * total_employees ∧
    sample_size = 15 :=
by sorry

end stratified_sample_size_l1830_183082


namespace complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l1830_183062

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems
theorem complement_intersection_A_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x | x < 3 ∨ x ≥ 6} := by sorry

theorem complement_B_union_A :
  ((Set.univ : Set ℝ) \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

theorem a_range_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = Set.Icc 2 8 := by sorry

end complement_intersection_A_B_complement_B_union_A_a_range_for_C_subset_B_l1830_183062


namespace binomial_18_10_l1830_183030

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end binomial_18_10_l1830_183030


namespace shaded_area_between_circles_l1830_183081

theorem shaded_area_between_circles (r : Real) : 
  r > 0 → -- radius of smaller circle is positive
  (2 * r = 6) → -- diameter of smaller circle is 6 units
  π * (3 * r)^2 - π * r^2 = 72 * π := by
  sorry

end shaded_area_between_circles_l1830_183081


namespace double_iced_cubes_count_l1830_183090

/-- Represents a cube cake -/
structure CubeCake where
  size : Nat
  top_iced : Bool
  front_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two sides -/
def count_double_iced_cubes (cake : CubeCake) : Nat :=
  if cake.top_iced && cake.front_iced then
    cake.size - 1
  else
    0

/-- Theorem: A 3x3x3 cake with top and front face iced has 2 subcubes with icing on two sides -/
theorem double_iced_cubes_count :
  let cake : CubeCake := { size := 3, top_iced := true, front_iced := true }
  count_double_iced_cubes cake = 2 := by
  sorry

end double_iced_cubes_count_l1830_183090


namespace sin_bounded_difference_l1830_183034

theorem sin_bounded_difference (a : ℝ) : 
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → |Real.sin x - a| ≤ |Real.sin x₀ - a| := by
  sorry

end sin_bounded_difference_l1830_183034


namespace remainder_problem_l1830_183055

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0)
  (h2 : k < 168)
  (h3 : k % 5 = 2)
  (h4 : k % 6 = 5)
  (h5 : k % 8 = 7)
  (h6 : k % 11 = 3) :
  k % 13 = 8 := by
  sorry

end remainder_problem_l1830_183055


namespace polynomial_sum_l1830_183095

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end polynomial_sum_l1830_183095


namespace max_regular_lines_six_points_l1830_183043

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Enumeration of possible regular line types -/
inductive RegularLineType
  | Horizontal
  | Vertical
  | LeftDiagonal
  | RightDiagonal

/-- A regular line in the 2D plane -/
structure RegularLine where
  type : RegularLineType
  offset : ℝ

/-- Function to check if a point lies on a regular line -/
def pointOnRegularLine (p : Point2D) (l : RegularLine) : Prop :=
  match l.type with
  | RegularLineType.Horizontal => p.y = l.offset
  | RegularLineType.Vertical => p.x = l.offset
  | RegularLineType.LeftDiagonal => p.y - p.x = l.offset
  | RegularLineType.RightDiagonal => p.y + p.x = l.offset

/-- The main theorem stating the maximum number of regular lines -/
theorem max_regular_lines_six_points (points : Fin 6 → Point2D) :
  (∃ (lines : Finset RegularLine), 
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) ∧
    lines.card = 11) ∧
  (∀ (lines : Finset RegularLine),
    (∀ l ∈ lines, ∃ i j, i ≠ j ∧ pointOnRegularLine (points i) l ∧ pointOnRegularLine (points j) l) →
    lines.card ≤ 11) :=
sorry

end max_regular_lines_six_points_l1830_183043


namespace circle_equation_l1830_183053

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -4 * x

-- Define the tangent point
def tangent_point : ℝ × ℝ := (3, -2)

-- State the theorem
theorem circle_equation 
  (C : Circle) 
  (h1 : line_l (tangent_point.1) (tangent_point.2))
  (h2 : center_line C.center.1 C.center.2)
  (h3 : ∃ (t : ℝ), C.center.1 + t * (tangent_point.1 - C.center.1) = tangent_point.1 ∧
                   C.center.2 + t * (tangent_point.2 - C.center.2) = tangent_point.2 ∧
                   t = 1) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 4)^2 = 8 ↔ 
    (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 :=
sorry

end circle_equation_l1830_183053


namespace ellipse_C_equation_min_OP_OQ_sum_l1830_183097

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Theorem for the equation of ellipse C
theorem ellipse_C_equation :
  ∀ a b : ℝ, (ellipse_C a b 1 (Real.sqrt 6 / 3)) →
  (∀ x y : ℝ, hyperbola x y ↔ hyperbola x y) →
  (∀ x y : ℝ, ellipse_C a b x y ↔ x^2 / 3 + y^2 = 1) :=
sorry

-- Define a line passing through two points on the ellipse
def line_through_ellipse (a b : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_C a b x1 y1 ∧ ellipse_C a b x2 y2

-- Define points P and Q on the x-axis
def point_P (x : ℝ) : Prop := x ≠ 0
def point_Q (x : ℝ) : Prop := x ≠ 0

-- Theorem for the minimum value of |OP| + |OQ|
theorem min_OP_OQ_sum :
  ∀ a b x1 y1 x2 y2 p q : ℝ,
  line_through_ellipse a b x1 y1 x2 y2 →
  point_P p → point_Q q →
  |p| + |q| ≥ 2 * Real.sqrt 3 :=
sorry

end ellipse_C_equation_min_OP_OQ_sum_l1830_183097


namespace ellipse_properties_l1830_183041

-- Define the ellipse C
def ellipse_C (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the eccentricity
def eccentricity (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  standard_ellipse x y

-- Define a perpendicular line to x-axis
def perpendicular_to_x (x y x₁ : ℝ) : Prop :=
  x = x₁

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (1, 0)

-- State the theorem
theorem ellipse_properties (a x y x₁ y₁ x₂ y₂ k : ℝ) :
  ellipse_C x y a →
  eccentricity a = 2 →
  line_through_R x y k →
  point_on_ellipse x₁ y₁ →
  point_on_ellipse x₂ y₂ →
  perpendicular_to_x x y x₁ →
  point_on_ellipse x₁ (-y₁) →
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∃ t, t * (x₁, -y₁) + (1 - t) * right_focus = (x₂, y₂)) :=
sorry

end ellipse_properties_l1830_183041


namespace smallest_common_factor_l1830_183083

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 5 → Nat.gcd (11 * m - 3) (8 * m + 4) = 1) ∧ 
  Nat.gcd (11 * 5 - 3) (8 * 5 + 4) > 1 := by
  sorry

end smallest_common_factor_l1830_183083


namespace cube_edge_length_l1830_183069

theorem cube_edge_length (V : ℝ) (h : V = 32 * Real.pi / 3) :
  ∃ s : ℝ, s = 4 * Real.sqrt 3 / 3 ∧ V = 4 * Real.pi * (s * Real.sqrt 3 / 2)^3 / 3 := by
  sorry

end cube_edge_length_l1830_183069


namespace increase_by_percentage_l1830_183060

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 784.3 →
  percentage = 28.5 →
  final = initial * (1 + percentage / 100) →
  final = 1007.8255 := by
  sorry

end increase_by_percentage_l1830_183060


namespace rachel_essay_editing_time_l1830_183064

/-- Rachel's essay writing problem -/
theorem rachel_essay_editing_time 
  (writing_rate : ℕ → ℕ)  -- Function mapping pages to minutes
  (research_time : ℕ)     -- Time spent researching in minutes
  (total_pages : ℕ)       -- Total pages written
  (total_time : ℕ)        -- Total time spent on the essay in minutes
  (h1 : writing_rate 1 = 30)  -- Writing rate is 1 page per 30 minutes
  (h2 : research_time = 45)   -- 45 minutes spent researching
  (h3 : total_pages = 6)      -- 6 pages written in total
  (h4 : total_time = 5 * 60)  -- Total time is 5 hours (300 minutes)
  : total_time - (research_time + writing_rate total_pages) = 75 := by
  sorry

#check rachel_essay_editing_time

end rachel_essay_editing_time_l1830_183064


namespace tom_current_blue_tickets_l1830_183035

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the conversion rates between ticket types -/
structure TicketConversion where
  yellow_to_red : ℕ
  red_to_blue : ℕ

/-- Theorem: Given the conditions, Tom currently has 7 blue tickets -/
theorem tom_current_blue_tickets 
  (total_yellow_needed : ℕ)
  (conversion : TicketConversion)
  (tom_tickets : TomTickets)
  (additional_blue_needed : ℕ)
  (h1 : total_yellow_needed = 10)
  (h2 : conversion.yellow_to_red = 10)
  (h3 : conversion.red_to_blue = 10)
  (h4 : tom_tickets.yellow = 8)
  (h5 : tom_tickets.red = 3)
  (h6 : additional_blue_needed = 163) :
  tom_tickets.blue = 7 := by
  sorry

#check tom_current_blue_tickets

end tom_current_blue_tickets_l1830_183035


namespace max_product_with_sum_2016_l1830_183087

theorem max_product_with_sum_2016 :
  ∀ x y : ℤ, x + y = 2016 → x * y ≤ 1016064 := by
  sorry

end max_product_with_sum_2016_l1830_183087


namespace smallest_union_size_l1830_183076

theorem smallest_union_size (X Y : Finset ℕ) : 
  Finset.card X = 30 → 
  Finset.card Y = 25 → 
  Finset.card (X ∩ Y) ≥ 10 → 
  45 ≤ Finset.card (X ∪ Y) ∧ ∃ X' Y' : Finset ℕ, 
    Finset.card X' = 30 ∧ 
    Finset.card Y' = 25 ∧ 
    Finset.card (X' ∩ Y') ≥ 10 ∧ 
    Finset.card (X' ∪ Y') = 45 :=
by
  sorry

end smallest_union_size_l1830_183076


namespace ant_walk_probability_l1830_183011

/-- The probability of returning to the starting vertex after n moves on a square,
    given the probability of moving clockwise and counter-clockwise. -/
def return_probability (n : ℕ) (p_cw : ℚ) (p_ccw : ℚ) : ℚ :=
  sorry

/-- The number of ways to choose k items from n items. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  sorry

theorem ant_walk_probability :
  let n : ℕ := 6
  let p_cw : ℚ := 2/3
  let p_ccw : ℚ := 1/3
  return_probability n p_cw p_ccw = 160/729 :=
sorry

end ant_walk_probability_l1830_183011


namespace max_sum_with_reciprocals_l1830_183073

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 := by
  sorry

#check max_sum_with_reciprocals

end max_sum_with_reciprocals_l1830_183073


namespace aquarium_height_is_three_l1830_183006

/-- Represents an aquarium with given dimensions and water filling process --/
structure Aquarium where
  length : ℝ
  width : ℝ
  height : ℝ
  initialFillFraction : ℝ
  spillFraction : ℝ
  finalMultiplier : ℝ

/-- Calculates the final volume of water in the aquarium after the described process --/
def finalVolume (a : Aquarium) : ℝ :=
  a.length * a.width * a.height * a.initialFillFraction * (1 - a.spillFraction) * a.finalMultiplier

/-- Theorem stating that an aquarium with the given properties has a height of 3 feet --/
theorem aquarium_height_is_three :
  ∀ (a : Aquarium),
    a.length = 4 →
    a.width = 6 →
    a.initialFillFraction = 1/2 →
    a.spillFraction = 1/2 →
    a.finalMultiplier = 3 →
    finalVolume a = 54 →
    a.height = 3 := by sorry

end aquarium_height_is_three_l1830_183006


namespace vector_AB_equals_zero_three_l1830_183037

-- Define points A and B
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (1, 2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_AB_equals_zero_three : vectorAB = (0, 3) := by
  sorry

end vector_AB_equals_zero_three_l1830_183037


namespace value_of_120abc_l1830_183071

theorem value_of_120abc (a b c d : ℝ) 
  (h1 : 10 * a = 20) 
  (h2 : 6 * b = 20) 
  (h3 : c^2 + d^2 = 50) : 
  120 * a * b * c = 800 * Real.sqrt (50 - d^2) := by
  sorry

end value_of_120abc_l1830_183071


namespace stratified_sampling_first_grade_l1830_183093

theorem stratified_sampling_first_grade (total_students : ℕ) (sample_size : ℕ) 
  (grade_1_ratio grade_2_ratio grade_3_ratio : ℕ) :
  total_students = 2400 →
  sample_size = 120 →
  grade_1_ratio = 5 →
  grade_2_ratio = 4 →
  grade_3_ratio = 3 →
  (grade_1_ratio * sample_size) / (grade_1_ratio + grade_2_ratio + grade_3_ratio) = 50 := by
  sorry

#check stratified_sampling_first_grade

end stratified_sampling_first_grade_l1830_183093


namespace exp_cos_inequality_l1830_183096

theorem exp_cos_inequality : 
  (Real.exp (Real.cos 1)) / (Real.cos 2 + 1) < 2 * Real.sqrt (Real.exp 1) := by
  sorry

end exp_cos_inequality_l1830_183096


namespace two_x_eq_zero_is_linear_l1830_183052

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem: The equation 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry


end two_x_eq_zero_is_linear_l1830_183052


namespace expression_equals_minus_15i_l1830_183049

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (1 + i) / (1 - i)

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression to be evaluated -/
noncomputable def expression : ℂ := 
  binomial 8 1 + 
  binomial 8 2 * z + 
  binomial 8 3 * z^2 + 
  binomial 8 4 * z^3 + 
  binomial 8 5 * z^4 + 
  binomial 8 6 * z^5 + 
  binomial 8 7 * z^6 + 
  binomial 8 8 * z^7

theorem expression_equals_minus_15i : expression = -15 * i := by sorry

end expression_equals_minus_15i_l1830_183049


namespace range_of_a_l1830_183051

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (h : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end range_of_a_l1830_183051


namespace cubes_volume_ratio_l1830_183001

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along each dimension -/
def cubesFit (boxDim : ℕ) (cubeDim : ℕ) : ℕ :=
  boxDim / cubeDim

/-- Calculates the volume occupied by cubes in the box -/
def cubesVolume (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  let l := cubesFit box.length cubeDim
  let w := cubesFit box.width cubeDim
  let h := cubesFit box.height cubeDim
  l * w * h * (cubeDim ^ 3)

/-- The main theorem to be proved -/
theorem cubes_volume_ratio (box : BoxDimensions) (cubeDim : ℕ) : 
  box.length = 8 → box.width = 6 → box.height = 12 → cubeDim = 4 →
  (cubesVolume box cubeDim : ℚ) / (boxVolume box : ℚ) = 2 / 3 := by
  sorry


end cubes_volume_ratio_l1830_183001


namespace total_value_is_305_l1830_183077

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total value of all coins and cash -/
def total_value : ℕ := num_gold_coins * gold_coin_value + num_silver_coins * silver_coin_value + cash

theorem total_value_is_305 : total_value = 305 := by
  sorry

end total_value_is_305_l1830_183077


namespace paper_pack_sheets_l1830_183003

theorem paper_pack_sheets : ∃ (S P : ℕ), S = 115 ∧ S - P = 100 ∧ 5 * P + 35 = S := by
  sorry

end paper_pack_sheets_l1830_183003


namespace smallest_radii_sum_squares_l1830_183066

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if four points lie on a circle -/
def onCircle (A B C D : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem smallest_radii_sum_squares
  (A : Point) (B : Point) (C : Point) (D : Point)
  (h1 : A = ⟨0, 0⟩)
  (h2 : B = ⟨-1, -1⟩)
  (h3 : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ C = ⟨x, y⟩ ∧ D = ⟨x + 1, y⟩)
  (h4 : onCircle A B C D) :
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > r₁ ∧
    (∀ (r : ℝ), onCircle A B C D → r ≥ r₁) ∧
    (∀ (r : ℝ), onCircle A B C D ∧ r ≠ r₁ → r ≥ r₂) ∧
    r₁^2 + r₂^2 = 1381 := by
  sorry

end smallest_radii_sum_squares_l1830_183066


namespace smallest_factor_for_square_l1830_183047

theorem smallest_factor_for_square (a : ℕ) : 
  3150 = 2 * 3^2 * 5^2 * 7 → 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬ ∃ m : ℕ, 3150 * k = m^2) ∧
  (∃ m : ℕ, 3150 * 14 = m^2) ∧
  (14 > 0) :=
by sorry

end smallest_factor_for_square_l1830_183047


namespace greatest_x_with_lcm_l1830_183027

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 15 21) ∧ lcm = 105) →
  x ≤ 105 :=
by
  sorry

end greatest_x_with_lcm_l1830_183027


namespace stating_tournament_orderings_l1830_183022

/-- Represents the number of players in the tournament -/
def num_players : Nat := 6

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Calculates the number of possible orderings in the tournament -/
def num_orderings : Nat := outcomes_per_game ^ (num_players - 1)

/-- 
Theorem stating that the number of possible orderings in the tournament is 32
given the specified number of players and outcomes per game.
-/
theorem tournament_orderings :
  num_orderings = 32 :=
by sorry

end stating_tournament_orderings_l1830_183022


namespace hyperbola_equation_l1830_183058

theorem hyperbola_equation (m a b : ℝ) :
  (∀ x y : ℝ, x^2 / (4 + m^2) + y^2 / m^2 = 1 → x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (a^2 / b^2 = 4) →
  x^2 - y^2 / 3 = 1 := by
  sorry

end hyperbola_equation_l1830_183058


namespace cab_travel_time_l1830_183061

/-- Proves that if a cab travels at 5/6 of its usual speed and arrives 6 minutes late, its usual travel time is 30 minutes. -/
theorem cab_travel_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/10)) : 
  usual_time = 1/2 := by
sorry

end cab_travel_time_l1830_183061


namespace M_mod_1000_l1830_183033

/-- Number of blue flags -/
def blue_flags : ℕ := 12

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
noncomputable def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end M_mod_1000_l1830_183033


namespace knights_adjacent_probability_l1830_183031

def numKnights : ℕ := 20
def chosenKnights : ℕ := 4

def probability_no_adjacent (n k : ℕ) : ℚ :=
  (n - 3) * (n - 5) * (n - 7) * (n - 9) / (n.choose k)

theorem knights_adjacent_probability :
  ∃ (Q : ℚ), Q = 1 - probability_no_adjacent numKnights chosenKnights :=
sorry

end knights_adjacent_probability_l1830_183031


namespace polynomial_equality_l1830_183094

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 2) = x^2 + b*x - 6) → (a = 3 ∧ b = 1) :=
by
  sorry

end polynomial_equality_l1830_183094


namespace digit_sum_divisible_by_11_l1830_183029

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any 39 successive natural numbers, at least one has a digit sum divisible by 11 -/
theorem digit_sum_divisible_by_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (n + k) % 11 = 0) := by sorry

end digit_sum_divisible_by_11_l1830_183029


namespace no_right_triangle_with_75_median_l1830_183008

theorem no_right_triangle_with_75_median (a b c : ℕ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (Nat.gcd a (Nat.gcd b c) = 1) →  -- (a, b, c) = 1
  ¬(((a * a + 4 * b * b : ℚ) / 4 = 15 * 15 / 4) ∨  -- median to leg
    (2 * a * a + 2 * b * b - c * c : ℚ) / 4 = 15 * 15 / 4)  -- median to hypotenuse
:= by sorry

end no_right_triangle_with_75_median_l1830_183008


namespace new_student_height_l1830_183007

def original_heights : List ℝ := [145, 139, 155, 160, 143]

def average_increase : ℝ := 1.2

theorem new_student_height :
  let original_sum := original_heights.sum
  let original_count := original_heights.length
  let original_average := original_sum / original_count
  let new_average := original_average + average_increase
  let new_count := original_count + 1
  let new_sum := new_average * new_count
  new_sum - original_sum = 155.6 := by sorry

end new_student_height_l1830_183007


namespace combined_painting_time_l1830_183067

/-- Given Shawn's and Karen's individual painting rates, calculate their combined time to paint one house -/
theorem combined_painting_time (shawn_rate karen_rate : ℝ) (h1 : shawn_rate = 1 / 18) (h2 : karen_rate = 1 / 12) :
  1 / (shawn_rate + karen_rate) = 7.2 := by
  sorry

end combined_painting_time_l1830_183067


namespace new_person_weight_l1830_183036

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the weight of the new person is 65 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end new_person_weight_l1830_183036


namespace morning_afternoon_difference_l1830_183005

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 31

theorem morning_afternoon_difference :
  morning_campers - afternoon_campers = 5 := by
  sorry

end morning_afternoon_difference_l1830_183005


namespace first_interest_rate_is_five_percent_l1830_183092

/-- Proves that the first interest rate is 5% given the problem conditions -/
theorem first_interest_rate_is_five_percent 
  (total_amount : ℝ)
  (first_amount : ℝ)
  (second_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2600)
  (h2 : first_amount = 1600)
  (h3 : second_rate = 6)
  (h4 : total_income = 140)
  (h5 : ∃ r, (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income) :
  ∃ r, r = 5 ∧ (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income :=
by sorry

end first_interest_rate_is_five_percent_l1830_183092


namespace equation_solution_l1830_183063

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 2 * ((1 / x) + (3 / x) / (6 / x)) - (1 / x) = 1.5 ∧ x = 2 := by
  sorry

end equation_solution_l1830_183063


namespace complementary_angles_ratio_l1830_183040

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 4 * b →   -- ratio of angles is 4:1
  b = 18 :=     -- smaller angle is 18 degrees
by
  sorry

end complementary_angles_ratio_l1830_183040


namespace cubic_expansion_coefficient_l1830_183012

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end cubic_expansion_coefficient_l1830_183012


namespace necessary_but_not_sufficient_l1830_183017

theorem necessary_but_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end necessary_but_not_sufficient_l1830_183017


namespace max_log_sum_max_log_sum_attained_l1830_183016

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) ≤ 1 :=
by sorry

theorem max_log_sum_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 6) :
  (Real.log x / Real.log (3/2) + Real.log y / Real.log (3/2)) = 1 ↔ x = 3/2 ∧ y = 1 :=
by sorry

end max_log_sum_max_log_sum_attained_l1830_183016


namespace arithmetic_sequence_sum_l1830_183099

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 1 + a 6 = 12 →
  a 7 + a 8 + a 9 = 45 := by
sorry

end arithmetic_sequence_sum_l1830_183099


namespace number_difference_proof_l1830_183046

theorem number_difference_proof (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end number_difference_proof_l1830_183046


namespace sum_of_roots_l1830_183015

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 17 = 0)
  (hb : b^3 - 3*b^2 + 5*b + 11 = 0) : 
  a + b = 2 := by sorry

end sum_of_roots_l1830_183015


namespace min_balls_for_fifteen_colors_l1830_183019

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_for_fifteen_colors (counts : BallCounts) 
  (h_red : counts.red = 35)
  (h_green : counts.green = 18)
  (h_yellow : counts.yellow = 15)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 12)
  (h_black : counts.black = 12)
  (h_purple : counts.purple = 8) :
  minBallsForColor counts 15 = 89 := by
  sorry

end min_balls_for_fifteen_colors_l1830_183019


namespace wrong_mark_calculation_l1830_183018

/-- Given a class of students with an incorrect average and one student's mark wrongly noted,
    calculate the wrongly noted mark. -/
theorem wrong_mark_calculation 
  (n : ℕ) -- number of students
  (initial_avg : ℚ) -- initial (incorrect) average
  (correct_mark : ℚ) -- correct mark for the student
  (correct_avg : ℚ) -- correct average after fixing the mark
  (h1 : n = 25) -- there are 25 students
  (h2 : initial_avg = 100) -- initial average is 100
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : correct_avg = 98) -- the correct average is 98
  : 
  -- The wrongly noted mark
  (n : ℚ) * initial_avg - ((n : ℚ) * correct_avg - correct_mark) = 60 :=
by sorry

end wrong_mark_calculation_l1830_183018


namespace polynomial_divisibility_l1830_183078

theorem polynomial_divisibility (C D : ℚ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D = 0) → 
  C = -1 ∧ D = -1 := by
sorry

end polynomial_divisibility_l1830_183078


namespace fudge_pan_dimension_l1830_183021

/-- Represents a rectangular pan of fudge --/
structure FudgePan where
  side1 : ℕ
  side2 : ℕ
  pieces : ℕ

/-- Theorem stating the relationship between pan dimensions and number of fudge pieces --/
theorem fudge_pan_dimension (pan : FudgePan) 
  (h1 : pan.side1 = 18)
  (h2 : pan.pieces = 522) :
  pan.side2 = 29 := by
  sorry

#check fudge_pan_dimension

end fudge_pan_dimension_l1830_183021


namespace rectangle_ratio_extension_l1830_183013

theorem rectangle_ratio_extension (x : ℝ) :
  (2*x > 0) →
  (5*x > 0) →
  ((2*x + 9) / (5*x + 9) = 3/7) →
  ((2*x + 18) / (5*x + 18) = 5/11) :=
by sorry

end rectangle_ratio_extension_l1830_183013


namespace inequality_proof_l1830_183065

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end inequality_proof_l1830_183065


namespace sector_angle_measure_l1830_183085

theorem sector_angle_measure (r : ℝ) (l : ℝ) :
  (2 * r + l = 12) →
  (1 / 2 * l * r = 8) →
  (l / r = 1 ∨ l / r = 4) :=
sorry

end sector_angle_measure_l1830_183085


namespace valid_configurations_count_l1830_183075

/-- Represents a configuration of lit and unlit bulbs -/
def BulbConfiguration := List Bool

/-- Checks if a configuration is valid (no adjacent lit bulbs) -/
def isValidConfiguration (config : BulbConfiguration) : Bool :=
  match config with
  | [] => true
  | [_] => true
  | true :: true :: _ => false
  | _ :: rest => isValidConfiguration rest

/-- Counts the number of lit bulbs in a configuration -/
def countLitBulbs (config : BulbConfiguration) : Nat :=
  config.filter id |>.length

/-- Generates all possible configurations for n bulbs -/
def allConfigurations (n : Nat) : List BulbConfiguration :=
  sorry

/-- Counts valid configurations with at least k lit bulbs out of n total bulbs -/
def countValidConfigurations (n k : Nat) : Nat :=
  (allConfigurations n).filter (fun config => 
    isValidConfiguration config && countLitBulbs config ≥ k
  ) |>.length

theorem valid_configurations_count : 
  countValidConfigurations 7 3 = 11 := by sorry

end valid_configurations_count_l1830_183075
