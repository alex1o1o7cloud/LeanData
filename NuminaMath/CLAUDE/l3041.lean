import Mathlib

namespace product_of_square_roots_equals_one_l3041_304108

theorem product_of_square_roots_equals_one :
  let P := Real.sqrt 2012 + Real.sqrt 2013
  let Q := -Real.sqrt 2012 - Real.sqrt 2013
  let R := Real.sqrt 2012 - Real.sqrt 2013
  let S := Real.sqrt 2013 - Real.sqrt 2012
  P * Q * R * S = 1 := by
  sorry

end product_of_square_roots_equals_one_l3041_304108


namespace max_blocks_fit_l3041_304171

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨6, 3, 4⟩

/-- The small block dimensions -/
def smallBlock : BoxDimensions := ⟨3, 1, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Theorem: The maximum number of small blocks that can fit in the large box is 12 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 12 ∧ 
  largeBox.length / smallBlock.length * 
  largeBox.width / smallBlock.width * 
  largeBox.height / smallBlock.height = 12 := by
  sorry

end max_blocks_fit_l3041_304171


namespace largest_divisor_of_p_cubed_minus_p_l3041_304190

theorem largest_divisor_of_p_cubed_minus_p (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) :
  (∃ (k : ℕ), k * 12 = p^3 - p) ∧
  (∀ (d : ℕ), d > 12 → ¬(∀ (q : ℕ), Prime q → q ≥ 5 → ∃ (k : ℕ), k * d = q^3 - q)) :=
by sorry

end largest_divisor_of_p_cubed_minus_p_l3041_304190


namespace largest_two_digit_one_less_multiple_l3041_304109

theorem largest_two_digit_one_less_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 60 * k) ∧
  (∀ m : ℕ, m > n → m < 100 → ¬∃ j : ℕ, m + 1 = 60 * j) ∧
  n = 59 := by
  sorry

end largest_two_digit_one_less_multiple_l3041_304109


namespace sequence_properties_l3041_304160

def is_arithmetic_progression (s : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, s (n + 1) - s n = d

theorem sequence_properties
  (a b c : ℕ+ → ℝ)
  (h1 : ∀ n : ℕ+, b n = a n - 2 * a (n + 1))
  (h2 : ∀ n : ℕ+, c n = a (n + 1) + 2 * a (n + 2) - 2) :
  (is_arithmetic_progression a → is_arithmetic_progression b) ∧
  (is_arithmetic_progression b ∧ is_arithmetic_progression c →
    ∃ d : ℝ, ∀ n : ℕ+, n ≥ 2 → a (n + 1) - a n = d) ∧
  (is_arithmetic_progression b ∧ b 1 + a 3 = 0 → is_arithmetic_progression a) :=
sorry

end sequence_properties_l3041_304160


namespace digits_of_s_1000_l3041_304129

/-- s(n) is an n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (m : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in s(1000) is 2893 -/
theorem digits_of_s_1000 : num_digits (s 1000) = 2893 := by sorry

end digits_of_s_1000_l3041_304129


namespace line_slope_l3041_304128

theorem line_slope (x y : ℝ) : 3 * y = 4 * x - 12 → (y - (-4)) / (x - 0) = 4 / 3 := by
  sorry

end line_slope_l3041_304128


namespace sqrt_neg_five_squared_l3041_304143

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end sqrt_neg_five_squared_l3041_304143


namespace zoe_pop_albums_l3041_304173

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 3

/-- Represents the number of country albums bought -/
def country_albums : ℕ := 3

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 24

/-- Calculates the number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem zoe_pop_albums : pop_albums = 5 := by
  sorry

end zoe_pop_albums_l3041_304173


namespace max_k_value_l3041_304169

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2/y^2 + y^2/x^2) + 2*k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 56) / 2 :=
sorry

end max_k_value_l3041_304169


namespace square_roots_equality_l3041_304142

theorem square_roots_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
sorry

end square_roots_equality_l3041_304142


namespace quadratic_equation_range_l3041_304170

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x + m - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 - 4*x₁ + m - 1 = 0) →
  (x₂^2 - 4*x₂ + m - 1 = 0) →
  (3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end quadratic_equation_range_l3041_304170


namespace sequence_properties_l3041_304157

def a : ℕ → ℕ
  | n => if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)

def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

theorem sequence_properties :
  (∀ k : ℕ, a (2 * k + 1) = a 1 + k * (a 3 - a 1)) ∧
  (∀ k : ℕ, k > 0 → a (2 * k) = a 2 * (a 4 / a 2) ^ (k - 1)) ∧
  (S 5 = 2 * a 4 + a 5) ∧
  (a 9 = a 3 + a 4) →
  (∀ n : ℕ, n > 0 → a n = if n % 2 = 1 then n else 2 * 3^((n / 2) - 1)) ∧
  (∀ m : ℕ, m > 0 → (a m * a (m + 1) = a (m + 2)) ↔ m = 2) ∧
  (∀ m : ℕ, m > 0 → (∃ k : ℕ, k > 0 ∧ S (2 * m) / S (2 * m - 1) = a k) ↔ (m = 1 ∨ m = 2)) := by
  sorry

#check sequence_properties

end sequence_properties_l3041_304157


namespace book_distribution_l3041_304176

theorem book_distribution (x : ℕ) (total_books : ℕ) : 
  (9 * x + 7 ≤ total_books) ∧ (total_books < 11 * x) →
  (9 * x + 7 = total_books) :=
by sorry

end book_distribution_l3041_304176


namespace arithmetic_calculation_l3041_304153

theorem arithmetic_calculation : (((3.242 * (14 + 6)) - (7.234 * 7)) / 20) = 0.7101 := by
  sorry

end arithmetic_calculation_l3041_304153


namespace unique_number_product_sum_digits_l3041_304146

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the only number satisfying the condition -/
theorem unique_number_product_sum_digits : 
  ∃! n : ℕ, n * sum_of_digits n = 2008 ∧ n > 0 := by sorry

end unique_number_product_sum_digits_l3041_304146


namespace percentage_of_female_brunettes_l3041_304186

theorem percentage_of_female_brunettes 
  (total_students : ℕ) 
  (female_percentage : ℚ)
  (short_brunette_percentage : ℚ)
  (short_brunette_count : ℕ) :
  total_students = 200 →
  female_percentage = 3/5 →
  short_brunette_percentage = 1/2 →
  short_brunette_count = 30 →
  (short_brunette_count : ℚ) / (short_brunette_percentage * (female_percentage * total_students)) = 1/2 :=
by sorry

end percentage_of_female_brunettes_l3041_304186


namespace olivia_money_made_l3041_304104

/-- Represents the types of chocolate bars -/
inductive ChocolateType
| A
| B
| C

/-- The cost of each type of chocolate bar -/
def cost (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 3
  | ChocolateType.B => 4
  | ChocolateType.C => 5

/-- The total number of bars in the box -/
def total_bars : ℕ := 15

/-- The number of bars of each type in the box -/
def bars_in_box (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 7
  | ChocolateType.B => 5
  | ChocolateType.C => 3

/-- The number of bars sold of each type -/
def bars_sold (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 4
  | ChocolateType.B => 3
  | ChocolateType.C => 2

/-- The total money made from selling the chocolate bars -/
def total_money : ℕ :=
  (bars_sold ChocolateType.A * cost ChocolateType.A) +
  (bars_sold ChocolateType.B * cost ChocolateType.B) +
  (bars_sold ChocolateType.C * cost ChocolateType.C)

theorem olivia_money_made :
  total_money = 34 :=
by sorry

end olivia_money_made_l3041_304104


namespace cost_of_hundred_nuggets_l3041_304133

/-- Calculates the total cost of chicken nuggets -/
def chicken_nugget_cost (total_nuggets : ℕ) (nuggets_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_nuggets / nuggets_per_box) * cost_per_box

/-- Theorem: The cost of 100 chicken nuggets is $20 -/
theorem cost_of_hundred_nuggets :
  chicken_nugget_cost 100 20 4 = 20 := by
  sorry

end cost_of_hundred_nuggets_l3041_304133


namespace math_club_probability_l3041_304100

def club_sizes : List Nat := [6, 9, 10]
def co_presidents_per_club : Nat := 3
def members_selected : Nat := 4

def probability_two_copresidents (n : Nat) : Rat :=
  (Nat.choose co_presidents_per_club 2 * Nat.choose (n - co_presidents_per_club) 2) /
  Nat.choose n members_selected

theorem math_club_probability : 
  (1 / 3 : Rat) * (club_sizes.map probability_two_copresidents).sum = 44 / 105 := by
  sorry

end math_club_probability_l3041_304100


namespace square_removal_domino_tiling_l3041_304107

theorem square_removal_domino_tiling (n m : ℕ) (hn : n = 2011) (hm : m = 11) :
  (∃ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2)) ∧
  (∀ (k : ℕ), k = (n - m + 1)^2 / 2 + ((n - m + 1)^2 % 2) → k = 2002001) :=
by sorry

end square_removal_domino_tiling_l3041_304107


namespace hall_breadth_proof_l3041_304180

/-- Given a rectangular hall and stones with specified dimensions, 
    prove that the breadth of the hall is 15 meters. -/
theorem hall_breadth_proof (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) 
                            (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.4 →
  stone_width = 0.5 →
  num_stones = 2700 →
  hall_length * (num_stones * stone_length * stone_width / hall_length) = 15 := by
  sorry

#check hall_breadth_proof

end hall_breadth_proof_l3041_304180


namespace nickel_difference_l3041_304167

/-- Given that Alice has 3p + 2 nickels and Bob has 2p + 6 nickels,
    the difference in their money in pennies is 5p - 20 --/
theorem nickel_difference (p : ℤ) : 
  let alice_nickels : ℤ := 3 * p + 2
  let bob_nickels : ℤ := 2 * p + 6
  let nickel_value : ℤ := 5  -- value of a nickel in pennies
  5 * p - 20 = nickel_value * (alice_nickels - bob_nickels) :=
by sorry

end nickel_difference_l3041_304167


namespace cubic_integer_roots_l3041_304112

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a cubic polynomial, counting multiplicity -/
def num_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- The theorem stating the possible values for the number of integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 3 := by
  sorry

end cubic_integer_roots_l3041_304112


namespace dividend_calculation_l3041_304113

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 47.5)
  (h2 : quotient = 24.3)
  (h3 : remainder = 32.4) :
  divisor * quotient + remainder = 1186.15 := by
  sorry

end dividend_calculation_l3041_304113


namespace browser_tabs_remaining_l3041_304183

theorem browser_tabs_remaining (initial_tabs : ℕ) : 
  initial_tabs = 400 → 
  (initial_tabs - initial_tabs / 4 - (initial_tabs - initial_tabs / 4) * 2 / 5) / 2 = 90 := by
  sorry

end browser_tabs_remaining_l3041_304183


namespace cosine_sine_sum_identity_l3041_304125

theorem cosine_sine_sum_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end cosine_sine_sum_identity_l3041_304125


namespace inequality_holds_iff_l3041_304141

theorem inequality_holds_iff (n k : ℕ) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → 
    a^k * b^k * (a^2 + b^2)^n ≤ (a + b)^(2*k + 2*n) / 2^(2*k + n)) ↔ 
  k ≥ n := by
sorry

end inequality_holds_iff_l3041_304141


namespace min_value_expression_l3041_304131

theorem min_value_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a + 2 * b = 1) :
  ∃ (m : ℝ), m = 2/3 ∧ ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 3 * x + 2 * y = 1 → 
    1 / (12 * x + 1) + 1 / (8 * y + 1) ≥ m :=
by sorry

end min_value_expression_l3041_304131


namespace constant_term_equals_96_l3041_304163

/-- The constant term in the expansion of (2x + a/x)^4 -/
def constantTerm (a : ℝ) : ℝ := a^2 * 2^2 * 6

theorem constant_term_equals_96 (a : ℝ) (h : a > 0) : 
  constantTerm a = 96 → a = 2 := by sorry

end constant_term_equals_96_l3041_304163


namespace conference_handshakes_l3041_304168

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total_people : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h_total : total_people = group1_size + group2_size)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  let group2_external := conf.group2_size * (conf.total_people - 1)
  let group2_internal := (conf.group2_size * (conf.group2_size - 1)) / 2
  group2_external + group2_internal

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total_people = 50 ∧
    conf.group1_size = 30 ∧
    conf.group2_size = 20 ∧
    handshakes conf = 1170 := by
  sorry

end conference_handshakes_l3041_304168


namespace joan_books_count_l3041_304127

/-- The number of books Joan sold in the yard sale -/
def books_sold : ℕ := 26

/-- The number of books Joan has left after the sale -/
def books_left : ℕ := 7

/-- The total number of books Joan gathered to sell -/
def total_books : ℕ := books_sold + books_left

theorem joan_books_count : total_books = 33 := by sorry

end joan_books_count_l3041_304127


namespace square_tiles_count_l3041_304158

theorem square_tiles_count (h s : ℕ) : 
  h + s = 30 →  -- Total number of tiles
  6 * h + 4 * s = 128 →  -- Total number of edges
  s = 26 :=  -- Number of square tiles
by
  sorry

end square_tiles_count_l3041_304158


namespace decrease_six_l3041_304175

def temperature_change : ℝ → ℝ := id

axiom positive_rise (x : ℝ) : x > 0 → temperature_change x > 0

axiom rise_three : temperature_change 3 = 3

theorem decrease_six : temperature_change (-6) = -6 := by sorry

end decrease_six_l3041_304175


namespace remainder_after_adding_1008_l3041_304105

theorem remainder_after_adding_1008 (n : ℤ) : 
  n % 4 = 1 → (n + 1008) % 4 = 1 := by
sorry

end remainder_after_adding_1008_l3041_304105


namespace distance_to_softball_park_l3041_304182

/-- Represents the problem of calculating the distance to the softball park -/
def softball_park_distance (efficiency : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_restaurant : ℝ) (to_friend : ℝ) (to_home : ℝ) : ℝ :=
  efficiency * initial_gas - (to_school + to_restaurant + to_friend + to_home)

/-- Theorem stating that the distance to the softball park is 6 miles -/
theorem distance_to_softball_park :
  softball_park_distance 19 2 15 2 4 11 = 6 := by
  sorry

end distance_to_softball_park_l3041_304182


namespace division_with_same_remainder_l3041_304156

theorem division_with_same_remainder (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℤ, 200 = k * x + 2) :
  ∀ n : ℤ, ∃ k : ℤ, 200 = k * x + 2 ∧ n ≠ k → ∃ m : ℤ, n * x + 2 = m * x + (n * x + 2) % x ∧ (n * x + 2) % x = 2 :=
by sorry

end division_with_same_remainder_l3041_304156


namespace cube_sum_problem_l3041_304126

theorem cube_sum_problem (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : p * q + p * r + q * r = 7)
  (h3 : p * q * r = -10) :
  p^3 + q^3 + r^3 = -10 := by
  sorry

end cube_sum_problem_l3041_304126


namespace problem_2005_squared_minus_2003_times_2007_l3041_304189

theorem problem_2005_squared_minus_2003_times_2007 : 2005^2 - 2003 * 2007 = 4 := by
  sorry

end problem_2005_squared_minus_2003_times_2007_l3041_304189


namespace problem_solution_l3041_304154

theorem problem_solution (A B : ℝ) (hB : B ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x^2
  f (g 1) = 0 → A = 3 := by
sorry

end problem_solution_l3041_304154


namespace fraction_problem_l3041_304139

theorem fraction_problem (x : ℚ) :
  (x / (4 * x + 5) = 3 / 7) → x = -3 := by
  sorry

end fraction_problem_l3041_304139


namespace age_ratio_problem_l3041_304199

theorem age_ratio_problem (amy jeremy chris : ℕ) : 
  amy + jeremy + chris = 132 →
  amy = jeremy / 3 →
  jeremy = 66 →
  ∃ k : ℕ, chris = k * amy →
  chris / amy = 2 := by
sorry

end age_ratio_problem_l3041_304199


namespace mary_sugar_added_l3041_304122

/-- Given a recipe that requires a certain amount of sugar and the amount of sugar still needed,
    calculate the amount of sugar already added. -/
def sugar_already_added (recipe_required : ℕ) (sugar_needed : ℕ) : ℕ :=
  recipe_required - sugar_needed

/-- Theorem stating that Mary has already added 10 cups of sugar. -/
theorem mary_sugar_added :
  sugar_already_added 11 1 = 10 := by
  sorry

end mary_sugar_added_l3041_304122


namespace license_plate_count_l3041_304192

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of alphanumeric characters (letters + digits) -/
def num_alphanumeric : ℕ := num_letters + num_digits

/-- The number of different license plates that can be formed -/
def num_license_plates : ℕ := num_letters * num_digits * num_alphanumeric

theorem license_plate_count : num_license_plates = 9360 := by
  sorry

end license_plate_count_l3041_304192


namespace triangle_side_length_l3041_304117

/-- Given a triangle ABC with side a = 8, angle B = 30°, and angle C = 105°, 
    prove that the length of side b is equal to 4√2. -/
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) : 
  a = 8 → B = 30 * π / 180 → C = 105 * π / 180 → 
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l3041_304117


namespace maria_friends_money_l3041_304136

/-- The amount of money Maria gave to her three friends -/
def total_given (maria_money : ℝ) (isha_share : ℝ) (florence_share : ℝ) (rene_share : ℝ) : ℝ :=
  isha_share + florence_share + rene_share

/-- Theorem stating the total amount Maria gave to her friends -/
theorem maria_friends_money :
  ∀ (maria_money : ℝ),
  maria_money > 0 →
  let isha_share := (1/3) * maria_money
  let florence_share := (1/2) * isha_share
  let rene_share := 300
  florence_share = 3 * rene_share →
  total_given maria_money isha_share florence_share rene_share = 3000 := by
  sorry

end maria_friends_money_l3041_304136


namespace rectangle_width_three_l3041_304147

/-- A rectangle with length twice its width and area equal to perimeter has width 3. -/
theorem rectangle_width_three (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 6 * w) → w = 3 := by
  sorry

end rectangle_width_three_l3041_304147


namespace quadratic_inequality_solution_l3041_304119

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the second inequality
def g (a b c : ℝ) (x : ℝ) := a * (x^2 + 1) + b * (x + 1) + c - 3 * a * x

theorem quadratic_inequality_solution 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : solution_set a b c = Set.Ioo (-2) 1) :
  {x : ℝ | g a b c x < 0} = Set.Iic 0 ∪ Set.Ioi 2 :=
sorry

end quadratic_inequality_solution_l3041_304119


namespace complex_equation_solution_l3041_304137

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = 2) : z = 1 - i := by
  sorry

end complex_equation_solution_l3041_304137


namespace quadratic_equations_solutions_l3041_304162

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2 * x^2 - 2 * Real.sqrt 2 * x + 1 = 0 ∧ x = Real.sqrt 2 / 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ * (2 * x₁ - 5) = 4 * x₁ - 10 ∧
                x₂ * (2 * x₂ - 5) = 4 * x₂ - 10 ∧
                x₁ = 5 / 2 ∧ x₂ = 2) :=
by sorry

end quadratic_equations_solutions_l3041_304162


namespace original_paint_intensity_l3041_304164

-- Define the paint mixing problem
def paint_mixing (original_intensity : ℝ) : Prop :=
  let f : ℝ := 1/3  -- fraction of original paint replaced
  let replacement_intensity : ℝ := 20  -- 20% solution
  let final_intensity : ℝ := 40  -- 40% final intensity
  (1 - f) * original_intensity + f * replacement_intensity = final_intensity

-- Theorem statement
theorem original_paint_intensity :
  ∃ (original_intensity : ℝ), paint_mixing original_intensity ∧ original_intensity = 50 := by
  sorry

end original_paint_intensity_l3041_304164


namespace discounted_shoe_price_l3041_304102

/-- Given a pair of shoes bought at a 20% discount for $480, 
    prove that the original price was $600. -/
theorem discounted_shoe_price (discount_rate : ℝ) (discounted_price : ℝ) :
  discount_rate = 0.20 →
  discounted_price = 480 →
  discounted_price = (1 - discount_rate) * 600 :=
by sorry

end discounted_shoe_price_l3041_304102


namespace number_difference_l3041_304120

theorem number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 30000) :
  b - a = 24543 := by
  sorry

end number_difference_l3041_304120


namespace velocity_center_of_mass_before_collision_l3041_304188

/-- Velocity of the center of mass of a two-cart system before collision -/
theorem velocity_center_of_mass_before_collision 
  (m : ℝ) -- mass of cart 1
  (v1_initial : ℝ) -- initial velocity of cart 1
  (m2 : ℝ) -- mass of cart 2
  (v2_initial : ℝ) -- initial velocity of cart 2
  (v1_final : ℝ) -- final velocity of cart 1
  (h1 : v1_initial = 12) -- initial velocity of cart 1 is 12 m/s
  (h2 : m2 = 4) -- mass of cart 2 is 4 kg
  (h3 : v2_initial = 0) -- cart 2 is initially at rest
  (h4 : v1_final = -6) -- final velocity of cart 1 is 6 m/s to the left
  (h5 : m > 0) -- mass of cart 1 is positive
  (h6 : m2 > 0) -- mass of cart 2 is positive
  : (m * v1_initial + m2 * v2_initial) / (m + m2) = 3 := by
  sorry

end velocity_center_of_mass_before_collision_l3041_304188


namespace octal_53_to_decimal_l3041_304123

/-- Converts an octal digit to its decimal value -/
def octal_to_decimal (d : ℕ) : ℕ := d

/-- Converts a two-digit octal number to its decimal equivalent -/
def octal_2digit_to_decimal (d1 d0 : ℕ) : ℕ :=
  octal_to_decimal d1 * 8 + octal_to_decimal d0

/-- The decimal representation of the octal number 53 is 43 -/
theorem octal_53_to_decimal :
  octal_2digit_to_decimal 5 3 = 43 := by sorry

end octal_53_to_decimal_l3041_304123


namespace division_decimal_l3041_304116

theorem division_decimal : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by sorry

end division_decimal_l3041_304116


namespace triangle_area_l3041_304198

-- Define the curve
def curve (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define the y-intercept
def y_intercept : ℝ := curve 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept1 - x_intercept2
  let height := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end triangle_area_l3041_304198


namespace white_ball_players_l3041_304130

theorem white_ball_players (total : ℕ) (yellow : ℕ) (both : ℕ) (h1 : total = 35) (h2 : yellow = 28) (h3 : both = 19) :
  total = (yellow - both) + (total - yellow + both) :=
by sorry

#check white_ball_players

end white_ball_players_l3041_304130


namespace work_completion_time_equivalence_l3041_304114

/-- Represents the work rate of a single worker per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, work rate, and days -/
def work_done (workers : ℕ) (rate : ℝ) (days : ℕ) : ℝ :=
  (workers : ℝ) * rate * (days : ℝ)

/-- Theorem stating that if the work is completed in 40 days with varying workforce,
    it would take 45 days with a constant workforce -/
theorem work_completion_time_equivalence :
  let total_work := work_done 100 work_rate 35 + work_done 200 work_rate 5
  ∃ (days : ℕ), days = 45 ∧ work_done 100 work_rate days = total_work :=
by sorry

end work_completion_time_equivalence_l3041_304114


namespace largest_prime_divisor_of_factorial_sum_l3041_304159

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 12 + factorial 13) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 12 + factorial 13) → q ≤ p :=
by sorry

end largest_prime_divisor_of_factorial_sum_l3041_304159


namespace max_ballpoint_pens_l3041_304181

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- The cost of each type of pen in rubles -/
def penCosts : PenCounts := { ballpoint := 10, gel := 30, fountain := 60 }

/-- The total cost of a given combination of pens -/
def totalCost (counts : PenCounts) : ℕ :=
  counts.ballpoint * penCosts.ballpoint +
  counts.gel * penCosts.gel +
  counts.fountain * penCosts.fountain

/-- The total number of pens -/
def totalPens (counts : PenCounts) : ℕ :=
  counts.ballpoint + counts.gel + counts.fountain

/-- Predicate for a valid pen combination -/
def isValidCombination (counts : PenCounts) : Prop :=
  totalPens counts = 20 ∧
  totalCost counts = 500 ∧
  counts.ballpoint > 0 ∧
  counts.gel > 0 ∧
  counts.fountain > 0

/-- Theorem: The maximum number of ballpoint pens is 11 -/
theorem max_ballpoint_pens :
  ∃ (counts : PenCounts), isValidCombination counts ∧
    counts.ballpoint = 11 ∧
    ∀ (other : PenCounts), isValidCombination other →
      other.ballpoint ≤ counts.ballpoint :=
by sorry

end max_ballpoint_pens_l3041_304181


namespace absolute_value_equation_solution_difference_l3041_304150

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end absolute_value_equation_solution_difference_l3041_304150


namespace circle_properties_l3041_304106

/-- Given a circle with area 16π, prove its diameter is 8 and circumference is 8π -/
theorem circle_properties (r : ℝ) (h : π * r^2 = 16 * π) :
  2 * r = 8 ∧ 2 * π * r = 8 * π := by
  sorry


end circle_properties_l3041_304106


namespace mans_rate_l3041_304134

def with_stream : ℝ := 25
def against_stream : ℝ := 13

theorem mans_rate (with_stream against_stream : ℝ) :
  with_stream = 25 →
  against_stream = 13 →
  (with_stream + against_stream) / 2 = 19 := by
sorry

end mans_rate_l3041_304134


namespace smallest_square_area_for_rectangles_l3041_304140

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a square given its side length -/
def squareArea (side : ℕ) : ℕ := side * side

/-- Checks if two rectangles can fit side by side within a given width -/
def canFitSideBySide (r1 r2 : Rectangle) (width : ℕ) : Prop :=
  r1.width + r2.width ≤ width

/-- Checks if two rectangles can fit one above the other within a given height -/
def canFitStackedVertically (r1 r2 : Rectangle) (height : ℕ) : Prop :=
  r1.height + r2.height ≤ height

/-- The main theorem stating the smallest possible area of the square -/
theorem smallest_square_area_for_rectangles : 
  let r1 : Rectangle := ⟨2, 4⟩
  let r2 : Rectangle := ⟨3, 5⟩
  let minSideLength : ℕ := max (r1.width + r2.width) (r1.height + r2.height)
  ∃ (side : ℕ), 
    side ≥ minSideLength ∧
    canFitSideBySide r1 r2 side ∧
    canFitStackedVertically r1 r2 side ∧
    squareArea side = 81 ∧
    ∀ (s : ℕ), s < side → ¬(canFitSideBySide r1 r2 s ∧ canFitStackedVertically r1 r2 s) :=
by sorry

end smallest_square_area_for_rectangles_l3041_304140


namespace inequality_proof_l3041_304144

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + ((1 / b + 6 * c) ^ (1/3 : ℝ)) + ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) := by
  sorry

end inequality_proof_l3041_304144


namespace sqrt_x_minus_one_real_l3041_304179

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l3041_304179


namespace quadratic_inequality_solution_l3041_304187

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, mx^2 + 8*m*x + 28 < 0 ↔ -7 < x ∧ x < -1) →
  m = 4 := by
  sorry

end quadratic_inequality_solution_l3041_304187


namespace music_class_size_l3041_304184

theorem music_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 29 := by
  sorry

end music_class_size_l3041_304184


namespace consecutive_integers_product_l3041_304124

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 2520 → n = 5 := by
  sorry

end consecutive_integers_product_l3041_304124


namespace weight_difference_l3041_304103

/-- Given the weights of three people (Ishmael, Ponce, and Jalen), prove that Ishmael is 20 pounds heavier than Ponce. -/
theorem weight_difference (I P J : ℝ) : 
  J = 160 →  -- Jalen's weight
  P = J - 10 →  -- Ponce is 10 pounds lighter than Jalen
  (I + P + J) / 3 = 160 →  -- Average weight is 160 pounds
  I - P = 20 :=  -- Ishmael is 20 pounds heavier than Ponce
by sorry

end weight_difference_l3041_304103


namespace polynomial_identity_sum_l3041_304172

theorem polynomial_identity_sum (a b c d e f : ℤ) :
  (∀ x : ℤ, (3 * x + 1)^5 = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end polynomial_identity_sum_l3041_304172


namespace geometric_sequence_sum_bounds_l3041_304155

theorem geometric_sequence_sum_bounds (a : ℕ → ℚ) (S : ℕ → ℚ) (A B : ℚ) :
  (∀ n : ℕ, a n = 4/3 * (-1/3)^n) →
  (∀ n : ℕ, S (n+1) = (4/3 * (1 - (-1/3)^(n+1))) / (1 + 1/3)) →
  (∀ n : ℕ, n > 0 → A ≤ S n - 1 / S n ∧ S n - 1 / S n ≤ B) →
  59/72 ≤ B - A :=
by sorry

end geometric_sequence_sum_bounds_l3041_304155


namespace quadratic_equation_real_roots_l3041_304178

theorem quadratic_equation_real_roots (a : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + (a - 1) = 0 := by
  sorry

end quadratic_equation_real_roots_l3041_304178


namespace weight_of_a_l3041_304177

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 50 →
  (a + b + c + d) / 4 = 53 →
  (b + c + d + e) / 4 = 51 →
  e = d + 3 →
  a = 73 := by
sorry

end weight_of_a_l3041_304177


namespace divisibility_of_power_plus_exponent_l3041_304197

theorem divisibility_of_power_plus_exponent (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, n ∣ (2^m + m) :=
sorry

end divisibility_of_power_plus_exponent_l3041_304197


namespace months_with_average_salary_8900_l3041_304165

def average_salary_jan_to_apr : ℕ := 8000
def average_salary_some_months : ℕ := 8900
def salary_may : ℕ := 6500
def salary_jan : ℕ := 2900

theorem months_with_average_salary_8900 :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / average_salary_some_months = 4 := by
sorry

end months_with_average_salary_8900_l3041_304165


namespace smallest_number_l3041_304191

theorem smallest_number (s : Set ℝ) (hs : s = {0, -2, 1, (1/2 : ℝ)}) :
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -2 :=
by sorry

end smallest_number_l3041_304191


namespace length_ae_is_10_l3041_304152

/-- A quadrilateral with the properties of an isosceles trapezoid and a rectangle -/
structure QuadrilateralABCDE where
  /-- AB is a side of the quadrilateral -/
  ab : ℝ
  /-- EC is a side of the quadrilateral -/
  ec : ℝ
  /-- ABCE is an isosceles trapezoid -/
  abce_isosceles_trapezoid : Bool
  /-- ACDE is a rectangle -/
  acde_rectangle : Bool

/-- The length of AE in the quadrilateral ABCDE -/
def length_ae (q : QuadrilateralABCDE) : ℝ :=
  sorry

/-- Theorem stating that the length of AE is 10 under given conditions -/
theorem length_ae_is_10 (q : QuadrilateralABCDE) 
  (h1 : q.ab = 10) 
  (h2 : q.ec = 20) 
  (h3 : q.abce_isosceles_trapezoid = true) 
  (h4 : q.acde_rectangle = true) : 
  length_ae q = 10 :=
sorry

end length_ae_is_10_l3041_304152


namespace mary_sold_at_least_12_boxes_l3041_304194

/-- The number of cases Mary needs to deliver -/
def cases : ℕ := 2

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The minimum number of boxes Mary sold -/
def min_boxes_sold : ℕ := cases * boxes_per_case

/-- Mary has some extra boxes (number unspecified) -/
axiom has_extra_boxes : ∃ n : ℕ, n > 0

theorem mary_sold_at_least_12_boxes :
  min_boxes_sold ≥ 12 ∧ ∃ total : ℕ, total > min_boxes_sold :=
sorry

end mary_sold_at_least_12_boxes_l3041_304194


namespace arithmetic_sequence_sum_l3041_304101

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : a 1 = 1
  h3 : ∀ n : ℕ, a (n + 1) = a n + d
  h4 : (a 5) ^ 2 = (a 3) * (a 10)

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1) + (n * (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = -3/4 * n^2 + 7/4 * n := by
  sorry

end arithmetic_sequence_sum_l3041_304101


namespace odd_mult_odd_is_odd_l3041_304149

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def P : Set ℕ := {n : ℕ | is_odd n}

theorem odd_mult_odd_is_odd (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

end odd_mult_odd_is_odd_l3041_304149


namespace clock_shows_ten_to_five_l3041_304166

/-- Represents a clock hand --/
inductive ClockHand
  | A
  | B
  | C

/-- Represents the position of a clock hand --/
inductive HandPosition
  | ExactHourMark
  | SlightlyOffHourMark

/-- Represents a clock with three hands --/
structure Clock :=
  (hands : Fin 3 → ClockHand)
  (positions : ClockHand → HandPosition)

/-- The time shown on the clock --/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Checks if the given clock configuration is valid --/
def isValidClock (c : Clock) : Prop :=
  ∃ (h1 h2 : ClockHand), h1 ≠ h2 ∧ 
    c.positions h1 = HandPosition.ExactHourMark ∧
    c.positions h2 = HandPosition.ExactHourMark ∧
    (∀ h, h ≠ h1 → h ≠ h2 → c.positions h = HandPosition.SlightlyOffHourMark)

/-- The main theorem --/
theorem clock_shows_ten_to_five (c : Clock) : 
  isValidClock c → ∃ (t : Time), t.hours = 4 ∧ t.minutes = 50 :=
sorry

end clock_shows_ten_to_five_l3041_304166


namespace quadratic_roots_equality_l3041_304161

theorem quadratic_roots_equality (α β γ p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) → 
  (x₂^2 + p*x₂ + q = 0) → 
  (α * x₁^2 + β * x₁ + γ = α * x₂^2 + β * x₂ + γ) ↔ 
  (p^2 = 4*q ∨ p = -β/α) :=
by sorry

end quadratic_roots_equality_l3041_304161


namespace hyperbola_foci_distance_l3041_304135

theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), 
  (y^2 / 75) - (x^2 / 11) = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 86 :=
by sorry

end hyperbola_foci_distance_l3041_304135


namespace waiter_customer_count_l3041_304121

/-- Represents the scenario of a waiter serving customers and receiving tips -/
structure WaiterScenario where
  total_customers : ℕ
  non_tipping_customers : ℕ
  tip_amount : ℕ
  total_tips : ℕ

/-- Theorem stating that given the conditions, the waiter had 7 customers in total -/
theorem waiter_customer_count (scenario : WaiterScenario) 
  (h1 : scenario.non_tipping_customers = 5)
  (h2 : scenario.tip_amount = 3)
  (h3 : scenario.total_tips = 6) :
  scenario.total_customers = 7 := by
  sorry


end waiter_customer_count_l3041_304121


namespace range_of_m_l3041_304110

theorem range_of_m (m : ℝ) : 
  (m + 4)^(-1/2 : ℝ) < (3 - 2*m)^(-1/2 : ℝ) → 
  -1/3 < m ∧ m < 3/2 :=
by
  sorry

end range_of_m_l3041_304110


namespace taco_truck_profit_l3041_304138

/-- Calculate the profit for a taco truck given the total beef, beef per taco, selling price, and cost to make. -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_to_make : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_to_make = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_to_make) = 200 :=
by
  sorry

#check taco_truck_profit

end taco_truck_profit_l3041_304138


namespace square_of_sum_l3041_304118

theorem square_of_sum (x y : ℝ) : (x + 2*y)^2 = x^2 + 4*x*y + 4*y^2 := by
  sorry

end square_of_sum_l3041_304118


namespace bryan_annual_commute_hours_l3041_304196

/-- Represents the time in minutes for each segment of Bryan's commute -/
structure CommuteSegment where
  walk_to_bus : ℕ
  bus_ride : ℕ
  walk_to_work : ℕ

/-- Represents Bryan's daily commute -/
def daily_commute : CommuteSegment :=
  { walk_to_bus := 5
  , bus_ride := 20
  , walk_to_work := 5 }

/-- Calculates the total time for a one-way commute in minutes -/
def one_way_commute_time (c : CommuteSegment) : ℕ :=
  c.walk_to_bus + c.bus_ride + c.walk_to_work

/-- Calculates the total daily commute time in hours -/
def daily_commute_hours (c : CommuteSegment) : ℚ :=
  (2 * one_way_commute_time c : ℚ) / 60

/-- The number of days Bryan works per year -/
def work_days_per_year : ℕ := 365

/-- Theorem stating that Bryan spends 365 hours per year commuting -/
theorem bryan_annual_commute_hours :
  (daily_commute_hours daily_commute * work_days_per_year : ℚ) = 365 := by
  sorry


end bryan_annual_commute_hours_l3041_304196


namespace log_4_30_l3041_304174

theorem log_4_30 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 30 / Real.log 4 = 1 / (2 * a) := by
  sorry

end log_4_30_l3041_304174


namespace multiplicative_inverse_modulo_l3041_304185

def A : Nat := 123456
def B : Nat := 162738
def M : Nat := 1000000
def N : Nat := 503339

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by
  sorry

end multiplicative_inverse_modulo_l3041_304185


namespace hiker_distance_problem_l3041_304145

theorem hiker_distance_problem (v t d : ℝ) :
  v > 0 ∧ t > 0 ∧ d > 0 ∧
  d = v * t ∧
  d = (v + 1) * (3 * t / 4) ∧
  d = (v - 1) * (t + 3) →
  d = 18 := by
sorry

end hiker_distance_problem_l3041_304145


namespace ray_gave_peter_30_cents_l3041_304195

/-- Given that Ray has 175 cents in nickels, gives twice as many cents to Randi as to Peter,
    and Randi has 6 more nickels than Peter, prove that Ray gave 30 cents to Peter. -/
theorem ray_gave_peter_30_cents (total : ℕ) (peter_cents : ℕ) (randi_cents : ℕ) : 
  total = 175 →
  randi_cents = 2 * peter_cents →
  randi_cents = peter_cents + 6 * 5 →
  peter_cents = 30 := by
sorry

end ray_gave_peter_30_cents_l3041_304195


namespace traveler_time_difference_l3041_304148

/-- Proof of the time difference between two travelers meeting at a point -/
theorem traveler_time_difference 
  (speed_A speed_B meeting_distance : ℝ) 
  (h1 : speed_A > 0)
  (h2 : speed_B > speed_A)
  (h3 : meeting_distance > 0) :
  meeting_distance / speed_A - meeting_distance / speed_B = 7 :=
by sorry

end traveler_time_difference_l3041_304148


namespace raj_ravi_age_difference_l3041_304132

/-- Represents the ages of individuals in the problem -/
structure Ages where
  raj : ℕ
  ravi : ℕ
  hema : ℕ
  rahul : ℕ

/-- Conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ∃ (x : ℕ),
    ages.raj = ages.ravi + x ∧
    ages.hema = ages.ravi - 2 ∧
    ages.raj = 3 * ages.rahul ∧
    ages.hema = (3 / 2 : ℚ) * ages.rahul ∧
    20 = ages.hema + (1 / 3 : ℚ) * ages.hema

/-- The theorem to be proved -/
theorem raj_ravi_age_difference (ages : Ages) :
  problem_conditions ages → ages.raj - ages.ravi = 13 :=
by sorry

end raj_ravi_age_difference_l3041_304132


namespace brass_players_count_l3041_304111

/-- Represents the composition of a marching band -/
structure MarchingBand where
  brass : ℕ
  woodwind : ℕ
  percussion : ℕ

/-- The total number of members in the marching band -/
def MarchingBand.total (band : MarchingBand) : ℕ :=
  band.brass + band.woodwind + band.percussion

/-- Theorem: The number of brass players in the marching band is 10 -/
theorem brass_players_count (band : MarchingBand) :
  band.total = 110 →
  band.woodwind = 2 * band.brass →
  band.percussion = 4 * band.woodwind →
  band.brass = 10 := by
  sorry

end brass_players_count_l3041_304111


namespace nell_card_collection_l3041_304151

/-- Represents the number and types of cards Nell has --/
structure CardCollection where
  baseball : ℕ
  ace : ℕ
  pokemon : ℕ

/-- Represents the initial state of Nell's card collection --/
def initial_collection : CardCollection := {
  baseball := 438,
  ace := 18,
  pokemon := 312
}

/-- Represents the state of Nell's card collection after giving away cards --/
def after_giveaway (c : CardCollection) : CardCollection := {
  baseball := c.baseball - c.baseball / 2,
  ace := c.ace - c.ace / 3,
  pokemon := c.pokemon
}

/-- Represents the final state of Nell's card collection after trading --/
def final_collection (c : CardCollection) : CardCollection := {
  baseball := c.baseball,
  ace := c.ace + 37,
  pokemon := c.pokemon - 52
}

/-- The main theorem to prove --/
theorem nell_card_collection :
  let final := final_collection (after_giveaway initial_collection)
  (final.baseball - final.ace = 170) ∧
  (final.baseball : ℚ) / 219 = (final.ace : ℚ) / 49 ∧
  (final.ace : ℚ) / 49 = (final.pokemon : ℚ) / 260 := by
  sorry


end nell_card_collection_l3041_304151


namespace garden_perimeter_l3041_304115

theorem garden_perimeter : 
  ∀ (width length : ℝ),
  length = 3 * width + 2 →
  length = 38 →
  2 * length + 2 * width = 100 :=
by
  sorry

end garden_perimeter_l3041_304115


namespace grading_implications_l3041_304193

-- Define the type for grades
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade
| D : Grade
| F : Grade

-- Define the ordering on grades
instance : LE Grade where
  le := λ g₁ g₂ => match g₁, g₂ with
    | Grade.F, _ => true
    | Grade.D, Grade.D | Grade.D, Grade.C | Grade.D, Grade.B | Grade.D, Grade.A => true
    | Grade.C, Grade.C | Grade.C, Grade.B | Grade.C, Grade.A => true
    | Grade.B, Grade.B | Grade.B, Grade.A => true
    | Grade.A, Grade.A => true
    | _, _ => false

instance : LT Grade where
  lt := λ g₁ g₂ => g₁ ≤ g₂ ∧ g₁ ≠ g₂

-- Define the grading function
def grading_function (score : ℚ) : Grade :=
  if score ≥ 90 then Grade.B
  else if score < 70 then Grade.C
  else Grade.C  -- Default case, can be any grade between B and C

-- State the theorem
theorem grading_implications :
  (∀ (score : ℚ) (grade : Grade),
    (grading_function score = grade → 
      (grade < Grade.B → score < 90) ∧
      (grade > Grade.C → score ≥ 70))) :=
sorry

end grading_implications_l3041_304193
