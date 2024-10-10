import Mathlib

namespace gcd_65536_49152_l3012_301254

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 := by
  sorry

end gcd_65536_49152_l3012_301254


namespace trigonometric_identities_l3012_301217

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -2) : 
  ((2 * Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi - α)) / 
   (Real.sin (Real.pi / 2 - α) - 3 * Real.sin (Real.pi + α)) = -1) ∧
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11/5) := by
  sorry

end trigonometric_identities_l3012_301217


namespace sqrt_D_irrational_l3012_301273

def D (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 4

theorem sqrt_D_irrational : ∀ x : ℝ, Irrational (Real.sqrt (D x)) := by
  sorry

end sqrt_D_irrational_l3012_301273


namespace partnership_profit_l3012_301265

/-- Given the investments of three partners and the profit share of one partner, 
    calculate the total profit of the partnership. -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 2400)
  (h2 : investment_B = 7200)
  (h3 : investment_C = 9600)
  (h4 : profit_share_A = 1125) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 9000 :=
by sorry

end partnership_profit_l3012_301265


namespace prob_at_least_one_boy_one_girl_l3012_301276

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_one_girl_l3012_301276


namespace intersection_of_A_and_B_l3012_301252

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end intersection_of_A_and_B_l3012_301252


namespace correct_remaining_time_l3012_301282

/-- Represents a food item with its cooking times -/
structure FoodItem where
  name : String
  recommendedTime : Nat
  actualTime : Nat

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingTimeInSeconds (food : FoodItem) : Nat :=
  (food.recommendedTime - food.actualTime) * 60

/-- The main theorem to prove -/
theorem correct_remaining_time (frenchFries chickenNuggets mozzarellaSticks : FoodItem)
  (h1 : frenchFries.name = "French Fries" ∧ frenchFries.recommendedTime = 12 ∧ frenchFries.actualTime = 2)
  (h2 : chickenNuggets.name = "Chicken Nuggets" ∧ chickenNuggets.recommendedTime = 18 ∧ chickenNuggets.actualTime = 5)
  (h3 : mozzarellaSticks.name = "Mozzarella Sticks" ∧ mozzarellaSticks.recommendedTime = 8 ∧ mozzarellaSticks.actualTime = 3) :
  remainingTimeInSeconds frenchFries = 600 ∧
  remainingTimeInSeconds chickenNuggets = 780 ∧
  remainingTimeInSeconds mozzarellaSticks = 300 := by
  sorry


end correct_remaining_time_l3012_301282


namespace copies_equal_totient_l3012_301223

/-- The pattern function that generates the next row -/
def nextRow (row : List (List Nat)) : List (List Nat) := sorry

/-- The number of copies of n in row n of the pattern -/
def copiesInRow (n : Nat) : Nat := sorry

/-- Euler's Totient function -/
def φ (n : Nat) : Nat := sorry

/-- Theorem stating that the number of copies of 2019 in row 2019 is equal to φ(2019) -/
theorem copies_equal_totient :
  copiesInRow 2019 = φ 2019 := by sorry

end copies_equal_totient_l3012_301223


namespace parabola_a_values_l3012_301297

/-- The parabola equation y = ax^2 -/
def parabola (a : ℝ) (x y : ℝ) : Prop := y = a * x^2

/-- The point M with coordinates (2, 1) -/
def point_M : ℝ × ℝ := (2, 1)

/-- The distance from point M to the directrix is 2 -/
def distance_to_directrix : ℝ := 2

/-- The possible values of a -/
def possible_a_values : Set ℝ := {1/4, -1/12}

/-- Theorem stating the possible values of a for the given conditions -/
theorem parabola_a_values :
  ∀ a : ℝ,
  (∃ y : ℝ, parabola a (point_M.1) y) →
  (∃ d : ℝ, d = distance_to_directrix ∧ 
    ((a > 0 ∧ d = point_M.2 + 1/(4*a)) ∨
     (a < 0 ∧ d = -1/(4*a) - point_M.2))) →
  a ∈ possible_a_values :=
sorry

end parabola_a_values_l3012_301297


namespace platform_length_l3012_301214

/-- Given a train crossing a platform and a signal pole, calculate the platform length -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_platform = 54) 
  (h3 : time_pole = 18) : 
  ∃ platform_length : ℝ, platform_length = 600 := by
  sorry

end platform_length_l3012_301214


namespace quadratic_congruence_solution_unique_solution_modulo_l3012_301227

theorem quadratic_congruence_solution :
  ∃ (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD 7] ∧ x ≡ 4 [ZMOD 7] := by sorry

theorem unique_solution_modulo :
  ∀ (n : ℕ), n ≥ 2 →
    (∃! (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD n]) ↔ n = 7 := by sorry

end quadratic_congruence_solution_unique_solution_modulo_l3012_301227


namespace inequality_condition_l3012_301205

theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) := by
sorry

end inequality_condition_l3012_301205


namespace equation_solution_l3012_301296

theorem equation_solution :
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -5 / 3 :=
by sorry

end equation_solution_l3012_301296


namespace complex_equation_solution_l3012_301281

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end complex_equation_solution_l3012_301281


namespace cost_of_500_pencils_l3012_301264

/-- The cost of n pencils in dollars, given the price of one pencil in cents and the number of cents in a dollar. -/
def cost_of_pencils (n : ℕ) (price_per_pencil : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (n * price_per_pencil : ℚ) / cents_per_dollar

/-- Theorem stating that the cost of 500 pencils is 10 dollars, given the specified conditions. -/
theorem cost_of_500_pencils : 
  cost_of_pencils 500 2 100 = 10 := by
  sorry

end cost_of_500_pencils_l3012_301264


namespace five_objects_three_categories_l3012_301253

/-- The number of ways to distribute n distinguishable objects into k distinct categories -/
def distributionWays (n k : ℕ) : ℕ := k ^ n

/-- Theorem: There are 243 ways to distribute 5 distinguishable objects into 3 distinct categories -/
theorem five_objects_three_categories : distributionWays 5 3 = 243 := by
  sorry

end five_objects_three_categories_l3012_301253


namespace sum_of_four_primes_divisible_by_60_l3012_301210

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  ∃ k : ℕ, p + q + r + s = 60 * (2 * k + 1) :=
sorry

end sum_of_four_primes_divisible_by_60_l3012_301210


namespace rectangular_solid_volume_l3012_301248

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end rectangular_solid_volume_l3012_301248


namespace nabla_example_l3012_301286

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end nabla_example_l3012_301286


namespace factorization_of_four_minus_n_squared_l3012_301218

theorem factorization_of_four_minus_n_squared (n : ℝ) : 4 - n^2 = (2 + n) * (2 - n) := by
  sorry

end factorization_of_four_minus_n_squared_l3012_301218


namespace arithmetic_sequence_sum_l3012_301235

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 10 = 3) →
  (a 3 * a 10 = -5) →
  a 5 + a 8 = 3 := by
sorry

end arithmetic_sequence_sum_l3012_301235


namespace sum_of_common_elements_l3012_301239

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_seq (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def geometric_seq (k : ℕ) : ℕ := 10 * 2^k

/-- Sequence of common elements in both progressions -/
def common_seq (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum common_seq = 3495250 := by sorry

end sum_of_common_elements_l3012_301239


namespace sum_of_variables_l3012_301259

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : a + 2*b + 3*c = 13) 
  (eq2 : 4*a + 3*b + 2*c = 17) : 
  a + b + c = 6 := by
sorry

end sum_of_variables_l3012_301259


namespace red_light_probability_is_two_fifths_l3012_301234

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle duration of a traffic light -/
def cycleDuration (d : TrafficLightDurations) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : TrafficLightDurations) : ℚ :=
  d.red / (cycleDuration d)

/-- Theorem stating that the probability of seeing a red light is 2/5 -/
theorem red_light_probability_is_two_fifths (d : TrafficLightDurations)
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) :
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end red_light_probability_is_two_fifths_l3012_301234


namespace alcohol_percentage_first_vessel_l3012_301242

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_alcohol_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel2_capacity = 5)
  (h3 : vessel2_alcohol_percentage = 40)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_alcohol_percentage = 27.5) :
  ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 25 ∧
    (vessel1_alcohol_percentage / 100) * vessel1_capacity +
    (vessel2_alcohol_percentage / 100) * vessel2_capacity =
    (final_alcohol_percentage / 100) * final_vessel_capacity :=
by sorry

end alcohol_percentage_first_vessel_l3012_301242


namespace hammingDistance_bounds_hammingDistance_triangle_inequality_l3012_301220

/-- A byte is a list of booleans representing binary digits. -/
def Byte := List Bool

/-- The Hamming distance between two bytes is the number of positions at which they differ. -/
def hammingDistance (u v : Byte) : Nat :=
  (u.zip v).filter (fun (a, b) => a ≠ b) |>.length

/-- Theorem stating that the Hamming distance between two bytes is bounded by 0 and the length of the bytes. -/
theorem hammingDistance_bounds (u v : Byte) (h : u.length = v.length) :
    0 ≤ hammingDistance u v ∧ hammingDistance u v ≤ u.length := by
  sorry

/-- Theorem stating the triangle inequality for Hamming distance. -/
theorem hammingDistance_triangle_inequality (u v w : Byte) 
    (hu : u.length = v.length) (hv : v.length = w.length) :
    hammingDistance u v ≤ hammingDistance w u + hammingDistance w v := by
  sorry

end hammingDistance_bounds_hammingDistance_triangle_inequality_l3012_301220


namespace marcia_wardrobe_cost_l3012_301246

/-- Calculates the total cost of Marcia's wardrobe --/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) 
                 (numSkirts numBlouses numPants : ℕ) : ℚ :=
  let skirtCost := skirtPrice * numSkirts
  let blouseCost := blousePrice * numBlouses
  let pantCost := pantPrice * (numPants - 1) + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 --/
theorem marcia_wardrobe_cost :
  wardrobeCost 20 15 30 3 5 2 = 180 := by
  sorry

end marcia_wardrobe_cost_l3012_301246


namespace arithmetic_sequence_common_difference_l3012_301228

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_a5 : a 5 = 12) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3012_301228


namespace add_1723_minutes_to_midnight_l3012_301289

-- Define a custom datatype for date and time
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

-- Define the starting date and time
def startDateTime : DateTime :=
  { year := 2023, month := 5, day := 5, hour := 0, minute := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 1723

-- Theorem to prove
theorem add_1723_minutes_to_midnight :
  addMinutes startDateTime minutesToAdd =
    { year := 2023, month := 5, day := 6, hour := 4, minute := 43 } :=
  sorry

end add_1723_minutes_to_midnight_l3012_301289


namespace total_weight_is_8040_l3012_301212

/-- Represents the catering setup for an event -/
structure CateringSetup where
  numTables : Nat
  settingsPerTable : Nat
  backupPercentage : Rat
  forkWeight : Rat
  knifeWeight : Rat
  spoonWeight : Rat
  largePlateWeight : Rat
  smallPlateWeight : Rat
  wineGlassWeight : Rat
  waterGlassWeight : Rat
  tableDecorationWeight : Rat

/-- Calculates the total weight of all items for the catering setup -/
def totalWeight (setup : CateringSetup) : Rat :=
  let totalSettings := setup.numTables * setup.settingsPerTable * (1 + setup.backupPercentage)
  let silverwareWeight := totalSettings * (setup.forkWeight + setup.knifeWeight + setup.spoonWeight)
  let plateWeight := totalSettings * (setup.largePlateWeight + setup.smallPlateWeight)
  let glassWeight := totalSettings * (setup.wineGlassWeight + setup.waterGlassWeight)
  let decorationWeight := setup.numTables * setup.tableDecorationWeight
  silverwareWeight + plateWeight + glassWeight + decorationWeight

/-- Theorem stating that the total weight for the given setup is 8040 ounces -/
theorem total_weight_is_8040 (setup : CateringSetup) 
    (h1 : setup.numTables = 15)
    (h2 : setup.settingsPerTable = 8)
    (h3 : setup.backupPercentage = 1/4)
    (h4 : setup.forkWeight = 7/2)
    (h5 : setup.knifeWeight = 4)
    (h6 : setup.spoonWeight = 9/2)
    (h7 : setup.largePlateWeight = 14)
    (h8 : setup.smallPlateWeight = 10)
    (h9 : setup.wineGlassWeight = 7)
    (h10 : setup.waterGlassWeight = 9)
    (h11 : setup.tableDecorationWeight = 16) :
    totalWeight setup = 8040 := by
  sorry


end total_weight_is_8040_l3012_301212


namespace valid_arrangement_exists_l3012_301290

/-- Represents the arrangement of numbers in the square with a center circle -/
structure Arrangement :=
  (top_left : ℕ)
  (top_right : ℕ)
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (center : ℕ)

/-- The set of numbers to be arranged -/
def numbers : Finset ℕ := {2, 4, 6, 8, 10}

/-- Checks if the given arrangement satisfies the diagonal and vertex sum condition -/
def is_valid_arrangement (a : Arrangement) : Prop :=
  a.top_left + a.center + a.bottom_right = 
  a.top_right + a.center + a.bottom_left ∧
  a.top_left + a.center + a.bottom_right = 
  a.top_left + a.top_right + a.bottom_left + a.bottom_right

/-- Checks if the given arrangement uses all the required numbers -/
def uses_all_numbers (a : Arrangement) : Prop :=
  {a.top_left, a.top_right, a.bottom_left, a.bottom_right, a.center} = numbers

/-- Theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : 
  ∃ (a : Arrangement), is_valid_arrangement a ∧ uses_all_numbers a :=
sorry

end valid_arrangement_exists_l3012_301290


namespace vasya_reads_entire_book_l3012_301211

theorem vasya_reads_entire_book :
  let first_day : ℚ := 1/2
  let second_day : ℚ := (1/3) * (1 - first_day)
  let first_two_days : ℚ := first_day + second_day
  let third_day : ℚ := (1/2) * first_two_days
  first_day + second_day + third_day = 1 := by sorry

end vasya_reads_entire_book_l3012_301211


namespace spinsters_to_cats_ratio_l3012_301244

theorem spinsters_to_cats_ratio 
  (spinsters : ℕ) 
  (cats : ℕ) 
  (x : ℚ)
  (ratio_condition : spinsters / cats = x / 9)
  (difference_condition : cats = spinsters + 63)
  (spinsters_count : spinsters = 18) :
  spinsters / cats = 2 / 9 := by
sorry

end spinsters_to_cats_ratio_l3012_301244


namespace two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l3012_301295

-- Define the number of female and male students
def num_female : ℕ := 5
def num_male : ℕ := 4

-- Define the number of students to be selected
def num_selected : ℕ := 4

-- Theorem for scenario 1
theorem two_male_two_female_selection_methods : 
  (num_male.choose 2 * num_female.choose 2) * num_selected.factorial = 1440 := by sorry

-- Theorem for scenario 2
theorem at_least_one_male_one_female_selection_methods :
  (num_male.choose 1 * num_female.choose 3 + 
   num_male.choose 2 * num_female.choose 2 + 
   num_male.choose 3 * num_female.choose 1) * num_selected.factorial = 2880 := by sorry

end two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l3012_301295


namespace simplified_ratio_l3012_301249

def initial_money : ℕ := 91
def spent_money : ℕ := 21

def money_left : ℕ := initial_money - spent_money

def ratio_numerator : ℕ := money_left
def ratio_denominator : ℕ := spent_money

theorem simplified_ratio :
  (ratio_numerator / (Nat.gcd ratio_numerator ratio_denominator)) = 10 ∧
  (ratio_denominator / (Nat.gcd ratio_numerator ratio_denominator)) = 3 :=
by sorry

end simplified_ratio_l3012_301249


namespace phone_number_probability_l3012_301277

theorem phone_number_probability :
  let first_three_options : ℕ := 2
  let last_four_arrangements : ℕ := 24
  let total_numbers : ℕ := first_three_options * last_four_arrangements
  let correct_numbers : ℕ := 1
  (correct_numbers : ℚ) / total_numbers = 1 / 48 := by
sorry

end phone_number_probability_l3012_301277


namespace two_numbers_difference_l3012_301274

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 → 
  3 * y - 2 * x = 10 → 
  |x - y| = 4 := by
sorry

end two_numbers_difference_l3012_301274


namespace original_price_proof_l3012_301209

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculates the original price before discounts -/
def original_price (final_price : ℚ) : ℚ :=
  final_price / (1 - discount_rate)

/-- The final price after discounts -/
def final_price : ℚ := 230

theorem original_price_proof :
  ∃ (price : ℕ), price ≥ 256 ∧ price < 257 ∧ 
  (original_price final_price).num / (original_price final_price).den = price / 1 := by
  sorry

#eval (original_price final_price).num / (original_price final_price).den

end original_price_proof_l3012_301209


namespace eight_stairs_climbs_l3012_301288

def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

theorem eight_stairs_climbs : climbStairs 8 = 81 := by
  sorry

end eight_stairs_climbs_l3012_301288


namespace prob_not_red_is_six_sevenths_l3012_301229

/-- The number of red jelly beans in the bag -/
def red_beans : ℕ := 4

/-- The number of green jelly beans in the bag -/
def green_beans : ℕ := 7

/-- The number of yellow jelly beans in the bag -/
def yellow_beans : ℕ := 5

/-- The number of blue jelly beans in the bag -/
def blue_beans : ℕ := 9

/-- The number of purple jelly beans in the bag -/
def purple_beans : ℕ := 3

/-- The total number of jelly beans in the bag -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The probability of selecting a jelly bean that is not red -/
def prob_not_red : ℚ := (green_beans + yellow_beans + blue_beans + purple_beans : ℚ) / total_beans

theorem prob_not_red_is_six_sevenths : prob_not_red = 6 / 7 := by
  sorry

end prob_not_red_is_six_sevenths_l3012_301229


namespace collinear_vectors_magnitude_l3012_301216

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-2, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

theorem collinear_vectors_magnitude (k : ℝ) :
  collinear a (b k) →
  ‖(3 • a) + (b k)‖ = Real.sqrt 5 := by
  sorry

end collinear_vectors_magnitude_l3012_301216


namespace complex_number_difference_l3012_301260

theorem complex_number_difference : 
  let z : ℂ := (Complex.I * (-6 + Complex.I)) / Complex.abs (3 - 4 * Complex.I)
  (z.re - z.im) = 1 := by sorry

end complex_number_difference_l3012_301260


namespace fifteenth_term_of_inverse_proportional_sequence_l3012_301236

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℚ) : Prop :=
  ∃ k : ℚ, k ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem fifteenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℚ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 15 = 3 := by
  sorry

end fifteenth_term_of_inverse_proportional_sequence_l3012_301236


namespace matrix_N_satisfies_conditions_l3012_301208

def N : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 8; 4, 6, -2; -9, -3, 5]

def i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
def j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
def k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]

theorem matrix_N_satisfies_conditions :
  N * i = !![3; 4; -9] ∧
  N * j = !![1; 6; -3] ∧
  N * k = !![8; -2; 5] := by
  sorry

end matrix_N_satisfies_conditions_l3012_301208


namespace solution_to_equation_l3012_301293

theorem solution_to_equation : ∃ y : ℝ, (7 - y = 10) ∧ (y = -3) := by sorry

end solution_to_equation_l3012_301293


namespace rectangle_composition_l3012_301247

/-- Given a rectangle ABCD composed of six identical smaller rectangles,
    prove that the length y is 20 -/
theorem rectangle_composition (x y : ℝ) : 
  (3 * y) * (2 * x) = 2400 →  -- Area of ABCD
  y = 20 := by
  sorry

end rectangle_composition_l3012_301247


namespace smallest_steps_l3012_301213

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 2 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 2 → n ≤ m) → 
  n = 58 := by
sorry

end smallest_steps_l3012_301213


namespace bill_more_sticks_than_ted_l3012_301255

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- Calculates the total number of objects thrown -/
def ThrowCount.total (t : ThrowCount) : ℕ := t.sticks + t.rocks

theorem bill_more_sticks_than_ted (bill : ThrowCount) (ted : ThrowCount) : 
  bill.total = 21 → 
  ted.rocks = 2 * bill.rocks → 
  ted.sticks = 10 → 
  ted.rocks = 10 → 
  bill.sticks - ted.sticks = 6 := by
sorry

end bill_more_sticks_than_ted_l3012_301255


namespace adjacent_sides_equal_not_implies_parallelogram_l3012_301257

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Definition of a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1 = q.D.1 - q.C.1 ∧ q.A.2 - q.B.2 = q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1 = q.B.1 - q.C.1 ∧ q.A.2 - q.D.2 = q.B.2 - q.C.2)

/-- Definition of equality of two sides -/
def sides_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2

/-- Theorem: Adjacent sides being equal does not imply parallelogram -/
theorem adjacent_sides_equal_not_implies_parallelogram :
  ¬∀ (q : Quadrilateral), 
    (sides_equal q.A q.B q.A q.D ∧ sides_equal q.B q.C q.C q.D) → 
    is_parallelogram q :=
sorry

end adjacent_sides_equal_not_implies_parallelogram_l3012_301257


namespace part1_solution_set_part2_a_range_l3012_301240

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part 2
theorem part2_a_range :
  ∀ a : ℝ, (∀ y : ℝ, -1 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, g a x = y) →
  (a ≤ 1 ∨ a ≥ 5) := by sorry

end part1_solution_set_part2_a_range_l3012_301240


namespace perpendicular_vectors_x_value_l3012_301225

-- Define the vectors
def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- The main theorem
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, perpendicular (a + b x) a ∧ x = 16 :=
sorry

end perpendicular_vectors_x_value_l3012_301225


namespace constant_k_value_l3012_301261

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end constant_k_value_l3012_301261


namespace total_cookies_l3012_301222

/-- Given 26 bags of cookies with 2 cookies in each bag, prove that the total number of cookies is 52. -/
theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : num_bags = 26) 
  (h2 : cookies_per_bag = 2) : 
  num_bags * cookies_per_bag = 52 := by
  sorry

end total_cookies_l3012_301222


namespace symmetric_points_line_equation_l3012_301230

/-- Given two points P and Q that are symmetric about a line l, 
    prove that the equation of line l is x - y + 1 = 0 -/
theorem symmetric_points_line_equation 
  (a b : ℝ) 
  (h : a ≠ b - 1) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 1 = 0) →
    (∀ (R : ℝ × ℝ), R ∈ l → (dist P R = dist Q R)) →
    true :=
by sorry


end symmetric_points_line_equation_l3012_301230


namespace right_triangle_hypotenuse_median_relation_l3012_301245

/-- In a right triangle, the square of the hypotenuse is equal to four-fifths of the sum of squares of the medians to the other two sides. -/
theorem right_triangle_hypotenuse_median_relation (a b c k_a k_b : ℝ) :
  a > 0 → b > 0 → c > 0 → k_a > 0 → k_b > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  k_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 →  -- Definition of k_a
  k_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Definition of k_b
  c^2 = (4/5) * (k_a^2 + k_b^2) := by
sorry

end right_triangle_hypotenuse_median_relation_l3012_301245


namespace mans_age_to_sons_age_ratio_l3012_301219

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 46 years older than his son and the son's current age is 44. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 44 →
    man_age = son_age + 46 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end mans_age_to_sons_age_ratio_l3012_301219


namespace drug_price_reduction_l3012_301279

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 60)
  (h2 : final_price = 48.6)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) : 
  x = 0.1 := by
sorry

end drug_price_reduction_l3012_301279


namespace f_properties_f_inv_property_l3012_301243

/-- A function f(x) that is directly proportional to x-3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k

/-- The theorem stating the properties of f -/
theorem f_properties (k : ℝ) :
  f k 4 = 3 →
  (∀ x, f k x = 3 * x - 9) ∧
  (∃ x, f k x = -12 ∧ x = -1) := by
  sorry

/-- The inverse function of f -/
noncomputable def f_inv (k : ℝ) (y : ℝ) : ℝ := (y + 3 * k) / k

/-- Theorem stating that f_inv(-12) = -1 when f(4) = 3 -/
theorem f_inv_property (k : ℝ) :
  f k 4 = 3 →
  f_inv k (-12) = -1 := by
  sorry

end f_properties_f_inv_property_l3012_301243


namespace quadratic_roots_negative_real_part_l3012_301291

theorem quadratic_roots_negative_real_part (p q : ℝ) :
  (∃ x : ℂ, p * x^2 + (p^2 - q) * x - (2*p - q - 1) = 0 ∧ x.re < 0) ↔
  (p = 0 ∧ -1 < q ∧ q < 0) ∨ (p > 0 ∧ q < p^2 ∧ q > 2*p - 1) :=
by sorry

end quadratic_roots_negative_real_part_l3012_301291


namespace sum_of_integers_l3012_301298

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  (Prime a ∨ Prime b ∨ Prime c ∨ Prime d ∨ Prime e) →
  a + b + c + d + e = 34 := by
sorry

end sum_of_integers_l3012_301298


namespace area_is_twenty_l3012_301232

/-- The equation of the graph --/
def graph_equation (x y : ℝ) : Prop := abs (5 * x) + abs (2 * y) = 10

/-- The set of points satisfying the graph equation --/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph --/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the area enclosed by the graph is 20 --/
theorem area_is_twenty : enclosed_area = 20 := by sorry

end area_is_twenty_l3012_301232


namespace pumpkin_count_l3012_301204

/-- The total number of pumpkins grown by Sandy, Mike, Maria, and Sam -/
def total_pumpkins (sandy mike maria sam : ℕ) : ℕ := sandy + mike + maria + sam

/-- Theorem stating that the total number of pumpkins is 157 -/
theorem pumpkin_count : total_pumpkins 51 23 37 46 = 157 := by
  sorry

end pumpkin_count_l3012_301204


namespace focus_of_our_parabola_l3012_301202

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- A parabola defined by its equation -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola given its equation -/
def focus_of_parabola (p : Parabola) : Focus := sorry

/-- The parabola x^2 + y = 0 -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 + y = 0 }

theorem focus_of_our_parabola :
  focus_of_parabola our_parabola = ⟨0, -1/4⟩ := by sorry

end focus_of_our_parabola_l3012_301202


namespace money_division_l3012_301256

theorem money_division (p q r : ℕ) (total : ℝ) (h1 : p + q + r = 22) (h2 : 12 * total / 22 - 7 * total / 22 = 5000) :
  7 * total / 22 - 3 * total / 22 = 4000 := by
  sorry

end money_division_l3012_301256


namespace total_cartridge_cost_l3012_301270

def black_and_white_cost : ℕ := 27
def color_cost : ℕ := 32
def num_black_and_white : ℕ := 1
def num_color : ℕ := 3

theorem total_cartridge_cost :
  num_black_and_white * black_and_white_cost + num_color * color_cost = 123 :=
by sorry

end total_cartridge_cost_l3012_301270


namespace fifth_term_is_x_l3012_301269

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define our specific sequence
def our_sequence (x y : ℝ) : ℕ → ℝ
| 0 => x + 2*y
| 1 => x - 2*y
| 2 => x + y
| 3 => x - y
| n + 4 => our_sequence x y 3 + (n + 1) * (our_sequence x y 1 - our_sequence x y 0)

theorem fifth_term_is_x (x y : ℝ) :
  is_arithmetic_sequence (our_sequence x y) →
  our_sequence x y 4 = x :=
by
  sorry

end fifth_term_is_x_l3012_301269


namespace total_birds_is_148_l3012_301241

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end total_birds_is_148_l3012_301241


namespace complex_fraction_evaluation_l3012_301206

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 1/32 := by
  sorry

end complex_fraction_evaluation_l3012_301206


namespace tax_rate_is_ten_percent_l3012_301287

/-- The tax rate for properties in Township K -/
def tax_rate : ℝ := sorry

/-- The initial assessed value of the property -/
def initial_value : ℝ := 20000

/-- The final assessed value of the property -/
def final_value : ℝ := 28000

/-- The increase in property tax -/
def tax_increase : ℝ := 800

/-- Theorem stating that the tax rate is 10% of the assessed value -/
theorem tax_rate_is_ten_percent :
  tax_rate = 0.1 :=
by
  sorry

#check tax_rate_is_ten_percent

end tax_rate_is_ten_percent_l3012_301287


namespace special_triangle_sum_range_l3012_301250

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Ensure angles are positive and sum to π
  angle_sum : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  -- Ensure sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 + t.a * t.b = 4 ∧ t.c = 2

-- State the theorem
theorem special_triangle_sum_range (t : Triangle) (h : SpecialTriangle t) :
  2 < 2 * t.a + t.b ∧ 2 * t.a + t.b < 4 := by
  sorry

end special_triangle_sum_range_l3012_301250


namespace arithmetic_square_root_of_four_l3012_301200

theorem arithmetic_square_root_of_four :
  ∃ (x : ℝ), x > 0 ∧ x * x = 4 ∧ ∀ y : ℝ, y > 0 ∧ y * y = 4 → y = x :=
by sorry

end arithmetic_square_root_of_four_l3012_301200


namespace maximum_marks_l3012_301262

theorem maximum_marks (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  obtained_marks = 184 → 
  percentage * max_marks = obtained_marks → 
  max_marks = 200 := by
sorry

end maximum_marks_l3012_301262


namespace men_in_first_group_l3012_301285

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 6

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 12

/-- The theorem stating that the number of men in the first group is 10 -/
theorem men_in_first_group : 
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group * hours_per_day = 
    men_second_group * days_second_group * hours_per_day ∧
    men_first_group = 10 :=
by
  sorry


end men_in_first_group_l3012_301285


namespace trig_identity_l3012_301238

theorem trig_identity : 
  Real.cos (54 * π / 180) * Real.cos (24 * π / 180) + 
  2 * Real.sin (12 * π / 180) * Real.cos (12 * π / 180) * Real.sin (126 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end trig_identity_l3012_301238


namespace savings_proof_l3012_301299

def original_savings (furniture_fraction : ℚ) (tv_cost : ℕ) : ℕ :=
  4 * tv_cost

theorem savings_proof (furniture_fraction : ℚ) (tv_cost : ℕ) 
  (h1 : furniture_fraction = 3/4) 
  (h2 : tv_cost = 210) : 
  original_savings furniture_fraction tv_cost = 840 := by
  sorry

end savings_proof_l3012_301299


namespace max_sets_production_l3012_301294

/-- Represents the number of sets produced given the number of workers assigned to bolts and nuts -/
def sets_produced (bolt_workers : ℕ) (nut_workers : ℕ) : ℕ :=
  min (25 * bolt_workers) ((20 * nut_workers) / 2)

/-- Theorem stating that 40 bolt workers and 100 nut workers maximize set production -/
theorem max_sets_production :
  ∀ (b n : ℕ),
    b + n = 140 →
    sets_produced b n ≤ sets_produced 40 100 :=
by sorry

end max_sets_production_l3012_301294


namespace calculate_expression_l3012_301226

theorem calculate_expression : -5^2 - (-3)^3 * (2/9) - 9 * |-(2/3)| = -25 := by
  sorry

end calculate_expression_l3012_301226


namespace angle_between_3_and_7_l3012_301224

/-- Represents a clock with equally spaced rays -/
structure Clock :=
  (num_rays : ℕ)
  (ray_spacing : ℝ)
  (h_positive_rays : 0 < num_rays)
  (h_spacing : ray_spacing = 360 / num_rays)

/-- Calculates the angle between two hour positions on a clock -/
def angle_between_hours (clock : Clock) (hour1 hour2 : ℕ) : ℝ :=
  let diff := (hour2 - hour1 + clock.num_rays) % clock.num_rays
  clock.ray_spacing * min diff (clock.num_rays - diff)

/-- Theorem: The smaller angle between 3 o'clock and 7 o'clock on a 12-hour clock is 120 degrees -/
theorem angle_between_3_and_7 :
  ∀ (c : Clock), c.num_rays = 12 → angle_between_hours c 3 7 = 120 :=
by sorry

end angle_between_3_and_7_l3012_301224


namespace disk_arrangement_area_l3012_301280

theorem disk_arrangement_area :
  ∀ (r : ℝ),
  r > 0 →
  r = 2 - Real.sqrt 3 →
  (12 : ℝ) * π * r^2 = π * (84 - 48 * Real.sqrt 3) := by
  sorry

end disk_arrangement_area_l3012_301280


namespace expression_simplification_and_evaluation_l3012_301258

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 5 + 1
  let y : ℝ := Real.sqrt 5 - 1
  ((5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (1 / (x^2 * y - x * y^2)) = 12 :=
by sorry

end expression_simplification_and_evaluation_l3012_301258


namespace p_satisfies_conditions_l3012_301251

/-- The quadratic polynomial p(x) that satisfies given conditions -/
def p (x : ℚ) : ℚ := (12/5) * x^2 - (36/5) * x - 216/5

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions : 
  p (-3) = 0 ∧ p 6 = 0 ∧ p 2 = -48 := by
  sorry

end p_satisfies_conditions_l3012_301251


namespace initial_amount_proof_l3012_301266

/-- Proves that Rs 100 at 5% interest for 48 years produces the same interest as Rs 600 at 10% interest for 4 years -/
theorem initial_amount_proof (amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (time2 : ℝ) : 
  amount = 100 ∧ rate1 = 0.05 ∧ rate2 = 0.10 ∧ time1 = 48 ∧ time2 = 4 →
  amount * rate1 * time1 = 600 * rate2 * time2 :=
by
  sorry

#check initial_amount_proof

end initial_amount_proof_l3012_301266


namespace lcm_of_20_45_75_l3012_301278

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_of_20_45_75_l3012_301278


namespace ratio_of_angles_l3012_301271

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def O : Point := sorry

-- Define the vertices of the triangle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point E
def E : Point := sorry

-- Define the inscribed triangle
def triangle_ABC : Triangle := sorry

-- Define the arcs
def arc_AB : ℝ := 100
def arc_BC : ℝ := 80

-- Define the perpendicular condition
def OE_perp_AC : Prop := sorry

-- Define the angles
def angle_OBE : ℝ := sorry
def angle_BAC : ℝ := sorry

-- State the theorem
theorem ratio_of_angles (circle : Circle) (triangle_ABC : Triangle) 
  (h1 : arc_AB = 100)
  (h2 : arc_BC = 80)
  (h3 : OE_perp_AC) :
  angle_OBE / angle_BAC = 2 / 5 := by sorry

end ratio_of_angles_l3012_301271


namespace arcsin_equation_solution_l3012_301221

theorem arcsin_equation_solution (x : ℝ) :
  (Real.arcsin x + Real.arcsin (3 * x) = π / 4) →
  (x = Real.sqrt (1 / (9 + 4 * Real.sqrt 2)) ∨
   x = -Real.sqrt (1 / (9 + 4 * Real.sqrt 2))) ∧
  (x ≥ -1 ∧ x ≤ 1) ∧ (3 * x ≥ -1 ∧ 3 * x ≤ 1) :=
by sorry

end arcsin_equation_solution_l3012_301221


namespace jovana_shells_l3012_301292

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Jovana added 12 pounds of shells to her bucket -/
theorem jovana_shells : shells_added 5 17 = 12 := by
  sorry

end jovana_shells_l3012_301292


namespace smallest_stair_count_l3012_301233

theorem smallest_stair_count (n : ℕ) : 
  (n > 20 ∧ n % 3 = 1 ∧ n % 5 = 4) → 
  (∀ m : ℕ, m > 20 ∧ m % 3 = 1 ∧ m % 5 = 4 → m ≥ n) → 
  n = 34 := by
sorry

end smallest_stair_count_l3012_301233


namespace no_infinite_sequence_exists_l3012_301275

theorem no_infinite_sequence_exists : ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, 
  (a (n + 2) : ℝ) = (a (n + 1) : ℝ) + Real.sqrt ((a (n + 1) : ℝ) + (a n : ℝ)) :=
by sorry

end no_infinite_sequence_exists_l3012_301275


namespace no_such_function_l3012_301215

theorem no_such_function : ¬∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end no_such_function_l3012_301215


namespace ladies_walk_l3012_301272

/-- The combined distance walked by two ladies in Central Park -/
theorem ladies_walk (distance_lady2 : ℝ) (h1 : distance_lady2 = 4) :
  let distance_lady1 : ℝ := 2 * distance_lady2
  distance_lady1 + distance_lady2 = 12 := by
sorry

end ladies_walk_l3012_301272


namespace jan_beth_money_difference_l3012_301207

theorem jan_beth_money_difference (beth_money jan_money : ℕ) : 
  beth_money + 35 = 105 →
  beth_money + jan_money = 150 →
  jan_money - beth_money = 10 := by
sorry

end jan_beth_money_difference_l3012_301207


namespace integer_roots_of_cubic_l3012_301284

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 5*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-71, -42, -24, -14, 4, 14, 22, 41} : Set ℤ) := by
  sorry

end integer_roots_of_cubic_l3012_301284


namespace order_cost_is_43_l3012_301201

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of fries in dollars -/
def fries_cost : ℕ := 2

/-- The number of sandwiches ordered -/
def num_sandwiches : ℕ := 3

/-- The number of sodas ordered -/
def num_sodas : ℕ := 7

/-- The number of fries ordered -/
def num_fries : ℕ := 5

/-- The total cost of the order -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas + fries_cost * num_fries

theorem order_cost_is_43 : total_cost = 43 := by
  sorry

end order_cost_is_43_l3012_301201


namespace complex_power_sum_l3012_301203

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1500 + 1/(z^1500) = 1 := by
  sorry

end complex_power_sum_l3012_301203


namespace tangent_implies_positive_derivative_l3012_301237

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (2,3) passes through (-1,2)
def tangent_condition (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), k * (-1 - 2) + 3 = 2 ∧ HasDerivAt f k 2

-- State the theorem
theorem tangent_implies_positive_derivative (f : ℝ → ℝ) 
  (h : tangent_condition f) : 
  ∃ (d : ℝ), HasDerivAt f d 2 ∧ d > 0 := by
  sorry

end tangent_implies_positive_derivative_l3012_301237


namespace weight_loss_per_month_l3012_301283

def initial_weight : ℝ := 250
def final_weight : ℝ := 154
def months_in_year : ℕ := 12

theorem weight_loss_per_month :
  (initial_weight - final_weight) / months_in_year = 8 := by
  sorry

end weight_loss_per_month_l3012_301283


namespace jerry_firecracker_fraction_l3012_301267

/-- Given:
  * Jerry bought 48 firecrackers initially
  * 12 firecrackers were confiscated
  * 1/6 of the remaining firecrackers were defective
  * Jerry set off 15 good firecrackers
Prove that Jerry set off 1/2 of the good firecrackers -/
theorem jerry_firecracker_fraction :
  let initial_firecrackers : ℕ := 48
  let confiscated_firecrackers : ℕ := 12
  let defective_fraction : ℚ := 1/6
  let set_off_firecrackers : ℕ := 15
  let remaining_firecrackers := initial_firecrackers - confiscated_firecrackers
  let good_firecrackers := remaining_firecrackers - (defective_fraction * remaining_firecrackers).num
  (set_off_firecrackers : ℚ) / good_firecrackers = 1/2 := by
  sorry

end jerry_firecracker_fraction_l3012_301267


namespace cos_eight_arccos_one_fourth_l3012_301263

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 172546/1048576 := by
  sorry

end cos_eight_arccos_one_fourth_l3012_301263


namespace grape_difference_l3012_301268

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := 83

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := (total_grapes - robs_grapes - 4) / 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

theorem grape_difference : allies_grapes - robs_grapes = 2 := by
  sorry

end grape_difference_l3012_301268


namespace cos_alpha_value_l3012_301231

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α - Real.pi/4) = -Real.sqrt 2 / 10)
  (h2 : 0 < α) (h3 : α < Real.pi/2) : 
  Real.cos α = 4/5 := by
sorry

end cos_alpha_value_l3012_301231
