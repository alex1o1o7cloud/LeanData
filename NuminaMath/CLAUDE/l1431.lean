import Mathlib

namespace NUMINAMATH_CALUDE_savings_ratio_l1431_143171

def january_amount : ℕ := 19
def march_amount : ℕ := 8
def total_amount : ℕ := 46

def february_amount : ℕ := total_amount - january_amount - march_amount

theorem savings_ratio : 
  (january_amount : ℚ) / (february_amount : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_savings_ratio_l1431_143171


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1431_143165

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : z / (2 + a * I) = 2 / (1 + I)) 
  (h2 : z.im = -3) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1431_143165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1431_143104

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = S seq 5)
  (h2 : seq.a 2 * seq.a 4 = S seq 4) :
  (∀ n, seq.a n = 2 * n - 6) ∧
  (∀ n < 7, S seq n ≤ seq.a n) ∧
  (S seq 7 > seq.a 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1431_143104


namespace NUMINAMATH_CALUDE_millet_in_brand_b_l1431_143105

/-- Represents the composition of a bird seed brand -/
structure BirdSeed where
  millet : ℝ
  other : ℝ
  composition_sum : millet + other = 1

/-- Represents a mix of two bird seed brands -/
structure BirdSeedMix where
  brandA : BirdSeed
  brandB : BirdSeed
  proportionA : ℝ
  proportionB : ℝ
  mix_sum : proportionA + proportionB = 1

/-- Theorem stating the millet percentage in Brand B given the conditions -/
theorem millet_in_brand_b 
  (mix : BirdSeedMix)
  (brandA_millet : mix.brandA.millet = 0.6)
  (mix_proportionA : mix.proportionA = 0.6)
  (mix_millet : mix.proportionA * mix.brandA.millet + mix.proportionB * mix.brandB.millet = 0.5) :
  mix.brandB.millet = 0.35 := by
  sorry


end NUMINAMATH_CALUDE_millet_in_brand_b_l1431_143105


namespace NUMINAMATH_CALUDE_feathers_per_pound_is_300_l1431_143183

/-- Represents the number of feathers in a goose -/
def goose_feathers : ℕ := 3600

/-- Represents the number of pillows that can be stuffed with one goose's feathers -/
def pillows_per_goose : ℕ := 6

/-- Represents the number of pounds of feathers needed for each pillow -/
def pounds_per_pillow : ℕ := 2

/-- Calculates the number of feathers in a pound of goose feathers -/
def feathers_per_pound : ℕ := goose_feathers / (pillows_per_goose * pounds_per_pillow)

theorem feathers_per_pound_is_300 : feathers_per_pound = 300 := by
  sorry

end NUMINAMATH_CALUDE_feathers_per_pound_is_300_l1431_143183


namespace NUMINAMATH_CALUDE_max_value_under_constraint_l1431_143148

theorem max_value_under_constraint (x y : ℝ) :
  x^2 + y^2 ≤ 5 →
  3*|x + y| + |4*y + 9| + |7*y - 3*x - 18| ≤ 27 + 6*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l1431_143148


namespace NUMINAMATH_CALUDE_inequality_proof_l1431_143123

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + 2*y + 3*z = 12) : x^2 + 2*y^3 + 3*z^2 > 24 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1431_143123


namespace NUMINAMATH_CALUDE_martin_goldfish_purchase_l1431_143114

/-- The number of new goldfish Martin purchases every week -/
def new_goldfish_per_week : ℕ := sorry

theorem martin_goldfish_purchase :
  let initial_goldfish : ℕ := 18
  let dying_goldfish_per_week : ℕ := 5
  let weeks : ℕ := 7
  let final_goldfish : ℕ := 4
  final_goldfish = initial_goldfish + (new_goldfish_per_week - dying_goldfish_per_week) * weeks →
  new_goldfish_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_martin_goldfish_purchase_l1431_143114


namespace NUMINAMATH_CALUDE_sons_age_l1431_143135

/-- Prove that the son's current age is 24 years given the conditions -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  father_age - 8 = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1431_143135


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1431_143146

theorem unique_congruence_in_range : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 7 ∧ n ≡ 12345 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1431_143146


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1431_143191

variable (a : ℝ)

theorem negation_of_proposition (p : Prop) :
  (∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1431_143191


namespace NUMINAMATH_CALUDE_election_result_l1431_143157

/-- The total number of polled votes in an election with two candidates,
    where one candidate got 45% of votes and was defeated by 9,000 votes,
    and there were 83 invalid votes. -/
def total_polled_votes : ℕ := by sorry

/-- The number of valid votes in the election. -/
def valid_votes : ℕ := by sorry

/-- The percentage of votes received by the losing candidate. -/
def losing_candidate_percentage : ℚ := 45 / 100

/-- The number of votes by which the winning candidate defeated the losing candidate. -/
def vote_difference : ℕ := 9000

/-- The number of invalid votes in the election. -/
def invalid_votes : ℕ := 83

theorem election_result :
  total_polled_votes = 90083 ∧
  valid_votes = 90000 ∧
  (↑valid_votes : ℚ) * losing_candidate_percentage + vote_difference = valid_votes ∧
  total_polled_votes = valid_votes + invalid_votes := by sorry

end NUMINAMATH_CALUDE_election_result_l1431_143157


namespace NUMINAMATH_CALUDE_irene_age_is_46_l1431_143184

/-- Given Eddie's age, calculate Irene's age based on the relationships between Eddie, Becky, and Irene. -/
def calculate_irene_age (eddie_age : ℕ) : ℕ :=
  let becky_age := eddie_age / 4
  2 * becky_age

/-- Theorem stating that given the conditions, Irene's age is 46. -/
theorem irene_age_is_46 :
  let eddie_age : ℕ := 92
  calculate_irene_age eddie_age = 46 := by
  sorry

#eval calculate_irene_age 92

end NUMINAMATH_CALUDE_irene_age_is_46_l1431_143184


namespace NUMINAMATH_CALUDE_cone_radius_l1431_143149

theorem cone_radius (r l : ℝ) : 
  r > 0 → l > 0 →
  π * l = 2 * π * r →
  π * r^2 + π * r * l = 3 * π →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l1431_143149


namespace NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l1431_143101

theorem smallest_positive_angle_with_same_terminal_side (angle : Real) : 
  angle = -660 * Real.pi / 180 → 
  ∃ (k : ℤ), (angle + 2 * Real.pi * k) % (2 * Real.pi) = Real.pi / 3 ∧ 
  ∀ (x : Real), 0 < x ∧ x < Real.pi / 3 → 
  ¬∃ (m : ℤ), (angle + 2 * Real.pi * m) % (2 * Real.pi) = x :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_with_same_terminal_side_l1431_143101


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l1431_143193

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (init := 0) fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))

/-- The base-7 representation of the number --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem: The base-10 equivalent of 65432 in base-7 is 16340 --/
theorem base7_to_base10_65432 : base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l1431_143193


namespace NUMINAMATH_CALUDE_green_corner_plants_l1431_143185

theorem green_corner_plants (total_pots : ℕ) (green_lily_cost spider_plant_cost : ℕ) (total_budget : ℕ)
  (h1 : total_pots = 46)
  (h2 : green_lily_cost = 9)
  (h3 : spider_plant_cost = 6)
  (h4 : total_budget = 390) :
  ∃ (green_lily_pots spider_plant_pots : ℕ),
    green_lily_pots + spider_plant_pots = total_pots ∧
    green_lily_cost * green_lily_pots + spider_plant_cost * spider_plant_pots = total_budget ∧
    green_lily_pots = 38 ∧
    spider_plant_pots = 8 :=
by sorry

end NUMINAMATH_CALUDE_green_corner_plants_l1431_143185


namespace NUMINAMATH_CALUDE_circle_center_on_line_l1431_143192

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center_on_line (x y : ℚ) : 
  (5 * x - 4 * y = 40) ∧ 
  (5 * x - 4 * y = -20) ∧ 
  (3 * x - y = 0) →
  x = -10/7 ∧ y = -30/7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l1431_143192


namespace NUMINAMATH_CALUDE_binomial_coefficient_9_5_l1431_143158

theorem binomial_coefficient_9_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_9_5_l1431_143158


namespace NUMINAMATH_CALUDE_f_negative_five_halves_l1431_143107

def f (x : ℝ) : ℝ := sorry

theorem f_negative_five_halves :
  (∀ x, f (-x) = -f x) →                     -- f is odd
  (∀ x, f (x + 2) = f x) →                   -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f(x) = 2x(1-x) for 0 ≤ x ≤ 1
  f (-5/2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_negative_five_halves_l1431_143107


namespace NUMINAMATH_CALUDE_function_shift_l1431_143153

/-- Given a function f(x) = (x(x+3))/2, prove that f(x-1) = (x^2 + x - 2)/2 -/
theorem function_shift (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x * (x + 3)) / 2
  f (x - 1) = (x^2 + x - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l1431_143153


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1431_143110

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1431_143110


namespace NUMINAMATH_CALUDE_cubic_derivative_value_l1431_143113

def f (x : ℝ) := x^3

theorem cubic_derivative_value (x₀ : ℝ) :
  (deriv f) x₀ = 3 → x₀ = 1 ∨ x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_cubic_derivative_value_l1431_143113


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1431_143130

theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ 2 ∧ x ≠ -1 ∧ |x + 2| = |x| + a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1431_143130


namespace NUMINAMATH_CALUDE_missing_number_proof_l1431_143186

theorem missing_number_proof : ∃ x : ℤ, (10111 - x * 2 * 5 = 10011) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1431_143186


namespace NUMINAMATH_CALUDE_tanner_remaining_money_l1431_143151

def september_savings : ℕ := 17
def october_savings : ℕ := 48
def november_savings : ℕ := 25
def video_game_cost : ℕ := 49

theorem tanner_remaining_money :
  september_savings + october_savings + november_savings - video_game_cost = 41 := by
  sorry

end NUMINAMATH_CALUDE_tanner_remaining_money_l1431_143151


namespace NUMINAMATH_CALUDE_diagonal_passes_through_840_cubes_l1431_143137

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 360 × 450 rectangular solid passes through 840 cubes -/
theorem diagonal_passes_through_840_cubes :
  cubes_passed_through 200 360 450 = 840 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_840_cubes_l1431_143137


namespace NUMINAMATH_CALUDE_hexadecimal_to_decimal_l1431_143125

theorem hexadecimal_to_decimal (m : ℕ) : 
  1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexadecimal_to_decimal_l1431_143125


namespace NUMINAMATH_CALUDE_watches_synchronize_after_1600_days_l1431_143131

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The rate at which Glafira's watch gains time (in seconds per day) -/
def glafira_gain : ℕ := 36

/-- The rate at which Gavrila's watch loses time (in seconds per day) -/
def gavrila_loss : ℕ := 18

/-- The theorem stating that the watches will display the correct time simultaneously after 1600 days -/
theorem watches_synchronize_after_1600_days :
  (seconds_per_day * 1600) % (glafira_gain + gavrila_loss) = 0 := by
  sorry

end NUMINAMATH_CALUDE_watches_synchronize_after_1600_days_l1431_143131


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l1431_143133

theorem perfect_squares_condition (k : ℤ) : 
  (∃ n : ℤ, k + 1 = n^2) ∧ (∃ m : ℤ, 16*k + 1 = m^2) ↔ k = 0 ∨ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l1431_143133


namespace NUMINAMATH_CALUDE_f_properties_l1431_143190

noncomputable def f (x m : ℝ) : ℝ := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) + m

theorem f_properties :
  ∀ m : ℝ,
  (∃ T > 0, ∀ x : ℝ, f x m = f (x + T) m) ∧
  (T = π → ∀ T' > 0, (∀ x : ℝ, f x m = f (x + T') m) → T' ≥ T) ∧
  (∃ x₀ ∈ Set.Icc 0 (π / 2), f x₀ m = 6 →
    ∃ x₁ ∈ Set.Icc 0 (π / 2), ∀ x ∈ Set.Icc 0 (π / 2), f x m ≥ f x₁ m ∧ f x₁ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1431_143190


namespace NUMINAMATH_CALUDE_min_sum_of_product_l1431_143106

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 3960) :
  ∃ (x y z : ℕ+), x * y * z = 3960 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 150 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l1431_143106


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1431_143187

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1431_143187


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1431_143108

/- Define the imaginary unit -/
variable (i : ℂ)

/- Define real numbers m and n -/
variable (m n : ℝ)

/- State the theorem -/
theorem complex_fraction_equals_i
  (h1 : i * i = -1)
  (h2 : m * (1 + i) = 11 + n * i) :
  (m + n * i) / (m - n * i) = i :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1431_143108


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1431_143127

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The incident ray passes through these points -/
def M : Point := { x := 3, y := -2 }
def P : Point := { x := 0, y := 1 }

/-- P is on the y-axis -/
axiom P_on_y_axis : P.x = 0

/-- Function to check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

/-- The reflected ray -/
def reflected_ray : Line := { A := 1, B := -1, C := 1 }

/-- Theorem stating that the reflected ray has the equation x - y + 1 = 0 -/
theorem reflected_ray_equation :
  point_on_line P reflected_ray ∧
  point_on_line { x := -M.x, y := M.y } reflected_ray :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1431_143127


namespace NUMINAMATH_CALUDE_average_growth_rate_is_20_percent_l1431_143147

/-- Represents the monthly revenue growth rate as a real number between 0 and 1 -/
def MonthlyGrowthRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The revenue in February in millions of yuan -/
def february_revenue : ℝ := 4

/-- The revenue increase rate from February to March -/
def march_increase_rate : ℝ := 0.1

/-- The revenue in May in millions of yuan -/
def may_revenue : ℝ := 633.6

/-- The number of months between March and May -/
def months_between : ℕ := 2

/-- Calculate the average monthly growth rate from March to May -/
def calculate_growth_rate (feb_rev : ℝ) (march_inc : ℝ) (may_rev : ℝ) (months : ℕ) : MonthlyGrowthRate :=
  sorry

theorem average_growth_rate_is_20_percent :
  calculate_growth_rate february_revenue march_increase_rate may_revenue months_between = ⟨0.2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_average_growth_rate_is_20_percent_l1431_143147


namespace NUMINAMATH_CALUDE_amount_spent_on_books_l1431_143102

/-- Calculates the amount spent on books given the total allowance and percentages spent on other items --/
theorem amount_spent_on_books
  (total_allowance : ℚ)
  (games_percentage : ℚ)
  (clothes_percentage : ℚ)
  (snacks_percentage : ℚ)
  (h1 : total_allowance = 50)
  (h2 : games_percentage = 1/4)
  (h3 : clothes_percentage = 2/5)
  (h4 : snacks_percentage = 3/20) :
  total_allowance - (games_percentage + clothes_percentage + snacks_percentage) * total_allowance = 10 :=
by sorry

end NUMINAMATH_CALUDE_amount_spent_on_books_l1431_143102


namespace NUMINAMATH_CALUDE_pet_store_cages_l1431_143111

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 56)
  (h2 : sold_puppies = 24)
  (h3 : puppies_per_cage = 4) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1431_143111


namespace NUMINAMATH_CALUDE_perfect_square_proof_l1431_143109

theorem perfect_square_proof (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_odd_b : Odd b) (h_int : ∃ k : ℤ, (a + b)^2 + 4*a = k * a * b) : 
  ∃ u : ℕ, a = u^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l1431_143109


namespace NUMINAMATH_CALUDE_divisor_problem_l1431_143199

theorem divisor_problem (x d : ℝ) (h1 : x = 33) (h2 : x / d + 9 = 15) : d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1431_143199


namespace NUMINAMATH_CALUDE_shopping_with_refund_l1431_143179

/-- Calculates the remaining money after shopping with a partial refund --/
theorem shopping_with_refund 
  (initial_amount : ℕ) 
  (sweater_cost t_shirt_cost shoes_cost : ℕ) 
  (refund_percentage : ℚ) : 
  initial_amount = 74 →
  sweater_cost = 9 →
  t_shirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 / 100 →
  initial_amount - (sweater_cost + t_shirt_cost + (shoes_cost * (1 - refund_percentage))) = 51 := by
  sorry

end NUMINAMATH_CALUDE_shopping_with_refund_l1431_143179


namespace NUMINAMATH_CALUDE_product_of_special_set_l1431_143120

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) (h_odd : Odd n) (h_n_gt_1 : n > 1) 
  (h_card : M.card = n) (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + x) : 
  M.prod id = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_set_l1431_143120


namespace NUMINAMATH_CALUDE_line_direction_vector_l1431_143198

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (2, 0) + t • d
  let d : ℝ × ℝ := direction_vector line
  let y (x : ℝ) : ℝ := (5 * x - 7) / 6
  ∀ x ≥ 2, (x - 2) ^ 2 + (y x) ^ 2 = t ^ 2 →
  d = (6 / Real.sqrt 61, 5 / Real.sqrt 61) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1431_143198


namespace NUMINAMATH_CALUDE_isosceles_triangle_same_color_l1431_143112

-- Define a circle
def Circle : Type := Unit

-- Define a color type
inductive Color
| C1
| C2

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  p1 : Point c
  p2 : Point c
  p3 : Point c
  isIsosceles : True  -- We assume this property without proving it

-- State the theorem
theorem isosceles_triangle_same_color (c : Circle) 
  (coloring : Point c → Color) :
  ∃ (t : IsoscelesTriangle c), 
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_same_color_l1431_143112


namespace NUMINAMATH_CALUDE_lcm_of_18_and_50_l1431_143119

theorem lcm_of_18_and_50 : Nat.lcm 18 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_and_50_l1431_143119


namespace NUMINAMATH_CALUDE_sum_of_special_function_l1431_143100

theorem sum_of_special_function (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) : 
  f (1/8) + f (2/8) + f (3/8) + f (4/8) + f (5/8) + f (6/8) + f (7/8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_function_l1431_143100


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l1431_143194

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ
  area_eq : area = (d1 * d2) / 2

/-- Theorem: In a rhombus with area 110 cm² and one diagonal 11 cm, the other diagonal is 20 cm -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.d1 = 11) 
    (h2 : r.area = 110) : 
    r.d2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l1431_143194


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l1431_143103

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l1431_143103


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1431_143138

theorem factor_of_polynomial (c d : ℤ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → c * x^19 + d * x^18 + 1 = 0) ↔ 
  (c = 1597 ∧ d = -2584) :=
by sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1431_143138


namespace NUMINAMATH_CALUDE_fahrenheit_95_equals_celsius_35_l1431_143176

-- Define the conversion function from Fahrenheit to Celsius
def fahrenheit_to_celsius (f : ℚ) : ℚ := (f - 32) * (5/9)

-- Theorem statement
theorem fahrenheit_95_equals_celsius_35 : fahrenheit_to_celsius 95 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_95_equals_celsius_35_l1431_143176


namespace NUMINAMATH_CALUDE_old_lamp_height_l1431_143118

theorem old_lamp_height (new_lamp_height : Real) (height_difference : Real) :
  new_lamp_height = 2.33 →
  height_difference = 1.33 →
  new_lamp_height - height_difference = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_old_lamp_height_l1431_143118


namespace NUMINAMATH_CALUDE_men_in_first_group_l1431_143134

/-- Represents the daily work done by a boy -/
def boy_work : ℝ := 1

/-- Represents the daily work done by a man -/
def man_work : ℝ := 2 * boy_work

/-- The number of days taken by the first group to complete the work -/
def days_group1 : ℕ := 5

/-- The number of days taken by the second group to complete the work -/
def days_group2 : ℕ := 4

/-- The number of boys in the first group -/
def boys_group1 : ℕ := 16

/-- The number of men in the second group -/
def men_group2 : ℕ := 13

/-- The number of boys in the second group -/
def boys_group2 : ℕ := 24

/-- The theorem stating that the number of men in the first group is 12 -/
theorem men_in_first_group :
  ∃ (m : ℕ), 
    (days_group1 : ℝ) * (m * man_work + boys_group1 * boy_work) = 
    (days_group2 : ℝ) * (men_group2 * man_work + boys_group2 * boy_work) ∧
    m = 12 := by
  sorry

end NUMINAMATH_CALUDE_men_in_first_group_l1431_143134


namespace NUMINAMATH_CALUDE_geometric_series_sum_127_128_l1431_143181

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_127_128 : 
  geometric_series_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_127_128_l1431_143181


namespace NUMINAMATH_CALUDE_cruise_ship_cabins_l1431_143129

/-- Represents the total number of cabins on a cruise ship -/
def total_cabins : ℕ := 600

/-- Represents the number of Deluxe cabins -/
def deluxe_cabins : ℕ := 30

/-- Theorem stating that the total number of cabins on the cruise ship is 600 -/
theorem cruise_ship_cabins :
  (deluxe_cabins : ℝ) + 0.2 * total_cabins + 3/4 * total_cabins = total_cabins :=
by sorry

end NUMINAMATH_CALUDE_cruise_ship_cabins_l1431_143129


namespace NUMINAMATH_CALUDE_count_single_colored_face_for_given_cube_l1431_143175

/-- Represents a cube cut in half and then into smaller cubes --/
structure CutCube where
  half_size : ℕ -- The number of small cubes along one edge of a half
  total_small_cubes : ℕ -- Total number of small cubes in each half

/-- Calculates the number of small cubes with only one colored face --/
def count_single_colored_face (c : CutCube) : ℕ :=
  4 * (c.half_size - 2) * (c.half_size - 2) * 2

/-- The theorem to be proved --/
theorem count_single_colored_face_for_given_cube :
  ∃ (c : CutCube), c.half_size = 4 ∧ c.total_small_cubes = 64 ∧ count_single_colored_face c = 32 :=
sorry

end NUMINAMATH_CALUDE_count_single_colored_face_for_given_cube_l1431_143175


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1431_143115

theorem min_value_sum_of_reciprocals (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1431_143115


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l1431_143139

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l1431_143139


namespace NUMINAMATH_CALUDE_parabola_shift_left_one_l1431_143197

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * shift + p.b,
    c := p.a * shift^2 - p.b * shift + p.c }

theorem parabola_shift_left_one :
  let original := Parabola.mk 1 0 2
  let shifted := shift_parabola original 1
  shifted = Parabola.mk 1 2 3 := by
  sorry

#check parabola_shift_left_one

end NUMINAMATH_CALUDE_parabola_shift_left_one_l1431_143197


namespace NUMINAMATH_CALUDE_equation_solution_l1431_143145

theorem equation_solution : 
  {x : ℝ | x^6 + (3 - x)^6 = 730} = {1.5 + Real.sqrt 5, 1.5 - Real.sqrt 5} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1431_143145


namespace NUMINAMATH_CALUDE_lian_lian_sales_properties_l1431_143169

/-- Represents the sales data and growth rate for the "Lian Lian" store -/
structure SalesData where
  june_sales : ℕ
  august_sales : ℕ
  months_between : ℕ
  growth_rate : ℝ

/-- Calculates the projected sales for the next month -/
def project_next_month_sales (data : SalesData) : ℝ :=
  data.august_sales * (1 + data.growth_rate)

/-- Theorem stating the properties of the "Lian Lian" store's sales data -/
theorem lian_lian_sales_properties (data : SalesData) 
  (h1 : data.june_sales = 30000)
  (h2 : data.august_sales = 36300)
  (h3 : data.months_between = 2)
  (h4 : data.growth_rate = (Real.sqrt 1.21 - 1)) :
  data.growth_rate = 0.1 ∧ project_next_month_sales data < 40000 := by
  sorry

#check lian_lian_sales_properties

end NUMINAMATH_CALUDE_lian_lian_sales_properties_l1431_143169


namespace NUMINAMATH_CALUDE_A_greater_than_B_l1431_143177

def A : ℕ → ℕ
  | 0 => 3
  | n+1 => 3^(A n)

def B : ℕ → ℕ
  | 0 => 8
  | n+1 => 8^(B n)

theorem A_greater_than_B (n : ℕ) : A (n + 1) > B n := by
  sorry

end NUMINAMATH_CALUDE_A_greater_than_B_l1431_143177


namespace NUMINAMATH_CALUDE_ink_blot_is_circle_l1431_143140

/-- A closed, bounded set in a plane -/
def InkBlot : Type := Set (ℝ × ℝ)

/-- The minimum distance from a point to the boundary of the ink blot -/
def min_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The maximum distance from a point to the boundary of the ink blot -/
def max_distance (S : InkBlot) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest of all minimum distances -/
def largest_min_distance (S : InkBlot) : ℝ := sorry

/-- The smallest of all maximum distances -/
def smallest_max_distance (S : InkBlot) : ℝ := sorry

/-- A circle in the plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : InkBlot := sorry

theorem ink_blot_is_circle (S : InkBlot) :
  largest_min_distance S = smallest_max_distance S →
  ∃ (center : ℝ × ℝ) (radius : ℝ), S = Circle center radius :=
sorry

end NUMINAMATH_CALUDE_ink_blot_is_circle_l1431_143140


namespace NUMINAMATH_CALUDE_degree_of_g_l1431_143182

/-- Given a polynomial f(x) = -7x^4 + 3x^3 + x - 5 and another polynomial g(x) such that 
    the degree of f(x) + g(x) is 2, prove that the degree of g(x) is 4. -/
theorem degree_of_g (f g : Polynomial ℝ) : 
  f = -7 * X^4 + 3 * X^3 + X - 5 →
  (f + g).degree = 2 →
  g.degree = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_g_l1431_143182


namespace NUMINAMATH_CALUDE_binary_multiplication_and_shift_l1431_143121

theorem binary_multiplication_and_shift :
  let a : Nat := 109  -- 1101101₂ in decimal
  let b : Nat := 15   -- 1111₂ in decimal
  let product : Nat := a * b
  let shifted : Rat := (product : Rat) / 4  -- Shifting 2 places right is equivalent to dividing by 4
  shifted = 1010011111.25 := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_and_shift_l1431_143121


namespace NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l1431_143116

/-- The probability of exactly one ball landing in the last box when 100 balls
    are randomly distributed among 100 boxes. -/
theorem probability_one_ball_in_last_box :
  let n : ℕ := 100
  let p : ℝ := 1 / n
  (n : ℝ) * p * (1 - p) ^ (n - 1) = (1 - 1 / n) ^ (n - 1) := by sorry

end NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l1431_143116


namespace NUMINAMATH_CALUDE_circle_properties_l1431_143173

/-- A circle with diameter endpoints (2, -3) and (8, 9) has center (5, 3) and radius 3√5 -/
theorem circle_properties :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (8, 9)
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let r : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  C = (5, 3) ∧ r = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l1431_143173


namespace NUMINAMATH_CALUDE_range_of_a_l1431_143117

-- Define the conditions
def p (x : ℝ) : Prop := 1 / (x - 3) ≥ 1
def q (x a : ℝ) : Prop := |x - a| < 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- Theorem statement
theorem range_of_a :
  ∃ a_lower a_upper : ℝ, a_lower = 3 ∧ a_upper = 4 ∧
  ∀ a : ℝ, sufficient_not_necessary a ↔ a_lower < a ∧ a ≤ a_upper :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1431_143117


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l1431_143141

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices (d : Decagon) : ℕ := 10

/-- The number of pairs of adjacent vertices in a decagon -/
def num_adjacent_pairs (d : Decagon) : ℕ := 20

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_vertex_pairs (d : Decagon) : ℕ := (num_vertices d).choose 2

/-- The probability of choosing two adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  (num_adjacent_pairs d : ℚ) / (total_vertex_pairs d : ℚ)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 4/9 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l1431_143141


namespace NUMINAMATH_CALUDE_may_savings_l1431_143126

def savings (month : ℕ) : ℕ :=
  match month with
  | 0 => 20  -- January
  | 1 => 3 * 20  -- February
  | n + 2 => 3 * savings (n + 1) + 50  -- March onwards

theorem may_savings : savings 4 = 2270 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l1431_143126


namespace NUMINAMATH_CALUDE_min_value_expression_l1431_143144

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1431_143144


namespace NUMINAMATH_CALUDE_square_root_problem_l1431_143174

theorem square_root_problem (x a : ℝ) : 
  ((2 * a + 1) ^ 2 = x ∧ (4 - a) ^ 2 = x) → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1431_143174


namespace NUMINAMATH_CALUDE_apartment_rent_calculation_required_rent_is_correct_l1431_143136

/-- Calculate the required monthly rent for an apartment investment --/
theorem apartment_rent_calculation (investment : ℝ) (maintenance_rate : ℝ) 
  (annual_taxes : ℝ) (desired_return_rate : ℝ) : ℝ :=
  let annual_return := investment * desired_return_rate
  let total_annual_requirement := annual_return + annual_taxes
  let monthly_net_requirement := total_annual_requirement / 12
  let monthly_rent := monthly_net_requirement / (1 - maintenance_rate)
  monthly_rent

/-- The required monthly rent is approximately $153.70 --/
theorem required_rent_is_correct : 
  ∃ ε > 0, |apartment_rent_calculation 20000 0.1 460 0.06 - 153.70| < ε :=
sorry

end NUMINAMATH_CALUDE_apartment_rent_calculation_required_rent_is_correct_l1431_143136


namespace NUMINAMATH_CALUDE_quadratic_sum_l1431_143167

/-- Given a quadratic function f(x) = -2x^2 + 16x - 72, prove that when expressed
    in the form a(x+b)^2 + c, the sum of a, b, and c is equal to -46. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), 
    (∀ x, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) ∧
    (a + b + c = -46) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1431_143167


namespace NUMINAMATH_CALUDE_leahs_coins_value_l1431_143154

theorem leahs_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  p = 2 * (n + 3) → 
  5 * n + p = 27 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l1431_143154


namespace NUMINAMATH_CALUDE_printer_fraction_of_total_l1431_143162

/-- The price of the printer as a fraction of the total price with an enhanced computer -/
theorem printer_fraction_of_total (basic_computer_price printer_price enhanced_computer_price total_price_basic total_price_enhanced : ℚ) : 
  total_price_basic = basic_computer_price + printer_price →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2000 →
  total_price_enhanced = enhanced_computer_price + printer_price →
  printer_price / total_price_enhanced = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_printer_fraction_of_total_l1431_143162


namespace NUMINAMATH_CALUDE_line_slope_l1431_143124

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l1431_143124


namespace NUMINAMATH_CALUDE_johns_apartment_rental_l1431_143172

/-- John's apartment rental problem -/
theorem johns_apartment_rental 
  (num_subletters : ℕ) 
  (subletter_payment : ℕ) 
  (annual_profit : ℕ) 
  (monthly_rent : ℕ) : 
  num_subletters = 3 → 
  subletter_payment = 400 → 
  annual_profit = 3600 → 
  monthly_rent = 900 → 
  (num_subletters * subletter_payment - monthly_rent) * 12 = annual_profit :=
by sorry

end NUMINAMATH_CALUDE_johns_apartment_rental_l1431_143172


namespace NUMINAMATH_CALUDE_simplified_expression_l1431_143155

theorem simplified_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplified_expression_l1431_143155


namespace NUMINAMATH_CALUDE_robe_cost_is_two_l1431_143122

/-- Calculates the cost per robe given the total number of singers, existing robes, and total cost for new robes. -/
def cost_per_robe (total_singers : ℕ) (existing_robes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_singers - existing_robes)

/-- Proves that the cost per robe is $2 given the specific conditions of the problem. -/
theorem robe_cost_is_two :
  cost_per_robe 30 12 36 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robe_cost_is_two_l1431_143122


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l1431_143163

/-- Given a 2x2 matrix N, prove that its inverse can be expressed as c * N + d * I -/
theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h₁ : N 0 0 = 3) (h₂ : N 0 1 = 1) (h₃ : N 1 0 = -2) (h₄ : N 1 1 = 4) :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = -1/14 ∧ d = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l1431_143163


namespace NUMINAMATH_CALUDE_allan_total_balloons_l1431_143196

def initial_balloons : Nat := 5
def additional_balloons : Nat := 3

theorem allan_total_balloons : 
  initial_balloons + additional_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_allan_total_balloons_l1431_143196


namespace NUMINAMATH_CALUDE_only_1_and_4_perpendicular_l1431_143189

-- Define the slopes of the lines
def m1 : ℚ := 2/3
def m2 : ℚ := -2/3
def m3 : ℚ := -2/3
def m4 : ℚ := -3/2

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem only_1_and_4_perpendicular :
  (are_perpendicular m1 m4) ∧
  ¬(are_perpendicular m1 m2) ∧
  ¬(are_perpendicular m1 m3) ∧
  ¬(are_perpendicular m2 m3) ∧
  ¬(are_perpendicular m2 m4) ∧
  ¬(are_perpendicular m3 m4) :=
by sorry

end NUMINAMATH_CALUDE_only_1_and_4_perpendicular_l1431_143189


namespace NUMINAMATH_CALUDE_range_of_p_or_q_range_of_a_intersection_l1431_143178

-- Define sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) : Prop := x ∈ B

-- Theorem 1: The set of x satisfying p ∨ q is equal to [-2, 5)
theorem range_of_p_or_q : {x : ℝ | p x ∨ q x} = Set.Ico (-2) 5 := by sorry

-- Theorem 2: The set of a satisfying A ∩ C = C is equal to (-∞, -4] ∪ [-1, 1/2]
theorem range_of_a_intersection : 
  {a : ℝ | A ∩ C a = C a} = Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by sorry

end NUMINAMATH_CALUDE_range_of_p_or_q_range_of_a_intersection_l1431_143178


namespace NUMINAMATH_CALUDE_arc_length_for_specific_circle_l1431_143156

theorem arc_length_for_specific_circle (r : ℝ) (α : ℝ) (l : ℝ) : 
  r = π → α = 2 * π / 3 → l = r * α → l = 2 * π^2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_specific_circle_l1431_143156


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l1431_143160

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in the form y^2 = 2px -/
structure Parabola where
  p : ℝ

def Parabola.contains (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * p.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem parabola_chord_intersection (p : Parabola) (m : Point) (d e : Point) :
  p.p = 2 →
  p.contains m →
  m.y = 4 →
  ∃ (l_md l_me l_de : Line),
    l_md.contains m ∧ l_md.contains d ∧
    l_me.contains m ∧ l_me.contains e ∧
    l_de.contains d ∧ l_de.contains e ∧
    perpendicular l_md l_me →
    l_de.contains (Point.mk 8 (-4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l1431_143160


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1431_143159

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf_odd : OddFunction f)
    (hf_2 : f 2 = 0) (hf_deriv : ∀ x > 0, (x * (deriv f x) - f x) / x^2 < 0) :
    {x : ℝ | x^2 * f x > 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1431_143159


namespace NUMINAMATH_CALUDE_license_plate_count_l1431_143164

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of prime digits under 10 -/
def num_prime_digits : ℕ := 4

/-- The total number of license plates -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_digits * num_prime_digits

theorem license_plate_count :
  total_license_plates = 351520 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1431_143164


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l1431_143170

theorem cryptarithmetic_puzzle :
  ∀ (E F D : ℕ),
    E + F + D = E * F - 3 →
    E - F = 2 →
    E ≠ F ∧ E ≠ D ∧ F ≠ D →
    E ≤ 9 ∧ F ≤ 9 ∧ D ≤ 9 →
    D = 4 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l1431_143170


namespace NUMINAMATH_CALUDE_blue_jellybean_probability_l1431_143150

/-- The probability of drawing 3 blue jellybeans in succession from a bag of 10 red and 10 blue jellybeans without replacement is 1/9.5. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  let prob_first : ℚ := blue_jellybeans / total_jellybeans
  let prob_second : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1)
  let prob_third : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2)
  prob_first * prob_second * prob_third = 1 / (19 / 2) :=
by sorry

end NUMINAMATH_CALUDE_blue_jellybean_probability_l1431_143150


namespace NUMINAMATH_CALUDE_painting_selection_theorem_l1431_143188

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 6

/-- The number of oil paintings -/
def oil_paintings : Nat := 4

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 5

/-- The number of ways to select one painting from each type -/
def select_one_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to select two paintings of different types -/
def select_two_different : Nat :=
  traditional_paintings * oil_paintings +
  traditional_paintings * watercolor_paintings +
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem :
  select_one_each = 120 ∧ select_two_different = 74 := by
  sorry

end NUMINAMATH_CALUDE_painting_selection_theorem_l1431_143188


namespace NUMINAMATH_CALUDE_probability_red_or_white_l1431_143152

/-- Probability of selecting a red or white marble from a bag -/
theorem probability_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h_total : total = 30)
  (h_blue : blue = 5)
  (h_red : red = 9)
  (h_positive : total > 0) :
  (red + (total - blue - red)) / total = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l1431_143152


namespace NUMINAMATH_CALUDE_min_sum_squares_l1431_143132

def f (x : ℝ) := |x + 1| - |x - 4|

def m₀ : ℝ := 5

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 3 * a + 4 * b + 5 * c = m₀) :
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1431_143132


namespace NUMINAMATH_CALUDE_first_group_students_l1431_143195

theorem first_group_students (total : ℕ) (group2 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group2 = 8)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group2 + group3 + group4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_students_l1431_143195


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1431_143168

def U : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}
def A : Finset ℕ := {0,1,3,5,8}
def B : Finset ℕ := {2,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7,9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1431_143168


namespace NUMINAMATH_CALUDE_woman_work_days_value_l1431_143143

/-- The number of days it takes for a woman to complete the work -/
def woman_work_days (total_members : ℕ) (man_work_days : ℕ) (combined_work_days : ℕ) (num_women : ℕ) : ℚ :=
  let num_men := total_members - num_women
  let man_work_rate := 1 / man_work_days
  let total_man_work := (combined_work_days / 2 : ℚ) * man_work_rate * num_men
  let total_woman_work := 1 - total_man_work
  let woman_work_rate := (total_woman_work * 3) / (combined_work_days * num_women)
  1 / woman_work_rate

/-- Theorem stating the number of days it takes for a woman to complete the work -/
theorem woman_work_days_value :
  woman_work_days 15 120 17 3 = 5100 / 83 :=
by sorry

end NUMINAMATH_CALUDE_woman_work_days_value_l1431_143143


namespace NUMINAMATH_CALUDE_homework_problem_l1431_143161

theorem homework_problem (p t : ℕ) (h1 : p > 15) (h2 : p * t = (2*p - 6) * (t - 3)) : p * t = 126 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l1431_143161


namespace NUMINAMATH_CALUDE_max_intersection_area_l1431_143128

/-- A right prism with a square base centered at the origin --/
structure Prism :=
  (side_length : ℝ)
  (center : ℝ × ℝ × ℝ := (0, 0, 0))

/-- A plane in 3D space defined by its equation coefficients --/
structure Plane :=
  (a b c d : ℝ)

/-- The intersection of a prism and a plane --/
def intersection (p : Prism) (plane : Plane) : Set (ℝ × ℝ × ℝ) :=
  {pt : ℝ × ℝ × ℝ | 
    let (x, y, z) := pt
    plane.a * x + plane.b * y + plane.c * z = plane.d ∧
    |x| ≤ p.side_length / 2 ∧
    |y| ≤ p.side_length / 2}

/-- The area of a set in 3D space --/
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of the intersection is equal to the area of the square base --/
theorem max_intersection_area (p : Prism) (plane : Plane) :
  p.side_length = 12 ∧
  plane = {a := 3, b := -6, c := 2, d := 24} →
  area (intersection p plane) ≤ p.side_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_area_l1431_143128


namespace NUMINAMATH_CALUDE_larger_number_proof_l1431_143142

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1431_143142


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1431_143180

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1431_143180


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l1431_143166

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l1431_143166
