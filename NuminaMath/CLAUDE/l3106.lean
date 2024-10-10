import Mathlib

namespace f_properties_l3106_310641

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∀ x, f (Real.pi - x) = f (Real.pi + x)) :=
by sorry

end f_properties_l3106_310641


namespace parabola_directrix_coefficient_l3106_310633

/-- For a parabola y = ax^2 with directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient : 
  ∀ (a : ℝ), (∀ x y : ℝ, y = a * x^2) → 
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k = -(1 / (4 * a))) → 
  a = -1/8 := by
sorry

end parabola_directrix_coefficient_l3106_310633


namespace sqrt_sum_equals_two_sqrt_two_l3106_310697

theorem sqrt_sum_equals_two_sqrt_two :
  Real.sqrt (5 - 2 * Real.sqrt 6) + Real.sqrt (5 + 2 * Real.sqrt 6) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equals_two_sqrt_two_l3106_310697


namespace product_of_numbers_l3106_310650

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
  sorry

end product_of_numbers_l3106_310650


namespace mps_to_kmph_conversion_l3106_310619

-- Define the conversion factor from mps to kmph
def mps_to_kmph_factor : ℝ := 3.6

-- Define the speed in mps
def speed_mps : ℝ := 15

-- Define the speed in kmph
def speed_kmph : ℝ := 54

-- Theorem to prove the conversion
theorem mps_to_kmph_conversion :
  speed_mps * mps_to_kmph_factor = speed_kmph := by
  sorry

end mps_to_kmph_conversion_l3106_310619


namespace expand_and_simplify_l3106_310694

theorem expand_and_simplify (x : ℝ) : 5 * (x + 6) * (x + 2) * (x + 7) = 5*x^3 + 75*x^2 + 340*x + 420 := by
  sorry

end expand_and_simplify_l3106_310694


namespace completed_square_q_value_l3106_310613

theorem completed_square_q_value (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = 5) :
  ∃ (p q : ℝ), ∀ x, (x^2 + b*x + c = 0 ↔ (x + p)^2 = q) ∧ q = 4 := by
  sorry

end completed_square_q_value_l3106_310613


namespace mandy_nutmeg_amount_l3106_310678

/-- The amount of cinnamon Mandy used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The difference between cinnamon and nutmeg in tablespoons -/
def difference : ℚ := 0.16666666666666666

/-- The amount of nutmeg Mandy used in tablespoons -/
def nutmeg : ℚ := cinnamon - difference

theorem mandy_nutmeg_amount : nutmeg = 0.5 := by sorry

end mandy_nutmeg_amount_l3106_310678


namespace sqrt_sum_inequality_l3106_310649

theorem sqrt_sum_inequality (a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : (a + 1/2) * (b + 1/2) ≥ 0) : 
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end sqrt_sum_inequality_l3106_310649


namespace shark_sightings_difference_l3106_310688

theorem shark_sightings_difference (daytona_sightings cape_may_sightings : ℕ) 
  (h1 : daytona_sightings = 26)
  (h2 : cape_may_sightings = 7)
  (h3 : daytona_sightings > 3 * cape_may_sightings) :
  daytona_sightings - 3 * cape_may_sightings = 5 := by
sorry

end shark_sightings_difference_l3106_310688


namespace function_properties_l3106_310601

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 3 * (a * x^3 + b * x^2)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 9 * a * x^2 + 6 * b * x

theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b 1) ∧
    (f a b 1 = 3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = -2) ∧
    (b = 3) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≤ 15) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = 15) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≥ -81) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = -81) := by
  sorry

end function_properties_l3106_310601


namespace semicircle_perimeter_approx_l3106_310605

/-- The perimeter of a semicircle with radius 9 is approximately 46.27 units. -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((9 * Real.pi + 18) : ℝ) - 46.27| < ε := by
  sorry

end semicircle_perimeter_approx_l3106_310605


namespace average_net_sales_proof_l3106_310674

def monthly_sales : List ℕ := [120, 80, 50, 130, 90, 160]
def monthly_expense : ℕ := 30
def num_months : ℕ := 6

theorem average_net_sales_proof :
  (List.sum monthly_sales - monthly_expense * num_months) / num_months = 75 := by
  sorry

end average_net_sales_proof_l3106_310674


namespace unique_integer_solution_l3106_310692

theorem unique_integer_solution : ∃! x : ℤ, x + 12 > 14 ∧ -3*x > -9 := by sorry

end unique_integer_solution_l3106_310692


namespace carson_clawed_39_times_l3106_310617

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (claws_per_wombat : ℕ) (claws_per_rhea : ℕ) : ℕ :=
  num_wombats * claws_per_wombat + num_rheas * claws_per_rhea

/-- Theorem stating that Carson gets clawed 39 times. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end carson_clawed_39_times_l3106_310617


namespace special_integers_count_l3106_310686

/-- The sum of all positive divisors of n including twice the greatest prime divisor of n -/
def g (n : ℕ) : ℕ := sorry

/-- The count of integers j such that 1 ≤ j ≤ 5000 and g(j) = j + 2√j + 1 -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 19 := by sorry

end special_integers_count_l3106_310686


namespace sqrt_rational_sum_l3106_310636

theorem sqrt_rational_sum (a b r : ℚ) (h : Real.sqrt a + Real.sqrt b = r) :
  ∃ (c d : ℚ), Real.sqrt a = c ∧ Real.sqrt b = d := by
  sorry

end sqrt_rational_sum_l3106_310636


namespace rectangular_prism_diagonals_l3106_310673

/-- A rectangular prism with dimensions 3, 4, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: The total number of diagonals in a rectangular prism with dimensions 3, 4, and 5 is 16. -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end rectangular_prism_diagonals_l3106_310673


namespace initial_men_count_l3106_310677

/-- The number of men initially doing the work -/
def initial_men : ℕ := 36

/-- The time taken by the initial group of men to complete the work -/
def initial_time : ℕ := 25

/-- The number of men in the second group -/
def second_group : ℕ := 15

/-- The time taken by the second group to complete the work -/
def second_time : ℕ := 60

/-- Theorem stating that the initial number of men is 36 -/
theorem initial_men_count : initial_men = 36 := by
  sorry

#check initial_men_count

end initial_men_count_l3106_310677


namespace fraction_simplification_l3106_310630

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end fraction_simplification_l3106_310630


namespace dave_shows_per_week_l3106_310643

theorem dave_shows_per_week :
  let strings_per_show : ℕ := 2
  let total_weeks : ℕ := 12
  let total_strings : ℕ := 144
  let shows_per_week : ℕ := total_strings / (strings_per_show * total_weeks)
  shows_per_week = 6 :=
by sorry

end dave_shows_per_week_l3106_310643


namespace binomial_1493_1492_l3106_310664

theorem binomial_1493_1492 : Nat.choose 1493 1492 = 1493 := by
  sorry

end binomial_1493_1492_l3106_310664


namespace banana_permutations_l3106_310622

-- Define the word BANANA
def word : String := "BANANA"

-- Define the total number of letters
def total_letters : Nat := word.length

-- Define the number of As
def num_A : Nat := 3

-- Define the number of Ns
def num_N : Nat := 2

-- Theorem statement
theorem banana_permutations : 
  (Nat.factorial total_letters) / (Nat.factorial num_A * Nat.factorial num_N) = 60 :=
by sorry

end banana_permutations_l3106_310622


namespace tank_capacity_l3106_310610

theorem tank_capacity (C : ℝ) 
  (h1 : (3/4 : ℝ) * C + 8 = (7/8 : ℝ) * C) : C = 64 := by
  sorry

end tank_capacity_l3106_310610


namespace expression_simplification_l3106_310681

theorem expression_simplification (a b : ℝ) 
  (ha : a = 3 + Real.sqrt 5) 
  (hb : b = 3 - Real.sqrt 5) : 
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * (a*b / (a - b)) = 2/3 := by
  sorry

end expression_simplification_l3106_310681


namespace base6_addition_proof_l3106_310614

/-- Convert a base 6 number to base 10 -/
def base6to10 (x y z : Nat) : Nat :=
  x * 36 + y * 6 + z

/-- Addition in base 6 -/
def addBase6 (x₁ y₁ z₁ x₂ y₂ z₂ : Nat) : Nat × Nat × Nat :=
  let sum := base6to10 x₁ y₁ z₁ + base6to10 x₂ y₂ z₂
  (sum / 36, (sum % 36) / 6, sum % 6)

theorem base6_addition_proof (C D : Nat) :
  C < 6 ∧ D < 6 ∧
  addBase6 5 C D 0 5 2 = (1, 2, C) →
  C + D = 5 := by
  sorry

end base6_addition_proof_l3106_310614


namespace spiral_grid_third_row_sum_l3106_310661

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid := Position → ℕ

/-- Creates a spiral grid with the given dimensions -/
def create_spiral_grid (n : ℕ) : SpiralGrid :=
  sorry

/-- Returns the numbers in a given row of the grid -/
def numbers_in_row (grid : SpiralGrid) (row : ℕ) : List ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := create_spiral_grid 17
  let third_row_numbers := numbers_in_row grid 3
  let min_number := third_row_numbers.minimum?
  let max_number := third_row_numbers.maximum?
  ∀ min max, min_number = some min → max_number = some max →
    min + max = 577 := by
  sorry

end spiral_grid_third_row_sum_l3106_310661


namespace snooker_tournament_ticket_sales_l3106_310612

/-- Proves that the total number of tickets sold is 336 given the specified conditions -/
theorem snooker_tournament_ticket_sales
  (vip_cost : ℕ)
  (general_cost : ℕ)
  (total_revenue : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_cost = 45)
  (h2 : general_cost = 20)
  (h3 : total_revenue = 7500)
  (h4 : ticket_difference = 276)
  (h5 : ∃ (vip general : ℕ),
    vip_cost * vip + general_cost * general = total_revenue ∧
    vip + ticket_difference = general) :
  ∃ (vip general : ℕ), vip + general = 336 := by
  sorry

end snooker_tournament_ticket_sales_l3106_310612


namespace difference_of_squares_special_case_l3106_310639

theorem difference_of_squares_special_case (m : ℝ) : 
  (2 * m + 1/2) * (2 * m - 1/2) = 4 * m^2 - 1/4 := by
  sorry

end difference_of_squares_special_case_l3106_310639


namespace abs_neg_ten_eq_ten_l3106_310683

theorem abs_neg_ten_eq_ten : |(-10 : ℤ)| = 10 := by
  sorry

end abs_neg_ten_eq_ten_l3106_310683


namespace race_track_width_l3106_310670

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 140.0563499208679 →
  ∃ width : ℝ, abs (width - ((2 * Real.pi * outer_radius - inner_circumference) / 2)) < 0.001 :=
by
  sorry

end race_track_width_l3106_310670


namespace proposition_properties_l3106_310696

theorem proposition_properties :
  -- 1. Negation of existential quantifier
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  -- 2. Sufficient but not necessary condition
  (∃ x : ℝ, x = 1 → x^2 - 4*x + 3 = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x ≠ 1) ∧
  -- 3. Converse of implication
  ((∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = 1) →
   (∀ x : ℝ, x ≠ 1 → x^2 - 4*x + 3 ≠ 0)) ∧
  -- 4. Falsity of conjunction doesn't imply falsity of both propositions
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
by sorry

end proposition_properties_l3106_310696


namespace desired_depth_is_18_l3106_310618

/-- Represents the digging scenario with given parameters -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth for a given digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℚ :=
  (scenario.initial_men * scenario.initial_hours * scenario.initial_depth : ℚ) /
  ((scenario.initial_men + scenario.extra_men) * scenario.new_hours)

/-- Theorem stating that the desired depth for the given scenario is 18 meters -/
theorem desired_depth_is_18 (scenario : DiggingScenario) 
    (h1 : scenario.initial_men = 9)
    (h2 : scenario.initial_hours = 8)
    (h3 : scenario.initial_depth = 30)
    (h4 : scenario.new_hours = 6)
    (h5 : scenario.extra_men = 11) :
  desired_depth scenario = 18 := by
  sorry

#eval desired_depth { initial_men := 9, initial_hours := 8, initial_depth := 30, new_hours := 6, extra_men := 11 }

end desired_depth_is_18_l3106_310618


namespace arithmetic_sequence_20th_term_l3106_310669

/-- Given an arithmetic sequence with first term 3 and common difference 4,
    the 20th term of the sequence is 79. -/
theorem arithmetic_sequence_20th_term :
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 4   -- common difference
  let n : ℕ := 20  -- term number we're looking for
  let aₙ : ℕ := a₁ + (n - 1) * d  -- formula for nth term of arithmetic sequence
  aₙ = 79 := by sorry

end arithmetic_sequence_20th_term_l3106_310669


namespace professor_k_lectures_l3106_310615

def num_jokes : ℕ := 8

theorem professor_k_lectures (num_jokes : ℕ) (h : num_jokes = 8) :
  (Finset.sum (Finset.range 2) (λ i => Nat.choose num_jokes (i + 2))) = 84 := by
  sorry

end professor_k_lectures_l3106_310615


namespace china_population_in_scientific_notation_l3106_310634

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The population of China according to the sixth national census -/
def china_population : ℝ := 1370540000

/-- The scientific notation representation of China's population -/
def china_population_scientific : ℝ := 1.37054 * (10 : ℝ) ^ 9

theorem china_population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n china_population ∧
  china_population_scientific = a * (10 : ℝ) ^ n :=
sorry

end china_population_in_scientific_notation_l3106_310634


namespace zhang_hong_weight_estimate_l3106_310603

/-- Regression equation for weight based on height -/
def weight_estimate (height : ℝ) : ℝ := 0.72 * height - 58.2

/-- Age range for which the regression equation is valid -/
def valid_age_range : Set ℝ := Set.Icc 18 38

theorem zhang_hong_weight_estimate :
  20 ∈ valid_age_range →
  weight_estimate 178 = 69.96 := by
  sorry

end zhang_hong_weight_estimate_l3106_310603


namespace blue_to_purple_ratio_l3106_310656

/-- Represents the number of beads of each color in a necklace. -/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ

/-- The properties of the necklace as described in the problem. -/
def necklace_properties (b : BeadCounts) : Prop :=
  b.purple = 7 ∧
  b.green = b.blue + 11 ∧
  b.purple + b.blue + b.green = 46

/-- The theorem stating the ratio of blue to purple beads is 2:1. -/
theorem blue_to_purple_ratio (b : BeadCounts) :
  necklace_properties b → b.blue = 2 * b.purple := by
  sorry

end blue_to_purple_ratio_l3106_310656


namespace geometric_sequence_proof_l3106_310642

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_proof (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_ratio : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  is_geometric a ∧ a 6 = 16 / 81 := by
  sorry

end geometric_sequence_proof_l3106_310642


namespace largest_room_length_l3106_310604

theorem largest_room_length 
  (largest_width : ℝ) 
  (smallest_width smallest_length : ℝ) 
  (area_difference : ℝ) 
  (h1 : largest_width = 45)
  (h2 : smallest_width = 15)
  (h3 : smallest_length = 8)
  (h4 : area_difference = 1230)
  (h5 : largest_width * largest_length - smallest_width * smallest_length = area_difference) :
  largest_length = 30 :=
by
  sorry

end largest_room_length_l3106_310604


namespace most_sweets_l3106_310675

/-- Given the distribution of sweets among three people, prove who received the most. -/
theorem most_sweets (total : ℕ) (minsu jaeyoung heesu : ℕ) 
  (h_total : total = 30)
  (h_minsu : minsu = 12)
  (h_jaeyoung : jaeyoung = 3)
  (h_heesu : heesu = 15)
  (h_sum : minsu + jaeyoung + heesu = total) :
  heesu > minsu ∧ heesu > jaeyoung := by
  sorry

end most_sweets_l3106_310675


namespace isosceles_triangle_perimeter_l3106_310621

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 6*a + 5 = 0 →
  b^2 - 6*b + 5 = 0 →
  a ≠ b →
  (a + a + b = 11 ∨ b + b + a = 11) :=
by sorry

end isosceles_triangle_perimeter_l3106_310621


namespace pasta_preference_ratio_l3106_310628

theorem pasta_preference_ratio :
  ∀ (total students : ℕ) (pasta_types : ℕ) (spaghetti_pref manicotti_pref : ℕ),
    total = 800 →
    pasta_types = 5 →
    spaghetti_pref = 300 →
    manicotti_pref = 120 →
    (spaghetti_pref : ℚ) / manicotti_pref = 5 / 2 := by
  sorry

end pasta_preference_ratio_l3106_310628


namespace die_product_divisibility_l3106_310671

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem die_product_divisibility :
  let die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  ∀ S : Finset ℕ, S ⊆ die_numbers → S.card = 7 →
    let product := S.prod id
    (is_divisible product 192) ∧
    (∀ n > 192, ∃ T : Finset ℕ, T ⊆ die_numbers ∧ T.card = 7 ∧ ¬(is_divisible (T.prod id) n)) :=
by sorry

end die_product_divisibility_l3106_310671


namespace point_in_third_quadrant_l3106_310611

theorem point_in_third_quadrant (a b : ℝ) : a + b < 0 → a * b > 0 → a < 0 ∧ b < 0 := by
  sorry

end point_in_third_quadrant_l3106_310611


namespace bathing_suits_for_women_l3106_310657

theorem bathing_suits_for_women (total : ℕ) (men : ℕ) (women : ℕ) : 
  total = 19766 → men = 14797 → women = total - men → women = 4969 := by
sorry

end bathing_suits_for_women_l3106_310657


namespace inner_square_probability_l3106_310623

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Checks if a square is on the perimeter or center lines -/
def is_on_perimeter_or_center (b : Board) (row col : ℕ) : Prop :=
  row = 1 ∨ row = b.size ∨ col = 1 ∨ col = b.size ∨
  row = b.size / 2 ∨ row = b.size / 2 + 1 ∨
  col = b.size / 2 ∨ col = b.size / 2 + 1

/-- Counts squares not on perimeter or center lines -/
def count_inner_squares (b : Board) : ℕ :=
  (b.size - 4) * (b.size - 4)

/-- The main theorem -/
theorem inner_square_probability (b : Board) (h : b.size = 10) :
  (count_inner_squares b : ℚ) / (b.size * b.size : ℚ) = 3 / 5 :=
sorry

end inner_square_probability_l3106_310623


namespace possible_x_values_l3106_310648

def M (x : ℝ) : Set ℝ := {3, 9, 3*x}
def N (x : ℝ) : Set ℝ := {3, x^2}

theorem possible_x_values :
  ∀ x : ℝ, N x ⊆ M x → x = -3 ∨ x = 0 :=
by sorry

end possible_x_values_l3106_310648


namespace sugar_delivery_problem_l3106_310667

/-- Represents the sugar delivery problem -/
def SugarDelivery (total_bags : ℕ) (total_weight : ℝ) (granulated_ratio : ℝ) (sugar_mass_ratio : ℝ) : Prop :=
  ∃ (sugar_bags : ℕ) (granulated_bags : ℕ) (sugar_weight : ℝ) (granulated_weight : ℝ),
    -- Total number of bags
    sugar_bags + granulated_bags = total_bags ∧
    -- Granulated sugar bags ratio
    granulated_bags = (1 + granulated_ratio) * sugar_bags ∧
    -- Total weight
    sugar_weight + granulated_weight = total_weight ∧
    -- Mass ratio between sugar and granulated sugar bags
    sugar_weight * granulated_bags = sugar_mass_ratio * granulated_weight * sugar_bags ∧
    -- Correct weights
    sugar_weight = 3 ∧ granulated_weight = 1.8

theorem sugar_delivery_problem :
  SugarDelivery 63 4.8 0.25 0.75 :=
sorry

end sugar_delivery_problem_l3106_310667


namespace segment_length_on_ellipse_l3106_310626

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem segment_length_on_ellipse :
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  (∃ t : ℝ, A = F₁ + t • (B - F₁)) →  -- A, B, and F₁ are collinear
  distance F₂ A + distance F₂ B = 12 →
  distance A B = 8 := by
  sorry

end segment_length_on_ellipse_l3106_310626


namespace largest_possible_z_value_l3106_310620

theorem largest_possible_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = 2 * Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 := by sorry

end largest_possible_z_value_l3106_310620


namespace wood_measurement_l3106_310685

theorem wood_measurement (x y : ℝ) : 
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔ 
  (∃ (wood_length rope_length : ℝ), 
    wood_length = x ∧ 
    rope_length = y ∧ 
    rope_length - wood_length = 4.5 ∧ 
    0.5 * rope_length - wood_length = -1) :=
by sorry

end wood_measurement_l3106_310685


namespace paul_total_crayons_l3106_310640

/-- The number of crayons Paul received for his birthday -/
def birthday_crayons : ℝ := 479.0

/-- The number of crayons Paul received at the end of the school year -/
def school_year_crayons : ℝ := 134.0

/-- The total number of crayons Paul has now -/
def total_crayons : ℝ := birthday_crayons + school_year_crayons

/-- Theorem stating that Paul's total number of crayons is 613.0 -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end paul_total_crayons_l3106_310640


namespace kelly_games_left_l3106_310698

/-- Calculates the number of games left after finding more and giving some away -/
def games_left (initial : ℕ) (found : ℕ) (given_away : ℕ) : ℕ :=
  initial + found - given_away

/-- Proves that Kelly will have 6 games left -/
theorem kelly_games_left : games_left 80 31 105 = 6 := by
  sorry

end kelly_games_left_l3106_310698


namespace tammys_climbing_speed_l3106_310689

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (v : ℝ), 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end tammys_climbing_speed_l3106_310689


namespace mean_of_fractions_l3106_310655

theorem mean_of_fractions (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 1/8) :
  (a + b + c) / 3 = 7/24 := by
sorry

end mean_of_fractions_l3106_310655


namespace percentage_relationships_l3106_310654

/-- Given the relationships between a, b, c, d, and e, prove the relative percentages. -/
theorem percentage_relationships (a b c d e : ℝ) 
  (hc_a : c = 0.25 * a)
  (hc_b : c = 0.5 * b)
  (hd_a : d = 0.4 * a)
  (hd_b : d = 0.2 * b)
  (he_d : e = 0.35 * d)
  (he_c : e = 0.15 * c) :
  b = 0.5 * a ∧ c = 0.625 * d ∧ d = (1 / 0.35) * e := by
  sorry


end percentage_relationships_l3106_310654


namespace pizza_cost_l3106_310637

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end pizza_cost_l3106_310637


namespace multiplication_fraction_equality_l3106_310659

theorem multiplication_fraction_equality : 12 * (1 / 8) * 32 = 48 := by
  sorry

end multiplication_fraction_equality_l3106_310659


namespace leap_year_inequality_l3106_310690

/-- Represents the dataset for a leap year as described in the problem -/
def leapYearData : List ℕ := sorry

/-- Calculates the median of modes for the leap year dataset -/
def medianOfModes (data : List ℕ) : ℚ := sorry

/-- Calculates the mean for the leap year dataset -/
def mean (data : List ℕ) : ℚ := sorry

/-- Calculates the median for the leap year dataset -/
def median (data : List ℕ) : ℚ := sorry

theorem leap_year_inequality :
  let d := medianOfModes leapYearData
  let μ := mean leapYearData
  let M := median leapYearData
  d < μ ∧ μ < M := by sorry

end leap_year_inequality_l3106_310690


namespace sqrt_equation_solution_l3106_310607

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 * x + 9) = 11 → x = 28 := by
  sorry

end sqrt_equation_solution_l3106_310607


namespace geometric_series_ratio_l3106_310624

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 10) →
  ((a + 4) / (1 - r) = 15) →
  r = 1/5 := by
sorry

end geometric_series_ratio_l3106_310624


namespace geometric_sequence_product_l3106_310627

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end geometric_sequence_product_l3106_310627


namespace nine_b_equals_eighteen_l3106_310668

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end nine_b_equals_eighteen_l3106_310668


namespace cos_negative_eleven_fourths_pi_l3106_310684

theorem cos_negative_eleven_fourths_pi :
  Real.cos (-11/4 * Real.pi) = -Real.sqrt 2 / 2 := by
  sorry

end cos_negative_eleven_fourths_pi_l3106_310684


namespace dealer_gross_profit_l3106_310629

-- Define the parameters
def purchase_price : ℝ := 150
def markup_rate : ℝ := 0.25
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Define the initial selling price
noncomputable def initial_selling_price : ℝ :=
  purchase_price / (1 - markup_rate)

-- Define the discounted price
noncomputable def discounted_price : ℝ :=
  initial_selling_price * (1 - discount_rate)

-- Define the final selling price (including tax)
noncomputable def final_selling_price : ℝ :=
  discounted_price * (1 + tax_rate)

-- Define the gross profit
noncomputable def gross_profit : ℝ :=
  final_selling_price - purchase_price

-- Theorem statement
theorem dealer_gross_profit :
  gross_profit = 19 := by sorry

end dealer_gross_profit_l3106_310629


namespace log21_not_calculable_l3106_310608

-- Define the given logarithm values
def log5 : ℝ := 0.6990
def log7 : ℝ := 0.8451

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := ∃ (a b : ℝ), x = a * log5 + b * log7

-- Theorem stating that log 21 cannot be calculated directly
theorem log21_not_calculable : ¬(can_calculate (Real.log 21)) :=
sorry

end log21_not_calculable_l3106_310608


namespace common_point_linear_functions_l3106_310660

/-- Three linear functions with a common point -/
theorem common_point_linear_functions
  (a b c d : ℝ)
  (h1 : a ≠ b)
  (h2 : ∃ (x y : ℝ), (y = a * x + a) ∧ (y = b * x + b) ∧ (y = c * x + d)) :
  c = d :=
sorry

end common_point_linear_functions_l3106_310660


namespace hundredth_term_is_397_l3106_310647

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 100th term of the specific arithmetic sequence -/
def hundredthTerm : ℝ := arithmeticSequenceTerm 1 4 100

theorem hundredth_term_is_397 : hundredthTerm = 397 := by sorry

end hundredth_term_is_397_l3106_310647


namespace fishing_catches_proof_l3106_310638

theorem fishing_catches_proof (a b c d : ℕ) : 
  a + b = 7 ∧ 
  a + c = 9 ∧ 
  a + d = 14 ∧ 
  b + c = 14 ∧ 
  b + d = 19 ∧ 
  c + d = 21 →
  (a = 1 ∧ b = 6 ∧ c = 8 ∧ d = 13) ∨
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 13) ∨
  (a = 6 ∧ b = 1 ∧ c = 8 ∧ d = 13) ∨
  (a = 6 ∧ b = 8 ∧ c = 1 ∧ d = 13) ∨
  (a = 8 ∧ b = 1 ∧ c = 6 ∧ d = 13) ∨
  (a = 8 ∧ b = 6 ∧ c = 1 ∧ d = 13) :=
by sorry

end fishing_catches_proof_l3106_310638


namespace angle_inequality_l3106_310662

open Real

theorem angle_inequality (θ : Real) (h1 : 3 * π / 4 < θ) (h2 : θ < π) :
  ∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * sin θ - x * (1 - x) + (1 - x)^2 * cos θ + 2 * x * (1 - x) * sqrt (cos θ * sin θ) > 0 :=
by sorry

end angle_inequality_l3106_310662


namespace pizza_consumption_proof_l3106_310645

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem pizza_consumption_proof :
  let initial_fraction : ℚ := 1/3
  let remaining_fraction : ℚ := 2/3
  let num_trips : ℕ := 6
  geometric_sum initial_fraction remaining_fraction num_trips = 364/729 := by
  sorry

end pizza_consumption_proof_l3106_310645


namespace sin_18_cos_36_equals_quarter_l3106_310652

theorem sin_18_cos_36_equals_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1 / 4 := by
  sorry

end sin_18_cos_36_equals_quarter_l3106_310652


namespace geometric_arithmetic_sequence_common_difference_l3106_310632

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end geometric_arithmetic_sequence_common_difference_l3106_310632


namespace associate_professor_pencils_l3106_310695

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (associate_prof_charts : ℕ) 
  (assistant_prof_pencils : ℕ) 
  (assistant_prof_charts : ℕ) :
  total_people = 6 →
  total_pencils = 7 →
  total_charts = 11 →
  associate_profs + assistant_profs = total_people →
  associate_prof_charts = 1 →
  assistant_prof_pencils = 1 →
  assistant_prof_charts = 2 →
  ∃ (associate_prof_pencils : ℕ),
    associate_prof_pencils * associate_profs + assistant_prof_pencils * assistant_profs = total_pencils ∧
    associate_prof_charts * associate_profs + assistant_prof_charts * assistant_profs = total_charts ∧
    associate_prof_pencils = 2 :=
by sorry

end associate_professor_pencils_l3106_310695


namespace justin_and_tim_same_game_l3106_310625

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem justin_and_tim_same_game :
  let total_combinations := Nat.choose total_players players_per_game
  let games_with_justin_and_tim := Nat.choose (total_players - 2) (players_per_game - 2)
  games_with_justin_and_tim = 210 := by
  sorry

end justin_and_tim_same_game_l3106_310625


namespace blue_sequin_rows_l3106_310644

/-- The number of sequins in each row of blue sequins -/
def blue_sequins_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of sequins in each row of purple sequins -/
def purple_sequins_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of sequins in each row of green sequins -/
def green_sequins_per_row : ℕ := 6

/-- The total number of sequins -/
def total_sequins : ℕ := 162

/-- Theorem: The number of rows of blue sequins is 6 -/
theorem blue_sequin_rows : 
  (total_sequins - (purple_rows * purple_sequins_per_row + green_rows * green_sequins_per_row)) / blue_sequins_per_row = 6 := by
  sorry

end blue_sequin_rows_l3106_310644


namespace total_material_calculation_l3106_310693

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := 0.16666666666666666

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.16666666666666666

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := concrete + bricks + stone

theorem total_material_calculation : total_material = 0.8333333333333332 := by
  sorry

end total_material_calculation_l3106_310693


namespace range_of_S_l3106_310606

theorem range_of_S (a b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 2 := by
  sorry

end range_of_S_l3106_310606


namespace exists_monochromatic_triplet_l3106_310663

/-- A coloring of natural numbers using two colors. -/
def Coloring := ℕ → Bool

/-- Predicate to check if three natural numbers form a valid triplet. -/
def ValidTriplet (x y z : ℕ) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x * y = z^2

/-- Theorem stating that for any two-color painting of natural numbers,
    there always exist three distinct natural numbers x, y, and z
    of the same color such that xy = z^2. -/
theorem exists_monochromatic_triplet (c : Coloring) :
  ∃ x y z : ℕ, ValidTriplet x y z ∧ c x = c y ∧ c y = c z :=
sorry

end exists_monochromatic_triplet_l3106_310663


namespace min_value_of_expression_equality_achieved_l3106_310635

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 ≥ 3032 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 3033 = 3032 :=
sorry

end min_value_of_expression_equality_achieved_l3106_310635


namespace same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l3106_310680

-- Define the quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (angle : ℝ) : Quadrant := sorry

-- Define the principle that angles with the same terminal side are in the same quadrant
theorem same_terminal_side_same_quadrant (angle1 angle2 : ℝ) :
  angle1 % 360 = angle2 % 360 → angle_quadrant angle1 = angle_quadrant angle2 := sorry

-- State the theorem
theorem angle_2010_in_third_quadrant :
  let angle_2010 : ℝ := 2010
  let angle_210 : ℝ := 210
  angle_2010 = 5 * 360 + angle_210 →
  angle_quadrant angle_210 = Quadrant.third →
  angle_quadrant angle_2010 = Quadrant.third := by
    sorry

end same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l3106_310680


namespace root_difference_l3106_310666

theorem root_difference (r s : ℝ) : 
  (∃ x, (1984 * x)^2 - 1983 * 1985 * x - 1 = 0 ∧ r = x ∧ 
    ∀ y, ((1984 * y)^2 - 1983 * 1985 * y - 1 = 0 → y ≤ r)) →
  (∃ x, 1983 * x^2 - 1984 * x + 1 = 0 ∧ s = x ∧ 
    ∀ y, (1983 * y^2 - 1984 * y + 1 = 0 → s ≤ y)) →
  r - s = 1982 / 1983 := by
sorry

end root_difference_l3106_310666


namespace water_bucket_ratio_l3106_310653

/-- Given two partially filled buckets of water, a and b, prove that the ratio of water in bucket b 
    to bucket a after transferring 6 liters from b to a is 1:2, given the initial conditions. -/
theorem water_bucket_ratio : 
  ∀ (a b : ℝ),
  a = 13.2 →
  a - 6 = (1/3) * (b + 6) →
  (b - 6) / (a + 6) = 1/2 := by
sorry

end water_bucket_ratio_l3106_310653


namespace optimal_planting_strategy_l3106_310616

/-- Represents the cost and planting details for a flower planting project --/
structure FlowerPlanting where
  costA : ℝ  -- Cost per pot of type A flowers
  costB : ℝ  -- Cost per pot of type B flowers
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℝ  -- Survival rate of type A flowers
  survivalRateB : ℝ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Calculates the total cost of planting flowers --/
def totalCost (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  fp.costA * potsA + fp.costB * (fp.totalPots - potsA)

/-- Calculates the number of pots to be replaced next year --/
def replacementPots (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  (1 - fp.survivalRateA) * potsA + (1 - fp.survivalRateB) * (fp.totalPots - potsA)

/-- Theorem stating the optimal planting strategy and minimum cost --/
theorem optimal_planting_strategy (fp : FlowerPlanting) 
    (h1 : 3 * fp.costA + 4 * fp.costB = 360)
    (h2 : 4 * fp.costA + 3 * fp.costB = 340)
    (h3 : fp.totalPots = 600)
    (h4 : fp.survivalRateA = 0.7)
    (h5 : fp.survivalRateB = 0.9)
    (h6 : fp.maxReplacement = 100) :
    ∃ (optimalA : ℕ), 
      optimalA = 200 ∧ 
      replacementPots fp optimalA ≤ fp.maxReplacement ∧
      ∀ (potsA : ℕ), replacementPots fp potsA ≤ fp.maxReplacement → 
        totalCost fp optimalA ≤ totalCost fp potsA ∧
      totalCost fp optimalA = 32000 := by
  sorry

end optimal_planting_strategy_l3106_310616


namespace roots_difference_abs_l3106_310600

theorem roots_difference_abs (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → 
  r₂^2 - 7*r₂ + 12 = 0 → 
  |r₁ - r₂| = 1 := by
sorry

end roots_difference_abs_l3106_310600


namespace product_from_lcm_gcd_l3106_310679

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 60) 
  (h_gcd : Nat.gcd a b = 5) : 
  a * b = 300 := by
  sorry

end product_from_lcm_gcd_l3106_310679


namespace arithmetic_sequence_problem_l3106_310602

theorem arithmetic_sequence_problem (n : ℕ) : 
  let a₁ : ℤ := 1
  let d : ℤ := 3
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 298 → n = 100 := by
sorry

end arithmetic_sequence_problem_l3106_310602


namespace min_value_expression_equality_achieved_l3106_310691

theorem min_value_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 ≥ 2021.75 :=
sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 5) * (x + 6) + 2024 = 2021.75 :=
sorry

end min_value_expression_equality_achieved_l3106_310691


namespace white_animals_more_than_cats_l3106_310687

theorem white_animals_more_than_cats (C W : ℕ) (h1 : C > 0) (h2 : W > 0) : W > C :=
  by
  -- Define the number of white cats (WC)
  have h3 : C / 3 = W / 6 :=
    -- Every third cat is white and every sixth white animal is a cat
    sorry
  
  -- Prove that W = 2C
  have h4 : W = 2 * C :=
    sorry

  -- Conclude that W > C
  sorry


end white_animals_more_than_cats_l3106_310687


namespace problem_solution_l3106_310672

-- Define the function f
def f (a b x : ℝ) := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_exists : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) : 
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
sorry


end problem_solution_l3106_310672


namespace algebraic_expression_value_l3106_310646

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - x - 1 = 0) :
  (2 * x + 3) * (2 * x - 3) - 2 * x * (1 - x) = -7 := by
  sorry

end algebraic_expression_value_l3106_310646


namespace expression_bound_l3106_310699

theorem expression_bound (x : ℝ) (h : x^2 - 7*x + 12 ≤ 0) : 
  40 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 54 := by
sorry

end expression_bound_l3106_310699


namespace circle_properties_l3106_310651

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define the center coordinates
def center : ℝ × ℝ := (-1, 2)

-- Define the radius
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_properties_l3106_310651


namespace n_squared_minus_one_divisible_by_24_l3106_310665

theorem n_squared_minus_one_divisible_by_24 (n : ℤ) 
  (h1 : ¬ 2 ∣ n) (h2 : ¬ 3 ∣ n) : 24 ∣ (n^2 - 1) := by
  sorry

end n_squared_minus_one_divisible_by_24_l3106_310665


namespace combined_experience_l3106_310676

def james_experience : ℕ := 20

def john_experience (james_current : ℕ) : ℕ := 2 * (james_current - 8) + 8

def mike_experience (john_current : ℕ) : ℕ := john_current - 16

theorem combined_experience :
  james_experience + john_experience james_experience + mike_experience (john_experience james_experience) = 68 :=
by sorry

end combined_experience_l3106_310676


namespace complex_number_quadrant_l3106_310609

theorem complex_number_quadrant (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 := by
  sorry

end complex_number_quadrant_l3106_310609


namespace club_members_after_four_years_l3106_310682

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then
    20
  else
    4 * club_members (k - 1) - 12

/-- The theorem stating the number of club members after 4 years -/
theorem club_members_after_four_years :
  club_members 4 = 4100 := by
  sorry

end club_members_after_four_years_l3106_310682


namespace function_inequality_l3106_310658

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x < 1) (h_f3 : f 3 = 4) :
  ∀ x, f (x + 1) < x + 2 ↔ x > 2 := by
  sorry

end function_inequality_l3106_310658


namespace prob_two_of_three_suits_l3106_310631

/-- The probability of drawing a specific suit from a standard 52-card deck -/
def prob_suit : ℚ := 1/4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of desired cards for each suit (hearts, diamonds, clubs) -/
def num_each_suit : ℕ := 2

/-- The probability of drawing exactly two hearts, two diamonds, and two clubs
    when drawing six cards with replacement from a standard 52-card deck -/
theorem prob_two_of_three_suits : 
  (num_draws.choose num_each_suit * num_draws.choose num_each_suit * num_draws.choose num_each_suit) *
  (prob_suit ^ num_each_suit * prob_suit ^ num_each_suit * prob_suit ^ num_each_suit) = 90/4096 := by
  sorry

end prob_two_of_three_suits_l3106_310631
