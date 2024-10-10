import Mathlib

namespace tile_count_equivalence_l286_28641

theorem tile_count_equivalence (area : ℝ) : 
  area = (0.3 : ℝ)^2 * 720 → area = (0.4 : ℝ)^2 * 405 := by
  sorry

end tile_count_equivalence_l286_28641


namespace tv_weight_difference_l286_28635

def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℚ := 4 / 1
def ounces_per_pound : ℕ := 16

theorem tv_weight_difference :
  let bill_area := bill_tv_length * bill_tv_width
  let bob_area := bob_tv_length * bob_tv_width
  let area_difference := max bill_area bob_area - min bill_area bob_area
  let weight_difference_oz := area_difference * weight_per_square_inch
  let weight_difference_lbs := weight_difference_oz / ounces_per_pound
  weight_difference_lbs = 150 := by sorry

end tv_weight_difference_l286_28635


namespace square_perimeter_l286_28615

theorem square_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) : 
  area_A = 121 →
  prob = 0.8677685950413223 →
  prob = (area_A - (perimeter_B / 4)^2) / area_A →
  perimeter_B = 16 := by
sorry

end square_perimeter_l286_28615


namespace cube_root_and_square_root_l286_28670

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the square root function
noncomputable def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x ∧ (y ≥ 0 ∨ y ≤ 0)}

theorem cube_root_and_square_root :
  (cubeRoot (1/8) = 1/2) ∧
  (squareRoot ((-6)^2) = {6, -6}) :=
sorry

end cube_root_and_square_root_l286_28670


namespace political_science_majors_l286_28680

/-- Represents the number of applicants who majored in political science -/
def P : ℕ := sorry

theorem political_science_majors :
  let total_applicants : ℕ := 40
  let high_gpa : ℕ := 20
  let not_ps_low_gpa : ℕ := 10
  let ps_high_gpa : ℕ := 5
  P = 15 := by sorry

end political_science_majors_l286_28680


namespace hiker_distance_l286_28653

theorem hiker_distance (north east south east2 : ℝ) 
  (h_north : north = 15)
  (h_east : east = 8)
  (h_south : south = 9)
  (h_east2 : east2 = 2) :
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end hiker_distance_l286_28653


namespace brittany_age_after_vacation_l286_28698

/-- Given that Rebecca is 25 years old and Brittany is 3 years older than Rebecca,
    prove that Brittany's age after returning from a 4-year vacation is 32 years old. -/
theorem brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  rebecca_age + age_difference + vacation_duration = 32 :=
by sorry

end brittany_age_after_vacation_l286_28698


namespace apple_distribution_l286_28626

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end apple_distribution_l286_28626


namespace sophies_perceived_height_l286_28621

/-- Calculates the perceived height in centimeters when doubled in a mirror reflection. -/
def perceivedHeightCm (actualHeightInches : ℝ) (conversionRate : ℝ) : ℝ :=
  2 * actualHeightInches * conversionRate

/-- Theorem stating that Sophie's perceived height in the mirror is 250.0 cm. -/
theorem sophies_perceived_height :
  let actualHeight : ℝ := 50
  let conversionRate : ℝ := 2.50
  perceivedHeightCm actualHeight conversionRate = 250.0 := by
  sorry

end sophies_perceived_height_l286_28621


namespace sum_product_equality_l286_28662

theorem sum_product_equality : (153 + 39 + 27 + 21) * 2 = 480 := by
  sorry

end sum_product_equality_l286_28662


namespace gold_families_count_l286_28693

def fundraiser (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) : Prop :=
  let bronze_donation := 25
  let silver_donation := 50
  let gold_donation := 100
  let total_goal := 750
  let final_day_goal := 50
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation = 
  total_goal - final_day_goal

theorem gold_families_count : 
  ∃! gold_families : ℕ, fundraiser 10 7 gold_families :=
sorry

end gold_families_count_l286_28693


namespace no_solution_implies_n_greater_than_one_l286_28630

theorem no_solution_implies_n_greater_than_one (n : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 1 ∧ x ≥ n)) → n > 1 := by
  sorry

end no_solution_implies_n_greater_than_one_l286_28630


namespace laura_change_l286_28613

def change_calculation (pants_cost : ℕ) (pants_count : ℕ) (shirt_cost : ℕ) (shirt_count : ℕ) (amount_given : ℕ) : ℕ :=
  amount_given - (pants_cost * pants_count + shirt_cost * shirt_count)

theorem laura_change : change_calculation 54 2 33 4 250 = 10 := by
  sorry

end laura_change_l286_28613


namespace sum_parity_when_sum_of_squares_even_l286_28660

theorem sum_parity_when_sum_of_squares_even (m n : ℤ) : 
  Even (m^2 + n^2) → Even (m + n) :=
by sorry

end sum_parity_when_sum_of_squares_even_l286_28660


namespace min_value_a_l286_28655

theorem min_value_a (a b : ℕ) (h : 1176 * a = b^3) : 63 ≤ a := by
  sorry

end min_value_a_l286_28655


namespace one_solution_r_product_l286_28603

theorem one_solution_r_product (r : ℝ) : 
  (∃! x : ℝ, (1 / (2 * x) = (r - x) / 9)) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ (r₁ * r₂ = -18) :=
sorry

end one_solution_r_product_l286_28603


namespace two_digit_square_last_two_digits_l286_28632

theorem two_digit_square_last_two_digits (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ x^2 % 100 = x % 100 ↔ x = 25 ∨ x = 76 := by
  sorry

end two_digit_square_last_two_digits_l286_28632


namespace triangle_side_length_expression_l286_28679

/-- For any triangle with side lengths a, b, and c, |a+b-c|-|a-b-c| = 2a-2c -/
theorem triangle_side_length_expression (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a + b - c| - |a - b - c| = 2*a - 2*c := by
  sorry

end triangle_side_length_expression_l286_28679


namespace total_pieces_is_4000_l286_28619

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all three puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

/-- Theorem stating that the total number of pieces in all three puzzles is 4000 -/
theorem total_pieces_is_4000 : total_pieces = 4000 := by
  sorry

end total_pieces_is_4000_l286_28619


namespace store_revenue_l286_28677

theorem store_revenue (N D J : ℝ) 
  (h1 : N = (3/5) * D) 
  (h2 : D = (20/7) * ((N + J) / 2)) : 
  J = (1/6) * N := by
sorry

end store_revenue_l286_28677


namespace catia_speed_theorem_l286_28633

/-- The speed at which Cátia should travel to reach home at 5:00 PM -/
def required_speed : ℝ := 12

/-- The time Cátia leaves school every day -/
def departure_time : ℝ := 3.75 -- 3:45 PM in decimal hours

/-- The distance from school to Cátia's home -/
def distance : ℝ := 15

/-- Arrival time when traveling at 20 km/h -/
def arrival_time_fast : ℝ := 4.5 -- 4:30 PM in decimal hours

/-- Arrival time when traveling at 10 km/h -/
def arrival_time_slow : ℝ := 5.25 -- 5:15 PM in decimal hours

/-- The desired arrival time -/
def desired_arrival_time : ℝ := 5 -- 5:00 PM in decimal hours

theorem catia_speed_theorem :
  (distance / (arrival_time_fast - departure_time) = 20) →
  (distance / (arrival_time_slow - departure_time) = 10) →
  (distance / (desired_arrival_time - departure_time) = required_speed) :=
by sorry

end catia_speed_theorem_l286_28633


namespace sum_of_fifth_powers_zero_l286_28639

theorem sum_of_fifth_powers_zero (a b c : ℚ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_nonzero : a^3 + b^3 + c^3 ≠ 0) : 
  a^5 + b^5 + c^5 = 0 := by
  sorry

end sum_of_fifth_powers_zero_l286_28639


namespace right_triangle_special_angles_l286_28690

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- leg 1
  b : ℝ  -- leg 2
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_condition : h = c / 4  -- altitude is 4 times smaller than hypotenuse

-- Define the theorem
theorem right_triangle_special_angles (t : RightTriangle) :
  let angle1 := Real.arcsin (t.h / t.c)
  let angle2 := Real.arcsin (t.a / t.c)
  (angle1 = 15 * π / 180 ∧ angle2 = 75 * π / 180) ∨
  (angle1 = 75 * π / 180 ∧ angle2 = 15 * π / 180) :=
sorry

end right_triangle_special_angles_l286_28690


namespace sin_alpha_plus_pi_half_l286_28667

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the angle α
variable (α : ℝ)

-- State the theorem
theorem sin_alpha_plus_pi_half (h : ∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) : 
  Real.sin (α + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end sin_alpha_plus_pi_half_l286_28667


namespace smallest_special_integer_l286_28694

theorem smallest_special_integer (N : ℕ) : N = 793 ↔ 
  N > 1 ∧
  (∀ M : ℕ, M > 1 → 
    (M ≡ 1 [ZMOD 8] ∧
     M ≡ 1 [ZMOD 9] ∧
     (∃ k : ℕ, 8^k ≤ M ∧ M < 2 * 8^k) ∧
     (∃ m : ℕ, 9^m ≤ M ∧ M < 2 * 9^m)) →
    N ≤ M) ∧
  N ≡ 1 [ZMOD 8] ∧
  N ≡ 1 [ZMOD 9] ∧
  (∃ k : ℕ, 8^k ≤ N ∧ N < 2 * 8^k) ∧
  (∃ m : ℕ, 9^m ≤ N ∧ N < 2 * 9^m) :=
by sorry

end smallest_special_integer_l286_28694


namespace circle_radius_from_circumference_l286_28636

/-- The radius of a circle with circumference 100π cm is 50 cm. -/
theorem circle_radius_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 100 * π → r = 50 :=
by sorry

end circle_radius_from_circumference_l286_28636


namespace coefficient_sum_is_five_sixths_l286_28697

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The property that f(x) - f(x-2) = (2x-1)^2 for all x -/
def SatisfiesEquation (f : PolynomialFunction) : Prop :=
  ∀ x, f x - f (x - 2) = (2 * x - 1)^2

/-- The coefficient of x^2 in a polynomial function -/
def CoefficientOfXSquared (f : PolynomialFunction) : ℝ := sorry

/-- The coefficient of x in a polynomial function -/
def CoefficientOfX (f : PolynomialFunction) : ℝ := sorry

theorem coefficient_sum_is_five_sixths (f : PolynomialFunction) 
  (h : SatisfiesEquation f) : 
  CoefficientOfXSquared f + CoefficientOfX f = 5/6 := by
  sorry

end coefficient_sum_is_five_sixths_l286_28697


namespace max_visible_cubes_12_10_9_l286_28607

/-- Represents a rectangular block formed by unit cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point for a given block -/
def max_visible_cubes (b : Block) : ℕ :=
  b.length * b.width + b.width * b.height + b.length * b.height -
  (b.length + b.width + b.height) + 1

/-- The theorem stating that for a 12 × 10 × 9 block, the maximum number of visible unit cubes is 288 -/
theorem max_visible_cubes_12_10_9 :
  max_visible_cubes ⟨12, 10, 9⟩ = 288 := by
  sorry

#eval max_visible_cubes ⟨12, 10, 9⟩

end max_visible_cubes_12_10_9_l286_28607


namespace systematic_sampling_example_l286_28681

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (startGroup : ℕ) (startNum : ℕ) (targetGroup : ℕ) : ℕ :=
  startNum + (targetGroup - startGroup) * (totalItems / numGroups)

/-- Theorem: In systematic sampling of 200 items into 40 groups, 
    if the 5th group draws 24, then the 9th group draws 44 -/
theorem systematic_sampling_example :
  systematicSample 200 40 5 24 9 = 44 := by
  sorry

#eval systematicSample 200 40 5 24 9

end systematic_sampling_example_l286_28681


namespace parallel_vectors_y_value_l286_28629

/-- Given two parallel vectors a and b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
  (h1 : a = (2, 3)) 
  (h2 : b = (4, -1 + y)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  y = 7 := by sorry

end parallel_vectors_y_value_l286_28629


namespace factorization_correctness_l286_28652

theorem factorization_correctness (x y : ℝ) : 
  (∃! n : ℕ, n = (if x^3 + 2*x*y + x = x*(x^2 + 2*y) then 1 else 0) + 
             (if x^2 + 4*x + 4 = (x + 2)^2 then 1 else 0) + 
             (if -x^2 + y^2 = (x + y)*(x - y) then 1 else 0) ∧ 
             n = 1) := by sorry

end factorization_correctness_l286_28652


namespace alcohol_solution_percentage_l286_28657

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 6.5 →
  added_water = 3.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 17 := by
sorry

end alcohol_solution_percentage_l286_28657


namespace polynomial_divisibility_l286_28671

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
sorry

end polynomial_divisibility_l286_28671


namespace train_speed_is_6_l286_28616

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
def train_speed (length : Float) (time : Float) : Float :=
  (length / time) * 3.6

/-- Theorem: The speed of the train is 6 km/hr -/
theorem train_speed_is_6 :
  let length : Float := 3.3333333333333335
  let time : Float := 2
  train_speed length time = 6 := by
  sorry

end train_speed_is_6_l286_28616


namespace periodic_product_quotient_iff_commensurable_l286_28665

theorem periodic_product_quotient_iff_commensurable 
  (f g : ℝ → ℝ) (T₁ T₂ : ℝ) 
  (hf : ∀ x, f (x + T₁) = f x) 
  (hg : ∀ x, g (x + T₂) = g x)
  (hpos_f : ∀ x, f x > 0)
  (hpos_g : ∀ x, g x > 0) :
  (∃ T, ∀ x, (f x * g x) = (f (x + T) * g (x + T)) ∧ 
            (f x / g x) = (f (x + T) / g (x + T))) ↔ 
  (∃ m n : ℤ, m ≠ 0 ∧ n ≠ 0 ∧ m * T₁ = n * T₂) :=
sorry

end periodic_product_quotient_iff_commensurable_l286_28665


namespace air_conditioner_power_consumption_l286_28669

/-- Power consumption of three air conditioners over specified periods -/
theorem air_conditioner_power_consumption 
  (power_A : Real) (hours_A : Real) (days_A : Real)
  (power_B : Real) (hours_B : Real) (days_B : Real)
  (power_C : Real) (hours_C : Real) (days_C : Real) :
  power_A = 7.2 →
  power_B = 9.6 →
  power_C = 12 →
  hours_A = 6 →
  hours_B = 4 →
  hours_C = 3 →
  days_A = 5 →
  days_B = 7 →
  days_C = 10 →
  (power_A / 8 * hours_A * days_A) +
  (power_B / 10 * hours_B * days_B) +
  (power_C / 12 * hours_C * days_C) = 83.88 := by
  sorry

#eval (7.2 / 8 * 6 * 5) + (9.6 / 10 * 4 * 7) + (12 / 12 * 3 * 10)

end air_conditioner_power_consumption_l286_28669


namespace arithmetic_sequence_count_3_4_2012_l286_28645

def arithmetic_sequence_count (a₁ : ℕ) (d : ℕ) (max : ℕ) : ℕ :=
  (max - a₁) / d + 1

theorem arithmetic_sequence_count_3_4_2012 :
  arithmetic_sequence_count 3 4 2012 = 502 := by
  sorry

end arithmetic_sequence_count_3_4_2012_l286_28645


namespace two_different_color_chips_probability_l286_28682

/-- Represents the colors of chips in the bag -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag of chips -/
structure ChipBag where
  blue : Nat
  red : Nat
  yellow : Nat

/-- Calculates the total number of chips in the bag -/
def ChipBag.total (bag : ChipBag) : Nat :=
  bag.blue + bag.red + bag.yellow

/-- Calculates the probability of drawing a specific color -/
def drawProbability (bag : ChipBag) (color : ChipColor) : Rat :=
  match color with
  | ChipColor.Blue => bag.blue / bag.total
  | ChipColor.Red => bag.red / bag.total
  | ChipColor.Yellow => bag.yellow / bag.total

/-- Calculates the probability of drawing two different colored chips -/
def differentColorProbability (bag : ChipBag) : Rat :=
  let blueFirst := drawProbability bag ChipColor.Blue * (1 - drawProbability bag ChipColor.Blue / 2)
  let redFirst := drawProbability bag ChipColor.Red * (1 - drawProbability bag ChipColor.Red)
  let yellowFirst := drawProbability bag ChipColor.Yellow * (1 - drawProbability bag ChipColor.Yellow)
  blueFirst + redFirst + yellowFirst

theorem two_different_color_chips_probability :
  let initialBag : ChipBag := { blue := 7, red := 5, yellow := 4 }
  differentColorProbability initialBag = 381 / 512 := by
  sorry


end two_different_color_chips_probability_l286_28682


namespace cupboard_has_35_slots_l286_28659

/-- Represents a cupboard with shelves and slots -/
structure Cupboard where
  shelves : ℕ
  slots_per_shelf : ℕ

/-- Represents the position of a plate in the cupboard -/
structure PlatePosition where
  shelf_from_top : ℕ
  shelf_from_bottom : ℕ
  slot_from_left : ℕ
  slot_from_right : ℕ

/-- Calculates the total number of slots in a cupboard -/
def total_slots (c : Cupboard) : ℕ := c.shelves * c.slots_per_shelf

/-- Theorem: Given the position of a plate, the cupboard has 35 slots -/
theorem cupboard_has_35_slots (pos : PlatePosition) 
  (h1 : pos.shelf_from_top = 2)
  (h2 : pos.shelf_from_bottom = 4)
  (h3 : pos.slot_from_left = 1)
  (h4 : pos.slot_from_right = 7) :
  ∃ c : Cupboard, total_slots c = 35 := by
  sorry

end cupboard_has_35_slots_l286_28659


namespace four_point_lines_l286_28650

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Count the number of distinct lines through four points -/
def count_lines (p1 p2 p3 p4 : Point) : ℕ :=
  sorry

/-- Theorem: The number of distinct lines through four points is either 1, 4, or 6 -/
theorem four_point_lines (p1 p2 p3 p4 : Point) :
  count_lines p1 p2 p3 p4 = 1 ∨ count_lines p1 p2 p3 p4 = 4 ∨ count_lines p1 p2 p3 p4 = 6 :=
by sorry

end four_point_lines_l286_28650


namespace arithmetic_sequence_difference_l286_28643

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

-- Define the theorem
theorem arithmetic_sequence_difference
  (d : ℝ) (m n : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence d 2 * arithmetic_sequence d 6 = (arithmetic_sequence d 4 - 2)^2)
  (h3 : m > n)
  (h4 : m - n = 10) :
  arithmetic_sequence d m - arithmetic_sequence d n = 30 := by
sorry

end arithmetic_sequence_difference_l286_28643


namespace eleven_operations_to_equal_l286_28688

/-- The number of operations required to make two numbers equal --/
def operations_to_equal (a b : ℕ) (sub_a add_b : ℕ) : ℕ :=
  (a - b) / (sub_a + add_b)

/-- Theorem stating that it takes 11 operations to make the numbers equal --/
theorem eleven_operations_to_equal :
  operations_to_equal 365 24 19 12 = 11 := by
  sorry

end eleven_operations_to_equal_l286_28688


namespace shannons_to_olivias_scoops_ratio_l286_28678

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of scoops Lucas, Danny, and Connor want in total -/
def lucas_danny_connor_scoops : ℕ := 6

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents the number of scoops left -/
def scoops_left : ℕ := 16

/-- Theorem stating that the ratio of Shannon's scoops to Olivia's scoops is 2:1 -/
theorem shannons_to_olivias_scoops_ratio : 
  ∃ (shannons_scoops : ℕ), 
    shannons_scoops = marys_cartons * scoops_per_carton - 
      (ethans_scoops + lucas_danny_connor_scoops + olivias_scoops + scoops_left) ∧
    shannons_scoops = 2 * olivias_scoops :=
by sorry

end shannons_to_olivias_scoops_ratio_l286_28678


namespace hyperbola_equation_l286_28668

/-- Represents a hyperbola with foci on the y-axis -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The length of the semi-major axis -/
  a : ℝ
  /-- The length of the semi-minor axis -/
  b : ℝ
  /-- One focus lies on the line 5x-2y+20=0 -/
  focus_on_line : c = 10
  /-- The ratio of c to a is 5/3 -/
  c_a_ratio : c / a = 5 / 3
  /-- Relationship between a, b, and c -/
  abc_relation : b^2 = c^2 - a^2

/-- The equation of the hyperbola is x²/64 - y²/36 = -1 -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ x y : ℝ, (x^2 / 64 - y^2 / 36 = -1) ↔ h.b^2 * y^2 - h.a^2 * x^2 = h.a^2 * h.b^2 :=
by sorry

end hyperbola_equation_l286_28668


namespace triangle_angle_measure_l286_28620

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin A + b * Real.sin B - c * Real.sin C) / (a * Real.sin B) = 2 * Real.sqrt 3 * Real.sin C →
  C = π / 6 :=
sorry

end triangle_angle_measure_l286_28620


namespace max_base_seven_digit_sum_l286_28656

/-- Represents a positive integer in base 7 --/
def BaseSevenDigits := List Nat

/-- Converts a positive integer to its base 7 representation --/
def toBaseSeven (n : Nat) : BaseSevenDigits :=
  sorry

/-- Calculates the sum of digits in a base 7 representation --/
def sumBaseSevenDigits (digits : BaseSevenDigits) : Nat :=
  sorry

/-- Checks if a base 7 representation is valid (all digits < 7) --/
def isValidBaseSeven (digits : BaseSevenDigits) : Prop :=
  sorry

/-- Converts a base 7 representation back to a natural number --/
def fromBaseSeven (digits : BaseSevenDigits) : Nat :=
  sorry

/-- The main theorem --/
theorem max_base_seven_digit_sum :
  ∀ n : Nat, n > 0 → n < 3000 →
    ∃ (max : Nat),
      max = 24 ∧
      sumBaseSevenDigits (toBaseSeven n) ≤ max ∧
      (∀ m : Nat, m > 0 → m < 3000 →
        sumBaseSevenDigits (toBaseSeven m) ≤ max) :=
by sorry

end max_base_seven_digit_sum_l286_28656


namespace loan_amount_proof_l286_28649

/-- The annual interest rate A charges B -/
def interest_rate_A : ℝ := 0.10

/-- The annual interest rate B charges C -/
def interest_rate_B : ℝ := 0.115

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's gain over the loan period -/
def gain : ℝ := 1125

/-- The amount lent by A to B -/
def amount : ℝ := 25000

theorem loan_amount_proof :
  gain = (interest_rate_B - interest_rate_A) * years * amount := by sorry

end loan_amount_proof_l286_28649


namespace evaluate_f_l286_28676

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end evaluate_f_l286_28676


namespace supplier_A_lower_variance_l286_28685

-- Define the purity data for Supplier A
def purity_A : List Nat := [72, 73, 74, 74, 74, 74, 74, 75, 75, 75, 76, 76, 76, 78, 79]

-- Define the purity data for Supplier B
def purity_B : List Nat := [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

-- Define the statistical measures for Supplier A
def mean_A : Nat := 75
def median_A : Nat := 75
def mode_A : Nat := 74
def variance_A : Float := 3.7

-- Define the statistical measures for Supplier B
def mean_B : Nat := 75
def median_B : Nat := 75
def mode_B : Nat := 75
def variance_B : Float := 6.0

-- Theorem statement
theorem supplier_A_lower_variance :
  variance_A < variance_B ∧
  List.length purity_A = 15 ∧
  List.length purity_B = 15 ∧
  mean_A = mean_B ∧
  median_A = median_B :=
sorry

end supplier_A_lower_variance_l286_28685


namespace parabola_line_intersection_l286_28654

/-- 
A line x = m intersects a parabola x = -3y^2 - 4y + 7 at exactly one point 
if and only if m = 25/3
-/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25/3 := by
  sorry

end parabola_line_intersection_l286_28654


namespace pencil_cost_l286_28600

/-- The cost of a pencil given total money and number of pencils that can be bought --/
theorem pencil_cost (total_money : ℚ) (num_pencils : ℕ) (h : total_money = 50 ∧ num_pencils = 10) : 
  total_money / num_pencils = 5 := by
  sorry

#check pencil_cost

end pencil_cost_l286_28600


namespace florist_roses_theorem_l286_28672

/-- Represents the number of roses picked in the first picking -/
def first_picking : ℝ := 16.0

theorem florist_roses_theorem (initial : ℝ) (second_picking : ℝ) (final_total : ℝ) :
  initial = 37.0 →
  second_picking = 19.0 →
  final_total = 72 →
  initial + first_picking + second_picking = final_total :=
by
  sorry

#check florist_roses_theorem

end florist_roses_theorem_l286_28672


namespace cubic_equation_solutions_l286_28614

/-- Given a cubic polynomial with three distinct real roots, 
    the equation formed by its product with its derivative 
    equals the square of its derivative has exactly two distinct real solutions. -/
theorem cubic_equation_solutions (a b c d : ℝ) (h : ∃ α β γ : ℝ, α < β ∧ β < γ ∧ 
  ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  ∃! (s t : ℝ), s < t ∧ 
    ∀ x, 4 * (a * x^3 + b * x^2 + c * x + d) * (3 * a * x + b) = (3 * a * x^2 + 2 * b * x + c)^2 
    ↔ x = s ∨ x = t :=
by sorry

end cubic_equation_solutions_l286_28614


namespace nabla_example_l286_28692

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end nabla_example_l286_28692


namespace two_point_form_always_valid_two_point_form_works_for_vertical_lines_l286_28602

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line passing through two points --/
def line_equation (p1 p2 : Point) (x y : ℝ) : Prop :=
  (y - p1.y) * (p2.x - p1.x) = (x - p1.x) * (p2.y - p1.y)

/-- Theorem: The two-point form of a line equation is always valid --/
theorem two_point_form_always_valid (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃ (l : Line), ∀ (x y : ℝ), (y = l.slope * x + l.intercept) ↔ line_equation p1 p2 x y :=
sorry

/-- Corollary: The two-point form works even for vertical lines --/
theorem two_point_form_works_for_vertical_lines (p1 p2 : Point) (h : p1.x = p2.x) (h' : p1 ≠ p2) :
  ∀ (y : ℝ), ∃ (x : ℝ), line_equation p1 p2 x y :=
sorry

end two_point_form_always_valid_two_point_form_works_for_vertical_lines_l286_28602


namespace cubic_polynomial_integer_root_l286_28642

theorem cubic_polynomial_integer_root 
  (d e : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + d*x + e = 0 ∧ x = 2 - Real.sqrt 5)
  (h2 : ∃ n : ℤ, n^3 + d*n + e = 0) :
  ∃ n : ℤ, n^3 + d*n + e = 0 ∧ n = -4 :=
sorry

end cubic_polynomial_integer_root_l286_28642


namespace solution_set_f_positive_range_of_m_l286_28686

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > 1/3} := by sorry

-- Define the theorem for part II
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, |m + 1| ≥ f x + 3*|x - 2|) ↔ m ≤ -6 ∨ m ≥ 4 := by sorry

end solution_set_f_positive_range_of_m_l286_28686


namespace cross_country_winning_scores_l286_28647

/-- Represents a cross-country meet between two teams -/
structure CrossCountryMeet where
  /-- Total number of runners -/
  total_runners : Nat
  /-- Number of runners per team -/
  runners_per_team : Nat
  /-- Minimum possible team score -/
  min_score : Nat
  /-- Maximum possible team score -/
  max_score : Nat

/-- Calculates the number of different winning scores possible in a cross-country meet -/
def count_winning_scores (meet : CrossCountryMeet) : Nat :=
  sorry

/-- Theorem stating the number of different winning scores in the given cross-country meet -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.total_runners = 10 ∧
    meet.runners_per_team = 5 ∧
    meet.min_score = 15 ∧
    meet.max_score = 40 ∧
    count_winning_scores meet = 13 :=
  sorry

end cross_country_winning_scores_l286_28647


namespace sphere_volume_reduction_line_tangent_to_circle_l286_28664

-- Proposition 1
theorem sphere_volume_reduction (r : ℝ) (V : ℝ → ℝ) (h : V r = (4/3) * π * r^3) :
  V (r/2) = (1/8) * V r := by sorry

-- Proposition 3
theorem line_tangent_to_circle :
  let d := (1 : ℝ) / Real.sqrt 2
  (d = Real.sqrt ((1/2) : ℝ)) ∧ 
  (∀ x y : ℝ, x + y + 1 = 0 → x^2 + y^2 = 1/2 → 
    (x^2 + y^2 = d^2 ∨ x^2 + y^2 > d^2)) := by sorry

end sphere_volume_reduction_line_tangent_to_circle_l286_28664


namespace trigonometric_equation_solutions_l286_28624

theorem trigonometric_equation_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ solutions ↔
      (c ∈ Set.Icc 0 (2 * Real.pi) ∧
       ∀ x : ℝ, 2 * Real.sin (3 * x - Real.pi / 3) = a * Real.sin (b * x + c))) ∧
    Finset.card solutions = 4 := by
  sorry

end trigonometric_equation_solutions_l286_28624


namespace baseball_cards_problem_l286_28689

theorem baseball_cards_problem (X : ℚ) : 3 * (X - (X + 1) / 2 - 1) = 18 ↔ X = 15 := by
  sorry

end baseball_cards_problem_l286_28689


namespace product_increase_thirteen_times_l286_28618

theorem product_increase_thirteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3)) / 
    (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ : ℚ) = 13 :=
by sorry

end product_increase_thirteen_times_l286_28618


namespace intersection_distance_l286_28687

/-- Two lines intersecting at 60 degrees --/
structure IntersectingLines :=
  (angle : ℝ)
  (h_angle : angle = 60)

/-- Points on the intersecting lines --/
structure PointsOnLines (l : IntersectingLines) :=
  (A B : ℝ × ℝ)
  (dist_initial : ℝ)
  (dist_after_move : ℝ)
  (move_distance : ℝ)
  (h_initial_dist : dist_initial = 31)
  (h_after_move_dist : dist_after_move = 21)
  (h_move_distance : move_distance = 20)

/-- The theorem to be proved --/
theorem intersection_distance (l : IntersectingLines) (p : PointsOnLines l) :
  ∃ (dist_A dist_B : ℝ),
    dist_A = 35 ∧ dist_B = 24 ∧
    (dist_A - p.move_distance)^2 + dist_B^2 = p.dist_initial^2 ∧
    dist_A^2 + dist_B^2 = p.dist_after_move^2 + p.move_distance^2 :=
sorry

end intersection_distance_l286_28687


namespace unique_parallel_line_l286_28605

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (on_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (M : Point) :
  parallel α β → 
  lies_on a α → 
  on_plane M β → 
  ∃! l : Line, passes_through l M ∧ line_parallel l a :=
sorry

end unique_parallel_line_l286_28605


namespace sunnydale_walk_home_fraction_l286_28622

/-- The fraction of students who walk home at Sunnydale Middle School -/
theorem sunnydale_walk_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/5
  let bike_fraction : ℚ := 1/8
  let walk_fraction : ℚ := 1 - (bus_fraction + auto_fraction + bike_fraction)
  walk_fraction = 41/120 := by
  sorry

end sunnydale_walk_home_fraction_l286_28622


namespace brothers_ages_l286_28627

theorem brothers_ages (x y : ℕ) : 
  x + y = 16 → 
  2 * (x + 4) = y + 4 → 
  ∃ (younger older : ℕ), younger = x ∧ older = y ∧ younger < older :=
by
  sorry

end brothers_ages_l286_28627


namespace negation_of_universal_proposition_l286_28691

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 > x^2) ↔ (∃ x : ℝ, x^3 ≤ x^2) :=
by sorry

end negation_of_universal_proposition_l286_28691


namespace total_clothing_items_l286_28638

theorem total_clothing_items (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) 
  (h1 : short_sleeve = 7)
  (h2 : long_sleeve = 9)
  (h3 : pants = 4)
  (h4 : jackets = 2) :
  short_sleeve + long_sleeve + pants + jackets = 22 := by
  sorry

end total_clothing_items_l286_28638


namespace square_area_ratio_l286_28609

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := 4 * s₂
  (s₁ * s₁) / (s₂ * s₂) = 16 := by
  sorry

end square_area_ratio_l286_28609


namespace sin_2alpha_minus_pi_3_l286_28617

theorem sin_2alpha_minus_pi_3 (α : ℝ) (h : Real.cos (α + π / 12) = -3 / 4) :
  Real.sin (2 * α - π / 3) = -1 / 8 := by
  sorry

end sin_2alpha_minus_pi_3_l286_28617


namespace litter_patrol_theorem_l286_28651

/-- The total number of litter items picked up by the Litter Patrol -/
def total_litter : ℕ := 40

/-- The number of non-miscellaneous items (glass bottles + aluminum cans + plastic bags) -/
def non_misc_items : ℕ := 30

/-- The percentage of non-miscellaneous items in the total litter -/
def non_misc_percentage : ℚ := 3/4

theorem litter_patrol_theorem :
  (non_misc_items : ℚ) / non_misc_percentage = total_litter := by sorry

end litter_patrol_theorem_l286_28651


namespace mary_sticker_problem_l286_28684

/-- Given the conditions about Mary's stickers, prove the total number of students in the class. -/
theorem mary_sticker_problem (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) :
  total_stickers = 250 →
  friends = 10 →
  stickers_per_friend = 15 →
  stickers_per_other = 5 →
  leftover_stickers = 25 →
  ∃ (total_students : ℕ), total_students = 26 ∧ 
    total_stickers = friends * stickers_per_friend + 
    (total_students - friends - 1) * stickers_per_other + leftover_stickers :=
by sorry

end mary_sticker_problem_l286_28684


namespace chocolate_chip_per_recipe_l286_28610

/-- Given that 23 recipes require 46 cups of chocolate chips in total,
    prove that the number of cups of chocolate chips needed for one recipe is 2. -/
theorem chocolate_chip_per_recipe :
  let total_recipes : ℕ := 23
  let total_chips : ℕ := 46
  (total_chips / total_recipes : ℚ) = 2 := by sorry

end chocolate_chip_per_recipe_l286_28610


namespace candy_probability_l286_28674

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def same_color_probability : ℚ :=
  678 / 1735

theorem candy_probability :
  let first_pick := 2
  let second_pick := 2
  let remaining_candies := total_candies - first_pick
  (red_candies.choose first_pick * (red_candies - first_pick).choose second_pick +
   blue_candies.choose first_pick * (blue_candies - first_pick).choose second_pick +
   (red_candies.choose 1 * blue_candies.choose 1) * 
   ((red_candies - 1).choose 1 * (blue_candies - 1).choose 1)) /
  (total_candies.choose first_pick * remaining_candies.choose second_pick) =
  same_color_probability := by
sorry

end candy_probability_l286_28674


namespace solution_of_cubic_system_l286_28625

theorem solution_of_cubic_system :
  ∀ x y : ℝ, x + y = 1 ∧ x^3 + y^3 = 19 →
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
sorry

end solution_of_cubic_system_l286_28625


namespace fraction_of_fraction_of_fraction_l286_28601

theorem fraction_of_fraction_of_fraction (n : ℚ) : n = 72 → (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * n = 2 := by
  sorry

end fraction_of_fraction_of_fraction_l286_28601


namespace largest_product_sum_1976_l286_28606

theorem largest_product_sum_1976 (n : ℕ) (h : n > 0) :
  (∃ (factors : List ℕ), factors.sum = 1976 ∧ factors.prod = n) →
  n ≤ 2 * 3^658 := by
sorry

end largest_product_sum_1976_l286_28606


namespace quadratic_function_property_l286_28675

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℤ) 
  (h1 : QuadraticFunction a b c 2 = 5)
  (h2 : ∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c 1)
  (h3 : QuadraticFunction a b c 1 = 3) :
  a - b + c = 11 := by
  sorry

end quadratic_function_property_l286_28675


namespace blue_candy_count_l286_28673

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) :
  blue = 3264 := by
  sorry

end blue_candy_count_l286_28673


namespace mushroom_collection_l286_28695

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem mushroom_collection :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ sum_of_digits n = 14 ∧ n % 50 = 0 :=
by
  -- The proof would go here
  sorry

#eval sum_of_digits 950  -- Should output 14
#eval 950 % 50           -- Should output 0

end mushroom_collection_l286_28695


namespace sequence_problem_l286_28631

theorem sequence_problem (x : ℕ → ℝ) 
  (h_distinct : ∀ n m, n ≥ 2 → m ≥ 2 → n ≠ m → x n ≠ x m)
  (h_relation : ∀ n, n ≥ 2 → x n = (x (n-1) + 398 * x n + x (n+1)) / 400) :
  Real.sqrt ((x 2023 - x 2) / 2021 * (2022 / (x 2023 - x 1))) + 2021 = 2022 := by
  sorry

end sequence_problem_l286_28631


namespace special_permutation_exists_l286_28604

/-- A permutation of numbers from 1 to 2^n satisfying the special property -/
def SpecialPermutation (n : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list satisfies the special property -/
def SatisfiesProperty (lst : List ℕ) : Prop :=
  ∀ i j, i < j → i < lst.length → j < lst.length →
    ∀ k, i < k ∧ k < j →
      (lst.get ⟨i, sorry⟩ + lst.get ⟨j, sorry⟩) / 2 ≠ lst.get ⟨k, sorry⟩

/-- Theorem stating that for any n, there exists a permutation of numbers
    from 1 to 2^n satisfying the special property -/
theorem special_permutation_exists (n : ℕ) :
  ∃ (perm : List ℕ), perm.length = 2^n ∧
    (∀ i, i ∈ perm ↔ 1 ≤ i ∧ i ≤ 2^n) ∧
    SatisfiesProperty perm :=
  sorry

end special_permutation_exists_l286_28604


namespace good_goods_sufficient_condition_l286_28699

-- Define propositions
variable (G : Prop) -- G represents "goods are good"
variable (C : Prop) -- C represents "goods are cheap"

-- Define the statement "Good goods are not cheap"
def good_goods_not_cheap : Prop := G → ¬C

-- Theorem to prove
theorem good_goods_sufficient_condition (h : good_goods_not_cheap G C) : 
  G → ¬C :=
by
  sorry


end good_goods_sufficient_condition_l286_28699


namespace inaccurate_tape_measurement_l286_28666

theorem inaccurate_tape_measurement 
  (wholesale_price : ℝ) 
  (tape_length : ℝ) 
  (retail_markup : ℝ) 
  (actual_profit : ℝ) 
  (h1 : retail_markup = 0.4)
  (h2 : actual_profit = 0.39)
  (h3 : ((1 + retail_markup) * wholesale_price - tape_length * wholesale_price) / (tape_length * wholesale_price) = actual_profit) :
  tape_length = 140 / 139 :=
sorry

end inaccurate_tape_measurement_l286_28666


namespace salary_reduction_percentage_l286_28646

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) : 
  S = (S - (R/100) * S) * (1 + 25/100) → R = 20 :=
by sorry

end salary_reduction_percentage_l286_28646


namespace student_marks_average_l286_28644

/-- Given a student's marks in mathematics, physics, and chemistry, 
    prove that the average of mathematics and chemistry marks is 20. -/
theorem student_marks_average (M P C : ℝ) 
  (h1 : M + P = 20)
  (h2 : C = P + 20) : 
  (M + C) / 2 = 20 := by
  sorry

end student_marks_average_l286_28644


namespace petya_wins_l286_28640

/-- Represents a 7x7 game board --/
def GameBoard := Fin 7 → Fin 7 → Option (Fin 7)

/-- Checks if a move is valid on the given board --/
def is_valid_move (board : GameBoard) (row col : Fin 7) (digit : Fin 7) : Prop :=
  (∀ i : Fin 7, board i col ≠ some digit) ∧
  (∀ j : Fin 7, board row j ≠ some digit)

/-- Represents a player's strategy --/
def Strategy := GameBoard → Option (Fin 7 × Fin 7 × Fin 7)

/-- Defines a winning strategy for the first player --/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (board : GameBoard),
    (∃ row col digit, is_valid_move board row col digit) →
    ∃ row col digit, s board = some (row, col, digit) ∧ is_valid_move board row col digit

theorem petya_wins : ∃ s : Strategy, winning_strategy s :=
  sorry

end petya_wins_l286_28640


namespace paul_school_supplies_l286_28612

/-- Given Paul's initial crayons and erasers, and the number of crayons left,
    prove the difference between erasers and crayons left is 70. -/
theorem paul_school_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (crayons_left : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : crayons_left = 336) :
    initial_erasers - crayons_left = 70 := by
  sorry

end paul_school_supplies_l286_28612


namespace set_operation_example_l286_28696

def set_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∪ N ∧ x ∉ M ∩ N}

theorem set_operation_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  set_operation M N = {1, 4} := by
  sorry

end set_operation_example_l286_28696


namespace outside_trash_count_l286_28658

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end outside_trash_count_l286_28658


namespace weight_at_170cm_l286_28648

/-- Represents the weight of a student in kg -/
def weight : ℝ → ℝ := λ x => 0.75 * x - 68.2

/-- Theorem stating that for a height of 170 cm, the weight is 59.3 kg -/
theorem weight_at_170cm : weight 170 = 59.3 := by
  sorry

end weight_at_170cm_l286_28648


namespace biased_coin_expected_value_l286_28683

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_head : ℚ := 1/4  -- Probability of getting a head
  let p_tail : ℚ := 3/4  -- Probability of getting a tail
  let win_head : ℚ := 4  -- Amount won for flipping a head
  let lose_tail : ℚ := 3 -- Amount lost for flipping a tail
  p_head * win_head - p_tail * lose_tail = -5/4 := by
sorry

end biased_coin_expected_value_l286_28683


namespace peanut_butter_weight_calculation_l286_28611

-- Define the ratio of oil to peanuts
def oil_to_peanuts_ratio : ℚ := 2 / 8

-- Define the amount of oil used
def oil_used : ℚ := 4

-- Define the function to calculate the total weight of peanut butter
def peanut_butter_weight (oil_amount : ℚ) : ℚ :=
  oil_amount + (oil_amount / oil_to_peanuts_ratio) * 8

-- Theorem statement
theorem peanut_butter_weight_calculation :
  peanut_butter_weight oil_used = 20 := by
  sorry

end peanut_butter_weight_calculation_l286_28611


namespace point_Q_coordinate_l286_28623

theorem point_Q_coordinate (Q : ℝ) : (|Q - 0| = 3) → (Q = 3 ∨ Q = -3) := by
  sorry

end point_Q_coordinate_l286_28623


namespace tiles_needed_l286_28628

/-- Given a rectangular room and tiling specifications, calculate the number of tiles needed --/
theorem tiles_needed (room_length room_width tile_size fraction_to_tile : ℝ) 
  (h1 : room_length = 12)
  (h2 : room_width = 20)
  (h3 : tile_size = 1)
  (h4 : fraction_to_tile = 1/6) :
  (room_length * room_width * fraction_to_tile) / tile_size = 40 := by
  sorry

end tiles_needed_l286_28628


namespace smallest_m_is_13_l286_28637

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

theorem smallest_m_is_13 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m < 13, ∃ n ≥ m, ¬has_nth_root_of_unity n) ∧
  (∀ n ≥ 13, has_nth_root_of_unity n) :=
sorry

end smallest_m_is_13_l286_28637


namespace boat_savings_l286_28634

/-- The cost of traveling by plane in dollars -/
def plane_cost : ℚ := 600

/-- The cost of traveling by boat in dollars -/
def boat_cost : ℚ := 254

/-- The amount saved by taking a boat instead of a plane -/
def money_saved : ℚ := plane_cost - boat_cost

theorem boat_savings : money_saved = 346 := by
  sorry

end boat_savings_l286_28634


namespace rectangle_circle_union_area_l286_28663

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let circle_radius : ℝ := 12
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 108 * π := by
sorry

end rectangle_circle_union_area_l286_28663


namespace sum_of_cubes_l286_28608

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
sorry

end sum_of_cubes_l286_28608


namespace geometric_sequence_shift_l286_28661

/-- 
Given a geometric sequence {a_n} with common ratio q ≠ 1, 
if {a_n + c} is also a geometric sequence, then c = 0.
-/
theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) : 
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence
  q ≠ 1 →  -- common ratio q ≠ 1
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →  -- {a_n + c} is also a geometric sequence
  c = 0 := by
sorry

end geometric_sequence_shift_l286_28661
