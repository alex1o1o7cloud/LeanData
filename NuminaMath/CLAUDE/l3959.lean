import Mathlib

namespace landscape_ratio_l3959_395993

/-- Given a rectangular landscape with the following properties:
  - length is 120 meters
  - contains a playground of 1200 square meters
  - playground occupies 1/3 of the total landscape area
  Prove that the ratio of length to breadth is 4:1 -/
theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (breadth : ℝ) : 
  length = 120 →
  playground_area = 1200 →
  playground_area = (1/3) * (length * breadth) →
  length / breadth = 4 := by
sorry


end landscape_ratio_l3959_395993


namespace perpendicular_vectors_m_equals_five_l3959_395906

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 5 -/
theorem perpendicular_vectors_m_equals_five :
  let a : ℝ × ℝ := (1, -1)
  let b : ℝ × ℝ := (m + 1, 2 * m - 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 5 := by
  sorry

end perpendicular_vectors_m_equals_five_l3959_395906


namespace two_numbers_difference_l3959_395930

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 200 → x - y = 10 := by
  sorry

end two_numbers_difference_l3959_395930


namespace quadrilateral_offset_l3959_395969

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 30 →
  offset1 = 9 →
  area = 225 →
  ∃ offset2 : ℝ, 
    offset2 = 6 ∧ 
    area = (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2 :=
by sorry

end quadrilateral_offset_l3959_395969


namespace cube_equation_solution_l3959_395940

theorem cube_equation_solution (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end cube_equation_solution_l3959_395940


namespace difference_of_two_greatest_values_l3959_395904

def is_three_digit_integer (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

def hundreds_digit (x : ℕ) : ℕ :=
  (x / 100) % 10

def tens_digit (x : ℕ) : ℕ :=
  (x / 10) % 10

def units_digit (x : ℕ) : ℕ :=
  x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_three_digit_integer x ∧ 4 * a = 2 * b ∧ 2 * b = c ∧ a > 0

def two_greatest_values (x y : ℕ) : Prop :=
  satisfies_conditions x ∧ satisfies_conditions y ∧
  ∀ z, satisfies_conditions z → z ≤ x ∧ (z ≠ x → z ≤ y)

theorem difference_of_two_greatest_values :
  ∃ x y, two_greatest_values x y ∧ x - y = 124 :=
sorry

end difference_of_two_greatest_values_l3959_395904


namespace jimmy_remaining_cards_l3959_395935

/-- Calculates the number of cards Jimmy has left after giving cards to Bob and Mary. -/
def cards_left (initial_cards : ℕ) (cards_to_bob : ℕ) : ℕ :=
  initial_cards - cards_to_bob - (2 * cards_to_bob)

/-- Theorem stating that Jimmy has 9 cards left after giving cards to Bob and Mary. -/
theorem jimmy_remaining_cards :
  cards_left 18 3 = 9 := by
  sorry

#eval cards_left 18 3

end jimmy_remaining_cards_l3959_395935


namespace full_servings_count_l3959_395901

-- Define the initial amount of peanut butter
def initial_amount : Rat := 34 + 2/3

-- Define the additional amount of peanut butter
def additional_amount : Rat := 15 + 1/3

-- Define the serving size
def serving_size : Rat := 3

-- Theorem to prove
theorem full_servings_count :
  ⌊(initial_amount + additional_amount) / serving_size⌋ = 16 := by
  sorry

end full_servings_count_l3959_395901


namespace topsoil_cost_l3959_395952

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 7

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := 1512

theorem topsoil_cost : 
  cost_per_cubic_foot * volume_in_cubic_yards * cubic_yards_to_cubic_feet = total_cost := by
  sorry

end topsoil_cost_l3959_395952


namespace rectangle_length_from_square_wire_l3959_395973

/-- Given a square with side length 20 cm and a rectangle with width 14 cm made from the same total wire length, the length of the rectangle is 26 cm. -/
theorem rectangle_length_from_square_wire (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end rectangle_length_from_square_wire_l3959_395973


namespace sixth_term_of_geometric_sequence_l3959_395987

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 243) :
  a 6 = 729 := by
sorry

end sixth_term_of_geometric_sequence_l3959_395987


namespace rectangle_length_equality_l3959_395967

/-- Given a figure composed of rectangles with right angles, prove that the unknown length Y is 1 cm --/
theorem rectangle_length_equality (Y : ℝ) : Y = 1 := by
  -- Define the sum of top segment lengths
  let top_sum := 3 + 2 + 3 + 4 + Y
  -- Define the sum of bottom segment lengths
  let bottom_sum := 7 + 4 + 2
  -- Assert that the sums are equal (property of rectangles)
  have sum_equality : top_sum = bottom_sum := by sorry
  -- Solve for Y
  sorry


end rectangle_length_equality_l3959_395967


namespace speaker_arrangement_count_l3959_395971

-- Define the number of speakers
def n : ℕ := 6

-- Theorem statement
theorem speaker_arrangement_count :
  (n.factorial / 2 : ℕ) = (n.factorial / 2 : ℕ) := by
  sorry

end speaker_arrangement_count_l3959_395971


namespace translation_theorem_l3959_395991

/-- The original function -/
def f (x : ℝ) : ℝ := x^2 + x

/-- The translated function -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The translation amount -/
def a : ℝ := 2

theorem translation_theorem (h : a > 0) : 
  ∀ x, g x = f (x - a) :=
by sorry

end translation_theorem_l3959_395991


namespace fish_count_theorem_l3959_395965

def is_valid_fish_count (t : ℕ) : Prop :=
  (t > 10 ∧ t > 15 ∧ t ≤ 18) ∨
  (t > 10 ∧ t ≤ 15 ∧ t > 18) ∨
  (t ≤ 10 ∧ t > 15 ∧ t > 18)

theorem fish_count_theorem :
  ∀ t : ℕ, is_valid_fish_count t ↔ (t = 16 ∨ t = 17 ∨ t = 18) :=
by sorry

end fish_count_theorem_l3959_395965


namespace sphere_radius_range_l3959_395900

/-- Represents the parabola x^2 = 2y where 0 ≤ y ≤ 20 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 2*p.2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 20}

/-- A sphere touching the bottom of the parabola with its center on the y-axis -/
structure Sphere :=
  (center : ℝ)
  (radius : ℝ)
  (touches_bottom : radius = center)
  (inside_parabola : ∀ x y, (x, y) ∈ Parabola → x^2 + (y - center)^2 ≥ radius^2)

/-- The theorem stating the range of the sphere's radius -/
theorem sphere_radius_range (s : Sphere) : 0 < s.radius ∧ s.radius ≤ 1 := by
  sorry


end sphere_radius_range_l3959_395900


namespace probability_multiple_of_three_l3959_395924

def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

def count_multiples_of_three (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_three |>.length

theorem probability_multiple_of_three : 
  (count_multiples_of_three 24 : ℚ) / 24 = 1 / 3 := by
  sorry

end probability_multiple_of_three_l3959_395924


namespace choir_singing_problem_l3959_395957

theorem choir_singing_problem (total_singers : ℕ) 
  (h1 : total_singers = 30)
  (first_verse : ℕ) 
  (h2 : first_verse = total_singers / 2)
  (second_verse : ℕ)
  (h3 : second_verse = (total_singers - first_verse) / 3)
  (final_verse : ℕ)
  (h4 : final_verse = total_singers - first_verse - second_verse) :
  final_verse = 10 := by
  sorry

end choir_singing_problem_l3959_395957


namespace plants_original_cost_l3959_395976

/-- Given a discount and the amount spent on plants, calculate the original cost. -/
def original_cost (discount : ℚ) (amount_spent : ℚ) : ℚ :=
  discount + amount_spent

/-- Theorem stating that given the specific discount and amount spent, the original cost is $467.00 -/
theorem plants_original_cost :
  let discount : ℚ := 399
  let amount_spent : ℚ := 68
  original_cost discount amount_spent = 467 := by
sorry

end plants_original_cost_l3959_395976


namespace gcd_increase_l3959_395912

theorem gcd_increase (m n : ℕ) (h : Nat.gcd (m + 6) n = 9 * Nat.gcd m n) :
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
  sorry

end gcd_increase_l3959_395912


namespace square_b_minus_d_l3959_395999

theorem square_b_minus_d (a b c d : ℝ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end square_b_minus_d_l3959_395999


namespace square_sum_given_difference_and_product_l3959_395953

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 := by
  sorry

end square_sum_given_difference_and_product_l3959_395953


namespace prob_fewer_tails_eight_coins_l3959_395922

/-- The number of coins flipped -/
def n : ℕ := 8

/-- The probability of getting fewer tails than heads when flipping n coins -/
def prob_fewer_tails (n : ℕ) : ℚ :=
  (1 - (n.choose (n / 2) : ℚ) / 2^n) / 2

theorem prob_fewer_tails_eight_coins : 
  prob_fewer_tails n = 93 / 256 := by
  sorry

end prob_fewer_tails_eight_coins_l3959_395922


namespace f₁_eq_f₂_l3959_395943

/-- Function f₁ that always returns 1 -/
def f₁ : ℝ → ℝ := λ _ => 1

/-- Function f₂ that returns x^0 -/
def f₂ : ℝ → ℝ := λ x => x^0

/-- Theorem stating that f₁ and f₂ are the same function -/
theorem f₁_eq_f₂ : f₁ = f₂ := by sorry

end f₁_eq_f₂_l3959_395943


namespace golden_ratio_equation_l3959_395946

theorem golden_ratio_equation : 
  let x : ℝ := (Real.sqrt 5 + 1) / 2
  let y : ℝ := (Real.sqrt 5 - 1) / 2
  x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := by sorry

end golden_ratio_equation_l3959_395946


namespace count_prime_pairs_sum_80_l3959_395959

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The sum of the pair is 80 -/
def sumIs80 (p q : ℕ) : Prop := p + q = 80

/-- The statement to be proved -/
theorem count_prime_pairs_sum_80 :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs → isPrime p ∧ isPrime q ∧ sumIs80 p q) ∧
    (∀ (p q : ℕ), isPrime p → isPrime q → sumIs80 p q → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) :=
sorry

end count_prime_pairs_sum_80_l3959_395959


namespace sum_of_squares_divisibility_l3959_395918

theorem sum_of_squares_divisibility (a b c : ℤ) :
  9 ∣ (a^2 + b^2 + c^2) → 
  (9 ∣ (a^2 - b^2)) ∨ (9 ∣ (a^2 - c^2)) ∨ (9 ∣ (b^2 - c^2)) := by
  sorry

end sum_of_squares_divisibility_l3959_395918


namespace train_b_start_time_l3959_395941

/-- The time when trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when train A starts, in hours after midnight -/
def train_a_start : ℝ := 8

/-- The distance between city A and city B in kilometers -/
def total_distance : ℝ := 465

/-- The speed of train A in km/hr -/
def train_a_speed : ℝ := 60

/-- The speed of train B in km/hr -/
def train_b_speed : ℝ := 75

/-- The theorem stating that the train from city B starts at 9 a.m. -/
theorem train_b_start_time :
  ∃ (t : ℝ),
    t = 9 ∧
    (meeting_time - train_a_start) * train_a_speed +
      (meeting_time - t) * train_b_speed = total_distance :=
by sorry

end train_b_start_time_l3959_395941


namespace fraction_calculation_l3959_395923

theorem fraction_calculation : 
  (7/6) / ((1/6) - (1/3)) * (3/14) / (3/5) = -5/2 := by
  sorry

end fraction_calculation_l3959_395923


namespace egg_usage_ratio_l3959_395981

/-- Proves that the ratio of eggs used to total eggs bought is 1:2 --/
theorem egg_usage_ratio (total_dozen : ℕ) (broken : ℕ) (left : ℕ) : 
  total_dozen = 6 → broken = 15 → left = 21 → 
  (total_dozen * 12 - (left + broken)) * 2 = total_dozen * 12 := by
  sorry

end egg_usage_ratio_l3959_395981


namespace log_8_4_equals_twice_log_8_2_l3959_395974

-- Define log_8 as a function
noncomputable def log_8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log_8_4_equals_twice_log_8_2 :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.00005 ∧ 
  ∃ (δ : ℝ), δ ≥ 0 ∧ δ < 0.00005 ∧
  |log_8 2 - 0.2525| ≤ ε →
  |log_8 4 - 2 * log_8 2| ≤ δ :=
sorry

end log_8_4_equals_twice_log_8_2_l3959_395974


namespace marks_deck_cost_l3959_395926

/-- The total cost of a rectangular deck with sealant -/
def deck_cost (length width base_cost sealant_cost : ℝ) : ℝ :=
  let area := length * width
  let total_cost_per_sqft := base_cost + sealant_cost
  area * total_cost_per_sqft

/-- Theorem: The cost of Mark's deck is $4800 -/
theorem marks_deck_cost :
  deck_cost 30 40 3 1 = 4800 := by
  sorry

end marks_deck_cost_l3959_395926


namespace total_tulips_count_l3959_395968

def tulips_per_eye : ℕ := 8
def number_of_eyes : ℕ := 2
def tulips_for_smile : ℕ := 18
def background_multiplier : ℕ := 9

def total_tulips : ℕ := 
  (tulips_per_eye * number_of_eyes + tulips_for_smile) + 
  (background_multiplier * tulips_for_smile)

theorem total_tulips_count : total_tulips = 196 := by
  sorry

end total_tulips_count_l3959_395968


namespace same_color_probability_l3959_395916

def totalBalls : ℕ := 20
def greenBalls : ℕ := 8
def redBalls : ℕ := 5
def blueBalls : ℕ := 7

theorem same_color_probability : 
  (greenBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (redBalls : ℚ) ^ 2 / totalBalls ^ 2 + 
  (blueBalls : ℚ) ^ 2 / totalBalls ^ 2 = 345 / 1000 := by
  sorry

end same_color_probability_l3959_395916


namespace max_qpn_value_l3959_395988

/-- Represents a two-digit number with equal digits -/
def TwoDigitEqualDigits (n : Nat) : Prop :=
  n ≥ 11 ∧ n ≤ 99 ∧ n % 11 = 0

/-- Represents a one-digit number -/
def OneDigit (n : Nat) : Prop :=
  n ≥ 1 ∧ n ≤ 9

/-- Represents a three-digit number -/
def ThreeDigits (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999

theorem max_qpn_value (nn n qpn : Nat) 
  (h1 : TwoDigitEqualDigits nn)
  (h2 : OneDigit n)
  (h3 : ThreeDigits qpn)
  (h4 : nn * n = qpn) :
  qpn ≤ 396 :=
sorry

end max_qpn_value_l3959_395988


namespace clock_rings_count_l3959_395914

/-- Represents the number of times a clock rings in a day -/
def clock_rings (ring_interval : ℕ) (start_hour : ℕ) (day_length : ℕ) : ℕ :=
  (day_length - start_hour) / ring_interval + 1

/-- Theorem stating that a clock ringing every 3 hours starting at 1 A.M. will ring 8 times in a day -/
theorem clock_rings_count : clock_rings 3 1 24 = 8 := by
  sorry

end clock_rings_count_l3959_395914


namespace square_value_theorem_l3959_395936

theorem square_value_theorem (a b : ℝ) (h : a > b) :
  ∃ square : ℝ, (-2*a - 1 < -2*b + square) ∧ (square = 0) := by
sorry

end square_value_theorem_l3959_395936


namespace prob_two_even_in_six_dice_l3959_395929

/-- A fair 10-sided die with faces numbered from 1 to 10 -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a 10-sided die -/
def probOdd : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The number of dice that should show an even number -/
def numEven : ℕ := 2

/-- The probability of rolling exactly two even numbers when rolling six fair 10-sided dice -/
theorem prob_two_even_in_six_dice : 
  (numDice.choose numEven : ℚ) * probEven ^ numEven * probOdd ^ (numDice - numEven) = 15/64 := by
  sorry

end prob_two_even_in_six_dice_l3959_395929


namespace expression_evaluation_l3959_395919

theorem expression_evaluation : 
  let a : ℚ := 5
  let b : ℚ := a + 4
  let c : ℚ := b - 12
  (a + 2 ≠ 0) → (b - 3 ≠ 0) → (c + 7 ≠ 0) →
  ((a + 4) / (a + 2)) * ((b + 1) / (b - 3)) * ((c + 10) / (c + 7)) = 3.75 := by
  sorry

end expression_evaluation_l3959_395919


namespace expression_equality_l3959_395947

theorem expression_equality : Real.sqrt 8 ^ (1/3) - |2 - Real.sqrt 3| + (1/2)^0 - Real.sqrt 3 = 1 := by
  sorry

end expression_equality_l3959_395947


namespace fraction_transformation_l3959_395983

theorem fraction_transformation (x : ℝ) (h : x ≠ 2) : 2 / (2 - x) = -(2 / (x - 2)) := by
  sorry

end fraction_transformation_l3959_395983


namespace workshop_average_salary_l3959_395972

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 35)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 16000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 :=
by
  sorry

end workshop_average_salary_l3959_395972


namespace tomato_plant_ratio_l3959_395939

/-- Proves that the ratio of dead tomato plants to initial tomato plants is 1/2 --/
theorem tomato_plant_ratio (total_vegetables : ℕ) (vegetables_per_plant : ℕ) 
  (initial_tomato : ℕ) (initial_eggplant : ℕ) (initial_pepper : ℕ) 
  (dead_pepper : ℕ) : 
  total_vegetables = 56 →
  vegetables_per_plant = 7 →
  initial_tomato = 6 →
  initial_eggplant = 2 →
  initial_pepper = 4 →
  dead_pepper = 1 →
  (initial_tomato - (total_vegetables / vegetables_per_plant - initial_eggplant - (initial_pepper - dead_pepper))) / initial_tomato = 1 / 2 := by
  sorry

end tomato_plant_ratio_l3959_395939


namespace base6_addition_subtraction_l3959_395985

-- Define a function to convert from base 6 to decimal
def base6ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Define a function to convert from decimal to base 6
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition_subtraction :
  let a := [2, 4, 5, 3, 1]  -- 13542₆ in reverse order
  let b := [5, 3, 4, 3, 2]  -- 23435₆ in reverse order
  let c := [2, 1, 3, 4]     -- 4312₆ in reverse order
  let result := [5, 0, 4, 1, 3]  -- 31405₆ in reverse order
  decimalToBase6 ((base6ToDecimal a + base6ToDecimal b) - base6ToDecimal c) = result := by
  sorry


end base6_addition_subtraction_l3959_395985


namespace papi_calot_plants_l3959_395910

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end papi_calot_plants_l3959_395910


namespace books_given_correct_l3959_395980

/-- The number of books Melissa gives to Jordan --/
def books_given : ℝ := 10.5

/-- Initial number of books Melissa had --/
def melissa_initial : ℕ := 123

/-- Initial number of books Jordan had --/
def jordan_initial : ℕ := 27

theorem books_given_correct :
  let melissa_final := melissa_initial - books_given
  let jordan_final := jordan_initial + books_given
  (melissa_initial + jordan_initial : ℝ) = melissa_final + jordan_final ∧
  melissa_final = 3 * jordan_final := by
  sorry

end books_given_correct_l3959_395980


namespace truck_loading_time_l3959_395908

theorem truck_loading_time (worker1_time worker2_time combined_time : ℝ) 
  (h1 : worker1_time = 6)
  (h2 : combined_time = 2.4)
  (h3 : 1 / worker1_time + 1 / worker2_time = 1 / combined_time) :
  worker2_time = 4 := by
  sorry

end truck_loading_time_l3959_395908


namespace lcm_from_hcf_and_product_l3959_395989

theorem lcm_from_hcf_and_product (a b : ℕ+) :
  Nat.gcd a b = 20 →
  a * b = 2560 →
  Nat.lcm a b = 128 := by
sorry

end lcm_from_hcf_and_product_l3959_395989


namespace range_of_x_plus_cos_y_l3959_395996

theorem range_of_x_plus_cos_y (x y : ℝ) (h : 2 * x + Real.cos (2 * y) = 1) :
  ∃ (z : ℝ), z = x + Real.cos y ∧ -1 ≤ z ∧ z ≤ 5/4 ∧
  (∃ (x' y' : ℝ), 2 * x' + Real.cos (2 * y') = 1 ∧ x' + Real.cos y' = -1) ∧
  (∃ (x'' y'' : ℝ), 2 * x'' + Real.cos (2 * y'') = 1 ∧ x'' + Real.cos y'' = 5/4) :=
sorry

end range_of_x_plus_cos_y_l3959_395996


namespace sin_cos_sum_equivalent_l3959_395927

theorem sin_cos_sum_equivalent (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) := by
  sorry

end sin_cos_sum_equivalent_l3959_395927


namespace dot_product_of_intersection_vectors_l3959_395950

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type representing y = 2/3(x+2) -/
structure Line where
  x : ℝ
  y : ℝ
  eq : y = 2/3*(x+2)

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Intersection points of the parabola and the line -/
def intersection_points (p : Parabola) (l : Line) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Vector from focus to a point -/
def vector_from_focus (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - focus.1, p.2 - focus.2)

/-- Dot product of two vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Main theorem -/
theorem dot_product_of_intersection_vectors (p : Parabola) (l : Line) :
  let (m, n) := intersection_points p l
  dot_product (vector_from_focus m) (vector_from_focus n) = 8 := by sorry

end dot_product_of_intersection_vectors_l3959_395950


namespace four_digit_number_count_l3959_395979

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- A function that returns true if all digits in a natural number are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the leftmost digit of a natural number -/
def leftmostDigit (n : ℕ) : ℕ := sorry

/-- A function that returns the rightmost digit of a natural number -/
def rightmostDigit (n : ℕ) : ℕ := sorry

theorem four_digit_number_count :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1000 ≤ n ∧ n < 10000) ∧ 
    (∀ n ∈ S, isPrime (leftmostDigit n)) ∧
    (∀ n ∈ S, isPerfectSquare (rightmostDigit n)) ∧
    (∀ n ∈ S, allDigitsDifferent n) ∧
    Finset.card S ≥ 288 := by sorry

end four_digit_number_count_l3959_395979


namespace mango_rate_is_55_l3959_395992

/-- The rate per kg of mangoes given Bruce's purchase --/
def mango_rate (grape_kg : ℕ) (grape_rate : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - grape_kg * grape_rate) / mango_kg

/-- Theorem stating that the rate per kg of mangoes is 55 --/
theorem mango_rate_is_55 :
  mango_rate 8 70 10 1110 = 55 := by
  sorry

end mango_rate_is_55_l3959_395992


namespace book_cost_l3959_395966

/-- If three identical books cost $45, then seven of these books cost $105. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  7 * (cost_of_three / 3) = 105 := by
  sorry

end book_cost_l3959_395966


namespace shell_distribution_l3959_395975

theorem shell_distribution (jillian savannah clayton friends_share : ℕ) : 
  jillian = 29 →
  savannah = 17 →
  clayton = 8 →
  friends_share = 27 →
  (jillian + savannah + clayton) / friends_share = 2 :=
by sorry

end shell_distribution_l3959_395975


namespace last_three_digits_l3959_395945

/-- A function that generates the list of positive integers with first digit 2 in increasing order -/
def digit2List : ℕ → ℕ 
| 0 => 2
| (n + 1) => 
  let prev := digit2List n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- The 998th digit in the digit2List -/
def digit998 : ℕ := sorry

/-- The 999th digit in the digit2List -/
def digit999 : ℕ := sorry

/-- The 1000th digit in the digit2List -/
def digit1000 : ℕ := sorry

/-- Theorem stating that the 998th, 999th, and 1000th digits form the number 216 -/
theorem last_three_digits : 
  digit998 * 100 + digit999 * 10 + digit1000 = 216 := by sorry

end last_three_digits_l3959_395945


namespace prob_two_red_cards_l3959_395986

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Defines a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem: The probability of drawing two red cards in succession with replacement is 1/4 -/
theorem prob_two_red_cards (d : Deck) (h : d = standard_deck) :
  (prob_red_card d) * (prob_red_card d) = 1 / 4 := by
  sorry


end prob_two_red_cards_l3959_395986


namespace garden_ratio_l3959_395913

/-- Given a rectangular garden with perimeter 180 yards and length 60 yards,
    prove that the ratio of length to width is 2:1 -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ)
    (h1 : perimeter = 180)
    (h2 : length = 60)
    (h3 : perimeter = 2 * length + 2 * width) :
    length / width = 2 := by
  sorry

end garden_ratio_l3959_395913


namespace quadratic_rational_root_even_denominator_l3959_395921

theorem quadratic_rational_root_even_denominator
  (a b c : ℤ)  -- Coefficients are integers
  (h_even_sum : Even (a + b))  -- Sum of a and b is even
  (h_odd_c : Odd c)  -- c is odd
  (p q : ℤ)  -- p/q is a rational root in simplest form
  (h_coprime : Nat.Coprime p.natAbs q.natAbs)  -- p and q are coprime
  (h_root : a * p^2 + b * p * q + c * q^2 = 0)  -- p/q is a root
  : Even q  -- q is even
:= by sorry

end quadratic_rational_root_even_denominator_l3959_395921


namespace find_number_l3959_395982

theorem find_number : ∃ x : ℝ, 8 * x = 0.4 * 900 ∧ x = 45 := by
  sorry

end find_number_l3959_395982


namespace triangle_angles_l3959_395990

/-- Triangle angles theorem -/
theorem triangle_angles (ω φ θ : ℝ) : 
  ω + φ + θ = 180 → 
  2 * ω + θ = 180 → 
  φ = 2 * θ → 
  θ = 36 ∧ φ = 72 ∧ ω = 72 := by
sorry

end triangle_angles_l3959_395990


namespace factory_working_days_l3959_395960

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 6500

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1300

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 5 :=
sorry

end factory_working_days_l3959_395960


namespace unique_solution_is_sqrt_two_l3959_395902

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- State the theorem
theorem unique_solution_is_sqrt_two :
  ∃! x, x > 1 ∧ f x = 1/4 ∧ x = Real.sqrt 2 := by sorry

end unique_solution_is_sqrt_two_l3959_395902


namespace sum_of_coefficients_l3959_395915

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end sum_of_coefficients_l3959_395915


namespace chord_segment_lengths_l3959_395995

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (h1 : R = 15) (h2 : OM = 13) (h3 : AB = 18) :
  ∃ (AM MB : ℝ), AM + MB = AB ∧ AM = 14 ∧ MB = 4 := by
  sorry

end chord_segment_lengths_l3959_395995


namespace chess_tournament_matches_l3959_395977

/-- The number of matches in a chess tournament -/
def tournament_matches (n : ℕ) (matches_per_pair : ℕ) : ℕ :=
  matches_per_pair * n * (n - 1) / 2

/-- Theorem: In a chess tournament with 150 players, where each player plays 3 matches
    against every other player, the total number of matches is 33,750 -/
theorem chess_tournament_matches :
  tournament_matches 150 3 = 33750 := by
  sorry

end chess_tournament_matches_l3959_395977


namespace problem_1_problem_2_l3959_395907

-- Problem 1
theorem problem_1 : -17 - (-6) + 8 - 2 = -5 := by sorry

-- Problem 2
theorem problem_2 : -1^2024 + 16 / (-2)^3 * |(-3) - 1| = -9 := by sorry

end problem_1_problem_2_l3959_395907


namespace min_value_of_f_l3959_395948

noncomputable section

def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

theorem min_value_of_f :
  ∃ (min : ℝ), min = π/6 - Real.sqrt 3/2 ∧
  ∀ x ∈ Set.Ioo 0 π, f x ≥ min :=
sorry

end

end min_value_of_f_l3959_395948


namespace percentage_not_sophomores_l3959_395909

theorem percentage_not_sophomores :
  ∀ (total juniors seniors freshmen sophomores : ℕ),
    total = 800 →
    juniors = (22 * total) / 100 →
    seniors = 160 →
    freshmen = sophomores + 48 →
    total = freshmen + sophomores + juniors + seniors →
    (100 * (total - sophomores)) / total = 74 := by
  sorry

end percentage_not_sophomores_l3959_395909


namespace billy_coin_count_l3959_395932

/-- Represents the number of piles for each coin type -/
structure PileCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Represents the number of coins in each pile for each coin type -/
structure CoinsPerPile where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total number of coins given the pile counts and coins per pile -/
def totalCoins (piles : PileCount) (coinsPerPile : CoinsPerPile) : Nat :=
  piles.quarters * coinsPerPile.quarters +
  piles.dimes * coinsPerPile.dimes +
  piles.nickels * coinsPerPile.nickels +
  piles.pennies * coinsPerPile.pennies

/-- Billy's coin sorting problem -/
theorem billy_coin_count :
  let piles : PileCount := { quarters := 3, dimes := 2, nickels := 4, pennies := 6 }
  let coinsPerPile : CoinsPerPile := { quarters := 5, dimes := 7, nickels := 3, pennies := 9 }
  totalCoins piles coinsPerPile = 95 := by
  sorry

end billy_coin_count_l3959_395932


namespace max_q_plus_2r_l3959_395955

theorem max_q_plus_2r (q r : ℕ+) (h : 1230 = 28 * q + r) : 
  (∀ q' r' : ℕ+, 1230 = 28 * q' + r' → q' + 2 * r' ≤ q + 2 * r) ∧ q + 2 * r = 95 := by
  sorry

end max_q_plus_2r_l3959_395955


namespace a_formula_l3959_395970

noncomputable section

/-- The function f(x) = x / sqrt(1 + x^2) -/
def f (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

/-- The sequence a_n defined recursively -/
def a (x : ℝ) : ℕ → ℝ
  | 0 => f x
  | n + 1 => f (a x n)

/-- The theorem stating the general formula for a_n -/
theorem a_formula (x : ℝ) (h : x > 0) (n : ℕ) :
  a x n = x / Real.sqrt (1 + n * x^2) := by
  sorry

end

end a_formula_l3959_395970


namespace largest_decimal_l3959_395942

theorem largest_decimal : ∀ (a b c d e : ℚ),
  a = 97/100 ∧ b = 979/1000 ∧ c = 9709/10000 ∧ d = 907/1000 ∧ e = 9089/10000 →
  b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e :=
by sorry

end largest_decimal_l3959_395942


namespace spadesuit_calculation_l3959_395911

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_calculation :
  (spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6))) = 3 := by
  sorry

end spadesuit_calculation_l3959_395911


namespace total_blue_balloons_l3959_395944

/-- The number of blue balloons Joan and Melanie have in total is 81, 
    given that Joan has 40 and Melanie has 41. -/
theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l3959_395944


namespace alicia_remaining_masks_l3959_395905

/-- The number of mask sets remaining in Alicia's collection after donation -/
def remaining_masks (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem stating that Alicia has 39 mask sets left after donating to the museum -/
theorem alicia_remaining_masks :
  remaining_masks 90 51 = 39 := by
  sorry

end alicia_remaining_masks_l3959_395905


namespace hyperbola_line_intersection_eccentricity_l3959_395903

/-- The eccentricity of a hyperbola that has a common point with the line y = 2x --/
def eccentricity_range (a b : ℝ) : Prop :=
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ x y : ℝ, y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) →
  1 < e ∧ e ≤ Real.sqrt 5

theorem hyperbola_line_intersection_eccentricity :
  ∀ a b : ℝ, a > 0 ∧ b > 0 → eccentricity_range a b :=
sorry

end hyperbola_line_intersection_eccentricity_l3959_395903


namespace tangent_parallel_at_minus_one_minus_four_l3959_395934

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one_minus_four :
  let P₀ : ℝ × ℝ := (-1, -4)
  (f' (P₀.1) = 4) ∧ (f P₀.1 = P₀.2) := by
  sorry

end tangent_parallel_at_minus_one_minus_four_l3959_395934


namespace mad_hatter_waiting_time_l3959_395931

/-- Represents a clock with a rate different from real time -/
structure AdjustedClock where
  rate : ℚ  -- Rate of the clock compared to real time

/-- The Mad Hatter's clock -/
def madHatterClock : AdjustedClock :=
  { rate := 75 / 60 }

/-- The March Hare's clock -/
def marchHareClock : AdjustedClock :=
  { rate := 50 / 60 }

/-- Calculates the real time passed for a given clock time -/
def realTimePassed (clock : AdjustedClock) (clockTime : ℚ) : ℚ :=
  clockTime / clock.rate

/-- The agreed meeting time on their clocks -/
def meetingTime : ℚ := 5

theorem mad_hatter_waiting_time :
  realTimePassed madHatterClock meetingTime + 2 = realTimePassed marchHareClock meetingTime :=
by sorry

end mad_hatter_waiting_time_l3959_395931


namespace annika_return_time_l3959_395994

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  rate : ℝ  -- Hiking rate in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalDistance : ℝ  -- Total distance to hike east in kilometers

/-- Calculates the time needed to return to the start of the trail -/
def timeToReturn (scenario : HikingScenario) : ℝ :=
  let remainingDistance := scenario.totalDistance - scenario.initialDistance
  let timeToCompleteEast := remainingDistance * scenario.rate
  let timeToReturnWest := scenario.totalDistance * scenario.rate
  timeToCompleteEast + timeToReturnWest

/-- Theorem stating that Annika needs 35 minutes to return to the start -/
theorem annika_return_time :
  let scenario : HikingScenario := {
    rate := 10,
    initialDistance := 2.5,
    totalDistance := 3
  }
  timeToReturn scenario = 35 := by sorry

end annika_return_time_l3959_395994


namespace fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l3959_395933

theorem fifteen_times_thirtysix_plus_fifteen_times_three_cubed : 15 * 36 + 15 * 3^3 = 945 := by
  sorry

end fifteen_times_thirtysix_plus_fifteen_times_three_cubed_l3959_395933


namespace greatest_integer_with_gcf_six_exists_138_unique_greatest_l3959_395956

theorem greatest_integer_with_gcf_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem exists_138 : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem unique_greatest : ∀ m : ℕ, m > 138 → m < 150 → Nat.gcd m 18 ≠ 6 :=
by sorry

end greatest_integer_with_gcf_six_exists_138_unique_greatest_l3959_395956


namespace milk_consumption_l3959_395937

theorem milk_consumption (bottle_milk : ℚ) (pour_fraction : ℚ) (drink_fraction : ℚ) :
  bottle_milk = 3/4 →
  pour_fraction = 1/2 →
  drink_fraction = 1/3 →
  drink_fraction * (pour_fraction * bottle_milk) = 1/8 := by
  sorry

end milk_consumption_l3959_395937


namespace time_to_work_l3959_395997

def round_trip_time : ℝ := 2
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120

theorem time_to_work :
  let distance := (round_trip_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home)
  let time_to_work := distance / speed_to_work
  time_to_work * 60 = 72 := by sorry

end time_to_work_l3959_395997


namespace sequence_next_term_l3959_395984

theorem sequence_next_term (a₁ a₂ a₃ a₄ a₅ x : ℕ) : 
  a₁ = 2 ∧ a₂ = 5 ∧ a₃ = 11 ∧ a₄ = 20 ∧ a₅ = 32 ∧
  (a₂ - a₁) = 3 ∧ (a₃ - a₂) = 6 ∧ (a₄ - a₃) = 9 ∧ (a₅ - a₄) = 12 ∧
  (x - a₅) = (a₅ - a₄) + 3 →
  x = 47 := by
sorry

end sequence_next_term_l3959_395984


namespace perpendicular_lines_triangle_area_l3959_395949

/-- Two perpendicular lines intersecting at (8,6) with y-intercepts differing by 14 form a triangle with area 56 -/
theorem perpendicular_lines_triangle_area :
  ∀ (m₁ m₂ b₁ b₂ : ℝ),
  m₁ * m₂ = -1 →                         -- perpendicular lines
  8 * m₁ + b₁ = 6 →                      -- line 1 passes through (8,6)
  8 * m₂ + b₂ = 6 →                      -- line 2 passes through (8,6)
  b₁ - b₂ = 14 →                         -- difference between y-intercepts
  (1/2) * 8 * |b₁ - b₂| = 56 :=          -- area of triangle
by sorry

end perpendicular_lines_triangle_area_l3959_395949


namespace s_square_sum_l3959_395978

/-- The sequence s_n is defined by the power series expansion of 1 / (1 - 2x - x^2) -/
noncomputable def s : ℕ → ℝ := sorry

/-- The power series expansion of 1 / (1 - 2x - x^2) -/
axiom power_series_expansion (x : ℝ) (h : x ≠ 0) : 
  (1 : ℝ) / (1 - 2*x - x^2) = ∑' (n : ℕ), s n * x^n

/-- The main theorem: s_n^2 + s_{n+1}^2 = s_{2n+2} for all non-negative integers n -/
theorem s_square_sum (n : ℕ) : (s n)^2 + (s (n+1))^2 = s (2*n+2) := by sorry

end s_square_sum_l3959_395978


namespace min_pieces_for_horizontal_four_l3959_395925

/-- Represents a chessboard as a list of 8 rows, each containing 8 cells --/
def Chessboard := List (List Bool)

/-- Checks if a row contains 4 consecutive true values --/
def hasFourConsecutive (row : List Bool) : Bool :=
  sorry

/-- Checks if any row in the chessboard has 4 consecutive pieces --/
def hasHorizontalFour (board : Chessboard) : Bool :=
  sorry

/-- Generates all possible arrangements of n pieces on a chessboard --/
def allArrangements (n : Nat) : List Chessboard :=
  sorry

theorem min_pieces_for_horizontal_four :
  ∀ n : Nat, (n ≥ 49 ↔ ∀ board ∈ allArrangements n, hasHorizontalFour board) :=
by sorry

end min_pieces_for_horizontal_four_l3959_395925


namespace work_together_proof_l3959_395998

/-- The number of days after which Alice, Bob, Carol, and Dave work together again -/
def days_until_work_together_again : ℕ := 360

/-- Alice's work schedule (every 5th day) -/
def alice_schedule : ℕ := 5

/-- Bob's work schedule (every 6th day) -/
def bob_schedule : ℕ := 6

/-- Carol's work schedule (every 8th day) -/
def carol_schedule : ℕ := 8

/-- Dave's work schedule (every 9th day) -/
def dave_schedule : ℕ := 9

theorem work_together_proof :
  days_until_work_together_again = Nat.lcm alice_schedule (Nat.lcm bob_schedule (Nat.lcm carol_schedule dave_schedule)) :=
by
  sorry

#eval days_until_work_together_again

end work_together_proof_l3959_395998


namespace modified_fibonacci_series_sum_l3959_395964

/-- Modified Fibonacci sequence -/
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => F (n + 1) + F n

/-- The sum of the series F_n / 5^n from n = 0 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, (F n : ℝ) / 5^n

theorem modified_fibonacci_series_sum : seriesSum = 35 / 18 := by
  sorry

end modified_fibonacci_series_sum_l3959_395964


namespace continued_fraction_solution_l3959_395917

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / y ∧ y = (3 + Real.sqrt 29) / 2 := by
  sorry

end continued_fraction_solution_l3959_395917


namespace cell_count_after_ten_days_l3959_395954

/-- Represents the cell division process over 10 days -/
def cellDivision (initialCells : ℕ) (firstSplitFactor : ℕ) (laterSplitFactor : ℕ) (totalDays : ℕ) : ℕ :=
  let afterFirstTwoDays := initialCells * firstSplitFactor
  let remainingDivisions := (totalDays - 2) / 2
  afterFirstTwoDays * laterSplitFactor ^ remainingDivisions

/-- Theorem stating the number of cells after 10 days -/
theorem cell_count_after_ten_days :
  cellDivision 5 3 2 10 = 240 := by
  sorry

#eval cellDivision 5 3 2 10

end cell_count_after_ten_days_l3959_395954


namespace min_production_to_meet_demand_l3959_395963

/-- Total market demand function -/
def f (x : ℕ) : ℕ := x * (x + 1) * (35 - 2 * x)

/-- Monthly demand function -/
def g (x : ℕ) : ℤ := f x - f (x - 1)

/-- The range of valid month numbers -/
def valid_months : Set ℕ := {x | 1 ≤ x ∧ x ≤ 12}

theorem min_production_to_meet_demand :
  ∃ (a : ℕ), (∀ x ∈ valid_months, (g x : ℝ) ≤ a) ∧
  (∀ b : ℕ, (∀ x ∈ valid_months, (g x : ℝ) ≤ b) → a ≤ b) ∧
  a = 171 := by
  sorry

end min_production_to_meet_demand_l3959_395963


namespace orange_apple_weight_equivalence_l3959_395961

/-- Given that 8 oranges weigh the same as 6 apples, 
    prove that 48 oranges weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℕ → ℝ),
  (∀ n : ℕ, orange_weight n > 0 ∧ apple_weight n > 0) →
  (orange_weight 8 = apple_weight 6) →
  (orange_weight 48 = apple_weight 36) :=
by sorry

end orange_apple_weight_equivalence_l3959_395961


namespace opposite_direction_time_calculation_l3959_395962

/-- Given two people moving in opposite directions from the same starting point,
    calculate the time taken to reach a specific distance between them. -/
theorem opposite_direction_time_calculation 
  (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) 
  (h1 : speed1 = 2) 
  (h2 : speed2 = 3) 
  (h3 : distance = 20) : 
  distance / (speed1 + speed2) = 4 := by
  sorry

#check opposite_direction_time_calculation

end opposite_direction_time_calculation_l3959_395962


namespace sum_of_cubes_l3959_395920

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = -1) (h2 : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end sum_of_cubes_l3959_395920


namespace tenth_term_of_a_sum_of_2023rd_terms_l3959_395928

/-- Sequence a_n defined as (-2)^n -/
def a (n : ℕ) : ℤ := (-2) ^ n

/-- Sequence b_n defined as (-2)^n + (n+1) -/
def b (n : ℕ) : ℤ := (-2) ^ n + (n + 1)

/-- The 10th term of sequence a_n is (-2)^10 -/
theorem tenth_term_of_a : a 10 = (-2) ^ 10 := by sorry

/-- The sum of the 2023rd terms of sequences a_n and b_n is -2^2024 + 2024 -/
theorem sum_of_2023rd_terms : a 2023 + b 2023 = -2 ^ 2024 + 2024 := by sorry

end tenth_term_of_a_sum_of_2023rd_terms_l3959_395928


namespace A_power_15_minus_3_power_14_is_zero_l3959_395938

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 3]

theorem A_power_15_minus_3_power_14_is_zero :
  A^15 - 3 • A^14 = 0 := by sorry

end A_power_15_minus_3_power_14_is_zero_l3959_395938


namespace fruit_trees_space_l3959_395951

/-- The total space needed for fruit trees in Quinton's yard -/
theorem fruit_trees_space (apple_width peach_width : ℕ) 
  (apple_space peach_space : ℕ) (apple_count peach_count : ℕ) : 
  apple_width = 10 → 
  peach_width = 12 → 
  apple_space = 12 → 
  peach_space = 15 → 
  apple_count = 2 → 
  peach_count = 2 → 
  (apple_count * apple_width + apple_space) + 
  (peach_count * peach_width + peach_space) = 71 := by
sorry

end fruit_trees_space_l3959_395951


namespace gcd_factorial_eight_and_factorial_six_squared_l3959_395958

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l3959_395958
