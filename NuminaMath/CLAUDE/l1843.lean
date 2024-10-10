import Mathlib

namespace autograph_distribution_theorem_l1843_184364

/-- Represents a set of autographs from 11 players -/
def Autographs := Fin 11 → Bool

/-- The set of all residents -/
def Residents := Fin 1111

/-- Distribution of autographs to residents -/
def AutographDistribution := Residents → Autographs

theorem autograph_distribution_theorem (d : AutographDistribution) 
  (h : ∀ (i j : Residents), i ≠ j → d i ≠ d j) :
  ∃ (i j : Residents), i ≠ j ∧ 
    (∀ (k : Fin 11), (d i k = true ∧ d j k = false) ∨ (d i k = false ∧ d j k = true)) :=
sorry

end autograph_distribution_theorem_l1843_184364


namespace ball_drawing_problem_l1843_184321

-- Define the sample space
def Ω : Type := Fin 4

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def A : Set Ω := sorry -- Both balls are the same color
def B : Set Ω := sorry -- Both balls are different colors
def C : Set Ω := sorry -- The first ball drawn is red
def D : Set Ω := sorry -- The second ball drawn is red

-- State the theorem
theorem ball_drawing_problem :
  (P (A ∩ B) = 0) ∧
  (P (A ∩ C) = P A * P C) ∧
  (P (B ∩ C) = P B * P C) := by
  sorry

end ball_drawing_problem_l1843_184321


namespace twice_not_equal_squared_l1843_184389

theorem twice_not_equal_squared (m : ℝ) : 2 * m ≠ m ^ 2 := by
  sorry

end twice_not_equal_squared_l1843_184389


namespace line_length_difference_l1843_184319

theorem line_length_difference (white_line blue_line : ℝ) 
  (h1 : white_line = 7.67) 
  (h2 : blue_line = 3.33) : 
  white_line - blue_line = 4.34 := by
sorry

end line_length_difference_l1843_184319


namespace puppy_discount_percentage_l1843_184359

/-- Calculates the discount percentage given the total cost before discount and the amount spent after discount -/
def discount_percentage (total_cost : ℚ) (amount_spent : ℚ) : ℚ :=
  (total_cost - amount_spent) / total_cost * 100

/-- Proves that the new-customer discount percentage is 20% for Julia's puppy purchases -/
theorem puppy_discount_percentage :
  let adoption_fee : ℚ := 20
  let dog_food : ℚ := 20
  let treats : ℚ := 2 * 2.5
  let toys : ℚ := 15
  let crate : ℚ := 20
  let bed : ℚ := 20
  let collar_leash : ℚ := 15
  let total_cost : ℚ := dog_food + treats + toys + crate + bed + collar_leash
  let total_spent : ℚ := 96
  let store_spent : ℚ := total_spent - adoption_fee
  discount_percentage total_cost store_spent = 20 := by
sorry

#eval discount_percentage 95 76

end puppy_discount_percentage_l1843_184359


namespace divisibility_of_concatenated_integers_l1843_184349

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

theorem divisibility_of_concatenated_integers :
  ∃ M : ℕ, M = concatenate_integers 50 ∧ M % 51 = 0 := by
  sorry

end divisibility_of_concatenated_integers_l1843_184349


namespace product_plus_number_equals_result_l1843_184375

theorem product_plus_number_equals_result : ∃ x : ℝ,
  12.05 * 5.4 + x = 108.45000000000003 ∧ x = 43.38000000000003 := by
  sorry

end product_plus_number_equals_result_l1843_184375


namespace possible_values_of_a_l1843_184333

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a^3 - b^3 = 27*x^3) 
  (h2 : a - b = 2*x) : 
  a = x + 5*x/Real.sqrt 6 ∨ a = x - 5*x/Real.sqrt 6 :=
sorry

end possible_values_of_a_l1843_184333


namespace bc_is_one_sixth_of_ad_l1843_184322

/-- Given a line segment AD with points E and B on it, prove that BC is 1/6 of AD -/
theorem bc_is_one_sixth_of_ad (A B C D E : ℝ) : 
  A < E ∧ E < D ∧   -- E is on AD
  A < B ∧ B < D ∧   -- B is on AD
  E - A = 3 * (D - E) ∧   -- AE is 3 times ED
  B - A = 5 * (D - B) ∧   -- AB is 5 times BD
  C = (B + E) / 2   -- C is midpoint of BE
  → 
  (C - B) / (D - A) = 1 / 6 :=
by sorry

end bc_is_one_sixth_of_ad_l1843_184322


namespace min_value_quadratic_sum_l1843_184350

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 2) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 24 / 11 :=
by sorry

end min_value_quadratic_sum_l1843_184350


namespace solution_set_f_geq_4_range_of_a_l1843_184399

-- Define the function f
def f (x : ℝ) := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 : 
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  {a : ℝ | ∃ x, a^2 + 2*a + |1 + x| < f x} = {a : ℝ | -3 < a ∧ a < 1} := by sorry

end solution_set_f_geq_4_range_of_a_l1843_184399


namespace quadratic_inequality_range_l1843_184325

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end quadratic_inequality_range_l1843_184325


namespace orcs_per_squad_l1843_184363

theorem orcs_per_squad (total_weight : ℕ) (num_squads : ℕ) (weight_per_orc : ℕ) :
  total_weight = 1200 →
  num_squads = 10 →
  weight_per_orc = 15 →
  (total_weight / weight_per_orc) / num_squads = 8 :=
by
  sorry

end orcs_per_squad_l1843_184363


namespace marys_maximum_earnings_l1843_184394

/-- Mary's maximum weekly earnings problem -/
theorem marys_maximum_earnings :
  let max_hours : ℕ := 60
  let regular_rate : ℚ := 12
  let regular_hours : ℕ := 30
  let overtime_rate : ℚ := regular_rate * (3/2)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_rate * regular_hours
  let overtime_earnings : ℚ := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 900 := by
  sorry

end marys_maximum_earnings_l1843_184394


namespace license_plate_count_l1843_184383

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 21

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The total number of possible license plates -/
def total_plates : ℕ := alphabet_size * consonant_count * vowel_count * odd_digit_count * odd_digit_count * even_digit_count * even_digit_count

theorem license_plate_count : total_plates = 1706250 := by
  sorry

end license_plate_count_l1843_184383


namespace base_angle_measure_l1843_184308

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  isosceles : angle2 = angle3

-- Theorem statement
theorem base_angle_measure (t : IsoscelesTriangle) (h : t.angle1 = 80 ∨ t.angle2 = 80) :
  t.angle2 = 50 ∨ t.angle2 = 80 := by
  sorry


end base_angle_measure_l1843_184308


namespace rancher_animals_count_l1843_184358

/-- Proves that a rancher with 5 times as many cows as horses and 140 cows has 168 animals in total -/
theorem rancher_animals_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses → cows = 140 → horses + cows = 168 := by
  sorry

end rancher_animals_count_l1843_184358


namespace min_value_on_interval_l1843_184311

/-- The function f(x) = -x³ + 3x² + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a y ≤ f a x) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) :=
by sorry

end min_value_on_interval_l1843_184311


namespace partial_fraction_sum_l1843_184330

theorem partial_fraction_sum (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_l1843_184330


namespace jonathan_saved_eight_l1843_184386

/-- Calculates the amount of money saved given the costs of three books and the additional amount needed. -/
def money_saved (book1_cost book2_cost book3_cost additional_needed : ℕ) : ℕ :=
  (book1_cost + book2_cost + book3_cost) - additional_needed

/-- Proves that given the specific costs and additional amount needed, the money saved is 8. -/
theorem jonathan_saved_eight :
  money_saved 11 19 7 29 = 8 := by
  sorry

end jonathan_saved_eight_l1843_184386


namespace parallel_lines_m_value_l1843_184370

/-- Given two lines AB and CD, where:
    - AB passes through points A(-2,m) and B(m,4)
    - CD passes through points C(m+1,1) and D(m,3)
    - AB is parallel to CD
    Prove that m = -8 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let C : ℝ × ℝ := (m + 1, 1)
  let D : ℝ × ℝ := (m, 3)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_CD := (D.2 - C.2) / (D.1 - C.1)
  slope_AB = slope_CD →
  m = -8 := by
sorry

end parallel_lines_m_value_l1843_184370


namespace max_piles_660_l1843_184354

/-- The maximum number of piles that can be created from a given number of stones,
    where any two pile sizes differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30 -- The actual implementation is not provided, just the result

/-- The condition that any two pile sizes differ by strictly less than 2 times -/
def validPileSizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → (a : ℝ) < 2 * b ∧ (b : ℝ) < 2 * a

theorem max_piles_660 :
  let n := maxPiles 660
  ∃ (piles : List ℕ), 
    piles.length = n ∧ 
    validPileSizes piles ∧ 
    piles.sum = 660 ∧
    ∀ (m : ℕ), m > n → 
      ¬∃ (largerPiles : List ℕ), 
        largerPiles.length = m ∧ 
        validPileSizes largerPiles ∧ 
        largerPiles.sum = 660 :=
by
  sorry

#eval maxPiles 660

end max_piles_660_l1843_184354


namespace f_properties_l1843_184315

def f_property (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ x y, f (x * y) = x * f y + y * f x) ∧
  (∀ x, x > 1 → f x < 0)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (f 1 = 0 ∧ f (-1) = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ > x₂ ∧ x₂ > 1 → f x₁ < f x₂) := by
  sorry

end f_properties_l1843_184315


namespace arithmetic_sequence_property_l1843_184303

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_condition : a 2 + a 4 = 10
  geometric_condition : (a 2) ^ 2 = a 1 * a 5
  arithmetic_property : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_property_l1843_184303


namespace money_calculation_l1843_184312

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def total_amount (n_50 : ℕ) (n_500 : ℕ) : ℕ :=
  50 * n_50 + 500 * n_500

/-- Proves that given 72 total notes with 57 being 50 rupee notes, the total amount is 10350 rupees -/
theorem money_calculation : total_amount 57 (72 - 57) = 10350 := by
  sorry

end money_calculation_l1843_184312


namespace max_product_sum_300_l1843_184304

theorem max_product_sum_300 : 
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ 
  (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) := by
  sorry

end max_product_sum_300_l1843_184304


namespace probability_red_or_blue_l1843_184346

theorem probability_red_or_blue 
  (prob_red : ℝ) 
  (prob_red_or_yellow : ℝ) 
  (h1 : prob_red = 0.45) 
  (h2 : prob_red_or_yellow = 0.65) 
  : prob_red + (1 - prob_red_or_yellow) = 0.80 := by
  sorry

end probability_red_or_blue_l1843_184346


namespace quadratic_coefficient_theorem_l1843_184353

theorem quadratic_coefficient_theorem (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -8) → 
  b = 6 ∧ c = -16 := by
sorry

end quadratic_coefficient_theorem_l1843_184353


namespace max_value_of_J_l1843_184379

def consecutive_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (· + 1)

def sum_equals_21 (a b c d : ℕ) : Prop :=
  a + b + c + d = 21

theorem max_value_of_J (nums : List ℕ) (A B C D E F G H I J K : ℕ) :
  nums = consecutive_numbers 11 →
  D ∈ nums → G ∈ nums → I ∈ nums → F ∈ nums → A ∈ nums →
  B ∈ nums → C ∈ nums → E ∈ nums → H ∈ nums → J ∈ nums → K ∈ nums →
  D > G → G > I → I > F → F > A →
  sum_equals_21 A B C D →
  sum_equals_21 D E F G →
  sum_equals_21 G H F I →
  sum_equals_21 I J K A →
  J ≤ 9 :=
sorry

end max_value_of_J_l1843_184379


namespace smallest_integer_square_triple_plus_75_l1843_184398

theorem smallest_integer_square_triple_plus_75 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 75 → x ≤ y) ∧ x^2 = 3*x + 75 :=
by
  -- The proof goes here
  sorry

end smallest_integer_square_triple_plus_75_l1843_184398


namespace divisibility_by_264_l1843_184341

theorem divisibility_by_264 (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (7 : ℤ)^(2*n) - (4 : ℤ)^(2*n) - 297 = 264 * k := by
  sorry

end divisibility_by_264_l1843_184341


namespace probability_of_letter_in_mathematics_l1843_184360

def alphabet : Finset Char := sorry

def mathematics : Finset Char := sorry

theorem probability_of_letter_in_mathematics :
  (mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by
  sorry

end probability_of_letter_in_mathematics_l1843_184360


namespace removed_sector_angle_l1843_184302

/-- Given a circular piece of paper with radius 15 cm, if a cone is formed from the remaining sector
    after removing a part, and this cone has a radius of 10 cm and a volume of 500π cm³,
    then the angle measure of the removed sector is 120°. -/
theorem removed_sector_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 15 →
  cone_radius = 10 →
  cone_volume = 500 * Real.pi →
  ∃ (removed_angle : ℝ), removed_angle = 120 ∧ 0 ≤ removed_angle ∧ removed_angle ≤ 360 :=
by sorry

end removed_sector_angle_l1843_184302


namespace inequality_solution_set_l1843_184324

theorem inequality_solution_set (x : ℝ) :
  (3 * x^2 - 1 > 13 - 5 * x) ↔ (x < -7 ∨ x > 2) :=
by sorry

end inequality_solution_set_l1843_184324


namespace wire_length_around_square_field_wire_length_is_15840_l1843_184392

/-- The length of wire required to go 15 times around a square field with area 69696 m² -/
theorem wire_length_around_square_field : ℝ :=
  let field_area : ℝ := 69696
  let side_length : ℝ := Real.sqrt field_area
  let perimeter : ℝ := 4 * side_length
  let num_rounds : ℝ := 15
  num_rounds * perimeter

/-- Proof that the wire length is 15840 m -/
theorem wire_length_is_15840 : wire_length_around_square_field = 15840 := by
  sorry


end wire_length_around_square_field_wire_length_is_15840_l1843_184392


namespace quadratic_function_property_l1843_184391

theorem quadratic_function_property 
  (a c y₁ y₂ y₃ y₄ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-2)^2 - 4 * a * (-2) + c)
  (h_y₂ : y₂ = c)
  (h_y₃ : y₃ = a * 3^2 - 4 * a * 3 + c)
  (h_y₄ : y₄ = a * 5^2 - 4 * a * 5 + c)
  (h_y₂y₄ : y₂ * y₄ < 0) :
  y₁ * y₃ < 0 := by
  sorry

end quadratic_function_property_l1843_184391


namespace book_selling_price_l1843_184396

theorem book_selling_price (cost_price selling_price : ℝ) : 
  cost_price = 200 →
  selling_price - cost_price = (340 - cost_price) + 0.05 * cost_price →
  selling_price = 350 := by
sorry

end book_selling_price_l1843_184396


namespace child_tickets_sold_l1843_184336

theorem child_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_receipts ∧
    child_tickets = 90 := by
  sorry

end child_tickets_sold_l1843_184336


namespace rectangle_tiling_no_walls_l1843_184338

/-- A domino tiling of a rectangle. -/
def DominoTiling (m n : ℕ) := Unit

/-- A wall in a domino tiling. -/
def Wall (m n : ℕ) (tiling : DominoTiling m n) := Unit

/-- Predicate indicating if a tiling has no walls. -/
def HasNoWalls (m n : ℕ) (tiling : DominoTiling m n) : Prop :=
  ∀ w : Wall m n tiling, False

theorem rectangle_tiling_no_walls 
  (m n : ℕ) 
  (h_even : Even (m * n))
  (h_m : m ≥ 5)
  (h_n : n ≥ 5)
  (h_not_six : ¬(m = 6 ∧ n = 6)) :
  ∃ (tiling : DominoTiling m n), HasNoWalls m n tiling :=
sorry

end rectangle_tiling_no_walls_l1843_184338


namespace absolute_value_equation_product_l1843_184323

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 3 = 47 ∧ |5 * x₂| + 3 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1936/25) := by
  sorry

end absolute_value_equation_product_l1843_184323


namespace at_least_one_not_less_than_two_l1843_184305

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l1843_184305


namespace parabola_vertex_specific_parabola_vertex_l1843_184368

/-- The vertex of a parabola defined by y = a(x-h)^2 + k has coordinates (h, k) -/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x => a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k :=
by sorry

/-- The vertex of the parabola y = 3(x-5)^2 + 4 has coordinates (5, 4) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x => 3 * (x - 5)^2 + 4
  (∀ x, f x ≥ f 5) ∧ f 5 = 4 :=
by sorry

end parabola_vertex_specific_parabola_vertex_l1843_184368


namespace steves_gold_bars_l1843_184343

theorem steves_gold_bars (friends : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : 
  friends = 4 → lost_bars = 20 → bars_per_friend = 20 →
  friends * bars_per_friend + lost_bars = 100 := by
  sorry

end steves_gold_bars_l1843_184343


namespace equation_solution_l1843_184345

theorem equation_solution :
  ∀ x y : ℝ, 
    y ≠ 0 →
    (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1) →
    ((x = 0 ∧ y = 1/2) ∨ (x = 0 ∧ y = -1/2)) :=
by sorry

end equation_solution_l1843_184345


namespace tiling_colors_l1843_184362

/-- Represents the type of tiling: squares or hexagons -/
inductive TilingType
  | Squares
  | Hexagons

/-- Calculates the number of colors needed for a specific tiling type and grid parameters -/
def number_of_colors (t : TilingType) (k l : ℕ) : ℕ :=
  match t with
  | TilingType.Squares => k^2 + l^2
  | TilingType.Hexagons => k^2 + k*l + l^2

/-- Theorem stating the number of colors needed for a valid tiling -/
theorem tiling_colors (t : TilingType) (k l : ℕ) (h : k ≠ 0 ∨ l ≠ 0) :
  ∃ (n : ℕ), n = number_of_colors t k l ∧ n > 0 :=
by sorry

end tiling_colors_l1843_184362


namespace xiao_bing_winning_probability_l1843_184369

-- Define the game parameters
def dice_outcomes : ℕ := 6 * 6
def same_number_outcomes : ℕ := 6
def xiao_cong_score : ℕ := 10
def xiao_bing_score : ℕ := 2

-- Define the probabilities
def prob_same_numbers : ℚ := same_number_outcomes / dice_outcomes
def prob_different_numbers : ℚ := 1 - prob_same_numbers

-- Define the expected scores
def xiao_cong_expected_score : ℚ := prob_same_numbers * xiao_cong_score
def xiao_bing_expected_score : ℚ := prob_different_numbers * xiao_bing_score

-- Theorem: The probability of Xiao Bing winning is 1/2
theorem xiao_bing_winning_probability : 
  xiao_cong_expected_score = xiao_bing_expected_score → 
  (1 : ℚ) / 2 = prob_different_numbers := by
  sorry

end xiao_bing_winning_probability_l1843_184369


namespace sqrt_17_property_l1843_184331

theorem sqrt_17_property (a b : ℝ) : 
  (∀ x : ℤ, (x : ℝ) ≤ Real.sqrt 17 → (x + 1 : ℝ) > Real.sqrt 17 → a = x) →
  b = Real.sqrt 17 - a →
  b ^ 2020 * (a + Real.sqrt 17) ^ 2021 = Real.sqrt 17 + 4 := by
  sorry

end sqrt_17_property_l1843_184331


namespace school_costume_problem_l1843_184352

/-- Represents the price of a costume set based on the quantity purchased -/
def price (n : ℕ) : ℕ :=
  if n ≤ 45 then 60
  else if n ≤ 90 then 50
  else 40

/-- The problem statement -/
theorem school_costume_problem :
  ∃ (a b : ℕ),
    a + b = 92 ∧
    a > b ∧
    a < 90 ∧
    a * price a + b * price b = 5020 ∧
    a = 50 ∧
    b = 42 ∧
    92 * 40 = a * price a + b * price b - 480 :=
by
  sorry


end school_costume_problem_l1843_184352


namespace union_eq_univ_complement_inter_B_a_range_l1843_184385

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 9*x + 18 ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorems to be proved
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_B : (Aᶜ) ∩ B = {x : ℝ | 3 < x ∧ x < 6} := by sorry

theorem a_range (a : ℝ) : C a ⊆ B → -2 ≤ a ∧ a ≤ 8 := by sorry

end union_eq_univ_complement_inter_B_a_range_l1843_184385


namespace isosceles_triangles_remainder_l1843_184361

/-- The number of vertices in the regular polygon --/
def n : ℕ := 2019

/-- The number of isosceles triangles in a regular n-gon --/
def num_isosceles (n : ℕ) : ℕ := (n * (n - 1) / 2 : ℕ) - (2 * n / 3 : ℕ)

/-- The theorem stating that the remainder when the number of isosceles triangles
    in a regular 2019-gon is divided by 100 is equal to 25 --/
theorem isosceles_triangles_remainder :
  num_isosceles n % 100 = 25 := by sorry

end isosceles_triangles_remainder_l1843_184361


namespace right_triangle_area_l1843_184397

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters, its area is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) : 
  a = 13 → -- hypotenuse is 13 meters
  b = 5 → -- one side is 5 meters
  c^2 + b^2 = a^2 → -- Pythagorean theorem (right triangle condition)
  (1/2 : ℝ) * b * c = 30 := by -- area formula
sorry

end right_triangle_area_l1843_184397


namespace ab_difference_l1843_184376

theorem ab_difference (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a - b = 3 := by
  sorry

end ab_difference_l1843_184376


namespace negative_one_third_squared_l1843_184351

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end negative_one_third_squared_l1843_184351


namespace square_area_ratio_l1843_184316

/-- Given three squares with the specified relationships, prove that the ratio of the areas of the first and second squares is 1/2. -/
theorem square_area_ratio (s₃ : ℝ) (h₃ : s₃ > 0) : 
  let s₁ := s₃ * Real.sqrt 2
  let s₂ := s₁ * Real.sqrt 2
  (s₁^2) / (s₂^2) = 1/2 := by sorry

end square_area_ratio_l1843_184316


namespace inequality_theorem_l1843_184335

theorem inequality_theorem (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, (a / (2^x + 1)) > (b / (2^x + 1)) := by
sorry

end inequality_theorem_l1843_184335


namespace quadratic_function_property_l1843_184301

theorem quadratic_function_property (b c m n : ℝ) :
  let f := fun (x : ℝ) => x^2 + b*x + c
  (f m = n ∧ f (m + 1) = n) →
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 2 → f x₁ > f x₂) →
  m ≥ 3/2 :=
by sorry

end quadratic_function_property_l1843_184301


namespace subcommittee_formation_ways_l1843_184372

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_ways : 
  number_of_ways_to_form_subcommittee 10 8 4 3 = 11760 := by
  sorry

end subcommittee_formation_ways_l1843_184372


namespace figure_b_cannot_be_formed_l1843_184314

/-- A piece is represented by its width and height -/
structure Piece where
  width : ℕ
  height : ℕ

/-- A figure is represented by its width and height -/
structure Figure where
  width : ℕ
  height : ℕ

/-- The set of available pieces -/
def pieces : Finset Piece := sorry

/-- The set of figures to be formed -/
def figures : Finset Figure := sorry

/-- Function to check if a figure can be formed from the given pieces -/
def canFormFigure (p : Finset Piece) (f : Figure) : Prop := sorry

/-- Theorem stating that Figure B cannot be formed while others can -/
theorem figure_b_cannot_be_formed :
  ∃ (b : Figure),
    b ∈ figures ∧
    ¬(canFormFigure pieces b) ∧
    ∀ (f : Figure), f ∈ figures ∧ f ≠ b → canFormFigure pieces f :=
sorry

end figure_b_cannot_be_formed_l1843_184314


namespace rectangles_in_4x4_grid_l1843_184306

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  num_rectangles_4x4 = choose_2_from_4 * choose_2_from_4 :=
by sorry

end rectangles_in_4x4_grid_l1843_184306


namespace acute_triangle_special_angles_l1843_184395

theorem acute_triangle_special_angles :
  ∃ (α β γ : ℕ),
    α + β + γ = 180 ∧
    0 < γ ∧ γ < β ∧ β < α ∧ α < 90 ∧
    α = 5 * γ ∧
    (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

end acute_triangle_special_angles_l1843_184395


namespace fish_pond_population_l1843_184390

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 50 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#eval (50 * 50) / 2  -- Should evaluate to 1250

end fish_pond_population_l1843_184390


namespace function_composition_implies_sum_l1843_184382

/-- Given two functions f and g, where f(x) = ax + b and g(x) = 3x - 6,
    and the condition that g(f(x)) = 4x + 3 for all x,
    prove that a + b = 13/3 -/
theorem function_composition_implies_sum (a b : ℝ) :
  (∀ x, 3 * (a * x + b) - 6 = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end function_composition_implies_sum_l1843_184382


namespace intersection_complement_equality_l1843_184339

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end intersection_complement_equality_l1843_184339


namespace non_union_women_percentage_l1843_184310

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℝ
  men : ℝ
  unionized : ℝ
  unionized_men : ℝ

/-- Conditions given in the problem -/
def company_conditions (c : CompanyEmployees) : Prop :=
  c.men / c.total = 0.54 ∧
  c.unionized / c.total = 0.6 ∧
  c.unionized_men / c.unionized = 0.7

/-- The theorem to be proved -/
theorem non_union_women_percentage (c : CompanyEmployees) 
  (h : company_conditions c) : 
  (c.total - c.unionized - (c.men - c.unionized_men)) / (c.total - c.unionized) = 0.7 := by
  sorry

end non_union_women_percentage_l1843_184310


namespace sum_digits_base_8_999_l1843_184326

def base_8_representation (n : ℕ) : List ℕ := sorry

theorem sum_digits_base_8_999 : 
  (base_8_representation 999).sum = 19 := by sorry

end sum_digits_base_8_999_l1843_184326


namespace symmetry_implies_linear_plus_periodic_l1843_184329

/-- A function has two centers of symmetry if there exist two distinct points
    such that reflecting the graph through these points leaves it unchanged. -/
def has_two_centers_of_symmetry (f : ℝ → ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧
  ∀ (x y : ℝ), f y = x ↔ f (2 * C₁.1 - y) = 2 * C₁.2 - x ∧
                      f (2 * C₂.1 - y) = 2 * C₂.2 - x

/-- A function is the sum of a linear function and a periodic function if
    there exist real numbers b and a ≠ 0, and a periodic function g with period a,
    such that f(x) = bx + g(x) for all x. -/
def is_sum_of_linear_and_periodic (f : ℝ → ℝ) : Prop :=
  ∃ (b : ℝ) (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧
  (∀ x, g (x + a) = g x) ∧
  (∀ x, f x = b * x + g x)

/-- Theorem: If a function has two centers of symmetry,
    then it can be expressed as the sum of a linear function and a periodic function. -/
theorem symmetry_implies_linear_plus_periodic (f : ℝ → ℝ) :
  has_two_centers_of_symmetry f → is_sum_of_linear_and_periodic f := by
  sorry

end symmetry_implies_linear_plus_periodic_l1843_184329


namespace no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1843_184393

theorem no_three_primes_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  p₁ > 3 → p₂ > 3 → p₃ > 3 → 
  Nat.Prime p₁ → Nat.Prime p₂ → Nat.Prime p₃ → 
  d < 5 → 
  ¬(p₂ = p₁ + d ∧ p₃ = p₁ + 2*d) :=
sorry

theorem no_k_primes_arithmetic_progression (k : ℕ) (p : ℕ → ℕ) (d : ℕ) :
  k > 3 → 
  (∀ i, i ≤ k → p i > k) →
  (∀ i, i ≤ k → Nat.Prime (p i)) →
  d ≤ k + 1 →
  ¬(∀ i, i ≤ k → p i = p 1 + (i - 1) * d) :=
sorry

end no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1843_184393


namespace ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l1843_184380

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after_n_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => next_day (day_after_n_days start n)

theorem ten_to_ninety_mod_seven : 10^90 % 7 = 1 := by sorry

theorem day_after_ten_to_ninety (start : DayOfWeek) :
  day_after_n_days start (10^90) = next_day start := by sorry

theorem friday_after_ten_to_ninety_is_saturday :
  day_after_n_days DayOfWeek.Friday (10^90) = DayOfWeek.Saturday := by sorry

end ten_to_ninety_mod_seven_day_after_ten_to_ninety_friday_after_ten_to_ninety_is_saturday_l1843_184380


namespace largest_three_digit_divisible_by_digits_l1843_184337

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def tens_digit_less_than_5 (n : ℕ) : Prop := (n / 10) % 10 < 5

def divisible_by_its_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0 ∧
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧
  n % hundreds = 0 ∧ n % tens = 0 ∧ n % ones = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, is_three_digit n →
    tens_digit_less_than_5 n →
    divisible_by_its_digits n →
    n ≤ 936 :=
by sorry

end largest_three_digit_divisible_by_digits_l1843_184337


namespace log_function_unique_parameters_l1843_184320

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+b)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := log_base a (x + b)

-- State the theorem
theorem log_function_unique_parameters :
  ∀ a b : ℝ, a > 0 → a ≠ 1 →
  (f a b (-1) = 0 ∧ f a b 0 = 1) →
  (a = 2 ∧ b = 2) :=
by sorry

end log_function_unique_parameters_l1843_184320


namespace arithmetic_mean_of_three_numbers_l1843_184307

theorem arithmetic_mean_of_three_numbers (a b c : ℝ) (h : a = 14 ∧ b = 22 ∧ c = 36) :
  (a + b + c) / 3 = 24 := by
  sorry

end arithmetic_mean_of_three_numbers_l1843_184307


namespace find_number_of_elements_number_of_elements_is_ten_l1843_184317

/-- Given an incorrect average and a correction, find the number of elements -/
theorem find_number_of_elements (incorrect_avg correct_avg : ℚ) 
  (incorrect_value correct_value : ℚ) : ℚ :=
  let n := (correct_value - incorrect_value) / (correct_avg - incorrect_avg)
  n

/-- Proof that the number of elements is 10 given the specific conditions -/
theorem number_of_elements_is_ten : 
  find_number_of_elements 20 26 26 86 = 10 := by
  sorry

end find_number_of_elements_number_of_elements_is_ten_l1843_184317


namespace highest_score_l1843_184355

theorem highest_score (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_ineq : b + d > a + c)
  (a_gt_bc : a > b + c) :
  d > a ∧ d > b ∧ d > c := by
  sorry

end highest_score_l1843_184355


namespace bottles_per_case_is_ten_l1843_184373

/-- The number of bottles produced per day -/
def bottles_per_day : ℕ := 72000

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 7200

/-- The number of bottles that a case can hold -/
def bottles_per_case : ℕ := bottles_per_day / cases_per_day

theorem bottles_per_case_is_ten : bottles_per_case = 10 := by
  sorry

end bottles_per_case_is_ten_l1843_184373


namespace circle_symmetry_l1843_184328

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation
def symmetry (x y x' y' : ℝ) : Prop :=
  symmetry_line ((x + x') / 2) ((y + y') / 2) ∧ 
  (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2

-- State the theorem
theorem circle_symmetry :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), symmetry x y x' y' ∧ given_circle x' y') ↔
    x^2 + (y + 1)^2 = 1 :=
by sorry

end circle_symmetry_l1843_184328


namespace point_above_line_l1843_184313

theorem point_above_line (a : ℝ) : 
  (2*a - (-1) + 1 < 0) ↔ (a < -1) := by sorry

end point_above_line_l1843_184313


namespace fraction_product_theorem_l1843_184381

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 16 / 24 * 32 / 48 * 28 / 7 * 15 / 9 * 50 / 25 * 21 / 35 = 32 / 3 := by
  sorry

end fraction_product_theorem_l1843_184381


namespace betty_height_in_feet_l1843_184334

/-- Given the heights of Carter, his dog, and Betty, prove Betty's height in feet. -/
theorem betty_height_in_feet :
  ∀ (carter_height dog_height betty_height : ℕ),
    carter_height = 2 * dog_height →
    dog_height = 24 →
    betty_height = carter_height - 12 →
    betty_height / 12 = 3 := by
  sorry

end betty_height_in_feet_l1843_184334


namespace rectangle_overlap_theorem_l1843_184357

/-- A rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- A configuration of rectangles placed within a larger rectangle -/
structure Configuration where
  outer : Rectangle
  inner : List Rectangle

/-- Predicate to check if two rectangles overlap by at least a given area -/
def overlaps (r1 r2 : Rectangle) (min_overlap : ℝ) : Prop :=
  ∃ (overlap_area : ℝ), overlap_area ≥ min_overlap

theorem rectangle_overlap_theorem (config : Configuration) :
  config.outer.area = 5 →
  config.inner.length = 9 →
  ∀ r ∈ config.inner, r.area = 1 →
  ∃ (r1 r2 : Rectangle), r1 ∈ config.inner ∧ r2 ∈ config.inner ∧ r1 ≠ r2 ∧ overlaps r1 r2 (1/9) :=
by sorry

end rectangle_overlap_theorem_l1843_184357


namespace triangle_options_l1843_184371

/-- Represents a triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

theorem triangle_options (t : Triangle) :
  (t.b^2 = t.a^2 - t.c^2 → isRightAngled t) ∧
  (t.a / t.b = 3 / 4 ∧ t.a / t.c = 3 / 5 ∧ t.b / t.c = 4 / 5 → isRightAngled t) ∧
  (t.C = t.A - t.B → isRightAngled t) ∧
  (t.A / t.B = 3 / 4 ∧ t.A / t.C = 3 / 5 ∧ t.B / t.C = 4 / 5 → ¬isRightAngled t) :=
by sorry


end triangle_options_l1843_184371


namespace sufficient_not_necessary_condition_l1843_184378

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0) →
  (a + a * b < 0) ∧
  ∃ (x y : ℝ), x + x * y < 0 ∧ ¬(x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end sufficient_not_necessary_condition_l1843_184378


namespace function_always_positive_l1843_184340

theorem function_always_positive (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (∀ x, x < 1 ∨ x > 3) :=
sorry

end function_always_positive_l1843_184340


namespace work_completion_proof_l1843_184366

/-- The number of days Matt and Peter worked together -/
def days_worked_together : ℕ := 12

/-- The time (in days) it takes Matt and Peter to complete the work together -/
def time_together : ℕ := 20

/-- The time (in days) it takes Peter to complete the work alone -/
def time_peter_alone : ℕ := 35

/-- The time (in days) it takes Peter to complete the remaining work after Matt stops -/
def time_peter_remaining : ℕ := 14

theorem work_completion_proof :
  (days_worked_together : ℚ) / time_together + 
  time_peter_remaining / time_peter_alone = 1 := by sorry

end work_completion_proof_l1843_184366


namespace earnings_per_lawn_l1843_184348

theorem earnings_per_lawn (total_lawns forgotten_lawns : ℕ) (total_earnings : ℚ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earnings = 36 →
  total_earnings / (total_lawns - forgotten_lawns) = 9 := by
sorry

end earnings_per_lawn_l1843_184348


namespace fraction_simplification_l1843_184356

theorem fraction_simplification :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) = 1 / 4 := by
  sorry

end fraction_simplification_l1843_184356


namespace fraction_simplification_l1843_184374

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end fraction_simplification_l1843_184374


namespace chord_length_theorem_l1843_184318

theorem chord_length_theorem (R AB BC : ℝ) (h_R : R = 12) (h_AB : AB = 6) (h_BC : BC = 4) :
  ∃ (AC : ℝ), (AC = Real.sqrt 35 + Real.sqrt 15) ∨ (AC = Real.sqrt 35 - Real.sqrt 15) := by
  sorry

end chord_length_theorem_l1843_184318


namespace reflection_of_circle_center_l1843_184342

/-- Reflects a point (x, y) across the line y = -x -/
def reflect_across_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -7)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (7, -3)

theorem reflection_of_circle_center :
  reflect_across_y_neg_x original_center = expected_reflected_center :=
by sorry

end reflection_of_circle_center_l1843_184342


namespace instrument_probability_l1843_184367

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 16 / 100 := by
  sorry

end instrument_probability_l1843_184367


namespace max_value_of_f_l1843_184384

-- Define the function
def f (x : ℝ) : ℝ := x * (1 - 3 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (max_y : ℝ), max_y = 1/12 ∧
  ∀ (x : ℝ), 0 < x → x < 1/3 → f x ≤ max_y :=
sorry

end max_value_of_f_l1843_184384


namespace scientific_notation_of_113700_l1843_184332

theorem scientific_notation_of_113700 :
  (113700 : ℝ) = 1.137 * (10 ^ 5) := by
  sorry

end scientific_notation_of_113700_l1843_184332


namespace last_week_tv_hours_l1843_184388

/-- The number of hours of television watched last week -/
def last_week_hours : ℝ := sorry

/-- The average number of hours watched over three weeks -/
def average_hours : ℝ := 10

/-- The number of hours watched the week before last -/
def week_before_hours : ℝ := 8

/-- The number of hours to be watched next week -/
def next_week_hours : ℝ := 12

theorem last_week_tv_hours : last_week_hours = 10 :=
  by
    have h1 : (week_before_hours + last_week_hours + next_week_hours) / 3 = average_hours := by sorry
    sorry


end last_week_tv_hours_l1843_184388


namespace inscribed_sphere_volume_l1843_184365

/-- The volume of a sphere inscribed in a right circular cone with specific properties -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) : 
  let r := d / 4
  4 / 3 * π * r^3 = 288 * π := by
  sorry

end inscribed_sphere_volume_l1843_184365


namespace choose_three_from_ten_l1843_184309

theorem choose_three_from_ten (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose n k = 120 := by sorry

end choose_three_from_ten_l1843_184309


namespace investment_roi_difference_l1843_184387

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℕ := 2

theorem investment_roi_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end investment_roi_difference_l1843_184387


namespace candy_cookie_packs_l1843_184344

-- Define the problem parameters
def num_trays : ℕ := 4
def cookies_per_tray : ℕ := 24
def cookies_per_pack : ℕ := 12

-- Define the theorem
theorem candy_cookie_packs : 
  (num_trays * cookies_per_tray) / cookies_per_pack = 8 := by
  sorry

end candy_cookie_packs_l1843_184344


namespace specific_bike_ride_north_distance_l1843_184377

/-- Represents a bike ride with given distances and final position -/
structure BikeRide where
  west : ℝ
  initialNorth : ℝ
  east : ℝ
  finalDistance : ℝ

/-- Calculates the final northward distance after going east for a given bike ride -/
def finalNorthDistance (ride : BikeRide) : ℝ :=
  sorry

/-- Theorem stating that for the specific bike ride described, the final northward distance after going east is 15 miles -/
theorem specific_bike_ride_north_distance :
  let ride : BikeRide := {
    west := 8,
    initialNorth := 5,
    east := 4,
    finalDistance := 20.396078054371138
  }
  finalNorthDistance ride = 15 := by
  sorry

end specific_bike_ride_north_distance_l1843_184377


namespace cube_equation_solution_l1843_184300

theorem cube_equation_solution (x : ℝ) : (x + 3)^3 = -64 → x = -7 := by
  sorry

end cube_equation_solution_l1843_184300


namespace no_statements_imply_negation_l1843_184327

theorem no_statements_imply_negation (p q : Prop) : 
  ¬((p ∨ q) → ¬(p ∨ q)) ∧
  ¬((p ∨ ¬q) → ¬(p ∨ q)) ∧
  ¬((¬p ∨ q) → ¬(p ∨ q)) ∧
  ¬((¬p ∧ q) → ¬(p ∨ q)) := by
  sorry

end no_statements_imply_negation_l1843_184327


namespace equation_positive_root_l1843_184347

theorem equation_positive_root (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 / (x - 1) - k / (1 - x) = 1)) → k = -2 := by
  sorry

end equation_positive_root_l1843_184347
