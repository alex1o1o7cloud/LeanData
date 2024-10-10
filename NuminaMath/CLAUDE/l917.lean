import Mathlib

namespace quadratic_polynomial_root_l917_91750

theorem quadratic_polynomial_root (x : ℂ) : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 24 * z + 51
  (p (4 + I) = 0) ∧ (∀ z : ℂ, p z = 3 * z^2 + ((-24) * z + 51)) := by
  sorry

end quadratic_polynomial_root_l917_91750


namespace range_of_H_l917_91770

/-- The function H defined as the difference of absolute values -/
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

/-- Theorem stating the range of function H -/
theorem range_of_H :
  (∀ x : ℝ, H x ≥ -4 ∧ H x ≤ 4) ∧
  (∃ x : ℝ, H x = -4) ∧
  (∃ x : ℝ, H x = 4) :=
sorry

end range_of_H_l917_91770


namespace oranges_in_basket_l917_91756

/-- The number of oranges in a fruit basket -/
def num_oranges : ℕ := 6

/-- The number of apples in the fruit basket -/
def num_apples : ℕ := num_oranges - 2

/-- The number of bananas in the fruit basket -/
def num_bananas : ℕ := 3 * num_apples

/-- The number of peaches in the fruit basket -/
def num_peaches : ℕ := num_bananas / 2

/-- Theorem: The number of oranges in the fruit basket is 6 -/
theorem oranges_in_basket : 
  num_oranges + num_apples + num_bananas + num_peaches = 28 → num_oranges = 6 := by
  sorry


end oranges_in_basket_l917_91756


namespace subtraction_problem_l917_91780

theorem subtraction_problem : 3.609 - 2.5 - 0.193 = 0.916 := by sorry

end subtraction_problem_l917_91780


namespace diophantine_equation_solutions_l917_91787

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ 
    (x = -1 ∧ y = -9) ∨ 
    (x = 1 ∧ y = 5) ∨ 
    (x = 7 ∧ y = -97) := by
  sorry

end diophantine_equation_solutions_l917_91787


namespace race_time_proof_l917_91709

/-- A runner completes a race -/
structure Runner where
  distance : ℝ  -- distance covered
  time : ℝ      -- time taken

/-- A race between two runners -/
structure Race where
  length : ℝ           -- race length
  runner_a : Runner    -- runner A
  runner_b : Runner    -- runner B

/-- Given a race satisfying the problem conditions, prove that runner A's time is 7 seconds -/
theorem race_time_proof (race : Race) 
  (h1 : race.length = 200)
  (h2 : race.runner_a.distance - race.runner_b.distance = 35)
  (h3 : race.runner_a.distance = race.length)
  (h4 : race.runner_a.time = 7) :
  race.runner_a.time = 7 := by sorry

end race_time_proof_l917_91709


namespace tree_height_difference_l917_91739

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let birch_height : ℚ := 37/2
  birch_height - pine_height = 25/4 := by sorry

#check tree_height_difference

end tree_height_difference_l917_91739


namespace total_pencils_l917_91729

/-- The number of pencils Jessica, Sandy, and Jason have in total is 24, 
    given that each of them has 8 pencils. -/
theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l917_91729


namespace opposite_direction_speed_l917_91721

/-- Proves that given two people moving in opposite directions for 4 hours,
    with one person moving at 3 km/hr and the distance between them after 4 hours being 20 km,
    the speed of the other person is 2 km/hr. -/
theorem opposite_direction_speed
  (time : ℝ)
  (pooja_speed : ℝ)
  (distance : ℝ)
  (h1 : time = 4)
  (h2 : pooja_speed = 3)
  (h3 : distance = 20) :
  ∃ (other_speed : ℝ), other_speed = 2 ∧ distance = (other_speed + pooja_speed) * time :=
sorry

end opposite_direction_speed_l917_91721


namespace lisa_marbles_problem_l917_91730

def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) :
  min_additional_marbles num_friends initial_marbles = 28 := by
  sorry

end lisa_marbles_problem_l917_91730


namespace new_person_weight_l917_91781

theorem new_person_weight (W : ℝ) (new_weight : ℝ) :
  (W + new_weight - 25) / 12 = W / 12 + 3 →
  new_weight = 61 :=
by
  sorry

end new_person_weight_l917_91781


namespace even_expression_l917_91767

theorem even_expression (x : ℤ) (h : x = 3) : 
  ∃ k : ℤ, 2 * (x^2 + 9) = 2 * k := by
  sorry

end even_expression_l917_91767


namespace hexagon_side_sum_l917_91772

/-- A polygon with six vertices -/
structure Hexagon :=
  (P Q R S T U : ℝ × ℝ)

/-- The area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For a hexagon PQRSTU with area 40, PQ = 6, QR = 7, and TU = 4, ST + TU = 7 -/
theorem hexagon_side_sum (h : Hexagon) 
  (h_area : area h = 40)
  (h_PQ : distance h.P h.Q = 6)
  (h_QR : distance h.Q h.R = 7)
  (h_TU : distance h.T h.U = 4) :
  distance h.S h.T + distance h.T h.U = 7 := by sorry

end hexagon_side_sum_l917_91772


namespace card_selection_count_l917_91759

theorem card_selection_count (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) n : ℚ) = (Nat.factorial (2 * n)) / ((Nat.factorial n) * (Nat.factorial n)) :=
by sorry

end card_selection_count_l917_91759


namespace gum_sharing_theorem_l917_91769

def john_gum : ℝ := 54.5
def cole_gum : ℝ := 45.75
def aubrey_gum : ℝ := 37
def maria_gum : ℝ := 70.25
def liam_gum : ℝ := 28.5
def emma_gum : ℝ := 32.5

def total_people : ℕ := 6

def total_gum : ℝ := 2 * (john_gum + cole_gum + aubrey_gum + maria_gum + liam_gum + emma_gum)

theorem gum_sharing_theorem : 
  total_gum / total_people = 89.5 := by
  sorry

end gum_sharing_theorem_l917_91769


namespace inverse_proportion_problem_l917_91789

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * (-5) = k → x = -8/5 := by
sorry

end inverse_proportion_problem_l917_91789


namespace gas_price_calculation_l917_91774

theorem gas_price_calculation (rental_cost mileage_rate total_miles total_expense gas_gallons : ℚ)
  (h1 : rental_cost = 150)
  (h2 : mileage_rate = 1/2)
  (h3 : total_miles = 320)
  (h4 : total_expense = 338)
  (h5 : gas_gallons = 8) :
  (total_expense - rental_cost - mileage_rate * total_miles) / gas_gallons = 7/2 := by
  sorry

#eval (338 : ℚ) - 150 - 1/2 * 320
#eval ((338 : ℚ) - 150 - 1/2 * 320) / 8

end gas_price_calculation_l917_91774


namespace circle_radius_increase_l917_91778

/-- Given a circle with radius r, prove that when the radius is increased by 5 and the area is quadrupled, the original radius was 5 and the new perimeter is 20π. -/
theorem circle_radius_increase (r : ℝ) : 
  (π * (r + 5)^2 = 4 * π * r^2) → 
  (r = 5 ∧ 2 * π * (r + 5) = 20 * π) := by
  sorry

end circle_radius_increase_l917_91778


namespace special_function_properties_l917_91712

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x

theorem special_function_properties
  (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 12) :
  (f 0 = 4) ∧
  (∀ a : ℝ, (∃ x₀ : ℝ, 1 < x₀ ∧ x₀ < 4 ∧ f x₀ - 8 = a * x₀) ↔ -1 < a ∧ a < 5) :=
by sorry

end special_function_properties_l917_91712


namespace complex_modulus_product_l917_91773

theorem complex_modulus_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end complex_modulus_product_l917_91773


namespace customers_left_l917_91728

theorem customers_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 33 → new = 26 → final = 28 → initial - (initial - new + final) = 31 := by
sorry

end customers_left_l917_91728


namespace irregular_shape_impossible_l917_91737

/-- Represents a shape formed by two equilateral triangles --/
structure TwoTriangleShape where
  -- Add necessary fields to describe the shape

/-- Predicate to check if a shape is regular (has symmetry or regularity) --/
def is_regular (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition of regularity

/-- Predicate to check if a shape can be formed by two equilateral triangles --/
def can_be_formed_by_triangles (s : TwoTriangleShape) : Prop :=
  sorry  -- Definition based on triangle placement rules

theorem irregular_shape_impossible (s : TwoTriangleShape) :
  ¬(is_regular s) → ¬(can_be_formed_by_triangles s) :=
  sorry  -- The proof would go here

end irregular_shape_impossible_l917_91737


namespace rebecca_eggs_count_l917_91711

theorem rebecca_eggs_count (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
  sorry

end rebecca_eggs_count_l917_91711


namespace function_multiple_preimages_l917_91701

theorem function_multiple_preimages :
  ∃ (f : ℝ → ℝ) (y : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = y ∧ f x₂ = y := by
  sorry

end function_multiple_preimages_l917_91701


namespace digits_of_two_power_fifteen_times_five_power_ten_l917_91723

theorem digits_of_two_power_fifteen_times_five_power_ten : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by
  sorry

end digits_of_two_power_fifteen_times_five_power_ten_l917_91723


namespace min_value_theorem_min_value_achievable_l917_91754

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -3 ∧ 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end min_value_theorem_min_value_achievable_l917_91754


namespace factorization_proof_l917_91765

theorem factorization_proof (x : ℝ) : 18 * x^3 + 12 * x^2 = 6 * x^2 * (3 * x + 2) := by
  sorry

end factorization_proof_l917_91765


namespace triangle_inequality_bounds_l917_91708

theorem triangle_inequality_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (htri : a + b > c) :
  4 ≤ (a + b + c)^2 / (b * c) ∧ (a + b + c)^2 / (b * c) ≤ 9 := by
  sorry

end triangle_inequality_bounds_l917_91708


namespace chocolate_eggs_problem_l917_91749

theorem chocolate_eggs_problem (egg_weight : ℕ) (num_boxes : ℕ) (remaining_weight : ℕ) : 
  egg_weight = 10 →
  num_boxes = 4 →
  remaining_weight = 90 →
  ∃ (total_eggs : ℕ), 
    total_eggs = num_boxes * (remaining_weight / (egg_weight * (num_boxes - 1))) ∧
    total_eggs = 12 := by
sorry

end chocolate_eggs_problem_l917_91749


namespace equation_solution_l917_91724

theorem equation_solution :
  ∃ s : ℚ, (s - 60) / 3 = (6 - 3 * s) / 4 ∧ s = 258 / 13 := by
  sorry

end equation_solution_l917_91724


namespace cricket_matches_count_l917_91725

theorem cricket_matches_count (total_average : ℝ) (first_four_average : ℝ) (last_three_average : ℝ) :
  total_average = 56 →
  first_four_average = 46 →
  last_three_average = 69.33333333333333 →
  ∃ (n : ℕ), n = 7 ∧ n * total_average = 4 * first_four_average + 3 * last_three_average :=
by sorry

end cricket_matches_count_l917_91725


namespace trig_identity_l917_91719

theorem trig_identity (a : ℝ) (h : Real.sin (π * Real.cos a) = Real.cos (π * Real.sin a)) :
  35 * (Real.sin (2 * a))^2 + 84 * (Real.cos (4 * a))^2 = 21 := by
  sorry

end trig_identity_l917_91719


namespace circle_whisper_game_l917_91777

theorem circle_whisper_game (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 18 →
  a + b = 16 →
  b + c = 12 →
  e + f = 8 →
  d = 6 := by
sorry

end circle_whisper_game_l917_91777


namespace boxes_with_no_items_l917_91784

/-- Given the following conditions:
  - There are 15 boxes in total
  - 8 boxes contain pencils
  - 5 boxes contain pens
  - 3 boxes contain markers
  - 4 boxes contain both pens and pencils
  - 1 box contains all three items (pencils, pens, and markers)
  Prove that the number of boxes containing neither pens, pencils, nor markers is 5. -/
theorem boxes_with_no_items (total : ℕ) (pencil : ℕ) (pen : ℕ) (marker : ℕ) 
  (pen_and_pencil : ℕ) (all_three : ℕ) :
  total = 15 →
  pencil = 8 →
  pen = 5 →
  marker = 3 →
  pen_and_pencil = 4 →
  all_three = 1 →
  total - (pen_and_pencil + (pencil - pen_and_pencil) + 
    (pen - pen_and_pencil) + (marker - all_three)) = 5 :=
by sorry

end boxes_with_no_items_l917_91784


namespace least_common_multiple_first_ten_l917_91722

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ ¬(k ∣ m)) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l917_91722


namespace system_solution_l917_91771

theorem system_solution (x y z : ℤ) : 
  (x^2 = y*z + 1 ∧ y^2 = z*x + 1 ∧ z^2 = x*y + 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = 0) ∨
   (x = 0 ∧ y = 1 ∧ z = -1) ∨
   (x = 0 ∧ y = -1 ∧ z = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 0) ∨
   (x = -1 ∧ y = 0 ∧ z = 1)) :=
by sorry

end system_solution_l917_91771


namespace systematic_sampling_theorem_l917_91761

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startNumber : ℕ
  hTotalPositive : 0 < totalStudents
  hSamplePositive : 0 < sampleSize
  hStartValid : startNumber ≤ totalStudents

/-- Generates the sequence of selected student numbers -/
def generateSequence (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating that the systematic sampling generates the expected sequence -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : s.startNumber = 3) :
  generateSequence s = [3, 13, 23, 33, 43] := by
  sorry

end systematic_sampling_theorem_l917_91761


namespace smallest_of_five_consecutive_even_integers_l917_91740

theorem smallest_of_five_consecutive_even_integers : 
  ∃ (n : ℕ), 
    (5 * n + 20 = 30 * 31) ∧ 
    (∀ m : ℕ, m < n → ¬(5 * m + 20 = 30 * 31)) ∧
    (n % 2 = 0) :=
by sorry

end smallest_of_five_consecutive_even_integers_l917_91740


namespace constant_molecular_weight_l917_91744

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 792

/-- The number of moles of the acid -/
def num_moles : ℝ := 9

/-- Theorem stating that the molecular weight remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  ∀ n : ℝ, n > 0 → molecular_weight = molecular_weight := by sorry

end constant_molecular_weight_l917_91744


namespace range_of_m_given_quadratic_inequality_l917_91776

theorem range_of_m_given_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end range_of_m_given_quadratic_inequality_l917_91776


namespace ascending_six_digit_numbers_count_l917_91753

/-- The number of six-digit natural numbers with digits in ascending order -/
def ascending_six_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem ascending_six_digit_numbers_count : ascending_six_digit_numbers = 84 := by
  sorry

end ascending_six_digit_numbers_count_l917_91753


namespace af_equals_kc_l917_91757

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
variable (O : ℝ × ℝ)  -- Center of the circle
variable (G H E D B A C F K : ℝ × ℝ)

-- Define the circle
variable (circle : Circle)

-- Define conditions
variable (gh_diameter : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 4 * circle.radius^2)
variable (ed_diameter : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 4 * circle.radius^2)
variable (perpendicular_diameters : (G.1 - H.1) * (E.1 - D.1) + (G.2 - H.2) * (E.2 - D.2) = 0)
variable (b_outside : (B.1 - circle.center.1)^2 + (B.2 - circle.center.2)^2 > circle.radius^2)
variable (a_on_circle : (A.1 - circle.center.1)^2 + (A.2 - circle.center.2)^2 = circle.radius^2)
variable (c_on_circle : (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2)
variable (a_on_gh : A.2 = G.2 ∧ A.2 = H.2)
variable (c_on_gh : C.2 = G.2 ∧ C.2 = H.2)
variable (f_on_gh : F.2 = G.2 ∧ F.2 = H.2)
variable (k_on_gh : K.2 = G.2 ∧ K.2 = H.2)
variable (ba_tangent : (B.1 - A.1) * (A.1 - circle.center.1) + (B.2 - A.2) * (A.2 - circle.center.2) = 0)
variable (bc_tangent : (B.1 - C.1) * (C.1 - circle.center.1) + (B.2 - C.2) * (C.2 - circle.center.2) = 0)
variable (be_intersects_gh_at_f : (B.1 - E.1) * (F.2 - B.2) = (F.1 - B.1) * (B.2 - E.2))
variable (bd_intersects_gh_at_k : (B.1 - D.1) * (K.2 - B.2) = (K.1 - B.1) * (B.2 - D.2))

-- Theorem statement
theorem af_equals_kc : (A.1 - F.1)^2 + (A.2 - F.2)^2 = (K.1 - C.1)^2 + (K.2 - C.2)^2 := by sorry

end af_equals_kc_l917_91757


namespace solution_set_part1_range_of_a_part2_l917_91726

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Icc (1/2) 1, f x a ≤ |2*x + 1|) →
  (0 ≤ a ∧ a ≤ 3) :=
sorry

end solution_set_part1_range_of_a_part2_l917_91726


namespace sentences_at_start_l917_91763

-- Define the typing rate
def typing_rate : ℕ := 6

-- Define the typing durations
def first_session : ℕ := 20
def second_session : ℕ := 15
def third_session : ℕ := 18

-- Define the number of erased sentences
def erased_sentences : ℕ := 40

-- Define the total number of sentences at the end of the day
def total_sentences : ℕ := 536

-- Theorem to prove
theorem sentences_at_start : 
  total_sentences - (typing_rate * (first_session + second_session + third_session) - erased_sentences) = 258 :=
by sorry

end sentences_at_start_l917_91763


namespace no_y_intercepts_l917_91786

theorem no_y_intercepts (y : ℝ) : ¬ ∃ y, 3 * y^2 - y + 4 = 0 := by
  sorry

end no_y_intercepts_l917_91786


namespace interest_rate_beyond_five_years_l917_91743

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_beyond_five_years 
  (principal : ℝ)
  (rate_first_two_years : ℝ)
  (rate_next_three_years : ℝ)
  (total_interest : ℝ)
  (h1 : principal = 12000)
  (h2 : rate_first_two_years = 0.06)
  (h3 : rate_next_three_years = 0.09)
  (h4 : total_interest = 11400)
  : ∃ (rate_beyond_five_years : ℝ),
    rate_beyond_five_years = 0.14 ∧
    total_interest = 
      simple_interest principal rate_first_two_years 2 +
      simple_interest principal rate_next_three_years 3 +
      simple_interest principal rate_beyond_five_years 4 :=
sorry

end interest_rate_beyond_five_years_l917_91743


namespace binary_conversion_l917_91736

-- Define the binary number
def binary_num : List Nat := [1, 0, 1, 1, 0, 0, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bin : List Nat) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Define the function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 89 ∧
  decimal_to_octal (binary_to_decimal binary_num) = [1, 3, 1] := by
  sorry

end binary_conversion_l917_91736


namespace pentagon_area_condition_l917_91788

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon given its vertices -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_area_condition (y : ℝ) : 
  let p := Pentagon.mk (0, 0) (0, 5) (3, y) (6, 5) (6, 0)
  hasVerticalSymmetry p ∧ pentagonArea p = 50 → y = 35/3 := by
  sorry

end pentagon_area_condition_l917_91788


namespace nine_more_knives_l917_91732

/-- Represents the number of each type of cutlery -/
structure Cutlery where
  forks : ℕ
  knives : ℕ
  spoons : ℕ
  teaspoons : ℕ

/-- The initial state of the cutlery drawer -/
def initial : Cutlery :=
  { forks := 6
  , knives := 6 + 9  -- We're proving this 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- The state after adding 2 of each cutlery -/
def after_adding (c : Cutlery) : Cutlery :=
  { forks := c.forks + 2
  , knives := c.knives + 2
  , spoons := c.spoons + 2
  , teaspoons := c.teaspoons + 2 }

/-- The total number of cutlery pieces -/
def total (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Main theorem: There are 9 more knives than forks initially -/
theorem nine_more_knives :
  initial.knives = initial.forks + 9 ∧
  initial.spoons = 2 * initial.knives ∧
  initial.teaspoons = initial.forks / 2 ∧
  total (after_adding initial) = 62 :=
by sorry

end nine_more_knives_l917_91732


namespace mean_median_difference_l917_91710

def frequency_distribution : List (ℕ × ℕ) := [
  (0, 2), (1, 3), (2, 4), (3, 5), (4, 3), (5, 1)
]

def total_students : ℕ := 18

def median (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

def mean (fd : List (ℕ × ℕ)) (total : ℕ) : ℚ :=
  sorry

theorem mean_median_difference :
  let m := mean frequency_distribution total_students
  let med := median frequency_distribution total_students
  |m - med| = 11 / 18 := by
  sorry

end mean_median_difference_l917_91710


namespace gerald_remaining_money_l917_91702

/-- Represents the cost of items and currency conversions --/
structure Costs where
  meat_pie : ℕ
  sausage_roll : ℕ
  farthings_per_pfennig : ℕ
  pfennigs_per_groat : ℕ
  groats_per_florin : ℕ

/-- Represents Gerald's initial money --/
structure GeraldMoney where
  farthings : ℕ
  groats : ℕ
  florins : ℕ

/-- Calculates the remaining pfennigs after purchase --/
def remaining_pfennigs (c : Costs) (m : GeraldMoney) : ℕ :=
  let total_pfennigs := 
    m.farthings / c.farthings_per_pfennig +
    m.groats * c.pfennigs_per_groat +
    m.florins * c.groats_per_florin * c.pfennigs_per_groat
  total_pfennigs - (c.meat_pie + c.sausage_roll)

/-- Theorem stating Gerald's remaining pfennigs --/
theorem gerald_remaining_money (c : Costs) (m : GeraldMoney) 
  (h1 : c.meat_pie = 120)
  (h2 : c.sausage_roll = 75)
  (h3 : m.farthings = 54)
  (h4 : m.groats = 8)
  (h5 : m.florins = 17)
  (h6 : c.farthings_per_pfennig = 6)
  (h7 : c.pfennigs_per_groat = 4)
  (h8 : c.groats_per_florin = 10) :
  remaining_pfennigs c m = 526 := by
  sorry

end gerald_remaining_money_l917_91702


namespace smallest_number_l917_91714

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : Nat := to_decimal [8, 5] 9
def num2 : Nat := to_decimal [2, 1, 0] 6
def num3 : Nat := to_decimal [1, 0, 0, 0] 4
def num4 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

-- Theorem statement
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end smallest_number_l917_91714


namespace probability_two_red_balls_l917_91764

/-- The probability of selecting 2 red balls from a bag containing 6 red, 4 blue, and 2 green balls -/
theorem probability_two_red_balls (red : ℕ) (blue : ℕ) (green : ℕ) 
  (h_red : red = 6) (h_blue : blue = 4) (h_green : green = 2) : 
  (Nat.choose red 2 : ℚ) / (Nat.choose (red + blue + green) 2) = 5 / 22 :=
sorry

end probability_two_red_balls_l917_91764


namespace power_product_equals_eight_l917_91733

theorem power_product_equals_eight (m n : ℤ) (h : 2 * m + n - 3 = 0) :
  (4 : ℝ) ^ m * (2 : ℝ) ^ n = 8 := by
  sorry

end power_product_equals_eight_l917_91733


namespace super_bowl_probability_sum_l917_91768

theorem super_bowl_probability_sum :
  ∀ (p_play p_not_play : ℝ),
  p_play = 9 * p_not_play →
  p_play ≥ 0 →
  p_not_play ≥ 0 →
  p_play + p_not_play = 1 :=
by
  sorry

end super_bowl_probability_sum_l917_91768


namespace bag_of_balls_l917_91779

/-- Given a bag of balls, prove that the total number of balls is 15 -/
theorem bag_of_balls (total_balls : ℕ) 
  (prob_red : ℚ) 
  (num_red : ℕ) : 
  prob_red = 1/5 → 
  num_red = 3 → 
  total_balls = 15 :=
by
  sorry

end bag_of_balls_l917_91779


namespace size_relationship_l917_91727

theorem size_relationship (a b c : ℝ) 
  (ha : a = Real.sqrt 3)
  (hb : b = Real.sqrt 15 - Real.sqrt 7)
  (hc : c = Real.sqrt 11 - Real.sqrt 3) :
  a > c ∧ c > b := by
  sorry

end size_relationship_l917_91727


namespace pencil_count_l917_91717

/-- Given an initial number of pencils and a number of pencils added, 
    calculate the total number of pencils after addition. -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that given 33 initial pencils and 27 added pencils, 
    the total number of pencils is 60. -/
theorem pencil_count : total_pencils 33 27 = 60 := by
  sorry

end pencil_count_l917_91717


namespace marion_ella_score_ratio_l917_91735

/-- Prove that the ratio of Marion's score to Ella's score is 2:3 -/
theorem marion_ella_score_ratio :
  let total_items : ℕ := 40
  let ella_incorrect : ℕ := 4
  let marion_score : ℕ := 24
  let ella_score : ℕ := total_items - ella_incorrect
  (marion_score : ℚ) / ella_score = 2 / 3 := by
  sorry

end marion_ella_score_ratio_l917_91735


namespace total_trees_planted_l917_91798

/-- The total number of trees planted by a family in spring -/
theorem total_trees_planted (apricot peach cherry : ℕ) : 
  apricot = 58 →
  peach = 3 * apricot →
  cherry = 5 * peach →
  apricot + peach + cherry = 1102 := by
  sorry

end total_trees_planted_l917_91798


namespace cone_base_radius_l917_91799

/-- Given a cone formed from a sector with arc length 8π, its base radius is 4. -/
theorem cone_base_radius (cone : Real) (sector : Real) :
  (sector = 8 * Real.pi) →    -- arc length of sector
  (sector = 2 * Real.pi * cone) →    -- arc length equals circumference of base
  (cone = 4) :=    -- radius of base
by sorry

end cone_base_radius_l917_91799


namespace remainder_theorem_polynomial_division_remainder_l917_91755

def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 15 * x - 20

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = divisor x * q x + 10 := by sorry

end remainder_theorem_polynomial_division_remainder_l917_91755


namespace probability_at_least_one_male_l917_91752

/-- The probability of selecting at least one male out of 3 contestants from a group of 8 finalists (5 females and 3 males) is 23/28. -/
theorem probability_at_least_one_male (total : ℕ) (females : ℕ) (males : ℕ) (selected : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  selected = 3 →
  (Nat.choose total selected - Nat.choose females selected : ℚ) / Nat.choose total selected = 23 / 28 := by
  sorry

#eval (Nat.choose 8 3 - Nat.choose 5 3 : ℚ) / Nat.choose 8 3 == 23 / 28

end probability_at_least_one_male_l917_91752


namespace smallest_representable_numbers_l917_91748

def is_representable (c : ℕ) : Prop :=
  ∃ m n : ℕ, c = 7 * m^2 - 11 * n^2

theorem smallest_representable_numbers :
  (is_representable 1 ∧ is_representable 5) ∧
  (∀ c : ℕ, c < 1 → ¬is_representable c) ∧
  (∀ c : ℕ, 1 < c → c < 5 → ¬is_representable c) :=
sorry

end smallest_representable_numbers_l917_91748


namespace total_money_l917_91700

/-- The total amount of money A, B, and C have together is 500, given the specified conditions. -/
theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 330 → c = 30 → a + b + c = 500 := by
  sorry

end total_money_l917_91700


namespace smallest_difference_fractions_l917_91792

theorem smallest_difference_fractions :
  ∃ (x y a b : ℤ),
    (0 < x) ∧ (x < 8) ∧ (0 < y) ∧ (y < 13) ∧
    (0 < a) ∧ (a < 8) ∧ (0 < b) ∧ (b < 13) ∧
    (x / 8 ≠ y / 13) ∧ (a / 8 ≠ b / 13) ∧
    (|x / 8 - y / 13| = |13 * x - 8 * y| / 104) ∧
    (|a / 8 - b / 13| = |13 * a - 8 * b| / 104) ∧
    (|13 * x - 8 * y| = 1) ∧ (|13 * a - 8 * b| = 1) ∧
    ∀ (p q : ℤ), (0 < p) → (p < 8) → (0 < q) → (q < 13) → (p / 8 ≠ q / 13) →
      |p / 8 - q / 13| ≥ |x / 8 - y / 13| ∧
      |p / 8 - q / 13| ≥ |a / 8 - b / 13| :=
by
  sorry

#check smallest_difference_fractions

end smallest_difference_fractions_l917_91792


namespace sin_double_angle_minus_pi_half_l917_91703

/-- Given an angle α in the Cartesian coordinate system with the specified properties,
    prove that sin(2α - π/2) = -1/2 -/
theorem sin_double_angle_minus_pi_half (α : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = -1 ∧ 
   x * Real.cos α = x ∧ x * Real.sin α = y) →
  Real.sin (2 * α - π / 2) = -1 / 2 := by sorry

end sin_double_angle_minus_pi_half_l917_91703


namespace union_and_subset_l917_91747

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < 1 + 3*m}

-- Define the complement of A
def A_complement : Set ℝ := {x | x ≤ -1 ∨ x > 3}

theorem union_and_subset :
  (∀ m : ℝ, m = 1 → A ∪ B m = {x | -1 < x ∧ x < 4}) ∧
  (∀ m : ℝ, B m ⊆ A_complement ↔ m ≤ -1/2 ∨ m > 3) :=
sorry

end union_and_subset_l917_91747


namespace basketball_pricing_solution_l917_91791

/-- Represents the cost and pricing of basketballs --/
structure BasketballPricing where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for remaining A brand basketballs

/-- Theorem stating the correct pricing and discount for the basketball problem --/
theorem basketball_pricing_solution (p : BasketballPricing) : 
  (40 * p.cost_a + 40 * p.cost_b = 7200) →
  (50 * p.cost_a + 30 * p.cost_b = 7400) →
  (p.price_a = 140) →
  (p.markup_b = 0.3) →
  (40 * (p.price_a - p.cost_a) + 10 * (p.price_a * (1 - p.discount_a / 100) - p.cost_a) + 30 * p.cost_b * p.markup_b = 2440) →
  (p.cost_a = 100 ∧ p.cost_b = 80 ∧ p.discount_a = 20) := by
  sorry

end basketball_pricing_solution_l917_91791


namespace angle_of_inclination_negative_slope_one_l917_91796

/-- The angle of inclination of a line given by the equation x + y + 3 = 0 is 3π/4 -/
theorem angle_of_inclination_negative_slope_one (x y : ℝ) :
  x + y + 3 = 0 → Real.arctan (-1) = 3 * Real.pi / 4 := by
  sorry

end angle_of_inclination_negative_slope_one_l917_91796


namespace min_a_value_l917_91790

theorem min_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 9 * x + y = x * y) :
  ∃ (a : ℝ), a > 0 ∧ (∀ (x y : ℝ), x > 0 → y > 0 → a * x + y ≥ 25) ∧
  (∀ (b : ℝ), b > 0 → (∀ (x y : ℝ), x > 0 → y > 0 → b * x + y ≥ 25) → b ≥ a) ∧
  a = 4 :=
sorry

end min_a_value_l917_91790


namespace parallel_lines_a_value_l917_91738

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, (a + 1) * x - y + 2 = 0 ↔ x + (a - 1) * y - 1 = 0) → 
  a = 0 := by
  sorry

end parallel_lines_a_value_l917_91738


namespace max_value_cos_sin_l917_91734

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end max_value_cos_sin_l917_91734


namespace factorization_equality_l917_91782

theorem factorization_equality (x : ℝ) :
  (x - 1)^4 + x * (2*x + 1) * (2*x - 1) + 5*x = (x^2 + 3 + 2*Real.sqrt 2) * (x^2 + 3 - 2*Real.sqrt 2) := by
  sorry

end factorization_equality_l917_91782


namespace min_value_theorem_l917_91716

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + 3 * y = 8) :
  (2 / x + 3 / y) ≥ 25 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + 3 * y = 8 ∧ 2 / x + 3 / y = 25 / 8 := by
  sorry

end min_value_theorem_l917_91716


namespace original_triangle_area_l917_91760

theorem original_triangle_area (original_side : ℝ) (new_side : ℝ) (new_area : ℝ) :
  new_side = 5 * original_side →
  new_area = 125 →
  (original_side^2 * Real.sqrt 3) / 4 = 5 := by
  sorry

end original_triangle_area_l917_91760


namespace vector_addition_l917_91741

/-- Given two vectors OA and AB in R², prove that OB = OA + AB -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end vector_addition_l917_91741


namespace bruce_fruit_purchase_cost_l917_91715

/-- Calculates the total cost of Bruce's fruit purchase in US dollars -/
def fruit_purchase_cost (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (apples_kg : ℝ) (apples_price : ℝ)
  (grapes_discount : ℝ) (mangoes_tax : ℝ) (oranges_premium : ℝ)
  (euro_to_usd : ℝ) (pound_to_usd : ℝ) (yen_to_usd : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price * (1 - grapes_discount)
  let mangoes_cost := mangoes_kg * mangoes_price * euro_to_usd * (1 + mangoes_tax)
  let oranges_cost := oranges_kg * oranges_price * pound_to_usd * (1 + oranges_premium)
  let apples_cost := apples_kg * apples_price * yen_to_usd
  grapes_cost + mangoes_cost + oranges_cost + apples_cost

/-- Theorem stating that Bruce's fruit purchase cost is $1563.10 -/
theorem bruce_fruit_purchase_cost :
  fruit_purchase_cost 8 70 8 55 5 40 10 3000 0.1 0.05 0.03 1.15 1.25 0.009 = 1563.10 := by
  sorry

end bruce_fruit_purchase_cost_l917_91715


namespace convergence_and_bound_l917_91704

def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 3 * u k - 3 * (u k)^2

theorem convergence_and_bound :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1/2| < ε) ∧
  (∀ k < 9, |u k - 1/2| > 1/2^500) ∧
  |u 9 - 1/2| ≤ 1/2^500 :=
sorry

end convergence_and_bound_l917_91704


namespace pentagon_perimeter_calculation_l917_91783

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (FG GH HI IJ JF : ℝ) : ℝ := FG + GH + HI + IJ + JF

/-- Theorem: The perimeter of pentagon FGHIJ is 7 + 2√5 -/
theorem pentagon_perimeter_calculation :
  pentagon_perimeter 2 2 (Real.sqrt 5) (Real.sqrt 5) 3 = 7 + 2 * Real.sqrt 5 := by
  sorry

end pentagon_perimeter_calculation_l917_91783


namespace janes_tulip_bulbs_l917_91742

theorem janes_tulip_bulbs :
  ∀ (T : ℕ),
    (T + T / 2 + 30 + 90 = 150) →
    T = 20 := by
  sorry

end janes_tulip_bulbs_l917_91742


namespace max_value_equation_l917_91793

theorem max_value_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 9 * x * y = p * (p + 3 * x + 6 * y)) :
  p^2 + x^2 + y^2 ≤ 29 ∧ ∃ (p' x' y' : ℕ), 
    Nat.Prime p' ∧ x' > 0 ∧ y' > 0 ∧ 
    9 * x' * y' = p' * (p' + 3 * x' + 6 * y') ∧
    p'^2 + x'^2 + y'^2 = 29 :=
by sorry


end max_value_equation_l917_91793


namespace geometric_sequence_sixth_term_l917_91762

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = -3)
  (h_4 : a 4 = 6) :
  a 6 = 24 := by
  sorry

end geometric_sequence_sixth_term_l917_91762


namespace proposition_false_iff_a_in_range_l917_91795

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end proposition_false_iff_a_in_range_l917_91795


namespace monotonic_decreasing_implies_a_leq_one_l917_91751

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1

theorem monotonic_decreasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) →
  a ≤ 1 := by
  sorry

end monotonic_decreasing_implies_a_leq_one_l917_91751


namespace perpendicular_lines_coefficient_l917_91705

/-- Given two lines in the xy-plane, this theorem proves that if they are perpendicular,
    then the coefficient 'a' in the first line's equation must equal 2/3. -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → 
   ((-1 : ℝ) / a) * (-2 / 3) = -1) →
  a = 2 / 3 := by
sorry


end perpendicular_lines_coefficient_l917_91705


namespace all_x_greater_than_two_l917_91731

theorem all_x_greater_than_two : ∀ x ∈ Set.Ioo 0 π, x + 1 / Real.sin x > 2 := by sorry

end all_x_greater_than_two_l917_91731


namespace mrs_heine_items_l917_91706

/-- The number of items Mrs. Heine will buy for her dogs -/
def total_items (num_dogs : ℕ) (biscuits_per_dog : ℕ) (boots_per_set : ℕ) : ℕ :=
  num_dogs * (biscuits_per_dog + boots_per_set)

/-- Proof that Mrs. Heine will buy 18 items -/
theorem mrs_heine_items : total_items 2 5 4 = 18 := by
  sorry

end mrs_heine_items_l917_91706


namespace remainder_7325_div_11_l917_91797

theorem remainder_7325_div_11 : 7325 % 11 = 6 := by
  sorry

end remainder_7325_div_11_l917_91797


namespace sum_of_even_integers_l917_91713

theorem sum_of_even_integers (first last : ℕ) (n : ℕ) (sum : ℕ) : 
  first = 202 →
  last = 300 →
  n = 50 →
  sum = 12550 →
  (last - first) / 2 + 1 = n →
  sum = n / 2 * (first + last) :=
by sorry

end sum_of_even_integers_l917_91713


namespace perpendicular_vectors_magnitude_l917_91746

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the magnitude of b is √5 -/
theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = 2 → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end perpendicular_vectors_magnitude_l917_91746


namespace fifteenth_digit_is_one_l917_91766

/-- The decimal representation of 1/9 as a sequence of digits after the decimal point -/
def decimal_1_9 : ℕ → ℕ
  | n => 1

/-- The decimal representation of 1/11 as a sequence of digits after the decimal point -/
def decimal_1_11 : ℕ → ℕ
  | n => if n % 3 = 0 then 0 else 9

/-- The sum of the decimal representations of 1/9 and 1/11 as a sequence of digits after the decimal point -/
def sum_decimals : ℕ → ℕ
  | n => (decimal_1_9 n + decimal_1_11 n) % 10

theorem fifteenth_digit_is_one :
  sum_decimals 14 = 1 := by sorry

end fifteenth_digit_is_one_l917_91766


namespace sum_area_ABC_DEF_l917_91720

-- Define the points and lengths
variable (A B C D E F G : ℝ × ℝ)
variable (AB BG GE DE : ℝ)

-- Define the areas of triangles
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
axiom AB_length : AB = 2
axiom BG_length : BG = 3
axiom GE_length : GE = 4
axiom DE_length : DE = 5

axiom sum_area_BCG_EFG : area_triangle B C G + area_triangle E F G = 24
axiom sum_area_AGF_CDG : area_triangle A G F + area_triangle C D G = 51

-- State the theorem
theorem sum_area_ABC_DEF :
  area_triangle A B C + area_triangle D E F = 23 :=
sorry

end sum_area_ABC_DEF_l917_91720


namespace min_value_product_l917_91785

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 := by
  sorry

end min_value_product_l917_91785


namespace candy_problem_l917_91707

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 4) - 5
  day2_remaining = 10 →
  initial_candies = 84 := by
sorry

end candy_problem_l917_91707


namespace four_fixed_points_iff_c_in_range_l917_91758

/-- A quadratic function f(x) = x^2 - cx + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - c*x + c

/-- The composition of f with itself -/
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate for f ∘ f having four distinct fixed points -/
def has_four_distinct_fixed_points (c : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f_comp_f c x₁ = x₁ ∧ f_comp_f c x₂ = x₂ ∧ f_comp_f c x₃ = x₃ ∧ f_comp_f c x₄ = x₄

theorem four_fixed_points_iff_c_in_range :
  ∀ c : ℝ, has_four_distinct_fixed_points c ↔ (c < -1 ∨ c > 3) :=
sorry

end four_fixed_points_iff_c_in_range_l917_91758


namespace max_value_of_sum_products_l917_91794

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 40 → 
  a * b + b * c + c * d + d * a ≤ 800 := by
sorry

end max_value_of_sum_products_l917_91794


namespace remainder_theorem_l917_91718

theorem remainder_theorem : 
  10002000400080016003200640128025605121024204840968192 % 100020004000800160032 = 40968192 := by
  sorry

end remainder_theorem_l917_91718


namespace smallest_shift_for_scaled_function_l917_91775

/-- A function with period 15 -/
def isPeriodic15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

/-- The property we want to prove -/
def hasShiftProperty (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

theorem smallest_shift_for_scaled_function 
  (f : ℝ → ℝ) (h : isPeriodic15 f) :
  (∃ b > 0, hasShiftProperty f b) ∧ 
  (∀ b > 0, hasShiftProperty f b → b ≥ 45) :=
sorry

end smallest_shift_for_scaled_function_l917_91775


namespace distance_on_line_l917_91745

/-- Given two points (a, b) and (c, d) on the line x + y = px + q,
    prove that the distance between them is |a-c|√(1 + (p-1)²) -/
theorem distance_on_line (p q a b c d : ℝ) :
  (a + b = p * a + q) →
  (c + d = p * c + q) →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + (p - 1)^2) := by
  sorry

end distance_on_line_l917_91745
