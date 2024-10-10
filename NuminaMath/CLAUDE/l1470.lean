import Mathlib

namespace factorization_problems_l1470_147065

theorem factorization_problems :
  (∀ m : ℝ, m * (m - 3) + 3 * (3 - m) = (m - 3)^2) ∧
  (∀ x : ℝ, 4 * x^3 - 12 * x^2 + 9 * x = x * (2 * x - 3)^2) := by
  sorry

end factorization_problems_l1470_147065


namespace remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l1470_147035

theorem remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen :
  (13^13 + 13) % 14 = 12 := by
  sorry

end remainder_thirteen_power_thirteen_plus_thirteen_mod_fourteen_l1470_147035


namespace symmetry_axis_implies_a_equals_one_l1470_147045

/-- The line equation -/
def line_equation (x y a : ℝ) : Prop := x - 2*a*y - 3 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The theorem stating that if the line is a symmetry axis of the circle, then a = 1 -/
theorem symmetry_axis_implies_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, line_equation x y a → circle_equation x y) →
  (line_equation (circle_center.1) (circle_center.2) a) →
  a = 1 :=
by sorry

end symmetry_axis_implies_a_equals_one_l1470_147045


namespace log_product_simplification_l1470_147031

theorem log_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x / Real.log (y^6)) * (Real.log (y^2) / Real.log (x^5)) *
  (Real.log (x^3) / Real.log (y^4)) * (Real.log (y^4) / Real.log (x^3)) *
  (Real.log (x^5) / Real.log (y^2)) = (1/6) * (Real.log x / Real.log y) := by
  sorry

end log_product_simplification_l1470_147031


namespace sequence_gcd_property_l1470_147068

/-- Given a sequence of natural numbers satisfying the GCD property, prove that a_i = i for all i. -/
theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i :=
by sorry

end sequence_gcd_property_l1470_147068


namespace expression_value_l1470_147003

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end expression_value_l1470_147003


namespace number_pair_problem_l1470_147039

theorem number_pair_problem (a b : ℕ) : 
  a + b = 62 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 25 ∨ b = 25) → 
  (a = 37 ∨ b = 37) :=
by sorry

end number_pair_problem_l1470_147039


namespace problem_1_problem_2_l1470_147071

-- Problem 1
theorem problem_1 : Real.sqrt 27 / Real.sqrt 3 + Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 5 = 5 - Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end problem_1_problem_2_l1470_147071


namespace davids_chemistry_marks_l1470_147062

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 76)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 75)
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 67 := by
sorry

end davids_chemistry_marks_l1470_147062


namespace negation_of_proposition_l1470_147095

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, Real.exp x ≤ 1)) ↔ (∃ x : ℝ, Real.exp x > 1) := by
  sorry

end negation_of_proposition_l1470_147095


namespace amy_soup_count_l1470_147094

/-- The number of chicken soup cans Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of tomato soup cans Amy bought -/
def tomato_soup : ℕ := 3

/-- The total number of soup cans Amy bought -/
def total_soup : ℕ := chicken_soup + tomato_soup

theorem amy_soup_count : total_soup = 9 := by
  sorry

end amy_soup_count_l1470_147094


namespace clinic_cats_count_l1470_147006

theorem clinic_cats_count (dog_cost cat_cost dog_count total_cost : ℕ) 
  (h1 : dog_cost = 60)
  (h2 : cat_cost = 40)
  (h3 : dog_count = 20)
  (h4 : total_cost = 3600)
  : ∃ cat_count : ℕ, dog_cost * dog_count + cat_cost * cat_count = total_cost ∧ cat_count = 60 := by
  sorry

end clinic_cats_count_l1470_147006


namespace right_triangle_and_inverse_mod_l1470_147042

theorem right_triangle_and_inverse_mod : 
  (60^2 + 144^2 = 156^2) ∧ 
  (∃ n : ℕ, n < 3751 ∧ (300 * n) % 3751 = 1) :=
by sorry

end right_triangle_and_inverse_mod_l1470_147042


namespace moores_law_2010_l1470_147007

def transistor_count (year : ℕ) : ℕ :=
  if year ≤ 2000 then
    2000000 * 2^((year - 1992) / 2)
  else
    2000000 * 2^4 * 4^((year - 2000) / 2)

theorem moores_law_2010 : transistor_count 2010 = 32768000000 := by
  sorry

end moores_law_2010_l1470_147007


namespace reciprocal_inequality_l1470_147021

theorem reciprocal_inequality (a b : ℝ) :
  (a > b ∧ a * b > 0 → 1 / a < 1 / b) ∧
  (a > b ∧ a * b < 0 → 1 / a > 1 / b) := by
  sorry

end reciprocal_inequality_l1470_147021


namespace greatest_non_expressible_as_sum_of_composites_l1470_147087

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬(Nat.Prime n)

-- Define the property of being expressible as the sum of two composite numbers
def ExpressibleAsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ (a b : ℕ), IsComposite a ∧ IsComposite b ∧ n = a + b

-- State the theorem
theorem greatest_non_expressible_as_sum_of_composites :
  (∀ n > 11, ExpressibleAsSumOfTwoComposites n) ∧
  ¬(ExpressibleAsSumOfTwoComposites 11) := by sorry

end greatest_non_expressible_as_sum_of_composites_l1470_147087


namespace coffee_milk_ratio_result_l1470_147029

/-- Represents the coffee and milk consumption problem -/
def coffee_milk_ratio (thermos_capacity : ℚ) (fills_per_day : ℕ) (school_days : ℕ) 
  (coffee_reduction_factor : ℚ) (current_coffee_per_week : ℚ) : Prop :=
  let total_capacity_per_week := thermos_capacity * fills_per_day * school_days
  let previous_coffee_per_week := current_coffee_per_week / coffee_reduction_factor
  let milk_per_week := total_capacity_per_week - previous_coffee_per_week
  let milk_per_fill := milk_per_week / (fills_per_day * school_days)
  (milk_per_fill : ℚ) / thermos_capacity = 1 / 5

/-- The main theorem stating the ratio of milk to thermos capacity -/
theorem coffee_milk_ratio_result : 
  coffee_milk_ratio 20 2 5 (1/4) 40 := by sorry

end coffee_milk_ratio_result_l1470_147029


namespace wrapping_paper_area_l1470_147088

/-- A rectangular box with dimensions l, w, and h, wrapped with a square sheet of paper -/
structure Box where
  l : ℝ  -- length
  w : ℝ  -- width
  h : ℝ  -- height
  l_gt_w : l > w

/-- The square sheet of wrapping paper -/
structure WrappingPaper where
  side : ℝ  -- side length of the square sheet

/-- The wrapping configuration -/
structure WrappingConfig (box : Box) (paper : WrappingPaper) where
  centered : Bool  -- box is centered on the paper
  vertices_on_midlines : Bool  -- vertices of longer side on paper midlines
  corners_meet_at_top : Bool  -- unoccupied corners meet at top center

theorem wrapping_paper_area (box : Box) (paper : WrappingPaper) 
    (config : WrappingConfig box paper) : paper.side^2 = 4 * box.l^2 := by
  sorry

#check wrapping_paper_area

end wrapping_paper_area_l1470_147088


namespace initial_tickets_l1470_147098

/-- 
Given that a person spends some tickets and has some tickets left,
this theorem proves the total number of tickets they initially had.
-/
theorem initial_tickets (spent : ℕ) (left : ℕ) : 
  spent = 3 → left = 8 → spent + left = 11 := by
  sorry

end initial_tickets_l1470_147098


namespace arithmetic_mean_reciprocals_l1470_147010

theorem arithmetic_mean_reciprocals : 
  let numbers := [2, 3, 7, 11]
  let reciprocals := numbers.map (λ x => 1 / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 493 / 1848 := by
  sorry

end arithmetic_mean_reciprocals_l1470_147010


namespace negative_quarter_to_11_times_negative_four_to_12_l1470_147070

theorem negative_quarter_to_11_times_negative_four_to_12 :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end negative_quarter_to_11_times_negative_four_to_12_l1470_147070


namespace inverse_of_three_mod_191_l1470_147033

theorem inverse_of_three_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by
  sorry

end inverse_of_three_mod_191_l1470_147033


namespace hall_width_proof_l1470_147005

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 17 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 60 →
  total_cost = 57000 →
  ∃ w : ℝ, (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧ w = 17 :=
by sorry

end hall_width_proof_l1470_147005


namespace abc_sum_sqrt_l1470_147049

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end abc_sum_sqrt_l1470_147049


namespace zero_is_natural_number_zero_not_natural_is_false_l1470_147057

-- Define the set of natural numbers including 0
def NaturalNumbers : Set ℕ := {n : ℕ | True}

-- State the theorem
theorem zero_is_natural_number : (0 : ℕ) ∈ NaturalNumbers := by
  sorry

-- Prove that the statement "0 is not a natural number" is false
theorem zero_not_natural_is_false : ¬(0 ∉ NaturalNumbers) := by
  sorry

end zero_is_natural_number_zero_not_natural_is_false_l1470_147057


namespace complex_modulus_l1470_147076

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_modulus_l1470_147076


namespace tangent_line_equation_chord_line_equation_l1470_147086

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define the point P
def P : ℝ × ℝ := (-2, 0)

-- Define a line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define tangent line condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ line_through_P k x y ∧
  ∀ (x' y' : ℝ), C x' y' ∧ line_through_P k x' y' → (x', y') = (x, y)

-- Define chord length condition
def has_chord_length (k : ℝ) (len : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧
  line_through_P k x₁ y₁ ∧ line_through_P k x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = len^2

-- Theorem for part (1)
theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k ↔ (k = 0 ∨ 3 * k = 4) :=
sorry

-- Theorem for part (2)
theorem chord_line_equation :
  ∀ k : ℝ, has_chord_length k (2 * Real.sqrt 2) ↔ (k = 1 ∨ k = 7) :=
sorry

end tangent_line_equation_chord_line_equation_l1470_147086


namespace wild_animal_picture_difference_l1470_147013

/-- The number of wild animal pictures Ralph has -/
def ralph_wild_animals : ℕ := 58

/-- The number of wild animal pictures Derrick has -/
def derrick_wild_animals : ℕ := 76

/-- Theorem stating the difference in wild animal pictures between Derrick and Ralph -/
theorem wild_animal_picture_difference :
  derrick_wild_animals - ralph_wild_animals = 18 := by sorry

end wild_animal_picture_difference_l1470_147013


namespace taxi_fare_distance_l1470_147009

/-- Represents the fare structure and total charge for a taxi ride -/
structure TaxiFare where
  initialCharge : ℚ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℚ  -- Charge for each additional 1/5 mile
  totalCharge : ℚ  -- Total charge for the ride

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def calculateDistance (fare : TaxiFare) : ℚ :=
  let additionalDistance := (fare.totalCharge - fare.initialCharge) / fare.additionalCharge
  (additionalDistance + 1) / 5

/-- Theorem stating that for the given fare structure and total charge, the ride distance is 8 miles -/
theorem taxi_fare_distance (fare : TaxiFare) 
    (h1 : fare.initialCharge = 280/100)
    (h2 : fare.additionalCharge = 40/100)
    (h3 : fare.totalCharge = 1840/100) : 
  calculateDistance fare = 8 := by
  sorry

#eval calculateDistance { initialCharge := 280/100, additionalCharge := 40/100, totalCharge := 1840/100 }

end taxi_fare_distance_l1470_147009


namespace betta_fish_guppies_l1470_147059

/-- The number of guppies eaten by each betta fish per day -/
def guppies_per_betta : ℕ := sorry

/-- The number of guppies eaten by the moray eel per day -/
def moray_guppies : ℕ := 20

/-- The number of betta fish -/
def num_bettas : ℕ := 5

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := 55

theorem betta_fish_guppies :
  guppies_per_betta = 7 ∧
  moray_guppies + num_bettas * guppies_per_betta = total_guppies :=
sorry

end betta_fish_guppies_l1470_147059


namespace average_difference_l1470_147025

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 6

-- Define the class enrollments
def class_enrollments : List ℕ := [40, 30, 30, 10, 5, 5]

-- Define t (average number of students per teacher)
def t : ℚ := (num_students : ℚ) / num_teachers

-- Define s (average number of students per student)
def s : ℚ := (List.sum (List.map (λ x => x * x) class_enrollments) : ℚ) / num_students

-- Theorem to prove
theorem average_difference : t - s = -29/3 := by
  sorry

end average_difference_l1470_147025


namespace trains_catch_up_catch_up_at_ten_pm_l1470_147063

/-- The time (in hours after 3:00 pm) when the second train catches the first train -/
def catch_up_time : ℝ := 7

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 70

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 80

/-- The time difference between the trains' departure times in hours -/
def time_difference : ℝ := 1

theorem trains_catch_up : 
  speed_train1 * (catch_up_time + time_difference) = speed_train2 * catch_up_time := by
  sorry

theorem catch_up_at_ten_pm : 
  catch_up_time = 7 := by
  sorry

end trains_catch_up_catch_up_at_ten_pm_l1470_147063


namespace number_equation_l1470_147055

theorem number_equation (x : ℝ) (n : ℝ) : x = 32 → (35 - (n - (15 - x)) = 12 * 2 / (1 / 2)) ↔ n = -30 := by
  sorry

end number_equation_l1470_147055


namespace smallest_positive_solution_sqrt_equation_l1470_147050

theorem smallest_positive_solution_sqrt_equation :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt (3 * x) - (5 * x - 1)
  ∃! x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end smallest_positive_solution_sqrt_equation_l1470_147050


namespace leahs_coins_value_l1470_147043

theorem leahs_coins_value (d n : ℕ) : 
  d + n = 15 ∧ 
  d = 2 * (n + 3) → 
  10 * d + 5 * n = 135 :=
by sorry

end leahs_coins_value_l1470_147043


namespace increasing_quadratic_implies_a_bound_l1470_147080

/-- Given a quadratic function f(x) = 2x^2 - 4(1-a)x + 1, 
    if f is increasing on [3,+∞), then a ≥ -2 -/
theorem increasing_quadratic_implies_a_bound (a : ℝ) : 
  (∀ x ≥ 3, ∀ y ≥ x, (2*y^2 - 4*(1-a)*y + 1) ≥ (2*x^2 - 4*(1-a)*x + 1)) →
  a ≥ -2 :=
by sorry

end increasing_quadratic_implies_a_bound_l1470_147080


namespace simplify_expression_l1470_147084

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a * (a^3)^(1/2))^(1/3) = a^(7/6) := by sorry

end simplify_expression_l1470_147084


namespace popsicle_sticks_left_l1470_147077

/-- Calculates the number of popsicle sticks Miss Davis has left after distribution -/
theorem popsicle_sticks_left (initial_sticks : ℕ) (sticks_per_group : ℕ) (num_groups : ℕ) : 
  initial_sticks = 170 → sticks_per_group = 15 → num_groups = 10 → 
  initial_sticks - (sticks_per_group * num_groups) = 20 := by
sorry

end popsicle_sticks_left_l1470_147077


namespace calculate_train_speed_goods_train_speed_l1470_147038

/-- Calculates the speed of a train given the speed of another train traveling in the opposite direction, the length of the train, and the time it takes to pass. -/
theorem calculate_train_speed (speed_a : ℝ) (length_b : ℝ) (pass_time : ℝ) : ℝ :=
  let speed_a_ms := speed_a * 1000 / 3600
  let relative_speed := length_b / pass_time
  let speed_b_ms := relative_speed - speed_a_ms
  let speed_b_kmh := speed_b_ms * 3600 / 1000
  speed_b_kmh

/-- Proves that given a train A traveling at 50 km/h and a goods train B of length 280 m passing train A in the opposite direction in 9 seconds, the speed of train B is approximately 62 km/h. -/
theorem goods_train_speed : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |calculate_train_speed 50 280 9 - 62| < ε :=
sorry

end calculate_train_speed_goods_train_speed_l1470_147038


namespace max_M_value_l1470_147044

def J (k : ℕ) : ℕ := 10^(k+2) + 100

def M (k : ℕ) : ℕ := (J k).factorization 2

theorem max_M_value :
  ∃ (k : ℕ), k > 0 ∧ M k = 4 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 4 :=
sorry

end max_M_value_l1470_147044


namespace average_marks_math_chem_l1470_147037

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 10 → 
  (math + chem) / 2 = 35 := by
sorry

end average_marks_math_chem_l1470_147037


namespace delores_remaining_money_l1470_147022

/-- Calculates the remaining money after purchasing a computer and printer -/
def remaining_money (initial_amount computer_cost printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

/-- Theorem: Given the specific amounts, the remaining money is $10 -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end delores_remaining_money_l1470_147022


namespace smallest_n_value_l1470_147020

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  is_even c →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  n ≥ 501 ∧ ∃ (a' b' c' m' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 2010 ∧
    is_even c' ∧
    a'.factorial * b'.factorial * c'.factorial = m' * (10 ^ 501) ∧
    ¬(10 ∣ m') :=
by sorry

end smallest_n_value_l1470_147020


namespace determinant_special_matrix_l1470_147008

open Matrix

theorem determinant_special_matrix (x : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x, x; x, x + 2, x; x, x, x + 2]
  det A = 16 * (x + 1) := by
  sorry

end determinant_special_matrix_l1470_147008


namespace feeding_sequences_count_l1470_147089

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : Nat := 4

/-- Represents the constraint of alternating genders when feeding -/
def alternating_genders : Bool := true

/-- Represents the condition of starting with a specific male animal -/
def starts_with_male : Bool := true

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : Nat :=
  (num_pairs) * (num_pairs - 1) * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 2)

/-- Theorem stating that the number of possible feeding sequences is 144 -/
theorem feeding_sequences_count :
  alternating_genders ∧ starts_with_male → feeding_sequences = 144 := by
  sorry

end feeding_sequences_count_l1470_147089


namespace triangle_side_value_l1470_147040

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 3, c = 2√3, and bsinA = acos(B + π/6), then b = √3 -/
theorem triangle_side_value (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  c = 2 * Real.sqrt 3 →
  b * Real.sin A = a * Real.cos (B + π/6) →
  b = Real.sqrt 3 := by
sorry

end triangle_side_value_l1470_147040


namespace rectangle_frame_area_l1470_147060

theorem rectangle_frame_area (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  a * b = ((a + 2) * (b + 2) - a * b) → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end rectangle_frame_area_l1470_147060


namespace min_sum_squares_l1470_147015

theorem min_sum_squares (x y : ℝ) (h : (x - 1)^2 + y^2 = 16) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a - 1)^2 + b^2 = 16 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 9 := by
  sorry

end min_sum_squares_l1470_147015


namespace computation_proof_l1470_147054

theorem computation_proof : 
  20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1 / 5 := by
  sorry

end computation_proof_l1470_147054


namespace intersection_of_A_and_B_l1470_147075

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l1470_147075


namespace binomial_product_l1470_147024

/-- The product of (2x² + 3y - 4) and (y + 6) is equal to 2x²y + 12x² + 3y² + 14y - 24 -/
theorem binomial_product (x y : ℝ) :
  (2 * x^2 + 3 * y - 4) * (y + 6) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 := by
  sorry

end binomial_product_l1470_147024


namespace elder_person_age_l1470_147041

/-- Proves that given two persons whose ages differ by 16 years, and 6 years ago the elder one was 3 times as old as the younger one, the present age of the elder person is 30 years. -/
theorem elder_person_age (y e : ℕ) : 
  e = y + 16 → 
  e - 6 = 3 * (y - 6) → 
  e = 30 :=
by sorry

end elder_person_age_l1470_147041


namespace school_purchase_options_l1470_147047

theorem school_purchase_options : 
  let valid_purchase := λ (x y : ℕ) => x ≥ 8 ∧ y ≥ 2 ∧ 120 * x + 140 * y ≤ 1500
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    S.card = n ∧ 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_purchase p.1 p.2) ∧
    n = 5 :=
by sorry

end school_purchase_options_l1470_147047


namespace intersection_when_m_2_B_subset_A_iff_l1470_147018

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}

-- Part 1: Intersection when m = 2
theorem intersection_when_m_2 : A ∩ B 2 = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_iff (m : ℝ) : B m ⊆ A ↔ m ≤ 7/3 := by
  sorry

end intersection_when_m_2_B_subset_A_iff_l1470_147018


namespace garrison_reinforcement_departure_reinforcement_left_after_27_days_l1470_147058

/-- Represents the problem of determining when reinforcements left a garrison --/
theorem garrison_reinforcement_departure (initial_men : ℕ) (initial_days : ℕ) 
  (departed_men : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_men := initial_men - departed_men
  let x := (total_provisions - remaining_men * remaining_days) / initial_men
  x

/-- Proves that the reinforcements left after 27 days given the problem conditions --/
theorem reinforcement_left_after_27_days : 
  garrison_reinforcement_departure 400 31 200 8 = 27 := by
  sorry

end garrison_reinforcement_departure_reinforcement_left_after_27_days_l1470_147058


namespace smallest_angle_in_special_triangle_l1470_147028

theorem smallest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b + c = 180 →
    b = 3 * a →
    c = 5 * a →
    a = 20 := by
  sorry

end smallest_angle_in_special_triangle_l1470_147028


namespace expo_artworks_arrangements_l1470_147052

/-- Represents the number of artworks of each type -/
structure ArtworkCounts where
  calligraphy : Nat
  paintings : Nat
  architectural : Nat

/-- Calculates the number of arrangements for the given artwork counts -/
def arrangeArtworks (counts : ArtworkCounts) : Nat :=
  sorry

/-- The specific artwork counts for the problem -/
def expoArtworks : ArtworkCounts :=
  { calligraphy := 2, paintings := 2, architectural := 1 }

/-- Theorem stating that the number of arrangements for the expo artworks is 36 -/
theorem expo_artworks_arrangements :
  arrangeArtworks expoArtworks = 36 := by
  sorry

end expo_artworks_arrangements_l1470_147052


namespace weekly_sales_equals_63_l1470_147002

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def avg_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * avg_daily_sales

theorem weekly_sales_equals_63 : total_weekly_sales = 63 := by
  sorry

end weekly_sales_equals_63_l1470_147002


namespace printer_equation_l1470_147092

theorem printer_equation (y : ℝ) : y > 0 →
  (300 : ℝ) / 6 + 300 / y = 300 / 3 ↔ 1 / 6 + 1 / y = 1 / 3 := by
  sorry

end printer_equation_l1470_147092


namespace sum_of_roots_greater_than_five_l1470_147034

theorem sum_of_roots_greater_than_five (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 5 := by
  sorry

end sum_of_roots_greater_than_five_l1470_147034


namespace suitcase_electronics_weight_l1470_147074

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 6) = 2 * (B / C)) -- Ratio doubles after removing 6 pounds of clothes
  : E = 9 := by
  sorry

end suitcase_electronics_weight_l1470_147074


namespace school_sections_l1470_147036

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 192) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 25 := by
sorry

end school_sections_l1470_147036


namespace angle_of_inclination_sqrt3_l1470_147072

/-- The angle of inclination of a line with slope √3 is 60°. -/
theorem angle_of_inclination_sqrt3 :
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  Real.tan angle = slope := by sorry

end angle_of_inclination_sqrt3_l1470_147072


namespace mean_correction_l1470_147014

def correct_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean - wrong_value + correct_value

theorem mean_correction (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : correct_value = 45) :
  (correct_mean n original_mean wrong_value correct_value) / n = 36.44 := by
  sorry

end mean_correction_l1470_147014


namespace arithmetic_sequence_sum_l1470_147001

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end arithmetic_sequence_sum_l1470_147001


namespace sarah_sells_more_than_tamara_l1470_147016

/-- Represents the bake sale competition between Tamara and Sarah -/
structure BakeSale where
  -- Tamara's baked goods
  tamara_brownie_pans : ℕ
  tamara_cookie_trays : ℕ
  tamara_brownie_pieces_per_pan : ℕ
  tamara_cookie_pieces_per_tray : ℕ
  tamara_small_brownie_price : ℚ
  tamara_large_brownie_price : ℚ
  tamara_cookie_price : ℚ
  tamara_small_brownies_sold : ℕ

  -- Sarah's baked goods
  sarah_cupcake_batches : ℕ
  sarah_muffin_dozens : ℕ
  sarah_cupcakes_per_batch : ℕ
  sarah_chocolate_cupcake_price : ℚ
  sarah_vanilla_cupcake_price : ℚ
  sarah_strawberry_cupcake_price : ℚ
  sarah_muffin_price : ℚ
  sarah_chocolate_cupcakes_sold : ℕ
  sarah_vanilla_cupcakes_sold : ℕ

/-- Calculates the total sales for Tamara -/
def tamara_total_sales (bs : BakeSale) : ℚ :=
  let total_brownies := bs.tamara_brownie_pans * bs.tamara_brownie_pieces_per_pan
  let large_brownies_sold := total_brownies - bs.tamara_small_brownies_sold
  let total_cookies := bs.tamara_cookie_trays * bs.tamara_cookie_pieces_per_tray
  bs.tamara_small_brownies_sold * bs.tamara_small_brownie_price +
  large_brownies_sold * bs.tamara_large_brownie_price +
  total_cookies * bs.tamara_cookie_price

/-- Calculates the total sales for Sarah -/
def sarah_total_sales (bs : BakeSale) : ℚ :=
  let total_cupcakes := bs.sarah_cupcake_batches * bs.sarah_cupcakes_per_batch
  let strawberry_cupcakes_sold := total_cupcakes - bs.sarah_chocolate_cupcakes_sold - bs.sarah_vanilla_cupcakes_sold
  let total_muffins := bs.sarah_muffin_dozens * 12
  total_muffins * bs.sarah_muffin_price +
  bs.sarah_chocolate_cupcakes_sold * bs.sarah_chocolate_cupcake_price +
  bs.sarah_vanilla_cupcakes_sold * bs.sarah_vanilla_cupcake_price +
  strawberry_cupcakes_sold * bs.sarah_strawberry_cupcake_price

/-- Theorem stating the difference in sales between Sarah and Tamara -/
theorem sarah_sells_more_than_tamara (bs : BakeSale) :
  bs.tamara_brownie_pans = 2 ∧
  bs.tamara_cookie_trays = 3 ∧
  bs.tamara_brownie_pieces_per_pan = 8 ∧
  bs.tamara_cookie_pieces_per_tray = 12 ∧
  bs.tamara_small_brownie_price = 2 ∧
  bs.tamara_large_brownie_price = 3 ∧
  bs.tamara_cookie_price = 3/2 ∧
  bs.tamara_small_brownies_sold = 4 ∧
  bs.sarah_cupcake_batches = 3 ∧
  bs.sarah_muffin_dozens = 2 ∧
  bs.sarah_cupcakes_per_batch = 10 ∧
  bs.sarah_chocolate_cupcake_price = 5/2 ∧
  bs.sarah_vanilla_cupcake_price = 2 ∧
  bs.sarah_strawberry_cupcake_price = 11/4 ∧
  bs.sarah_muffin_price = 7/4 ∧
  bs.sarah_chocolate_cupcakes_sold = 7 ∧
  bs.sarah_vanilla_cupcakes_sold = 8 →
  sarah_total_sales bs - tamara_total_sales bs = 75/4 := by
  sorry

end sarah_sells_more_than_tamara_l1470_147016


namespace dislike_radio_and_music_l1470_147046

theorem dislike_radio_and_music (total_people : ℕ) 
  (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) :
  total_people = 1500 →
  radio_dislike_percent = 25 / 100 →
  music_dislike_percent = 15 / 100 →
  ⌊(total_people : ℚ) * radio_dislike_percent * music_dislike_percent⌋ = 56 :=
by sorry

end dislike_radio_and_music_l1470_147046


namespace pat_calculation_error_l1470_147097

theorem pat_calculation_error (x : ℝ) : 
  (x / 7 - 20 = 13) → (7 * x + 20 > 1100) := by
  sorry

end pat_calculation_error_l1470_147097


namespace sum_of_reciprocals_of_roots_l1470_147093

theorem sum_of_reciprocals_of_roots (m n : ℝ) 
  (hm : m^2 + 3*m + 5 = 0) 
  (hn : n^2 + 3*n + 5 = 0) : 
  1/n + 1/m = -3/5 := by sorry

end sum_of_reciprocals_of_roots_l1470_147093


namespace set_A_nonempty_iff_a_negative_l1470_147019

theorem set_A_nonempty_iff_a_negative (a : ℝ) :
  (∃ x : ℝ, (Real.sqrt x)^2 ≠ a) ↔ a < 0 := by sorry

end set_A_nonempty_iff_a_negative_l1470_147019


namespace debby_total_messages_l1470_147079

/-- The total number of text messages Debby received -/
def total_messages (before_noon after_noon : ℕ) : ℕ := before_noon + after_noon

/-- Proof that Debby received 39 text messages in total -/
theorem debby_total_messages :
  total_messages 21 18 = 39 := by
  sorry

end debby_total_messages_l1470_147079


namespace negation_equivalence_l1470_147053

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) :=
by sorry

end negation_equivalence_l1470_147053


namespace distance_P_to_xaxis_l1470_147091

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- The point P -/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 -/
theorem distance_P_to_xaxis : distanceToXAxis P.1 P.2 = 3 := by
  sorry

end distance_P_to_xaxis_l1470_147091


namespace opposite_of_negative_2023_l1470_147069

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of -2023 is 2023
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l1470_147069


namespace prism_in_sphere_lateral_edge_l1470_147017

/-- A prism with a square base and lateral edges perpendicular to the base -/
structure Prism where
  base_side : ℝ
  lateral_edge : ℝ

/-- A sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem: The length of the lateral edge of a prism inscribed in a sphere -/
theorem prism_in_sphere_lateral_edge 
  (p : Prism) 
  (s : Sphere) 
  (h1 : p.base_side = 1) 
  (h2 : s.radius = 1) 
  (h3 : s.radius = Real.sqrt (p.base_side^2 + p.base_side^2 + p.lateral_edge^2) / 2) : 
  p.lateral_edge = Real.sqrt 2 := by
  sorry

end prism_in_sphere_lateral_edge_l1470_147017


namespace petyas_run_l1470_147090

theorem petyas_run (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  D / (2 * 1.25 * V) + D / (2 * 0.8 * V) > D / V := by
  sorry

end petyas_run_l1470_147090


namespace ratio_and_equation_solution_l1470_147078

theorem ratio_and_equation_solution (a b : ℝ) : 
  b / a = 4 → b = 16 - 6 * a + a^2 → (a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41) :=
by sorry

end ratio_and_equation_solution_l1470_147078


namespace train_crossing_time_l1470_147051

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 500 ∧ 
  train_speed = 63 * 1000 / 3600 ∧ 
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
sorry

end train_crossing_time_l1470_147051


namespace zoe_family_members_l1470_147032

/-- Proves that Zoe is buying for 5 family members given the problem conditions -/
theorem zoe_family_members :
  let cost_per_person : ℚ := 3/2  -- $1.50
  let total_cost : ℚ := 9
  ∀ x : ℚ, (x + 1) * cost_per_person = total_cost → x = 5 :=
by
  sorry

end zoe_family_members_l1470_147032


namespace circle_center_tangent_parabola_l1470_147081

/-- A circle that passes through (1,0) and is tangent to y = x^2 at (1,1) has its center at (1,1) -/
theorem circle_center_tangent_parabola : 
  ∀ (center : ℝ × ℝ),
  (∀ (p : ℝ × ℝ), p.1^2 = p.2 → (center.1 - p.1)^2 + (center.2 - p.2)^2 = (center.1 - 1)^2 + center.2^2) →
  (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + center.2^2 →
  center = (1, 1) :=
by sorry

end circle_center_tangent_parabola_l1470_147081


namespace bill_bouquets_to_buy_l1470_147085

/-- Represents the rose business scenario for Bill --/
structure RoseBusiness where
  buy_roses_per_bouquet : ℕ
  buy_price_per_bouquet : ℕ
  sell_roses_per_bouquet : ℕ
  sell_price_per_bouquet : ℕ

/-- Calculates the number of bouquets Bill needs to buy to earn a specific profit --/
def bouquets_to_buy (rb : RoseBusiness) (target_profit : ℕ) : ℕ :=
  let buy_bouquets := rb.sell_roses_per_bouquet
  let sell_bouquets := rb.buy_roses_per_bouquet
  let profit_per_operation := sell_bouquets * rb.sell_price_per_bouquet - buy_bouquets * rb.buy_price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * buy_bouquets

/-- Theorem stating that Bill needs to buy 125 bouquets to earn $1000 --/
theorem bill_bouquets_to_buy :
  let rb : RoseBusiness := {
    buy_roses_per_bouquet := 7,
    buy_price_per_bouquet := 20,
    sell_roses_per_bouquet := 5,
    sell_price_per_bouquet := 20
  }
  bouquets_to_buy rb 1000 = 125 := by sorry

end bill_bouquets_to_buy_l1470_147085


namespace relationship_between_a_b_c_l1470_147004

theorem relationship_between_a_b_c : ∀ (a b c : ℝ),
  a = -(1^2) →
  b = (3 - Real.pi)^0 →
  c = (-0.25)^2023 * 4^2024 →
  b > a ∧ a > c := by
  sorry

end relationship_between_a_b_c_l1470_147004


namespace rectangular_prism_paint_l1470_147030

theorem rectangular_prism_paint (m n r : ℕ) : 
  0 < m ∧ 0 < n ∧ 0 < r →
  m ≤ n ∧ n ≤ r →
  (m - 2) * (n - 2) * (r - 2) + 
  (4 * (m - 2) + 4 * (n - 2) + 4 * (r - 2)) - 
  (2 * (m - 2) * (n - 2) + 2 * (m - 2) * (r - 2) + 2 * (n - 2) * (r - 2)) = 1985 →
  ((m = 5 ∧ n = 7 ∧ r = 663) ∨
   (m = 5 ∧ n = 5 ∧ r = 1981) ∨
   (m = 3 ∧ n = 3 ∧ r = 1981) ∨
   (m = 1 ∧ n = 7 ∧ r = 399) ∨
   (m = 1 ∧ n = 3 ∧ r = 1987)) := by
sorry

end rectangular_prism_paint_l1470_147030


namespace quadratic_inequality_solution_l1470_147066

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 7*x + 10 < 0 ↔ 2 < x ∧ x < 5 := by
  sorry

end quadratic_inequality_solution_l1470_147066


namespace crackers_distribution_l1470_147099

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 8 →
  num_friends = 4 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end crackers_distribution_l1470_147099


namespace officer_assignment_count_l1470_147027

def group_members : Nat := 4
def officer_positions : Nat := 3

theorem officer_assignment_count : 
  group_members ^ officer_positions = 64 := by
  sorry

end officer_assignment_count_l1470_147027


namespace dakota_medical_bill_l1470_147061

def hospital_stay_days : ℕ := 3
def bed_cost_per_day : ℚ := 900
def specialist_cost_per_hour : ℚ := 250
def specialist_time_hours : ℚ := 1/4
def num_specialists : ℕ := 2
def ambulance_cost : ℚ := 1800

theorem dakota_medical_bill : 
  hospital_stay_days * bed_cost_per_day + 
  specialist_cost_per_hour * specialist_time_hours * num_specialists +
  ambulance_cost = 4750 := by
  sorry

end dakota_medical_bill_l1470_147061


namespace c_investment_is_half_l1470_147064

/-- Represents the investment of a partner in a partnership --/
structure Investment where
  capital : ℚ  -- Fraction of total capital invested
  time : ℚ     -- Fraction of total time invested

/-- Represents a partnership with three investors --/
structure Partnership where
  a : Investment
  b : Investment
  c : Investment
  total_profit : ℚ
  a_share : ℚ

/-- The theorem stating that given the conditions of the problem, C's investment is 1/2 of the total capital --/
theorem c_investment_is_half (p : Partnership) : 
  p.a = ⟨1/6, 1/6⟩ → 
  p.b = ⟨1/3, 1/3⟩ → 
  p.c.time = 1 →
  p.total_profit = 2300 →
  p.a_share = 100 →
  p.c.capital = 1/2 := by
  sorry


end c_investment_is_half_l1470_147064


namespace larger_number_proof_l1470_147000

/-- Given two positive integers with HCF 23 and LCM factors 13 and 14, prove the larger number is 322 -/
theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) → 
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 13 * 14 * k) → 
  (max a b = 322) := by
sorry

end larger_number_proof_l1470_147000


namespace largest_power_of_five_dividing_sum_l1470_147096

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials : ℕ := factorial 50 + factorial 52 + factorial 54

theorem largest_power_of_five_dividing_sum : 
  (∃ (k : ℕ), sum_of_factorials = 5^12 * k ∧ ¬(∃ (m : ℕ), sum_of_factorials = 5^13 * m)) := by
  sorry

end largest_power_of_five_dividing_sum_l1470_147096


namespace sequence_a_11_l1470_147026

theorem sequence_a_11 (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, 4 * S n = 2 * a n - n.val^2 + 7 * n.val) : 
  a 11 = -2 := by
  sorry

end sequence_a_11_l1470_147026


namespace largest_divisor_of_P_l1470_147082

def P (n : ℕ) : ℕ := (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_P (n : ℕ) (h : Even n) (k : ℕ) :
  (∀ m : ℕ, Even m → k ∣ P m) → k ≤ 15 :=
by sorry

end largest_divisor_of_P_l1470_147082


namespace triangle_angle_obtuse_l1470_147073

theorem triangle_angle_obtuse (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : α > π/2 := by
  sorry

end triangle_angle_obtuse_l1470_147073


namespace candy_probability_l1470_147083

def total_candies : ℕ := 20
def red_candies : ℕ := 10
def blue_candies : ℕ := 10

def probability_same_combination : ℚ := 118 / 323

theorem candy_probability : 
  total_candies = red_candies + blue_candies →
  probability_same_combination = 
    (2 * (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3) + 
          blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) + 
     6 * red_candies * (red_candies - 1) * blue_candies * (blue_candies - 1)) / 
    (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3)) :=
by sorry

end candy_probability_l1470_147083


namespace d_value_when_x_plus_3_is_factor_l1470_147023

/-- The polynomial Q(x) with parameter d -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 27

/-- Theorem stating that d = -27 when x+3 is a factor of Q(x) -/
theorem d_value_when_x_plus_3_is_factor :
  ∃ d : ℝ, (∀ x : ℝ, Q d x = 0 ↔ x = -3) → d = -27 := by
  sorry

end d_value_when_x_plus_3_is_factor_l1470_147023


namespace largest_n_for_product_1764_l1470_147012

/-- Represents an arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem largest_n_for_product_1764 :
  ∀ u v : ℕ,
  u ≥ 1 → v ≥ 1 → u ≤ v →
  ∃ n : ℕ, n ≥ 1 ∧
    (arithmeticSequence 3 u n) * (arithmeticSequence 3 v n) = 1764 →
  ∀ m : ℕ, m > 40 →
    (arithmeticSequence 3 u m) * (arithmeticSequence 3 v m) ≠ 1764 :=
by sorry

end largest_n_for_product_1764_l1470_147012


namespace sequence_nonpositive_l1470_147011

theorem sequence_nonpositive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_convex : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2*a k + a (k+1) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end sequence_nonpositive_l1470_147011


namespace foci_distance_of_hyperbola_l1470_147067

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = -144

theorem foci_distance_of_hyperbola :
  ∃ (h : ℝ → ℝ → ℝ), 
    (∀ x y, hyperbola_equation x y → 
      h x y = (let a := 4; let b := 3; Real.sqrt (a^2 + b^2) * 2)) ∧
    (∀ x y, hyperbola_equation x y → h x y = 10) :=
sorry

end foci_distance_of_hyperbola_l1470_147067


namespace triangle_properties_l1470_147056

/-- Given a triangle ABC with interior angles A, B, and C, prove the magnitude of A and the maximum perimeter. -/
theorem triangle_properties (A B C : Real) (R : Real) : 
  -- Conditions
  A + B + C = π ∧ 
  (Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2) ∧
  R = 2 →
  -- Conclusions
  A = 2*π/3 ∧ 
  ∃ (L : Real), L = 2*Real.sqrt 3 + 4 ∧ 
    ∀ (a b c : Real), 
      a / Real.sin A = 2*R → 
      a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
      a + b + c ≤ L :=
by sorry

end triangle_properties_l1470_147056


namespace line_perpendicular_to_plane_l1470_147048

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

end line_perpendicular_to_plane_l1470_147048
