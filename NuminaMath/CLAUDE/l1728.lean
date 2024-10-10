import Mathlib

namespace problem_solution_l1728_172898

def sequence1 (n : ℕ) : ℤ := (-1)^n * (2*n - 1)
def sequence2 (n : ℕ) : ℤ := (-1)^n * (2*n - 1) - 2
def sequence3 (n : ℕ) : ℤ := 3 * (2*n - 1) * (-1)^(n+1)

theorem problem_solution :
  (sequence1 10 = 19) ∧
  (sequence2 15 = -31) ∧
  (∀ n : ℕ, sequence2 n + sequence2 (n+1) + sequence2 (n+2) ≠ 1001) ∧
  (∃! k : ℕ, k % 2 = 1 ∧ sequence1 k + sequence2 k + sequence3 k = 599 ∧ k = 301) :=
by sorry

end problem_solution_l1728_172898


namespace xyz_product_l1728_172830

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 180)
  (h2 : y * (z + x) = 192)
  (h3 : z * (x + y) = 204) :
  x * y * z = 168 * Real.sqrt 6 := by
sorry

end xyz_product_l1728_172830


namespace chord_length_in_circle_l1728_172886

/-- The length of the chord cut by the line x = 1/2 from the circle (x-1)^2 + y^2 = 1 is √3 -/
theorem chord_length_in_circle (x y : ℝ) : 
  (x = 1/2) → ((x - 1)^2 + y^2 = 1) → 
  ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ 
    ((1/2 - 1)^2 + y1^2 = 1) ∧ 
    ((1/2 - 1)^2 + y2^2 = 1) ∧
    ((1/2 - 1/2)^2 + (y1 - y2)^2 = 3) :=
by sorry

end chord_length_in_circle_l1728_172886


namespace mabels_daisies_l1728_172821

/-- Given initial daisies, petals per daisy, and daisies given away, 
    calculate the number of petals on remaining daisies -/
def remaining_petals (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ) : ℕ :=
  (initial_daisies - daisies_given) * petals_per_daisy

/-- Theorem: Given 5 initial daisies with 8 petals each, 
    after giving away 2 daisies, 24 petals remain -/
theorem mabels_daisies : remaining_petals 5 8 2 = 24 := by
  sorry

end mabels_daisies_l1728_172821


namespace cube_of_square_of_second_smallest_prime_l1728_172836

-- Define the second smallest prime number
def second_smallest_prime : Nat := 3

-- Theorem statement
theorem cube_of_square_of_second_smallest_prime : 
  (second_smallest_prime ^ 2) ^ 3 = 729 := by
  sorry

end cube_of_square_of_second_smallest_prime_l1728_172836


namespace jennis_age_l1728_172877

theorem jennis_age (sum diff : ℕ) (h_sum : sum = 70) (h_diff : diff = 32) :
  ∃ (age_jenni age_bai : ℕ), age_jenni + age_bai = sum ∧ age_bai - age_jenni = diff ∧ age_jenni = 19 :=
by sorry

end jennis_age_l1728_172877


namespace pencil_difference_proof_l1728_172880

def pencil_distribution (total : ℕ) (kept : ℕ) (given_to_manny : ℕ) : Prop :=
  let given_away := total - kept
  let given_to_nilo := given_away - given_to_manny
  given_to_nilo - given_to_manny = 10

theorem pencil_difference_proof :
  pencil_distribution 50 20 10 := by
  sorry

end pencil_difference_proof_l1728_172880


namespace unrestricted_arrangements_count_restricted_arrangements_count_l1728_172859

/-- Represents the number of singers in the chorus -/
def total_singers : ℕ := 8

/-- Represents the number of female singers -/
def female_singers : ℕ := 6

/-- Represents the number of male singers -/
def male_singers : ℕ := 2

/-- Represents the number of people per row -/
def people_per_row : ℕ := 4

/-- Represents the number of rows -/
def num_rows : ℕ := 2

/-- Calculates the number of arrangements with no restrictions -/
def unrestricted_arrangements : ℕ := Nat.factorial total_singers

/-- Calculates the number of arrangements with lead singer in front and male singers in back -/
def restricted_arrangements : ℕ :=
  (Nat.choose (female_singers - 1) (people_per_row - 1)) *
  (Nat.factorial people_per_row) *
  (Nat.factorial people_per_row)

/-- Theorem stating the number of unrestricted arrangements -/
theorem unrestricted_arrangements_count :
  unrestricted_arrangements = 40320 := by sorry

/-- Theorem stating the number of restricted arrangements -/
theorem restricted_arrangements_count :
  restricted_arrangements = 5760 := by sorry

end unrestricted_arrangements_count_restricted_arrangements_count_l1728_172859


namespace book_pricing_and_cost_theorem_l1728_172845

/-- Represents the price and quantity of books --/
structure BookInfo where
  edu_price : ℝ
  ele_price : ℝ
  edu_quantity : ℕ
  ele_quantity : ℕ

/-- Calculates the total cost of books --/
def total_cost (info : BookInfo) : ℝ :=
  info.edu_price * info.edu_quantity + info.ele_price * info.ele_quantity

/-- Checks if the quantity constraint is satisfied --/
def quantity_constraint (info : BookInfo) : Prop :=
  info.edu_quantity ≤ 3 * info.ele_quantity ∧ info.edu_quantity ≥ 70

/-- The main theorem to be proven --/
theorem book_pricing_and_cost_theorem (info : BookInfo) : 
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 2, ele_quantity := 3} = 126) →
  (total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := 3, ele_quantity := 2} = 109) →
  (info.edu_price = 15 ∧ info.ele_price = 32) ∧
  (∀ m : ℕ, m + info.ele_quantity = 200 → quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} →
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ≥ 3850) ∧
  (∃ m : ℕ, m + info.ele_quantity = 200 ∧ 
    quantity_constraint {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} ∧
    total_cost {edu_price := info.edu_price, ele_price := info.ele_price, edu_quantity := m, ele_quantity := 200 - m} = 3850) :=
by sorry

end book_pricing_and_cost_theorem_l1728_172845


namespace abs_product_plus_four_gt_abs_sum_l1728_172854

def f (x : ℝ) := |x - 1| + |x + 1|

def M : Set ℝ := {x | f x < 4}

theorem abs_product_plus_four_gt_abs_sum {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M) :
  |a * b + 4| > |a + b| := by
  sorry

end abs_product_plus_four_gt_abs_sum_l1728_172854


namespace mikes_ride_length_l1728_172883

/-- Represents the taxi ride problem --/
structure TaxiRide where
  startingAmount : ℝ
  costPerMile : ℝ
  anniesMiles : ℝ
  bridgeToll : ℝ

/-- The theorem stating that Mike's ride was 46 miles long --/
theorem mikes_ride_length (ride : TaxiRide) 
  (h1 : ride.startingAmount = 2.5)
  (h2 : ride.costPerMile = 0.25)
  (h3 : ride.anniesMiles = 26)
  (h4 : ride.bridgeToll = 5) :
  ∃ (mikesMiles : ℝ), 
    mikesMiles = 46 ∧ 
    ride.startingAmount + ride.costPerMile * mikesMiles = 
    ride.startingAmount + ride.bridgeToll + ride.costPerMile * ride.anniesMiles :=
by
  sorry


end mikes_ride_length_l1728_172883


namespace remainder_and_smallest_integer_l1728_172884

theorem remainder_and_smallest_integer (n : ℤ) : n % 20 = 11 →
  ((n % 4 + n % 5 = 4) ∧
   (∀ m : ℤ, m > 50 ∧ m % 20 = 11 → m ≥ 51) ∧
   (51 % 20 = 11)) :=
by sorry

end remainder_and_smallest_integer_l1728_172884


namespace park_outer_diameter_l1728_172865

/-- Represents the structure of a circular park with concentric regions -/
structure CircularPark where
  fountain_diameter : ℝ
  garden_width : ℝ
  inner_path_width : ℝ
  outer_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.fountain_diameter + 2 * (park.garden_width + park.inner_path_width + park.outer_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 48 feet -/
theorem park_outer_diameter :
  let park : CircularPark := {
    fountain_diameter := 10,
    garden_width := 12,
    inner_path_width := 3,
    outer_path_width := 4
  }
  outer_boundary_diameter park = 48 := by sorry

end park_outer_diameter_l1728_172865


namespace problem_1_l1728_172824

theorem problem_1 : -3 + 8 - 15 - 6 = -16 := by
  sorry

end problem_1_l1728_172824


namespace shortest_tree_height_l1728_172806

/-- The heights of four trees satisfying certain conditions -/
structure TreeHeights where
  tallest : ℝ
  second_tallest : ℝ
  third_tallest : ℝ
  shortest : ℝ
  tallest_height : tallest = 108
  second_tallest_height : second_tallest = tallest / 2 - 6
  third_tallest_height : third_tallest = second_tallest / 4
  shortest_height : shortest = second_tallest + third_tallest - 2

/-- The height of the shortest tree is 58 feet -/
theorem shortest_tree_height (t : TreeHeights) : t.shortest = 58 := by
  sorry

end shortest_tree_height_l1728_172806


namespace notebook_cost_l1728_172881

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 42 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  total_cost = 3213 ∧
  cost_per_notebook = 17 :=
by sorry

end notebook_cost_l1728_172881


namespace green_shirt_percentage_l1728_172826

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of students wearing blue shirts
def blue_percentage : ℚ := 45 / 100

-- Define the percentage of students wearing red shirts
def red_percentage : ℚ := 23 / 100

-- Define the number of students wearing other colors
def other_colors : ℕ := 136

-- Theorem to prove
theorem green_shirt_percentage :
  (total_students - (blue_percentage * total_students).floor - 
   (red_percentage * total_students).floor - other_colors) / total_students = 15 / 100 := by
sorry

end green_shirt_percentage_l1728_172826


namespace tutor_schedule_lcm_l1728_172862

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 9 10))) = 630 := by
  sorry

end tutor_schedule_lcm_l1728_172862


namespace correct_calculation_l1728_172887

theorem correct_calculation (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 := by
  sorry

end correct_calculation_l1728_172887


namespace determinant_of_geometric_sequence_l1728_172840

-- Define a geometric sequence of four terms
def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

-- State the theorem
theorem determinant_of_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) :
  is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ - a₂ * a₃ = 0 := by
  sorry

end determinant_of_geometric_sequence_l1728_172840


namespace smallest_n_for_real_power_l1728_172833

def complex_i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem smallest_n_for_real_power :
  ∃ (n : ℕ), n > 0 ∧ is_real ((1 + complex_i) ^ n) ∧
  ∀ (m : ℕ), 0 < m → m < n → ¬ is_real ((1 + complex_i) ^ m) :=
by sorry

end smallest_n_for_real_power_l1728_172833


namespace reinforcement_theorem_l1728_172895

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) 
    (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  (initial_garrison * initial_duration - initial_garrison * days_before_reinforcement) / remaining_duration - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 2000. -/
theorem reinforcement_theorem : 
  reinforcement_size 2000 40 20 10 = 2000 := by
  sorry

end reinforcement_theorem_l1728_172895


namespace sum_of_coordinates_D_l1728_172893

-- Define the points
def C : ℝ × ℝ := (10, 6)
def N : ℝ × ℝ := (4, 8)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_coordinates_D :
  is_midpoint N C D → D.1 + D.2 = 16 := by sorry

end sum_of_coordinates_D_l1728_172893


namespace solve_for_x_l1728_172832

theorem solve_for_x (x y z : ℝ) 
  (eq1 : x + y = 75)
  (eq2 : (x + y) + y + z = 130)
  (eq3 : z = y + 10) :
  x = 52.5 := by sorry

end solve_for_x_l1728_172832


namespace smallest_base_for_101_l1728_172808

/-- A number n can be expressed in base b using only two digits if b ≤ n < b^2 -/
def expressibleInTwoDigits (n : ℕ) (b : ℕ) : Prop :=
  b ≤ n ∧ n < b^2

/-- The smallest whole number b such that 101 can be expressed in base b using only two digits -/
def smallestBase : ℕ := 10

theorem smallest_base_for_101 :
  (∀ b : ℕ, b < smallestBase → ¬expressibleInTwoDigits 101 b) ∧
  expressibleInTwoDigits 101 smallestBase := by
  sorry

end smallest_base_for_101_l1728_172808


namespace smallest_k_with_remainder_one_existence_of_547_smallest_k_is_547_l1728_172829

theorem smallest_k_with_remainder_one (k : ℕ) : k > 1 ∧ 
  k % 13 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ k % 2 = 1 → k ≥ 547 := by
  sorry

theorem existence_of_547 : 
  547 > 1 ∧ 547 % 13 = 1 ∧ 547 % 7 = 1 ∧ 547 % 3 = 1 ∧ 547 % 2 = 1 := by
  sorry

theorem smallest_k_is_547 : ∃! k : ℕ, k > 1 ∧ 
  k % 13 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ k % 2 = 1 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ∧ m % 2 = 1) → k ≤ m := by
  sorry

end smallest_k_with_remainder_one_existence_of_547_smallest_k_is_547_l1728_172829


namespace triangle_angle_C_l1728_172849

theorem triangle_angle_C (A B C : ℝ) (h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) :
  A + B + C = Real.pi → C = Real.pi / 4 := by
  sorry

end triangle_angle_C_l1728_172849


namespace alloy_mixture_theorem_l1728_172843

/-- The amount of the first alloy used to create the third alloy -/
def first_alloy_amount : ℝ := 15

/-- The percentage of chromium in the first alloy -/
def first_alloy_chromium_percent : ℝ := 0.10

/-- The percentage of chromium in the second alloy -/
def second_alloy_chromium_percent : ℝ := 0.06

/-- The amount of the second alloy used to create the third alloy -/
def second_alloy_amount : ℝ := 35

/-- The percentage of chromium in the resulting third alloy -/
def third_alloy_chromium_percent : ℝ := 0.072

theorem alloy_mixture_theorem :
  first_alloy_amount * first_alloy_chromium_percent +
  second_alloy_amount * second_alloy_chromium_percent =
  (first_alloy_amount + second_alloy_amount) * third_alloy_chromium_percent :=
by sorry

end alloy_mixture_theorem_l1728_172843


namespace f_derivative_at_one_l1728_172834

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2 * (Real.log 2 - 1) * x)

theorem f_derivative_at_one :
  deriv f 1 = 1 := by sorry

end f_derivative_at_one_l1728_172834


namespace marks_lost_is_one_l1728_172810

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  correct_score : ℕ
  total_score : ℤ
  correct_answers : ℕ

/-- Calculates the marks lost for each wrong answer in the examination -/
def marks_lost_per_wrong_answer (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_correct_score := exam.correct_score * exam.correct_answers
  (total_correct_score - exam.total_score) / wrong_answers

/-- Theorem stating that the marks lost for each wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
    (h1 : exam.total_questions = 60)
    (h2 : exam.correct_score = 4)
    (h3 : exam.total_score = 110)
    (h4 : exam.correct_answers = 34) :
  marks_lost_per_wrong_answer exam = 1 := by
  sorry

#eval marks_lost_per_wrong_answer { 
  total_questions := 60, 
  correct_score := 4, 
  total_score := 110, 
  correct_answers := 34 
}

end marks_lost_is_one_l1728_172810


namespace conversation_on_weekday_l1728_172828

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to check if a day is a weekday
def isWeekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

-- Define the brothers
structure Brother :=
  (liesOnSaturday : Bool)
  (liesOnSunday : Bool)
  (willLieTomorrow : Bool)

-- Define the conversation
def conversation (day : Day) (brother1 brother2 : Brother) : Prop :=
  brother1.liesOnSaturday = true
  ∧ brother1.liesOnSunday = true
  ∧ brother2.willLieTomorrow = true
  ∧ (day = Day.Saturday → ¬brother1.liesOnSaturday)
  ∧ (day = Day.Sunday → ¬brother1.liesOnSunday)
  ∧ (isWeekday day → ¬brother2.willLieTomorrow)

-- Theorem: The conversation occurs on a weekday
theorem conversation_on_weekday (day : Day) (brother1 brother2 : Brother) :
  conversation day brother1 brother2 → isWeekday day :=
by sorry

end conversation_on_weekday_l1728_172828


namespace expected_worth_unfair_coin_expected_worth_is_zero_l1728_172860

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : ℝ :=
  let p_heads : ℝ := 2/3
  let p_tails : ℝ := 1/3
  let gain_heads : ℝ := 5
  let loss_tails : ℝ := 10
  p_heads * gain_heads + p_tails * (-loss_tails)

/-- Proof that the expected worth of the unfair coin flip is 0 -/
theorem expected_worth_is_zero : expected_worth_unfair_coin = 0 := by
  sorry

end expected_worth_unfair_coin_expected_worth_is_zero_l1728_172860


namespace probability_of_king_is_one_thirteenth_l1728_172850

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (kings : ℕ)

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : Deck) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Theorem: The probability of drawing a King from a standard deck is 1/13 -/
theorem probability_of_king_is_one_thirteenth (deck : Deck) 
  (h1 : deck.total_cards = 52)
  (h2 : deck.ranks = 13)
  (h3 : deck.suits = 4)
  (h4 : deck.kings = 4) :
  probability_of_draw deck deck.kings = 1 / 13 := by
  sorry

end probability_of_king_is_one_thirteenth_l1728_172850


namespace triangle_side_calculation_l1728_172825

/-- Given a triangle ABC with side a = 4, angle B = π/3, and area S = 6√3,
    prove that side b = 2√7 -/
theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Conditions
  a = 4 →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 →
  -- Definition of cosine law
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  -- Conclusion
  b = 2 * Real.sqrt 7 := by
sorry

end triangle_side_calculation_l1728_172825


namespace circle_area_ratio_l1728_172888

theorem circle_area_ratio (R_C R_D : ℝ) (h : R_C > 0 ∧ R_D > 0) :
  (60 / 360 * (2 * Real.pi * R_C) = 2 * (40 / 360 * (2 * Real.pi * R_D))) →
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 16 / 9 := by
  sorry

end circle_area_ratio_l1728_172888


namespace geometric_sequence_property_l1728_172892

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end geometric_sequence_property_l1728_172892


namespace find_divisor_l1728_172812

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 3 + remainder) :
  3 = dividend / quotient :=
by
  sorry

end find_divisor_l1728_172812


namespace multiples_of_five_average_l1728_172873

theorem multiples_of_five_average (n : ℕ) : 
  (((n : ℝ) / 2) * (5 + 5 * n)) / n = 55 → n = 21 := by
  sorry

end multiples_of_five_average_l1728_172873


namespace scalar_product_formula_l1728_172803

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def scalar_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem scalar_product_formula (x₁ y₁ x₂ y₂ : ℝ) :
  scalar_product (vector_2d x₁ y₁) (vector_2d x₂ y₂) = x₁ * x₂ + y₁ * y₂ := by
  sorry

end scalar_product_formula_l1728_172803


namespace dhoni_spending_l1728_172863

theorem dhoni_spending (total_earnings : ℝ) (rent_percent dishwasher_percent leftover_percent : ℝ) :
  rent_percent = 25 →
  leftover_percent = 52.5 →
  dishwasher_percent = 100 - rent_percent - leftover_percent →
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 10 :=
by sorry

end dhoni_spending_l1728_172863


namespace hyperbola_n_range_l1728_172871

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the condition for the distance between foci
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

-- Define the range of n
def n_range (n : ℝ) : Prop :=
  -1 < n ∧ n < 3

-- Theorem statement
theorem hyperbola_n_range :
  ∀ m n : ℝ,
  (∃ x y : ℝ, hyperbola_equation x y m n) →
  foci_distance m n →
  n_range n :=
sorry

end hyperbola_n_range_l1728_172871


namespace polygon_diagonals_l1728_172866

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n - 3 ≤ 6) → n = 9 := by
  sorry

end polygon_diagonals_l1728_172866


namespace range_of_m_for_root_in_interval_l1728_172853

/-- Given a function f(x) = 2x - m with a root in the interval (1, 2), 
    prove that the range of m is 2 < m < 4 -/
theorem range_of_m_for_root_in_interval 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = 2 * x - m) 
  (h2 : ∃ x ∈ Set.Ioo 1 2, f x = 0) : 
  2 < m ∧ m < 4 := by
  sorry

end range_of_m_for_root_in_interval_l1728_172853


namespace m_range_l1728_172805

theorem m_range (m : ℝ) : 
  let M := Set.Iic m
  let P := {x : ℝ | x ≥ -1}
  M ∩ P = ∅ → m < -1 :=
by
  sorry

end m_range_l1728_172805


namespace mean_equality_implies_z_value_l1728_172835

theorem mean_equality_implies_z_value : 
  let mean1 := (8 + 14 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 44 / 3 := by
  sorry

end mean_equality_implies_z_value_l1728_172835


namespace f_at_4_l1728_172864

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f : List ℤ := [11, -9, 7, -5, 3, 1]

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : horner f 4 = 1559 := by
  sorry

end f_at_4_l1728_172864


namespace complex_equality_l1728_172896

theorem complex_equality (a : ℝ) (z : ℂ) : 
  z = (a + 3*I) / (1 + 2*I) → z.re = z.im → a = -1 := by sorry

end complex_equality_l1728_172896


namespace bobby_candy_problem_l1728_172869

theorem bobby_candy_problem (initial_candy : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ)
  (h1 : initial_candy = 36)
  (h2 : eaten_later = 15)
  (h3 : remaining_candy = 4) :
  initial_candy - remaining_candy - eaten_later = 17 :=
by sorry

end bobby_candy_problem_l1728_172869


namespace quadratic_equal_roots_coefficient_l1728_172879

theorem quadratic_equal_roots_coefficient (k : ℝ) (h : k = 1.7777777777777777) : 
  let eq := fun x : ℝ => 2 * k * x^2 + 3 * k * x + 2
  let discriminant := (3 * k)^2 - 4 * (2 * k) * 2
  discriminant = 0 → 3 * k = 5.333333333333333 :=
by
  sorry

#eval (3 : Float) * 1.7777777777777777

end quadratic_equal_roots_coefficient_l1728_172879


namespace tangent_line_of_odd_function_l1728_172858

/-- Given function f(x) = (a-1)x^2 - a*sin(x) is odd, 
    prove that its tangent line at (0,0) is y = -x -/
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, ((a - 1) * x^2 - a * Real.sin x) = -((a - 1) * (-x)^2 - a * Real.sin (-x))) →
  (∃ f : ℝ → ℝ, (∀ x, f x = (a - 1) * x^2 - a * Real.sin x) ∧ 
    (∃ f' : ℝ → ℝ, (∀ x, HasDerivAt f (f' x) x) ∧ f' 0 = -1)) :=
by sorry

end tangent_line_of_odd_function_l1728_172858


namespace counterpositive_equivalence_l1728_172813

theorem counterpositive_equivalence (a b c : ℝ) :
  (a^2 + b^2 + c^2 < 3 → a + b + c ≠ 3) ↔
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) :=
by sorry

end counterpositive_equivalence_l1728_172813


namespace right_triangle_area_l1728_172804

/-- The area of a right triangle with vertices at (-3,0), (0,2), and (0,0) is 3 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (0, 0)
  -- Assume the triangle is right-angled
  (B.1 - C.1) * (A.2 - C.2) = (A.1 - C.1) * (B.2 - C.2) →
  -- The area of the triangle
  1/2 * |A.1 - C.1| * |B.2 - C.2| = 3 := by
sorry


end right_triangle_area_l1728_172804


namespace monthly_interest_rate_equation_l1728_172899

/-- The monthly interest rate that satisfies the compound interest equation for a loan of $200 with $22 interest charged in the second month. -/
theorem monthly_interest_rate_equation : ∃ r : ℝ, 200 * (1 + r)^2 = 222 := by
  sorry

end monthly_interest_rate_equation_l1728_172899


namespace circle_and_tangent_line_l1728_172870

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  9*x + 40*y + 18 = 0 ∨ x = -2

-- Theorem statement
theorem circle_and_tangent_line :
  -- Given conditions
  (circle_C 0 0) ∧
  (circle_C 6 0) ∧
  (circle_C 0 8) ∧
  -- Line l passes through (-2, 0)
  (line_l (-2) 0) ∧
  -- Line l is tangent to circle C
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y ∧
    (∀ (x' y' : ℝ), line_l x' y' → (x' - x)^2 + (y' - y)^2 > 0 ∨ (x' = x ∧ y' = y))) →
  -- Conclusion: The equations of C and l are correct
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ (x y : ℝ), line_l x y ↔ (9*x + 40*y + 18 = 0 ∨ x = -2)) :=
by sorry


end circle_and_tangent_line_l1728_172870


namespace area_bounded_by_curves_l1728_172818

-- Define the function f(x) = x^3 - 4x
def f (x : ℝ) : ℝ := x^3 - 4*x

-- State the theorem
theorem area_bounded_by_curves : 
  ∃ (a b : ℝ), a ≥ 0 ∧ b > a ∧ f a = 0 ∧ f b = 0 ∧ 
  (∫ (x : ℝ) in a..b, |f x|) = 4 := by
  sorry

end area_bounded_by_curves_l1728_172818


namespace valid_division_l1728_172807

theorem valid_division (divisor quotient remainder dividend : ℕ) : 
  divisor = 3040 →
  quotient = 8 →
  remainder = 7 →
  dividend = 24327 →
  dividend = divisor * quotient + remainder :=
by sorry

end valid_division_l1728_172807


namespace magnitude_of_z_is_one_l1728_172894

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem magnitude_of_z_is_one (h : (1 - z) / (1 + z) = 2 * i) : Complex.abs z = 1 := by
  sorry

end magnitude_of_z_is_one_l1728_172894


namespace limit_at_zero_l1728_172819

-- Define the function f
def f (x : ℝ) := x^2

-- State the theorem
theorem limit_at_zero (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ → 
    |(f Δx - f 0) / Δx - 0| < ε := by
  sorry

end limit_at_zero_l1728_172819


namespace valid_topping_combinations_l1728_172848

/-- Represents the number of cheese options --/
def cheese_options : ℕ := 3

/-- Represents the number of meat options --/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options --/
def vegetable_options : ℕ := 5

/-- Represents that peppers is one of the vegetable options --/
axiom peppers_is_vegetable : vegetable_options > 0

/-- Represents that pepperoni is one of the meat options --/
axiom pepperoni_is_meat : meat_options > 0

/-- Calculates the total number of combinations without restrictions --/
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

/-- Represents the number of invalid combinations (pepperoni with peppers) --/
def invalid_combinations : ℕ := 1

/-- Theorem stating the total number of valid topping combinations --/
theorem valid_topping_combinations : 
  total_combinations - invalid_combinations = 59 := by sorry

end valid_topping_combinations_l1728_172848


namespace base2_to_base4_conversion_l1728_172856

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_to_base4_conversion :
  decimal_to_base4 (base2_to_decimal [true, false, true, true, false, true, true, true, false]) =
  [1, 1, 2, 3, 2] :=
by sorry

end base2_to_base4_conversion_l1728_172856


namespace income_average_difference_l1728_172874

theorem income_average_difference (n : ℕ) (min_income max_income error_income : ℝ) :
  n > 0 ∧
  min_income = 8200 ∧
  max_income = 98000 ∧
  error_income = 980000 ∧
  n = 28 * 201000 →
  (error_income - max_income) / n = 882 :=
by sorry

end income_average_difference_l1728_172874


namespace meghan_money_l1728_172878

/-- The total amount of money Meghan has, given the number of bills of each denomination -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Theorem stating that Meghan's total money is $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end meghan_money_l1728_172878


namespace distributive_property_subtraction_l1728_172847

theorem distributive_property_subtraction (a b c : ℝ) : a - (b + c) = a - b - c := by
  sorry

end distributive_property_subtraction_l1728_172847


namespace decagon_diagonal_intersections_l1728_172842

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def diagonal_intersections (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  diagonal_intersections 10 = 210 := by sorry

end decagon_diagonal_intersections_l1728_172842


namespace no_valid_operation_l1728_172885

def basic_op (x y : ℝ) : Set ℝ :=
  {x + y, x - y, x * y, x / y}

theorem no_valid_operation :
  ∀ op ∈ basic_op 9 2, (op * 3 + (4 * 2) - 6) ≠ 21 := by
  sorry

end no_valid_operation_l1728_172885


namespace complex_equation_solution_l1728_172857

theorem complex_equation_solution :
  ∀ y : ℝ,
  let z₁ : ℂ := 3 + y * Complex.I
  let z₂ : ℂ := 2 - Complex.I
  z₁ / z₂ = 1 + Complex.I →
  y = 1 :=
by
  sorry

end complex_equation_solution_l1728_172857


namespace expression_value_l1728_172841

theorem expression_value (a b c d x : ℝ) : 
  a = -b → cd = 1 → abs x = 2 → 
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 3 ∨
  x^2 - (a + b + cd) * x + (a + b)^2021 + (-cd)^2022 = 7 := by
sorry

end expression_value_l1728_172841


namespace total_amount_is_80000_l1728_172855

/-- Represents the problem of dividing money between two investments with different interest rates -/
def MoneyDivisionProblem (total_profit interest_10_amount : ℕ) : Prop :=
  ∃ (total_amount interest_20_amount : ℕ),
    -- Total amount is the sum of both investments
    total_amount = interest_10_amount + interest_20_amount ∧
    -- Profit calculation
    total_profit = (interest_10_amount * 10 / 100) + (interest_20_amount * 20 / 100)

/-- Theorem stating that given the problem conditions, the total amount is 80000 -/
theorem total_amount_is_80000 :
  MoneyDivisionProblem 9000 70000 → ∃ total_amount : ℕ, total_amount = 80000 :=
sorry

end total_amount_is_80000_l1728_172855


namespace arithmetic_mean_of_scores_l1728_172868

def scores : List ℝ := [84, 90, 87, 93, 88, 92]

theorem arithmetic_mean_of_scores : 
  (scores.sum / scores.length : ℝ) = 89 := by sorry

end arithmetic_mean_of_scores_l1728_172868


namespace davids_physics_marks_l1728_172827

/-- Calculates the marks in Physics given marks in other subjects and the average --/
def physics_marks (english : ℕ) (math : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

/-- Proves that David's marks in Physics are 82 given his other marks and average --/
theorem davids_physics_marks :
  physics_marks 61 65 67 85 72 = 82 := by
  sorry

end davids_physics_marks_l1728_172827


namespace inequality_solution_set_l1728_172876

theorem inequality_solution_set (d : ℝ) : 
  (d / 4 ≤ 3 - d ∧ 3 - d < 1 - 2*d) ↔ (-2 < d ∧ d ≤ 12/5) :=
by sorry

end inequality_solution_set_l1728_172876


namespace non_right_triangles_count_l1728_172875

-- Define the points on the grid
def Point := Fin 6

-- Define the grid
def Grid := Point → ℝ × ℝ

-- Define the specific grid layout
def grid_layout : Grid := sorry

-- Define a function to check if a triangle is right-angled
def is_right_angled (p q r : Point) (g : Grid) : Prop := sorry

-- Define a function to count non-right-angled triangles
def count_non_right_triangles (g : Grid) : ℕ := sorry

-- Theorem statement
theorem non_right_triangles_count :
  count_non_right_triangles grid_layout = 4 := by sorry

end non_right_triangles_count_l1728_172875


namespace wine_consumption_problem_l1728_172837

/-- Represents the wine consumption problem from the Ming Dynasty's "The Great Compendium of Mathematics" -/
theorem wine_consumption_problem (x y : ℚ) : 
  (x + y = 19 ∧ 3 * x + (1/3) * y = 33) ↔ 
  (x ≥ 0 ∧ y ≥ 0 ∧ 
   ∃ (good_wine weak_wine guests : ℕ),
     good_wine = x ∧
     weak_wine = y ∧
     guests = 33 ∧
     good_wine + weak_wine = 19 ∧
     (3 * good_wine + (weak_wine / 3 : ℚ)) = guests) :=
by sorry

end wine_consumption_problem_l1728_172837


namespace function_composition_sqrt2_l1728_172867

/-- Given a function f(x) = a * x^2 - √2, where a is a constant,
    if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
  sorry

end function_composition_sqrt2_l1728_172867


namespace select_five_from_eight_with_book_a_l1728_172816

/-- The number of ways to select 5 books from 8 books, always including "Book A" -/
def select_books (total_books : ℕ) (books_to_select : ℕ) : ℕ :=
  Nat.choose (total_books - 1) (books_to_select - 1)

/-- Theorem: Selecting 5 books from 8 books, always including "Book A", can be done in 35 ways -/
theorem select_five_from_eight_with_book_a : select_books 8 5 = 35 := by
  sorry

end select_five_from_eight_with_book_a_l1728_172816


namespace profit_calculation_l1728_172846

-- Define the variables
def charge_per_lawn : ℕ := 12
def lawns_mowed : ℕ := 3
def gas_expense : ℕ := 17
def extra_income : ℕ := 10

-- Define Tom's profit
def toms_profit : ℕ := charge_per_lawn * lawns_mowed + extra_income - gas_expense

-- Theorem statement
theorem profit_calculation : toms_profit = 29 := by
  sorry

end profit_calculation_l1728_172846


namespace book_arrangement_count_l1728_172872

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangeBooksCount (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * factorial mathBooks * factorial englishBooks

theorem book_arrangement_count :
  arrangeBooksCount 3 5 = 1440 := by
  sorry

end book_arrangement_count_l1728_172872


namespace no_natural_solutions_l1728_172823

theorem no_natural_solutions : ∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z := by
  sorry

end no_natural_solutions_l1728_172823


namespace shopkeeper_pricing_l1728_172890

theorem shopkeeper_pricing (CP : ℝ) 
  (h1 : 0.65 * CP = 416) : 1.25 * CP = 800 := by
  sorry

end shopkeeper_pricing_l1728_172890


namespace holiday_rain_probability_l1728_172822

/-- Probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.3

/-- Probability of rain on Tuesday -/
def prob_rain_tuesday : ℝ := 0.6

/-- Probability of rain continuing to the next day -/
def prob_rain_continue : ℝ := 0.8

/-- Probability of rain on at least one day during the two-day holiday period -/
def prob_rain_at_least_one_day : ℝ :=
  1 - (1 - prob_rain_monday) * (1 - prob_rain_tuesday)

theorem holiday_rain_probability :
  prob_rain_at_least_one_day = 0.72 := by
  sorry

end holiday_rain_probability_l1728_172822


namespace polly_mirror_rate_l1728_172838

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ      -- tweets per minute when happy
  hungry_rate : ℕ     -- tweets per minute when hungry
  mirror_rate : ℕ     -- tweets per minute when watching mirror
  happy_time : ℕ      -- time spent being happy (in minutes)
  hungry_time : ℕ     -- time spent being hungry (in minutes)
  mirror_time : ℕ     -- time spent watching mirror (in minutes)
  total_tweets : ℕ    -- total number of tweets

/-- Theorem about Polly's tweeting rate when watching the mirror -/
theorem polly_mirror_rate (p : PollyTweets)
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.happy_time = 20)
  (h4 : p.hungry_time = 20)
  (h5 : p.mirror_time = 20)
  (h6 : p.total_tweets = 1340)
  (h7 : p.total_tweets = p.happy_rate * p.happy_time + p.hungry_rate * p.hungry_time + p.mirror_rate * p.mirror_time) :
  p.mirror_rate = 45 := by
  sorry

end polly_mirror_rate_l1728_172838


namespace area_cyclic_quadrilateral_l1728_172814

/-- Given a quadrilateral ABCD inscribed in a circle with radius R,
    where φ is the angle between its diagonals,
    the area S of the quadrilateral is equal to 2R^2 * sin(A) * sin(B) * sin(φ). -/
theorem area_cyclic_quadrilateral (R : ℝ) (A B φ : ℝ) (S : ℝ) 
    (hR : R > 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hφ : 0 < φ ∧ φ < π) :
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin φ := by
  sorry

end area_cyclic_quadrilateral_l1728_172814


namespace exactly_six_expressions_l1728_172802

/-- Represents an expression using three identical digits --/
inductive ThreeDigitExpr (d : ℕ)
| add : ThreeDigitExpr d
| sub : ThreeDigitExpr d
| mul : ThreeDigitExpr d
| div : ThreeDigitExpr d
| exp : ThreeDigitExpr d
| sqrt : ThreeDigitExpr d
| floor : ThreeDigitExpr d
| fact : ThreeDigitExpr d

/-- Evaluates a ThreeDigitExpr to a real number --/
def eval {d : ℕ} : ThreeDigitExpr d → ℝ
| ThreeDigitExpr.add => sorry
| ThreeDigitExpr.sub => sorry
| ThreeDigitExpr.mul => sorry
| ThreeDigitExpr.div => sorry
| ThreeDigitExpr.exp => sorry
| ThreeDigitExpr.sqrt => sorry
| ThreeDigitExpr.floor => sorry
| ThreeDigitExpr.fact => sorry

/-- Predicate for valid expressions that evaluate to 24 --/
def isValid (d : ℕ) (e : ThreeDigitExpr d) : Prop :=
  d ≠ 8 ∧ eval e = 24

/-- The main theorem stating there are exactly 6 valid expressions --/
theorem exactly_six_expressions :
  ∃ (exprs : Finset (Σ (d : ℕ), ThreeDigitExpr d)),
    exprs.card = 6 ∧
    (∀ (d : ℕ) (e : ThreeDigitExpr d), isValid d e ↔ (⟨d, e⟩ : Σ (d : ℕ), ThreeDigitExpr d) ∈ exprs) :=
sorry

end exactly_six_expressions_l1728_172802


namespace last_four_digits_of_5_pow_2011_l1728_172820

-- Define the function to get the last four digits
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle of last four digits
def lastFourDigitsCycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2011 :
  lastFourDigits (5^2011) = 8125 := by
  sorry

end last_four_digits_of_5_pow_2011_l1728_172820


namespace purchase_price_calculation_l1728_172809

/-- Given a markup of $50, which includes 25% of cost for overhead and $12 of net profit,
    the purchase price of the article is $152. -/
theorem purchase_price_calculation (markup overhead_percentage net_profit : ℚ) 
    (h1 : markup = 50)
    (h2 : overhead_percentage = 25 / 100)
    (h3 : net_profit = 12)
    (h4 : markup = overhead_percentage * purchase_price + net_profit) :
  purchase_price = 152 :=
by sorry


end purchase_price_calculation_l1728_172809


namespace additional_distance_for_average_speed_l1728_172800

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (target_average_speed : ℝ)
  (h : initial_distance = 20)
  (h1 : initial_speed = 40)
  (h2 : second_speed = 60)
  (h3 : target_average_speed = 55)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end additional_distance_for_average_speed_l1728_172800


namespace geometric_sequence_sum_8_l1728_172861

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (A : ℝ) : ℝ := 2 - A * 2^(n - 1)

/-- The geometric sequence {a_n} -/
def a (n : ℕ) (A : ℝ) : ℝ := S n A - S (n-1) A

/-- Theorem stating that S_8 equals -510 for the given geometric sequence -/
theorem geometric_sequence_sum_8 (A : ℝ) (h1 : ∀ n : ℕ, n ≥ 1 → S n A = 2 - A * 2^(n - 1))
  (h2 : ∀ k : ℕ, k ≥ 1 → a (k+1) A / a k A = a (k+2) A / a (k+1) A) :
  S 8 A = -510 := by sorry

end geometric_sequence_sum_8_l1728_172861


namespace smallest_part_of_proportional_division_l1728_172839

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 124)
  (h_prop : a = 2 ∧ b = 1/2 ∧ c = 1/4) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 124 / 11 := by
sorry

end smallest_part_of_proportional_division_l1728_172839


namespace complex_power_magnitude_l1728_172851

theorem complex_power_magnitude (z : ℂ) :
  z = (1 / Real.sqrt 2 : ℂ) + (Complex.I / Real.sqrt 2) →
  Complex.abs (z^8) = 1 := by
sorry

end complex_power_magnitude_l1728_172851


namespace symmetry_axis_l1728_172811

/-- Given two lines l₁ and l₂ in a 2D plane, this function returns true if they are symmetric about a third line l. -/
def are_symmetric (l₁ l₂ l : ℝ → ℝ → Prop) : Prop := sorry

/-- The line with equation y = -x -/
def line_l₁ (x y : ℝ) : Prop := y = -x

/-- The line with equation x + y - 2 = 0 -/
def line_l₂ (x y : ℝ) : Prop := x + y - 2 = 0

/-- The proposed axis of symmetry -/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

theorem symmetry_axis :
  are_symmetric line_l₁ line_l₂ line_l :=
sorry

end symmetry_axis_l1728_172811


namespace john_distance_l1728_172897

/-- Calculates the total distance John travels given his speeds and running times -/
def total_distance (solo_speed : ℝ) (dog_speed : ℝ) (time_with_dog : ℝ) (time_solo : ℝ) : ℝ :=
  dog_speed * time_with_dog + solo_speed * time_solo

/-- Proves that John travels 5 miles given the specified conditions -/
theorem john_distance :
  let solo_speed : ℝ := 4
  let dog_speed : ℝ := 6
  let time_with_dog : ℝ := 0.5
  let time_solo : ℝ := 0.5
  total_distance solo_speed dog_speed time_with_dog time_solo = 5 := by
  sorry

#eval total_distance 4 6 0.5 0.5

end john_distance_l1728_172897


namespace union_equality_implies_a_values_l1728_172815

def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {1, 2}

theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1/2 ∨ a = 1 := by
  sorry

end union_equality_implies_a_values_l1728_172815


namespace descending_order_of_powers_l1728_172831

theorem descending_order_of_powers : 
  2^(2/3) > (-1.8)^(2/3) ∧ (-1.8)^(2/3) > (-2)^(1/3) := by sorry

end descending_order_of_powers_l1728_172831


namespace library_to_post_office_l1728_172889

def total_distance : ℝ := 0.8
def house_to_library : ℝ := 0.3
def post_office_to_house : ℝ := 0.4

theorem library_to_post_office :
  total_distance - house_to_library - post_office_to_house = 0.1 := by
  sorry

end library_to_post_office_l1728_172889


namespace bridge_length_calculation_l1728_172817

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 250 →
  crossing_time = 20 →
  train_speed = 66.6 →
  (train_speed * crossing_time) - train_length = 1082 :=
by sorry

end bridge_length_calculation_l1728_172817


namespace notebook_cost_l1728_172801

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 36 →
  total_cost = 2772 →
  ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
    buying_students > total_students / 2 ∧
    notebooks_per_student > 2 ∧
    cost_per_notebook = 2 * notebooks_per_student ∧
    buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 12 :=
by sorry

end notebook_cost_l1728_172801


namespace sum_with_radical_conjugate_l1728_172882

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by sorry

end sum_with_radical_conjugate_l1728_172882


namespace geometric_sequence_s4_l1728_172852

/-- A geometric sequence with partial sums S_n -/
structure GeometricSequence where
  S : ℕ → ℝ
  is_geometric : ∀ n : ℕ, S (n + 2) - S (n + 1) = (S (n + 1) - S n) * (S (n + 1) - S n) / (S n - S (n - 1))

/-- Theorem: In a geometric sequence where S_2 = 7 and S_6 = 91, S_4 = 28 -/
theorem geometric_sequence_s4 (seq : GeometricSequence) 
  (h2 : seq.S 2 = 7) (h6 : seq.S 6 = 91) : seq.S 4 = 28 := by
  sorry

end geometric_sequence_s4_l1728_172852


namespace max_min_values_on_interval_l1728_172891

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem max_min_values_on_interval :
  (∃ x ∈ interval, f x = 50 ∧ ∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = 33 ∧ ∀ y ∈ interval, f y ≥ 33) := by
  sorry

end max_min_values_on_interval_l1728_172891


namespace largest_factorable_m_l1728_172844

/-- A quadratic expression of the form 3x^2 + mx - 60 -/
def quadratic (m : ℤ) (x : ℤ) : ℤ := 3 * x^2 + m * x - 60

/-- Checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def is_factorable (m : ℤ) : Prop :=
  ∃ (a b c d : ℤ), ∀ x, quadratic m x = (a * x + b) * (c * x + d)

/-- The largest value of m for which the quadratic is factorable -/
def largest_m : ℤ := 57

theorem largest_factorable_m :
  (is_factorable largest_m) ∧
  (∀ m : ℤ, m > largest_m → ¬(is_factorable m)) := by sorry

end largest_factorable_m_l1728_172844
