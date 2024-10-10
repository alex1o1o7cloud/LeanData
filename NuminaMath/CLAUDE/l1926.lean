import Mathlib

namespace polygon_has_13_sides_l1926_192634

/-- A polygon has n sides. The number of diagonals is equal to 5 times the number of sides. -/
def polygon_diagonals (n : ℕ) : Prop :=
  n * (n - 3) = 5 * n

/-- The polygon satisfying the given condition has 13 sides. -/
theorem polygon_has_13_sides : 
  ∃ (n : ℕ), polygon_diagonals n ∧ n = 13 :=
sorry

end polygon_has_13_sides_l1926_192634


namespace inequality_solution_set_l1926_192699

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (3 - x) ≤ 1 ↔ x > 3 ∨ x ≤ 5/2 :=
sorry

end inequality_solution_set_l1926_192699


namespace kindergarten_tissues_l1926_192611

/-- The number of tissues brought by kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end kindergarten_tissues_l1926_192611


namespace arithmetic_sequence_property_l1926_192646

/-- An arithmetic sequence. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 16)
  (h_a4 : a 4 = 6) :
  a 6 = 10 :=
sorry

end arithmetic_sequence_property_l1926_192646


namespace big_bonsai_cost_l1926_192671

/-- Represents the cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- Represents the number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- Represents the number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 190

/-- Proves that the cost of a big bonsai is $20 -/
theorem big_bonsai_cost : 
  ∃ (big_bonsai_cost : ℕ), 
    small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings ∧ 
    big_bonsai_cost = 20 := by
  sorry

end big_bonsai_cost_l1926_192671


namespace probability_of_specific_outcome_l1926_192652

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- A specific outcome we're interested in -/
def specificOutcome : SixCoins :=
  { penny := CoinFlip.Heads,
    nickel := CoinFlip.Heads,
    dime := CoinFlip.Heads,
    quarter := CoinFlip.Tails,
    halfDollar := CoinFlip.Tails,
    oneDollar := CoinFlip.Tails }

/-- The probability of getting the specific outcome when flipping six coins -/
theorem probability_of_specific_outcome :
  (1 : ℚ) / totalOutcomes = 1 / 64 := by sorry

end probability_of_specific_outcome_l1926_192652


namespace sum_with_radical_conjugate_l1926_192695

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 50
  let y : ℝ := 12 + Real.sqrt 50
  x + y = 24 := by sorry

end sum_with_radical_conjugate_l1926_192695


namespace parallel_lines_a_values_l1926_192657

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 10 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y → l₂ a x y

-- Theorem statement
theorem parallel_lines_a_values :
  ∀ a : ℝ, parallel a → (a = -1 ∨ a = 2) :=
by sorry

end parallel_lines_a_values_l1926_192657


namespace combined_job_time_l1926_192668

def job_time_A : ℝ := 8
def job_time_B : ℝ := 12

theorem combined_job_time : 
  let rate_A := 1 / job_time_A
  let rate_B := 1 / job_time_B
  let combined_rate := rate_A + rate_B
  1 / combined_rate = 4.8 := by sorry

end combined_job_time_l1926_192668


namespace hyperbola_eccentricity_l1926_192645

/-- The eccentricity of a hyperbola with asymptotic lines y = ±(3/2)x is either √13/2 or √13/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (b / a = 3 / 2 ∨ a / b = 3 / 2) →
  c^2 = a^2 + b^2 →
  (c / a = Real.sqrt 13 / 2 ∨ c / a = Real.sqrt 13 / 3) :=
by sorry

end hyperbola_eccentricity_l1926_192645


namespace reading_activity_results_l1926_192694

def characters_per_day : ℕ := 850
def days_per_week : ℕ := 7
def total_weeks : ℕ := 20

def characters_per_week : ℕ := characters_per_day * days_per_week
def total_characters : ℕ := characters_per_week * total_weeks

def approximate_ten_thousands (n : ℕ) : ℕ :=
  (n + 5000) / 10000

theorem reading_activity_results :
  characters_per_week = 5950 ∧
  total_characters = 119000 ∧
  approximate_ten_thousands total_characters = 12 :=
by sorry

end reading_activity_results_l1926_192694


namespace probability_of_correct_number_l1926_192603

def first_three_options : ℕ := 3

def last_five_digits : ℕ := 5
def repeating_digits : ℕ := 2

def total_combinations : ℕ := first_three_options * (Nat.factorial last_five_digits / Nat.factorial repeating_digits)

theorem probability_of_correct_number :
  (1 : ℚ) / total_combinations = 1 / 180 := by
  sorry

end probability_of_correct_number_l1926_192603


namespace negation_of_square_positivity_l1926_192623

theorem negation_of_square_positivity :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
sorry

end negation_of_square_positivity_l1926_192623


namespace solve_flour_problem_l1926_192658

def flour_problem (total_flour sugar flour_to_add flour_already_in : ℕ) : Prop :=
  total_flour = 10 ∧
  sugar = 2 ∧
  flour_to_add = sugar + 1 ∧
  flour_already_in + flour_to_add = total_flour

theorem solve_flour_problem :
  ∃ (flour_already_in : ℕ), flour_problem 10 2 3 flour_already_in ∧ flour_already_in = 7 :=
by sorry

end solve_flour_problem_l1926_192658


namespace homework_time_decrease_l1926_192610

theorem homework_time_decrease (initial_time final_time : ℝ) (x : ℝ) 
  (h_initial : initial_time = 100)
  (h_final : final_time = 70)
  (h_positive : 0 < x ∧ x < 1) :
  initial_time * (1 - x)^2 = final_time := by
  sorry

end homework_time_decrease_l1926_192610


namespace additional_miles_for_average_speed_l1926_192650

theorem additional_miles_for_average_speed 
  (initial_distance : ℝ) 
  (initial_speed : ℝ) 
  (desired_average_speed : ℝ) 
  (additional_speed : ℝ) : 
  initial_distance = 20 ∧ 
  initial_speed = 40 ∧ 
  desired_average_speed = 55 ∧ 
  additional_speed = 60 → 
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / 
    (initial_distance / initial_speed + additional_distance / additional_speed) = 
    desired_average_speed ∧ 
    additional_distance = 90 := by
sorry

end additional_miles_for_average_speed_l1926_192650


namespace min_y_intercept_l1926_192663

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 11

-- Define the y-intercept of the tangent line as a function of x
def r (x : ℝ) : ℝ := -2*x^3 + 6*x^2 - 6

-- Theorem statement
theorem min_y_intercept :
  ∀ x ∈ Set.Icc 0 2, r 0 ≤ r x :=
sorry

end min_y_intercept_l1926_192663


namespace division_of_four_by_negative_two_l1926_192686

theorem division_of_four_by_negative_two : 4 / (-2 : ℚ) = -2 := by sorry

end division_of_four_by_negative_two_l1926_192686


namespace least_possible_radios_l1926_192676

theorem least_possible_radios (n d : ℕ) (h1 : d > 0) : 
  (d + 8 * n - 16 - d = 72) → (∃ (m : ℕ), m ≥ n ∧ m ≥ 12) := by
  sorry

end least_possible_radios_l1926_192676


namespace percentage_failed_both_l1926_192616

/-- Percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 35

/-- Percentage of students who failed in English -/
def failed_english : ℝ := 45

/-- Percentage of students who passed in both subjects -/
def passed_both : ℝ := 40

/-- Percentage of students who failed in both subjects -/
def failed_both : ℝ := 20

theorem percentage_failed_both :
  failed_both = failed_hindi + failed_english - (100 - passed_both) := by
  sorry

end percentage_failed_both_l1926_192616


namespace child_ticket_cost_l1926_192641

theorem child_ticket_cost 
  (adult_price : ℕ) 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_attendance : ℕ) 
  (h1 : adult_price = 9)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_attendance = 175) :
  ∃ (child_price : ℕ), 
    child_price * (total_tickets - adult_attendance) + 
    adult_price * adult_attendance = total_revenue ∧ 
    child_price = 6 :=
by
  sorry

end child_ticket_cost_l1926_192641


namespace equation_solutions_l1926_192669

theorem equation_solutions :
  (∃ x : ℝ, x * (x + 10) = -9 ↔ x = -9 ∨ x = -1) ∧
  (∃ x : ℝ, x * (2 * x + 3) = 8 * x + 12 ↔ x = -3/2 ∨ x = 4) := by
  sorry

end equation_solutions_l1926_192669


namespace regular_polygon_properties_l1926_192648

/-- A regular polygon with interior angles of 160 degrees and side length of 4 units has 18 sides and a perimeter of 72 units. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (side_length : ℝ),
    n > 2 →
    side_length = 4 →
    (180 * (n - 2) : ℝ) / n = 160 →
    n = 18 ∧ n * side_length = 72 := by
  sorry

#check regular_polygon_properties

end regular_polygon_properties_l1926_192648


namespace correct_statements_l1926_192687

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| GeneralToGeneral
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific
| SpecificToGeneral

-- Define a function to describe the correct direction for each reasoning type
def correct_direction (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Define the statements
def statement (n : Nat) : ReasoningType × ReasoningDirection :=
  match n with
  | 1 => (ReasoningType.Inductive, ReasoningDirection.GeneralToGeneral)
  | 2 => (ReasoningType.Inductive, ReasoningDirection.PartToWhole)
  | 3 => (ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific)
  | 4 => (ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific)
  | 5 => (ReasoningType.Analogical, ReasoningDirection.SpecificToGeneral)
  | _ => (ReasoningType.Inductive, ReasoningDirection.PartToWhole) -- Default case

-- Define a function to check if a statement is correct
def is_correct (n : Nat) : Prop :=
  let (rt, rd) := statement n
  rd = correct_direction rt

-- Theorem stating that statements 2, 3, and 4 are the correct ones
theorem correct_statements :
  (is_correct 2 ∧ is_correct 3 ∧ is_correct 4) ∧
  (∀ n, n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 → ¬is_correct n) :=
sorry


end correct_statements_l1926_192687


namespace max_alpha_squared_l1926_192617

theorem max_alpha_squared (a b x y : ℝ) : 
  a > 0 → b > 0 → a = 2 * b →
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2 →
  (∀ α : ℝ, α = a / b → α^2 ≤ 4) ∧ (∃ α : ℝ, α = a / b ∧ α^2 = 4) :=
by sorry

end max_alpha_squared_l1926_192617


namespace nonagon_intersection_points_l1926_192678

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of distinct interior intersection points of diagonals in a regular nonagon -/
def intersectionPoints (n : RegularNonagon) : ℕ := sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def chooseFromNine : ℕ := Nat.choose 9 4

/-- Theorem stating that the number of intersection points in a regular nonagon
    is equal to the number of ways to choose 4 vertices from 9 -/
theorem nonagon_intersection_points (n : RegularNonagon) :
  intersectionPoints n = chooseFromNine := by sorry

end nonagon_intersection_points_l1926_192678


namespace evergreen_marching_band_max_size_l1926_192691

theorem evergreen_marching_band_max_size :
  ∃ (n : ℕ),
    (∀ k : ℕ, 15 * k < 800 → 15 * k ≤ 15 * n) ∧
    (15 * n < 800) ∧
    (15 * n % 19 = 2) ∧
    (15 * n = 750) := by
  sorry

end evergreen_marching_band_max_size_l1926_192691


namespace inequality_solution_set_l1926_192689

theorem inequality_solution_set (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) 1 ∪ Set.Icc 2 9 :=
by sorry

end inequality_solution_set_l1926_192689


namespace simplify_expression_l1926_192619

theorem simplify_expression (x : ℝ) : (3 * x)^4 - (4 * x) * (x^3) = 77 * x^4 := by
  sorry

end simplify_expression_l1926_192619


namespace arithmetic_sequence_sum_l1926_192602

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
  sorry

end arithmetic_sequence_sum_l1926_192602


namespace trigonometric_sum_l1926_192622

theorem trigonometric_sum (x : ℝ) : 
  (Real.cos x + Real.cos (x + 2 * Real.pi / 3) + Real.cos (x + 4 * Real.pi / 3) = 0) ∧
  (Real.sin x + Real.sin (x + 2 * Real.pi / 3) + Real.sin (x + 4 * Real.pi / 3) = 0) := by
  sorry

end trigonometric_sum_l1926_192622


namespace system_one_solution_system_two_solution_l1926_192682

-- System 1
theorem system_one_solution : 
  ∃ (x y : ℝ), y = x - 4 ∧ x + y = 6 ∧ x = 5 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℝ), 2*x + y = 1 ∧ 4*x - y = 5 ∧ x = 1 ∧ y = -1 := by sorry

end system_one_solution_system_two_solution_l1926_192682


namespace ones_digit_of_triple_4567_l1926_192660

theorem ones_digit_of_triple_4567 : (3 * 4567) % 10 = 1 := by
  sorry

end ones_digit_of_triple_4567_l1926_192660


namespace geometric_sequence_ratio_l1926_192631

/-- Given a geometric sequence {a_n} with positive terms where a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, 
    the ratio a_10 / a_8 is equal to 3 + 2√2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
    (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_l1926_192631


namespace point_in_third_quadrant_l1926_192654

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) : 
  (m < 0 ∧ 4 + 2*m < 0) ↔ m < -2 :=
sorry

end point_in_third_quadrant_l1926_192654


namespace inequality_sign_change_l1926_192637

theorem inequality_sign_change (a b : ℝ) (c : ℝ) (h1 : c < 0) (h2 : a < b) : c * b < c * a := by
  sorry

end inequality_sign_change_l1926_192637


namespace inequality_proof_l1926_192636

theorem inequality_proof (a b : ℝ) (ha : |a| ≤ Real.sqrt 3) (hb : |b| ≤ Real.sqrt 3) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end inequality_proof_l1926_192636


namespace arithmetic_sequence_formula_l1926_192677

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean_2_6 : (a 2 + a 6) / 2 = 5)
  (h_mean_3_7 : (a 3 + a 7) / 2 = 7) :
  ∃ f : ℕ → ℝ, (∀ n, a n = f n) ∧ (∀ n, f n = 2 * n - 3) :=
sorry

end arithmetic_sequence_formula_l1926_192677


namespace y_intercept_of_line_l1926_192635

/-- A line with slope -3 passing through (3,0) has y-intercept (0,9) -/
theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) :
  m = -3 →
  x₀ = 3 →
  y₀ = 0 →
  ∃ (b : ℝ), ∀ (x y : ℝ), y = m * (x - x₀) + y₀ → y = m * x + b →
  b = 9 ∧ 9 = m * 0 + b :=
by sorry

end y_intercept_of_line_l1926_192635


namespace largest_mersenne_prime_under_500_l1926_192624

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end largest_mersenne_prime_under_500_l1926_192624


namespace sum_lower_bound_l1926_192647

theorem sum_lower_bound (x : ℕ → ℝ) (h_incr : ∀ n, x n ≤ x (n + 1)) (h_x0 : x 0 = 1) :
  (∑' n, x (n + 1) / (x n)^3) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end sum_lower_bound_l1926_192647


namespace inverse_g_at_negative_one_l1926_192604

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 - 5

-- State the theorem
theorem inverse_g_at_negative_one :
  Function.invFun g (-1) = 1 :=
sorry

end inverse_g_at_negative_one_l1926_192604


namespace largest_prime_2010_digits_divisibility_l1926_192684

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_2010_digits (n : ℕ) : Prop := 10^2009 ≤ n ∧ n < 10^2010

def largest_prime_with_2010_digits (p : ℕ) : Prop :=
  is_prime p ∧ has_2010_digits p ∧ ∀ q : ℕ, is_prime q → has_2010_digits q → q ≤ p

theorem largest_prime_2010_digits_divisibility (p : ℕ) 
  (h : largest_prime_with_2010_digits p) : 
  12 ∣ (p^2 - 1) :=
sorry

end largest_prime_2010_digits_divisibility_l1926_192684


namespace rd_funding_exceeds_2_million_l1926_192673

/-- R&D funding function -/
def rd_funding (x : ℕ) : ℝ := 1.3 * (1 + 0.12)^x

/-- Year when funding exceeds 2 million -/
def exceed_year : ℕ := 4

theorem rd_funding_exceeds_2_million : 
  rd_funding exceed_year > 2 ∧ 
  ∀ y : ℕ, y < exceed_year → rd_funding y ≤ 2 := by
  sorry

#eval exceed_year + 2015

end rd_funding_exceeds_2_million_l1926_192673


namespace disc_purchase_problem_l1926_192685

theorem disc_purchase_problem (price_a price_b total_spent : ℚ) (num_b : ℕ) :
  price_a = 21/2 ∧ 
  price_b = 17/2 ∧ 
  total_spent = 93 ∧ 
  num_b = 6 →
  ∃ (num_a : ℕ), num_a + num_b = 10 ∧ 
    num_a * price_a + num_b * price_b = total_spent :=
by sorry

end disc_purchase_problem_l1926_192685


namespace green_face_probability_l1926_192630

/-- A structure representing a die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green_faces : ℕ
  red_faces : ℕ
  blue_faces : ℕ
  yellow_faces : ℕ
  total_eq_sum : sides = green_faces + red_faces + blue_faces + yellow_faces

/-- The probability of rolling a specific color on a colored die -/
def roll_probability (d : ColoredDie) (color_faces : ℕ) : ℚ :=
  color_faces / d.sides

/-- Theorem: The probability of rolling a green face on our specific 12-sided die is 1/12 -/
theorem green_face_probability :
  let d : ColoredDie := {
    sides := 12,
    green_faces := 1,
    red_faces := 5,
    blue_faces := 4,
    yellow_faces := 2,
    total_eq_sum := by simp
  }
  roll_probability d d.green_faces = 1 / 12 := by
  sorry

end green_face_probability_l1926_192630


namespace determinant_property_l1926_192625

theorem determinant_property (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 3) : 
  Matrix.det !![2*p, 2*p + 5*q; 2*r, 2*r + 5*s] = 30 := by
  sorry

end determinant_property_l1926_192625


namespace stock_price_change_l1926_192679

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let day1 := initial_price * (1 - 0.25)
  let day2 := day1 * (1 + 0.40)
  let day3 := day2 * (1 - 0.10)
  (day3 - initial_price) / initial_price = -0.055 := by
sorry

end stock_price_change_l1926_192679


namespace same_total_price_implies_sams_price_l1926_192651

/-- The price per sheet charged by Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := sorry

/-- The price per sheet charged by John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The sitting fee charged by John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The sitting fee charged by Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The number of sheets in the package -/
def num_sheets : ℕ := 12

theorem same_total_price_implies_sams_price (h : johns_price_per_sheet * num_sheets + johns_sitting_fee = sams_price_per_sheet * num_sheets + sams_sitting_fee) : 
  sams_price_per_sheet = 1.5 := by
  sorry

end same_total_price_implies_sams_price_l1926_192651


namespace smallest_resolvable_debt_is_gcd_l1926_192621

/-- The value of a pig in dollars -/
def pig_value : ℕ := 300

/-- The value of a goat in dollars -/
def goat_value : ℕ := 210

/-- The smallest positive debt that can be resolved using pigs and goats -/
def smallest_resolvable_debt : ℕ := 30

/-- Theorem stating that the smallest_resolvable_debt is the smallest positive integer
    that can be expressed as a linear combination of pig_value and goat_value -/
theorem smallest_resolvable_debt_is_gcd :
  smallest_resolvable_debt = Nat.gcd pig_value goat_value ∧
  ∀ d : ℕ, d > 0 → (∃ a b : ℤ, d = a * pig_value + b * goat_value) →
    d ≥ smallest_resolvable_debt :=
by sorry

end smallest_resolvable_debt_is_gcd_l1926_192621


namespace quadratic_function_properties_l1926_192674

/-- A quadratic function f(x) = ax² + bx satisfying specific conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ (x : ℝ), f a b x = x ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  (∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
  (Set.Icc 0 3).image (f a b) = Set.Icc (-3/2) (1/2) := by
  sorry

end quadratic_function_properties_l1926_192674


namespace intersection_when_a_is_one_range_of_a_for_empty_intersection_l1926_192620

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Part 1: Intersection when a = 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 < x ∧ x < 3} := by sorry

-- Part 2: Range of a when intersection is empty
theorem range_of_a_for_empty_intersection :
  ∀ a, A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 := by sorry

end intersection_when_a_is_one_range_of_a_for_empty_intersection_l1926_192620


namespace magnitude_of_vector_sum_l1926_192672

def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

theorem magnitude_of_vector_sum :
  ‖a + 3 • b‖ = Real.sqrt 58 := by
  sorry

end magnitude_of_vector_sum_l1926_192672


namespace crazy_silly_school_series_difference_l1926_192627

theorem crazy_silly_school_series_difference : 
  let num_books : ℕ := 15
  let num_movies : ℕ := 14
  num_books - num_movies = 1 := by
  sorry

end crazy_silly_school_series_difference_l1926_192627


namespace married_couples_with_2_to_4_children_l1926_192656

/-- The fraction of married couples with 2 to 4 children in a population with given characteristics -/
theorem married_couples_with_2_to_4_children (total_population : ℕ) 
  (married_couple_percentage : ℚ) (one_child : ℚ) (two_children : ℚ) 
  (three_children : ℚ) (four_children : ℚ) (five_children : ℚ) :
  total_population = 10000 →
  married_couple_percentage = 1/5 →
  one_child = 1/5 →
  two_children = 1/4 →
  three_children = 3/20 →
  four_children = 1/6 →
  five_children = 1/10 →
  two_children + three_children + four_children = 17/30 := by
  sorry


end married_couples_with_2_to_4_children_l1926_192656


namespace min_cars_in_group_l1926_192661

/-- Represents the properties of a group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ
  withACNoStripes : ℕ

/-- The conditions of the car group problem -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 47 ∧
  g.withStripes ≥ 55 ∧
  g.withACNoStripes ≤ 45

/-- The theorem stating the minimum number of cars in the group -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 102 := by
  sorry

#check min_cars_in_group

end min_cars_in_group_l1926_192661


namespace wire_resistance_theorem_l1926_192640

/-- The resistance of a wire loop -/
def wire_loop_resistance (R : ℝ) : ℝ := R

/-- The distance between points A and B -/
def distance_AB : ℝ := 2

/-- The resistance of one meter of wire -/
def wire_resistance_per_meter (R : ℝ) : ℝ := R

/-- Theorem: The resistance of one meter of wire is equal to the total resistance of the wire loop -/
theorem wire_resistance_theorem (R : ℝ) :
  wire_loop_resistance R = wire_resistance_per_meter R :=
by sorry

end wire_resistance_theorem_l1926_192640


namespace tims_kittens_l1926_192643

theorem tims_kittens (initial_kittens : ℕ) : 
  (initial_kittens > 0) →
  (initial_kittens * 2 / 3 * 3 / 5 = 12) →
  initial_kittens = 30 := by
sorry

end tims_kittens_l1926_192643


namespace cube_sum_magnitude_l1926_192633

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end cube_sum_magnitude_l1926_192633


namespace expression_simplification_l1926_192649

theorem expression_simplification :
  let x := (1 : ℝ) / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2)))
  let y := (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / ((9 * Real.sqrt 5 + 4 * Real.sqrt 7)^2 - 100)
  x = y := by sorry

end expression_simplification_l1926_192649


namespace parabola_translation_l1926_192626

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 3x² -/
def original_parabola : Parabola := ⟨3, 0, 0⟩

/-- The vertical translation amount -/
def translation : ℝ := -2

/-- Translates a parabola vertically by a given amount -/
def translate_vertically (p : Parabola) (t : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + t⟩

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  translate_vertically original_parabola translation

theorem parabola_translation :
  resulting_parabola = ⟨3, 0, -2⟩ := by sorry

end parabola_translation_l1926_192626


namespace hyperbola_intersection_range_l1926_192600

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, 
    (intersects_at_two_points k ∧ 
     ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧ 
                        line_l k x₁ y₁ ∧ line_l k x₂ y₂ → 
                        dot_product_condition x₁ y₁ x₂ y₂) →
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end hyperbola_intersection_range_l1926_192600


namespace quadratic_polynomial_from_roots_and_point_l1926_192675

/-- Given a quadratic polynomial q(x) with roots at x = -2 and x = 3, and q(1) = -10,
    prove that q(x) = 5/3x^2 - 5/3x - 10 -/
theorem quadratic_polynomial_from_roots_and_point (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = -2 ∨ x = 3) →  -- roots at x = -2 and x = 3
  (∃ a b c, ∀ x, q x = a * x^2 + b * x + c) →  -- q is a quadratic polynomial
  q 1 = -10 →  -- q(1) = -10
  ∀ x, q x = 5/3 * x^2 - 5/3 * x - 10 :=
by sorry

end quadratic_polynomial_from_roots_and_point_l1926_192675


namespace equation_solution_l1926_192614

theorem equation_solution : ∃ x : ℝ, 61 + 5 * x / (180 / 3) = 62 ∧ x = 12 := by
  sorry

end equation_solution_l1926_192614


namespace sod_coverage_theorem_l1926_192680

/-- The number of square sod pieces needed to cover two rectangular areas -/
def sod_squares_needed (length1 width1 length2 width2 sod_size : ℕ) : ℕ :=
  ((length1 * width1 + length2 * width2) : ℕ) / (sod_size * sod_size)

/-- Theorem stating that 1500 squares of 2x2-foot sod are needed to cover two areas of 30x40 feet and 60x80 feet -/
theorem sod_coverage_theorem :
  sod_squares_needed 30 40 60 80 2 = 1500 :=
by sorry

end sod_coverage_theorem_l1926_192680


namespace subtracted_number_l1926_192605

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem subtracted_number (x : Nat) : 
  (sum_of_digits (10^38 - x) = 330) → 
  (x = 10^37 + 3 * 10^36) :=
by sorry

end subtracted_number_l1926_192605


namespace letter_150_is_Z_l1926_192662

/-- Represents the letters in the repeating pattern -/
inductive Letter
| X
| Y
| Z

/-- The length of the repeating pattern -/
def pattern_length : Nat := 3

/-- Function to determine the nth letter in the repeating pattern -/
def nth_letter (n : Nat) : Letter :=
  match n % pattern_length with
  | 0 => Letter.Z
  | 1 => Letter.X
  | _ => Letter.Y

/-- Theorem stating that the 150th letter in the pattern is Z -/
theorem letter_150_is_Z : nth_letter 150 = Letter.Z := by
  sorry

end letter_150_is_Z_l1926_192662


namespace cricketer_matches_l1926_192615

theorem cricketer_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) :
  score1 = 40 →
  score2 = 10 →
  matches1 = 2 →
  matches2 = 3 →
  overall_avg = 22 →
  (score1 * matches1 + score2 * matches2) / (matches1 + matches2) = overall_avg →
  matches1 + matches2 = 5 := by
  sorry

end cricketer_matches_l1926_192615


namespace unrepaired_road_not_thirty_percent_l1926_192642

/-- Represents the percentage of road repaired in the first phase -/
def first_phase_repair : ℝ := 0.4

/-- Represents the percentage of remaining road repaired in the second phase -/
def second_phase_repair : ℝ := 0.3

/-- Represents the total length of the road in meters -/
def total_road_length : ℝ := 200

/-- Theorem stating that the unrepaired portion of the road is not 30% -/
theorem unrepaired_road_not_thirty_percent :
  let remaining_after_first := 1 - first_phase_repair
  let repaired_in_second := remaining_after_first * second_phase_repair
  let total_repaired := first_phase_repair + repaired_in_second
  total_repaired ≠ 0.7 := by sorry

end unrepaired_road_not_thirty_percent_l1926_192642


namespace triangle_max_area_l1926_192664

/-- The maximum area of a triangle with medians satisfying certain conditions -/
theorem triangle_max_area (m_a m_b m_c : ℝ) 
  (h_a : m_a ≤ 2) (h_b : m_b ≤ 3) (h_c : m_c ≤ 4) : 
  (∃ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) ∧
  (∀ (E' : ℝ), E' = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E' ≤ E)) →
  (∃ (E_max : ℝ), E_max = 4 ∧
  (∀ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E ≤ E_max)) :=
by sorry

end triangle_max_area_l1926_192664


namespace equation_solutions_l1926_192659

theorem equation_solutions :
  (∃ x : ℚ, (3 : ℚ) / 5 - (5 : ℚ) / 8 * x = (2 : ℚ) / 5 ∧ x = (8 : ℚ) / 25) ∧
  (∃ x : ℚ, 7 * (x - 2) = 8 * (x - 4) ∧ x = 18) := by
  sorry

end equation_solutions_l1926_192659


namespace definite_integral_abs_x_squared_minus_two_l1926_192629

theorem definite_integral_abs_x_squared_minus_two :
  ∫ x in (-2)..1, |x^2 - 2| = 1/3 + 8*Real.sqrt 2/3 := by sorry

end definite_integral_abs_x_squared_minus_two_l1926_192629


namespace multiply_18396_9999_l1926_192644

theorem multiply_18396_9999 : 18396 * 9999 = 183941604 := by
  sorry

end multiply_18396_9999_l1926_192644


namespace paul_buys_two_toys_l1926_192607

/-- The number of toys Paul can buy given his savings, allowance, and toy price -/
def toys_paul_can_buy (savings : ℕ) (allowance : ℕ) (toy_price : ℕ) : ℕ :=
  (savings + allowance) / toy_price

/-- Theorem: Paul can buy 2 toys with his savings and allowance -/
theorem paul_buys_two_toys :
  toys_paul_can_buy 3 7 5 = 2 := by
  sorry

end paul_buys_two_toys_l1926_192607


namespace circle_line_intersection_theorem_l1926_192618

/-- Circle C with equation x^2 + (y-4)^2 = 4 -/
def C (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

/-- Line l with equation y = kx -/
def l (k x y : ℝ) : Prop := y = k * x

/-- Point Q(m, n) is on segment MN -/
def Q_on_MN (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ m = t * x₁ + (1 - t) * x₂ ∧ n = t * y₁ + (1 - t) * y₂

/-- The condition 2/|OQ|^2 = 1/|OM|^2 + 1/|ON|^2 -/
def harmonic_condition (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  2 / (m^2 + n^2) = 1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2)

theorem circle_line_intersection_theorem
  (k m n x₁ y₁ x₂ y₂ : ℝ)
  (hC₁ : C x₁ y₁)
  (hC₂ : C x₂ y₂)
  (hl₁ : l k x₁ y₁)
  (hl₂ : l k x₂ y₂)
  (hQ : Q_on_MN m n x₁ y₁ x₂ y₂)
  (hHarmonic : harmonic_condition m n x₁ y₁ x₂ y₂)
  (hm : m ∈ Set.Ioo (-Real.sqrt 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3)) :
  n = Real.sqrt (15 * m^2 + 180) / 5 :=
sorry

end circle_line_intersection_theorem_l1926_192618


namespace smallest_n_value_l1926_192628

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def quadruplet_count : ℕ := 84000

/-- The given GCD value -/
def gcd_value : ℕ := 84

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    satisfying gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n that satisfies the conditions -/
theorem smallest_n_value : 
  (∀ m < 1555848, count_quadruplets m ≠ quadruplet_count) ∧ 
  count_quadruplets 1555848 = quadruplet_count := by sorry

end smallest_n_value_l1926_192628


namespace team_selection_count_l1926_192609

def num_boys : ℕ := 7
def num_girls : ℕ := 10
def team_size : ℕ := 5
def min_girls : ℕ := 2

theorem team_selection_count :
  (Finset.sum (Finset.range (team_size - min_girls + 1))
    (λ k => Nat.choose num_girls (min_girls + k) * Nat.choose num_boys (team_size - (min_girls + k)))) = 5817 :=
by sorry

end team_selection_count_l1926_192609


namespace geometric_sequence_ratio_l1926_192690

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  a₃ * a₅ * a₇ * a₉ * a₁₁ = 243 → a₁₀^2 / a₁₃ = 3 := by
  sorry

end geometric_sequence_ratio_l1926_192690


namespace house_height_calculation_l1926_192639

/-- Given two trees with consistent shadow ratios and a house with a known shadow length,
    prove that the height of the house can be determined. -/
theorem house_height_calculation (tree1_height tree1_shadow tree2_height tree2_shadow house_shadow : ℝ)
    (h1 : tree1_height > 0)
    (h2 : tree2_height > 0)
    (h3 : tree1_shadow > 0)
    (h4 : tree2_shadow > 0)
    (h5 : house_shadow > 0)
    (h6 : tree1_shadow / tree1_height = tree2_shadow / tree2_height) :
    ∃ (house_height : ℝ), house_height = tree1_height * (house_shadow / tree1_shadow) :=
  sorry

end house_height_calculation_l1926_192639


namespace largest_power_of_five_in_sum_of_factorials_l1926_192688

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 98 + factorial 99 + factorial 100

theorem largest_power_of_five_in_sum_of_factorials :
  (∃ k : ℕ, sum_of_factorials = 5^26 * k ∧ ¬∃ m : ℕ, sum_of_factorials = 5^27 * m) := by
  sorry

end largest_power_of_five_in_sum_of_factorials_l1926_192688


namespace chocolate_bars_count_l1926_192697

theorem chocolate_bars_count (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 21) 
  (h2 : bars_per_box = 25) : 
  small_boxes * bars_per_box = 525 := by
  sorry

end chocolate_bars_count_l1926_192697


namespace sqrt_2_pow_12_l1926_192670

theorem sqrt_2_pow_12 : Real.sqrt (2^12) = 64 := by
  sorry

end sqrt_2_pow_12_l1926_192670


namespace actual_sleep_time_l1926_192638

/-- The required sleep time for middle school students -/
def requiredSleepTime : ℝ := 9

/-- The recorded excess sleep time for Xiao Ming -/
def recordedExcessTime : ℝ := 0.4

/-- Theorem: Actual sleep time is the sum of required sleep time and recorded excess time -/
theorem actual_sleep_time : 
  requiredSleepTime + recordedExcessTime = 9.4 := by
  sorry

end actual_sleep_time_l1926_192638


namespace ben_egg_count_l1926_192632

/-- The number of trays Ben was given -/
def num_trays : ℕ := 7

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 10

/-- The total number of eggs Ben examined -/
def total_eggs : ℕ := num_trays * eggs_per_tray

theorem ben_egg_count : total_eggs = 70 := by
  sorry

end ben_egg_count_l1926_192632


namespace decision_box_is_diamond_l1926_192601

/-- A type representing different shapes that can be used in a flowchart --/
inductive FlowchartShape
  | Rectangle
  | Diamond
  | Oval
  | Parallelogram

/-- A function that returns the shape used for decision boxes in a flowchart --/
def decisionBoxShape : FlowchartShape := FlowchartShape.Diamond

/-- Theorem stating that the decision box in a flowchart is represented by a diamond shape --/
theorem decision_box_is_diamond : decisionBoxShape = FlowchartShape.Diamond := by
  sorry

end decision_box_is_diamond_l1926_192601


namespace inverse_proportion_problem_l1926_192608

/-- Given that x and y are inversely proportional, x + y = 30, and x - y = 10,
    prove that y = 200/7 when x = 7 -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 7 → y = 200 / 7 := by
  sorry

end inverse_proportion_problem_l1926_192608


namespace f_range_f_period_one_l1926_192667

-- Define the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if x - ⌊x⌋ ≤ 1/2 then ⌊x⌋ else ⌈x⌉

-- Define the function f(x) = x - {x}
noncomputable def f (x : ℝ) : ℝ := x - nearest_integer x

-- Theorem stating the range of f(x)
theorem f_range : Set.range f = Set.Ioc (-1/2) (1/2) := by sorry

-- Theorem stating that f(x) has a period of 1
theorem f_period_one (x : ℝ) : f (x + 1) = f x := by sorry

end f_range_f_period_one_l1926_192667


namespace curve_self_intersection_l1926_192698

-- Define the parametric equations
def x (t : ℝ) : ℝ := t^2 + 3
def y (t : ℝ) : ℝ := t^3 - 6*t + 4

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 9 ∧ y a = 4 := by
  sorry

end curve_self_intersection_l1926_192698


namespace sqrt_twelve_equals_two_sqrt_three_l1926_192693

theorem sqrt_twelve_equals_two_sqrt_three :
  Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

end sqrt_twelve_equals_two_sqrt_three_l1926_192693


namespace perimeter_difference_l1926_192681

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of a modified rectangle with a vertical shift --/
def modifiedRectanglePerimeter (length width shift : ℕ) : ℕ :=
  2 * length + 2 * width + 2 * shift

/-- The positive difference between the perimeter of a 6x1 rectangle with a vertical shift
    and the perimeter of a 4x1 rectangle is 6 units --/
theorem perimeter_difference : 
  modifiedRectanglePerimeter 6 1 1 - rectanglePerimeter 4 1 = 6 := by
  sorry

end perimeter_difference_l1926_192681


namespace function_inequality_l1926_192606

/-- Given functions f and g, prove that if 2f(x) ≥ g(x) for all x > 0, then a ≤ 4 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, 2 * (x * Real.log x) ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry


end function_inequality_l1926_192606


namespace cos_165_degrees_l1926_192683

theorem cos_165_degrees : Real.cos (165 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end cos_165_degrees_l1926_192683


namespace min_value_trig_expression_min_value_trig_expression_achievable_l1926_192613

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 ≥ 121 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ θ φ : ℝ, (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
by sorry

end min_value_trig_expression_min_value_trig_expression_achievable_l1926_192613


namespace alex_jane_pen_difference_l1926_192666

/-- Calculates the number of pens Alex has after a given number of weeks -/
def alex_pens (initial_pens : ℕ) (weeks : ℕ) : ℕ :=
  initial_pens * (2 ^ weeks)

/-- The number of pens Jane has after a month -/
def jane_pens : ℕ := 16

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The initial number of pens Alex has -/
def alex_initial_pens : ℕ := 4

theorem alex_jane_pen_difference :
  alex_pens alex_initial_pens weeks_in_month - jane_pens = 16 := by
  sorry


end alex_jane_pen_difference_l1926_192666


namespace expression_value_l1926_192612

theorem expression_value (x : ℤ) (h : x = -3) : x^2 - 4*(x - 5) = 41 := by
  sorry

end expression_value_l1926_192612


namespace arithmetic_sequence_sum_l1926_192665

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15, 
    prove that a_3 + a_4 + ... + a_9 equals 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                                -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=        -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l1926_192665


namespace triangle_cos_C_l1926_192655

theorem triangle_cos_C (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h2 : A + B + C = Real.pi) (h3 : Real.sin A = 4/5) (h4 : Real.cos B = 3/5) : 
  Real.cos C = 7/25 := by
sorry

end triangle_cos_C_l1926_192655


namespace minimal_extensive_h_21_l1926_192692

/-- An extensive function is a function from positive integers to integers
    such that f(x) + f(y) ≥ x² + y² for all positive integers x and y. -/
def Extensive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ (x.val ^ 2 : ℤ) + (y.val ^ 2 : ℤ)

/-- The sum of the first 30 values of an extensive function -/
def SumFirst30 (f : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f ⟨i + 1, Nat.succ_pos i⟩)

/-- An extensive function with minimal sum of first 30 values -/
def MinimalExtensive (h : ℕ+ → ℤ) : Prop :=
  Extensive h ∧ ∀ g : ℕ+ → ℤ, Extensive g → SumFirst30 h ≤ SumFirst30 g

theorem minimal_extensive_h_21 (h : ℕ+ → ℤ) (hmin : MinimalExtensive h) :
    h ⟨21, by norm_num⟩ ≥ 301 := by
  sorry

end minimal_extensive_h_21_l1926_192692


namespace f_properties_l1926_192653

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧  -- f is increasing on (-∞, 0)
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧  -- f is decreasing on (0, 2)
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧  -- f is increasing on (2, +∞)
  (∀ x, x ≠ 0 → f x ≤ f 0) ∧  -- f(0) is a local maximum
  (∀ x, x ≠ 2 → f x ≥ f 2) ∧  -- f(2) is a local minimum
  f 0 = 0 ∧  -- value at x = 0
  f 2 = -4  -- value at x = 2
  := by sorry

end f_properties_l1926_192653


namespace election_majority_l1926_192696

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 160 := by
sorry

end election_majority_l1926_192696
