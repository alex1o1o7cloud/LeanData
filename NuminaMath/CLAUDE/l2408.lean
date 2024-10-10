import Mathlib

namespace unique_box_configuration_l2408_240866

/-- Represents a square piece --/
structure Square :=
  (id : Nat)

/-- Represents the F-shaped figure --/
def FShape := List Square

/-- Represents additional squares --/
def AdditionalSquares := List Square

/-- Represents a possible configuration of an open rectangular box --/
def BoxConfiguration := List Square

/-- A function that attempts to form an open rectangular box --/
def formBox (f : FShape) (add : AdditionalSquares) : Option BoxConfiguration :=
  sorry

/-- The main theorem stating there's exactly one valid configuration --/
theorem unique_box_configuration 
  (f : FShape) 
  (add : AdditionalSquares) 
  (h1 : f.length = 7) 
  (h2 : add.length = 3) : 
  ∃! (box : BoxConfiguration), formBox f add = some box :=
sorry

end unique_box_configuration_l2408_240866


namespace simplify_sqrt_expression_l2408_240804

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = x^3 / 3 + 1 / (3 * x^3) := by
  sorry

end simplify_sqrt_expression_l2408_240804


namespace angle_COD_measure_l2408_240836

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (belongs_to : Point → Circle → Prop)
variable (intersect_at : Circle → Circle → Point → Point → Prop)
variable (tangent_to : Point → Circle → Prop)
variable (lies_on_ray : Point → Point → Point → Prop)
variable (is_midpoint : Point → Point → Point → Prop)
variable (is_circumcenter : Point → Point → Point → Point → Prop)
variable (angle_measure : Point → Point → Point → ℝ)

-- Define the given points and circles
variable (ω₁ ω₂ : Circle)
variable (A B P Q R S O C D : Point)

-- State the theorem
theorem angle_COD_measure :
  intersect_at ω₁ ω₂ A B →
  tangent_to P ω₁ →
  tangent_to Q ω₂ →
  lies_on_ray R P A →
  lies_on_ray S Q A →
  angle_measure A P Q = 45 →
  angle_measure A Q P = 30 →
  is_circumcenter O A S R →
  is_midpoint C A P →
  is_midpoint D A Q →
  angle_measure C O D = 142.5 := by
  sorry

end angle_COD_measure_l2408_240836


namespace f_decreasing_implies_a_range_l2408_240807

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6) (1/2) ∧ a ≠ 1/2 :=
sorry

end f_decreasing_implies_a_range_l2408_240807


namespace difference_of_squares_l2408_240835

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end difference_of_squares_l2408_240835


namespace six_doctors_three_days_l2408_240891

/-- The number of ways for a given number of doctors to each choose one rest day from a given number of days. -/
def restDayChoices (numDoctors : ℕ) (numDays : ℕ) : ℕ :=
  numDays ^ numDoctors

/-- Theorem stating that for 6 doctors choosing from 3 days, the number of choices is 3^6. -/
theorem six_doctors_three_days : 
  restDayChoices 6 3 = 3^6 := by
  sorry

end six_doctors_three_days_l2408_240891


namespace rectangular_field_area_l2408_240840

/-- Theorem: Area of a rectangular field -/
theorem rectangular_field_area (width : ℝ) : 
  (width ≥ 0) →
  (16 * width + 54 = 22 * width) → 
  (16 * width = 144) :=
by
  sorry

end rectangular_field_area_l2408_240840


namespace triangle_most_stable_l2408_240889

-- Define the possible shapes
inductive Shape
  | Heptagon
  | Hexagon
  | Pentagon
  | Triangle

-- Define a property for stability
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Theorem stating that the triangle is the most stable shape
theorem triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.Triangle :=
by
  sorry

end triangle_most_stable_l2408_240889


namespace food_bank_donation_ratio_l2408_240808

theorem food_bank_donation_ratio :
  let foster_chickens : ℕ := 45
  let american_water := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_water := american_water - 30
  let total_items : ℕ := 375
  let boudin_chickens := total_items - (foster_chickens + american_water + hormel_chickens + del_monte_water)
  (boudin_chickens : ℚ) / hormel_chickens = 1 / 3 :=
by sorry

end food_bank_donation_ratio_l2408_240808


namespace license_plate_count_l2408_240861

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 4

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

theorem license_plate_count : 
  num_digits ^ digits_count * num_letters ^ letters_count * block_positions = 878800000 := by
  sorry

end license_plate_count_l2408_240861


namespace problem_solution_l2408_240829

theorem problem_solution (a b : ℕ+) (h : (a.val^3 - a.val^2 + 1) * (b.val^3 - b.val^2 + 2) = 2020) :
  10 * a.val + b.val = 53 := by
  sorry

end problem_solution_l2408_240829


namespace allowance_spending_l2408_240881

theorem allowance_spending (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_amount : ℚ) : 
  weekly_allowance = 3.75 →
  arcade_fraction = 3/5 →
  candy_amount = 1 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
  sorry

end allowance_spending_l2408_240881


namespace leahs_calculation_l2408_240846

theorem leahs_calculation (y : ℕ) (h : (y + 4) * 5 = 140) : y * 5 + 4 = 124 := by
  sorry

end leahs_calculation_l2408_240846


namespace product_sum_multiplier_l2408_240865

theorem product_sum_multiplier (a b k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = k * (a * b)) (h2 : 1 / a + 1 / b = 6) : k = 6 := by
  sorry

end product_sum_multiplier_l2408_240865


namespace baseball_glove_discount_percentage_l2408_240817

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_cleats_price : ℝ := 10
def baseball_cleats_pairs : ℕ := 2
def total_amount : ℝ := 79

theorem baseball_glove_discount_percentage :
  let other_items_total : ℝ := baseball_cards_price + baseball_bat_price + baseball_cleats_price * baseball_cleats_pairs
  let glove_sale_price : ℝ := total_amount - other_items_total
  let discount_percentage : ℝ := (baseball_glove_original_price - glove_sale_price) / baseball_glove_original_price * 100
  discount_percentage = 20 := by sorry

end baseball_glove_discount_percentage_l2408_240817


namespace problem_statement_l2408_240828

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2*a^2 + 4*a*b + 2*b^2 - 6 = 12 := by
  sorry

end problem_statement_l2408_240828


namespace household_coffee_expense_l2408_240855

def weekly_coffee_expense (person_a_cups : ℕ) (person_a_ounces : ℝ)
                          (person_b_cups : ℕ) (person_b_ounces : ℝ)
                          (person_c_cups : ℕ) (person_c_ounces : ℝ)
                          (person_c_days : ℕ) (coffee_cost : ℝ) : ℝ :=
  let person_a_weekly := person_a_cups * person_a_ounces * 7
  let person_b_weekly := person_b_cups * person_b_ounces * 7
  let person_c_weekly := person_c_cups * person_c_ounces * person_c_days
  let total_weekly_ounces := person_a_weekly + person_b_weekly + person_c_weekly
  total_weekly_ounces * coffee_cost

theorem household_coffee_expense :
  weekly_coffee_expense 3 0.4 1 0.6 2 0.5 5 1.25 = 22 := by
  sorry

end household_coffee_expense_l2408_240855


namespace white_washing_cost_is_4530_l2408_240821

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_length * door_width
  let window_area := window_length * window_width * num_windows
  let paintable_area := wall_area - door_area - window_area
  paintable_area * cost_per_sqft

/-- The cost of white washing the room is 4530 rupees -/
theorem white_washing_cost_is_4530 :
  white_washing_cost 25 15 12 6 3 4 3 3 5 = 4530 := by
  sorry

end white_washing_cost_is_4530_l2408_240821


namespace parallelogram_base_length_l2408_240806

/-- Proves that a parallelogram with area 44 cm² and height 11 cm has a base of 4 cm -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (is_parallelogram : Bool) 
  (h1 : is_parallelogram = true)
  (h2 : area = 44)
  (h3 : height = 11) :
  area / height = 4 := by
  sorry

end parallelogram_base_length_l2408_240806


namespace exactly_three_sets_l2408_240832

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (length_ge_two : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 150 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_sets : 
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    sets.card = 3 := by sorry

end exactly_three_sets_l2408_240832


namespace fraction_multiplication_l2408_240872

theorem fraction_multiplication : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_multiplication_l2408_240872


namespace jasons_games_this_month_l2408_240818

/-- 
Given that:
- Jason went to 17 games last month
- Jason plans to go to 16 games next month
- Jason will attend 44 games in all

Prove that Jason went to 11 games this month.
-/
theorem jasons_games_this_month 
  (games_last_month : ℕ) 
  (games_next_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44) :
  total_games - (games_last_month + games_next_month) = 11 := by
  sorry

end jasons_games_this_month_l2408_240818


namespace analysis_seeks_sufficient_condition_l2408_240816

/-- Represents a mathematical method for proving inequalities -/
inductive ProofMethod
| Analysis
| Synthesis

/-- Represents types of conditions in mathematical proofs -/
inductive ConditionType
| Sufficient
| Necessary
| NecessaryAndSufficient
| Neither

/-- Represents an inequality to be proved -/
structure Inequality where
  -- We don't need to specify the actual inequality, just that it exists
  dummy : Unit

/-- Function that represents the process of seeking a condition in the analysis method -/
def seekCondition (m : ProofMethod) (i : Inequality) : ConditionType :=
  match m with
  | ProofMethod.Analysis => ConditionType.Sufficient
  | ProofMethod.Synthesis => ConditionType.Neither -- This is arbitrary for non-Analysis methods

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_seeks_sufficient_condition (i : Inequality) :
  seekCondition ProofMethod.Analysis i = ConditionType.Sufficient := by
  sorry

#check analysis_seeks_sufficient_condition

end analysis_seeks_sufficient_condition_l2408_240816


namespace live_flowers_l2408_240862

theorem live_flowers (total : ℕ) (withered : ℕ) (h1 : total = 13) (h2 : withered = 7) :
  total - withered = 6 := by
  sorry

end live_flowers_l2408_240862


namespace ap_sum_terms_l2408_240868

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℤ     -- First term
  d : ℤ      -- Common difference

/-- Calculates the sum of the first n terms of an arithmetic progression -/
def sum_of_terms (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  n * (2 * ap.a₁ + (n - 1) * ap.d) / 2

/-- Theorem: The number of terms needed for the sum to equal 3069 in the given arithmetic progression is either 9 or 31 -/
theorem ap_sum_terms (ap : ArithmeticProgression) 
  (h1 : ap.a₁ = 429) 
  (h2 : ap.d = -22) : 
  (∃ n : ℕ, sum_of_terms ap n = 3069) → (n = 9 ∨ n = 31) :=
sorry

end ap_sum_terms_l2408_240868


namespace valid_parameterizations_l2408_240800

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- A point is on the line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) : Prop :=
  line_equation p.1 p.2

/-- The direction vector is valid if it's parallel to (1, -3) -/
def valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, -3*k)

/-- A parameterization is valid if its point is on the line and its direction is valid -/
def valid_parameterization (p v : ℝ × ℝ) : Prop :=
  point_on_line p ∧ valid_direction v

theorem valid_parameterizations :
  valid_parameterization (4/3, 0) (1, -3) ∧
  valid_parameterization (-2, 10) (-3, 9) :=
by sorry

end valid_parameterizations_l2408_240800


namespace mason_hotdog_weight_l2408_240857

/-- Represents the weight of different food items in ounces -/
def food_weight : (String → Nat)
| "hotdog" => 2
| "burger" => 5
| "pie" => 10
| _ => 0

/-- The number of burgers Noah ate -/
def noah_burgers : Nat := 8

/-- Calculates the number of pies Jacob ate -/
def jacob_pies : Nat := noah_burgers - 3

/-- Calculates the number of hotdogs Mason ate -/
def mason_hotdogs : Nat := 3 * jacob_pies

/-- Theorem stating that the total weight of hotdogs Mason ate is 30 ounces -/
theorem mason_hotdog_weight :
  mason_hotdogs * food_weight "hotdog" = 30 := by
  sorry

end mason_hotdog_weight_l2408_240857


namespace ellipse_foci_distance_l2408_240842

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 20 = 0

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 0

/-- Theorem stating that the distance between the foci of the given ellipse is 0 -/
theorem ellipse_foci_distance :
  ∀ x y : ℝ, ellipse_equation x y → foci_distance = 0 := by
  sorry

end ellipse_foci_distance_l2408_240842


namespace original_savings_amount_l2408_240885

/-- Proves that the original savings amount was $11,000 given the spending pattern and remaining balance. -/
theorem original_savings_amount (initial_savings : ℝ) : 
  initial_savings * (1 - 0.2 - 0.4) - 1500 = 2900 → 
  initial_savings = 11000 := by
sorry

end original_savings_amount_l2408_240885


namespace category_A_sample_size_l2408_240860

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  categoryA : ℕ
  categoryB : ℕ
  categoryC : ℕ

/-- Calculates the number of students selected from a category in stratified sampling -/
def stratifiedSample (categories : SchoolCategories) (totalSample : ℕ) (categorySize : ℕ) : ℕ :=
  (categorySize * totalSample) / (categories.categoryA + categories.categoryB + categories.categoryC)

/-- Theorem: The number of students selected from Category A in the given scenario is 200 -/
theorem category_A_sample_size :
  let categories := SchoolCategories.mk 2000 3000 4000
  let totalSample := 900
  stratifiedSample categories totalSample categories.categoryA = 200 := by
  sorry

end category_A_sample_size_l2408_240860


namespace indeterminate_existence_l2408_240805

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Q : U → Prop)  -- Q(x) means x is a quadrilateral
variable (A : U → Prop)  -- A(x) means x has property A

-- State the theorem
theorem indeterminate_existence (h : ¬(∀ x, Q x → A x)) :
  ¬(∀ p q : Prop, p = (∃ x, Q x ∧ A x) → (q = True ∨ q = False)) :=
sorry

end indeterminate_existence_l2408_240805


namespace coronavirus_case_increase_l2408_240870

theorem coronavirus_case_increase (initial_cases : ℕ) 
  (second_day_recoveries : ℕ) (third_day_new_cases : ℕ) 
  (third_day_recoveries : ℕ) (final_total_cases : ℕ) :
  initial_cases = 2000 →
  second_day_recoveries = 50 →
  third_day_new_cases = 1500 →
  third_day_recoveries = 200 →
  final_total_cases = 3750 →
  ∃ (second_day_increase : ℕ),
    final_total_cases = initial_cases + second_day_increase - second_day_recoveries + 
      third_day_new_cases - third_day_recoveries ∧
    second_day_increase = 750 :=
by sorry

end coronavirus_case_increase_l2408_240870


namespace arithmetic_sequence_property_l2408_240877

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the theorem
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (l m n : ℕ) (a' b c : ℝ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_l : a l = 1 / a')
  (h_m : a m = 1 / b)
  (h_n : a n = 1 / c) :
  (l - m : ℝ) * a' * b + (m - n : ℝ) * b * c + (n - l : ℝ) * c * a' = 0 := by
  sorry

end arithmetic_sequence_property_l2408_240877


namespace multiply_by_48_equals_173_times_240_l2408_240899

theorem multiply_by_48_equals_173_times_240 : 48 * 865 = 173 * 240 := by
  sorry

end multiply_by_48_equals_173_times_240_l2408_240899


namespace complex_on_y_axis_l2408_240848

theorem complex_on_y_axis (a : ℝ) : 
  let z : ℂ := (a - 3 * Complex.I) / (1 - Complex.I)
  (Complex.re z = 0) → a = -3 := by
sorry

end complex_on_y_axis_l2408_240848


namespace identical_solutions_iff_k_neg_one_l2408_240809

/-- 
Proves that the equations y = x^2 and y = 2x + k have two identical solutions 
if and only if k = -1.
-/
theorem identical_solutions_iff_k_neg_one (k : ℝ) : 
  (∃ x y : ℝ, y = x^2 ∧ y = 2*x + k ∧ 
   (∀ x' y' : ℝ, y' = x'^2 ∧ y' = 2*x' + k → x' = x ∧ y' = y)) ↔ 
  k = -1 := by
  sorry

end identical_solutions_iff_k_neg_one_l2408_240809


namespace negative_twenty_is_spend_l2408_240893

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℕ)
| spend (amount : ℕ)

/-- Converts a transaction to its signed representation -/
def signedAmount (t : Transaction) : ℤ :=
  match t with
  | Transaction.receive n => n
  | Transaction.spend n => -n

/-- The convention of representing transactions -/
structure TransactionConvention where
  positiveIsReceive : ∀ (n : ℕ), signedAmount (Transaction.receive n) > 0
  negativeIsSpend : ∀ (n : ℕ), signedAmount (Transaction.spend n) < 0

/-- The main theorem -/
theorem negative_twenty_is_spend (conv : TransactionConvention) :
  signedAmount (Transaction.spend 20) = -20 :=
by sorry

end negative_twenty_is_spend_l2408_240893


namespace max_value_of_reciprocal_sum_l2408_240831

theorem max_value_of_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' a' b' : ℝ, a' > 1 → b' > 1 → a'^x' = 3 → b'^y' = 3 → a' + b' = 2 * Real.sqrt 3 → 
    1/x' + 1/y' ≤ 1/x + 1/y) ∧ 1/x + 1/y = 1 :=
by sorry

end max_value_of_reciprocal_sum_l2408_240831


namespace max_area_triangle_l2408_240803

/-- Given points A, B, C, and P in a plane with specific distances, 
    prove that the maximum possible area of triangle ABC is 18.5 -/
theorem max_area_triangle (A B C P : ℝ × ℝ) : 
  let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
  let PC := Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  PA = 5 ∧ PB = 4 ∧ PC = 3 ∧ BC = 5 →
  (∀ A' : ℝ × ℝ, 
    let PA' := Real.sqrt ((A'.1 - P.1)^2 + (A'.2 - P.2)^2)
    PA' = 5 →
    let area := abs ((A'.1 - B.1) * (C.2 - B.2) - (A'.2 - B.2) * (C.1 - B.1)) / 2
    area ≤ 18.5) :=
by sorry


end max_area_triangle_l2408_240803


namespace pizza_problem_l2408_240853

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The fraction of pizza eaten after n trips to the refrigerator -/
def pizza_eaten (n : ℕ) : ℚ :=
  geometric_sum (1/3) (1/3) n

theorem pizza_problem : pizza_eaten 6 = 364/729 := by
  sorry

end pizza_problem_l2408_240853


namespace inverse_proportion_ratios_l2408_240837

/-- Given that c is inversely proportional to d, prove the ratios of their values -/
theorem inverse_proportion_ratios 
  (k : ℝ) 
  (c d : ℝ → ℝ) 
  (h1 : ∀ x, c x * d x = k) 
  (c1 c2 d1 d2 c3 d3 : ℝ) 
  (h2 : c1 / c2 = 4 / 5) 
  (h3 : c3 = 2 * c1) :
  d1 / d2 = 5 / 4 ∧ d3 = d1 / 2 := by
sorry

end inverse_proportion_ratios_l2408_240837


namespace largest_n_for_product_l2408_240878

/-- Arithmetic sequence (a_n) with initial value 1 and common difference x -/
def a (n : ℕ) (x : ℤ) : ℤ := 1 + (n - 1 : ℤ) * x

/-- Arithmetic sequence (b_n) with initial value 1 and common difference y -/
def b (n : ℕ) (y : ℤ) : ℤ := 1 + (n - 1 : ℤ) * y

theorem largest_n_for_product (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h_a2_b2 : 1 < a 2 x ∧ a 2 x ≤ b 2 y) :
  (∃ n : ℕ, a n x * b n y = 1764) →
  (∀ m : ℕ, a m x * b m y = 1764 → m ≤ 44) ∧
  (∃ n : ℕ, a n x * b n y = 1764 ∧ n = 44) :=
by sorry

end largest_n_for_product_l2408_240878


namespace acute_triangle_existence_l2408_240884

theorem acute_triangle_existence (d : Fin 12 → ℝ) 
  (h_range : ∀ i, 1 < d i ∧ d i < 12) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (d i < d j + d k) ∧ 
    (d j < d i + d k) ∧ 
    (d k < d i + d j) ∧
    (d i)^2 + (d j)^2 > (d k)^2 :=
by sorry

end acute_triangle_existence_l2408_240884


namespace one_third_vector_AB_l2408_240892

/-- Given two vectors OA and OB in 2D space, prove that 1/3 of vector AB equals (-3, -2) -/
theorem one_third_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (2, 8) → OB = (-7, 2) → (1 / 3 : ℝ) • (OB - OA) = (-3, -2) := by sorry

end one_third_vector_AB_l2408_240892


namespace total_pastries_is_97_l2408_240838

/-- Given the number of pastries for Grace, calculate the total number of pastries for Grace, Calvin, Phoebe, and Frank. -/
def totalPastries (grace : ℕ) : ℕ :=
  let calvin := grace - 5
  let phoebe := grace - 5
  let frank := calvin - 8
  grace + calvin + phoebe + frank

/-- Theorem stating that given Grace has 30 pastries, the total number of pastries for all four is 97. -/
theorem total_pastries_is_97 : totalPastries 30 = 97 := by
  sorry

#eval totalPastries 30

end total_pastries_is_97_l2408_240838


namespace fraction_comparison_l2408_240897

theorem fraction_comparison : (10^1984 + 1) / (10^1985) > (10^1985 + 1) / (10^1986) := by
  sorry

end fraction_comparison_l2408_240897


namespace union_of_M_and_N_l2408_240812

def M (a : ℕ) : Set ℕ := {3, 2^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_M_and_N (a b : ℕ) :
  M a ∩ N a b = {2} →
  M a ∪ N a b = {1, 2, 3} := by
  sorry

end union_of_M_and_N_l2408_240812


namespace hyperbola_equation_l2408_240879

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0 →
    ∃ t : ℝ, (b*x + a*y = 0 ∨ b*x - a*y = 0) →
      (x - 3)^2 + y^2 = 4) →
  3^2 = a^2 - b^2 →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end hyperbola_equation_l2408_240879


namespace seven_thirteenths_repeating_block_length_l2408_240880

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7 is prime -/
axiom seven_prime : Nat.Prime 7

/-- 13 is prime -/
axiom thirteen_prime : Nat.Prime 13

/-- The theorem stating that the length of the repeating block in the decimal expansion of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  repeating_block_length = 6 := by sorry

end seven_thirteenths_repeating_block_length_l2408_240880


namespace length_AC_l2408_240882

-- Define the right triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Condition 1 and 2: ABC is a right triangle with angle C = 90°
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- Condition 3: AB = 9
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9 ∧
  -- Condition 4: cos B = 2/3
  (C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2/3

-- Theorem statement
theorem length_AC (A B C : ℝ × ℝ) (h : triangle_ABC A B C) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end length_AC_l2408_240882


namespace rectangle_length_l2408_240875

/-- Given a rectangle with perimeter 42 and width 4, its length is 17. -/
theorem rectangle_length (P w l : ℝ) (h1 : P = 42) (h2 : w = 4) (h3 : P = 2 * (l + w)) : l = 17 := by
  sorry

end rectangle_length_l2408_240875


namespace magistrate_seating_arrangements_l2408_240851

/-- Represents the number of people of each nationality -/
def magistrates : Finset (Nat × Nat) := {(2, 3), (1, 4)}

/-- The total number of magistrate members -/
def total_members : Nat := (magistrates.sum (λ x => x.1))

/-- Calculates the number of valid seating arrangements -/
def valid_arrangements (m : Finset (Nat × Nat)) (total : Nat) : Nat :=
  sorry

theorem magistrate_seating_arrangements :
  valid_arrangements magistrates total_members = 1895040 := by
  sorry

end magistrate_seating_arrangements_l2408_240851


namespace power_simplification_l2408_240856

theorem power_simplification : 16^6 * 4^6 * 16^10 * 4^10 = 64^16 := by
  sorry

end power_simplification_l2408_240856


namespace sqrt_sum_equation_l2408_240876

theorem sqrt_sum_equation (a b : ℚ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
by sorry

end sqrt_sum_equation_l2408_240876


namespace cos_330_deg_l2408_240844

/-- Cosine of 330 degrees is equal to √3/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_deg_l2408_240844


namespace zacks_marbles_l2408_240874

theorem zacks_marbles (friends : ℕ) (ratio : List ℕ) (leftover : ℕ) (initial : ℕ) :
  friends = 9 →
  ratio = [5, 6, 7, 8, 9, 10, 11, 12, 13] →
  leftover = 27 →
  initial = (ratio.sum * 3) + leftover →
  initial = 270 :=
by sorry

end zacks_marbles_l2408_240874


namespace inequality_solution_l2408_240867

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-2) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 6 ∪ Set.Ioi 7) ↔ 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ 
   (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 24)) :=
by sorry

end inequality_solution_l2408_240867


namespace vector_equality_properties_l2408_240843

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def same_direction (a b : E) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_equality_properties (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (heq : a = b) :
  same_direction a b ∧ ‖a‖ = ‖b‖ := by sorry

end vector_equality_properties_l2408_240843


namespace largest_difference_l2408_240896

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) := by
  sorry

end largest_difference_l2408_240896


namespace complex_equation_solution_l2408_240822

theorem complex_equation_solution :
  ∀ (z : ℂ), z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by sorry

end complex_equation_solution_l2408_240822


namespace triangle_properties_l2408_240810

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * c * Real.sin A = Real.sqrt 3 * a →
  b = 2 →
  c = Real.sqrt 7 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end triangle_properties_l2408_240810


namespace kgonal_number_formula_l2408_240886

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
  | 4 => (n^2 : ℚ)
  | 5 => (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
  | 6 => (2 : ℚ) * n^2 - (n : ℚ)
  | _ => ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n

theorem kgonal_number_formula (n k : ℕ) (h : k ≥ 3) :
  N n k = ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n := by
  sorry

end kgonal_number_formula_l2408_240886


namespace squared_difference_product_l2408_240849

theorem squared_difference_product (a b : ℝ) : 
  a = 4 + 2 * Real.sqrt 5 → 
  b = 4 - 2 * Real.sqrt 5 → 
  a^2 * b - a * b^2 = -16 * Real.sqrt 5 := by
  sorry

end squared_difference_product_l2408_240849


namespace ones_digit_of_6_pow_45_l2408_240814

theorem ones_digit_of_6_pow_45 : ∃ n : ℕ, 6^45 ≡ 6 [ZMOD 10] :=
sorry

end ones_digit_of_6_pow_45_l2408_240814


namespace symmetric_probability_is_one_over_429_l2408_240834

/-- Represents a coloring of the 13-square array -/
def Coloring := Fin 13 → Bool

/-- The total number of squares in the array -/
def totalSquares : ℕ := 13

/-- The number of red squares -/
def redSquares : ℕ := 8

/-- The number of blue squares -/
def blueSquares : ℕ := 5

/-- Predicate to check if a coloring is symmetric under 90-degree rotation -/
def isSymmetric (c : Coloring) : Prop := sorry

/-- The number of symmetric colorings -/
def symmetricColorings : ℕ := 3

/-- The total number of possible colorings -/
def totalColorings : ℕ := Nat.choose totalSquares blueSquares

/-- The probability of selecting a symmetric coloring -/
def symmetricProbability : ℚ := symmetricColorings / totalColorings

theorem symmetric_probability_is_one_over_429 : 
  symmetricProbability = 1 / 429 := by sorry

end symmetric_probability_is_one_over_429_l2408_240834


namespace currency_exchange_problem_l2408_240841

/-- The exchange rate from U.S. dollars to British pounds -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in British pounds -/
def amount_spent : ℚ := 72

/-- The remaining amount in British pounds as a function of initial U.S. dollars -/
def remaining (d : ℚ) : ℚ := 4 * d

theorem currency_exchange_problem (d : ℚ) : 
  (exchange_rate * d - amount_spent = remaining d) → d = -30 := by
  sorry

end currency_exchange_problem_l2408_240841


namespace rice_bag_weight_qualification_l2408_240898

def is_qualified (weight : ℝ) : Prop :=
  9.9 ≤ weight ∧ weight ≤ 10.1

theorem rice_bag_weight_qualification :
  is_qualified 10 ∧
  ¬ is_qualified 9.2 ∧
  ¬ is_qualified 10.2 ∧
  ¬ is_qualified 9.8 :=
by sorry

end rice_bag_weight_qualification_l2408_240898


namespace prob_six_queen_is_4_663_l2408_240854

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of sixes in a standard deck -/
def NumSixes : ℕ := 4

/-- Number of queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a 6 as the first card and a Queen as the second card -/
def ProbSixQueen : ℚ := (NumSixes : ℚ) / StandardDeck * NumQueens / (StandardDeck - 1)

theorem prob_six_queen_is_4_663 : ProbSixQueen = 4 / 663 := by
  sorry

end prob_six_queen_is_4_663_l2408_240854


namespace complex_product_sum_l2408_240825

theorem complex_product_sum (a b : ℝ) : 
  (1 + Complex.I) * (2 + Complex.I) = Complex.mk a b → a + b = 4 := by
  sorry

end complex_product_sum_l2408_240825


namespace unique_solution_from_distinct_points_l2408_240847

/-- Given two distinct points on a line, the system of equations formed by these points always has a unique solution -/
theorem unique_solution_from_distinct_points 
  (k : ℝ) (a b₁ b₂ a₁ a₂ : ℝ) :
  (a ≠ 2) →  -- P₁ and P₂ are distinct
  (b₁ = k * a + 1) →  -- P₁ is on the line
  (b₂ = k * 2 + 1) →  -- P₂ is on the line
  ∃! (x y : ℝ), a₁ * x + b₁ * y = 1 ∧ a₂ * x + b₂ * y = 1 :=
by sorry

end unique_solution_from_distinct_points_l2408_240847


namespace initial_customers_l2408_240871

theorem initial_customers (initial leaving new final : ℕ) : 
  leaving = 8 → new = 4 → final = 9 → 
  initial - leaving + new = final → 
  initial = 13 := by sorry

end initial_customers_l2408_240871


namespace amoeba_population_after_10_days_l2408_240873

/-- The number of amoebas after n days, given an initial population of 2 -/
def amoeba_population (n : ℕ) : ℕ := 2 * 3^n

/-- Theorem stating that the amoeba population after 10 days is 118098 -/
theorem amoeba_population_after_10_days : amoeba_population 10 = 118098 := by
  sorry

end amoeba_population_after_10_days_l2408_240873


namespace min_value_of_F_l2408_240802

theorem min_value_of_F (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min : ℝ), min = 6/11 ∧ ∀ (F : ℝ), F = 2*x^2 + 3*y^2 + z^2 → F ≥ min :=
sorry

end min_value_of_F_l2408_240802


namespace cosine_sum_120_l2408_240839

theorem cosine_sum_120 (α : ℝ) : 
  Real.cos (α - 120 * π / 180) + Real.cos α + Real.cos (α + 120 * π / 180) = 0 := by
  sorry

end cosine_sum_120_l2408_240839


namespace triangle_perpendicular_segment_length_l2408_240820

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 5
  xz_length : Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 12

-- Define the perpendicular segment LM
def perpendicular_segment (X Y Z M : ℝ × ℝ) : Prop :=
  (M.1 - X.1) * (Y.1 - X.1) + (M.2 - X.2) * (Y.2 - X.2) = 0

-- Theorem statement
theorem triangle_perpendicular_segment_length 
  (X Y Z M : ℝ × ℝ) (h : Triangle X Y Z) (h_perp : perpendicular_segment X Y Z M) :
  Real.sqrt ((M.1 - Y.1)^2 + (M.2 - Y.2)^2) = (5 * Real.sqrt 119) / 12 := by
  sorry

end triangle_perpendicular_segment_length_l2408_240820


namespace samir_stairs_count_l2408_240801

theorem samir_stairs_count (s : ℕ) : 
  (s + (s / 2 + 18) = 495) → s = 318 := by
  sorry

end samir_stairs_count_l2408_240801


namespace min_remote_uses_l2408_240823

/-- Represents the state of lamps --/
def LampState := Fin 169 → Bool

/-- The remote control operation --/
def remote_control (s : LampState) (switches : Finset (Fin 169)) : LampState :=
  λ i => if i ∈ switches then !s i else s i

/-- All lamps are initially on --/
def initial_state : LampState := λ _ => true

/-- All lamps are off --/
def all_off (s : LampState) : Prop := ∀ i, s i = false

/-- The remote control changes exactly 19 switches --/
def valid_remote_use (switches : Finset (Fin 169)) : Prop :=
  switches.card = 19

theorem min_remote_uses :
  ∃ (sequence : List (Finset (Fin 169))),
    sequence.length = 9 ∧
    (∀ switches ∈ sequence, valid_remote_use switches) ∧
    all_off (sequence.foldl remote_control initial_state) ∧
    (∀ (shorter_sequence : List (Finset (Fin 169))),
      shorter_sequence.length < 9 →
      (∀ switches ∈ shorter_sequence, valid_remote_use switches) →
      ¬ all_off (shorter_sequence.foldl remote_control initial_state)) :=
sorry

end min_remote_uses_l2408_240823


namespace negative_power_division_l2408_240830

theorem negative_power_division : -3^7 / 3^2 = -3^5 := by sorry

end negative_power_division_l2408_240830


namespace a_range_theorem_l2408_240895

/-- Proposition p: (a-2)x^2 + 2(a-2)x - 4 < 0 for all x ∈ ℝ -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

/-- Proposition q: One root of x^2 + (a-1)x + 1 = 0 is in (0,1), and the other is in (1,2) -/
def prop_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ y^2 + (a - 1) * y + 1 = 0 ∧
    0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

/-- The range of values for a -/
def a_range (a : ℝ) : Prop :=
  (a > -2 ∧ a ≤ -3/2) ∨ (a ≥ -1 ∧ a ≤ 2)

theorem a_range_theorem (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) → a_range a := by
  sorry

end a_range_theorem_l2408_240895


namespace ball_count_proof_l2408_240826

theorem ball_count_proof (total : ℕ) (p_yellow p_blue p_red : ℚ) 
  (h_total : total = 80)
  (h_yellow : p_yellow = 1/4)
  (h_blue : p_blue = 7/20)
  (h_red : p_red = 2/5)
  (h_sum : p_yellow + p_blue + p_red = 1) :
  ∃ (yellow blue red : ℕ),
    yellow = 20 ∧ 
    blue = 28 ∧ 
    red = 32 ∧
    yellow + blue + red = total ∧
    (yellow : ℚ) / total = p_yellow ∧
    (blue : ℚ) / total = p_blue ∧
    (red : ℚ) / total = p_red :=
by
  sorry

end ball_count_proof_l2408_240826


namespace library_books_fraction_l2408_240833

theorem library_books_fraction (total : ℕ) (sold : ℕ) (h1 : total = 9900) (h2 : sold = 3300) :
  (total - sold : ℚ) / total = 2 / 3 := by
  sorry

end library_books_fraction_l2408_240833


namespace problem_solution_l2408_240869

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 := by
  sorry

end problem_solution_l2408_240869


namespace max_distance_trig_points_l2408_240859

theorem max_distance_trig_points (α β : ℝ) : 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (max_dist : ℝ), max_dist = 2 ∧ ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ max_dist :=
by sorry

end max_distance_trig_points_l2408_240859


namespace work_completion_time_l2408_240863

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 12) (hab : a_and_b = 3) : 
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 4 := by sorry

end work_completion_time_l2408_240863


namespace intersection_complement_equal_l2408_240852

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5*x ≥ 0}

theorem intersection_complement_equal : A ∩ (U \ B) = {2, 3} := by sorry

end intersection_complement_equal_l2408_240852


namespace cat_speed_l2408_240894

/-- Proves that a cat's speed is 90 km/h given specific conditions -/
theorem cat_speed (rat_speed : ℝ) (head_start : ℝ) (catch_time : ℝ) :
  rat_speed = 36 →
  head_start = 6 →
  catch_time = 4 →
  rat_speed * (head_start + catch_time) = 90 * catch_time :=
by
  sorry

#check cat_speed

end cat_speed_l2408_240894


namespace eulers_formula_l2408_240813

/-- A convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices - p.edges + p.faces = 2 := by
  sorry

end eulers_formula_l2408_240813


namespace initial_overs_played_l2408_240850

/-- Proves that the number of overs played initially is 15, given the target score,
    initial run rate, required run rate for remaining overs, and the number of remaining overs. -/
theorem initial_overs_played (target_score : ℝ) (initial_run_rate : ℝ) (required_run_rate : ℝ) (remaining_overs : ℝ) :
  target_score = 275 →
  initial_run_rate = 3.2 →
  required_run_rate = 6.485714285714286 →
  remaining_overs = 35 →
  ∃ (initial_overs : ℝ), initial_overs = 15 ∧
    target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by
  sorry


end initial_overs_played_l2408_240850


namespace min_additional_flights_for_40_percent_rate_l2408_240858

/-- Calculates the on-time rate given the number of on-time flights and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Represents the airport's flight departure scenario -/
structure AirportScenario where
  lateFlights : ℕ
  initialOnTimeFlights : ℕ
  additionalOnTimeFlights : ℕ

/-- Theorem: At least 1 additional on-time flight is needed for the on-time rate to exceed 40% -/
theorem min_additional_flights_for_40_percent_rate 
  (scenario : AirportScenario) 
  (h1 : scenario.lateFlights = 1)
  (h2 : scenario.initialOnTimeFlights = 3) :
  (∀ x : ℕ, x < 1 → 
    onTimeRate (scenario.initialOnTimeFlights + x) 
               (scenario.lateFlights + scenario.initialOnTimeFlights + x) ≤ 2/5) ∧
  (onTimeRate (scenario.initialOnTimeFlights + 1) 
              (scenario.lateFlights + scenario.initialOnTimeFlights + 1) > 2/5) :=
by sorry

end min_additional_flights_for_40_percent_rate_l2408_240858


namespace inequality_equivalence_l2408_240883

theorem inequality_equivalence (x : ℝ) : 
  (1/3 : ℝ)^(x^2 - 8) > 3^(-2*x) ↔ -2 < x ∧ x < 4 := by sorry

end inequality_equivalence_l2408_240883


namespace solve_exponential_equation_l2408_240827

theorem solve_exponential_equation :
  ∃ x : ℝ, 4^x = Real.sqrt 64 ∧ x = 3/2 := by
  sorry

end solve_exponential_equation_l2408_240827


namespace rope_length_proof_l2408_240824

/-- The length of the rope in meters -/
def rope_length : ℝ := 1.15

/-- The fraction of the rope that was used -/
def used_fraction : ℝ := 0.4

/-- The remaining length of the rope in meters -/
def remaining_length : ℝ := 0.69

theorem rope_length_proof : 
  rope_length * (1 - used_fraction) = remaining_length :=
by sorry

end rope_length_proof_l2408_240824


namespace chocolate_distribution_l2408_240815

/-- The number of students -/
def num_students : ℕ := 211

/-- The number of possible combinations of chocolate choices -/
def num_combinations : ℕ := 35

/-- The minimum number of students in the largest group -/
def min_largest_group : ℕ := 7

theorem chocolate_distribution :
  ∃ (group : Finset (Fin num_students)),
    group.card ≥ min_largest_group ∧
    ∀ (s₁ s₂ : Fin num_students),
      s₁ ∈ group → s₂ ∈ group →
      ∃ (c : Fin num_combinations), true :=
by
  sorry

end chocolate_distribution_l2408_240815


namespace trigonometric_identities_l2408_240811

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos α - Real.sin α = -Real.sqrt 5 / 5) : 
  (Real.sin α * Real.cos α = 2 / 5) ∧ 
  (Real.sin α + Real.cos α = 3 * Real.sqrt 5 / 5) ∧ 
  ((2 * Real.sin α * Real.cos α - Real.cos α + 1) / (1 - Real.tan α) = (-9 + Real.sqrt 5) / 5) := by
  sorry

end trigonometric_identities_l2408_240811


namespace thirtieth_term_of_sequence_l2408_240819

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 148 := by
  sorry

end thirtieth_term_of_sequence_l2408_240819


namespace imaginary_part_of_complex_l2408_240845

theorem imaginary_part_of_complex (z : ℂ) :
  z = 2 + (1 / (3 * I)) → z.im = -1/3 := by sorry

end imaginary_part_of_complex_l2408_240845


namespace imaginary_part_of_i_minus_2_squared_l2408_240887

theorem imaginary_part_of_i_minus_2_squared (i : ℂ) : 
  (i * i = -1) → Complex.im ((i - 2) ^ 2) = -4 := by sorry

end imaginary_part_of_i_minus_2_squared_l2408_240887


namespace nested_even_function_is_even_l2408_240864

-- Define an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem nested_even_function_is_even
  (g : ℝ → ℝ)
  (h : even_function g) :
  even_function (fun x ↦ g (g (g (g x)))) :=
by sorry

end nested_even_function_is_even_l2408_240864


namespace stock_yield_calculation_l2408_240890

theorem stock_yield_calculation (a_price b_price b_yield : ℝ) 
  (h1 : a_price = 96)
  (h2 : b_price = 115.2)
  (h3 : b_yield = 0.12)
  (h4 : a_price * b_yield = b_price * (a_yield : ℝ)) :
  a_yield = 0.10 :=
by
  sorry

end stock_yield_calculation_l2408_240890


namespace angle_PQR_is_60_degrees_l2408_240888

-- Define the points
def P : ℝ × ℝ × ℝ := (-3, 1, 7)
def Q : ℝ × ℝ × ℝ := (-4, 0, 3)
def R : ℝ × ℝ × ℝ := (-5, 0, 4)

-- Define the angle PQR in radians
def angle_PQR : ℝ := sorry

theorem angle_PQR_is_60_degrees :
  angle_PQR = π / 3 := by sorry

end angle_PQR_is_60_degrees_l2408_240888
