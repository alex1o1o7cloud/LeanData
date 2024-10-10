import Mathlib

namespace negation_quadratic_roots_l3515_351581

theorem negation_quadratic_roots (a b c : ℝ) :
  (¬(b^2 - 4*a*c < 0 → ∀ x, a*x^2 + b*x + c ≠ 0)) ↔
  (b^2 - 4*a*c ≥ 0 → ∃ x, a*x^2 + b*x + c = 0) :=
by sorry

end negation_quadratic_roots_l3515_351581


namespace equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l3515_351528

/-- The length of one side of an equilateral triangle whose perimeter equals 
    the perimeter of a 125 cm × 115 cm rectangle is 160 cm. -/
theorem equilateral_triangle_side_length : ℝ → Prop :=
  λ side_length : ℝ =>
    let rectangle_width : ℝ := 125
    let rectangle_length : ℝ := 115
    let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
    let triangle_perimeter : ℝ := 3 * side_length
    (triangle_perimeter = rectangle_perimeter) → (side_length = 160)

theorem equilateral_triangle_side_length_proof : 
  equilateral_triangle_side_length 160 := by sorry

end equilateral_triangle_side_length_equilateral_triangle_side_length_proof_l3515_351528


namespace f_2007_equals_zero_l3515_351587

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2007_equals_zero
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_fg : ∀ x, g x = f (x - 1)) :
  f 2007 = 0 := by
  sorry

end f_2007_equals_zero_l3515_351587


namespace sum_of_i_powers_l3515_351548

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every four powers
axiom i_period (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem sum_of_i_powers : i^23 + i^221 + i^20 = 1 := by
  sorry

end sum_of_i_powers_l3515_351548


namespace a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l3515_351577

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∃ a : ℝ, a > 1 ∧ a^2 > 1) ∧ 
  (∃ a : ℝ, a^2 > 1 ∧ ¬(a > 1)) := by
  sorry

end a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l3515_351577


namespace group_count_divisible_by_27_l3515_351593

/-- Represents the number of groups of each size -/
structure GroupCounts where
  size2 : ℕ
  size5 : ℕ
  size11 : ℕ

/-- The mean size of a group is 4 -/
def mean_size_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 + 5 * g.size5 + 11 * g.size11) / (g.size2 + g.size5 + g.size11) = 4

/-- The mean of answers when each person is asked how many others are in their group is 6 -/
def mean_answer_condition (g : GroupCounts) : Prop :=
  (2 * g.size2 * 1 + 5 * g.size5 * 4 + 11 * g.size11 * 10) / (2 * g.size2 + 5 * g.size5 + 11 * g.size11) = 6

/-- The main theorem to prove -/
theorem group_count_divisible_by_27 (g : GroupCounts) 
  (h1 : mean_size_condition g) (h2 : mean_answer_condition g) : 
  ∃ k : ℕ, g.size2 + g.size5 + g.size11 = 27 * k := by
  sorry

end group_count_divisible_by_27_l3515_351593


namespace condition_necessary_not_sufficient_l3515_351568

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧
  (∃ x : ℝ, (2*x - 1)*x = 0 ∧ x ≠ 0) :=
by sorry

end condition_necessary_not_sufficient_l3515_351568


namespace det_transformation_l3515_351553

/-- Given a 2x2 matrix with determinant 7, prove that the determinant of a related matrix is also 7 -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 7) :
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end det_transformation_l3515_351553


namespace waitress_income_fraction_from_tips_l3515_351550

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Theorem: Given a waitress's income where tips are 9/4 of salary,
    the fraction of income from tips is 9/13 -/
theorem waitress_income_fraction_from_tips 
  (income : WaitressIncome) 
  (h : income.tips = (9 : ℚ) / 4 * income.salary) : 
  income.tips / (income.salary + income.tips) = (9 : ℚ) / 13 := by
  sorry


end waitress_income_fraction_from_tips_l3515_351550


namespace max_milk_bags_theorem_l3515_351552

/-- Calculates the maximum number of bags of milk that can be purchased given the cost per bag, 
    the promotion rule, and the total available money. -/
def max_milk_bags (cost_per_bag : ℚ) (promotion_rule : ℕ → ℕ) (total_money : ℚ) : ℕ :=
  sorry

/-- The promotion rule: for every 2 bags purchased, 1 additional bag is given for free -/
def buy_two_get_one_free (n : ℕ) : ℕ :=
  n + n / 2

theorem max_milk_bags_theorem :
  max_milk_bags 2.5 buy_two_get_one_free 30 = 18 := by
  sorry

end max_milk_bags_theorem_l3515_351552


namespace line_points_relationship_l3515_351595

theorem line_points_relationship (m a b : ℝ) :
  ((-2 : ℝ), a) ∈ {(x, y) | y = -2*x + m} →
  ((2 : ℝ), b) ∈ {(x, y) | y = -2*x + m} →
  a > b := by
  sorry

end line_points_relationship_l3515_351595


namespace prob_two_females_is_two_fifths_l3515_351540

/-- Represents the survey data for students' preferences on breeding small animal A -/
structure SurveyData where
  male_like : ℕ
  male_dislike : ℕ
  female_like : ℕ
  female_dislike : ℕ

/-- Calculates the probability of selecting two females from a stratified sample -/
def prob_two_females (data : SurveyData) : ℚ :=
  let total_like := data.male_like + data.female_like
  let female_ratio := data.female_like / total_like
  let num_females_selected := 6 * female_ratio
  (num_females_selected * (num_females_selected - 1)) / (6 * 5)

/-- The main theorem to be proved -/
theorem prob_two_females_is_two_fifths (data : SurveyData) 
  (h1 : data.male_like = 20)
  (h2 : data.male_dislike = 30)
  (h3 : data.female_like = 40)
  (h4 : data.female_dislike = 10) :
  prob_two_females data = 2/5 := by
  sorry

end prob_two_females_is_two_fifths_l3515_351540


namespace odd_function_property_l3515_351560

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_recursive : ∀ x, f (x + 4) = f x + 3 * f 2)
  (h_f_1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
sorry

end odd_function_property_l3515_351560


namespace triangle_cosine_sum_l3515_351594

theorem triangle_cosine_sum (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = 3 * B) (h3 : A = 9 * C) :
  Real.cos A * Real.cos B + Real.cos B * Real.cos C + Real.cos C * Real.cos A = -1/4 := by
  sorry

end triangle_cosine_sum_l3515_351594


namespace pure_imaginary_m_equals_four_l3515_351509

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number formed by the given expression. -/
def ComplexExpression (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m - 4, m^2 - 5*m - 6⟩

theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, IsPureImaginary (ComplexExpression m) → m = 4 :=
sorry

end pure_imaginary_m_equals_four_l3515_351509


namespace six_meter_logs_more_efficient_l3515_351517

/-- Represents the number of pieces obtained from a log of given length -/
def pieces_from_log (log_length : ℕ) : ℕ := log_length

/-- Represents the number of cuts needed to divide a log into 1-meter pieces -/
def cuts_for_log (log_length : ℕ) : ℕ := log_length - 1

/-- Represents the efficiency of cutting a log, measured as pieces per cut -/
def cutting_efficiency (log_length : ℕ) : ℚ :=
  (pieces_from_log log_length : ℚ) / (cuts_for_log log_length : ℚ)

/-- Theorem stating that 6-meter logs are more efficient to cut than 8-meter logs -/
theorem six_meter_logs_more_efficient :
  cutting_efficiency 6 > cutting_efficiency 8 := by
  sorry

end six_meter_logs_more_efficient_l3515_351517


namespace f_range_l3515_351551

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+2) - 3

theorem f_range : Set.range f = Set.Ici (-7) := by sorry

end f_range_l3515_351551


namespace number_exceeding_percentage_l3515_351557

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end number_exceeding_percentage_l3515_351557


namespace f_properties_l3515_351501

noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then x + 2
  else if x = 0 then 0
  else x - 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f x < 2 ↔ x < 4) := by sorry

end f_properties_l3515_351501


namespace jerry_speed_is_40_l3515_351546

-- Define the given conditions
def jerry_time : ℚ := 1/2  -- 30 minutes in hours
def beth_time : ℚ := 5/6   -- 50 minutes in hours
def beth_speed : ℚ := 30   -- miles per hour
def route_difference : ℚ := 5  -- miles

-- Theorem to prove
theorem jerry_speed_is_40 :
  let beth_distance : ℚ := beth_speed * beth_time
  let jerry_distance : ℚ := beth_distance - route_difference
  jerry_distance / jerry_time = 40 := by
  sorry


end jerry_speed_is_40_l3515_351546


namespace base_conversion_problem_l3515_351562

theorem base_conversion_problem (a b : ℕ) : 
  (a < 10 ∧ b < 10) → -- Ensuring a and b are single digits
  (6 * 7^2 + 5 * 7 + 6 = 300 + 10 * a + b) → -- 656₇ = 3ab₁₀
  (a * b) / 15 = 1 := by
sorry

end base_conversion_problem_l3515_351562


namespace set_B_determination_l3515_351531

open Set

theorem set_B_determination (A B : Set ℕ) : 
  A = {1, 2} → 
  A ∩ B = {1} → 
  A ∪ B = {0, 1, 2} → 
  B = {0, 1} := by
  sorry

end set_B_determination_l3515_351531


namespace second_player_wins_l3515_351584

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the state of the game board -/
structure GameBoard where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a move in the game -/
structure Move where
  player : Player
  position : Fin 3
  value : ℝ

/-- Checks if a move is valid -/
def isValidMove (board : GameBoard) (move : Move) : Prop :=
  match move.position with
  | 0 => move.value ≠ 0  -- a ≠ 0
  | _ => True

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move.position with
  | 0 => { board with a := move.value }
  | 1 => { board with b := move.value }
  | 2 => { board with c := move.value }

/-- Checks if the quadratic equation has real roots -/
def hasRealRoots (board : GameBoard) : Prop :=
  board.b * board.b - 4 * board.a * board.c ≥ 0

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (firstMove : Move),
    isValidMove { a := 0, b := 0, c := 0 } firstMove →
    ∃ (secondMove : Move),
      isValidMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove ∧
      hasRealRoots (applyMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove) :=
sorry


end second_player_wins_l3515_351584


namespace table_length_is_77_l3515_351556

/-- The length of the rectangular table -/
def table_length : ℕ := 77

/-- The width of the rectangular table -/
def table_width : ℕ := 80

/-- The height of each sheet of paper -/
def sheet_height : ℕ := 5

/-- The width of each sheet of paper -/
def sheet_width : ℕ := 8

/-- The horizontal and vertical increment for each subsequent sheet -/
def increment : ℕ := 1

theorem table_length_is_77 :
  ∃ (n : ℕ), 
    table_length = sheet_height + n * increment ∧
    table_width = sheet_width + n * increment ∧
    table_width - table_length = sheet_width - sheet_height := by
  sorry

end table_length_is_77_l3515_351556


namespace leila_payment_l3515_351570

/-- The total cost Leila should pay Ali for the cakes -/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
               (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Theorem stating that Leila should pay Ali $168 for the cakes -/
theorem leila_payment : total_cost 3 12 6 22 = 168 := by
  sorry

end leila_payment_l3515_351570


namespace emily_meal_combinations_l3515_351516

/-- The number of protein options available --/
def num_proteins : ℕ := 4

/-- The number of side options available --/
def num_sides : ℕ := 5

/-- The number of dessert options available --/
def num_desserts : ℕ := 5

/-- The number of sides Emily must choose --/
def sides_to_choose : ℕ := 3

/-- The total number of different meal combinations Emily can choose --/
def total_meals : ℕ := num_proteins * (num_sides.choose sides_to_choose) * num_desserts

theorem emily_meal_combinations :
  total_meals = 200 :=
sorry

end emily_meal_combinations_l3515_351516


namespace min_value_of_fraction_l3515_351532

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y)^2 ≥ (1 : ℝ) / 2 ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a^2 + b^2) / (a + b)^2 = (1 : ℝ) / 2 := by
  sorry

end min_value_of_fraction_l3515_351532


namespace exponent_division_l3515_351563

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by sorry

end exponent_division_l3515_351563


namespace inverse_function_point_and_sum_l3515_351535

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 6) →  -- This condition is derived from (2,3) being on y = f(x)/2
  (f_inv 6 = 2) →  -- This is the definition of the inverse function
  (∃ (x y : ℝ), x = 6 ∧ y = 1 ∧ y = (f_inv x) / 2) ∧  -- Point (6,1) is on y = f^(-1)(x)/2
  (6 + 1 = 7)  -- Sum of coordinates
  := by sorry

end inverse_function_point_and_sum_l3515_351535


namespace ellipse_focal_distance_difference_l3515_351523

/-- Given an ellipse with semi-major axis a and semi-minor axis b = √96,
    and a point P on the ellipse such that |PF₁| : |PF₂| : |OF₂| = 8 : 6 : 5,
    prove that |PF₁| - |PF₂| = 4 -/
theorem ellipse_focal_distance_difference 
  (a : ℝ) 
  (h_a : a > 4 * Real.sqrt 6) 
  (P : ℝ × ℝ) 
  (h_P : (P.1 / a)^2 + P.2^2 / 96 = 1) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_foci : ∃ (k : ℝ), k > 0 ∧ dist P F₁ = 8*k ∧ dist P F₂ = 6*k ∧ dist (0, 0) F₂ = 5*k) :
  dist P F₁ - dist P F₂ = 4 := by
  sorry


end ellipse_focal_distance_difference_l3515_351523


namespace bolt_defect_probability_l3515_351504

theorem bolt_defect_probability :
  let machine1_production : ℝ := 0.30
  let machine2_production : ℝ := 0.25
  let machine3_production : ℝ := 0.45
  let machine1_defect_rate : ℝ := 0.02
  let machine2_defect_rate : ℝ := 0.01
  let machine3_defect_rate : ℝ := 0.03
  machine1_production + machine2_production + machine3_production = 1 →
  machine1_production * machine1_defect_rate +
  machine2_production * machine2_defect_rate +
  machine3_production * machine3_defect_rate = 0.022 := by
sorry

end bolt_defect_probability_l3515_351504


namespace charity_event_result_l3515_351576

/-- Represents the number of books of each type brought by a participant -/
structure BookContribution where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents the total number of books collected -/
structure TotalBooks where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents how books are distributed on shelves -/
structure ShelfDistribution where
  first_shelf : ℕ
  second_shelf : ℕ

def charity_event_books (total : TotalBooks) (shelf : ShelfDistribution) : Prop :=
  -- Each participant brings either 1 encyclopedia, 3 fiction books, or 2 reference books
  ∃ (participants : ℕ) (encyc_part fict_part ref_part : ℕ),
    participants = encyc_part + fict_part + ref_part ∧
    total.encyclopedias = encyc_part * 1 ∧
    total.fiction = fict_part * 3 ∧
    total.reference = ref_part * 2 ∧
    -- 150 encyclopedias were collected
    total.encyclopedias = 150 ∧
    -- Two bookshelves were filled with an equal number of books
    shelf.first_shelf = shelf.second_shelf ∧
    -- The first shelf contained 1/5 of all reference books, 1/7 of all fiction books, and all encyclopedias
    shelf.first_shelf = total.encyclopedias + total.reference / 5 + total.fiction / 7 ∧
    -- Total books on both shelves
    shelf.first_shelf + shelf.second_shelf = total.encyclopedias + total.fiction + total.reference

theorem charity_event_result :
  ∀ (total : TotalBooks) (shelf : ShelfDistribution),
    charity_event_books total shelf →
    ∃ (participants : ℕ),
      participants = 416 ∧
      total.encyclopedias + total.fiction + total.reference = 738 :=
sorry

end charity_event_result_l3515_351576


namespace exterior_angle_theorem_l3515_351538

/-- The measure of the exterior angle BAC in a coplanar arrangement 
    where a square and a regular nonagon share a common side AD -/
def exterior_angle_BAC : ℝ := 130

/-- The measure of the interior angle of a regular nonagon -/
def nonagon_interior_angle : ℝ := 140

/-- The measure of the interior angle of a square -/
def square_interior_angle : ℝ := 90

theorem exterior_angle_theorem :
  exterior_angle_BAC = 360 - nonagon_interior_angle - square_interior_angle :=
by sorry

end exterior_angle_theorem_l3515_351538


namespace no_square_free_arithmetic_sequence_l3515_351561

theorem no_square_free_arithmetic_sequence :
  ∀ (a d : ℕ+), d ≠ 1 →
  ∃ (n : ℕ), ∃ (k : ℕ), k > 1 ∧ (k * k ∣ (a + n * d)) :=
by sorry

end no_square_free_arithmetic_sequence_l3515_351561


namespace number_transformation_l3515_351582

theorem number_transformation (x : ℝ) : 2 * ((2 * (x + 1)) - 1) = 2 * x + 2 := by
  sorry

#check number_transformation

end number_transformation_l3515_351582


namespace largest_expression_l3515_351520

theorem largest_expression : 
  let a := 3 + 1 + 0 + 5
  let b := 3 * 1 + 0 + 5
  let c := 3 + 1 * 0 + 5
  let d := 3 * 1 + 0 * 5
  let e := 3 * 1 + 0 * 5 * 3
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end largest_expression_l3515_351520


namespace gold_ratio_l3515_351558

theorem gold_ratio (total_gold : ℕ) (greg_gold : ℕ) (h1 : total_gold = 100) (h2 : greg_gold = 20) :
  greg_gold / (total_gold - greg_gold) = 1 / 4 := by
  sorry

end gold_ratio_l3515_351558


namespace line_slope_intercept_sum_l3515_351539

/-- Given a line with slope 4 passing through (5, -2), prove that m + b = -18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 ∧ 
  -2 = 4 * 5 + b → 
  m + b = -18 := by
sorry

end line_slope_intercept_sum_l3515_351539


namespace area_of_inscribed_rectangle_l3515_351533

/-- The area of a rectangle inscribed in a square, given other inscribed shapes -/
theorem area_of_inscribed_rectangle (s : ℝ) (r1_length r1_width : ℝ) (sq_side : ℝ) :
  s = 4 →
  r1_length = 2 →
  r1_width = 4 →
  sq_side = 1 →
  ∃ (r2_length r2_width : ℝ),
    r2_length * r2_width = s^2 - (r1_length * r1_width + sq_side^2) :=
by sorry

end area_of_inscribed_rectangle_l3515_351533


namespace min_value_3a_plus_1_l3515_351575

theorem min_value_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 9 * a + 6 = 2) :
  ∃ (x : ℝ), (3 * a + 1 ≥ x) ∧ (∀ y, 3 * a + 1 ≥ y → x ≥ y) ∧ x = -2 := by
  sorry

end min_value_3a_plus_1_l3515_351575


namespace sequence_ratio_theorem_l3515_351542

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Property of the sequence: a_{n+1} - 2a_n = 0 for all n -/
def HasConstantRatio (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) - 2 * a n = 0

/-- Property of the sequence: a_n ≠ 0 for all n -/
def IsNonZero (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≠ 0

/-- The main theorem -/
theorem sequence_ratio_theorem (a : Sequence) 
  (h1 : HasConstantRatio a) (h2 : IsNonZero a) : 
  (2 * a 1 + a 2) / (a 3 + a 5) = 1 / 5 := by
  sorry

end sequence_ratio_theorem_l3515_351542


namespace fraction_value_l3515_351599

theorem fraction_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, x + (2 * q - p) / (2 * q + p) = 2 ∧ x = 11 / 7 := by
  sorry

end fraction_value_l3515_351599


namespace top_is_multiple_of_four_l3515_351545

/-- Represents a number pyramid with 4 rows -/
structure NumberPyramid where
  bottom_row : Fin 4 → ℤ
  second_row : Fin 3 → ℤ
  third_row : Fin 2 → ℤ
  top : ℤ

/-- Defines a valid number pyramid where each cell above the bottom row
    is the sum of the two cells below it, and the second row contains equal integers -/
def is_valid_pyramid (p : NumberPyramid) : Prop :=
  (∃ n : ℤ, ∀ i : Fin 3, p.second_row i = n) ∧
  (∀ i : Fin 2, p.third_row i = p.second_row i + p.second_row (i + 1)) ∧
  p.top = p.third_row 0 + p.third_row 1

theorem top_is_multiple_of_four (p : NumberPyramid) (h : is_valid_pyramid p) :
  ∃ k : ℤ, p.top = 4 * k :=
sorry

end top_is_multiple_of_four_l3515_351545


namespace sum_rounded_to_hundredth_l3515_351572

-- Define the numbers
def a : Float := 45.378
def b : Float := 13.897
def c : Float := 29.4567

-- Define the sum
def sum : Float := a + b + c

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem sum_rounded_to_hundredth :
  round_to_hundredth sum = 88.74 := by sorry

end sum_rounded_to_hundredth_l3515_351572


namespace hamburgers_count_l3515_351502

/-- The number of hamburgers initially made -/
def initial_hamburgers : ℝ := 9.0

/-- The number of additional hamburgers made -/
def additional_hamburgers : ℝ := 3.0

/-- The total number of hamburgers made -/
def total_hamburgers : ℝ := initial_hamburgers + additional_hamburgers

theorem hamburgers_count : total_hamburgers = 12.0 := by
  sorry

end hamburgers_count_l3515_351502


namespace initial_weasels_count_l3515_351518

/-- Represents the number of weasels caught by one fox in one week -/
def weasels_per_fox_per_week : ℕ := 4

/-- Represents the number of rabbits caught by one fox in one week -/
def rabbits_per_fox_per_week : ℕ := 2

/-- Represents the number of foxes -/
def num_foxes : ℕ := 3

/-- Represents the number of weeks the foxes hunt -/
def num_weeks : ℕ := 3

/-- Represents the initial number of rabbits -/
def initial_rabbits : ℕ := 50

/-- Represents the number of rabbits and weasels left after hunting -/
def remaining_animals : ℕ := 96

/-- Theorem stating that the initial number of weasels is 100 -/
theorem initial_weasels_count : 
  ∃ (initial_weasels : ℕ), 
    initial_weasels = 100 ∧
    initial_weasels + initial_rabbits = 
      remaining_animals + 
      (weasels_per_fox_per_week * num_foxes * num_weeks) + 
      (rabbits_per_fox_per_week * num_foxes * num_weeks) := by
  sorry

end initial_weasels_count_l3515_351518


namespace rectangle_area_excluding_hole_l3515_351519

variable (x : ℝ)

def large_rectangle_length : ℝ := 2 * x + 4
def large_rectangle_width : ℝ := x + 7
def hole_length : ℝ := x + 2
def hole_width : ℝ := 3 * x - 5

theorem rectangle_area_excluding_hole (h : x > 5/3) :
  large_rectangle_length x * large_rectangle_width x - hole_length x * hole_width x = -x^2 + 17*x + 38 := by
  sorry

end rectangle_area_excluding_hole_l3515_351519


namespace arithmetic_sequence_common_difference_l3515_351521

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9)
  (h3 : ∀ n, a (n + 1) = a n + d)
  (h4 : (a 4) ^ 2 = (a 1) * (a 8)) :
  d = 1 := by
sorry

end arithmetic_sequence_common_difference_l3515_351521


namespace special_hyperbola_equation_l3515_351573

/-- A hyperbola with center at the origin, foci on the x-axis, and specific properties. -/
structure SpecialHyperbola where
  -- The equation of the hyperbola in the form x²/a² - y²/b² = 1
  a : ℝ
  b : ℝ
  -- The right focus is at (c, 0) where c² = a² + b²
  c : ℝ
  h_c : c^2 = a^2 + b^2
  -- A line through the right focus with slope √(3/5)
  line_slope : ℝ
  h_slope : line_slope^2 = 3/5
  -- The line intersects the hyperbola at P and Q
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_hyperbola : (P.1/a)^2 - (P.2/b)^2 = 1
  h_Q_on_hyperbola : (Q.1/a)^2 - (Q.2/b)^2 = 1
  h_P_on_line : P.2 = line_slope * (P.1 - c)
  h_Q_on_line : Q.2 = line_slope * (Q.1 - c)
  -- PO ⊥ OQ
  h_perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0
  -- |PQ| = 4
  h_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 16

/-- The theorem stating that the special hyperbola has the equation x² - y²/3 = 1 -/
theorem special_hyperbola_equation (h : SpecialHyperbola) : h.a^2 = 1 ∧ h.b^2 = 3 := by
  sorry

#check special_hyperbola_equation

end special_hyperbola_equation_l3515_351573


namespace sum_specific_arithmetic_progression_l3515_351515

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of terms in an arithmetic progression -/
def num_terms_arithmetic_progression (a : ℤ) (d : ℤ) (l : ℤ) : ℕ :=
  ((l - a) / d).toNat + 1

theorem sum_specific_arithmetic_progression :
  let a : ℤ := -45  -- First term
  let d : ℤ := 2    -- Common difference
  let l : ℤ := 23   -- Last term
  let n : ℕ := num_terms_arithmetic_progression a d l
  sum_arithmetic_progression a d n = -385 := by
sorry

end sum_specific_arithmetic_progression_l3515_351515


namespace security_check_comprehensive_l3515_351569

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario for which a survey method is chosen -/
structure Scenario where
  requiresAllChecked : Bool
  noExceptions : Bool
  populationAccessible : Bool
  populationFinite : Bool

/-- Determines the correct survey method for a given scenario -/
def correctSurveyMethod (s : Scenario) : SurveyMethod :=
  if s.requiresAllChecked && s.noExceptions && s.populationAccessible && s.populationFinite then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sample

/-- The scenario of security checks before boarding a plane -/
def securityCheckScenario : Scenario :=
  { requiresAllChecked := true
    noExceptions := true
    populationAccessible := true
    populationFinite := true }

theorem security_check_comprehensive :
  correctSurveyMethod securityCheckScenario = SurveyMethod.Comprehensive := by
  sorry


end security_check_comprehensive_l3515_351569


namespace sqrt_equation_solution_l3515_351505

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 6 ∧ Real.sqrt (4 + 9 + x^2) = 7 :=
by sorry

end sqrt_equation_solution_l3515_351505


namespace expression_bounds_l3515_351534

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end expression_bounds_l3515_351534


namespace parallel_vectors_second_component_l3515_351583

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors_second_component (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3) :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + b)) → b.2 = -3 := by
  sorry

end parallel_vectors_second_component_l3515_351583


namespace solution_y_l3515_351526

-- Define the function G
def G (a b c d : ℕ) : ℕ := a^b + c * d

-- Define the theorem
theorem solution_y : ∃ y : ℕ, G 3 y 6 15 = 300 ∧ 
  ∀ z : ℕ, G 3 z 6 15 = 300 → y = z :=
by
  sorry

end solution_y_l3515_351526


namespace neither_necessary_nor_sufficient_l3515_351541

theorem neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
  ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) :=
by sorry

end neither_necessary_nor_sufficient_l3515_351541


namespace f_of_g_5_l3515_351571

-- Define the functions f and g
def g (x : ℝ) : ℝ := 4 * x + 10
def f (x : ℝ) : ℝ := 6 * x - 12

-- State the theorem
theorem f_of_g_5 : f (g 5) = 168 := by
  sorry

end f_of_g_5_l3515_351571


namespace sin_cos_sum_equals_half_l3515_351554

theorem sin_cos_sum_equals_half : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (167 * π / 180) = 1 / 2 := by
sorry

end sin_cos_sum_equals_half_l3515_351554


namespace triangle_side_range_l3515_351555

theorem triangle_side_range (p : ℝ) : 
  (∃ r s : ℝ, r * s = 4 * 26 ∧ 
              r^2 + p*r + 1 = 0 ∧ 
              s^2 + p*s + 1 = 0 ∧ 
              r > 0 ∧ s > 0 ∧
              r + s > 2 ∧ r + 2 > s ∧ s + 2 > r) →
  -2 * Real.sqrt 2 < p ∧ p < -2 :=
by sorry

end triangle_side_range_l3515_351555


namespace f_equal_implies_sum_negative_l3515_351578

noncomputable def f (x : ℝ) : ℝ := ((1 - x) / (1 + x^2)) * Real.exp x

theorem f_equal_implies_sum_negative (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ < 0 := by
  sorry

end f_equal_implies_sum_negative_l3515_351578


namespace always_possible_scatter_plot_l3515_351525

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- We don't need to specify the internal structure of the data
  -- as the problem doesn't provide details about it

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- We don't need to specify the internal structure of the scatter plot
  -- as the problem doesn't provide details about it

/-- States that it's always possible to create a scatter plot from two-variable data -/
theorem always_possible_scatter_plot (data : TwoVariableData) : 
  ∃ (plot : ScatterPlot), true :=
sorry

end always_possible_scatter_plot_l3515_351525


namespace seed_germination_problem_l3515_351512

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.25 * x + 0.35 * 200) / (x + 200) = 0.28999999999999996 → 
  x = 300 := by
  sorry

end seed_germination_problem_l3515_351512


namespace second_number_value_l3515_351591

theorem second_number_value : 
  ∀ (a b c d : ℝ),
  a + b + c + d = 280 →
  a = 2 * b →
  c = (1/3) * a →
  d = b + c →
  b = 52.5 := by
sorry

end second_number_value_l3515_351591


namespace bridge_length_calculation_l3515_351567

/-- Calculate the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  ∃ (bridge_length : ℝ), bridge_length = 160 ∧
    bridge_length = train_speed_kmh * 1000 / 3600 * passing_time - train_length :=
by sorry

end bridge_length_calculation_l3515_351567


namespace fourth_student_score_l3515_351536

theorem fourth_student_score (s1 s2 s3 s4 : ℕ) : 
  s1 = 70 → s2 = 80 → s3 = 90 → (s1 + s2 + s3 + s4) / 4 = 70 → s4 = 40 := by
  sorry

end fourth_student_score_l3515_351536


namespace unique_m_value_l3515_351511

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + m - 1 = 0}

theorem unique_m_value : ∃! m : ℝ, A ∪ B m = A := by sorry

end unique_m_value_l3515_351511


namespace rams_and_ravis_selection_probability_l3515_351598

theorem rams_and_ravis_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 2/7)
  (h2 : p_both = 0.05714285714285714)
  (h3 : p_both = p_ram * (p_ravi : ℝ)) : 
  p_ravi = 1/5 := by
sorry

end rams_and_ravis_selection_probability_l3515_351598


namespace inverse_75_mod_76_l3515_351597

theorem inverse_75_mod_76 : ∃ x : ℕ, x < 76 ∧ (75 * x) % 76 = 1 :=
by
  use 75
  sorry

end inverse_75_mod_76_l3515_351597


namespace intersection_point_of_lines_l3515_351574

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) ↔ x = 25/11 ∧ y = 48/11 := by
  sorry

end intersection_point_of_lines_l3515_351574


namespace consecutive_odd_integers_multiplier_l3515_351522

theorem consecutive_odd_integers_multiplier (x : ℤ) (m : ℚ) : 
  x + 4 = 15 →  -- Third integer is 15
  (∀ k : ℤ, x + 2*k ∈ {n : ℤ | n % 2 = 1}) →  -- All three are odd integers
  x * m = 2 * (x + 4) + 3 →  -- First integer times m equals 3 more than twice the third
  m = 3 := by
sorry

end consecutive_odd_integers_multiplier_l3515_351522


namespace quadratic_inequality_l3515_351585

theorem quadratic_inequality (x : ℝ) : x^2 - 48*x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 := by
  sorry

end quadratic_inequality_l3515_351585


namespace intersection_of_A_and_B_l3515_351529

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - 2*x - 8 ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | x ≥ 4} := by sorry

end intersection_of_A_and_B_l3515_351529


namespace min_area_triangle_m_sum_l3515_351589

/-- The sum of m values for minimum area triangle -/
theorem min_area_triangle_m_sum : 
  ∀ (m : ℤ), 
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (14, 13)
  let C : ℝ × ℝ := (6, m)
  let triangle_area (m : ℤ) : ℝ := sorry -- Function to calculate triangle area
  let min_area : ℝ := sorry -- Minimum area of the triangle
  (∃ (m₁ m₂ : ℤ), 
    m₁ ≠ m₂ ∧ 
    triangle_area m₁ = min_area ∧ 
    triangle_area m₂ = min_area ∧ 
    m₁ + m₂ = 16) := by sorry


end min_area_triangle_m_sum_l3515_351589


namespace even_times_odd_is_even_l3515_351503

/-- An integer is even if it's divisible by 2 -/
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

/-- An integer is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- The product of an even integer and an odd integer is always even -/
theorem even_times_odd_is_even (a b : ℤ) (ha : IsEven a) (hb : IsOdd b) : IsEven (a * b) := by
  sorry


end even_times_odd_is_even_l3515_351503


namespace negation_of_existence_equals_forall_not_equal_l3515_351500

theorem negation_of_existence_equals_forall_not_equal (x : ℝ) :
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, x > 0 → Real.log x ≠ x - 1 :=
by sorry

end negation_of_existence_equals_forall_not_equal_l3515_351500


namespace star_four_five_l3515_351506

def star (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

theorem star_four_five : star 4 5 = -84 := by
  sorry

end star_four_five_l3515_351506


namespace park_diameter_l3515_351596

/-- Given a circular park with a fountain, garden, and walking path, 
    calculate the diameter of the outer boundary of the walking path. -/
theorem park_diameter (fountain_diameter garden_width path_width : ℝ) 
    (h1 : fountain_diameter = 12)
    (h2 : garden_width = 10)
    (h3 : path_width = 6) :
    fountain_diameter + 2 * garden_width + 2 * path_width = 44 :=
by sorry

end park_diameter_l3515_351596


namespace custom_operation_result_l3515_351544

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b - c)^2

/-- Main theorem -/
theorem custom_operation_result (x y z : ℝ) :
  dollar ((x - z)^2) ((y - x)^2) ((y - z)^2) = (-2*x*z + z^2 + 2*y*x - 2*y*z)^2 := by
  sorry

end custom_operation_result_l3515_351544


namespace linear_equation_condition_l3515_351549

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ m k, (a - 2) * x^(|a| - 1) + 6 = m * x + k) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
sorry

end linear_equation_condition_l3515_351549


namespace sin_cos_equality_implies_ten_degrees_l3515_351566

theorem sin_cos_equality_implies_ten_degrees (x : ℝ) :
  Real.sin (4 * x * π / 180) * Real.sin (5 * x * π / 180) = 
  Real.cos (4 * x * π / 180) * Real.cos (5 * x * π / 180) →
  x = 10 := by
  sorry

end sin_cos_equality_implies_ten_degrees_l3515_351566


namespace work_completion_men_difference_l3515_351559

theorem work_completion_men_difference (work : ℕ) : 
  ∀ (m n : ℕ), 
    m = 20 → 
    m * 10 = n * 20 → 
    m - n = 10 :=
by sorry

end work_completion_men_difference_l3515_351559


namespace diplomats_speaking_french_l3515_351543

theorem diplomats_speaking_french (T : ℕ) (F R B : ℕ) : 
  T = 70 →
  R = 38 →
  B = 7 →
  (T - F - R + B : ℤ) = 14 →
  F = 25 :=
by sorry

end diplomats_speaking_french_l3515_351543


namespace overlapping_circles_area_ratio_l3515_351565

/-- Given two overlapping circles, this theorem proves the ratio of their areas. -/
theorem overlapping_circles_area_ratio
  (L S A : ℝ)  -- L: area of large circle, S: area of small circle, A: overlapped area
  (h1 : A = 3/5 * S)  -- Overlapped area is 3/5 of small circle
  (h2 : A = 6/25 * L)  -- Overlapped area is 6/25 of large circle
  : S / L = 2/5 := by
  sorry

end overlapping_circles_area_ratio_l3515_351565


namespace merchant_markup_percentage_l3515_351592

theorem merchant_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  ((1 + M / 100) * C - 0.4 * ((1 + M / 100) * C) = 1.05 * C) →
  M = 75 := by
sorry

end merchant_markup_percentage_l3515_351592


namespace absolute_value_sum_difference_l3515_351590

theorem absolute_value_sum_difference (a b c : ℚ) :
  a = -1/4 → b = -2 → c = -11/4 → |a| + |b| - |c| = -1/2 := by
  sorry

end absolute_value_sum_difference_l3515_351590


namespace gcf_60_75_l3515_351508

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l3515_351508


namespace arithmetic_sequence_ratio_l3515_351586

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : ℚ
  common_diff : ℚ
  seq_def : ∀ n, a n = first_term + (n - 1) * common_diff

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first_term + (n - 1) * seq.common_diff) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, n > 0 → sum_n_terms a n / sum_n_terms b n = (2 * n + 3 : ℚ) / (3 * n - 1)) →
  a.a 9 / b.a 9 = 37 / 50 := by
  sorry

end arithmetic_sequence_ratio_l3515_351586


namespace arithmetic_sequence_general_term_l3515_351579

def arithmeticSequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∀ n : ℕ, a n = 2 * n - 3 :=
by sorry

end arithmetic_sequence_general_term_l3515_351579


namespace quadratic_integer_roots_count_l3515_351537

theorem quadratic_integer_roots_count :
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = m)
  (∃! s : Finset ℤ, (∀ m : ℤ, m ∈ s ↔ f m) ∧ s.card = 8) :=
sorry

end quadratic_integer_roots_count_l3515_351537


namespace magnitude_of_vector_combination_l3515_351524

-- Define the vector type
def Vec2D := ℝ × ℝ

-- Define the angle between vectors a and b
def angle_between (a b : Vec2D) : ℝ := sorry

-- Define the magnitude of a vector
def magnitude (v : Vec2D) : ℝ := sorry

-- Define the dot product of two vectors
def dot_product (a b : Vec2D) : ℝ := sorry

-- Define the vector subtraction
def vec_sub (a b : Vec2D) : Vec2D := sorry

-- Define the vector scalar multiplication
def vec_scalar_mul (r : ℝ) (v : Vec2D) : Vec2D := sorry

theorem magnitude_of_vector_combination (a b : Vec2D) :
  angle_between a b = 2 * Real.pi / 3 →
  a = (3/5, -4/5) →
  magnitude b = 2 →
  magnitude (vec_sub (vec_scalar_mul 2 a) b) = 2 * Real.sqrt 3 := by
  sorry

end magnitude_of_vector_combination_l3515_351524


namespace susan_average_speed_l3515_351588

/-- Calculates the average speed of a trip with four segments -/
def average_speed (d1 d2 d3 d4 v1 v2 v3 v4 : ℚ) : ℚ :=
  let total_distance := d1 + d2 + d3 + d4
  let total_time := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  total_distance / total_time

/-- Theorem stating that the average speed for Susan's trip is 480/19 mph -/
theorem susan_average_speed :
  average_speed 40 40 60 20 30 15 45 20 = 480 / 19 := by
  sorry

end susan_average_speed_l3515_351588


namespace count_numbers_divisible_by_291_l3515_351510

theorem count_numbers_divisible_by_291 :
  let max_k : ℕ := 291000
  let is_valid : ℕ → Prop := λ k => k ≤ max_k ∧ (k^2 - 1) % 291 = 0
  (Finset.filter is_valid (Finset.range (max_k + 1))).card = 4000 := by
  sorry

end count_numbers_divisible_by_291_l3515_351510


namespace triangle_angle_measure_l3515_351547

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b^2 = a^2 - 2*b*c →
  A = 2*π/3 →
  C = π/6 := by
sorry

end triangle_angle_measure_l3515_351547


namespace rain_hours_calculation_l3515_351514

/-- Given a 9-hour period where it rained for 4 hours, prove that it did not rain for 5 hours. -/
theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
  sorry

end rain_hours_calculation_l3515_351514


namespace fractional_inequality_condition_l3515_351530

theorem fractional_inequality_condition (x : ℝ) :
  (∀ x, 1 / x < 1 → x > 1) ∧ (∃ x, x > 1 ∧ ¬(1 / x < 1)) :=
by sorry

end fractional_inequality_condition_l3515_351530


namespace nancy_pencils_proof_l3515_351513

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial_pencils total_pencils : ℕ) : ℕ :=
  total_pencils - initial_pencils

theorem nancy_pencils_proof (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : total_pencils = 72) :
  pencils_added initial_pencils total_pencils = 45 := by
  sorry

#eval pencils_added 27 72

end nancy_pencils_proof_l3515_351513


namespace ratio_w_to_y_l3515_351580

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 5) :
  w / y = 4 / 1 := by
sorry

end ratio_w_to_y_l3515_351580


namespace number_of_boxes_l3515_351507

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) :
  total_eggs / eggs_per_box = 2 := by
  sorry

end number_of_boxes_l3515_351507


namespace expansion_theorem_l3515_351564

-- Define the sum of binomial coefficients
def sum_binomial_coeff (m : ℝ) (n : ℕ) : ℝ := 2^n

-- Define the coefficient of x in the expansion
def coeff_x (m : ℝ) (n : ℕ) : ℝ := (n.choose 2) * m^2

theorem expansion_theorem (m : ℝ) (n : ℕ) (h_m : m > 0) 
  (h_sum : sum_binomial_coeff m n = 256)
  (h_coeff : coeff_x m n = 112) :
  n = 8 ∧ m = 2 ∧ 
  (Nat.choose 8 4 * 2^4 - Nat.choose 8 2 * 2^2 : ℝ) = 1008 :=
sorry

end expansion_theorem_l3515_351564


namespace line_equation_proof_l3515_351527

theorem line_equation_proof (x y : ℝ) :
  let point_A : ℝ × ℝ := (1, 3)
  let slope_reference : ℝ := -4
  let slope_line : ℝ := slope_reference / 3
  (4 * x + 3 * y - 13 = 0) ↔
    (y - point_A.2 = slope_line * (x - point_A.1) ∧
     slope_line = slope_reference / 3) :=
by sorry

end line_equation_proof_l3515_351527
