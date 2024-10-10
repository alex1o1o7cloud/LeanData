import Mathlib

namespace lilliputian_matchboxes_in_gulliver_matchbox_l1207_120768

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scaleFactor : ℕ := 12

/-- The dimensions of a matchbox (length, width, height) -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a matchbox given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The number of Lilliputian matchboxes that fit in one dimension -/
def fitInOneDimension : ℕ := scaleFactor

theorem lilliputian_matchboxes_in_gulliver_matchbox (g : Dimensions) (l : Dimensions)
    (h_scale : l.length = g.length / scaleFactor ∧ 
               l.width = g.width / scaleFactor ∧ 
               l.height = g.height / scaleFactor) :
    (volume g) / (volume l) = 1728 := by
  sorry

end lilliputian_matchboxes_in_gulliver_matchbox_l1207_120768


namespace no_solutions_exist_l1207_120713

-- Define the greatest prime factor function
def greatest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solutions_exist : 
  ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Real.sqrt n) ∧ 
  (greatest_prime_factor (n + 60) = Real.sqrt (n + 60)) := by
  sorry

end no_solutions_exist_l1207_120713


namespace line_equation_for_triangle_l1207_120769

/-- Given a line passing through (a, 0) that forms a triangle with area T' in the first quadrant,
    prove that its equation is 2T'x - a^2y + 2aT' = 0 --/
theorem line_equation_for_triangle (a T' : ℝ) (h_a : a > 0) (h_T' : T' > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ,
    (x t = a ∧ y t = 0) ∨
    (x t = 0 ∧ y t = 2 * T' / a) ∨
    (x t ≥ 0 ∧ y t ≥ 0 ∧ 2 * T' * x t - a^2 * y t + 2 * a * T' = 0) :=
sorry

end line_equation_for_triangle_l1207_120769


namespace arithmetic_calculations_l1207_120762

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  ((-6) * (-2) / (1/8) = 96) ∧
  ((-24) * ((-3/4) - (5/6) + (7/8)) = 17) ∧
  (-(1^4) - (1 - 0.5) * (1/3) * ((-3)^2) = -5/2) := by
  sorry

end arithmetic_calculations_l1207_120762


namespace sum_of_integers_l1207_120710

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 135) : 
  x.val + y.val = 26 := by
sorry

end sum_of_integers_l1207_120710


namespace marble_problem_l1207_120708

theorem marble_problem (total : ℝ) (red blue yellow purple white : ℝ) : 
  red + blue + yellow + purple + white = total ∧
  red = 0.25 * total ∧
  blue = 0.15 * total ∧
  yellow = 0.20 * total ∧
  purple = 0.05 * total ∧
  white = 50 ∧
  total = 143 →
  blue + (red / 3) = 33 := by
sorry

end marble_problem_l1207_120708


namespace polygon_sides_l1207_120766

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles :=
by sorry

end polygon_sides_l1207_120766


namespace closest_fraction_l1207_120724

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ options ∧
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| :=
by
  sorry

end closest_fraction_l1207_120724


namespace pencil_cost_to_selling_ratio_l1207_120782

/-- Given a purchase of 90 pencils sold at a loss equal to the selling price of 40 pencils,
    the ratio of the cost of 90 pencils to the selling price of 90 pencils is 13:1. -/
theorem pencil_cost_to_selling_ratio :
  ∀ (C S : ℝ),
  C > 0 → S > 0 →
  90 * C - 40 * S = 90 * S →
  (90 * C) / (90 * S) = 13 / 1 := by
sorry

end pencil_cost_to_selling_ratio_l1207_120782


namespace square_of_binomial_formula_l1207_120787

theorem square_of_binomial_formula (x y : ℝ) :
  (2*x + y) * (y - 2*x) = y^2 - (2*x)^2 :=
by sorry

end square_of_binomial_formula_l1207_120787


namespace star_op_example_l1207_120793

/-- Custom binary operation ☼ defined for rational numbers -/
def star_op (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

/-- Theorem stating that 4 ☼ (-9) = 140 -/
theorem star_op_example : star_op 4 (-9) = 140 := by
  sorry

end star_op_example_l1207_120793


namespace function_passes_through_point_l1207_120754

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) when a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end function_passes_through_point_l1207_120754


namespace expand_product_l1207_120711

theorem expand_product (x : ℝ) : (x - 4) * (x^2 + 2*x + 1) = x^3 - 2*x^2 - 7*x - 4 := by
  sorry

end expand_product_l1207_120711


namespace final_student_score_problem_solution_l1207_120776

theorem final_student_score (total_students : ℕ) (graded_students : ℕ) 
  (initial_average : ℚ) (final_average : ℚ) : ℚ :=
  let remaining_students := total_students - graded_students
  let initial_total := initial_average * graded_students
  let final_total := final_average * total_students
  (final_total - initial_total) / remaining_students

theorem problem_solution :
  final_student_score 20 19 75 78 = 135 := by sorry

end final_student_score_problem_solution_l1207_120776


namespace darnell_call_minutes_l1207_120761

/-- Represents the monthly phone usage and plans for Darnell -/
structure PhoneUsage where
  unlimited_plan_cost : ℝ
  alt_plan_text_cost : ℝ
  alt_plan_text_limit : ℝ
  alt_plan_call_cost : ℝ
  alt_plan_call_limit : ℝ
  texts_sent : ℝ
  alt_plan_savings : ℝ

/-- Calculates the number of minutes Darnell spends on the phone each month -/
def calculate_call_minutes (usage : PhoneUsage) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, Darnell spends 60 minutes on the phone each month -/
theorem darnell_call_minutes (usage : PhoneUsage) 
  (h1 : usage.unlimited_plan_cost = 12)
  (h2 : usage.alt_plan_text_cost = 1)
  (h3 : usage.alt_plan_text_limit = 30)
  (h4 : usage.alt_plan_call_cost = 3)
  (h5 : usage.alt_plan_call_limit = 20)
  (h6 : usage.texts_sent = 60)
  (h7 : usage.alt_plan_savings = 1) :
  calculate_call_minutes usage = 60 :=
sorry

end darnell_call_minutes_l1207_120761


namespace ellipse_major_axis_length_l1207_120752

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_axis_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_axis_ratio

/-- Theorem: The length of the major axis of the ellipse is 5 -/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.25 = 5 := by
  sorry

end ellipse_major_axis_length_l1207_120752


namespace max_value_implies_a_equals_three_l1207_120783

/-- Given a function y = x(1-ax) with maximum value 1/12 for 0 < x < 1/a, prove that a = 3 -/
theorem max_value_implies_a_equals_three (a : ℝ) : 
  (∃ (max_y : ℝ), max_y = 1/12 ∧ 
    (∀ x : ℝ, 0 < x → x < 1/a → x * (1 - a*x) ≤ max_y) ∧
    (∃ x : ℝ, 0 < x ∧ x < 1/a ∧ x * (1 - a*x) = max_y)) →
  a = 3 :=
sorry

end max_value_implies_a_equals_three_l1207_120783


namespace window_purchase_savings_l1207_120794

/-- Represents the window store's pricing and discount structure -/
structure WindowStore where
  regularPrice : ℕ := 120
  freeWindowThreshold : ℕ := 5
  bulkDiscountThreshold : ℕ := 10
  bulkDiscountRate : ℚ := 0.05

/-- Calculates the cost of windows for an individual purchase -/
def individualCost (store : WindowStore) (quantity : ℕ) : ℚ :=
  let freeWindows := quantity / store.freeWindowThreshold
  let paidWindows := quantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  if quantity > store.bulkDiscountThreshold
  then basePrice * (1 - store.bulkDiscountRate)
  else basePrice

/-- Calculates the cost of windows for a collective purchase -/
def collectiveCost (store : WindowStore) (quantities : List ℕ) : ℚ :=
  let totalQuantity := quantities.sum
  let freeWindows := totalQuantity / store.freeWindowThreshold
  let paidWindows := totalQuantity - freeWindows
  let basePrice := paidWindows * store.regularPrice
  basePrice * (1 - store.bulkDiscountRate)

/-- Theorem statement for the window purchase problem -/
theorem window_purchase_savings (store : WindowStore) :
  let gregQuantity := 9
  let susanQuantity := 13
  let individualTotal := individualCost store gregQuantity + individualCost store susanQuantity
  let collectiveTotal := collectiveCost store [gregQuantity, susanQuantity]
  individualTotal - collectiveTotal = 162 := by
  sorry

end window_purchase_savings_l1207_120794


namespace equation_solutions_l1207_120792

theorem equation_solutions : 
  ∀ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ↔ 
  ((m = 5 ∧ n = -3) ∨ (m = -5 ∧ n = -3) ∨ (m = 5 ∧ n = 3) ∨ (m = -5 ∧ n = 3)) :=
by sorry

end equation_solutions_l1207_120792


namespace fish_selection_probabilities_l1207_120732

/-- The number of fish in the aquarium -/
def total_fish : ℕ := 6

/-- The number of black fish in the aquarium -/
def black_fish : ℕ := 4

/-- The number of red fish in the aquarium -/
def red_fish : ℕ := 2

/-- The number of days the teacher has classes -/
def class_days : ℕ := 4

/-- The probability of selecting fish of the same color in two consecutive draws -/
def prob_same_color : ℚ := 5 / 9

/-- The probability of selecting fish of different colors in two consecutive draws on exactly 2 out of 4 days -/
def prob_diff_color_two_days : ℚ := 800 / 2187

theorem fish_selection_probabilities :
  (prob_same_color = (black_fish / total_fish) ^ 2 + (red_fish / total_fish) ^ 2) ∧
  (prob_diff_color_two_days = 
    (class_days.choose 2 : ℚ) * prob_same_color ^ 2 * (1 - prob_same_color) ^ 2) := by
  sorry

end fish_selection_probabilities_l1207_120732


namespace magnitude_of_complex_power_l1207_120747

theorem magnitude_of_complex_power : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I) ^ 8 = (41^4 : ℝ) / 1679616 := by
  sorry

end magnitude_of_complex_power_l1207_120747


namespace marker_notebook_cost_l1207_120746

theorem marker_notebook_cost :
  ∀ (m n : ℕ),
  (10 * m + 5 * n = 120) →
  (m > n) →
  (m = 10 ∧ n = 4) →
  (m + n = 14) :=
by sorry

end marker_notebook_cost_l1207_120746


namespace binomial_coefficient_equality_l1207_120707

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) → (x = 2 ∨ x = 4) := by
  sorry

end binomial_coefficient_equality_l1207_120707


namespace parameterization_valid_iff_l1207_120750

/-- A parameterization of a line is represented by an initial point and a direction vector -/
structure Parameterization where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 2x - 4 -/
def line (x : ℝ) : ℝ := 2 * x - 4

/-- A parameterization is valid for the line y = 2x - 4 -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2

/-- Theorem: A parameterization is valid if and only if it satisfies the conditions -/
theorem parameterization_valid_iff (p : Parameterization) :
  is_valid_parameterization p ↔ 
  (line p.x₀ = p.y₀ ∧ ∃ (t : ℝ), p.dx = t * 1 ∧ p.dy = t * 2) :=
by sorry

end parameterization_valid_iff_l1207_120750


namespace diameter_of_circle_with_radius_seven_l1207_120760

/-- The diameter of a circle is twice its radius -/
def diameter (radius : ℝ) : ℝ := 2 * radius

/-- For a circle with radius 7, the diameter is 14 -/
theorem diameter_of_circle_with_radius_seven :
  diameter 7 = 14 := by sorry

end diameter_of_circle_with_radius_seven_l1207_120760


namespace equal_roots_quadratic_equation_l1207_120706

theorem equal_roots_quadratic_equation (x m : ℝ) : 
  (∃ r : ℝ, ∀ x, x^2 - 5*x + m = 0 ↔ x = r) → m = 25/4 := by
  sorry

end equal_roots_quadratic_equation_l1207_120706


namespace onions_sold_l1207_120741

theorem onions_sold (initial : ℕ) (left : ℕ) (sold : ℕ) : 
  initial = 98 → left = 33 → sold = initial - left → sold = 65 := by
sorry

end onions_sold_l1207_120741


namespace annie_candy_cost_l1207_120712

/-- Calculates the total cost of candies Annie bought for her classmates --/
def total_candy_cost (candy_a_cost candy_b_cost candy_c_cost : ℚ) 
                     (classmates : ℕ) 
                     (a_per_person b_per_person c_per_person : ℕ) : ℚ :=
  let cost_per_person := a_per_person * candy_a_cost + 
                         b_per_person * candy_b_cost + 
                         c_per_person * candy_c_cost
  cost_per_person * classmates

theorem annie_candy_cost : 
  total_candy_cost 0.1 0.15 0.2 35 3 2 1 = 28 := by
  sorry

end annie_candy_cost_l1207_120712


namespace min_value_problem_l1207_120770

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (y : ℝ), y = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) ∧
    y ≥ (2 * Real.sqrt 2016 - 2) / 2015 ∧
    (∀ (z : ℝ), z = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) → z ≥ y) := by
  sorry

end min_value_problem_l1207_120770


namespace inverse_125_mod_79_l1207_120722

theorem inverse_125_mod_79 (h : (5⁻¹ : ZMod 79) = 39) : (125⁻¹ : ZMod 79) = 69 := by
  sorry

end inverse_125_mod_79_l1207_120722


namespace customer_payment_l1207_120785

def cost_price : ℝ := 6425
def markup_percentage : ℝ := 24

theorem customer_payment (cost : ℝ) (markup : ℝ) :
  cost = cost_price →
  markup = markup_percentage →
  cost * (1 + markup / 100) = 7967 := by
  sorry

end customer_payment_l1207_120785


namespace quadratic_minimum_interval_l1207_120756

theorem quadratic_minimum_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 ≥ 5/4) ∧ 
  (∃ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 = 5/4) →
  m = -3/2 ∨ m = 7/2 := by
sorry

end quadratic_minimum_interval_l1207_120756


namespace bus_seat_capacity_l1207_120717

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seat_capacity (bus : BusSeating) : Nat :=
  sorry

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 10,
    total_capacity := 91
  }
  seat_capacity bus = 3 := by sorry

end bus_seat_capacity_l1207_120717


namespace binomial_20_17_l1207_120714

theorem binomial_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end binomial_20_17_l1207_120714


namespace concert_revenue_calculation_l1207_120773

/-- Calculates the total revenue from concert ticket sales given specific conditions --/
theorem concert_revenue_calculation (ticket_price : ℝ) 
  (first_ten_discount : ℝ) (next_twenty_discount : ℝ)
  (military_discount : ℝ) (student_discount : ℝ) (senior_discount : ℝ)
  (total_buyers : ℕ) (military_buyers : ℕ) (student_buyers : ℕ) (senior_buyers : ℕ) :
  ticket_price = 20 →
  first_ten_discount = 0.4 →
  next_twenty_discount = 0.15 →
  military_discount = 0.25 →
  student_discount = 0.2 →
  senior_discount = 0.1 →
  total_buyers = 85 →
  military_buyers = 8 →
  student_buyers = 12 →
  senior_buyers = 9 →
  (10 * (ticket_price * (1 - first_ten_discount)) +
   20 * (ticket_price * (1 - next_twenty_discount)) +
   military_buyers * (ticket_price * (1 - military_discount)) +
   student_buyers * (ticket_price * (1 - student_discount)) +
   senior_buyers * (ticket_price * (1 - senior_discount)) +
   (total_buyers - (10 + 20 + military_buyers + student_buyers + senior_buyers)) * ticket_price) = 1454 := by
  sorry


end concert_revenue_calculation_l1207_120773


namespace permutation_exists_16_no_permutation_exists_15_l1207_120704

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_valid_permutation (perm : List ℕ) (max_sum : ℕ) : Prop :=
  perm.length = numbers.length ∧
  perm.toFinset = numbers.toFinset ∧
  ∀ i, i + 2 < perm.length → perm[i]! + perm[i+1]! + perm[i+2]! ≤ max_sum

theorem permutation_exists_16 : ∃ perm, is_valid_permutation perm 16 :=
sorry

theorem no_permutation_exists_15 : ¬∃ perm, is_valid_permutation perm 15 :=
sorry

end permutation_exists_16_no_permutation_exists_15_l1207_120704


namespace quadratic_inequality_solution_l1207_120734

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → m = 1 := by
  sorry

end quadratic_inequality_solution_l1207_120734


namespace solve_for_a_l1207_120744

theorem solve_for_a : ∃ (a : ℝ), 
  let A : Set ℝ := {2, 3, a^2 + 2*a - 3}
  let B : Set ℝ := {|a + 3|, 2}
  5 ∈ A ∧ 5 ∉ B ∧ a = -4 := by
  sorry

end solve_for_a_l1207_120744


namespace total_goals_scored_l1207_120790

def soccer_match (team_a_first_half : ℕ) (team_b_second_half : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let team_a_second_half := team_b_second_half - 2
  let team_a_total := team_a_first_half + team_a_second_half
  let team_b_total := team_b_first_half + team_b_second_half
  team_a_total + team_b_total = 26

theorem total_goals_scored :
  soccer_match 8 8 := by sorry

end total_goals_scored_l1207_120790


namespace corn_yield_theorem_l1207_120702

/-- Calculates the total corn yield for Johnson and his neighbor after 6 months -/
def total_corn_yield (johnson_yield : ℕ) (johnson_area : ℕ) (neighbor_area : ℕ) (months : ℕ) : ℕ :=
  let johnson_total := johnson_yield * (months / 2)
  let neighbor_yield := 2 * johnson_yield
  let neighbor_total := neighbor_yield * neighbor_area * (months / 2)
  johnson_total + neighbor_total

/-- Theorem stating that the total corn yield is 1200 under given conditions -/
theorem corn_yield_theorem :
  total_corn_yield 80 1 2 6 = 1200 :=
by
  sorry

end corn_yield_theorem_l1207_120702


namespace jill_net_salary_l1207_120759

/-- Represents Jill's financial situation --/
structure JillFinances where
  net_salary : ℝ
  discretionary_income : ℝ
  vacation_fund_percent : ℝ
  savings_percent : ℝ
  socializing_percent : ℝ
  remaining_amount : ℝ

/-- Theorem stating Jill's net monthly salary given her financial conditions --/
theorem jill_net_salary (j : JillFinances) 
  (h1 : j.discretionary_income = j.net_salary / 5)
  (h2 : j.vacation_fund_percent = 0.3)
  (h3 : j.savings_percent = 0.2)
  (h4 : j.socializing_percent = 0.35)
  (h5 : j.remaining_amount = 108)
  (h6 : (1 - (j.vacation_fund_percent + j.savings_percent + j.socializing_percent)) * j.discretionary_income = j.remaining_amount) :
  j.net_salary = 3600 := by
  sorry

#check jill_net_salary

end jill_net_salary_l1207_120759


namespace square_sum_problem_l1207_120758

theorem square_sum_problem (a b c d m n : ℕ+) 
  (sum_eq : a + b + c + d = m^2)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 1989)
  (max_eq : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end square_sum_problem_l1207_120758


namespace brother_got_two_l1207_120749

-- Define the type for grades
inductive Grade : Type
  | one : Grade
  | two : Grade
  | three : Grade
  | four : Grade
  | five : Grade

-- Define the sneezing function
def grandmother_sneezes (statement : Prop) : Prop := sorry

-- Define the brother's grade
def brothers_grade : Grade := sorry

-- Theorem statement
theorem brother_got_two :
  -- Condition 1: When the brother tells the truth, the grandmother sneezes
  (∀ (statement : Prop), statement → grandmother_sneezes statement) →
  -- Condition 2: The brother said he got a "5", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.five)) →
  -- Condition 3: The brother said he got a "4", and the grandmother sneezed
  (grandmother_sneezes (brothers_grade = Grade.four)) →
  -- Condition 4: The brother said he got at least a "3", but the grandmother didn't sneeze
  (¬ grandmother_sneezes (brothers_grade = Grade.three ∨ brothers_grade = Grade.four ∨ brothers_grade = Grade.five)) →
  -- Conclusion: The brother's grade is 2
  brothers_grade = Grade.two :=
by
  sorry

end brother_got_two_l1207_120749


namespace vector_angle_problem_l1207_120725

theorem vector_angle_problem (α β : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (Real.cos α, Real.sin α))
  (h2 : b = (Real.cos β, Real.sin β))
  (h3 : ‖a - b‖ = (2 / 5) * Real.sqrt 5)
  (h4 : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h5 : Real.sin β = -5/13) :
  Real.cos (α - β) = 3/5 ∧ Real.sin α = 33/65 := by
  sorry

end vector_angle_problem_l1207_120725


namespace smallest_factor_of_4814_l1207_120738

theorem smallest_factor_of_4814 (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧
  10 ≤ b ∧ b ≤ 99 ∧
  a * b = 4814 ∧
  a ≤ b →
  a = 53 := by sorry

end smallest_factor_of_4814_l1207_120738


namespace intersection_of_sets_l1207_120718

theorem intersection_of_sets : 
  let P : Set ℤ := {-3, -2, 0, 2}
  let Q : Set ℤ := {-1, -2, -3, 0, 1}
  P ∩ Q = {-3, -2, 0} := by
sorry

end intersection_of_sets_l1207_120718


namespace trajectory_of_M_l1207_120731

/-- The trajectory of point M satisfying the given conditions -/
theorem trajectory_of_M (x y : ℝ) (h : x ≥ 3/2) :
  (∀ (t : ℝ), t^2 + y^2 = 1 → 
    Real.sqrt ((x - t)^2 + y^2) = Real.sqrt ((x - 2)^2 + y^2) + 1) →
  3 * x^2 - y^2 - 8 * x + 5 = 0 := by
  sorry

end trajectory_of_M_l1207_120731


namespace sequence_x_perfect_square_l1207_120733

def perfect_square_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, s n = m^2

def sequence_x : ℕ → ℤ
| 0 => 0
| 1 => 3
| (n + 2) => 4 * sequence_x (n + 1) - sequence_x n

theorem sequence_x_perfect_square :
  perfect_square_sequence (λ n => (sequence_x (n + 1) * sequence_x (n - 1) + 9).natAbs) := by
  sorry

end sequence_x_perfect_square_l1207_120733


namespace football_progress_l1207_120700

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

theorem football_progress : yard_changes.sum = 11 := by
  sorry

end football_progress_l1207_120700


namespace trivia_team_members_l1207_120786

/-- Represents a trivia team with its total members and points scored. -/
structure TriviaTeam where
  totalMembers : ℕ
  absentMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Theorem stating the total members in the trivia team -/
theorem trivia_team_members (team : TriviaTeam)
  (h1 : team.absentMembers = 4)
  (h2 : team.pointsPerMember = 8)
  (h3 : team.totalPoints = 64)
  : team.totalMembers = 12 := by
  sorry

#check trivia_team_members

end trivia_team_members_l1207_120786


namespace trig_expression_equals_32_l1207_120701

theorem trig_expression_equals_32 : 
  3 / (Real.sin (20 * π / 180))^2 - 1 / (Real.cos (20 * π / 180))^2 + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end trig_expression_equals_32_l1207_120701


namespace value_of_expression_l1207_120739

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end value_of_expression_l1207_120739


namespace points_one_unit_from_negative_two_l1207_120729

theorem points_one_unit_from_negative_two : 
  ∀ x : ℝ, abs (x - (-2)) = 1 ↔ x = -3 ∨ x = -1 := by sorry

end points_one_unit_from_negative_two_l1207_120729


namespace factorization_equality_l1207_120757

theorem factorization_equality (x y : ℝ) : 5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) := by
  sorry

end factorization_equality_l1207_120757


namespace rectangle_ellipse_theorem_l1207_120775

/-- Represents a rectangle ABCD with an inscribed ellipse K -/
structure RectangleWithEllipse where
  -- Length of side AB
  ab : ℝ
  -- Length of side AD
  ad : ℝ
  -- Point M on AB where the minor axis of K intersects
  am : ℝ
  -- Point L on AB where the minor axis of K intersects
  lb : ℝ
  -- Ensure AB = 2
  ab_eq_two : ab = 2
  -- Ensure AD < √2
  ad_lt_sqrt_two : ad < Real.sqrt 2
  -- Ensure M and L are on AB
  m_l_on_ab : am + lb = ab

/-- The theorem to be proved -/
theorem rectangle_ellipse_theorem (rect : RectangleWithEllipse) :
  rect.am^2 - rect.lb^2 = -8 := by
  sorry

end rectangle_ellipse_theorem_l1207_120775


namespace smith_family_seating_arrangements_l1207_120796

/-- Represents a family with parents and children -/
structure Family :=
  (num_parents : Nat)
  (num_children : Nat)

/-- Represents a car with front and back seats -/
structure Car :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of seating arrangements for a family in a car -/
def seating_arrangements (f : Family) (c : Car) (parent_driver : Bool) : Nat :=
  sorry

/-- The Smith family with 2 parents and 3 children -/
def smith_family : Family :=
  { num_parents := 2, num_children := 3 }

/-- The Smith family car with 2 front seats and 3 back seats -/
def smith_car : Car :=
  { front_seats := 2, back_seats := 3 }

theorem smith_family_seating_arrangements :
  seating_arrangements smith_family smith_car true = 48 := by
  sorry

end smith_family_seating_arrangements_l1207_120796


namespace line_translation_l1207_120740

-- Define the original line
def original_line (x : ℝ) : ℝ := 2 * x + 1

-- Define the translation amount
def translation : ℝ := -2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := 2 * x - 1

-- Theorem stating that the translation of the original line results in the translated line
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x + translation :=
sorry

end line_translation_l1207_120740


namespace operation_terminates_l1207_120721

/-- A sequence of positive integers -/
def Sequence := List Nat

/-- Represents the operation of replacing adjacent numbers -/
inductive Operation
  | replaceLeft (x y : Nat) : Operation  -- Replaces (x, y) with (y+1, x)
  | replaceRight (x y : Nat) : Operation -- Replaces (x, y) with (x-1, x)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match s, op with
  | x::y::rest, Operation.replaceLeft x' y' => if x > y then (y+1)::x::rest else s
  | x::y::rest, Operation.replaceRight x' y' => if x > y then (x-1)::x::rest else s
  | _, _ => s

/-- Theorem: The process of applying operations terminates after finite iterations -/
theorem operation_terminates (s : Sequence) : 
  ∃ (n : Nat), ∀ (ops : List Operation), ops.length > n → 
    (ops.foldl applyOperation s = s) := by
  sorry


end operation_terminates_l1207_120721


namespace arithmetic_sequence_property_l1207_120781

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 12) :
  a 8 = 21 :=
sorry

end arithmetic_sequence_property_l1207_120781


namespace triangle_inequality_l1207_120703

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2)
  ≤ Real.sqrt (5*a^2 + 5*b^2 + 5*c^2 + 4*a*b + 4*b*c + 4*c*a) :=
by sorry

end triangle_inequality_l1207_120703


namespace water_bottle_shortage_l1207_120764

/-- Represents the water bottle consumption during a soccer match --/
structure WaterBottleConsumption where
  initial_bottles : ℕ
  first_break_players : ℕ
  first_break_bottles_per_player : ℕ
  second_break_players : ℕ
  second_break_bottles_per_player : ℕ
  second_break_extra_bottles : ℕ
  third_break_players : ℕ
  third_break_bottles_per_player : ℕ

/-- Calculates the shortage of water bottles after the match --/
def calculate_shortage (consumption : WaterBottleConsumption) : ℤ :=
  let total_used := 
    consumption.first_break_players * consumption.first_break_bottles_per_player +
    consumption.second_break_players * consumption.second_break_bottles_per_player +
    consumption.second_break_extra_bottles +
    consumption.third_break_players * consumption.third_break_bottles_per_player
  consumption.initial_bottles - total_used

/-- Theorem stating that there is a shortage of 4 bottles given the match conditions --/
theorem water_bottle_shortage : 
  ∃ (consumption : WaterBottleConsumption), 
    consumption.initial_bottles = 48 ∧
    consumption.first_break_players = 11 ∧
    consumption.first_break_bottles_per_player = 2 ∧
    consumption.second_break_players = 14 ∧
    consumption.second_break_bottles_per_player = 1 ∧
    consumption.second_break_extra_bottles = 4 ∧
    consumption.third_break_players = 12 ∧
    consumption.third_break_bottles_per_player = 1 ∧
    calculate_shortage consumption = -4 := by
  sorry

end water_bottle_shortage_l1207_120764


namespace perpendicular_vectors_l1207_120797

def vector_a : ℝ × ℝ := (4, -5)
def vector_b (b : ℝ) : ℝ × ℝ := (b, 3)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem perpendicular_vectors (b : ℝ) :
  dot_product vector_a (vector_b b) = 0 → b = 15/4 := by
  sorry

end perpendicular_vectors_l1207_120797


namespace ava_apple_trees_l1207_120791

theorem ava_apple_trees (lily_trees : ℕ) : 
  (lily_trees + 3) + lily_trees = 15 → (lily_trees + 3) = 9 := by
  sorry

end ava_apple_trees_l1207_120791


namespace irrational_between_neg_one_and_two_l1207_120772

theorem irrational_between_neg_one_and_two :
  (-1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 3 ∧ -Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 5 ∧ -Real.sqrt 5 < 2) ∧
  ¬(-1 < Real.sqrt 5 ∧ Real.sqrt 5 < 2) :=
by
  sorry

end irrational_between_neg_one_and_two_l1207_120772


namespace theresa_video_games_l1207_120730

/-- The number of video games each person has -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.tory = 2 * vg.alex ∧
  vg.tory = 6

/-- The theorem to prove -/
theorem theresa_video_games (vg : VideoGames) (h : satisfies_conditions vg) : vg.theresa = 11 := by
  sorry

#check theresa_video_games

end theresa_video_games_l1207_120730


namespace juggling_improvement_l1207_120753

/-- 
Given:
- start_objects: The number of objects Jeanette starts juggling with
- weeks: The number of weeks Jeanette practices
- end_objects: The number of objects Jeanette can juggle at the end
- weekly_improvement: The number of additional objects Jeanette can juggle each week

Prove that with the given conditions, the weekly improvement is 2.
-/
theorem juggling_improvement 
  (start_objects : ℕ) 
  (weeks : ℕ) 
  (end_objects : ℕ) 
  (weekly_improvement : ℕ) 
  (h1 : start_objects = 3)
  (h2 : weeks = 5)
  (h3 : end_objects = 13)
  (h4 : end_objects = start_objects + weeks * weekly_improvement) : 
  weekly_improvement = 2 := by
  sorry


end juggling_improvement_l1207_120753


namespace simplify_and_rationalize_l1207_120716

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l1207_120716


namespace video_game_problem_l1207_120788

theorem video_game_problem (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) :
  total_games - (total_earnings / price_per_game) = total_games - (total_earnings / price_per_game) :=
by sorry

end video_game_problem_l1207_120788


namespace negation_of_forall_cubic_l1207_120723

theorem negation_of_forall_cubic (P : ℝ → Prop) :
  (¬ ∀ x < 0, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x < 0, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_forall_cubic_l1207_120723


namespace time_after_adding_seconds_l1207_120720

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (4:45:00 a.m.) -/
def initialTime : Time :=
  { hours := 4, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The resulting time after adding seconds -/
def resultTime : Time :=
  { hours := 8, minutes := 30, seconds := 45 }

theorem time_after_adding_seconds :
  addSeconds initialTime secondsToAdd = resultTime := by
  sorry

end time_after_adding_seconds_l1207_120720


namespace passing_percentage_l1207_120767

def max_marks : ℕ := 300
def obtained_marks : ℕ := 160
def failed_by : ℕ := 20

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end passing_percentage_l1207_120767


namespace deck_size_proof_l1207_120736

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end deck_size_proof_l1207_120736


namespace weight_replacement_l1207_120755

theorem weight_replacement (n : ℕ) (new_weight : ℝ) (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 68)
  (h3 : avg_increase = 1) :
  n * avg_increase = new_weight - (new_weight - n * avg_increase) :=
by
  sorry

#check weight_replacement

end weight_replacement_l1207_120755


namespace vegetable_ghee_mixture_ratio_l1207_120771

/-- The ratio of volumes of two brands of vegetable ghee in a mixture -/
theorem vegetable_ghee_mixture_ratio :
  ∀ (Va Vb : ℝ),
  Va + Vb = 4 →
  900 * Va + 850 * Vb = 3520 →
  Va / Vb = 3 / 2 := by
  sorry

end vegetable_ghee_mixture_ratio_l1207_120771


namespace sum_of_squares_16_to_30_l1207_120774

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 := by
  sorry

end sum_of_squares_16_to_30_l1207_120774


namespace more_ones_than_zeros_mod_500_l1207_120789

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ := sorry

/-- The set of positive integers less than or equal to 1000 whose binary representation has more 1's than 0's -/
def M : Finset ℕ := sorry

theorem more_ones_than_zeros_mod_500 : M.card % 500 = 61 := by sorry

end more_ones_than_zeros_mod_500_l1207_120789


namespace eighth_row_interior_sum_l1207_120784

/-- Sum of all elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end eighth_row_interior_sum_l1207_120784


namespace complement_of_union_l1207_120745

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end complement_of_union_l1207_120745


namespace max_N_value_l1207_120777

def N (a b c : ℕ) : ℕ := a * b * c + a * b + b * c + a - b - c

theorem max_N_value :
  ∃ (a b c : ℕ),
    a ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    b ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    c ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ),
      x ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      y ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      z ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      x ≠ y → y ≠ z → x ≠ z →
      N a b c ≥ N x y z ∧
    N a b c = 167 :=
  sorry

end max_N_value_l1207_120777


namespace cosine_sine_inequality_l1207_120780

theorem cosine_sine_inequality (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = π / 2) : 
  Real.cos a + Real.cos b + Real.cos c > Real.sin a + Real.sin b + Real.sin c := by
  sorry

end cosine_sine_inequality_l1207_120780


namespace product_of_exponents_l1207_120715

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 252 → 
  2^r + 58 = 122 → 
  5^3 * 6^s = 117000 → 
  p * r * s = 36 := by
  sorry

end product_of_exponents_l1207_120715


namespace cos_pi_4_minus_alpha_l1207_120748

theorem cos_pi_4_minus_alpha (α : Real) (h : Real.sin (α - 7 * Real.pi / 4) = 1 / 2) :
  Real.cos (Real.pi / 4 - α) = 1 / 2 := by
  sorry

end cos_pi_4_minus_alpha_l1207_120748


namespace expression_evaluation_l1207_120726

theorem expression_evaluation : (1 + (3 * 5)) / 2 = 8 := by
  sorry

end expression_evaluation_l1207_120726


namespace expand_product_l1207_120798

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end expand_product_l1207_120798


namespace complex_subtraction_l1207_120735

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 4 - I) :
  a - 2*b = -3 - I := by sorry

end complex_subtraction_l1207_120735


namespace cone_sphere_ratio_l1207_120799

-- Define the radius, height, and volumes
variable (r : ℝ) -- radius of the base (shared by cone and sphere)
variable (h : ℝ) -- height of the cone
variable (V_sphere V_cone : ℝ) -- volumes of sphere and cone

-- Define the theorem
theorem cone_sphere_ratio :
  (V_sphere = (4/3) * Real.pi * r^3) →  -- volume formula for sphere
  (V_cone = (1/3) * Real.pi * r^2 * h) →  -- volume formula for cone
  (V_cone = (1/3) * V_sphere) →  -- given condition
  (h / r = 4/3) :=  -- conclusion to prove
by
  sorry  -- proof omitted

end cone_sphere_ratio_l1207_120799


namespace r_daily_earning_l1207_120705

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 40 per day. -/
theorem r_daily_earning (p q r : ℕ) : 
  (9 * (p + q + r) = 1890) →
  (5 * (p + r) = 600) →
  (7 * (q + r) = 910) →
  r = 40 := by
  sorry

end r_daily_earning_l1207_120705


namespace total_population_theorem_l1207_120709

/-- Represents the population of a school -/
structure SchoolPopulation where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls
  t : ℕ  -- number of teachers

/-- Checks if the school population satisfies the given conditions -/
def isValidPopulation (p : SchoolPopulation) : Prop :=
  p.b = 4 * p.g ∧ p.g = 2 * p.t

/-- Calculates the total population of the school -/
def totalPopulation (p : SchoolPopulation) : ℕ :=
  p.b + p.g + p.t

/-- Theorem stating that for a valid school population, 
    the total population is equal to 11b/8 -/
theorem total_population_theorem (p : SchoolPopulation) 
  (h : isValidPopulation p) : 
  (totalPopulation p : ℚ) = 11 * (p.b : ℚ) / 8 := by
  sorry

end total_population_theorem_l1207_120709


namespace max_daily_profit_l1207_120779

/-- The daily profit function for a store selling an item -/
noncomputable def daily_profit (x : ℕ) : ℝ :=
  if x ≥ 1 ∧ x ≤ 30 then
    -x^2 + 52*x + 620
  else if x ≥ 31 ∧ x ≤ 60 then
    -40*x + 2480
  else
    0

/-- The maximum daily profit and the day it occurs -/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1296 ∧
    max_day = 26 ∧
    (∀ x : ℕ, x ≥ 1 ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
by sorry

end max_daily_profit_l1207_120779


namespace intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l1207_120719

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+1)*x + a = 0}

-- Define the complement of B in ℝ
def C_ℝB (a : ℝ) : Set ℝ := {x : ℝ | x ∉ B a}

-- Statement 1
theorem intersection_A_complement_B_when_a_2 :
  A ∩ C_ℝB 2 = {-3} :=
sorry

-- Statement 2
theorem a_values_when_A_union_B_equals_A :
  {a : ℝ | A ∪ B a = A} = {-3, 1} :=
sorry

end intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l1207_120719


namespace chromatic_flow_duality_l1207_120763

/-- A planar multigraph -/
structure PlanarMultigraph where
  -- Add necessary fields

/-- The dual of a planar multigraph -/
def dual (G : PlanarMultigraph) : PlanarMultigraph :=
  sorry

/-- The chromatic number of a planar multigraph -/
def chromaticNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- The flow number of a planar multigraph -/
def flowNumber (G : PlanarMultigraph) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a planar multigraph equals the flow number of its dual -/
theorem chromatic_flow_duality (G : PlanarMultigraph) :
    chromaticNumber G = flowNumber (dual G) :=
  sorry

end chromatic_flow_duality_l1207_120763


namespace grade_average_condition_l1207_120778

theorem grade_average_condition (grades : List ℤ) (n : ℕ) :
  n > 0 →
  n = grades.length →
  (grades.sum : ℚ) / n = 46 / 10 →
  ∃ k : ℕ, n = 5 * k :=
by sorry

end grade_average_condition_l1207_120778


namespace cos_equality_solution_l1207_120727

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1230 * π / 180) → n = 150 := by
  sorry

end cos_equality_solution_l1207_120727


namespace alcohol_mixture_proof_l1207_120795

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_proof :
  let solution_x_volume : ℝ := 300
  let solution_x_concentration : ℝ := 0.10
  let solution_y_volume : ℝ := 100
  let solution_y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.15
  let total_volume := solution_x_volume + solution_y_volume
  let total_alcohol := solution_x_volume * solution_x_concentration + solution_y_volume * solution_y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

end alcohol_mixture_proof_l1207_120795


namespace constant_speed_calculation_l1207_120751

/-- Proves that a journey of 2304 kilometers completed in 36 hours at a constant speed results in a speed of 64 km/h -/
theorem constant_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 2304 →
  time = 36 →
  speed = distance / time →
  speed = 64 := by
sorry

end constant_speed_calculation_l1207_120751


namespace hyperbola_focal_length_l1207_120742

/-- Represents a hyperbola in the form x²/a² - y²/b² = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a parabola in the form y² = 2px --/
structure Parabola where
  p : ℝ
  h_pos_p : p > 0

/-- The focal length of a hyperbola is the distance from the center to a focus --/
def focal_length (h : Hyperbola) : ℝ := sorry

/-- The left vertex of a hyperbola --/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- The focus of a parabola --/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The directrix of a parabola --/
def parabola_directrix (p : Parabola) : ℝ × ℝ → Prop := sorry

/-- An asymptote of a hyperbola --/
def hyperbola_asymptote (h : Hyperbola) : ℝ × ℝ → Prop := sorry

/-- The distance between two points --/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_length 
  (h : Hyperbola) 
  (p : Parabola) 
  (h_distance : distance (left_vertex h) (parabola_focus p) = 4)
  (h_intersection : ∃ (pt : ℝ × ℝ), hyperbola_asymptote h pt ∧ parabola_directrix p pt ∧ pt = (-2, -1)) :
  focal_length h = 2 := by sorry

end hyperbola_focal_length_l1207_120742


namespace proportionality_problem_l1207_120737

/-- Given that x is directly proportional to y², y is inversely proportional to z²,
    and x = 5 when z = 8, prove that x = 5/256 when z = 32 -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^2)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 5 ∧ z = 8) :
  x = 5/256 ∧ z = 32 := by
  sorry

end proportionality_problem_l1207_120737


namespace y_intercept_for_specific_line_l1207_120728

/-- A line in the 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 7 }
  y_intercept l = (0, 21) := by
  sorry

end y_intercept_for_specific_line_l1207_120728


namespace sqrt_sum_difference_equality_l1207_120743

theorem sqrt_sum_difference_equality : 
  Real.sqrt 27 + Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = (4 * Real.sqrt 3) / 3 := by
  sorry

end sqrt_sum_difference_equality_l1207_120743


namespace strengthened_inequality_l1207_120765

theorem strengthened_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) := by
  sorry

end strengthened_inequality_l1207_120765
