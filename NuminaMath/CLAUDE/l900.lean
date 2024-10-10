import Mathlib

namespace half_of_half_equals_half_l900_90065

theorem half_of_half_equals_half (x : ℝ) : (1/2 * (1/2 * x) = 1/2) → x = 2 := by
  sorry

end half_of_half_equals_half_l900_90065


namespace susan_chairs_l900_90098

/-- The number of chairs in Susan's house -/
def total_chairs : ℕ :=
  let red_chairs : ℕ := 5
  let yellow_chairs : ℕ := 4 * red_chairs
  let blue_chairs : ℕ := yellow_chairs - 2
  let green_chairs : ℕ := (red_chairs + blue_chairs) / 2
  red_chairs + yellow_chairs + blue_chairs + green_chairs

/-- Theorem stating the total number of chairs in Susan's house -/
theorem susan_chairs : total_chairs = 54 := by
  sorry

end susan_chairs_l900_90098


namespace pencil_cost_l900_90064

/-- Calculates the cost of a pencil given shopping information -/
theorem pencil_cost (initial_amount : ℚ) (hat_cost : ℚ) (num_cookies : ℕ) (cookie_cost : ℚ) (remaining_amount : ℚ) : 
  initial_amount = 20 →
  hat_cost = 10 →
  num_cookies = 4 →
  cookie_cost = 5/4 →
  remaining_amount = 3 →
  initial_amount - (hat_cost + num_cookies * cookie_cost + remaining_amount) = 2 :=
by sorry

end pencil_cost_l900_90064


namespace product_sum_difference_l900_90081

theorem product_sum_difference (a b : ℤ) (h1 : b = 8) (h2 : b - a = 3) :
  a * b - 2 * (a + b) = 14 := by sorry

end product_sum_difference_l900_90081


namespace power_difference_l900_90036

theorem power_difference (a : ℝ) (m n : ℤ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m-n) = 3 := by
  sorry

end power_difference_l900_90036


namespace point_not_on_line_l900_90051

theorem point_not_on_line (m b : ℝ) (h : m + b < 0) : 
  ¬(∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = 20) :=
by sorry

end point_not_on_line_l900_90051


namespace triangle_cookie_cutters_l900_90011

theorem triangle_cookie_cutters (total_sides : ℕ) (square_cutters : ℕ) (hexagon_cutters : ℕ) 
  (h1 : total_sides = 46)
  (h2 : square_cutters = 4)
  (h3 : hexagon_cutters = 2) :
  ∃ (triangle_cutters : ℕ), 
    triangle_cutters * 3 + square_cutters * 4 + hexagon_cutters * 6 = total_sides ∧ 
    triangle_cutters = 6 := by
  sorry

end triangle_cookie_cutters_l900_90011


namespace fractional_equation_solution_l900_90068

theorem fractional_equation_solution :
  ∃ x : ℚ, x = -3/4 ∧ x / (x + 1) = 2 * x / (3 * x + 3) - 1 :=
sorry

end fractional_equation_solution_l900_90068


namespace smallest_n_divisibility_l900_90003

theorem smallest_n_divisibility : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(540 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 
  540 ∣ n^3 ∧ 
  n = 60 := by
  sorry

end smallest_n_divisibility_l900_90003


namespace expression_equals_one_l900_90033

theorem expression_equals_one : 
  (150^2 - 9^2) / (110^2 - 13^2) * ((110 - 13) * (110 + 13)) / ((150 - 9) * (150 + 9)) = 1 := by
  sorry

end expression_equals_one_l900_90033


namespace max_missable_problems_l900_90045

theorem max_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 40) :
  ⌊total_problems * (1 - passing_percentage)⌋ = 6 :=
sorry

end max_missable_problems_l900_90045


namespace convince_jury_l900_90061

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Normal

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the statement made by the person
def statement (p : Person) : Prop :=
  p.guilty ∧ p.type = PersonType.Liar

-- Define what it means for a person to be consistent with their statement
def consistent (p : Person) : Prop :=
  (p.type = PersonType.Knight ∧ statement p) ∨
  (p.type = PersonType.Liar ∧ ¬statement p) ∨
  (p.type = PersonType.Normal ∧ statement p)

-- Theorem to prove
theorem convince_jury :
  ∃ (p : Person), consistent p ∧ ¬p.guilty ∧ p.type ≠ PersonType.Knight :=
sorry

end convince_jury_l900_90061


namespace area_AXYD_area_AXYD_is_72_l900_90042

/-- Rectangle ABCD with given dimensions and point E -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (AB : ℝ)
  (BC : ℝ)

/-- Point Z on the extension of BC -/
def Z (rect : Rectangle) : ℝ × ℝ := (rect.C.1, rect.C.2 + 18)

/-- Conditions for the rectangle and point E -/
def validRectangle (rect : Rectangle) : Prop :=
  rect.AB = 20 ∧
  rect.BC = 12 ∧
  rect.A = (0, 0) ∧
  rect.B = (20, 0) ∧
  rect.C = (20, 12) ∧
  rect.D = (0, 12) ∧
  rect.E = (6, 6)

/-- Theorem: Area of quadrilateral AXYD is 72 -/
theorem area_AXYD (rect : Rectangle) (h : validRectangle rect) : ℝ :=
  72

/-- Main theorem: If the rectangle satisfies the conditions, then the area of AXYD is 72 -/
theorem area_AXYD_is_72 (rect : Rectangle) (h : validRectangle rect) : 
  area_AXYD rect h = 72 := by
  sorry

end area_AXYD_area_AXYD_is_72_l900_90042


namespace original_denominator_proof_l900_90093

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 23 := by
sorry

end original_denominator_proof_l900_90093


namespace mutually_exclusive_events_l900_90056

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Blue

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 white balls and 2 blue balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.White} + 2 • {BallColor.Blue}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: All drawn balls are blue -/
def allBlue (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Blue ∧ outcome.second = BallColor.Blue

/-- The probability of an event occurring when drawing two balls from the bag -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ := sorry

/-- Theorem: "At least one white ball" and "All are blue balls" are mutually exclusive -/
theorem mutually_exclusive_events :
  probability (λ outcome => atLeastOneWhite outcome ∧ allBlue outcome) = 0 :=
sorry

end mutually_exclusive_events_l900_90056


namespace min_value_parallel_vectors_l900_90054

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 3^x + 9^y + 2 -/
theorem min_value_parallel_vectors (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3 - x, y]
  let b : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), a = k • b) →
  (∀ (x' y' : ℝ), 3^x' + 9^y' + 2 ≥ 6 * Real.sqrt 3 + 2) ∧
  (∃ (x₀ y₀ : ℝ), 3^x₀ + 9^y₀ + 2 = 6 * Real.sqrt 3 + 2) :=
by sorry

end min_value_parallel_vectors_l900_90054


namespace percent_equality_l900_90073

theorem percent_equality (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end percent_equality_l900_90073


namespace sum_of_max_min_is_10_l900_90077

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem sum_of_max_min_is_10 :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a + f b = 10 :=
sorry

end sum_of_max_min_is_10_l900_90077


namespace simplify_expression_l900_90010

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b^2) - 2*b^2 + 5 = 9*b^4 + 6*b^3 - 2*b^2 + 5 := by
  sorry

end simplify_expression_l900_90010


namespace tree_spacing_l900_90049

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (spacing : ℕ) :
  yard_length = 434 →
  num_trees = 32 →
  spacing * (num_trees - 1) = yard_length →
  spacing = 14 :=
by
  sorry

end tree_spacing_l900_90049


namespace markup_calculation_l900_90032

theorem markup_calculation (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : 
  purchase_price = 48 →
  overhead_percentage = 0.25 →
  net_profit = 12 →
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 24 := by
sorry

end markup_calculation_l900_90032


namespace darks_drying_time_l900_90024

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  wash : Nat
  dry : Nat

/-- Calculates the total time for a load of laundry -/
def totalTime (lt : LaundryTime) : Nat :=
  lt.wash + lt.dry

theorem darks_drying_time (whites : LaundryTime) (darks_wash : Nat) (colors : LaundryTime) 
    (total_time : Nat) (h1 : whites.wash = 72) (h2 : whites.dry = 50)
    (h3 : darks_wash = 58) (h4 : colors.wash = 45) (h5 : colors.dry = 54)
    (h6 : total_time = 344) :
    ∃ (darks_dry : Nat), darks_dry = 65 ∧ 
    total_time = totalTime whites + totalTime colors + darks_wash + darks_dry := by
  sorry

#check darks_drying_time

end darks_drying_time_l900_90024


namespace negative_sqrt_four_equals_negative_two_l900_90052

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end negative_sqrt_four_equals_negative_two_l900_90052


namespace jack_age_l900_90040

/-- Given that Jack's age is 20 years less than twice Jane's age,
    and the sum of their ages is 60, prove that Jack is 33 years old. -/
theorem jack_age (j a : ℕ) 
  (h1 : j = 2 * a - 20)  -- Jack's age is 20 years less than twice Jane's age
  (h2 : j + a = 60)      -- The sum of their ages is 60
  : j = 33 := by
  sorry

end jack_age_l900_90040


namespace income_ratio_is_seven_to_six_l900_90008

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- Given the conditions of the problem, prove that the ratio of Rajan's income to Balan's income is 7:6 -/
theorem income_ratio_is_seven_to_six 
  (rajan balan : Person)
  (h1 : rajan.expenditure * 5 = balan.expenditure * 6)
  (h2 : rajan.income - rajan.expenditure = 1000)
  (h3 : balan.income - balan.expenditure = 1000)
  (h4 : rajan.income = 7000) :
  7 * balan.income = 6 * rajan.income := by
  sorry

#check income_ratio_is_seven_to_six

end income_ratio_is_seven_to_six_l900_90008


namespace certain_number_proof_l900_90015

theorem certain_number_proof (n : ℕ) : 
  n % 10 = 6 ∧ 1442 % 10 = 12 → n = 1446 :=
by sorry

end certain_number_proof_l900_90015


namespace inequality_proof_l900_90072

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end inequality_proof_l900_90072


namespace quadratic_real_roots_l900_90055

theorem quadratic_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) ↔ (a = 1 ∧ b = -1/2) :=
by sorry

end quadratic_real_roots_l900_90055


namespace missy_claims_count_l900_90095

/-- The number of insurance claims that can be handled by three agents --/
def insurance_claims (jan_claims : ℕ) : ℕ × ℕ × ℕ :=
  let john_claims := jan_claims + (jan_claims * 30 / 100)
  let missy_claims := john_claims + 15
  (jan_claims, john_claims, missy_claims)

/-- Theorem stating that Missy can handle 41 claims given the conditions --/
theorem missy_claims_count :
  let (jan, john, missy) := insurance_claims 20
  missy = 41 := by sorry

end missy_claims_count_l900_90095


namespace cube_edge_length_range_l900_90017

theorem cube_edge_length_range (volume : ℝ) (h : volume = 100) :
  ∃ (edge : ℝ), edge ^ 3 = volume ∧ 4 < edge ∧ edge < 5 := by
  sorry

end cube_edge_length_range_l900_90017


namespace parallelogram_area_l900_90070

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height horiz_diff : ℝ) 
  (h_base : base = 20)
  (h_slant : slant_height = 6)
  (h_diff : horiz_diff = 5) :
  base * Real.sqrt (slant_height^2 - horiz_diff^2) = 20 * Real.sqrt 11 := by
  sorry

#check parallelogram_area

end parallelogram_area_l900_90070


namespace candy_bar_cost_l900_90097

/-- The cost of candy bars purchased by Dan -/
def total_cost : ℚ := 6

/-- The number of candy bars Dan bought -/
def number_of_bars : ℕ := 2

/-- The cost of each candy bar -/
def cost_per_bar : ℚ := total_cost / number_of_bars

/-- Theorem stating that the cost of each candy bar is $3 -/
theorem candy_bar_cost : cost_per_bar = 3 := by
  sorry

end candy_bar_cost_l900_90097


namespace line_through_circle_center_l900_90048

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*x' - 4*y' = 0 → 
   (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) → 
  a = 1 := by
sorry

end line_through_circle_center_l900_90048


namespace triangle_area_in_circle_l900_90001

-- Define the triangle and circle
def triangle_ratio : Vector ℝ 3 := ⟨[6, 8, 10], by simp⟩
def circle_radius : ℝ := 5

-- Theorem statement
theorem triangle_area_in_circle (sides : Vector ℝ 3) (r : ℝ) 
  (h1 : sides = triangle_ratio) 
  (h2 : r = circle_radius) : 
  ∃ (a b c : ℝ), 
    a * sides[0] = b * sides[1] ∧ 
    a * sides[0] = c * sides[2] ∧
    b * sides[1] = c * sides[2] ∧
    (a * sides[0])^2 + (b * sides[1])^2 = (c * sides[2])^2 ∧
    c * sides[2] = 2 * r ∧
    (1/2) * (a * sides[0]) * (b * sides[1]) = 24 := by
  sorry

end triangle_area_in_circle_l900_90001


namespace solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l900_90076

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for part I
theorem solution_set_f_less_than_5 :
  {x : ℝ | f x < 5} = Set.Ioo (-7/4) (3/4) :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_greater_than_abs_1_minus_a :
  {a : ℝ | ∀ x, f x > |1 - a|} = Set.Ioo (-3) 5 :=
sorry

end solution_set_f_less_than_5_range_of_a_for_f_greater_than_abs_1_minus_a_l900_90076


namespace b_plus_c_equals_six_l900_90091

theorem b_plus_c_equals_six (a b c d : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c + d = 3) 
  (h3 : a + d = 2) : 
  b + c = 6 := by
  sorry

end b_plus_c_equals_six_l900_90091


namespace cube_side_length_l900_90004

/-- Proves that given the cost of paint, coverage, and total cost to paint a cube,
    the side length of the cube is 8 feet. -/
theorem cube_side_length 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 36.50)
  (h2 : paint_coverage = 16)
  (h3 : total_cost = 876) :
  ∃ (s : ℝ), s = 8 ∧ 
  total_cost = (6 * s^2 / paint_coverage) * paint_cost :=
sorry

end cube_side_length_l900_90004


namespace randy_initial_money_l900_90069

theorem randy_initial_money :
  ∀ M : ℝ,
  (M - 10 - (M - 10) / 4 = 15) →
  M = 30 := by
sorry

end randy_initial_money_l900_90069


namespace parabola_equation_l900_90039

/-- Given a parabola y^2 = 2px where p > 0, if a point P(2, y_0) on the parabola
    has a distance of 4 from the directrix, then the equation of the parabola is y^2 = 8x -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) 
  (h3 : p/2 + 2 = 4) : 
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 8*x :=
by sorry

end parabola_equation_l900_90039


namespace no_real_solutions_l900_90031

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x*y - z^2 = 4) := by
  sorry

end no_real_solutions_l900_90031


namespace no_such_sequence_l900_90044

theorem no_such_sequence : ¬∃ (a : ℕ → ℕ),
  (∀ n > 1, a n > a (n - 1)) ∧
  (∀ m n : ℕ, a (m * n) = a m + a n) :=
sorry

end no_such_sequence_l900_90044


namespace max_expression_l900_90030

theorem max_expression (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hy : 0 < y₁ ∧ y₁ < y₂) 
  (hsum_x : x₁ + x₂ = 1) 
  (hsum_y : y₁ + y₂ = 1) : 
  x₁ * y₁ + x₂ * y₂ ≥ max (x₁ * x₂ + y₁ * y₂) (max (x₁ * y₂ + x₂ * y₁) (1/2)) :=
sorry

end max_expression_l900_90030


namespace correct_result_l900_90079

variables (a b c : ℝ)

def A : ℝ := 3 * a * b - 2 * a * c + 5 * b * c + 2 * (a * b + 2 * b * c - 4 * a * c)

theorem correct_result :
  A a b c - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end correct_result_l900_90079


namespace nickel_difference_formula_l900_90035

/-- The number of nickels equivalent to one quarter -/
def nickels_per_quarter : ℕ := 5

/-- Alice's quarters as a function of q -/
def alice_quarters (q : ℕ) : ℕ := 10 * q + 2

/-- Bob's quarters as a function of q -/
def bob_quarters (q : ℕ) : ℕ := 2 * q + 10

/-- The difference in nickels between Alice and Bob -/
def nickel_difference (q : ℕ) : ℤ :=
  (alice_quarters q - bob_quarters q) * nickels_per_quarter

theorem nickel_difference_formula (q : ℕ) :
  nickel_difference q = 40 * (q - 1) := by sorry

end nickel_difference_formula_l900_90035


namespace imaginary_part_of_reciprocal_l900_90041

theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  Complex.im (z⁻¹) = 2/5 := by
  sorry

end imaginary_part_of_reciprocal_l900_90041


namespace certain_number_proof_l900_90060

theorem certain_number_proof (y : ℝ) : 
  (0.20 * 1050 = 0.15 * y - 15) → y = 1500 := by
  sorry

end certain_number_proof_l900_90060


namespace solution_g_less_than_6_range_of_a_l900_90013

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -|x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1| + |2*x + 4|

-- Theorem for the solution of g(x) < 6
theorem solution_g_less_than_6 : 
  {x : ℝ | g x < 6} = Set.Ioo (-9/4) (3/4) := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x₁, ∃ x₂, -g x₁ = f a x₂} = Set.Ici (-5) := by sorry

end solution_g_less_than_6_range_of_a_l900_90013


namespace zane_picked_up_62_pounds_l900_90043

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_up_62_pounds : zane_garbage = 62 := by
  sorry

end zane_picked_up_62_pounds_l900_90043


namespace aftershave_dilution_l900_90094

theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 →
  initial_concentration = 0.6 →
  target_concentration = 0.4 →
  water_added = 6 →
  initial_volume * initial_concentration = 
    target_concentration * (initial_volume + water_added) :=
by
  sorry

#check aftershave_dilution

end aftershave_dilution_l900_90094


namespace train_speed_l900_90022

/-- The speed of a train given specific conditions involving a jogger --/
theorem train_speed (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 120 →
  train_length = 120 →
  passing_time = 24 →
  ∃ (train_speed : ℝ), train_speed = 36 := by
  sorry

end train_speed_l900_90022


namespace monotonic_decreasing_interval_l900_90023

def f (x : ℝ) := x^3 - 3*x^2

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (∀ y : ℝ, x < y → f x > f y) ↔ x ∈ Set.Ioo 0 2 := by
  sorry

end monotonic_decreasing_interval_l900_90023


namespace bake_sale_pastries_sold_l900_90029

/-- Represents the number of pastries sold at a bake sale. -/
def pastries_sold (cupcakes cookies taken_home : ℕ) : ℕ :=
  cupcakes + cookies - taken_home

/-- Proves that the number of pastries sold is correct given the conditions. -/
theorem bake_sale_pastries_sold :
  pastries_sold 4 29 24 = 9 := by
  sorry

end bake_sale_pastries_sold_l900_90029


namespace least_integer_absolute_value_l900_90014

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2 * y + 3| ≤ 12)) ∧ (|2 * x + 3| ≤ 12) → x = -7 := by
  sorry

end least_integer_absolute_value_l900_90014


namespace system_equation_solution_l900_90067

theorem system_equation_solution (x y some_number : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = 5) → 
  (2 * x * y / some_number = 2) →
  some_number = 3 := by
  sorry

end system_equation_solution_l900_90067


namespace rectangular_prism_sum_l900_90059

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define any specific properties here, as we're only interested in the general structure

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

theorem rectangular_prism_sum (rp : RectangularPrism) : 
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end rectangular_prism_sum_l900_90059


namespace max_sphere_radius_in_intersecting_cones_l900_90074

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem stating the maximum sphere radius for the given configuration -/
theorem max_sphere_radius_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 5, height := 12 },
    cone2 := { baseRadius := 5, height := 12 },
    intersectionDistance := 4
  }
  maxSphereRadius ic = 40 / 13 := by sorry

end max_sphere_radius_in_intersecting_cones_l900_90074


namespace factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l900_90019

-- Statement 1
theorem factorization_3x2_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

-- Statement 2
theorem factorization_ax2_4axy_4ay2 (a x y : ℝ) : a * x^2 - 4 * a * x * y + 4 * a * y^2 = a * (x - 2 * y)^2 := by
  sorry

end factorization_3x2_minus_12_factorization_ax2_4axy_4ay2_l900_90019


namespace quad_pyramid_volume_l900_90007

noncomputable section

/-- A quadrilateral pyramid with a square base -/
structure QuadPyramid where
  /-- Side length of the square base -/
  a : ℝ
  /-- Dihedral angle at edge SA -/
  α : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The dihedral angle is within the valid range -/
  α_range : π / 2 < α ∧ α ≤ 2 * π / 3
  /-- Angles between opposite lateral faces are right angles -/
  opposite_faces_right : True

/-- Volume of the quadrilateral pyramid -/
def volume (p : QuadPyramid) : ℝ := (p.a ^ 3 * |Real.cos p.α|) / 3

/-- Theorem stating the volume of the quadrilateral pyramid -/
theorem quad_pyramid_volume (p : QuadPyramid) : 
  volume p = (p.a ^ 3 * |Real.cos p.α|) / 3 := by sorry

end

end quad_pyramid_volume_l900_90007


namespace f_min_value_l900_90086

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end f_min_value_l900_90086


namespace stratified_sampling_survey_l900_90016

theorem stratified_sampling_survey (young_population middle_aged_population elderly_population : ℕ)
  (elderly_sampled : ℕ) (young_sampled : ℕ) :
  young_population = 800 →
  middle_aged_population = 1600 →
  elderly_population = 1400 →
  elderly_sampled = 70 →
  (elderly_sampled : ℚ) / elderly_population = (young_sampled : ℚ) / young_population →
  young_sampled = 40 :=
by sorry

end stratified_sampling_survey_l900_90016


namespace jills_net_salary_l900_90090

/-- Calculates the net monthly salary given the discretionary income percentage and the amount left after allocations -/
def calculate_net_salary (discretionary_income_percentage : ℚ) (amount_left : ℚ) : ℚ :=
  (amount_left / (discretionary_income_percentage * (1 - 0.3 - 0.2 - 0.35))) * 100

/-- Proves that under the given conditions, Jill's net monthly salary is $3700 -/
theorem jills_net_salary :
  let discretionary_income_percentage : ℚ := 1/5
  let amount_left : ℚ := 111
  calculate_net_salary discretionary_income_percentage amount_left = 3700 := by
  sorry

#eval calculate_net_salary (1/5) 111

end jills_net_salary_l900_90090


namespace initial_cabinets_l900_90080

theorem initial_cabinets (total : ℕ) (additional : ℕ) (counters : ℕ) : 
  total = 26 → 
  additional = 5 → 
  counters = 3 → 
  ∃ initial : ℕ, initial + counters * (2 * initial) + additional = total ∧ initial = 3 := by
  sorry

end initial_cabinets_l900_90080


namespace sum_of_coefficients_eq_64_l900_90050

/-- The sum of the numerical coefficients in the complete expansion of (x^2 - 3xy + y^2)^6 -/
def sum_of_coefficients : ℕ :=
  (1 - 3)^6

theorem sum_of_coefficients_eq_64 : sum_of_coefficients = 64 := by
  sorry

#eval sum_of_coefficients

end sum_of_coefficients_eq_64_l900_90050


namespace paige_homework_problems_l900_90026

/-- The number of math problems Paige had for homework -/
def math_problems : ℕ := 43

/-- The number of science problems Paige had for homework -/
def science_problems : ℕ := 12

/-- The number of problems Paige finished at school -/
def finished_problems : ℕ := 44

/-- The number of problems Paige had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_problems

theorem paige_homework_problems :
  homework_problems = 11 := by sorry

end paige_homework_problems_l900_90026


namespace inequality_solution_range_of_a_l900_90075

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, ∃ y, Real.log y = f x + a} = {a : ℝ | a > -2} := by sorry

end inequality_solution_range_of_a_l900_90075


namespace total_pencils_l900_90025

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end total_pencils_l900_90025


namespace miranda_pillows_l900_90057

/-- Calculates the number of pillows Miranda can stuff given the following conditions:
  * Each pillow needs 2 pounds of feathers
  * 1 pound of goose feathers is approximately 300 feathers
  * Miranda's goose has approximately 3600 feathers
-/
def pillows_from_goose (feathers_per_pillow : ℕ) (feathers_per_pound : ℕ) (goose_feathers : ℕ) : ℕ :=
  (goose_feathers / feathers_per_pound) / feathers_per_pillow

/-- Proves that Miranda can stuff 6 pillows given the conditions -/
theorem miranda_pillows : 
  pillows_from_goose 2 300 3600 = 6 := by
  sorry

end miranda_pillows_l900_90057


namespace final_selling_price_l900_90078

/-- Calculate the final selling price of items with given conditions -/
theorem final_selling_price :
  let cycle_price : ℚ := 1400
  let helmet_price : ℚ := 400
  let safety_light_price : ℚ := 200
  let cycle_discount : ℚ := 0.1
  let helmet_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.05
  let cycle_loss : ℚ := 0.12
  let helmet_profit : ℚ := 0.25
  let lock_price : ℚ := 300
  let transaction_fee : ℚ := 0.03

  let discounted_cycle := cycle_price * (1 - cycle_discount)
  let discounted_helmet := helmet_price * (1 - helmet_discount)
  let total_safety_lights := 2 * safety_light_price

  let total_before_tax := discounted_cycle + discounted_helmet + total_safety_lights
  let total_after_tax := total_before_tax * (1 + tax_rate)

  let selling_cycle := discounted_cycle * (1 - cycle_loss)
  let selling_helmet := discounted_helmet * (1 + helmet_profit)
  let selling_safety_lights := total_safety_lights

  let total_selling_before_fee := selling_cycle + selling_helmet + selling_safety_lights + lock_price
  let fee_amount := total_selling_before_fee * transaction_fee
  let total_selling_after_fee := total_selling_before_fee - fee_amount

  let final_price := ⌊total_selling_after_fee⌋

  final_price = 2215 := by sorry

end final_selling_price_l900_90078


namespace total_students_in_schools_l900_90083

theorem total_students_in_schools (capacity1 capacity2 : ℕ) 
  (h1 : capacity1 = 400) 
  (h2 : capacity2 = 340) : 
  2 * capacity1 + 2 * capacity2 = 1480 :=
by sorry

end total_students_in_schools_l900_90083


namespace derivative_f_at_1_l900_90021

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2*x)^10

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 20 := by sorry

end derivative_f_at_1_l900_90021


namespace family_reunion_attendance_l900_90089

/-- Calculates the number of people served given the amount of pasta used,
    based on a recipe where 2 pounds of pasta serves 7 people. -/
def people_served (pasta_pounds : ℚ) : ℚ :=
  (pasta_pounds / 2) * 7

/-- Theorem stating that 10 pounds of pasta will serve 35 people,
    given a recipe where 2 pounds of pasta serves 7 people. -/
theorem family_reunion_attendance :
  people_served 10 = 35 := by
  sorry

end family_reunion_attendance_l900_90089


namespace union_and_subset_conditions_l900_90028

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_and_subset_conditions :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ 3) := by sorry

end union_and_subset_conditions_l900_90028


namespace sum_of_three_consecutive_integers_l900_90066

theorem sum_of_three_consecutive_integers (a b c : ℤ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 14 → a + b + c = 39 := by
  sorry

end sum_of_three_consecutive_integers_l900_90066


namespace solution_to_system_l900_90082

theorem solution_to_system : ∃ (x y : ℚ), 
  (7 * x - 50 * y = 3) ∧ (3 * y - x = 5) ∧ 
  (x = -259/29) ∧ (y = -38/29) := by
  sorry

end solution_to_system_l900_90082


namespace tip_is_24_dollars_l900_90020

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℚ := 48

/-- The cost of a child's haircut in dollars -/
def childrens_haircut_cost : ℚ := 36

/-- The number of women getting haircuts -/
def num_women : ℕ := 1

/-- The number of children getting haircuts -/
def num_children : ℕ := 2

/-- The tip percentage as a decimal -/
def tip_percentage : ℚ := 0.20

/-- The total cost of haircuts before tip -/
def total_cost : ℚ := womens_haircut_cost * num_women + childrens_haircut_cost * num_children

/-- The tip amount in dollars -/
def tip_amount : ℚ := total_cost * tip_percentage

theorem tip_is_24_dollars : tip_amount = 24 := by
  sorry

end tip_is_24_dollars_l900_90020


namespace geometric_sequence_first_term_l900_90084

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 18) -- second term is 18
  (h2 : a * r^2 = 24) -- third term is 24
  : a = 27/2 := by
sorry

end geometric_sequence_first_term_l900_90084


namespace smallest_cut_length_l900_90027

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 13 →
  (∀ y : ℕ, y > 0 ∧ y ≤ 13 → (13 - y) + (20 - y) ≤ 25 - y → y ≥ x) →
  (13 - x) + (20 - x) ≤ 25 - x →
  x = 8 := by
  sorry

end smallest_cut_length_l900_90027


namespace fraction_upper_bound_l900_90062

theorem fraction_upper_bound (x : ℝ) (h : x > 0) : x / (x^2 + 3*x + 1) ≤ 1/5 := by
  sorry

end fraction_upper_bound_l900_90062


namespace sams_money_l900_90034

/-- Given that Sam and Erica have $91 together and Erica has $53, 
    prove that Sam has $38. -/
theorem sams_money (total : ℕ) (ericas_money : ℕ) (sams_money : ℕ) : 
  total = 91 → ericas_money = 53 → sams_money = total - ericas_money → sams_money = 38 := by
  sorry

end sams_money_l900_90034


namespace chord_length_theorem_l900_90009

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 25

-- Theorem statement
theorem chord_length_theorem :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), line_equation x y → circle_equation x y → 
      chord_length = 4 * Real.sqrt 5) :=
sorry

end chord_length_theorem_l900_90009


namespace quadratic_expression_values_l900_90047

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3)
  (hn : |n| = 2)
  (hmn : m < n) :
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end quadratic_expression_values_l900_90047


namespace fraction_inequality_solution_set_l900_90000

theorem fraction_inequality_solution_set : 
  {x : ℝ | x / (x + 1) < 0} = Set.Ioo (-1) 0 := by sorry

end fraction_inequality_solution_set_l900_90000


namespace tom_total_weight_l900_90088

/-- Calculates the total weight Tom is moving with given his body weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight (tom_weight : ℝ) (hand_multiplier : ℝ) (vest_multiplier : ℝ) : ℝ :=
  tom_weight * hand_multiplier * 2 + tom_weight * vest_multiplier

/-- Theorem stating that Tom's total weight moved is 525 kg given the problem conditions. -/
theorem tom_total_weight :
  let tom_weight : ℝ := 150
  let hand_multiplier : ℝ := 1.5
  let vest_multiplier : ℝ := 0.5
  total_weight tom_weight hand_multiplier vest_multiplier = 525 := by
  sorry

end tom_total_weight_l900_90088


namespace percentage_difference_l900_90096

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.25)) :
  y = x * (1 + 0.25) := by
  sorry

end percentage_difference_l900_90096


namespace probability_theorem_l900_90071

/-- Represents a brother with a name of a certain length -/
structure Brother where
  name : String
  name_length : Nat

/-- Represents the problem setup -/
structure LetterCardProblem where
  adam : Brother
  brian : Brother
  total_letters : Nat
  (total_is_sum : total_letters = adam.name_length + brian.name_length)
  (total_is_twelve : total_letters = 12)

/-- The probability of selecting one letter from each brother's name -/
def probability_one_from_each (problem : LetterCardProblem) : Rat :=
  4 / 11

theorem probability_theorem (problem : LetterCardProblem) :
  probability_one_from_each problem = 4 / 11 := by
  sorry

end probability_theorem_l900_90071


namespace xy_neq_one_condition_l900_90092

theorem xy_neq_one_condition (x y : ℝ) :
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 1) ∧ x * y = 1) ∧
  (x * y ≠ 1 → (x ≠ 1 ∨ y ≠ 1)) :=
by sorry

end xy_neq_one_condition_l900_90092


namespace solve_sock_problem_l900_90002

def sock_problem (initial_pairs : ℕ) (lost_pairs : ℕ) (purchased_pairs : ℕ) (gifted_pairs : ℕ) (final_pairs : ℕ) : Prop :=
  let remaining_pairs := initial_pairs - lost_pairs
  ∃ (donated_fraction : ℚ),
    0 ≤ donated_fraction ∧
    donated_fraction ≤ 1 ∧
    remaining_pairs * (1 - donated_fraction) + purchased_pairs + gifted_pairs = final_pairs ∧
    donated_fraction = 2/3

theorem solve_sock_problem :
  sock_problem 40 4 10 3 25 :=
sorry

end solve_sock_problem_l900_90002


namespace escalator_standing_time_l900_90099

/-- Represents the time it takes Clea to ride down an escalator under different conditions -/
def escalator_time (non_operating_time walking_time standing_time : ℝ) : Prop :=
  -- Distance of the escalator
  ∃ d : ℝ,
  -- Speed of Clea walking down the escalator
  ∃ c : ℝ,
  -- Speed of the escalator
  ∃ s : ℝ,
  -- Conditions
  (d = 70 * c) ∧  -- Time to walk down non-operating escalator
  (d = 28 * (c + s)) ∧  -- Time to walk down operating escalator
  (standing_time = d / s) ∧  -- Time to stand on operating escalator
  (standing_time = 47)  -- The result we want to prove

/-- Theorem stating that given the conditions, the standing time on the operating escalator is 47 seconds -/
theorem escalator_standing_time :
  escalator_time 70 28 47 :=
sorry

end escalator_standing_time_l900_90099


namespace sports_lottery_combinations_and_cost_l900_90038

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def consecutive_three : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def consecutive_two : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def single_number : ℕ := 7

/-- The cost of each bet in yuan -/
def bet_cost : ℕ := 2

/-- The total number of combinations -/
def total_combinations : ℕ := consecutive_three * consecutive_two * single_number

/-- The total cost in yuan -/
def total_cost : ℕ := total_combinations * bet_cost

theorem sports_lottery_combinations_and_cost :
  total_combinations = 1050 ∧ total_cost = 2100 := by
  sorry

end sports_lottery_combinations_and_cost_l900_90038


namespace b_51_equals_5151_l900_90012

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def is_not_even (n : ℕ) : Prop := ¬(2 ∣ n)

def b : ℕ → ℕ := sorry

theorem b_51_equals_5151 : b 51 = 5151 := by sorry

end b_51_equals_5151_l900_90012


namespace quadrilateral_area_in_regular_octagon_l900_90006

-- Define a regular octagon
structure RegularOctagon :=
  (side_length : ℝ)

-- Define the area of a quadrilateral formed by two adjacent vertices and two diagonal intersections
def quadrilateral_area (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem quadrilateral_area_in_regular_octagon 
  (octagon : RegularOctagon) 
  (h : octagon.side_length = 5) : 
  quadrilateral_area octagon = 25 * Real.sqrt 2 := by
  sorry

end quadrilateral_area_in_regular_octagon_l900_90006


namespace proportion_of_dogs_l900_90085

theorem proportion_of_dogs (C G : ℝ) 
  (h1 : 0.8 * G + 0.25 * C = 0.3 * (G + C)) 
  (h2 : C > 0) 
  (h3 : G > 0) : 
  C / (C + G) = 10 / 11 := by
  sorry

end proportion_of_dogs_l900_90085


namespace jordans_initial_weight_jordans_weight_proof_l900_90046

/-- Calculates Jordan's initial weight based on his weight loss program and current weight -/
theorem jordans_initial_weight (initial_loss_rate : ℕ) (initial_weeks : ℕ) 
  (subsequent_loss_rate : ℕ) (subsequent_weeks : ℕ) (current_weight : ℕ) : ℕ :=
  let total_loss := initial_loss_rate * initial_weeks + subsequent_loss_rate * subsequent_weeks
  current_weight + total_loss

/-- Proves that Jordan's initial weight was 250 pounds -/
theorem jordans_weight_proof : 
  jordans_initial_weight 3 4 2 8 222 = 250 := by
  sorry

end jordans_initial_weight_jordans_weight_proof_l900_90046


namespace other_bill_denomination_l900_90018

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of $10 bills used
def num_ten_bills : ℕ := 2

-- Define the function to calculate the number of other bills
def num_other_bills (n : ℕ) : ℕ := n + 1

-- Define the theorem
theorem other_bill_denomination :
  ∃ (x : ℕ), 
    x * num_other_bills num_ten_bills + 10 * num_ten_bills = total_spent ∧
    x = 20 := by
  sorry

end other_bill_denomination_l900_90018


namespace at_least_one_not_less_than_two_l900_90058

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l900_90058


namespace ellipse_geometric_sequence_l900_90087

/-- Given an ellipse E with equation x²/a² + y²/b² = 1 (a > b > 0),
    eccentricity e = √2/2, and left vertex at (-2,0),
    prove that for points B and C on E, where AB is parallel to OC
    and AB intersects the y-axis at D, |AB|, √2|OC|, and |AD|
    form a geometric sequence. -/
theorem ellipse_geometric_sequence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (E : Set (ℝ × ℝ))
  (hE : E = {(x, y) | x^2/a^2 + y^2/b^2 = 1})
  (he : (a^2 - b^2)/a^2 = 1/2)
  (hA : (-2, 0) ∈ E)
  (B C : ℝ × ℝ) (hB : B ∈ E) (hC : C ∈ E)
  (hparallel : ∃ (k : ℝ), (B.2 + 2*k = B.1 ∧ C.2 = k*C.1))
  (D : ℝ × ℝ) (hD : D.1 = 0 ∧ D.2 = B.2 - B.1/2*B.2) :
  ∃ (r : ℝ), abs (B.1 - (-2)) * abs (D.2) = r * (abs (C.1) * abs (C.1) + abs (C.2) * abs (C.2))
    ∧ abs (D.2)^2 = r * (abs (B.1 - (-2)) * abs (D.2)) :=
by sorry

end ellipse_geometric_sequence_l900_90087


namespace angle_QRS_is_150_degrees_l900_90005

/-- A quadrilateral PQRS with specific side lengths and angles -/
structure Quadrilateral :=
  (PQ : ℝ)
  (RS : ℝ)
  (PS : ℝ)
  (angle_QPS : ℝ)
  (angle_RSP : ℝ)

/-- The theorem stating the condition for ∠QRS in the given quadrilateral -/
theorem angle_QRS_is_150_degrees (q : Quadrilateral) 
  (h1 : q.PQ = 40)
  (h2 : q.RS = 20)
  (h3 : q.PS = 60)
  (h4 : q.angle_QPS = 60)
  (h5 : q.angle_RSP = 60) :
  ∃ (angle_QRS : ℝ), angle_QRS = 150 := by
  sorry

end angle_QRS_is_150_degrees_l900_90005


namespace simplify_fraction_integer_decimal_parts_l900_90053

-- Part 1
theorem simplify_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  2 / (Real.sqrt x + Real.sqrt y) = Real.sqrt x - Real.sqrt y :=
sorry

-- Part 2
theorem integer_decimal_parts (a : ℤ) (b : ℝ) 
  (h : 1 / (2 - Real.sqrt 3) = ↑a + b) (h_b : 0 ≤ b ∧ b < 1) :
  (a : ℝ)^2 + b^2 = 13 - 2 * Real.sqrt 3 :=
sorry

end simplify_fraction_integer_decimal_parts_l900_90053


namespace largest_6k_plus_1_factor_of_11_factorial_l900_90037

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def is_of_form_6k_plus_1 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 1

theorem largest_6k_plus_1_factor_of_11_factorial :
  ∀ n : ℕ, is_factor n (factorial 11) → is_of_form_6k_plus_1 n → n ≤ 385 :=
by sorry

end largest_6k_plus_1_factor_of_11_factorial_l900_90037


namespace man_and_son_work_time_l900_90063

/-- The time taken for a man and his son to complete a task together, given their individual completion times -/
theorem man_and_son_work_time (man_time son_time : ℝ) (h1 : man_time = 5) (h2 : son_time = 20) :
  1 / (1 / man_time + 1 / son_time) = 4 := by
  sorry

end man_and_son_work_time_l900_90063
