import Mathlib

namespace sphere_surface_area_and_volume_l1845_184570

/-- Given a sphere with diameter 18 inches, prove its surface area and volume -/
theorem sphere_surface_area_and_volume :
  let diameter : ℝ := 18
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  let volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  surface_area = 324 * Real.pi ∧ volume = 972 * Real.pi := by
  sorry


end sphere_surface_area_and_volume_l1845_184570


namespace no_perfect_squares_with_conditions_l1845_184533

theorem no_perfect_squares_with_conditions : 
  ¬∃ (n : ℕ), 
    n^2 < 20000 ∧ 
    4 ∣ n^2 ∧ 
    ∃ (k : ℕ), n^2 = (k + 1)^2 - k^2 :=
by sorry

end no_perfect_squares_with_conditions_l1845_184533


namespace power_of_two_sum_l1845_184563

theorem power_of_two_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := by
  sorry

end power_of_two_sum_l1845_184563


namespace diana_wins_probability_l1845_184580

/-- The number of sides on Apollo's die -/
def apollo_sides : ℕ := 8

/-- The number of sides on Diana's die -/
def diana_sides : ℕ := 5

/-- The probability that Diana's roll is larger than Apollo's roll -/
def probability_diana_wins : ℚ := 1/4

/-- Theorem stating that the probability of Diana winning is 1/4 -/
theorem diana_wins_probability : 
  probability_diana_wins = 1/4 := by sorry

end diana_wins_probability_l1845_184580


namespace complex_fraction_equals_neg_i_l1845_184578

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_fraction_equals_neg_i : (1 - i) / (1 + i) = -i := by sorry

end complex_fraction_equals_neg_i_l1845_184578


namespace direction_vector_form_l1845_184508

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p₁ p₂ : ℝ × ℝ) (b : ℝ) : 
  p₁ = (-3, 4) →
  p₂ = (2, -1) →
  (p₂.1 - p₁.1, p₂.2 - p₁.2) = (b * (p₂.2 - p₁.2), p₂.2 - p₁.2) →
  b = 1 := by
  sorry

#check direction_vector_form

end direction_vector_form_l1845_184508


namespace doll_distribution_theorem_l1845_184568

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_pair := n_dolls.choose 2
  let choose_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_pair * choose_house * arrange_rest

theorem doll_distribution_theorem :
  distribute_dolls 7 6 = 15120 :=
sorry

end doll_distribution_theorem_l1845_184568


namespace min_value_of_expression_l1845_184537

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ y : ℝ, y = 1/a + 4/b → y ≥ 9/2 :=
by sorry

end min_value_of_expression_l1845_184537


namespace fourth_term_value_l1845_184576

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem fourth_term_value : a 4 = 11 := by
  sorry

end fourth_term_value_l1845_184576


namespace car_travel_distance_l1845_184584

/-- Given a car that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas. -/
theorem car_travel_distance (miles : ℝ) (gallons : ℝ) 
  (h1 : miles = 300) (h2 : gallons = 10) :
  (miles / gallons) * 15 = 450 := by
  sorry

end car_travel_distance_l1845_184584


namespace parallelogram_base_length_l1845_184573

/-- Given a parallelogram with area 44 cm² and height 11 cm, its base length is 4 cm. -/
theorem parallelogram_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
    (h1 : area = 44) 
    (h2 : height = 11) 
    (h3 : area = base * height) : base = 4 := by
  sorry

end parallelogram_base_length_l1845_184573


namespace three_solutions_cubic_l1845_184597

theorem three_solutions_cubic (n : ℕ+) (x y : ℤ) 
  (h : x^3 - 3*x*y^2 + y^3 = n) : 
  ∃ (a b c d e f : ℤ), 
    (a^3 - 3*a*b^2 + b^3 = n) ∧ 
    (c^3 - 3*c*d^2 + d^3 = n) ∧ 
    (e^3 - 3*e*f^2 + f^3 = n) ∧ 
    (a ≠ c ∨ b ≠ d) ∧ 
    (a ≠ e ∨ b ≠ f) ∧ 
    (c ≠ e ∨ d ≠ f) := by
  sorry

end three_solutions_cubic_l1845_184597


namespace range_of_m_l1845_184571

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 15 / 2) * Real.sin (Real.pi * x)

theorem range_of_m (x₀ : ℝ) (h₁ : x₀ ∈ Set.Ioo (-1) 1)
  (h₂ : ∀ x : ℝ, f x ≤ f x₀)
  (h₃ : ∃ m : ℝ, x₀^2 + (f x₀)^2 < m^2) :
  ∃ m : ℝ, m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end range_of_m_l1845_184571


namespace tic_tac_toe_winnings_l1845_184588

theorem tic_tac_toe_winnings
  (total_games : ℕ)
  (tied_games : ℕ)
  (net_loss : ℤ)
  (h_total : total_games = 100)
  (h_tied : tied_games = 40)
  (h_loss : net_loss = 30)
  (h_win_value : ℤ)
  (h_tie_value : ℤ)
  (h_lose_value : ℤ)
  (h_win_val : h_win_value = 1)
  (h_tie_val : h_tie_value = 0)
  (h_lose_val : h_lose_value = -2)
  : ∃ (won_games : ℕ),
    won_games = 30 ∧
    won_games + tied_games + (total_games - won_games - tied_games) = total_games ∧
    h_win_value * won_games + h_tie_value * tied_games + h_lose_value * (total_games - won_games - tied_games) = -net_loss :=
by sorry

end tic_tac_toe_winnings_l1845_184588


namespace ellipse_product_l1845_184589

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- An ellipse defined by its center, major axis, minor axis, and a focus -/
structure Ellipse :=
  (center : Point)
  (majorAxis : ℝ)
  (minorAxis : ℝ)
  (focus : Point)

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Diameter of the incircle of a right triangle -/
def incircleDiameter (leg1 leg2 hypotenuse : ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_product (e : Ellipse) :
  distance e.center e.focus = 8 →
  incircleDiameter e.minorAxis 8 e.majorAxis = 4 →
  e.majorAxis * e.minorAxis = 240 := by sorry

end ellipse_product_l1845_184589


namespace largest_valid_number_l1845_184558

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number :
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end largest_valid_number_l1845_184558


namespace min_red_chips_l1845_184552

/-- Represents a box of colored chips -/
structure ChipBox where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if a ChipBox satisfies the given conditions -/
def isValidChipBox (box : ChipBox) : Prop :=
  box.blue ≥ box.white / 2 ∧
  box.blue ≤ box.red / 3 ∧
  box.white + box.blue ≥ 55

/-- The theorem stating the minimum number of red chips -/
theorem min_red_chips (box : ChipBox) :
  isValidChipBox box → box.red ≥ 57 := by
  sorry

end min_red_chips_l1845_184552


namespace dogwood_tree_count_l1845_184512

/-- The total number of dogwood trees after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem dogwood_tree_count :
  total_trees 39 41 20 = 100 := by
  sorry

end dogwood_tree_count_l1845_184512


namespace a_greater_than_zero_when_a_greater_than_b_l1845_184583

theorem a_greater_than_zero_when_a_greater_than_b (a b : ℝ) 
  (h1 : a^2 > b^2) (h2 : a > b) : a > 0 := by
  sorry

end a_greater_than_zero_when_a_greater_than_b_l1845_184583


namespace oranges_picked_sum_l1845_184551

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_sum :
  total_oranges = 55 := by sorry

end oranges_picked_sum_l1845_184551


namespace no_linear_term_implies_a_eq_neg_two_l1845_184594

/-- If the simplified result of (3x+2)(3x+a) does not contain a linear term of x, then a = -2 -/
theorem no_linear_term_implies_a_eq_neg_two (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (3*x + 2) * (3*x + a) = b*x^2 + c) → a = -2 := by
sorry

end no_linear_term_implies_a_eq_neg_two_l1845_184594


namespace expand_expression_l1845_184520

theorem expand_expression (x : ℝ) : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 := by
  sorry

end expand_expression_l1845_184520


namespace extra_marks_for_second_candidate_l1845_184559

/-- The total number of marks in the exam -/
def T : ℝ := 300

/-- The passing marks -/
def P : ℝ := 120

/-- The percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the second candidate -/
def second_candidate_percentage : ℝ := 0.45

/-- The number of marks by which the first candidate fails -/
def failing_margin : ℝ := 30

theorem extra_marks_for_second_candidate : 
  second_candidate_percentage * T - P = 15 := by sorry

end extra_marks_for_second_candidate_l1845_184559


namespace quadratic_factorization_l1845_184501

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 74 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end quadratic_factorization_l1845_184501


namespace total_is_255_l1845_184540

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a b c : ℕ)

/-- Calculates the total amount of money given a ratio and the first person's share -/
def totalAmount (ratio : MoneyRatio) (firstShare : ℕ) : ℕ :=
  let multiplier := firstShare / ratio.a
  multiplier * (ratio.a + ratio.b + ratio.c)

/-- Theorem stating that for the given ratio and first share, the total amount is 255 -/
theorem total_is_255 (ratio : MoneyRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 5) (h3 : ratio.c = 9) 
    (h4 : totalAmount ratio 45 = 255) : totalAmount ratio 45 = 255 := by
  sorry

end total_is_255_l1845_184540


namespace upstream_speed_is_27_l1845_184502

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the upstream speed given downstream and still water speeds -/
def calculateUpstreamSpeed (downstream stillWater : ℝ) : ℝ :=
  2 * stillWater - downstream

/-- Theorem stating that given the conditions, the upstream speed is 27 kmph -/
theorem upstream_speed_is_27 (speed : RowingSpeed) 
    (h1 : speed.downstream = 35)
    (h2 : speed.stillWater = 31) :
    speed.upstream = 27 :=
  by sorry

end upstream_speed_is_27_l1845_184502


namespace chinese_sexagenary_cycle_properties_l1845_184538

/-- Represents the Chinese sexagenary cycle -/
structure SexagenaryCycle where
  heavenly_stems : Fin 10
  earthly_branches : Fin 12

/-- Calculates the next year with the same combination in the sexagenary cycle -/
def next_same_year (year : Int) : Int :=
  year + 60

/-- Calculates the previous year with the same combination in the sexagenary cycle -/
def prev_same_year (year : Int) : Int :=
  year - 60

/-- Calculates a year with a specific offset in the cycle -/
def year_with_offset (base_year : Int) (offset : Int) : Int :=
  base_year + offset

theorem chinese_sexagenary_cycle_properties :
  let ren_wu_2002 : SexagenaryCycle := ⟨9, 7⟩ -- Ren (9th stem), Wu (7th branch)
  -- 1. Next Ren Wu year
  (next_same_year 2002 = 2062) ∧
  -- 2. Jiawu War year (Jia Wu)
  (year_with_offset 2002 (-108) = 1894) ∧
  -- 3. Wuxu Reform year (Wu Xu)
  (year_with_offset 2002 (-104) = 1898) ∧
  -- 4. Geng Shen years in the 20th century
  (year_with_offset 2002 (-82) = 1920) ∧
  (year_with_offset 2002 (-22) = 1980) := by
  sorry

end chinese_sexagenary_cycle_properties_l1845_184538


namespace inequality_proof_l1845_184525

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end inequality_proof_l1845_184525


namespace cycle_selling_price_l1845_184596

/-- The selling price of a cycle after applying successive discounts -/
def selling_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem: The selling price of a cycle originally priced at Rs. 3,600, 
    after applying successive discounts of 15%, 10%, and 5%, is equal to Rs. 2,616.30 -/
theorem cycle_selling_price :
  selling_price 3600 0.15 0.10 0.05 = 2616.30 := by
  sorry

end cycle_selling_price_l1845_184596


namespace parabola_directrix_l1845_184595

/-- Given a parabola with equation y = ax² and directrix y = -1, prove that a = 1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = -1/4/k ∧ k = a)) → 
  a = 1/4 := by
sorry

end parabola_directrix_l1845_184595


namespace sequence_sum_property_l1845_184530

theorem sequence_sum_property (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = n^2 + n) →
  (∀ n : ℕ, a n = 2 * n) :=
by sorry

end sequence_sum_property_l1845_184530


namespace employee_earnings_theorem_l1845_184587

/-- Calculates the total earnings for an employee based on their work schedule and pay rates -/
def calculate_earnings (task_a_rate : ℚ) (task_b_rate : ℚ) (overtime_multiplier : ℚ) 
                       (commission_rate : ℚ) (task_a_hours : List ℚ) (task_b_hours : List ℚ) : ℚ :=
  let task_a_total_hours := task_a_hours.sum
  let task_b_total_hours := task_b_hours.sum
  let task_a_regular_hours := min task_a_total_hours 40
  let task_a_overtime_hours := max (task_a_total_hours - 40) 0
  let task_a_earnings := task_a_regular_hours * task_a_rate + 
                         task_a_overtime_hours * task_a_rate * overtime_multiplier
  let task_b_earnings := task_b_total_hours * task_b_rate
  let total_before_commission := task_a_earnings + task_b_earnings
  let commission := if task_b_total_hours ≥ 10 then total_before_commission * commission_rate else 0
  total_before_commission + commission

/-- Theorem stating that the employee's earnings for the given work schedule and pay rates equal $2211 -/
theorem employee_earnings_theorem :
  let task_a_rate : ℚ := 30
  let task_b_rate : ℚ := 40
  let overtime_multiplier : ℚ := 1.5
  let commission_rate : ℚ := 0.1
  let task_a_hours : List ℚ := [6, 6, 6, 12, 12]
  let task_b_hours : List ℚ := [4, 4, 4, 3, 3]
  calculate_earnings task_a_rate task_b_rate overtime_multiplier commission_rate task_a_hours task_b_hours = 2211 := by
  sorry

end employee_earnings_theorem_l1845_184587


namespace weight_difference_l1845_184557

-- Define the weights as natural numbers
def sam_weight : ℕ := 105
def peter_weight : ℕ := 65

-- Define Tyler's weight based on Peter's weight
def tyler_weight : ℕ := 2 * peter_weight

-- Theorem to prove
theorem weight_difference : tyler_weight - sam_weight = 25 := by
  sorry

end weight_difference_l1845_184557


namespace solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l1845_184507

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Solution set when a = -1
theorem solution_set_when_a_neg_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 3 for all x
theorem range_of_a_when_always_ge_three :
  {a : ℝ | ∀ x, f a x ≥ 3} = {a : ℝ | a ≤ -2 ∨ a ≥ 4} := by sorry

end solution_set_when_a_neg_one_range_of_a_when_always_ge_three_l1845_184507


namespace arithmetic_sequence_common_difference_l1845_184586

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_a2 : a 2 = 9)  -- Given: a_2 = 9
  (h_a5 : a 5 = 33)  -- Given: a_5 = 33
  : a 2 - a 1 = 8 :=  -- Conclusion: The common difference is 8
by sorry

end arithmetic_sequence_common_difference_l1845_184586


namespace juggling_show_balls_l1845_184562

/-- The number of balls needed for a juggling show -/
def balls_needed (jugglers : ℕ) (balls_per_juggler : ℕ) : ℕ :=
  jugglers * balls_per_juggler

/-- Theorem: 378 jugglers each juggling 6 balls require 2268 balls in total -/
theorem juggling_show_balls : balls_needed 378 6 = 2268 := by
  sorry

end juggling_show_balls_l1845_184562


namespace area_union_rotated_triangle_l1845_184599

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The centroid of a triangle -/
def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Rotation of a point around another point by 180 degrees -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The union of two regions -/
def unionArea (area1 : ℝ) (area2 : ℝ) : ℝ := sorry

theorem area_union_rotated_triangle (t : Triangle) :
  let m := t.centroid
  let t' := Triangle.mk t.a t.b t.c t.h_positive
  unionArea t.area t'.area = t.area := by sorry

end area_union_rotated_triangle_l1845_184599


namespace quadratic_inequality_l1845_184503

theorem quadratic_inequality (x : ℝ) : x^2 - 7*x + 6 < 0 ↔ 1 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l1845_184503


namespace range_of_m_l1845_184521

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 → x^2 / (a^2) + y^2 / (b^2) = 1 ∧ a > b

def q (m : ℝ) : Prop := ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ ∀ (x y : ℝ), y^2 / 5 - x^2 / m = 1 → x^2 / (5*e^2) - y^2 / 5 = 1

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(p m) ∧ ¬(q m)) ∧ (p m ∨ q m) → 1/3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l1845_184521


namespace shaded_area_of_circle_with_perpendicular_diameters_l1845_184528

theorem shaded_area_of_circle_with_perpendicular_diameters (r : ℝ) (h : r = 4) :
  let circle_area := π * r^2
  let quarter_circle_area := circle_area / 4
  let right_triangle_area := r^2 / 2
  let shaded_area := 2 * quarter_circle_area + 2 * right_triangle_area
  shaded_area = 16 + 8 * π := by
  sorry

end shaded_area_of_circle_with_perpendicular_diameters_l1845_184528


namespace ABC_equality_l1845_184514

variables (u v w : ℝ)
variables (A B C : ℝ)

def A_def : A = u * v + u + 1 := by sorry
def B_def : B = v * w + v + 1 := by sorry
def C_def : C = w * u + w + 1 := by sorry
def uvw_condition : u * v * w = 1 := by sorry

theorem ABC_equality : A * B * C = A * B + B * C + C * A := by sorry

end ABC_equality_l1845_184514


namespace inequality_relationship_l1845_184515

theorem inequality_relationship (a b : ℝ) (h : a < 1 / b) :
  (a > 0 ∧ b > 0 → 1 / a > b) ∧
  (a < 0 ∧ b < 0 → 1 / a > b) ∧
  (a < 0 ∧ b > 0 → 1 / a < b) :=
by sorry

end inequality_relationship_l1845_184515


namespace restaurant_bill_calculation_l1845_184593

theorem restaurant_bill_calculation
  (check_amount : ℝ)
  (tax_rate : ℝ)
  (tip : ℝ)
  (h1 : check_amount = 15)
  (h2 : tax_rate = 0.20)
  (h3 : tip = 2) :
  check_amount + check_amount * tax_rate + tip = 20 :=
by sorry

end restaurant_bill_calculation_l1845_184593


namespace locus_of_right_angle_vertex_l1845_184510

-- Define the right triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the perpendicular lines (x-axis and y-axis)
def OnXAxis (P : ℝ × ℝ) : Prop := P.2 = 0
def OnYAxis (P : ℝ × ℝ) : Prop := P.1 = 0

-- Define the locus of point C
def LocusC (C : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), RightTriangle A B C ∧ OnXAxis A ∧ OnYAxis B

-- State the theorem
theorem locus_of_right_angle_vertex :
  ∃ (S₁ S₂ : Set (ℝ × ℝ)), 
    (∀ C, LocusC C ↔ C ∈ S₁ ∨ C ∈ S₂) ∧ 
    (∃ (a b c d : ℝ × ℝ), S₁ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • a + t • b}) ∧
    (∃ (e f g h : ℝ × ℝ), S₂ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • e + t • f}) :=
sorry

end locus_of_right_angle_vertex_l1845_184510


namespace fraction_simplification_l1845_184532

theorem fraction_simplification (x : ℝ) : (x - 2) / 4 - (3 * x + 1) / 3 = (-9 * x - 10) / 12 := by
  sorry

end fraction_simplification_l1845_184532


namespace probability_ace_spade_three_correct_l1845_184585

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Number of Aces in a standard deck -/
def NumAces : Nat := 4

/-- Number of Spades in a standard deck -/
def NumSpades : Nat := 13

/-- Number of 3s in a standard deck -/
def NumThrees : Nat := 4

/-- Probability of drawing an Ace as the first card, a Spade as the second card,
    and a 3 as the third card when dealing three cards at random from a standard deck -/
def probability_ace_spade_three : ℚ :=
  17 / 11050

theorem probability_ace_spade_three_correct :
  probability_ace_spade_three = 17 / 11050 := by
  sorry

end probability_ace_spade_three_correct_l1845_184585


namespace bookcase_length_in_feet_l1845_184550

/-- Converts inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem bookcase_length_in_feet :
  inches_to_feet 48 = 4 :=
by sorry

end bookcase_length_in_feet_l1845_184550


namespace min_benches_in_hall_l1845_184506

/-- The minimum number of benches required in a school hall -/
def min_benches (male_students : ℕ) (female_ratio : ℕ) (students_per_bench : ℕ) : ℕ :=
  ((male_students * (female_ratio + 1) + students_per_bench - 1) / students_per_bench : ℕ)

/-- Theorem: Given the conditions, the minimum number of benches required is 29 -/
theorem min_benches_in_hall :
  min_benches 29 4 5 = 29 := by
  sorry

end min_benches_in_hall_l1845_184506


namespace correct_operation_l1845_184554

theorem correct_operation : 
  (5 * Real.sqrt 3 - 2 * Real.sqrt 3 ≠ 3) ∧ 
  (2 * Real.sqrt 2 * 3 * Real.sqrt 2 ≠ 6) ∧ 
  (3 * Real.sqrt 3 / Real.sqrt 3 = 3) ∧ 
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5) := by
  sorry

end correct_operation_l1845_184554


namespace vector_properties_l1845_184524

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def opposite_direction (a b : V) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ b = k • a

theorem vector_properties (a : V) (h : a ≠ 0) :
  opposite_direction a (-3 • a) ∧ a - 3 • a = -2 • a := by
  sorry

end vector_properties_l1845_184524


namespace fraction_value_l1845_184567

theorem fraction_value (a b c : ℤ) 
  (eq1 : a + b = 20) 
  (eq2 : b + c = 22) 
  (eq3 : c + a = 2022) : 
  (a - b) / (c - a) = 1000 := by
sorry

end fraction_value_l1845_184567


namespace grain_mixture_pricing_l1845_184555

/-- Calculates the selling price of a grain given its cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the grain mixture problem -/
theorem grain_mixture_pricing
  (wheat_weight : ℚ) (wheat_price : ℚ) (wheat_profit : ℚ)
  (rice_weight : ℚ) (rice_price : ℚ) (rice_profit : ℚ)
  (barley_weight : ℚ) (barley_price : ℚ) (barley_profit : ℚ)
  (h_wheat_weight : wheat_weight = 30)
  (h_wheat_price : wheat_price = 11.5)
  (h_wheat_profit : wheat_profit = 30)
  (h_rice_weight : rice_weight = 20)
  (h_rice_price : rice_price = 14.25)
  (h_rice_profit : rice_profit = 25)
  (h_barley_weight : barley_weight = 15)
  (h_barley_price : barley_price = 10)
  (h_barley_profit : barley_profit = 35) :
  let total_weight := wheat_weight + rice_weight + barley_weight
  let total_selling_price := sellingPrice (wheat_weight * wheat_price) wheat_profit +
                             sellingPrice (rice_weight * rice_price) rice_profit +
                             sellingPrice (barley_weight * barley_price) barley_profit
  total_selling_price / total_weight = 15.5 := by
  sorry

end grain_mixture_pricing_l1845_184555


namespace solve_equation_l1845_184542

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end solve_equation_l1845_184542


namespace color_coat_drying_time_l1845_184561

/-- Represents the drying time for nail polish coats -/
structure NailPolishDryingTime where
  base_coat : ℕ
  color_coat : ℕ
  top_coat : ℕ
  total_time : ℕ

/-- Theorem: Given the conditions of Jane's nail polish application,
    prove that each color coat takes 3 minutes to dry -/
theorem color_coat_drying_time (t : NailPolishDryingTime)
  (h1 : t.base_coat = 2)
  (h2 : t.top_coat = 5)
  (h3 : t.total_time = 13)
  (h4 : t.total_time = t.base_coat + 2 * t.color_coat + t.top_coat) :
  t.color_coat = 3 := by
  sorry

end color_coat_drying_time_l1845_184561


namespace percentage_problem_l1845_184543

theorem percentage_problem :
  let total := 500
  let unknown_percentage := 50
  let given_percentage := 10
  let result := 25
  (given_percentage / 100) * (unknown_percentage / 100) * total = result :=
by sorry

end percentage_problem_l1845_184543


namespace power_of_three_mod_ten_l1845_184505

theorem power_of_three_mod_ten (k : ℕ) : 3^(4*k + 3) % 10 = 7 := by
  sorry

end power_of_three_mod_ten_l1845_184505


namespace average_of_eleven_numbers_l1845_184539

theorem average_of_eleven_numbers 
  (n : ℕ) 
  (first_six_avg : ℚ) 
  (last_six_avg : ℚ) 
  (sixth_number : ℚ) 
  (h1 : n = 11) 
  (h2 : first_six_avg = 98) 
  (h3 : last_six_avg = 65) 
  (h4 : sixth_number = 318) : 
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / n = 60 := by
  sorry

end average_of_eleven_numbers_l1845_184539


namespace solution_set_nonempty_iff_a_gt_one_l1845_184598

theorem solution_set_nonempty_iff_a_gt_one :
  ∀ a : ℝ, (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := by
  sorry

end solution_set_nonempty_iff_a_gt_one_l1845_184598


namespace board_numbers_prime_or_one_l1845_184519

theorem board_numbers_prime_or_one (a : ℕ) (h_a_odd : Odd a) (h_a_gt_100 : a > 100)
  (h_prime : ∀ n : ℕ, n ≤ Real.sqrt (a / 5) → Nat.Prime ((a - n^2) / 4)) :
  ∀ n : ℕ, Nat.Prime ((a - n^2) / 4) ∨ (a - n^2) / 4 = 1 :=
by sorry

end board_numbers_prime_or_one_l1845_184519


namespace hyperbola_equation_l1845_184534

/-- Represents a hyperbola with focus on the y-axis -/
structure Hyperbola where
  transverse_axis_length : ℝ
  focal_length : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / (h.transverse_axis_length/2)^2) - (x^2 / ((h.focal_length/2)^2 - (h.transverse_axis_length/2)^2)) = 1

theorem hyperbola_equation (h : Hyperbola) 
  (h_transverse : h.transverse_axis_length = 6)
  (h_focal : h.focal_length = 10) :
  ∀ x y : ℝ, standard_equation h x y ↔ y^2/9 - x^2/16 = 1 :=
sorry

end hyperbola_equation_l1845_184534


namespace expression_simplification_and_evaluation_l1845_184553

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  (1 - 1 / x) / ((x^2 - 2*x + 1) / x^2) = 1 + Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_and_evaluation_l1845_184553


namespace square_root_problem_l1845_184574

theorem square_root_problem (n : ℝ) (x : ℝ) (h1 : n > 0) (h2 : Real.sqrt n = x + 3) (h3 : Real.sqrt n = 2*x - 6) :
  x = 1 ∧ n = 16 := by
  sorry

end square_root_problem_l1845_184574


namespace one_large_one_small_capacity_l1845_184572

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 3 large trucks and 4 small trucks is 22 tons -/
axiom condition1 : 3 * large_truck_capacity + 4 * small_truck_capacity = 22

/-- The total capacity of 2 large trucks and 6 small trucks is 23 tons -/
axiom condition2 : 2 * large_truck_capacity + 6 * small_truck_capacity = 23

/-- Theorem: One large truck and one small truck can transport 6.5 tons together -/
theorem one_large_one_small_capacity : 
  large_truck_capacity + small_truck_capacity = 6.5 := by sorry

end one_large_one_small_capacity_l1845_184572


namespace sum_remainder_l1845_184516

theorem sum_remainder (x y z : ℕ) 
  (hx : x % 53 = 31) 
  (hy : y % 53 = 45) 
  (hz : z % 53 = 6) : 
  (x + y + z) % 53 = 29 := by
sorry

end sum_remainder_l1845_184516


namespace golden_ratio_system_solution_l1845_184547

theorem golden_ratio_system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1)
  (eq2 : 2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1)
  (eq3 : 2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1) :
  x = (1 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2 ∧ z = (1 + Real.sqrt 5) / 2 :=
sorry

end golden_ratio_system_solution_l1845_184547


namespace rational_square_decomposition_l1845_184544

theorem rational_square_decomposition (r : ℚ) :
  ∃ (S : Set (ℚ × ℚ)), (Set.Infinite S) ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^2 + y^2 = r^2) :=
sorry

end rational_square_decomposition_l1845_184544


namespace apple_pear_equivalence_l1845_184590

/-- Represents the worth of apples in terms of pears -/
def apple_worth (apples : ℚ) (pears : ℚ) : Prop :=
  apples = pears

theorem apple_pear_equivalence :
  apple_worth (3/4 * 12) 9 →
  apple_worth (2/3 * 6) 4 :=
by
  sorry

end apple_pear_equivalence_l1845_184590


namespace robotics_club_enrollment_l1845_184579

theorem robotics_club_enrollment (total : ℕ) (engineering : ℕ) (computer_science : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : engineering = 45)
  (h3 : computer_science = 35)
  (h4 : both = 25) :
  total - (engineering + computer_science - both) = 25 := by
  sorry

end robotics_club_enrollment_l1845_184579


namespace curve_and_tangent_line_l1845_184560

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the line l1
def l1 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the property of point P
def P_property (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the minimization condition
def min_distance (Q M : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (mx, my) := M
  l1 qx qy ∧ C mx my ∧
  ∀ M' : ℝ × ℝ, C M'.1 M'.2 → (qx - mx)^2 + (qy - my)^2 ≤ (qx - M'.1)^2 + (qy - M'.2)^2

-- State the theorem
theorem curve_and_tangent_line :
  (∀ P : ℝ × ℝ, P_property P → C P.1 P.2) ∧
  (∀ Q M : ℝ × ℝ, min_distance Q M → (M.1 = 1 ∨ M.2 = -4)) :=
sorry

end curve_and_tangent_line_l1845_184560


namespace unique_solution_abc_squared_l1845_184556

theorem unique_solution_abc_squared (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end unique_solution_abc_squared_l1845_184556


namespace sphere_surface_area_l1845_184522

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end sphere_surface_area_l1845_184522


namespace smallest_v_in_consecutive_cubes_sum_l1845_184569

theorem smallest_v_in_consecutive_cubes_sum (w x y u v : ℕ) :
  w < x ∧ x < y ∧ y < u ∧ u < v →
  (∃ a, w = a^3) ∧ (∃ b, x = b^3) ∧ (∃ c, y = c^3) ∧ (∃ d, u = d^3) ∧ (∃ e, v = e^3) →
  w^3 + x^3 + y^3 + u^3 = v^3 →
  v ≥ 6 :=
by sorry

end smallest_v_in_consecutive_cubes_sum_l1845_184569


namespace complex_product_l1845_184581

/-- Given complex numbers Q, E, and D, prove their product is 116i -/
theorem complex_product (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2*I ∧ D = 7 - 3*I → Q * E * D = 116*I := by
  sorry

end complex_product_l1845_184581


namespace slope_point_relation_l1845_184511

theorem slope_point_relation (m : ℝ) : 
  m > 0 → 
  ((m + 1 - 4) / (2 - m) = Real.sqrt 5) → 
  m = (10 - Real.sqrt 5) / 4 := by
sorry

end slope_point_relation_l1845_184511


namespace max_min_difference_z_l1845_184575

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
sorry

end max_min_difference_z_l1845_184575


namespace last_digit_sum_powers_l1845_184582

theorem last_digit_sum_powers : 
  (1993^2002 + 1995^2002) % 10 = 4 := by sorry

end last_digit_sum_powers_l1845_184582


namespace set_A_properties_l1845_184566

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end set_A_properties_l1845_184566


namespace paint_per_statue_l1845_184513

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7 / 16)
  (h2 : num_statues = 7) :
  total_paint / num_statues = 1 / 16 := by
  sorry

end paint_per_statue_l1845_184513


namespace sum_is_composite_l1845_184518

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : 
  ∃ (k : ℕ), k > 1 ∧ k < m + n ∧ k ∣ (m + n) :=
by sorry

end sum_is_composite_l1845_184518


namespace sum_and_count_theorem_l1845_184564

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_integers 10 30 + count_even_integers 10 30 = 431 := by
  sorry

end sum_and_count_theorem_l1845_184564


namespace gcd_of_squares_l1845_184565

theorem gcd_of_squares : Nat.gcd (101^2 + 202^2 + 303^2) (100^2 + 201^2 + 304^2) = 3 := by
  sorry

end gcd_of_squares_l1845_184565


namespace geometric_sequence_exists_l1845_184548

theorem geometric_sequence_exists : ∃ (a r : ℝ), 
  a ≠ 0 ∧ r ≠ 0 ∧ 
  a * r^2 = 3 ∧
  a * r^4 = 27 ∧
  a = -1/3 := by
  sorry

end geometric_sequence_exists_l1845_184548


namespace divisor_sum_property_l1845_184517

def divisors (n : ℕ) : List ℕ := sorry

def D (n : ℕ) : ℕ := sorry

theorem divisor_sum_property (n : ℕ) (h : n > 1) :
  let d := divisors n
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end divisor_sum_property_l1845_184517


namespace triangle_properties_l1845_184527

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * Real.sin t.A = t.b + t.c ∧
  t.a = 6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = π/3 ∧ t.a + t.b + t.c = 18 := by
  sorry

end triangle_properties_l1845_184527


namespace arrangements_count_l1845_184535

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of arrangements for 7 students in a row, 
    where one student (A) must be in the center and 
    two students (B and C) must stand together -/
def arrangements : ℕ := 192

/-- Theorem stating that the number of arrangements is 192 -/
theorem arrangements_count : 
  (∀ (n : ℕ), n = total_students → 
   ∃ (center : Fin n) (together : Fin n → Fin n → Prop),
   (∀ (i j : Fin n), together i j ↔ together j i) ∧
   (∃! (pair : Fin n × Fin n), together pair.1 pair.2) ∧
   (center = ⟨(n - 1) / 2, by sorry⟩) →
   (arrangements = 192)) :=
by sorry

end arrangements_count_l1845_184535


namespace least_three_digit_seven_heavy_l1845_184591

def is_seven_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_seven_heavy : 
  (∀ n : ℕ, is_three_digit n → is_seven_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_seven_heavy 103 :=
sorry

end least_three_digit_seven_heavy_l1845_184591


namespace solve_problem_l1845_184526

-- Define the type for gender
inductive Gender
| Boy
| Girl

-- Define the type for a child
structure Child :=
  (name : String)
  (gender : Gender)
  (statement : Gender)

-- Define the problem setup
def problem_setup (sasha zhenya : Child) : Prop :=
  (sasha.name = "Sasha" ∧ zhenya.name = "Zhenya") ∧
  (sasha.gender ≠ zhenya.gender) ∧
  (sasha.statement = Gender.Boy) ∧
  (zhenya.statement = Gender.Girl) ∧
  (sasha.statement ≠ sasha.gender ∨ zhenya.statement ≠ zhenya.gender)

-- Theorem to prove
theorem solve_problem (sasha zhenya : Child) :
  problem_setup sasha zhenya →
  sasha.gender = Gender.Girl ∧ zhenya.gender = Gender.Boy :=
by
  sorry

end solve_problem_l1845_184526


namespace min_value_theorem_l1845_184509

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let f := fun x : ℝ => a * x^3 + b * x + 2^x
  (∀ x ∈ Set.Icc 0 1, f x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 1, f x = 4) →
  (∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = -3/2) :=
by sorry

end min_value_theorem_l1845_184509


namespace rectangles_in_6x6_grid_l1845_184545

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a grid of size n x n -/
def rectangles_in_grid (n : ℕ) : ℕ := (choose_two n) ^ 2

/-- Theorem: In a 6x6 grid, the number of rectangles is 225 -/
theorem rectangles_in_6x6_grid : rectangles_in_grid 6 = 225 := by
  sorry

end rectangles_in_6x6_grid_l1845_184545


namespace min_value_theorem_l1845_184592

/-- An even function f defined on ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m + 1| - 2

/-- The property of f being an even function -/
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem min_value_theorem (m : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  is_even_function (f m) →
  f m a + f m (2 * b) = m →
  (∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 9/5) →
  ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 9/5 :=
by sorry

end min_value_theorem_l1845_184592


namespace negation_of_universal_quantification_l1845_184504

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end negation_of_universal_quantification_l1845_184504


namespace diophantine_equation_solutions_l1845_184500

def has_solution (n : ℕ) : Prop :=
  ∃ (a b c : ℤ), a^n + b^n = c^n + n

theorem diophantine_equation_solutions :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → (has_solution n ↔ n = 1 ∨ n = 2 ∨ n = 3)) :=
sorry

end diophantine_equation_solutions_l1845_184500


namespace sum_squares_theorem_l1845_184523

theorem sum_squares_theorem (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end sum_squares_theorem_l1845_184523


namespace journey_distance_l1845_184529

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = 224 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 := by
  sorry

end journey_distance_l1845_184529


namespace quilt_shaded_fraction_l1845_184531

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  shaded_squares : ℕ
  half_shaded_squares : ℕ

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  (q.shaded_squares + q.half_shaded_squares / 2) / q.total_squares

/-- Theorem stating that the given quilt block has 3/8 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := {
    total_squares := 16,
    shaded_squares := 4,
    half_shaded_squares := 4
  }
  shaded_fraction q = 3 / 8 := by sorry

end quilt_shaded_fraction_l1845_184531


namespace infinite_solutions_l1845_184536

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℝ) : Prop := 4 * x - 6 * y = 10

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (t : ℝ),
    let (x, y) := f t
    equation1 x y ∧ equation2 x y :=
sorry

end infinite_solutions_l1845_184536


namespace magnitude_of_complex_power_l1845_184577

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2*Complex.I)^(3+3)) = 512 := by sorry

end magnitude_of_complex_power_l1845_184577


namespace initial_bananas_count_l1845_184549

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end initial_bananas_count_l1845_184549


namespace geometry_angle_probability_l1845_184546

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def angle_letters : Finset Char := {'A', 'N', 'G', 'L', 'E'}

theorem geometry_angle_probability : 
  (geometry_letters ∩ angle_letters).card / geometry_letters.card = 1 / 4 := by
sorry

end geometry_angle_probability_l1845_184546


namespace line_equation_conversion_l1845_184541

/-- Given a line in the form (3, 7) · ((x, y) - (-2, 4)) = 0, 
    prove that its slope-intercept form y = mx + b 
    has m = -3/7 and b = 22/7 -/
theorem line_equation_conversion :
  ∀ (x y : ℝ), 
  (3 : ℝ) * (x + 2) + (7 : ℝ) * (y - 4) = 0 →
  ∃ (m b : ℝ), y = m * x + b ∧ m = -3/7 ∧ b = 22/7 := by
  sorry

end line_equation_conversion_l1845_184541
