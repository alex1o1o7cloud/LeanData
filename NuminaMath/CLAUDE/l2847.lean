import Mathlib

namespace solution_set_inequality_inequality_for_positive_mn_l2847_284748

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x ≤ 10 - |x - 3| ↔ x ∈ Set.Icc (-8/3) 4 := by sorry

-- Theorem for part 2
theorem inequality_for_positive_mn (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_mn : m + 2 * n = m * n) :
  f m + f (-2 * n) ≥ 16 := by sorry

end solution_set_inequality_inequality_for_positive_mn_l2847_284748


namespace tom_has_24_blue_marbles_l2847_284718

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The difference between Jason's and Tom's blue marbles -/
def marble_difference : ℕ := 20

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := jason_blue_marbles - marble_difference

theorem tom_has_24_blue_marbles : tom_blue_marbles = 24 := by
  sorry

end tom_has_24_blue_marbles_l2847_284718


namespace sum_of_reciprocals_bound_l2847_284704

theorem sum_of_reciprocals_bound (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 1) : 
  1 / (4*a + 3*b + c) + 1 / (3*a + b + 4*d) + 
  1 / (a + 4*c + 3*d) + 1 / (4*b + 3*c + d) ≥ 2 := by
sorry

end sum_of_reciprocals_bound_l2847_284704


namespace total_animal_sightings_l2847_284720

def week1_sightings : List Nat := [8, 7, 8, 11, 8, 7, 13]
def week2_sightings : List Nat := [7, 9, 10, 21, 11, 7, 17]

theorem total_animal_sightings :
  (week1_sightings.sum + week2_sightings.sum) = 144 := by
  sorry

end total_animal_sightings_l2847_284720


namespace three_digit_number_problem_l2847_284717

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def swap_hundreds_units (n : ℕ) : ℕ :=
  (units_digit n) * 100 + (tens_digit n) * 10 + (hundreds_digit n)

theorem three_digit_number_problem :
  ∀ n : ℕ, is_three_digit_number n →
    (tens_digit n)^2 = (hundreds_digit n) * (units_digit n) →
    n - (swap_hundreds_units n) = 297 →
    (n = 300 ∨ n = 421) :=
sorry

end three_digit_number_problem_l2847_284717


namespace nancy_antacids_per_month_l2847_284782

/-- Calculates the number of antacids Nancy takes per month -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let days_per_week := 7
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Proves that Nancy takes 60 antacids per month given the specified conditions -/
theorem nancy_antacids_per_month :
  antacids_per_month 3 2 1 3 2 4 = 60 := by
  sorry

end nancy_antacids_per_month_l2847_284782


namespace unique_solution_condition_l2847_284745

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0) ↔ b < 7/4 :=
by sorry

end unique_solution_condition_l2847_284745


namespace pupusa_minimum_l2847_284708

theorem pupusa_minimum (a b : ℕ+) (h1 : a < 391) (h2 : Nat.lcm a b > Nat.lcm a 391) : 
  ∀ b' : ℕ+, (∃ a' : ℕ+, a' < 391 ∧ Nat.lcm a' b' > Nat.lcm a' 391) → b' ≥ 18 := by
sorry

end pupusa_minimum_l2847_284708


namespace parallelogram_base_l2847_284786

/-- The base of a parallelogram with area 240 square cm and height 10 cm is 24 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 240 ∧ height = 10 ∧ area = base * height → base = 24 := by
  sorry

end parallelogram_base_l2847_284786


namespace sean_needs_six_packs_l2847_284796

/-- Calculates the number of light bulb packs needed given the number of bulbs required for each room --/
def calculate_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let other_rooms_total := bedroom + bathroom + kitchen + basement
  let garage := other_rooms_total / 2
  let total_bulbs := other_rooms_total + garage
  (total_bulbs + 1) / 2

/-- Proves that Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  calculate_packs_needed 2 1 1 4 = 6 := by
  sorry

end sean_needs_six_packs_l2847_284796


namespace min_height_for_box_l2847_284747

/-- Represents the dimensions of a rectangular box with square base --/
structure BoxDimensions where
  base : ℕ  -- side length of the square base
  height : ℕ -- height of the box

/-- Calculates the surface area of the box --/
def surfaceArea (d : BoxDimensions) : ℕ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the height condition --/
def satisfiesHeightCondition (d : BoxDimensions) : Prop :=
  d.height = 2 * d.base + 1

/-- Checks if the box dimensions satisfy the surface area condition --/
def satisfiesSurfaceAreaCondition (d : BoxDimensions) : Prop :=
  surfaceArea d ≥ 130

/-- The main theorem stating the minimum height that satisfies all conditions --/
theorem min_height_for_box : 
  ∃ (d : BoxDimensions), 
    satisfiesHeightCondition d ∧ 
    satisfiesSurfaceAreaCondition d ∧ 
    (∀ (d' : BoxDimensions), 
      satisfiesHeightCondition d' ∧ 
      satisfiesSurfaceAreaCondition d' → 
      d.height ≤ d'.height) ∧
    d.height = 9 := by
  sorry

end min_height_for_box_l2847_284747


namespace four_is_eight_percent_of_fifty_l2847_284727

theorem four_is_eight_percent_of_fifty :
  (4 : ℝ) / 50 * 100 = 8 := by sorry

end four_is_eight_percent_of_fifty_l2847_284727


namespace equation_solution_l2847_284703

theorem equation_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end equation_solution_l2847_284703


namespace estimate_2_sqrt_5_l2847_284784

theorem estimate_2_sqrt_5 : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end estimate_2_sqrt_5_l2847_284784


namespace school_population_l2847_284753

/-- Given a school with 42 boys and a boy-to-girl ratio of 7:1, 
    prove that the total number of students is 48. -/
theorem school_population (num_boys : ℕ) (ratio : ℚ) : 
  num_boys = 42 → ratio = 7/1 → num_boys + (num_boys / ratio.num) = 48 := by
  sorry

end school_population_l2847_284753


namespace sum_of_reciprocals_of_roots_l2847_284730

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 8) :=
by sorry

end sum_of_reciprocals_of_roots_l2847_284730


namespace product_equals_difference_of_squares_l2847_284737

theorem product_equals_difference_of_squares (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end product_equals_difference_of_squares_l2847_284737


namespace fraction_equivalence_l2847_284744

theorem fraction_equivalence (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by
  sorry

end fraction_equivalence_l2847_284744


namespace range_of_y_over_x_l2847_284761

theorem range_of_y_over_x (x y : ℝ) (h1 : 3 * x - 2 * y - 5 = 0) (h2 : 1 ≤ x) (h3 : x ≤ 2) :
  ∃ (z : ℝ), z = y / x ∧ -1 ≤ z ∧ z ≤ 1/4 :=
sorry

end range_of_y_over_x_l2847_284761


namespace complement_union_theorem_complement_intersect_theorem_l2847_284759

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_theorem : 
  (Set.univ \ (A ∪ B)) = {x : ℝ | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_intersect_theorem :
  ((Set.univ \ A) ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end complement_union_theorem_complement_intersect_theorem_l2847_284759


namespace job_crop_production_l2847_284732

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production given a FarmLand allocation -/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that for Job's specific land allocation, the crop production area is 70 hectares -/
theorem job_crop_production :
  let job_farm := FarmLand.mk 150 25 15 40
  crop_production job_farm = 70 := by
  sorry

end job_crop_production_l2847_284732


namespace functions_inequality_l2847_284740

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_inequality (hf : f 0 = 0) 
  (hg : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y) :
  ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 := by
sorry

end functions_inequality_l2847_284740


namespace repeating_decimal_sum_l2847_284773

theorem repeating_decimal_sum (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 12 / 13 →
  a = 4 ∧ b = 6 ∧ c = 3 := by
sorry

end repeating_decimal_sum_l2847_284773


namespace max_age_on_aubrey_birthday_l2847_284789

/-- The age difference between Luka and Aubrey -/
def age_difference : ℕ := 2

/-- Luka's age when Max was born -/
def luka_age_at_max_birth : ℕ := 4

/-- Aubrey's age for which we want to find Max's age -/
def aubrey_target_age : ℕ := 8

/-- Max's age when Aubrey reaches the target age -/
def max_age : ℕ := aubrey_target_age - age_difference

theorem max_age_on_aubrey_birthday :
  max_age = 6 := by sorry

end max_age_on_aubrey_birthday_l2847_284789


namespace cement_mixture_weight_l2847_284766

theorem cement_mixture_weight : 
  ∀ W : ℝ, 
    (5/14 + 3/10 + 2/9 + 1/7) * W + 2.5 = W → 
    W = 112.5 := by
  sorry

end cement_mixture_weight_l2847_284766


namespace oranges_per_box_l2847_284785

/-- Given 45 oranges and 9 boxes, prove that the number of oranges per box is 5 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 45) (h2 : num_boxes = 9) : 
  total_oranges / num_boxes = 5 := by
  sorry

end oranges_per_box_l2847_284785


namespace min_vertical_distance_l2847_284738

-- Define the two functions
def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 3*x - 2

-- Define the vertical distance between the two functions
def verticalDistance (x : ℝ) : ℝ := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), verticalDistance x = 1 ∧ ∀ (y : ℝ), verticalDistance y ≥ 1 :=
sorry

end min_vertical_distance_l2847_284738


namespace faucet_leak_approx_l2847_284714

/-- The volume of water leaked by an untightened faucet in 4 hours -/
def faucet_leak_volume : ℝ :=
  let drops_per_second : ℝ := 2
  let milliliters_per_drop : ℝ := 0.05
  let hours : ℝ := 4
  let seconds_per_hour : ℝ := 3600
  drops_per_second * milliliters_per_drop * hours * seconds_per_hour

/-- Assertion that the faucet leak volume is approximately 1.4 × 10^3 milliliters -/
theorem faucet_leak_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 10 ∧ |faucet_leak_volume - 1.4e3| < ε :=
sorry

end faucet_leak_approx_l2847_284714


namespace adult_ticket_price_l2847_284728

theorem adult_ticket_price
  (total_tickets : ℕ)
  (senior_price : ℚ)
  (total_receipts : ℚ)
  (senior_tickets : ℕ)
  (h1 : total_tickets = 529)
  (h2 : senior_price = 15)
  (h3 : total_receipts = 9745)
  (h4 : senior_tickets = 348) :
  (total_receipts - senior_price * senior_tickets) / (total_tickets - senior_tickets) = 25 := by
  sorry

end adult_ticket_price_l2847_284728


namespace min_value_of_f_l2847_284736

-- Define second-order product sum
def second_order_sum (a b c d : ℤ) : ℤ := a * d + b * c

-- Define third-order product sum
def third_order_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℤ) : ℤ :=
  a1 * (second_order_sum b2 b3 c2 c3) +
  a2 * (second_order_sum b1 b3 c1 c3) +
  a3 * (second_order_sum b1 b2 c1 c2)

-- Define the function f
def f (n : ℕ+) : ℤ := third_order_sum n 2 (-9) n 1 n 1 2 n

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℤ), m = -21 ∧ ∀ (n : ℕ+), f n ≥ m :=
sorry

end min_value_of_f_l2847_284736


namespace permutation_combination_sum_l2847_284750

/-- Given that A_n^m = 272 and C_n^m = 136, prove that m + n = 19 -/
theorem permutation_combination_sum (m n : ℕ) 
  (h1 : m.factorial * (n - m).factorial * 272 = n.factorial)
  (h2 : m.factorial * (n - m).factorial * 136 = n.factorial) : 
  m + n = 19 := by
  sorry

end permutation_combination_sum_l2847_284750


namespace petya_can_buy_ice_cream_l2847_284746

theorem petya_can_buy_ice_cream (total : ℕ) (kolya vasya petya : ℕ) : 
  total = 2200 →
  kolya * 18 = vasya →
  total = kolya + vasya + petya →
  petya ≥ 15 :=
by sorry

end petya_can_buy_ice_cream_l2847_284746


namespace sqrt_x_minus_2_meaningful_l2847_284724

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_meaningful_l2847_284724


namespace arithmetic_sequence_sum_l2847_284765

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term 2, last term 102, and common difference 5 is 1092 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 2 102 5 = 1092 := by
  sorry

end arithmetic_sequence_sum_l2847_284765


namespace votes_against_percentage_l2847_284764

theorem votes_against_percentage (total_votes : ℕ) (difference : ℕ) :
  total_votes = 330 →
  difference = 66 →
  let votes_against := (total_votes - difference) / 2
  let percentage_against := (votes_against : ℚ) / total_votes * 100
  percentage_against = 40 := by
  sorry

end votes_against_percentage_l2847_284764


namespace jimmy_action_figures_sale_discount_l2847_284776

theorem jimmy_action_figures_sale_discount (total_figures : ℕ) 
  (regular_figure_value : ℚ) (special_figure_value : ℚ) (total_earned : ℚ) :
  total_figures = 5 →
  regular_figure_value = 15 →
  special_figure_value = 20 →
  total_earned = 55 →
  (4 * regular_figure_value + special_figure_value - total_earned) / total_figures = 5 := by
  sorry

end jimmy_action_figures_sale_discount_l2847_284776


namespace inscribed_circle_rectangle_area_l2847_284751

/-- A circle inscribed in a rectangle --/
structure InscribedCircle where
  radius : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ
  inscribed : rectangle_width = 2 * radius
  ratio : rectangle_length = 3 * rectangle_width

/-- The area of a rectangle with an inscribed circle of radius 8 and length-to-width ratio of 3:1 is 768 --/
theorem inscribed_circle_rectangle_area (c : InscribedCircle) 
  (h1 : c.radius = 8) : c.rectangle_length * c.rectangle_width = 768 := by
  sorry

#check inscribed_circle_rectangle_area

end inscribed_circle_rectangle_area_l2847_284751


namespace seventh_term_of_geometric_sequence_l2847_284710

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_a4 : a 4 = 8)
  (h_q : q = -2) :
  a 7 = -64 := by
sorry

end seventh_term_of_geometric_sequence_l2847_284710


namespace prob_odd_divisor_15_factorial_l2847_284706

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def prob_odd_divisor (n : ℕ) : ℚ :=
  (num_odd_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  prob_odd_divisor (factorial 15) = 1 / 6 := by sorry

end prob_odd_divisor_15_factorial_l2847_284706


namespace triangle_angle_measure_l2847_284775

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = c and C = π/5, then B = 3π/10 -/
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end triangle_angle_measure_l2847_284775


namespace overtime_hours_calculation_l2847_284733

theorem overtime_hours_calculation (regular_rate overtime_rate total_pay : ℚ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_rate = 2 * regular_rate)
  (h3 : total_pay = 186) : 
  (total_pay - 40 * regular_rate) / overtime_rate = 11 := by
  sorry

end overtime_hours_calculation_l2847_284733


namespace xi_eq_4_equiv_events_l2847_284715

/-- Represents the outcome of a single die roll -/
def DieRoll : Type := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceRoll : Type := DieRoll × DieRoll

/-- The sum of the points obtained when rolling two dice -/
def ξ : TwoDiceRoll → Nat :=
  fun (d1, d2) => d1.val + 1 + d2.val + 1

/-- The event where one die shows 3 and the other shows 1 -/
def event_3_1 : Set TwoDiceRoll :=
  {roll | (roll.1.val = 2 ∧ roll.2.val = 0) ∨ (roll.1.val = 0 ∧ roll.2.val = 2)}

/-- The event where both dice show 2 -/
def event_2_2 : Set TwoDiceRoll :=
  {roll | roll.1.val = 1 ∧ roll.2.val = 1}

/-- The theorem stating that ξ = 4 is equivalent to the union of event_3_1 and event_2_2 -/
theorem xi_eq_4_equiv_events :
  {roll : TwoDiceRoll | ξ roll = 4} = event_3_1 ∪ event_2_2 := by
  sorry

end xi_eq_4_equiv_events_l2847_284715


namespace min_intersection_length_l2847_284762

def set_length (a b : ℝ) := b - a

def M (m : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ m + 7/10}
def N (n : ℝ) := {x : ℝ | n - 2/5 ≤ x ∧ x ≤ n}

theorem min_intersection_length :
  ∃ (min_length : ℝ),
    min_length = 1/10 ∧
    ∀ (m n : ℝ),
      0 ≤ m → m ≤ 3/10 →
      2/5 ≤ n → n ≤ 1 →
      ∃ (a b : ℝ),
        (∀ x, x ∈ M m ∩ N n ↔ a ≤ x ∧ x ≤ b) ∧
        set_length a b ≥ min_length :=
by sorry

end min_intersection_length_l2847_284762


namespace remainder_problem_l2847_284707

theorem remainder_problem (x : ℤ) : x % 61 = 24 → x % 5 = 4 := by
  sorry

end remainder_problem_l2847_284707


namespace v_2002_equals_4_l2847_284798

def g : ℕ → ℕ
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | 5 => 5
  | _ => 0  -- default case for completeness

def v : ℕ → ℕ
  | 0 => 3
  | n + 1 => g (v n)

theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end v_2002_equals_4_l2847_284798


namespace cubic_root_sum_l2847_284778

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 25 * a - 1 = 0) →
  (40 * b^3 - 60 * b^2 + 25 * b - 1 = 0) →
  (40 * c^3 - 60 * c^2 + 25 * c - 1 = 0) →
  (0 < a) ∧ (a < 1) →
  (0 < b) ∧ (b < 1) →
  (0 < c) ∧ (c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end cubic_root_sum_l2847_284778


namespace food_bank_remaining_l2847_284767

/-- Calculates the amount of food remaining in the food bank after four weeks of donations and distributions. -/
theorem food_bank_remaining (week1_donation : ℝ) (week2_factor : ℝ) (week3_increase : ℝ) (week4_decrease : ℝ)
  (week1_given_out : ℝ) (week2_given_out : ℝ) (week3_given_out : ℝ) (week4_given_out : ℝ) :
  week1_donation = 40 →
  week2_factor = 1.5 →
  week3_increase = 1.25 →
  week4_decrease = 0.9 →
  week1_given_out = 0.6 →
  week2_given_out = 0.7 →
  week3_given_out = 0.8 →
  week4_given_out = 0.5 →
  let week2_donation := week1_donation * week2_factor
  let week3_donation := week2_donation * week3_increase
  let week4_donation := week3_donation * week4_decrease
  let week1_remaining := week1_donation * (1 - week1_given_out)
  let week2_remaining := week2_donation * (1 - week2_given_out)
  let week3_remaining := week3_donation * (1 - week3_given_out)
  let week4_remaining := week4_donation * (1 - week4_given_out)
  week1_remaining + week2_remaining + week3_remaining + week4_remaining = 82.75 := by
  sorry

end food_bank_remaining_l2847_284767


namespace actual_time_when_clock_shows_7pm_l2847_284779

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Converts Time to minutes since midnight -/
def timeToMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- Represents a clock that may gain or lose time -/
structure Clock where
  rate : ℚ  -- Rate of time gain/loss (1 means accurate, >1 means gaining time)

theorem actual_time_when_clock_shows_7pm 
  (c : Clock) 
  (h1 : c.rate = 7 / 6)  -- Clock gains 5 minutes in 30 minutes
  (h2 : timeToMinutes { hours := 7, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
        c.rate * timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num }) :
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } := by
  sorry

end actual_time_when_clock_shows_7pm_l2847_284779


namespace smallest_number_of_eggs_l2847_284701

theorem smallest_number_of_eggs : ∀ (n : ℕ),
  n > 200 ∧
  ∃ (c : ℕ), n = 15 * c - 3 ∧
  c ≥ 14 →
  n ≥ 207 ∧
  ∃ (m : ℕ), m = 207 ∧ m > 200 ∧ ∃ (d : ℕ), m = 15 * d - 3 ∧ d ≥ 14 :=
by
  sorry

end smallest_number_of_eggs_l2847_284701


namespace outfit_count_l2847_284793

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 8

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- A function that calculates the number of valid outfits -/
def valid_outfits : ℕ := 
  num_colors * num_colors * num_colors - 
  (num_colors * (num_colors - 1) * 3)

/-- Theorem stating that the number of valid outfits is 344 -/
theorem outfit_count : valid_outfits = 344 := by
  sorry

end outfit_count_l2847_284793


namespace cube_of_eight_l2847_284783

theorem cube_of_eight : 8^3 = 512 := by
  sorry

end cube_of_eight_l2847_284783


namespace chord_length_specific_case_l2847_284770

/-- The length of the chord cut by a circle on a line -/
def chord_length (a b c d e f : ℝ) : ℝ :=
  let circle := fun (x y : ℝ) => x^2 + y^2 + a*x + b*y + c
  let line := fun (x y : ℝ) => d*x + e*y + f
  -- The actual calculation of the chord length would go here
  0  -- Placeholder

theorem chord_length_specific_case :
  chord_length 0 (-2) (-1) 2 (-1) (-1) = 2 * Real.sqrt 30 / 5 := by
  sorry

#check chord_length_specific_case

end chord_length_specific_case_l2847_284770


namespace probability_at_most_six_distinct_numbers_l2847_284756

theorem probability_at_most_six_distinct_numbers : 
  let n_dice : ℕ := 8
  let n_faces : ℕ := 6
  let total_outcomes : ℕ := n_faces ^ n_dice
  let favorable_outcomes : ℕ := 3628800
  (favorable_outcomes : ℚ) / total_outcomes = 45 / 52 :=
by sorry

end probability_at_most_six_distinct_numbers_l2847_284756


namespace journey_time_theorem_l2847_284772

/-- Represents the time and distance relationship for a journey to the supermarket -/
structure JourneyTime where
  bike_speed : ℝ
  walk_speed : ℝ
  total_distance : ℝ

/-- The journey time satisfies the given conditions -/
def satisfies_conditions (j : JourneyTime) : Prop :=
  j.bike_speed * 12 + j.walk_speed * 20 = j.total_distance ∧
  j.bike_speed * 8 + j.walk_speed * 36 = j.total_distance

/-- The theorem to be proved -/
theorem journey_time_theorem (j : JourneyTime) (h : satisfies_conditions j) :
  (j.total_distance - j.bike_speed * 2) / j.walk_speed = 60 := by
  sorry

end journey_time_theorem_l2847_284772


namespace greatest_c_value_l2847_284731

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end greatest_c_value_l2847_284731


namespace parallel_vectors_dot_product_l2847_284729

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem parallel_vectors_dot_product :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → dot_product a b = -10 := by
  sorry

end parallel_vectors_dot_product_l2847_284729


namespace largest_n_with_lcm_property_l2847_284734

theorem largest_n_with_lcm_property : 
  ∃ (m : ℕ+), Nat.lcm m.val 972 = 3 * m.val * Nat.gcd m.val 972 ∧ 
  ∀ (n : ℕ) (m : ℕ+), n > 972 → n < 1000 → 
    Nat.lcm m.val n ≠ 3 * m.val * Nat.gcd m.val n := by
  sorry

end largest_n_with_lcm_property_l2847_284734


namespace minibus_seats_l2847_284723

/-- The number of seats in a minibus given specific seating arrangements -/
theorem minibus_seats (total_children : ℕ) (three_child_seats : ℕ) : 
  total_children = 19 →
  three_child_seats = 5 →
  (∃ (two_child_seats : ℕ), 
    total_children = three_child_seats * 3 + two_child_seats * 2 ∧
    three_child_seats + two_child_seats = 7) := by
  sorry

end minibus_seats_l2847_284723


namespace exists_graph_with_short_paths_l2847_284743

/-- A directed graph with n vertices -/
def DirectedGraph (n : ℕ) := Fin n → Fin n → Prop

/-- A path of length at most 2 exists between two vertices in a directed graph -/
def HasPathAtMost2 (G : DirectedGraph n) (u v : Fin n) : Prop :=
  G u v ∨ ∃ w, G u w ∧ G w v

/-- For any n > 4, there exists a directed graph with n vertices
    such that any two vertices have a path of length at most 2 between them -/
theorem exists_graph_with_short_paths (n : ℕ) (h : n > 4) :
  ∃ G : DirectedGraph n, ∀ u v : Fin n, HasPathAtMost2 G u v :=
sorry

end exists_graph_with_short_paths_l2847_284743


namespace disjoint_sets_imply_m_leq_neg_one_l2847_284705

def A : Set (ℝ × ℝ) := {p | p.2 = Real.log (p.1 + 1) - 1}

def B (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 = m}

theorem disjoint_sets_imply_m_leq_neg_one (m : ℝ) :
  A ∩ B m = ∅ → m ≤ -1 := by sorry

end disjoint_sets_imply_m_leq_neg_one_l2847_284705


namespace action_figures_removed_l2847_284709

/-- 
Given:
- initial_figures: The initial number of action figures
- added_figures: The number of action figures added
- final_figures: The final number of action figures on the shelf

Prove that the number of removed figures is 1.
-/
theorem action_figures_removed 
  (initial_figures : ℕ) 
  (added_figures : ℕ) 
  (final_figures : ℕ) 
  (h1 : initial_figures = 3)
  (h2 : added_figures = 4)
  (h3 : final_figures = 6) :
  initial_figures + added_figures - final_figures = 1 := by
  sorry

end action_figures_removed_l2847_284709


namespace multiple_p_solutions_l2847_284777

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 heads in 5 tosses -/
def w (p : ℝ) : ℝ := binomial_probability 5 3 p

/-- There exist at least two distinct values of p in (0, 1) that satisfy w(p) = 144/625 -/
theorem multiple_p_solutions : ∃ p₁ p₂ : ℝ, 0 < p₁ ∧ p₁ < 1 ∧ 0 < p₂ ∧ p₂ < 1 ∧ p₁ ≠ p₂ ∧ w p₁ = 144/625 ∧ w p₂ = 144/625 := by
  sorry

end multiple_p_solutions_l2847_284777


namespace cookies_per_person_l2847_284757

/-- The number of cookies in a dozen --/
def dozen : ℕ := 12

/-- The number of batches Beth bakes --/
def batches : ℕ := 4

/-- The number of dozens per batch --/
def dozens_per_batch : ℕ := 2

/-- The number of people sharing the cookies --/
def people : ℕ := 16

/-- Theorem: Each person consumes 6 cookies when 4 batches of 2 dozen cookies are shared equally among 16 people --/
theorem cookies_per_person :
  (batches * dozens_per_batch * dozen) / people = 6 := by
  sorry

end cookies_per_person_l2847_284757


namespace book_distribution_theorem_l2847_284769

def num_books : ℕ := 6
def num_people : ℕ := 3

/-- The number of ways to divide 6 books into three parts of 2 books each -/
def divide_equal_parts : ℕ := 15

/-- The number of ways to distribute 6 books to three people, each receiving 2 books -/
def distribute_equal : ℕ := 90

/-- The number of ways to distribute 6 books to three people without restrictions -/
def distribute_unrestricted : ℕ := 729

/-- The number of ways to distribute 6 books to three people, with each person receiving at least 1 book -/
def distribute_at_least_one : ℕ := 481

theorem book_distribution_theorem :
  divide_equal_parts = 15 ∧
  distribute_equal = 90 ∧
  distribute_unrestricted = 729 ∧
  distribute_at_least_one = 481 :=
by sorry

end book_distribution_theorem_l2847_284769


namespace multiplication_mistake_l2847_284716

theorem multiplication_mistake (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 138 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1242 := by
sorry

end multiplication_mistake_l2847_284716


namespace gym_visitors_l2847_284780

theorem gym_visitors (initial_count : ℕ) (left_count : ℕ) (final_count : ℕ) :
  final_count ≥ initial_count - left_count →
  (final_count - (initial_count - left_count)) = 
  (final_count + left_count - initial_count) :=
by sorry

end gym_visitors_l2847_284780


namespace valid_a_values_l2847_284781

def A (a : ℝ) : Set ℝ := {2, 1 - a, a^2 - a + 2}

theorem valid_a_values : ∀ a : ℝ, 4 ∈ A a ↔ a = -3 ∨ a = 2 := by sorry

end valid_a_values_l2847_284781


namespace circumscribed_circle_diameter_l2847_284749

/-- Given a triangle with one side of length 10 and the opposite angle of 45°,
    the diameter of its circumscribed circle is 10√2. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 10) (h_angle : angle = Real.pi / 4) :
  (side / Real.sin angle) = 10 * Real.sqrt 2 := by
  sorry

end circumscribed_circle_diameter_l2847_284749


namespace gadget_marked_price_l2847_284763

/-- The marked price of a gadget under specific conditions -/
theorem gadget_marked_price 
  (original_price : ℝ)
  (purchase_discount : ℝ)
  (desired_gain_percentage : ℝ)
  (operating_cost : ℝ)
  (selling_discount : ℝ)
  (h1 : original_price = 50)
  (h2 : purchase_discount = 0.15)
  (h3 : desired_gain_percentage = 0.4)
  (h4 : operating_cost = 5)
  (h5 : selling_discount = 0.25) :
  ∃ (marked_price : ℝ), 
    marked_price = 86 ∧ 
    marked_price * (1 - selling_discount) = 
      (original_price * (1 - purchase_discount) * (1 + desired_gain_percentage) + operating_cost) := by
  sorry


end gadget_marked_price_l2847_284763


namespace complement_of_P_in_U_l2847_284788

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  U \ P = Set.Ioo (-1) 6 := by sorry

end complement_of_P_in_U_l2847_284788


namespace larger_number_proof_l2847_284787

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (x : ℕ+), Nat.lcm A B = 20 * 11 * 15 * x) → 
  (A ≤ B) →
  B = 300 := by
sorry

end larger_number_proof_l2847_284787


namespace correct_parentheses_removal_l2847_284713

theorem correct_parentheses_removal (a b c d : ℝ) : 
  (a^2 - (1 - 2*a) ≠ a^2 - 1 - 2*a) ∧ 
  (a^2 + (-1 - 2*a) ≠ a^2 - 1 + 2*a) ∧ 
  (a - (5*b - (2*c - 1)) = a - 5*b + 2*c - 1) ∧ 
  (-(a + b) + (c - d) ≠ -a - b + c + d) :=
by sorry

end correct_parentheses_removal_l2847_284713


namespace language_selection_theorem_l2847_284739

theorem language_selection_theorem (n : ℕ) :
  ∀ (employees : Finset (Finset ℕ)),
    (employees.card = 500) →
    (∀ e ∈ employees, e ⊆ Finset.range (2 * n)) →
    (∀ e ∈ employees, e.card ≥ n) →
    ∃ (selected : Finset ℕ),
      selected.card = 14 ∧
      selected ⊆ Finset.range (2 * n) ∧
      ∀ e ∈ employees, ∃ l ∈ selected, l ∈ e :=
by sorry

end language_selection_theorem_l2847_284739


namespace complex_fraction_simplification_l2847_284795

theorem complex_fraction_simplification (a b : ℝ) (ha : a = 4.91) (hb : b = 0.09) :
  (((a^2 - b^2) * (a^2 + b^(2/3) + a * b^(1/3))) / (a * b^(1/3) + a * a^(1/2) - b * b^(1/3) - (a * b^2)^(1/2))) /
  ((a^3 - b) / (a * b^(1/3) - (a^3 * b^2)^(1/6) - b^(2/3) + a * a^(1/2))) = a + b :=
by sorry

end complex_fraction_simplification_l2847_284795


namespace unique_function_solution_l2847_284755

theorem unique_function_solution :
  ∃! f : ℝ → ℝ, (∀ x y : ℝ, f (x + f y - 1) = x + y) ∧ (∀ x : ℝ, f x = x + 1/2) :=
by sorry

end unique_function_solution_l2847_284755


namespace smallest_distance_between_points_on_circles_l2847_284760

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 5*I)) = 2)
  (hw : Complex.abs (w - (-3 + 4*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 106 - 6 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 5*I)) = 2 → 
      Complex.abs (w' - (-3 + 4*I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end smallest_distance_between_points_on_circles_l2847_284760


namespace max_min_values_l2847_284774

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  ∃ (m n : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x a ≤ m) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = m) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → n ≤ f x a) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = n) ∧
    m = 1 + a ∧
    n = -3 + a :=
by
  sorry

end max_min_values_l2847_284774


namespace divisibility_equivalence_l2847_284754

theorem divisibility_equivalence (m n : ℕ) : 
  (((2^m : ℕ) - 1)^2 ∣ ((2^n : ℕ) - 1)) ↔ (m * ((2^m : ℕ) - 1) ∣ n) := by
  sorry

end divisibility_equivalence_l2847_284754


namespace games_played_so_far_l2847_284711

/-- Proves that the number of games played so far is 15, given the conditions of the problem -/
theorem games_played_so_far 
  (total_games : ℕ) 
  (current_average : ℚ) 
  (goal_average : ℚ) 
  (required_average : ℚ) 
  (h1 : total_games = 20)
  (h2 : current_average = 26)
  (h3 : goal_average = 30)
  (h4 : required_average = 42)
  : ∃ (x : ℕ), x = 15 ∧ 
    x * current_average + (total_games - x) * required_average = total_games * goal_average := by
  sorry

end games_played_so_far_l2847_284711


namespace line_equation_l2847_284721

/-- A line parameterized by (x,y) = (3t + 6, 5t - 7) where t is a real number -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem line_equation :
  ∀ (t x y : ℝ), parameterized_line t = (x, y) →
  y = slope_intercept_form (5/3) (-17) x := by
sorry

end line_equation_l2847_284721


namespace smallest_k_with_remainders_l2847_284719

theorem smallest_k_with_remainders : ∃ k : ℕ, 
  k > 1 ∧
  k % 17 = 1 ∧
  k % 11 = 1 ∧
  k % 6 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 17 = 1 → m % 11 = 1 → m % 6 = 2 → k ≤ m :=
by
  use 188
  sorry

end smallest_k_with_remainders_l2847_284719


namespace initial_average_problem_l2847_284702

theorem initial_average_problem (n : ℕ) (A : ℝ) (added_value : ℝ) (new_average : ℝ) 
  (h1 : n = 15)
  (h2 : added_value = 14)
  (h3 : new_average = 54)
  (h4 : (n : ℝ) * A + n * added_value = n * new_average) :
  A = 40 := by
sorry

end initial_average_problem_l2847_284702


namespace shortest_side_theorem_l2847_284791

theorem shortest_side_theorem (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → a + c > b → 
  a^2 + b^2 > 5*c^2 → 
  c < a ∧ c < b :=
sorry

end shortest_side_theorem_l2847_284791


namespace equation_solution_l2847_284790

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / x) ∧ x = 9 := by
  sorry

end equation_solution_l2847_284790


namespace unique_two_digit_number_l2847_284792

/-- A function that returns true if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns true if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A function that returns true if a number is a multiple of 9 -/
def isMultipleOf9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that returns true if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem unique_two_digit_number :
  ∃! n : ℕ, isTwoDigit n ∧ isOdd n ∧ isMultipleOf9 n ∧
    isPerfectSquare (tensDigit n * onesDigit n) ∧ n = 99 :=
sorry

end unique_two_digit_number_l2847_284792


namespace polygon_sides_l2847_284712

/-- The number of diagonals that can be drawn from one vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: If 2018 diagonals can be drawn from one vertex of an n-sided polygon, then n = 2021 -/
theorem polygon_sides (n : ℕ) (h : diagonals_from_vertex n = 2018) : n = 2021 := by
  sorry

end polygon_sides_l2847_284712


namespace isosceles_triangle_perimeter_l2847_284799

/-- An isosceles triangle with two sides of length 7 and one side of length 3 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 3 → -- Two sides are 7cm and one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) → -- Isosceles condition
  a + b + c = 17 := by -- Perimeter is 17cm
sorry


end isosceles_triangle_perimeter_l2847_284799


namespace quadratic_root_range_l2847_284758

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x > 1 ∧ y < -1 ∧ 
   x^2 + (a^2 + 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 + 1)*y + a - 2 = 0) →
  -1 < a ∧ a < 0 :=
by sorry

end quadratic_root_range_l2847_284758


namespace tangent_line_value_l2847_284752

/-- A function f: ℝ → ℝ is tangent to the line y = -x + 8 at x = 5 if:
    1. f(5) = -5 + 8
    2. f'(5) = -1
-/
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = 3 ∧ deriv f 5 = -1

theorem tangent_line_value (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : is_tangent_at_5 f) : f 5 = 3 := by
  sorry

end tangent_line_value_l2847_284752


namespace opposite_of_negative_five_l2847_284741

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end opposite_of_negative_five_l2847_284741


namespace fraction_sum_inequality_l2847_284768

theorem fraction_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (x + y) + y / (y + z) + z / (z + x) ≤ 2 := by
  sorry

end fraction_sum_inequality_l2847_284768


namespace airplane_passengers_l2847_284771

theorem airplane_passengers (P : ℕ) 
  (h1 : P - 58 + 24 - 47 + 14 + 10 = 67) : P = 124 := by
  sorry

end airplane_passengers_l2847_284771


namespace envelope_game_properties_l2847_284726

/-- A game with envelopes and two evenly matched teams -/
structure EnvelopeGame where
  num_envelopes : ℕ
  win_points : ℕ
  win_probability : ℝ

/-- Calculate the expected number of points for one team in a single game -/
noncomputable def expected_points (game : EnvelopeGame) : ℝ :=
  sorry

/-- Calculate the probability of a specific envelope being chosen in a game -/
noncomputable def envelope_probability (game : EnvelopeGame) : ℝ :=
  sorry

/-- Theorem about the expected points and envelope probability in the specific game -/
theorem envelope_game_properties :
  let game : EnvelopeGame := ⟨13, 6, 1/2⟩
  (100 * expected_points game = 465) ∧
  (envelope_probability game = 12/13) := by
  sorry

end envelope_game_properties_l2847_284726


namespace exponent_division_l2847_284742

theorem exponent_division (a : ℝ) : a^6 / a^4 = a^2 := by
  sorry

end exponent_division_l2847_284742


namespace reciprocal_difference_fractions_l2847_284700

theorem reciprocal_difference_fractions : (1 / (1/4 - 1/5) : ℚ) = 20 := by
  sorry

end reciprocal_difference_fractions_l2847_284700


namespace hyperbola_equation_l2847_284725

/-- A hyperbola with center at the origin, focus at (3,0), and a line passing through
    the focus intersecting the hyperbola at two points whose midpoint is (-12,-15) -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The center of the hyperbola is at the origin -/
  center_at_origin : equation 0 0
  /-- One focus of the hyperbola is at (3,0) -/
  focus_at_3_0 : ∃ (x y : ℝ), equation x y ∧ (x - 3)^2 + y^2 = (x + 3)^2 + y^2
  /-- There exists a line passing through (3,0) that intersects the hyperbola at two points -/
  intersecting_line : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ equation x₂ y₂ ∧ 
    (y₁ - 0) / (x₁ - 3) = (y₂ - 0) / (x₂ - 3)
  /-- The midpoint of the two intersection points is (-12,-15) -/
  midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ equation x₂ y₂ ∧
    (x₁ + x₂) / 2 = -12 ∧ (y₁ + y₂) / 2 = -15

/-- The equation of the hyperbola is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.equation = fun x y => x^2 / 4 - y^2 / 5 = 1 := by sorry

end hyperbola_equation_l2847_284725


namespace cube_preserves_inequality_l2847_284794

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_inequality_l2847_284794


namespace family_reunion_children_l2847_284797

theorem family_reunion_children (adults children : ℕ) : 
  adults = children / 3 →
  adults / 3 + 10 = adults →
  children = 45 := by
sorry

end family_reunion_children_l2847_284797


namespace expected_closest_distance_five_points_l2847_284735

/-- The expected distance between the closest pair of points when five points are chosen uniformly at random on a segment of length 1 -/
theorem expected_closest_distance_five_points (segment_length : ℝ) 
  (h_segment : segment_length = 1) : ℝ :=
by
  sorry

end expected_closest_distance_five_points_l2847_284735


namespace travel_theorem_l2847_284722

-- Define the cities and distances
def XY : ℝ := 4500
def XZ : ℝ := 4000

-- Define travel costs
def bus_cost_per_km : ℝ := 0.20
def plane_cost_per_km : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the theorem
theorem travel_theorem :
  let YZ : ℝ := Real.sqrt (XY^2 - XZ^2)
  let total_distance : ℝ := XY + YZ + XZ
  let bus_total_cost : ℝ := bus_cost_per_km * total_distance
  let plane_total_cost : ℝ := plane_booking_fee + plane_cost_per_km * total_distance
  total_distance = 10562 ∧ plane_total_cost < bus_total_cost := by
  sorry

end travel_theorem_l2847_284722
