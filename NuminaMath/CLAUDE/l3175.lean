import Mathlib

namespace NUMINAMATH_CALUDE_oil_spend_is_500_l3175_317522

/-- Represents the price reduction, amount difference, and reduced price of oil --/
structure OilPriceData where
  reduction_percent : ℚ
  amount_difference : ℚ
  reduced_price : ℚ

/-- Calculates the amount spent on oil given the price reduction data --/
def calculate_oil_spend (data : OilPriceData) : ℚ :=
  let original_price := data.reduced_price / (1 - data.reduction_percent)
  let m := data.amount_difference * (data.reduced_price * original_price) / (original_price - data.reduced_price)
  m

/-- Theorem stating that given the specific conditions, the amount spent on oil is 500 --/
theorem oil_spend_is_500 (data : OilPriceData) 
  (h1 : data.reduction_percent = 1/4)
  (h2 : data.amount_difference = 5)
  (h3 : data.reduced_price = 25) : 
  calculate_oil_spend data = 500 := by
  sorry

end NUMINAMATH_CALUDE_oil_spend_is_500_l3175_317522


namespace NUMINAMATH_CALUDE_cracked_marbles_count_l3175_317514

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [18, 19, 21, 23, 25, 34]

/-- The total number of marbles -/
def total : Nat := bags.sum

/-- Predicate to check if a number is divisible by 3 -/
def divisibleBy3 (n : Nat) : Prop := n % 3 = 0

/-- Predicate to check if a number leaves remainder 2 when divided by 3 -/
def remainder2Mod3 (n : Nat) : Prop := n % 3 = 2

theorem cracked_marbles_count : 
  ∃ (jennyBags georgeBags : List Nat) (crackedBag : Nat),
    jennyBags.length = 3 ∧
    georgeBags.length = 2 ∧
    crackedBag ∈ bags ∧
    jennyBags ⊆ bags ∧
    georgeBags ⊆ bags ∧
    crackedBag ∉ jennyBags ∧
    crackedBag ∉ georgeBags ∧
    divisibleBy3 (jennyBags.sum + georgeBags.sum) ∧
    remainder2Mod3 crackedBag ∧
    crackedBag = 23 :=
  sorry

end NUMINAMATH_CALUDE_cracked_marbles_count_l3175_317514


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3175_317551

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ x = 3) →
  (∃ x y : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ y^3 - 6*y^2 + a*y - 6 = 0 ∧ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3175_317551


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_six_l3175_317572

theorem sqrt_difference_equals_six :
  Real.sqrt (21 + 12 * Real.sqrt 3) - Real.sqrt (21 - 12 * Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_six_l3175_317572


namespace NUMINAMATH_CALUDE_vanilla_percentage_is_30_percent_l3175_317517

def chocolate : ℕ := 70
def vanilla : ℕ := 90
def strawberry : ℕ := 50
def mint : ℕ := 30
def cookieDough : ℕ := 60

def totalResponses : ℕ := chocolate + vanilla + strawberry + mint + cookieDough

theorem vanilla_percentage_is_30_percent :
  (vanilla : ℚ) / (totalResponses : ℚ) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_percentage_is_30_percent_l3175_317517


namespace NUMINAMATH_CALUDE_max_parts_three_planes_correct_l3175_317528

/-- The maximum number of parts into which three non-overlapping planes can divide space -/
def max_parts_three_planes : ℕ := 8

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- A configuration of three non-overlapping planes in 3D space -/
structure ThreePlaneConfiguration where
  plane1 : Plane3D
  plane2 : Plane3D
  plane3 : Plane3D
  non_overlapping : plane1 ≠ plane2 ∧ plane1 ≠ plane3 ∧ plane2 ≠ plane3

/-- The number of parts into which a configuration of three planes divides space -/
def num_parts (config : ThreePlaneConfiguration) : ℕ :=
  sorry -- The actual calculation would go here

theorem max_parts_three_planes_correct :
  ∀ (config : ThreePlaneConfiguration), num_parts config ≤ max_parts_three_planes ∧
  ∃ (config : ThreePlaneConfiguration), num_parts config = max_parts_three_planes :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_correct_l3175_317528


namespace NUMINAMATH_CALUDE_division_problem_l3175_317554

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 222 →
  quotient = 17 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 13 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3175_317554


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3175_317578

/-- An arithmetic sequence with general term formula aₙ = -n + 5 -/
def arithmeticSequence (n : ℕ) : ℤ := -n + 5

/-- The common difference of an arithmetic sequence -/
def commonDifference (a : ℕ → ℤ) : ℤ := a (1 : ℕ) - a 0

theorem arithmetic_sequence_common_difference :
  commonDifference arithmeticSequence = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3175_317578


namespace NUMINAMATH_CALUDE_isosceles_perpendicular_division_l3175_317523

/-- An isosceles triangle with base 32 and legs 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_eq : base = 32
  leg_eq : leg = 20

/-- The perpendicular from the apex to the base divides the base into two segments -/
def perpendicular_segments (t : IsoscelesTriangle) : ℝ × ℝ :=
  (7, 25)

/-- Theorem: The perpendicular from the apex of the isosceles triangle
    divides the base into segments of 7 and 25 units -/
theorem isosceles_perpendicular_division (t : IsoscelesTriangle) :
  perpendicular_segments t = (7, 25) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_perpendicular_division_l3175_317523


namespace NUMINAMATH_CALUDE_video_vote_ratio_l3175_317553

theorem video_vote_ratio : 
  let up_votes : ℕ := 18
  let down_votes : ℕ := 4
  let ratio : ℚ := up_votes / down_votes
  ratio = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_video_vote_ratio_l3175_317553


namespace NUMINAMATH_CALUDE_history_paper_pages_l3175_317500

/-- Calculates the total number of pages in a paper given the number of days and pages per day. -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days, requiring 27 pages per day, has 81 pages in total. -/
theorem history_paper_pages : total_pages 3 27 = 81 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3175_317500


namespace NUMINAMATH_CALUDE_silent_reading_ratio_l3175_317536

theorem silent_reading_ratio (total : ℕ) (board_games : ℕ) (homework : ℕ) 
  (h1 : total = 24)
  (h2 : board_games = total / 3)
  (h3 : homework = 4)
  : (total - board_games - homework) * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_silent_reading_ratio_l3175_317536


namespace NUMINAMATH_CALUDE_triangle_acute_if_tan_product_positive_l3175_317563

/-- Given a triangle ABC with internal angles A, B, and C, 
    if the product of their tangents is positive, 
    then the triangle is acute. -/
theorem triangle_acute_if_tan_product_positive 
  (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : Real.tan A * Real.tan B * Real.tan C > 0) : 
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_if_tan_product_positive_l3175_317563


namespace NUMINAMATH_CALUDE_full_price_revenue_is_1250_l3175_317505

/-- Represents the revenue from full-price tickets in a school club's ticket sale. -/
def revenue_full_price (full_price : ℚ) (num_full_price : ℕ) : ℚ :=
  full_price * num_full_price

/-- Represents the total revenue from all tickets sold. -/
def total_revenue (full_price : ℚ) (num_full_price : ℕ) (num_discount_price : ℕ) : ℚ :=
  revenue_full_price full_price num_full_price + (full_price / 3) * num_discount_price

/-- Theorem stating that the revenue from full-price tickets is $1250. -/
theorem full_price_revenue_is_1250 :
  ∃ (full_price : ℚ) (num_full_price num_discount_price : ℕ),
    num_full_price + num_discount_price = 200 ∧
    total_revenue full_price num_full_price num_discount_price = 2500 ∧
    revenue_full_price full_price num_full_price = 1250 :=
sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_1250_l3175_317505


namespace NUMINAMATH_CALUDE_computer_purchase_cost_l3175_317592

theorem computer_purchase_cost (computer_cost : ℕ) (base_video_card_cost : ℕ) 
  (h1 : computer_cost = 1500)
  (h2 : base_video_card_cost = 300) : 
  computer_cost + 
  (computer_cost / 5) + 
  (2 * base_video_card_cost - base_video_card_cost) = 2100 := by
  sorry

#check computer_purchase_cost

end NUMINAMATH_CALUDE_computer_purchase_cost_l3175_317592


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3175_317511

-- Define the discount rates and profit margin as real numbers between 0 and 1
def purchase_discount : Real := 0.3
def sale_discount : Real := 0.2
def profit_margin : Real := 0.3

-- Define the list price as an arbitrary positive real number
def list_price : Real := 100

-- Define the purchase price as a function of the list price and purchase discount
def purchase_price (lp : Real) : Real := lp * (1 - purchase_discount)

-- Define the marked price as a function of the list price
def marked_price (lp : Real) : Real := 1.25 * lp

-- Define the selling price as a function of the marked price and sale discount
def selling_price (mp : Real) : Real := mp * (1 - sale_discount)

-- Define the profit as a function of the selling price and purchase price
def profit (sp : Real) (pp : Real) : Real := sp - pp

-- Theorem statement
theorem merchant_pricing_strategy :
  profit (selling_price (marked_price list_price)) (purchase_price list_price) =
  profit_margin * selling_price (marked_price list_price) := by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3175_317511


namespace NUMINAMATH_CALUDE_exists_valid_sign_assignment_l3175_317565

/-- Represents a vertex in the triangular grid --/
structure Vertex :=
  (x : ℤ)
  (y : ℤ)

/-- Represents a triangle in the grid --/
structure Triangle :=
  (a : Vertex)
  (b : Vertex)
  (c : Vertex)

/-- The type of sign assignment functions --/
def SignAssignment := Vertex → Int

/-- Predicate to check if a triangle satisfies the sign rule --/
def satisfiesRule (f : SignAssignment) (t : Triangle) : Prop :=
  (f t.a = f t.b → f t.c = 1) ∧
  (f t.a ≠ f t.b → f t.c = -1)

/-- The set of all triangles in the grid --/
def allTriangles : Set Triangle := sorry

/-- Statement of the theorem --/
theorem exists_valid_sign_assignment :
  ∃ (f : SignAssignment),
    (∀ t ∈ allTriangles, satisfiesRule f t) ∧
    (∃ v w : Vertex, f v ≠ f w) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_sign_assignment_l3175_317565


namespace NUMINAMATH_CALUDE_joan_gemstone_count_l3175_317556

/-- Proves that Joan has 21 gemstone samples given the conditions of the problem -/
theorem joan_gemstone_count :
  ∀ (minerals_yesterday minerals_today gemstones : ℕ),
    gemstones = minerals_yesterday / 2 →
    minerals_today = minerals_yesterday + 6 →
    minerals_today = 48 →
    gemstones = 21 := by
  sorry

end NUMINAMATH_CALUDE_joan_gemstone_count_l3175_317556


namespace NUMINAMATH_CALUDE_marys_story_characters_l3175_317521

theorem marys_story_characters (total : ℕ) (init_a init_c init_d init_e : ℕ) : 
  total = 60 →
  init_a = total / 2 →
  init_c = init_a / 2 →
  init_d + init_e = total - init_a - init_c →
  init_d = 2 * init_e →
  init_d = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_story_characters_l3175_317521


namespace NUMINAMATH_CALUDE_a_100_value_l3175_317567

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a n - a (n + 1) = 2

theorem a_100_value (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 6) : 
  a 100 = -188 := by
  sorry

end NUMINAMATH_CALUDE_a_100_value_l3175_317567


namespace NUMINAMATH_CALUDE_snake_paint_requirement_l3175_317552

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in one segment of the snake -/
def cubes_per_segment : ℕ := 6

/-- The amount of paint needed for one segment -/
def paint_per_segment : ℕ := 240

/-- The amount of extra paint needed for the ends of the snake -/
def extra_paint_for_ends : ℕ := 20

/-- Theorem stating the total amount of paint needed for the snake -/
theorem snake_paint_requirement :
  let segments := total_cubes / cubes_per_segment
  let paint_for_segments := segments * paint_per_segment
  paint_for_segments + extra_paint_for_ends = 80660 :=
by sorry

end NUMINAMATH_CALUDE_snake_paint_requirement_l3175_317552


namespace NUMINAMATH_CALUDE_remaining_cards_l3175_317548

def initial_cards : ℕ := 13
def cards_given_away : ℕ := 9

theorem remaining_cards : initial_cards - cards_given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l3175_317548


namespace NUMINAMATH_CALUDE_final_walnut_count_l3175_317541

/-- The number of walnuts left in the burrow after the squirrels' actions --/
def walnuts_left (initial_walnuts boy_walnuts boy_dropped girl_walnuts girl_eaten : ℕ) : ℕ :=
  initial_walnuts + (boy_walnuts - boy_dropped) + girl_walnuts - girl_eaten

/-- Theorem stating the final number of walnuts in the burrow --/
theorem final_walnut_count :
  walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_final_walnut_count_l3175_317541


namespace NUMINAMATH_CALUDE_alex_cell_phone_cost_l3175_317538

/-- Represents the cell phone plan cost structure and usage --/
structure CellPhonePlan where
  baseCost : ℝ
  textCost : ℝ
  extraMinuteCost : ℝ
  freeHours : ℝ
  textsSent : ℝ
  hoursUsed : ℝ

/-- Calculates the total cost of the cell phone plan --/
def totalCost (plan : CellPhonePlan) : ℝ :=
  plan.baseCost +
  plan.textCost * plan.textsSent +
  plan.extraMinuteCost * (plan.hoursUsed - plan.freeHours) * 60

/-- Theorem stating that Alex's total cost is $45.00 --/
theorem alex_cell_phone_cost :
  let plan : CellPhonePlan := {
    baseCost := 30
    textCost := 0.04
    extraMinuteCost := 0.15
    freeHours := 25
    textsSent := 150
    hoursUsed := 26
  }
  totalCost plan = 45 := by
  sorry


end NUMINAMATH_CALUDE_alex_cell_phone_cost_l3175_317538


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3175_317539

def total_republicans : ℕ := 12
def total_democrats : ℕ := 10
def subcommittee_republicans : ℕ := 5
def subcommittee_democrats : ℕ := 4

theorem subcommittee_formation_count :
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 166320 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3175_317539


namespace NUMINAMATH_CALUDE_arrangements_eq_two_pow_l3175_317597

/-- The number of arrangements of the sequence 1, 2, ..., n, where each number
    is either strictly greater than all the numbers before it or strictly less
    than all the numbers before it. -/
def arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * arrangements (n - 1)

/-- Theorem stating that the number of arrangements for n numbers is 2^(n-1) -/
theorem arrangements_eq_two_pow (n : ℕ) : arrangements n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_two_pow_l3175_317597


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l3175_317596

theorem smallest_multiples_sum :
  ∀ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0) → d ≤ y) →
  c + d = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l3175_317596


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l3175_317574

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of students selected -/
def selected_students : ℕ := 2

/-- Represents the event of exactly one boy being selected -/
def one_boy_selected (k : ℕ) : Prop := k = 1

/-- Represents the event of exactly two boys being selected -/
def two_boys_selected (k : ℕ) : Prop := k = 2

/-- States that the events are mutually exclusive -/
theorem events_mutually_exclusive : 
  ∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k) :=
sorry

/-- States that the events are not opposite -/
theorem events_not_opposite : 
  ∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k) :=
sorry

/-- Main theorem stating that the events are mutually exclusive but not opposite -/
theorem mutually_exclusive_not_opposite : 
  (∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k)) ∧
  (∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l3175_317574


namespace NUMINAMATH_CALUDE_initial_erasers_eq_taken_plus_left_l3175_317526

/-- The initial number of erasers in the box -/
def initial_erasers : ℕ := 69

/-- The number of erasers Doris took out of the box -/
def erasers_taken : ℕ := 54

/-- The number of erasers left in the box -/
def erasers_left : ℕ := 15

/-- Theorem stating that the initial number of erasers is equal to
    the sum of erasers taken and erasers left -/
theorem initial_erasers_eq_taken_plus_left :
  initial_erasers = erasers_taken + erasers_left := by
  sorry

end NUMINAMATH_CALUDE_initial_erasers_eq_taken_plus_left_l3175_317526


namespace NUMINAMATH_CALUDE_f_value_at_5_l3175_317533

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

-- State the theorem
theorem f_value_at_5 (a b c m : ℝ) :
  f a b c (-5) = m → f a b c 5 = -m + 4 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5_l3175_317533


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l3175_317577

theorem consecutive_integers_sum_of_cubes (a b c : ℕ) : 
  (a > 0) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (a^2 + b^2 + c^2 = 2450) → 
  (a^3 + b^3 + c^3 = 73341) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l3175_317577


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l3175_317546

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) - 
  Real.sin (43 * π / 180) * Real.sin (77 * π / 180) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l3175_317546


namespace NUMINAMATH_CALUDE_min_value_of_f_l3175_317501

def f (x a : ℝ) : ℝ := 2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

theorem min_value_of_f (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x a ≥ 2) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f x a = 2) →
  a = -3 - Real.sqrt 7 ∨ a = 0 ∨ a = 2 ∨ a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3175_317501


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l3175_317520

/-- The speed of the first cyclist in meters per second -/
def v : ℝ := 7

/-- The speed of the second cyclist in meters per second -/
def second_cyclist_speed : ℝ := 8

/-- The circumference of the circular track in meters -/
def track_circumference : ℝ := 300

/-- The time taken for the cyclists to meet at the starting point in seconds -/
def meeting_time : ℝ := 20

theorem cyclist_speed_proof :
  v * meeting_time + second_cyclist_speed * meeting_time = track_circumference :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l3175_317520


namespace NUMINAMATH_CALUDE_distance_between_x_and_y_l3175_317516

-- Define the walking speeds
def yolanda_speed : ℝ := 2
def bob_speed : ℝ := 4

-- Define the time difference in starting
def time_difference : ℝ := 1

-- Define Bob's distance walked when they meet
def bob_distance : ℝ := 25.333333333333332

-- Define the total distance between X and Y
def total_distance : ℝ := 40

-- Theorem statement
theorem distance_between_x_and_y :
  let time_bob_walked := bob_distance / bob_speed
  let yolanda_distance := yolanda_speed * (time_bob_walked + time_difference)
  yolanda_distance + bob_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_distance_between_x_and_y_l3175_317516


namespace NUMINAMATH_CALUDE_radian_measure_60_degrees_l3175_317508

theorem radian_measure_60_degrees : 
  (60 : ℝ) * (π / 180) = π / 3 := by sorry

end NUMINAMATH_CALUDE_radian_measure_60_degrees_l3175_317508


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3175_317530

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  (a = 2) →
  (b = Real.sqrt 3) →
  (B = π / 3) →
  -- Triangle definition (implicitly assuming it's a valid triangle)
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  A = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3175_317530


namespace NUMINAMATH_CALUDE_stating_last_passenger_seat_probability_l3175_317581

/-- 
Represents the probability that the last passenger sits in their assigned seat
given n seats and n passengers, where the first passenger (Absent-Minded Scientist)
takes a random seat, and subsequent passengers follow the described seating rules.
-/
def last_passenger_correct_seat_prob (n : ℕ) : ℚ :=
  if n ≥ 2 then 1/2 else 0

/-- 
Theorem stating that for any number of seats n ≥ 2, the probability that 
the last passenger sits in their assigned seat is 1/2.
-/
theorem last_passenger_seat_probability (n : ℕ) (h : n ≥ 2) : 
  last_passenger_correct_seat_prob n = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_stating_last_passenger_seat_probability_l3175_317581


namespace NUMINAMATH_CALUDE_correct_answer_l3175_317527

theorem correct_answer (x : ℝ) (h : x / 3 = 27) : x * 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l3175_317527


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3175_317590

/-- The number of unique games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 7 players, where each player plays every other player once,
    the total number of games played is 21. -/
theorem chess_tournament_games :
  num_games 7 = 21 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3175_317590


namespace NUMINAMATH_CALUDE_bear_climbing_problem_l3175_317534

/-- Represents the mountain climbing problem with two bears -/
structure MountainClimb where
  S : ℝ  -- Total distance from base to summit in meters
  VA : ℝ  -- Bear A's ascending speed
  VB : ℝ  -- Bear B's ascending speed
  meetingTime : ℝ  -- Time when bears meet (in hours)
  meetingDistance : ℝ  -- Distance from summit where bears meet (in meters)

/-- The theorem statement for the mountain climbing problem -/
theorem bear_climbing_problem (m : MountainClimb) : 
  m.VA > m.VB ∧  -- Bear A is faster than Bear B
  m.meetingTime = 2 ∧  -- Bears meet after 2 hours
  m.meetingDistance = 1600 ∧  -- Bears meet 1600 meters from summit
  m.S - 1600 = 2 * m.meetingTime * (m.VA + m.VB) ∧  -- Meeting condition
  (m.S + 800) / (m.S - 1600) = 5 / 4 →  -- Condition when Bear B reaches summit
  (m.S / m.VA + m.S / (2 * m.VA)) = 14 / 5  -- Total time for Bear A
  := by sorry

end NUMINAMATH_CALUDE_bear_climbing_problem_l3175_317534


namespace NUMINAMATH_CALUDE_spatial_relations_theorem_l3175_317566

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between a line and a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def plane_parallel_plane (p1 p2 : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def line_parallel_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem spatial_relations_theorem 
  (m n : Line3D) 
  (α β : Plane3D) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ l : Line3D, line_parallel_plane m α → 
    (line_in_plane l α → line_parallel_line m l)) ∧
  (¬ (plane_parallel_plane α β → line_in_plane m α → 
    line_in_plane n β → line_parallel_line m n)) ∧
  (line_perp_plane m α → line_perp_plane n β → 
    line_parallel_line m n → plane_parallel_plane α β) ∧
  (plane_parallel_plane α β → line_in_plane m α → 
    line_parallel_plane m β) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relations_theorem_l3175_317566


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l3175_317510

def total_lamps : ℕ := 6
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 2
def lamps_turned_on : ℕ := 3

def probability_specific_arrangement : ℚ := 2 / 25

theorem specific_arrangement_probability :
  (Nat.choose total_lamps blue_lamps * Nat.choose total_lamps lamps_turned_on) *
  probability_specific_arrangement =
  (Nat.choose (total_lamps - 2) (blue_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_turned_on - 1)) :=
by sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l3175_317510


namespace NUMINAMATH_CALUDE_price_reduction_equality_l3175_317525

theorem price_reduction_equality (z : ℝ) (h : z > 0) : 
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (z * (1 - 15/100) * (1 - 15/100) = z * (1 - x/100)) ∧
  x = 27.75 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equality_l3175_317525


namespace NUMINAMATH_CALUDE_mile_to_rod_l3175_317599

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversions
axiom mile_to_furlong : mile = 10 * furlong
axiom furlong_to_rod : furlong = 40 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 400 * rod := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l3175_317599


namespace NUMINAMATH_CALUDE_bracelet_price_is_15_l3175_317584

/-- The price of a gold heart necklace in dollars -/
def gold_heart_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def coffee_mug_price : ℕ := 20

/-- The number of bracelets bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces bought -/
def necklaces_bought : ℕ := 2

/-- The number of coffee mugs bought -/
def mugs_bought : ℕ := 1

/-- The amount paid in dollars -/
def amount_paid : ℕ := 100

/-- The change received in dollars -/
def change_received : ℕ := 15

theorem bracelet_price_is_15 :
  ∃ (bracelet_price : ℕ),
    bracelet_price * bracelets_bought +
    gold_heart_price * necklaces_bought +
    coffee_mug_price * mugs_bought =
    amount_paid - change_received ∧
    bracelet_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_price_is_15_l3175_317584


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3175_317524

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 → x ≠ -4 →
    (7 * x + 3) / (x^2 - 5*x - 36) = (66/13) / (x - 9) + (25/13) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3175_317524


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3175_317585

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 4*a| ≤ 4) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3175_317585


namespace NUMINAMATH_CALUDE_inequality_proof_l3175_317535

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (hd : 0 < d ∧ d ≤ 1) : 
  1 / (a^2 + b^2 + c^2 + d^2) ≥ 1/4 + (1-a)*(1-b)*(1-c)*(1-d) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3175_317535


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3175_317591

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, 725*m ≡ 1025*m [ZMOD 40] → n ≤ m) ∧ 
  (725*n ≡ 1025*n [ZMOD 40]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3175_317591


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3175_317583

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3175_317583


namespace NUMINAMATH_CALUDE_min_value_expression_l3175_317570

theorem min_value_expression (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (a + 3) * (b + 2) ≤ (x + 3) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 2 * a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 3) * (b₀ + 2) = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3175_317570


namespace NUMINAMATH_CALUDE_inequality_proof_l3175_317558

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) :
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3175_317558


namespace NUMINAMATH_CALUDE_fence_bricks_l3175_317540

/-- Calculates the number of bricks needed for a rectangular fence --/
def bricks_needed (length width height depth : ℕ) : ℕ :=
  4 * length * width * depth

theorem fence_bricks :
  bricks_needed 20 5 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_fence_bricks_l3175_317540


namespace NUMINAMATH_CALUDE_equal_savings_after_25_weeks_l3175_317588

/-- Proves that saving 7 dollars per week results in equal savings after 25 weeks --/
theorem equal_savings_after_25_weeks 
  (your_initial_savings : ℕ := 160)
  (friend_initial_savings : ℕ := 210)
  (friend_weekly_savings : ℕ := 5)
  (weeks : ℕ := 25)
  (your_weekly_savings : ℕ := 7) : 
  your_initial_savings + weeks * your_weekly_savings = 
  friend_initial_savings + weeks * friend_weekly_savings := by
sorry

end NUMINAMATH_CALUDE_equal_savings_after_25_weeks_l3175_317588


namespace NUMINAMATH_CALUDE_jed_cards_40_after_4_weeks_l3175_317582

/-- The number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + 6 * weeks - 2 * (weeks / 2)

/-- The theorem stating that Jed will have 40 cards after 4 weeks -/
theorem jed_cards_40_after_4_weeks :
  cards_after_weeks 20 4 = 40 := by sorry

end NUMINAMATH_CALUDE_jed_cards_40_after_4_weeks_l3175_317582


namespace NUMINAMATH_CALUDE_expression_evaluation_l3175_317568

theorem expression_evaluation :
  let x : ℕ := 3
  (x + x * x^(x^2)) * 3 = 177156 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3175_317568


namespace NUMINAMATH_CALUDE_range_of_g_l3175_317594

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1

-- Define the range of a function
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y ∈ S, ∃ x, f x = y

-- State the theorem
theorem range_of_g (a : ℝ) :
  (has_range (f a) Set.univ) → (has_range (g a) {y | y ≥ 1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l3175_317594


namespace NUMINAMATH_CALUDE_customized_bowling_ball_volume_l3175_317532

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  let total_hole_volume := 2 * small_hole_volume + 2 * large_hole_volume
  sphere_volume - total_hole_volume = 2239 * π :=
by sorry

end NUMINAMATH_CALUDE_customized_bowling_ball_volume_l3175_317532


namespace NUMINAMATH_CALUDE_xy_inequality_l3175_317543

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2) : x * y + 3 ≥ 2 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l3175_317543


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l3175_317589

theorem unique_three_digit_divisible_by_seven :
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 5 ∧          -- units digit is 5
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0             -- divisible by 7
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l3175_317589


namespace NUMINAMATH_CALUDE_remainder_theorem_l3175_317542

theorem remainder_theorem (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 4 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3175_317542


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l3175_317557

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the first type of tray -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the second type of tray -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l3175_317557


namespace NUMINAMATH_CALUDE_number_of_late_classmates_l3175_317559

/-- The number of late classmates given Charlize's lateness, classmates' additional lateness, and total late time -/
def late_classmates (charlize_lateness : ℕ) (classmate_additional_lateness : ℕ) (total_late_time : ℕ) : ℕ :=
  (total_late_time - charlize_lateness) / (charlize_lateness + classmate_additional_lateness)

/-- Theorem stating that the number of late classmates is 4 given the specific conditions -/
theorem number_of_late_classmates :
  late_classmates 20 10 140 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_late_classmates_l3175_317559


namespace NUMINAMATH_CALUDE_units_digit_product_l3175_317576

theorem units_digit_product (n : ℕ) : 
  (4^150 * 9^151 * 16^152) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_product_l3175_317576


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l3175_317555

/-- The cost per page for revision in a manuscript typing service --/
def revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (initial_cost : ℕ) (total_cost : ℕ) : ℚ :=
  let pages_not_revised := total_pages - revised_once - revised_twice
  let initial_typing_cost := total_pages * initial_cost
  let revision_pages := revised_once + 2 * revised_twice
  (total_cost - initial_typing_cost : ℚ) / revision_pages

/-- Theorem stating the revision cost for the given manuscript --/
theorem manuscript_revision_cost :
  revision_cost 200 80 20 5 1360 = 3 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l3175_317555


namespace NUMINAMATH_CALUDE_third_side_length_valid_l3175_317544

theorem third_side_length_valid (a b c : ℝ) : 
  a = 2 → b = 4 → c = 4 → 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_valid_l3175_317544


namespace NUMINAMATH_CALUDE_age_difference_proof_l3175_317531

theorem age_difference_proof (people : Fin 5 → ℕ) 
  (h1 : people 0 = people 1 + 1)
  (h2 : people 2 = people 3 + 2)
  (h3 : people 4 = people 5 + 3)
  (h4 : people 6 = people 7 + 4) :
  people 9 = people 8 + 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3175_317531


namespace NUMINAMATH_CALUDE_fraction_equality_l3175_317573

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3175_317573


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3175_317515

theorem inequality_solution_set (x : ℝ) : 
  |5*x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3175_317515


namespace NUMINAMATH_CALUDE_triangle_properties_l3175_317529

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (S : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.sqrt 3 / 2) * (t.a * t.b * Real.cos t.C) = 2 * t.S)
  (h2 : t.c = Real.sqrt 6) :
  t.C = π / 3 ∧ t.a + t.b + t.c ≤ 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3175_317529


namespace NUMINAMATH_CALUDE_number_line_relations_l3175_317513

theorem number_line_relations (a b m n p q : ℝ) 
  (ha : 1/2 < a ∧ a < 1) (hb : 1/2 < b ∧ b < 1) :
  (1 < a + b ∧ a + b < 2) ∧ 
  (a - b < 0) ∧
  (1/4 < a * b ∧ a * b < 1) := by
  sorry

end NUMINAMATH_CALUDE_number_line_relations_l3175_317513


namespace NUMINAMATH_CALUDE_moms_ice_cream_scoops_pierre_ice_cream_problem_l3175_317519

/-- Given the cost of ice cream scoops and the total bill, calculate the number of scoops Pierre's mom gets. -/
theorem moms_ice_cream_scoops (cost_per_scoop : ℕ) (pierres_scoops : ℕ) (total_bill : ℕ) : ℕ :=
  let moms_scoops := (total_bill - cost_per_scoop * pierres_scoops) / cost_per_scoop
  moms_scoops

/-- Prove that given the specific conditions, Pierre's mom gets 4 scoops of ice cream. -/
theorem pierre_ice_cream_problem :
  moms_ice_cream_scoops 2 3 14 = 4 := by
  sorry

end NUMINAMATH_CALUDE_moms_ice_cream_scoops_pierre_ice_cream_problem_l3175_317519


namespace NUMINAMATH_CALUDE_shoe_matching_problem_l3175_317593

/-- Represents a collection of shoes -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (color_count : ℕ)
  (indistinguishable : Bool)

/-- 
Given a collection of shoes, returns the minimum number of shoes
needed to guarantee at least one matching pair of the same color
-/
def minShoesForMatch (collection : ShoeCollection) : ℕ :=
  collection.total_pairs + 1

/-- Theorem statement for the shoe matching problem -/
theorem shoe_matching_problem (collection : ShoeCollection) 
  (h1 : collection.total_pairs = 24)
  (h2 : collection.color_count = 2)
  (h3 : collection.indistinguishable = true) :
  minShoesForMatch collection = 25 := by
  sorry

#check shoe_matching_problem

end NUMINAMATH_CALUDE_shoe_matching_problem_l3175_317593


namespace NUMINAMATH_CALUDE_discounted_soda_price_l3175_317562

/-- Calculates the price of discounted soda cans -/
theorem discounted_soda_price (regular_price : ℝ) (discount_percent : ℝ) (num_cans : ℕ) : 
  regular_price = 0.30 →
  discount_percent = 15 →
  num_cans = 72 →
  num_cans * (regular_price * (1 - discount_percent / 100)) = 18.36 :=
by sorry

end NUMINAMATH_CALUDE_discounted_soda_price_l3175_317562


namespace NUMINAMATH_CALUDE_pattern_D_cannot_form_cube_l3175_317512

/-- A pattern is a collection of connected squares. -/
structure Pattern where
  squares : ℕ
  connected : Bool

/-- A cube requires exactly 6 faces. -/
def cube_faces : ℕ := 6

/-- A pattern can form a cube if it has exactly 6 squares and they are connected. -/
def can_form_cube (p : Pattern) : Prop :=
  p.squares = cube_faces ∧ p.connected

/-- Pattern D has 7 squares arranged in a "+" shape with an extra square. -/
def pattern_D : Pattern :=
  { squares := 7, connected := true }

/-- Theorem: Pattern D cannot form a cube. -/
theorem pattern_D_cannot_form_cube : ¬(can_form_cube pattern_D) := by
  sorry


end NUMINAMATH_CALUDE_pattern_D_cannot_form_cube_l3175_317512


namespace NUMINAMATH_CALUDE_order_of_trigonometric_functions_l3175_317507

theorem order_of_trigonometric_functions : 
  let a := Real.sin (Real.sin (2008 * π / 180))
  let b := Real.sin (Real.cos (2008 * π / 180))
  let c := Real.cos (Real.sin (2008 * π / 180))
  let d := Real.cos (Real.cos (2008 * π / 180))
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_order_of_trigonometric_functions_l3175_317507


namespace NUMINAMATH_CALUDE_area_at_stage_4_l3175_317575

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the growth of the rectangle at each stage --/
def grow (r : Rectangle) : Rectangle :=
  { width := r.width + 2, length := r.length + 3 }

/-- Calculates the rectangle at a given stage --/
def rectangleAtStage (n : ℕ) : Rectangle :=
  match n with
  | 0 => { width := 2, length := 3 }
  | n + 1 => grow (rectangleAtStage n)

theorem area_at_stage_4 : area (rectangleAtStage 4) = 150 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_4_l3175_317575


namespace NUMINAMATH_CALUDE_impossible_to_gather_all_stones_l3175_317586

/-- Represents the number of stones in each pile -/
structure PileState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a valid move -/
inductive Move
  | move12 : Move  -- Move from pile 1 and 2 to 3
  | move13 : Move  -- Move from pile 1 and 3 to 2
  | move23 : Move  -- Move from pile 2 and 3 to 1

/-- Apply a move to a PileState -/
def applyMove (state : PileState) (move : Move) : PileState :=
  match move with
  | Move.move12 => PileState.mk (state.pile1 - 1) (state.pile2 - 1) (state.pile3 + 2)
  | Move.move13 => PileState.mk (state.pile1 - 1) (state.pile2 + 2) (state.pile3 - 1)
  | Move.move23 => PileState.mk (state.pile1 + 2) (state.pile2 - 1) (state.pile3 - 1)

/-- Check if all stones are in one pile -/
def isAllInOnePile (state : PileState) : Prop :=
  (state.pile1 = 0 ∧ state.pile2 = 0) ∨
  (state.pile1 = 0 ∧ state.pile3 = 0) ∨
  (state.pile2 = 0 ∧ state.pile3 = 0)

/-- Initial state of the piles -/
def initialState : PileState := PileState.mk 20 1 9

/-- Theorem stating it's impossible to gather all stones in one pile -/
theorem impossible_to_gather_all_stones :
  ¬∃ (moves : List Move), isAllInOnePile (moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_gather_all_stones_l3175_317586


namespace NUMINAMATH_CALUDE_absolute_value_symmetry_l3175_317595

/-- A function f : ℝ → ℝ is symmetric about the line x = c if f(c + x) = f(c - x) for all x ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  SymmetricAbout (fun x ↦ |x - a|) 3 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_symmetry_l3175_317595


namespace NUMINAMATH_CALUDE_substitution_remainder_l3175_317506

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the number of substitute players --/
def substitute_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 11^2 + 11^2 * 10^2 + 11^2 * 10^2 * 9^2 + 11^2 * 10^2 * 9^2 * 8^2

/-- Theorem stating that the remainder when the number of substitution ways
    is divided by 1000 is 722 --/
theorem substitution_remainder :
  substitution_ways % 1000 = 722 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l3175_317506


namespace NUMINAMATH_CALUDE_three_students_got_A_l3175_317587

structure Student :=
  (name : String)
  (gotA : Bool)

def Emily : Student := ⟨"Emily", false⟩
def Frank : Student := ⟨"Frank", false⟩
def Grace : Student := ⟨"Grace", false⟩
def Harry : Student := ⟨"Harry", false⟩

def students : List Student := [Emily, Frank, Grace, Harry]

def emilyStatement (s : List Student) : Prop :=
  (Emily.gotA = true) → (Frank.gotA = true)

def frankStatement (s : List Student) : Prop :=
  (Frank.gotA = true) → (Grace.gotA = true)

def graceStatement (s : List Student) : Prop :=
  (Grace.gotA = true) → (Harry.gotA = true)

def harryStatement (s : List Student) : Prop :=
  (Harry.gotA = true) → (Emily.gotA = false)

def exactlyThreeGotA (s : List Student) : Prop :=
  (s.filter (λ x => x.gotA)).length = 3

theorem three_students_got_A :
  ∀ s : List Student,
    s = students →
    emilyStatement s →
    frankStatement s →
    graceStatement s →
    harryStatement s →
    exactlyThreeGotA s →
    (Frank.gotA = true ∧ Grace.gotA = true ∧ Harry.gotA = true ∧ Emily.gotA = false) :=
by sorry


end NUMINAMATH_CALUDE_three_students_got_A_l3175_317587


namespace NUMINAMATH_CALUDE_sum_of_f_values_negative_l3175_317502

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem sum_of_f_values_negative
  (f : ℝ → ℝ)
  (h_decreasing : is_monotonically_decreasing f)
  (h_odd : is_odd_function f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_f_values_negative_l3175_317502


namespace NUMINAMATH_CALUDE_total_practice_time_is_135_l3175_317509

/-- The number of minutes Daniel practices on a school day -/
def school_day_practice : ℕ := 15

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The number of minutes Daniel practices on a weekend day -/
def weekend_day_practice : ℕ := 2 * school_day_practice

/-- The total practice time for a whole week in minutes -/
def total_practice_time : ℕ := school_day_practice * school_days + weekend_day_practice * weekend_days

theorem total_practice_time_is_135 : total_practice_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_135_l3175_317509


namespace NUMINAMATH_CALUDE_bulb_longevity_probability_l3175_317569

/-- Probability that a bulb from Factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- Probability that a bulb from Factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- Proportion of bulbs supplied by Factory X -/
def supply_x : ℝ := 0.60

/-- Proportion of bulbs supplied by Factory Y -/
def supply_y : ℝ := 1 - supply_x

/-- Theorem stating the probability that a purchased bulb will work for longer than 4000 hours -/
theorem bulb_longevity_probability :
  prob_x * supply_x + prob_y * supply_y = 0.614 := by
  sorry

end NUMINAMATH_CALUDE_bulb_longevity_probability_l3175_317569


namespace NUMINAMATH_CALUDE_coordinate_system_proof_l3175_317549

def M (m : ℝ) : ℝ × ℝ := (m - 2, 2 * m - 7)
def N (n : ℝ) : ℝ × ℝ := (n, 3)

theorem coordinate_system_proof :
  (∀ m : ℝ, M m = (m - 2, 2 * m - 7)) ∧
  (∀ n : ℝ, N n = (n, 3)) →
  (∀ m : ℝ, (M m).2 = 0 → m = 7/2 ∧ M m = (3/2, 0)) ∧
  (∀ m : ℝ, |m - 2| = |2 * m - 7| → m = 5 ∨ m = 3) ∧
  (∀ m n : ℝ, (M m).1 = (N n).1 ∧ |(M m).2 - (N n).2| = 2 → n = 4 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_coordinate_system_proof_l3175_317549


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l3175_317560

/-- The function f(x) defined as a^(x-2) + 3 for some base a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

/-- Theorem stating that (2, 4) is a fixed point of f(x) for any base a > 0 -/
theorem fixed_point_of_f (a : ℝ) (h : a > 0) : f a 2 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l3175_317560


namespace NUMINAMATH_CALUDE_square_of_101_l3175_317545

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l3175_317545


namespace NUMINAMATH_CALUDE_birds_remaining_proof_l3175_317547

/-- Calculates the number of birds remaining on a fence after some fly away. -/
def birdsRemaining (initialBirds flownAway : ℝ) : ℝ :=
  initialBirds - flownAway

/-- Theorem stating that the number of birds remaining is the difference between
    the initial number and the number that flew away. -/
theorem birds_remaining_proof (initialBirds flownAway : ℝ) :
  birdsRemaining initialBirds flownAway = initialBirds - flownAway := by
  sorry

/-- Example calculation for the specific problem -/
example : birdsRemaining 15.3 6.5 = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_birds_remaining_proof_l3175_317547


namespace NUMINAMATH_CALUDE_kitchen_floor_theorem_l3175_317561

/-- Calculates the area of the kitchen floor given the total mopping time,
    mopping rate, and bathroom floor area. -/
def kitchen_floor_area (total_time : ℕ) (mopping_rate : ℕ) (bathroom_area : ℕ) : ℕ :=
  total_time * mopping_rate - bathroom_area

/-- Proves that the kitchen floor area is 80 square feet given the specified conditions. -/
theorem kitchen_floor_theorem :
  kitchen_floor_area 13 8 24 = 80 := by
  sorry

#eval kitchen_floor_area 13 8 24

end NUMINAMATH_CALUDE_kitchen_floor_theorem_l3175_317561


namespace NUMINAMATH_CALUDE_find_multiple_of_ages_l3175_317598

/-- Given Hiram's age and Allyson's age, find the multiple M that satisfies the equation. -/
theorem find_multiple_of_ages (hiram_age allyson_age : ℕ) (M : ℚ)
  (h1 : hiram_age = 40)
  (h2 : allyson_age = 28)
  (h3 : hiram_age + 12 = M * allyson_age - 4) :
  M = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_of_ages_l3175_317598


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3175_317579

theorem initial_markup_percentage (initial_price : ℝ) (price_increase : ℝ) : 
  initial_price = 24 →
  price_increase = 6 →
  let final_price := initial_price + price_increase
  let wholesale_price := final_price / 2
  let initial_markup := initial_price - wholesale_price
  initial_markup / wholesale_price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l3175_317579


namespace NUMINAMATH_CALUDE_prime_divisor_problem_l3175_317580

theorem prime_divisor_problem (p : ℕ) (h_prime : Nat.Prime p) : 
  (∃ k : ℕ, 635 = 7 * k * p + 11) → p = 89 := by sorry

end NUMINAMATH_CALUDE_prime_divisor_problem_l3175_317580


namespace NUMINAMATH_CALUDE_line_equation_l3175_317504

/-- Given a line L with slope -3 and y-intercept 7, its equation is y = -3x + 7 -/
theorem line_equation (L : Set (ℝ × ℝ)) (slope : ℝ) (y_intercept : ℝ)
  (h1 : slope = -3)
  (h2 : y_intercept = 7)
  (h3 : ∀ (x y : ℝ), (x, y) ∈ L ↔ y = slope * x + y_intercept) :
  ∀ (x y : ℝ), (x, y) ∈ L ↔ y = -3 * x + 7 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3175_317504


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l3175_317518

theorem complex_absolute_value_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l3175_317518


namespace NUMINAMATH_CALUDE_fractional_method_optimization_l3175_317564

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The theorem for the fractional method optimization -/
theorem fractional_method_optimization (range : ℕ) (division_points : ℕ) (n : ℕ) :
  range = 21 →
  division_points = 20 →
  fib (n + 1) - 1 = division_points →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_fractional_method_optimization_l3175_317564


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l3175_317537

theorem existence_of_special_polynomial (n : ℕ) (hn : n > 0) :
  ∃ P : Polynomial ℕ,
    (∀ (i : ℕ), Polynomial.coeff P i ∈ ({0, 1} : Set ℕ)) ∧
    (∀ (d : ℕ), d ≥ 2 → P.eval d ≠ 0 ∧ (P.eval d) % n = 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l3175_317537


namespace NUMINAMATH_CALUDE_lcm_and_sum_first_ten_l3175_317571

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

/-- The least common multiple of the first ten positive integers -/
def lcmFirstTen : ℕ := Finset.lcm firstTenIntegers id

/-- The sum of the first ten positive integers -/
def sumFirstTen : ℕ := Finset.sum firstTenIntegers id

theorem lcm_and_sum_first_ten :
  lcmFirstTen = 2520 ∧ sumFirstTen = 55 := by sorry

end NUMINAMATH_CALUDE_lcm_and_sum_first_ten_l3175_317571


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3175_317503

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- Condition that the asymptote slope is positive -/
  asymptote_slope_pos : asymptote_slope > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptote slope √2/2 is √6/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = Real.sqrt 2 / 2) : 
    eccentricity h = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3175_317503


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3175_317550

/-- The radius of an inscribed circle in a right triangle with specific dimensions -/
theorem inscribed_circle_radius (A B C P : ℝ × ℝ) (r : ℝ) : 
  -- ABC is a right triangle with ∠BAC = 90°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →
  -- AB = 8
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 64 →
  -- BC = 10
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 100 →
  -- P is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) →
  -- Circle with radius r is tangent to BC at P and to AB
  ∃ O : ℝ × ℝ, 
    ((O.1 - P.1)^2 + (O.2 - P.2)^2 = r^2) ∧
    ((O.1 - B.1)^2 + (O.2 - B.2)^2 = (r + 8)^2) ∧
    ((O.1 - A.1)^2 + (O.2 - A.2)^2 = r^2) →
  -- Then the radius is √11
  r = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3175_317550
